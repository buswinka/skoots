import time
from scipy.ndimage import convolve, gaussian_filter
from typing import Tuple, List, Dict, Optional

import torch
from torch import Tensor
import torch.nn.functional as F


def __outer_product(input_array: Tensor) -> Tensor:
    """
    Computs the unique values of the outer products of the first dim (coord dim) for input.
    Ripped straight from the LSD implementation, but in torch

    Shapes:
        - input_array: :math:`(C, Z, X, Y)`
        - returns: :math:`(C*C, Z, X, Y)`

    :param: input array
    """
    k = input_array.shape[0]
    outer = torch.einsum("i...,j...->ij...", input_array, input_array)

    return outer.reshape((k ** 2,) + input_array.shape[1:])


def __get_stats(coords: Tensor, mask: Tensor,
                sigma_voxel: Tuple[float, ...],
                sigma: Tuple[float, ...]) -> Tensor:
    """
    Function computes unscaled shape statistics.

    Stats [0, 1, 2] are for the mean offset
    Stats [3, 4, 5] are the variance
    Stats [6, 7, 8] are the pearson covariance
    Stats [9] is the distance

    None are nomalized!

    Shapes:
        - coords: :math:`(3, Z_{in}, X_{in}, Y_{in}`
        - mask: :math:`(Z_{in}, X_{in}, Y_{in}`
        - sigma_voxel: :math:`(3)`
        - sigma: :math:`(3)`
        - returns: :math:`(10, Z_{in}, X_{in}, Y_{in}`

    :param coords: Meshgrid of indicies
    :param mask: torch.int instance segmentation mask
    :param sigma_voxel: sigma / voxel for each dim
    :param sigma: standard deviation for bluring at each spatial dim
    :return: Statistics for each instance of the instance segmentation mask
    """

    assert coords.ndim == 4
    assert mask.ndim == 3
    assert coords.device == mask.device

    masked_coords: Tensor = coords * mask
    count: Tensor = __aggregate(mask, sigma_voxel)

    count[count == 0] = 1  # done for numerical stability.

    n_spatial_dims = len(count.shape)  # should always be 3

    mean: List[Tensor] = [__aggregate(masked_coords[d], sigma_voxel).unsqueeze(0) for d in range(n_spatial_dims)]
    mean: Tensor = torch.concatenate(mean, dim=0).div(count)

    mean_offset: Tensor = mean - coords

    # covariance
    coords_outer: Tensor = __outer_product(masked_coords)  # [9, X, Y, Z]

    entries = [0, 4, 8, 1, 2, 5]
    covariance: Tensor = torch.concatenate([__aggregate(coords_outer[d], sigma_voxel).unsqueeze(0) for d in entries], dim=0)
    covariance.div_(count).sub_(__outer_product(mean)[entries])  # in place for memory

    variance = covariance[[0, 1, 2], ...]

    # Pearson coefficients of zy, zx, yx
    pearson = covariance[[3, 4, 5], ...]

    # normalize Pearson correlation coefficient
    variance[variance < 1e-3] = 1e-3  # numerical stability
    pearson[0, ...] /= torch.sqrt(variance[0, ...] * variance[1, ...])
    pearson[1, ...] /= torch.sqrt(variance[0, ...] * variance[2, ...])
    pearson[2, ...] /= torch.sqrt(variance[1, ...] * variance[2, ...])

    # normalize variances to interval [0, 1]
    variance[0, ...] /= sigma[0] ** 2
    variance[1, ...] /= sigma[1] ** 2
    variance[2, ...] /= sigma[2] ** 2

    return torch.concatenate((mean_offset, variance, pearson, count.unsqueeze(0)), dim=0)


def __aggregate_default(input_array: Tensor, sigma: Tuple[float, ...]):
    """
    Basically just blurs. Different if doing with a sphere, but fuck it

    :param input_array: (X, Y, Z) tensor
    :param sigma:
    :return:
    """

    device = input_array.device
    return torch.from_numpy(
        gaussian_filter(input_array.cpu().numpy(), sigma=sigma, mode="constant", cval=0.0, truncate=3.0)).to(device)


def make_gaussian_kernel_1d(sigma: float, device: torch.device = 'cpu') -> Tensor:
    """
    ripped from https://stackoverflow.com/questions/67633879/implementing-a-3d-gaussian-blur-using-separable-2d-convolutions-in-pytorch

    :param sigma: standard deviation of gaussian kernel
    :param device: torch device to put kernel on
    :return: 1d gaussian kernel
    """
    kernel_size = int(2 * round(sigma * 3.0) + 1)
    ts = torch.linspace(-kernel_size // 2, kernel_size // 2 + 1, kernel_size, device=device)
    gauss = torch.exp((-(ts / sigma) ** 2 / 2))
    kernel = gauss / gauss.sum()

    return kernel


def __aggregate(array: Tensor, sigma: Tuple[float, ...]) -> Tensor:
    """
    Performs a 3D gaussian blur on the input tensor using repeated 1D convolutions.
    Has slight numerical differences to the native LSD implementation, but Im not paid enough
    to figure that out.

    Shapes:
        - array: :math:`(Z_{in}, X_{in}, Y_{in})`
        - sigma: :math: `(3)`
        - returns :math:`(Z_{in}, X_{in}, Y_{in})`

    :param array: input array to blur
    :param sigma: tuple of standard deviations for each dimension
    :return: blurred array
    """

    assert array.ndim == 3, 'Array must be 3D with shape (Z, X, Y)'
    assert len(sigma) == 3, 'Must provide 3 sigma values, one for each spatial dimension'

    z, x, y = array.shape
    array = array.reshape(1, 1, z, x, y)
    device = array.device

    # Separable 1D convolution
    for i in range(3):

        # 3d convolution need 5D tensor (B, C, X, Y, Z)
        kernel: Tensor = make_gaussian_kernel_1d(sigma=sigma[i], device=device).view(1, 1, -1, 1, 1)
        pad: Tuple[int] = tuple(int((k - 1) // 2) for k in kernel.shape[2::])

        array = array.permute(0, 1, 4, 2, 3)
        array = F.conv3d(array, kernel, stride=1, padding=pad)

    return array.squeeze(0).squeeze(0)


def lsd(segmentation: Tensor, sigma: Tuple[float, float, float], voxel_size: Tuple[int, int, int]):
    """
    Pytorch reimplementation of local-shape-descriptors without gunpowder.
    Credit goes to Jan and Arlo.

    Never downsamples, always computes the lsd's for every label. Uses a guassian instead of sphere

    Base implementation assumes numpy ordering (Z, X, Y), therefore all code uses this ordering, however we
    expect inputs to be in the form (1, X, Y, Z) and outputs to be in the form: (10, X, Y, Z)

    Shapes:
        - segmentation: (X, Y, Z)
        - sigma: (3)
        - voxel_size: (3)
        - returns: (C=10, X, Y, Z)

    :param segmentation:  label array to compute the local shape descriptors for
    :param sigma: The radius to consider for the local shape descriptor.
    :param voxel_size:
    :return: local shape descriptors
    """

    segmentation = segmentation.squeeze(0).permute(2, 0, 1)

    device = segmentation.device

    shape = segmentation.shape
    labels = torch.unique(segmentation)

    descriptors = torch.zeros((10, shape[0], shape[1], shape[2]), dtype=torch.float, device=device)
    sigma_voxel = [s / v for s, v in zip(sigma, voxel_size)]

    # Grid of indexes for computing the descriptors. Can be cached.
    grid = torch.meshgrid(
        torch.arange(0, shape[0] * voxel_size[0], voxel_size[0], device=device),
        torch.arange(0, shape[1] * voxel_size[1], voxel_size[1], device=device),
        torch.arange(0, shape[2] * voxel_size[2], voxel_size[2], device=device),
        indexing='ij')
    grid = [g.unsqueeze(0) for g in grid]
    grid = torch.concatenate(grid, dim=0)

    for label in labels:  # do this for each instance
        if label == 0: continue

        mask = (segmentation == label).float()
        descriptor: Tensor = __get_stats(coords=grid, mask=mask, sigma_voxel=sigma_voxel, sigma=sigma)
        descriptors.add_(descriptor * mask)

    max_distance = torch.tensor(sigma, dtype=torch.float, device=device)

    # correct descriptors for proper scaling
    descriptors[[0, 1, 2], ...] = (
            descriptors[[0, 1, 2], ...] / max_distance[:, None, None, None] * 0.5
            + 0.5
    )
    # pearsons in [0, 1]
    descriptors[[6, 7, 8], ...] = descriptors[[6, 7, 8], ...] * 0.5 + 0.5

    # reset background to 0
    descriptors[[0, 1, 2, 6, 7, 8], ...] *= segmentation != 0

    # Clamp to reasonable values
    torch.clamp(descriptors, 0, 1, out=descriptors)

    return descriptors.permute(0, 2, 3, 1)


if __name__ == '__main__':
    import h5py
    import io
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    import requests
    import skimage

    from scipy.ndimage import median_filter
    from skimage.filters import sobel, threshold_li
    from skimage.measure import label
    from skimage.morphology import remove_small_holes, remove_small_objects

    # get 3d cells data, use nuceli channel
    data = skimage.data.cells3d()[:, 1]

    # ignore end sections that don't contain cells
    data = data[25:45]

    # denoise
    denoised = median_filter(data, size=3)

    # create binary mask
    thresholded = denoised > threshold_li(denoised)

    # remove small holes and objects
    remove_holes = remove_small_holes(thresholded, 20 ** 3)
    remove_objects = remove_small_objects(remove_holes, 20 ** 3)

    # relabel connected components
    labels = label(remove_objects).astype(np.int64)


    # take a random crop for efficiency
    def random_crop(labels, crop_size):
        y = random.randint(0, labels.shape[1] - crop_size)
        x = random.randint(0, labels.shape[2] - crop_size)
        labels = labels[:, y:y + crop_size, x:x + crop_size]
        return labels


    crop = random_crop(labels, 100)


    crop = torch.tensor(crop).to('cuda')

    lsds = lsd(crop, (5,) * 3, (1, 1, 4)).cpu().numpy()

    fig, axes = plt.subplots(
                1,
                4,
                figsize=(25, 10),
                sharex=True,
                sharey=True,
                squeeze=False)

    # lsds are shape: c,z,y,x (where channels is now 10 dimensions)
    # first 3 components can be rendered as rgb, matplotlib expects channels last
    axes[0][0].imshow(lsds[0:3,10].T)
    axes[0][0].title.set_text('Mean offset')

    axes[0][1].imshow(lsds[3:6,10].T)
    axes[0][1].title.set_text('Covariance')

    axes[0][2].imshow(lsds[6:9,10].T)
    axes[0][2].title.set_text('Pearsons')

    axes[0][3].imshow(lsds[9,10].T, cmap='jet')
    axes[0][3].title.set_text('Size')
    plt.show()




