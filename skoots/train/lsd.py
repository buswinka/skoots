import gunpowder as gp
import numpy as np
import time
import logging
from scipy.ndimage import convolve, gaussian_filter
from numpy.lib.stride_tricks import as_strided
from typing import Tuple, List, Dict, Optional

import torch
from torch import Tensor

from skoots.lib.morphology import gauss_filter

logger = logging.getLogger(__name__)


def get_local_shape_descriptors(
        segmentation,
        sigma,
        components=None,
        voxel_size=None,
        roi=None,
        labels=None,
        mode="gaussian",
        downsample=1,
):
    """
    Compute local shape descriptors for the given segmentation.
    Args:
        segmentation (``np.array`` of ``int``):
            A label array to compute the local shape descriptors for.
        sigma (``tuple`` of ``float``):
            The radius to consider for the local shape descriptor.
        components (``string`` of ``int``, optional):
            The components of the local shape descriptors to compute and return.
            "012" returns the first three components. "0129" returns the first
            three and last components if 3D, "0125" if 2D. Components must be in
            ascending order. Defaults to all components. Valid component
            combinations can be seen in tests folder (components test).
            Component string lookup, where example component : "3D axes", "2D axes"
                mean offset (mean) : "012", "01"
                orthogonal covariance (ortho) : "345", "23"
                diagonal covariance (diag) : "678", "4"
                size : "9", "5"
            example combinations:
                diag + size : "6789", "45"
                mean + diag + size : "0126789", "0145"
                mean + ortho + diag : "012345678", "01234"
                ortho + diag : "345678", "234"
        voxel_size (``tuple`` of ``int``, optional):
            The voxel size of ``segmentation``. Defaults to 1.
        roi (``gunpowder.Roi``, optional):
            Restrict the computation to the given ROI.
        labels (array-like of ``int``, optional):
            Restrict the computation to the given labels. Defaults to all
            labels inside the ``roi`` of ``segmentation``.
        mode (``string``, optional):
            Either ``gaussian`` or ``sphere``. Determines over what region
            the local shape descriptor is computed. For ``gaussian``, a
            Gaussian with the given ``sigma`` is used, and statistics are
            averaged with corresponding weights. For ``sphere``, a sphere
            with radius ``sigma`` is used. Defaults to 'gaussian'.
        downsample (``int``, optional):
            Compute the local shape descriptor on a downsampled volume for
            faster processing. Defaults to 1 (no downsampling).
    """
    return LsdExtractor(sigma, mode, downsample).get_descriptors(
        segmentation, components, voxel_size, roi, labels
    )


def __outer_prodcut(input_array: Tensor) -> Tensor:
    """ computs the unique values of the outer products of the first dim (coord dim) for input """
    k = input_array.shape[0]
    outer = torch.einsum("i...,j...->ij...", input_array, input_array)

    return outer.reshape((k ** 2,) + input_array.shape[1:])


def __get_stats(coords: Tensor, mask: Tensor, sigma_voxel: Tuple[float, ...], sigma: Tuple[float, ...]):
    """


    :param coords: Meshgrid with shape (3, Z, X, Y)
    :param mask: mask with shape (Z, X, Y)
    :param sigma_voxel: sigma / voxel for each dim
    :param sub_roi:
    :param components:
    :return:
    """
    print(coords.shape, mask.shape)
    masked_coords = coords * mask
    count = __aggregate(mask, sigma_voxel)

    count_len = len(count.shape)  # should always be 3 right?
    count[count == 0] = 1

    mean = torch.tensor([
        __aggregate(masked_coords[d], sigma_voxel) for d in range(count_len)
    ]).div(count)
    
    mean_offset = mean - coords

    # covariance
    coords_outer = __outer_prodcut(masked_coords)

    entries = [0, 4, 8, 1, 2, 5]
    covariance = torch.tensor([
        __aggregate(coords_outer[d], sigma_voxel) for d in entries
    ])

    covariance._div(count)._sub(
        __outer_prodcut(mean)[entries]
    ) # in place for memory

    variance = covariance[[0, 1, 2], ...]

    # Pearson coefficients of zy, zx, yx
    pearson = covariance[[3, 4, 5], ...]

    # normalize Pearson correlation coefficient
    variance[variance < 1e-3] = 1e-3  # numerical stability
    pearson[0] /= np.sqrt(variance[0, ...] * variance[1, ...])
    pearson[1] /= np.sqrt(variance[0, ...] * variance[2, ...])
    pearson[2] /= np.sqrt(variance[1, ...] * variance[2, ...])

    # normalize variances to interval [0, 1]
    variance[0] /= sigma[0] ** 2
    variance[1] /= sigma[1] ** 2
    variance[2] /= sigma[2] ** 2

    return (mean_offset, variance, pearson, count[None, :])

def __aggregate(input_array: Tensor, sigma: Tuple[float, ...]):
    """
    Basically just blurs. Different if doing with a sphere, but fuck it

    :param input_array: (X, Y, Z) tensor
    :param sigma:
    :return:
    """
    """
    from lsds
    return gaussian_filter(
                    array, sigma=sigma, mode="constant", cval=0.0, truncate=3.0
                )[roi_slices]

    """
    input_array._unsqueeze(0)._unsqueeze(0)
    return gauss_filter(input_array, kernel=[3, 3, 3], sigma=sigma).squeeze()


def lsd(segmentation: Tensor, sigma: Tuple[float, float, float], voxel_size: Tuple[int, int, int] | None = None):
    """
    Pytorch reimplementation of local-shape-descriptors without gunpowder.
    Credit goes to Jan and Arlo

    Never downsamples, always computes the lsd's for every label. Uses a guassian instead of sphere.

    base implementation assumes numpy ordering (Z, X, Y), we will also do so.

    Shapes:
        - segmentation: (Z, X, Y)

    :param segmentation:  label array to compute the local shape descriptors for
    :param sigma: The radius to consider for the local shape descriptor.
    :param voxel_size:
    :return: lsd
    """

    device = segmentation.device

    shape = segmentation.shape
    labels = torch.unique(segmentation)

    descriptors = torch.zeros((10, shape[0], shape[1], shape[2]), dtype=torch.float, device=device)
    sigma_voxel: Tuple[float, ...] = tuple(s / v for s, v in zip(sigma, voxel_size))

    grid = torch.meshgrid(
        torch.arange(0, shape[0], 1, device=device),
        torch.arange(0, shape[1], 1, device=device),
        torch.arange(0, shape[2], 1, device=device),
        indexing='ij')
    grid = torch.tensor(grid, device=device)
    print(grid.shape)

    for label in labels:
        if label == 0: continue

        mask = (segmentation == label).float()

        descriptor: Tensor = __get_stats(coords=grid, mask=mask, sigma_voxel=sigma_voxel, sigma=sigma)
        descriptors._add(descriptor * mask)

    max_distance = torch.tensor(sigma, dtype=torch.float, device=device)

    # correct descriptors for propper scaling
    descriptors[[0, 1, 2], ...]._div(max_distance.view(3, 1, 1, 1) * 0.5 + 0.5)

    # pearsons in [0, 1]
    descriptors[[6,7,8], ...]._mul(0.5)._add(0.5)

    # reset background to 0
    descriptors[[0,1,2,6,7,8], ...]._mul(segmentation != 0)

    torch.clamp(descriptors, 0, 1, out=descriptors)

    return descriptors




class LsdExtractor(object):
    def __init__(self, sigma, mode="gaussian", downsample=1):
        """
        Create an extractor for local shape descriptors. The extractor caches
        the data repeatedly needed for segmentations of the same size. If this
        is not desired, `func:get_local_shape_descriptors` should be used
        instead.
        Args:
            sigma (``tuple`` of ``float``):
                The radius to consider for the local shape descriptor.
            mode (``string``, optional):
                Either ``gaussian`` or ``sphere``. Determines over what region
                the local shape descriptor is computed. For ``gaussian``, a
                Gaussian with the given ``sigma`` is used, and statistics are
                averaged with corresponding weights. For ``sphere``, a sphere
                with radius ``sigma`` is used. Defaults to 'gaussian'.
            downsample (``int``, optional):
                Compute the local shape descriptor on a downsampled volume for
                faster processing. Defaults to 1 (no downsampling).
        """
        self.sigma = sigma
        self.mode = mode
        self.downsample = downsample
        self.coords = {}

    def get_descriptors(
            self, segmentation, components=None, voxel_size: Tuple[int] = None, roi=None, labels=None
    ):
        """Compute local shape descriptors for a given segmentation.
        Args:
            segmentation (``np.array`` of ``int``):
                A label array to compute the local shape descriptors for.
            components (``string`` of ``int``, optional):
                The components of the local shape descriptors to compute and return.
                "012" returns the first three components. "0129" returns the first three and
                last components if 3D, "0125" if 2D. Components must be in ascending order.
                Defaults to all components.
            voxel_size (``tuple`` of ``int``, optional):
                The voxel size of ``segmentation``. Defaults to 1.
            roi (``gunpowder.Roi``, optional):
                Restrict the computation to the given ROI in voxels.
            labels (array-like of ``int``, optional):
                Restrict the computation to the given labels. Defaults to all
                labels inside the ``roi`` of ``segmentation``.
        """

        dims = len(segmentation.shape)

        if voxel_size is None:
            voxel_size = gp.Coordinate((1,) * dims)
        else:
            voxel_size = gp.Coordinate(voxel_size)

        if roi is None:
            roi = gp.Roi((0,) * dims, segmentation.shape)

        roi_slices = roi.to_slices()
        print(f'{roi_slices=}')

        if labels is None:
            labels = np.unique(segmentation[roi_slices])

        # get number of channels
        if components is None:
            if dims == 2:
                self.sigma = self.sigma[0:2]
                channels = 6
            elif dims == 3:
                channels = 10
            else:
                raise AssertionError(f"Segmentation shape has {dims} dims.")

        else:
            channels = len(components)

        # prepare full-res descriptor volumes for roi
        descriptors = np.zeros((channels,) + roi.get_shape(), dtype=np.float32)
        print(f'{descriptors.shape=}')
        # get sub-sampled shape, roi, voxel size and sigma
        df = self.downsample
        logger.debug(
            "Downsampling segmentation %s with factor %f", segmentation.shape, df
        )

        sub_shape = tuple(s / df for s in segmentation.shape)
        sub_roi = roi / df
        print(f'{sub_roi=}')
        assert sub_roi * df == roi, (
                "Segmentation shape %s is not a multiple of downsampling factor "
                "%d (sub_roi=%s, roi=%s)."
                % (segmentation.shape, self.downsample, sub_roi, roi)
        )
        sub_voxel_size = tuple(v * df for v in voxel_size)
        sub_sigma_voxel = tuple(s / v for s, v in zip(self.sigma, sub_voxel_size))

        logger.debug("Downsampled shape: %s", sub_shape)
        logger.debug("Downsampled voxel size: %s", sub_voxel_size)
        logger.debug("Sigma in voxels: %s", sub_sigma_voxel)

        print(f'{sub_voxel_size=}, {sub_sigma_voxel=}')
        print(f'{sub_shape=}')
        # prepare coords volume (reuse if we already have one)
        if (sub_shape, sub_voxel_size) not in self.coords:

            logger.debug("Create meshgrid...")

            try:
                # 3d by default
                grid = np.meshgrid(
                    np.arange(0, sub_shape[0] * sub_voxel_size[0], sub_voxel_size[0]),
                    np.arange(0, sub_shape[1] * sub_voxel_size[1], sub_voxel_size[1]),
                    np.arange(0, sub_shape[2] * sub_voxel_size[2], sub_voxel_size[2]),
                    indexing="ij",
                )

            except:

                grid = np.meshgrid(
                    np.arange(0, sub_shape[0] * sub_voxel_size[0], sub_voxel_size[0]),
                    np.arange(0, sub_shape[1] * sub_voxel_size[1], sub_voxel_size[1]),
                    indexing="ij",
                )

            self.coords[(sub_shape, sub_voxel_size)] = np.array(grid, dtype=np.float32)

        coords = self.coords[(sub_shape, sub_voxel_size)]

        # for all labels
        for label in labels:

            if label == 0:
                continue

            logger.debug("Creating shape descriptors for label %d", label)

            mask = (segmentation == label).astype(np.float32)
            logger.debug("Label mask %s", mask.shape)

            try:
                # 3d by default
                print(df)
                sub_mask = mask[::df, ::df, ::df]

            except:
                sub_mask = mask[::df, ::df]

            logger.debug("Downsampled label mask %s", sub_mask.shape)

            print(
                f'Input into __get_stats: {coords.shape}, {sub_mask.shape=}, {sub_sigma_voxel=}, {sub_roi=}, {components=}')
            sub_descriptor = np.concatenate(
                self.__get_stats(coords, sub_mask, sub_sigma_voxel, sub_roi, components)
            )

            logger.debug("Upscaling descriptors...")
            start = time.time()
            descriptor = self.__upsample(sub_descriptor, df)
            logger.debug("%f seconds", time.time() - start)

            logger.debug("Accumulating descriptors...")
            start = time.time()
            descriptors += descriptor * mask[roi_slices]
            logger.debug("%f seconds", time.time() - start)

        # normalize stats

        # get max possible mean offset for normalization
        if self.mode == "gaussian":
            # farthest voxel in context is 3*sigma away, but due to Gaussian
            # weighting, sigma itself is probably a better upper bound
            max_distance = np.array([s for s in self.sigma], dtype=np.float32)
        elif self.mode == "sphere":
            # farthest voxel in context is sigma away, but this is almost
            # impossible to reach as offset -- let's take half sigma
            max_distance = np.array([0.5 * s for s in self.sigma], dtype=np.float32)

        if dims == 3:

            # mean offsets (z,y,x) = [0,1,2]
            # covariance (zz,yy,xx) = [3,4,5]
            # pearsons (zy,zx,yx) = [6,7,8]
            # size = [9]

            if components is None:

                # mean offsets in [0, 1]
                descriptors[[0, 1, 2]] = (
                        descriptors[[0, 1, 2]] / max_distance[:, None, None, None] * 0.5
                        + 0.5
                )
                # pearsons in [0, 1]
                descriptors[[6, 7, 8]] = descriptors[[6, 7, 8]] * 0.5 + 0.5
                # reset background to 0
                descriptors[[0, 1, 2, 6, 7, 8]] *= segmentation[roi_slices] != 0

            else:

                for i, c in enumerate(components):

                    c = int(c)

                    if c in range(0, 3):
                        descriptors[[i]] = (
                                descriptors[[i]] / max_distance[c, None, None, None] * 0.5
                                + 0.5
                        )
                        descriptors[[i]] *= segmentation[roi_slices] != 0

                    elif c in range(6, 9):
                        descriptors[[i]] = descriptors[[i]] * 0.5 + 0.5
                        descriptors[[i]] *= segmentation[roi_slices] != 0

                    else:
                        pass

        else:

            # mean offsets (y,x) = [0,1]
            # covariance (yy,xx) = [2,3]
            # pearsons (yx) = [4]
            # size = [5]

            if components is None:

                # mean offsets in [0, 1]
                descriptors[[0, 1]] = (
                        descriptors[[0, 1]] / max_distance[:, None, None] * 0.5 + 0.5
                )
                # pearsons in [0, 1]
                descriptors[[4]] = descriptors[[4]] * 0.5 + 0.5
                # reset background to 0
                descriptors[[0, 1, 4]] *= segmentation[roi_slices] != 0

            else:

                for i, c in enumerate(components):

                    c = int(c)

                    if c in range(0, 2):
                        descriptors[[i]] = (
                                descriptors[[i]] / max_distance[c, None, None] * 0.5 + 0.5
                        )
                        descriptors[[i]] *= segmentation[roi_slices] != 0

                    elif c == 4:
                        descriptors[[i]] = descriptors[[i]] * 0.5 + 0.5
                        descriptors[[i]] *= segmentation[roi_slices] != 0

        # clip outliers
        np.clip(descriptors, 0.0, 1.0, out=descriptors)
        print(f'{descriptors.shape}')
        return descriptors

    def __get_stats(self, coords, mask, sigma_voxel, roi, components):

        # mask for object
        print(f'IN __get_stats: {coords.shape}, {mask.shape=}')
        masked_coords = coords * mask

        # number of inside voxels
        logger.debug("Counting inside voxels...")
        start = time.time()
        count = self.__aggregate(mask, sigma_voxel, self.mode, roi)

        count_len = len(count.shape)

        # avoid division by zero
        count[count == 0] = 1
        logger.debug("%f seconds", time.time() - start)

        # mean
        logger.debug("Computing mean position of inside voxels...")
        start = time.time()

        mean = np.array(
            [
                self.__aggregate(masked_coords[d], sigma_voxel, self.mode, roi)
                for d in range(count_len)
            ]
        )

        mean /= count
        logger.debug("%f seconds", time.time() - start)

        if components is not None:
            calc_mean_offset = True in [
                str(comp) in components for comp in range(count_len)
            ]
            calc_covariance = True in [
                str(comp) in components for comp in range(count_len, 4 * count_len - 3)
            ]

        if components is None or calc_mean_offset:
            logger.debug("Computing offset of mean position...")
            start = time.time()
            mean_offset = mean - coords[(slice(None),) + roi.to_slices()]
            print(f'{mean.shape=}, {coords.shape=}')
            print(f'{mean_offset.shape=}, {mean_offset.max()=}, {mean_offset.min()=}')

        # covariance
        if components is None or calc_covariance:
            logger.debug("Computing covariance...")
            coords_outer = self.__outer_product(masked_coords)

            # remove duplicate entries in covariance
            entries = [0, 4, 8, 1, 2, 5] if count_len == 3 else [0, 3, 1]

            covariance = np.array(
                [
                    self.__aggregate(coords_outer[d], sigma_voxel, self.mode, roi)
                    # 3d:
                    # 0 1 2
                    # 3 4 5
                    # 6 7 8
                    # 2d:
                    # 0 1
                    # 2 3
                    for d in entries
                ]
            )

            covariance /= count
            covariance -= self.__outer_product(mean)[entries]

            logger.debug("%f seconds", time.time() - start)

            if count_len == 3:

                # variances of z, y, x coordinates
                variance = covariance[[0, 1, 2]]

                # Pearson coefficients of zy, zx, yx
                pearson = covariance[[3, 4, 5]]

                # normalize Pearson correlation coefficient
                variance[variance < 1e-3] = 1e-3  # numerical stability
                pearson[0] /= np.sqrt(variance[0] * variance[1])
                pearson[1] /= np.sqrt(variance[0] * variance[2])
                pearson[2] /= np.sqrt(variance[1] * variance[2])

                # normalize variances to interval [0, 1]
                variance[0] /= self.sigma[0] ** 2
                variance[1] /= self.sigma[1] ** 2
                variance[2] /= self.sigma[2] ** 2

            else:

                # variances of y, x coordinates
                variance = covariance[[0, 1]]

                # Pearson coefficients of yx
                pearson = covariance[[2]]

                # normalize Pearson correlation coefficient
                variance[variance < 1e-3] = 1e-3  # numerical stability
                pearson /= np.sqrt(variance[0] * variance[1])

                # normalize variances to interval [0, 1]
                variance[0] /= self.sigma[0] ** 2
                variance[1] /= self.sigma[1] ** 2

        if components is not None:

            ret = tuple()

            for i in components:

                i = int(i)

                if count_len == 3:

                    if i in range(0, 3):
                        ret += (mean_offset[[i]],)
                    elif i in range(3, 6):
                        ret += (variance[[i - 3]],)
                    elif i in range(6, 9):
                        ret += (pearson[[i - 6]],)
                    elif i == 9:
                        ret += (count[None, :],)
                    else:
                        raise AssertionError(
                            f"3D lsds have components in range(0,10), encountered {i}"
                        )

                elif count_len == 2:

                    if i in range(0, 2):
                        ret += (mean_offset[[i]],)
                    elif i in range(2, 4):
                        ret += (variance[[i - 2]],)
                    elif i == 4:
                        ret += (pearson,)
                    elif i == 5:
                        ret += (count[None, :],)
                    else:
                        raise AssertionError(
                            f"2D lsds have components in range(0,6), encountered {i}"
                        )

                else:
                    raise AssertionError(f"Number of dims was found to be {count_len}")

        else:
            ret = (mean_offset, variance, pearson, count[None, :])

        return ret

    def __make_sphere(self, radius):

        logger.debug("Creating sphere with radius %d...", radius)

        r2 = np.arange(-radius, radius) ** 2
        dist2 = r2[:, None, None] + r2[:, None] + r2
        return (dist2 <= radius ** 2).astype(np.float32)

    def __aggregate(self, array, sigma, mode="gaussian", roi=None):

        if roi is None:
            roi_slices = (slice(None),)
        else:
            roi_slices = roi.to_slices()

        if mode == "gaussian":

            out = gaussian_filter(
                array, sigma=sigma, mode="constant", cval=0.0, truncate=3.0
            )[roi_slices]


            return out
        elif mode == "sphere":

            radius = sigma[0]
            for d in range(len(sigma)):
                assert (
                        radius == sigma[d]
                ), "For mode 'sphere', only isotropic sigma is allowed."

            sphere = self.__make_sphere(radius)
            return convolve(array, sphere, mode="constant", cval=0.0)[roi_slices]

        else:
            raise RuntimeError("Unknown mode %s" % mode)

    def get_context(self):

        """Return the context needed to compute the LSDs."""

        if self.mode == "gaussian":
            return tuple((3.0 * s for s in self.sigma))
        elif self.mode == "sphere":
            return self.sigma

    def __outer_product(self, array):

        """Computes the unique values of the outer products of the first dimension
        of ``array``. If ``array`` has shape ``(k, d, h, w)``, for example, the
        output will be of shape ``(k*(k+1)/2, d, h, w)``.
        """
        k = array.shape[0]
        outer = np.einsum("i...,j...->ij...", array, array)

        print(f'__outer_product: {array.shape=}, {outer.shape=}')
        return outer.reshape((k ** 2,) + array.shape[1:])

    def __upsample(self, array, f):

        shape = array.shape
        stride = array.strides

        if len(array.shape) == 4:
            sh = (shape[0], shape[1], f, shape[2], f, shape[3], f)
            st = (stride[0], stride[1], 0, stride[2], 0, stride[3], 0)
        else:
            sh = (shape[0], shape[1], f, shape[2], f)
            st = (stride[0], stride[1], 0, stride[2], 0)

        view = as_strided(array, sh, st)

        l = [shape[0]]
        [l.append(shape[i + 1] * f) for i, j in enumerate(shape[1:])]

        return view.reshape(l)


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

    # lsds = get_local_shape_descriptors(
    #     segmentation=random_crop(labels, 100),
    #     sigma=(5,) * 3,
    #     voxel_size=(1, 1, 4))
    #
    # labels.shape
# def lsd(segmentation: Tensor, sigma: Tuple[float, float, float], voxel_size: Tuple[int, int, int] | None = None):
    crop = torch.tensor(crop)
    torch_lsd = lsd(crop, (5,) * 3, (1,1,4))