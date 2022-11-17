import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from typing import List, Tuple, Dict, Optional

from skoots.lib.morphology import gauss_filter, binary_dilation, _compute_zero_padding, _get_binary_kernel3d


@torch.jit.script
def average_baked_skeletons(baked_skeleton: Tensor, kernel_size: int = 3) -> Tensor:
    """
    Takes a baked skeleton computed by skoots.lib.skeleton.bake_skeleton and averages all the values such that
    there is a smooth transition from one location to another.

    Shapes:
        - baked_skeleton :math:`(B_{in}, 3, X_{in}, Y_{in}, Z_{in})`
        - returns :math:`(B_{in}, 3, X_{in}, Y_{in}, Z_{in})`

    :param baked_skeleton: Baked skeleton tensor
    :param kernel_size: Kernel size for smoothing
    :return: smoothed skeleton: Smoothed Tensor
    """
    padding: Tuple[int, int, int] = _compute_zero_padding((kernel_size, kernel_size, kernel_size))
    kernel: Tensor = _get_binary_kernel3d(kernel_size, str(baked_skeleton.device))
    b, c, h, w, d = baked_skeleton.shape

    # map the local window to single vector
    features: Tensor = F.conv3d(baked_skeleton.reshape(b * c, 1, h, w, d), kernel,
                                padding=padding, stride=1)
    features: Tensor = features.view(b, c, -1, h, w, d)  # B, C, -1, X, Y, Z

    nonzero = features.gt(0).sum(2)  # B, C, X, Y, Z
    nonzero[nonzero.eq(0)] = 1.
    features = features.sum(2)  # This is the average of the kernel window... without zeros
    features = features / nonzero

    return features


@torch.jit.ignore
def bake_skeleton(masks: Tensor,
                  skeletons: Dict[int, Tensor],
                  anisotropy: List[float] = (1., 1., 1.),
                  average: bool = True,
                  device: str = 'cpu'
                  ) -> Tensor:
    """
    For each pixel :math:`p_ik` of object :math:`k` at index :math:`i\in[x,y,z]` in masks, returns a baked skeleton
    where the value at each index is the closest skeleton point :math:`s_{jk}` of any instance :math:`k`.

    This should reflect the ACTUAL spatial distance of your dataset for best results...These models tend to like XY
    embedding vectors more than Z. For anisotropic datasets, you should roughly provide the anisotropic correction
    factor of each voxel. For instance anisotropy of (1.0, 1.0, 5.0) means that the Z dimension is 5x larger than XY.


    Formally, the value at each position :math:`i\in[x,y,z]` of the baked skeleton tensor :math:`S` is the minimum of the
    euclidean distance function :math:`f(a, b)` and the skeleton point of any instance:

    .. math::
        S_{i} = min\left( f(i, s_{k}) \right) for k \in [1, 2, ..., N]

    Shapes:
        - masks: :math:`(1, X_{in}, Y_{in}, Z_{in})`
        - skeletons: :math:`(3, N_i)`
        - anisotropy: :math:`(3)`
        - returns: :math:`(3, X_{in}, Y_{in}, Z_{in})`

    :param masks: Ground Truth instance mask of shape [1, X, Y, Z] of objects where each pixel is an integer id value.
    :param skeletons: Dict of skeleton indicies where each key is a unique instance of an object in mask.
        - Each skeleton has a shape [3, N] where N is the number of pixels constituting the skeleton
    :param anisotropy: Anisotropic correction factor for min distance calculation
    :param average: Average the skeletons such that there is a smooth transition form one area to the next
    :param device: torch.Device by which to run calculations

    :return: Baked skeleton
    """

    baked = torch.zeros((3, masks.shape[1], masks.shape[2], masks.shape[3]), device=device)

    assert baked.device == masks.device, 'Masks device must equal kwarg device'

    unique = torch.unique(masks)

    anisotropy = torch.tensor(anisotropy, device=device).view(1, 1, 3)

    for id in unique[unique != 0]:
        nonzero: Tensor = masks[0, ...].eq(id).nonzero().unsqueeze(0).float()  # 1, N, 3
        skel: Tensor = skeletons[int(id)].to(device).unsqueeze(0).float()  # 1, N, 3

        # Calculate the distance between the skeleton of object 'id'
        # and all nonzero pixels of the binary mask of instance 'id'
        # print(skel.shape, nonzero.shape, id)
        ind: Tensor = torch.cdist(x1=skel.mul(anisotropy),
                                  x2=nonzero.mul(anisotropy)
                                  ).squeeze(0).argmin(dim=0)

        nonzero = nonzero.squeeze(0).long()  # Can only index with long dtype

        baked[:, nonzero[:, 0], nonzero[:, 1], nonzero[:, 2]] = skel[0, ind, :].float().T
        del ind

    baked = average_baked_skeletons(baked.unsqueeze(0)).squeeze(0) if average else baked
    baked[baked==0] = -100 # otherwise 0,0,0 is positive... weird...

    return baked


def skeleton_to_mask(skeletons: Dict[int, Tensor],
                     shape: Tuple[int, int, int],
                     kernel_size: Tuple[int, int, int] = (15,15,1),
                     n: int = 2) -> Tensor:
    """
    Converts a skeleton Dict to a skeleton mask which can simply be regressed against via Dice loss or whatever...

    Shapes:
        - skeletons: [N, 3]
        - shape :math:`(3)`
        - returns: math:`(1, X_{in}, Y_{in}, Z_{in})`

    :param skeletons: Dict of skeletons
    :param shape: Shape of final mask
    :return: Maks of embedded skeleton px
    """
    skeleton_mask = torch.zeros(shape)

    for k in skeletons:
        x = skeletons[k][:, 0].long()
        y = skeletons[k][:, 1].long()
        z = skeletons[k][:, 2].long()

        ind_x = torch.logical_and(x >= 0, x < shape[0])
        ind_y = torch.logical_and(y >= 0, y < shape[1])
        ind_z = torch.logical_and(z >= 0, z < shape[2])

        ind = (ind_x.float() + ind_y.float() + ind_z.float()) == 3  # Equivalent to a 3 way logical and

        skeleton_mask[x[ind], y[ind], z[ind]] = 1

    skeleton_mask = skeleton_mask.unsqueeze(0).unsqueeze(0)

    for _ in range(n):  # this might make things a bit better on the skeleton side of things...
        skeleton_mask = gauss_filter(binary_dilation(skeleton_mask.gt(0.5).float()), kernel_size, [0.8, 0.8, 0.8])

    return skeleton_mask.squeeze(0)


def index_skeleton_by_embed(skeleton: Tensor, embed: Tensor) -> Tensor:
    """
    Returns an instance mask by indexing skeleton with an embedding tensor
    For memory efficiency, skeleton is only ever Referenced! Never copied (I hope)

    Shapes:
        -skeleton: :math:`(B_{in}=1, 1, X_{in}, Y_{in}, Z_{in})`
        -embed: :math:`(B_{in}=1, 3, X_{in}, Y_{in}, Z_{in})`

    :param skeleton: Skeleton of a single instance
    :param embed: Embedding
    :return: torch.int instance mask
    """
    assert embed.device == skeleton.device, 'embed and skeleton must be on same device'
    # assert embed.shape[2::] == skeleton.shape[2::], 'embed and skeleton must have identical spatial dimensions'
    assert embed.ndim == 5 and skeleton.ndim == 5, 'Embed and skeleton must be a 5D tensor'

    b, c, x, y, z = embed.shape                     # get the shape of the embedding
    embed = embed.view((c, -1)).round()             # flatten the embedding to extract it as an index

    # We need to only select indicies which lie within the skeleton
    x_ind = embed[0, :].clamp(0, skeleton.shape[2]-1).long()
    y_ind = embed[1, :].clamp(0, skeleton.shape[3]-1).long()
    z_ind = embed[2, :].clamp(0, skeleton.shape[4]-1).long()

    out = torch.zeros((1, 1, x, y, z), device=embed.device, dtype=torch.int).flatten()   # For indexing to work, the out tensor
                                                                           # has to be flat

    ind = torch.arange(0, x_ind.shape[-1], device=embed.device)            # For each out pixel, we take the embedding at
                                                                           # that loc and assign it to skeleton

    out[ind] = skeleton[:, :, x_ind, y_ind, z_ind].int()  # assign the skeleton ind to the out tensor

    return out.view(1, 1, x, y, z)                  # return the re-shaped out tensor


if __name__ == '__main__':
    skeleton = {1: torch.tensor([[10, 10, 2], [20, 20, 4]])}
    mask = torch.ones((1, 50, 50, 20))

    baked = bake_skeleton(mask, skeleton)

    print(f'{baked[:,11,11,3]=}, {baked[:,21,21,5]=}')
