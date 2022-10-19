import torch
from torch import Tensor
import torch.nn.functional as F

from typing import List, Tuple, Dict, Optional 

from skoots.lib.morphology import _compute_zero_padding, _get_binary_kernel3d



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


@torch.jit.script
def bake_skeleton(masks: Tensor, skeletons: Dict[int, Tensor],
                  anisotropy: List[float] = (1., 1., 1.),
                  average: bool = True,
                  device: str = 'cpu') -> Tensor:
    """
    For each pixel :math:`p_ik` of object :math:`k` at index :math:`i\in[x,y,z]` in masks, returns a baked_skeleton tensor where
    the value at each index is the closest skeleton point :math:`s_{jk}` of any instance :math:`k`.

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
    :param device: torch.Device by which to run calculations

    :return: Baked skeleton
    """

    baked = torch.zeros((3, masks.shape[1], masks.shape[2], masks.shape[3]), device=device)

    unique = torch.unique(masks)

    anisotropy = torch.tensor(anisotropy, device=device).view(1, 1, 3)

    for id in unique[unique != 0]:

        nonzero: Tensor = masks[0, ...].eq(id).nonzero().unsqueeze(0).float().mul(anisotropy)  # 1, N, 3
        skel: Tensor = skeletons[int(id)].to(device).unsqueeze(0).float().mul(anisotropy)  # 1, N, 3

        # Calculate the distance between the skeleton of object 'id'
        # and all nonzero pixels of the binary mask of instance 'id'
        ind: Tensor = torch.cdist(skel, nonzero).squeeze(0).argmin(dim=0)

        nonzero = nonzero.squeeze(0).long()  # Can only index with long dtype

        baked[:, nonzero[:, 0], nonzero[:, 1], nonzero[:, 2]] = skel[0, ind, :].float().T
    
    baked = average_baked_skeletons(baked.unsqueeze(0)).squeeze(0) if average else baked

    return baked


if __name__ == '__main__':
    
    skeleton = {1: torch.tensor([[10,10,2], [20,20,4]])}
    mask = torch.ones((1,50,50,20))

    baked = bake_skeleton(mask, skeleton)

    print(f'{baked[:,11,11,3]=}, {baked[:,21,21,5]=}')
