import torch
from torch import Tensor
import torch.nn.functional as F

from typing import List, Tuple, Dict

from skoots.lib.morphology import _compute_zero_padding, _get_binary_kernel3d


@torch.jit.script
def average_baked_skeletons(input: Tensor) -> Tensor:
    """
    Takes a baked skeleton computed by skoots.lib.skeleton.bake_skeleton and averages all the values such that
    there is a smooth transition from one location to another.

    :param input: [1, 3, X, Y, Z]
    :return: smoothed skeleton [1, 3, X, Y, Z]
    """
    padding: Tuple[int, int, int] = _compute_zero_padding((3, 3, 3))
    kernel: Tensor = _get_binary_kernel3d(3, str(input.device))
    b, c, h, w, d = input.shape

    # map the local window to single vector
    features: Tensor = F.conv3d(input.reshape(b * c, 1, h, w, d), kernel,
                                padding=padding, stride=1)
    features: Tensor = features.view(b, c, -1, h, w, d)  # B, C, -1, X, Y, Z

    # print(features.shape)

    nonzero = features.gt(0).sum(2)  # B, C, X, Y, Z
    nonzero[nonzero.eq(0)] = 1.
    features = features.sum(2)  # This is the average of the kernel window... without zeros
    features = features / nonzero

    return features


@torch.jit.script
def bake_skeleton(masks: Tensor, skeletons: Dict[int, Tensor],
                  anisotropy: List[float] = (1., 1., 3.),
                  device: str = 'cpu') -> Tensor:
    """
    For each pixel (p) index {x,y,z} in masks with shape [1, X, Y, Z], returns a new tensor [3, X, Y, Z] where
    the value at each {X, Y, Z} is the closest skeleton point (s) of any instance.

    .. math::
        p_{xyz} = min(\ \Phi(s_{xyz},  \{x, y, z\})\ )\ for\ x,y,z \in\ mask.shape

    where

    .. math::
        \Phi(a,\ b)
    is the euclidean distance function.


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
        if id == 0: continue

        nonzero: Tensor = masks[0, ...].eq(id).nonzero().unsqueeze(0).float().mul(anisotropy)  # 1, N, 3
        skel: Tensor = skeletons[int(id)].unsqueeze(0).float().mul(anisotropy)  # 1, N, 3

        ind = torch.cdist(skel, nonzero).squeeze(0).argmin(dim=0)

        baked[:, nonzero[:, 0], nonzero[:, 1], nonzero[:, 2]] = skel[ind, :].float().T

    return average_baked_skeletons(baked.unsqueeze(0)).squeeze(0)  # requires batching...
