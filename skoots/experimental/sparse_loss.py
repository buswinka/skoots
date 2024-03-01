import logging
from typing import Tuple, Dict

import torch
from skoots.lib.embedding_to_prob import baked_embed_to_prob
from skoots.lib.skeleton import bake_skeleton
from skoots.train.loss import _dice, tversky
from torch import Tensor
import torch.nn.functional as F
from skoots.lib.morphology import (
    gauss_filter,
    binary_dilation_2d,
    _compute_zero_padding,
    _get_binary_kernel3d,
)
from yacs.config import CfgNode


def vector_direction_penalty(vectors: Tensor, kernel_size: int = 3) -> Tensor:
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
    padding: Tuple[int, int, int] = _compute_zero_padding(
        (kernel_size, kernel_size, kernel_size)
    )
    kernel: Tensor = _get_binary_kernel3d(kernel_size, str(vectors.device))
    b, c, h, w, d = vectors.shape

    # map the local window to single vector
    features: Tensor = F.conv3d(
        vectors.reshape(b * c, 1, h, w, d), kernel, padding=padding, stride=1
    )
    # middle is index 9 + 4

    features: Tensor = features.view(b, c, -1, h, w, d)  # B, C, KS**2, X, Y, Z

    center = features[:, :, [13], ...]

    dot = torch.sum(features * center, dim=1, keepdim=True)  # B, c=1, 27, ...
    magnitude = torch.sqrt(features.pow(2).sum(1, keepdim=True).add(1e-8))
    sin_of = dot.div((magnitude * magnitude[:, :, [13], ...]).add(1e-8))
    sin_of = sin_of.pow(2)
    sin_of = sin_of.mul(-1).add(1.000001)

    sin_of[
        :, :, [13], ...
    ] = 0  # zero out the middle because we only want to average everything else

    ind = magnitude > 1e-8  # B, 1, 27, X, Y, Z

    nonzero = ind.sum(2)  # B, C, X, Y, Z
    nonzero[nonzero.eq(0)] = 1.0
    sin_of[torch.logical_not(ind)] = 0
    sin_of = sin_of.sum(2) / nonzero

    return sin_of


@torch.no_grad()
def closest_skeleton(
    shape: Tuple[int, int, int, int, int],
    skeletons: Dict[int, Tensor],
    anisotropy: Tuple[float, float, float],
) -> Tuple[Tensor, Tensor]:
    """
    Calculates the distance of each pixel index from the nearest skeleton

    :param shape:
    :param skeletons:
    :return:
    """

    # Concat all skeletons to one massive tensor
    ninst = len(skeletons)
    skeletons = torch.concat(tuple(v for v in skeletons.values()), dim=0)  # N, 3

    DEVICE = skeletons.device

    x, y, z = shape[2::]
    indx = torch.logical_and(skeletons[:, 0].lt(x + 50), skeletons[:, 0].gt(-50))
    indy = torch.logical_and(skeletons[:, 1].lt(y + 50), skeletons[:, 1].gt(-50))
    indz = torch.logical_and(skeletons[:, 2].lt(z + 10), skeletons[:, 2].gt(-10))

    ind = torch.logical_and(torch.logical_and(indx, indy), indz)
    skeletons = skeletons[ind, :]

    if skeletons.numel() > 0:
        logging.debug(
            f"merged skeletons have shape: {skeletons.shape} from {ninst} instances"
        )

        # bake skeletons using triton kernel. May be too slow...
        baked, distance = bake_skeleton(
            masks=torch.ones(
                (shape[2], shape[3], shape[4]), device=DEVICE, dtype=torch.float32
            ),
            skeletons={1: skeletons.float()},
            return_distance=True,
            average=True,
            device=DEVICE,
            anisotropy=anisotropy,
        )
    else:
        distance = torch.empty(
            (1, shape[2], shape[3], shape[4]), dtype=torch.float32, device=DEVICE
        )
        distance.fill_(100.00)

        baked = torch.ones(
            (3, shape[2], shape[3], shape[4]), dtype=torch.float32, device=DEVICE
        )
        baked.fill_(1000.00)

    return baked, distance


def embed_distance(embed: Tensor, baked_skeleton: Tensor) -> Tensor:
    """
    Takes an embedding, and baked_skeleton tensor and returns the distance between the two

    Shapes:
        - embed: :math:`(B, C=3, X, Y, Z)'
        - baked_skeleton: :math:`(B, C=3, X, Y, Z)'
        - returns: :math:`(B, C=1, X, Y, Z)'

    :param embed: the locations of each pixel in embedding space
    :param baked_skeleton: the closest skeleton index to each voxel index
    :return: distance between embedding and baked_skeleton
    """

    dist = (embed - baked_skeleton).pow(2).sum(0, keepdim=True).sqrt()

    return dist


def sparse_background_loss(
    embed_loss: Tensor, background: Tensor, multiplier: 10
) -> Tensor:
    """
    Returns the embedding loss indexed by the background. All values of embed_loss where background == 1 should be zero.
    Background voxels with value 0 **may** be background, but is not considered here.

    Shapes:
        - embed_loss: :math:`(B, C=1, X, Y, Z)'
        - background: :math:`(B, C=1, X, Y, Z)'
        - returns: :math:`(B, C=1, X, Y, Z)'


    :param embed_loss: loss value from each voxel calculated by skoots.lib.embedding_to_prob.baked_embedding_to_prob()
    :param background: manually labeled background tensor. value of 1 is assured to be background, 0 may be background.
    :return: mean embed_loss indexed by background.
    """
    # we want this val to always be zero, that's why we don't subtract anything...
    # this is just mean squared error.

    index = background.gt(0.5)

    if index.max() == 0: # nothing so return 0.0
        return 0.0

    error = embed_loss[background.gt(0.5)].pow(2).mean()

    return error * multiplier


def sparse_embed_loss(
    embed_loss: Tensor,
    skeleton_distance: Tensor,
    background: Tensor,
    distance_thr: float,
) -> Tensor:
    """
    This loss calculates the embedding loss of all voxels that are a certain distance away from distance_thr. Only
    voxels with a background value of 0 may be considerd. Background values of 1 are assured to be background, values of
    zero may be background.

    Shapes:
        - embed_loss: :math:`(B, C=1, X, Y, Z)'
        - skeleton_distance: :math:`(B, C=1, X, Y, Z)'
        - background: :math:`(B, C=1, X, Y, Z)'
        - returns: :math:`(B, C=1, X, Y, Z)'


    :param embed_loss: Embedding loss from skoots.lib.embedding_to_prob.baked_embedding_to_prob()
    :param skeleton_distance: distance of each voxel from its nearest skeleton
    :param distance_thr: maximum distance for which we consider loss...
    :return: Dice loss of embedding from distance thresholded by distance_thr
    """

    distance_mask = skeleton_distance.lt(distance_thr)

    # zero out background because we want that to always be zero
    logging.debug(f"{distance_mask.shape=}, {background.shape=}")
    distance_mask[background.gt(0.5)] = 0

    # here we want each embeding loss to be one. This is just mean squared error.
    if distance_mask.max() > 0:
        loss = (1 - embed_loss[distance_mask]).pow(2).mean()
    else:
        index = skeleton_distance.argmin()
        loss = (1 - embed_loss.flatten()[index]).pow(2).mean()

    return loss


def vec_adjacency_penalty_loss(vectors: Tensor) -> Tensor:
    """
    This should penalize adjacent vectors for being radically different from each other...


    uses the sign of the two vectors to condition against a ms

    :param vectors:
    :return:
    """

    raise NotImplementedError


def _tversky(pred, gt, alpha, beta, eps):
    """
    alpha penalizes fp,

    beta penalizes fn

    alpha == beta == 1 is dice loss

    :param pred:
    :param gt:
    :param alpha:
    :param beta:
    :param eps:
    :return:
    """
    true_positive: Tensor = pred.mul(gt).sum()
    false_positive: Tensor = torch.logical_not(gt).mul(pred).sum().add(1e-10).mul(alpha)
    false_negative: Tensor = (torch.logical_not(pred) * gt).sum() * beta
    loss = (true_positive + eps) / (
        true_positive + false_positive + false_negative + eps
    )

    return 1 - loss


def sparse_loss(
    embed: Tensor,
    vectors: Tensor,
    skeletons: Tensor,
    background: Tensor,
    semantic_mask: Tensor,
    sigma: Tensor,
    anisotropy: Tuple[float, float, float],
    distance_thr: float,
    cfg: CfgNode,
) -> Tuple[Tensor, Tensor, Tensor]:
    """

    :param embed:
    :param skeletons:
    :param background:
    :param sigma:
    :param anisotropy:
    :param distance_thr:
    :return: background_loss, embed_loss
    """

    b, c, x, y, z = background.shape

    background_loss = torch.zeros((b), device=background.device)
    embed_loss = torch.zeros((b), device=background.device)

    embed_prob = torch.zeros_like(background, dtype=torch.float)

    logging.debug(f"{len(skeletons)=}, {background.shape=}")

    for _b in range(b):
        baked, distance = closest_skeleton(
            shape=(1, c, x, y, z), skeletons=skeletons[_b], anisotropy=anisotropy
        )

        embed_prob[[_b], ...] = baked_embed_to_prob(
            embedding=embed[[_b], ...], baked_skeletons=baked, sigma=sigma
        )

        logging.debug(
            f"attempting batch {_b + 1}/{b} -|- {baked.shape=}, {distance.shape=}, {embed_prob.shape=}"
        )
        logging.debug(f"->{embed_prob.max()=}, {embed_prob.min()=}, {sigma=}")
        logging.debug(f"->{baked.max()=}, {baked.min()=}")
        logging.debug(f"->{distance.max()=}, {distance.min()=}")

        a = sparse_background_loss(
            embed_loss=embed_prob[_b, ...],
            background=background[_b, ...],
            multiplier=cfg.EXPERIMENTAL.SPARSE_BACKGROUND_PENALTY_MULTIPLIER,
        )
        b = sparse_embed_loss(
            embed_prob[_b, ...],
            embed_distance(embed[_b, ...], baked),
            background[_b, ...],
            distance_thr=distance_thr,
        )

        e = sparse_embed_loss(
            embed_prob[_b, ...],
            distance,
            background[_b, ...],
            distance_thr=distance_thr,
        )

        f = vector_direction_penalty(vectors=vectors).mean()

        # assert not torch.any(torch.isnan(a))
        # assert not torch.any(torch.isinf(a))
        #
        # assert not torch.any(torch.isnan(b))
        # assert not torch.any(torch.isinf(b))
        #
        # assert not torch.any(torch.isnan(e))
        # assert not torch.any(torch.isinf(e))
        #
        # assert not torch.any(torch.isnan(f))
        # assert not torch.any(torch.isinf(f))
        # assert not torch.isnan(f), f'{f=}'
        # print(a, b, e, f)

        embed_loss[_b] = a + b + e + f

    # mse of embed+background
    # index = background < 0.5

    # c = sparse_background_loss(semantic_mask, background, multiplier=1)
    # d = (embed_prob[index] - semantic_mask[index].gt(0.2).float()).pow(2).mean()
    background_loss = _dice(embed_prob.gt(0.2), semantic_mask, 1e-8)  # 3.0, 1.0,1e-8)
    # background_loss = ((1 - background[index]) - semantic_mask[index]).pow(2).mean() + (
    #         embed_prob[not_index] - (pred_background[not_index])).pow(2).mean()

    # background_loss = ( 2/10 * c) + (8/10 * d) # ratios to account for class imbalance..

    # assert not torch.any(torch.isnan(background_loss))
    # assert not torch.any(torch.isinf(background_loss))

    # div by 2 to make loss scale between 0 and 1
    return background_loss, embed_loss.mean() / 2, embed_prob


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    testin = torch.randn((1, 3, 10, 10, 10))
    out = vector_direction_penalty(testin)

    print(out.shape)

    plt.imshow(out[0, 0, :, :, 5])
    plt.show()
