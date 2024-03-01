import torch
from torch import Tensor
import logging

from skoots.lib.morphology import binary_erosion
from typing import List


def erode_bg_masks(background: Tensor, n_erode: int, log: bool = True) -> Tensor:
    """
    erodes a background mask

    :param background: 3D binary tensor [X, Y, Z]
    :param n_erode: num times to apply erosion

    :return: eroded masks [X, Y, Z]
    """
    if log:
        logging.info(f"eroding background masks with {n_erode=}")

    if n_erode == 0:
        return background

    assert background.ndim == 4, f"background ndim != 3, {background.shape=}"
    assert background.max() <= 1, f"background max != 1 {background.max()=}"
    assert background.dtype == torch.uint8, f"background dtype: {background.dtype}"

    background = background.unsqueeze(0).float()
    device = background.device

    background.to('cpu')
    for _ in range(int(n_erode)):
        background = binary_erosion(background)

    return background.squeeze(0).to(torch.uint8).to(device)


def ablate_bg_masks(background, alpha: float, log: bool = True):
    """
    zeros out background mask slices until round(Z*alpha) slices are left


    :param background: background masks
    :param alpha: float between 0 and 1 which determines the ablation percentage.
    :return: ablated background mask
    """

    assert background.ndim == 4, f"background ndim != 3, {background.shape=}"
    assert background.max() <= 1, f"background max != 1 {background.max()=}"
    assert background.dtype == torch.uint8, f"background dtype: {background.dtype}"

    assert alpha > 0, "ablating with alpha=0 removes all background masks"
    assert alpha <= 1, f"alpha must be 0<alpha<=1, not {alpha=}"

    if log:
        logging.info(f"ablating background masks with {alpha=}")
    c, x, y, z = background.shape

    for i in reversed(range(z)):
        if i + 1 > (z * alpha):
            background[:, :, :, i].mul_(0)

    return background
