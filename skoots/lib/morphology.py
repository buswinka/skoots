import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, Tuple, Union, Sequence, List

@torch.jit.script
def _compute_zero_padding(kernel_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r"""Utility function that computes zero padding tuple.
    Adapted from Kornia
    """
    computed: List[int] = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1], computed[2]

@torch.jit.script
def _get_binary_kernel3d(window_size: int, device: str) -> Tensor:
    r"""Creates a symetric binary kernel to extract the patches. If the window size
    is HxWxD will create a (H*W)xHxW kernel.

    Adapted from Kornia

    """
    window_range: int = int(window_size ** 3)
    kernel: Tensor = torch.zeros((window_range, window_range, window_range), device=device)
    for i in range(window_range):
        kernel[i, i, i] += 1.0
    kernel = kernel.view(-1, 1, window_size, window_size, window_size)

    # get rid of all zero kernels
    ind = torch.nonzero(kernel.view(kernel.shape[0], -1).sum(1))
    return kernel[ind[:, 0], ...]


@torch.jit.script
def binary_erosion(image: Tensor) -> Tensor:
    """
    Performs binary erosion on a 5D Tensor.

    :param image: torch.Tensor[B, C, X, Y, Z]
    :return: eroded image
    """
    device = str(image.device)
    kernel = _get_binary_kernel3d(3, device)
    padding = _compute_zero_padding((3, 3, 3))

    b, c, h, w, d = image.shape
    # map the local window to single vector
    features: Tensor = F.conv3d(image.reshape(b * c, 1, h, w, d), kernel, padding=padding, stride=1)
    return features.min(dim=1)[0].unsqueeze(0)


@torch.jit.script
def binary_dilation(image: Tensor) -> Tensor:
    """
    Performs binary dilation on a 5D Tensor.

    :param image: torch.Tensor[B, C, X, Y, Z]
    :return: dilated image
    """
    padding: Tuple[int, int, int] = _compute_zero_padding((3, 3, 3))
    kernel: Tensor = _get_binary_kernel3d(3, str(image.device))

    b, c, h, w, d = image.shape
    # map the local window to single vector
    features = F.conv3d(image.reshape(b * c, 1, h, w, d), kernel,
                        padding=padding, stride=1)
    return torch.max(features.view(b, c, -1, h, w, d), dim=2)[0]