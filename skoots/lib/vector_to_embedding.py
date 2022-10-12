import torch
from torch import Tensor
from skoots.lib.utils import _crop3d as _crop
import torch.nn as nn

from typing import Tuple, Dict, Optional, List
import torch
from torch import Tensor
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple

import triton
import triton.language as tl

import matplotlib.pyplot as plt

@torch.jit.script
def _vec2embed2D(scale: Tensor, vector: Tensor) -> Tensor:
    """
    2D or 3D vector to embedding

    Could be a faster way to do this with strides but idk...

    :param scale: [N=2/3]
    :param vector: [B, C, X, Y, Z?]
    :return:
    """

    num: Tensor = torch.clone(scale.float())

    newshape: Tuple[int, int, int, int] = (1, 2, 1, 1)

    axis_ind: List[Tensor] = [
        torch.linspace(0, vector.shape[3] - 1, vector.shape[3], device=vector.device),
        torch.linspace(0, vector.shape[4] - 1, vector.shape[4], device=vector.device)
    ]

    mesh = torch.meshgrid(axis_ind, indexing='ij')
    mesh = [m.unsqueeze(0).unsqueeze(0) for m in mesh]
    mesh = torch.cat(mesh, dim=1)

    vector = vector.mul(num.view(newshape))

    return mesh + vector

@torch.jit.script
def _vec2embed3D(scale: Tensor, vector: Tensor) -> Tensor:
    """
    2D or 3D vector to embedding

    Could be a faster way to do this with strides but idk...

    :param scale: [N=2/3]
    :param vector: [B, C, X, Y, Z?]
    :return:
    """

    num: Tensor = torch.clone(scale.float())

    newshape: Tuple[int, int, int, int, int] = (1, 3, 1, 1, 1)

    axis_ind: List[Tensor] = [
        torch.linspace(0, vector.shape[3] - 1, vector.shape[3], device=vector.device),
        torch.linspace(0, vector.shape[4] - 1, vector.shape[4], device=vector.device),
        torch.linspace(0, vector.shape[5] - 1, vector.shape[5], device=vector.device)
    ]

    mesh = torch.meshgrid(axis_ind, indexing='ij')
    mesh = [m.unsqueeze(0).unsqueeze(0) for m in mesh]
    mesh = torch.cat(mesh, dim=1)

    vector = vector.mul(num.view(newshape))

    return mesh + vector


@torch.jit.ignore
def vector_to_embedding(scale: Tensor, vector: Tensor):
    return _vec2embed3D(scale, vector) if vector.ndim==5 else _vec2embed2D(scale, vector)


def vec2embedND(scale, vector):
    """
    2D or 3D vector to embedding

    Could be a faster way to do this with strides but idk...

    :param scale: [N=2/3]
    :param vector: [B, C, X, Y, Z?]
    :return:
    """
    assert scale.shape[0] == vector.shape[1], f'Cannot use {scale.shape[0]}D scale with vector shape: {vector.shape}'
    assert scale.shape[0] == vector.ndim-2, f'Cannot use {scale.shape[0]}D scale with {vector.ndim - 2}D vector shape [B, C, ...]: {vector.shape}'

    num: Tensor = torch.clone(scale.float())

    newshape: Tuple[int] = tuple([1, scale.shape[0]] + [1,] * (vector.ndim - 2))

    axis_ind: List[Tensor] = []
    for i in range(vector.ndim - 2):
        axis_ind.append(
            torch.linspace(0, vector.shape[2 + i] - 1, vector.shape[2 + i], device=vector.device)
        )

    mesh = torch.meshgrid(axis_ind, indexing='ij')
    mesh = [m.unsqueeze(0).unsqueeze(0) for m in mesh]
    mesh = torch.cat(mesh, dim=1)

    vector = vector.mul(num.view(newshape))

    return mesh + vector


