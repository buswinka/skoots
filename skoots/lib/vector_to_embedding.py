from typing import List, Tuple
import torch
from torch import Tensor

from typing import Tuple

"""
2D and 3D implementations of vector to embedding. 

vector is predicted by a network and is an array of values ranging from -1 to 1. These values can be scaled by
"scale" and are used to project pixels from their location, to a new location in embedding space. 

Functionally, a trained network will push pixels to the center of an object, forming clusers which can be used to infer
object locations. 

"""


@torch.jit.script
def _vec2embed2D(scale: Tensor, vector: Tensor) -> Tensor:
    """
    2D vector to embedding

    :param scale: The offest in XY of the vectors.
    :param vector: [B, C, X, Y] the vector matrix predicted by the unet

    :return: Pixel Spatial Embeddings (i.e. vector + pixel_indicies)
    """

    num: Tensor = torch.clone(scale.float())

    newshape: Tuple[int, int, int, int] = (1, 2, 1, 1)

    axis_ind: List[Tensor] = [
        torch.linspace(0, vector.shape[2] - 1, vector.shape[2], device=vector.device),
        torch.linspace(0, vector.shape[3] - 1, vector.shape[3], device=vector.device)
    ]

    mesh = torch.meshgrid(axis_ind, indexing='ij')
    mesh: List[Tensor] = [m.unsqueeze(0).unsqueeze(0) for m in mesh]
    mesh: Tensor = torch.cat(mesh, dim=1)

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
        torch.linspace(0, vector.shape[2] - 1, vector.shape[2], device=vector.device),
        torch.linspace(0, vector.shape[3] - 1, vector.shape[3], device=vector.device),
        torch.linspace(0, vector.shape[4] - 1, vector.shape[4], device=vector.device)
    ]

    mesh = torch.meshgrid(axis_ind, indexing='ij')
    mesh: List[Tensor] = [m.unsqueeze(0).unsqueeze(0) for m in mesh]
    mesh: Tensor = torch.cat(mesh, dim=1)

    vector = vector.mul(num.view(newshape))

    return mesh + vector


@torch.jit.ignore
def vector_to_embedding(scale: Tensor, vector: Tensor) -> Tensor:
    """
    2D or 3D vector to embedding procedure.

    :param scale: Vector scaling factors
    :param vector: Vector field predicted by a neural network with shape [B, C=(2 or 3), X, Y, Z?]

    :return: Pixel spatial embeddings of the same shape as vector
    """
    assert vector.ndim in [4, 5], f'Vector must be a 4D or 5D tensor, not {vector.ndim}D: {vector.shape=}'
    return _vec2embed3D(scale, vector) if vector.ndim == 5 else _vec2embed2D(scale, vector)


def vec2embedND(scale, vector):
    """
    Generic N dimmensional vector to embedding

    Could be a faster way to do this with strides but idk...

    :param scale: [N=2/3]
    :param vector: [B, C, X, Y, Z?]
    :return:
    """
    assert scale.shape[0] == vector.shape[1], f'Cannot use {scale.shape[0]}D scale with vector shape: {vector.shape}'
    assert scale.shape[
               0] == vector.ndim - 2, f'Cannot use {scale.shape[0]}D scale with {vector.ndim - 2}D vector shape [B, C, ...]: {vector.shape}'

    num: Tensor = torch.clone(scale.float())

    newshape: Tuple[int] = tuple([1, scale.shape[0]] + [1, ] * (vector.ndim - 2))

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
