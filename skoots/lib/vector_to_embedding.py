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
    :param vector: [B, C=2, X, Y] the vector matrix predicted by the unet

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


@torch.jit.scrip
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

    mesh: Tuple[Tensor, ...] = torch.meshgrid(axis_ind, indexing='ij')
    mesh: List[Tensor] = [m.unsqueeze(0).unsqueeze(0) for m in mesh]
    mesh: Tensor = torch.cat(mesh, dim=1)

    vector = vector.mul(num.view(newshape))

    return mesh + vector

def vector_to_embedding(scale: Tensor, vector: Tensor) -> Tensor:
    """
    Converts a 2D or 3D vector field to a spatial embedding by adding the vector at any position to its own position.

    vector is a 2D or 3D vector field of shape :math:`(B, 2, X, Y)` for 2D or :math:`(B, 3, X, Y, Z)` for 3D.
    Each vector ":math:`v`" lies within the range -1 and 1 and is scaled by scale ":math:`s`". The scaled vector is then
    added to its own position to form a spatial embedding ":math:`\phi`":

    Formally:
        .. math::
            i,j,k \in \mathbb{Z}_{≥0} \n
            v_{i,j,k} \in [-1, 1] \n
            s = [s_i, s_j, s_k]

            \phi_{i,j,k} = v_{i,j,k} * s + [i, j, k]


    Shapes:
        - scale: :math:`(2)` or :math:`(3)`
        - vector: :math:`(B_{in}, 2, X_{in}, Y_{in})` or :math:`(B_{in}, 3, X_{in}, Y_{in}, Z_{in})`
        - Returns: :math:`(B_{in}, 2, X_{in}, Y_{in})` or :math:`(B_{in}, 3, X_{in}, Y_{in}, Z_{in})`


    :param scale: Scaling factors for each vector spatial dimension
    :param vector: Vector field predicted by a neural network

    :return: Pixel spatial embeddings
    """

    # assert vector.ndim in [4, 5], f'Vector must be a 4D or 5D tensor, not {vector.ndim}D: {vector.shape=}'
    return _vec2embed3D(scale, vector) if vector.ndim == 5 else _vec2embed2D(scale, vector)


def vec2embedND(scale, vector):
    """
    Generic N dimmensional vector to embedding

    Could be a faster way to do this with strides but idk...

    :param scale: [N=2/3]
    :param vector: [B, C, X, Y, Z?]
    :return:
    """
    assert scale.shape[0] == vector.shape[1], \
        f'Cannot use {scale.shape[0]}D scale with vector shape: {vector.shape}'
    assert scale.shape[0] == vector.ndim - 2, \
        f'Cannot use {scale.shape[0]}D scale with {vector.ndim - 2}D vector shape [B, C, ...]: {vector.shape}'

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
