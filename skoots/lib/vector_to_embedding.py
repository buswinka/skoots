from typing import List
from typing import Tuple

import torch
from torch import Tensor

"""
2D and 3D implementations of vector to embedding. 

vector is predicted by a network and is an array of values ranging from -1 to 1. These values can be scaled by
"scale" and are used to project pixels from their location, to a new location in embedding space. 

Functionally, a trained network will push pixels to the center of an object, forming clusers which can be used to infer
object locations. 

"""


@torch.jit.script
def get_vector_mesh(shape: Tuple[int, int, int, int, int], device: str) -> Tensor:
    """generates a 3d mesh from a vector"""
    axis_ind: List[Tensor] = [
        torch.linspace(0, shape[2] - 1, shape[2], device=device),
        torch.linspace(0, shape[3] - 1, shape[3], device=device),
        torch.linspace(0, shape[4] - 1, shape[4], device=device),
    ]

    mesh: List[Tensor] = torch.meshgrid(axis_ind, indexing="ij")
    mesh: List[Tensor] = [m.unsqueeze(0).unsqueeze(0) for m in mesh]
    mesh: Tensor = torch.cat(mesh, dim=1)

    return mesh


@torch.jit.script
def _vec2embed3D_graphable(static_scale: Tensor, vector: Tensor, static_mesh) -> Tensor:
    """
    3D vector to embedding which uses static inputs for cuda graphs

    Could be a faster way to do this with strides but idk...

    :param scale: Tensor with shape (3)
    :param vector: [B, C, X, Y, Z]
    :return: embedding vector
    """

    return static_mesh + vector.mul(static_scale.view((1, 3, 1, 1, 1)))


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
        torch.linspace(0, vector.shape[3] - 1, vector.shape[3], device=vector.device),
    ]

    mesh = torch.meshgrid(axis_ind, indexing="ij")
    mesh: List[Tensor] = [m.unsqueeze(0).unsqueeze(0) for m in mesh]
    mesh: Tensor = torch.cat(mesh, dim=1)

    vector = vector.mul(num.view(newshape))

    return mesh + vector


@torch.jit.script
def _vec2embed3D(scale: Tensor, vector: Tensor, n: int = 1) -> Tensor:
    """
    2D or 3D vector to embedding
    Could be a faster way to do this with strides but idk...
    :param scale: [N=2/3]
    :param vector: [B, C, X, Y, Z?]
    :param n: number of times to apply vectors
    :return:
    """

    num: Tensor = torch.clone(scale.float())

    newshape: Tuple[int, int, int, int, int] = (1, 3, 1, 1, 1)

    axis_ind: List[Tensor] = [
        torch.linspace(0, vector.shape[2] - 1, vector.shape[2], device=vector.device),
        torch.linspace(0, vector.shape[3] - 1, vector.shape[3], device=vector.device),
        torch.linspace(0, vector.shape[4] - 1, vector.shape[4], device=vector.device),
    ]

    mesh: List[Tensor] = torch.meshgrid(axis_ind, indexing="ij")
    mesh: List[Tensor] = [m.unsqueeze(0).unsqueeze(0) for m in mesh]
    mesh: Tensor = torch.cat(mesh, dim=1)

    scaled_vector = vector.mul(num.view(newshape))
    mesh = mesh + scaled_vector

    scale = 1.0
    for _ in range(n - 1):  # Only executes if n > 1
        # convert to index.

        scale *= 1.0

        scaled_vector = vector.mul(scale * num.view(newshape))

        index = mesh.round()
        b, c, x, y, z = index.shape
        for i, k in enumerate([x, y, z]):
            index[:, i, ...] = torch.clamp(index[:, i, ...], 0, k)

        # 3d index to raveled
        index = (
            (index[:, [0], ...] * y * z)
            + (index[:, [1], ...] * z)
            + (index[:, [2], ...])
        )
        index = index.clamp(0, x * y * z - 1).long()

        for i in range(c):
            mesh[:, [i], ...] = mesh[:, [i], ...] + scaled_vector[:, [i], ...].take(index)

    return mesh


def vector_to_embedding(scale: Tensor, vector: Tensor, N: int = 1) -> Tensor:
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
    return (
        _vec2embed3D(scale, vector, N)
        if vector.ndim == 5
        else _vec2embed2D(scale, vector)
    )


def vec2embedND(scale, vector):
    """
    Generic N dimmensional vector to embedding

    Could be a faster way to do this with strides but idk...

    :param scale: [N=2/3]
    :param vector: [B, C, X, Y, Z?]
    :return:
    """
    assert (
        scale.shape[0] == vector.shape[1]
    ), f"Cannot use {scale.shape[0]}D scale with vector shape: {vector.shape}"
    assert (
        scale.shape[0] == vector.ndim - 2
    ), f"Cannot use {scale.shape[0]}D scale with {vector.ndim - 2}D vector shape [B, C, ...]: {vector.shape}"

    num: Tensor = torch.clone(scale.float())

    newshape: Tuple[int] = tuple(
        [1, scale.shape[0]]
        + [
            1,
        ]
        * (vector.ndim - 2)
    )

    axis_ind: List[Tensor] = []
    for i in range(vector.ndim - 2):
        axis_ind.append(
            torch.linspace(
                0, vector.shape[2 + i] - 1, vector.shape[2 + i], device=vector.device
            )
        )

    mesh = torch.meshgrid(axis_ind, indexing="ij")
    mesh = [m.unsqueeze(0).unsqueeze(0) for m in mesh]
    mesh = torch.cat(mesh, dim=1)

    vector = vector.mul(num.view(newshape))

    return mesh + vector


if __name__ == "__main__":
    vector = torch.ones((1, 3, 10, 10, 10)).float()
    vector[:, 0, 5, 5, 5] = -1
    vector[:, 1, 5, 5, 5] = -1
    vector[:, 2, 5, 5, 5] = -1

    vector[:, [0,1,2], 4, 4, 4] = torch.tensor((2, 2, 2)).float()


    out = vector_to_embedding(torch.tensor((1, 1, 1)), vector, N=2)

    print(f"{out[0, :, 5, 5, 5]=}") # should equal 6 6 6
