from typing import Tuple, Dict, List

import numpy as np
import torch
from numba import njit, prange
from scipy.ndimage import label
from torch import Tensor
from tqdm import tqdm

from skoots.lib.cropper import crops, get_total_num_crops


def efficient_flood_fill(skeleton: Tensor) -> Tensor:
    """
    Efficiently floods a binary skeleton mask in place by first flood filling small regions,
    then merging connected components later. Avoids memory copies when possible. Returns a skeleton mask where
    each connected component has a unique label, however these labels may not be sequential.
    I.e. unique(skeleton) -> [4, 16, 23, 24, 96]

    :param skeleton: binary skeleton mask to flood fill
    :return: Flood filled tensor
    """
    skeleton = skeleton.unsqueeze(0) if skeleton.ndim == 3 else skeleton
    total = get_total_num_crops(
        skeleton.shape, crop_size=[1000, 1000, 100], overlap=[0, 0, 0]
    )
    iterator = tqdm(
        crops(skeleton, crop_size=[1000, 1000, 200]), desc="Flood-filling small crops: "
    )
    max_id = 1
    skeletons_dict = {}

    seams_x = []  # this will be all the seams of the crops we need to check later.
    seams_y = []
    seams_z = []

    # Iterate over crops of the skeleton to perform a flood fill
    for crop, (x, y, z) in iterator:
        seams_x = seams_x + [x] if x not in seams_x else seams_x
        seams_y = seams_y + [y] if y not in seams_y else seams_y
        seams_z = seams_z + [z] if z not in seams_z else seams_z

        # crop should only contain ones and zeros and be an int. Not guaranteed by skoots.lib.cropper.crops
        crop = crop.squeeze().gt(0).int()

        crop, max_id = flood_all(
            crop, max_id + 1
        )  # max_id is the previous max - update by 1!
        w, h, d = crop.shape
        skeleton[0, x : x + w, y : y + h, z : z + d] = crop

    # We now check each crop seam for double labeled instances. Here, a collision is the same object having label (x)
    # in one crop, then label (y) in another.  We must check in all dims, x, y, and z.

    # print(f'[      ] Detecting collisions...', end='')
    # X
    collisions: List[Tuple[int, int]] = []
    for x in tqdm(seams_x, desc="Checking for duplicate IDs in X", total=len(seams_x)):
        if x > 0:
            slice_0 = skeleton[0, x, :, :]
            slice_1 = skeleton[0, x - 1, :, :]
            collisions.extend(get_adjacent_labels(slice_0, slice_1))

    # Y
    for y in tqdm(seams_y, desc="Checking for duplicate IDs in Y", total=len(seams_y)):
        if y > 0:
            slice_0 = skeleton[0, :, y, :]
            slice_1 = skeleton[0, :, y - 1, :]
            collisions.extend(get_adjacent_labels(slice_0, slice_1))

    # Z
    for z in tqdm(seams_z, desc="Checking for duplicate IDs in Z", total=len(seams_z)):
        if z > 0:
            slice_0 = skeleton[0, :, :, z]
            slice_1 = skeleton[0, :, :, z - 1]
            collisions.extend(get_adjacent_labels(slice_0, slice_1))

    # print("\r[\x1b[1;32;40m DONE \x1b[0m] Detecting collisions...")

    # Multiple collisions for each id value may exist, so we construct a graph of id values
    print(f"[      ] Constructing collision graph...", end="")
    graph: Dict[int, List[int]] = {}
    for a, b in collisions:
        if a not in graph:
            graph[a] = [b]
        else:
            graph[a].append(b)
        if b not in graph:
            graph[b] = [a]
        else:
            graph[b].append(a)
    print("\r[\x1b[1;32;40m DONE \x1b[0m] Constructing collision graph...")

    # Each skeleton, of multiple potential id values, forms a connected component in the graph
    print(f"[      ] Identifying connected components...", end="")
    cc: List[List[int]] = connected_components(graph)
    print("\r[\x1b[1;32;40m DONE \x1b[0m] Identifying connected components...")

    # We need to decide which node of the graph, (id value of a skeleton) will represent the rest of the skeleton
    # For simplicity we just choose the last value.
    to_replace: List[int] = []
    replace_with: List[int] = []
    for component in cc:
        # we always replace with the LAST value of a bunch of connected components
        a = component.pop(-1)
        replace_with.extend([a for _ in component])
        to_replace.extend(component)

    # for every pixel at position j, if it is identical to a value at position i in two_replace,
    # we replace pixel j with the new value: replace_with[i]

    collisions = [
        (a, b) for a, b in zip(to_replace, replace_with)
    ]  # replace needs a set of collisions

    print(f"[      ] Performing in place replacement of collisions...", end="")
    skeleton = replace(skeleton, collisions)  # in place replace
    print(
        "\r[\x1b[1;32;40m DONE \x1b[0m] Performing in place replacement of collisions..."
    )

    return skeleton.squeeze(0)


def flood_all(x: Tensor, id: int) -> Tuple[Tensor, int, Dict[int, Tensor]]:
    """
    Finds all features (connected components in an ndarray) and gives it a unique label from 1 to N for N components

    :param x: torch.bool tensor
    :param id: previous max id value
    :return:
    """
    x = x.gt(0).int()

    mask, max_id = label(input=x.cpu().numpy())
    mask = torch.from_numpy(mask).to(torch.int16)

    mask = mask + x.mul(id)

    return mask, mask.max()


def dfs(
    connected: List[int],
    node: int,
    graph: Dict[int, List[int]],
    visited: Dict[int, bool],
) -> List[int]:
    """depth first search for finding connected components of a graph"""
    visited[node] = True
    connected.append(node)
    for n in graph[node]:
        if not visited[n]:
            connected = dfs(connected, n, graph, visited)
    return connected


def connected_components(graph: Dict[int, List[int]]) -> List[List[int]]:
    """
    Finds all connected components in a graph of id values by performing depth
    first search.

    :param graph: input graph where each key is a node, and each value is a list of edges
    :return: list of all connected notes
    """
    cc: List[List[int]] = []

    nodes = graph.keys()
    visited = {n: False for n in graph.keys()}
    for n in nodes:
        if not visited[n]:
            connected = []
            cc.append(dfs(connected, n, graph, visited))
    return cc


@njit(parallel=True)
def _in_place_replace(
    x: np.ndarray, to_replace: np.ndarray, replace_with: np.ndarray
) -> None:
    """
    Performs an in place replacement of values in tensor x.

    Checks each location in x for a value in to_replace. if a value is in to_replace, the value is
    swapped with the associated value in replace_with.

    :param x: input nd.array
    :param to_replace: array of values to be replaced in x
    :param replace_with: array of values by which should be replaced
    :return:
    """
    assert (
        to_replace.shape == replace_with.shape
    ), "to_replace must be the same size as replace with"
    assert x.ndim == 1, "input tensor must be reveled"

    for i in prange(x.shape[0]):
        # ind = to_replace == x[i]
        # if np.any(ind):
        for j, v in enumerate(to_replace):
            if x[i] == v:
                x[i] = replace_with[j]  # replace_with[ind][0]
                break  # presumably most of the time we'll hit the actual value much sooner...


def replace(x: Tensor, collisions: List[Tuple[int, int]]) -> Tensor:
    """
    Performs an in place replacement of values in the input tensor :math:`x`.
    This function calls a just-in-time compiled numba kernel which parallelizes the replacement.

    Roughly performs this algorith:
        for i in range(x):
            for (to_replace, replace_with) in collisions:
                if x[i] == to_replace:
                    x[i] = replace_with
        return x


    :param x: Input torch.Tensor of any size
    :param collisions: List of collisions where c[0] is the value to replace, and c[1] is the value to replace with
    :return: Original tensor with modified memory
    """

    assert x.dtype == torch.int16, f"Input tensor datatype must be int16 not {x.dtype}"

    shape = x.shape
    x = x.flatten().numpy()

    to_replace = np.array([a[0] for a in collisions], dtype=np.int16).flatten()
    replace_with = np.array([a[1] for a in collisions], dtype=np.int16).flatten()

    _in_place_replace(x, to_replace=to_replace, replace_with=replace_with)

    return torch.from_numpy(x).view(shape)


def get_adjacent_labels(x: Tensor, y: Tensor) -> List[Tuple[int, int]]:
    """
    calculates which masks of a signle object have two labels (due to the border)

    :param x:
    :param y:
    :return:
    """

    # Could ideally use the cantor pairing function but might overflow???
    # this seems to be safe and memory efficient
    z0 = (x + y).unique().tolist()
    z1 = (x * y).unique().tolist()

    identical = []

    for _x in x.unique():
        for _y in y.unique():
            if _x == 0 or _y == 0:
                continue

            if (_x + _y) in z0 and (_x * _y) in z1:
                identical.append((_x.item(), _y.item()))
    # identical = identical if len(identical) > 0 else None
    return identical


if __name__ == "__main__":
    graph = {
        1: [2, 3],
        2: [1],
        3: [1, 5, 4],
        4: [5],
        5: [3],
        6: [7],
        7: [6, 8, 9],
        8: [7],
        9: [7],
    }
    cc = connected_components(graph)
    print(cc)
