import torch
from torch import Tensor
from skimage.morphology import flood_fill, flood
from tqdm import tqdm
from typing import Tuple, Optional, Union, Dict, List

from skimage.morphology import skeletonize as sk_skeletonize
from skoots.lib.cropper import crops
from skoots.lib.merge import get_adjacent_labels

from scipy.ndimage import label

from numba import njit, prange
import numpy as np


def efficient_flood_fill(skeleton: Tensor,
                         min_skeleton_size: Optional[int] = 100,
                         skeletonize: bool = False,
                         device: Optional[Union[str, torch.device]] = 'cpu'
                         ) -> Tuple[Tensor, Dict[int, Tensor]]:
    """
    Efficiently flood fills a skeleton tensor

    :param skeleton:
    :param min_skeleton_size:
    :param skeletonize:
    :param device:
    :return:
    """
    # This is the w/h/d on EITHER side of a center seed point.
    # resulting crop size will be:  [2W, 2H, 2D]
    w, h, d = [550, 550, 50]

    unlabeled: list = skeleton.squeeze().eq(
        1).nonzero().tolist()  # Get ALL unlabeled pixels. This is potentially SUPER inefficient...
    unlabeled: Dict[str, Tensor] = {str(x): x for x in unlabeled}  # convert to hash table for memory efficiency

    shape = skeleton.shape  # [X, Y, Z]

    id = 2
    skeleton_dict = {}

    pbar = tqdm()
    """
    OK WHAT IF WE JUST GET THE NONZERO VALUES FROM CURRENT CROP.
    then, there is no need to calculate the nonzero for *everything*
    we only calculate nonzero for everything when the crop contains no nonzero data. 
    
    """
    while len(unlabeled) > 0:  # hash map could improve performance...
        # ind = torch.randint(unlabeled.shape[0], (1,)).squeeze()
        key, seed = unlabeled.popitem()
        # seed = unlabeled[ind, :].tolist()  # [X, Y ,Z]

        # Get the indices of each crop. Respect the boundaries of the image
        x0 = torch.tensor(seed[0] - w).clamp(0, shape[0])
        x1 = torch.tensor(seed[0] + w).clamp(0, shape[0])

        y0 = torch.tensor(seed[1] - h).clamp(0, shape[1])
        y1 = torch.tensor(seed[1] + h).clamp(0, shape[1])

        z0 = torch.tensor(seed[2] - d).clamp(0, shape[2])
        z1 = torch.tensor(seed[2] + d).clamp(0, shape[2])

        # Create a crop which to perform the flood fill. This helps with speed
        # cannot necessarily be sure that an object is 100% encompassed by the window
        # The window is pretty large though... It works will in practice.
        crop = skeleton[x0:x1, y0:y1, z0:z1]

        # The seed values we calculated before need to be corrected to this local window
        seed = tuple(int(s - offset) for s, offset in zip(seed, [x0, y0, z0]))

        # Fill the image with the new id value by a flood_fill algorithm
        mask = torch.from_numpy(flood(crop.cpu().numpy(), seed_point=seed)).to(device)

        # Assign the new pixels to the original tensor
        # ind = crop == id
        ind_nonzero = mask.nonzero()

        if ind_nonzero.shape[0] > min_skeleton_size:
            skeleton[x0:x1, y0:y1, z0:z1][mask] = id

            # Experimental
            if skeletonize:
                # scale_factor = 5
                # a = torch.tensor((scale_factor, scale_factor, 1), device=ind.device).view(1, 3)
                # _skeleton = torch.nonzero(ind[::scale_factor, ::scale_factor, :]) * a
                _skeleton = torch.from_numpy(sk_skeletonize(mask.cpu().numpy())).nonzero().to(device)

                if _skeleton.numel() == 0:
                    _skeleton = ind_nonzero  # If there are no skeletons, just return mean

            else:
                _skeleton = ind_nonzero

            skeleton_dict[id] = _skeleton.to(device) + torch.tensor([x0, y0, z0], device=device)  # [N, 3]

        else:
            skeleton[x0:x1, y0:y1, z0:z1][mask] = 0.  # crop[crop == id]

        # drop all nonzero elements in dict
        ind_nonzero = ind_nonzero + torch.tensor([x0, y0, z0], device=ind_nonzero.device)
        for v in ind_nonzero.tolist():
            unlabeled.pop(str(v), None)  # each nonzero element should be a key in the unlabeled hash map

        del mask, ind_nonzero

        id += 1
        pbar.desc = f'ID: {id} | Remaining Skeletons: {len(unlabeled)}'
        pbar.update(1)

    pbar.close()

    return skeleton, skeleton_dict


def efficient_flood_fill_v2(skeleton: Tensor,
                            min_skeleton_size: Optional[int] = 100,
                            skeletonize: bool = False,
                            device: Optional[Union[str, torch.device]] = 'cpu'
                            ) -> Tuple[Tensor, Dict[int, Tensor]]:
    """
    Efficiently flood fills a skeleton tensor

    :param skeleton:
    :param min_skeleton_size:
    :param skeletonize:
    :param device:
    :return:
    """
    # This is the w/h/d on EITHER side of a center seed point.
    # resulting crop size will be:  [2W, 2H, 2D]
    w, h, d = [300, 300, 60]

    nonzero: Tensor = skeleton.squeeze().eq(1).nonzero()
    shape = skeleton.shape  # [X, Y, Z]

    id = 2
    skeleton_dict = {}

    pbar = tqdm()
    """
    OK WHAT IF WE JUST GET THE NONZERO VALUES FROM CURRENT CROP.
    then, there is no need to calculate the nonzero for *everything*
    we only calculate nonzero for everything when the crop contains no nonzero data. 
    """
    while nonzero.numel() > 0:  # hash map could improve performance...
        # key, seed = unlabeled.popitem()
        seed = nonzero[0, :].tolist()  # [x, y, z]
        # seed = unlabeled[ind, :].tolist()  # [X, Y ,Z]

        # Get the indices of each crop. Respect the boundaries of the image
        x0 = torch.tensor(seed[0] - w).clamp(0, shape[0])
        x1 = torch.tensor(seed[0] + w).clamp(0, shape[0])

        y0 = torch.tensor(seed[1] - h).clamp(0, shape[1])
        y1 = torch.tensor(seed[1] + h).clamp(0, shape[1])

        z0 = torch.tensor(seed[2] - d).clamp(0, shape[2])
        z1 = torch.tensor(seed[2] + d).clamp(0, shape[2])

        # Create a crop which to perform the flood fill. This helps with speed
        # cannot necessarily be sure that an object is 100% encompassed by the window
        # The window is pretty large though... It works will in practice.
        """ 
        HYPOTHESIS: This is super slow... 
        
        Better to do a crop n merge type of thing? 
        
        """
        crop = skeleton[x0:x1, y0:y1, z0:z1]

        # The seed values we calculated before need to be corrected to this local window
        seed = tuple(int(s - offset) for s, offset in zip(seed, [x0, y0, z0]))

        # Fill the image with the new id value by a flood_fill algorithm
        mask = torch.from_numpy(flood(crop.cpu().numpy(), seed_point=seed)).to(device)

        # Assign the new pixels to the original tensor
        # ind = crop == id
        ind_nonzero = mask.nonzero()

        if ind_nonzero.shape[0] > min_skeleton_size:
            skeleton[x0:x1, y0:y1, z0:z1][mask] = id

            # Experimental
            if skeletonize:
                # scale_factor = 5
                # a = torch.tensor((scale_factor, scale_factor, 1), device=ind.device).view(1, 3)
                # _skeleton = torch.nonzero(ind[::scale_factor, ::scale_factor, :]) * a
                _skeleton = torch.from_numpy(sk_skeletonize(mask.cpu().numpy())).nonzero().to(device)

                if _skeleton.numel() == 0:
                    _skeleton = ind_nonzero  # If there are no skeletons, just return mean

            else:
                _skeleton = ind_nonzero

            skeleton_dict[id] = _skeleton.to(device) + torch.tensor([x0, y0, z0], device=device)  # [N, 3]

        else:
            skeleton[x0:x1, y0:y1, z0:z1][mask] = 0.  # crop[crop == id]

        nonzero = skeleton[x0:x1, y0:y1, z0:z1].eq(1).nonzero() + torch.tensor([x0, y0, z0])
        # nonzero = skeleton.eq(1).nonzero()

        if nonzero.numel() == 0:
            print('had to calculate a huge nonzero')
            nonzero = skeleton.eq(1).nonzero()

        del mask, ind_nonzero

        id += 1
        pbar.desc = f'ID: {id} | Remaining Skeletons: {len(nonzero)}'
        pbar.update(1)

    pbar.close()

    return skeleton, skeleton_dict


def efficient_flood_fill_v3(
        skeleton: Tensor,
        min_skeleton_size: Optional[int] = 100,
        skeletonize: bool = False,
        device: Optional[Union[str, torch.device]] = 'cpu'
) -> Tuple[Tensor, Dict[int, Tensor]]:
    skeleton = skeleton.unsqueeze(0) if skeleton.ndim == 3 else skeleton

    # Crop size has to be exactly identical to eval!!!
    iterator = tqdm(crops(skeleton, crop_size=[300, 300, 10]), desc='Assigning Instances:')
    max_id = 1
    skeletons_dict = {}

    seams_x = [] # this will be all the seams of the crops we need to check later.
    seams_y = []
    seams_z = []

    for crop, (x, y, z) in iterator:
        seams_x = seams_x + [x] if x not in seams_x else seams_x
        seams_y = seams_y + [y] if y not in seams_y else seams_y
        seams_z = seams_z + [z] if z not in seams_z else seams_z

        # print(f'{skeleton.shape=}, {(x,y,z)=}')
        crop = crop.squeeze().gt(0).int()
        crop, max_id, _skeletons = flood_all(crop, max_id + 1)
        w, h, d = crop.shape
        skeleton[0, x:x + w, y:y + h, z:z + d] = crop

    # X
    collisions = []
    for x in seams_x:
        if x > 0:
            slice_0 = skeleton[0, x, :, :]
            slice_1 = skeleton[0, x - 1, :, :]
            # torch.save(slice_0, '/home/chris/Desktop/slice_0.trch')
            # torch.save(slice_1, '/home/chris/Desktop/slice_1.trch')
            collisions.extend(get_adjacent_labels(slice_0, slice_1))
            # print(collisions)
            # raise ValueError
            # for a, b in collisions:
            #     if b == 5 and a == 96:
            #         print('WRONG: x', x)
    # Y
    for y in seams_y:
        if y > 0:
            slice_0 = skeleton[0, :, y, :]
            slice_1 = skeleton[0, :, y-1, :]
            collisions.extend(get_adjacent_labels(slice_0, slice_1))
            for a, b in collisions:
                if b == 5 and a == 96:
                    print('WRONG: y', y)

    # Z
    for z in seams_z:
        if z > 0:
            slice_0 = skeleton[0, :, :, z]
            slice_1 = skeleton[0, :, :, z - 1]
            collisions.extend(get_adjacent_labels(slice_0, slice_1))
            for a, b in collisions:
                if b == 5 and a == 96:
                    print('WRONG: z', z)

    for a, b in collisions:
        if a in [73, 5, 28, 101, 9, 81]:
            print(a, b)


    # CHECK EVERYTHING AGAIN HERE! some collisions for a graphs where more than 2 labels are connected
    """
    1. check all borders and find all connected components
    2. construct graph of all nodes 
    3. find all closed paths
    4. loop over all pixels in the final image and replace
        - do this by modifying values of a sparse tensor...
        
    """
    graph = {}
    for (a, b) in collisions:
        if a not in graph:
            graph[a] = [b]
        else:
            graph[a].append(b)
        if b not in graph:
            graph[b] = [a]
        else:
            graph[b].append(a)

        if a == 96 or b == 96:
            print('GRAPH CREATION:', a, b)

    cc: List[List[int]] = connected_components(graph)

    print('graph!')
    for k,v in graph.items():
        print(k, v)


    print('connected components!')
    for _cc in cc:
        print(_cc)

    to_replace: List[int] = []
    replace_with: List[int] = []

    for component in cc:
        # we always replace with the LAST value of a bunch of connected components
        a = component.pop(-1)
        replace_with.extend([a for _ in component])
        to_replace.extend(component)

    collisions = [(a, b) for a, b in zip(to_replace, replace_with)]
    skeleton = replace(skeleton, collisions)


    return skeleton.squeeze(0), skeletons_dict


def flood_all(x: Tensor, id: int) -> Tuple[Tensor, int, Dict[int, Tensor]]:
    """
    Floods a crop of a larger image

    :param x: torch.bool tensor
    :param id: previous max id value
    :return:
    """
    x = x.gt(0).int()

    mask, max_id = label(input=x.cpu().numpy())
    # print(f'FLOOD: {np.unique(mask)=}, {id=},', end='')
    mask = torch.from_numpy(mask).to(torch.int16)

    mask = mask + x.mul(id)
    # print(f'after: {mask.max().item()=}, {mask[mask != 0].min().item()=}')

    return mask, mask.max(), None


def dfs(connected: List[int], node: int, graph: Dict[int, List[int]], visited: Dict[int, bool]) -> List[int]:
    """ depth first search """
    visited[node] = True
    connected.append(node)
    for n in graph[node]:
        if not visited[n]:
            connected = dfs(connected, n, graph, visited)
    return connected


def connected_components(graph: Dict[int, List[int]]) -> List[List[int]]:
    """
    find all connected components in a graph
    iterative depth first search
    i dont know how to do this...

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
def _in_place_replace(x: np.ndarray, to_replace: np.ndarray, replace_with: np.ndarray) -> None:
    """
    Performs an in place replacement of values in tensor x.

    checks each loaction in x for a value in to_replace. if a value is in to_replace, the value is
    swapped with the associated value in replace_with.

    :param x: input nd.array
    :param to_replace: array of values to be replaced in x
    :param replace_with: array of values by which should be replaced
    :return:
    """
    assert to_replace.shape == replace_with.shape, 'to_replace must be the same size as replace with'
    assert x.ndim == 1, 'input tensor must be reveled'

    for i in prange(x.shape[0]):
        ind = to_replace == x[i]
        if np.any(ind):
            x[i] = replace_with[ind][0]


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

    assert x.dtype == torch.int16, f'Input tensor datatype must be int16 not {x.dtype}'

    shape = x.shape
    x = x.flatten().numpy()

    to_replace = np.array([a[0] for a in collisions], dtype=np.int16).flatten()
    replace_with = np.array([a[1] for a in collisions], dtype=np.int16).flatten()

    _in_place_replace(x, to_replace=to_replace, replace_with=replace_with)

    return torch.from_numpy(x).view(shape)


if __name__ == '__main__':
    graph = {
        1: [2, 3],
        2: [1],
        3: [1,5,4],
        4: [5],
        5: [3],
        6: [7],
        7: [6,8,9],
        8: [7],
        9: [7]
    }
    cc = connected_components(graph)
    print(cc)

