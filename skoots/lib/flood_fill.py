import torch
from torch import Tensor
from skimage.morphology import flood_fill, flood
from tqdm import tqdm
from typing import Tuple, Optional, Union, Dict, List

from skimage.morphology import skeletonize as sk_skeletonize
from skoots.lib.cropper import crops
from skoots.lib.merge import get_adjacent_labels

from scipy.ndimage import label


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

    unlabeled: list = skeleton.squeeze().eq(1).nonzero().tolist()  # Get ALL unlabeled pixels. This is potentially SUPER inefficient...
    unlabeled: Dict[str, Tensor] = {str(x): x for x in unlabeled}            # convert to hash table for memory efficiency

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
        seed = nonzero[0, :].tolist() # [x, y, z]
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
)-> Tuple[Tensor, Dict[int, Tensor]]:

    skeleton = skeleton.unsqueeze(0) if skeleton.ndim == 3 else skeleton

    # Crop size has to be exactly identical to eval!!!
    iterator = tqdm(crops(skeleton, crop_size=[300, 300, 10]), desc='Assigning Instances:')
    max_id = 2
    skeletons_dict = {}

    for crop, (x, y, z) in iterator:
        # print(f'{skeleton.shape=}, {(x,y,z)=}')
        crop = crop.squeeze().gt(0)
        crop, max_id, _skeletons = flood_all(crop, max_id)
        w, h, d = crop.shape

        # Check the boundries of each crop here.
        collisions = []
        if x > 0:   # X - dont need to check edges
            slice_0 = crop[0, :, :]
            slice_1 = skeleton[0, x-1, y:y+h, z:z+d]
            collisions.extend(get_adjacent_labels(slice_0, slice_1))

        if y > 0:  # X - dont need to check edges
            slice_0 = crop[:, 0, :]
            slice_1 = skeleton[0, x:x+w, y-1, z:z+d]
            collisions.extend(get_adjacent_labels(slice_0, slice_1))

        if z > 0:  # X - dont need to check edges
            slice_0 = crop[:, :, 0]
            slice_1 = skeleton[0, x:x+w, y:y+h, z-1]
            collisions.extend(get_adjacent_labels(slice_0, slice_1))

        """ 
        A pair of collisions may already have been dealt with. We need to both remember what
        collision has been addressed, but which it's been replaced with...
        """
        collisions = torch.tensor(collisions) # N x 2

        for i, (a, b) in enumerate(collisions):
            crop[crop.eq(a)] = b
            # ind = collisions[:, 0] == a
            # collisions[ind, 0] = b
            # skeleton[skeleton.eq(a)] = b


        # previously_handled_collision = {}
        # for (_a, _b) in collisions:
        #
        #     if _a not in previously_handled_collision:
        #         print(f'replacing {_a=} with {_b=}')
        #         previously_handled_collision[_a] = _b
        #         crop[crop.eq(_a)] = _b
        #
        #
        #     else:
        #         while _a in previously_handled_collision:
        #             _a = previously_handled_collision[_a] # reassign to previously used
        #         #
        #         # print('PREVIOUS COLLISION', _a, ' -> ', previously_handled_collision[_a], ' | ', _b)
        #         # print(f'\t{_a} was previously assigned to {previously_handled_collision[_a]}, now {_b}')
        #
        #         crop[crop.eq(_a)] = _b
        #         s = skeleton[0, x:x + w, y:y + h, z:z + d]
        #         skeleton[0, x:x + w, y:y + h, z:z + d][s.eq(_a)] = _b
        #         previously_handled_collision[_a] = _b
        #         print(f'replacing {_a=} with {_b=}')

        skeleton[0, x:x + w, y:y + h, z:z + d] = crop

    return skeleton.squeeze(0), skeletons_dict




def flood_all(x: Tensor, id: int) -> Tuple[Tensor, int, Dict[int, Tensor]]:
    """
    Floods a crop of a larger image

    :param x: torch.bool tensor
    :param id: previous max id value
    :return:
    """

    mask, max_id = label(input=x.cpu().numpy())
    mask = torch.from_numpy(mask)

    mask = mask + x.mul(id)

    return mask, id + max_id, None


