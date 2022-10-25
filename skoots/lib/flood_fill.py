import torch
from torch import Tensor
from skimage.morphology import flood_fill, flood
from tqdm import tqdm
from typing import Tuple, Optional, Union, Dict

from skimage.morphology import skeletonize as sk_skeletonize


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

        id += 1
        pbar.desc = f'ID: {id} | Remaining Skeletons: {len(unlabeled)}'
        pbar.update(1)

    pbar.close()

    return skeleton, skeleton_dict
