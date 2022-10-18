import torch
from torch import Tensor
from skimage.morphology import flood_fill
from tqdm import tqdm
from typing import Tuple, Optional, Union, Dict


def efficient_flood_fill(skeleton: Tensor,
                         min_skeleton_size: Optional[int] = 100,
                         device: Optional[Union[str, torch.device]] = 'cpu') -> Tuple[Tensor, Dict[int, Tensor]]:
    """
    get a

    :param skeleton:
    :param device:
    :return:
    """
    # This is the w/h/d on EITHER side of a center seed point.
    # resulting crop size will be:  [2W, 2H, 2D]
    w, h, d = [550, 550, 20]

    unlabeled = skeleton.eq(1).nonzero()  # Get ALL unlabeled pixels

    shape = skeleton.shape  # [X, Y, Z]

    id = 2
    skeleton_dict = {}

    pbar = tqdm()
    while unlabeled.numel() > 0:
        ind = torch.randint(unlabeled.shape[0], (1,)).squeeze()
        seed = unlabeled[ind, :].tolist()  # [X, Y ,Z]

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
        crop = torch.from_numpy(flood_fill(crop.cpu().numpy(), seed_point=seed, new_value=id)).to(device)

        # Assign the new pixels to the original tensor
        ind = crop == id
        if ind.nonzero().shape[0] > min_skeleton_size:
            skeleton[x0:x1, y0:y1, z0:z1][ind] = crop[ind].to(skeleton.device)
            skeleton_dict[id] = torch.nonzero(ind) + torch.tensor([x0, y0, z0], device=device)  # [N, 3]

        else:
            skeleton[x0:x1, y0:y1, z0:z1][ind] = 0.  # crop[crop == id]

        # Recalculate the unlabeled pixels. This is unfortunately the fastest way to do this...
        unlabeled = skeleton.eq(1).nonzero()

        id += 1
        pbar.desc = f'ID: {id} | Remaining Skeletons: {unlabeled.shape[0]}'
        pbar.update(1)

    pbar.close()

    return skeleton, skeleton_dict
