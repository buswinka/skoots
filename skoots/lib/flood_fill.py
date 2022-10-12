import torch
from torch import Tensor
from skimage.morphology import flood_fill
from tqdm import tqdm
from typing import Tuple


def efficient_flood_fill(skeleton: Tensor, device='cpu') -> Tuple[Tensor, Tensor]:
    """
    get a

    :param skeleton:
    :return:
    """

    w, h, d = [550, 550, 20]

    unlabeled = skeleton.eq(1).nonzero()

    shape = skeleton.shape  # [X, Y, Z]

    id = 2
    skeleton_dict = {}

    pbar = tqdm()
    while unlabeled.numel() > 0:
        ind = torch.randint(unlabeled.shape[0], (1,)).squeeze()
        seed = unlabeled[ind, :].tolist()  # [X, Y ,Z]

        x0 = torch.tensor(seed[0] - w).clamp(0, shape[0])
        x1 = torch.tensor(seed[0] + w).clamp(0, shape[0])

        y0 = torch.tensor(seed[1] - h).clamp(0, shape[1])
        y1 = torch.tensor(seed[1] + h).clamp(0, shape[1])

        z0 = torch.tensor(seed[2] - d).clamp(0, shape[2])
        z1 = torch.tensor(seed[2] + d).clamp(0, shape[2])

        crop = skeleton[x0:x1, y0:y1, z0:z1]
        seed = tuple(int(s - offset) for s, offset in zip(seed, [x0, y0, z0]))

        crop = torch.from_numpy(flood_fill(crop.cpu().numpy(), seed_point=seed, new_value=id)).to(device)

        ind = crop == id
        if ind.nonzero().shape[0] > 100:
            skeleton[x0:x1, y0:y1, z0:z1][ind] = crop[ind].to(skeleton.device)
            skeleton_dict[id] = torch.nonzero(ind) + torch.tensor([x0, y0, z0], device=device)  # [N, 3]

        else:
            skeleton[x0:x1, y0:y1, z0:z1][ind] = 0. #crop[crop == id]

        unlabeled = skeleton.eq(1).nonzero()

        id += 1
        pbar.desc = f'ID: {id} | Remaining Skeletons: {unlabeled.shape[0]}'
        pbar.update(1)
    pbar.close()

    return skeleton, skeleton_dict
