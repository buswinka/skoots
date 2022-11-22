import torch
from torch import Tensor
import torchvision.transforms.functional as ttf
from typing import Dict, Tuple, Union, Sequence, List, Callable, Optional
from skoots.lib.morphology import binary_erosion, _get_binary_kernel3d
from skoots.lib.skeleton import bake_skeleton, skeleton_to_mask

import math
from copy import deepcopy


@torch.jit.script
def _get_box(mask: Tensor, device: str, threshold: int) -> Tuple[Tensor, Tensor]:
    # mask in shape of 300, 400, 1 [H, W, z=1]
    nonzero = torch.nonzero(mask)  # Q, 3=[x,y,z]
    label = mask.max()

    box = torch.tensor([-1, -1, -1, -1], dtype=torch.long, device=device)

    # Recall, image in shape of [C, H, W]

    if nonzero.numel() > threshold:
        x0 = torch.min(nonzero[:, 1])
        x1 = torch.max(nonzero[:, 1])
        y0 = torch.min(nonzero[:, 0])
        y1 = torch.max(nonzero[:, 0])

        if (x1 - x0 > 0) and (y1 - y0 > 0):
            box[0] = x0
            box[1] = y0
            box[2] = x1
            box[3] = y1

    return label, box


@torch.jit.script
def _get_affine_matrix(
        center: List[float], angle: float, translate: List[float], scale: float, shear: List[float], device: str,
) -> Tensor:
    # We need compute the affine transformation matrix: M = T * C * RSS * C^-1

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    T: Tensor = torch.eye(3, device=device)
    T[0, -1] = translate[0]
    T[1, -1] = translate[1]

    C: Tensor = torch.eye(3, device=device)
    C[0, -1] = center[0]
    C[1, -1] = center[1]

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    RSS = torch.tensor(
        [[a, b, 0.],
         [c, d, 0.],
         [0., 0., 1.]],
        device=device)
    RSS = RSS * scale
    RSS[-1, -1] = 1

    return T @ C @ RSS @ torch.inverse(C)


def calc_centroid(mask: torch.Tensor, id: int) -> torch.Tensor:
    temp = (mask == id).float()

    # crop the region to just erode small region...
    lower = torch.nonzero(temp).min(0)[0]
    upper = torch.nonzero(temp).max(0)[0]

    temp = temp[
           lower[0].item():upper[0].item(),  # x
           lower[1].item():upper[1].item(),  # y
           lower[2].item():upper[2].item(),  # z
           ]

    if temp.numel() == 0:
        return torch.tensor((-1, -1, -1), device=mask.device)

    x, y, z = temp.shape
    temp = temp.view((1, x, y, z))

    nonzero = torch.nonzero(temp)

    if nonzero.numel() == 0:
        return torch.tensor((-1, -1, -1), device=mask.device)

    old_temp = temp
    while nonzero.numel() > 0:
        old_temp = temp
        temp = temp.unsqueeze(0) if temp.ndim == 4 else temp
        temp = binary_erosion(temp)
        nonzero = torch.nonzero(temp)

    # except Exception
    #     print(temp.shape)
    #     print('threw error for some reason - merged_transform.py line 209')
    #     raise ValueError

    center = torch.nonzero(old_temp.view(x, y, z)).float().mean(0).add(lower)

    if torch.any(torch.isnan(center)):
        print(f'{id=}, {temp.shape=}, {old_temp.shape=}, {old_temp.numel()=}, {nonzero.numel()=}')
        print(f'nonzero old_temp {torch.nonzero(old_temp.view(x, y, z)).float()}')
        print(upper, lower)
        raise ValueError

    return center


@torch.no_grad()
def merged_transform_3D(data_dict: Dict[str, Tensor],
                        device: Optional[str] = None,
                        bake_skeleton_anisotropy: Tuple[float, float, float] = (1.0, 1.0, 3.0),
                        smooth_skeleton_kernel_size: Tuple[int, int, int] = (3, 3, 1)
                        ) -> Dict[str, Tensor]:
    DEVICE: str = str(data_dict['image'].device) if device is None else device

    # Image should be in shape of [C, H, W, D]
    CROP_WIDTH = torch.tensor([300], device=DEVICE)
    CROP_HEIGHT = torch.tensor([300], device=DEVICE)
    CROP_DEPTH = torch.tensor([20], device=DEVICE)

    FLIP_RATE = torch.tensor(0.5, device=DEVICE)

    BRIGHTNESS_RATE = torch.tensor(0.4, device=DEVICE)
    BRIGHTNESS_RANGE = torch.tensor((-0.1, 0.1), device=DEVICE)

    NOISE_GAMMA = torch.tensor(0.1, device=DEVICE)
    NOISE_RATE = torch.tensor(0.2, device=DEVICE)

    FILTER_RATE = torch.tensor(0.5, device=DEVICE)

    CONTRAST_RATE = torch.tensor(0.33, device=DEVICE)
    CONTRAST_RANGE = torch.tensor((0.75, 2.), device=DEVICE)

    AFFINE_RATE = torch.tensor(0.66, device=DEVICE)
    AFFINE_SCALE = torch.tensor((0.85, 1.1), device=DEVICE)
    AFFINE_YAW = torch.tensor((-180, 180), device=DEVICE)
    AFFINE_SHEAR = torch.tensor((-7, 7), device=DEVICE)

    masks = torch.clone(data_dict['masks'])  # .to(DEVICE))
    image = torch.clone(data_dict['image'])  #

    skeletons = deepcopy(data_dict['skeletons'])
    skeletons = {k: v.float().to(DEVICE) for k, v in skeletons.items()}

    # ------------ Random Crop 1
    extra = 300
    w = CROP_WIDTH + extra if CROP_WIDTH + extra <= image.shape[1] else torch.tensor(image.shape[1])
    h = CROP_HEIGHT + extra if CROP_HEIGHT + extra <= image.shape[2] else torch.tensor(image.shape[2])
    d = CROP_DEPTH if CROP_DEPTH <= image.shape[3] else torch.tensor(image.shape[3])

    # Randomly select a centroid to center in frame
    ind: int = torch.randint(len(skeletons.keys()), (1,), dtype=torch.long, device=DEVICE).item()
    key: int = list(skeletons)[ind]
    center: Tensor = skeletons[key].mean(0).squeeze()

    # Center that instance
    x0 = center[0].sub(torch.floor(w / 2)).long().clamp(min=0, max=image.shape[1] - w.item())
    y0 = center[1].sub(torch.floor(h / 2)).long().clamp(min=0, max=image.shape[2] - h.item())
    z0 = center[2].sub(torch.floor(d / 2)).long().clamp(min=0, max=image.shape[3] - d.item())

    x1 = x0 + w
    y1 = y0 + h
    z1 = z0 + d

    image = image[:, x0.item():x1.item(), y0.item():y1.item(), z0.item():z1.item()].to(DEVICE)
    masks = masks[:, x0.item():x1.item(), y0.item():y1.item(), z0.item():z1.item()].to(DEVICE)

    # Correct the skeleton positions
    unique = torch.unique(masks)
    if -1 not in skeletons:
        keys = list(skeletons.keys())
        for k in keys:
            if not torch.any(unique == k):
                skeletons.pop(k)
            else:
                skeletons[k] = skeletons[k] - torch.tensor([x0, y0, z0], device=DEVICE)

    # -------------------affine (Cant use baked skeletons)
    if torch.rand(1, device=DEVICE) < AFFINE_RATE:
        angle = (AFFINE_YAW[1] - AFFINE_YAW[0]) * torch.rand(1, device=DEVICE) + AFFINE_YAW[0]
        shear = (AFFINE_SHEAR[1] - AFFINE_SHEAR[0]) * torch.rand(1, device=DEVICE) + AFFINE_SHEAR[0]
        scale = (AFFINE_SCALE[1] - AFFINE_SCALE[0]) * torch.rand(1, device=DEVICE) + AFFINE_SCALE[0]

        mat: Tensor = _get_affine_matrix(center=[image.shape[1] / 2, image.shape[2] / 2],
                                         angle=-angle.item(),
                                         translate=[0., 0.],
                                         scale=scale.item(),
                                         shear=[float(shear.item()), float(shear.item())],
                                         device=str(image.device))

        # Rotate the skeletons by the affine matrix
        for k, v in skeletons.items():
            skeleton_xy = v[:, [0, 1]].permute(1, 0).unsqueeze(0)  # [N, 3] -> [1, 2, N]
            _ones = torch.ones((1, 1, skeleton_xy.shape[-1]), device=DEVICE)  # [1, 1, N]
            skeleton_xy = torch.cat((skeleton_xy, _ones), dim=1)  # [1, 3, N]
            rotated_skeleton = mat @ skeleton_xy  # [1,3,N]
            skeletons[k][:, [0, 1]] = rotated_skeleton[0, [0, 1], :].T.float()

        image = ttf.affine(image.permute(0, 3, 1, 2).float(),
                           angle=angle.item(),
                           shear=[float(shear.item())],
                           scale=scale.item(),
                           translate=[0, 0]).permute(0, 2, 3, 1)

        unique_before = masks.unique().long().sub(1)
        unique_before = unique_before[unique_before.ge(0)]

        masks = ttf.affine(masks.permute(0, 3, 1, 2).float(),
                           angle=angle.item(),
                           shear=[float(shear.item())],
                           scale=scale.item(),
                           translate=[0, 0]).permute(0, 2, 3, 1)

        unique_after = masks.unique().long().sub(1)
        unique_after = unique_after[unique_after.ge(0)]

        skeletons = {k: v for k, v in skeletons.items() if torch.any(unique_after.eq(k - 1))}

        assert len(skeletons.keys()) > 0, f'{unique_after=}, {unique_before=}'

    # ------------ Center Crop 2
    w = CROP_WIDTH if CROP_WIDTH <= image.shape[1] else torch.tensor(image.shape[1])
    h = CROP_HEIGHT if CROP_HEIGHT <= image.shape[2] else torch.tensor(image.shape[2])
    d = CROP_DEPTH if CROP_DEPTH <= image.shape[3] else torch.tensor(image.shape[3])

    center = center - torch.tensor([x0, y0, z0], device=DEVICE)

    # Center that instance
    x0 = center[0].sub(torch.floor(w / 2)).long().clamp(min=0, max=image.shape[1] - w.item())
    y0 = center[1].sub(torch.floor(h / 2)).long().clamp(min=0, max=image.shape[2] - h.item())
    z0 = center[2].sub(torch.floor(d / 2)).long().clamp(min=0, max=image.shape[3] - d.item())

    x1 = x0 + w
    y1 = y0 + h
    z1 = z0 + d

    image = image[:, x0.item():x1.item(), y0.item():y1.item(), z0.item():z1.item()]
    masks = masks[:, x0.item():x1.item(), y0.item():y1.item(), z0.item():z1.item()]

    unique = torch.unique(masks)
    if -1 not in skeletons:
        keys = list(skeletons.keys())
        for k in keys:
            if not torch.any(unique == k):
                skeletons.pop(k)
            else:
                skeletons[k] = skeletons[k] - torch.tensor([x0, y0, z0], device=DEVICE)

    # ------------------- x flip
    if torch.rand(1, device=DEVICE) < FLIP_RATE:
        image = image.flip(1)
        masks = masks.flip(1)

        if -1 not in skeletons:
            for k, v in skeletons.items():
                skeletons[k][:, 0] = image.shape[1] - v[:, 0]

    # ------------------- y flip
    if torch.rand(1, device=DEVICE) < FLIP_RATE:
        image = image.flip(2)
        masks = masks.flip(2)
        if -1 not in skeletons:
            for k, v in skeletons.items():
                skeletons[k][:, 1] = image.shape[2] - v[:, 1]

    # ------------------- z flip
    if torch.rand(1, device=DEVICE) < FLIP_RATE:
        image = image.flip(3)
        masks = masks.flip(3)
        if -1 not in skeletons:
            for k, v in skeletons.items():
                skeletons[k][:, 2] = image.shape[3] - v[:, 2]

    # # ------------------- Random Invert
    if torch.rand(1, device=DEVICE) < BRIGHTNESS_RATE:
        image = image.sub(1).mul(-1)

    # ------------------- Adjust Brightness
    if torch.rand(1, device=DEVICE) < BRIGHTNESS_RATE:
        # funky looking but FAST
        val = torch.empty(image.shape[0], device=DEVICE).uniform_(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
        image = image.add(val.reshape(image.shape[0], 1, 1, 1)).clamp(0, 1)

    # ------------------- Adjust Contrast
    if torch.rand(1, device=DEVICE) < CONTRAST_RATE:
        contrast_val = (CONTRAST_RANGE[1] - CONTRAST_RANGE[0]) * torch.rand((image.shape[0]), device=DEVICE) + \
                       CONTRAST_RANGE[0]

        for z in range(image.shape[-1]):
            image[..., z] = ttf.adjust_contrast(image[..., z], contrast_val[0].item()).squeeze(0)

    # ------------------- Noise
    if torch.rand(1, device=DEVICE) < NOISE_RATE:
        noise = torch.rand(image.shape, device=DEVICE) * NOISE_GAMMA

        image = image.add(noise).clamp(0, 1)

    data_dict['image'] = image
    data_dict['masks'] = masks
    data_dict['skeletons'] = skeletons

    baked: Tensor = bake_skeleton(masks, skeletons, anisotropy=bake_skeleton_anisotropy, average=True, device=DEVICE)
    data_dict['baked-skeleton']: Union[Tensor, None] = baked

    _, x, y, z = masks.shape
    data_dict['skele_masks']: Tensor = skeleton_to_mask(skeletons, (x, y, z), kernel_size=smooth_skeleton_kernel_size, n=1)

    return data_dict


def background_transform_3D(data_dict: Dict[str, Tensor], device: Optional[str] = None) -> Dict[str, Tensor]:
    DEVICE: str = str(data_dict['image'].device) if device is None else device

    # Image should be in shape of [C, H, W, D]
    CROP_WIDTH = torch.tensor([300], device=DEVICE)
    CROP_HEIGHT = torch.tensor([300], device=DEVICE)
    CROP_DEPTH = torch.tensor([20], device=DEVICE)

    FLIP_RATE = torch.tensor(0.5, device=DEVICE)

    BRIGHTNESS_RATE = torch.tensor(0.4, device=DEVICE)
    INVERT_RATE = torch.tensor(0.5, device=DEVICE)
    BRIGHTNESS_RANGE = torch.tensor((-0.1, 0.1), device=DEVICE)

    NOISE_GAMMA = torch.tensor(0.1, device=DEVICE)
    NOISE_RATE = torch.tensor(0.2, device=DEVICE)

    FILTER_RATE = torch.tensor(0.5, device=DEVICE)

    CONTRAST_RATE = torch.tensor(0.33, device=DEVICE)
    CONTRAST_RANGE = torch.tensor((0.75, 2.), device=DEVICE)

    AFFINE_RATE = torch.tensor(0.66, device=DEVICE)
    AFFINE_SCALE = torch.tensor((0.85, 1.1), device=DEVICE)
    AFFINE_YAW = torch.tensor((-180, 180), device=DEVICE)
    AFFINE_SHEAR = torch.tensor((-7, 7), device=DEVICE)

    image = torch.clone(data_dict['image'])  #

    # ------------ Random Crop 1
    extra = 300
    w = CROP_WIDTH + extra if CROP_WIDTH + extra <= image.shape[1] else torch.tensor(image.shape[1])
    h = CROP_HEIGHT + extra if CROP_HEIGHT + extra <= image.shape[2] else torch.tensor(image.shape[2])
    d = CROP_DEPTH if CROP_DEPTH <= image.shape[3] else torch.tensor(image.shape[3])

    shape = torch.tensor(image.shape[1::], device=DEVICE) - torch.tensor([w, h, d], device=DEVICE)
    center = torch.tensor([torch.randint(0, s, (1,)).item() if s > 0 else 0 for s in shape], device=DEVICE)

    # Center that instance
    x0 = center[0].sub(torch.floor(w / 2)).long().clamp(min=0, max=image.shape[1] - w.item())
    y0 = center[1].sub(torch.floor(h / 2)).long().clamp(min=0, max=image.shape[2] - h.item())
    z0 = center[2].sub(torch.floor(d / 2)).long().clamp(min=0, max=image.shape[3] - d.item())

    x1 = x0 + w
    y1 = y0 + h
    z1 = z0 + d

    image = image[:, x0.item():x1.item(), y0.item():y1.item(), z0.item():z1.item()].to(DEVICE)

    # -------------------affine (Cant use baked skeletons)
    if torch.rand(1, device=DEVICE) < AFFINE_RATE:
        angle = (AFFINE_YAW[1] - AFFINE_YAW[0]) * torch.rand(1, device=DEVICE) + AFFINE_YAW[0]
        shear = (AFFINE_SHEAR[1] - AFFINE_SHEAR[0]) * torch.rand(1, device=DEVICE) + AFFINE_SHEAR[0]
        scale = (AFFINE_SCALE[1] - AFFINE_SCALE[0]) * torch.rand(1, device=DEVICE) + AFFINE_SCALE[0]

        image = ttf.affine(image.permute(0, 3, 1, 2).float(),
                           angle=angle.item(),
                           shear=[float(shear.item())],
                           scale=scale.item(),
                           translate=[0, 0]).permute(0, 2, 3, 1)

    # ------------ Center Crop 2
    w = CROP_WIDTH if CROP_WIDTH <= image.shape[1] else torch.tensor(image.shape[1])
    h = CROP_HEIGHT if CROP_HEIGHT <= image.shape[2] else torch.tensor(image.shape[2])
    d = CROP_DEPTH if CROP_DEPTH <= image.shape[3] else torch.tensor(image.shape[3])

    center = center - torch.tensor([x0, y0, z0], device=DEVICE)

    # Center that instance
    x0 = center[0].sub(torch.floor(w / 2)).long().clamp(min=0, max=image.shape[1] - w.item())
    y0 = center[1].sub(torch.floor(h / 2)).long().clamp(min=0, max=image.shape[2] - h.item())
    z0 = center[2].sub(torch.floor(d / 2)).long().clamp(min=0, max=image.shape[3] - d.item())

    x1 = x0 + w
    y1 = y0 + h
    z1 = z0 + d

    image = image[:, x0.item():x1.item(), y0.item():y1.item(), z0.item():z1.item()]

    # ------------------- x flip
    if torch.rand(1, device=DEVICE) < FLIP_RATE:
        image = image.flip(1)

    # ------------------- y flip
    if torch.rand(1, device=DEVICE) < FLIP_RATE:
        image = image.flip(2)

    # ------------------- z flip
    if torch.rand(1, device=DEVICE) < FLIP_RATE:
        image = image.flip(3)

    # # ------------------- Random Invert
    if torch.rand(1, device=DEVICE) < INVERT_RATE:
        image = image.sub(1).mul(-1)

    # ------------------- Adjust Brightness
    if torch.rand(1, device=DEVICE) < BRIGHTNESS_RATE:
        # funky looking but FAST
        val = torch.empty(image.shape[0], device=DEVICE).uniform_(BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])
        image = image.add(val.reshape(image.shape[0], 1, 1, 1)).clamp(0, 1)

    # ------------------- Adjust Contrast
    if torch.rand(1, device=DEVICE) < CONTRAST_RATE:
        contrast_val = (CONTRAST_RANGE[1] - CONTRAST_RANGE[0]) * torch.rand((image.shape[0]), device=DEVICE) + \
                       CONTRAST_RANGE[0]

        for z in range(image.shape[-1]):
            image[..., z] = ttf.adjust_contrast(image[..., z], contrast_val[0].item()).squeeze(0)

    # ------------------- Noise
    if torch.rand(1, device=DEVICE) < NOISE_RATE:
        noise = torch.rand(image.shape, device=DEVICE) * NOISE_GAMMA

        image = image.add(noise).clamp(0, 1)

    data_dict['image'] = image
    data_dict['masks'] = torch.zeros_like(image, device=DEVICE)
    data_dict['skeletons']: Dict[int, Tensor] = {-1: torch.empty((0, 3), device=DEVICE)}
    data_dict['baked-skeleton'] = torch.zeros((3, image.shape[1], image.shape[2], image.shape[3]), device=DEVICE)
    data_dict['skele_masks'] = torch.zeros_like(image, device=DEVICE)

    return data_dict


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torchvision.utils
    import tqdm
    import torch.distributed as dist
    import torch.optim.lr_scheduler
    from skoots.train.dataloader import dataset, colate, MultiDataset

    from torch.utils.data import DataLoader

    torch.manual_seed(0)

    path = '/home/chris/Dropbox (Partners HealthCare)/trainHairCellSegmentation/data/'
    data = dataset(path=path, transforms=merged_transform_3D, sample_per_image=4).to('cuda')
    dataloader = DataLoader(data, num_workers=0, batch_size=4, collate_fn=colate)

    for im, ma, cen in dataloader:
        pass
