import math
import random
from copy import deepcopy
from typing import Dict, Tuple, Union, List, Optional, Callable

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms.functional as ttf
from skoots.lib.morphology import binary_erosion
from skoots.lib.skeleton import bake_skeleton, skeleton_to_mask

from skoots.lib.custom_types import DataDict
from torch import Tensor
from yacs.config import CfgNode

import logging

def elastic_deform(
        image: Tensor,
        mask: Tensor,
        skeleton: Dict[int, Tensor],
        displacement_shape: Tuple[int, int, int] = (6, 6, 2),
        max_displacement: float = 0.2,
) -> Tuple[Tensor, Tensor, Dict[int, Tensor]]:
    """
    Randomly creates a deformation grid and applies to image, mask, and skeletons

    :param image: Tensor
    :param mask:  Tensor
    :param skeleton:  Dict[int, Tensor[3, N]]
    :param displacement_shape: Tuple of len 3
    :param max_displacement: float between 0 and 1
    :return:
    """

    assert image.shape == mask.shape, "image and mask shape must be the same"
    assert image.ndim == 5, f"image must be in shape: [B, C, X, Y, Z], not {image.shape}"
    assert image.device == mask.device
    assert (
            len(displacement_shape) == 3
    ), "displacement_shape must be a tuple of integers with len == 3"
    assert max_displacement < 1.0, "max displacement must not exceed 1.0"

    # for k, v in skeleton.items():
    #     assert v.shape[1] == 3, f'skeleton shape wrong: {v.shape}'

    device = image.device

    b, c, x, y, z = image.shape

    # offset are the random directon vectors
    offset: Tensor = (
        F.interpolate(
            torch.rand((1, 3, 2, 2, 2), device=device), (x, y, z), mode="trilinear"
        ).permute((0, 2, 3, 4, 1)).mul(
            torch.tensor((0.05, 0.2, 0.2), device=device).view(1, 1, 1, 1, 3)
        )

    )

    # base_grid is the identity grid. Applying base_grid alone will result in an identical image as input
    base_grid: Tensor = torch.stack(
        (
            torch.linspace(-1, 1, z, dtype=torch.float, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(x, y, 1),
            torch.linspace(-1, 1, y, dtype=torch.float, device=device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(x, 1, z),
            torch.linspace(-1, 1, x, dtype=torch.float, device=device)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, y, z),
        ),
        dim=-1,
    ).unsqueeze(0)

    grid = (base_grid + offset).float()

    io.imsave('/home/chris/Desktop/grid.tif', grid.squeeze().permute(2, 0, 1, 3).cpu().numpy())

    # apply the deformation
    image = F.grid_sample(image.float(), grid.float(), align_corners=True, mode='nearest')
    mask = F.grid_sample(mask.float(), grid.float(), align_corners=True, mode='nearest')

    # remap grid from -1->1 to 0->shape
    # grid = (base_grid - offset).float()
    grid = grid.add(1).div(2).mul(
        torch.tensor((z, y, x), device=device).view(1, 1, 1, 1, 3)
    )
    print(f'{grid[0,34,54,22,:]=}')

    for k, skel in skeleton.items():
        sx: Tensor = skel[:, 0].long()
        sy: Tensor = skel[:, 1].long()
        sz: Tensor = skel[:, 2].long()

        ind_x = torch.logical_and(sx >= 0, sx < x)
        ind_y = torch.logical_and(sy >= 0, sy < y)
        ind_z = torch.logical_and(sz >= 0, sz < z)

        # Equivalent to a 3 way logical and
        ind = (ind_x.float() + ind_y.float() + ind_z.float()) == 3

        # grid last dim is Z, Y, X for some reason
        print(skel)
        skel[ind, :] = grid[0, sx[ind], sy[ind], sz[ind], :][:, [2, 1, 0]]
        print(skel)

        skeleton[k] = skel

    return image, mask, skeleton


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


# @torch.jit.script
def _get_affine_matrix(
        center: List[float],
        angle: float,
        translate: List[float],
        scale: float,
        shear: List[float],
        device: str,
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
    # a = math.cos(rot - sy) / math.cos(sy)
    # b = -1 * math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    # c = math.sin(rot - sy) / math.cos(sy)
    # d = math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(
        rot
    )  # # rotated scale shear

    RSS = torch.tensor([[a, b, 0.0], [c, d, 0.0], [0.0, 0.0, 1.0]], device=device)
    RSS = RSS * scale
    # print(RSS)

    RSS[-1, -1] = 1

    # cx, cy = center
    # tx, ty = translate
    #
    # matrix = [a, b, 0.0, c, d, 0.0]
    # matrix = [x * scale for x in matrix]
    # # Apply inverse of center translation: RSS * C^-1
    # matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
    # matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
    # # Apply translation and center : T * C * RSS * C^-1
    # matrix[2] += cx + tx
    # matrix[5] += cy + ty
    #
    #
    # # matrix = [d, -b, 0.0, -c, a, 0.0]
    # # matrix = [x / scale for x in matrix]
    # # # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    # # matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    # # matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
    # # # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    # # matrix[2] += cx
    # # matrix[5] += cy
    #
    # matrix += [0, 0, 1]
    # matrix = torch.tensor(matrix, device=device).view(3, 3)
    # return matrix
    #
    #
    #
    return T @ C @ RSS @ torch.inverse(C)
    # return C @ torch.inverse(C) @ torch.inverse(T) @ torch.inverse(RSS)


def _get_inverse_affine_matrix(
        center: List[float],
        angle: float,
        translate: List[float],
        scale: float,
        shear: List[float],
        inverted: bool = True,
) -> List[float]:
    # Helper method to compute inverse matrix for affine transformation

    # Pillow requires inverse affine transformation matrix:
    # Affine matrix is : M = T * C * RotateScaleShear * C^-1
    #
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RotateScaleShear is rotation with scale and shear matrix
    #
    #       RotateScaleShear(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
    #         [ s*sin(a - sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
    #         [ 0                    , 0                                        , 1 ]
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    if inverted:
        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        matrix = [d, -b, 0.0, -c, a, 0.0]
        matrix = [x / scale for x in matrix]
        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
        matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += cx
        matrix[5] += cy
    else:
        matrix = [a, b, 0.0, c, d, 0.0]
        matrix = [x * scale for x in matrix]
        # Apply inverse of center translation: RSS * C^-1
        matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
        matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
        # Apply translation and center : T * C * RSS * C^-1
        matrix[2] += cx + tx
        matrix[5] += cy + ty

    return matrix


def calc_centroid(mask: Tensor, id: int) -> Tensor:
    temp = (mask == id).float()

    # crop the region to just erode small region...
    lower = torch.nonzero(temp).min(0)[0]
    upper = torch.nonzero(temp).max(0)[0]

    temp = temp[
           lower[0].item(): upper[0].item(),  # x
           lower[1].item(): upper[1].item(),  # y
           lower[2].item(): upper[2].item(),  # z
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
        print(
            f"{id=}, {temp.shape=}, {old_temp.shape=}, {old_temp.numel()=}, {nonzero.numel()=}"
        )
        print(f"nonzero old_temp {torch.nonzero(old_temp.view(x, y, z)).float()}")
        print(upper, lower)
        raise ValueError

    return center


class TransformFromCfg(nn.Module):
    def __init__(self, cfg: CfgNode, device: torch.device, scale: float = 255.0):
        super(TransformFromCfg, self).__init__()
        """
        Why? Apparently a huge amount of overhead is just initializing this from cfg
        If we preinitalize, then we can save on overhead, to do this, we need a class...
        Probably a reasonalbe functional way to do this. Ill think on it later

        """

        self.prefix_function = self._identity
        self.posfix_function = self._identity

        self.dataset_mean = 0
        self.dataset_std = 1

        self.cfg = cfg

        self.DEVICE = device
        self.SCALE = scale

        self.CROP_WIDTH = cfg.AUGMENTATION.CROP_WIDTH
        self.CROP_HEIGHT = cfg.AUGMENTATION.CROP_HEIGHT

        self.CROP_DEPTH = cfg.AUGMENTATION.CROP_DEPTH

        self.FLIP_RATE = cfg.AUGMENTATION.FLIP_RATE

        self.BRIGHTNESS_RATE = cfg.AUGMENTATION.BRIGHTNESS_RATE
        self.BRIGHTNESS_RANGE = cfg.AUGMENTATION.BRIGHTNESS_RANGE
        self.NOISE_GAMMA = cfg.AUGMENTATION.NOISE_GAMMA
        self.NOISE_RATE = cfg.AUGMENTATION.NOISE_RATE

        self.FILTER_RATE = 0.5

        self.CONTRAST_RATE = cfg.AUGMENTATION.CONTRAST_RATE
        self.CONTRAST_RANGE = cfg.AUGMENTATION.CONTRAST_RANGE

        self.AFFINE_RATE = cfg.AUGMENTATION.AFFINE_RATE
        self.AFFINE_SCALE = cfg.AUGMENTATION.AFFINE_SCALE
        self.AFFINE_SHEAR = cfg.AUGMENTATION.AFFINE_SHEAR
        self.AFFINE_YAW = cfg.AUGMENTATION.AFFINE_YAW

        self.BAKE_SKELETON_ANISOTROPY = cfg.AUGMENTATION.BAKE_SKELETON_ANISOTROPY

        self._center = None
        self._xyz = (None, None, None)

    def _identity(self, *args):
        return args if len(args) > 1 else args[0]

    def _crop1(self, image, masks, skeletons):
        # ------------ Random Crop 1
        extra = 300
        C, X, Y, Z = image.shape
        # ------------ Random Crop 1
        extra = 300
        w = self.CROP_WIDTH + extra if self.CROP_WIDTH + extra <= X else X
        h = self.CROP_HEIGHT + extra if self.CROP_HEIGHT + extra <= Y else Y
        d = self.CROP_DEPTH if self.CROP_DEPTH <= Z else Z

        key = random.choice(list(skeletons.keys()))
        self._center: Tensor = skeletons[key].float().mean(0).squeeze()

        # Center that instance
        x0 = (
            self._center[0]
            .sub(w // 2)
            .long()
            .clamp(min=0, max=image.shape[1] - w)
        )
        y0 = (
            self._center[1]
            .sub(h // 2)
            .long()
            .clamp(min=0, max=image.shape[2] - h)
        )
        z0 = (
            self._center[2]
            .sub(d // 2)
            .long()
            .clamp(min=0, max=image.shape[3] - d)
        )

        self._xyz = (x0, y0, z0)

        x1 = x0 + w
        y1 = y0 + h
        z1 = z0 + d

        image = image[:, x0: x1, y0: y1, z0: z1]
        masks = masks[:, x0: x1, y0: y1, z0: z1]

        # Correct the skeleton positions
        unique = torch.unique(masks)
        if -1 not in skeletons:
            keys = list(skeletons.keys())
            for k in keys:
                if not torch.any(unique == k):
                    skeletons.pop(k)
                else:
                    skeletons[k] = skeletons[k] - torch.tensor([x0, y0, z0], device=self.DEVICE)

        # send to device
        if image.device != self.DEVICE:
            image = image.to(self.DEVICE)

        if masks.device != self.DEVICE:
            masks = masks.to(self.DEVICE)

        for k, v in skeletons.items():
            skeletons[k] = v.float().to(self.DEVICE)

        return image, masks, skeletons

    def _affine(self, image, masks, skeletons):
        angle = (self.AFFINE_YAW[1] - self.AFFINE_YAW[0]) * torch.rand(
            1, device=self.DEVICE
        ) + self.AFFINE_YAW[0]
        shear = (self.AFFINE_SHEAR[1] - self.AFFINE_SHEAR[0]) * torch.rand(
            1, device=self.DEVICE
        ) + self.AFFINE_SHEAR[0]
        scale = (self.AFFINE_SCALE[1] - self.AFFINE_SCALE[0]) * torch.rand(
            1, device=self.DEVICE
        ) + self.AFFINE_SCALE[0]

        # shear = torch.tensor((35), device=DEVICE)
        # angle = torch.tensor((45), device=DEVICE)
        # scale = torch.tensor((1), device=DEVICE)

        mat: Tensor = _get_affine_matrix(
            center=[image.shape[1] / 2, image.shape[2] / 2],
            angle=-angle.item(),
            translate=[0.0, 0.0],
            scale=scale.item(),
            shear=[0.0, float(shear.item())],
            device=str(image.device),
        )  # Rotate the skeletons by the affine matrix

        for k, v in skeletons.items():
            skeleton_xy = v[:, [0, 1]].permute(1, 0).unsqueeze(0)  # [N, 3] -> [1, 2, N]
            _ones = torch.ones(
                (1, 1, skeleton_xy.shape[-1]), device=self.DEVICE
            )  # [1, 1, N]
            skeleton_xy = torch.cat((skeleton_xy, _ones), dim=1)  # [1, 3, N]
            rotated_skeleton = mat @ skeleton_xy  # [1,3,N]
            skeletons[k][:, [0, 1]] = rotated_skeleton[0, [0, 1], :].T.float()

        image = ttf.affine(
            image.permute(0, 3, 1, 2).float(),
            angle=angle.item(),
            shear=float(shear.item()),
            scale=scale.item(),
            translate=[0, 0],
        ).permute(0, 2, 3, 1)

        # unique_before = masks.unique().long().sub(1)
        # unique_before = unique_before[unique_before.ge(0)]

        masks = ttf.affine(
            masks.permute(0, 3, 1, 2).float(),
            angle=angle.item(),
            shear=float(shear.item()),
            scale=scale.item(),
            translate=[0, 0],
        ).permute(0, 2, 3, 1)

        unique_after = masks.unique().long().sub(1)
        unique_after = unique_after[unique_after.ge(0)]

        skeletons = {
            k: v for k, v in skeletons.items() if torch.any(unique_after.eq(k - 1))
        }

        return image, masks, skeletons
        # angle = random.uniform(*self.AFFINE_YAW)
        # shear = random.uniform(*self.AFFINE_YAW)
        # scale = random.uniform(*self.AFFINE_SCALE)
        #
        # mat: Tensor = _get_affine_matrix(
        #     center=[image.shape[1] / 2, image.shape[2] / 2],
        #     angle=-angle,
        #     translate=[0.0, 0.0],
        #     scale=scale,
        #     shear=[0.0, shear],
        #     device=str(image.device),
        # )  # Rotate the skeletons by the affine matrix
        #
        # if -1 not in skeletons:
        #     for k, v in skeletons.items():
        #         skeleton_xy = v[:, [0, 1]].permute(1, 0).unsqueeze(0)  # [N, 3] -> [1, 2, N]
        #         _ones = torch.ones((1, 1, skeleton_xy.shape[-1]), device=self.DEVICE)  # [1, 1, N]
        #         skeleton_xy = torch.cat((skeleton_xy, _ones), dim=1)  # [1, 3, N]
        #         rotated_skeleton = mat @ skeleton_xy  # [1,3,N]
        #         skeletons[k][:, [0, 1]] = rotated_skeleton[0, [0, 1], :].T.float()
        #
        # image = ttf.affine(
        #     image.permute(0, 3, 1, 2).float(),
        #     angle=angle,
        #     shear=[float(shear)],
        #     scale=scale,
        #     translate=[0, 0],
        # ).permute(0, 2, 3, 1)
        #
        # masks = ttf.affine(
        #     masks.permute(0, 3, 1, 2).float(),
        #     angle=angle,
        #     shear=[float(shear)],
        #     scale=scale,
        #     translate=[0, 0],
        # ).permute(0, 2, 3, 1)
        #
        # return image, masks, skeletons

    def _crop2(self, image, masks, skeletons):
        C, X, Y, Z = image.shape
        w = self.CROP_WIDTH if self.CROP_WIDTH < X else X
        h = self.CROP_HEIGHT if self.CROP_HEIGHT < Y else Y
        d = self.CROP_DEPTH if self.CROP_DEPTH < Z else Z

        x0, y0, z0 = self._xyz

        self._center = self._center - torch.tensor([x0, y0, z0], device=self.DEVICE)

        # Center that instance
        x0 = (
            self._center[0]
            .sub(w // 2)
            .long()
            .clamp(min=0, max=image.shape[1] - w)
        )
        y0 = (
            self._center[1]
            .sub(h // 2)
            .long()
            .clamp(min=0, max=image.shape[2] - h)
        )
        z0 = (
            self._center[2]
            .sub(d // 2)
            .long()
            .clamp(min=0, max=image.shape[3] - d)
        )

        x1 = x0 + w
        y1 = y0 + h
        z1 = z0 + d

        image = image[:, x0:x1, y0:y1, z0:z1]
        masks = masks[:, x0:x1, y0:y1, z0:z1]

        unique = torch.unique(masks)
        if -1 not in skeletons:
            keys = list(skeletons.keys())
            for k in keys:
                if not torch.any(unique == k):
                    skeletons.pop(k)
                else:
                    skeletons[k] = skeletons[k] - torch.tensor([x0, y0, z0], device=self.DEVICE)

        return image, masks, skeletons

    def _flipX(self, image, masks, skeletons):
        image = image.flip(1)
        masks = masks.flip(1)

        if -1 not in skeletons:
            for k, v in skeletons.items():
                skeletons[k][:, 0] = image.shape[1] - v[:, 0]

        return image, masks

    def _flipY(self, image, masks, skeletons):
        image = image.flip(2)
        masks = masks.flip(2)
        if -1 not in skeletons:
            for k, v in skeletons.items():
                skeletons[k][:, 1] = image.shape[2] - v[:, 1]
        return image, masks

    def _flipZ(self, image, masks, skeletons):
        image = image.flip(3)
        masks = masks.flip(3)
        if -1 not in skeletons:
            for k, v in skeletons.items():
                skeletons[k][:, 2] = image.shape[3] - v[:, 2]
        return image, masks

    def _invert(self, image, masks):
        image.sub_(1).mul_(-1)
        return image, masks

    def _brightness(self, image, masks):
        val = random.uniform(*self.BRIGHTNESS_RANGE)
        # in place ok because flip always returns a copy
        image = image.add(val)
        return image, masks

    def _contrast(self, image, masks):
        contrast_val = random.uniform(*self.CONTRAST_RANGE)
        # [ C, X, Y, Z ] -> [Z, C, X, Y]
        image = ttf.adjust_contrast(image.permute(3, 0, 1, 2), contrast_val).permute(1, 2, 3, 0)

        return image, masks

    def _noise(self, image, masks):
        noise = torch.rand(image.shape, device=self.DEVICE) * self.NOISE_GAMMA
        image = image.add(noise)
        return image, masks

    def _normalize(self, image, masks):
        # mean = image.float().mean()
        # std = image.float().std()
        mean = image.float().mean() if not self.dataset_mean else self.dataset_mean
        std = image.float().std() if not self.dataset_std else self.dataset_std

        image = image.float().sub(mean).div(std)
        return image, masks

    def set_dataset_mean(self, mean):
        self.dataset_mean = mean
        return self

    def set_dataset_std(self, std):
        self.dataset_std = std
        return self

    @torch.no_grad()
    def forward(self, data_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:

        assert "masks" in data_dict, 'keyword "masks" not in data_dict'
        assert "image" in data_dict, 'keyword "image" not in data_dict'
        assert "skeletons" in data_dict, 'keyword "skeletons" not in data_dict'

        logging.debug(f"TransformFromCfg.forward() | starting transforms on device: {self.DEVICE}")


        logging.debug("TransformFromCfg.forward() | applying prefix function")
        data_dict = self.prefix_function(data_dict)

        masks = data_dict["masks"]
        image = data_dict["image"]
        skeletons = deepcopy(data_dict["skeletons"])

        spatial_dims = masks.ndim - 1

        logging.debug("TransformFromCfg.forward() | applying crop 1")
        image, masks, skeletons = self._crop1(image, masks, skeletons)

        # scale: int = 2 ** 16 if image.max() > 256 else 255  # Our images might be 16 bit, or 8 bit
        # scale = scale if image.max() > 1 else 1.0

        logging.debug("TransformFromCfg.forward() | normalizing")
        image, masks = self._normalize(image, masks)

        # affine
        if random.random() < self.AFFINE_RATE:
            logging.debug("TransformFromCfg.forward() | applying affine transform")
            image, masks, skeletons = self._affine(image, masks, skeletons)

        # ------------ Center Crop 2
        logging.debug("TransformFromCfg.forward() | applying crop 2")
        image, masks, skeletons = self._crop2(image, masks, skeletons)

        # ------------------- x flip
        if random.random() < self.FLIP_RATE:
            logging.debug("TransformFromCfg.forward() | flipping in x")
            image, masks = self._flipX(image, masks, skeletons)

        # ------------------- y flip
        if random.random() < self.FLIP_RATE:
            logging.debug("TransformFromCfg.forward() | flipping in y")
            image, masks = self._flipY(image, masks, skeletons)

        # ------------------- z flip
        if random.random() < self.FLIP_RATE:
            logging.debug("TransformFromCfg.forward() | flipping in z")
            image, masks = self._flipZ(image, masks, skeletons)

        # # ------------------- Random Invert
        if random.random() < self.BRIGHTNESS_RATE:
            logging.debug("TransformFromCfg.forward() | inverting")
            image, masks = self._invert(image, masks)

        # ------------------- Adjust Brightness
        if random.random() < self.BRIGHTNESS_RATE:
            logging.debug("TransformFromCfg.forward() | adjusting brightness")
            image, masks = self._brightness(image, masks)

        # ------------------- Adjust Contrast
        if random.random() < self.CONTRAST_RATE:
            logging.debug("TransformFromCfg.forward() | adjusting contrast")
            image, masks = self._contrast(image, masks)

        # ------------------- Noise
        if random.random() < self.NOISE_RATE:
            logging.debug("TransformFromCfg.forward() | adding noise")
            image, masks = self._noise(image, masks)

        if spatial_dims == 2:
            image = image[..., 0]
            masks = masks[..., 0]

        data_dict["image"] = image
        data_dict["masks"] = masks

        baked: Tensor = bake_skeleton(
            masks,
            skeletons,
            anisotropy=self.BAKE_SKELETON_ANISOTROPY,
            average=True,
            device=self.DEVICE,
        )
        data_dict["baked_skeleton"]: Union[Tensor, None] = baked

        _, x, y, z = masks.shape
        data_dict["skele_masks"]: Tensor = skeleton_to_mask(
            skeletons,
            (x, y, z),
            device=self.DEVICE
            # kernel_size=cfg.AUGMENTATION.SMOOTH_SKELETON_KERNEL_SIZE,
            # n=cfg.AUGMENTATION.N_SKELETON_MASK_DILATE,
        )

        data_dict = self.posfix_function(data_dict)

        return data_dict

    def pre_fn(self, fn: Callable[[Dict[str, Tensor]], Dict[str, Tensor]]):
        self.prefix_function = fn
        return self

    def post_fn(self, fn: Callable[[Dict[str, Tensor]], Dict[str, Tensor]]):
        self.posfix_function = fn
        return self

    def post_crop_fn(self, fn):
        self.postcrop_function = fn
        return self

    def __repr__(self):
        return f"TransformFromCfg[Device:{self.DEVICE}]\ncfg.AUGMENTATION:\n=================\n{self.cfg.AUGMENTATION}]"

#
# @torch.no_grad()
# # @torch.jit.ignore()
# def transform_from_cfg(
#         data_dict: Dict[str, Tensor], cfg: CfgNode, device: Optional[str] = None
# ) -> DataDict:
#     DEVICE: str = str(data_dict["image"].device) if device is None else device
#
#     # Image should be in shape of [C, H, W, D]
#     CROP_WIDTH = torch.tensor(cfg.AUGMENTATION.CROP_WIDTH, device=DEVICE)
#     CROP_HEIGHT = torch.tensor(cfg.AUGMENTATION.CROP_HEIGHT, device=DEVICE)
#     CROP_DEPTH = torch.tensor(cfg.AUGMENTATION.CROP_DEPTH, device=DEVICE)
#
#     FLIP_RATE = torch.tensor(cfg.AUGMENTATION.FLIP_RATE, device=DEVICE)
#
#     BRIGHTNESS_RATE = torch.tensor(cfg.AUGMENTATION.BRIGHTNESS_RATE, device=DEVICE)
#     BRIGHTNESS_RANGE = torch.tensor(cfg.AUGMENTATION.BRIGHTNESS_RANGE, device=DEVICE)
#
#     NOISE_GAMMA = torch.tensor(cfg.AUGMENTATION.NOISE_GAMMA, device=DEVICE)
#     NOISE_RATE = torch.tensor(cfg.AUGMENTATION.NOISE_RATE, device=DEVICE)
#
#     FILTER_RATE = torch.tensor(0.5, device=DEVICE)
#
#     CONTRAST_RATE = torch.tensor(cfg.AUGMENTATION.CONTRAST_RATE, device=DEVICE)
#     CONTRAST_RANGE = torch.tensor(cfg.AUGMENTATION.CONTRAST_RANGE, device=DEVICE)
#
#     AFFINE_RATE = torch.tensor(cfg.AUGMENTATION.AFFINE_RATE, device=DEVICE)
#     AFFINE_SCALE = torch.tensor(cfg.AUGMENTATION.AFFINE_SCALE, device=DEVICE)
#     AFFINE_YAW = torch.tensor(cfg.AUGMENTATION.AFFINE_YAW, device=DEVICE)
#     AFFINE_SHEAR = torch.tensor(cfg.AUGMENTATION.AFFINE_SHEAR, device=DEVICE)
#
#     masks = data_dict["masks"]  # .to(DEVI
#     image = data_dict["image"]
#     #
#     skeletons = deepcopy(data_dict["skeletons"])
#     skeletons = {k: v.float() for k, v in skeletons.items()}
#
#     # ------------ Random Crop 1
#     extra = 300
#     w = (
#         CROP_WIDTH + extra
#         if CROP_WIDTH + extra <= image.shape[1]
#         else torch.tensor(image.shape[1])
#     )
#     h = (
#         CROP_HEIGHT + extra
#         if CROP_HEIGHT + extra <= image.shape[2]
#         else torch.tensor(image.shape[2])
#     )
#     d = CROP_DEPTH if CROP_DEPTH <= image.shape[3] else torch.tensor(image.shape[3])
#
#     # Randomly select a centroid to center in frame
#     ind: int = torch.randint(
#         len(skeletons.keys()), (1,), dtype=torch.long, device=DEVICE
#     ).item()
#     key: int = list(skeletons)[ind]
#     center: Tensor = skeletons[key].mean(0).squeeze()
#
#     # Center that instance
#     x0 = (
#         center[0]
#         .sub(torch.floor(w / 2))
#         .long()
#         .clamp(min=0, max=image.shape[1] - w.item())
#     )
#     y0 = (
#         center[1]
#         .sub(torch.floor(h / 2))
#         .long()
#         .clamp(min=0, max=image.shape[2] - h.item())
#     )
#     z0 = (
#         center[2]
#         .sub(torch.floor(d / 2))
#         .long()
#         .clamp(min=0, max=image.shape[3] - d.item())
#     )
#
#     x1 = x0 + w
#     y1 = y0 + h
#     z1 = z0 + d
#
#     image = (
#         image[:, x0.item(): x1.item(), y0.item(): y1.item(), z0.item(): z1.item()]
#         .to(DEVICE)
#         .div(255)
#         .half()
#     )
#
#     masks = masks[
#             :, x0.item(): x1.item(), y0.item(): y1.item(), z0.item(): z1.item()
#             ].to(DEVICE)
#
#     # Correct the skeleton positions
#     unique = torch.unique(masks)
#     if -1 not in skeletons:
#         keys = list(skeletons.keys())
#         for k in keys:
#             if not torch.any(unique == k):
#                 skeletons.pop(k)
#             else:
#                 skeletons[k] = skeletons[k].to(DEVICE) - torch.tensor([x0, y0, z0], device=DEVICE)
#
#     # --------------------------- elastic transform
#     # image, masks, skeletons = elastic_deform(image.unsqueeze(0), masks.unsqueeze(0), skeletons)
#     # image = image.squeeze(0)
#     # masks = masks.squeeze(0)
#
#     # -------------------affine (Cant use baked skeletons)
#     if torch.rand(1, device=DEVICE) < AFFINE_RATE:
#         angle = (AFFINE_YAW[1] - AFFINE_YAW[0]) * torch.rand(
#             1, device=DEVICE
#         ) + AFFINE_YAW[0]
#         shear = (AFFINE_SHEAR[1] - AFFINE_SHEAR[0]) * torch.rand(
#             1, device=DEVICE
#         ) + AFFINE_SHEAR[0]
#         scale = (AFFINE_SCALE[1] - AFFINE_SCALE[0]) * torch.rand(
#             1, device=DEVICE
#         ) + AFFINE_SCALE[0]
#
#         # shear = torch.tensor((35), device=DEVICE)
#         # angle = torch.tensor((45), device=DEVICE)
#         # scale = torch.tensor((1), device=DEVICE)
#
#         mat: Tensor = _get_affine_matrix(
#             center=[image.shape[1] / 2, image.shape[2] / 2],
#             angle=-angle.item(),
#             translate=[0.0, 0.0],
#             scale=scale.item(),
#             shear=[0.0, float(shear.item())],
#             device=str(image.device),
#         )  # Rotate the skeletons by the affine matrix
#
#         for k, v in skeletons.items():
#             skeleton_xy = v[:, [0, 1]].permute(1, 0).unsqueeze(0)  # [N, 3] -> [1, 2, N]
#             _ones = torch.ones(
#                 (1, 1, skeleton_xy.shape[-1]), device=DEVICE
#             )  # [1, 1, N]
#             skeleton_xy = torch.cat((skeleton_xy, _ones), dim=1)  # [1, 3, N]
#             rotated_skeleton = mat @ skeleton_xy  # [1,3,N]
#             skeletons[k][:, [0, 1]] = rotated_skeleton[0, [0, 1], :].T.float()
#
#         image = ttf.affine(
#             image.permute(0, 3, 1, 2).float(),
#             angle=angle.item(),
#             shear=float(shear.item()),
#             scale=scale.item(),
#             translate=[0, 0],
#         ).permute(0, 2, 3, 1)
#
#         # unique_before = masks.unique().long().sub(1)
#         # unique_before = unique_before[unique_before.ge(0)]
#
#         masks = ttf.affine(
#             masks.permute(0, 3, 1, 2).float(),
#             angle=angle.item(),
#             shear=float(shear.item()),
#             scale=scale.item(),
#             translate=[0, 0],
#         ).permute(0, 2, 3, 1)
#
#         unique_after = masks.unique().long().sub(1)
#         unique_after = unique_after[unique_after.ge(0)]
#
#         skeletons = {
#             k: v for k, v in skeletons.items() if torch.any(unique_after.eq(k - 1))
#         }
#
#         # assert len(skeletons.keys()) > 0, f"{unique_after=}, {unique_before=}"
#
#     # # ------------ Center Crop 2
#     w = CROP_WIDTH if CROP_WIDTH <= image.shape[1] else torch.tensor(image.shape[1])
#     h = CROP_HEIGHT if CROP_HEIGHT <= image.shape[2] else torch.tensor(image.shape[2])
#     d = CROP_DEPTH if CROP_DEPTH <= image.shape[3] else torch.tensor(image.shape[3])
#
#     center = center - torch.tensor([x0, y0, z0], device=DEVICE)
#
#     # Center that instance
#     x0 = (
#         center[0]
#         .sub(torch.floor(w / 2))
#         .long()
#         .clamp(min=0, max=image.shape[1] - w.item())
#     )
#     y0 = (
#         center[1]
#         .sub(torch.floor(h / 2))
#         .long()
#         .clamp(min=0, max=image.shape[2] - h.item())
#     )
#     z0 = (
#         center[2]
#         .sub(torch.floor(d / 2))
#         .long()
#         .clamp(min=0, max=image.shape[3] - d.item())
#     )
#
#     x1 = x0 + w
#     y1 = y0 + h
#     z1 = z0 + d
#
#     image = image[
#             :, x0.item(): x1.item(), y0.item(): y1.item(), z0.item(): z1.item()
#             ]
#     masks = masks[
#             :, x0.item(): x1.item(), y0.item(): y1.item(), z0.item(): z1.item()
#             ]
#
#     unique = torch.unique(masks)
#     if -1 not in skeletons:
#         keys = list(skeletons.keys())
#         for k in keys:
#             if not torch.any(unique == k):
#                 skeletons.pop(k)
#             else:
#                 skeletons[k] = skeletons[k] - torch.tensor([x0, y0, z0], device=DEVICE)
#
#     # ------------------- x flip
#     if torch.rand(1, device=DEVICE) < FLIP_RATE:
#         image = image.flip(1)
#         masks = masks.flip(1)
#
#         if -1 not in skeletons:
#             for k, v in skeletons.items():
#                 skeletons[k][:, 0] = image.shape[1] - v[:, 0]
#
#     # ------------------- y flip
#     if torch.rand(1, device=DEVICE) < FLIP_RATE:
#         image = image.flip(2)
#         masks = masks.flip(2)
#         if -1 not in skeletons:
#             for k, v in skeletons.items():
#                 skeletons[k][:, 1] = image.shape[2] - v[:, 1]
#
#     # ------------------- z flip
#     if torch.rand(1, device=DEVICE) < FLIP_RATE:
#         image = image.flip(3)
#         masks = masks.flip(3)
#         if -1 not in skeletons:
#             for k, v in skeletons.items():
#                 skeletons[k][:, 2] = image.shape[3] - v[:, 2]
#
#     # # ------------------- Random Invert
#     if torch.rand(1, device=DEVICE) < BRIGHTNESS_RATE:
#         image = image.sub(1).mul(-1)
#
#     # ------------------- Adjust Brightness
#     if torch.rand(1, device=DEVICE) < BRIGHTNESS_RATE:
#         # funky looking but FAST
#         val = torch.empty(image.shape[0], device=DEVICE).uniform_(
#             BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1]
#         )
#         image = image.add(val.reshape(image.shape[0], 1, 1, 1)).clamp(0, 1)
#
#     # ------------------- Adjust Contrast
#     if torch.rand(1, device=DEVICE) < CONTRAST_RATE:
#         contrast_val = (CONTRAST_RANGE[1] - CONTRAST_RANGE[0]) * torch.rand(
#             (image.shape[0]), device=DEVICE
#         ) + CONTRAST_RANGE[0]
#
#         for z in range(image.shape[-1]):
#             image[..., z] = ttf.adjust_contrast(
#                 image[..., z], contrast_val[0].item()
#             ).squeeze(0)
#
#     # ------------------- Noise
#     if torch.rand(1, device=DEVICE) < NOISE_RATE:
#         noise = torch.rand(image.shape, device=DEVICE) * NOISE_GAMMA
#
#         image = image.add(noise).clamp(0, 1)
#     #
#     data_dict["image"] = image
#     data_dict["masks"] = masks
#     data_dict["skeletons"] = skeletons
#
#     baked: Tensor = bake_skeleton(
#         masks,
#         skeletons,
#         anisotropy=cfg.AUGMENTATION.BAKE_SKELETON_ANISOTROPY,
#         average=True,
#         device=DEVICE,
#     )
#     data_dict["baked_skeleton"]: Union[Tensor, None] = baked
#
#     _, x, y, z = masks.shape
#     data_dict["skele_masks"]: Tensor = skeleton_to_mask(
#         skeletons,
#         (x, y, z),
#         # kernel_size=cfg.AUGMENTATION.SMOOTH_SKELETON_KERNEL_SIZE,
#         # n=cfg.AUGMENTATION.N_SKELETON_MASK_DILATE,
#     )
#
#     return data_dict

class BackgroundTransformFromCfg(TransformFromCfg):
    def __init__(self, cfg: CfgNode, device: torch.device | str):
        super(BackgroundTransformFromCfg, self).__init__(cfg, device)

    def _crop1(self, image, masks, skeletons):
        masks = masks.unsqueeze(-1) if masks.ndim == 3 else masks
        image = image.unsqueeze(-1) if image.ndim == 3 else image

        C, X, Y, Z = image.shape
        # ------------ Random Crop 1
        extra = 300
        w = self.CROP_WIDTH + extra if self.CROP_WIDTH + extra <= X else X
        h = self.CROP_HEIGHT + extra if self.CROP_HEIGHT + extra <= Y else Y
        d = self.CROP_DEPTH if self.CROP_DEPTH <= Z else Z

        # select a random point for croping
        x0 = random.randint(0, X - w)
        y0 = random.randint(0, Y - h)
        z0 = random.randint(0, Z - d)

        x1 = x0 + w
        y1 = y0 + h
        z1 = z0 + d

        image = image[:, x0:x1, y0:y1, z0:z1]
        masks = masks[:, x0:x1, y0:y1, z0:z1]

        if image.device != self.DEVICE:
            image = image.to(self.DEVICE)

        if masks.device != self.DEVICE:
            masks = masks.to(self.DEVICE)

        return image, masks, skeletons

    def forward(self, data_dict):
        data_dict['masks'] = torch.ones_like(data_dict['image'])
        data_dict['skeletons'] = {-1: None}

        data_dict = super().forward(data_dict)
        return

    def __repr__(self):
        return f"BackgroundTransformFromCfg[Device:{self.DEVICE}]\ncfg.AUGMENTATION:\n=================\n{self.cfg.AUGMENTATION}]"

#
# def background_transform_from_cfg(
#         data_dict: Dict[str, Tensor], cfg: CfgNode, device: Optional[str] = None
# ) -> DataDict:
#     # Image should be in shape of [C, H, W, D]
#     DEVICE: str = str(data_dict["image"].device) if device is None else device
#
#     # Image should be in shape of [C, H, W, D]
#     CROP_WIDTH = torch.tensor(cfg.AUGMENTATION.CROP_WIDTH, device=DEVICE)
#     CROP_HEIGHT = torch.tensor(cfg.AUGMENTATION.CROP_HEIGHT, device=DEVICE)
#     CROP_DEPTH = torch.tensor(cfg.AUGMENTATION.CROP_DEPTH, device=DEVICE)
#
#     FLIP_RATE = torch.tensor(cfg.AUGMENTATION.FLIP_RATE, device=DEVICE)
#
#     BRIGHTNESS_RATE = torch.tensor(cfg.AUGMENTATION.BRIGHTNESS_RATE, device=DEVICE)
#     BRIGHTNESS_RANGE = torch.tensor(cfg.AUGMENTATION.BRIGHTNESS_RANGE, device=DEVICE)
#
#     NOISE_GAMMA = torch.tensor(cfg.AUGMENTATION.NOISE_GAMMA, device=DEVICE)
#     NOISE_RATE = torch.tensor(cfg.AUGMENTATION.NOISE_RATE, device=DEVICE)
#
#     FILTER_RATE = torch.tensor(0.5, device=DEVICE)
#
#     CONTRAST_RATE = torch.tensor(cfg.AUGMENTATION.CONTRAST_RATE, device=DEVICE)
#     CONTRAST_RANGE = torch.tensor(cfg.AUGMENTATION.CONTRAST_RANGE, device=DEVICE)
#
#     AFFINE_RATE = torch.tensor(cfg.AUGMENTATION.AFFINE_RATE, device=DEVICE)
#     AFFINE_SCALE = torch.tensor(cfg.AUGMENTATION.AFFINE_SCALE, device=DEVICE)
#     AFFINE_YAW = torch.tensor(cfg.AUGMENTATION.AFFINE_YAW, device=DEVICE)
#     AFFINE_SHEAR = torch.tensor(cfg.AUGMENTATION.AFFINE_SHEAR, device=DEVICE)
#
#     image = torch.clone(data_dict["image"])  #
#
#     # ------------ Random Crop 1
#     extra = 300
#     w = (
#         CROP_WIDTH + extra
#         if CROP_WIDTH + extra <= image.shape[1]
#         else torch.tensor(image.shape[1])
#     )
#     h = (
#         CROP_HEIGHT + extra
#         if CROP_HEIGHT + extra <= image.shape[2]
#         else torch.tensor(image.shape[2])
#     )
#     d = CROP_DEPTH if CROP_DEPTH <= image.shape[3] else torch.tensor(image.shape[3])
#
#     shape = torch.tensor(image.shape[1::], device=DEVICE) - torch.tensor(
#         [w, h, d], device=DEVICE
#     )
#     center = torch.tensor(
#         [torch.randint(0, s, (1,)).item() if s > 0 else 0 for s in shape], device=DEVICE
#     )
#
#     # Center that instance
#     x0 = (
#         center[0]
#         .sub(torch.floor(w / 2))
#         .long()
#         .clamp(min=0, max=image.shape[1] - w.item())
#     )
#     y0 = (
#         center[1]
#         .sub(torch.floor(h / 2))
#         .long()
#         .clamp(min=0, max=image.shape[2] - h.item())
#     )
#     z0 = (
#         center[2]
#         .sub(torch.floor(d / 2))
#         .long()
#         .clamp(min=0, max=image.shape[3] - d.item())
#     )
#
#     x1 = x0 + w
#     y1 = y0 + h
#     z1 = z0 + d
#
#     image = image[
#             :, x0.item(): x1.item(), y0.item(): y1.item(), z0.item(): z1.item()
#             ].to(DEVICE)
#
#     # -------------------affine (Cant use baked skeletons)
#     if torch.rand(1, device=DEVICE) < AFFINE_RATE:
#         angle = (AFFINE_YAW[1] - AFFINE_YAW[0]) * torch.rand(
#             1, device=DEVICE
#         ) + AFFINE_YAW[0]
#         shear = (AFFINE_SHEAR[1] - AFFINE_SHEAR[0]) * torch.rand(
#             1, device=DEVICE
#         ) + AFFINE_SHEAR[0]
#         scale = (AFFINE_SCALE[1] - AFFINE_SCALE[0]) * torch.rand(
#             1, device=DEVICE
#         ) + AFFINE_SCALE[0]
#
#         image = ttf.affine(
#             image.permute(0, 3, 1, 2).float(),
#             angle=angle.item(),
#             shear=[float(shear.item())],
#             scale=scale.item(),
#             translate=[0, 0],
#         ).permute(0, 2, 3, 1)
#
#     # ------------ Center Crop 2
#     w = CROP_WIDTH if CROP_WIDTH <= image.shape[1] else torch.tensor(image.shape[1])
#     h = CROP_HEIGHT if CROP_HEIGHT <= image.shape[2] else torch.tensor(image.shape[2])
#     d = CROP_DEPTH if CROP_DEPTH <= image.shape[3] else torch.tensor(image.shape[3])
#
#     center = center - torch.tensor([x0, y0, z0], device=DEVICE)
#
#     # Center that instance
#     x0 = (
#         center[0]
#         .sub(torch.floor(w / 2))
#         .long()
#         .clamp(min=0, max=image.shape[1] - w.item())
#     )
#     y0 = (
#         center[1]
#         .sub(torch.floor(h / 2))
#         .long()
#         .clamp(min=0, max=image.shape[2] - h.item())
#     )
#     z0 = (
#         center[2]
#         .sub(torch.floor(d / 2))
#         .long()
#         .clamp(min=0, max=image.shape[3] - d.item())
#     )
#
#     x1 = x0 + w
#     y1 = y0 + h
#     z1 = z0 + d
#
#     image = image[
#             :, x0.item(): x1.item(), y0.item(): y1.item(), z0.item(): z1.item()
#             ]
#
#     # ------------------- x flip
#     if torch.rand(1, device=DEVICE) < FLIP_RATE:
#         image = image.flip(1)
#
#     # ------------------- y flip
#     if torch.rand(1, device=DEVICE) < FLIP_RATE:
#         image = image.flip(2)
#
#     # ------------------- z flip
#     if torch.rand(1, device=DEVICE) < FLIP_RATE:
#         image = image.flip(3)
#
#     # # ------------------- Random Invert
#     if torch.rand(1, device=DEVICE) < BRIGHTNESS_RATE:
#         image = image.sub(1).mul(-1)
#
#     # ------------------- Adjust Brightness
#     if torch.rand(1, device=DEVICE) < BRIGHTNESS_RATE:
#         # funky looking but FAST
#         val = torch.empty(image.shape[0], device=DEVICE).uniform_(
#             BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1]
#         )
#         image = image.add(val.reshape(image.shape[0], 1, 1, 1)).clamp(0, 1)
#
#     # ------------------- Adjust Contrast
#     if torch.rand(1, device=DEVICE) < CONTRAST_RATE:
#         contrast_val = (CONTRAST_RANGE[1] - CONTRAST_RANGE[0]) * torch.rand(
#             (image.shape[0]), device=DEVICE
#         ) + CONTRAST_RANGE[0]
#
#         for z in range(image.shape[-1]):
#             image[..., z] = ttf.adjust_contrast(
#                 image[..., z], contrast_val[0].item()
#             ).squeeze(0)
#
#     # ------------------- Noise
#     if torch.rand(1, device=DEVICE) < NOISE_RATE:
#         noise = torch.rand(image.shape, device=DEVICE) * NOISE_GAMMA
#
#         image = image.add(noise).clamp(0, 1)
#
#     data_dict["image"] = image
#     data_dict["masks"] = torch.zeros_like(image, device=DEVICE)
#     data_dict["skeletons"]: Dict[int, Tensor] = {-1: torch.empty((0, 3), device=DEVICE)}
#     data_dict["baked_skeleton"] = torch.zeros(
#         (3, image.shape[1], image.shape[2], image.shape[3]), device=DEVICE
#     )
#     data_dict["skele_masks"] = torch.zeros_like(image, device=DEVICE)
#
#
#     return data_dict

#
# @torch.no_grad()
# def merged_transform_3D(
#         data_dict: Dict[str, Tensor],
#         device: Optional[str] = None,
#         bake_skeleton_anisotropy: Tuple[float, float, float] = (1.0, 1.0, 3.0),
#         smooth_skeleton_kernel_size: Tuple[int, int, int] = (3, 3, 1),
# ) -> DataDict:
#     DEVICE: str = str(data_dict["image"].device) if device is None else device
#
#     # Image should be in shape of [C, H, W, D]
#     CROP_WIDTH = torch.tensor([300], device=DEVICE)
#     CROP_HEIGHT = torch.tensor([300], device=DEVICE)
#     CROP_DEPTH = torch.tensor([20], device=DEVICE)
#
#     FLIP_RATE = torch.tensor(0.5, device=DEVICE)
#
#     BRIGHTNESS_RATE = torch.tensor(0.4, device=DEVICE)
#     BRIGHTNESS_RANGE = torch.tensor((-0.1, 0.1), device=DEVICE)
#
#     NOISE_GAMMA = torch.tensor(0.1, device=DEVICE)
#     NOISE_RATE = torch.tensor(0.2, device=DEVICE)
#
#     FILTER_RATE = torch.tensor(0.5, device=DEVICE)
#
#     CONTRAST_RATE = torch.tensor(0.33, device=DEVICE)
#     CONTRAST_RANGE = torch.tensor((0.75, 2.0), device=DEVICE)
#
#     AFFINE_RATE = torch.tensor(0.66, device=DEVICE)
#     AFFINE_SCALE = torch.tensor((0.85, 1.1), device=DEVICE)
#     AFFINE_YAW = torch.tensor((-180, 180), device=DEVICE)
#     AFFINE_SHEAR = torch.tensor((-7, 7), device=DEVICE)
#
#     masks = torch.clone(data_dict["masks"])  # .to(DEVICE))
#     image = torch.clone(data_dict["image"])  #
#
#     skeletons = deepcopy(data_dict["skeletons"])
#     skeletons = {k: v.float().to(DEVICE) for k, v in skeletons.items()}
#
#     # ------------ Random Crop 1
#     extra = 300
#     w = (
#         CROP_WIDTH + extra
#         if CROP_WIDTH + extra <= image.shape[1]
#         else torch.tensor(image.shape[1])
#     )
#     h = (
#         CROP_HEIGHT + extra
#         if CROP_HEIGHT + extra <= image.shape[2]
#         else torch.tensor(image.shape[2])
#     )
#     d = CROP_DEPTH if CROP_DEPTH <= image.shape[3] else torch.tensor(image.shape[3])
#
#     # Randomly select a centroid to center in frame
#     ind: int = torch.randint(
#         len(skeletons.keys()), (1,), dtype=torch.long, device=DEVICE
#     ).item()
#     key: int = list(skeletons)[ind]
#     center: Tensor = skeletons[key].mean(0).squeeze()
#
#     # Center that instance
#     x0 = (
#         center[0]
#         .sub(torch.floor(w / 2))
#         .long()
#         .clamp(min=0, max=image.shape[1] - w.item())
#     )
#     y0 = (
#         center[1]
#         .sub(torch.floor(h / 2))
#         .long()
#         .clamp(min=0, max=image.shape[2] - h.item())
#     )
#     z0 = (
#         center[2]
#         .sub(torch.floor(d / 2))
#         .long()
#         .clamp(min=0, max=image.shape[3] - d.item())
#     )
#
#     x1 = x0 + w
#     y1 = y0 + h
#     z1 = z0 + d
#
#     image = image[
#             :, x0.item(): x1.item(), y0.item(): y1.item(), z0.item(): z1.item()
#             ].to(DEVICE)
#     masks = masks[
#             :, x0.item(): x1.item(), y0.item(): y1.item(), z0.item(): z1.item()
#             ].to(DEVICE)
#
#     # Correct the skeleton positions
#     unique = torch.unique(masks)
#     if -1 not in skeletons:
#         keys = list(skeletons.keys())
#         for k in keys:
#             if not torch.any(unique == k):
#                 skeletons.pop(k)
#             else:
#                 skeletons[k] = skeletons[k] - torch.tensor([x0, y0, z0], device=DEVICE)
#
#     # -------------------affine (Cant use baked skeletons)
#     if torch.rand(1, device=DEVICE) < AFFINE_RATE:
#         angle = (AFFINE_YAW[1] - AFFINE_YAW[0]) * torch.rand(
#             1, device=DEVICE
#         ) + AFFINE_YAW[0]
#         shear = (AFFINE_SHEAR[1] - AFFINE_SHEAR[0]) * torch.rand(
#             1, device=DEVICE
#         ) + AFFINE_SHEAR[0]
#         scale = (AFFINE_SCALE[1] - AFFINE_SCALE[0]) * torch.rand(
#             1, device=DEVICE
#         ) + AFFINE_SCALE[0]
#
#         mat: Tensor = _get_affine_matrix(
#             center=[image.shape[1] / 2, image.shape[2] / 2],
#             angle=-angle.item(),
#             translate=[0.0, 0.0],
#             scale=scale.item(),
#             shear=[float(shear.item()), float(shear.item())],
#             device=str(image.device),
#         )
#
#         # Rotate the skeletons by the affine matrix
#         for k, v in skeletons.items():
#             skeleton_xy = v[:, [0, 1]].permute(1, 0).unsqueeze(0)  # [N, 3] -> [1, 2, N]
#             _ones = torch.ones(
#                 (1, 1, skeleton_xy.shape[-1]), device=DEVICE
#             )  # [1, 1, N]
#             skeleton_xy = torch.cat((skeleton_xy, _ones), dim=1)  # [1, 3, N]
#             rotated_skeleton = mat @ skeleton_xy  # [1,3,N]
#             skeletons[k][:, [0, 1]] = rotated_skeleton[0, [0, 1], :].T.float()
#
#         image = ttf.affine(
#             image.permute(0, 3, 1, 2).float(),
#             angle=angle.item(),
#             shear=[float(shear.item())],
#             scale=scale.item(),
#             translate=[0, 0],
#         ).permute(0, 2, 3, 1)
#
#         unique_before = masks.unique().long().sub(1)
#         unique_before = unique_before[unique_before.ge(0)]
#
#         masks = ttf.affine(
#             masks.permute(0, 3, 1, 2).float(),
#             angle=angle.item(),
#             shear=[float(shear.item())],
#             scale=scale.item(),
#             translate=[0, 0],
#         ).permute(0, 2, 3, 1)
#
#         unique_after = masks.unique().long().sub(1)
#         unique_after = unique_after[unique_after.ge(0)]
#
#         skeletons = {
#             k: v for k, v in skeletons.items() if torch.any(unique_after.eq(k - 1))
#         }
#
#         assert len(skeletons.keys()) > 0, f"{unique_after=}, {unique_before=}"
#
#     # ------------ Center Crop 2
#     w = CROP_WIDTH if CROP_WIDTH <= image.shape[1] else torch.tensor(image.shape[1])
#     h = CROP_HEIGHT if CROP_HEIGHT <= image.shape[2] else torch.tensor(image.shape[2])
#     d = CROP_DEPTH if CROP_DEPTH <= image.shape[3] else torch.tensor(image.shape[3])
#
#     center = center - torch.tensor([x0, y0, z0], device=DEVICE)
#
#     # Center that instance
#     x0 = (
#         center[0]
#         .sub(torch.floor(w / 2))
#         .long()
#         .clamp(min=0, max=image.shape[1] - w.item())
#     )
#     y0 = (
#         center[1]
#         .sub(torch.floor(h / 2))
#         .long()
#         .clamp(min=0, max=image.shape[2] - h.item())
#     )
#     z0 = (
#         center[2]
#         .sub(torch.floor(d / 2))
#         .long()
#         .clamp(min=0, max=image.shape[3] - d.item())
#     )
#
#     x1 = x0 + w
#     y1 = y0 + h
#     z1 = z0 + d
#
#     image = image[
#             :, x0.item(): x1.item(), y0.item(): y1.item(), z0.item(): z1.item()
#             ]
#     masks = masks[
#             :, x0.item(): x1.item(), y0.item(): y1.item(), z0.item(): z1.item()
#             ]
#
#     unique = torch.unique(masks)
#     if -1 not in skeletons:
#         keys = list(skeletons.keys())
#         for k in keys:
#             if not torch.any(unique == k):
#                 skeletons.pop(k)
#             else:
#                 skeletons[k] = skeletons[k] - torch.tensor([x0, y0, z0], device=DEVICE)
#
#     # ------------------- x flip
#     if torch.rand(1, device=DEVICE) < FLIP_RATE:
#         image = image.flip(1)
#         masks = masks.flip(1)
#
#         if -1 not in skeletons:
#             for k, v in skeletons.items():
#                 skeletons[k][:, 0] = image.shape[1] - v[:, 0]
#
#     # ------------------- y flip
#     if torch.rand(1, device=DEVICE) < FLIP_RATE:
#         image = image.flip(2)
#         masks = masks.flip(2)
#         if -1 not in skeletons:
#             for k, v in skeletons.items():
#                 skeletons[k][:, 1] = image.shape[2] - v[:, 1]
#
#     # ------------------- z flip
#     if torch.rand(1, device=DEVICE) < FLIP_RATE:
#         image = image.flip(3)
#         masks = masks.flip(3)
#         if -1 not in skeletons:
#             for k, v in skeletons.items():
#                 skeletons[k][:, 2] = image.shape[3] - v[:, 2]
#
#     # # ------------------- Random Invert
#     if torch.rand(1, device=DEVICE) < BRIGHTNESS_RATE:
#         image = image.sub(1).mul(-1)
#
#     # ------------------- Adjust Brightness
#     if torch.rand(1, device=DEVICE) < BRIGHTNESS_RATE:
#         # funky looking but FAST
#         val = torch.empty(image.shape[0], device=DEVICE).uniform_(
#             BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1]
#         )
#         image = image.add(val.reshape(image.shape[0], 1, 1, 1)).clamp(0, 1)
#
#     # ------------------- Adjust Contrast
#     if torch.rand(1, device=DEVICE) < CONTRAST_RATE:
#         contrast_val = (CONTRAST_RANGE[1] - CONTRAST_RANGE[0]) * torch.rand(
#             (image.shape[0]), device=DEVICE
#         ) + CONTRAST_RANGE[0]
#
#         for z in range(image.shape[-1]):
#             image[..., z] = ttf.adjust_contrast(
#                 image[..., z], contrast_val[0].item()
#             ).squeeze(0)
#
#     # ------------------- Noise
#     if torch.rand(1, device=DEVICE) < NOISE_RATE:
#         noise = torch.rand(image.shape, device=DEVICE) * NOISE_GAMMA
#
#         image = image.add(noise).clamp(0, 1)
#
#     data_dict["image"] = image
#     data_dict["masks"] = masks
#     data_dict["skeletons"] = skeletons
#
#     baked: Tensor = bake_skeleton(
#         masks.squeeze(0).contigous(),
#         skeletons,
#         anisotropy=bake_skeleton_anisotropy,
#         average=True,
#         device=DEVICE,
#     )
#     data_dict["baked_skeleton"]: Union[Tensor, None] = baked
#
#     _, x, y, z = masks.shape
#     data_dict["skele_masks"]: Tensor = skeleton_to_mask(
#         skeletons, (x, y, z), kernel_size=smooth_skeleton_kernel_size, n=1
#     )
#
#     return data_dict
#
#
# def background_transform_3D(
#         data_dict: Dict[str, Tensor], device: Optional[str] = None
# ) -> DataDict:
#     DEVICE: str = str(data_dict["image"].device) if device is None else device
#
#     # Image should be in shape of [C, H, W, D]
#     CROP_WIDTH = torch.tensor([300], device=DEVICE)
#     CROP_HEIGHT = torch.tensor([300], device=DEVICE)
#     CROP_DEPTH = torch.tensor([20], device=DEVICE)
#
#     FLIP_RATE = torch.tensor(0.5, device=DEVICE)
#
#     BRIGHTNESS_RATE = torch.tensor(0.4, device=DEVICE)
#     INVERT_RATE = torch.tensor(0.5, device=DEVICE)
#     BRIGHTNESS_RANGE = torch.tensor((-0.1, 0.1), device=DEVICE)
#
#     NOISE_GAMMA = torch.tensor(0.1, device=DEVICE)
#     NOISE_RATE = torch.tensor(0.2, device=DEVICE)
#
#     FILTER_RATE = torch.tensor(0.5, device=DEVICE)
#
#     CONTRAST_RATE = torch.tensor(0.33, device=DEVICE)
#     CONTRAST_RANGE = torch.tensor((0.75, 2.0), device=DEVICE)
#
#     AFFINE_RATE = torch.tensor(0.66, device=DEVICE)
#     AFFINE_SCALE = torch.tensor((0.85, 1.1), device=DEVICE)
#     AFFINE_YAW = torch.tensor((-180, 180), device=DEVICE)
#     AFFINE_SHEAR = torch.tensor((-7, 7), device=DEVICE)
#
#     image = torch.clone(data_dict["image"])  #
#
#     # ------------ Random Crop 1
#     extra = 300
#     w = (
#         CROP_WIDTH + extra
#         if CROP_WIDTH + extra <= image.shape[1]
#         else torch.tensor(image.shape[1])
#     )
#     h = (
#         CROP_HEIGHT + extra
#         if CROP_HEIGHT + extra <= image.shape[2]
#         else torch.tensor(image.shape[2])
#     )
#     d = CROP_DEPTH if CROP_DEPTH <= image.shape[3] else torch.tensor(image.shape[3])
#
#     shape = torch.tensor(image.shape[1::], device=DEVICE) - torch.tensor(
#         [w, h, d], device=DEVICE
#     )
#     center = torch.tensor(
#         [torch.randint(0, s, (1,)).item() if s > 0 else 0 for s in shape], device=DEVICE
#     )
#
#     # Center that instance
#     x0 = (
#         center[0]
#         .sub(torch.floor(w / 2))
#         .long()
#         .clamp(min=0, max=image.shape[1] - w.item())
#     )
#     y0 = (
#         center[1]
#         .sub(torch.floor(h / 2))
#         .long()
#         .clamp(min=0, max=image.shape[2] - h.item())
#     )
#     z0 = (
#         center[2]
#         .sub(torch.floor(d / 2))
#         .long()
#         .clamp(min=0, max=image.shape[3] - d.item())
#     )
#
#     x1 = x0 + w
#     y1 = y0 + h
#     z1 = z0 + d
#
#     image = image[
#             :, x0.item(): x1.item(), y0.item(): y1.item(), z0.item(): z1.item()
#             ].to(DEVICE)
#
#     # -------------------affine (Cant use baked skeletons)
#     if torch.rand(1, device=DEVICE) < AFFINE_RATE:
#         angle = (AFFINE_YAW[1] - AFFINE_YAW[0]) * torch.rand(
#             1, device=DEVICE
#         ) + AFFINE_YAW[0]
#         shear = (AFFINE_SHEAR[1] - AFFINE_SHEAR[0]) * torch.rand(
#             1, device=DEVICE
#         ) + AFFINE_SHEAR[0]
#         scale = (AFFINE_SCALE[1] - AFFINE_SCALE[0]) * torch.rand(
#             1, device=DEVICE
#         ) + AFFINE_SCALE[0]
#
#         image = ttf.affine(
#             image.permute(0, 3, 1, 2).float(),
#             angle=angle.item(),
#             shear=[float(shear.item())],
#             scale=scale.item(),
#             translate=[0, 0],
#         ).permute(0, 2, 3, 1)
#
#     # ------------ Center Crop 2
#     w = CROP_WIDTH if CROP_WIDTH <= image.shape[1] else torch.tensor(image.shape[1])
#     h = CROP_HEIGHT if CROP_HEIGHT <= image.shape[2] else torch.tensor(image.shape[2])
#     d = CROP_DEPTH if CROP_DEPTH <= image.shape[3] else torch.tensor(image.shape[3])
#
#     center = center - torch.tensor([x0, y0, z0], device=DEVICE)
#
#     # Center that instance
#     x0 = (
#         center[0]
#         .sub(torch.floor(w / 2))
#         .long()
#         .clamp(min=0, max=image.shape[1] - w.item())
#     )
#     y0 = (
#         center[1]
#         .sub(torch.floor(h / 2))
#         .long()
#         .clamp(min=0, max=image.shape[2] - h.item())
#     )
#     z0 = (
#         center[2]
#         .sub(torch.floor(d / 2))
#         .long()
#         .clamp(min=0, max=image.shape[3] - d.item())
#     )
#
#     x1 = x0 + w
#     y1 = y0 + h
#     z1 = z0 + d
#
#     image = image[
#             :, x0.item(): x1.item(), y0.item(): y1.item(), z0.item(): z1.item()
#             ]
#
#     # ------------------- x flip
#     if torch.rand(1, device=DEVICE) < FLIP_RATE:
#         image = image.flip(1)
#
#     # ------------------- y flip
#     if torch.rand(1, device=DEVICE) < FLIP_RATE:
#         image = image.flip(2)
#
#     # ------------------- z flip
#     if torch.rand(1, device=DEVICE) < FLIP_RATE:
#         image = image.flip(3)
#
#     # # ------------------- Random Invert
#     if torch.rand(1, device=DEVICE) < INVERT_RATE:
#         image = image.sub(1).mul(-1)
#
#     # ------------------- Adjust Brightness
#     if torch.rand(1, device=DEVICE) < BRIGHTNESS_RATE:
#         # funky looking but FAST
#         val = torch.empty(image.shape[0], device=DEVICE).uniform_(
#             BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1]
#         )
#         image = image.add(val.reshape(image.shape[0], 1, 1, 1)).clamp(0, 1)
#
#     # ------------------- Adjust Contrast
#     if torch.rand(1, device=DEVICE) < CONTRAST_RATE:
#         contrast_val = (CONTRAST_RANGE[1] - CONTRAST_RANGE[0]) * torch.rand(
#             (image.shape[0]), device=DEVICE
#         ) + CONTRAST_RANGE[0]
#
#         for z in range(image.shape[-1]):
#             image[..., z] = ttf.adjust_contrast(
#                 image[..., z], contrast_val[0].item()
#             ).squeeze(0)
#
#     # ------------------- Noise
#     if torch.rand(1, device=DEVICE) < NOISE_RATE:
#         noise = torch.rand(image.shape, device=DEVICE) * NOISE_GAMMA
#
#         image = image.add(noise).clamp(0, 1)
#
#     data_dict["image"] = image
#     data_dict["masks"] = torch.zeros_like(image, device=DEVICE)
#     data_dict["skeletons"]: Dict[int, Tensor] = {-1: torch.empty((0, 3), device=DEVICE)}
#     data_dict["baked_skeleton"] = torch.zeros(
#         (3, image.shape[1], image.shape[2], image.shape[3]), device=DEVICE
#     )
#     data_dict["skele_masks"] = torch.zeros_like(image, device=DEVICE)
#
#     return data_dict


if __name__ == "__main__":
    import torch.distributed as dist
    import torch.optim.lr_scheduler

    import skimage.io as io
    import matplotlib.pyplot as plt
    import skoots.train.generate_skeletons
    from skoots.lib.skeleton import skeleton_to_mask

    img = torch.from_numpy(
        io.imread(
            "/home/chris/Dropbox (Partners HealthCare)/skoots/tests/test_data/hide_validate_skeleton_instance_mask.tif"
        ).astype(float)
    )

    mask = torch.from_numpy(
        io.imread(
            "/home/chris/Dropbox (Partners HealthCare)/skoots/tests/test_data/hide_validate.labels.tif"
        ).astype(float)
    )

    img = img[:, 850:1100, 690:890].permute(1, 2, 0)
    mask = mask[:, 850:1100, 690:890].permute(1, 2, 0)

    skeletons = skoots.train.generate_skeletons.calculate_skeletons(mask, torch.tensor((0.3, 0.3, 1.0)))

    x, y, z = img.shape

    img = img.view(1, 1, x, y, z)
    mask = mask.view(1, 1, x, y, z)

    og = img.clone()

    og_skl = skeleton_to_mask(skeletons, (x, y, z))

    img, mask, skeletons = elastic_deform(img, mask, skeletons)

    skl_msk = skeleton_to_mask(skeletons, (x, y, z))

    plt.imshow(og[0, 0, :, :, 14])
    plt.show()

    plt.imshow(og_skl[0, :, :, 14])
    plt.show()

    plt.imshow(img[0, 0, :, :, 14])
    plt.show()

    plt.imshow(skl_msk[0, :, :, 14])
    plt.show()
