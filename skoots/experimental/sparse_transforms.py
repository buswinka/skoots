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

from skoots.train.merged_transform import elastic_deform, _get_affine_matrix
from skoots.lib.custom_types import SparseDataDict

from torch import Tensor
from yacs.config import CfgNode

import logging


class SparseTransformFromCfg(nn.Module):
    def __init__(self, cfg: CfgNode, device: torch.device, scale: float = 255.0):
        super(SparseTransformFromCfg, self).__init__()
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

        self.ELASTIC_GRID_SHAPE = cfg.AUGMENTATION.ELASTIC_GRID_SHAPE
        self.ELASTIC_GRID_MAGNITUDE = cfg.AUGMENTATION.ELASTIC_GRID_MAGNITUDE
        self.ELASTIC_RATE = cfg.AUGMENTATION.ELASTIC_RATE

        self.BAKE_SKELETON_ANISOTROPY = cfg.AUGMENTATION.BAKE_SKELETON_ANISOTROPY

        self._center = None
        self._xyz = (None, None, None)

    def _identity(self, *args):
        return args if len(args) > 1 else args[0]

    def _elastic(self, image, masks, skel_mask, skeletons):
        image = image.unsqueeze(0)
        masks = masks.unsqueeze(0)
        skel_mask = skel_mask.unsqueeze(0)

        warped_image, warped_masks, warped_skel_mask, skeletons = elastic_deform(
            image, masks, skel_mask, skeleton=skeletons
        )

        return warped_image.squeeze(0), warped_masks.squeeze(0), warped_skel_mask.squeeze(0), skeletons

    def _crop1(self, image, masks, skel_mask, skeletons):
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
        try:
            x0 = self._center[0].sub(w // 2).long().clamp(min=0, max=image.shape[1] - w)
            y0 = self._center[1].sub(h // 2).long().clamp(min=0, max=image.shape[2] - h)
            z0 = self._center[2].sub(d // 2).long().clamp(min=0, max=image.shape[3] - d)
        except:
            print(f'{key}, {skeletons[key].shape}, {self._center}')

        self._xyz = (x0, y0, z0)

        x1 = x0 + w
        y1 = y0 + h
        z1 = z0 + d

        logging.debug(
            f"TransformFromCfg.forward() | applying crop 1 [{x0}:{x1}, {y0}:{y1}, {z0}:{z1}]"
        )

        image = image[:, x0:x1, y0:y1, z0:z1].clone()
        masks = masks[:, x0:x1, y0:y1, z0:z1].clone()

        skel_mask = skel_mask[:, x0:x1, y0:y1, z0:z1].clone()

        # Correct the skeleton positions
        pruned_skeletons = {}
        if -1 not in skeletons:
            keys = list(skeletons.keys())
            for k in keys:
            #     if torch.any(unique == k):

                pruned_skeletons[k] = skeletons[k].clone() - torch.tensor(
                    [x0, y0, z0], device=image.device
                )

        else:
            raise RuntimeError

        # send to device
        if image.device != self.DEVICE:
            image = image.to(self.DEVICE)

        if masks.device != self.DEVICE:
            masks = masks.to(self.DEVICE)

        for k, v in pruned_skeletons.items():
            pruned_skeletons[k] = v.float().to(self.DEVICE)

        return image, masks, skel_mask, pruned_skeletons

    def _affine(self, image, masks, skel_mask, skeletons):
        angle = (self.AFFINE_YAW[1] - self.AFFINE_YAW[0]) * torch.rand(
            1, device=self.DEVICE
        ) + self.AFFINE_YAW[0]
        shear = (self.AFFINE_SHEAR[1] - self.AFFINE_SHEAR[0]) * torch.rand(
            1, device=self.DEVICE
        ) + self.AFFINE_SHEAR[0]
        scale = (self.AFFINE_SCALE[1] - self.AFFINE_SCALE[0]) * torch.rand(
            1, device=self.DEVICE
        ) + self.AFFINE_SCALE[0]

        logging.debug(
            f"TransformFromCfg.forward() | applying affine transform with {angle=}, {shear=}, {scale=}"
        )

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

        skel_mask = ttf.affine(
            skel_mask.permute(0, 3, 1, 2).float(),
            angle=angle.item(),
            shear=float(shear.item()),
            scale=scale.item(),
            translate=[0, 0],
        ).permute(0, 2, 3, 1)

        unique_after = masks.unique().long().sub(1)
        unique_after = unique_after[unique_after.ge(0)]

        # skeletons = {
        #     k: v for k, v in skeletons.items() if torch.any(unique_after.eq(k - 1))
        # }

        return image, masks, skel_mask, skeletons

    def _crop2(self, image, masks, skel_mask, skeletons):
        C, X, Y, Z = image.shape
        w = self.CROP_WIDTH if self.CROP_WIDTH < X else X
        h = self.CROP_HEIGHT if self.CROP_HEIGHT < Y else Y
        d = self.CROP_DEPTH if self.CROP_DEPTH < Z else Z

        x0, y0, z0 = self._xyz

        self._center = self._center - torch.tensor([x0, y0, z0], device=self.DEVICE)

        # Center that instance
        x0 = self._center[0].sub(w // 2).long().clamp(min=0, max=image.shape[1] - w)
        y0 = self._center[1].sub(h // 2).long().clamp(min=0, max=image.shape[2] - h)
        z0 = self._center[2].sub(d // 2).long().clamp(min=0, max=image.shape[3] - d)

        x1 = x0 + w
        y1 = y0 + h
        z1 = z0 + d

        logging.debug(
            f"TransformFromCfg.forward() | applying crop 2 [{x0}:{x1}, {y0}:{y1}, {z0}:{z1}]"
        )

        image = image[:, x0:x1, y0:y1, z0:z1]
        masks = masks[:, x0:x1, y0:y1, z0:z1]
        skel_mask = skel_mask[:, x0:x1, y0:y1, z0:z1]

        unique = torch.unique(masks)
        if -1 not in skeletons:
            keys = list(skeletons.keys())
            for k in keys:
            #     if not torch.any(unique == k):
            #         skeletons.pop(k)
            #     else:
                toinsert = skeletons[k] - torch.tensor(
                        [x0, y0, z0], device=self.DEVICE
                    )
                if toinsert.max() < w + 30:
                    skeletons[k] = toinsert
                elif toinsert.min() > -20:
                    skeletons[k] = toinsert

        return image, masks, skel_mask, skeletons

    def _flipX(self, image, masks, skel_mask, skeletons):
        image = image.flip(1)
        masks = masks.flip(1)
        skel_mask = skel_mask.flip(1)

        if -1 not in skeletons:
            for k, v in skeletons.items():
                skeletons[k][:, 0] = image.shape[1] - v[:, 0]
        return image, masks, skel_mask, skeletons

    def _flipY(self, image, masks, skel_mask, skeletons):
        image = image.flip(2)
        masks = masks.flip(2)
        skel_mask = skel_mask.flip(2)
        if -1 not in skeletons:
            for k, v in skeletons.items():
                skeletons[k][:, 1] = image.shape[2] - v[:, 1]
        return image, masks, skel_mask, skeletons

    def _flipZ(self, image, masks, skel_mask, skeletons):
        image = image.flip(3)
        masks = masks.flip(3)
        skel_mask = skel_mask.flip(3)
        if -1 not in skeletons:
            for k, v in skeletons.items():
                skeletons[k][:, 2] = image.shape[3] - v[:, 2]
        return image, masks, skel_mask, skeletons

    def _invert(self, image):
        image.sub_(255).mul_(-1)
        return image

    def _brightness(self, image):
        val = random.uniform(*self.BRIGHTNESS_RANGE)
        logging.debug(
            f"TransformFromCfg.forward() | adjusting brightness: {val=}"
        )
        # in place ok because flip always returns a copy
        image = image.add(val)
        image.clamp_(0, 255)
        return image

    def _contrast(self, image):
        contrast_val = random.uniform(*self.CONTRAST_RANGE)
        logging.debug(
            f"TransformFromCfg.forward() | adjusting contrast: {contrast_val=}"
        )
        # [ C, X, Y, Z ] -> [Z, C, X, Y]
        image = image.div(255)
        image = ttf.adjust_contrast(image.permute(3, 0, 1, 2), contrast_val).permute(
            1, 2, 3, 0
        )
        image = image.mul(255)


        return image

    def _noise(self, image):
        logging.debug(f"TransformFromCfg.forward() | adding noise with {self.NOISE_GAMMA=}")
        noise = torch.rand(image.shape, device=self.DEVICE) * self.NOISE_GAMMA
        image = image.add(noise)
        return image

    def _normalize(self, image):
        # mean = image.float().mean()
        # std = image.float().std()
        mean = image.float().mean() if not self.dataset_mean else self.dataset_mean
        std = image.float().std() if not self.dataset_std else self.dataset_std

        image = image.float().sub(mean).div(std)
        return image

    def set_dataset_mean(self, mean):
        self.dataset_mean = mean
        return self

    def set_dataset_std(self, std):
        self.dataset_std = std
        return self

    @torch.no_grad()
    def forward(self, data_dict: SparseDataDict) -> SparseDataDict:
        assert "background" in data_dict, 'keyword "background" not in data_dict'
        assert "skele_masks" in data_dict, 'keyword "skele_masks" not in data_dict'
        assert "image" in data_dict, 'keyword "image" not in data_dict'
        assert "skeletons" in data_dict, 'keyword "skeletons" not in data_dict'

        logging.debug(
            f"TransformFromCfg.forward() | starting transforms on device: {self.DEVICE}"
        )

        logging.debug("TransformFromCfg.forward() | applying prefix function")
        data_dict: SparseDataDict = self.prefix_function(data_dict)

        masks = data_dict["background"]
        image = data_dict["image"]
        skel_mask = data_dict["skele_masks"]
        skeletons = data_dict["skeletons"]

        spatial_dims = masks.ndim - 1

        assert len(skeletons.keys()) > 0
        assert skel_mask.shape == masks.shape
        assert image.shape == masks.shape

        image, masks, skel_mask, skeletons = self._crop1(
            image, masks, skel_mask, skeletons
        )

        # scale: int = 2 ** 16 if image.max() > 256 else 255  # Our images might be 16 bit, or 8 bit
        # scale = scale if image.max() > 1 else 1.0

        if random.random() < self.ELASTIC_RATE:
            image, masks, skel_mask, skeletons = self._elastic(
                image, masks, skel_mask, skeletons
            )

        # affine
        if random.random() < self.AFFINE_RATE:
            image, masks, skel_mask, skeletons = self._affine(
                image, masks, skel_mask, skeletons
            )

        # ------------ Center Crop 2
        image, masks, skel_mask, skeletons = self._crop2(
            image, masks, skel_mask, skeletons
        )

        # ------------------- x flip
        if random.random() < self.FLIP_RATE:
            logging.debug("TransformFromCfg.forward() | flipping in x")
            image, masks, skel_mask, skeletons = self._flipX(image, masks, skel_mask, skeletons)

        # ------------------- y flip
        if random.random() < self.FLIP_RATE:
            logging.debug("TransformFromCfg.forward() | flipping in y")
            image, masks, skel_mask, skeletons = self._flipY(image, masks, skel_mask, skeletons)

        # ------------------- z flip
        if random.random() < self.FLIP_RATE:
            logging.debug("TransformFromCfg.forward() | flipping in z")
            image, masks, skel_mask, skeletons = self._flipZ(image, masks, skel_mask, skeletons)

        # # ------------------- Random Invert
        if random.random() < self.BRIGHTNESS_RATE:
            logging.debug("TransformFromCfg.forward() | inverting")
            image = self._invert(image)

        # ------------------- Adjust Brightness
        if random.random() < self.BRIGHTNESS_RATE:
            image = self._brightness(image)

        # ------------------- Adjust Contrast
        if random.random() < self.CONTRAST_RATE:
            image = self._contrast(image)

        # ------------------- Noise
        if random.random() < self.NOISE_RATE:
            image = self._noise(image)

        logging.debug(
            f"TransformFromCfg.forward() | normalizing with {self.dataset_mean=}, {self.dataset_std=}"
        )
        image = self._normalize(image)

        data_dict["image"] = image
        data_dict["background"] = masks
        data_dict["skele_masks"] = skel_mask
        data_dict["skeletons"] = skeletons

        data_dict: SparseDataDict = self.posfix_function(data_dict)

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
        return f"SparseTransformFromCfg[Device:{self.DEVICE}]\ncfg.AUGMENTATION:\n=================\n{self.cfg.AUGMENTATION}]"
