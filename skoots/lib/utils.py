from functools import partial
from numbers import Number
from typing import List, Tuple, Optional

import bism.backends
import bism.modules
import torch
import torch.nn as nn
from bism.models.spatial_embedding import SpatialEmbedding
from torch import Tensor
from yacs.config import CfgNode


def cfg_to_bism_model(cfg: CfgNode) -> nn.Module:
    """utility function to get a bism model from cfg"""

    _valid_model_constructors = {
        "bism_unext": bism.backends.unext.UNeXT_3D,
        "bism_unet": bism.backends.unet.UNet_3D,
    }

    _valid_model_blocks = {"block3d": bism.modules.convnext_block.Block3D}

    _valid_upsample_layers = {
        "upsamplelayer3d": bism.modules.upsample_layer.UpSampleLayer3D
    }

    _valid_normalization = {
        "layernorm": partial(
            bism.modules.layer_norm.LayerNorm, data_format="channels_first"
        )
    }

    _valid_activations = {
        "gelu": torch.nn.GELU,
        "relu": torch.nn.ReLU,
        "silu": torch.nn.SiLU,
        "selu": torch.nn.SELU,
    }

    _valid_concat_blocks = {"concatconv3d": bism.modules.concat.ConcatConv3D}

    model_config = [
        cfg.MODEL.DIMS,
        cfg.MODEL.DEPTHS,
        cfg.MODEL.KERNEL_SIZE,
        cfg.MODEL.DROP_PATH_RATE,
        cfg.MODEL.LAYER_SCALE_INIT_VALUE,
        cfg.MODEL.ACTIVATION,
        cfg.MODEL.BLOCK,
        cfg.MODEL.CONCAT_BLOCK,
        cfg.MODEL.UPSAMPLE_BLOCK,
        cfg.MODEL.NORMALIZATION,
    ]

    model_kwargs = [
        "dims",
        "depths",
        "kernel_size",
        "drop_path_rate",
        "layer_scale_init_value",
        "activation",
        "block",
        "concat_conv",
        "upsample_layer",
        "normalization",
    ]

    valid_dicts = [
        None,
        None,
        None,
        None,
        None,
        _valid_activations,
        _valid_model_blocks,
        _valid_concat_blocks,
        _valid_upsample_layers,
        _valid_normalization,
    ]

    kwarg = {}
    for kw, config, vd in zip(model_kwargs, model_config, valid_dicts):
        if vd is not None:
            if config in vd:
                kwarg[kw] = vd[config]
            else:
                raise RuntimeError(
                    f"{config} is not a valid config option for {kw}. Valid options are: {vd.keys()}"
                )
        else:
            kwarg[kw] = config

    if cfg.MODEL.ARCHITECTURE in _valid_model_constructors:
        backbone = _valid_model_constructors[cfg.MODEL.ARCHITECTURE]
    else:
        raise RuntimeError(
            f"{cfg.MODEL.ARCHITECTURE} is not a valid model constructor, valid options are: {_valid_model_constructors.keys()}"
        )

    backbone = backbone(cfg.MODEL.IN_CHANNELS, cfg.MODEL.OUT_CHANNELS, **kwarg)
    model = SpatialEmbedding(backbone)

    return model


def calculate_indexes(
    pad_size: int, eval_image_size: int, image_shape: int, padded_image_shape: int
) -> List[List[int]]:
    """
    This calculates indexes for the complete evaluation of an arbitrarily large image by unet.
    each index is offset by eval_image_size, but has a width of eval_image_size + pad_size * 2.
    Unet needs padding on each side of the evaluation to ensure only full convolutions are used
    in generation of the final mask. If the algorithm cannot evenly create indexes for
    padded_image_shape, an additional index is added at the end of equal size.



    :param pad_size: int corresponding to the amount of padding on each side of the
                     padded image
    :param eval_image_size: int corresponding to the shape of the image to be used for
                            the final mask
    :param image_shape: int Shape of image before padding is applied
    :param padded_image_shape: int Shape of image after padding is applied

    :return: List of lists corresponding to the indexes
    """

    # We want to account for when the eval image size is super big, just return index for the whole image.
    if eval_image_size + (2 * pad_size) > image_shape:
        return [[0, image_shape - 1]]

    try:
        ind_list = torch.arange(0, image_shape, eval_image_size)
    except RuntimeError:
        raise RuntimeError(
            f"Calculate_indexes has incorrect values {pad_size} | {image_shape} | {eval_image_size}:\n"
            f"You are likely trying to have a chunk smaller than the set evaluation image size. "
            "Please decrease number of chunks."
        )
    ind = []
    for i, z in enumerate(ind_list):
        if i == 0:
            continue
        z1 = int(ind_list[i - 1])
        z2 = int(z - 1) + (2 * pad_size)
        if z2 < padded_image_shape:
            ind.append([z1, z2])
        else:
            break
    if (
        not ind
    ):  # Sometimes z is so small the first part doesnt work. Check if z_ind is empty, if it is do this!!!
        z1 = 0
        z2 = eval_image_size + pad_size * 2
        ind.append([z1, z2])
        ind.append(
            [padded_image_shape - (eval_image_size + pad_size * 2), padded_image_shape]
        )
    else:  # we always add at the end to ensure that the whole thing is covered.
        z1 = padded_image_shape - (eval_image_size + pad_size * 2)
        z2 = padded_image_shape - 1
        ind.append([z1, z2])
    return ind


def get_dtype_offset(dtype: str = "uint16", image_max: Optional[Number] = None) -> int:
    """
    Returns the scaling factor such that
    such that :math:`\frac{image}{f} \in [0, ..., 1]`

    Supports: uint16, uint8, uint12, float64

    :param dtype: String representation of the data type.
    :param image_max: Returns image max if the dtype is not supported.
    :return: Integer scale factor
    """

    encoding = {
        "uint16": 2**16,
        "uint8": 2**8,
        "uint12": 2**12,
        "float64": 1,
    }
    dtype = str(dtype)
    if dtype in encoding:
        scale = encoding[dtype]
    else:
        print(
            f"\x1b[1;31;40m"
            + f"ERROR: Unsupported dtype: {dtype}. Currently support: {[k for k in encoding]}"
            + "\x1b[0m"
        )
        scale = None
        if image_max:
            print(f"Appropriate Scale Factor inferred from image maximum: {image_max}")
            if image_max <= 256:
                scale = 256
            else:
                scale = image_max
    return scale


@torch.jit.script
def _crop3d(img: Tensor, x: int, y: int, z: int, w: int, h: int, d: int) -> Tensor:
    """
    torch scriptable function which crops an image

    :param img: torch.Tensor image of shape [C, X, Y, Z]
    :param x: x coord of crop box
    :param y: y coord of crop box
    :param z: z coord of crop box
    :param w: width of crop box
    :param h: height of crop box
    :param d: depth of crop box
    :return:
    """
    return img[..., x : x + w, y : y + h, z : z + d]


@torch.jit.script
def _crop2d(img: Tensor, x: int, y: int, w: int, h: int) -> Tensor:
    """
    torch scriptable function which crops an image

    :param img: torch.Tensor image of shape [C, X, Y, Z]
    :param x: x coord of crop box
    :param y: y coord of crop box
    :param z: z coord of crop box
    :param w: width of crop box
    :param h: height of crop box
    :param d: depth of crop box
    :return:
    """

    return img[..., x : x + w, y : y + h]


@torch.jit.script
def crop_to_identical_size(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Crops Tensor a to the shape of Tensor b, then crops Tensor b to the shape of Tensor a.

    :param a: torch.
    :param b:
    :return:
    """
    if a.ndim < 3:
        raise RuntimeError(
            "Only supports tensors with minimum 3dimmensions and shape [..., X, Y, Z]"
        )

    a = _crop3d(a, x=0, y=0, z=0, w=b.shape[-3], h=b.shape[-2], d=b.shape[-1])
    b = _crop3d(b, x=0, y=0, z=0, w=a.shape[-3], h=a.shape[-2], d=a.shape[-1])
    return a, b


@torch.jit.script
def cantor2(a: Tensor, b: Tensor) -> Tensor:
    "Hashes two combination of tensors together"
    return 0.5 * (a + b) * (a + b + 1) + b


@torch.jit.script
def cantor3(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    "Hashes three combination of tensors together"
    return (0.5 * (cantor2(a, b) + c) * (cantor2(a, b) + c + 1) + c).int()


@torch.jit.script
def identical_rows(a: Tensor, b: Tensor) -> Tensor:
    """
    Given two matrices of identical size, determines the indices of identical rows.

    :param a: [N, 3] torch.Tensor
    :param b: [N, 3] torch.Tensor

    :return: Indicies of identical rows
    """
    a = cantor3(a[:, 0], a[:, 1], a[:, 2])
    b = cantor3(b[:, 0], b[:, 1], b[:, 2])

    _inv = torch.reciprocal(b)

    ind = torch.sum(a.unsqueeze(-1) @ _inv.unsqueeze(0), dim=0).gt(0)

    return ind


def get_cached_ball_coords(device: str) -> Tensor:
    out = [
        [0, 3, 3],
        [1, 1, 2],
        [1, 1, 3],
        [1, 1, 4],
        [1, 2, 1],
        [1, 2, 2],
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 5],
        [1, 3, 1],
        [1, 3, 2],
        [1, 3, 3],
        [1, 3, 4],
        [1, 3, 5],
        [1, 4, 1],
        [1, 4, 2],
        [1, 4, 3],
        [1, 4, 4],
        [1, 4, 5],
        [1, 5, 2],
        [1, 5, 3],
        [1, 5, 4],
        [2, 1, 1],
        [2, 1, 2],
        [2, 1, 3],
        [2, 1, 4],
        [2, 1, 5],
        [2, 2, 1],
        [2, 2, 2],
        [2, 2, 3],
        [2, 2, 4],
        [2, 2, 5],
        [2, 3, 1],
        [2, 3, 2],
        [2, 3, 3],
        [2, 3, 4],
        [2, 3, 5],
        [2, 4, 1],
        [2, 4, 2],
        [2, 4, 3],
        [2, 4, 4],
        [2, 4, 5],
        [2, 5, 1],
        [2, 5, 2],
        [2, 5, 3],
        [2, 5, 4],
        [2, 5, 5],
        [3, 0, 3],
        [3, 1, 1],
        [3, 1, 2],
        [3, 1, 3],
        [3, 1, 4],
        [3, 1, 5],
        [3, 2, 1],
        [3, 2, 2],
        [3, 2, 3],
        [3, 2, 4],
        [3, 2, 5],
        [3, 3, 0],
        [3, 3, 1],
        [3, 3, 2],
        [3, 3, 3],
        [3, 3, 4],
        [3, 3, 5],
        [3, 3, 6],
        [3, 4, 1],
        [3, 4, 2],
        [3, 4, 3],
        [3, 4, 4],
        [3, 4, 5],
        [3, 5, 1],
        [3, 5, 2],
        [3, 5, 3],
        [3, 5, 4],
        [3, 5, 5],
        [3, 6, 3],
        [4, 1, 1],
        [4, 1, 2],
        [4, 1, 3],
        [4, 1, 4],
        [4, 1, 5],
        [4, 2, 1],
        [4, 2, 2],
        [4, 2, 3],
        [4, 2, 4],
        [4, 2, 5],
        [4, 3, 1],
        [4, 3, 2],
        [4, 3, 3],
        [4, 3, 4],
        [4, 3, 5],
        [4, 4, 1],
        [4, 4, 2],
        [4, 4, 3],
        [4, 4, 4],
        [4, 4, 5],
        [4, 5, 1],
        [4, 5, 2],
        [4, 5, 3],
        [4, 5, 4],
        [4, 5, 5],
        [5, 1, 2],
        [5, 1, 3],
        [5, 1, 4],
        [5, 2, 1],
        [5, 2, 2],
        [5, 2, 3],
        [5, 2, 4],
        [5, 2, 5],
        [5, 3, 1],
        [5, 3, 2],
        [5, 3, 3],
        [5, 3, 4],
        [5, 3, 5],
        [5, 4, 1],
        [5, 4, 2],
        [5, 4, 3],
        [5, 4, 4],
        [5, 4, 5],
        [5, 5, 2],
        [5, 5, 3],
        [5, 5, 4],
        [6, 3, 3],
    ]
    return torch.tensor(out, device=device).T

def get_cached_disk_coords(device: str) -> Tensor:
    out = [
        [0, 7, 0],
        [1, 4, 0],
        [1, 5, 0],
        [1, 6, 0],
        [1, 7, 0],
        [1, 8, 0],
        [1, 9, 0],
        [1, 10, 0],
        [2, 3, 0],
        [2, 4, 0],
        [2, 5, 0],
        [2, 6, 0],
        [2, 7, 0],
        [2, 8, 0],
        [2, 9, 0],
        [2, 10, 0],
        [2, 11, 0],
        [3, 2, 0],
        [3, 3, 0],
        [3, 4, 0],
        [3, 5, 0],
        [3, 6, 0],
        [3, 7, 0],
        [3, 8, 0],
        [3, 9, 0],
        [3, 10, 0],
        [3, 11, 0],
        [3, 12, 0],
        [4, 1, 0],
        [4, 2, 0],
        [4, 3, 0],
        [4, 4, 0],
        [4, 5, 0],
        [4, 6, 0],
        [4, 7, 0],
        [4, 8, 0],
        [4, 9, 0],
        [4, 10, 0],
        [4, 11, 0],
        [4, 12, 0],
        [4, 13, 0],
        [5, 1, 0],
        [5, 2, 0],
        [5, 3, 0],
        [5, 4, 0],
        [5, 5, 0],
        [5, 6, 0],
        [5, 7, 0],
        [5, 8, 0],
        [5, 9, 0],
        [5, 10, 0],
        [5, 11, 0],
        [5, 12, 0],
        [5, 13, 0],
        [6, 1, 0],
        [6, 2, 0],
        [6, 3, 0],
        [6, 4, 0],
        [6, 5, 0],
        [6, 6, 0],
        [6, 7, 0],
        [6, 8, 0],
        [6, 9, 0],
        [6, 10, 0],
        [6, 11, 0],
        [6, 12, 0],
        [6, 13, 0],
        [7, 0, 0],
        [7, 1, 0],
        [7, 2, 0],
        [7, 3, 0],
        [7, 4, 0],
        [7, 5, 0],
        [7, 6, 0],
        [7, 7, 0],
        [7, 8, 0],
        [7, 9, 0],
        [7, 10, 0],
        [7, 11, 0],
        [7, 12, 0],
        [7, 13, 0],
        [7, 14, 0],
        [8, 1, 0],
        [8, 2, 0],
        [8, 3, 0],
        [8, 4, 0],
        [8, 5, 0],
        [8, 6, 0],
        [8, 7, 0],
        [8, 8, 0],
        [8, 9, 0],
        [8, 10, 0],
        [8, 11, 0],
        [8, 12, 0],
        [8, 13, 0],
        [9, 1, 0],
        [9, 2, 0],
        [9, 3, 0],
        [9, 4, 0],
        [9, 5, 0],
        [9, 6, 0],
        [9, 7, 0],
        [9, 8, 0],
        [9, 9, 0],
        [9, 10, 0],
        [9, 11, 0],
        [9, 12, 0],
        [9, 13, 0],
        [10, 1, 0],
        [10, 2, 0],
        [10, 3, 0],
        [10, 4, 0],
        [10, 5, 0],
        [10, 6, 0],
        [10, 7, 0],
        [10, 8, 0],
        [10, 9, 0],
        [10, 10, 0],
        [10, 11, 0],
        [10, 12, 0],
        [10, 13, 0],
        [11, 2, 0],
        [11, 3, 0],
        [11, 4, 0],
        [11, 5, 0],
        [11, 6, 0],
        [11, 7, 0],
        [11, 8, 0],
        [11, 9, 0],
        [11, 10, 0],
        [11, 11, 0],
        [11, 12, 0],
        [12, 3, 0],
        [12, 4, 0],
        [12, 5, 0],
        [12, 6, 0],
        [12, 7, 0],
        [12, 8, 0],
        [12, 9, 0],
        [12, 10, 0],
        [12, 11, 0],
        [13, 4, 0],
        [13, 5, 0],
        [13, 6, 0],
        [13, 7, 0],
        [13, 8, 0],
        [13, 9, 0],
        [13, 10, 0],
        [14, 7, 0],
        [0, 3, 1],
        [1, 1, 1],
        [1, 2, 1],
        [1, 3, 1],
        [1, 4, 1],
        [1, 5, 1],
        [2, 1, 1],
        [2, 2, 1],
        [2, 3, 1],
        [2, 4, 1],
        [2, 5, 1],
        [3, 0, 1],
        [3, 1, 1],
        [3, 2, 1],
        [3, 3, 1],
        [3, 4, 1],
        [3, 5, 1],
        [3, 6, 1],
        [4, 1, 1],
        [4, 2, 1],
        [4, 3, 1],
        [4, 4, 1],
        [4, 5, 1],
        [5, 1, 1],
        [5, 2, 1],
        [5, 3, 1],
        [5, 4, 1],
        [5, 5, 1],
        [6, 3, 1],
        [0, 3, -1],
        [1, 1, -1],
        [1, 2, -1],
        [1, 3, -1],
        [1, 4, -1],
        [1, 5, -1],
        [2, 1, -1],
        [2, 2, -1],
        [2, 3, -1],
        [2, 4, -1],
        [2, 5, -1],
        [3, 0, -1],
        [3, 1, -1],
        [3, 2, -1],
        [3, 3, -1],
        [3, 4, -1],
        [3, 5, -1],
        [3, 6, -1],
        [4, 1, -1],
        [4, 2, -1],
        [4, 3, -1],
        [4, 4, -1],
        [4, 5, -1],
        [5, 1, -1],
        [5, 2, -1],
        [5, 3, -1],
        [5, 4, -1],
        [5, 5, -1],
        [6, 3, -1],
    ]

    out = torch.tensor(out, device=device).T
    out[0,:] -= 8
    out[1,:] -= 8
    return out