import torch
from torch import Tensor
import skimage.io as io
import numpy as np
from typing import Optional
from torch import nn
from yacs.config import CfgNode

import bism.models
import bism.modules


def imread(image_path: str,
           pin_memory: Optional[bool] = False) -> Tensor:
    """
    Imports an image from file and returns in torch format

    :param image_path: path to image
    :param pin_memory: saves torch tensor in pinned memory if true
    :return:
    """

    image: np.array = io.imread(image_path)  # [Z, X, Y, C]
    image: np.array = image[..., np.newaxis] if image.ndim == 3 else image
    image: np.array = image.transpose(-1, 1, 2, 0)
    image: np.array = image[[2], ...] if image.shape[0] > 3 else image  # [C=1, X, Y, Z]

    image: Tensor = torch.from_numpy(image.astype(np.int32))

    if pin_memory:
        image: Tensor = image.pin_memory()

    return image



if __name__ == '__main__':
    from skoots.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg_to_bism_model(cfg)