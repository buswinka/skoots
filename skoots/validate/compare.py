from typing import Dict

import torch
from skoots.validate.stats import get_volume, get_surface_area
from torch import Tensor


def stats_per_instance(x: Tensor, anisotropy: Tensor) -> Dict[str, Tensor]:
    """

    :param x:
    :param anisotropy:
    :return:
    """
    summary = {}

    object_id = torch.unique(x)
    object_id = object_id[object_id > 0]
    summary["id"] = object_id

    summary["volume"] = torch.empty_like(object_id)
    summary["surface_area"] = torch.empty_like(object_id)

    for i, id in enumerate(object_id):
        summary["volume"][i] = get_volume(x.eq(id), anisotropy)
        summary["surface_area"][i] = get_surface_area(x.eq(id), anisotropy)

    return summary


def compare(ground_truth, predictions) -> Dict[str, Tensor]:
    raise NotImplementedError
