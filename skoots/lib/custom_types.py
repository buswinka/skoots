from typing import Dict, TypedDict
from torch import Tensor


class DataDict(TypedDict):
    image: Tensor
    masks: Tensor
    skeletons: Dict[int, Tensor]
    baked_skeletons: Tensor
    skele_masks: Tensor

