from typing import Dict, TypedDict
from torch import Tensor


class DataDict(TypedDict):
    image: Tensor
    masks: Tensor
    skeletons: Dict[int, Tensor]
    baked_skeletons: Tensor
    skele_masks: Tensor

class SparseDataDict(TypedDict):
    image: Tensor
    background: Tensor
    skeletons: Dict[int, Tensor]
    skele_masks: Tensor