from torch import Tensor
from typing import Tuple, List

from torch import Tensor


def get_adjacent_labels(x: Tensor, y: Tensor) -> List[Tuple[int, int]]:
    """
    calculates which masks of a signle object have two labels (due to the border)

    :param x:
    :param y:
    :return:
    """

    # Could ideally use the cantor pairing function but might overflow???
    # this seems to be safe and memory efficient
    z0 = (x + y).unique().tolist()
    z1 = (x * y).unique().tolist()

    identical = []

    for _x in x.unique():
        for _y in y.unique():
            if _x == 0 or _y == 0:
                continue

            if (_x + _y) in z0 and (_x * _y) in z1:
                identical.append((_x.item(), _y.item()))
    # identical = identical if len(identical) > 0 else None
    return identical
