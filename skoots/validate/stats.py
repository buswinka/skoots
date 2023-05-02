from typing import Optional, List, Union

import skimage.io
import skimage.measure
import torch
import torch.nn
from fvcore.nn import FlopCountAnalysis
from fvcore.nn.parameter_count import parameter_count
from torch import Tensor


def get_volume(
    x: Tensor, spacing: Optional[Union[List[float], Tensor]] = None
) -> Tensor:
    """
    Returns the volume of all nonzero voxels in the tensor

    :param x: [X, Y, Z] bool tensor of an image
    :param spacing: Array of the voxel spacings with which to calculate the volume

    :return: Volume
    """

    volume = torch.sum(x.nonzero().shape[0])
    volume = volume * torch.tensor(spacing).prod() if spacing else volume

    return volume


def get_surface_area(x: Tensor, anisotropy_ratio: List[float]) -> Tensor:
    """
    Approximates object surface area by performing a marching cubes algorithm to get a collection of faces.
    Then calculates surface area of all polygons.

    :param x: [X, Y, Z] bool tensor of an image
    :param anisotropy_ratio:

    :return: surface area in arbitrary units.
    """

    vets, faces, normals, values = skimage.measure.marching_cubes(
        x.gt(0).mul(255).cpu().numpy(), spacing=anisotropy_ratio
    )

    surface_area = skimage.measure.mesh_surface_area(verts=vets, faces=faces)
    surface_area = torch.from_numpy(surface_area)

    return surface_area


def tversky(x: Tensor, y: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """

    :param x:
    :param y:
    :param a: Penalizes false positives
    :param b: Penalizes false negatives
    :return:
    """

    true_positive: Tensor = x.mul(y).sum()
    false_positive: Tensor = torch.logical_not(x).mul(y).sum().mul(a)
    false_negative: Tensor = ((1 - y) * x).sum() * b

    tversky = (true_positive + 1e-16) / (
        true_positive + false_positive + false_negative + 1e-16
    )

    return 1 - tversky


def get_flops(model: torch.nn.Module, example_input: Tensor):
    return FlopCountAnalysis(model, example_input).total()


def get_parameter_count(model: torch.nn.Module):
    param_dict = parameter_count(model)

    total = 0
    for k, v in param_dict.items():
        total += v

    return total


if __name__ == "__main__":
    from hcat.backends.unext_hydra import UNeXT as UNeXT_0
    from hcat.backends.unext import UNeXT as UNeXT_1
    import matplotlib.pyplot as plt

    f = []
    p = []
    for i in (5, 7, 9):
        model0 = UNeXT_0(depths=[2, 3, 4, 3, 2], kernel_size=i)
        model0 = model0.to("cuda:0")

        flops0 = get_flops(model0, torch.rand((1, 1, 256, 256, 10), device="cuda:0"))
        total_params0 = get_parameter_count(model0)
        f.append(flops0)
        p.append(total_params0)

    plt.plot((5, 7, 9), f)

    f = []
    p = []
    for i in (5, 7, 9):
        model1 = UNeXT_1(depths=[2, 3, 4, 3, 2], kernel_size=i)
        model1 = model1.to("cuda:0")

        flops1 = get_flops(model1, torch.rand((1, 1, 256, 256, 10), device="cuda:0"))
        total_params1 = get_parameter_count(model1)
        f.append(flops1)
        p.append(total_params1)

    plt.plot((5, 7, 9), f)
    plt.show()
