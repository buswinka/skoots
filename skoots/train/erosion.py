from typing import Dict, Tuple, Union, List

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def binary_erosion(image: Tensor) -> Tensor:
    kernel = _get_binary_kernel3d(3, image.device)
    padding = _compute_zero_padding((3, 3, 3))

    b, c, h, w, d = image.shape
    # map the local window to single vector
    features: Tensor = F.conv3d(
        image.reshape(b * c, 1, h, w, d), kernel, padding=padding, stride=1
    )
    return features.min(dim=1)[0].unsqueeze(0)


class erosion:
    def __init__(
        self, kernel_targets: int = 3, rate: float = 0.5, device: str = "cpu"
    ) -> None:
        super(erosion, self).__init__()
        self.device = device
        self.rate = rate
        self.kernel_targets = kernel_targets

        if kernel_targets % 2 != 1:
            raise ValueError("Expected Kernel target to be Odd")

        self.padding: Tuple[int, int, int] = self._compute_zero_padding(
            (kernel_targets, kernel_targets, kernel_targets)
        )
        self.kernel: torch.Tensor = self._get_binary_kernel3d(kernel_targets, device)

    def __call__(
        self, input: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Randomly applies a gaussian blur

        :param data_dict Dict[str, torch.Tensor]: data_dictionary from a dataloader. Has keys:
            key : val
            'image' : torch.Tensor of size [C, X, Y, Z] where C is the number of colors, X,Y,Z are the mask height,
                      width, and depth
            'masks' : torch.Tensor of size [I, X, Y, Z] where I is the number of identifiable objects in the mask
            'centroids' : torch.Tensor of size [I, 3] where dimension two is the [X, Y, Z] position of the centroid
                          for instance i

        :return: data_dict Dict[str, torch.Tensor]: dictionary with identical keys as input, but with transformed values
        """
        if isinstance(input, Dict):
            self.check_inputs(input)
            return self._is_dict(input)
        elif isinstance(input, torch.Tensor):
            return self._is_tensor(input)

    def _is_dict(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if data_dict["masks"].dtype != self.kernel.dtype:
            raise ValueError(
                f'Expected Image dtype to be {self.kernel.dtype} not {data_dict["masks"].dtype}'
            )
        # We expect the image to have ndim==4 but need a batch size of 1 for median filter

        device = data_dict["image"].device
        if self.kernel.device != device:
            self.kernel = self.kernel.to(device)

        if torch.rand(1) < self.rate:
            im = data_dict["masks"].unsqueeze(0)
            b, c, h, w, d = im.shape
            # map the local window to single vector
            features: torch.Tensor = F.conv3d(
                im.reshape(b * c, 1, h, w, d),
                self.kernel,
                padding=self.padding,
                stride=1,
            )
            data_dict["masks"] = features.min(dim=1)[0]
        return data_dict

    def _is_tensor(self, input: torch.Tensor) -> torch.Tensor:
        "Assumes [C, X, Y, Z]"
        if input.device != self.kernel.device:
            raise ValueError(
                f"Expected Image Device to be {self.kernel.device} not {input.device}"
            )
        if input.dtype != self.kernel.dtype:
            raise ValueError(
                f"Expected Image dtype to be {self.kernel.dtype} not {input.dtype}"
            )
        if input.ndim != 4:
            raise ValueError(
                f"Expected Image ndim to be 4 not {input.ndim}, with shape {input.shape}"
            )

        b, c, h, w, d = input.unsqueeze(0).shape
        # map the local window to single vector
        features: torch.Tensor = F.conv3d(
            input.reshape(b * c, 1, h, w, d),
            self.kernel,
            padding=self.padding,
            stride=1,
        )
        return features.min(dim=1)[0]


def _compute_zero_padding(kernel_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r"""Utility function that computes zero padding tuple.
    Adapted from Kornia
    """
    computed: List[int] = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1], computed[2]


def _get_binary_kernel3d(window_size: int, device: torch.device) -> torch.Tensor:
    r"""Creates a symetric binary kernel to extract the patches. If the window size
    is HxWxD will create a (H*W)xHxW kernel.

    Adapted from Kornia

    """
    window_range: int = int(window_size**3)
    kernel: torch.Tensor = torch.zeros(
        (window_range, window_range, window_range), device=device
    )
    for i in range(window_range):
        kernel[i, i, i] += 1.0
    kernel = kernel.view(-1, 1, window_size, window_size, window_size)

    # get rid of all zero kernels
    ind = torch.nonzero(kernel.view(kernel.shape[0], -1).sum(1))
    return kernel[ind[:, 0], ...]
