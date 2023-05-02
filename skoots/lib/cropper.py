from typing import Tuple, List, Optional

from torch import Tensor


def get_total_num_crops(
    image_shape: Tensor, crop_size: List[int], overlap: Optional[Tuple[int]]
) -> int:
    total = 0

    for i, size in enumerate(crop_size):
        crop_size[i] = (
            crop_size[i] if crop_size[i] < image_shape[i + 1] else image_shape[i + 1]
        )
        # overlap[i] = overlap[i] if cropsize[i] < image_shape[i+1] else 0

    assert len(image_shape) - 1 == len(crop_size) == len(overlap) == 3, (
        f"Image Shape must equal the shape of the crop.\n{image_shape=}, {crop_size=}"
        f"{overlap=}"
    )
    dim = ["x", "y", "z"]
    for c, o, d in zip(crop_size, overlap, dim):
        assert (
            c - o * 2 != 0
        ), f"Overlap in {d} dimmension cannot be equal to or larger than crop size... {o*2=} < {c}"

    x = 0
    while x < image_shape[1]:
        _x = x if x + crop_size[0] <= image_shape[1] else image_shape[1] - crop_size[0]

        y = 0
        while y < image_shape[2]:
            _y = (
                y
                if y + crop_size[1] <= image_shape[2]
                else image_shape[2] - crop_size[1]
            )

            z = 0
            while z < image_shape[3]:
                _z = (
                    z
                    if z + crop_size[2] <= image_shape[3]
                    else image_shape[3] - crop_size[2]
                )

                total += 1

                z += crop_size[2] - (overlap[2] * 2)
            y += crop_size[1] - (overlap[1] * 2)
        x += crop_size[0] - (overlap[0] * 2)

    return total


def crops(
    image: Tensor,
    crop_size: List[int],
    overlap: Optional[Tuple[int, int, int]] = (0, 0, 0),
    device="cpu",
) -> Tuple[Tensor, List[int]]:
    """
    Generator which takes an image and sends out crops of a certain size with overlap pixels
    Can send a crop to a torch.device ahead of use through caching.

    Shapes:
        - image: :math:`(C, X, Y, Z)`
        - yeilds: :math:`(B, C, X, Y, Z)`

    :param image: 4D torch.Tensor of shape [C, X, Y, Z]
    :param crop_size: Spatial dims of the resulting crops [X, Y, Z]
    :param overlap: Overlap between each crop
    :param device: torch device of the crop
    :return: Crop of the image, and the indicies of the crop
    """

    image_shape = image.shape  # C, X, Y, Z

    for i, size in enumerate(crop_size):
        crop_size[i] = (
            crop_size[i] if crop_size[i] < image_shape[i + 1] else image_shape[i + 1]
        )

    assert len(image_shape) - 1 == len(crop_size) == len(overlap) == 3, (
        f"Image Shape must equal the shape of the crop.\n{image.shape=}, {crop_size=}"
        f"{overlap=}"
    )
    dim = ["x", "y", "z"]
    for c, o, d in zip(crop_size, overlap, dim):
        assert (
            c - (o * 2) != 0
        ), f"Overlap in {d} dimmension cannot be equal to or larger than crop size... {c=} - {o*2=} = {c - (o * 2)} < {c}"

    # for i in range(image_shape[1] // cropsize[1] + 1):

    _output_cache = []

    x = 0
    while x < image_shape[1]:
        _x = x if x + crop_size[0] <= image_shape[1] else image_shape[1] - crop_size[0]

        y = 0
        while y < image_shape[2]:
            _y = (
                y
                if y + crop_size[1] <= image_shape[2]
                else image_shape[2] - crop_size[1]
            )

            z = 0
            while z < image_shape[3]:
                _z = (
                    z
                    if z + crop_size[2] <= image_shape[3]
                    else image_shape[3] - crop_size[2]
                )

                _output_cache.append(
                    (
                        image[
                            :,
                            _x : _x + crop_size[0],
                            _y : _y + crop_size[1],
                            _z : _z + crop_size[2],
                        ]
                        .unsqueeze(0)
                        .to(device, non_blocking=True),
                        [_x, _y, _z],
                    )
                )

                if len(_output_cache) > 1:
                    yield _output_cache.pop(0)

                z += crop_size[2] - (overlap[2] * 2)
            y += crop_size[1] - (overlap[1] * 2)
        x += crop_size[0] - (overlap[0] * 2)

    yield _output_cache.pop(0)  # final yield
