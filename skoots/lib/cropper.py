import functools
from torch import Tensor
from typing import Tuple, List, Optional


def crops(image: Tensor,
          cropsize: List[int],
          overlap: Optional[List[int]] = [0, 0, 0]) -> Tuple[Tensor, List[int]]:
    """
    Takes an image and sends out crops of a certain size with overlap pixels

    """



    image_shape = image.shape  # C, X, Y, Z

    for i, size in enumerate(cropsize):
        cropsize[i] = cropsize[i] if cropsize[i] < image_shape[i+1] else image_shape[i+1]
        # overlap[i] = overlap[i] if cropsize[i] < image_shape[i+1] else 0

    print(f'Crops: {image_shape}')
    print(cropsize)

    assert len(image_shape) - 1 == len(
        cropsize) == len(overlap) == 3, f'Image Shape must equal the shape of the crop.\n{image.shape=}, {cropsize=}' \
                                        f'{overlap=}'
    dim = ['x', 'y', 'z']
    for c, o, d in zip(cropsize, overlap, dim):
        assert c - o*2 != 0, f'Overlap in {d} dimmension cannot be equal to or larger than crop size... {o*2=} < {c}'

    # for i in range(image_shape[1] // cropsize[1] + 1):

    x = 0
    while x < image_shape[1]:
        _x = x if x + cropsize[0] <= image_shape[1] else image_shape[1] - cropsize[0]

        y = 0
        while y < image_shape[2]:
            _y = y if y + cropsize[1] <= image_shape[2] else image_shape[2] - cropsize[1]

            z = 0
            while z < image_shape[3]:
                _z = z if z + cropsize[2] <= image_shape[3] else image_shape[3] - cropsize[2]

                # print(f'{x}:{x + cropsize[0]}, {y}:{y + cropsize[1]}, {z}:{z + cropsize[2]}')

                yield image[:, _x:_x + cropsize[0], _y:_y + cropsize[1], _z:_z + cropsize[2]].unsqueeze(0), [_x, _y, _z]


                z += (cropsize[2] - (overlap[2] * 2))
            y += (cropsize[1] - (overlap[1] * 2))
        x += (cropsize[0] - (overlap[0] * 2))
