import glob
import os.path

import numpy as np
import skimage.io as io
import torch
import zarr
from torch import Tensor


def convert(base_dir):
    if os.path.isdir(base_dir) and not base_dir.endswith(".zarr"):
        files = glob.glob(os.path.join(base_dir, "*.trch"))

    elif '*' in base_dir:
        files = glob.glob(base_dir)
    else:
        files = [base_dir]
    print("Found the following files to convert:")
    for f in files:
        if os.path.exists(f):
            print(f"\t{f}")
        # print(f'\t{f}')
    print("------")
    for f in files:
        if not os.path.exists(f):
            continue
        print(f"Converting {f}...", end="")

        if f.endswith(".zarr"):
            x = zarr.load(f)
        else:
            x = torch.load(f, map_location="cpu")

        print(x.shape)

        if not isinstance(x, Tensor) and not isinstance(x, np.ndarray):
            continue

        new_file = os.path.splitext(f)[0] + ".tif"
        if isinstance(x, np.ndarray):
            if x.ndim == 3:
                x = x.transpose(2, 0, 1)
                io.imsave(new_file, x, compression="zlib")

            elif x.ndim == 4:
                x = x.transpose(3, 1, 2, 0).astype(np.uint16)
                io.imsave(new_file, x, compression="zlib")

        else:
            if x.min() < 0:
                print("doing this...", x.min())
                mask = x == 0
                x = (
                    x.add(1).div(2).mul(255).float().round().to(torch.uint8)
                )  # convert skeletons to uint8
                x[mask] = 0
                print(x.shape, x.min(), x.max())

            if x.ndim == 3:
                io.imsave(new_file, x.permute(2, 0, 1).numpy(), compression="zlib")
            elif x.ndim == 4:
                print(new_file)
                io.imsave(
                    new_file, x.permute(3, 1, 2, 0).numpy(), compression="zlib"
                )  # Z, C, X, Y

        del x
