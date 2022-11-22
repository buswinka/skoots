import torch
import skimage.io as io
import numpy as np
import glob
from torch import Tensor
import os.path


def convert(base_dir):
    for f in glob.glob(os.path.join(base_dir, '*.trch')):
        x = torch.load(f)
        if not isinstance(x, Tensor):
            continue
        print(x.shape, f)

        new_file = os.path.splitext(f)[0] + '.tif'
        io.imsave(new_file, x.permute(2, 0, 1).numpy())
        del x
