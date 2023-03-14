import skoots.lib.skeleton
import torch
import torch.nn.functional as F
from torch import Tensor
from skoots.lib.embedding_to_prob import baked_embed_to_prob
from skoots.lib.vector_to_embedding import vector_to_embedding
from skoots.lib.flood_fill import efficient_flood_fill
from skoots.lib.cropper import crops
from skoots.lib.skeleton import bake_skeleton
from skoots.lib.morphology import binary_erosion, binary_dilation

import skimage.io as io
from tqdm import tqdm
import numpy as np
from torch import Tensor

from bism.models import get_constructor
from bism.models.spatial_embedding import SpatialEmbedding
from torch.cuda.amp import GradScaler, autocast

import warnings
import os.path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from skimage.morphology import skeletonize


@torch.inference_mode()  # disables autograd and reference counting for SPEED
def eval(image_path: str,
         checkpoint_path: str = '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/models/Oct21_17-15-08_CHRISUBUNTU.trch') -> None:

    """
    Evaluates SKOOTS on an arbitrary image.

    :param image_path:
    :param checkpoint_path:
    :return:
    """

    checkpoint = torch.load(checkpoint_path)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    filename_without_extensions = os.path.splitext(image_path)[0]

    image: np.array = io.imread(image_path)  # [Z, X, Y, C]
    image: np.array = image[..., np.newaxis] if image.ndim == 3 else image
    image: np.array = image.transpose(-1, 1, 2, 0)
    image: np.array = image[[2], ...] if image.shape[0] > 3 else image  # [C=1, X, Y, Z]

    scale: int = 2 ** 16 if image.dtype == np.uint16 else 2 ** 8

    image: Tensor = torch.from_numpy(image).pin_memory()
    # print(f'Image Shape: {image.shape}, Dtype: {image.dtype}, Scale Factor: {scale}')

    # Allocate a bunch or things...
    if image.dtype == torch.float:
        pad3d = (5, 5, 30, 30, 30, 30)  # Pads last dim first!
        image = F.pad(image, pad3d, mode='reflect') if image.dtype == torch.float else image
    else:
        pad3d = False

    vector_scale = torch.tensor(checkpoint['vector_scale'])

    c, x, y, z = image.shape

    skeleton = torch.zeros(size=(1, x, y, z), dtype=torch.int16)
    vectors = torch.zeros((3, x, y, z), dtype=torch.half)

    model_constructor = get_constructor('unext', spatial_dim=3)  # gets the model from a name...
    backbone = model_constructor(in_channels=1, out_channels=32, dims=[32, 64, 128, 64, 32])
    model = SpatialEmbedding(
        backbone=backbone
    )
    model.load_state_dict(state_dict=checkpoint['model_state_dict'])
    model = model.to(device).train()
    # model = torch.jit.optimize_for_inference(torch.jit.script(model))

    model = torch.jit.script(model)

    cropsize = [300, 300, 20]
    overlap = [30, 30, 2]

    total = skoots.lib.cropper.get_total_num_crops(image.shape, cropsize, overlap)
    iterator = tqdm(crops(image, cropsize, overlap), desc='', total=total)

    id = 1
    for slice, (x, y, z) in iterator:
        with autocast(enabled=True):  # Saves Memory!
            out = model(slice.div(scale).float().cuda())

        probability_map = out[:, [-1], ...].cpu()
        skeleton_map = out[:, [-2], ...].float()

        skeleton_map = skeleton_map.cpu()
        vec = out[:, 0:3:1, ...].cpu()

        vec = vec * probability_map.gt(0.5)
        skeleton_map = skeleton_map * probability_map.gt(0.5)

        # put the predictions into the preallocated tensors...
        skeleton[:,
        x + overlap[0]: x + cropsize[0] - overlap[0],
        y + overlap[1]: y + cropsize[1] - overlap[1],
        z + overlap[2]: z + cropsize[2] - overlap[2]] = skeleton_map[0, :,
                                                        overlap[0]: -overlap[0],
                                                        overlap[1]: -overlap[1],
                                                        overlap[2]: -overlap[2]
                                                        :].gt(0.8)

        vectors[:,
        x + overlap[0]: x + cropsize[0] - overlap[0],
        y + overlap[1]: y + cropsize[1] - overlap[1],
        z + overlap[2]: z + cropsize[2] - overlap[2]] = vec[0, :,
                                                        overlap[0]: -overlap[0],
                                                        overlap[1]: -overlap[1],
                                                        overlap[2]: -overlap[2]
                                                        :].half()

        iterator.desc = f'Evaluating UNet on slice [x{x}:y{y}:z{z}]'

    # _x, _y, _z
    if pad3d:
        skeleton = skeleton[0, pad3d[2]:-pad3d[3], pad3d[4]:-pad3d[5], pad3d[0]:-pad3d[1]]
        vectors = vectors[:, pad3d[2]:-pad3d[3], pad3d[4]:-pad3d[5], pad3d[0]:-pad3d[1]]

    else:
        skeleton = skeleton[0, ...]
        vectors = vectors[:, ...]

    del image  # we don't need the image anymore

    # torch.save(vectors, filename_without_extensions + '_vectors.trch')
    # torch.save(skeleton, filename_without_extensions + '_unlabeled_skeletons.trch')

    skeleton: Tensor = efficient_flood_fill(skeleton)

    # torch.save(skeleton, filename_without_extensions + '_skeletons.trch')

    instance_mask = torch.zeros_like(skeleton, dtype=torch.int16)
    skeleton = skeleton.unsqueeze(0).unsqueeze(0)

    iterator = tqdm(crops(vectors, crop_size=[500, 500, 50]), desc='Assigning Instances:')

    print(f'[      ] identifying connected components...', end='')
    for _vec, (x, y, z) in iterator:

        _embed = skoots.lib.vector_to_embedding.vector_to_embedding(scale=vector_scale, vector=_vec)
        _embed += torch.tensor((x, y, z)).view(1, 3, 1, 1, 1)  # We adjust embedding to region of the crop
        _inst_maks = skoots.lib.skeleton.index_skeleton_by_embed(skeleton=skeleton,
                                                                 embed=_embed).squeeze()
        w, h, d = _inst_maks.shape
        instance_mask[x:x + w, y:y + h, z:z + d] = _inst_maks

    print('DONE')
    # print(instance_mask.unique().shape[0] - 1, ' Unique mito')

    del skeleton, vectors  # explicitly delete unnecessary tensors for memory

    # # io.imsave(filename_without_extensions + '_instance_mask.tif',
    #           instance_mask.cpu().numpy().transpose(2, 0, 1))


if __name__ == '__main__':
    # image_path = '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/hide_validate-1.tif'
    image_path = '/home/chris/Documents/threeOHC_registered_8bit_cell2.tif'
    # image_path = '/home/chris/Dropbox (Partners HealthCare)/Manuscripts - Buswinka/Mitochondria Segmentation/Figures/Fig X - compare to affinity/data/hide_validate.tif'
    # image_path = '/home/chris/Documents/threeOHC_registered_8bit.tif'
    # image_path = '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/onemito.tif'
    # image_path = '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/cell_apex-1.tif'
    # image_path = '/home/chris/Dropbox (Partners HealthCare)/Manuscripts - Buswinka/Mitochondria Segmentation/Figures/Figure X6X  - Whole image analysis/crop.tif'
    import time
    t1 = time.time()
    eval(image_path)
    t2 = time.time()

    print(f"TOOK {t1-t2} seconds")
