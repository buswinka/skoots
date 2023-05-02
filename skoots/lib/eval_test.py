import os.path
import warnings

import numpy as np
import skimage.io as io
import torch
import torch.nn.functional as F
from bism.models import get_constructor
from bism.models.spatial_embedding import SpatialEmbedding
from torch import Tensor
from torch.cuda.amp import autocast
from tqdm import tqdm

import skoots.lib.skeleton
from skoots.lib.cropper import crops
from skoots.lib.flood_fill import efficient_flood_fill
from skoots.lib.vector_to_embedding import vector_to_embedding

warnings.filterwarnings("ignore")

"""
Instance Segmentation more or less...
---------------------
    vectors, skeletons, semantic_masks = model(image)
    embedding_x, embedding_y, embedding_z = vectors_to_embedding(vectors * semantic_masks)
    instance_skeletons = efficient_flood_fill(skeletons)
    instance_mask = instance_skeleton[embedding_x, embedding_y, embedding_z]

"""


@torch.inference_mode()  # disables autograd and reference counting for SPEED
def eval(
    image_path: str,
    num_tuple=(60, 60, 60 // 5),
    checkpoint_path: str = "/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/models/Oct21_17-15-08_CHRISUBUNTU.trch",
) -> None:
    scale = -99999

    image: np.array = io.imread(image_path)  # [Z, X, Y, C]
    image: np.array = image[..., np.newaxis] if image.ndim == 3 else image
    image: np.array = image.transpose(-1, 1, 2, 0)
    image: np.array = image[[2], ...] if image.shape[0] > 3 else image  # [C=1, X, Y, Z]

    scale: int = 2**16 if image.dtype == np.uint16 else 2**8

    image: Tensor = torch.from_numpy(image.astype(np.int32)).pin_memory()
    print(f"Image Shape: {image.shape}, Dtype: {image.dtype}, Scale Factor: {scale}")

    # Allocate a bunch or things...
    if image.dtype == torch.float:
        pad3d = (5, 5, 30, 30, 30, 30)  # Pads last dim first!
        image = (
            F.pad(image, pad3d, mode="reflect") if image.dtype == torch.float else image
        )
    else:
        pad3d = False

    # num_tuple = (60, 60, 60 // 5)
    num = torch.tensor(num_tuple)

    c, x, y, z = image.shape

    skeleton = torch.zeros(size=(1, x, y, z), dtype=torch.int16)
    # semantic = torch.zeros(size=(1, x, y, z), dtype=torch.int16)
    vectors = torch.zeros((3, x, y, z), dtype=torch.half)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path)

    state_dict = (
        checkpoint
        if not "model_state_dict" in checkpoint
        else checkpoint["model_state_dict"]
    )

    model_constructor = get_constructor(
        "unext", spatial_dim=3
    )  # gets the model from a name...
    backbone = model_constructor(
        in_channels=1, out_channels=16, dims=[16, 32, 64, 32, 16]
    )
    model = SpatialEmbedding(backbone=backbone)
    model.load_state_dict(state_dict=state_dict)
    model = model.to(device).train()
    # model = torch.jit.optimize_for_inference(torch.jit.script(model))

    cropsize = [300, 300, 20]
    overlap = [30, 30, 2]

    total = skoots.lib.cropper.get_total_num_crops(image.shape, cropsize, overlap)
    iterator = tqdm(crops(image, cropsize, overlap), desc="", total=total)

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

        skeleton[
            :,
            x + overlap[0] : x + cropsize[0] - overlap[0],
            y + overlap[1] : y + cropsize[1] - overlap[1],
            z + overlap[2] : z + cropsize[2] - overlap[2],
        ] = skeleton_map[
            0,
            :,
            overlap[0] : -overlap[0],
            overlap[1] : -overlap[1],
            overlap[2] : -overlap[2] :,
        ].gt(
            0.8
        )
        # semantic[:,
        # x + overlap[0]: x + cropsize[0] - overlap[0],
        # y + overlap[1]: y + cropsize[1] - overlap[1],
        # z + overlap[2]: z + cropsize[2] - overlap[2]] = probability_map[0, :,
        #                                                 overlap[0]: -overlap[0],
        #                                                 overlap[1]: -overlap[1],
        #                                                 overlap[2]: -overlap[2]
        #                                                 :].gt(0.5)

        vectors[
            :,
            x + overlap[0] : x + cropsize[0] - overlap[0],
            y + overlap[1] : y + cropsize[1] - overlap[1],
            z + overlap[2] : z + cropsize[2] - overlap[2],
        ] = vec[
            0,
            :,
            overlap[0] : -overlap[0],
            overlap[1] : -overlap[1],
            overlap[2] : -overlap[2] :,
        ].half()

        iterator.desc = f"Evaluating UNet on slice [x{x}:y{y}:z{z}]"

    # _x, _y, _z
    if pad3d:
        skeleton = skeleton[
            0, pad3d[2] : -pad3d[3], pad3d[4] : -pad3d[5], pad3d[0] : -pad3d[1]
        ]
        # semantic = semantic[0, pad3d[2]:-pad3d[3], pad3d[4]:-pad3d[5], pad3d[0]:-pad3d[1]]
        vectors = vectors[
            :, pad3d[2] : -pad3d[3], pad3d[4] : -pad3d[5], pad3d[0] : -pad3d[1]
        ]

    else:
        skeleton = skeleton[0, ...]
        vectors = vectors[:, ...]

    del image

    # io.imsave('/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/semantic.tif',
    #           semantic.squeeze().mul(255).int().cpu().numpy().astype(np.uint8).transpose(2, 0, 1))

    # io.imsave is memory costly when saving large tensors... :(
    torch.save(vectors, os.path.split(image_path)[0] + "/vectors.trch")
    torch.save(skeleton, os.path.split(image_path)[0] + "/skeleton_unlabeled.trch")

    # io.imsave(
    #     '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/skeleton_unlabeled.tif',
    #     skeleton.squeeze().cpu().numpy().transpose(2, 0, 1))

    skeleton = efficient_flood_fill(skeleton)

    # io.imsave('/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/skeleton.tif',
    #           skeleton.cpu().numpy().transpose(2, 0, 1))

    torch.save(skeleton, os.path.split(image_path)[0] + "/skeleton.trch")

    instance_mask = torch.zeros_like(skeleton, dtype=torch.int16)
    skeleton = skeleton.unsqueeze(0).unsqueeze(0)

    iterator = tqdm(
        crops(vectors, crop_size=[500, 500, 50]), desc="Assigning Instances:"
    )

    print(f"[      ] identifying connected components...", end="")
    for _vec, (x, y, z) in iterator:
        _embed = skoots.lib.vector_to_embedding.vector_to_embedding(
            scale=num, vector=_vec, N=1
        )
        _embed += torch.tensor((x, y, z)).view(
            1, 3, 1, 1, 1
        )  # We adjust embedding to region of the crop
        _inst_maks = skoots.lib.skeleton.index_skeleton_by_embed(
            skeleton=skeleton, embed=_embed
        ).squeeze()
        w, h, d = _inst_maks.shape
        instance_mask[x : x + w, y : y + h, z : z + d] = _inst_maks

    print("DONE")
    # print(instance_mask.unique().shape[0] - 1, ' Unique mito')

    del skeleton, vectors  # explicitly delete unnecessary tensors for memory

    io.imsave(
        os.path.split(image_path)[0] + "/instance_mask.tif",
        instance_mask.cpu().numpy().astype(np.uint16).transpose(2, 0, 1),
    )


if __name__ == "__main__":
    # image_path = '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/hide_validate-1.tif'
    # image_path = '/home/chris/Documents/threeOHC_registered_8bit_cell2.tif'
    image_path = "/home/chris/Documents/threeOHC_registered_8bit.tif"
    # image_path = '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/onemito.tif'
    # image_path = '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/cell_apex-1.tif'
    # image_path = '/home/chris/Dropbox (Partners HealthCare)/Manuscripts - Buswinka/Mitochondria Segmentation/Figures/Figure X6X  - Whole image analysis/crop.tif'
    eval(image_path)
