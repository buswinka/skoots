import logging
import os.path
import tracemalloc
import warnings

import fastremap
import numpy as np
import skimage.io as io
import torch
import torch._dynamo
import torch.nn as nn
import zarr
from torch import Tensor
from torch.cuda.amp import autocast
from tqdm import tqdm
from yacs.config import CfgNode

import skoots.lib.skeleton
from skoots.lib.cropper import crops
from skoots.lib.flood_fill import efficient_flood_fill
from skoots.lib.morphology import binary_dilation, binary_dilation_2d
from skoots.lib.utils import cfg_to_bism_model
from skoots.lib.vector_to_embedding import vector_to_embedding

import time

warnings.filterwarnings("ignore")


@torch.inference_mode()  # disables autograd and reference counting for SPEED
def eval(
    image_path: str,
    checkpoint_path: str = "/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/models/Oct21_17-15-08_CHRISUBUNTU.trch",
) -> None:
    """
    Evaluates SKOOTS on an arbitrary image.

    :param image_path:
    :param checkpoint_path:
    :return:
    """
    tracemalloc.start()
    start = time.time()

    torch._dynamo.config.log_level = logging.ERROR
    logging.info(f"Loading model file: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    if "cfg" in checkpoint:
        cfg: CfgNode = checkpoint["cfg"]
    else:
        raise RuntimeError("Attempting to evaluate skoots on a legacy model file.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    filename_without_extensions = os.path.splitext(image_path)[0]

    logging.info(f"Loading image from file: {image_path}")
    image: np.array = io.imread(image_path)  # [Z, X, Y, C]
    image: np.array = image[..., np.newaxis] if image.ndim == 3 else image
    image: np.array = image.transpose(-1, 1, 2, 0)
    image: np.array = image[[2], ...] if image.shape[0] > 3 else image  # [C=1, X, Y, Z]

    # if True:#image.dtype == torch.float:
    #     # pad3d = (5, 5, 30, 30, 30, 30)  # Pads last dim first!
    #     pad3d = ((0,0), (30, 30), (30, 30), (5, 5))
    #     image = np.pad(image, pad3d, mode='reflect') #if image.dtype == torch.float else image
    # else:
    #     pad3d = False

    c, x, y, z = image.shape
    logging.info(
        f"Loaded an image with shape: {(c, x, y, z)}, dtype: {image.dtype}, min: {image.min()}, max: {image.max()}"
    )

    scale: int = 2**16 if image.dtype == np.uint16 else 2**8

    image: Tensor = torch.from_numpy(image).to(torch.uint8).pin_memory()
    # print(f'Image Shape: {image.shape}, Dtype: {image.dtype}, Scale Factor: {scale}')

    # Allocate a bunch or things...

    vector_scale = torch.tensor(cfg.SKOOTS.VECTOR_SCALING)
    logging.info("Pre-allocating output arrays...")
    skeleton = torch.zeros(size=(1, x, y, z), dtype=torch.uint8)
    vectors = torch.zeros((3, x, y, z), dtype=torch.half)

    logging.info(f"Constructing SKOOTS model")
    base_model: nn.Module = cfg_to_bism_model(cfg)  # This is our skoots torch model
    base_model.load_state_dict(state_dict=checkpoint["model_state_dict"])
    base_model = base_model.to(device).train()

    logging.info(f"Compiling SKOOTS model with torch inductor")
    model = torch.compile(base_model)
    for _ in range(10):
        _ = model(torch.rand((1, 1, 300, 300, 20), device=device, dtype=torch.float))

    cropsize = [300, 300, 20]  # DEFAULT (300, 300, 20)
    overlap = [50, 50, 5]

    total = skoots.lib.cropper.get_total_num_crops(image.shape, cropsize, overlap)
    iterator = tqdm(
        crops(image, cropsize, overlap, device=device), desc="", total=total
    )
    benchmark_start = time.time()

    id = 1
    for crop, (x, y, z) in iterator:
        with autocast(enabled=True):  # Saves Memory!
            out = model(crop.div(scale).float().cuda())

        probability_map = out[:, [-1], ...]
        skeleton_map = out[:, [-2], ...].float()
        vec = out[:, 0:3:1, ...]

        vec = vec * probability_map.gt(0.5)
        skeleton_map = skeleton_map * probability_map.gt(0.5)

        for _ in range(
            2
        ):  # expand the skeletons in x/y/z. Only  because they can get too skinny
            skeleton_map = binary_dilation(skeleton_map)
            for _ in range(0):  # expand 2 times just in x/y
                skeleton_map = binary_dilation_2d(skeleton_map)

        # put the predictions into the preallocated tensors...
        _destination = (
            ...,
            slice(x + overlap[0], x + cropsize[0] - overlap[0]),
            slice(y + overlap[1], y + cropsize[1] - overlap[1]),
            slice(z + overlap[2], z + cropsize[2] - overlap[2]),
        )

        _source = (
            0,
            ...,
            slice(overlap[0], -overlap[0]),
            slice(overlap[1], -overlap[1]),
            slice(overlap[2], -overlap[2]),
        )

        skeleton[_destination] = skeleton_map[_source].gt(0.8).cpu()
        vectors[_destination] = vec[_source].half().cpu()

        iterator.desc = f"Evaluating UNet on slice [x{x}:y{y}:z{z}]"

    # _x, _y, _z
    # if pad3d:
    #     _, padx, pady, padz = pad3d
    #
    #     skeleton = skeleton[0, padx[0]:-padx[1], pady[0]:-pady[1], padz[0]:-padz[1]]
    #     vectors = vectors[:, padx[0]:-padx[1], pady[0]:-pady[1], padz[0]:-padz[1]]
    #
    # else:
    #     skeleton = skeleton[0, ...]
    #     vectors = vectors[:, ...]

    del image  # we don't need the image anymore

    # torch.save(vectors, filename_without_extensions + '_vectors.trch')
    # torch.save(skeleton, filename_without_extensions + '_unlabeled_skeletons.trch')

    # logging.info(f"Saving vector output")
    # io.imsave(
    #     filename_without_extensions + f"_skoots_vectors.tif",
    #     vectors.mul(127)
    #     .add(127)
    #     .int()
    #     .cpu()
    #     .numpy()
    #     .transpose(3, 1, 2, 0)
    #     .astype(np.uint16),
    #     compression="zlib",
    # )
    #
    # logging.info(f"Saving unlabeledc skeleton output")
    # io.imsave(
    #     filename_without_extensions + f"_unlabeled_skeletons.tif",
    #     skeleton.mul(255)
    #     .squeeze(0)
    #     .clamp(0, 255)
    #     .int()
    #     .cpu()
    #     .numpy()
    #     .transpose(2, 0, 1)
    #     .astype(np.uint8),
    #     compression="zlib",
    # )

    zarr.save_array(
        filename_without_extensions + f"_skoots_vectors.zarr",
        vectors.mul(127).add(127).int().cpu().numpy(),
    )

    zarr.save_array(
        filename_without_extensions + f"_skoots_skeleton_unlabeled.zarr",
        skeleton.cpu().numpy(),
    )

    logging.info(f"Performing an flood fill on skeletons")
    skeleton: Tensor = efficient_flood_fill(skeleton.to(torch.int16))

    # io.imsave(
    #     filename_without_extensions + f"_skeletons.tif",
    #     skeleton
    #     .squeeze(0)
    #     .clamp(0, 255)
    #     .int()
    #     .cpu()
    #     .numpy()
    #     .transpose(2, 0, 1)
    #     .astype(np.uint16),
    #     compression="zlib",
    # )

    zarr.save_array(
        filename_without_extensions + f"_skoots_skeleton.zarr", skeleton.cpu().numpy()
    )

    logging.info(f"Saving labeled skeletons")
    # torch.save(skeleton, filename_without_extensions + '_skeletons.trch')

    instance_mask = torch.zeros_like(skeleton, dtype=torch.int16)
    skeleton = skeleton.unsqueeze(0).unsqueeze(0)

    cropsize = [500, 500, 50]
    overlap = (50, 50, 5)
    total = skoots.lib.cropper.get_total_num_crops(vectors.shape, cropsize, overlap)
    iterator = tqdm(
        crops(vectors, crop_size=cropsize, overlap=overlap),
        desc="Assigning Instances:",
        total=total,
    )

    logging.info(f"Identifying connected components...")
    for _vec, (x, y, z) in iterator:
        _destination = (
            slice(x + overlap[0], x + cropsize[0] - overlap[0]),
            slice(y + overlap[1], y + cropsize[1] - overlap[1]),
            slice(z + overlap[2], z + cropsize[2] - overlap[2]),
        )

        _source = (
            slice(overlap[0], -overlap[0]),
            slice(overlap[1], -overlap[1]),
            slice(overlap[2], -overlap[2]),
        )

        _embed = skoots.lib.vector_to_embedding.vector_to_embedding(
            scale=vector_scale, vector=_vec, N=1
        )
        _embed += torch.tensor((x, y, z)).view(
            1, 3, 1, 1, 1
        )  # We adjust embedding to region of the crop
        _inst_maks = skoots.lib.skeleton.index_skeleton_by_embed(
            skeleton=skeleton, embed=_embed
        ).squeeze()
        w, h, d = _inst_maks.shape

        instance_mask[_destination] = (
            _inst_maks[_source] if torch.tensor(overlap).gt(0).all() else _inst_maks
        )

    with open(filename_without_extensions + "_skoots_benchmark.txt", "w") as f:
        ss = tracemalloc.get_traced_memory()
        f.write(f"SKOOTS Segmentation Benchmark:\n")
        f.write(f"------------------------------\n")
        f.write(f"Time: {time.time() - benchmark_start} seconds\n")
        f.write(f"Memory (current/max): {tracemalloc.get_traced_memory()}\n\n")

    print("DONE")
    # print(instance_mask.unique().shape[0] - 1, ' Unique mito')

    del skeleton, vectors  # explicitly delete unnecessary tensors for memory
    instance_mask, remapping = fastremap.renumber(
        instance_mask.cpu().numpy(), in_place=True
    )

    io.imsave(filename_without_extensions + '_instance_mask.tif',
              instance_mask.transpose(2, 0, 1), compression='zlib')
    # zarr.save_array(
    #     filename_without_extensions + f"_skoots_instance_mask.zarr", instance_mask
    # )

    end = time.time()
    elapsed = end - start

    logging.info(
        f"DONE: Process took {elapsed} seconds, {elapsed / 60} minutes, {elapsed / (60 ** 2)}, hours"
    )


if __name__ == "__main__":
    # image_path = '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/hide_validate-1.tif'
    image_path = "/home/chris/Documents/threeOHC_registered_8bit_cell2.tif"
    # image_path = '/home/chris/Dropbox (Partners HealthCare)/Manuscripts - Buswinka/Mitochondria Segmentation/Figures/Fig X - compare to affinity/data/hide_validate.tif'
    # image_path = '/home/chris/Documents/threeOHC_registered_8bit.tif'
    # image_path = '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/onemito.tif'
    # image_path = '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/outputs/cell_apex-1.tif'
    # image_path = '/home/chris/Dropbox (Partners HealthCare)/Manuscripts - Buswinka/Mitochondria Segmentation/Figures/Figure X6X  - Whole image analysis/crop.tif'
    import time

    t1 = time.time()
    eval(image_path)
    t2 = time.time()

    print(f"TOOK {t1 - t2} seconds")
