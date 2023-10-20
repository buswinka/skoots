import os.path
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize
from torch import Tensor
from tqdm import tqdm
import kimimaro
import numpy as np
import skimage.io as io
import glob


def save_train_test_split(
    mask: Tensor, skeleton: Dict[int, Tensor], z_split: int, base: str
):
    """
    Splits a volume of binary masks and skeletons. You CANNOT naively just split the mask in two as skeletons of objects
    on the border might not be properly calculated.

    Saves pickled Dict[int, Tensor] to base+'_train.skeletons.trch' and base+'_validate.skeletons.trch'

    :param mask: Instance masks
    :param skeleton: Dict of skeleton of EVERY object in mask
    :param z_split: Z index of the train test split
    :param base: base filepath by which to save.

    :return: None
    """

    # train
    _mask = mask[..., 0 : z_split + 1 :]

    # assert 486 in _mask.unique()

    _skel = {}
    for u in _mask.unique():
        u = int(u)
        if u == 486:
            print("We got em...")
        if u == 0:
            continue
        if u in skeleton:
            _skel[u] = skeleton[u]
        # else:
        # print(f'Not in Train: {u}')

    torch.save(_skel, base + "_train.skeletons.trch")

    _mask = mask[..., z_split::]
    _skel = {}
    for u in _mask.unique():
        u = int(u)
        if u == 0:
            continue
        if u in skeleton:
            x = skeleton[u]
            x[:, 2] -= 150
            _skel[u] = x

    torch.save(_skel, base + "_validate.skeletons.trch")


def calculate_skeletons(mask: Tensor, scale: Tensor) -> Dict[int, Tensor]:
    """
    Calculates the skeleton of each object in mask

    :param mask: [C, X, Y, Z]
    :return: Dict[int, Tensor] dict of masks where int is the object id and Tensor is [3, K] skeletons
    """
    # raise DeprecationWarning('This function is slow and should no longer be used...')
    unique = torch.unique(mask)

    x, y, z = mask.shape

    if scale.sum() != 3:
        print(scale, x, y, z)

        large_mask = (
            F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=torch.tensor([x, y, z]).mul(scale).float().round().int().tolist(),
                mode="nearest",
            )
            .squeeze()
            # .cuda()
            .int()
        )
    else:
        large_mask = mask.clone()

    large_mask_unique = large_mask.unique().tolist()
    unique_list = unique.tolist()

    for u in unique:
        assert u in large_mask_unique, 'Downscaled too much!'

    for u in large_mask_unique:
        assert u in unique_list, 'Downscaled too much!'

    # assert torch.allclose(
    #     unique.cuda(), torch.unique(large_mask)
    # ), f"{unique=}, {large_mask.unique()=}"

    # large_mask = mask

    unique, counts = torch.unique(large_mask, return_counts=True)

    output = {}

    for id in tqdm(unique):
        if id == 0:
            continue

        temp = large_mask == id
        nonzero = torch.nonzero(temp)

        lower = nonzero.min(0)[0]
        upper = nonzero.max(0)[0]

        # print(f'{id.item()=}, {lower=}, {upper=}')

        upper[upper.sub(lower) == 0] += 1

        # Get just the region of a particular instance of the binary image...
        temp = temp[
            lower[0].item() : upper[0].item(),  # x
            lower[1].item() : upper[1].item(),  # y
            lower[2].item() : upper[2].item(),  # z
        ].float()

        _x = upper[0] - lower[0]
        _y = upper[1] - lower[1]
        _z = upper[2] - lower[2]

        # Calculate the binary skeleton of that image...
        skeleton = skeletonize(temp.cpu().numpy(), method="lee")
        skeleton = torch.from_numpy(skeleton).unsqueeze(0).unsqueeze(0)

        offset = lower.cpu().div(scale)  # , rounding_mode='trunc')
        # offset = offset

        if torch.nonzero(skeleton).shape[0] != 0:
            skel = torch.nonzero(skeleton.squeeze(0).squeeze(0)).div(scale).add(offset)
            # print(skel)
            output[int(id)] = skel
        else:
            _nonzoer = torch.nonzero(temp.cpu()).float()
            _nonzoer = _nonzoer.unsqueeze(0) if _nonzoer.ndim == 1 else _nonzoer
            output[int(id)] = _nonzoer.mean(0).div(scale).add(offset).unsqueeze(0)

        assert (
            output[int(id)].shape[0] > 0 and output[int(id)].ndim > 1
        ), f"{temp.nonzero().shape=}, {lower=} {id}, {output[int(id)].shape}"

    return output


def _calculate_skeletons(mask: Tensor, scale: Tensor) -> Dict[int, Tensor]:
    """

    image of shape [X, Y, Z,] with int masks
    returns dict[int, Tensor]

    :param mask:
    :param scale:
    :return:
    """

    skels = kimimaro.skeletonize(
        mask.numpy(),
        anisotropy=scale,
        progress=True,
        dust_threshold=0,
        fill_holes=False,
        fix_avocados=False,
        parallel=0
    )
    print('test1234')
    skels = {k:torch.from_numpy(v.vertices).div(torch.tensor(scale)).round().long() for k, v in skels.items()}


    return skels


def create_gt_skeletons(base_dir, mask_filter, scale: Tuple[float, float, float]):
    files = glob.glob(os.path.join(base_dir, f"*{mask_filter}.tif"))
    files = [b[:-11:] for b in files]

    scale = torch.tensor(scale)

    for f in files:
        print('attempting to skeletonize ', f+f"{mask_filter}.tif" )
        mask = io.imread(f + f"{mask_filter}.tif")
        mask = torch.from_numpy(mask.astype(np.int32))
        mask = mask.permute((1, 2, 0))

        output = calculate_skeletons(mask, scale)

        for u in mask.unique():
            if u == 0:
                continue
            assert int(u) in output, f"{f}, {u}, {output.keys()=}"

        torch.save(output, f + ".skeletons.trch")
        print("SAVED", f + ".skeletons.trch")


if __name__ == "__main__":
    import skimage.io as io
    import numpy as np
    import glob

    """
    The move -> Calculate the skeleton of each instance and save it as a tensor of nonzero indicies
    For each pixel in the instance mask, we now need to calculate which skeleton point we need to point to
    To do this, we take the index of each point, find the skeleton point closest to it, andreplace it
    """

    image = io.imread('/home/chris/Dropbox (Partners HealthCare)/skoots-experiments/data/mitochondria/train/external/kasthuri.labels.tif')
    image = torch.from_numpy(image.transpose((1, 2, 0)))
    skels = calculate_skeletons(image, (1, 1, 7))

    skeletons = torch.zeros_like(image)
    print(image.shape, skeletons.shape)

    for id, skel in skels.items():
        print(skel.shape)
        skeletons[skel[:,0].long(), skel[:,1].long(), skel[:,2].div(7).round().long()] = 1

    io.imsave('/home/chris/Desktop/skeletons_test.tif', skeletons.permute((2, 0 ,1).numpy()))

    # bases = [
    #     # '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/data/train/',
    #     "/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/data/toBeSkeletonized/",
    # ]
    # if not torch.cuda.is_available():
    #     raise RuntimeError("NO CUDA")
    #
    # # Z_SCALE = 2
    # SCALE = torch.tensor([0.2, 0.2, 3])
    # # SCALE = torch.tensor([1,1,1])
    #
    # bases = glob.glob(bases[0] + "*.labels.tif")
    # bases = [b[:-11:] for b in bases]
    #
    # for base in bases:
    #     mask = io.imread(base + ".labels.tif")
    #     mask = torch.from_numpy(mask.astype(np.int32))
    #
    #     mask = mask.permute((1, 2, 0))
    #     x, y, z = mask.shape
    #
    #     output = calculate_skeletons(mask, SCALE)
    #
    #     torch.save(output, base + ".skeletons.trch")
    #     print("SAVED", base + ".skeletons.trch")
    #
    #     save_train_test_split(mask, output, 150, base)
