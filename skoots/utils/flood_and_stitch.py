import argparse
import logging

import fastremap
import numpy as np
from scipy.ndimage import label

import tqdm
from tqdm import trange
import io
import skimage.io

"""
This is software which tries to flood each 2D z slice of a mask, then stitch together a 3D mask...
"""


class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
        self.buf = ""

    def write(self, buf: str):
        assert isinstance(buf, str), f"self.buf is not a str"
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)


def watershed_and_stitch(mask: np.ndarray, dim: int) -> np.ndarray:
    assert mask.ndim == 3, f"mask ndim must be 3, not shape {mask.shape=}"
    assert mask.dtype == np.uint8, f"{mask.dtype=} != np.uint8"
    assert dim < 3, f"dim argument must be less than 3, not {dim=}"

    logging.info(f"Performing watershed on loaded image with shape: {mask.shape=}")
    logging.info(f"Mask Max: {mask.max()=}")
    logging.info(f"Mask Min: {mask.min()=}")
    logging.info(f"Num voxels to label: {mask.__gt__(0).sum()}")

    # This is basically a dimensional agnostic way to slice a matrix
    if dim == 0:
        index = [None, slice(None), slice(None)]
    elif dim == 1:
        index = [slice(None), None, slice(None)]
    elif dim == 2:
        index = [slice(None), slice(None), None]
    else:
        raise RuntimeError("dim argument must be 0 indexed between 0<=dim<=2")

    mask = mask.__gt__(0).astype(np.int32)

    # tqdm logging wrapper
    tqdm_out = TqdmToLogger(logger=logging.getLogger(), level=logging.INFO)

    for i in trange(
        mask.shape[dim], file=tqdm_out, desc="Performing Flooding of mask: "
    ):
        index[dim] = i

        mask_plane = mask[tuple(index)]  # 2D
        label(mask_plane, output=mask_plane)  # label in place

    if mask.shape[dim] == 1:  # one slice flood fill...
        return mask

    for _ in range(2):
        newind = np.max(mask)  # this is the new index we start with
        index[dim] = 0
        slice_a = mask[tuple(index)]

        for i in trange(
            1, mask.shape[dim], file=tqdm_out, desc="Identifying Connected Components"
        ):
            index[dim] = i
            slice_b = mask[tuple(index)]

            a_unique = np.unique(slice_a)
            logging.debug(f"on slice {i} found {a_unique=}")
            for u in a_unique:
                if u == 0:
                    continue

                logging.debug(f"{u=}, {slice_a.shape=}, {slice_b.shape=}")

                collision, counts = np.unique(
                    slice_b[slice_a.__eq__(u)], return_counts=True
                )

                counts = counts[collision != 0]
                collision = collision[collision != 0]

                counts = counts[collision != u]
                collision = collision[collision != u]

                if len(collision) == 0:
                    continue

                assert len(counts) == len(collision)

                to_replace = collision[np.argmax(counts)]

                if to_replace == u:
                    continue

                if to_replace == 0:
                    raise RuntimeError("to_replace is zero")

                logging.debug(f"unique: {collision=}, {counts=}")
                logging.debug(f"replacing {u} with {newind} in slice {i-1}")
                logging.debug(f"replacing {to_replace} with {newind} in slice {i}")

                index[dim] = slice(i)
                mask[tuple(index)][mask[tuple(index)] == u] = newind

                slice_b[slice_b == to_replace] = newind
                newind += 1

            slice_a = slice_b

        mask = np.flip(mask, axis=dim)

    mask = fastremap.renumber(mask)
    unique = np.unique(mask)
    logging.info(f"found {len(unique)} new objects with ids: {unique.tolist()}")
    return mask


if __name__ == "__main__":
    desc = """
    takes a 3D image of a semantic mask with one label (0 is background, ~0 is foreground)
    and performs a flood fill over each slice of a given dimension. Then attempts to stitch
    together the masks into a coherent 3D mask...
    """
    parser = argparse.ArgumentParser(
        prog="skoots.utils.flood_and_stitch.py", description=desc
    )

    parser.add_argument("image_path", type=str, help="input image tif")
    parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        default=0,
        help="spatial dimension to slice over. default=0",
    )

    parser.add_argument(
        "--distance",
        action="store_true",
        help="applies distance transform before watershed",
    )

    parser.add_argument(
        "--log",
        type=int,
        default=3,
        help="Log Level: 0-Debug, 1-Info, 2-Warning, 3-Error, 4-Critical",
    )

    args = parser.parse_args()

    # Set logging level
    _log_map = [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]

    logging.basicConfig(
        level=_log_map[args.log],
        format="[%(asctime)s] skoots/utils/flood_and_stitch.py [%(levelname)s]: %(message)s",
        force=True,
    )

    mask = skimage.io.imread(args.image_path).__gt__(0).astype(np.uint8)
    mask = watershed_and_stitch(mask, args.dimension)
    fastremap.renumber(mask, in_place=True)
    mask = fastremap.refit(mask)
    skimage.io.imsave(args.image_path.replace(".tif", "_replaced.tif"), mask)
