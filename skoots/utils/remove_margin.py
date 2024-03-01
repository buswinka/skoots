import os.path
import warnings

import skimage.io as io


def remove_margin(im_path: str, mask_path: str):
    """
    simple util to remove the auto margins of the skoots eval process. only useful when trying to correct for
    training data.

    :param path: path to image
    :param overwrite: if true, rewrites the filename
    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        assert os.path.exists(im_path), f"{im_path} does not exist"
        assert os.path.exists(mask_path), f"{mask_path} does not exist"

        print("[      ] Loading Image...", end="", flush=True)
        im = io.imread(im_path)
        print("\r[\x1b[1;32;40m DONE \x1b[0m] ", end="")
        print(f"Image loaded with shape: {im.shape} and dtype: {im.dtype}.", flush=True)

        # --------------------------------------------------------------------------------------------------------------

        print("[      ] Loading Mask...", end="", flush=True)
        ma = io.imread(mask_path)
        print("\r[\x1b[1;32;40m DONE \x1b[0m] ", end="")
        print(f"Mask loaded with shape: {ma.shape} and dtype: {ma.dtype}.", flush=True)

        # --------------------------------------------------------------------------------------------------------------
        print("[      ] Cropping to skoots margin of [50, 50, 5]...", end="", flush=True)
        assert im.shape == ma.shape, f"{im.shape=} != {ma.shape=}"

        assert im.ndim == 3, f'{im.ndim=} != 3'
        assert ma.ndim == 3, f'{ma.ndim=} != 3'

        assert im.shape[0] > 10, f'[Z, X, Y]: {im.shape} | Z !< 100'
        assert im.shape[1] > 100, f'[Z, X, Y]: {im.shape} | X !< 100'
        assert im.shape[2] > 100, f'[Z, X, Y]: {im.shape} | Y !< 10'

        assert ma.shape[0] > 10, f'[Z, X, Y]: {ma.shape} | Z !< 100'
        assert ma.shape[1] > 100, f'[Z, X, Y]: {ma.shape} | X !< 100'
        assert ma.shape[2] > 100, f'[Z, X, Y]: {ma.shape} | Y !< 10'

        im = im[5:-5, 50:-50, 50:-50]
        ma = ma[5: -5, 50:-50, 50:-50]
        print("\r[\x1b[1;32;40m DONE \x1b[0m] ", end="")
        print(f"Image shape: {im.shape}, Mask shape: {ma.shape}", flush=True)

        # --------------------------------------------------------------------------------------------------------------

        print("[      ] Saving Image...", end="", flush=True)
        file, ext = os.path.splitext(im_path)
        im_path = file + '_removed_margins' + ext
        io.imsave(im_path, im, compression="zlib")
        print("\r[\x1b[1;32;40m DONE \x1b[0m] ", end="")
        print(f"Saved to path: {im_path}", flush=True)

        # --------------------------------------------------------------------------------------------------------------

        print("[      ] Saving Mask...", end="", flush=True)
        file, ext = os.path.splitext(mask_path)
        mask_path = file + '_removed_margins' + ext
        io.imsave(mask_path, ma, compression="zlib")
        print("\r[\x1b[1;32;40m DONE \x1b[0m] ", end="")
        print(f"Saved to path: {mask_path}", flush=True)

        # --------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SKOOTS Utils Renumber")
    parser.add_argument("image_filepath", type=str, help="Path to image")
    parser.add_argument("mask_filepath", type=str, help="path to mask")

    args = parser.parse_args()

    remove_margin(args.image_filepath, args.mask_filepath)
