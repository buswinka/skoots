import fastremap
import skimage.io as io
import os.path
import warnings


def load_renumber_save(path: str, overwrite: bool):
    """
    Simple util to renumber instance masks. Callable from a bash alias: remap $IMAGE_PATH
    Overwrites! Error!

    :param path:
    :param overwrite: if true, rewrites the filename
    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        assert os.path.exists(path), f"{path} does not exist"

        print("[      ] Loading Image...", end="", flush=True)
        im = io.imread(path)
        print("\r[\x1b[1;32;40m DONE \x1b[0m] ", end="")
        print(f"Image loaded with shape: {im.shape} and dtype: {im.dtype}", flush=True)

        # --------------------------------------------------------------------------------------------------------------

        print("[      ] Finding Unique...", end="", flush=True)
        unique = fastremap.unique(im)
        unique = unique.tolist()
        n_unique = len(unique) - 1
        mapping = {0: 0}
        i = 1
        for k in unique:
            if k == 0:
                continue
            while i in mapping.values():
                i += 1
            mapping[k] = i
            i += 1
        print("\r[\x1b[1;32;40m DONE \x1b[0m] ", end="")
        print(f"Found {n_unique} numbers to remap")

        # --------------------------------------------------------------------------------------------------------------

        print("[      ] Remapping...", end="", flush=True)
        im = fastremap.remap(im, mapping)
        print("\r[\x1b[1;32;40m DONE \x1b[0m] ", end="")
        print(f"Remapped {n_unique} numbers with max label {i}")

        # --------------------------------------------------------------------------------------------------------------

        print("[      ] Renumbering...", end="", flush=True)
        im, remapping = fastremap.renumber(im)
        print("\r[\x1b[1;32;40m DONE \x1b[0m] ", end="")
        print(f"Remapped to shape: {im.shape} and dtype: {im.dtype}", flush=True)

        # --------------------------------------------------------------------------------------------------------------

        print("[      ] Saving...", end="", flush=True)

        if not overwrite:
            file, ext = os.path.splitext(path)
            path = file + '_remapped' + ext

        io.imsave(path, im, compression="zlib")
        print("\r[\x1b[1;32;40m DONE \x1b[0m] ", end="")
        print(f"Saved to path: {path}", flush=True)

        # --------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SKOOTS Utils Renumber")
    parser.add_argument("image_filepath", type=str, help="Path to image")
    parser.add_argument("-o", "--overwrite", action='store_true', help="Path to image")

    args = parser.parse_args()

    load_renumber_save(args.image_filepath, args.overwrite)
