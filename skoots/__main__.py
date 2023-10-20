import argparse
import glob
import logging
import os.path

import skoots.lib.eval
import skoots.train.generate_skeletons
import skoots.utils.convert_trch_to_tif


def main():
    parser = argparse.ArgumentParser(
        prog="SKOOTS",
        description="skoots parameters",
    )

    # general script arguments
    eval_args = parser.add_argument_group("eval arguments")
    eval_args.add_argument("--image", type=str, help="path to image")
    eval_args.add_argument(
        "--pretrained-checkpoint",
        type=str,
        help="path to a pretrained skoots model. Will be used"
        "as a starting point for training",
    )

    eval_args.add_argument(
        "--log",
        type=int,
        default=3,
        help="Log Level: 0-Debug, 1-Info, 2-Warning, 3-Error, 4-Critical",
    )

    _log_map = [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]

    # accessory script arguments
    accessory_args = parser.add_argument_group("scripting arguments")
    accessory_args.add_argument(
        "--skeletonize-train-data", help="calculate skeletons of training data"
    )

    accessory_args.add_argument(
        "--mask-filter", default='.labels', help="filter of mask file"
    )

    accessory_args.add_argument(
        "--anisotropyXY",
        type=float,
        default=1.0,
        help="calculate skeletons of training data",
    )
    accessory_args.add_argument(
        "--anisotropyZ",
        type=float,
        default=1.0,
        help="calculate skeletons of training data",
    )
    accessory_args.add_argument(
        "--convert",
        type=str,
        help="converts all skoots eval outputs in directory to a tif image",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=_log_map[args.log],
        format="[%(asctime)s] skoots-eval [%(levelname)s]: %(message)s",
    )

    # Eval
    if args.skeletonize_train_data is None and args.convert is None:
        assert args.pretrained_checkpoint is not None, (
            f"Cannot evaluate SKOOTS wihtout pretrained model. "
            f"--pretrained_checkpoint must not be None"
        )
        if args.pretrained_checkpoint == 'base':
            args.pretrained_checkpoint = "/home/chris/Dropbox (Partners HealthCare)/skoots-experiments/models/mito/Mar28_16-26-14_CHRISUBUNTU.trch"

        if os.path.isdir(args.image):
            files = glob.glob(args.image + "/*.tif")
            files.sort()
        else:
            files = [args.image]
        for f in files:
            skoots.lib.eval.eval(f, args.pretrained_checkpoint)

    # accessory scripts
    if args.skeletonize_train_data:
        downscale = (args.anisotropyXY, args.anisotropyXY, args.anisotropyZ)
        skoots.train.generate_skeletons.create_gt_skeletons(
            args.skeletonize_train_data, args.mask_filter, downscale
        )

    if args.convert:
        skoots.utils.convert_trch_to_tif.convert(args.convert)


if __name__ == "__main__":
    main()
