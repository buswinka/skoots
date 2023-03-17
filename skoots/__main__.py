import skoots.utils.convert_trch_to_tif
import skoots.lib.eval
import skoots.train.generate_skeletons
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog='SKOOTS',
        description='skoots parameters',
    )


    # general script arguments
    general_args = parser.add_argument_group('general arguments')
    general_args.add_argument('--pretrained_checkpoint', help='path to a pretrained skoots model. Will be used'
                                                            'as a starting point for training')


    # evaluation arguments
    eval_args = parser.add_argument_group('evaluation Arguments')
    eval_args.add_argument('--eval', help='path to image file to segment')


    # accessory script arguments
    accessory_args = parser.add_argument_group('scripting arguments')
    accessory_args.add_argument('--skeletonize_train_data', help='calculate skeletons of training data')
    accessory_args.add_argument('--downscaleXY', type=float, default=1.0, help='calculate skeletons of training data')
    accessory_args.add_argument('--downscaleZ',type=float, default=1.0, help='calculate skeletons of training data')
    accessory_args.add_argument('--convert_output_to_tiff', help='converts all skoots eval outputs in directory to a tif image')

    args = parser.parse_args()

    # Eval
    if args.eval is not None:
        assert args.pretrained_checkpoint is not None, f'Cannot evaluate SKOOTS wihtout pretrained model. ' \
                                                       f'--pretrained_checkpoint must not be None'
        skoots.lib.eval.eval(args.eval, args.pretrained_checkpoint)

    # accessory scripts
    if args.skeletonize_train_data:
        downscale = (args.downscaleXY, args.downscaleXY, args.downscaleZ)
        skoots.train.generate_skeletons.create_gt_skeletons(args.skeletonize_train_data, args.mask_filter, downscale)

    if args.convert_output_to_tiff:
        skoots.utils.convert_trch_to_tif.convert(args.convert_output_to_tiff)

if __name__ == '__main__':
    main()
