import skoots.lib.eval
import skoots.train.distributed
import skoots.train.generate_skeletons
import skoots.utils.convert_trch_to_tif

from skoots.train.setup import setup_process, cleanup, find_free_port

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from bism.models import get_constructor
from bism.models.spatial_embedding import SpatialEmbedding

from skoots.train.distributed import train
from skoots.lib.mp_utils import setup_process, cleanup, find_free_port

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

    # training arguments
    train_args = parser.add_argument_group('training arguments')
    train_args.add_argument('--train', action='store_true', help='train network using images in dir')
    train_args.add_argument('--anisotropy', type=float, default=5, help='Z dimmensions anisotropy.')
    train_args.add_argument('--vector_scale', type=float, default=60, help='XY embedding vector scale. ')
    train_args.add_argument('--mask_filter',
                            default='.labels', type=str, help='end string for masks to run on. Default: %(default)s')
    train_args.add_argument('--test_dir',
                            default=[], type=str, help='folder containing test data (optional)')
    train_args.add_argument('--learning_rate',
                            default=1e-4, type=float, help='learning rate. Default: %(default)s')
    train_args.add_argument('--num_image_samples',
                            default=20, type=float,
                            help='Number of samples of a training image per epoch Default: %(default)s')
    train_args.add_argument('--n_epochs',
                            default=5000, type=int, help='number of epochs. Default: %(default)s')
    train_args.add_argument('--batch_size',
                            default=3, type=int, help='batch size. Default: %(default)s')
    train_args.add_argument('--checkpoint', help='pretrained model location')
    train_args.add_argument('--train_dir', help='directory of training data')
    train_args.add_argument('--validation_dir', help='directory of validation data')
    train_args.add_argument('--background_dir', help='directory of background images')
    train_args.add_argument('--model_save_path', help='directory to save trained model file')

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
    # Preprocess Training Data

    # Eval
    if args.eval is not None:
        assert args.pretrained_checkpoint is not None, f'Cannot evaluate SKOOTS wihtout pretrained model. ' \
                                                       f'--pretrained_checkpoint must not be None'
        skoots.lib.eval.eval(args.eval, args.pretrained_checkpoint)

    # Training
    if args.train:
        dims = [32, 64, 128, 64, 32]

        model_constructor = get_constructor('unext', spatial_dim=3)  # gets the model from a name...
        backbone = model_constructor(in_channels=1, out_channels=dims[-1], dims=dims)
        model = SpatialEmbedding(
            backbone=backbone)

        if args.pretrained_checkpoint:
            model.load_state_dict(torch.load(args.pretrained_checkpoint)['model_state_dict'])

        hyperparams = {
            'model': 'unext',
            'depths': '[2,2,2,2,2]',
            'dims': str(dims),
        }

        anisotropy = (1., 1., args.anisotropy)
        vector_scale = (args.vector_scale, args.vector_scale, args.vector_scale/args.anisotropy)

        port = find_free_port()
        world_size = torch.cuda.device_count()
        mp.spawn(skoots.train.distributed.train,
                 args=(port, world_size, model, hyperparams,
                       args.train_dir, args.validation_dir, args.background_dir,
                       vector_scale, anisotropy),
                 nprocs=world_size, join=True)

    # accessory scripts
    if args.skeletonize_train_data:
        downscale = (args.downscaleXY, args.downscaleXY, args.downscaleZ)
        skoots.train.generate_skeletons.create_gt_skeletons(args.skeletonize_train_data, args.mask_filter, downscale)

    if args.convert_output_to_tiff:
        skoots.utils.convert_trch_to_tif.convert(args.convert_output_to_tiff)

if __name__ == '__main__':
    main()
