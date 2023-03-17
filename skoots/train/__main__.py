import argparse
import os.path
import sys
import warnings

from skoots.config import get_cfg_defaults

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from skoots.lib.mp_utils import setup_process, cleanup, find_free_port
from skoots.train.engine import train
from skoots.validate.stats import get_flops, get_parameter_count

from skoots.lib.utils import cfg_to_bism_model
from skoots.config import get_cfg_defaults

from yacs.config import CfgNode

from functools import partial

torch.set_float32_matmul_precision('high')


def load_cfg(args: argparse.Namespace):
    """Load configurations.
    """
    # Set configurations
    cfg = get_cfg_defaults()
    if os.path.exists(args.config_file):
        cfg.merge_from_file(args.config_file)
    else:
        warnings.warn('Could not find config file from path, training using default configuration')
    cfg.freeze()

    return cfg


def main():
    parser = argparse.ArgumentParser(description='SKOOTS Training Parameters')
    parser.add_argument('--config-file', type=str, help='YAML config file for training')
    args = parser.parse_args()

    cfg = load_cfg(args)
    model: nn.Module = cfg_to_bism_model(cfg)  # This is our skoots torch model

    checkpoint = torch.load(cfg.TRAIN.PRETRAINED_MODEL_PATH)
    state_dict = checkpoint if not 'model_state_dict' in checkpoint else checkpoint['model_state_dict']
    model.load_state_dict(state_dict)

    cfg, model = main()
    port = find_free_port()
    world_size = cfg.SYSTEM.NUM_GPUS if cfg.TRAIN.DISTRIBUTED else 1

    mp.spawn(train, args=(port, world_size, model, cfg), nprocs=world_size, join=True)



if __name__ == '__main__':
    import sys
    sys.exit(main())
