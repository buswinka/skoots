import argparse
import os.path

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from skoots.config import get_cfg_defaults
from skoots.lib.mp_utils import find_free_port
from skoots.lib.utils import cfg_to_bism_model
from skoots.train.engine import train
import warnings
import logging

torch.set_float32_matmul_precision("high")


def load_cfg_from_file(args: argparse.Namespace):
    """Load configurations."""
    # Set configurations
    cfg = get_cfg_defaults()
    if os.path.exists(args.config_file):
        cfg.merge_from_file(args.config_file)
    else:
        raise ValueError("Could not find config file from path!")
    cfg.freeze()

    return cfg


def main():
    parser = argparse.ArgumentParser(description="SKOOTS Training Parameters")
    parser.add_argument("--config-file", type=str, help="YAML config file for training")
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
        format="[%(asctime)s] skoots-train [%(levelname)s]: %(message)s",
        force=True,
    )

    cfg = load_cfg_from_file(args)
    model: nn.Module = cfg_to_bism_model(cfg)  # This is our skoots torch model

    if cfg.TRAIN.PRETRAINED_MODEL_PATH and os.path.exists(cfg.TRAIN.PRETRAINED_MODEL_PATH[0]):
        checkpoint = torch.load(cfg.TRAIN.PRETRAINED_MODEL_PATH[0])
        state_dict = (
            checkpoint
            if not "model_state_dict" in checkpoint
            else checkpoint["model_state_dict"]
        )
        model.load_state_dict(state_dict)
    else:
        warnings.warn(f'Could not find file at path: {cfg.TRAIN.PRETRAINED_MODEL_PATH[0]}. Model has not been loaded.')

    port = find_free_port()
    world_size = cfg.SYSTEM.NUM_GPUS if cfg.TRAIN.DISTRIBUTED else 1

    mp.spawn(train, args=(port, world_size, model, cfg, args.log), nprocs=world_size, join=True)


if __name__ == "__main__":
    import sys

    sys.exit(main())
