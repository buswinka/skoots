import argparse
import os.path

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from skoots.config import get_cfg_defaults, validate_cfg
from skoots.lib.mp_utils import find_free_port
from skoots.lib.utils import cfg_to_bism_model
from skoots.train.engine import train
from skoots.experimental.sparse_engine import train as sparse_train

import warnings
import logging
from typing import List
import glob

torch.set_float32_matmul_precision("high")


def load_cfg_from_file(file: str):
    """Load configurations."""
    # Set configurations
    cfg = get_cfg_defaults()
    if os.path.exists(file):
        cfg.merge_from_file(file)
    else:
        raise ValueError("Could not find config file from path!")
    cfg.freeze()

    validate_cfg(cfg)

    return cfg


def main():
    parser = argparse.ArgumentParser(description="SKOOTS Training Parameters")
    parser.add_argument("--config-file", type=str, help="YAML config file for training")
    parser.add_argument('-b', '--batch', action='store_true',
                        help='Batch execute a folder of training config files')
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

    configs: List[str] = glob.glob(os.path.join(args.config_file, '*.yaml')) if args.batch else [args.config_file]

    if args.batch:
        logging.info(f'found {len(configs)} config files at: {args.config_file}')
    else:
        logging.info(f'performing training run with config: {args.config_file}')

    for f in configs:
        cfg = load_cfg_from_file(f)
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

        if cfg.EXPERIMENTAL.IS_SPARSE:
            mp.spawn(sparse_train, args=(port, world_size, model, cfg, args.log), nprocs=world_size, join=True)
            logging.critical(f'joined all processes for config file: {f}')

        else:
            mp.spawn(train, args=(port, world_size, model, cfg, args.log), nprocs=world_size, join=True)


if __name__ == "__main__":
    import sys

    sys.exit(main())
