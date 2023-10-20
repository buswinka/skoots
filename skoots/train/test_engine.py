import argparse
import os.path

import torch
import torch.multiprocessing as mp
import torch.nn as nn

from skoots.config import get_cfg_defaults
from skoots.lib.mp_utils import find_free_port
from skoots.lib.utils import cfg_to_bism_model
from skoots.train.engine import train

if __name__ == '__main__':

    torch.set_float32_matmul_precision("high")

    cfg = get_cfg_defaults()
    file = '/home/chris/Dropbox (Partners HealthCare)/skoots-experiments/configs/mitochondria/train_skoots_on_mitochondria.yaml'
    if os.path.exists(file):
        cfg.merge_from_file(file)
    cfg.freeze()

    model: nn.Module = cfg_to_bism_model(cfg)  # This is our skoots torch model

    if cfg.TRAIN.PRETRAINED_MODEL_PATH:
        checkpoint = torch.load(cfg.TRAIN.PRETRAINED_MODEL_PATH[0])
        state_dict = (
            checkpoint
            if not "model_state_dict" in checkpoint
            else checkpoint["model_state_dict"]
        )
        model.load_state_dict(state_dict)

    port = find_free_port()
    world_size = cfg.SYSTEM.NUM_GPUS if cfg.TRAIN.DISTRIBUTED else 1

    mp.spawn(train, args=(port, world_size, model, cfg), nprocs=world_size, join=True)
