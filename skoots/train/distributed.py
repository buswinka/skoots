import warnings
from functools import partial
from typing import Tuple, Callable, Dict
import os.path

from skoots.train.dataloader import dataset, MultiDataset, skeleton_colate
from skoots.train.sigma import Sigma, init_sigma
from skoots.train.loss import tversky, soft_dice_cldice
from skoots.train.merged_transform import transform_from_cfg, background_transform_from_cfg
from skoots.train.engine import engine
from skoots.train.setup import setup_process, cleanup, find_free_port

from torch import Tensor
import torch.nn as nn
import torch.optim.lr_scheduler
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lion_pytorch import Lion

from yacs.config import CfgNode

torch.manual_seed(101196)


def train(rank: str,
          port: str,
          world_size: int,
          model: nn.Module,
          cfg: CfgNode
          ):
    setup_process(rank, world_size, port, backend='nccl')

    device = f'cuda:{rank}'

    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model)
    # model = torch.compile(model)

    _ = model(torch.rand((1, 1, 300, 300, 20), device=device))

    augmentations: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] = partial(transform_from_cfg, cfg=cfg,
                                                                              device=device)
    background_agumentations: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] = partial(background_transform_from_cfg,
                                                                                         cfg=cfg, device=device)
    # Training Dataset - MultiDataset[Mitochondria, Background]
    _datasets = []
    for path, N in zip(cfg.TRAIN.TRAIN_DATA_DIR, cfg.TRAIN.TRAIN_SAMPLE_PER_IMAGE):
        _device = device if cfg.TRAIN.STRORE_DATA_ON_GPU else 'cpu'
        _datasets.append(dataset(path=path,
                                 transforms=augmentations,
                                 sample_per_image=N,
                                 device=device,
                                 pad_size=10).to(_device))

    for path, N in zip(cfg.TRAIN.TRAIN_DATA_DIR, cfg.TRAIN.TRAIN_SAMPLE_PER_IMAGE):
        _datasets.append(dataset(path=path,
                                 transforms=partial(background_agumentations, device=device), sample_per_image=N,
                                 device=device,
                                 pad_size=100).to('cpu'))

    merged_train = MultiDataset(*_datasets)

    train_sampler = torch.utils.data.distributed.DistributedSampler(merged_train)
    dataloader = DataLoader(merged_train, num_workers=0, batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE,
                            sampler=train_sampler, collate_fn=skeleton_colate)

    # Validation Dataset
    _datasets = []
    for path, N in zip(cfg.TRAIN.VALIDATION_DATA_DIR, cfg.TRAIN.VALIDATION_SAMPLE_PER_IMAGE):
        _device = device if cfg.TRAIN.STRORE_DATA_ON_GPU else 'cpu'
        _datasets.append(dataset(path=path,
                                 transforms=augmentations,
                                 sample_per_image=N,
                                 device=device,
                                 pad_size=10).to(_device))

    merged_validation = MultiDataset(*_datasets)
    test_sampler = torch.utils.data.distributed.DistributedSampler(merged_validation)
    valdiation_dataloader = DataLoader(merged_validation, num_workers=0, batch_size=cfg.TRAIN.VALIDATION_BATCH_SIZE,
                                       sampler=test_sampler,
                                       collate_fn=skeleton_colate)

    torch.backends.cudnn.benchmark = cfg.TRAIN.CUDNN_BENCHMARK
    torch.autograd.profiler.profile = cfg.TRAIN.AUTOGRAD_PROFILE
    torch.autograd.profiler.emit_nvtx(enabled=cfg.TRAIN.AUTOGRAD_EMIT_NVTX)
    torch.autograd.set_detect_anomaly(cfg.TRAIN.AUTOGRAD_DETECT_ANOMALY)

    sigma: Sigma = init_sigma(cfg, device)

    # The constants dict contains everything needed to replicate a training run.
    # will get serialized and saved.
    epochs = 10000
    constants = {
        'model': model,
        'vector_scale': vector_scale,
        'anisotropy': anisotropy,
        'lr': 5e-4 / 3,  # 5e-4,
        'wd': 1e-6 / 0.33,  # 1e-6,
        'optimizer': partial(Lion, use_triton=False),  # partial(torch.optim.AdamW, eps=1e-16),
        'scheduler': partial(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, T_0=epochs + 1),
        'sigma': sigma,
        'loss_embed': soft_dice_cldice(),  # tversky(alpha=0.25, beta=0.75, eps=1e-8, device=device),
        'loss_prob': soft_dice_cldice(),  # tversky(alpha=0.5, beta=0.5, eps=1e-8, device=device),
        'loss_skele': tversky(alpha=0.5, beta=1.5, eps=1e-8, device=device),
        'epochs': epochs,
        'device': device,
        'train_data': dataloader,
        'val_data': valdiation_dataloader,
        'train_sampler': train_sampler,
        'test_sampler': test_sampler,
        'distributed': True,
        'mixed_precision': True,
        'rank': rank,
        'n_warmup': 1500,
        'savepath': '/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/models',
    }

    writer = SummaryWriter() if rank == 0 else None
    if writer:
        print('SUMMARY WRITER LOG DIR: ', writer.get_logdir())
    model_state_dict, optimizer_state_dict, avg_loss = engine(writer=writer, verbose=True, force=True, **constants)
    avg_loss = torch.tensor(avg_loss)
    if writer:
        writer.add_hparams(hyperparams,
                           {'hparam/loss': avg_loss[-20:-1].mean().item()})

    if rank == 0:
        for k in constants:
            if k in ['model', 'train_data', 'val_data', 'train_sampler', 'test_sampler', 'loss_embed', 'loss_prob']:
                constants[k] = str(constants[k])

        constants['model_state_dict'] = model_state_dict
        constants['optimizer_state_dict'] = optimizer_state_dict

        torch.save(constants,
                   f'/home/chris/Dropbox (Partners HealthCare)/trainMitochondriaSegmentation/models/{os.path.split(writer.log_dir)[-1]}.trch')

    cleanup(rank)


if __name__ == '__main__':
    port = find_free_port()
    world_size = 2
    mp.spawn(train, args=(port, world_size), nprocs=world_size, join=True)
