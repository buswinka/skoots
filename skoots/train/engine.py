import os
import os.path
from functools import partial
from statistics import mean
from typing import Callable, Union, Dict

import skoots.train.loss
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
import torch.optim.swa_utils
from skoots.lib.embedding_to_prob import baked_embed_to_prob
from skoots.lib.vector_to_embedding import vector_to_embedding
from skoots.train.dataloader import dataset, MultiDataset, skeleton_colate
from skoots.train.merged_transform import (
    TransformFromCfg,
    BackgroundTransformFromCfg,
)
from skoots.train.setup import setup_process
from skoots.train.sigma import Sigma, init_sigma
from skoots.train.utils import write_progress
from torch import Tensor
from torch import profiler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from yacs.config import CfgNode

import logging

Dataset = Union[Dataset, DataLoader]

_valid_optimizers = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "adamax": torch.optim.Adamax,
}

_valid_loss_functions = {
    "soft_cldice": skoots.train.loss.soft_dice_cldice,
    "tversky": skoots.train.loss.tversky,
}

_valid_lr_schedulers = {
    "cosine_annealing_warm_restarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
}

torch.manual_seed(101196)
torch.set_float32_matmul_precision("high")


def train(rank: str, port: str, world_size: int, base_model: nn.Module, cfg: CfgNode, logging_level: int):
    setup_process(rank, world_size, port, backend="nccl")
    device = f"cuda:{rank}"

    _log_map = [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]
    logging.basicConfig(
        level=_log_map[logging_level],
        format=f"[%(asctime)s] skoots-train rank{rank} [%(levelname)s]: %(message)s",
        force=True,
    )

    base_model = base_model.to(device)
    base_model = torch.nn.parallel.DistributedDataParallel(base_model)

    if int(rank) == 0:
        print(cfg)

    if int(torch.__version__[0]) >= 2:
        logging.info('Compiling model with torch inductor')
        model = torch.compile(base_model)
    model=base_model

    augmentations = TransformFromCfg(cfg=cfg, device=device).set_dataset_std(255).set_dataset_mean(0)
    background_augmentations = BackgroundTransformFromCfg(cfg=cfg, device=device).set_dataset_std(255).set_dataset_mean(0)


    assert len(cfg.TRAIN.TRAIN_STORE_DATA_ON_GPU) == len(
        cfg.TRAIN.TRAIN_DATA_DIR), 'must set identical length gpu storage argument'
    assert len(cfg.TRAIN.VALIDATION_STORE_DATA_ON_GPU) == len(
        cfg.TRAIN.VALIDATION_DATA_DIR), 'must set identical length gpu storage argument'
    assert len(cfg.TRAIN.BACKGROUND_STORE_DATA_ON_GPU) == len(
        cfg.TRAIN.BACKGROUND_DATA_DIR), 'must set identical length gpu storage argument'

    _datasets = []
    for path, N, on_cuda in zip(cfg.TRAIN.TRAIN_DATA_DIR, cfg.TRAIN.TRAIN_SAMPLE_PER_IMAGE,
                                cfg.TRAIN.TRAIN_STORE_DATA_ON_GPU):
        _device = device if on_cuda else 'cpu'
        _datasets.append(
            dataset(
                path=path,
                transforms=augmentations,
                sample_per_image=N,
                device=device,
                pad_size=10,
            )
            .pin_memory()
            .to(_device)
        )

    for path, N, on_cuda in zip(
            cfg.TRAIN.BACKGROUND_DATA_DIR, cfg.TRAIN.BACKGROUND_SAMPLE_PER_IMAGE, cfg.TRAIN.BACKGROUND_STORE_DATA_ON_GPU
    ):
        _device = device if on_cuda else 'cpu'
        _datasets.append(
            dataset(
                path=path,
                transforms=background_augmentations,
                sample_per_image=N,
                device=device,
                pad_size=100,
            )
            .pin_memory()
            .to(_device)
        )

    merged_train = MultiDataset(*_datasets)

    train_sampler = torch.utils.data.distributed.DistributedSampler(merged_train)
    _n_workers = 0  # if _device != 'cpu' else 2
    dataloader = DataLoader(
        merged_train,
        num_workers=_n_workers,
        batch_size=cfg.TRAIN.TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=skeleton_colate,
    )

    # Validation Dataset
    _datasets = []
    for path, N, on_cuda in zip(
            cfg.TRAIN.VALIDATION_DATA_DIR, cfg.TRAIN.VALIDATION_SAMPLE_PER_IMAGE, cfg.TRAIN.VALIDATION_STORE_DATA_ON_GPU
    ):
        _device = device if on_cuda else 'cpu'
        _datasets.append(
            dataset(
                path=path,
                transforms=augmentations,
                sample_per_image=N,
                device=device,
                pad_size=10,
            )
            .pin_memory()
            .to(_device)
        )

    merged_validation = MultiDataset(*_datasets)
    test_sampler = torch.utils.data.distributed.DistributedSampler(merged_validation)
    if _datasets or cfg.TRAIN.VALIDATION_BATCH_SIZE >= 1:
        _n_workers = 0  # if _device != 'cpu' else 2
        _device = device if cfg.TRAIN.STORE_DATA_ON_GPU else "cpu"
        valdiation_dataloader = DataLoader(
            merged_validation,
            num_workers=_n_workers,
            batch_size=cfg.TRAIN.VALIDATION_BATCH_SIZE,
            sampler=test_sampler,
            collate_fn=skeleton_colate,
        )

    else:  # we might not want to run validation...
        valdiation_dataloader = None

    torch.backends.cudnn.benchmark = cfg.TRAIN.CUDNN_BENCHMARK
    torch.autograd.profiler.profile = cfg.TRAIN.AUTOGRAD_PROFILE
    torch.autograd.profiler.emit_nvtx(enabled=cfg.TRAIN.AUTOGRAD_EMIT_NVTX)
    torch.autograd.set_detect_anomaly(cfg.TRAIN.AUTOGRAD_DETECT_ANOMALY)

    sigma: Sigma = init_sigma(cfg, device)
    epochs = cfg.TRAIN.NUM_EPOCHS

    writer = SummaryWriter() if rank == 0 else None
    if writer:
        print("SUMMARY WRITER LOG DIR: ", writer.get_logdir())

    # TRAIN LOOP ----------------------------

    vector_scale = torch.tensor(cfg.SKOOTS.VECTOR_SCALING, device=device)

    kwargs = {k:v for k, v in zip(cfg.TRAIN.OPTIMIZER_KEYWORD_ARGUMENTS, cfg.TRAIN.OPTIMIZER_KEYWORD_VALUES)}
    logging.info('creating optimizer')
    optimizer = _valid_optimizers[cfg.TRAIN.OPTIMIZER](
        model.parameters(),
        lr=cfg.TRAIN.LEARNING_RATE,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY, **kwargs
    )

    logging.info('creating scheduler')
    scheduler = _valid_lr_schedulers[cfg.TRAIN.SCHEDULER](
        optimizer, T_0=cfg.TRAIN.SCHEDULER_T0
    )
    logging.info('creating grad scaler')
    scaler = GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = 100
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.05)

    logging.info('creating loss functions')
    _kwarg = {
        k: v for k, v in zip(cfg.TRAIN.LOSS_EMBED_KEYWORDS, cfg.TRAIN.LOSS_EMBED_VALUES)
    }
    loss_embed: Callable = _valid_loss_functions[cfg.TRAIN.LOSS_EMBED](**_kwarg)

    _kwarg = {
        k: v
        for k, v in zip(
            cfg.TRAIN.LOSS_PROBABILITY_KEYWORDS, cfg.TRAIN.LOSS_PROBABILITY_VALUES
        )
    }
    loss_prob: Callable = _valid_loss_functions[cfg.TRAIN.LOSS_PROBABILITY](**_kwarg)

    _kwarg = {
        k: v
        for k, v in zip(
            cfg.TRAIN.LOSS_SKELETON_KEYWORDS, cfg.TRAIN.LOSS_SKELETON_VALUES
        )
    }
    loss_skele: Callable = _valid_loss_functions[cfg.TRAIN.LOSS_SKELETON](**_kwarg)

    # Save each loss value in a list...
    avg_epoch_loss = [9999999999.9999999999]
    avg_epoch_embed_loss = [9999999999.9999999999]
    avg_epoch_prob_loss = [9999999999.9999999999]
    avg_epoch_skele_loss = [9999999999.9999999999]

    avg_val_loss = [9999999999.9999999999]
    avg_val_embed_loss = [9999999999.9999999999]
    avg_val_prob_loss = [9999999999.9999999999]
    avg_val_skele_loss = [9999999999.9999999999]

    # skel_crossover_loss = skoots.train.loss.split(n_iter=3, alpha=2)

    # Warmup... Get the first from train_data
    logging.info('generating warmup data')
    for images, masks, skeleton, skele_masks, baked in dataloader:
        break

    assert images is not None, len(dataloader)

    logging.info('starting warmup process')
    warmup_range = trange(cfg.TRAIN.N_WARMUP, desc="Warmup: {}")
    for w in warmup_range:
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
            out: Tensor = model(images)

            probability_map: Tensor = out[:, [-1], ...]
            vector: Tensor = out[:, 0:3:1, ...]
            predicted_skeleton: Tensor = out[:, [-2], ...]

            embedding: Tensor = vector_to_embedding(vector_scale, vector)
            out: Tensor = baked_embed_to_prob(embedding, baked, sigma(0))

            _loss_embed = loss_embed(
                out, masks.gt(0).float()
            )  # out = [B, 2/3, X, Y, Z?]
            _loss_prob = loss_prob(probability_map, masks.gt(0).float())
            _loss_skeleton = loss_skele(
                predicted_skeleton, skele_masks.gt(0).float()
            )  # + skel_crossover_loss(predicted_skeleton, skele_masks.gt(0).float())

            loss = (
                    (cfg.TRAIN.LOSS_EMBED_RELATIVE_WEIGHT * _loss_embed)
                    + (cfg.TRAIN.LOSS_PROBABILITY_RELATIVE_WEIGHT * _loss_prob)
                    + (cfg.TRAIN.LOSS_SKELETON_RELATIVE_WEIGHT * _loss_skeleton)
            )

            warmup_range.desc = f"{loss.item()}"

            if torch.isnan(loss):
                print(
                    f"Found NaN value in loss.\n\tLoss Embed: {_loss_embed}\n\tLoss Probability: {_loss_prob}"
                )
                print(f"\t{torch.any(torch.isnan(vector))}")
                print(f"\t{torch.any(torch.isnan(embedding))}")
                continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Train Step...
    epoch_range = (
        trange(epochs, desc=f"Loss = {1.0000000}") if rank == 0 else range(epochs)
    )
    for e in epoch_range:
        logging.info(f'Starting Epoch: {e+1}/{epochs}')
        _loss, _embed, _prob, _skele = [], [], [], []

        if cfg.TRAIN.DISTRIBUTED:
            train_sampler.set_epoch(e)

        for index, (images, masks, skeleton, skele_masks, baked) in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            logging.info(f'Starting Batch: {index+1}/{len(dataloader)}')

            with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
                out: Tensor = model(images)

                probability_map: Tensor = out[:, [-1], ...]
                vector: Tensor = out[:, 0:3:1, ...]
                predicted_skeleton: Tensor = out[:, [-2], ...]

                embedding: Tensor = vector_to_embedding(vector_scale, vector)
                out: Tensor = baked_embed_to_prob(embedding, baked, sigma(e))

                _loss_embed = loss_embed(
                    out, masks.gt(0).float()
                )  # out = [B, 2/3, X, Y, :w
                # Z?]
                _loss_prob = loss_prob(probability_map, masks.gt(0).float())
                _loss_skeleton = loss_skele(
                    predicted_skeleton, skele_masks.gt(0).float()
                )  # + skel_crossover_loss(predicted_skeleton, skele_masks.gt(0).float())

                logging.debug(f'\t{_loss_embed=}, {_loss_prob=}, {_loss_skeleton=}')

                # fuck this small amount of code.
                loss = (
                        (
                                cfg.TRAIN.LOSS_EMBED_RELATIVE_WEIGHT
                                * (1 if e > cfg.TRAIN.LOSS_EMBED_START_EPOCH else 0)
                                * _loss_embed
                        )
                        + (
                                cfg.TRAIN.LOSS_PROBABILITY_RELATIVE_WEIGHT
                                * (1 if e > cfg.TRAIN.LOSS_PROBABILITY_START_EPOCH else 0)
                                * _loss_prob
                        )
                        + (
                                cfg.TRAIN.LOSS_SKELETON_RELATIVE_WEIGHT
                                * (1 if e > cfg.TRAIN.LOSS_SKELETON_START_EPOCH else 0)
                                * _loss_skeleton
                        )
                )

                if torch.isnan(loss):
                    print(
                        f"Found NaN value in loss.\n\tLoss Embed: {_loss_embed}\n\tLoss Probability: {_loss_prob}"
                    )
                    print(f"\t{torch.any(torch.isnan(vector))}")
                    print(f"\t{torch.any(torch.isnan(embedding))}")
                    continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if e > swa_start:
                swa_model.update_parameters(model)

            _loss.append(loss.item())
            _embed.append(_loss_embed.item())
            _prob.append(_loss_prob.item())
            _skele.append(_loss_skeleton.item())

        avg_epoch_loss.append(mean(_loss))
        avg_epoch_embed_loss.append(mean(_embed))
        avg_epoch_prob_loss.append(mean(_prob))
        avg_epoch_skele_loss.append(mean(_skele))
        scheduler.step()

        if writer and (rank == 0):
            logging.info('writing to tensorboard')
            writer.add_scalar("lr", scheduler.get_last_lr()[-1], e)
            writer.add_scalar("Loss/train", avg_epoch_loss[-1], e)
            writer.add_scalar("Loss/embed", avg_epoch_embed_loss[-1], e)
            writer.add_scalar("Loss/prob", avg_epoch_prob_loss[-1], e)
            writer.add_scalar("Loss/skele-mask", avg_epoch_skele_loss[-1], e)
            write_progress(
                writer=writer,
                tag="Train",
                epoch=e,
                images=images,
                masks=masks,
                probability_map=probability_map,
                vector=vector,
                out=out,
                skeleton=skeleton,
                predicted_skeleton=predicted_skeleton,
                gt_skeleton=skele_masks,
            )

        # # Validation Step
        if e % 10 == 0 and valdiation_dataloader:
            logging.info('starting validation step')
            _loss, _embed, _prob, _skele = [], [], [], []
            for images, masks, skeleton, skele_masks, baked in valdiation_dataloader:
                with autocast(enabled=cfg.TRAIN.MIXED_PRECISION):  # Saves Memory!
                    with torch.no_grad():
                        out: Tensor = model(images)

                        probability_map: Tensor = out[:, [-1], ...]
                        predicted_skeleton: Tensor = out[:, [-2], ...]
                        vector: Tensor = out[:, 0:3:1, ...]

                        embedding: Tensor = vector_to_embedding(vector_scale, vector)
                        out: Tensor = baked_embed_to_prob(embedding, baked, sigma(e))

                        _loss_embed = loss_embed(out, masks.gt(0).float())
                        _loss_prob = loss_prob(probability_map, masks.gt(0).float())
                        _loss_skeleton = loss_prob(
                            predicted_skeleton, skele_masks.gt(0).float()
                        )

                        loss = (2 * _loss_embed) + (2 * _loss_prob) + _loss_skeleton

                        if torch.isnan(loss):
                            print(
                                f"Found NaN value in loss.\n\tLoss Embed: {_loss_embed}\n\tLoss Probability: {_loss_prob}"
                            )
                            print(f"\t{torch.any(torch.isnan(vector))}")
                            print(f"\t{torch.any(torch.isnan(embedding))}")
                            continue

                scaler.scale(loss)
                _loss.append(loss.item())
                _embed.append(_loss_embed.item())
                _prob.append(_loss_prob.item())
                _skele.append(_loss_skeleton.item())

            avg_val_loss.append(mean(_loss))
            avg_val_embed_loss.append(mean(_embed))
            avg_val_prob_loss.append(mean(_prob))
            avg_val_skele_loss.append(mean(_skele))

            if writer and (rank == 0):
                logging.info('writing validation to tensorboard')
                writer.add_scalar("Validation/train", avg_val_loss[-1], e)
                writer.add_scalar("Validation/embed", avg_val_embed_loss[-1], e)
                writer.add_scalar("Validation/prob", avg_val_prob_loss[-1], e)
                write_progress(
                    writer=writer,
                    tag="Validation",
                    epoch=e,
                    images=images,
                    masks=masks,
                    probability_map=probability_map,
                    vector=vector,
                    out=out,
                    skeleton=skeleton,
                    predicted_skeleton=predicted_skeleton,
                    gt_skeleton=skele_masks,
                )

        if rank == 0:
            epoch_range.desc = (
                    f"lr={scheduler.get_last_lr()[-1]:.3e}, Loss (train | val): "
                    + f"{avg_epoch_loss[-1]:.5f} | {avg_val_loss[-1]:.5f}"
            )

        logging.info('creating model state dict')
        state_dict = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )
        if e % 100 == 0:
            logging.info('writing intermediary state')
            torch.save(state_dict, cfg.TRAIN.SAVE_PATH + f"/test_{e}.trch")

    if rank == 0:
        state_dict = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )
        constants = {
            "cfg": cfg,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_epoch_loss": avg_epoch_loss,
            "avg_epoch_embed_loss": avg_epoch_embed_loss,
            "avg_epoch_prob_loss": avg_epoch_prob_loss,
            "avg_epoch_skele_loss": avg_epoch_skele_loss,
            "avg_val_loss": avg_epoch_loss,
            "avg_val_embed_loss": avg_epoch_embed_loss,
            "avg_val_prob_loss": avg_epoch_prob_loss,
            "avg_val_skele_loss": avg_epoch_skele_loss,
        }
        try:
            torch.save(
                constants,
                f"{cfg.TRAIN.SAVE_PATH}/{os.path.split(writer.log_dir)[-1]}.trch",
            )
        except:
            print(
                f"Could not save at: {cfg.TRAIN.SAVE_PATH}/{os.path.split(writer.log_dir)[-1]}.trch"
                f"Saving at {os.getcwd()}/{os.path.split(writer.log_dir)[-1]}.trch instead"
            )

            torch.save(
                constants,
                f"{os.getcwd()}/{os.path.split(writer.log_dir)[-1]}.trch",
            )
