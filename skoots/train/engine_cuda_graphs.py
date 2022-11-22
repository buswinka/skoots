import os.path
import numpy as np
import skimage.io as io
import skoots.train.loss
from skoots.train.utils import sum_loss, show_box_pred
from skoots.train.sigma import Sigma
from skoots.lib.vector_to_embedding import vector_to_embedding
from skoots.lib.embedding_to_prob import baked_embed_to_prob

from skoots.train.utils import mask_overlay, write_progress

from typing import List, Tuple, Callable, Union, OrderedDict, Optional
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.detection import FasterRCNN

from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import trange
from torch.cuda.amp import GradScaler, autocast
from statistics import mean
from torchvision.utils import flow_to_image, draw_keypoints, make_grid
import matplotlib.pyplot as plt
import torch.optim.swa_utils
Dataset = Union[Dataset, DataLoader]

# SKELETON TRAINING ENGINE
def engine(
        model: FasterRCNN,
        lr: float,
        wd: float,
        vector_scale: Tuple[int, int, int],
        epochs: int,
        optimizer: Optimizer,
        scheduler,
        sigma: Sigma,
        loss_embed,
        loss_prob,
        loss_skele,
        device: str,
        savepath: str,
        train_data: Dataset,
        rank: int,
        val_data: Optional[Dataset] = None,
        train_sampler=None,
        test_sampler=None,
        writer=None,
        verbose=False,
        distributed=True,
        mixed_precision=False,
        n_warmup: int = 100,
        force=False,
        **kwargs,
) -> Tuple[OrderedDict, OrderedDict, List[float]]:

    # Print out each kwarg to std out
    if verbose and rank == 0:
        print('Initiating Training Run', flush=False)
        vars = locals()
        for k in vars:
            if k != 'model':
                print(f'\t> {k}: {vars[k]}', flush=False)
        print('', flush=True)

    num = torch.tensor(vector_scale, device=device)


    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = scheduler(optimizer)
    scaler = GradScaler(enabled=mixed_precision)

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = 100
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.05)

    # Save each loss value in a list...
    avg_epoch_loss = []
    avg_epoch_embed_loss = []
    avg_epoch_prob_loss = []
    avg_epoch_skele_loss = []

    avg_val_loss = []
    avg_val_embed_loss = []
    avg_val_prob_loss = []
    avg_val_skele_loss = []

    # skel_crossover_loss = skoots.train.loss.split(n_iter=3, alpha=2)


    # Warmup... Get the first from train_data
    for static_images, static_masks, skeleton, static_skele_masks, static_baked in train_data:
        pass

    static_sigma = sigma(epochs-1).to(device=device)

    # warmup...
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream(device))
    warmup_range = trange(n_warmup, desc='Warmup: {}')
    with torch.cuda.stream(s):
        for w in warmup_range:
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=mixed_precision):  # Saves Memory!
                static_out: Tensor = model(static_images)
                # static_probability_map: Tensor = static_out[:, [-1], ...]
                # static_vector: Tensor = static_out[:, 0:3:1, ...]
                # static_predicted_skeleton: Tensor = static_out[:, [-2], ...]

                static_embedding: Tensor = vector_to_embedding(num, static_out[:, 0:3:1, ...])
                static_predicted_prob: Tensor = baked_embed_to_prob(static_embedding, static_baked, static_sigma)

                static_loss_embed = loss_embed(static_out[:, [-1], ...],
                                               static_masks.gt(0).float())  # out = [B, 2/3, X, Y, Z?]
                static_loss_prob = loss_prob(static_out[:, [-1], ...], static_masks.gt(0).float())
                static_loss_skeleton = loss_skele(static_out[:, [-2], ...], static_skele_masks.gt(0).float())
                static_loss = static_loss_embed + (1 * static_loss_prob) + (1 * static_loss_skeleton)

                warmup_range.desc = f'{static_loss.item()}'

            scaler.scale(static_loss).backward()
            scaler.step(optimizer)
            scaler.update()

    torch.cuda.current_stream().wait_stream(s)

    # capture
    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        with autocast(enabled=mixed_precision):
            static_out: Tensor = model(static_images)
            # static_probability_map: Tensor = static_out[:, [-1], ...]
            # static_vector: Tensor = static_out[:, 0:3:1, ...]
            # static_predicted_skeleton: Tensor = static_out[:, [-2], ...]

            static_embedding: Tensor = vector_to_embedding(num, static_out[:, 0:3:1, ...])
            static_predicted_prob: Tensor = baked_embed_to_prob(static_embedding, static_baked, static_sigma)

            static_loss_embed = loss_embed(static_out[:, [-1], ...], static_masks.gt(0).float())  # out = [B, 2/3, X, Y, Z?]
            static_loss_prob = loss_prob(static_out[:, [-1], ...], static_masks.gt(0).float())
            static_loss_skeleton = loss_skele(static_out[:, [-2], ...], static_skele_masks.gt(0).float())
            static_loss = static_loss_embed + (1 * static_loss_prob) + (1 * static_loss_skeleton)

            warmup_range.desc = f'{static_loss.item()}'

            scaler.scale(static_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            swa_model.update_parameters(model)


    # Train Step...
    epoch_range = trange(epochs, desc=f'Loss = {1.0000000}') if rank == 0 else range(epochs)
    for e in epoch_range:
        _loss, _embed, _prob, _skele = [], [], [], []

        if distributed:
            train_sampler.set_epoch(e)

        for images, masks, skeleton, skele_masks, baked in train_data:

            static_images.copy_(images)
            static_masks.copy_(masks)
            static_skele_masks.copy_(skele_masks)
            static_baked.copy_(baked)

            g.replay()

            # optimizer.zero_grad(set_to_none=True)
            #
            # with autocast(enabled=mixed_precision):  # Saves Memory!
            #     out: Tensor = model(images)
            #
            #     probability_map: Tensor = out[:, [-1], ...]
            #     vector: Tensor = out[:, 0:3:1, ...]
            #     predicted_skeleton: Tensor = out[:, [-2], ...]
            #
            #     embedding: Tensor = vector_to_embedding(num, vector)
            #     out: Tensor = baked_embed_to_prob(embedding, baked, sigma(e))
            #
            #     _loss_embed = loss_embed(out, masks.gt(0).float())  # out = [B, 2/3, X, Y, :w
            #     # Z?]
            #     _loss_prob = loss_prob(probability_map, masks.gt(0).float())
            #     _loss_skeleton = loss_skele(predicted_skeleton, skele_masks.gt(
            #         0).float())  # + skel_crossover_loss(predicted_skeleton, skele_masks.gt(0).float())
            #
            #     loss = _loss_embed + (1 * _loss_prob) + ((1 if e > 10 else 0) * _loss_skeleton)
            #

            #
            #     if torch.isnan(loss):
            #         print(f'Found NaN value in loss.\n\tLoss Embed: {_loss_embed}\n\tLoss Probability: {_loss_prob}')
            #         print(f'\t{torch.any(torch.isnan(vector))}')
            #         print(f'\t{torch.any(torch.isnan(embedding))}')
            #         continue
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            if e > swa_start:
                swa_model.update_parameters(model)

            _loss.append(static_loss.item())
            _embed.append(static_loss_embed.item())
            _prob.append(static_loss_prob.item())
            _skele.append(static_loss_skeleton.item())

        avg_epoch_loss.append(mean(_loss))
        avg_epoch_embed_loss.append(mean(_embed))
        avg_epoch_prob_loss.append(mean(_prob))
        avg_epoch_skele_loss.append(mean(_skele))
        scheduler.step()

        if writer and (rank == 0):
            writer.add_scalar('lr', scheduler.get_last_lr()[-1], e)
            writer.add_scalar('Loss/train', avg_epoch_loss[-1], e)
            writer.add_scalar('Loss/embed', avg_epoch_embed_loss[-1], e)
            writer.add_scalar('Loss/prob', avg_epoch_prob_loss[-1], e)
            writer.add_scalar('Loss/skele-mask', avg_epoch_skele_loss[-1], e)
            write_progress(writer=writer, tag='Train', epoch=e, images=images, masks=masks,
                           probability_map=probability_map,
                           vector=vector, out=out, skeleton=skeleton,
                           predicted_skeleton=predicted_skeleton, gt_skeleton=skele_masks)

        # # Validation Step
        if e % 10 == 0 and val_data:
            _loss, _embed, _prob, _skele = [], [], [], []
            for images, masks, skeleton, skele_masks, baked in val_data:
                with autocast(enabled=mixed_precision):  # Saves Memory!
                    with torch.no_grad():
                        out: Tensor = swa_model(images)

                        probability_map: Tensor = out[:, [-1], ...]
                        predicted_skeleton: Tensor = out[:, [-2], ...]
                        vector: Tensor = out[:, 0:3:1, ...]

                        embedding: Tensor = vector_to_embedding(num, vector)
                        out: Tensor = baked_embed_to_prob(embedding, baked, sigma(e))

                        _loss_embed = loss_embed(out, masks.gt(0).float())
                        _loss_prob = loss_prob(probability_map, masks.gt(0).float())
                        _loss_skeleton = loss_prob(predicted_skeleton, skele_masks.gt(0).float())

                        loss = (2 * _loss_embed) + (2 * _loss_prob) + _loss_skeleton

                        if torch.isnan(loss):
                            print(
                                f'Found NaN value in loss.\n\tLoss Embed: {_loss_embed}\n\tLoss Probability: {_loss_prob}')
                            print(f'\t{torch.any(torch.isnan(vector))}')
                            print(f'\t{torch.any(torch.isnan(embedding))}')
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
                writer.add_scalar('Validation/train', avg_val_loss[-1], e)
                writer.add_scalar('Validation/embed', avg_val_embed_loss[-1], e)
                writer.add_scalar('Validation/prob', avg_val_prob_loss[-1], e)
                write_progress(writer=writer, tag='Validation', epoch=e, images=images, masks=masks,
                               probability_map=probability_map,
                               vector=vector, out=out, skeleton=skeleton,
                               predicted_skeleton=predicted_skeleton, gt_skeleton=skele_masks)

        if rank == 0:
            epoch_range.desc = f'lr={scheduler.get_last_lr()[-1]:.3e}, Loss (train | val): ' + f'{avg_epoch_loss[-1]:.5f} | {avg_val_loss[-1]:.5f}'

        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        if e % 100 == 0:
            torch.save(state_dict, savepath + f'/test_{e}.trch')

    return state_dict, optimizer.state_dict(), avg_val_loss
