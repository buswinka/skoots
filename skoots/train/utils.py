from typing import Dict, Optional, Union, List

import matplotlib.pyplot as plt
import torch
import torch.nn.modules.batchnorm
from torch import Tensor
from torchvision.utils import flow_to_image, draw_keypoints, make_grid


@torch.no_grad()
def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for batch_ind, data_dict in enumerate(loader):
        images, data_dict = hcat.lib.utils.prep_dict(
            data_dict, device
        )  # Preps output to handle potential batched data
        output = model(images, data_dict)

    # for input in loader:
    #     if isinstance(input, (list, tuple)):
    #         input = input[0]
    #     if device is not None:
    #         input = input.to(device)
    #
    #     model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


@torch.jit.script
def sum_loss(input: Dict[str, Tensor]) -> Optional[Tensor]:
    loss: Union[None, Tensor] = None
    for key in input:
        loss = loss + input[key] if loss is not None else input[key]
    return loss


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def show_box_pred(image: List, output: List, thr=0.90):
    c = ["nul", "r", "b", "y", "w"]

    boxes = output[0]["boxes"].detach().cpu().numpy().tolist()
    labels = output[0]["labels"].detach().cpu().int().numpy().tolist()
    scores = (
        output[0]["scores"].detach().cpu().numpy().tolist()
        if "scores" in output[0]
        else [1] * len(labels)
    )

    image = image.cpu()

    # x1, y1, x2, y2
    inp = image.numpy().transpose((1, 2, 0))
    if inp.shape[-1] == 1:
        inp = inp[:, :, 0]
        plt.imshow(inp, origin="lower", cmap="Greys_r")
    else:
        plt.imshow(inp, origin="lower")

    for i, box in enumerate(boxes):
        if scores[i] < thr:
            continue

        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        plt.plot([x1, x2], [y2, y2], c[labels[i]], alpha=1 * scores[i])
        plt.plot([x1, x2], [y1, y1], c[labels[i]], alpha=1 * scores[i])
        plt.plot([x1, x1], [y1, y2], c[labels[i]], alpha=1 * scores[i])
        plt.plot([x2, x2], [y1, y2], c[labels[i]], alpha=1 * scores[i])

    return plt.gcf()


def mask_overlay(image: Tensor, mask: Tensor, thr: Optional[float] = 0.5) -> Tensor:
    if image.ndim > 3:
        raise RuntimeError("3D Tensors not supported!!!", image.shape)

    _image = image.gt(thr)
    _mask = mask.gt(thr)

    _, x, y = _image.shape

    overlap = _image * _mask
    false_positive = torch.logical_not(_image) * mask
    false_negative = torch.logical_not(_mask) * image

    out = torch.zeros((3, x, y), device=image.device)

    out[:, overlap[0, ...].gt(0.5)] = 1.0
    out[0, false_positive[0, ...].gt(0.5)] = 0.5
    out[2, false_negative[0, ...].gt(0.5)] = 0.5

    return out


def write_progress(
    writer,
    tag,
    epoch,
    images,
    masks,
    probability_map,
    vector,
    out,
    skeleton: Optional = None,
    predicted_skeleton: Optional = None,
    gt_skeleton: Optional = None,
):
    _a = images[0, [0, 0, 0], :, :, 7].cpu()
    _b = masks[0, [0, 0, 0], :, :, 7].gt(0.5).float().cpu()
    _overlay = mask_overlay(_b[[0], ...], probability_map[0, :, :, :, 7].cpu())
    _c = _overlay.mul(255).round().type(torch.uint8).cpu()

    if skeleton:
        for k in skeleton[0]:  # List[Dict[str, Tensor]]
            keypoints: Tensor = skeleton[0][k][:, [1, 0]].unsqueeze(1)
            _c = draw_keypoints(_c, keypoints, colors="blue", radius=1)

    _d = flow_to_image(vector[0, [1, 0], :, :, 7].float()).cpu()
    if skeleton:
        for k in skeleton[0]:  # List[Dict[str, Tensor]]
            keypoints: Tensor = skeleton[0][k][:, [1, 0]].unsqueeze(1)
            _d = draw_keypoints(_d, keypoints, colors="blue", radius=1)

    # print(out.shape)
    out: List[Tensor] = (
        [out[[0], ...]] if not isinstance(out, list) else out
    )  # [B, 1 ,x ,y,z] ?
    # print(type(out), out[0].shape)
    _f = out[0][0, ..., 7].max(0)[0].float().cpu().unsqueeze(0)
    # print(_f.shape)
    _f = torch.concat((_f, _f, _f), dim=0)
    img_list = [_a, _b, _c, _d, _f]

    if predicted_skeleton is not None:
        # print(predicted_skeleton.shape)
        _g = predicted_skeleton[0, 0, ..., 7].float().cpu().unsqueeze(0)
        _g = torch.concat((_g, _g, _g), dim=0).mul(255).round().type(torch.uint8)
        # print(_g.shape)
        assert _g.ndim == 3, f"{_g.shape=}"
        if skeleton is not None:
            for k in skeleton[0]:  # List[Dict[str, Tensor]]
                keypoints: Tensor = skeleton[0][k][:, [1, 0]].unsqueeze(1)
                _g = draw_keypoints(_g, keypoints, colors="blue", radius=1)
        img_list.append(_g)

    if gt_skeleton is not None:
        # print(predicted_skeleton.shape)
        _g = gt_skeleton[0, 0, ..., 7].float().cpu().unsqueeze(0)
        _g = torch.concat((_g, _g, _g), dim=0).mul(255).round().type(torch.uint8)
        # print(_g.shape)
        assert _g.ndim == 3, f"{_g.shape=}"
        img_list.append(_g)

    for i, im in enumerate(img_list):
        assert isinstance(
            im, torch.Tensor
        ), f"im {i} is not a Tensor instead it is a {type(im)}, {img_list[i]}"
        assert (
            img_list[0].shape == img_list[i].shape
        ), f"im {i} is has shape {im.shape}, not {img_list[0].shape}"

    _img = make_grid(img_list, nrow=1, normalize=True, scale_each=True)

    writer.add_image(tag, _img, epoch, dataformats="CWH")
