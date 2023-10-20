from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from skoots.lib.morphology import binary_erosion
from skoots.lib.utils import crop_to_identical_size


def tversky_graphable(pred, gt, alpha, beta):
    true_positive: Tensor = pred.mul(gt).sum()
    false_positive: Tensor = torch.logical_not(gt).mul(pred).sum().add(1e-10).mul(alpha)
    false_negative: Tensor = ((1 - pred) * gt).sum() * beta

    tversky = (true_positive + 1e-10) / (
        true_positive + false_positive + false_negative + 1e-10
    )

    return 1 - tversky


class jaccard(nn.Module):
    def __init__(self):
        super(jaccard, self).__init__()

    def forward(
        self, predicted: torch.Tensor, ground_truth: torch.Tensor, eps: float = 1e-10
    ) -> torch.Tensor:
        """
        Returns jaccard index of two torch.Tensors

        :param predicted: [B, I, X, Y, Z] torch.Tensor
                - probabilities calculated from hcat.utils.embedding_to_probability
                  where B: is batch size, I: instances in image
        :param ground_truth: [B, I, X, Y, Z] torch.Tensor
                - segmentation mask for each instance (I).
        :param eps: float
                - Very small number to ensure numerical stability. Default 1e-10
        :return: jaccard_loss: [1] torch.Tensor
                - Result of Loss Function Calculation
        """

        # Crop both tensors to the same shape
        predicted, ground_truth = crop_to_identical_size(predicted, ground_truth)

        intersection = (predicted * ground_truth).sum().add(eps)
        union = (predicted + ground_truth).sum().sub(intersection).add(eps)

        return 1.0 - (intersection / union)


class dice(nn.Module):
    def __init__(self):
        super(dice, self).__init__()

    def forward(
        self, predicted: torch.Tensor, ground_truth: torch.Tensor, smooth: float = 1e-10
    ) -> torch.Tensor:
        """
        Returns dice index of two torch.Tensors

        :param predicted: [B, I, X, Y, Z] torch.Tensor
                - probabilities calculated from hcat.utils.embedding_to_probability
                  where B: is batch size, I: instances in image
        :param ground_truth: [B, I, X, Y, Z] torch.Tensor
                - segmentation mask for each instance (I).
        :param smooth: float
                - Very small number to ensure numerical stability. Default 1e-10
        :return: dice_loss: [1] torch.Tensor
                - Result of Loss Function Calculation
        """

        # Crop both tensors to the same shape
        predicted, ground_truth = crop_to_identical_size(predicted, ground_truth)

        intersection = (predicted * ground_truth).sum().add(smooth)
        denominator = (predicted + ground_truth).sum().add(smooth)
        loss = 2 * intersection / denominator

        return 1 - loss


class tversky(nn.Module):
    def __init__(self, alpha: float, beta: float, eps: float):
        """
        Returns dice index of two torch.Tensors

        :param alpha: float
                - Value which penalizes False Positive Values
        :param beta: float
                - Value which penalizes False Negatives
        :param eps: float
                - Numerical stability term
        """
        super(tversky, self).__init__()

        self.alpha = torch.tensor(float(alpha))
        self.beta = torch.tensor(float(beta))
        self.eps = torch.tensor(float(eps))

    def forward(
        self, predicted: Union[Tensor, List[Tensor]], ground_truth: Tensor
    ) -> Tensor:
        if self.alpha.device != predicted.device:  # silently caches device
            self.alpha.to(predicted.device)
            self.beta.to(predicted.device)
            self.eps.to(predicted.device)

        futures: List[torch.jit.Future[torch.Tensor]] = []

        # List of Tensors
        if isinstance(predicted, list):
            for i, pred in enumerate(predicted):
                futures.append(
                    torch.jit.fork(
                        self._tversky,
                        pred,
                        ground_truth[i, ...],
                        self.alpha,
                        self.beta,
                        self.eps,
                    )
                )

        # Already Batched Tensor
        elif isinstance(predicted, Tensor):
            for i in range(predicted.shape[0]):
                futures.append(
                    torch.jit.fork(
                        self._tversky,
                        predicted[i, ...],
                        ground_truth[i, ...],
                        self.alpha,
                        self.beta,
                        self.eps,
                    )
                )

        results: List[Tensor] = []
        for future in futures:
            results.append(torch.jit.wait(future))

        return torch.mean(torch.stack(results))

    @staticmethod
    def _tversky(
        pred: Tensor, gt: Tensor, alpha: Tensor, beta: Tensor, eps: float = 1e-8
    ):
        """
        tversky loss on per image basis.


        Args:
            pred: [N, X, Y, Z] Tensor of predicted segmentation masks (N instances)
            gt: [N, X, Y, Z] Tensor of ground truth segmentation masks (N instances)
            alpha: Penalty to false positives
            beta: Penalty to false negatives
            eps: stability parameter

        Returns:

        """

        # ------------------- Expand Masks
        unique = torch.unique(gt)
        unique = unique[unique != 0]

        # assert gt.ndim == 4, f'{gt.shape=}'

        _, x, y, z = gt.shape
        nd_masks = torch.zeros((unique.shape[0], x, y, z), device=pred.device)
        for i, id in enumerate(unique):
            nd_masks[i, ...] = (gt == id).float().squeeze(0)

        pred, nd_masks = crop_to_identical_size(pred, nd_masks)

        # assert not torch.any(torch.isnan(pred)), torch.sum(torch.isnan(pred))

        true_positive: Tensor = pred.mul(nd_masks).sum()
        false_positive: Tensor = (
            torch.logical_not(nd_masks).mul(pred).sum().add(1e-10).mul(alpha)
        )
        false_negative: Tensor = ((1 - pred) * nd_masks).sum() * beta

        # assert not torch.any(torch.isnan(true_positive)), torch.sum(torch.isnan(true_positive))
        # assert not torch.any(torch.isnan(false_negative))
        # assert not torch.any(torch.isnan(false_positive))
        #
        # assert not torch.any(torch.isinf(true_positive))
        # assert not torch.any(torch.isinf(false_negative))
        # assert not torch.any(torch.isinf(false_positive))

        tversky = (true_positive + eps) / (
            true_positive + false_positive + false_negative + eps
        )

        return 1 - tversky

    def __repr__(self):
        return f"LossFn[name=tversky, alpha={self.alpha.item()}, beta={self.beta.item()}, eps={self.eps.item()}"


class split(nn.Module):
    def __init__(self, n_iter: int = 2, alpha: float = 2.0, device: str = "cpu"):
        """
        The "oh shit my skeletons have split" loss. This basically checks if an edge has crossed the middle
        of a GT object. If it does, it applies a crazy loss.

        Approximates the distance function by just eroding and adding a bunch.
        Approximates the edge function by subtracting the prediction by the eroded

        For speed, will only check for pixels :math:`n_{iter}` away. So if :math:`n_{iter} = 3' the maximum distance
        any pixel might be from an edge would be 3.

        Formally:
            if :math:`E` is the edge function and :math:`\Phi` is the distance function, we compute the loss :math:`L(s, p)`
            where :math:`s` is the ground truth skeleton, and :math:`p` is the predicted skeleton where

        .. math::
            L(s, p) = E(p)^{ \alpha \Phi(s)} - 1


        :param n_iter: Number of times to perform erosion for distance calculation.
        :param alpha: Scale factor for exponential loss. Large values penalize breakages more.
        :param device: a torch.device - 'cuda' or 'cpu'
        """

        super(split, self).__init__()
        self.n = n_iter
        self._device = device
        self.a = torch.tensor(alpha, device=device)

    def forward(self, pred, gt):
        distance = torch.zeros_like(gt) + gt

        for _ in range(self.n):
            gt = binary_erosion(gt)
            distance = distance + gt  # psuedo distance function...

        distance = distance.div(self.n - 1)

        pred = pred.sub(binary_erosion(pred)).mul(2)  # cheeky edge detection function

        _split_loss = self._split_loss(edges=pred, distance=distance, a=self.a)

        return _split_loss.mean()

    @staticmethod
    # @torch.jit.script
    def _split_loss(edges: Tensor, distance: Tensor, a: Tensor):
        return torch.pow(edges, a * distance)


############## clDICE LOSS FROM: https://github.com/jocpae/clDice/blob/master/cldice_loss/pytorch/soft_skeleton.py


def soft_erode(img: Tensor) -> Tensor:
    """approximates morphological operations through max_pooling for 2D and 3D"""
    if len(img.shape) == 4:
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)
    elif len(img.shape) == 5:
        p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
        p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
        p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img: Tensor) -> Tensor:
    """approximates morphological operations through max_pooling for 2D and 3D"""
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def soft_open(img: Tensor) -> Tensor:
    """approximates morphological operations through max_pooling for 2D and 3D"""
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: Tensor, iter_: int) -> Tensor:
    """
    Performs a soft-skeletonization by terativly performing "soft morphological operations"

    :param img: Image to perform operation on
    :param iter_: Number of times to perform the operation
    :return: Soft-skeleton
    """
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth=1.0):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, predicted: Tensor, ground_truth: Tensor) -> Tensor:
        """
        Calculates the soft-clDice metric on a true and predicted value

        :param ground_truth:
        :param predicted:
        :return:
        """
        skeleton_predicted = soft_skeletonize(predicted, self.iter)
        skeleton_true = soft_skeletonize(ground_truth, self.iter)

        tprec = (
            torch.sum(torch.multiply(skeleton_predicted, ground_truth)[:, 1:, ...])
            + self.smooth
        ) / (torch.sum(skeleton_predicted[:, 1:, ...]) + self.smooth)
        tsens = (
            torch.sum(torch.multiply(skeleton_true, predicted)[:, 1:, ...])
            + self.smooth
        ) / (torch.sum(skeleton_true[:, 1:, ...]) + self.smooth)

        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)

        return cl_dice


def soft_dice(predicted: Tensor, ground_truth: Tensor, smooth: int = 1) -> Tensor:
    """
    Computes the soft dice metric

    :param ground_truth:
    :param predicted:
    :param smooth: smoothing factor to prevent division by zero
    :return:
    """
    intersection = torch.sum((ground_truth * predicted))
    coeff = (2.0 * intersection + smooth) / (
        torch.sum(ground_truth) + torch.sum(predicted) + smooth
    )

    return 1.0 - coeff


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth=1.0):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, predicted: Tensor, ground_truth: Tensor) -> Tensor:
        """
        Calculates a singular loss value combining soft-Dice and soft-clDice which can be used to train
        a neural network

        :param predicted: Input tensor
        :param ground_truth: Ground Truth Tensor
        :return: Single value which to perform a backwards pass
        """

        dice = soft_dice(ground_truth, predicted)

        skel_pred = soft_skeletonize(predicted, self.iter)
        skel_true = soft_skeletonize(ground_truth, self.iter)

        tprec = (torch.sum(skel_pred * ground_truth) + self.smooth) / (
            torch.sum(skel_pred) + self.smooth
        )
        tsens = (torch.sum(skel_true * predicted) + self.smooth) / (
            torch.sum(skel_true) + self.smooth
        )

        cl_dice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)

        return (1.0 - self.alpha) * dice + self.alpha * cl_dice


if __name__ == "__main__":
    lossfn = soft_dice_cldice()

    predicted = torch.rand((1, 1, 20, 20, 10), device="cpu")
    gt = torch.rand((1, 1, 20, 20, 10), device="cpu").round().float()
    a = lossfn(predicted, gt)
    print(a)
