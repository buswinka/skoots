import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Union

from skoots.lib.utils import crop_to_identical_size
from skoots.lib.morphology import binary_erosion


class jaccard(nn.Module):
    def __init__(self):
        super(jaccard, self).__init__()

    def forward(self, predicted: torch.Tensor, ground_truth: torch.Tensor,
                eps: float = 1e-10) -> torch.Tensor:
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

    def forward(self, predicted: torch.Tensor, ground_truth: torch.Tensor,
                smooth: float = 1e-10) -> torch.Tensor:
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
    def __init__(self, alpha, beta, eps, device: str = 'cpu'):
        """
        Returns dice index of two torch.Tensors

        :param smooth: float
                - Very small number to ensure numerical stability. Default 1e-10
        :param alpha: float
                - Value which penalizes False Positive Values
        :param beta: float
                - Value which penalizes False Negatives
        :param gamma: float
                - Focal loss term
        """
        super(tversky, self).__init__()

        self.alpha = torch.tensor(alpha, device=device)
        self.beta = torch.tensor(beta, device=device)
        self.eps = torch.tensor(eps, device=device)

    def forward(self, predicted: Union[Tensor, List[Tensor]], ground_truth: Tensor) -> Tensor:
        # assert isinstance(ground_truth, Tensor)
        # assert ground_truth.shape[0] == len(
        #     predicted), f'Batch sizes are note the same!, {len(predicted)}, {ground_truth.shape}'

        futures: List[torch.jit.Future[torch.Tensor]] = []

        # List of Tensors
        if isinstance(predicted, list):
            for i, pred in enumerate(predicted):
                futures.append(
                    torch.jit.fork(self._tversky, pred, ground_truth[i, ...], self.alpha, self.beta, self.eps))

        # Already Batched Tensor
        elif isinstance(predicted, Tensor):
            for i in range(predicted.shape[0]):
                futures.append(
                    torch.jit.fork(self._tversky, predicted[i, ...], ground_truth[i, ...], self.alpha, self.beta,
                                   self.eps))

        results: List[Tensor] = []
        for future in futures:
            results.append(torch.jit.wait(future))

        return torch.mean(torch.stack(results))

    @staticmethod
    def _tversky(pred: Tensor, gt: Tensor, alpha: Tensor, beta: Tensor, eps: float = 1e-8):
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
        false_positive: Tensor = torch.logical_not(nd_masks).mul(pred).sum().add(1e-10).mul(alpha)
        false_negative: Tensor = ((1 - pred) * nd_masks).sum() * beta

        # assert not torch.any(torch.isnan(true_positive)), torch.sum(torch.isnan(true_positive))
        # assert not torch.any(torch.isnan(false_negative))
        # assert not torch.any(torch.isnan(false_positive))
        #
        # assert not torch.any(torch.isinf(true_positive))
        # assert not torch.any(torch.isinf(false_negative))
        # assert not torch.any(torch.isinf(false_positive))

        tversky = (true_positive + eps) / (true_positive + false_positive + false_negative + eps)

        return 1 - tversky

    def __repr__(self):
        return f'LossFn[name=tversky, alpha={self.alpha.item()}, beta={self.beta.item()}, eps={self.eps.item()}'


class split(nn.Module):
    def __init__(self, n_iter: int = 2, alpha: float = 2.0, device: str = 'cpu'):
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
            print("yeet")
            gt = binary_erosion(gt)
            distance = distance + gt   # psuedo distance function...

        distance = distance.div(self.n)
        assert distance.max() <= 1

        pred = pred.sub(binary_erosion(pred)) \
                   .mul(2)  # cheeky edge detection function

        return self._split_loss(edges=pred, distance=distance, a=self.a).mean()

    @staticmethod
    @torch.jit.script
    def _split_loss(edges: Tensor, distance: Tensor, a: Tensor):
        return torch.pow(edges, a * distance)


if __name__ == '__main__':
    lossfn = torch.jit.script(split(2, 1.5, device='cpu'))
    print(lossfn)

    predicted = torch.rand((1, 1, 20, 20, 10), device='cpu')
    gt = torch.rand((1, 1, 20, 20, 10), device='cpu').mul(10).round().float()

    a = lossfn(predicted, gt)
    print(a)