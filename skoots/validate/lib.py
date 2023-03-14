import numpy as np
import torch
from torch import Tensor
import torchvision
from skoots.validate.utils import imread
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


def mask_to_bbox(mask: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Calculates the 3D bbox for each instance of an instance segmentation mask.

    Assumes each positive integer is an instance for class label. Returns a tensor of id labels and a tensor of bboxes
    bboxes are in format: [x0,y0,z0,x1,y1,z1]

    Assigns a bbox to each unique lablel! Does not mean each label has a valid bbox!!!

    Shapes:
        - mask: :math:`(1, X_{in}, Y_{in}, Z_{in})`
        - return[0]: Id labels: :math: `(N)`
        - return[1]: bboxes: :math: `(6, N)`


    :param mask: Input instance segmentation mask
    :return: id labels and bboxes
    """
    assert mask.ndim == 4, 'Mask ndim != 4'

    sparse_mask = mask.to_sparse()

    indices = sparse_mask.indices()
    values = sparse_mask.values()

    unique = torch.unique(values)  # assured to have no zeros because tensor is sparse

    bboxes = torch.empty((6, unique.shape[0]), device=mask.device, dtype=torch.int16)  # preallocate for speed

    for i, u in enumerate(unique):
        ind = indices[1::, values == u]  # just x,y,z dim of indicies

        bboxes[0, i] = ind[0].min()  # x0
        bboxes[1, i] = ind[1].min()  # y0
        bboxes[2, i] = ind[2].min()  # z0

        bboxes[3, i] = ind[0].max()  # x1
        bboxes[4, i] = ind[1].max()  # y1
        bboxes[5, i] = ind[2].max()  # z1

    return unique, bboxes


def valid_box_inds(boxes):
    """
    returns the inds of all valid boxes

    :param boxes: [6, N]
    :return: [N]
    """

    x0, y0, z0, x1, y1, z1 = boxes[0, :], boxes[1, :], boxes[2, :], boxes[3, :], boxes[4, :], boxes[5, :]
    inds = torch.logical_and(torch.logical_and((x1 > x0), (y1 > y0)), (z1 > z0))
    return inds


def box_iou(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute the IoU of the cartesian product of two sets of boxes.

    Each box in each set shall be (x0, y0, z0, x0, y0, z0).

    Shapes:
        a: :math:`(6, N)`.
        b: :math:`(6, M)`.
        returns: :math: `(N, M)`


    :param a: box 1
    :param b: box 2
    :return: iou of each box in 1 against all boxes in 2
    """
    if not valid_box_inds(a).bool().all():
        raise AssertionError("a does not follow (x0, y0, z0, x1, y1, z1) format.")

    if not valid_box_inds(b).bool().all():
        raise AssertionError("b does not follow (x0, y0, z0, x1, y2, z1) format.")

    a = a.T.float()  # we'll have buffer overflow otherwise
    b = b.T.float()  # we'll have buffer overflow otherwise

    # find intersection
    lower_bounds = torch.max(a[:, :3].unsqueeze(1), b[:, :3].unsqueeze(0)).float()  # (n, m, 3)
    upper_bounds = torch.min(a[:, 3:].unsqueeze(1), b[:, 3:].unsqueeze(0)).float()  # (n, m, 3)

    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 3)

    intersection = intersection_dims[:, :, 0].float() * \
                   intersection_dims[:, :, 1].float() * \
                   intersection_dims[:, :, 2].float()  # (n, m)

    # Find areas of each box in both sets
    areas_a = (a[:, 3] - a[:, 0]) * \
              (a[:, 4] - a[:, 1]) * \
              (a[:, 5] - a[:, 2])  # (n)

    areas_b = (b[:, 3] - b[:, 0]) * \
              (b[:, 4] - b[:, 1]) * \
              (b[:, 5] - b[:, 2])  # (m)

    union = areas_a.unsqueeze(1) + areas_b.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


def calculate_accuracies_from_bbox(ground_truth: Dict[str, Tensor], predictions: Dict[str, Tensor],
                                   device: str | None = None, threshold=0.1):
    """
    Calculates True positive, False Positive, False Negative from data_dict of segmentation 3d bboxes

    :param ground_truth:
    :param predictions:
    :param device:
    :param threshold:
    :return:
    """

    device = device if device else ground_truth['boxes'].device

    _gt = ground_truth['boxes'].to(device)
    _pred = predictions['boxes'].to(device)

    iou = box_iou(_gt, _pred)

    gt_max, gt_indicies = iou.max(dim=1)
    gt = torch.logical_not(gt_max.gt(threshold)) if iou.shape[1] > 0 else torch.ones(0)
    pred = torch.logical_not(iou.max(dim=0)[0].gt(threshold)) if iou.shape[0] > 0 else torch.ones(0)

    true_positive = torch.sum(torch.logical_not(gt))
    false_positive = torch.sum(pred)
    false_negative = torch.sum(gt)

    return true_positive, false_positive, false_negative,


if __name__ == "__main__":
    gt = imread('/home/chris/Dropbox (Partners HealthCare)/skoots/tests/test_data/hide_validate.labels.tif')
    pred = imread(
        '/home/chris/Dropbox (Partners HealthCare)/skoots/tests/test_data/hide_validate_skeleton_instance_mask.tif')
    aff_pred = imread(
        '/home/chris/Dropbox (Partners HealthCare)/skoots/tests/test_data/hide_validation_affinity_instance_segmentaiton.tif')
    # print(gt.shape)
    #
    # # labels, bboxes = mask_to_bbox(gt.cuda())
    labels, boxes = mask_to_bbox(gt.cuda())
    ind = valid_box_inds(boxes)
    ground_truth = {'labels': labels[ind], 'boxes': boxes[:, ind]}

    labels, boxes = mask_to_bbox(pred.cuda())
    ind = valid_box_inds(boxes)
    predictions = {'labels': labels[ind], 'boxes': boxes[:, ind]}

    labels, boxes = mask_to_bbox(aff_pred.cuda())
    ind = valid_box_inds(boxes)
    aff_predictions = {'labels': labels[ind], 'boxes': boxes[:, ind]}

    ap = []
    recall = []
    for i in range(100):
        i /= 100
        tp, fp, fn = calculate_accuracies_from_bbox(ground_truth, predictions, threshold=i)
        ap.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))

    ap = [a.cpu().numpy() for a in ap]
    recall = [r.cpu().numpy() for r in recall]
    plt.plot(np.arange(0, 100), ap)

    ap = []
    recall = []
    for i in range(100):
        i /= 100
        tp, fp, fn = calculate_accuracies_from_bbox(ground_truth, aff_predictions, threshold=i)
        ap.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))

    ap = [a.cpu().numpy() for a in ap]
    recall = [r.cpu().numpy() for r in recall]
    plt.plot(np.arange(0, 100), ap)

    plt.ylabel('AP')
    plt.xlabel('IoU Threshold: (%)')
    plt.show()
