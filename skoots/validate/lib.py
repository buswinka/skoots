import skoots.train.loss
from typing import Dict, Tuple

import torch
from torch import Tensor
from tqdm import tqdm

import skoots.train.loss
from skoots.validate.utils import imread


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
    assert mask.ndim == 4, "Mask ndim != 4"

    sparse_mask = mask.to_sparse()

    indices = sparse_mask.indices()
    values = sparse_mask.values()

    unique = torch.unique(values)  # assured to have no zeros because tensor is sparse

    bboxes = torch.empty(
        (6, unique.shape[0]), device=mask.device, dtype=torch.int16
    )  # preallocate for speed

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

    x0, y0, z0, x1, y1, z1 = (
        boxes[0, :],
        boxes[1, :],
        boxes[2, :],
        boxes[3, :],
        boxes[4, :],
        boxes[5, :],
    )
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
    lower_bounds = torch.max(
        a[:, :3].unsqueeze(1), b[:, :3].unsqueeze(0)
    ).float()  # (n, m, 3)
    upper_bounds = torch.min(
        a[:, 3:].unsqueeze(1), b[:, 3:].unsqueeze(0)
    ).float()  # (n, m, 3)

    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 3)

    intersection = (
        intersection_dims[:, :, 0].float()
        * intersection_dims[:, :, 1].float()
        * intersection_dims[:, :, 2].float()
    )  # (n, m)

    # Find areas of each box in both sets
    areas_a = (a[:, 3] - a[:, 0]) * (a[:, 4] - a[:, 1]) * (a[:, 5] - a[:, 2])  # (n)

    areas_b = (b[:, 3] - b[:, 0]) * (b[:, 4] - b[:, 1]) * (b[:, 5] - b[:, 2])  # (m)

    union = areas_a.unsqueeze(1) + areas_b.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


def calculate_accuracies_from_bbox(
    ground_truth: Dict[str, Tensor],
    predictions: Dict[str, Tensor],
    device: str | None = None,
    threshold=0.1,
):
    """
    Calculates True positive, False Positive, False Negative from data_dict of segmentation 3d bboxes

    :param ground_truth:
    :param predictions:
    :param device:
    :param threshold:
    :return:
    """

    device = device if device else ground_truth["boxes"].device

    _gt = ground_truth["boxes"].to(device)
    _pred = predictions["boxes"].to(device)

    iou = box_iou(_gt, _pred)

    gt_max, gt_indicies = iou.max(dim=1)
    gt = torch.logical_not(gt_max.gt(threshold)) if iou.shape[1] > 0 else torch.ones(0)
    pred = (
        torch.logical_not(iou.max(dim=0)[0].gt(threshold))
        if iou.shape[0] > 0
        else torch.ones(0)
    )

    true_positive = torch.sum(torch.logical_not(gt))
    false_positive = torch.sum(pred)
    false_negative = torch.sum(gt)

    return (
        true_positive,
        false_positive,
        false_negative,
    )


def accuracies_from_iou(iou: Tensor, thr: float = 0.1) -> Tensor:
    gt_max, gt_indicies = iou.max(dim=1)
    gt = torch.logical_not(gt_max.gt(thr)) if iou.shape[1] > 0 else torch.ones(0)
    pred = (
        torch.logical_not(iou.max(dim=0)[0].gt(thr))
        if iou.shape[0] > 0
        else torch.ones(0)
    )

    true_positive = torch.sum(torch.logical_not(gt))
    false_positive = torch.sum(pred)
    false_negative = torch.sum(gt)

    return (
        true_positive.cpu().item(),
        false_positive.cpu().item(),
        false_negative.cpu().item(),
    )


def mask_iou(gt: Tensor, pred: Tensor):
    """
    Calculates the IoU of each object on a per-mask-basis.

    :param gt: mask 1 with N instances
    :param pred: mask 2 with M instances
    :return: NxM matrix of IoU's
    """
    assert gt.shape == pred.shape, "Input tensors must be the same shape"
    assert gt.device == pred.device, "Input tensors must be on the same device"

    a_unique = gt.unique()
    a_unique = a_unique[a_unique > 0]

    b_unique = pred.unique()
    b_unique = b_unique[b_unique > 0]

    iou = torch.zeros(
        (a_unique.shape[0], b_unique.shape[0]), dtype=torch.float, device=gt.device
    )

    for i, au in tqdm(enumerate(a_unique), total=len(a_unique)):
        _a = gt == au
        touching = pred[
            _a
        ].unique()  # we only calculate iou of lables which have "contact with" our mask
        touching = touching[touching != 0]

        for j, bu in enumerate(b_unique):
            if torch.any(touching == bu):
                _b = pred == bu

                intersection = torch.logical_and(_a, _b).sum()
                union = torch.logical_or(_a, _b).sum()

                iou[i, j] = intersection / union
            else:
                iou[i, j] = 0.0

    return iou


def mask_dice(gt: Tensor, pred: Tensor):
    """
    Calculates the Dice Index of each object on a per-mask-basis.

    :param gt: mask 1 with N instances
    :param pred: mask 2 with M instances
    :return: NxM matrix of IoU's
    """
    assert gt.shape == pred.shape, "Input tensors must be the same shape"
    assert gt.device == pred.device, "Input tensors must be on the same device"

    a_unique = gt.unique()
    a_unique = a_unique[a_unique > 0]

    b_unique = pred.unique()
    b_unique = b_unique[b_unique > 0]

    dice = torch.zeros(
        (a_unique.shape[0], b_unique.shape[0]), dtype=torch.float, device=gt.device
    )

    for i, au in tqdm(enumerate(a_unique), total=len(a_unique)):
        _a = gt == au
        touching = pred[
            _a
        ].unique()  # we only calculate iou of lables which have "contact with" our mask
        touching = touching[touching != 0]

        for j, bu in enumerate(b_unique):
            if torch.any(touching == bu):
                _b = pred == bu

                numerator = torch.logical_and(_a, _b).sum() * 2
                denominator = _a.sum() + _b.sum()

                assert (
                    numerator < denominator
                ), f"{numerator=}, {denominator=}, {_a.sum()=}, {_b.sum()=}, {(_a*_b).sum()=}"

                dice[i, j] = numerator / denominator
            else:
                dice[i, j] = 0.0

    return dice


def mask_soft_cldice(gt: Tensor, pred: Tensor):
    """
    Calculates the Dice Index of each object on a per-mask-basis.

    :param gt: mask 1 with N instances
    :param pred: mask 2 with M instances
    :return: NxM matrix of IoU's
    """
    assert gt.shape == pred.shape, "Input tensors must be the same shape"
    assert gt.device == pred.device, "Input tensors must be on the same device"

    a_unique = gt.unique()
    a_unique = a_unique[a_unique > 0]

    b_unique = pred.unique()
    b_unique = b_unique[b_unique > 0]

    criterion = torch.compile(skoots.train.loss.soft_cldice())
    cldice = torch.zeros(
        (a_unique.shape[0], b_unique.shape[0]), dtype=torch.float, device=gt.device
    )

    for i, au in tqdm(enumerate(a_unique), total=len(a_unique)):
        _a = gt == au
        touching = pred[
            _a
        ].unique()  # we only calculate iou of lables which have "contact with" our mask
        touching = touching[touching != 0]

        for j, bu in enumerate(b_unique):
            if torch.any(touching == bu):
                _b = pred == bu
                cldice[i, j] = criterion(_b.float(), _a.float())
            else:
                cldice[i, j] = 0.0

    return cldice


def sparse_mask_iou(a: Tensor, b: Tensor) -> Tensor:
    """
    Calculates the IoU of each object on a per-mask-basis using sparse tensors.

    :param a: mask 1 with N instances
    :param b: mask 2 with M instances
    :return: NxM matrix of IoU's
    """
    raise NotImplementedError("In Development...")

    assert a.shape == b.shape, "Input tensors must be the same shape"
    assert a.device == b.device, "Input tensors must be on the same device"

    shape = a.shape

    a = a.to_sparse_coo()
    b = b.to_sparse_coo()

    a_unique = a.labels().unique()
    b_unique = b.labels().unique()

    a_indicies = a.indicies()
    b_indicies = b.indicies()

    iou = torch.zeros(
        (a_unique.shape[0], b_unique.shape[0]), dtype=torch.float, device=a.device
    )

    for i, au in tqdm(enumerate(a_unique)):
        for j, bu in enumerate(b_unique):
            _a_ind = a_indicies[a.lables() == au]
            _b_ind = b_indicies[b.lables() == bu]

            a_sparse = torch.sparse_coo_tensor(
                indices=_a_ind, labels=torch.ones_like(a.labels() == au), size=shape
            )
            b_sparse = torch.sparse_coo_tensor(
                indices=_b_ind, labels=torch.ones_like(b.labels() == bu), size=shape
            )


def f1_score(tp, fp, fn):
    num = 2 * tp
    dem = 2 * tp + fp + fn
    return num / dem


def _iou_instance_dict(a: Tensor, b: Tensor) -> Dict[int, Tensor]:
    """
    Given two instance masks, compares each instance in b against a. Usually assumes A is the ground truth.

    :param a: Mask A
    :param b: Mask B
    :return:  Dict of instances and every IOU for each instance
    """
    a_unique = a.unique()
    a_unique = a_unique[a_unique > 0]

    b_unique = b.unique()
    b_unique = b_unique[b_unique > 0]

    iou = {}

    for i, au in tqdm(enumerate(a_unique), total=len(a_unique)):
        _a = a == au

        touching = b[
            _a
        ].unique()  # we only calculate iou of lables which have "contact with" our mask
        touching = touching[touching != 0]
        iou[au] = []

        for j, bu in enumerate(b_unique):
            if torch.any(touching == bu):
                _b = b == bu

                intersection = torch.logical_and(_a, _b).sum()
                union = torch.logical_or(_a, _b).sum()
                iou[au].append((intersection / union).item())

    return iou


def get_segmentation_errors(ground_truth: Tensor, predicted: Tensor) -> float:
    """
    Calculates the IoU of each object on a per-mask-basis.

    :param ground_truth: mask 1 with N instances
    :param predicted: mask 2 with M instances
    :return: NxM matrix of IoU's
    """

    iou = _iou_instance_dict(ground_truth, predicted)
    for k, v in iou.items():
        iou[k] = torch.tensor(v)

    num_split = 0
    for k, v in iou.items():
        if v.gt(0.2).int().sum() > 1:
            num_split += 1

    over_segmentation_rate = num_split / len(iou)

    iou = _iou_instance_dict(predicted, ground_truth)
    for k, v in iou.items():
        iou[k] = torch.tensor(v)

    num_split = 0
    for k, v in iou.items():
        if v.gt(0.2).int().sum() > 1:
            num_split += 1

    under_segmentation_rate = num_split / len(iou)

    return over_segmentation_rate, under_segmentation_rate


if __name__ == "__main__":
    gt = imread("../../tests/test_data/hide_validate.labels.tif")[..., 2:-2]
    pred = imread("../../tests/test_data/hide_validate_skeleton_instance_mask.tif")[
        ..., 2:-2
    ]
    aff_pred = imread(
        "../../tests/test_data/hide_validation_affinity_instance_segmentaiton.tif"
    )[..., 2:-2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt = gt.to(device)
    pred = pred.to(device)
    aff_pred = aff_pred.to(device)

    u, c = aff_pred.unique(return_counts=True)
    for a, b in tqdm(zip(u, c)):
        if b < 500:
            aff_pred[aff_pred == a] = 0

    skoots_seg_errors = get_segmentation_errors(gt, pred)
    aff_seg_errors = get_segmentation_errors(gt, aff_pred)

    # print('SKOOTS IOU')
    # if not os.path.exists('../../tests/iou_gt_skoots_mask.trch'):
    #     iou_skoots = mask_iou(gt, pred)
    #     torch.save(iou_skoots, '../../tests/iou_gt_skoots_mask.trch')
    # else:
    #     iou_skoots = torch.load('../../tests/iou_gt_skoots_mask.trch')
    #
    # print('AFFINITES IOU')
    # if not os.path.exists('../../tests/iou_gt_affinites_mask.trch'):
    #     iou_aff = mask_iou(gt, aff_pred)
    #     torch.save(iou_aff, '../../tests/iou_gt_affinites_mask.trch')
    # else:
    #     iou_aff = torch.load('../../tests/iou_gt_affinites_mask.trch')
    #
    # # iou_skoots = torch.load('../../tests/iou_gt_skoots_mask.trch')
    # # iou_aff = torch.load('../../tests/iou_gt_affinites_mask.trch')
    #
    #
    #
    # tfp_skoots = [accuracies_from_iou(iou_skoots, thr/100) for thr in range(100)]
    # tfp_aff = [accuracies_from_iou(iou_aff, thr/100) for thr in range(100)]
    #
    # precision_skoots = [(tp /(tp + fp)) for (tp, fp, fn) in tfp_skoots]
    # recall_skoots = [(tp / (tp + fn)) for (tp, fp, fn) in tfp_skoots]
    #
    # precision_aff = [(tp/(tp+fp)) for (tp, fp, fn) in tfp_aff]
    # recall_aff = [(tp / (tp + fn)) for (tp, fp, fn) in tfp_aff]
    #
    # f1_skoots = [f1_score(*a) for a in tfp_skoots]
    # f1_aff = [f1_score(*a) for a in tfp_aff]
    #
    # plt.plot(np.arange(0, 100), precision_skoots)
    # plt.plot(np.arange(0, 100), precision_aff)
    # plt.legend(['SKOOTS', 'AFFINITIES'])
    # plt.ylabel('Precision')
    # plt.xlabel('IoU Threshold: (%)')
    # plt.show()
    #
    # plt.plot(np.arange(0, 100), recall_skoots)
    # plt.plot(np.arange(0, 100), recall_aff)
    # plt.legend(['SKOOTS', 'AFFINITIES'])
    # plt.ylabel('Recall')
    # plt.xlabel('IoU Threshold: (%)')
    # plt.show()
    #
    # plt.plot(np.arange(0, 100), f1_skoots)
    # plt.plot(np.arange(0, 100), f1_aff)
    # plt.legend(['SKOOTS', 'AFFINITIES'])
    # plt.ylabel('F1')
    # plt.xlabel('IoU Threshold: (%)')
    # plt.show()
