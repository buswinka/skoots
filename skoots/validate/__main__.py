import argparse
import logging
import os.path

import matplotlib.pyplot as plt
import torch

from skoots.validate.lib import (
    get_segmentation_errors,
    mask_iou,
    accuracies_from_iou,
    f1_score,
    mask_dice,
    mask_soft_cldice,
)
from skoots.validate.utils import imread


def main() -> None:
    """
    Runs validation stats on two isntance masks. GT was manually calculated, pred is predicted by an algorithm

    :param gt:
    :param pred:
    :return:
    """
    parser = argparse.ArgumentParser(description="SKOOTS Training Parameters")
    parser.add_argument(
        "--ground_truth", type=str, help="Path to ground truth instance mask"
    )
    parser.add_argument(
        "--predicted", type=str, help="Path to ground truth instance mask"
    )
    parser.add_argument(
        "--log",
        type=int,
        default=3,
        help="Log Level: 0-Debug, 1-Info, 2-Warning, 3-Error, 4-Critical",
    )
    args = parser.parse_args()

    _log_map = [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]
    logging.basicConfig(
        level=_log_map[args.log],
        format="[%(asctime)s] bism-eval [%(levelname)s]: %(message)s",
    )

    gt_path, pred_path = args.ground_truth, args.predicted

    # Load model
    if os.path.exists(gt_path) and os.path.exists(pred_path):
        file_without_extension = os.path.splitext(pred_path)[0]
        gt, pred = imread(gt_path), imread(pred_path)

        gt = gt[:, 50:-50, 50:-50, 5:-5]
        pred = pred[:, 50:-50, 50:-50, 5:-5]

        logging.debug(f"Ground Truth Shape: {gt.shape}, Predicted Shape: {pred.shape}")
    else:
        raise RuntimeError(f"{os.path.exists(gt_path)=}, {os.path.exists(pred_path)=}")

    # sent to device for speed
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gt = gt.to(device)
    pred = pred.to(device)

    print("Calculating Segmentation Errors...")
    over_segmentation_rate, under_segmentation_rate = get_segmentation_errors(gt, pred)

    print("Calculating Instance Intersection over Union...")
    iou = mask_iou(gt, pred)
    dice = mask_dice(gt, pred)
    cldice = mask_soft_cldice(gt, pred)

    print("Calculating Accuracy Statistics...")
    tfp = [accuracies_from_iou(iou, thr / 100) for thr in range(100)]
    precision = [(tp / (tp + fp)) for (tp, fp, fn) in tfp]
    recall = [(tp / (tp + fn)) for (tp, fp, fn) in tfp]
    f1 = [f1_score(*a) for a in tfp]

    # Precision
    _x = torch.arange(0, 100).numpy()
    fig = plt.figure()
    plt.plot(_x, precision, "k-")
    plt.title("Precision")
    plt.xlabel("Threshold (%)")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(f"{file_without_extension}_precision.png", dpi=300)
    plt.close()

    # Recall
    _x = torch.arange(0, 100).numpy()
    fig = plt.figure()
    plt.plot(_x, recall, "k-")
    plt.title("Precision")
    plt.xlabel("Threshold (%)")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(f"{file_without_extension}_recall.png", dpi=300)
    plt.close()

    # F1
    _x = torch.arange(0, 100).numpy()
    fig = plt.figure()
    plt.plot(_x, f1, "k-")
    plt.title("F1 Score")
    plt.xlabel("Threshold (%)")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(f"{file_without_extension}_f1.png", dpi=300)
    plt.close()

    print("Writing File...")
    with open(f"{file_without_extension}_accuracy_stats.csv", "w") as file:
        file.write(f"Ground Truth File: {gt_path}\n")
        file.write(f"Predicted File: {pred_path}\n")
        file.write(f"Over Segmentation Rate: {over_segmentation_rate}\n")
        file.write(f"Under Segmentation Rate: {under_segmentation_rate}\n")
        file.write(
            f"thr,true_positive,false_positive,false_negative,precision,recall,f1\n"
        )
        i = -1
        for (tp, fp, fn), _precision, _recall, _f1 in zip(tfp, precision, recall, f1):
            i += 1
            file.write(f"{i / 100},{tp},{fp},{fn},{_precision},{_recall},{_f1}\n")

    print(f"File Written: {file_without_extension}_accuracy_stats.csv")

    with open(f"{file_without_extension}_intersection_over_union.csv", "w") as file:
        file.write(f"Ground Truth File: {gt_path}\n")
        file.write(f"Predicted File: {pred_path}\n")
        file.write(f"Average IOU: {iou.max(1)[0].mean().item()}\n")
        file.write(f"Average Dice: {dice.max(1)[0].mean().item()}\n")
        file.write(f"Average clDice: {cldice.max(1)[0].mean().item()}\n")
        file.write(f"gt_label,best_iou,best_dice,best_cldice\n")

        for i, u in enumerate(gt.unique()):
            if u == 0:
                continue
            file.write(
                f"{u},{iou[i - 1, :].max().item()},{dice[i - 1, :].max().item()},{cldice[i - 1, :].max().item()}\n"
            )

    print(f"File Written: {file_without_extension}_intersection_over_union.csv")
