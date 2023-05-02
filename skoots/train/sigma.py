from typing import Dict, List

import torch
from yacs.config import CfgNode


# DOCUMENTED


class Sigma:
    """an adjustable sigma parameter for training."""

    def __init__(
        self,
        adjustments: List[Dict[str, float]],
        initial_sigma: List[float] = [0.1, 0.1, 0.8],
        device="cpu",
    ):
        """
        Creates an object which inputs an epoch, and returns a torch.tensor of [sigma_x, sigma_y, sigma_z]

        Sigma should reflect the Error of embedding vectors at each spatial dim.

        :param adjustments: lost of adjustments and when to apply them
        :param initial_sigma: initial values of sigma at epoch=0
        :param device: device to load sigma on ('cpu' or 'cuda')
        """
        self.adjutments = adjustments
        self.device = device

        self.initial_sigma = (
            torch.tensor(initial_sigma, device=device)
            if isinstance(initial_sigma, list)
            else initial_sigma
        )

        values = [1]  # Initial values set so sigma isnt zero at 3 zero
        epochs = [-1]
        for d in adjustments:
            values.append(d["multiplier"])
            epochs.append(d["epoch"])

        self.values = torch.tensor(values, device=self.device)
        self.epochs = torch.tensor(epochs, device=self.device)

    def __call__(self, e: int):
        """
        Returns sigma at epoch "e"
        :param e: epoch
        :return: sigma [x,y,z]
        """
        multiplier = self.values[self.epochs < e].prod()

        return self.initial_sigma * multiplier


def init_sigma(cfg: CfgNode, device: torch.device) -> Sigma:
    start = torch.tensor(cfg.TRAIN.INITIAL_SIGMA, device=device)
    multipliers = [{"multiplier": a, "epoch": b} for a, b in cfg.TRAIN.SIGMA_DECAY]
    return Sigma(initial_sigma=start, adjustments=multipliers)


if __name__ == "__main__":
    from skoots.config import get_cfg_defaults

    cfg = get_cfg_defaults()

    _ = init_sigma(cfg, "cpu")
