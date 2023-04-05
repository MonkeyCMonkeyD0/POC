import torch
from torch import nn


class DiceIndex(nn.Module):
    r"""Sørensen–Dice Score

    The Dice coefficient, or Dice-Sørensen coefficient, is a common metric for pixel segmentation that can also be
    modified to act as a loss function:

    .. math::
        DSC(X, Y) = \frac{2 \left| X + Y \right|}{\left| X \right| + \left| Y \right|}

    Inspired by https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """
    def __init__(self, mesure_background=False, smooth=1e-7):
        super(DiceIndex, self).__init__()
        self.mesure_background = mesure_background
        self.register_buffer("smooth", torch.tensor(smooth, dtype=torch.float))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        if not self.mesure_background:
            preds = preds[:,1:]
            targets = targets[:,1:]

        intersection = torch.sum(targets * preds, dim=(0, 2, 3), dtype=torch.float)
        total_pixels = torch.sum(targets, dim=(0, 2, 3), dtype=torch.float) + torch.sum(preds, dim=(0, 2, 3), dtype=torch.float)

        return (2 * intersection / (total_pixels + self.smooth)).mean()
