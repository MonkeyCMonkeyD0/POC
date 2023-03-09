import torch

from metrics import DiceIndex


class DiceLoss(DiceIndex):
    r"""Sørensen–Dice Loss

    The Dice coefficient, or Dice-Sørensen coefficient, is a common metric for pixel segmentation that can also be
    modified to act as a loss function:

    .. math::
        DSC(X, Y) = 1 - \frac{2 \left| X + Y \right|}{\left| X \right| + \left| Y \right|}

    The loss functions used the value of the probabilities rather than amount of member in sets as this calculation
    would not be derivable

    Inspired by https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """

    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__(smooth=smooth, mesure_background=True)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        dice_coefficient = super().forward(preds, targets).requires_grad_(True)
        return 1. - dice_coefficient

    def __repr__(self):
        return self.__class__.__name__
