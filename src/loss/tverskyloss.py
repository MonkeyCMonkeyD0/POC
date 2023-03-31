import torch

from metrics import TverskyIndex


class TverskyLoss(TverskyIndex):
    r"""Tversky Loss

    .. math::
        TL = 1 - TI

    The loss functions used the value of the probabilities rather than amount of member in sets as this calculation
    would not be derivable
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-7):
        super(TverskyLoss, self).__init__(alpha=alpha, beta=beta, smooth=smooth)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        tversky_index = super().forward(preds, targets).requires_grad_(True)
        return 1. - tversky_index

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class FocalTverskyLoss(TverskyLoss):
    r"""Focal Tversky Loss

    .. math::
        FTL = TL^\gamma

    The loss functions used the value of the probabilities rather than amount of member in sets as this calculation
    would not be derivable
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=2, smooth=1e-7):
        super(FocalTverskyLoss, self).__init__(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        tversky_loss = super().forward(preds, targets).requires_grad_(True)
        return tversky_loss ** self.gamma

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

