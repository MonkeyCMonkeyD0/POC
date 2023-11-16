import torch
from torch import nn



class TverskyIndex(nn.Module):
    r"""Tversky Index

    The Tversky Index (TI) is a asymmetric similarity measure that is a 
    generalisation of the dice coefficient and the Jaccard index.

    .. math::
        TI = \frac{TP}{TP + \alpha FP +  \beta FN}
    """

    def __init__(self, alpha=.3, beta=.7, smooth=1e-7):
        super(TverskyIndex, self).__init__()
        self.register_buffer("smooth", torch.tensor(smooth, dtype=torch.float))

        self.alpha = torch.tensor(alpha, dtype=torch.float)
        self.beta = torch.tensor(beta, dtype=torch.float)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        TP = torch.sum(preds[:,1] * targets[:,1], dtype=torch.float)
        FP = torch.sum(preds[:,1] * targets[:,0], dtype=torch.float)
        FN = torch.sum(preds[:,0] * targets[:,1], dtype=torch.float)

        return TP / (TP + (self.alpha * FP) + (self.beta * FN) + self.smooth)
