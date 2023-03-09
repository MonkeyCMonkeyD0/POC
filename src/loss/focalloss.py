from typing import Optional

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F



class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    y_hat is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - y_hat: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 weight: Optional[Tensor] = None,
                 gamma: float = 2.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        assert reduction in ['mean', 'sum', 'none']

        super().__init__()
        self.alpha = weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        # self.nll_loss = nn.NLLLoss(
        #     weight=alpha, reduction='none', ignore_index=ignore_index)

        self.ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        return self.__class__.__name__

    # def __repr__(self):
    #     arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
    #     arg_vals = [self.__dict__[k] for k in arg_keys]
    #     arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
    #     arg_str = ', '.join(arg_strs)
    #     return f'{type(self).__name__}({arg_str})'

    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.ce_loss)
        ce = self.ce_loss(y_hat, y)

        # compute pt = {p if y=1 or 1-p otherwise
        p = F.softmax(y_hat, dim=1)
        pt = torch.sum(y * p, dim=1)
        # pt = torch.where(y[:,1].round().bool(), p[:,1], p[:,0])

        # compute focal term: (1 - pt)^gamma
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
