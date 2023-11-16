import torch
from torch import nn
from torch.nn import functional as F


class MeanLoss(nn.Module):
    def __init__(self, loss1, loss2, ratio=.5):
        super(MeanLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.ratio = ratio

    def get_names(self):
        return self.__class__.__name__, self.loss1.__class__.__name__, self.loss2.__class__.__name__

    def __str__(self):
        return f"{self.__class__.__name__}({str(self.loss1)}+{str(self.loss2)})"

    def forward(self, *args):
        return self.ratio * self.loss1(*args) + (1 - self.ratio) * self.loss2(*args)
