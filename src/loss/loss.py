import torch
from torch import nn
from torch.nn import functional as F


class PixelLoss(nn.Module):
    def __init__(self, pixel_loss, volume_loss, ratio=None):
        super(PixelLoss, self).__init__()
        self.pixel_loss = pixel_loss

    def get_names(self):
        return self.__class__.__name__, self.pixel_loss.__class__.__name__, ' '

    def __str__(self):
        return f"{self.__class__.__name__}:{str(self.pixel_loss)}"

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        return self.pixel_loss(preds, targets)

class VolumeLoss(nn.Module):
    def __init__(self, pixel_loss, volume_loss, ratio=None):
        super(VolumeLoss, self).__init__()
        self.volume_loss = volume_loss

    def get_names(self):
        return self.__class__.__name__, ' ', self.volume_loss.__class__.__name__

    def __str__(self):
        return f"{self.__class__.__name__}:{str(self.volume_loss)}"

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        return self.volume_loss(preds, targets)


class MultiscaleLoss(nn.Module):
    """docstring for MultiscaleLoss"""
    def __init__(self, loss_function):
        super(MultiscaleLoss, self).__init__()
        self.loss_function = loss_function

    def get_names(self):
        return self.loss_function.get_names()

    def __str__(self):
        return f"{self.__class__.__name__}:{str(self.loss_function)}"

    def forward(self, preds_tuple: (torch.Tensor), targets: torch.Tensor):
        loss = torch.tensor(0.).to(preds_tuple[0].device)
        for pred in preds_tuple:
            loss += self.loss_function(pred, targets)
        return loss
