import torch
from torch import nn
from torch.nn import functional as F



class CombinedLoss(nn.Module):
    def __init__(self, loss1, loss2, ratio=.5):
        super(CombinedLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.ratio = ratio

    def __repr__(self):
        return self.__class__.__name__, self.loss1.__class__.__name__, self.loss2.__class__.__name__, f"{self.ratio:.2}"

    def __str__(self):
        return f"{self.__class__.__name__}({str(self.loss1)}+{str(self.loss2)})"

    def forward(self, *args):
        return self.ratio * self.loss1(*args) + (1 - self.ratio) * self.loss2(*args)


class BorderedLoss(nn.Module):
    def __init__(self, border_loss, volume_loss, ratio=.5):
        super(BorderedLoss, self).__init__()
        self.border_loss = border_loss
        self.volume_loss = volume_loss
        self.ratio = ratio
        self.register_buffer("kernel", torch.Tensor([[[[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]]]]), persistent=False)

    def __repr__(self):
        return self.__class__.__name__, self.border_loss.__class__.__name__, self.volume_loss.__class__.__name__, f"{self.ratio:.2}"

    def __str__(self):
        return f"{self.__class__.__name__}({str(self.border_loss)}+{str(self.volume_loss)})"

    def create_border_mask(self, masks: torch.Tensor):
        borders = F.conv2d(masks[:,-1:], weight=self.kernel, padding='same')
        borders.clamp_(min=0., max=1.)

        return borders


    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        volume_loss_val = self.ratio * self.volume_loss(preds, targets)

        borders = self.create_border_mask(targets)
        emphasized_preds = preds * borders
        emphasized_targets = targets * borders

        border_loss_val = (1 - self.ratio) * self.border_loss(emphasized_preds, emphasized_targets)

        return volume_loss_val + border_loss_val


class PixelLoss(nn.Module):
    def __init__(self, pixel_loss, volume_loss, ratio=None):
        super(PixelLoss, self).__init__()
        self.pixel_loss = pixel_loss

    def __repr__(self):
        return self.__class__.__name__, self.pixel_loss.__class__.__name__

    def __str__(self):
        return f"{self.__class__.__name__}:{str(self.pixel_loss)}"

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        return self.pixel_loss(preds, targets)

class VolumeLoss(nn.Module):
    def __init__(self, pixel_loss, volume_loss, ratio=None):
        super(VolumeLoss, self).__init__()
        self.volume_loss = volume_loss

    def __repr__(self):
        return self.__class__.__name__, self.volume_loss.__class__.__name__

    def __str__(self):
        return f"{self.__class__.__name__}:{str(self.volume_loss)}"

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        return self.volume_loss(preds, targets)

