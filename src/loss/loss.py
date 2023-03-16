import torch
from torch import nn
from torch.nn import functional as F



class CombinedLoss(nn.Module):
    def __init__(self, loss1, loss2, ratio=.5):
        super(CombinedLoss, self).__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.ratio = ratio
        self.__class__.__name__ = f"({self.loss1.__class__.__name__}+{self.loss2.__class__.__name__})"

    def __repr__(self):
        return self.__class__.__name__

    def forward(self, *args):
        return self.ratio * self.loss1(*args) + (1 - self.ratio) * self.loss2(*args)


class BorderedLoss(nn.Module):
    def __init__(self, border_loss, volume_loss, ratio=.5):
        super(BorderedLoss, self).__init__()
        self.border_loss = border_loss
        self.volume_loss = volume_loss
        self.ratio = ratio
        self.__class__.__name__ = f"(B:{self.border_loss.__class__.__name__}+V:{self.volume_loss.__class__.__name__})"
        self.register_buffer("kernel", torch.Tensor([[[[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]]]]), persistent=False)

    def __repr__(self):
        return self.__class__.__name__

    # def to(self, device: str):
    #     super(BorderedLoss, self).to(device)
    #     self.kernel = self.kernel.to(device)
    #     return self

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
