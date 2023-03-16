### Originated from https://github.com/chaddy1004/sobel-operator-pytorch/blob/master/model.py

import torch
from torch import nn
from torchvision.transforms import Pad


class SobelFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter_x = nn.Conv2d(in_channels=3, out_channels=1, groups=1, kernel_size=3, stride=1, padding='valid', bias=False)
        self.filter_y = nn.Conv2d(in_channels=3, out_channels=1, groups=1, kernel_size=3, stride=1, padding='valid', bias=False)

        kernel_x = torch.tensor([
            [1., 0., -1.],
            [2., 0., -2.],
            [1., 0., -1.]])
        kernel_y = torch.tensor([
            [1., 2., 1.],
            [0., 0., 0.],
            [-1., -2., -1.]])
        self.filter_x.weight = nn.Parameter(kernel_x.expand(1, 3, -1, -1), requires_grad=False)
        self.filter_y.weight = nn.Parameter(kernel_y.expand(1, 3, -1, -1), requires_grad=False)
        self.pad = Pad(padding=1, padding_mode='edge')

    def forward(self, img):
        Gx = self.filter_x(img)
        Gy = self.filter_y(img)
        Gx = self.pad(Gx)
        Gy = self.pad(Gy)
        Gx = torch.square(Gx)
        Gy = torch.square(Gy)
        return torch.sqrt(Gx + Gy)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
