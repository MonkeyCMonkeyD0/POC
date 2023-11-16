import torch
from torch import nn
from torchvision.transforms import GaussianBlur, Pad
from torch.nn.functional import threshold


class LaplacianFilter(nn.Module):
    def __init__(self, threshold: float = 2., gaussian_smoothed: bool = True):
        super().__init__()
        self.smooth = gaussian_smoothed
        self.threshold = 1. - threshold / 100
        self.gaussian_blur = GaussianBlur(kernel_size=7, sigma=2)
        self.laplacian_filter = nn.Conv2d(in_channels=3, out_channels=1, groups=1, kernel_size=5, stride=1, padding='valid', bias=False)

        kernel = torch.tensor([
            [0., 0., -1., 0., 0.],
            [0., -1., -2., -1., 0.],
            [-1., -2., 16., -2., -1],
            [0., -1., -2., -1., 0.],
            [0., 0., -1., 0., 0.]])

        self.laplacian_filter.weight = nn.Parameter(kernel.expand(1, 3, -1, -1), requires_grad=False)
        self.pad = Pad(padding=2, padding_mode='edge')

    @torch.inference_mode()
    def forward(self, img):
        if self.smooth:
            img = self.gaussian_blur(img)
        img = self.laplacian_filter(img)
        img = self.pad(img)

        img = threshold(img, threshold=img.quantile(self.threshold), value=0)
        img -= img.min()
        img /= img.max()
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
