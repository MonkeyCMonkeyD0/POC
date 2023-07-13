import numpy as np

from skimage.segmentation import watershed

import torch
from torch import nn
from torchvision.transforms.functional import rgb_to_grayscale


class WatershedFilter(nn.Module):
    def __init__(self, background_filter: nn.Module, foreground_filter: nn.Module):
        super().__init__()
        self.background_filter = background_filter
        self.foreground_filter = foreground_filter

    @torch.inference_mode()
    def forward(self, img):
        input_device = img.device

        background = self.background_filter(img)
        foreground = self.foreground_filter(img)
        markers = background + 2 * foreground
        markers = markers.squeeze(0).cpu().numpy()

        gray_img = rgb_to_grayscale(img)
        gray_img -= gray_img.min()
        gray_img /= gray_img.max()
        numpy_img = gray_img.squeeze(0).cpu().numpy()
        result = watershed(numpy_img, markers=markers, connectivity=1, compactness=1.)
        img = torch.from_numpy(result).expand(1, -1, -1).float()
        img -= img.min()
        img /= img.max()
        return img.to(input_device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
