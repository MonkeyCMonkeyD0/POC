from skimage.morphology import skeletonize

import torch
from torch import nn
from torchvision.transforms.functional import rgb_to_grayscale


class MedianPixelFilter(nn.Module):
    def __init__(self, binary_filter: nn.Module):
        super().__init__()
        self.filter = binary_filter

    @torch.inference_mode()
    def forward(self, img):
        input_device = img.device
        bin_mask = self.filter(img).bool()
        gray_img = rgb_to_grayscale(img)
        median_pixel = gray_img[bin_mask].median()
        res = torch.where(gray_img == median_pixel, 1, 0)
        return res.to(input_device)

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__
