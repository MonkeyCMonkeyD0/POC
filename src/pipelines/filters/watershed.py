import numpy as np

from skimage.filters import frangi
from skimage.segmentation import watershed
from skimage.morphology import skeletonize

import torch
from torch import nn
from torchvision.transforms.functional import rgb_to_grayscale


class WatershedFilter(nn.Module):
    def __init__(self, processing_filter: nn.Module):
        super().__init__()
        self.filter = processing_filter

    @torch.inference_mode()
    def forward(self, img):
        input_device = img.device

        filter_res = self.filter(img).squeeze(0)
        markers = torch.zeros_like(filter_res)
        markers[filter_res >= filter_res.quantile(.9925)] = 2
        markers[filter_res <= filter_res.quantile(.9825)] = 1
        markers = markers.cpu().numpy()

        gray_img = rgb_to_grayscale(img).squeeze(0)
        numpy_img = gray_img.cpu().numpy()
        result = watershed(numpy_img, markers=markers)
        img = torch.from_numpy(result).expand(1, -1, -1).float()
        img -= img.min()
        img /= img.max()
        return img.to(input_device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
