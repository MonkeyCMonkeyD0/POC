import numpy as np
from skimage.morphology import binary_opening

import torch
from torch import nn
from torchvision.transforms.functional import rgb_to_grayscale, gaussian_blur


class CrackBinaryFilter(nn.Module):
    def __init__(self, top_percent: float = 1.3):
        super().__init__()
        self.top_percent = top_percent

    @torch.inference_mode()
    def forward(self, img):
        input_device = img.device
        gray_img = gaussian_blur(rgb_to_grayscale(img), kernel_size=3)[0]
        threshold = gray_img.quantile((100. - self.top_percent) / 100.)
        res = (gray_img >= threshold).int().cpu().numpy()
        res = binary_opening(res, footprint=np.ones((5, 5)))
        res = torch.from_numpy(res).expand(1, -1, -1).int()
        return res.to(input_device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
