import torch
from torch import nn
from torchvision.transforms.functional import rgb_to_grayscale


class BinaryFilter(nn.Module):
    def __init__(self, top_percent: float = 0.25):
        super().__init__()
        self.top_percent = top_percent

    @torch.inference_mode()
    def forward(self, img):
        gray_img = rgb_to_grayscale(img)
        threshold = gray_img.quantile((100 - self.top_percent) / 100)
        res = (gray_img >= threshold).int()
        return res

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
