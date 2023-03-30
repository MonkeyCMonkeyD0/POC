from skimage.filters import sato

import torch
from torch import nn
from torchvision.transforms.functional import rgb_to_grayscale


class SatoFilter(nn.Module):
    def __init__(self, black_ridges: bool = True):
        super().__init__()
        self.black_ridges = black_ridges

    @torch.inference_mode()
    def forward(self, img):
        input_device = img.device
        gray_img = rgb_to_grayscale(img).squeeze(0)
        numpy_img = gray_img.cpu().numpy()
        result = sato(numpy_img, black_ridges=self.black_ridges)
        img = torch.from_numpy(result).expand(1, -1, -1)
        img -= img.min()
        img /= img.max()
        return img.to(input_device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
