from skimage.morphology import skeletonize

import torch
from torch import nn


class SkeletonFilter(nn.Module):
    def __init__(self, binary_filter: nn.Module):
        super().__init__()
        self.binary_filter = binary_filter

    @torch.inference_mode()
    def forward(self, img):
        input_device = img.device
        numpy_img = self.binary_filter(img).squeeze(0).cpu().numpy()
        skeleton = skeletonize(numpy_img)
        res = torch.from_numpy(skeleton).expand(1, -1, -1).int()
        return res.to(input_device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
