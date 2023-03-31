from skimage.morphology import skeletonize

import torch
from torch import nn
from torchvision.transforms.functional import rgb_to_grayscale


class SkeletonFilter(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def forward(self, img):
        input_device = img.device
        numpy_img = rgb_to_grayscale(img).squeeze(0).cpu().numpy()
        skeleton = skeletonize(numpy_img)
        res = torch.from_numpy(skeleton).expand(1, -1, -1).int()
        return res.to(input_device)

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__class__.__name__
