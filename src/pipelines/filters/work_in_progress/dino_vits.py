import torch
from torch import nn
from torch.nn.functional import interpolate
from torchvision.models import vision_transformer


class DINOFilter(nn.Module):
    def __init__(self, map_channel: int = 1):
        super().__init__()
        self.map_channel = map_channel
        self.net = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        self.net.eval()

    @torch.inference_mode()
    def forward(self, img):
        img_size = img.shape[-2:]
        if img.dim() < 4:
            img = img.unsqueeze(0)
        attentions = self.net.get_last_selfattention(img)
        attentions = attentions[0, self.map_channel, 0, 1:].reshape(1, 1, img_size[0] // 8, img_size[1] // 8) # keeping only one channel from feature map
        attentions = interpolate(attentions, scale_factor=8, mode="nearest")  # Upscaling to use as a channel
        attentions -= attentions.min()
        attentions /= attentions.max()
        return attentions.squeeze(0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
