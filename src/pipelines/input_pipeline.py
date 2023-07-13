import torch
from torch import nn
from torchvision.transforms import Compose


class InputPipeline(nn.Module):
    def __init__(self, filter = None, additional_channel = None):
        super().__init__()

        self.filter = filter

        if isinstance(additional_channel, list):
            self.additional_channel = nn.ModuleList(additional_channel)
        elif isinstance(additional_channel, nn.Module):
            self.additional_channel = nn.ModuleList([additional_channel])
        else:
            self.additional_channel = None

        self._nb_channel = 3 + (len(self.additional_channel) if self.additional_channel is not None else 0)

    @torch.inference_mode()
    def forward(self, img):
        if self.filter is not None:
            new_img = self.filter(img)
        else:
            new_img = img.clone()

        if self.additional_channel is not None:
            for transform in self.additional_channel:
                new_channel = transform(img)
                new_channel = (new_channel - new_channel.min()) / new_channel.max()
                new_img = torch.cat((new_img, new_channel), dim=-3)

        return new_img

    @property
    def nb_channel(self):
        return self._nb_channel

    def get_names(self):
        return (self.__class__.__name__,
            ",".join([t.__name__ for t in self.filter]) if self.filter is not None else " ",
            ",".join([str(t) for t in self.additional_channel]) if self.additional_channel is not None else " ")
        
    def __str__(self):
        return "{}({}+{})".format(
            self.__class__.__name__,
            ",".join([t.__name__ for t in self.filter]) if self.filter is not None else " ",
            ",".join([str(t) for t in self.additional_channel]) if self.additional_channel is not None else " ")
