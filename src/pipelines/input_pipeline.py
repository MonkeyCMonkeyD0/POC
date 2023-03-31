import torch
from torch import nn
from torchvision.transforms import Compose


class InputPipeline(nn.Module):
    def __init__(self, transformer = None, layer_transformer = None):
        super().__init__()

        if isinstance(transformer, list):
            self.transformer = transformer
        elif transformer is None:
            self.transformer = None
        else:
            self.transformer = [transformer]

        if isinstance(layer_transformer, list):
            self.layer_transformer = nn.ModuleList(layer_transformer)
        elif isinstance(layer_transformer, nn.Module):
            self.layer_transformer = nn.ModuleList([layer_transformer])
        else:
            self.layer_transformer = None

        self._nb_channel = 3 + (len(self.layer_transformer) if self.layer_transformer is not None else 0)

    @torch.inference_mode()
    def forward(self, img):
        n_channel = img.shape[-3]
        if self.layer_transformer is not None:
            for transform in self.layer_transformer:
                new_channel = transform(img[0:n_channel])
                new_channel = (new_channel - new_channel.min()) / new_channel.max()
                img = torch.cat((img, new_channel), dim=-3) 
        if self.transformer is not None:
            img[0:n_channel] = Compose(self.transformer)(img[0:n_channel])
        return img

    @property
    def nb_channel(self):
        return self._nb_channel

    def __repr__(self):
        return str((self.__class__.__name__,
            ",".join([t.__name__ for t in self.transformer]) if self.transformer is not None else "_",
            ",".join([str(t) for t in self.layer_transformer]) if self.layer_transformer is not None else "_"))
        
    def __str__(self):
        return "{}({}+{})".format(
            self.__class__.__name__,
            ",".join([t.__name__ for t in self.transformer]) if self.transformer is not None else "_",
            ",".join([str(t) for t in self.layer_transformer]) if self.layer_transformer is not None else "_")
