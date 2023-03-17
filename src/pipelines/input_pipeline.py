import torch
from torch import nn
from torchvision.transforms import Compose


class InputPipeline(nn.Module):
    def __init__(self, transformer = None, layer_transformer = None):
        super().__init__()
        if isinstance(transformer, list):
            self.transformer = transformer
        elif isinstance(transformer, nn.Module):
            self.transformer = [transformer]
        else:
            self.transformer = None

        if isinstance(layer_transformer, list):
            self.layer_transformer = nn.ModuleList(layer_transformer)
        elif isinstance(layer_transformer, nn.Module):
            self.layer_transformer = nn.ModuleList([layer_transformer])
        else:
            self.layer_transformer = None


    def forward(self, img):
        n_channel = img.size()[-3]
        if self.transformer is not None:
            img = Compose(self.transformer)(img)
        if self.layer_transformer is not None:
            for transform in self.layer_transformer:
                new_channel = transform(img[0:n_channel])
                img = torch.cat((img, new_channel), dim=-3) 
        return img

    def __repr__(self) -> str:
        return "{}({}+{})".format(
            self.__class__.__name__,
            ",".join([t.__name__ for t in self.transformer]) if self.transformer is not None else " ",
            ",".join([str(t) for t in self.layer_transformer]) if self.layer_transformer is not None else " ")
