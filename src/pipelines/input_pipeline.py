import torch
from torch import nn


class InputPipeline(nn.Module):
    def __init__(self, transformer = None, layer_transformer = None):
        super().__init__()
        if isinstance(transformer, list):
            self.transformer = transformer
        else:
            self.transformer = [transformer]

        self.layer_transformer = nn.ModuleList(layer_transformer)

    def forward(self, img):
        n_channel = img.size()[-3]
        if self.transformer is not None:
            for transform in self.transformer:
                img = transform(img)
        if self.layer_transformer is not None:
            for transform in self.layer_transformer:
                new_channel = transform(img[0:n_channel])
                img = torch.cat((img, new_channel), dim=-3) 
        return img

    def __repr__(self) -> str:
        return "{}({}+{})".format(
            self.__class__.__name__,
            self.transformer.__class__.__name__ if self.transformer is not None else " ",
            self.layer_transformer.__class__.__name__ if self.layer_transformer is not None else " ")
