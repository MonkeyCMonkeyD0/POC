import torch
from torch import nn


class InputPipeline(nn.Module):
    def __init__(self, transformer: nn.Module=None, layer_transformer: nn.Module=None):
        super().__init__()
        self.transformer = transformer
        self.layer_transformer = layer_transformer

    def forward(self, img):
        if self.transformer is not None:
            img = self.transformer(img)
        if self.layer_transformer is not None:
            new_chanel = self.layer_transformer(img)
            img = torch.cat((img, new_chanel), dim=-3) 
        return img

    def __repr__(self) -> str:
        # internal_mod = f"({self.transformer.__class__.__name__ if self.transformer is not None else ""} + {self.layer_transformer.__class__.__name__ if self.layer_transformer is not None else ""})"
        return f"{self.__class__.__name__}" # + internal_mod
