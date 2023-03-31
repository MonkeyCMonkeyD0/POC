import torch
from torch import nn


class SumFilters(nn.Module):
    def __init__(self, *filters):
        super().__init__()
        self.filters = filters

    @torch.inference_mode()
    def forward(self, img):
        res = torch.zeros_like(img)[0:1]
        for f in self.filters:
            res += f(img)
        res -= res.min()
        res /= res.max()
        return res

    def __repr__(self):
        return str((self.__class__.__name__, *[str(f) for f in self.filters]))
        
    def __str__(self):
        return "{}({})".format(self.__class__.__name__, "+".join([str(f) for f in self.filters]))
