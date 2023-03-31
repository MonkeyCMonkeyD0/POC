import torch
from torch import nn


class SequenceFilters(nn.Module):
    def __init__(self, *filters):
        super().__init__()
        self.filters_name = [str(f) for f in filters]
        self.filters = nn.Sequential(filters)

    @torch.inference_mode()
    def forward(self, img):
        return self.filters(img)

    def __repr__(self):
        return str((self.__class__.__name__, *self.filters_name))
        
    def __str__(self):
        return "<-".join(reversed(self.filters_name))
