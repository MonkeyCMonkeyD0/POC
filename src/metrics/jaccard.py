import torch
from torch import nn


class JaccardIndex(nn.Module):
    r"""The Jaccard index, also known as the Jaccard similarity coefficient or Intersection Over Union

    .. math::
        J(A,B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}.
    """
    def __init__(self, mesure_background=False, smooth=1e-7):
        super(JaccardIndex, self).__init__()
        self.mesure_background = mesure_background
        self.register_buffer("smooth", torch.tensor(smooth, dtype=torch.float))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        if not self.mesure_background:
            preds = preds[:,1:]
            targets = targets[:,1:]

        intersection = torch.sum(targets * preds, dim=(0, 2, 3), dtype=torch.float)
        union = torch.sum(targets, dim=(0, 2, 3), dtype=torch.float) + torch.sum(preds, dim=(0, 2, 3), dtype=torch.float) - intersection

        return (intersection / (union + self.smooth)).mean()


class WeightJaccardIndex(nn.Module):
    r"""The Jaccard index, also known as the Jaccard similarity coefficient or Intersection Over Union

    .. math::
        J(A,B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}.
    """
    def __init__(self, smooth=1e-7):
        super(WeightJaccardIndex, self).__init__()
        self.register_buffer("smooth", torch.tensor(smooth, dtype=torch.float))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        intersection = torch.sum(targets * preds, dim=(0, 2, 3), dtype=torch.float)
        union = torch.sum(targets, dim=(0, 2, 3), dtype=torch.float) + torch.sum(preds, dim=(0, 2, 3), dtype=torch.float) - intersection

        prop_class = targets.mean(dim=(0, 2, 3))

        return torch.sum(prop_class * intersection / (union + self.smooth))


class PowerJaccardIndex(nn.Module):
    r"""The Power Jaccard index, from https://hal.science/hal-03139997/file/On_Power_losses_for_semantic_segmentation.pdf

    .. math::
        J^p(A,B) = \frac{|A \cap B|}{|A|^p + |B|^p - |A \cap B|}.
    """
    def __init__(self, p=2, mesure_background=False, smooth=1e-7):
        super(JaccardIndex, self).__init__()
        self.mesure_background = mesure_background
        assert (1 <= p <= 2), "p should be between 1 and 2."
        self.register_buffer("p", torch.tensor(p, dtype=torch.float))
        self.register_buffer("smooth", torch.tensor(smooth, dtype=torch.float))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        if not self.mesure_background:
            preds = preds[:,1:]
            targets = targets[:,1:]

        intersection = torch.sum(targets * preds, dim=(0, 2, 3), dtype=torch.float)
        union = torch.sum(targets, dim=(0, 2, 3), dtype=torch.float)**self.p + torch.sum(preds, dim=(0, 2, 3), dtype=torch.float)**self.p - intersection

        return (intersection / (union + self.smooth)).mean()
