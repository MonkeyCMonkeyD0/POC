import torch

from metrics import JaccardIndex


class JaccardLoss(JaccardIndex):
    r"""The Jaccard loss

    Pytorch loss function based on the Jaccard Index.
    The Jaccard index, also known as the Jaccard similarity coefficient or Intersection Over Union

    .. math::
        JL(A,B) = 1 - \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}.

    The loss functions used the value of the probabilities rather than amount of member in sets as this calculation
    would not be derivable

    Inspired by https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
    """

    def __init__(self, smooth=1e-7):
        super(JaccardLoss, self).__init__(smooth=smooth, mesure_background=True)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        jaccard_index = super().forward(preds, targets).requires_grad_(True)
        return 1. - jaccard_index

    def __repr__(self):
        return self.__class__.__name__
