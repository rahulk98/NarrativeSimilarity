"""Loss functions: soft-margin triplet loss and CORAL domain adaptation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLossWithSoftMargin(nn.Module):
    """
    Soft margin triplet loss using cosine distance.
    L = log(1 + exp(scale * (d(a,p) - d(a,n))))
    """

    def __init__(self, scale: float = 10.0):
        super().__init__()
        self.scale = scale

    def forward(self, anchor, positive, negative):
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negative = F.normalize(negative, p=2, dim=-1)

        d_pos = 1 - torch.sum(anchor * positive, dim=-1)
        d_neg = 1 - torch.sum(anchor * negative, dim=-1)

        losses = torch.log(1 + torch.exp(self.scale * (d_pos - d_neg)))
        return losses.mean()


class CORALLoss(nn.Module):
    """
    CORAL (Correlation Alignment) loss for domain adaptation.
    Aligns second-order statistics (covariance) between source and target embeddings.
    L = ||C_S - C_T||_F^2 / (4 * d^2)
    """

    def forward(self, source_embs, target_embs):
        source_centered = source_embs - source_embs.mean(dim=0)
        target_centered = target_embs - target_embs.mean(dim=0)

        source_cov = torch.mm(source_centered.T, source_centered) / (source_embs.size(0) - 1)
        target_cov = torch.mm(target_centered.T, target_centered) / (target_embs.size(0) - 1)

        loss = torch.norm(source_cov - target_cov, p="fro") ** 2
        return loss / (4 * source_embs.size(1) ** 2)
