"""Solution 6 — Focal Loss: Down-weights easy examples, focuses on hard ones.

Focal Loss modifies the standard cross-entropy by adding a modulating factor
``(1 - p_t)^gamma`` that reduces the loss for well-classified examples and
focuses training on hard, misclassified samples.  Particularly useful for
class-imbalanced datasets like BreakHis (benign ~31% vs malignant ~69%).

Uses BatchNorm model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SOLUTION_NAME = "focal_loss"
NORM_TYPE = "batch"  # uses BatchNorm model
LOSS_TYPE = "focal"
OPTIMIZER = "adam"
WEIGHT_DECAY = 0.0
DESCRIPTION = (
    "CNN with BatchNorm + FocalLoss(α=0.25, γ=2.0). "
    "Down-weights easy examples to handle class imbalance."
)


class FocalLoss(nn.Module):
    """CrossEntropy-compatible focal loss for multi-class classification."""

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: 'none', 'mean', 'sum'.")

        if isinstance(alpha, (list, tuple)):
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = float(alpha)

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        alpha = self.alpha

        if isinstance(alpha, torch.Tensor) and alpha.ndim > 0:
            alpha = alpha.to(inputs.device)[targets]

        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()

        return focal_loss
