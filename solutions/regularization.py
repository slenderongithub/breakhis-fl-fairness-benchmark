"""Solution 7 — L2 Regularization + Stronger Data Augmentation.

Combines weight decay (L2 penalty) with aggressive data augmentation to
improve generalization.  The augmented transforms include heavier rotation,
color jitter, perspective warping, and Gaussian blur — simulating the
variability in histopathology slide preparation.

Uses BatchNorm model architecture.
"""

SOLUTION_NAME = "regularization"
NORM_TYPE = "batch"  # uses BatchNorm model
LOSS_TYPE = "cross_entropy"
OPTIMIZER = "adam"
WEIGHT_DECAY = 0.001  # L2 regularization
DESCRIPTION = (
    "CNN with BatchNorm + L2 weight decay (0.001) + stronger augmentation. "
    "Tests if regularization improves generalization under non-IID splits."
)
