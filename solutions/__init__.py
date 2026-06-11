"""Solutions package — normalization techniques, loss functions, and regularization.

Each solution module defines:
    SOLUTION_NAME : str   — unique identifier
    NORM_TYPE     : str   — which normalization the CNN uses
    LOSS_TYPE     : str   — 'cross_entropy' or 'focal'
    OPTIMIZER     : str   — 'adam' or 'sgd'
    WEIGHT_DECAY  : float — L2 regularization strength
    DESCRIPTION   : str   — human-readable summary
"""

from . import baseline
from . import batch_norm
from . import group_norm
from . import layer_norm
from . import instance_norm
from . import focal_loss
from . import regularization

from .focal_loss import FocalLoss

# Registry: solution_name → module
SOLUTION_REGISTRY = {
    baseline.SOLUTION_NAME: baseline,
    batch_norm.SOLUTION_NAME: batch_norm,
    group_norm.SOLUTION_NAME: group_norm,
    layer_norm.SOLUTION_NAME: layer_norm,
    instance_norm.SOLUTION_NAME: instance_norm,
    focal_loss.SOLUTION_NAME: focal_loss,
    regularization.SOLUTION_NAME: regularization,
}

__all__ = [
    "SOLUTION_REGISTRY",
    "FocalLoss",
    "baseline",
    "batch_norm",
    "group_norm",
    "layer_norm",
    "instance_norm",
    "focal_loss",
    "regularization",
]
