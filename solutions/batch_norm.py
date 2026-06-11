"""Solution 2 — Batch Normalization: ``nn.BatchNorm2d`` after each conv layer.

Batch Normalization normalizes activations across the mini-batch dimension.
It stabilizes training but can be problematic in federated settings where
each client's batch statistics differ due to non-IID data.
"""

SOLUTION_NAME = "batch_norm"
NORM_TYPE = "batch"
LOSS_TYPE = "cross_entropy"
OPTIMIZER = "adam"
WEIGHT_DECAY = 0.0
DESCRIPTION = (
    "CNN with BatchNorm2d. Normalizes per mini-batch — can degrade under "
    "non-IID federated splits because local batch statistics diverge."
)
