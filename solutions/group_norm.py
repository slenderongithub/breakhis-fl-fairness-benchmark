"""Solution 3 — Group Normalization: ``nn.GroupNorm`` after each conv layer.

Group Normalization divides channels into groups and normalizes within each
group independently. Unlike BatchNorm, it does **not** depend on batch
statistics, making it naturally suited for federated learning with non-IID
data and small local batch sizes.
"""

SOLUTION_NAME = "group_norm"
NORM_TYPE = "group"
LOSS_TYPE = "cross_entropy"
OPTIMIZER = "adam"
WEIGHT_DECAY = 0.0
DESCRIPTION = (
    "CNN with GroupNorm (8 groups). Batch-independent normalization — "
    "expected to perform well under non-IID federated splits."
)
