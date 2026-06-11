"""Solution 4 — Layer Normalization: ``nn.LayerNorm`` after each conv layer.

Layer Normalization normalizes across the full channel×H×W dimensions for
each sample independently.  Like GroupNorm it is batch-independent, but it
normalizes across all channels simultaneously which can lose per-channel
specialization.
"""

SOLUTION_NAME = "layer_norm"
NORM_TYPE = "layer"
LOSS_TYPE = "cross_entropy"
OPTIMIZER = "adam"
WEIGHT_DECAY = 0.0
DESCRIPTION = (
    "CNN with LayerNorm. Normalizes across C×H×W per sample — "
    "batch-independent but higher parameter count."
)
