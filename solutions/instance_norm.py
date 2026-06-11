"""Solution 5 — Instance Normalization: ``nn.InstanceNorm2d`` after each conv.

Instance Normalization normalizes each feature channel of each sample
independently (across H×W only).  Originally designed for style transfer,
it can help when input appearance varies significantly — relevant for
histopathology images with staining variation.
"""

SOLUTION_NAME = "instance_norm"
NORM_TYPE = "instance"
LOSS_TYPE = "cross_entropy"
OPTIMIZER = "adam"
WEIGHT_DECAY = 0.0
DESCRIPTION = (
    "CNN with InstanceNorm2d. Per-channel per-sample normalization — "
    "good for staining variation in histopathology."
)
