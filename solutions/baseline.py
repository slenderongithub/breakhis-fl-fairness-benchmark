"""Solution 1 — Baseline: No normalization layers.

Uses the raw CNN without any normalization. Serves as the control
experiment to compare all other techniques against.
"""

SOLUTION_NAME = "baseline"
NORM_TYPE = "none"
LOSS_TYPE = "cross_entropy"  # standard CE with class weights
OPTIMIZER = "adam"
WEIGHT_DECAY = 0.0
DESCRIPTION = (
    "Baseline CNN with no normalization layers. "
    "Uses Adam optimizer with class-weighted CrossEntropyLoss."
)
