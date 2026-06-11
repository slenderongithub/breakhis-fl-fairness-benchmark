"""Global configuration for the BreakHis federated learning benchmark."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset paths
BREAKHIS_PATH = PROJECT_ROOT / "data" / "breakhis"

# Dataset settings
NUM_CLASSES = 2  # benign vs malignant
IMG_SIZE = 128   # resize BreakHis images (smaller = faster training on CPU)
CLASS_NAMES = ["benign", "malignant"]

# BreakHis channel statistics (computed from histopathology images)
BREAKHIS_MEAN = (0.7862, 0.6261, 0.7654)
BREAKHIS_STD = (0.1065, 0.1396, 0.0910)

# Training settings
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
CLIENT_LEARNING_RATE = 0.001   # Adam-friendly LR
SERVER_LEARNING_RATE = 1.0
WEIGHT_DECAY = 0.001
NUM_ROUNDS = 20
LOCAL_EPOCHS = 2

# Federated settings
NUM_CLIENTS = 5
CLIENTS_PER_ROUND = 5
DIRICHLET_ALPHA = 0.5  # controls non-IID degree (lower = more skewed)

# Reproducibility
RANDOM_SEED = 42
