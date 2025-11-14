# projection_layer/config.py

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training hyperparameters
BATCH_SIZE = 128
LR = 1e-3
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-4

# Checkpoint defaults
DEFAULT_OUT_DIR = "projection_checkpoints"

# Loss: we'll use cosine-similarity-based loss
