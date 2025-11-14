# diffusion/config.py

import torch

# You can switch to SDXL or any other model later
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Base prompt used as semantic “anchor”
BASE_PROMPT = "an abstract generative painting"

# Device selection shared with perception
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# How strongly the perception-derived conditioning should influence text embedding
STYLE_STRENGTH = 0.8  # you can tweak this

# Default diffusion parameters
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 30
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
