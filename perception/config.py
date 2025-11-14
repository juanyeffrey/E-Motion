# ames_perception/config.py
import torch
FER_MODEL_NAME = "tahayf/resnet-50_ferplus"
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000

# Length of audio window (seconds) per embedding
AUDIO_WINDOW_SECONDS = 0.8

# Target FPS for visual embedding stream
TARGET_FPS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
