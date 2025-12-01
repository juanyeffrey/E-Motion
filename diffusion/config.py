# diffusion/config.py

import torch
import os

# Determine project root to ensure paths work regardless of where script is run
# This file is in diffusion/config.py, so project root is one level up
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Base model configuration
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"  # SD 1.5 is compatible with ControlNet
CONTROLNET_MODEL_ID = "lllyasviel/sd-controlnet-canny"
IP_ADAPTER_MODEL_ID = "h94/IP-Adapter"  # Pre-trained IP-Adapter weights
USE_LCM = True  # Use LCM-LoRA for 4-8 step fast generation
USE_IP_ADAPTER = True  # Disabled to prevent crashes

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default diffusion parameters
GUIDANCE_SCALE = 7.5
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# Style-specific configurations
STYLES = {
    'abstract': {
        'method': 'text',  # Text-based generation (no IP-Adapter)
        'base_prompt': 'vibrant abstract expressionist painting, dynamic brushstrokes, emotional energy',
        'num_inference_steps': 4,  # LCM allows 2-4 steps for text-only
        'use_controlnet': False,
        'use_ipadapter': False,  # IP-Adapter NOT used for abstract
        'style_strength': 0.8,
    },
    'realistic': {
        'method': 'reference',  # Uses ControlNet + IP-Adapter
        'reference_path': os.path.join(PROJECT_ROOT, 'final_demo', 'static', 'references', 'realistic.jpg'),
        'base_prompt': 'photorealistic portrait, natural lighting, detailed face',
        'num_inference_steps': 4,  # LCM allows 4-8 steps
        'use_controlnet': True,
        'use_ipadapter': True,  # IP-Adapter DISABLED
        'controlnet_type': 'canny',
        'controlnet_conditioning_scale': 0.8,  # Increased from 0.5 to force stronger structural adherence
        'ipadapter_scale': 0.6,
    },
    'scifi': {
        'method': 'reference',  # Uses ControlNet + IP-Adapter
        'reference_path': os.path.join(PROJECT_ROOT, 'final_demo', 'static', 'references', 'scifi.jpg'),
        'base_prompt': 'futuristic sci-fi portrait, neon lighting, cyberpunk aesthetic',
        'num_inference_steps': 4,  # LCM allows 4-8 steps
        'use_controlnet': True,
        'use_ipadapter': True,  # IP-Adapter DISABLED
        'controlnet_type': 'canny',
        'controlnet_conditioning_scale': 0.8,  # Increased from 0.5 to force stronger structural adherence
        'ipadapter_scale': 0.6,
    }
}

# Perception embedding dimension (without audio: facial landmarks + emotions)
# MediaPipe landmarks: 478 landmarks * 3 coords = 1434D, FER+ emotions: 7D, Audio: 0D (disabled)
PERCEPTION_DIM_NO_AUDIO = 1441  # 1434 + 7 (refine_landmarks=True gives 478 landmarks, FER+ has 7 classes)

# Projection layer configuration
USE_PROJECTION_LAYER = True  # Disabled to test pure prompt injection
PROJECTION_HIDDEN_DIM = 1024  # Hidden layer between input (1441) and output (768)

# CLIP embedding dimension for SD 1.5
CLIP_TEXT_DIM = 768  # SD 1.5 uses 768-dim text encoder
CLIP_IMAGE_DIM = 768

# Legacy config for backward compatibility
BASE_PROMPT = STYLES['abstract']['base_prompt']
STYLE_STRENGTH = STYLES['abstract']['style_strength']
NUM_INFERENCE_STEPS = STYLES['abstract']['num_inference_steps']
