# Diffusion Module - Style Transfer Pipeline

## Overview
This module implements a modular style transfer system using Stable Diffusion with three different generation modes.

## Architecture

```
User Image → Perception Pipeline → perception_vec
                                        ↓
                            StyleTransferDiffusion
                                        ↓
                    ┌──────────────────┴──────────────────┐
                    ↓                  ↓                  ↓
               ABSTRACT           REALISTIC           SCI-FI
            (text-based)      (ControlNet+Ref)   (ControlNet+Ref)
```

## Modules

### `config.py`
Central configuration for all diffusion parameters:
- **Model IDs**: SD-Turbo, ControlNet models
- **Style Configs**: Per-style settings (prompts, steps, conditioning scales)
- **Dimensions**: Perception dim, CLIP dims
- **Toggles**: Enable/disable projection layer, IP-Adapter

### `conditioning.py`
Modular conditioning components:

#### `PerceptionProjector`
- Projects perception embeddings → CLIP text space
- Can be used untrained or trained
- Normalizes and stabilizes embeddings

#### `TextConditioner`
- Handles text-based conditioning (abstract style)
- Encodes text prompts with CLIP
- Blends perception with text embeddings

#### `ReferenceConditioner`
- Manages reference images for realistic/sci-fi
- Pre-loads and caches reference images
- Prepares images for IP-Adapter (when implemented)

#### `ControlNetPreprocessor`
- Extracts structural information from user images
- Supports: Canny edges (default), OpenPose, Depth
- Outputs control signals for ControlNet

### `diffusion.py`
Main generation pipeline:

#### `StyleTransferDiffusion`
Unified class that orchestrates all components:

**Methods:**
- `generate_abstract()` - Text + perception conditioning
- `generate_reference_based()` - ControlNet + reference style
- `generate()` - Unified interface, auto-routes to correct method

**Initialization:**
- Loads SD-Turbo base model
- Loads ControlNet model
- Pre-caches reference images
- Initializes all conditioners

## Usage

### Basic Usage
```python
from perception.perception_core import PerceptionPipeline
from diffusion.diffusion import StyleTransferDiffusion

# Initialize
perception_pipeline = PerceptionPipeline()
diffusion = StyleTransferDiffusion(perception_dim=1412)

# Process user image
perception_output = perception_pipeline.process_image(user_image)
perception_vec = perception_output.as_vector()

# Generate styled image
styled_image = diffusion.generate(
    user_image=user_image,
    perception_vec=perception_vec,
    style_choice='realistic'  # or 'abstract', 'scifi'
)
```

### Style-Specific Generation
```python
# Abstract (text-based)
abstract_image = diffusion.generate_abstract(perception_vec)

# Realistic (with ControlNet)
realistic_image = diffusion.generate_reference_based(
    user_image, 
    perception_vec, 
    'realistic'
)
```

## Customization

### Adding New Styles
Edit `config.py`:
```python
STYLES['new_style'] = {
    'method': 'reference',  # or 'text'
    'base_prompt': 'your prompt here',
    'reference_path': 'path/to/reference.jpg',
    'num_inference_steps': 4,
    'use_controlnet': True,
    'use_ipadapter': True,
    'controlnet_conditioning_scale': 0.5,
}
```

### Adjusting Perception Influence
In `config.py`, modify `style_strength`:
```python
STYLES['abstract']['style_strength'] = 0.8  # 0-1, higher = more perception influence
```

### Using Trained Projection Layer
1. Train the projection layer (see `projection_layer/`)
2. Load weights in `conditioning.py`:
```python
projector = PerceptionProjector(perception_dim, CLIP_TEXT_DIM)
projector.load_state_dict(torch.load('path/to/weights.pth'))
```

### Different ControlNet Types
In `conditioning.py`, extend `ControlNetPreprocessor`:
```python
def preprocess_openpose(self, image):
    # Implement OpenPose detection
    pass
```

Update `config.py`:
```python
STYLES['realistic']['controlnet_type'] = 'openpose'
```

## Future Enhancements

### IP-Adapter Integration
Currently simplified to text conditioning. To add full IP-Adapter:
1. Install: `pip install ip-adapter`
2. In `diffusion.py`, add IP-Adapter pipeline wrapper
3. Use `ReferenceConditioner.get_reference()` as IP-Adapter input

### Multi-Reference Blending
Blend multiple reference styles:
```python
def generate_blended(self, user_image, perception_vec, style_weights):
    # style_weights = {'abstract': 0.3, 'realistic': 0.5, 'scifi': 0.2}
    # Blend embeddings based on weights
    pass
```

### Perception-to-Prompt Decoder
Convert perception embeddings back to text:
```python
emotion = perception_output.get_dominant_emotion()
prompt = f"{base_prompt}, {emotion} expression"
```

## Troubleshooting

**Issue**: Out of memory
- Reduce `IMAGE_HEIGHT` and `IMAGE_WIDTH` in `config.py`
- Use `torch.float32` instead of `float16` on CPU

**Issue**: Slow generation
- Reduce `num_inference_steps` (SD-Turbo works well with 1-4 steps)
- Disable ControlNet for testing: `use_controlnet = False`

**Issue**: Poor style transfer
- Adjust `controlnet_conditioning_scale` (0.3-0.7 works well)
- Try different reference images
- Increase `num_inference_steps`

## Dependencies
```
torch
diffusers
transformers
opencv-python
pillow
numpy
```
