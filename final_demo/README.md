# E-Motion Final Demo - Style Transfer Web App

## Overview
Web-based application for real-time style transfer using perception embeddings and ControlNet/IP-Adapter.

## User Flow
1. **Capture**: User views webcam feed and presses SPACE to capture a photo
2. **Style Selection**: User chooses from 3 reference styles (Abstract, Realistic, Sci-Fi)
3. **Generation**: System generates a styled portrait combining user's expression with chosen style
4. **Result**: User can view, download, or start over

## Setup

### Install Dependencies
```bash
pip install flask opencv-python pillow numpy
```

### Add Reference Images
Place 3 reference images in `static/references/`:
- `abstract.jpg` - Abstract/expressive style
- `realistic.jpg` - Realistic/photographic style
- `scifi.jpg` - Sci-fi/futuristic style

### Run the Demo
```bash
cd final_demo
python app.py
```

Then open http://localhost:5000 in your browser.

## Project Structure
```
final_demo/
├── app.py                  # Flask backend
├── templates/
│   └── index.html         # Main UI
├── static/
│   ├── style.css          # Styling
│   ├── script.js          # Frontend logic
│   ├── placeholder.jpg    # Placeholder image
│   └── references/        # Reference images (add your own)
│       ├── abstract.jpg
│       ├── realistic.jpg
│       └── scifi.jpg
└── README.md
```

## TODO: Integration Points

### 1. Perception Pipeline Integration
In `app.py`, replace the TODO in `generate_styled_image()`:
```python
# Add perception processing
from perception.perception_core import PerceptionPipeline
pipeline = PerceptionPipeline()
perception_out = pipeline.process(captured_image, audio=None, ts=0)
```

### 2. ControlNet/IP-Adapter Integration
Add the actual generation pipeline:
```python
# Add diffusion pipeline with ControlNet/IP-Adapter
from diffusion.diffusion_controlnet import ControlNetIPAdapterPipeline
gen_pipeline = ControlNetIPAdapterPipeline()
generated_image = gen_pipeline.generate(
    user_image=captured_image,
    style_reference=load_reference(style_choice),
    perception_vec=perception_out.as_vector()
)
```

### 3. Reference Image Selection
Add high-quality reference images to `static/references/` folder

## Features
- ✅ Real-time webcam capture
- ✅ Spacebar trigger for photo capture
- ✅ Three-option style selection
- ✅ Side-by-side result comparison
- ✅ Download functionality
- ✅ Reset/start over capability
- ⏳ Actual image generation (placeholder currently)

## Browser Compatibility
- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Requires HTTPS or localhost

## Notes
- Currently uses placeholder generation (returns original image)
- Ready for ControlNet/IP-Adapter integration
- Perception pipeline hookup point clearly marked
- All UI/UX flow complete and functional
