# diffusion/diffusion.py

import torch
import numpy as np
from PIL import Image
import os

from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDIMScheduler, LCMScheduler

from .config import (
    SD_MODEL_ID,
    CONTROLNET_MODEL_ID,
    IP_ADAPTER_MODEL_ID,
    DEVICE,
    STYLES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    USE_PROJECTION_LAYER,
    USE_LCM,
    USE_IP_ADAPTER,
)
from .conditioning import (
    TextConditioner,
    ReferenceConditioner,
    ControlNetPreprocessor,
)


class StyleTransferDiffusion:
    """
    Unified pipeline for style transfer with perception-aware conditioning.
    Supports three modes:
    - Abstract: Text-based generation with perception conditioning
    - Realistic/Sci-Fi: Reference image + ControlNet + perception modulation
    """

    def __init__(self, perception_dim: int):
        """
        Args:
            perception_dim: Dimension of perception embeddings (without audio)
        """
        print(f"[StyleTransferDiffusion] Initializing with perception_dim={perception_dim}")
        
        self.perception_dim = perception_dim
        self.dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        
        # Initialize base Stable Diffusion pipeline
        print(f"[StyleTransferDiffusion] Loading base SD model: {SD_MODEL_ID}")
        self.base_pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=self.dtype,
        ).to(DEVICE)
        self.base_pipe.safety_checker = None
        
        # Load LCM-LoRA for fast generation
        if USE_LCM:
            print("[StyleTransferDiffusion] Loading LCM-LoRA for fast generation...")
            self.base_pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
            self.base_pipe.fuse_lora()
            self.base_pipe.scheduler = LCMScheduler.from_config(self.base_pipe.scheduler.config)

        # Load IP-Adapter for reference image conditioning
        if USE_IP_ADAPTER:
            print("[StyleTransferDiffusion] Loading IP-Adapter...")
            from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
            
            # Load CLIP vision encoder separately
            print("  - Loading CLIP vision encoder...")
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=self.dtype
            ).to(DEVICE)
            
            feature_extractor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            
            # Register modules so load_ip_adapter finds them
            self.base_pipe.register_modules(
                image_encoder=image_encoder,
                feature_extractor=feature_extractor
            )
            
            # Load IP-Adapter weights
            print("  - Loading IP-Adapter weights...")
            self.base_pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin"
            )
            print("  ✓ IP-Adapter loaded successfully")
        
        # Initialize ControlNet pipeline (for realistic/scifi)
        print(f"[StyleTransferDiffusion] Loading ControlNet model: {CONTROLNET_MODEL_ID}")
        self.controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_MODEL_ID,
            torch_dtype=self.dtype,
        ).to(DEVICE)
        
        # Prepare arguments for ControlNet pipeline
        controlnet_pipe_kwargs = {
            "vae": self.base_pipe.vae,
            "text_encoder": self.base_pipe.text_encoder,
            "tokenizer": self.base_pipe.tokenizer,
            "unet": self.base_pipe.unet,
            "controlnet": self.controlnet,
            "scheduler": self.base_pipe.scheduler,
            "safety_checker": None,
            "feature_extractor": None,
        }
        
        # If IP-Adapter is used, pass the shared encoder and feature extractor
        if USE_IP_ADAPTER:
            controlnet_pipe_kwargs["image_encoder"] = self.base_pipe.image_encoder
            controlnet_pipe_kwargs["feature_extractor"] = self.base_pipe.feature_extractor
            
        self.controlnet_pipe = StableDiffusionControlNetPipeline(**controlnet_pipe_kwargs).to(DEVICE)
        
        # Explicitly register modules to ensure pipeline knows about them
        if USE_IP_ADAPTER:
            self.controlnet_pipe.register_modules(
                image_encoder=self.base_pipe.image_encoder,
                feature_extractor=self.base_pipe.feature_extractor
            )
        
        if USE_LCM:
            print("[StyleTransferDiffusion] Applying LCM-LoRA to ControlNet pipeline...")
            # LoRA weights already loaded in base_pipe's unet, just update scheduler
            # If we passed the scheduler in kwargs, it might already be LCM, but let's ensure config is consistent
            if not isinstance(self.controlnet_pipe.scheduler, LCMScheduler):
                 self.controlnet_pipe.scheduler = LCMScheduler.from_config(self.controlnet_pipe.scheduler.config)
        
        # Initialize conditioning modules
        print("[StyleTransferDiffusion] Initializing conditioners...")
        self.text_conditioner = TextConditioner(
            self.base_pipe, 
            perception_dim, 
            use_projection=USE_PROJECTION_LAYER
        )
        self.reference_conditioner = ReferenceConditioner(self.base_pipe)
        self.controlnet_preprocessor = ControlNetPreprocessor(controlnet_type='canny')
        
        # Pre-load and cache reference images
        self._preload_references()
        
        print("[StyleTransferDiffusion] Initialization complete!")
    
    def _preload_references(self):
        """Pre-load reference images for realistic and sci-fi styles"""
        if not USE_IP_ADAPTER:
            print("[StyleTransferDiffusion] IP-Adapter disabled, skipping reference preload.")
            return

        print("[StyleTransferDiffusion] Pre-loading reference images...")
        for style_name, style_config in STYLES.items():
            if style_config['method'] == 'reference':
                ref_path = style_config['reference_path']
                if os.path.exists(ref_path):
                    self.reference_conditioner.cache_reference(style_name, ref_path)
                    print(f"  [OK] Cached reference: {style_name} from {ref_path}")
                    
                    # Precompute IP-Adapter embeddings if enabled
                    if USE_IP_ADAPTER:
                        self.reference_conditioner.encode_reference(style_name)
                else:
                    print(f"  [!] Reference not found: {ref_path} (will skip IP-Adapter)")
    
    @torch.no_grad()
    def generate_abstract(self, perception_vec: np.ndarray, progress_callback=None, emotion_label: str = None):
        """
        Generate abstract art using text conditioning + perception.
        
        Args:
            perception_vec: Perception embeddings from PerceptionOutput.as_vector()
            progress_callback: Optional callback(step, total_steps) for progress tracking
            emotion_label: Detected emotion string (e.g., "happy", "sad")
        
        Returns:
            PIL.Image
        """
        style_config = STYLES['abstract']
        
        print(f"[Abstract] Generating with {style_config['num_inference_steps']} steps...")
        
        # Wrap progress callback for diffusers
        def callback_wrapper(step, timestep, latents):
            if progress_callback:
                progress_callback(step, style_config['num_inference_steps'])
        
        # Rich emotion descriptors for abstract art
        emotion_prompts = {
            "angry": "chaotic red and black strokes, sharp jagged lines, intense energy, aggressive texture, storm-like, fiery, high contrast",
            "disgust": "sickly green and purple, distorted shapes, uneven texture, repelling composition, murky atmosphere, unsettling",
            "fear": "dark shadows, trembling lines, cold blue and grey, claustrophobic composition, nervous energy, mysterious, shadowy figures",
            "happy": "bright yellow and orange, flowing curves, radiant light, harmonious composition, joyful energy, warm atmosphere, sunbursts",
            "neutral": "balanced composition, soft beige and grey, minimalist, calm lines, steady rhythm, peaceful, structured geometry",
            "sad": "melancholic blue and grey, raining texture, heavy downward strokes, lonely atmosphere, quiet composition, faded colors",
            "surprise": "explosive colors, dynamic radial lines, shockwaves, vibrant contrast, sudden burst, electric energy, neon accents"
        }

        # Construct prompt with emotion
        prompt = style_config['base_prompt']
        if emotion_label:
            # Add the specific artistic description for this emotion
            rich_desc = emotion_prompts.get(emotion_label.lower(), f"{emotion_label} feeling")
            prompt += f", {rich_desc}, abstract representation of {emotion_label}"
            print(f"  + Injected emotion into prompt: '{prompt}'")

        # Use simple text prompt (projection layer disabled for now)
        guidance = 1.0 if USE_LCM else 7.5  # LCM uses minimal guidance
        image = self.base_pipe(
            prompt=prompt,
            num_inference_steps=style_config['num_inference_steps'],
            guidance_scale=guidance,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            callback=callback_wrapper,
            callback_steps=1
        ).images[0]
        
        return image
    
    @torch.no_grad()
    def generate_reference_based(self, user_image: np.ndarray, perception_vec: np.ndarray, style_name: str, progress_callback=None, emotion_label: str = None):
        """
        Generate realistic/sci-fi style using ControlNet + reference image.
        
        Args:
            user_image: User's captured image (BGR numpy array)
            perception_vec: Perception embeddings
            style_name: 'realistic' or 'scifi'
            progress_callback: Optional callback(step, total_steps) for progress tracking
            emotion_label: Detected emotion string (e.g., "happy", "sad")
        
        Returns:
            PIL.Image
        """
        style_config = STYLES[style_name]
        
        print(f"[{style_name.capitalize()}] Generating with ControlNet + reference...")
        
        # Preprocess user image for ControlNet
        control_image = self.controlnet_preprocessor.preprocess(user_image)
        
        # Get reference image for IP-Adapter (only if enabled)
        reference_image = None
        if USE_IP_ADAPTER:
            reference_image = self.reference_conditioner.get_reference(style_name)
        
        # Wrap progress callback for diffusers
        def callback_wrapper(step, timestep, latents):
            if progress_callback:
                progress_callback(step, style_config['num_inference_steps'])
        
        # Generate with ControlNet + IP-Adapter
        guidance = 1.0 if USE_LCM else 7.5  # LCM uses minimal guidance
        
        # Construct prompt with emotion
        prompt = style_config['base_prompt']
        if emotion_label:
            prompt += f", {emotion_label} facial expression, emotional {emotion_label}"
            print(f"  + Injected emotion into prompt: '{prompt}'")
        
        generation_kwargs = {
            'prompt': prompt,
            'image': control_image,
            'num_inference_steps': style_config['num_inference_steps'],
            'controlnet_conditioning_scale': style_config['controlnet_conditioning_scale'],
            'guidance_scale': guidance,
            'height': IMAGE_HEIGHT,
            'width': IMAGE_WIDTH,
            'callback': callback_wrapper,
            'callback_steps': 1
        }
        
        # Add IP-Adapter conditioning if enabled
        if USE_IP_ADAPTER:
            # Use cached embeddings or fallback to raw image
            cached_embeds = self.reference_conditioner.encode_reference(style_name)
            if cached_embeds is not None:
                # Check if it's already embeddings or a PIL image
                import torch
                if isinstance(cached_embeds, torch.Tensor):
                    # Ensure it's a list of tensors as expected by diffusers
                    generation_kwargs['ip_adapter_image_embeds'] = [cached_embeds]
                else:
                    # It's a PIL image, pass directly
                    generation_kwargs['ip_adapter_image'] = cached_embeds
                
                # Scale controls how much the reference style influences (0.0 = none, 1.0 = full)
                # Use the base pipe to set scale, as it holds the UNet with IP-Adapter
                self.controlnet_pipe.set_ip_adapter_scale(style_config.get('ipadapter_scale', 0.6))
            else:
                # This should not happen if references are properly loaded
                print(f"  ⚠ No reference for {style_name}, skipping IP-Adapter")
                
        image = self.controlnet_pipe(**generation_kwargs).images[0]
        
        return image
    
    @torch.no_grad()
    def generate(self, user_image: np.ndarray, perception_vec: np.ndarray, style_choice: str, progress_callback=None, emotion_label: str = None):
        """
        Unified generation method - routes to appropriate style generator.
        
        Args:
            user_image: User's captured image (BGR numpy array)
            perception_vec: Perception embeddings from PerceptionOutput.as_vector()
            style_choice: 'abstract', 'realistic', or 'scifi'
            progress_callback: Optional callback(step, total_steps) for progress tracking
            emotion_label: Detected emotion string (e.g., "happy", "sad")
        
        Returns:
            PIL.Image
        """
        if style_choice not in STYLES:
            raise ValueError(f"Unknown style: {style_choice}. Choose from {list(STYLES.keys())}")
        
        style_config = STYLES[style_choice]
        
        if style_config['method'] == 'text':
            return self.generate_abstract(perception_vec, progress_callback, emotion_label)
        elif style_config['method'] == 'reference':
            return self.generate_reference_based(user_image, perception_vec, style_choice, progress_callback, emotion_label)
        else:
            raise ValueError(f"Unknown method: {style_config['method']}")
    
    # Backward compatibility
    def generate_image_from_perception(self, perception_vec: np.ndarray, guidance_scale=None, num_inference_steps=None):
        """Legacy method - generates abstract by default"""
        return self.generate_abstract(perception_vec)


# Alias for backward compatibility
AmesDiffusion = StyleTransferDiffusion
