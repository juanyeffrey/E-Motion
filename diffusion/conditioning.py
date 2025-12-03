# diffusion/conditioning.py

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
import os

from .config import DEVICE, CLIP_TEXT_DIM, CLIP_IMAGE_DIM, PROJECTION_HIDDEN_DIM
from projection_layer.model import ProjectionMLP as PerceptionProjector

# class PerceptionProjector(nn.Module):
#     """
#     Projects perception embeddings (facial landmarks + emotions) into CLIP text embedding space.
#     Can be used untrained (random projection) or trained for better semantic alignment.
#     """
#
#     def __init__(self, in_dim: int, clip_dim: int = CLIP_TEXT_DIM, hidden_dim: int = PROJECTION_HIDDEN_DIM):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, clip_dim),
#         )
#         
#         # Initialize with small weights for stability
#         for module in self.net.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight, gain=0.5)
#                 nn.init.zeros_(module.bias)
#
#     def forward(self, perception_vec: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             perception_vec: [batch, in_dim] or [in_dim]
#         Returns:
#             [batch, clip_dim] or [1, clip_dim]
#         """
#         if perception_vec.ndim == 1:
#             perception_vec = perception_vec.unsqueeze(0)
#         return self.net(perception_vec)


class TextConditioner:
    """
    Handles text-based conditioning for abstract style.
    Combines base text prompt with perception embeddings.
    """
    
    def __init__(self, pipe, perception_dim: int, use_projection: bool = True):
        self.pipe = pipe
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.use_projection = use_projection
        
        if use_projection:
            self.projector = PerceptionProjector(perception_dim, CLIP_TEXT_DIM)
            self.projector.to(DEVICE)
            
            # Try to load trained weights
            checkpoint_path = os.path.join("projection_layer", "checkpoints", "projection_best.pt")
            if not os.path.exists(checkpoint_path):
                # Fallback to old name
                checkpoint_path = os.path.join("projection_layer", "checkpoints", "best_model.pth")
            
            if os.path.exists(checkpoint_path):
                print(f"[TextConditioner] Loading trained projection model from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
                    
                self.projector.load_state_dict(state_dict)
            else:
                print("[TextConditioner] No trained projection model found, using random initialization.")
    
    @torch.no_grad()
    def encode_text(self, prompt: str):
        """Encode text prompt to CLIP embeddings"""
        text_inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(DEVICE)
        text_embeds = self.text_encoder(input_ids).last_hidden_state
        return text_embeds  # [1, seq_len, clip_dim]
    
    @torch.no_grad()
    def condition_with_perception(self, base_prompt: str, perception_vec: np.ndarray, strength: float = 0.8):
        """
        Conditions text embeddings with perception.
        
        Args:
            base_prompt: Base text description
            perception_vec: Perception embeddings from PerceptionOutput.as_vector()
            strength: How much perception influences the conditioning (0-1)
        
        Returns:
            Conditioned prompt embeddings [1, seq_len, clip_dim]
        """
        # Get base text embeddings
        text_embeds = self.encode_text(base_prompt)  # [1, seq_len, 768]
        
        if not self.use_projection:
            return text_embeds
        
        # Project perception to CLIP space
        perception_tensor = torch.from_numpy(perception_vec.astype(np.float32)).to(DEVICE)
        with torch.no_grad():
            perception_emb = self.projector(perception_tensor)  # [1, 768]
        
        # Apply strength and ensure correct shape
        perception_emb = torch.tanh(perception_emb) * strength  # [1, 768]
        
        # Expand to match sequence length
        seq_len = text_embeds.shape[1]
        perception_emb = perception_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [1, seq_len, 768]
        
        # Add perception to text embeddings
        conditioned_embeds = text_embeds + perception_emb
        
        return conditioned_embeds


class ReferenceConditioner:
    """
    Handles reference image-based conditioning for realistic/sci-fi styles.
    Works with IP-Adapter to blend reference style with perception.
    Caches both images and their embeddings for faster generation.
    """
    
    def __init__(self, pipe):
        self.pipe = pipe
        self.reference_cache = {}  # Cache loaded images
        self.embedding_cache = {}  # Cache IP-Adapter embeddings
    
    def load_and_preprocess_image(self, image_path: str, size: tuple = (512, 512)):
        """Load and preprocess reference image"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize(size, Image.Resampling.LANCZOS)
        return image
    
    def cache_reference(self, style_name: str, image_path: str):
        """Pre-compute and cache reference image for faster generation"""
        image = self.load_and_preprocess_image(image_path)
        self.reference_cache[style_name] = image
        return image
    
    def get_reference(self, style_name: str):
        """Retrieve cached reference image"""
        return self.reference_cache.get(style_name)
    
    def encode_reference(self, style_name: str):
        """Encode reference image to IP-Adapter embeddings and cache them"""
        if style_name in self.embedding_cache:
            return self.embedding_cache[style_name]
        
        image = self.reference_cache.get(style_name)
        if image is None:
            return None
        
        # Encode image using IP-Adapter's image encoder
        if hasattr(self.pipe, 'image_encoder') and self.pipe.image_encoder is not None:
            import torch
            
            # Use the feature_extractor (CLIPImageProcessor) to preprocess
            if hasattr(self.pipe, 'feature_extractor'):
                inputs = self.pipe.feature_extractor(images=image, return_tensors="pt")
                pixel_values = inputs.pixel_values.to(
                    device=self.pipe.image_encoder.device,
                    dtype=self.pipe.image_encoder.dtype
                )
                
                # Encode to embeddings
                with torch.no_grad():
                    image_embeds = self.pipe.image_encoder(pixel_values).image_embeds
                
                # Cache for reuse
                self.embedding_cache[style_name] = image_embeds
                print(f"  ✓ Cached IP-Adapter embeddings for {style_name}")
                return image_embeds
            else:
                # If no feature_extractor, just pass the PIL image directly
                # The pipeline will handle preprocessing internally
                return image
        
        return None


class ControlNetPreprocessor:
    """
    Preprocesses user images for ControlNet conditioning.
    Extracts structural information (edges, pose, depth).
    """
    
    def __init__(self, controlnet_type: str = 'canny'):
        self.controlnet_type = controlnet_type
    
    def preprocess_canny(self, image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150):
        """
        Extract Canny edges from image.
        Lower thresholds capture more facial details (wrinkles, subtle expressions).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL Image
        edges_pil = Image.fromarray(edges_rgb)
        return edges_pil
    
    def preprocess(self, image: np.ndarray):
        """
        Main preprocessing function.
        
        Args:
            image: numpy array in BGR format (OpenCV format)
        
        Returns:
            PIL Image with control signal
        """
        if self.controlnet_type == 'canny':
            return self.preprocess_canny(image)
        else:
            raise ValueError(f"Unknown controlnet_type: {self.controlnet_type}")
