# diffusion/ames_diffusion.py

import torch
from diffusers import StableDiffusionPipeline

from .config import (
    SD_MODEL_ID,
    DEVICE,
    BASE_PROMPT,
    GUIDANCE_SCALE,
    NUM_INFERENCE_STEPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)
from .conditioning import DiffusionConditioner


class AmesDiffusion:
    """
    Wrapper for Stable Diffusion with perception-aware conditioning.
    """

    def __init__(self, perception_dim: int):
        # Choose dtype based on device
        if DEVICE == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        self.pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=torch_dtype,
        ).to(DEVICE)

        # Optional: disable safety checker for experimentation (up to you / your policies)
        self.pipe.safety_checker = None

        self.conditioner = DiffusionConditioner(self.pipe, perception_dim)
        self.base_prompt = BASE_PROMPT

    @torch.no_grad()
    def generate_image_from_perception(self, perception_vec, guidance_scale=None, num_inference_steps=None):
        """
        perception_vec: np.ndarray[(D,)] from PerceptionOutput.as_vector()
        returns: PIL.Image.Image
        """
        if guidance_scale is None:
            guidance_scale = GUIDANCE_SCALE
        if num_inference_steps is None:
            num_inference_steps = NUM_INFERENCE_STEPS

        prompt_embeds = self.conditioner.build_conditioned_prompt_embeds(
            base_prompt=self.base_prompt,
            perception_vec=perception_vec,
        )

        # We can also construct negative_prompt_embeds if desired; here we keep it simple.
        image = self.pipe(
            prompt_embeds=prompt_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
        ).images[0]

        return image
