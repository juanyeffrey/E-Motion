# diffusion/conditioning.py

import torch
import torch.nn as nn
import numpy as np

from .config import STYLE_STRENGTH, DEVICE


class PerceptionNaiveProjector(nn.Module):
    """
    Naive projection from perception embedding -> CLIP text embedding space.

    - in_dim: dimension of perception_vec (e.g. ~2180)
    - clip_dim: CLIP text embedding dimension (e.g. 768)
    """

    def __init__(self, in_dim: int, clip_dim: int, hidden_dim: int = 1024):
        super().__init__()
        # Simple 2-layer MLP (untrained for now)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, clip_dim),
        )

    def forward(self, perception_vec: torch.Tensor) -> torch.Tensor:
        """
        perception_vec: [batch, in_dim] or [in_dim]
        returns: [batch, clip_dim]
        """
        if perception_vec.ndim == 1:
            perception_vec = perception_vec.unsqueeze(0)
        return self.net(perception_vec)


class DiffusionConditioner:
    """
    Wraps a Stable Diffusion pipeline and lets you inject perception embeddings
    directly into the text conditioning.

    Strategy (naive):
    - Encode a base text prompt with CLIP text encoder -> text_embeds [1, seq_len, clip_dim]
    - Project perception_vec -> style_emb [1, clip_dim]
    - Add STYLE_STRENGTH * style_emb to every token embedding.
    """

    def __init__(self, pipe, perception_dim: int):
        """
        pipe: StableDiffusionPipeline
        perception_dim: dimension of perception_vec from PerceptionOutput.as_vector()
        """
        self.pipe = pipe
        # Get CLIP embedding dim from text_encoder config
        clip_dim = pipe.text_encoder.config.hidden_size
        self.projector = PerceptionNaiveProjector(perception_dim, clip_dim)
        self.projector.to(DEVICE)

    @torch.no_grad()
    def build_conditioned_prompt_embeds(self, base_prompt: str, perception_vec: np.ndarray):
        """
        base_prompt: base text description (e.g. "an abstract generative painting")
        perception_vec: np.ndarray[(D,)] from PerceptionOutput.as_vector()
        returns: prompt_embeds tensor suitable for Stable Diffusion
        """
        # 1. Get text embeddings from pipeline
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder

        text_inputs = tokenizer(
            [base_prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_inputs.input_ids.to(DEVICE)

        text_outputs = text_encoder(input_ids)
        text_embeds = text_outputs.last_hidden_state  # [1, seq_len, clip_dim]

        # 2. Convert perception vector to tensor
        perception_tensor = torch.from_numpy(perception_vec.astype(np.float32)).to(DEVICE)

        # 3. Get style embedding
        style_emb = self.projector(perception_tensor)  # [1, clip_dim]
        style_emb = torch.tanh(style_emb)  # keep values bounded
        style_emb = STYLE_STRENGTH * style_emb  # scale

        # 4. Broadcast to all tokens and add
        style_emb_tokens = style_emb.unsqueeze(1)  # [1, 1, clip_dim]
        conditioned_embeds = text_embeds + style_emb_tokens  # broadcast over seq_len

        return conditioned_embeds
