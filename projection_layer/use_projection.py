# projection_layer/use_projection.py

import os
import json
from typing import Union

import numpy as np
import torch

from .model import ProjectionMLP
from .config import DEVICE


class ProjectionModel:
    """
    Convenience wrapper for applying a trained ProjectionMLP
    to new perception embeddings (e.g. from your live AMES perception layer).
    """

    def __init__(self, checkpoint_dir: str):
        meta_path = os.path.join(checkpoint_dir, "projection_meta.json")
        ckpt_path = os.path.join(checkpoint_dir, "projection_best.pt")

        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing meta file: {meta_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        input_dim = ckpt["input_dim"]
        output_dim = ckpt["output_dim"]
        hidden_dim = ckpt.get("hidden_dim", 1024)

        self.model = ProjectionMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
        ).to(DEVICE)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.input_dim = input_dim
        self.output_dim = output_dim

    @torch.no_grad()
    def project(self, perception_vec: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Map a perception embedding -> CLIP embedding (baseline MLP projection).

        perception_vec: shape (D_in,) or (B, D_in)
        returns: np.ndarray with shape (D_out,) or (B, D_out)
        """
        if isinstance(perception_vec, np.ndarray):
            x = torch.from_numpy(perception_vec.astype("float32"))
        else:
            x = perception_vec

        if x.ndim == 1:
            x = x.unsqueeze(0)

        x = x.to(DEVICE)
        y = self.model(x)  # [B, D_out]
        y = y.cpu().numpy()

        if y.shape[0] == 1:
            return y[0]
        return y
