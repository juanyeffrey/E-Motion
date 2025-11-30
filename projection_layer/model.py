# projection_layer/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionMLP(nn.Module):
    """
    Projects perception embeddings (facial landmarks + emotions) into CLIP text embedding space.
    Matches the architecture used in diffusion/conditioning.py.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Initialize with small weights for stability
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)
        return x
