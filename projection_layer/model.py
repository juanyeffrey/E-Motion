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
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize with small weights for stability
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.5)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x
