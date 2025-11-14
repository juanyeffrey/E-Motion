# projection_layer/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionMLP(nn.Module):
    """
    Baseline 2-layer MLP:
      input_dim -> hidden_dim -> output_dim

    We'll train it to map perception embeddings to CLIP embeddings.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Optional: initialize small to avoid exploding activations
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
