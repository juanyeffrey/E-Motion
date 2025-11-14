# projection_layer/dataset.py

import numpy as np
import torch
from torch.utils.data import Dataset


class ProjectionDataset(Dataset):
    """
    Dataset for perception -> CLIP projection.

    Expects .npz with:
      - perception_embeddings: shape (N, D_in)
      - clip_embeddings:       shape (N, D_out)
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.perception = data["perception_embeddings"].astype("float32")
        self.clip = data["clip_embeddings"].astype("float32")

        assert (
            self.perception.shape[0] == self.clip.shape[0]
        ), "perception and clip embeddings must have same N"

        self.n = self.perception.shape[0]
        self.in_dim = self.perception.shape[1]
        self.out_dim = self.clip.shape[1]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.from_numpy(self.perception[idx])
        y = torch.from_numpy(self.clip[idx])
        return x, y
