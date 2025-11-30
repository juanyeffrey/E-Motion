# ames_perception/visual_ferplus.py

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
import cv2

# Constants
FER_MODEL_NAME = "trpakov/vit-face-expression"
FER_PROCESSOR_NAME = "microsoft/resnet-50"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FERPlusEmotionEmbedder:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(FER_PROCESSOR_NAME)
        self.model = AutoModelForImageClassification.from_pretrained(FER_MODEL_NAME)
        self.model.to(DEVICE)
        self.model.eval()
        self.labels = list(self.model.config.id2label.values())

    def __call__(self, frame_bgr):
        """
        frame_bgr: np.ndarray[H, W, 3] (BGR)
        returns: (probs: (E,), logits: (E,))
        """
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=frame_rgb, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [1, E]
            probs = torch.softmax(logits, dim=-1)

        return probs.cpu().numpy().squeeze(0), logits.cpu().numpy().squeeze(0)
