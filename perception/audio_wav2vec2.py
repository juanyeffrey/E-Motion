# ames_perception/audio_wav2vec2.py

import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from .config import WAV2VEC2_MODEL_NAME, SAMPLE_RATE, DEVICE

class Wav2Vec2Embedder:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_NAME)
        self.model = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL_NAME)
        self.model.to(DEVICE)
        self.model.eval()

    def __call__(self, audio_waveform):
        """
        audio_waveform: 1D np.ndarray, float32, sampled at SAMPLE_RATE
        returns: np.ndarray[(D,)]
        """
        if audio_waveform.ndim > 1:
            audio_waveform = np.mean(audio_waveform, axis=0)

        inputs = self.processor(
            audio_waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # [1, T', D]

        # mean-pool over time
        emb = hidden_states.mean(dim=1)  # [1, D]
        return emb.cpu().numpy().squeeze(0)
