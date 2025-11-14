# ames_perception/perception_core.py

import numpy as np
from dataclasses import dataclass

from .visual_mediapipe import MediaPipeFaceEmbedder
from .visual_ferplus import FERPlusEmotionEmbedder
from .audio_wav2vec2 import Wav2Vec2Embedder

@dataclass
class PerceptionOutput:
    timestamp: float
    visual_landmarks: np.ndarray | None
    emotion_probs: np.ndarray | None
    audio_embedding: np.ndarray | None

    def as_vector(self):
        parts = []
        for x in [self.visual_landmarks, self.emotion_probs, self.audio_embedding]:
            if x is not None:
                parts.append(x)
        if not parts:
            return None
        return np.concatenate(parts, axis=0)

class PerceptionPipeline:
    def __init__(self):
        self.face_embedder = MediaPipeFaceEmbedder()
        self.emotion_embedder = FERPlusEmotionEmbedder()
        self.audio_embedder = Wav2Vec2Embedder()

    def process(self, frame_bgr, audio_window, timestamp):
        visual_landmarks = self.face_embedder(frame_bgr)

        if visual_landmarks is not None:
            emotion_probs, _ = self.emotion_embedder(frame_bgr)
        else:
            emotion_probs = None

        audio_embedding = None
        if audio_window is not None:
            audio_embedding = self.audio_embedder(audio_window)

        return PerceptionOutput(
            timestamp=timestamp,
            visual_landmarks=visual_landmarks,
            emotion_probs=emotion_probs,
            audio_embedding=audio_embedding,
        )
