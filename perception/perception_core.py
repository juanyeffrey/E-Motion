# ames_perception/perception_core.py

import numpy as np
from dataclasses import dataclass

from .visual_mediapipe import MediaPipeFaceEmbedder
from .visual_ferplus import FERPlusEmotionEmbedder
# from .audio_wav2vec2 import Wav2Vec2Embedder  # Audio disabled for single-image mode

@dataclass
class PerceptionOutput:
    timestamp: float
    visual_landmarks: np.ndarray | None
    emotion_probs: np.ndarray | None
    audio_embedding: np.ndarray | None  # Kept for compatibility, always None

    def as_vector(self):
        """Returns concatenated visual embeddings only (no audio)"""
        parts = []
        for x in [self.visual_landmarks, self.emotion_probs]:
            if x is not None:
                parts.append(x)
        if not parts:
            return None
        return np.concatenate(parts, axis=0)
    
    def get_dominant_emotion(self):
        """Returns the dominant emotion label"""
        if self.emotion_probs is None:
            return "neutral"
        # FER+ model (trpakov/vit-face-expression) has 7 emotion classes
        emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        return emotion_labels[self.emotion_probs.argmax()]

class PerceptionPipeline:
    def __init__(self):
        self.face_embedder = MediaPipeFaceEmbedder()
        self.emotion_embedder = FERPlusEmotionEmbedder()
        # self.audio_embedder = Wav2Vec2Embedder()  # Audio disabled for single-image mode

    def process_image(self, image_bgr, timestamp=0.0):
        """
        Process a single image (no audio, no video stream)
        
        Args:
            image_bgr: numpy array in BGR format (OpenCV format)
            timestamp: optional timestamp (default 0.0)
        
        Returns:
            PerceptionOutput with visual_landmarks, emotion_probs, and audio_embedding=None
        """
        visual_landmarks = self.face_embedder(image_bgr)

        if visual_landmarks is not None:
            emotion_probs, _ = self.emotion_embedder(image_bgr)
        else:
            emotion_probs = None

        return PerceptionOutput(
            timestamp=timestamp,
            visual_landmarks=visual_landmarks,
            emotion_probs=emotion_probs,
            audio_embedding=None,  # Always None in image-only mode
        )
    
    def process(self, frame_bgr, audio_window=None, timestamp=0.0):
        """
        Legacy method for backward compatibility - ignores audio_window
        Use process_image() instead for clarity
        """
        return self.process_image(frame_bgr, timestamp)
