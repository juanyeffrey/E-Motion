# ames_perception/visual_mediapipe.py

import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

class MediaPipeFaceEmbedder:
    def __init__(self, max_faces=1):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __call__(self, frame_bgr):
        """
        frame_bgr: np.ndarray[H, W, 3] from OpenCV
        returns: np.ndarray[(L*3,)] or None if no face
        """
        # Convert BGR → RGB
        frame_rgb = frame_bgr[:, :, ::-1]
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        # Use first face only for now
        landmarks = results.multi_face_landmarks[0].landmark
        arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        arr = arr.flatten()  # shape (L*3,)
        return arr
