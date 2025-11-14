# demos/live_stream_embeddings.py

import cv2
import time
import numpy as np

from perception.video_io import webcam_frames
from perception.audio_io import LiveAudioBuffer
from perception.perception_core import PerceptionPipeline

def main():
    audio_buf = LiveAudioBuffer()
    audio_buf.start()
    pipe = PerceptionPipeline()

    try:
        for ts, frame in webcam_frames():
            audio_window = audio_buf.get_window()
            out = pipe.process(frame, audio_window, ts)
            vec = out.as_vector()

            # Minimal live feedback
            if vec is not None:
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"len={vec.shape[0]}, "
                    f"||vec||={np.linalg.norm(vec):.2f}, "
                    f"audio={'yes' if out.audio_embedding is not None else 'no'}, "
                    f"face={'yes' if out.visual_landmarks is not None else 'no'}"
                )

            # (Optional) show webcam window
            cv2.imshow("AMES Perception - Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        audio_buf.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
