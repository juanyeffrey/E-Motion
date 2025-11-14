# demos/diffusion_naive_demo.py

import time
import numpy as np
import cv2

from perception.audio_io import LiveAudioBuffer
from perception.video_io import webcam_frames
from perception.perception_core import PerceptionPipeline
from diffusion.diffusion import AmesDiffusion


def collect_perception_embedding(
    duration_seconds: float = 5.0,
    min_vectors: int = 20,
):
    """
    Run perception for a short window and return an averaged perception vector.

    - Uses the same pattern as the live streaming demo.
    - Requires at least `min_vectors` collected or raises a RuntimeError.
    - No fallback to zero vector.
    """
    audio_buf = LiveAudioBuffer()
    audio_buf.start()
    pipe = PerceptionPipeline()

    start_time = time.time()
    vectors = []
    n_frames = 0

    cam = webcam_frames()  # create generator once

    try:
        while True:
            # stop conditions
            now = time.time()
            if now - start_time > duration_seconds and len(vectors) > 0:
                break
            if len(vectors) >= min_vectors:
                break

            try:
                ts, frame = next(cam)
            except StopIteration:
                print("webcam_frames: no more frames")
                break

            n_frames += 1
            audio_window = audio_buf.get_window()
            out = pipe.process(frame, audio_window, ts)
            vec = out.as_vector()

            # Debug info
            print(
                f"[capture] t={now - start_time:4.1f}s "
                f"face={'yes' if out.visual_landmarks is not None else 'no'}, "
                f"audio={'yes' if out.audio_embedding is not None else 'no'}, "
                f"vec={'ok' if vec is not None else 'none'}"
            )

            # Show webcam
            cv2.imshow("Perception Capture (press q to abort)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if vec is not None:
                vectors.append(vec)

    finally:
        audio_buf.stop()
        cv2.destroyAllWindows()

    if not vectors:
        raise RuntimeError(
            "No perception vectors collected. "
            "Make sure your face is in view for a few seconds and don't press 'q' immediately."
        )

    stacked = np.stack(vectors, axis=0)
    avg_vec = stacked.mean(axis=0)
    print(f"Collected {len(vectors)} perception vectors over {n_frames} frames.")
    return avg_vec



def show_image(image, window_name="AMES Diffusion Output"):
    """
    Show PIL image in an OpenCV window.
    """
    # Convert PIL -> OpenCV BGR
    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    cv2.imshow(window_name, img_bgr)
    print("Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("Collecting perception embedding from webcam + mic...")
    perception_vec = collect_perception_embedding(duration_seconds=5.0)
    print(f"Perception vector dim: {perception_vec.shape[0]}")

    print("Initializing diffusion model (this may take a bit the first time)...")
    perception_dim = perception_vec.shape[0]
    ames_diff = AmesDiffusion(perception_dim=perception_dim)

    print("Generating image conditioned on your expression...")
    image = ames_diff.generate_image_from_perception(perception_vec)

    show_image(image)


if __name__ == "__main__":
    main()
