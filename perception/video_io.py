# ames_perception/video_io.py

import cv2
import time

def webcam_frames(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts = time.time()
            yield ts, frame
    finally:
        cap.release()

def video_file_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_duration = 1.0 / fps
    ts = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield ts, frame
            ts += frame_duration
    finally:
        cap.release()
