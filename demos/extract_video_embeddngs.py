# demos/extract_video_embeddings.py

import argparse
import numpy as np
import subprocess
import tempfile
import os
import soundfile as sf

from ames_perception.video_io import video_file_frames
from ames_perception.perception_core import PerceptionPipeline
from ames_perception.config import SAMPLE_RATE, AUDIO_WINDOW_SECONDS

def extract_audio_to_wav(video_path, wav_path, sample_rate):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", str(sample_rate),
        wav_path,
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True, help="Output npz file")
    args = parser.parse_args()

    # 1. Extract audio to wav (temp)
    tmp_wav = tempfile.mktemp(suffix=".wav")
    extract_audio_to_wav(args.video, tmp_wav, SAMPLE_RATE)

    audio_waveform, sr = sf.read(tmp_wav)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sr}")

    # Precompute audio window for each frame timestamp
    window_size = int(SAMPLE_RATE * AUDIO_WINDOW_SECONDS)

    def get_audio_window_for_ts(ts):
        center = int(ts * SAMPLE_RATE)
        start = max(0, center - window_size // 2)
        end = min(audio_waveform.shape[0], start + window_size)
        if end - start < window_size:
            return None
        return audio_waveform[start:end].astype(np.float32)

    pipe = PerceptionPipeline()

    timestamps = []
    embeddings = []

    for ts, frame in video_file_frames(args.video):
        audio_window = get_audio_window_for_ts(ts)
        out = pipe.process(frame, audio_window, ts)
        vec = out.as_vector()
        if vec is not None:
            timestamps.append(ts)
            embeddings.append(vec)

    os.remove(tmp_wav)

    embeddings = np.stack(embeddings) if embeddings else np.zeros((0,))

    np.savez(
        args.out,
        timestamps=np.array(timestamps),
        embeddings=embeddings,
    )
    print(f"Saved {len(timestamps)} embeddings to {args.out}")

if __name__ == "__main__":
    main()
