# ames_perception/audio_io.py

import sounddevice as sd
import numpy as np
import queue
import threading
import time

from .config import SAMPLE_RATE, AUDIO_WINDOW_SECONDS

class LiveAudioBuffer:
    def __init__(self):
        self.q = queue.Queue()
        self.stream = None
        self.lock = threading.Lock()
        self.buffer = np.zeros(0, dtype=np.float32)
        self.window_size = int(SAMPLE_RATE * AUDIO_WINDOW_SECONDS)

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        self.q.put(indata.copy())

    def start(self):
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_window(self):
        """
        Returns latest audio window of length window_size (or None if insufficient)
        """
        with self.lock:
            while not self.q.empty():
                chunk = self.q.get_nowait().flatten()
                self.buffer = np.concatenate([self.buffer, chunk])

            if self.buffer.size < self.window_size:
                return None

            # Take last window_size samples
            window = self.buffer[-self.window_size:]
            return window.astype(np.float32)
