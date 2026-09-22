"""Scrcpy frame source. The window controller supplies the already-decoded frame."""

from capture.base import CaptureBackend


class ScrcpyCapture(CaptureBackend):
    name = "scrcpy"

    def __init__(self, source=None):
        self._source = source
        self._frame = None
        self._timestamp = 0.0
        self._started = False

    def start(self):
        self._started = True

    def push(self, frame, timestamp):
        self._frame = frame
        try:
            self._timestamp = float(timestamp)
        except (TypeError, ValueError):
            self._timestamp = 0.0

    def grab(self):
        if self._source is not None:
            try:
                frame, timestamp = self._source()
            except Exception:
                return None, 0.0
            try:
                timestamp = float(timestamp or 0.0)
            except (TypeError, ValueError):
                timestamp = 0.0
            return frame, timestamp
        return self._frame, self._timestamp

    def close(self):
        self._started = False
        self._frame = None
        self._timestamp = 0.0
