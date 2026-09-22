"""Capture backend interface shared by scrcpy and MuMu."""

import time


class CaptureBackend:
    """A frame source. ``grab`` returns ``(frame, timestamp)`` and never needs a display in tests."""

    name = "base"

    def start(self):
        return None

    def grab(self):
        return None, 0.0

    def close(self):
        return None

    def latency(self, now=None):
        _frame, timestamp = self.grab()
        if not timestamp:
            return None
        current = time.time() if now is None else now
        try:
            age = float(current) - float(timestamp)
        except (TypeError, ValueError):
            return None
        if age < 0.0:
            return 0.0
        return age
