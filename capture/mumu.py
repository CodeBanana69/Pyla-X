"""MuMu Player screen capture.

MuMu can hand the guest framebuffer to the host without scrcpy's H.264 encode,
which cuts the capture portion of input delay. Tests install a ``source``
callable and never open an emulator. With no source, ``grab`` returns no frame
so the window controller can keep the scrcpy feed.
"""

from capture.base import CaptureBackend


class MuMuScreenCapture(CaptureBackend):
    name = "mumu"

    def __init__(self, source=None, display_index=0, instance_index=0):
        self.display_index = _index(display_index)
        self.instance_index = _index(instance_index)
        self._source = source
        self._started = False

    @classmethod
    def from_config(cls, config=None, source=None):
        config = config or {}
        return cls(
            source=source,
            display_index=config.get("mumu_display_index", 0),
            instance_index=config.get("mumu_instance_index", 0),
        )

    def start(self):
        self._started = True
        if self._source is None:
            self._source = _empty_source

    def set_source(self, source):
        self._source = source

    def grab(self):
        source = self._source
        if source is None:
            return None, 0.0
        try:
            frame, timestamp = source()
        except Exception:
            return None, 0.0
        try:
            timestamp = float(timestamp or 0.0)
        except (TypeError, ValueError):
            timestamp = 0.0
        return frame, timestamp

    def close(self):
        self._started = False


def _empty_source():
    return None, 0.0


def _index(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
