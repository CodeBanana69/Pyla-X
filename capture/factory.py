"""Select a capture backend by name."""

from capture.mumu import MuMuScreenCapture
from capture.scrcpy_backend import ScrcpyCapture


def normalize_backend_name(name):
    text = str(name or "scrcpy").strip().lower().replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    aliases = {
        "scrcpy": "scrcpy",
        "default": "scrcpy",
        "mumu": "mumu",
        "mumu screen capture": "mumu",
    }
    if text not in aliases:
        raise ValueError(f"Unknown capture backend: {name}")
    return aliases[text]


def create_capture_backend(name="scrcpy", **kwargs):
    """Build a capture backend. ``mumu`` / ``MuMu Screen Capture`` select MuMu."""
    kind = normalize_backend_name(name)
    if kind == "mumu":
        return MuMuScreenCapture(
            source=kwargs.get("source"),
            display_index=kwargs.get("display_index", 0),
            instance_index=kwargs.get("instance_index", 0),
        )
    return ScrcpyCapture(source=kwargs.get("source"))
