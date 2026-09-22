"""Screen capture backends used by the window controller."""

from capture.factory import create_capture_backend, normalize_backend_name

__all__ = ["create_capture_backend", "normalize_backend_name"]
