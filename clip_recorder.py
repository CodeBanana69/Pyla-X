"""Gameplay and debug clip recording.

Normal clips store the clean gameplay frame. Debug clips store the same tick
after overlay graphics have been drawn. Encoding is optional: if an MP4 writer
cannot be opened, recording is skipped instead of raising.
"""

from __future__ import annotations

import atexit
import os
import time


def frame_for_clip(clean_frame, overlay_frame, *, include_overlays):
    """Pick the frame that should be muxed into a clip.

    Normal clips always keep the clean gameplay frame, even when a debug
    overlay frame exists for the same tick.
    """
    if include_overlays:
        if overlay_frame is None:
            raise ValueError("An overlay frame is required when include_overlays is set.")
        return overlay_frame
    return clean_frame


def clip_outputs_for_tick(clean_frame, overlay_frame, *, record_normal, record_debug):
    """Return ``(kind, frame)`` pairs to mux for one gameplay tick."""
    outputs = []
    if record_normal:
        outputs.append(("normal", frame_for_clip(clean_frame, overlay_frame, include_overlays=False)))
    if record_debug:
        outputs.append(("debug", frame_for_clip(clean_frame, overlay_frame, include_overlays=True)))
    return outputs


def open_mp4_writer(path, fps, size):
    """Open an MP4 writer, or return ``None`` when encoding is unavailable."""
    try:
        import cv2
    except ImportError:
        return None

    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        return None

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    except Exception:
        return None

    if writer is None or not writer.isOpened():
        try:
            writer.release()
        except Exception:
            pass
        return None
    return writer


def _copy_frame(image):
    copy = getattr(image, "copy", None)
    if callable(copy):
        try:
            return copy()
        except Exception:
            return image
    return image


def _frame_size(image, fallback_width, fallback_height):
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    if fallback_width and fallback_height:
        return int(fallback_width), int(fallback_height)
    raise ValueError("Clip frame size is unknown.")


class ClipRecorder:
    def __init__(
        self,
        width=None,
        height=None,
        fps=30.0,
        missing_player_grace=1.0,
        min_player_seen_before_recording=3.0,
        filename_prefix="clip",
        output_dir=None,
        writer_factory=None,
    ):
        self.width = int(width) if width else None
        self.height = int(height) if height else None
        try:
            self.fps = max(float(fps or 30.0), 1.0)
        except (TypeError, ValueError):
            self.fps = 30.0
        self.missing_player_grace = float(missing_player_grace)
        self.min_player_seen_before_recording = float(min_player_seen_before_recording)
        self.filename_prefix = filename_prefix
        self.output_dir = output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips")
        self.writer_factory = writer_factory or open_mp4_writer
        self.writer = None
        self.path = None
        self.frames_written = 0
        self._encoding_unavailable = False
        self.player_seen_since = None
        self.last_player_seen = None
        self.last_frame_written_at = None
        self.pending_frames = []

    def update(self, image, debug_data, frame_advanced):
        if self._encoding_unavailable or not debug_data or not frame_advanced or image is None:
            return

        now = time.time()
        player_detected = bool(debug_data.get("player"))
        if player_detected:
            if self.player_seen_since is None:
                self.player_seen_since = now
            self.last_player_seen = now
            if self.writer is None and now - self.player_seen_since >= self.min_player_seen_before_recording:
                self.start(now, image)
                if self._encoding_unavailable:
                    return
                self.flush_pending_frames()
        else:
            self.player_seen_since = None

        if self.writer is None:
            if player_detected:
                self.pending_frames.append((now, _copy_frame(image)))
                self.prune_pending_frames(now)
            else:
                self.pending_frames.clear()
            return

        self.write_frame(image, now)

        if (
            not player_detected
            and self.last_player_seen is not None
            and now - self.last_player_seen > self.missing_player_grace
        ):
            self.stop()

    def write_frame(self, image, timestamp):
        if self.writer is None:
            return

        if self.last_frame_written_at is None:
            frames_to_write = 1
        else:
            elapsed = max(timestamp - self.last_frame_written_at, 0)
            frames_to_write = max(1, int(round(elapsed * self.fps)))
            frames_to_write = min(frames_to_write, int(max(self.fps * 2, 1)))

        for _ in range(frames_to_write):
            self.writer.write(image)
            self.frames_written += 1
        self.last_frame_written_at = timestamp

    def prune_pending_frames(self, now):
        keep_seconds = self.min_player_seen_before_recording + self.missing_player_grace
        self.pending_frames = [
            (timestamp, frame)
            for timestamp, frame in self.pending_frames
            if now - timestamp <= keep_seconds
        ]

    def flush_pending_frames(self):
        for timestamp, frame in self.pending_frames:
            self.write_frame(frame, timestamp)
        self.pending_frames.clear()

    def start(self, now, image=None):
        try:
            width, height = _frame_size(image, self.width, self.height)
        except ValueError as error:
            print(f"Clip recorder skipped a clip: {error}")
            return

        self.width = width
        self.height = height
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        path = os.path.join(self.output_dir, f"{self.filename_prefix}_{timestamp}.mp4")
        writer = self.writer_factory(path, self.fps, (width, height))
        opened = True
        is_opened = getattr(writer, "isOpened", None)
        if writer is None:
            opened = False
        elif callable(is_opened):
            opened = bool(is_opened())

        if not opened:
            print(f"Clip recorder could not open {path}; encoding was skipped.")
            self._encoding_unavailable = True
            self.pending_frames.clear()
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            self.writer = None
            self.path = None
            return

        self.path = path
        self.writer = writer
        self.frames_written = 0
        self.last_frame_written_at = None

    def stop(self):
        if self.writer is None:
            self.pending_frames.clear()
            return

        try:
            self.writer.release()
        except Exception:
            pass
        saved_path = self.path
        frames_written = self.frames_written
        self.writer = None
        self.path = None
        self.frames_written = 0
        self.player_seen_since = None
        self.last_player_seen = None
        self.last_frame_written_at = None
        self.pending_frames.clear()
        if frames_written and saved_path:
            print(f"Saved clip: {saved_path}")

    def close(self):
        self.stop()


class NormalClipRecorder(ClipRecorder):
    """Record clean gameplay frames with no debug overlay."""

    def __init__(self, fps=30.0, **kwargs):
        kwargs.setdefault("filename_prefix", "gameplay_clip")
        kwargs.setdefault(
            "output_dir",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips"),
        )
        super().__init__(fps=fps, **kwargs)
        atexit.register(self.close)

    def record_gameplay(self, clean_frame, player_boxes, frame_advanced=True):
        for _kind, clip_frame in clip_outputs_for_tick(
            clean_frame,
            None,
            record_normal=True,
            record_debug=False,
        ):
            self.update(clip_frame, {"player": player_boxes or []}, frame_advanced)
