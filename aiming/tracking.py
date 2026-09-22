"""Enemy velocity estimated from successive detections."""

import math


class MotionTracker:
    """Match detections frame to frame and report velocity in units per second."""

    def __init__(self, match_distance=160.0):
        distance = match_distance
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            distance = 160.0
        if not math.isfinite(distance) or distance <= 0.0:
            distance = 160.0
        self.match_distance = distance
        self._tracks = []

    def update(self, positions, timestamp):
        try:
            now = float(timestamp)
        except (TypeError, ValueError):
            return list(self._tracks)
        if not math.isfinite(now):
            return list(self._tracks)

        unused = list(self._tracks)
        updated = []
        for position in positions or []:
            parsed = _point(position)
            if parsed is None:
                continue
            best_index = None
            best_distance = self.match_distance
            for index, track in enumerate(unused):
                gap = math.hypot(parsed[0] - track["pos"][0], parsed[1] - track["pos"][1])
                if gap < best_distance:
                    best_distance = gap
                    best_index = index
            if best_index is None:
                velocity = (0.0, 0.0)
            else:
                previous = unused.pop(best_index)
                dt = now - previous["t"]
                if dt > 1e-6:
                    velocity = (
                        (parsed[0] - previous["pos"][0]) / dt,
                        (parsed[1] - previous["pos"][1]) / dt,
                    )
                else:
                    velocity = previous["velocity"]
            updated.append({"pos": parsed, "velocity": velocity, "t": now})
        self._tracks = updated
        return list(updated)

    def velocity_near(self, position, max_distance=None):
        parsed = _point(position)
        if parsed is None or not self._tracks:
            return (0.0, 0.0)
        limit = self.match_distance if max_distance is None else max_distance
        try:
            limit = float(limit)
        except (TypeError, ValueError):
            limit = self.match_distance
        best = None
        best_distance = limit
        for track in self._tracks:
            gap = math.hypot(parsed[0] - track["pos"][0], parsed[1] - track["pos"][1])
            if gap <= best_distance:
                best_distance = gap
                best = track
        if best is None:
            return (0.0, 0.0)
        return best["velocity"]


def _point(position):
    if position is None or isinstance(position, (str, bytes)):
        return None
    try:
        x = float(position[0])
        y = float(position[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return (x, y)
