"""Input delay that follows measured capture and touch latency."""

import math

DEFAULT_AIM_DELAY = 0.1


class DynamicDelay:
    """Smooth capture plus input latency into a delay the aimer can use.

    ``current`` returns ``default`` until a finite, non-negative sample exists.
    Later samples are blended with ``smoothing`` (1 keeps only the newest
    sample) and clamped to ``[minimum, maximum]``.
    """

    def __init__(self, default=DEFAULT_AIM_DELAY, minimum=0.0, maximum=0.35, smoothing=0.5):
        parsed_default = _finite(default)
        self.default = DEFAULT_AIM_DELAY if parsed_default is None or parsed_default < 0.0 else parsed_default
        parsed_min = _finite(minimum)
        parsed_max = _finite(maximum)
        self.minimum = 0.0 if parsed_min is None or parsed_min < 0.0 else parsed_min
        self.maximum = 0.35 if parsed_max is None else parsed_max
        if self.maximum < self.minimum:
            self.maximum = self.minimum
        parsed_smoothing = _finite(smoothing)
        if parsed_smoothing is None:
            parsed_smoothing = 0.5
        self.smoothing = min(1.0, max(0.0, parsed_smoothing))
        self._estimate = None

    def has_measurement(self):
        return self._estimate is not None

    def observe(self, capture_latency=None, input_latency=None):
        parts = []
        for value in (capture_latency, input_latency):
            number = _finite(value)
            if number is not None and number >= 0.0:
                parts.append(number)
        if not parts:
            return self.current()

        sample = sum(parts)
        if self._estimate is None:
            self._estimate = sample
        else:
            weight = self.smoothing
            self._estimate = weight * sample + (1.0 - weight) * self._estimate
        return self.current()

    def current(self):
        if self._estimate is None:
            return self._clamp(self.default)
        return self._clamp(self._estimate)

    def reset(self):
        self._estimate = None

    def _clamp(self, value):
        return min(self.maximum, max(self.minimum, value))


def _finite(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
