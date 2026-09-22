"""Closed-form intercept math for predictive aiming.

The projectile is launched from the shooter and inherits the shooter's velocity.
A hit lands when the projectile comes within ``projectile_radius`` of the target
center. Impossible shots return a result instead of raising.
"""

import math

DEFAULT_PROJECTILE_SPEED = 2200.0
DEFAULT_PROJECTILE_RADIUS = 40.0
DEFAULT_BRAWLER_SPEED = 720.0


def _empty_solution(reason, trajectory="linear"):
    return {
        "feasible": False,
        "aim_point": None,
        "time_to_impact": None,
        "reason": reason,
        "trajectory": trajectory,
        "lead": (0.0, 0.0),
        "path": [],
        "paths": [],
        "in_range": False,
        "aim_vector": (0.0, 0.0),
    }


def _as_vec(value):
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _as_float(value, fallback=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(number):
        return fallback
    return number


def _positive_roots(a, b, c):
    """Real positive roots of ``a t^2 + b t + c = 0``."""
    eps = 1e-9
    if abs(a) < eps:
        if abs(b) < eps:
            return []
        root = -c / b
        return [root] if root > eps else []

    discriminant = b * b - 4.0 * a * c
    if discriminant < -1e-6:
        return []
    sqrt_disc = math.sqrt(max(0.0, discriminant))
    if b >= 0.0:
        q = -0.5 * (b + sqrt_disc)
    else:
        q = -0.5 * (b - sqrt_disc)

    roots = []
    if abs(q) > eps:
        first = q / a
        if first > eps:
            roots.append(first)
        second = c / q
        if second > eps:
            roots.append(second)
    else:
        root = -b / a
        if root > eps:
            roots.append(root)
    return roots


def stick_to_world_velocity(movement, brawler_speed, radius=100.0):
    """Convert a joystick vector into a world-space velocity.

    A full deflection (``radius``) is ``brawler_speed``. Partial stick scales
    linearly. Non-vectors and a still stick return zero.
    """
    vector = _as_vec(movement)
    speed = _as_float(brawler_speed, 0.0)
    stick_radius = _as_float(radius, 0.0)
    if vector is None or speed is None or speed <= 0.0 or stick_radius is None or stick_radius <= 0.0:
        return (0.0, 0.0)
    mag = math.hypot(vector[0], vector[1])
    if mag < 1.0:
        return (0.0, 0.0)
    fraction = min(1.0, mag / stick_radius)
    scale = speed * fraction / mag
    return (vector[0] * scale, vector[1] * scale)


def calculate_lead(
    shooter_pos,
    target_pos,
    target_velocity=(0.0, 0.0),
    projectile_speed=DEFAULT_PROJECTILE_SPEED,
    shooter_speed=0.0,
    shooter_velocity=None,
    projectile_radius=0.0,
    max_range=None,
    latency=0.0,
    trajectory="linear",
):
    """Solve the earliest intercept of a moving target.

    ``shooter_speed`` is used only when ``shooter_velocity`` is omitted. A
    positive speed moves the shooter toward the target and is added to the
    projectile by velocity inheritance. ``latency`` is seconds of capture and
    input delay applied before the projectile leaves the brawler.
    """
    shooter = _as_vec(shooter_pos)
    target = _as_vec(target_pos)
    velocity = _as_vec(target_velocity if target_velocity is not None else (0.0, 0.0))
    if shooter is None or target is None or velocity is None:
        return _empty_solution("invalid", trajectory)

    speed = _as_float(projectile_speed, None)
    if speed is None or speed <= 0.0:
        return _empty_solution("impossible_speed", trajectory)

    radius = _as_float(projectile_radius, 0.0)
    if radius is None:
        return _empty_solution("invalid", trajectory)
    radius = max(0.0, radius)

    delay = _as_float(latency, 0.0)
    if delay is None or delay < 0.0:
        return _empty_solution("invalid", trajectory)

    inherited = _as_vec(shooter_velocity) if shooter_velocity is not None else None
    if shooter_velocity is not None and inherited is None:
        return _empty_solution("invalid", trajectory)
    if inherited is None:
        scalar_speed = _as_float(shooter_speed, 0.0)
        if scalar_speed is None:
            return _empty_solution("invalid", trajectory)
        inherited = (0.0, 0.0)
        if scalar_speed != 0.0:
            look_x = target[0] + velocity[0] * delay - shooter[0]
            look_y = target[1] + velocity[1] * delay - shooter[1]
            look = math.hypot(look_x, look_y)
            if look > 1e-9:
                inherited = (scalar_speed * look_x / look, scalar_speed * look_y / look)

    range_limit = None
    if max_range is not None:
        range_limit = _as_float(max_range, None)
        if range_limit is None or range_limit < 0.0:
            return _empty_solution("invalid", trajectory)

    fire_x = shooter[0] + inherited[0] * delay
    fire_y = shooter[1] + inherited[1] * delay
    origin_x = target[0] + velocity[0] * delay
    origin_y = target[1] + velocity[1] * delay

    rel_x = origin_x - fire_x
    rel_y = origin_y - fire_y
    rel_vx = velocity[0] - inherited[0]
    rel_vy = velocity[1] - inherited[1]
    distance = math.hypot(rel_x, rel_y)

    if distance <= radius:
        aim_point = (origin_x, origin_y)
        lead = (aim_point[0] - target[0], aim_point[1] - target[1])
        aim_vector = (origin_x - fire_x, origin_y - fire_y)
        return {
            "feasible": True,
            "aim_point": aim_point,
            "time_to_impact": 0.0,
            "reason": "ok",
            "trajectory": trajectory,
            "lead": lead,
            "path": [],
            "paths": [],
            "in_range": True,
            "aim_vector": aim_vector,
        }

    a = rel_vx * rel_vx + rel_vy * rel_vy - speed * speed
    b = 2.0 * (rel_x * rel_vx + rel_y * rel_vy - speed * radius)
    c = distance * distance - radius * radius
    roots = _positive_roots(a, b, c)
    if not roots:
        return _empty_solution("impossible_speed", trajectory)

    flight = min(roots)
    if range_limit is not None and speed * flight > range_limit + 1e-6:
        return _empty_solution("out_of_range", trajectory)

    aim_x = origin_x + velocity[0] * flight
    aim_y = origin_y + velocity[1] * flight
    aim_point = (aim_x, aim_y)
    lead = (aim_x - target[0], aim_y - target[1])
    aim_vector = (
        aim_x - fire_x - inherited[0] * flight,
        aim_y - fire_y - inherited[1] * flight,
    )
    return {
        "feasible": True,
        "aim_point": aim_point,
        "time_to_impact": flight,
        "reason": "ok",
        "trajectory": trajectory,
        "lead": lead,
        "path": [],
        "paths": [],
        "in_range": True,
        "aim_vector": aim_vector,
    }
