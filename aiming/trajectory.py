"""Straight, arcing, and Nani projectile paths."""

import math

from aiming.lead import calculate_lead

THROWER_BRAWLERS = frozenset({
    "barley",
    "dynamike",
    "tick",
    "sprout",
    "grom",
    "willow",
    "berry",
    "juju",
})

LOB_ANGLE_RADIANS = math.radians(55.0)
NANI_SPREAD_RADIANS = math.radians(12.0)
NANI_SUPER_TURN_RATE = math.pi
NANI_SUPER_FLIGHT_SECONDS = 4.0
PATH_SAMPLES = 8


def trajectory_kind(brawler, skill="attack", brawler_info=None):
    """Pick the projectile model for a brawler and attack or super."""
    info = brawler_info or {}
    skill_name = "super" if skill == "super" else "attack"
    explicit = info.get(f"{skill_name}_trajectory")
    if not explicit and skill_name == "attack":
        explicit = info.get("attack_trajectory") or info.get("trajectory")
    if explicit:
        return str(explicit)

    name = str(brawler or "").strip().lower()
    if name == "nani" and skill_name == "super":
        return "nani_super"
    if name == "nani":
        return "nani_attack"
    if name in THROWER_BRAWLERS:
        return "arc"
    return "linear"


def _rotate(vector, angle):
    x, y = vector
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return (
        x * cos_angle - y * sin_angle,
        x * sin_angle + y * cos_angle,
    )


def _sample_line(start, end, height_at=None, samples=PATH_SAMPLES):
    sx, sy = start
    ex, ey = end
    points = []
    steps = max(1, samples)
    for index in range(steps + 1):
        u = index / steps
        x = sx + (ex - sx) * u
        y = sy + (ey - sy) * u
        height = 0.0 if height_at is None else height_at(u)
        points.append((x, y, height))
    return points


def _attach_path(solution, path, paths=None):
    solution = dict(solution)
    solution["path"] = path
    solution["paths"] = paths if paths is not None else ([path] if path else [])
    return solution


def solve_linear(shooter_pos, target_pos, **kwargs):
    solution = calculate_lead(shooter_pos, target_pos, trajectory="linear", **kwargs)
    if not solution["feasible"] or solution["aim_point"] is None:
        return solution
    path = _sample_line(shooter_pos, solution["aim_point"])
    return _attach_path(solution, path)


def solve_arc(shooter_pos, target_pos, projectile_speed, lob_angle=LOB_ANGLE_RADIANS, **kwargs):
    """Thrower lob. Horizontal speed is slower, so the lead is longer, and the path has height."""
    try:
        angle = float(lob_angle)
        speed = float(projectile_speed)
    except (TypeError, ValueError):
        return calculate_lead(shooter_pos, target_pos, projectile_speed=0, trajectory="arc", **kwargs)
    if not math.isfinite(angle) or not math.isfinite(speed):
        return calculate_lead(shooter_pos, target_pos, projectile_speed=0, trajectory="arc", **kwargs)

    horizontal_speed = speed * math.cos(angle)
    solution = calculate_lead(
        shooter_pos,
        target_pos,
        projectile_speed=horizontal_speed,
        trajectory="arc",
        **kwargs,
    )
    if not solution["feasible"] or solution["aim_point"] is None:
        return solution

    start = shooter_pos
    end = solution["aim_point"]
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    slope = math.tan(angle)

    def height_at(u):
        return distance * slope * u * (1.0 - u)

    path = _sample_line(start, end, height_at=height_at)
    return _attach_path(solution, path)


def _nani_ray(shooter, direction, distance):
    length = math.hypot(direction[0], direction[1])
    if length < 1e-9:
        return _sample_line(shooter, shooter)
    scale = distance / length
    end = (shooter[0] + direction[0] * scale, shooter[1] + direction[1] * scale)
    outbound = _sample_line(shooter, end, samples=4)
    returning = _sample_line(end, shooter, samples=4)[1:]
    return outbound + returning


def solve_nani_attack(shooter_pos, target_pos, projectile_speed, spread=NANI_SPREAD_RADIANS, **kwargs):
    """Nani's main attack: three orbs that travel out and return."""
    solution = calculate_lead(
        shooter_pos,
        target_pos,
        projectile_speed=projectile_speed,
        trajectory="nani_attack",
        **kwargs,
    )
    if not solution["feasible"]:
        return solution

    direction = solution["aim_vector"]
    flight = solution["time_to_impact"] or 0.0
    try:
        travel = abs(float(projectile_speed)) * flight
    except (TypeError, ValueError):
        travel = 0.0
    if travel < 1e-6:
        travel = math.hypot(direction[0], direction[1])

    paths = [
        _nani_ray(shooter_pos, _rotate(direction, -spread), travel),
        _nani_ray(shooter_pos, direction, travel),
        _nani_ray(shooter_pos, _rotate(direction, spread), travel),
    ]
    return _attach_path(solution, paths[1], paths)


def _normalize(vector):
    length = math.hypot(vector[0], vector[1])
    if length < 1e-9:
        return None
    return (vector[0] / length, vector[1] / length)


def _rotate_towards(current, desired, max_angle):
    current_n = _normalize(current)
    desired_n = _normalize(desired)
    if current_n is None:
        return desired_n or (1.0, 0.0)
    if desired_n is None:
        return current_n
    cross = current_n[0] * desired_n[1] - current_n[1] * desired_n[0]
    dot = max(-1.0, min(1.0, current_n[0] * desired_n[0] + current_n[1] * desired_n[1]))
    angle = math.atan2(cross, dot)
    angle = max(-max_angle, min(max_angle, angle))
    return _rotate(current_n, angle)


def solve_nani_super(
    shooter_pos,
    target_pos,
    target_velocity=(0.0, 0.0),
    projectile_speed=900.0,
    projectile_radius=40.0,
    max_range=None,
    latency=0.0,
    turn_rate=NANI_SUPER_TURN_RATE,
    dt=0.05,
    **_kwargs,
):
    """Peep: a steerable projectile that can curve toward the target."""
    from aiming.lead import _as_float, _as_vec, _empty_solution

    shooter = _as_vec(shooter_pos)
    target = _as_vec(target_pos)
    velocity = _as_vec(target_velocity if target_velocity is not None else (0.0, 0.0))
    speed = _as_float(projectile_speed, None)
    radius = _as_float(projectile_radius, 0.0)
    delay = _as_float(latency, 0.0)
    step = _as_float(dt, 0.05)
    rate = _as_float(turn_rate, NANI_SUPER_TURN_RATE)
    if (
        shooter is None
        or target is None
        or velocity is None
        or speed is None
        or speed <= 0.0
        or radius is None
        or delay is None
        or delay < 0.0
        or step is None
        or step <= 0.0
        or rate is None
        or rate < 0.0
    ):
        reason = "impossible_speed" if speed is not None and speed <= 0.0 else "invalid"
        return _empty_solution(reason, "nani_super")
    radius = max(0.0, radius)

    duration = NANI_SUPER_FLIGHT_SECONDS
    if max_range is not None:
        limit = _as_float(max_range, None)
        if limit is None or limit < 0.0:
            return _empty_solution("invalid", "nani_super")
        if limit > 0.0:
            duration = min(duration, limit / speed)

    pos_x = shooter[0]
    pos_y = shooter[1]
    aim = _normalize((
        target[0] + velocity[0] * delay - pos_x,
        target[1] + velocity[1] * delay - pos_y,
    )) or (1.0, 0.0)
    path = [(pos_x, pos_y, 0.0)]
    elapsed = 0.0
    closest = math.hypot(target[0] - pos_x, target[1] - pos_y)

    while elapsed < duration + 1e-9:
        remaining = min(step, duration - elapsed)
        if remaining <= 1e-9:
            break
        sample_t = delay + elapsed + remaining
        future = (target[0] + velocity[0] * sample_t, target[1] + velocity[1] * sample_t)
        aim = _rotate_towards(aim, (future[0] - pos_x, future[1] - pos_y), rate * remaining)
        pos_x += aim[0] * speed * remaining
        pos_y += aim[1] * speed * remaining
        elapsed += remaining
        path.append((pos_x, pos_y, 0.0))
        gap = math.hypot(future[0] - pos_x, future[1] - pos_y)
        closest = min(closest, gap)
        if gap <= radius:
            aim_point = future
            lead = (aim_point[0] - target[0], aim_point[1] - target[1])
            return {
                "feasible": True,
                "aim_point": aim_point,
                "time_to_impact": elapsed,
                "reason": "ok",
                "trajectory": "nani_super",
                "lead": lead,
                "path": path,
                "paths": [path],
                "in_range": True,
                "aim_vector": (aim[0] * speed, aim[1] * speed),
            }

    separating = (target[0] - shooter[0]) * velocity[0] + (target[1] - shooter[1]) * velocity[1] > 0.0
    target_speed = math.hypot(velocity[0], velocity[1])
    reason = "impossible_speed" if separating and target_speed >= speed else "out_of_range"
    solution = _empty_solution(reason, "nani_super")
    solution["path"] = path
    solution["paths"] = [path]
    return solution
