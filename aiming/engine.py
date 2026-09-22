"""Dispatch a brawler attack onto the matching trajectory solver."""

from aiming.lead import (
    DEFAULT_BRAWLER_SPEED,
    DEFAULT_PROJECTILE_RADIUS,
    DEFAULT_PROJECTILE_SPEED,
    _as_float,
    _empty_solution,
)
from aiming.trajectory import (
    solve_arc,
    solve_linear,
    solve_nani_attack,
    solve_nani_super,
    trajectory_kind,
)


def projectile_profile(brawler_info, skill="attack", brawler=None):
    """Read projectile speed, radius, brawler speed, and range from brawler info."""
    info = brawler_info or {}
    skill_name = "super" if skill == "super" else "attack"
    speed = _as_float(info.get(f"{skill_name}_projectile_speed"), DEFAULT_PROJECTILE_SPEED)
    radius = _as_float(info.get(f"{skill_name}_projectile_radius"), DEFAULT_PROJECTILE_RADIUS)
    brawler_speed = _as_float(info.get("speed"), DEFAULT_BRAWLER_SPEED)
    if speed is None or speed <= 0.0:
        speed = DEFAULT_PROJECTILE_SPEED
    if radius is None or radius < 0.0:
        radius = DEFAULT_PROJECTILE_RADIUS
    if brawler_speed is None or brawler_speed < 0.0:
        brawler_speed = DEFAULT_BRAWLER_SPEED

    raw_range = info.get(f"{skill_name}_range")
    if raw_range is None and skill_name == "attack":
        raw_range = info.get("attack_range")
    name = str(brawler or "").strip().lower()
    if name == "nani" and skill_name == "super":
        parsed = _as_float(raw_range, 0.0)
        max_range = None if parsed is None or parsed <= 0.0 else parsed
    elif raw_range is None:
        max_range = None
    else:
        max_range = _as_float(raw_range, None)
        if max_range is not None and max_range < 0.0:
            max_range = None

    return {
        "projectile_speed": speed,
        "projectile_radius": radius,
        "brawler_speed": brawler_speed,
        "max_range": max_range,
    }


def aim_at_target(
    shooter_pos,
    target_pos,
    target_velocity=(0.0, 0.0),
    projectile_speed=None,
    shooter_speed=0.0,
    shooter_velocity=None,
    projectile_radius=None,
    max_range=None,
    latency=0.0,
    brawler=None,
    brawler_info=None,
    skill="attack",
    trajectory=None,
    scale=1.0,
):
    """Build an aim solution for one target.

    Explicit projectile speed, radius, and range override the brawler profile.
    ``scale`` resizes profile values that were not passed explicitly, matching
    a non-1080p frame.
    """
    kind = trajectory or trajectory_kind(brawler, skill, brawler_info)
    profile = projectile_profile(brawler_info, skill, brawler)
    frame_scale = _as_float(scale, None)
    if frame_scale is None or frame_scale <= 0.0:
        return _empty_solution("invalid", kind)

    speed = projectile_speed
    if speed is None:
        speed = profile["projectile_speed"] * frame_scale
    radius = projectile_radius
    if radius is None:
        radius = profile["projectile_radius"] * frame_scale
    limited_range = max_range
    if limited_range is None and profile["max_range"] is not None:
        limited_range = profile["max_range"] * frame_scale

    common = {
        "target_velocity": target_velocity,
        "shooter_speed": shooter_speed,
        "shooter_velocity": shooter_velocity,
        "projectile_radius": radius,
        "max_range": limited_range,
        "latency": latency,
    }
    try:
        if kind == "arc":
            return solve_arc(shooter_pos, target_pos, projectile_speed=speed, **common)
        if kind == "nani_attack":
            return solve_nani_attack(shooter_pos, target_pos, projectile_speed=speed, **common)
        if kind == "nani_super":
            return solve_nani_super(
                shooter_pos,
                target_pos,
                projectile_speed=speed,
                **common,
            )
        if kind != "linear":
            solution = _empty_solution("invalid", str(kind))
            return solution
        return solve_linear(shooter_pos, target_pos, projectile_speed=speed, **common)
    except (TypeError, ValueError, OverflowError):
        return _empty_solution("invalid", kind)
