"""Predictive aiming for Pyla (v0.9.2).

Lead calculation uses target velocity, projectile speed, shooter speed, and
projectile size. Throwers and Nani use a different trajectory than a straight shot.
"""

from aiming.delay import DEFAULT_AIM_DELAY, DynamicDelay
from aiming.engine import aim_at_target, projectile_profile, trajectory_kind
from aiming.lead import calculate_lead, stick_to_world_velocity
from aiming.playstyles import AIMBOT_PLAYSTYLES, registered_aimbot_names
from aiming.tracking import MotionTracker

__all__ = [
    "AIMBOT_PLAYSTYLES",
    "DEFAULT_AIM_DELAY",
    "DynamicDelay",
    "MotionTracker",
    "aim_at_target",
    "calculate_lead",
    "projectile_profile",
    "registered_aimbot_names",
    "stick_to_world_velocity",
    "trajectory_kind",
]
