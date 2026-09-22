"""Decision logic for in-match safety and lobby recovery.

Every function takes already-detected state and returns movement, aim, or
UI actions. Callers do not need a live game.
"""

import math
import re

from utils import JOYSTICK_RADIUS, normalize_move


BOUNDARY_MODES = {"knockout", "brawlball", "brawlball_5v5"}
SHOWDOWN_MODES = {"showdown", "solo_showdown", "duo_showdown", "trio_showdown"}
UNWANTED_LOBBY_MODES = {"solo_showdown", "ranked"}

_MODE_ALIASES = {
    "solo showdown": "solo_showdown",
    "solo": "solo_showdown",
    "ranked": "ranked",
    "rank": "ranked",
    "trio showdown": "trio_showdown",
    "duo showdown": "duo_showdown",
    "showdown": "showdown",
    "brawl ball": "brawlball",
    "brawlball": "brawlball",
    "brawl ball 5v5": "brawlball_5v5",
    "knockout": "knockout",
    "knock out": "knockout",
    "gem grab": "gem_grab",
    "hot zone": "hot_zone",
    "heist": "heist",
    "bounty": "bounty",
    "wipeout": "wipeout",
}

_LOBBY_MODE_PATTERNS = (
    ("solo showdown", "solo_showdown"),
    ("trio showdown", "trio_showdown"),
    ("duo showdown", "duo_showdown"),
    ("brawl ball 5v5", "brawlball_5v5"),
    ("brawl ball", "brawlball"),
    ("brawlball", "brawlball"),
    ("knockout", "knockout"),
    ("gem grab", "gem_grab"),
    ("hot zone", "hot_zone"),
    ("ranked", "ranked"),
    ("heist", "heist"),
    ("bounty", "bounty"),
    ("wipeout", "wipeout"),
)


def normalize_gamemode(value):
    if value is None:
        return ""
    text = str(value).strip().lower().replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    if not text or text == "all":
        return ""
    if text in _MODE_ALIASES:
        return _MODE_ALIASES[text]
    underscored = text.replace(" ", "_")
    return _MODE_ALIASES.get(underscored, underscored)


def is_showdown_mode(mode):
    return normalize_gamemode(mode) in SHOWDOWN_MODES


def resolve_configured_gamemode(bot_config=None, playstyle_info=None):
    config = bot_config or {}
    explicit = str(config.get("configured_gamemode") or "").strip()
    if explicit:
        return explicit
    for mode in (playstyle_info or {}).get("gamemodes") or []:
        normalized = normalize_gamemode(mode)
        if normalized:
            return normalized
    return ""


def _text_of(item):
    if isinstance(item, dict):
        return str(item.get("text", ""))
    return str(item)


def _normalize_rect(rect):
    x1, y1, x2, y2 = rect[:4]
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def point_in_rect(point, rect):
    x, y = point
    x1, y1, x2, y2 = _normalize_rect(rect)
    return x1 <= x <= x2 and y1 <= y <= y2


def _rect_exit_toward(origin, target, rect):
    """First boundary hit on the segment from an interior point toward target."""
    x1, y1, x2, y2 = _normalize_rect(rect)
    ox, oy = origin
    tx, ty = target
    dx, dy = tx - ox, ty - oy
    hits = []
    if dx:
        for edge_x in (x1, x2):
            t = (edge_x - ox) / dx
            if 0 < t <= 1:
                y = oy + t * dy
                if y1 - 1e-6 <= y <= y2 + 1e-6:
                    hits.append((t, (edge_x, y)))
    if dy:
        for edge_y in (y1, y2):
            t = (edge_y - oy) / dy
            if 0 < t <= 1:
                x = ox + t * dx
                if x1 - 1e-6 <= x <= x2 + 1e-6:
                    hits.append((t, (x, edge_y)))
    if not hits:
        return (float(tx), float(ty))
    hits.sort(key=lambda item: item[0])
    exit_x, exit_y = hits[0][1]
    return (float(exit_x), float(exit_y))


def _rects_overlap(first, second):
    ax1, ay1, ax2, ay2 = _normalize_rect(first)
    bx1, by1, bx2, by2 = _normalize_rect(second)
    return ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1


def detect_brawlball_cage(player_pos, cages, map_center, radius=JOYSTICK_RADIUS):
    """Return a path to map center when the player is inside a goal cage."""
    center = (float(map_center[0]), float(map_center[1]))
    player = (float(player_pos[0]), float(player_pos[1]))
    for cage in cages or []:
        rect = cage.get("rect") if isinstance(cage, dict) else cage
        if rect is None or not point_in_rect(player, rect):
            continue
        opening = cage.get("opening") if isinstance(cage, dict) else None
        if opening is None:
            opening = _rect_exit_toward(player, center, rect)
        opening = (float(opening[0]), float(opening[1]))
        movement = normalize_move(center[0] - player[0], center[1] - player[1], radius)
        return {
            "trapped": True,
            "cage": _normalize_rect(rect),
            "opening": opening,
            "path": [opening, center],
            "movement": movement,
        }
    return {
        "trapped": False,
        "cage": None,
        "opening": None,
        "path": [],
        "movement": (0.0, 0.0),
    }


def detect_wall_cage(player_pos, walls, map_center, depth=180, flank=110, thickness=48, radius=JOYSTICK_RADIUS):
    """Treat a three-sided wall pocket that opens toward map center as a cage.

    The back slab and both flanks must contain walls. The side facing map
    center must be open, which is the mouth the bot paths through.
    """
    idle = {"trapped": False, "cage": None, "opening": None, "path": [], "movement": (0.0, 0.0)}
    if not walls or map_center is None:
        return idle
    px, py = float(player_pos[0]), float(player_pos[1])
    cx, cy = float(map_center[0]), float(map_center[1])
    dx, dy = cx - px, cy - py
    distance = math.hypot(dx, dy)
    if distance < 1:
        return idle
    ux, uy = dx / distance, dy / distance
    perpendicular = (-uy, ux)

    def slab(center_point):
        sx, sy = center_point
        return (sx - thickness, sy - thickness, sx + thickness, sy + thickness)

    def blocked(center_point):
        area = slab(center_point)
        return any(_rects_overlap(area, wall) for wall in walls)

    back = (px - ux * depth * 0.55, py - uy * depth * 0.55)
    flank_a = (px + perpendicular[0] * flank, py + perpendicular[1] * flank)
    flank_b = (px - perpendicular[0] * flank, py - perpendicular[1] * flank)
    front = (px + ux * depth * 0.45, py + uy * depth * 0.45)
    if not (blocked(back) and blocked(flank_a) and blocked(flank_b)) or blocked(front):
        return idle

    opening = (px + ux * depth, py + uy * depth)
    return {
        "trapped": True,
        "cage": slab(back),
        "opening": opening,
        "path": [opening, (cx, cy)],
        "movement": normalize_move(dx, dy, radius),
    }


def poison_gas_avoidance(player_pos, safe_center, safe_radius, edge_margin=80, radius=JOYSTICK_RADIUS):
    """Steer into the safe circle, and away from a shrinking edge."""
    px, py = float(player_pos[0]), float(player_pos[1])
    cx, cy = float(safe_center[0]), float(safe_center[1])
    safe_radius = float(safe_radius)
    dx, dy = cx - px, cy - py
    distance = math.hypot(dx, dy)
    distance_to_edge = safe_radius - distance
    outside = distance_to_edge < 0
    near_edge = (not outside) and distance_to_edge <= float(edge_margin)
    active = outside or near_edge
    movement = normalize_move(dx, dy, radius) if active and distance > 0 else None
    return {
        "active": active and movement is not None,
        "outside_safe_zone": outside,
        "near_edge": near_edge,
        "distance_to_edge": distance_to_edge,
        "movement": movement,
        "source": "circle",
    }


def avoid_directional_gas(gas, radius=JOYSTICK_RADIUS):
    """Move opposite the stronger poison side on each axis.

    Positive Y is downward, matching the existing showdown playstyles:
    gas above the player produces a positive Y movement.
    """
    if not gas:
        return None
    up = gas.get("up") or 0
    down = gas.get("down") or 0
    left = gas.get("left") or 0
    right = gas.get("right") or 0
    x = y = 0
    if up or down:
        y = radius if up > down else -radius
    if left or right:
        x = radius if left > right else -radius
    if x == 0 and y == 0:
        return None
    return (float(x), float(y))


def resolve_gas_movement(player_pos=None, safe_center=None, safe_radius=None, directional=None, edge_margin=80, radius=JOYSTICK_RADIUS):
    if safe_center is not None and safe_radius is not None and player_pos is not None:
        return poison_gas_avoidance(player_pos, safe_center, safe_radius, edge_margin, radius)
    movement = avoid_directional_gas(directional, radius)
    return {
        "active": movement is not None,
        "outside_safe_zone": movement is not None,
        "near_edge": False,
        "distance_to_edge": None,
        "movement": movement,
        "source": "directional" if movement is not None else None,
    }


def default_map_layout(mode, width=1920, height=1080):
    """Legal field, and brawlball goals, in the 1920x1080 reference frame."""
    field = (160, 90, width - 160, height - 140)
    center = ((field[0] + field[2]) / 2, (field[1] + field[3]) / 2)
    goals = []
    normalized = normalize_gamemode(mode)
    if normalized in {"brawlball", "brawlball_5v5"}:
        mouth = height * 0.22
        mid = height / 2
        goals = [
            (40, mid - mouth, field[0], mid + mouth),
            (field[2], mid - mouth, width - 40, mid + mouth),
        ]
    return {"field": field, "goals": goals, "cages": list(goals), "center": center}


def scale_layout(layout, width_ratio=1, height_ratio=1):
    def scale_rect(rect):
        x1, y1, x2, y2 = rect
        return (x1 * width_ratio, y1 * height_ratio, x2 * width_ratio, y2 * height_ratio)

    center = layout["center"]
    return {
        "field": scale_rect(layout["field"]),
        "goals": [scale_rect(goal) for goal in layout.get("goals") or []],
        "cages": [scale_rect(cage) for cage in layout.get("cages") or []],
        "center": (center[0] * width_ratio, center[1] * height_ratio),
    }


def legal_regions(mode, field, goals=None):
    regions = []
    if field is not None:
        regions.append(_normalize_rect(field))
    if normalize_gamemode(mode) in {"brawlball", "brawlball_5v5"}:
        regions.extend(_normalize_rect(goal) for goal in (goals or []))
    return regions


def clamp_point_to_regions(point, regions):
    if point is None:
        return None, False
    if not regions:
        return (float(point[0]), float(point[1])), False
    px, py = float(point[0]), float(point[1])
    for region in regions:
        if point_in_rect((px, py), region):
            return (px, py), False

    def project(region):
        x1, y1, x2, y2 = region
        return (min(max(px, x1), x2), min(max(py, y1), y2))

    projected = min(regions, key=lambda region: math.hypot(project(region)[0] - px, project(region)[1] - py))
    clamped = project(projected)
    return clamped, clamped != (px, py)


def clamp_movement_to_bounds(player_pos, movement, regions, step=120):
    """Shorten a joystick vector so its step does not leave the legal region."""
    if movement is None:
        return None
    mx, my = float(movement[0]), float(movement[1])
    length = math.hypot(mx, my)
    if length < 1 or not regions:
        return (mx, my)

    inside, outside = clamp_point_to_regions(player_pos, regions)
    if outside:
        return normalize_move(inside[0] - player_pos[0], inside[1] - player_pos[1], length)

    ux, uy = mx / length, my / length
    destination = (player_pos[0] + ux * step, player_pos[1] + uy * step)
    clamped, changed = clamp_point_to_regions(destination, regions)
    if not changed:
        return (mx, my)
    inward = math.hypot(clamped[0] - player_pos[0], clamped[1] - player_pos[1])
    if inward < 4:
        return (0.0, 0.0)
    scale = min(1.0, inward / float(step))
    return (mx * scale, my * scale)


def constrain_aim_and_movement(player_pos, aim_target, movement, mode, field, goals=None, step=120):
    regions = legal_regions(mode, field, goals)
    aim, aim_clamped = clamp_point_to_regions(aim_target, regions)
    clamped_movement = clamp_movement_to_bounds(player_pos, movement, regions, step)
    return {
        "aim": aim,
        "aim_clamped": aim_clamped,
        "movement": clamped_movement,
        "regions": regions,
    }


def _entity_position(entity):
    if entity is None:
        return None
    if len(entity) >= 4 and all(isinstance(entity[index], (int, float)) for index in range(4)):
        return ((entity[0] + entity[2]) / 2, (entity[1] + entity[3]) / 2)
    if len(entity) >= 2:
        return (float(entity[0]), float(entity[1]))
    return None


def closest_teammate(player_pos, teammates):
    best = None
    best_distance = float("inf")
    for teammate in teammates or []:
        position = _entity_position(teammate)
        if position is None:
            continue
        distance = math.hypot(position[0] - player_pos[0], position[1] - player_pos[1])
        if distance < best_distance:
            best = position
            best_distance = distance
    if best is None:
        return None, None
    return best, best_distance


def teammate_focus_movement(player_pos, teammates, enabled=True, glue_distance=70, radius=JOYSTICK_RADIUS):
    """Bias toward the nearest showdown teammate, and stop wandering once glued."""
    if not enabled:
        return {"active": False, "movement": None, "distance": None, "glued": False, "teammate": None}
    teammate, distance = closest_teammate(player_pos, teammates)
    if teammate is None:
        return {"active": False, "movement": None, "distance": None, "glued": False, "teammate": None}
    glued = distance <= glue_distance
    if glued:
        movement = (0.0, 0.0)
    else:
        movement = normalize_move(teammate[0] - player_pos[0], teammate[1] - player_pos[1], radius)
    return {
        "active": True,
        "movement": movement,
        "distance": distance,
        "glued": glued,
        "teammate": teammate,
    }


def apply_teammate_bias(playstyle_movement, player_pos, teammates, enabled=True, glue_distance=70, weight=0.7, radius=JOYSTICK_RADIUS):
    focus = teammate_focus_movement(player_pos, teammates, enabled, glue_distance, radius)
    if not focus["active"]:
        return playstyle_movement, focus
    if playstyle_movement is None:
        return focus["movement"], focus
    px, py = float(playstyle_movement[0]), float(playstyle_movement[1])
    if focus["glued"]:
        keep = 1 - weight
        return (px * keep, py * keep), focus
    fx, fy = focus["movement"]
    blend = 1 - weight
    return (px * blend + fx * weight, py * blend + fy * weight), focus


def apply_safety_policy(
    player_pos,
    playstyle_movement=None,
    aim_target=None,
    mode="",
    cages=None,
    map_center=None,
    field=None,
    goals=None,
    walls=None,
    safe_center=None,
    safe_radius=None,
    directional_gas=None,
    teammates=None,
    avoid_gas=True,
    cage_escape=True,
    boundary_awareness=True,
    teammate_focus=True,
    edge_margin=80,
    glue_distance=70,
    teammate_weight=0.7,
    step=120,
):
    """Apply cage escape, gas avoidance, teammate bias, then map limits.

    Cage escape wins over gas, and gas wins over teammate bias. Boundary
    clamping always runs for knockout and brawlball.
    """
    normalized = normalize_gamemode(mode)
    reasons = []
    movement = playstyle_movement
    cage_result = {"trapped": False, "path": [], "movement": (0.0, 0.0), "cage": None}

    if cage_escape and normalized in {"brawlball", "brawlball_5v5"}:
        if map_center is None and field is not None:
            rect = _normalize_rect(field)
            map_center = ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
        if cages and map_center is not None:
            cage_result = detect_brawlball_cage(player_pos, cages, map_center)
        if not cage_result["trapped"] and walls and map_center is not None:
            wall_result = detect_wall_cage(player_pos, walls, map_center)
            if wall_result["trapped"]:
                cage_result = wall_result
        if cage_result["trapped"]:
            movement = cage_result["movement"]
            reasons.append("brawlball_cage")

    gas_result = {"active": False, "movement": None}
    focus_result = {"active": False, "movement": None}
    if "brawlball_cage" not in reasons:
        if avoid_gas:
            gas_result = resolve_gas_movement(
                player_pos,
                safe_center=safe_center,
                safe_radius=safe_radius,
                directional=directional_gas,
                edge_margin=edge_margin,
            )
        if gas_result["active"]:
            movement = gas_result["movement"]
            reasons.append("poison_gas")
        elif teammate_focus and is_showdown_mode(normalized):
            movement, focus_result = apply_teammate_bias(
                movement,
                player_pos,
                teammates,
                enabled=True,
                glue_distance=glue_distance,
                weight=teammate_weight,
            )
            if focus_result["active"]:
                reasons.append("teammate_focus")

    aim = aim_target
    aim_clamped = False
    if boundary_awareness and normalized in BOUNDARY_MODES and field is not None:
        constrained = constrain_aim_and_movement(player_pos, aim_target, movement, normalized, field, goals, step)
        movement = constrained["movement"]
        aim = constrained["aim"]
        aim_clamped = constrained["aim_clamped"]
        reasons.append("map_bounds")

    return {
        "movement": movement,
        "aim": aim,
        "aim_clamped": aim_clamped,
        "reasons": reasons,
        "cage": cage_result,
        "gas": gas_result,
        "teammate_focus": focus_result,
    }


def latch_underdog(already_seen, screen_detected):
    """Stay true for the rest of the end screen once the underdog banner appears."""
    return bool(already_seen) or bool(screen_detected)


def _range_value(ranges, trophies):
    for max_trophies, value in ranges:
        if float(trophies) <= float(max_trophies):
            return value
    raise ValueError("Current trophies exceed all defined ranges")


def _result_name(result):
    if hasattr(result, "value"):
        return str(result.value)
    return str(result or "")


CLASSIC_WIN_RANGES = [(1999, 10), (2499, 8), (2799, 6), (2999, 4), (3099, 2), (float("inf"), 1)]
CLASSIC_LOSE_RANGES = [
    (49, 0), (299, 1), (599, 2), (799, 3), (999, 4), (1099, 5), (1199, 6), (1299, 7),
    (1499, 8), (1799, 9), (3999, 10), (float("inf"), 15),
]
SHOWDOWN_TRIO_RANGES = [
    (49, (11, 5, 5, 5)),
    (99, (11, 5, 4, -1)),
    (199, (11, 5, 3, -1)),
    (299, (11, 5, 2, -1)),
    (499, (11, 5, 2, -2)),
    (599, (11, 5, 1, -2)),
    (799, (11, 5, 1, -3)),
    (999, (11, 5, 1, -4)),
    (1099, (11, 5, 0, -6)),
    (1199, (11, 5, 0, -7)),
    (1299, (11, 5, 0, -8)),
    (1499, (11, 5, 0, -9)),
    (1799, (11, 5, -5, -10)),
    (1999, (11, 5, -5, -11)),
    (2199, (9, 4, -5, -11)),
    (float("inf"), (9, 4, -5, -11)),
]


def _streak_gain(trophies, win_streak):
    if trophies >= 2000:
        return 0
    return min(win_streak - 1, 10)


def compute_trophy_update(
    result,
    trophies,
    win_streak,
    underdog,
    place=None,
    multiplier=1,
    win_ranges=None,
    lose_ranges=None,
    showdown_ranges=None,
):
    """Mirror match accounting, including the underdog banner.

    Underdog below 2000 trophies adds the classic bonus and keeps the win
    streak after a loss. At 2000 or above the banner does not change the result.
    Showdown place payouts stay on the showdown table; the banner still
    controls whether a showdown loss clears the streak.
    """
    trophies = 0 if trophies is None else trophies
    streak = 0 if win_streak is None else win_streak
    win_ranges = win_ranges or CLASSIC_WIN_RANGES
    lose_ranges = lose_ranges or CLASSIC_LOSE_RANGES
    showdown_ranges = showdown_ranges or SHOWDOWN_TRIO_RANGES
    underdog_applied = bool(underdog) and trophies < 2000
    result_name = _result_name(result)

    def showdown_delta(current_streak):
        deltas = _range_value(showdown_ranges, trophies)
        bonus = _streak_gain(trophies, current_streak) if place < 2 else 0
        return deltas[place] * multiplier + bonus

    if result_name == "victory":
        streak += 1
        if place is not None:
            delta = showdown_delta(streak)
        else:
            gain = _range_value(win_ranges, trophies)
            delta = gain * multiplier + _streak_gain(trophies, streak) + (5 if underdog_applied else 0)
    elif result_name == "defeat":
        if not underdog_applied:
            streak = 0
        if place is not None:
            delta = showdown_delta(streak)
        else:
            loss = _range_value(lose_ranges, trophies)
            delta = -(loss - (3 if underdog_applied else 0))
    elif result_name == "draw":
        if place is not None:
            delta = showdown_delta(streak)
        else:
            delta = 4 if underdog_applied else 0
    else:
        delta = 0

    if trophies >= 1000 and trophies + delta < 1000:
        new_trophies = 1000
    elif trophies >= 2000 and trophies + delta < 2000:
        new_trophies = 2000
    else:
        new_trophies = trophies + delta
    return {
        "trophy_delta": delta,
        "new_trophies": new_trophies,
        "new_win_streak": streak,
        "underdog_applied": underdog_applied,
    }


def classify_lobby_mode(labels):
    joined = " ".join(_text_of(item) for item in (labels or [])).lower()
    if not joined.strip():
        return None
    for needle, mode in _LOBBY_MODE_PATTERNS:
        if re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", joined):
            return mode
    return None


def _label_center(item):
    bbox = item.get("bbox") if isinstance(item, dict) else None
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    return None


def _mode_label_matches(text, mode):
    normalized = normalize_gamemode(text)
    if normalized == mode:
        return True
    compact = " ".join(str(text).strip().lower().replace("_", " ").split())
    aliases = {
        "brawlball": ("brawl ball", "brawlball"),
        "brawlball_5v5": ("brawl ball 5v5", "brawlball 5v5"),
        "solo_showdown": ("solo showdown",),
        "trio_showdown": ("trio showdown",),
        "duo_showdown": ("duo showdown",),
        "knockout": ("knockout",),
        "ranked": ("ranked",),
        "gem_grab": ("gem grab",),
        "hot_zone": ("hot zone",),
    }
    return compact in aliases.get(mode, ())


def gamemode_recovery_action(detected_mode, configured_mode, menu_text=None, enabled=True):
    """Ask for a switch when the lobby has fallen into Solo Showdown or Ranked."""
    configured = normalize_gamemode(configured_mode)
    detected = normalize_gamemode(detected_mode) if detected_mode else None
    if not detected:
        detected = classify_lobby_mode(menu_text)
    base = {"needed": False, "detected": detected, "configured": configured, "click": None, "action": None}
    if not enabled:
        base["reason"] = "disabled"
        return base
    if not detected or detected not in UNWANTED_LOBBY_MODES or not configured or detected == configured:
        base["reason"] = "mode_ok"
        return base
    click = None
    for item in menu_text or []:
        if isinstance(item, dict) and _mode_label_matches(item.get("text", ""), configured):
            click = _label_center(item)
            if click is not None:
                break
    return {
        "needed": True,
        "action": "switch_gamemode",
        "detected": detected,
        "configured": configured,
        "click": click,
        "reason": "unwanted_lobby_mode",
    }


def buffie_detection_from_labels(labels):
    texts = []
    prize_center = None
    for item in labels or []:
        text = _text_of(item)
        texts.append(text)
        if prize_center is None and isinstance(item, dict):
            lowered = text.lower()
            if any(word in lowered for word in ("prize", "gift")):
                prize_center = _label_center(item)
    joined = " ".join(texts).lower()
    visible = "buffie" in joined or "claw" in joined
    return {
        "buffie_visible": visible,
        "claw_ready": "claw" in joined,
        "prize_aligned": any(token in joined for token in ("aligned", "grab now")),
        "reward_visible": any(token in joined for token in ("reward", "collect", "tap to continue")),
        "prize_center": prize_center,
        "screen": "buffie" if visible else None,
    }


class BuffieController:
    """Claw-machine sequence driven by UI detection flags.

    idle -> open (click the machine) -> position (move the claw) ->
    release -> collect -> done.
    """

    def __init__(self):
        self.state = "idle"

    def step(self, ui=None):
        ui = ui or {}
        visible = bool(ui.get("buffie_visible") or ui.get("screen") in {"buffie", "claw", "event_drop"})
        claw_ready = bool(ui.get("claw_ready"))
        aligned = bool(ui.get("prize_aligned") or ui.get("claw_over_prize"))
        reward = bool(ui.get("reward_visible") or ui.get("reward_open"))

        if reward:
            self.state = "collect"
            return self._action("collect", "proceed")
        if self.state == "collect":
            self.state = "done"
            return self._action("none", done=True)
        if self.state in {"release", "await_reward"}:
            self.state = "await_reward"
            return self._action("wait")
        if self.state == "done" and not visible:
            return self._action("none", done=True)
        if self.state == "done" and visible:
            self.state = "idle"
        if not visible and self.state == "idle":
            return self._action("none")
        if visible and not claw_ready and self.state in {"idle", "open"}:
            self.state = "open"
            return self._action("click_machine", "buffie_machine")
        if claw_ready and not aligned:
            self.state = "position"
            return self._action("move_claw", ui.get("prize_center"))
        if claw_ready and aligned:
            self.state = "release"
            return self._action("release_claw", "attack")
        if visible:
            self.state = "open"
            return self._action("click_machine", "buffie_machine")
        return self._action("wait")

    def _action(self, action, target=None, done=False):
        return {
            "state": self.state,
            "action": action,
            "target": target,
            "done": bool(done or self.state == "done"),
        }
