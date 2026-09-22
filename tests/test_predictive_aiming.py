"""Deterministic tests for predictive aiming, delay, capture, and aimbot playstyles."""

import ast
import json
import math
import random
import time
import unittest
from pathlib import Path

from aiming import (
    AIMBOT_PLAYSTYLES,
    DynamicDelay,
    MotionTracker,
    aim_at_target,
    calculate_lead,
    stick_to_world_velocity,
    trajectory_kind,
)
from aiming.trajectory import solve_arc, solve_nani_attack, solve_nani_super
from capture import create_capture_backend, normalize_backend_name
from capture.base import CaptureBackend

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "lead_cases.json"
PLAYSTYLES = ROOT / "playstyles"


def _close(actual, expected, tol=1e-6):
    return abs(actual - expected) <= tol


class LeadFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]

    def test_fixture_file_covers_required_shapes(self):
        ids = {case["id"] for case in self.cases}
        self.assertTrue({"stationary", "moving_toward", "moving_away", "too_fast", "out_of_range"} <= ids)

    def test_fixture_cases(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                solution = calculate_lead(
                    case["shooter"],
                    case["target"],
                    target_velocity=case["target_velocity"],
                    projectile_speed=case["projectile_speed"],
                    shooter_speed=case.get("shooter_speed", 0.0),
                    projectile_radius=case.get("projectile_radius", 0.0),
                    max_range=case.get("max_range"),
                )
                self.assertEqual(solution["feasible"], case["feasible"])
                self.assertEqual(solution["reason"], case["reason"])
                if case["feasible"]:
                    self.assertIsNotNone(solution["aim_point"])
                    self.assertTrue(_close(solution["time_to_impact"], case["time"]))
                    self.assertTrue(_close(solution["aim_point"][0], case["aim"][0]))
                    self.assertTrue(_close(solution["aim_point"][1], case["aim"][1]))
                    self.assertTrue(solution["in_range"])
                else:
                    self.assertIsNone(solution["aim_point"])
                    self.assertFalse(solution["in_range"])

    def test_toward_aim_is_closer_and_away_aim_is_farther(self):
        toward = calculate_lead((0, 0), (100, 0), target_velocity=(-40, 0), projectile_speed=100, projectile_radius=0)
        away = calculate_lead((0, 0), (100, 0), target_velocity=(40, 0), projectile_speed=100, projectile_radius=0)
        self.assertLess(toward["aim_point"][0], 100)
        self.assertGreater(away["aim_point"][0], 100)

    def test_negative_radius_matches_zero_radius(self):
        plain = calculate_lead((0, 0), (200, 0), projectile_speed=100, projectile_radius=0)
        negative = calculate_lead((0, 0), (200, 0), projectile_speed=100, projectile_radius=-25)
        self.assertTrue(plain["feasible"] and negative["feasible"])
        self.assertTrue(_close(plain["time_to_impact"], negative["time_to_impact"]))

    def test_invalid_inputs_do_not_raise(self):
        cases = [
            calculate_lead(None, (1, 0), projectile_speed=10),
            calculate_lead((0, 0), (float("nan"), 0), projectile_speed=10),
            calculate_lead((0, 0), (10, 0), target_velocity=(float("inf"), 0), projectile_speed=10),
            calculate_lead((0, 0), (10, 0), projectile_speed=-5),
            calculate_lead((0, 0), "nope", projectile_speed=10),
            calculate_lead((0, 0), (10, 0), shooter_velocity=("bad", 0), projectile_speed=10),
        ]
        reasons = [case["reason"] for case in cases]
        self.assertEqual(reasons[:3], ["invalid", "invalid", "invalid"])
        self.assertEqual(reasons[3], "impossible_speed")
        self.assertEqual(reasons[4:], ["invalid", "invalid"])
        for case in cases:
            self.assertFalse(case["feasible"])
            self.assertIsNone(case["aim_point"])

    def test_overlapping_target_is_immediate(self):
        solution = calculate_lead((0, 0), (5, 0), projectile_speed=100, projectile_radius=10)
        self.assertTrue(solution["feasible"])
        self.assertEqual(solution["time_to_impact"], 0.0)
        self.assertEqual(solution["aim_point"], (5.0, 0.0))

    def test_latency_pushes_a_crossing_target_further(self):
        base = calculate_lead((0, 0), (0, 100), target_velocity=(40, 0), projectile_speed=10000, projectile_radius=0)
        delayed = calculate_lead(
            (0, 0),
            (0, 100),
            target_velocity=(40, 0),
            projectile_speed=10000,
            projectile_radius=0,
            latency=0.5,
        )
        self.assertGreater(delayed["aim_point"][0], base["aim_point"][0])

    def test_inherited_shooter_velocity_changes_fire_direction(self):
        solution = calculate_lead(
            (0, 0),
            (100, 0),
            target_velocity=(0, 0),
            projectile_speed=100,
            shooter_velocity=(0, 30),
            projectile_radius=0,
        )
        self.assertTrue(solution["feasible"])
        self.assertLess(solution["aim_vector"][1], 0.0)
        self.assertTrue(_close(solution["aim_point"][0], 100.0))
        self.assertTrue(_close(solution["aim_point"][1], 0.0))


class TrajectoryTests(unittest.TestCase):
    def test_kind_selection(self):
        self.assertEqual(trajectory_kind("shelly"), "linear")
        self.assertEqual(trajectory_kind("barley"), "arc")
        self.assertEqual(trajectory_kind("nani", "attack"), "nani_attack")
        self.assertEqual(trajectory_kind("nani", "super"), "nani_super")
        self.assertEqual(trajectory_kind("shelly", brawler_info={"attack_trajectory": "arc"}), "arc")

    def test_arc_has_height_and_longer_flight_than_a_straight_shot(self):
        straight = calculate_lead((0, 0), (200, 0), projectile_speed=200, projectile_radius=0, max_range=500)
        arc = solve_arc((0, 0), (200, 0), projectile_speed=200, projectile_radius=0, max_range=500)
        self.assertEqual(arc["trajectory"], "arc")
        self.assertGreater(arc["time_to_impact"], straight["time_to_impact"])
        heights = [point[2] for point in arc["path"]]
        self.assertTrue(_close(heights[0], 0.0))
        self.assertTrue(_close(heights[-1], 0.0))
        self.assertGreater(max(heights), 0.0)

    def test_thrower_dispatch_uses_arc(self):
        solution = aim_at_target(
            (0, 0),
            (180, 0),
            projectile_speed=220,
            projectile_radius=10,
            max_range=400,
            brawler="dynamike",
        )
        self.assertEqual(solution["trajectory"], "arc")
        self.assertGreater(max(point[2] for point in solution["path"]), 0.0)

    def test_nani_attack_has_three_returning_rays(self):
        solution = solve_nani_attack(
            (0, 0),
            (150, 0),
            projectile_speed=200,
            projectile_radius=0,
            max_range=400,
        )
        self.assertTrue(solution["feasible"])
        self.assertEqual(solution["trajectory"], "nani_attack")
        self.assertEqual(len(solution["paths"]), 3)
        for path in solution["paths"]:
            self.assertGreater(len(path), 4)
            self.assertTrue(_close(path[-1][0], 0.0, tol=1e-6))
            self.assertTrue(_close(path[-1][1], 0.0, tol=1e-6))
        center = solution["paths"][1][2]
        side = solution["paths"][2][2]
        center_angle = math.atan2(center[1], center[0])
        side_angle = math.atan2(side[1], side[0])
        self.assertTrue(_close(abs(side_angle - center_angle), math.radians(12), tol=1e-6))

    def test_nani_super_curves_and_zero_range_still_flies(self):
        moving = solve_nani_super(
            (0, 0),
            (80, 80),
            target_velocity=(60, -30),
            projectile_speed=180,
            projectile_radius=12,
            max_range=None,
        )
        self.assertEqual(moving["trajectory"], "nani_super")
        self.assertGreater(len(moving["path"]), 3)
        start, mid, end = moving["path"][0], moving["path"][len(moving["path"]) // 2], moving["path"][-1]
        cross = (mid[0] - start[0]) * (end[1] - start[1]) - (mid[1] - start[1]) * (end[0] - start[0])
        self.assertGreater(abs(cross), 1.0)

        info = json.loads((ROOT / "cfg" / "brawlers_info.json").read_text(encoding="utf-8"))["nani"]
        self.assertEqual(info["super_range"], 0)
        guided = aim_at_target((0, 0), (120, 40), brawler="nani", brawler_info=info, skill="super", projectile_speed=500, projectile_radius=20)
        self.assertEqual(guided["trajectory"], "nani_super")
        self.assertTrue(guided["feasible"])

    def test_impossible_special_paths_do_not_raise(self):
        arc = aim_at_target((0, 0), (100, 0), target_velocity=(400, 0), projectile_speed=50, brawler="tick", max_range=1000)
        nani = aim_at_target((0, 0), (100, 0), projectile_speed=0, brawler="nani")
        peep = aim_at_target(
            (0, 0),
            (50, 0),
            target_velocity=(300, 0),
            projectile_speed=40,
            projectile_radius=1,
            brawler="nani",
            skill="super",
            max_range=80,
        )
        self.assertFalse(arc["feasible"])
        self.assertEqual(arc["trajectory"], "arc")
        self.assertFalse(nani["feasible"])
        self.assertEqual(nani["trajectory"], "nani_attack")
        self.assertFalse(peep["feasible"])
        self.assertIn(peep["reason"], {"impossible_speed", "out_of_range"})

    def test_linear_path_stays_on_the_ground_line(self):
        solution = aim_at_target((0, 0), (90, 0), projectile_speed=90, projectile_radius=0, brawler="colt", max_range=200)
        self.assertEqual(solution["trajectory"], "linear")
        for point in solution["path"]:
            self.assertTrue(_close(point[1], 0.0))
            self.assertTrue(_close(point[2], 0.0))


class DelayAndTrackingTests(unittest.TestCase):
    def test_delay_uses_default_until_measured(self):
        delay = DynamicDelay(default=0.1, maximum=0.35, smoothing=1.0)
        self.assertFalse(delay.has_measurement())
        self.assertTrue(_close(delay.current(), 0.1))
        self.assertTrue(_close(delay.observe(None, None), 0.1))
        self.assertTrue(_close(delay.observe("nope", float("nan")), 0.1))
        self.assertFalse(delay.has_measurement())

    def test_delay_follows_capture_and_input_and_clamps(self):
        delay = DynamicDelay(default=0.1, maximum=0.35, smoothing=0.5)
        self.assertTrue(_close(delay.observe(0.2, None), 0.2))
        self.assertTrue(delay.has_measurement())
        self.assertTrue(_close(delay.observe(0.0, None), 0.1))
        self.assertTrue(_close(delay.observe(0.05, 0.05), 0.1))
        clamped = DynamicDelay(default=0.1, maximum=0.35, smoothing=1.0)
        self.assertTrue(_close(clamped.observe(2.0, 1.0), 0.35))
        clamped.reset()
        self.assertFalse(clamped.has_measurement())
        self.assertTrue(_close(clamped.current(), 0.1))

    def test_motion_tracker_velocity_is_deterministic(self):
        tracker = MotionTracker()
        self.assertEqual(tracker.velocity_near((0, 0)), (0.0, 0.0))
        tracker.update([(0, 0)], 1.0)
        self.assertEqual(tracker.velocity_near((0, 0)), (0.0, 0.0))
        tracker.update([(30, -15)], 1.5)
        velocity = tracker.velocity_near((30, -15))
        self.assertTrue(_close(velocity[0], 60.0))
        self.assertTrue(_close(velocity[1], -30.0))
        self.assertEqual(tracker.velocity_near((500, 500)), (0.0, 0.0))

    def test_stick_velocity_caps_at_brawler_speed(self):
        self.assertEqual(stick_to_world_velocity("", 720), (0.0, 0.0))
        self.assertEqual(stick_to_world_velocity((0, 0), 720), (0.0, 0.0))
        full = stick_to_world_velocity((100, 0), 720)
        self.assertTrue(_close(full[0], 720.0))
        self.assertTrue(_close(full[1], 0.0))
        half = stick_to_world_velocity((0, -50), 720)
        self.assertTrue(_close(half[1], -360.0))


class CaptureBackendTests(unittest.TestCase):
    def test_mumu_is_selectable_and_reads_fixture_frames(self):
        frames = [
            ("frame-a", 10.0),
            ("frame-b", 12.5),
        ]

        def source():
            return frames.pop(0)

        for name in ("mumu", "MuMu Screen Capture", "mumu_screen_capture"):
            self.assertEqual(normalize_backend_name(name), "mumu")

        backend = create_capture_backend("MuMu Screen Capture", source=source, display_index=2)
        self.assertIsInstance(backend, CaptureBackend)
        self.assertEqual(backend.name, "mumu")
        self.assertEqual(backend.display_index, 2)
        backend.start()
        frame, timestamp = backend.grab()
        self.assertEqual((frame, timestamp), ("frame-a", 10.0))
        self.assertTrue(_close(backend.latency(now=12.5), 0.0))
        backend.close()

    def test_mumu_without_a_source_returns_no_frame(self):
        backend = create_capture_backend("mumu")
        backend.start()
        self.assertEqual(backend.grab(), (None, 0.0))
        self.assertIsNone(backend.latency(now=1.0))

    def test_scrcpy_backend_and_unknown_name(self):
        pushed = {}

        def source():
            return pushed.get("frame"), pushed.get("time", 0.0)

        backend = create_capture_backend("scrcpy", source=source)
        self.assertEqual(backend.name, "scrcpy")
        self.assertEqual(backend.grab(), (None, 0.0))
        pushed["frame"] = "pixels"
        pushed["time"] = 4.0
        self.assertEqual(backend.grab(), ("pixels", 4.0))
        with self.assertRaises(ValueError):
            create_capture_backend("not-a-backend")

    def test_source_errors_do_not_escape(self):
        def broken():
            raise RuntimeError("display missing")

        backend = create_capture_backend("mumu", source=broken)
        self.assertEqual(backend.grab(), (None, 0.0))


class PlaystyleRegistryTests(unittest.TestCase):
    def test_three_aimbot_playstyles_are_registered(self):
        names = [item["name"] for item in AIMBOT_PLAYSTYLES]
        self.assertEqual(
            names,
            [
                "Follower + Aimbot",
                "Slarckvul's Aggressive + Aimbot",
                "Showdown Survivor + Aimbot",
            ],
        )
        for item in AIMBOT_PLAYSTYLES:
            path = PLAYSTYLES / item["filename"]
            self.assertTrue(path.is_file(), item["filename"])
            metadata = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(metadata["name"], item["name"])
            script = "\n".join(path.read_text(encoding="utf-8").splitlines()[1:])
            ast.parse(script)
            self.assertNotIn("import ", script)
            self.assertIn("predict_aim", script)
            self.assertIn("aim_attack", script)

    def test_scripts_aim_when_an_enemy_is_present(self):
        for item in AIMBOT_PLAYSTYLES:
            with self.subTest(playstyle=item["name"]):
                calls = []
                namespace = _script_namespace(calls)
                script = "\n".join((PLAYSTYLES / item["filename"]).read_text(encoding="utf-8").splitlines()[1:])
                exec(script, namespace, namespace)
                self.assertIn("aim_attack", calls)
                self.assertTrue(any(call[0] == "predict_aim" and call[1] == "attack" for call in calls))
                self.assertIsNotNone(namespace.get("movement"))


def _script_namespace(calls):
    player = [0.0, 0.0, 20.0, 20.0]
    enemy = [200.0, 0.0, 220.0, 40.0]

    def get_entity_pos(box):
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    def predict_aim(target, target_velocity=None, skill_type="attack"):
        calls.append(("predict_aim", skill_type, target))
        return {
            "feasible": True,
            "in_range": True,
            "trajectory": "linear",
            "aim_point": target,
        }

    def aim_attack(solution):
        calls.append("aim_attack")
        return True

    def aim_super(solution):
        calls.append("aim_super")
        return True

    enemy_pos = get_entity_pos(enemy)
    player_pos = get_entity_pos(player)
    distance = math.hypot(enemy_pos[0] - player_pos[0], enemy_pos[1] - player_pos[1])
    return {
        "math": math,
        "random": random,
        "time": time,
        "brawler": "shelly",
        "current_brawler": "shelly",
        "brawlers_info": {
            "shelly": {
                "safe_range": 100,
                "attack_range": 500,
                "super_range": 500,
                "super_type": "damage",
                "hold_attack": 0,
            }
        },
        "player_data": player,
        "enemy_data": [enemy],
        "teammate_data": [],
        "walls": [],
        "persistent_data": {"time_since_holding_attack": None},
        "is_gadget_ready": False,
        "is_hypercharge_ready": False,
        "is_super_ready": False,
        "debug": False,
        "JOYSTICK_RADIUS": 100,
        "TILE_SIZE": 60,
        "width_ratio": 1,
        "height_ratio": 1,
        "seconds_to_hold_attack_after_reaching_max": 1.5,
        "get_entity_pos": get_entity_pos,
        "get_brawler_range": lambda _name: (100, 500, 500),
        "is_there_enemy": lambda enemies: bool(enemies),
        "find_closest_enemy": lambda enemies, player_coords, walls, skill: (enemy_pos, distance),
        "find_closest_teammate": lambda teammates, player_coords, walls: (None, None),
        "is_path_blocked": lambda *_args, **_kwargs: False,
        "is_enemy_hittable": lambda *_args, **_kwargs: True,
        "is_there_poison_gas": lambda *_args, **_kwargs: {"up": 0, "down": 0, "left": 0, "right": 0},
        "must_brawler_hold_attack": lambda *_args, **_kwargs: False,
        "attack": lambda **_kwargs: calls.append("attack"),
        "use_super": lambda: calls.append("super"),
        "use_gadget": lambda: calls.append("gadget"),
        "use_hypercharge": lambda: calls.append("hypercharge"),
        "predict_aim": predict_aim,
        "aim_attack": aim_attack,
        "aim_super": aim_super,
        "movement": None,
    }


if __name__ == "__main__":
    unittest.main()
