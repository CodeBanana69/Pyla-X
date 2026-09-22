import math
import unittest
from unittest.mock import MagicMock, patch

from gameplay_safety import (
    BuffieController,
    apply_safety_policy,
    apply_teammate_bias,
    avoid_directional_gas,
    buffie_detection_from_labels,
    clamp_point_to_regions,
    compute_trophy_update,
    constrain_aim_and_movement,
    default_map_layout,
    detect_brawlball_cage,
    detect_wall_cage,
    gamemode_recovery_action,
    latch_underdog,
    legal_regions,
    poison_gas_avoidance,
    teammate_focus_movement,
)
from lobby_automation import LobbyAutomation
from lobby_ocr import screen_from_grounding, validate_screen_read
from trophy_observer import MatchResult, TrophyObserver


def _layout(mode):
    return default_map_layout(mode)


class CageDetectionTests(unittest.TestCase):
    def test_trapped_player_paths_to_map_center(self):
        center = (960, 540)
        cage = (0, 400, 200, 680)
        decision = detect_brawlball_cage((80, 540), [cage], center)

        self.assertTrue(decision["trapped"])
        self.assertEqual(decision["path"][-1], center)
        self.assertEqual(decision["path"][0][0], 200)
        self.assertGreater(decision["movement"][0], 0)
        self.assertAlmostEqual(decision["movement"][1], 0, places=5)
        self.assertAlmostEqual(math.hypot(*decision["movement"]), 100, places=5)

    def test_player_outside_a_cage_is_not_trapped(self):
        decision = detect_brawlball_cage((500, 540), [(0, 400, 200, 680)], (960, 540))
        self.assertFalse(decision["trapped"])
        self.assertEqual(decision["path"], [])
        self.assertEqual(decision["movement"], (0.0, 0.0))

    def test_wall_pocket_opening_toward_center_is_a_cage(self):
        player = (120, 200)
        center = (520, 200)
        walls = [
            (0, 160, 40, 240),
            (80, 60, 160, 110),
            (80, 280, 160, 330),
        ]
        decision = detect_wall_cage(player, walls, center)
        self.assertTrue(decision["trapped"])
        self.assertGreater(decision["movement"][0], 0)
        self.assertEqual(decision["path"][-1], center)
        self.assertGreater(decision["opening"][0], player[0])

    def test_open_field_walls_are_not_a_cage(self):
        decision = detect_wall_cage((120, 200), [(0, 160, 40, 240)], (520, 200))
        self.assertFalse(decision["trapped"])


class PoisonGasTests(unittest.TestCase):
    def test_player_outside_the_safe_circle_steers_into_it(self):
        decision = poison_gas_avoidance((0, 0), (100, 0), 50)
        self.assertTrue(decision["active"])
        self.assertTrue(decision["outside_safe_zone"])
        self.assertGreater(decision["movement"][0], 0)
        self.assertLess(decision["distance_to_edge"], 0)

    def test_player_near_the_shrinking_edge_moves_away_from_it(self):
        decision = poison_gas_avoidance((80, 0), (100, 0), 50, edge_margin=80)
        self.assertTrue(decision["active"])
        self.assertFalse(decision["outside_safe_zone"])
        self.assertTrue(decision["near_edge"])
        self.assertGreater(decision["movement"][0], 0)

    def test_player_deep_inside_the_safe_zone_keeps_course(self):
        decision = poison_gas_avoidance((90, 0), (100, 0), 200, edge_margin=40)
        self.assertFalse(decision["active"])
        self.assertIsNone(decision["movement"])

    def test_directional_gas_moves_to_the_opposite_side(self):
        movement = avoid_directional_gas({"up": 0, "down": 0, "left": 12, "right": 1})
        self.assertGreater(movement[0], 0)
        self.assertEqual(movement[1], 0)

        upward_gas = avoid_directional_gas({"up": 9, "down": 1, "left": 0, "right": 0})
        self.assertGreater(upward_gas[1], 0)
        self.assertEqual(upward_gas[0], 0)


class MapBoundaryTests(unittest.TestCase):
    def test_knockout_aim_cannot_leave_the_field_or_enter_a_goal(self):
        layout = _layout("knockout")
        regions = legal_regions("knockout", layout["field"], layout["goals"])
        self.assertEqual(len(regions), 1)

        aim, clamped = clamp_point_to_regions((20, 500), regions)
        self.assertTrue(clamped)
        self.assertEqual(aim[0], layout["field"][0])

        inside, clamped_inside = clamp_point_to_regions((960, 500), regions)
        self.assertFalse(clamped_inside)
        self.assertEqual(inside, (960, 500))

    def test_brawlball_keeps_goals_and_rejects_the_area_beyond_them(self):
        layout = _layout("brawlball")
        regions = legal_regions("brawlball", layout["field"], layout["goals"])
        goal_point = (50, 540)
        kept, clamped = clamp_point_to_regions(goal_point, regions)
        self.assertFalse(clamped)
        self.assertEqual(kept, goal_point)

        beyond, beyond_clamped = clamp_point_to_regions((10, 540), regions)
        self.assertTrue(beyond_clamped)
        self.assertGreaterEqual(beyond[0], layout["goals"][0][0])
        self.assertLess(beyond[0], layout["field"][0])

    def test_movement_toward_an_off_map_point_is_shortened(self):
        layout = _layout("knockout")
        constrained = constrain_aim_and_movement(
            (200, 500),
            (10, 500),
            (-100, 0),
            "knockout",
            layout["field"],
            step=120,
        )
        self.assertEqual(constrained["aim"][0], layout["field"][0])
        self.assertLess(constrained["movement"][0], 0)
        self.assertGreater(constrained["movement"][0], -100)

        blocked = constrain_aim_and_movement(
            (layout["field"][0], 500),
            (10, 500),
            (-100, 0),
            "knockout",
            layout["field"],
            step=120,
        )
        self.assertEqual(blocked["movement"], (0.0, 0.0))


class UnderdogAccountingTests(unittest.TestCase):
    def test_banner_latches_for_the_rest_of_the_end_screen(self):
        seen = False
        for detected in (False, False, True, False):
            seen = latch_underdog(seen, detected)
        self.assertTrue(seen)
        self.assertFalse(latch_underdog(False, False))

    def test_underdog_win_adds_bonus_and_extends_streak(self):
        update = compute_trophy_update("victory", trophies=100, win_streak=2, underdog=True)
        self.assertTrue(update["underdog_applied"])
        self.assertEqual(update["new_win_streak"], 3)
        self.assertEqual(update["trophy_delta"], 17)
        self.assertEqual(update["new_trophies"], 117)

    def test_underdog_loss_keeps_the_streak_and_shrinks_the_loss(self):
        update = compute_trophy_update("defeat", trophies=100, win_streak=3, underdog=True)
        self.assertEqual(update["new_win_streak"], 3)
        self.assertEqual(update["trophy_delta"], 2)
        self.assertEqual(update["new_trophies"], 102)

    def test_normal_loss_clears_the_streak(self):
        update = compute_trophy_update("defeat", trophies=100, win_streak=3, underdog=False)
        self.assertEqual(update["new_win_streak"], 0)
        self.assertEqual(update["trophy_delta"], -1)
        self.assertEqual(update["new_trophies"], 99)

    def test_underdog_is_ignored_at_2000_trophies(self):
        win = compute_trophy_update("victory", trophies=2500, win_streak=0, underdog=True)
        loss = compute_trophy_update("defeat", trophies=2500, win_streak=4, underdog=True)
        self.assertFalse(win["underdog_applied"])
        self.assertEqual(win["trophy_delta"], 6)
        self.assertEqual(loss["new_win_streak"], 0)
        self.assertEqual(loss["trophy_delta"], -10)

    def test_underdog_draw_awards_the_small_bonus(self):
        update = compute_trophy_update("draw", trophies=100, win_streak=2, underdog=True)
        self.assertEqual(update["trophy_delta"], 4)
        self.assertEqual(update["new_win_streak"], 2)

    def test_showdown_loss_keeps_streak_only_when_underdog(self):
        underdog = compute_trophy_update("defeat", trophies=100, win_streak=4, underdog=True, place=3)
        normal = compute_trophy_update("defeat", trophies=100, win_streak=4, underdog=False, place=3)
        self.assertEqual(underdog["trophy_delta"], -1)
        self.assertEqual(underdog["new_win_streak"], 4)
        self.assertEqual(normal["new_win_streak"], 0)

    def test_observer_uses_the_same_underdog_update(self):
        with patch.object(TrophyObserver, "load_history", return_value=[]), \
             patch.object(TrophyObserver, "_atomic_save_history"), \
             patch.object(TrophyObserver, "send_results_to_api"):
            observer = TrophyObserver()
            observer.current_trophies = 100
            observer.win_streak = 3
            parsed = observer.parse_game_result("defeat")
            observer.add_trophies(parsed, "shelly", {}, True)
            self.assertEqual(observer.current_trophies, 102)
            self.assertEqual(observer.win_streak, 3)
            self.assertEqual(observer.match_history[-1]["trophy_delta"], 2)
            self.assertEqual(observer.match_history[-1]["result"], MatchResult.DEFEAT.value)


class GamemodeRecoveryTests(unittest.TestCase):
    def test_solo_showdown_requests_the_configured_mode(self):
        menu = [
            {"text": "Solo Showdown", "bbox": [0, 0, 10, 10]},
            {"text": "Brawl Ball", "bbox": [100, 40, 180, 80]},
        ]
        action = gamemode_recovery_action(None, "brawlball", menu_text=menu)
        self.assertTrue(action["needed"])
        self.assertEqual(action["action"], "switch_gamemode")
        self.assertEqual(action["detected"], "solo_showdown")
        self.assertEqual(action["configured"], "brawlball")
        self.assertEqual(action["click"], (140, 60))

    def test_ranked_lobby_requests_a_switch_without_a_visible_target(self):
        action = gamemode_recovery_action("Ranked", "knockout", menu_text=["Ranked"])
        self.assertTrue(action["needed"])
        self.assertEqual(action["detected"], "ranked")
        self.assertIsNone(action["click"])

    def test_configured_and_matching_modes_do_not_switch(self):
        self.assertFalse(gamemode_recovery_action("solo showdown", "Solo Showdown")["needed"])
        self.assertFalse(gamemode_recovery_action("trio showdown", "brawlball")["needed"])
        self.assertFalse(gamemode_recovery_action("ranked", "brawlball", enabled=False)["needed"])

    def test_lobby_automation_clicks_the_recovered_mode(self):
        window = MagicMock()
        lobby = LobbyAutomation(window, ocr_reader=MagicMock())
        action = gamemode_recovery_action(
            "solo_showdown",
            "knockout",
            menu_text=[{"text": "Knockout", "bbox": [10, 20, 30, 40]}],
        )
        self.assertEqual(lobby.perform_gamemode_switch(action), "clicked_label")
        window.click.assert_called_once_with(20, 30)

    def test_missing_click_target_still_requests_the_switch(self):
        window = MagicMock()
        lobby = LobbyAutomation(window, ocr_reader=MagicMock())
        action = gamemode_recovery_action("ranked", "gem grab")
        self.assertEqual(lobby.perform_gamemode_switch(action), "requested")
        window.click.assert_not_called()

    def test_ocr_screen_carries_lobby_mode_and_buffie(self):
        showdown = screen_from_grounding(
            "<|ref|>Solo Showdown<|/ref|><|det|>[[0, 0, 400, 100]]<|/det|>",
            200,
            100,
        )
        self.assertEqual(showdown["lobby_mode"], "solo_showdown")
        validate_screen_read(showdown)

        buffie = screen_from_grounding(
            "<|ref|>Buffie<|/ref|><|det|>[[0, 0, 100, 100]]<|/det|>",
            200,
            100,
        )
        self.assertEqual(buffie["screen"], "buffie")
        validate_screen_read(buffie)


class BuffieTests(unittest.TestCase):
    def test_claw_sequence_follows_ui_detections(self):
        claw = BuffieController()
        opened = claw.step({"buffie_visible": True})
        moving = claw.step({"buffie_visible": True, "claw_ready": True, "prize_center": (400, 500)})
        released = claw.step({"buffie_visible": True, "claw_ready": True, "prize_aligned": True})
        waiting = claw.step({"buffie_visible": True, "claw_ready": True})
        collected = claw.step({"reward_visible": True})
        finished = claw.step({})

        self.assertEqual(opened["action"], "click_machine")
        self.assertEqual(opened["state"], "open")
        self.assertEqual(moving["action"], "move_claw")
        self.assertEqual(moving["target"], (400, 500))
        self.assertEqual(released["action"], "release_claw")
        self.assertEqual(waiting["action"], "wait")
        self.assertEqual(collected["action"], "collect")
        self.assertTrue(finished["done"])
        self.assertEqual(finished["state"], "done")

    def test_label_detection_feeds_the_sequence(self):
        detection = buffie_detection_from_labels([
            {"text": "Buffie claw", "bbox": [0, 0, 20, 20]},
            {"text": "Prize", "bbox": [40, 40, 80, 80]},
        ])
        self.assertTrue(detection["buffie_visible"])
        self.assertTrue(detection["claw_ready"])
        self.assertEqual(detection["prize_center"], (60, 60))
        step = BuffieController().step(detection)
        self.assertEqual(step["action"], "move_claw")

    def test_machine_click_uses_the_configured_button(self):
        window = MagicMock()
        lobby = LobbyAutomation(window, ocr_reader=MagicMock())
        self.assertEqual(lobby.perform_buffie_action({"action": "click_machine"}), "click_machine")
        window.click.assert_called_once_with(1700, 860, already_include_ratio=False)
        self.assertEqual(lobby.perform_buffie_action({"action": "collect"}), "collect")
        window.press.assert_called_with("proceed")


class TeammateFocusTests(unittest.TestCase):
    def test_far_teammate_pulls_movement_toward_them(self):
        player = (100, 100)
        teammates = [[400, 80, 440, 120]]
        movement, focus = apply_teammate_bias((0, -100), player, teammates, weight=0.7)
        self.assertTrue(focus["active"])
        self.assertFalse(focus["glued"])
        self.assertGreater(movement[0], 0)
        self.assertLess(movement[1], 0)

    def test_glued_teammate_damps_wander(self):
        movement, focus = apply_teammate_bias(
            (100, 0),
            (100, 100),
            [(120, 100)],
            glue_distance=70,
            weight=0.7,
        )
        self.assertTrue(focus["glued"])
        self.assertAlmostEqual(movement[0], 30)
        self.assertEqual(movement[1], 0)

    def test_disabled_focus_leaves_movement_alone(self):
        focus = teammate_focus_movement((0, 0), [(50, 0)], enabled=False)
        self.assertFalse(focus["active"])
        self.assertIsNone(focus["movement"])

    def test_policy_prioritises_cage_then_gas_then_teammates(self):
        cage_and_gas = apply_safety_policy(
            player_pos=(20, 50),
            playstyle_movement=(0, -100),
            mode="brawlball",
            cages=[(0, 0, 100, 100)],
            map_center=(200, 50),
            field=(0, 0, 400, 200),
            goals=[(0, 0, 100, 100)],
            safe_center=(0, 50),
            safe_radius=5,
            teammates=[(10, 50)],
            teammate_focus=True,
        )
        self.assertIn("brawlball_cage", cage_and_gas["reasons"])
        self.assertNotIn("poison_gas", cage_and_gas["reasons"])
        self.assertGreater(cage_and_gas["movement"][0], 0)

        gas_over_team = apply_safety_policy(
            player_pos=(100, 100),
            playstyle_movement=(0, 0),
            mode="trio showdown",
            directional_gas={"up": 0, "down": 0, "left": 8, "right": 0},
            teammates=[(20, 100)],
        )
        self.assertEqual(gas_over_team["reasons"], ["poison_gas"])
        self.assertGreater(gas_over_team["movement"][0], 0)

        team = apply_safety_policy(
            player_pos=(100, 100),
            playstyle_movement=(0, -100),
            mode="trio_showdown",
            teammates=[[400, 80, 440, 120]],
        )
        self.assertIn("teammate_focus", team["reasons"])
        self.assertGreater(team["movement"][0], 0)


if __name__ == "__main__":
    unittest.main()
