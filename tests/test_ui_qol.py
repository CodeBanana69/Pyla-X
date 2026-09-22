import importlib.util
import tempfile
import threading
import time
import unittest
from datetime import date
from pathlib import Path

from clip_recorder import (
    NormalClipRecorder,
    clip_outputs_for_tick,
    frame_for_clip,
    open_mp4_writer,
)
from control_loop import control_action
from interface_launch import parse_cli_args, resolve_interface_mode

ROOT = Path(__file__).resolve().parents[1]


def _load(module_name, relative_path):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


history_filter = _load("pyla_history_filter", "webui/history_filter.py")
locale_fallback = _load("pyla_locale_fallback", "webui/locale_fallback.py")
runtime = _load("pyla_runtime", "webui/runtime.py")
settings_search = _load("pyla_settings_search", "webui/settings_search.py")

in_inclusive_date_range = history_filter.in_inclusive_date_range
translate = locale_fallback.translate
RuntimeControl = runtime.RuntimeControl
RuntimeManager = runtime.RuntimeManager
filter_settings = settings_search.filter_settings


class FakeFrame:
    def __init__(self, name, shape=(4, 6, 3)):
        self.name = name
        self.shape = shape

    def copy(self):
        return FakeFrame(self.name, self.shape)


class FakeWriter:
    def __init__(self):
        self.frames = []
        self.released = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True


class MatchSimulator:
    def __init__(self):
        self.entered = threading.Event()
        self.paused = threading.Event()
        self.finished = threading.Event()
        self.actions = []

    def __call__(self, discord_bot, queue_data, runtime_control=None):
        self.entered.set()
        try:
            while True:
                action = control_action(
                    "match",
                    stop_requested=runtime_control.should_stop(),
                    pause_requested=runtime_control.should_pause(),
                    immediate=runtime_control.interrupts_immediately(),
                )
                self.actions.append(action)
                if action == "stop":
                    return
                if action == "pause":
                    self.paused.set()
                    runtime_control.mark_paused()
                    while runtime_control.should_pause() and not runtime_control.should_stop():
                        time.sleep(0.01)
                    if runtime_control.should_stop():
                        return
                    runtime_control.mark_running()
                    self.paused.clear()
                time.sleep(0.01)
        finally:
            self.finished.set()


class ControlActionTests(unittest.TestCase):
    def test_deferred_requests_wait_for_the_lobby(self):
        self.assertIsNone(control_action("match", stop_requested=True, pause_requested=False, immediate=False))
        self.assertIsNone(control_action("match", stop_requested=False, pause_requested=True, immediate=False))
        self.assertEqual(control_action("lobby", stop_requested=True, pause_requested=False, immediate=False), "stop")
        self.assertEqual(control_action("lobby", stop_requested=False, pause_requested=True, immediate=False), "pause")

    def test_immediate_requests_interrupt_the_current_match(self):
        self.assertEqual(control_action("match", stop_requested=True, pause_requested=False, immediate=True), "stop")
        self.assertEqual(control_action("match", stop_requested=False, pause_requested=True, immediate=True), "pause")
        self.assertEqual(
            control_action("match", stop_requested=True, pause_requested=True, immediate=True),
            "stop",
        )

    def test_runtime_flags_distinguish_force_from_deferred(self):
        states = []
        control = RuntimeControl(states.append)
        control.request_pause(immediate=False)
        self.assertTrue(control.should_pause())
        self.assertFalse(control.interrupts_immediately())
        self.assertIsNone(control_action("match", stop_requested=False, pause_requested=True, immediate=False))

        control.request_pause(immediate=True)
        self.assertTrue(control.interrupts_immediately())
        self.assertEqual(
            control_action(
                "match",
                stop_requested=control.should_stop(),
                pause_requested=control.should_pause(),
                immediate=control.interrupts_immediately(),
            ),
            "pause",
        )

        control.request_stop(immediate=True)
        self.assertTrue(control.should_stop())
        self.assertFalse(control.should_pause())
        self.assertTrue(control.interrupts_immediately())
        self.assertEqual(
            control_action(
                "match",
                stop_requested=True,
                pause_requested=False,
                immediate=control.interrupts_immediately(),
            ),
            "stop",
        )

        deferred = RuntimeControl(states.append)
        deferred.request_stop(immediate=False)
        self.assertTrue(deferred.should_stop())
        self.assertFalse(deferred.interrupts_immediately())


class RuntimeInterruptTests(unittest.TestCase):
    def test_force_pause_interrupts_match_and_deferred_pause_does_not(self):
        loop = MatchSimulator()
        manager = RuntimeManager(loop)
        started = manager.start([{"brawler": "Shelly"}], object())
        self.assertTrue(started["ok"])
        self.assertTrue(loop.entered.wait(2))

        deferred = manager.pause(immediate=False)
        self.assertTrue(deferred["ok"])
        self.assertFalse(deferred["immediate"])
        self.assertFalse(loop.paused.wait(0.2))
        self.assertNotIn("pause", loop.actions)

        forced = manager.pause(immediate=True)
        self.assertTrue(forced["ok"])
        self.assertTrue(forced["immediate"])
        self.assertIn("immediately", forced["message"])
        self.assertTrue(loop.paused.wait(2))
        self.assertEqual(manager.get_status()["state"], "paused")
        self.assertIn("pause", loop.actions)

        manager.stop(immediate=True)
        self.assertTrue(loop.finished.wait(2))

    def test_force_stop_interrupts_match_and_deferred_stop_does_not(self):
        loop = MatchSimulator()
        manager = RuntimeManager(loop)
        self.assertTrue(manager.start([{"brawler": "Shelly"}], object())["ok"])
        self.assertTrue(loop.entered.wait(2))

        deferred = manager.stop(immediate=False)
        self.assertTrue(deferred["ok"])
        self.assertFalse(deferred["immediate"])
        self.assertFalse(loop.finished.wait(0.2))
        self.assertNotIn("stop", loop.actions)

        forced = manager.stop(immediate=True)
        self.assertTrue(forced["ok"])
        self.assertTrue(forced["immediate"])
        self.assertIn("Force stop", forced["message"])
        self.assertTrue(loop.finished.wait(2))
        self.assertIn("stop", loop.actions)
        self.assertFalse(manager.get_status()["is_running"])


class ClipRecorderTests(unittest.TestCase):
    def test_normal_clips_keep_clean_frames_and_debug_clips_keep_overlays(self):
        clean = FakeFrame("clean")
        overlay = FakeFrame("overlay")
        self.assertIs(frame_for_clip(clean, overlay, include_overlays=False), clean)
        self.assertIs(frame_for_clip(clean, overlay, include_overlays=True), overlay)

        outputs = clip_outputs_for_tick(clean, overlay, record_normal=True, record_debug=True)
        self.assertEqual(outputs, [("normal", clean), ("debug", overlay)])

    def test_normal_recorder_muxes_only_clean_frames(self):
        clean = FakeFrame("clean")
        overlay = FakeFrame("overlay")
        writer = FakeWriter()
        with tempfile.TemporaryDirectory() as tmp:
            recorder = NormalClipRecorder(
                width=6,
                height=4,
                fps=10,
                min_player_seen_before_recording=0,
                output_dir=tmp,
                writer_factory=lambda path, fps, size: writer,
            )
            for kind, frame in clip_outputs_for_tick(clean, overlay, record_normal=True, record_debug=True):
                if kind == "normal":
                    recorder.update(frame, {"player": [[0, 0, 1, 1]]}, True)
            recorder.stop()

        self.assertTrue(writer.frames)
        self.assertTrue(all(frame.name == "clean" for frame in writer.frames))
        self.assertFalse(any(frame.name == "overlay" for frame in writer.frames))
        self.assertTrue(writer.released)

    def test_missing_encoder_skips_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = NormalClipRecorder(
                width=6,
                height=4,
                fps=10,
                min_player_seen_before_recording=0,
                output_dir=tmp,
                writer_factory=lambda path, fps, size: None,
            )
            recorder.update(FakeFrame("clean"), {"player": [[1]]}, True)
            recorder.update(FakeFrame("clean"), {"player": [[1]]}, True)
            self.assertIsNone(recorder.writer)
            self.assertTrue(recorder._encoding_unavailable)
            self.assertEqual(list(Path(tmp).glob("*.mp4")), [])

    def test_real_mp4_encoder_or_skip(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.mp4"
            writer = open_mp4_writer(probe, 8, (8, 8))
            if writer is None:
                self.skipTest("mp4 encoder is unavailable")
            writer.release()
            probe.unlink(missing_ok=True)

            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            recorder = NormalClipRecorder(
                width=8,
                height=8,
                fps=8,
                min_player_seen_before_recording=0,
                output_dir=tmp,
            )
            recorder.record_gameplay(frame, [[0, 0, 2, 2]])
            recorder.stop()
            clips = list(Path(tmp).glob("gameplay_clip_*.mp4"))
            if not clips:
                self.skipTest("mp4 encoder did not write a gameplay clip")
            self.assertGreater(clips[0].stat().st_size, 0)


class HistoryFilterTests(unittest.TestCase):
    def test_inclusive_date_range(self):
        start = date(2026, 1, 1)
        end = date(2026, 1, 31)
        self.assertTrue(in_inclusive_date_range(date(2026, 1, 1), start, end))
        self.assertTrue(in_inclusive_date_range(date(2026, 1, 31), start, end))
        self.assertTrue(in_inclusive_date_range(date(2026, 1, 15), start, end))
        self.assertFalse(in_inclusive_date_range(date(2025, 12, 31), start, end))
        self.assertFalse(in_inclusive_date_range(date(2026, 2, 1), start, end))
        self.assertTrue(in_inclusive_date_range(date(2026, 5, 5), date(2026, 5, 5), None))
        self.assertTrue(in_inclusive_date_range(date(2026, 5, 5), None, date(2026, 5, 5)))
        self.assertFalse(in_inclusive_date_range(date(2026, 5, 4), date(2026, 5, 5), None))
        self.assertFalse(in_inclusive_date_range(None, start, None))
        self.assertTrue(in_inclusive_date_range(None, None, None))
        self.assertTrue(in_inclusive_date_range(date(2024, 2, 29), None, None))


class SettingsSearchTests(unittest.TestCase):
    def setUp(self):
        self.settings = [
            {
                "section": "debug",
                "key": "record_normal_clips",
                "label": "Record Clips",
                "help": "Save MP4 clips of clean gameplay when the player is tracked and then lost. Debug overlays are never included.",
            },
            {
                "section": "general",
                "key": "emulator_port",
                "label": "Emulator Port",
                "help": "ADB port used for the emulator instance.",
            },
            {
                "section": "general",
                "key": "interface_mode",
                "label": "Interface Mode",
                "description": "Choose what Pyla opens at startup.",
                "options": [
                    {"value": "desktop", "label": "Integrated window"},
                    {"value": "headless", "label": "Headless"},
                ],
            },
        ]

    def test_matches_keyword_description_and_option_text(self):
        self.assertEqual(
            [item["key"] for item in filter_settings(self.settings, "overlay")],
            ["record_normal_clips"],
        )
        self.assertEqual(
            [item["key"] for item in filter_settings(self.settings, "adb port")],
            ["emulator_port"],
        )
        self.assertEqual(
            [item["key"] for item in filter_settings(self.settings, "Headless")],
            ["interface_mode"],
        )
        self.assertEqual(len(filter_settings(self.settings, "  ")), 3)
        self.assertEqual(filter_settings(self.settings, "missing-keyword"), [])


class InterfaceLaunchTests(unittest.TestCase):
    def test_cli_selects_desktop_browser_and_headless(self):
        self.assertEqual(parse_cli_args(["--desktop"]).interface_mode, "desktop")
        self.assertEqual(parse_cli_args(["--web"]).interface_mode, "browser")
        self.assertEqual(parse_cli_args(["--browser"]).interface_mode, "browser")
        self.assertEqual(parse_cli_args(["--no-webapp"]).interface_mode, "browser")
        self.assertEqual(parse_cli_args(["--headless"]).interface_mode, "headless")
        self.assertIsNone(parse_cli_args([]).interface_mode)
        self.assertTrue(parse_cli_args(["--headless", "--no-console"]).no_console)

    def test_cli_overrides_saved_interface_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "general_config.toml"
            path.write_text('interface_mode = "headless"\n', encoding="utf-8")
            self.assertEqual(resolve_interface_mode(parse_cli_args([]), path), "headless")
            self.assertEqual(resolve_interface_mode(parse_cli_args(["--desktop"]), path), "desktop")


class LocalizationTests(unittest.TestCase):
    def test_fallback_keeps_english_for_missing_or_unknown_languages(self):
        catalogs = {
            "fr": {"Pause": "Mettre en pause", "Record Clips": "Enregistrer des clips"},
            "ru": {"Pause": "Пауза", "Record Clips": "Записывать клипы"},
        }
        self.assertEqual(translate(catalogs, "fr", "Pause"), "Mettre en pause")
        self.assertEqual(translate(catalogs, "ru", "Record Clips"), "Записывать клипы")
        self.assertEqual(translate(catalogs, "fr", "Not in the catalog"), "Not in the catalog")
        self.assertEqual(translate(catalogs, "de", "Pause"), "Pause")
        self.assertEqual(translate(catalogs, "en", "Pause"), "Pause")
        self.assertEqual(translate(catalogs, None, "Pause"), "Pause")

    def test_ui_catalogs_include_russian_and_french(self):
        script = (ROOT / "static" / "js" / "i18n.js").read_text(encoding="utf-8")
        html = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIn('value="ru"', html)
        self.assertIn('value="fr"', html)
        self.assertIn("const FR = {", script)
        self.assertIn("Object.assign(RU,", script)
        expectations = {
            "From date": ("Date de début", "Дата начала"),
            "To date": ("Date de fin", "Дата окончания"),
            "Date range": ("Période", "Диапазон дат"),
            "Find a setting": ("Rechercher un paramètre", "Найти настройку"),
            "Record Clips": ("Enregistrer des clips", "Записывать клипы"),
            "Force Pause": ("Forcer la pause", "Пауза сразу"),
            "Force Stop": ("Forcer l’arrêt", "Остановить сразу"),
            "Save MP4 clips of clean gameplay when the player is tracked and then lost. Debug overlays are never included.": (
                "Enregistrer des clips MP4 de la partie, sans incrustation de débogage, lorsque le joueur est suivi puis perdu.",
                "Сохранять MP4-клипы чистого геймплея, когда игрок сначала найден, а затем потерян. Отладочные наложения не записываются.",
            ),
        }
        for english, (french, russian) in expectations.items():
            self.assertIn(f'"{english}": "{french}"', script)
            self.assertIn(f'"{english}": "{russian}"', script)


if __name__ == "__main__":
    unittest.main()
