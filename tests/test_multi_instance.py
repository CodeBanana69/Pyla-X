import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from instance_profiles import (
    InstanceRegistry,
    bind_profile,
    clear_bound_profile,
    set_registry,
)
from port_finder import DEEP_SCAN_END, DEEP_SCAN_START, SHALLOW_ADB_PORTS, iter_scan_ports, scan_adb_ports


def _load_module(name: str, relative: str):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def registry(tmp_path: Path):
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "general_config.toml").write_text(
        'emulator_port = 5037\nplayer_tag = ""\nmax_fps = "auto"\nplay_order = "in_order"\n'
        "auto_load_queue_on_startup = false\n",
        encoding="utf-8",
    )
    (tmp_path / "cfg" / "bot_config.toml").write_text(
        'current_playstyle = "showdown_survivor.pyla"\n',
        encoding="utf-8",
    )
    created = InstanceRegistry(tmp_path)
    set_registry(created)
    yield created
    clear_bound_profile()
    set_registry(None)


def test_profiles_isolate_port_playstyle_queue_and_history(registry: InstanceRegistry):
    second = registry.create_profile("Second emulator")
    registry.set_adb_port(second["id"], 16384)
    registry.set_playstyle(second["id"], "default_up.pyla")
    registry.write_queue("default", [{"brawler": "shelly", "trophies": 100}])
    registry.write_queue(second["id"], [{"brawler": "colt", "trophies": 200}])
    registry.ensure_history("default").write_text(
        "date_time,brawler_name\n2026-01-01,Shelly\n",
        encoding="utf-8",
    )
    second_history = registry.history_path(second["id"])
    second_history.write_text("date_time,brawler_name\n2026-01-02,Colt\n", encoding="utf-8")

    assert registry.queue_path("default") != registry.queue_path(second["id"])
    assert registry.history_path("default") != second_history
    assert registry.read_queue("default")[0]["brawler"] == "shelly"
    assert registry.read_queue(second["id"])[0]["brawler"] == "colt"
    assert "Shelly" in registry.history_path("default").read_text(encoding="utf-8")
    assert "Colt" not in registry.history_path("default").read_text(encoding="utf-8")
    assert registry.get_profile("default")["adb_port"] != 16384
    assert registry.get_profile(second["id"])["playstyle"] == "default_up.pyla"
    assert registry.get_profile("default")["playstyle"] == "showdown_survivor.pyla"

    shared = {"emulator_port": 5037, "player_tag": ""}
    default_view = registry.apply_file_overrides("default", "cfg/general_config.toml", shared)
    second_view = registry.apply_file_overrides(second["id"], "cfg/general_config.toml", shared)
    assert default_view["emulator_port"] == 5037
    assert second_view["emulator_port"] == 16384


def test_synced_settings_follow_all_profiles_and_unsynced_stay_separate(registry: InstanceRegistry):
    second = registry.create_profile("Alt account")
    assert registry.is_synced("general", "max_fps")
    assert not registry.is_synced("general", "player_tag")
    assert not registry.is_synced("general", "emulator_port")
    assert not registry.is_synced("bot", "current_playstyle")

    registry.set_override(second["id"], "general", "player_tag", "#ALT")
    shared = {"max_fps": "auto", "player_tag": "", "emulator_port": 5037}
    assert registry.apply_file_overrides("default", "cfg/general_config.toml", shared)["player_tag"] == ""
    assert registry.apply_file_overrides(second["id"], "cfg/general_config.toml", shared)["player_tag"] == "#ALT"

    shared["max_fps"] = 30
    assert registry.apply_file_overrides("default", "cfg/general_config.toml", shared)["max_fps"] == 30
    assert registry.apply_file_overrides(second["id"], "cfg/general_config.toml", shared)["max_fps"] == 30

    action = registry.set_key_synced("general", "max_fps", False, 30, 30)
    assert action == "per_instance"
    registry.set_override(second["id"], "general", "max_fps", 15)
    unsynced_shared = {"max_fps": 30, "player_tag": "", "emulator_port": 5037}
    assert registry.apply_file_overrides("default", "cfg/general_config.toml", unsynced_shared)["max_fps"] == 30
    assert registry.apply_file_overrides(second["id"], "cfg/general_config.toml", unsynced_shared)["max_fps"] == 15

    action = registry.set_key_synced("general", "max_fps", True, 15, 30)
    assert action == "write_shared"
    resynced = {"max_fps": 15, "player_tag": "", "emulator_port": 5037}
    assert registry.apply_file_overrides("default", "cfg/general_config.toml", resynced)["max_fps"] == 15
    assert registry.apply_file_overrides(second["id"], "cfg/general_config.toml", resynced)["max_fps"] == 15


def test_default_profile_cannot_be_deleted_and_ports_cannot_collide(registry: InstanceRegistry):
    second = registry.create_profile("MuMu")
    with pytest.raises(ValueError, match="default profile cannot be deleted"):
        registry.delete_profile("default")
    with pytest.raises(ValueError, match="already used"):
        registry.set_adb_port(second["id"], registry.get_profile("default")["adb_port"])
    registry.delete_profile(second["id"])
    assert registry.get_profile(second["id"]) is None
    assert not registry.queue_path(second["id"]).exists()


def test_bound_profile_overrides_loaded_config(registry: InstanceRegistry, monkeypatch, tmp_path: Path):
    import utils

    second = registry.create_profile("Bound")
    registry.set_adb_port(second["id"], 7555)
    registry.set_playstyle(second["id"], "default_up.pyla")
    monkeypatch.setattr(utils, "PROJECT_ROOT", tmp_path)
    utils.cached_toml.clear()

    bind_profile(second["id"])
    general = utils.load_toml_as_dict("cfg/general_config.toml")
    bot = utils.load_toml_as_dict("cfg/bot_config.toml")
    clear_bound_profile()
    unbound = utils.load_toml_as_dict("cfg/general_config.toml")

    assert general["emulator_port"] == 7555
    assert bot["current_playstyle"] == "default_up.pyla"
    assert unbound["emulator_port"] == 5037


def test_shallow_scan_ignores_ports_outside_known_emulator_ranges():
    hidden_port = 42424
    assert hidden_port not in SHALLOW_ADB_PORTS
    assert 16384 in SHALLOW_ADB_PORTS
    assert 16416 in SHALLOW_ADB_PORTS

    def probe(port: int) -> bool:
        return port in {5555, hidden_port}

    shallow = scan_adb_ports(deep=False, probe=probe)
    shallow_ports = [device["port"] for device in shallow]
    assert shallow_ports == [5555]
    assert all(DEEP_SCAN_START <= port <= DEEP_SCAN_END for port in iter_scan_ports(deep=True))


def test_deep_scan_finds_ports_outside_the_shallow_candidate_list():
    hidden_port = 42424
    probed = []

    def probe(port: int) -> bool:
        probed.append(port)
        return port == hidden_port

    found = scan_adb_ports(deep=True, probe=probe)
    assert hidden_port in probed
    assert [device["port"] for device in found] == [hidden_port]
    assert found[0]["serial"] == "127.0.0.1:42424"
    assert hidden_port not in iter_scan_ports(deep=False)


def test_runtime_logs_and_workers_stay_isolated_per_profile():
    RuntimeManager = _load_module("pyla_runtime_under_test", "webui/runtime.py").RuntimeManager

    def fake_main(_discord, _queue, runtime_control=None, profile_id=None):
        print(f"worker {profile_id}")
        while runtime_control and not runtime_control.should_stop():
            time.sleep(0.01)

    manager = RuntimeManager(fake_main)
    manager.set_active_profile_provider(lambda: "default")

    def emit(name: str, text: str):
        thread = threading.Thread(target=lambda: print(text), name=name)
        thread.start()
        thread.join()

    emit("pyla-default", "default-log-line")
    emit("pyla-second", "second-log-line")
    assert any("default-log-line" in line for line in manager.get_logs("default"))
    assert all("second-log-line" not in line for line in manager.get_logs("default"))
    assert any("second-log-line" in line for line in manager.get_logs("second"))

    assert manager.start([{"brawler": "shelly"}], None, profile_id="default")["ok"]
    assert manager.start([{"brawler": "colt"}], None, profile_id="second")["ok"]
    deadline = time.time() + 2
    while time.time() < deadline and not (
        manager.get_status("default")["is_running"] and manager.get_status("second")["is_running"]
    ):
        time.sleep(0.01)
    assert manager.get_status("default")["is_running"]
    assert manager.get_status("second")["is_running"]
    assert manager.get_status("default")["state"] == "running"
    manager.stop("default")
    manager.stop("second")
    deadline = time.time() + 2
    while time.time() < deadline and (
        manager.get_status("default")["is_running"] or manager.get_status("second")["is_running"]
    ):
        time.sleep(0.01)
    assert not manager.get_status("default")["is_running"]
    assert not manager.get_status("second")["is_running"]
    manager.clear_logs("second")
    assert manager.get_logs("second") == []
    assert any("default-log-line" in line for line in manager.get_logs("default"))


def test_service_sync_and_profile_queues(registry: InstanceRegistry, monkeypatch, tmp_path: Path):
    import utils
    WebDataService = _load_module("pyla_services_under_test", "webui/services.py").WebDataService

    monkeypatch.setattr(utils, "PROJECT_ROOT", tmp_path)
    utils.cached_toml.clear()

    class FakeRuntime:
        def get_status(self, profile_id=None):
            return {
                "profile_id": profile_id or "default",
                "state": "idle",
                "is_running": False,
                "last_error": "",
                "session_started_at": None,
            }

    service = WebDataService(FakeRuntime())
    service.save_queue_data([{
        "brawler": "shelly",
        "type": "trophies",
        "push_until": 1000,
        "trophies": 10,
        "wins": 0,
        "automatically_pick": True,
        "win_streak": 0,
    }])
    created = service.create_profile("Second emulator profile")
    active = next(item for item in created["profiles"]["items"] if item["is_active"])
    assert active["name"] == "Second emulator profile"
    assert created["queue"] == []

    service.use_profile("default")
    assert service.get_queue_data()[0]["brawler"] == "shelly"
    service.update_settings("general", {"max_fps": 45, "player_tag": "#MAIN"})
    service.use_profile(active["id"])
    other = service.get_settings_payload("general")
    assert other["max_fps"] == 45
    assert other["player_tag"] != "#MAIN"
    service.update_settings("general", {"player_tag": "#ALT"})
    service.use_profile("default")
    assert service.get_settings_payload("general")["player_tag"] == "#MAIN"
    assert json.loads(registry.queue_path("default").read_text(encoding="utf-8"))[0]["brawler"] == "shelly"
