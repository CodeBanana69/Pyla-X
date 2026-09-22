"""In-process bot instances for Discord remote commands.

Commands always act on the instance selected by ``/activate_instance``.
A single default instance is registered when the bot starts, so existing
one-process setups keep working.
"""

from __future__ import annotations

import csv
import re
import threading
from pathlib import Path
from typing import Any, Callable

INVALID_PLAYER_TAG_MESSAGE = "Player tag is incorrect. Use your Brawl Stars player tag, not your Supercell ID."
MATCH_HISTORY_LIMIT = 25
PLAYER_TAG_BODY = re.compile(r"[0289PYLQGRJCUV]{3,15}")


def normalize_player_tag(raw_tag: Any) -> str:
    tag = str(raw_tag or "").strip().upper().replace("%23", "")
    if tag.startswith("#"):
        tag = tag[1:]
    if not PLAYER_TAG_BODY.fullmatch(tag):
        raise ValueError(INVALID_PLAYER_TAG_MESSAGE)
    return tag


def normalize_instance_name(raw_name: Any) -> str:
    name = " ".join(str(raw_name or "").strip().split())
    if not name:
        raise ValueError("Instance name is required.")
    if len(name) > 32:
        raise ValueError("Instance name must be 32 characters or fewer.")
    return name


def clamp_history_limit(limit: Any, maximum: int = MATCH_HISTORY_LIMIT) -> int:
    try:
        parsed = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("Match history limit must be a whole number.") from exc
    if parsed < 1:
        raise ValueError("Match history limit must be at least 1.")
    return min(parsed, maximum)


def merge_brawlers_below_threshold(queue: list[dict[str, Any]], selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_by_key = {str(item["brawler"]).lower(): item for item in selected}
    updated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in queue:
        key = str(item.get("brawler", "")).lower()
        if key in selected_by_key:
            updated.append(dict(selected_by_key[key]))
            seen.add(key)
        else:
            updated.append(dict(item))
    for key, item in selected_by_key.items():
        if key not in seen:
            updated.append(dict(item))
    return updated


def select_recent_matches(matches: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(
        matches,
        key=lambda match: str(match.get("date_sort") or match.get("date_time") or ""),
        reverse=True,
    )
    return ordered[:limit]


def read_recent_matches(csv_path: str | Path, limit: int = 10) -> list[dict[str, Any]]:
    """Return the newest matches from a match-history CSV.

    Empty rows are ignored. Missing files yield an empty list so callers
    can run before the observer has written history.
    """
    bounded = clamp_history_limit(limit, maximum=50)
    path = Path(csv_path)
    if not path.exists():
        return []

    matches: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            brawler = str(row.get("brawler_name") or "").strip()
            if not brawler:
                continue
            gamemodes = [part for part in str(row.get("playstyle_gamemodes") or "").split("|") if part]
            try:
                trophy_delta = int(float(row.get("trophy_delta")))
            except (TypeError, ValueError):
                trophy_delta = 0
            try:
                trophy_before = int(float(row.get("current_trophies")))
            except (TypeError, ValueError):
                trophy_before = None
            matches.append({
                "brawler": brawler,
                "date_time": str(row.get("date_time") or "").strip(),
                "date_sort": str(row.get("date_time") or "").strip(),
                "result": str(row.get("result") or "").strip().lower(),
                "trophy_before": trophy_before,
                "trophy_delta": trophy_delta,
                "win_streak": row.get("new_winstreak"),
                "playstyle_name": str(row.get("playstyle_name") or "").strip(),
                "mode": gamemodes[0] if gamemodes else "",
            })
    return select_recent_matches(matches, bounded)


class AdbScrcpyService:
    """Restart ADB and scrcpy for one instance.

    Tests inject ``restart_adb``, ``stop_scrcpy``, and ``reconnect_scrcpy``
    so the real server is never killed. The default callables are imported
    only when a restart actually runs without overrides.
    """

    def __init__(
        self,
        restart_adb: Callable[[Any], Any] | None = None,
        stop_scrcpy: Callable[[Any], Any] | None = None,
        reconnect_scrcpy: Callable[[Any], Any] | None = None,
    ):
        self._restart_adb = restart_adb
        self._stop_scrcpy = stop_scrcpy
        self._reconnect_scrcpy = reconnect_scrcpy

    def restart(self, instance: Any) -> dict[str, Any]:
        controller = getattr(instance, "window_controller", None)
        errors: list[str] = []
        adb_restarted = False
        scrcpy_restarted = False

        try:
            if self._stop_scrcpy is not None:
                self._stop_scrcpy(instance)
            else:
                self._stop_attached_scrcpy(controller)
        except Exception as exc:
            errors.append(f"scrcpy stop failed: {exc}")

        try:
            if self._restart_adb is not None:
                self._restart_adb(instance)
            else:
                self._restart_attached_adb(controller)
            adb_restarted = True
        except Exception as exc:
            errors.append(str(exc))

        try:
            if self._reconnect_scrcpy is not None:
                scrcpy_restarted = bool(self._reconnect_scrcpy(instance))
            else:
                scrcpy_restarted = self._reconnect_attached_scrcpy(controller)
        except Exception as exc:
            errors.append(f"scrcpy restart failed: {exc}")

        if not adb_restarted:
            message = "Failed to restart ADB."
        elif scrcpy_restarted:
            message = "ADB and scrcpy restarted."
        elif controller is None and self._reconnect_scrcpy is None:
            message = "ADB restarted. Scrcpy is not attached for this instance."
        else:
            message = "ADB restarted, but scrcpy did not come back."
        if errors:
            message = f"{message} {' '.join(errors)}".strip()

        no_client_to_restore = controller is None and self._reconnect_scrcpy is None
        return {
            "ok": adb_restarted and not errors and (scrcpy_restarted or no_client_to_restore),
            "message": message,
            "adb_restarted": adb_restarted,
            "scrcpy_restarted": scrcpy_restarted,
            "instance_id": getattr(instance, "instance_id", None),
        }

    @staticmethod
    def _stop_attached_scrcpy(controller: Any):
        client = getattr(controller, "scrcpy_client", None) if controller is not None else None
        if client is not None:
            client.stop()

    @staticmethod
    def _restart_attached_adb(controller: Any):
        if controller is not None and hasattr(controller, "force_rediscover"):
            if not controller.force_rediscover():
                raise RuntimeError("ADB restarted, but the device was not found.")
            return
        from window_controller import restart_adb_server

        restart_adb_server()

    @staticmethod
    def _reconnect_attached_scrcpy(controller: Any) -> bool:
        if controller is None or not hasattr(controller, "reconnect_scrcpy"):
            return False
        return bool(controller.reconnect_scrcpy())


class DetachedRuntime:
    """Placeholder runtime for an instance that has no local bot thread."""

    def get_status(self) -> dict[str, Any]:
        return {
            "state": "idle",
            "is_running": False,
            "last_error": "",
            "session_started_at": None,
        }

    def start_current_queue(self, _discord_bot) -> dict[str, Any]:
        return {"ok": False, "message": "This instance has no local bot process attached."}

    def stop(self) -> dict[str, Any]:
        return {"ok": True, "message": "Pyla is already stopped."}

    def pause(self) -> dict[str, Any]:
        return {"ok": False, "message": "Pyla is not running."}


class BotInstance:
    def __init__(
        self,
        instance_id: str,
        runtime_manager: Any = None,
        data_service: Any = None,
        adb_service: AdbScrcpyService | None = None,
        persist_player_tag: bool = False,
    ):
        self.instance_id = instance_id
        self.runtime_manager = runtime_manager if runtime_manager is not None else DetachedRuntime()
        self.data_service = data_service
        self.adb_service = adb_service if adb_service is not None else AdbScrcpyService()
        self.persist_player_tag = persist_player_tag
        self.window_controller = None
        self._player_tag: str | None = None
        self._queue: list[dict[str, Any]] | None = None
        self.matches: list[dict[str, Any]] | None = None
        self.brawler_trophies: list[dict[str, Any]] | None = None
        self.session_overlay: dict[str, Any] = {}

    def get_status(self) -> dict[str, Any]:
        if self.runtime_manager is None:
            return DetachedRuntime().get_status()
        return self.runtime_manager.get_status()

    def get_player_tag(self) -> str:
        if self._player_tag is not None:
            return self._player_tag
        service = self.data_service
        if service is not None and hasattr(service, "get_settings_payload"):
            general = service.get_settings_payload("general") or {}
            return str(general.get("player_tag") or "").replace("#", "").strip().upper()
        return ""

    def set_player_tag(self, raw_tag: Any) -> str:
        tag = normalize_player_tag(raw_tag)
        self._player_tag = tag
        if self.persist_player_tag and self.data_service is not None and hasattr(self.data_service, "update_settings"):
            self.data_service.update_settings("general", {"player_tag": tag})
        return tag

    def get_queue(self) -> list[dict[str, Any]]:
        if self._queue is not None:
            return [dict(item) for item in self._queue]
        if self.data_service is not None and hasattr(self.data_service, "get_queue_data"):
            return self.data_service.get_queue_data()
        return []

    def set_queue(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        copied = [dict(item) for item in items]
        if self._queue is not None or self.data_service is None or not hasattr(self.data_service, "save_queue_data"):
            self._queue = copied
            return [dict(item) for item in copied]
        if hasattr(self.data_service, "_assert_queue_editable"):
            self.data_service._assert_queue_editable()
        self.data_service.save_queue_data(copied)
        return self.get_queue()

    def iter_brawlers(self) -> list[dict[str, Any]]:
        if self.brawler_trophies is not None:
            return [dict(item) for item in self.brawler_trophies]
        service = self.data_service
        if service is not None and hasattr(service, "list_brawler_trophies"):
            return list(service.list_brawler_trophies(self.get_player_tag()))
        return []

    def push_all(self, trophy_threshold: int) -> dict[str, Any]:
        try:
            threshold = int(trophy_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("Trophy threshold must be a whole number.") from exc
        if threshold < 0:
            raise ValueError("Trophy threshold must be zero or greater.")

        source = self.iter_brawlers()
        if not source:
            return {
                "items": self.get_queue(),
                "added_count": 0,
                "brawlers": [],
                "reason": "no_trophy_data",
            }

        selected = []
        for item in source:
            name = str(item.get("brawler") or "").strip()
            if not name:
                continue
            trophies = int(item.get("trophies") or 0)
            if trophies >= threshold:
                continue
            selected.append({
                "brawler": name,
                "type": "trophies",
                "push_until": threshold,
                "trophies": trophies,
                "wins": int(item.get("wins") or 0),
                "automatically_pick": bool(item.get("automatically_pick", True)),
                "win_streak": int(item.get("win_streak") or 0),
            })

        if not selected:
            return {
                "items": self.get_queue(),
                "added_count": 0,
                "brawlers": [],
                "reason": "none_below_threshold",
            }

        updated = merge_brawlers_below_threshold(self.get_queue(), selected)
        self.set_queue(updated)
        return {
            "items": self.get_queue(),
            "added_count": len(selected),
            "brawlers": [item["brawler"] for item in selected],
            "reason": "queued",
        }

    def update_session(self, stats: dict[str, Any] | None):
        if not stats:
            return
        for key in ("trophies", "wins", "losses", "trophy_delta", "current_brawler", "mode"):
            if key in stats and stats[key] is not None:
                self.session_overlay[key] = stats[key]

    def get_session(self) -> dict[str, Any]:
        status = self.get_status()
        overlay = self.session_overlay
        queue = self.get_queue()
        current = queue[0] if queue else {}
        history_session: dict[str, Any] = {}
        mode = overlay.get("mode")
        service = self.data_service
        if service is not None:
            if mode is None and hasattr(service, "get_playstyles_payload"):
                try:
                    current_playstyle = (service.get_playstyles_payload() or {}).get("current") or {}
                    gamemodes = current_playstyle.get("gamemodes") or []
                    mode = gamemodes[0] if gamemodes else current_playstyle.get("name")
                except Exception:
                    mode = None
            if "wins" not in overlay and hasattr(service, "get_match_history_payload"):
                try:
                    history_session = (service.get_match_history_payload() or {}).get("session_summary") or {}
                except Exception:
                    history_session = {}

        trophies = overlay.get("trophies")
        if trophies is None:
            trophies = current.get("trophies")
        wins = overlay.get("wins")
        if wins is None:
            wins = history_session.get("wins", 0)
        losses = overlay.get("losses")
        if losses is None:
            losses = history_session.get("losses", 0)
        trophy_delta = overlay.get("trophy_delta")
        if trophy_delta is None:
            trophy_delta = history_session.get("trophy_delta", 0)

        return {
            "instance_id": self.instance_id,
            "state": status.get("state") or "idle",
            "is_running": bool(status.get("is_running")),
            "trophies": trophies,
            "wins": int(wins or 0),
            "losses": int(losses or 0),
            "trophy_delta": int(trophy_delta or 0),
            "current_brawler": overlay.get("current_brawler") or current.get("brawler"),
            "mode": mode,
            "player_tag": self.get_player_tag(),
            "session_started_at": status.get("session_started_at"),
        }

    def get_match_history(self, limit: int = 10) -> list[dict[str, Any]]:
        bounded = clamp_history_limit(limit)
        if self.matches is not None:
            return select_recent_matches(self.matches, bounded)
        service = self.data_service
        if service is not None and hasattr(service, "get_recent_matches"):
            return list(service.get_recent_matches(bounded))
        return []

    def restart_adb(self) -> dict[str, Any]:
        service = self.adb_service if self.adb_service is not None else AdbScrcpyService()
        return service.restart(self)


class InstanceRegistry:
    def __init__(self):
        self._instances: dict[str, BotInstance] = {}
        self._active_id: str | None = None
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._instances)

    def register(self, instance: BotInstance, activate: bool = False) -> BotInstance:
        with self._lock:
            self._instances[instance.instance_id] = instance
            if activate or self._active_id is None:
                self._active_id = instance.instance_id
        return instance

    def get(self, instance_id: str) -> BotInstance | None:
        return self._instances.get(instance_id)

    def activate(self, instance_id: str) -> BotInstance:
        with self._lock:
            instance = self._instances.get(instance_id)
            if instance is None:
                raise KeyError(instance_id)
            self._active_id = instance_id
            return instance

    def get_active(self) -> BotInstance:
        active_id = self._active_id
        if active_id is None or active_id not in self._instances:
            raise RuntimeError("No active bot instance is selected.")
        return self._instances[active_id]

    def list_instances(self) -> list[dict[str, Any]]:
        active_id = self._active_id
        listed = []
        for instance in self._instances.values():
            listed.append({
                "id": instance.instance_id,
                "active": instance.instance_id == active_id,
                "player_tag": instance.get_player_tag(),
            })
        return listed
