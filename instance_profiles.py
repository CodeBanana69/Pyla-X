"""Per-instance profiles for running several emulators from one project folder.

Each profile keeps its own ADB port, playstyle, brawler queue, and match
history. Other settings are synced across profiles unless listed in
``unsynced_keys``.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

DEFAULT_PROFILE_ID = "default"
DEFAULT_PROFILE_NAME = "Public profile"
INSTANCES_CONFIG_RELATIVE = Path("cfg") / "instances_config.json"
HISTORY_HEADER = (
    "date_time,brawler_name,result,current_trophies,trophy_delta,"
    "new_winstreak,playstyle_hash,playstyle_name,playstyle_gamemodes,"
    "playstyle_brawlers,pyla_version,power_level\n"
)

# These always belong to one instance. Queue and history are files, not keys.
ALWAYS_PER_INSTANCE = {("bot", "current_playstyle")}
DEFAULT_UNSYNCED_KEYS = ("general.emulator_port", "general.player_tag")

SECTION_FILES = {
    "general": "cfg/general_config.toml",
    "bot": "cfg/bot_config.toml",
    "timers": "cfg/time_tresholds.toml",
    "debug": "cfg/debug_settings.toml",
    "webhook": "cfg/webhook_config.toml",
}

_registry: "InstanceRegistry | None" = None
_registry_lock = threading.Lock()
_bound = threading.local()


def section_for_config_path(file_path: str) -> str | None:
    normalized = str(file_path or "").replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    marker = "cfg/"
    index = normalized.rfind(marker)
    if index != -1:
        normalized = normalized[index:]
    for section, relative in SECTION_FILES.items():
        if normalized == relative:
            return section
    return None


def setting_token(section: str, key: str) -> str:
    return f"{section}.{key}"


def bind_profile(profile_id: str | None) -> None:
    """Bind config, queue, and history reads on this thread to one profile."""
    if profile_id:
        _bound.profile_id = str(profile_id)
    elif hasattr(_bound, "profile_id"):
        del _bound.profile_id


def clear_bound_profile() -> None:
    if hasattr(_bound, "profile_id"):
        del _bound.profile_id


def current_bound_profile() -> str | None:
    profile_id = getattr(_bound, "profile_id", None)
    return str(profile_id) if profile_id else None


def get_registry() -> "InstanceRegistry":
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                from utils import PROJECT_ROOT
                _registry = InstanceRegistry(PROJECT_ROOT)
    return _registry


def set_registry(registry: "InstanceRegistry | None") -> None:
    global _registry
    _registry = registry


class InstanceRegistry:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self._lock = threading.RLock()
        self._data = self._load()

    @property
    def config_path(self) -> Path:
        return self.root / INSTANCES_CONFIG_RELATIVE

    def active_id(self) -> str:
        with self._lock:
            return str(self._data.get("active_profile_id") or DEFAULT_PROFILE_ID)

    def list_profiles(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._public_profile(profile) for profile in self._data["profiles"]]

    def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        with self._lock:
            profile = self._find(profile_id)
            return self._public_profile(profile) if profile else None

    def is_synced(self, section: str, key: str) -> bool:
        if (section, key) in ALWAYS_PER_INSTANCE:
            return False
        with self._lock:
            return setting_token(section, key) not in set(self._data.get("unsynced_keys") or [])

    def sync_map(self, section: str, keys: list[str]) -> dict[str, bool]:
        return {key: self.is_synced(section, key) for key in keys}

    def set_active(self, profile_id: str) -> dict[str, Any]:
        with self._lock:
            profile = self._require(profile_id)
            self._data["active_profile_id"] = profile["id"]
            self._save()
            return self._public_profile(profile)

    def create_profile(self, name: str) -> dict[str, Any]:
        clean_name = _clean_name(name)
        with self._lock:
            source = self._require(self.active_id())
            profile_id = self._unique_id(_slug(clean_name))
            profile = {
                "id": profile_id,
                "name": clean_name,
                "adb_port": self._next_port(),
                "playstyle": source.get("playstyle") or "default_up.pyla",
                "overrides": json.loads(json.dumps(source.get("overrides") or {})),
            }
            if self.is_synced("general", "emulator_port"):
                profile["adb_port"] = source.get("adb_port")
            self._data["profiles"].append(profile)
            self._save()
            self.write_queue(profile_id, [])
            self.ensure_history(profile_id)
            return self._public_profile(profile)

    def rename_profile(self, profile_id: str, name: str) -> dict[str, Any]:
        clean_name = _clean_name(name)
        with self._lock:
            profile = self._require(profile_id)
            profile["name"] = clean_name
            self._save()
            return self._public_profile(profile)

    def delete_profile(self, profile_id: str) -> None:
        if profile_id == DEFAULT_PROFILE_ID:
            raise ValueError("The default profile cannot be deleted")
        with self._lock:
            profile = self._require(profile_id)
            self._data["profiles"] = [item for item in self._data["profiles"] if item["id"] != profile["id"]]
            if self._data.get("active_profile_id") == profile["id"]:
                self._data["active_profile_id"] = DEFAULT_PROFILE_ID
            self._save()
        self._delete_profile_files(profile_id)

    def set_playstyle(self, profile_id: str, filename: str) -> None:
        filename = str(filename or "").strip()
        if not filename:
            raise ValueError("Playstyle filename is required.")
        with self._lock:
            self._require(profile_id)["playstyle"] = filename
            self._save()

    def set_adb_port(self, profile_id: str, port: int, *, allow_duplicate: bool = False) -> int:
        parsed = int(port)
        if parsed < 1 or parsed > 65535:
            raise ValueError("ADB port must be between 1 and 65535.")
        with self._lock:
            profile = self._require(profile_id)
            if not allow_duplicate and not self.is_synced("general", "emulator_port"):
                for other in self._data["profiles"]:
                    if other["id"] != profile["id"] and other.get("adb_port") == parsed:
                        raise ValueError(f"Port {parsed} is already used by {other['name']}.")
            profile["adb_port"] = parsed
            self._save()
            return parsed

    def set_override(self, profile_id: str, section: str, key: str, value: Any) -> None:
        if (section, key) in ALWAYS_PER_INSTANCE:
            raise ValueError("That setting is stored on the profile itself.")
        if self.is_synced(section, key):
            raise ValueError("Synced settings cannot be overridden on one profile.")
        with self._lock:
            profile = self._require(profile_id)
            if section == "general" and key == "emulator_port":
                self.set_adb_port(profile_id, int(value))
                return
            profile.setdefault("overrides", {}).setdefault(section, {})[key] = value
            self._save()

    def clear_overrides_for_key(self, section: str, key: str) -> None:
        with self._lock:
            for profile in self._data["profiles"]:
                section_overrides = (profile.get("overrides") or {}).get(section) or {}
                section_overrides.pop(key, None)
            self._save()

    def set_key_synced(self, section: str, key: str, synced: bool, active_value: Any, shared_value: Any) -> str:
        """Update the sync flag. Returns ``write_shared`` or ``per_instance``."""
        if (section, key) in ALWAYS_PER_INSTANCE:
            raise ValueError("Playstyle is always configured per instance.")
        token = setting_token(section, key)
        with self._lock:
            unsynced = list(self._data.get("unsynced_keys") or [])
            if synced:
                self._data["unsynced_keys"] = [item for item in unsynced if item != token]
                if section == "general" and key == "emulator_port" and active_value is not None:
                    for profile in self._data["profiles"]:
                        profile["adb_port"] = int(active_value)
                else:
                    for profile in self._data["profiles"]:
                        ((profile.get("overrides") or {}).get(section) or {}).pop(key, None)
                self._save()
                return "write_shared"
            if token not in unsynced:
                unsynced.append(token)
            self._data["unsynced_keys"] = unsynced
            if section == "general" and key == "emulator_port":
                for profile in self._data["profiles"]:
                    if profile.get("adb_port") is None and shared_value not in (None, ""):
                        profile["adb_port"] = int(shared_value)
            else:
                for profile in self._data["profiles"]:
                    overrides = profile.setdefault("overrides", {}).setdefault(section, {})
                    if key not in overrides:
                        overrides[key] = shared_value
            self._save()
            return "per_instance"

    def apply_file_overrides(self, profile_id: str, file_path: str, data: dict[str, Any]) -> dict[str, Any]:
        section = section_for_config_path(file_path)
        if section is None or not isinstance(data, dict):
            return data
        with self._lock:
            profile = self._find(profile_id)
            if profile is None:
                return data
            merged = dict(data)
            overrides = (profile.get("overrides") or {}).get(section) or {}
            for key, value in overrides.items():
                if not self.is_synced(section, key):
                    merged[key] = value
            if section == "general" and not self.is_synced("general", "emulator_port"):
                if profile.get("adb_port") is not None:
                    merged["emulator_port"] = profile["adb_port"]
            if section == "bot" and profile.get("playstyle"):
                merged["current_playstyle"] = profile["playstyle"]
            return merged

    def effective_adb_port(self, profile_id: str) -> int | None:
        with self._lock:
            profile = self._require(profile_id)
            if self.is_synced("general", "emulator_port"):
                shared = self._read_toml(SECTION_FILES["general"]).get("emulator_port")
                if shared not in (None, ""):
                    return int(shared)
            port = profile.get("adb_port")
            return int(port) if port not in (None, "") else None

    def queue_path(self, profile_id: str) -> Path:
        if profile_id == DEFAULT_PROFILE_ID:
            return self.root / "latest_brawler_data.json"
        return self.root / "cfg" / profile_id / "queue.json"

    def history_path(self, profile_id: str) -> Path:
        if profile_id == DEFAULT_PROFILE_ID:
            return self.root / "cfg" / "match_history.csv"
        return self.root / "cfg" / profile_id / "match_history.csv"

    def read_queue(self, profile_id: str) -> list:
        path = self.queue_path(profile_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def write_queue(self, profile_id: str, items: list) -> None:
        path = self.queue_path(profile_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, indent=4), encoding="utf-8")

    def ensure_history(self, profile_id: str) -> Path:
        path = self.history_path(profile_id)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(HISTORY_HEADER, encoding="utf-8")
        return path

    def _public_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": profile["id"],
            "name": profile["name"],
            "adb_port": profile.get("adb_port"),
            "playstyle": profile.get("playstyle") or "default_up.pyla",
            "is_default": profile["id"] == DEFAULT_PROFILE_ID,
            "overrides": json.loads(json.dumps(profile.get("overrides") or {})),
        }

    def _load(self) -> dict[str, Any]:
        if self.config_path.exists():
            try:
                loaded = json.loads(self.config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = {}
        else:
            loaded = {}
        profiles = loaded.get("profiles") if isinstance(loaded, dict) else None
        if not isinstance(profiles, list) or not profiles:
            profiles = [self._default_profile()]
        if not any(item.get("id") == DEFAULT_PROFILE_ID for item in profiles):
            profiles.insert(0, self._default_profile())
        normalized = []
        for item in profiles:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            normalized.append({
                "id": str(item["id"]),
                "name": str(item.get("name") or item["id"]),
                "adb_port": _optional_port(item.get("adb_port")),
                "playstyle": str(item.get("playstyle") or "default_up.pyla"),
                "overrides": item.get("overrides") if isinstance(item.get("overrides"), dict) else {},
            })
        active = str((loaded or {}).get("active_profile_id") or DEFAULT_PROFILE_ID)
        if not any(item["id"] == active for item in normalized):
            active = DEFAULT_PROFILE_ID
        unsynced = (loaded or {}).get("unsynced_keys")
        if not isinstance(unsynced, list):
            unsynced = list(DEFAULT_UNSYNCED_KEYS)
        data = {
            "active_profile_id": active,
            "unsynced_keys": [str(item) for item in unsynced],
            "profiles": normalized,
        }
        self._data = data
        if not self.config_path.exists():
            self._save()
        return data

    def _default_profile(self) -> dict[str, Any]:
        general = self._read_toml(SECTION_FILES["general"])
        bot = self._read_toml(SECTION_FILES["bot"])
        return {
            "id": DEFAULT_PROFILE_ID,
            "name": DEFAULT_PROFILE_NAME,
            "adb_port": _optional_port(general.get("emulator_port")) or 5037,
            "playstyle": str(bot.get("current_playstyle") or "default_up.pyla"),
            "overrides": {},
        }

    def _save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._data, indent=2)
        temp_path = self.config_path.with_suffix(".json.tmp")
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(self.config_path)

    def _find(self, profile_id: str) -> dict[str, Any] | None:
        for profile in self._data["profiles"]:
            if profile["id"] == profile_id:
                return profile
        return None

    def _require(self, profile_id: str) -> dict[str, Any]:
        profile = self._find(profile_id)
        if profile is None:
            raise ValueError(f"Unknown profile '{profile_id}'.")
        return profile

    def _unique_id(self, slug: str) -> str:
        base = slug or "profile"
        if base == DEFAULT_PROFILE_ID:
            base = "profile"
        candidate = base
        existing = {item["id"] for item in self._data["profiles"]}
        suffix = 2
        while candidate in existing:
            candidate = f"{base}-{suffix}"
            suffix += 1
        return candidate

    def _next_port(self) -> int:
        used = {item.get("adb_port") for item in self._data["profiles"] if item.get("adb_port")}
        from port_finder import SHALLOW_ADB_PORTS
        for port in SHALLOW_ADB_PORTS:
            if port not in used:
                return int(port)
        return max(used or {5037}) + 1

    def _read_toml(self, relative: str) -> dict[str, Any]:
        path = self.root / relative
        if not path.exists():
            return {}
        try:
            import toml
            with path.open("r", encoding="utf-8") as handle:
                data = toml.load(handle)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _delete_profile_files(self, profile_id: str) -> None:
        if profile_id == DEFAULT_PROFILE_ID:
            return
        folder = self.root / "cfg" / profile_id
        for relative in (self.queue_path(profile_id), self.history_path(profile_id)):
            if relative.exists() and relative.is_file():
                relative.unlink()
        if folder.exists() and folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


def _clean_name(name: str) -> str:
    clean = re.sub(r"\s+", " ", str(name or "")).strip()
    if not clean:
        raise ValueError("Profile name is required.")
    if len(clean) > 40:
        raise ValueError("Profile name must be 40 characters or fewer.")
    return clean


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:32] or "profile"


def _optional_port(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
