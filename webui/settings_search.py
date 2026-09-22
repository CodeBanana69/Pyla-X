"""Match settings by keyword against their label, key, and description."""

from __future__ import annotations

from typing import Any, Iterable


def normalize_settings_query(query: str | None) -> str:
    return " ".join(str(query or "").lower().split())


def setting_search_text(setting: dict[str, Any]) -> str:
    parts: list[str] = [
        str(setting.get("section") or ""),
        str(setting.get("label") or ""),
        str(setting.get("key") or ""),
        str(setting.get("help") or ""),
        str(setting.get("description") or ""),
    ]
    for option in setting.get("options") or []:
        if isinstance(option, dict):
            parts.append(str(option.get("label") or ""))
            parts.append(str(option.get("value") or ""))
        else:
            parts.append(str(option))
    return " ".join(part for part in parts if part).lower()


def setting_matches(setting: dict[str, Any], query: str | None) -> bool:
    normalized = normalize_settings_query(query)
    if not normalized:
        return True
    haystack = setting_search_text(setting)
    return all(token in haystack for token in normalized.split())


def filter_settings(settings: Iterable[dict[str, Any]], query: str | None) -> list[dict[str, Any]]:
    return [setting for setting in settings if setting_matches(setting, query)]
