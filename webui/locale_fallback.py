"""Fallback rules shared with the web UI language catalogs.

Unknown languages and missing strings stay in the source language (English).
"""

from __future__ import annotations


def translate(catalogs: dict, language: str | None, text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    if language in (None, "", "en"):
        return text
    catalog = catalogs.get(language) if isinstance(catalogs, dict) else None
    if not isinstance(catalog, dict):
        return text
    translated = catalog.get(text)
    if translated is None or translated == "":
        return text
    return translated
