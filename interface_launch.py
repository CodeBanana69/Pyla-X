"""Startup interface selection for the local PylaAI application.

The UI can run as a native pywebview desktop window, in the system browser,
or headless (the local web server stays available and nothing is opened).

Command-line flags override ``interface_mode`` in ``cfg/general_config.toml``:

* ``--desktop`` opens the integrated pywebview window
* ``--web`` / ``--browser`` / ``--no-webapp`` opens the system browser
* ``--headless`` serves the UI without opening a window or browser
* ``--no-console`` hides a console that belongs only to PylaAI
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


INTERFACE_MODES = frozenset({"desktop", "browser", "headless"})


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="PylaAI",
        description="PylaAI, the best free and open source brawl stars bot.",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Hide PylaAI's own console and write output to a log file.",
    )
    interface_group = parser.add_mutually_exclusive_group()
    interface_group.add_argument(
        "--desktop",
        dest="interface_mode",
        action="store_const",
        const="desktop",
        help="Force the UI to open in the integrated pywebview window.",
    )
    interface_group.add_argument(
        "--web",
        "--browser",
        "--no-webapp",
        dest="interface_mode",
        action="store_const",
        const="browser",
        help="Force the UI to open in the system browser instead of pywebview.",
    )
    interface_group.add_argument(
        "--headless",
        dest="interface_mode",
        action="store_const",
        const="headless",
        help="Force headless mode: serve the local web UI without opening it.",
    )
    args, _unknown_args = parser.parse_known_args(argv)
    return args


def load_saved_interface_mode(config_path=None):
    path = Path(config_path) if config_path is not None else Path.cwd() / "cfg" / "general_config.toml"
    try:
        with path.open("rb") as config_file:
            configured_mode = str(tomllib.load(config_file).get("interface_mode", "desktop")).strip().lower()
    except (OSError, tomllib.TOMLDecodeError) as error:
        print(f"Could not read interface_mode from {path}: {error}. Using desktop mode.")
        return "desktop"

    if configured_mode not in INTERFACE_MODES:
        print(f"Unknown interface_mode {configured_mode!r} in {path}. Using desktop mode.")
        return "desktop"
    return configured_mode


def resolve_interface_mode(cli_args, config_path=None):
    return cli_args.interface_mode or load_saved_interface_mode(config_path)
