"""ADB emulator port discovery.

Shallow scans probe the ports commonly used by MuMu, BlueStacks, LDPlayer,
Nox, and generic ADB. Deep scans walk every TCP port from 1000 through 99999.
Callers inject ``probe`` so tests never need a real emulator.
"""

from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

# Same candidate order the bot already used when connecting without a preferred port.
SHALLOW_ADB_PORTS: tuple[int, ...] = tuple(
    [5137, 5555, 16384, 7555, 5635, 62001, 62025, 62026, 7556, 7565, 16416]
    + list(range(5556, 5566))
    + list(range(5565, 5756, 10))
    + list(range(16385, 16415))
)

DEEP_SCAN_START = 1000
DEEP_SCAN_END = 99999

Probe = Callable[[int], bool]


def iter_scan_ports(deep: bool = False) -> list[int]:
    """Return the ports a scan should probe, shallow candidates first."""
    if not deep:
        return _dedupe(SHALLOW_ADB_PORTS)
    return list(range(DEEP_SCAN_START, DEEP_SCAN_END + 1))


def scan_adb_ports(
    *,
    deep: bool = False,
    probe: Probe,
    host: str = "127.0.0.1",
    max_workers: int = 128,
    ports: Iterable[int] | None = None,
) -> list[dict]:
    """Probe ports and return the ones that accept a connection.

    ``probe`` receives a port and returns True when something is listening.
    Results are sorted by port and contain host, port, and serial.
    """
    selected = list(ports) if ports is not None else iter_scan_ports(deep)
    if not selected:
        return []

    workers = max(1, min(max_workers, len(selected)))
    open_ports: list[int] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for port, is_open in pool.map(lambda port: (port, bool(probe(port))), selected):
            if is_open:
                open_ports.append(port)

    devices = []
    for port in sorted(set(open_ports)):
        devices.append({
            "host": host,
            "port": port,
            "serial": f"{host}:{port}",
        })
    return devices


def tcp_port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.05) -> bool:
    """Return True when a local TCP port accepts a connection."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        return sock.connect_ex((host, int(port))) == 0
    except OSError:
        return False
    finally:
        sock.close()


def parse_adb_serial_port(serial: str) -> int | None:
    """Extract an ADB TCP port from ``127.0.0.1:16384`` or ``emulator-5554``."""
    text = str(serial or "").strip()
    if ":" in text:
        maybe_port = text.rsplit(":", 1)[1]
        if maybe_port.isdigit():
            return int(maybe_port)
    if text.startswith("emulator-"):
        console_port = text.removeprefix("emulator-")
        if console_port.isdigit():
            return int(console_port) + 1
    return None


def scan_emulator_ports(deep: bool = False, host: str = "127.0.0.1") -> list[dict]:
    """Scan localhost and best-effort ``adb connect`` every open port.

    ADB failures do not hide ports that already accepted a TCP connection, so
    the UI can still assign them.
    """
    devices = scan_adb_ports(deep=deep, probe=lambda port: tcp_port_open(port, host=host), host=host)
    _adb_connect_devices(devices, host=host)
    for extra in _list_online_adb_devices(host):
        if all(item["port"] != extra["port"] for item in devices):
            devices.append(extra)
    devices.sort(key=lambda item: item["port"])
    return devices


def _adb_connect_devices(devices: list[dict], host: str) -> None:
    try:
        from adbutils import adb
    except Exception:
        return
    for device in devices:
        target = f"{host}:{device['port']}"
        try:
            adb.connect(target)
            device["serial"] = target
        except Exception:
            continue


def _list_online_adb_devices(host: str) -> list[dict]:
    try:
        from adbutils import adb
        listed = adb.device_list()
    except Exception:
        return []
    found = []
    for device in listed:
        serial = str(getattr(device, "serial", "") or "")
        port = parse_adb_serial_port(serial)
        if port is None:
            continue
        found.append({"host": host, "port": port, "serial": serial})
    return found


def _dedupe(ports: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for port in ports:
        if port in seen:
            continue
        seen.add(port)
        ordered.append(port)
    return ordered
