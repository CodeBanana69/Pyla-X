from __future__ import annotations

import collections
import re
import sys
import threading
import time
import traceback
from typing import Any, Callable


GLOBAL_LOGS = collections.deque(maxlen=2000)
GLOBAL_LOGS_LOCK = threading.Lock()


class ThreadFilterStream:
    ANSI_CLEAN_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

    def __init__(self, original_stream, prefix_filter="pyla-", is_stderr=False):
        self.original_stream = original_stream
        self.prefix_filter = prefix_filter
        self.is_stderr = is_stderr
        self.thread_buffers: dict[str, list[str]] = {}

    def write(self, text):
        self.original_stream.write(text)
        if not text:
            return

        thread_name = threading.current_thread().name
        if not thread_name.startswith(self.prefix_filter):
            return

        with GLOBAL_LOGS_LOCK:
            self.thread_buffers.setdefault(thread_name, []).append(text)
            combined = "".join(self.thread_buffers[thread_name])
            if "\n" not in combined:
                return

            lines = combined.split("\n")
            self.thread_buffers[thread_name] = [lines[-1]]
            for line in lines[:-1]:
                log_line = self.ANSI_CLEAN_RE.sub("", line)
                if self.is_stderr:
                    log_line = f"[stderr] {log_line}"
                GLOBAL_LOGS.append(log_line)

    def flush(self):
        self.original_stream.flush()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)


if not getattr(sys.stdout, "_is_pyla_redirected", False):
    sys.stdout = ThreadFilterStream(sys.stdout, prefix_filter="pyla-")
    sys.stdout._is_pyla_redirected = True

if not getattr(sys.stderr, "_is_pyla_redirected", False):
    sys.stderr = ThreadFilterStream(sys.stderr, prefix_filter="pyla-", is_stderr=True)
    sys.stderr._is_pyla_redirected = True


class RuntimeControl:
    def __init__(self, state_callback: Callable[[str], None]):
        self._state_callback = state_callback
        self._stop_event = threading.Event()
        self._pause_requested = threading.Event()
        self._immediate = threading.Event()

    def request_pause(self, immediate: bool = False):
        self._pause_requested.set()
        if immediate:
            self._immediate.set()
        else:
            self._immediate.clear()

    def resume(self):
        self._pause_requested.clear()
        self._immediate.clear()

    def request_stop(self, immediate: bool = False):
        keep_immediate = bool(immediate) or self._immediate.is_set()
        self._stop_event.set()
        self._pause_requested.clear()
        if keep_immediate:
            self._immediate.set()
        else:
            self._immediate.clear()

    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def should_pause(self) -> bool:
        return self._pause_requested.is_set() and not self._stop_event.is_set()

    def interrupts_immediately(self) -> bool:
        return self._immediate.is_set() and (self.should_stop() or self.should_pause())

    def mark_running(self):
        self._state_callback("running")

    def mark_paused(self):
        self._state_callback("paused")


class RuntimeManager:
    def __init__(self, pyla_main):
        self.pyla_main = pyla_main
        self._thread: threading.Thread | None = None
        self.rt_control: RuntimeControl | None = None
        self._lock = threading.Lock()
        self._state = "idle"
        self._last_error = ""
        self._session_started_at: float | None = None
        self.queue_provider: Callable[[], list[dict[str, Any]]] | None = None
        self._auth_provider: Callable[[], dict[str, Any]] | None = None

    def _set_state(self, state: str):
        with self._lock:
            self._state = state

    def configure_start_gate(
            self,
            queue_provider: Callable[[], list[dict[str, Any]]],
            auth_provider: Callable[[], dict[str, Any]],
    ):
        self.queue_provider = queue_provider
        self._auth_provider = auth_provider

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            thread_alive = self._thread.is_alive() if self._thread else False
            if not thread_alive and self._state != "error":
                self._state = "idle"
                self._thread = None
                self.rt_control = None
                self._session_started_at = None
            immediate = bool(thread_alive and self.rt_control and self.rt_control.interrupts_immediately())
            return {
                "state": self._state,
                "is_running": thread_alive,
                "last_error": self._last_error,
                "session_started_at": self._session_started_at if thread_alive else None,
                "immediate": immediate,
            }

    def start(self, queue_data: list[dict[str, Any]], discord_bot) -> dict[str, Any]:
        with self._lock:
            thread_alive = self._thread.is_alive() if self._thread else False

            if thread_alive:
                if self._state == "paused" and self.rt_control:
                    self.rt_control.resume()
                    self._state = "running"
                    self._last_error = ""
                    return {"ok": True, "message": "Pyla resumed."}
                return {"ok": False, "message": f"Pyla cannot start while state is {self._state}."}

            self.rt_control = RuntimeControl(self._set_state)
            self._state = "running"
            self._last_error = ""
            self._session_started_at = time.time()
            self._thread = threading.Thread(
                target=self._run_worker,
                args=(queue_data, self.rt_control, discord_bot),
                daemon=True,
                name="pyla-runtime",
            )
            self._thread.start()
            return {"ok": True, "message": "Pyla started."}

    def start_current_queue(self, discord_bot) -> dict[str, Any]:
        if not self.queue_provider or not self._auth_provider:
            return {
                "ok": False,
                "message": "Runtime start gate is not configured.",
                "code": "START_GATE_NOT_CONFIGURED",
            }

        runtime_state = self.get_status()["state"]
        queue_data = self.queue_provider()
        if runtime_state != "paused" and not queue_data:
            return {"ok": False, "message": "Queue is empty.", "code": "EMPTY_QUEUE"}

        auth_state = self._auth_provider()
        if auth_state.get("required") and not auth_state.get("authenticated"):
            return {
                "ok": False,
                "message": auth_state.get("message") or "Login required before starting.",
                "code": auth_state.get("code") or "LOGIN_REQUIRED",
                "auth": auth_state,
            }

        return self.start(queue_data, discord_bot)

    def _run_worker(self, queue_data: list[dict[str, Any]], control: RuntimeControl, discord_bot):
        try:
            self.pyla_main(discord_bot, queue_data, runtime_control=control)
            with self._lock:
                if self._state != "error":
                    self._state = "idle"
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0
            with self._lock:
                if code in (0, None):
                    self._state = "idle"
                    self._last_error = ""
                else:
                    self._state = "error"
                    self._last_error = f"Pyla exited with code {code}."
        except Exception as exc:
            with self._lock:
                self._state = "error"
                self._last_error = str(exc)
            print(str(exc))
            traceback.print_exc()
        finally:
            with self._lock:
                self._thread = None
                self.rt_control = None
                self._session_started_at = None

    def pause(self, immediate: bool = False) -> dict[str, Any]:
        with self._lock:
            thread_alive = self._thread.is_alive() if self._thread else False
            if not thread_alive or not self.rt_control:
                return {"ok": False, "message": "Pyla is not running.", "immediate": False}

            if self._state == "running":
                self.rt_control.request_pause(immediate=immediate)
                self._state = "pausing"
                if self.rt_control.interrupts_immediately():
                    return {
                        "ok": True,
                        "message": "Force pause requested. Pyla will pause immediately.",
                        "immediate": True,
                    }
                return {
                    "ok": True,
                    "message": "Pause requested. Pyla will pause in the lobby.",
                    "immediate": False,
                }

            if self._state == "pausing":
                if immediate:
                    self.rt_control.request_pause(immediate=True)
                    return {
                        "ok": True,
                        "message": "Force pause requested. Pyla will pause immediately.",
                        "immediate": True,
                    }
                return {
                    "ok": True,
                    "message": "Pause already requested.",
                    "immediate": self.rt_control.interrupts_immediately(),
                }

            if self._state == "paused":
                return {"ok": True, "message": "Pause already requested.", "immediate": False}

            return {"ok": False, "message": f"Pyla cannot pause while state is {self._state}.", "immediate": False}

    def stop(self, immediate: bool = False) -> dict[str, Any]:
        with self._lock:
            thread_alive = self._thread.is_alive() if self._thread else False
            if not thread_alive or not self.rt_control:
                self._state = "idle"
                self._session_started_at = None
                return {"ok": True, "message": "Pyla is already stopped.", "immediate": False}

            thread = self._thread
            was_paused = self._state == "paused"
            self.rt_control.request_stop(immediate=immediate)
            immediate_now = self.rt_control.interrupts_immediately()
            self._state = "stopping"
            message = (
                "Force stop requested. Pyla is shutting down."
                if immediate_now
                else "Stop requested. Pyla is shutting down."
            )

        if was_paused and thread:
            thread.join(timeout=2)
            if not thread.is_alive():
                with self._lock:
                    stopped_state = self._state
                    self._thread = None
                    self.rt_control = None
                    self._session_started_at = None
                    if self._state != "error":
                        self._state = "idle"
                        stopped_state = "idle"
                if stopped_state == "error":
                    return {"ok": False, "message": self._last_error or "Pyla stopped with an error.", "immediate": immediate_now}
                return {"ok": True, "message": "Pyla stopped.", "immediate": immediate_now}

        return {"ok": True, "message": message, "immediate": immediate_now}

    def get_logs(self) -> list[str]:
        with GLOBAL_LOGS_LOCK:
            return list(GLOBAL_LOGS)

    def clear_logs(self):
        with GLOBAL_LOGS_LOCK:
            GLOBAL_LOGS.clear()
