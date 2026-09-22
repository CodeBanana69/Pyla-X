from __future__ import annotations

import collections
import re
import sys
import threading
import time
import traceback
from typing import Any, Callable


GLOBAL_LOGS = collections.deque(maxlen=2000)
INSTANCE_LOGS: dict[str, collections.deque] = {}
GLOBAL_LOGS_LOCK = threading.Lock()


def logs_for_thread(thread_name: str) -> collections.deque:
    bucket = INSTANCE_LOGS.get(thread_name)
    if bucket is None:
        bucket = collections.deque(maxlen=2000)
        INSTANCE_LOGS[thread_name] = bucket
    return bucket


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
                logs_for_thread(thread_name).append(log_line)
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

    def request_pause(self):
        self._pause_requested.set()

    def resume(self):
        self._pause_requested.clear()

    def request_stop(self):
        self._stop_event.set()
        self._pause_requested.clear()

    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def should_pause(self) -> bool:
        return self._pause_requested.is_set() and not self._stop_event.is_set()

    def mark_running(self):
        self._state_callback("running")

    def mark_paused(self):
        self._state_callback("paused")


class RuntimeManager:
    def __init__(self, pyla_main):
        self.pyla_main = pyla_main
        self._lock = threading.Lock()
        self._slots: dict[str, dict[str, Any]] = {}
        self._active_profile_provider: Callable[[], str] = lambda: "default"
        self.queue_provider: Callable[[], list[dict[str, Any]]] | None = None
        self._auth_provider: Callable[[], dict[str, Any]] | None = None

    def set_active_profile_provider(self, provider: Callable[[], str]) -> None:
        self._active_profile_provider = provider

    def _active_profile_id(self) -> str:
        try:
            return str(self._active_profile_provider() or "default")
        except Exception:
            return "default"

    def _slot(self, profile_id: str) -> dict[str, Any]:
        slot = self._slots.get(profile_id)
        if slot is None:
            slot = {
                "thread": None,
                "rt_control": None,
                "state": "idle",
                "last_error": "",
                "session_started_at": None,
            }
            self._slots[profile_id] = slot
        return slot

    def _set_slot_state(self, profile_id: str, state: str):
        with self._lock:
            self._slot(profile_id)["state"] = state

    def configure_start_gate(
            self,
            queue_provider: Callable[[], list[dict[str, Any]]],
            auth_provider: Callable[[], dict[str, Any]],
    ):
        self.queue_provider = queue_provider
        self._auth_provider = auth_provider

    def get_status(self, profile_id: str | None = None) -> dict[str, Any]:
        profile_id = profile_id or self._active_profile_id()
        with self._lock:
            slot = self._slot(profile_id)
            thread = slot["thread"]
            thread_alive = thread.is_alive() if thread else False
            if not thread_alive and slot["state"] != "error":
                slot["state"] = "idle"
                slot["thread"] = None
                slot["rt_control"] = None
                slot["session_started_at"] = None
            return {
                "profile_id": profile_id,
                "state": slot["state"],
                "is_running": thread_alive,
                "last_error": slot["last_error"],
                "session_started_at": slot["session_started_at"] if thread_alive else None,
            }

    def start(self, queue_data: list[dict[str, Any]], discord_bot, profile_id: str | None = None) -> dict[str, Any]:
        profile_id = profile_id or self._active_profile_id()
        with self._lock:
            slot = self._slot(profile_id)
            thread_alive = slot["thread"].is_alive() if slot["thread"] else False

            if thread_alive:
                if slot["state"] == "paused" and slot["rt_control"]:
                    slot["rt_control"].resume()
                    slot["state"] = "running"
                    slot["last_error"] = ""
                    return {"ok": True, "message": "Pyla resumed.", "profile_id": profile_id}
                return {"ok": False, "message": f"Pyla cannot start while state is {slot['state']}.", "profile_id": profile_id}

            control = RuntimeControl(lambda state: self._set_slot_state(profile_id, state))
            slot["rt_control"] = control
            slot["state"] = "running"
            slot["last_error"] = ""
            slot["session_started_at"] = time.time()
            slot["thread"] = threading.Thread(
                target=self._run_worker,
                args=(queue_data, control, discord_bot, profile_id),
                daemon=True,
                name=f"pyla-{profile_id}",
            )
            slot["thread"].start()
            return {"ok": True, "message": "Pyla started.", "profile_id": profile_id}

    def start_current_queue(self, discord_bot, profile_id: str | None = None) -> dict[str, Any]:
        if not self.queue_provider or not self._auth_provider:
            return {
                "ok": False,
                "message": "Runtime start gate is not configured.",
                "code": "START_GATE_NOT_CONFIGURED",
            }

        profile_id = profile_id or self._active_profile_id()
        runtime_state = self.get_status(profile_id)["state"]
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

        return self.start(queue_data, discord_bot, profile_id=profile_id)

    def _run_worker(self, queue_data: list[dict[str, Any]], control: RuntimeControl, discord_bot, profile_id: str):
        from instance_profiles import bind_profile, clear_bound_profile
        bind_profile(profile_id)
        print(f"Starting profile {profile_id}.")
        try:
            self.pyla_main(discord_bot, queue_data, runtime_control=control, profile_id=profile_id)
            with self._lock:
                slot = self._slot(profile_id)
                if slot["state"] != "error":
                    slot["state"] = "idle"
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0
            with self._lock:
                slot = self._slot(profile_id)
                if code in (0, None):
                    slot["state"] = "idle"
                    slot["last_error"] = ""
                else:
                    slot["state"] = "error"
                    slot["last_error"] = f"Pyla exited with code {code}."
        except Exception as exc:
            with self._lock:
                slot = self._slot(profile_id)
                slot["state"] = "error"
                slot["last_error"] = str(exc)
            print(str(exc))
            traceback.print_exc()
        finally:
            clear_bound_profile()
            with self._lock:
                slot = self._slot(profile_id)
                if slot.get("thread") is threading.current_thread():
                    slot["thread"] = None
                    slot["rt_control"] = None
                    slot["session_started_at"] = None

    def pause(self, profile_id: str | None = None) -> dict[str, Any]:
        profile_id = profile_id or self._active_profile_id()
        with self._lock:
            slot = self._slot(profile_id)
            thread_alive = slot["thread"].is_alive() if slot["thread"] else False
            if not thread_alive or not slot["rt_control"]:
                return {"ok": False, "message": "Pyla is not running.", "profile_id": profile_id}

            if slot["state"] == "running":
                slot["rt_control"].request_pause()
                slot["state"] = "pausing"
                return {"ok": True, "message": "Pause requested. Pyla will pause in the lobby.", "profile_id": profile_id}

            if slot["state"] in {"pausing", "paused"}:
                return {"ok": True, "message": "Pause already requested.", "profile_id": profile_id}

            return {"ok": False, "message": f"Pyla cannot pause while state is {slot['state']}.", "profile_id": profile_id}

    def stop(self, profile_id: str | None = None) -> dict[str, Any]:
        profile_id = profile_id or self._active_profile_id()
        with self._lock:
            slot = self._slot(profile_id)
            thread_alive = slot["thread"].is_alive() if slot["thread"] else False
            if not thread_alive or not slot["rt_control"]:
                slot["state"] = "idle"
                slot["session_started_at"] = None
                return {"ok": True, "message": "Pyla is already stopped.", "profile_id": profile_id}

            thread = slot["thread"]
            was_paused = slot["state"] == "paused"
            slot["rt_control"].request_stop()
            slot["state"] = "stopping"

        if was_paused and thread:
            thread.join(timeout=2)
            if not thread.is_alive():
                with self._lock:
                    slot = self._slot(profile_id)
                    stopped_state = slot["state"]
                    if slot.get("thread") is thread:
                        slot["thread"] = None
                        slot["rt_control"] = None
                        slot["session_started_at"] = None
                    if slot["state"] != "error":
                        slot["state"] = "idle"
                        stopped_state = "idle"
                if stopped_state == "error":
                    return {"ok": False, "message": slot["last_error"] or "Pyla stopped with an error.", "profile_id": profile_id}
                return {"ok": True, "message": "Pyla stopped.", "profile_id": profile_id}

        return {"ok": True, "message": "Stop requested. Pyla is shutting down.", "profile_id": profile_id}

    def get_logs(self, profile_id: str | None = None) -> list[str]:
        profile_id = profile_id or self._active_profile_id()
        thread_name = f"pyla-{profile_id}"
        with GLOBAL_LOGS_LOCK:
            return list(INSTANCE_LOGS.get(thread_name, ()))

    def clear_logs(self, profile_id: str | None = None):
        profile_id = profile_id or self._active_profile_id()
        thread_name = f"pyla-{profile_id}"
        with GLOBAL_LOGS_LOCK:
            bucket = INSTANCE_LOGS.get(thread_name)
            if bucket is not None:
                bucket.clear()
