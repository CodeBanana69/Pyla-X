"""Decide when a running bot should halt, pause, or keep playing.

Deferred stop and pause wait until the match returns to the lobby. Immediate
requests are checked on every loop tick and interrupt the current match.
"""


def control_action(game_state, *, stop_requested, pause_requested, immediate):
    """Return ``stop``, ``pause``, or ``None`` for this loop tick."""
    stop_requested = bool(stop_requested)
    pause_requested = bool(pause_requested)
    if immediate and stop_requested:
        return "stop"
    if immediate and pause_requested:
        return "pause"
    if game_state == "lobby" and stop_requested:
        return "stop"
    if game_state == "lobby" and pause_requested:
        return "pause"
    return None
