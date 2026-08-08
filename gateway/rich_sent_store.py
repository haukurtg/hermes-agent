"""Local index of text we've sent via ``sendRichMessage`` (Bot API 10.1).

Telegram does NOT echo a rich message's content back in ``reply_to_message``
when a user replies to it (verified: ``.text``/``.caption`` empty,
``.api_kwargs`` None). So replies to the launchd briefings / any rich send
arrive with no quotable text and the agent is blind to what was referenced.

Fix: remember ``message_id -> text`` at send time, look it up by
``reply_to_id`` on inbound. This module is the single source of truth for that
index.

Best-effort and dependency-free: every operation swallows errors and degrades
to a no-op / ``None`` so it can never break a send or an inbound message.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

from utils import atomic_json_write

_MAX_ENTRIES = 1000
_MAX_TEXT_CHARS = 2000
_LOCK = threading.Lock()


def _store_path() -> str:
    # Resolve via get_hermes_home() so the active profile override is honored.
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    return os.path.join(str(home), "state", "rich_sent_index.json")


def _key(chat_id, message_id) -> str:
    return f"{chat_id}:{message_id}"


def _timestamp(entry) -> float:
    try:
        return float(entry.get("ts", 0)) if isinstance(entry, dict) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _store_paths() -> list[tuple[str | None, str]]:
    """Return default, current and named-profile index paths."""
    from hermes_constants import get_default_hermes_root, get_hermes_home

    root, current = get_default_hermes_root(), get_hermes_home()
    paths = [
        (None, root / "state" / "rich_sent_index.json"),
        *((home.name, home / "state" / "rich_sent_index.json")
          for home in sorted(root.glob("profiles/*"))[:128] if home.is_dir()),
    ]
    profile = current.name if current.parent == root / "profiles" else None
    paths.append((profile, _store_path()))
    return list({os.path.abspath(path): (profile, os.fspath(path))
                 for profile, path in paths}.values())


def record(
    chat_id, message_id, text: Optional[str], *, thread_id=None, sender_id=None
) -> None:
    with _LOCK:
        _record_unlocked(
            chat_id, message_id, text, thread_id=thread_id, sender_id=sender_id
        )


def _record_unlocked(
    chat_id, message_id, text: Optional[str], *, thread_id=None, sender_id=None
) -> None:
    """Persist text and optional reaction-routing metadata."""
    if message_id is None or chat_id is None:
        return
    path = _store_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                data = {}
        except (FileNotFoundError, ValueError):
            data = {}
        key = _key(chat_id, message_id)
        previous = data.get(key)
        entry = dict(previous) if isinstance(previous, dict) else {}
        entry.update({"t": (text or "")[:_MAX_TEXT_CHARS], "ts": int(time.time())})
        if thread_id not in {None, ""}:
            entry["thread_id"] = str(thread_id)
        if sender_id not in {None, ""}:
            entry["sender_id"] = str(sender_id)
        data[key] = entry
        # Trim oldest by timestamp when over cap.
        if len(data) > _MAX_ENTRIES:
            for k, _ in sorted(
                data.items(), key=lambda kv: _timestamp(kv[1])
            )[: len(data) - _MAX_ENTRIES]:
                data.pop(k, None)
        atomic_json_write(path, data)
    except Exception:
        return


def lookup_entry(chat_id, message_id, *, all_profiles: bool = False) -> Optional[dict]:
    """Return one unambiguous entry, optionally searching every profile."""
    if message_id is None or chat_id is None:
        return None
    matches = []
    for profile, path in _store_paths() if all_profiles else [(None, _store_path())]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                entry = json.load(fh).get(_key(chat_id, message_id))
            if isinstance(entry, dict):
                entry = dict(entry)
                entry["profile"] = profile
                matches.append(entry)
        except (FileNotFoundError, OSError, ValueError, AttributeError, TypeError):
            pass
    return matches[0] if len(matches) == 1 else None


def lookup(chat_id, message_id) -> Optional[str]:
    """Return stored text for ``(chat_id, message_id)`` or ``None``."""
    entry = lookup_entry(chat_id, message_id)
    return (entry.get("t") or None) if entry else None
