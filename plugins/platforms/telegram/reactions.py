"""Canonical Telegram standard reaction emoji."""


def _standard_reactions() -> tuple[str, ...]:
    try:
        from telegram.constants import ReactionEmoji

        return tuple(
            value
            for item in ReactionEmoji
            if (value := str(getattr(item, "value", item)))
        )
    except (ImportError, TypeError):
        return ()


def _without_presentation(value: str) -> str:
    return value.replace("\ufe0e", "").replace("\ufe0f", "")


def canonical_standard_emoji(emoji: str) -> str | None:
    """Return Telegram's canonical form for a standard reaction emoji."""
    raw = str(emoji or "").strip()
    if not raw:
        return None

    allowed = _standard_reactions()
    if not allowed:
        # Older PTB versions still validate the value at the Bot API boundary.
        return raw if "\u200d" in raw else raw.rstrip("\ufe0e\ufe0f") or raw
    if raw in allowed:
        return raw

    normalized = _without_presentation(raw)
    return next(
        (value for value in allowed if _without_presentation(value) == normalized),
        None,
    )
