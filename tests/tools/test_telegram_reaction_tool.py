"""Tests for Telegram's current-message reaction tool."""

import asyncio
import json
import subprocess
import sys
import threading

from gateway.config import Platform
from gateway.session import SessionSource


def _call_on_gateway(
    monkeypatch, handler, emoji="❤️", message_id="900", chat_id="-100"
):
    from tools import telegram_reaction_tool as module

    values = {
        "HERMES_SESSION_PLATFORM": "telegram",
        "HERMES_SESSION_KEY": "secondary-session",
        "HERMES_SESSION_CHAT_ID": chat_id,
        "HERMES_SESSION_MESSAGE_ID": message_id,
    }
    monkeypatch.setattr(
        module, "get_session_env", lambda name, default="": values.get(name, default)
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100",
        chat_type="group",
        user_id="42",
        profile="secondary",
    )
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def run_loop():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop)
    thread.start()
    ready.wait(timeout=2)
    adapter = type("Adapter", (), {"add_current_reaction": handler})()
    runner = type(
        "Runner",
        (),
        {
            "_gateway_loop": loop,
            "_get_cached_session_source": lambda self, key: source,
            "_adapter_for_source": lambda self, current: adapter,
        },
    )()
    import gateway.run

    monkeypatch.setattr(gateway.run, "_gateway_runner_ref", lambda: runner)
    try:
        return json.loads(module.telegram_reaction_tool(emoji)), loop, thread, source
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


def test_tool_uses_live_session_target_and_canonical_emoji(monkeypatch):
    observed = {}

    async def react(_adapter, chat_id, message_id, emoji):
        observed.update(
            chat_id=chat_id,
            message_id=message_id,
            emoji=emoji,
            loop=asyncio.get_running_loop(),
            thread=threading.get_ident(),
        )
        return True

    result, loop, thread, source = _call_on_gateway(monkeypatch, react)

    assert result == {"success": True}
    assert observed == {
        "chat_id": "-100",
        "message_id": "900",
        "emoji": "❤",
        "loop": loop,
        "thread": thread.ident,
    }
    assert source.profile == "secondary"


def test_tool_fails_when_turn_has_no_ordinary_message_target(monkeypatch):
    async def react(_adapter, chat_id, message_id, emoji):
        return False

    result, *_ = _call_on_gateway(monkeypatch, react, "👍", message_id="")
    assert result == {"error": "The current Telegram message context is unavailable."}
    result, *_ = _call_on_gateway(monkeypatch, react, "👍", chat_id="-200")
    assert result == {"error": "The current Telegram session is unavailable."}


def test_tool_rejects_non_telegram_and_unsupported_emoji(monkeypatch):
    from tools import telegram_reaction_tool as module

    monkeypatch.setattr(
        module,
        "get_session_env",
        lambda name, default="": "discord" if name == "HERMES_SESSION_PLATFORM" else default,
    )
    assert "Telegram session" in json.loads(module.telegram_reaction_tool("👍"))["error"]

    monkeypatch.setattr(
        module,
        "get_session_env",
        lambda name, default="": "telegram" if name == "HERMES_SESSION_PLATFORM" else default,
    )
    monkeypatch.setattr(module, "canonical_standard_emoji", lambda emoji: None)
    assert json.loads(module.telegram_reaction_tool("🧠")) == {
        "error": "Telegram does not support that standard reaction emoji."
    }


def test_every_installed_standard_reaction_and_display_alias_is_canonicalized():
    script = r'''
from telegram.constants import ReactionEmoji
from plugins.platforms.telegram.reactions import canonical_standard_emoji

reactions = [str(getattr(item, "value", item)) for item in ReactionEmoji]
for emoji in reactions:
    assert canonical_standard_emoji(emoji) == emoji
    if "\u200d" not in emoji and not emoji.endswith(("\ufe0e", "\ufe0f")):
        assert canonical_standard_emoji(f"{emoji}\ufe0f") == emoji
    if "\u200d" in emoji and "\ufe0f" in emoji:
        assert canonical_standard_emoji(emoji.replace("\ufe0f", "")) == emoji
assert canonical_standard_emoji("🧠") is None
'''
    subprocess.run([sys.executable, "-c", script], check=True)


def test_canonicalizer_falls_back_without_reaction_enum(monkeypatch):
    from plugins.platforms.telegram import reactions

    monkeypatch.setattr(reactions, "_standard_reactions", lambda: ())
    assert reactions.canonical_standard_emoji("❤️") == "❤"
    assert reactions.canonical_standard_emoji("👨‍💻") == "👨‍💻"


def test_tool_is_registered_only_for_telegram_and_kept_eager():
    from hermes_cli.tools_config import _get_platform_tools
    from tools import telegram_reaction_tool
    from tools.registry import registry
    from tools.tool_search import is_deferrable_tool_name
    from toolsets import resolve_toolset

    entry = registry.get_entry("telegram_react")
    assert entry is not None and entry.toolset == "telegram_reactions"
    assert set(telegram_reaction_tool.SCHEMA["parameters"]["properties"]) == {"emoji"}
    assert "telegram_react" in resolve_toolset("hermes-telegram")
    assert "telegram_react" in resolve_toolset("telegram_reactions")
    assert "telegram_react" not in resolve_toolset("hermes-cli")
    assert "telegram_react" not in resolve_toolset("hermes-discord")
    assert "telegram_reactions" in _get_platform_tools({}, "telegram")
    assert "telegram_reactions" not in _get_platform_tools({}, "discord")
    assert is_deferrable_tool_name("telegram_react") is False


def test_tool_does_not_emit_a_duplicate_progress_message():
    import queue

    from gateway.run import TurnRunner
    from gateway.turn_context import TurnContext

    progress = queue.Queue()
    context = TurnContext(
        progress_queue=progress,
        progress_mode="all",
        tool_progress_enabled=True,
        _run_still_current=lambda: True,
    )
    runner = TurnRunner(  # type: ignore[arg-type]
        type("Gateway", (), {"_adapter_for_source": lambda self, source: None})(),
        context,
    )

    runner.progress_callback(
        "tool.started", "telegram_react", preview="test", args={"emoji": "👍"}
    )

    assert progress.empty()
