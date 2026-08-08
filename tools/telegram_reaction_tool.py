"""Intentional reaction tool for the bundled Telegram plugin."""

import asyncio
import concurrent.futures
import json

from gateway.session_context import get_session_env
from plugins.platforms.telegram.reactions import canonical_standard_emoji
from tools.registry import registry, tool_error

_TIMEOUT = 10.0
SCHEMA = {
    "name": "telegram_react",
    "description": (
        "Add one standard emoji reaction to this turn's inbound Telegram message. "
        "Use it as a response, not a processing-status signal. The target is fixed; "
        "do not describe the reaction in text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "emoji": {
                "type": "string",
                "description": "A standard Telegram reaction, such as 👍, ❤, 😂, or 👀.",
            }
        },
        "required": ["emoji"],
        "additionalProperties": False,
    },
}


def telegram_reaction_tool(emoji: str) -> str:
    emoji = (emoji or "").strip()
    if not emoji:
        return tool_error("An emoji is required.")
    if str(get_session_env("HERMES_SESSION_PLATFORM", "")).lower() != "telegram":
        return tool_error("This tool is only available in a Telegram session.")
    canonical = canonical_standard_emoji(emoji)
    if canonical is None:
        return tool_error("Telegram does not support that standard reaction emoji.")
    emoji = canonical
    session_key = get_session_env("HERMES_SESSION_KEY", "")
    chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "")
    message_id = get_session_env("HERMES_SESSION_MESSAGE_ID", "")
    if not (session_key and chat_id and message_id):
        return tool_error("The current Telegram message context is unavailable.")

    try:
        from gateway.config import Platform
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        if runner is None:
            return tool_error("The Telegram gateway is not connected.")
        source = runner._get_cached_session_source(session_key)
        if (
            source is None
            or source.platform != Platform.TELEGRAM
            or str(source.chat_id) != str(chat_id)
        ):
            return tool_error("The current Telegram session is unavailable.")
        adapter = runner._adapter_for_source(source)
        react = getattr(adapter, "add_current_reaction", None)
        loop = getattr(runner, "_gateway_loop", None)
        if not callable(react) or loop is None or not loop.is_running() or loop.is_closed():
            return tool_error("The Telegram gateway is not connected.")
        try:
            if asyncio.get_running_loop() is loop:
                return tool_error("Telegram reaction is unavailable on the gateway loop.")
        except RuntimeError:
            pass

        coroutine = react(chat_id, message_id, emoji)
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        except Exception:
            coroutine.close()
            return tool_error("Telegram reaction failed.")
        try:
            success = future.result(timeout=_TIMEOUT)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return tool_error("Telegram reaction timed out.")
        except Exception:
            future.cancel()
            return tool_error("Telegram reaction failed.")
    except Exception:
        return tool_error("Telegram reaction failed.")

    if not success:
        return tool_error("The current Telegram message context is unavailable.")
    return json.dumps({"success": True})


registry.register(
    name="telegram_react",
    toolset="telegram_reactions",
    schema=SCHEMA,
    handler=lambda args, **_: telegram_reaction_tool(args.get("emoji", "")),
    emoji="💛",
)
