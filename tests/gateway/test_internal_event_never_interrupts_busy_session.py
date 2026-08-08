"""Regression test: internal synthetic events must never interrupt a busy session.

Reported by @Heeervas (June 2026): an ``async_delegation`` completion from a
``delegate_task(background=true)`` subagent re-enters the originating gateway
session as an internal ``MessageEvent``. When that session was busy running a
turn, the completion was treated exactly like a user TEXT message and hit the
default ``busy_input_mode='interrupt'`` path — calling
``running_agent.interrupt()`` and aborting the active turn, plus sending a
"⚡ Interrupting current task" ack. The same shape affects background-process
completions (terminal ``notify_on_complete``), which also re-enter as internal
events.

The fix: ``_handle_active_session_busy_message`` returns ``False`` early for any
event with ``internal=True``, so the base adapter queues it silently (no
interrupt, no ack) and it cascades as a new turn after the current one finishes.
This preserves strict message-role alternation and the design invariant that a
completion surfaces as a NEW turn only when idle, never spliced into a running
turn.
"""

from __future__ import annotations

import sys
import threading
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Minimal telegram stubs only when the optional dependency is truly absent.
try:
    import telegram  # noqa: F401
except ImportError:
    _tg = types.ModuleType("telegram")
    _constants = types.ModuleType("telegram.constants")
    setattr(_tg, "constants", _constants)
    _ct = MagicMock()
    _ct.SUPERGROUP = "supergroup"
    _ct.GROUP = "group"
    _ct.PRIVATE = "private"
    setattr(_constants, "ChatType", _ct)
    sys.modules.setdefault("telegram", _tg)
    sys.modules.setdefault("telegram.constants", _constants)
    sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.platforms.base import (  # noqa: E402
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SessionSource,
    _is_deferred_followup_event,
    build_session_key,
)
from gateway.run import GatewayRunner  # noqa: E402


def _make_internal_event(text: str = "[async delegation completed]") -> MessageEvent:
    source = SessionSource(
        platform=MagicMock(value="telegram"),
        chat_id="123",
        chat_type="private",
        user_id="user1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg1",
        internal=True,
    )


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner.adapters = {}
    runner.config = MagicMock()
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    return runner


def _make_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = MagicMock(value="telegram")
    return adapter


def _make_running_parent() -> MagicMock:
    parent = MagicMock()
    parent._active_children = []  # no active subagents at completion time
    parent._active_children_lock = threading.Lock()
    parent.get_activity_summary.return_value = {
        "api_call_count": 4,
        "max_iterations": 60,
        "current_tool": "terminal",
    }
    return parent


@pytest.mark.asyncio
async def test_internal_event_does_not_interrupt_busy_session() -> None:
    """The async-delegation completion must not abort the active turn."""
    runner = _make_runner()
    runner._busy_input_mode = "interrupt"  # the default that caused the bug
    adapter = _make_adapter()
    event = _make_internal_event()
    sk = build_session_key(event.source)
    parent = _make_running_parent()
    runner._running_agents[sk] = parent
    runner.adapters[event.source.platform] = adapter

    handled = await runner._handle_active_session_busy_message(event, sk)

    # Returns False so the base adapter silently queues the internal event
    # as a cascading next turn — it must NOT be handled-with-interrupt here.
    assert handled is False
    # The active turn must survive.
    parent.interrupt.assert_not_called()
    # No "⚡ Interrupting current task" (or any) ack for a synthetic event.
    adapter._send_with_retry.assert_not_called()


@pytest.mark.asyncio
async def test_reaction_event_does_not_interrupt_or_approve_busy_session(
    monkeypatch,
) -> None:
    """A weak reaction signal must wait for the active turn to finish."""
    runner = _make_runner()
    runner._busy_input_mode = "interrupt"
    adapter = _make_adapter()
    event = _make_internal_event("[Telegram reaction: added 👍]")
    event.internal = False
    event.metadata = {"deferred_followup_event": True}
    sk = build_session_key(event.source)
    parent = _make_running_parent()
    parent._supports_active_turn_redirect = True
    parent.redirect.return_value = True
    runner._running_agents[sk] = parent
    runner.adapters[event.source.platform] = adapter
    approval = MagicMock(return_value=True)
    monkeypatch.setattr("tools.approval.has_blocking_approval", approval)

    handled = await runner._handle_active_session_busy_message(event, sk)

    # False hands the event back to BasePlatformAdapter, whose busy-session
    # path queues it silently for the next FIFO turn.
    assert handled is False
    parent.redirect.assert_not_called()
    parent.interrupt.assert_not_called()
    adapter._send_with_retry.assert_not_called()
    approval.assert_called_once_with(sk)


def test_deferred_followup_classification_covers_internal_and_reaction_events() -> None:
    internal = _make_internal_event()
    reaction = _make_internal_event("[Telegram reaction: added 👍]")
    reaction.internal = False
    reaction.metadata = {"deferred_followup_event": True}
    ordinary = _make_internal_event("ordinary text")
    ordinary.internal = False

    assert _is_deferred_followup_event(internal) is True
    assert _is_deferred_followup_event(reaction) is True
    assert _is_deferred_followup_event(ordinary) is False


@pytest.mark.asyncio
async def test_reaction_event_cannot_satisfy_pending_clarify(monkeypatch) -> None:
    """A reaction waits as a follow-up instead of becoming a clarify answer."""
    from tools import clarify_gateway

    event = _make_internal_event("[Telegram reaction: added 👍]")
    event.internal = False
    event.metadata = {"deferred_followup_event": True}
    session_key = build_session_key(event.source)

    adapter = MagicMock()
    adapter.name = "test"
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter._message_handler = AsyncMock()
    adapter._topic_recovery_fn = None
    adapter._active_sessions = {session_key: object()}
    adapter._pending_messages = {}
    adapter._heal_stale_session_lock = MagicMock()
    adapter._busy_session_handler = AsyncMock(return_value=False)
    adapter._is_queue_text_debounce_candidate = MagicMock(return_value=False)

    monkeypatch.setattr(
        clarify_gateway,
        "get_pending_for_session",
        lambda *_args, **_kwargs: object(),
    )

    await BasePlatformAdapter.handle_message(adapter, event)

    adapter._message_handler.assert_not_awaited()
    adapter._busy_session_handler.assert_awaited_once_with(event, session_key)
    assert adapter._pending_messages[session_key] is event


@pytest.mark.asyncio
async def test_reaction_event_skips_update_and_slash_confirm_prompts(monkeypatch) -> None:
    from types import SimpleNamespace
    from tests.gateway.test_clarify_thread_followup_not_swallowed import (
        _FellThroughIntercept,
        _StubAdapter,
        _event,
        _make_runner,
    )

    runner = _make_runner(_StubAdapter())
    state = SimpleNamespace(
        persistent=SimpleNamespace(update_prompt_pending=True),
        turn=SimpleNamespace(started_ts=0, agent=None),
    )
    setattr(runner, "_peek_session_state", lambda session_key: state)
    event = _event("[Telegram reaction: added 👍]")
    event.metadata = {"deferred_followup_event": True}
    pending = MagicMock(side_effect=AssertionError("deferred event queried confirm"))
    monkeypatch.setattr("tools.slash_confirm.get_pending", pending)
    monkeypatch.setattr("tools.approval.has_blocking_approval", lambda key: False)
    setattr(runner, "_is_session_running", MagicMock(side_effect=_FellThroughIntercept))
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *args, **kwargs: [])

    with pytest.raises(_FellThroughIntercept):
        await runner._handle_message(event)
    assert state.persistent.update_prompt_pending is True
    pending.assert_not_called()
