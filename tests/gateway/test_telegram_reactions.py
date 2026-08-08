"""Tests for Telegram message reactions tied to processing lifecycle hooks."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource


def _make_adapter(**extra):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="fake-token", extra=extra)
    adapter._bot = AsyncMock()
    adapter._bot.set_message_reaction = AsyncMock()
    adapter._intentional_reaction_targets = set()
    return adapter


def _make_event(chat_id: str = "123", message_id: str = "456") -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            chat_type="private",
            user_id="42",
            user_name="TestUser",
        ),
        message_id=message_id,
    )


# ── _reactions_enabled ───────────────────────────────────────────────


def test_reactions_disabled_by_default(monkeypatch):
    """Telegram reactions should be disabled by default."""
    monkeypatch.delenv("TELEGRAM_REACTIONS", raising=False)
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is False


def test_reactions_enabled_when_set_true(monkeypatch):
    """Setting TELEGRAM_REACTIONS=true enables reactions."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    assert adapter._reactions_enabled() is True


# ── _set_reaction ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_reaction_calls_bot_api(monkeypatch):
    """_set_reaction should call bot.set_message_reaction with correct args."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()

    result = await adapter._set_reaction("123", "456", "\U0001f440")

    assert result is True
    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="\U0001f440",
    )


@pytest.mark.asyncio
async def test_lifecycle_reaction_stays_disabled(monkeypatch):
    """The existing automatic processing reactions still honor the flag."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    adapter = _make_adapter()

    await adapter.on_processing_start(_make_event())
    await adapter.on_processing_complete(_make_event(), ProcessingOutcome.SUCCESS)

    adapter._bot.set_message_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_intentional_reaction_ignores_lifecycle_flag(monkeypatch):
    """The explicit Telegram reaction tool path is independent of lifecycle gating."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "false")
    adapter = _make_adapter()
    assert not hasattr(adapter, "add_reaction")
    assert not hasattr(adapter, "remove_reaction")

    result = await adapter.add_current_reaction("123", "456", "❤️")

    assert result is True
    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction="❤",
    )


@pytest.mark.asyncio
async def test_lifecycle_completion_does_not_overwrite_intentional_reaction(monkeypatch):
    """A model-selected emoji must survive the automatic completion hook."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_start(event)
    assert await adapter.add_current_reaction("123", "456", "❤️") is True
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    bot = adapter._bot
    assert bot is not None
    assert bot.set_message_reaction.await_count == 2
    reactions = [call.kwargs["reaction"] for call in bot.set_message_reaction.await_args_list]
    assert reactions == ["👀", "❤"]
    assert adapter._intentional_reaction_targets == set()



@pytest.mark.parametrize(
    ("enabled", "expected"),
    [(False, False), (True, True), ("yes", True), ("off", False)],
)
def test_application_handler_registration_gates_inbound_reactions(
    monkeypatch, enabled, expected,
):
    """Inbound reaction updates are registered only after explicit opt-in."""
    from plugins.platforms.telegram import adapter as telegram_module

    class FakeReactionHandler:
        MESSAGE_REACTION_UPDATED = "updated"

        def __init__(self, callback, **kwargs):
            self.callback = callback
            self.kwargs = kwargs

    monkeypatch.setattr(telegram_module, "MessageReactionHandler", FakeReactionHandler)
    adapter = _make_adapter(inbound_reactions=enabled)
    handlers = []
    app = SimpleNamespace(add_handler=lambda handler: handlers.append(handler))

    adapter._register_reaction_handler(app)

    reaction_handlers = [
        handler for handler in handlers if isinstance(handler, FakeReactionHandler)
    ]
    assert bool(reaction_handlers) is expected
    if expected:
        assert reaction_handlers[0].callback == adapter._handle_message_reaction
        assert reaction_handlers[0].kwargs == {"message_reaction_types": "updated"}


@pytest.mark.parametrize("handler_cls", [None, type("LegacyHandler", (), {})])
def test_inbound_reactions_skip_cleanly_without_reaction_update_support(
    monkeypatch, caplog, handler_cls,
):
    """Older PTB installs keep the rest of Telegram connected when opted in."""
    from plugins.platforms.telegram import adapter as telegram_module

    monkeypatch.setattr(telegram_module, "MessageReactionHandler", handler_cls)
    adapter = _make_adapter(inbound_reactions=True)
    handlers = []
    app = SimpleNamespace(add_handler=lambda handler: handlers.append(handler))

    with caplog.at_level("WARNING"):
        adapter._register_reaction_handler(app)

    assert "reaction updates unsupported" in caplog.text


def test_sent_index_edit_preserves_existing_thread_id(monkeypatch, tmp_path):
    from gateway import rich_sent_store

    monkeypatch.setattr(
        rich_sent_store,
        "_store_path",
        lambda: str(tmp_path / "state" / "rich_sent_index.json"),
    )
    rich_sent_store.record("123", "456", "first", thread_id="77")
    rich_sent_store.record("123", "456", "edited")

    assert rich_sent_store.lookup_entry("123", "456")["thread_id"] == "77"


def test_sent_index_is_not_expanded_when_inbound_reactions_are_disabled(monkeypatch):
    from gateway import rich_sent_store

    adapter = _make_adapter()
    adapter._rich_messages_enabled = False
    record = Mock()
    monkeypatch.setattr(rich_sent_store, "record", record)

    adapter._record_sent_message("123", "456", "answer")

    record.assert_not_called()


def test_sent_index_records_telegram_bot_owner(monkeypatch):
    from gateway import rich_sent_store

    adapter = _make_adapter(inbound_reactions=True)
    adapter._bot = SimpleNamespace(id=12345)
    record = Mock()
    monkeypatch.setattr(rich_sent_store, "record", record)

    adapter._record_sent_message("123", "456", "answer")

    assert record.call_args.kwargs["sender_id"] == 12345


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("returned_thread", "is_forum", "wire_thread", "logical_thread", "expected", "retry"),
    [
        (88, True, 77, "77", 88, True),
        (None, False, 77, "77", None, True),
        (None, False, None, "1", "1", False),
    ],
)
async def test_thread_fallback_indexes_returned_effective_thread(
    returned_thread, is_forum, wire_thread, logical_thread, expected, retry
):
    adapter = _make_adapter()
    adapter._is_bad_request_error = lambda error: True
    adapter._is_thread_not_found_error = lambda error: True
    adapter._prune_stale_dm_topic_binding = Mock()
    adapter._record_sent_message = Mock()
    returned = SimpleNamespace(
        message_id=999,
        message_thread_id=returned_thread,
        is_topic_message=is_forum,
        chat=SimpleNamespace(is_forum=is_forum),
        text="fallback message",
    )
    adapter._bot.send_message = AsyncMock(
        side_effect=(
            [RuntimeError("Message thread not found"), returned]
            if retry else [returned]
        )
    )
    kwargs = {"chat_id": "123", "text": "fallback message"}
    if wire_thread is not None:
        kwargs["message_thread_id"] = wire_thread

    result = await adapter._send_message_with_thread_fallback(
        _logical_thread_id=logical_thread,
        **kwargs,
    )

    assert result is returned
    assert adapter._record_sent_message.call_count == 1
    call = adapter._record_sent_message.call_args
    assert call.kwargs["effective_thread_id"] == expected


# ── on_processing_start ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_processing_start_handles_missing_ids(monkeypatch):
    """Should handle events without chat_id or message_id gracefully."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SimpleNamespace(chat_id=None),
        message_id=None,
    )

    await adapter.on_processing_start(event)

    adapter._bot.set_message_reaction.assert_not_awaited()


# ── on_processing_complete ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_processing_complete_cancelled_clears_reaction(monkeypatch):
    """Cancelled processing should clear the in-progress reaction.

    Without this clear, the 👀 reaction lingers on the user's message
    indefinitely (until another agent run swaps it for 👍/👎). On a
    ``/stop`` that ends a session, that reaction never gets cleaned up.
    """
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_complete(event, ProcessingOutcome.CANCELLED)

    # set_message_reaction with reaction=None clears all reactions on the
    # message (Bot API documented semantics; equivalent to Bot API 10.0's
    # deleteMessageReaction but works on PTB 22.6 already).
    adapter._bot.set_message_reaction.assert_awaited_once_with(
        chat_id=123,
        message_id=456,
        reaction=None,
    )


@pytest.mark.asyncio
async def test_clear_reactions_handles_api_error_gracefully(monkeypatch):
    """API errors during clear should not propagate."""
    monkeypatch.setenv("TELEGRAM_REACTIONS", "true")
    adapter = _make_adapter()
    adapter._bot.set_message_reaction = AsyncMock(side_effect=RuntimeError("no perms"))

    result = await adapter._clear_reactions("123", "456")
    assert result is False


# ── config.py bridging ───────────────────────────────────────────────


def test_config_bridges_telegram_reactions(monkeypatch, tmp_path):
    """gateway/config.py bridges telegram.reactions to TELEGRAM_REACTIONS env var."""
    import yaml
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({
        "telegram": {
            "reactions": True,
        },
    }))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Use setenv (not delenv) so monkeypatch registers cleanup even when
    # the var doesn't exist yet — load_gateway_config will overwrite it.
    monkeypatch.setenv("TELEGRAM_REACTIONS", "")

    from gateway.config import load_gateway_config
    load_gateway_config()

    import os
    assert os.getenv("TELEGRAM_REACTIONS") == "true"


# Inbound reaction routing

def _make_reaction_adapter(
    monkeypatch,
    tmp_path,
    *,
    authorized=True,
    thread_id: str | None = "77",
    sender_id="111",
):
    from gateway import rich_sent_store
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setattr(
        rich_sent_store,
        "_store_path",
        lambda: str(tmp_path / "state" / "rich_sent_index.json"),
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: tmp_path / "base",
    )
    rich_sent_store.record(
        "-100",
        "900",
        "A bot-authored answer",
        thread_id=thread_id,
        sender_id=sender_id,
    )

    origin = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-100",
        chat_type="forum",
        user_id="1000",
        user_name="Hermes user",
        thread_id=thread_id,
        profile="default",
    )
    session_entry = SimpleNamespace(origin=origin)
    runner = SimpleNamespace(
        session_store=SimpleNamespace(_entries={"telegram-session": session_entry}),
        _is_user_authorized=Mock(return_value=authorized),
        _session_key_for_source=Mock(
            side_effect=lambda source: f"telegram:{source.user_id}:{source.thread_id or 'root'}"
        ),
        _set_pending_turn_sidecar_notes=Mock(),
        _peek_session_state=lambda key: None,
        _profile_name_for_source=lambda source: "default",
        _profile_adapters={},
    )
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.gateway_runner = runner
    adapter._bot = SimpleNamespace(id=111)
    adapter.handle_message = AsyncMock()
    runner._authorization_adapter = lambda platform, profile=None: adapter
    return adapter, runner


def _update(*, old, new, user_id="42", is_bot=False, chat_id="-100", update_id=123):
    return SimpleNamespace(
        update_id=update_id,
        message_reaction=SimpleNamespace(
            chat=SimpleNamespace(id=chat_id, type="supergroup", is_forum=True),
            message_id=900,
            user=SimpleNamespace(
                id=user_id,
                username="authorized-user",
                full_name="Authorized User",
                is_bot=is_bot,
            ),
            old_reaction=[SimpleNamespace(emoji=value) for value in old],
            new_reaction=[SimpleNamespace(emoji=value) for value in new],
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("old", "new", "added", "removed"),
    [
        ([], ["❤️"], "added ❤️", None),
        (["👍"], ["❤️"], "added ❤️", "removed 👍"),
        (["❤️"], [], None, "removed ❤️"),
    ],
)
async def test_reaction_delta_starts_immediate_authenticated_turn(
    monkeypatch, tmp_path, old, new, added, removed,
):
    adapter, runner = _make_reaction_adapter(monkeypatch, tmp_path)

    await adapter._handle_message_reaction(_update(old=old, new=new))

    runner._is_user_authorized.assert_called_once()
    source = runner._is_user_authorized.call_args.args[0]
    assert source.chat_id == "-100"
    assert source.chat_type == "group"
    assert source.thread_id == "77"
    assert source.user_id == "42"
    runner._set_pending_turn_sidecar_notes.assert_not_called()
    adapter.handle_message.assert_awaited_once()
    event = getattr(adapter.handle_message, "await_args").args[0]
    assert event.source is source
    assert event.user_id == "42"
    assert event.user_name == "authorized-user"
    assert event.message_id == "900"
    assert event.platform_update_id == 123
    assert event.reply_to_message_id == "900"
    assert event.reply_to_text == "A bot-authored answer"
    assert event.reply_to_is_own_message is True
    assert event.internal is False
    assert event.metadata["deferred_followup_event"] is True
    note = event.text
    if added:
        assert added in note
    if removed:
        assert removed in note
    assert '"A bot-authored answer"' in note
    assert "Return NO_REPLY" in event.channel_prompt
    assert "consequential or risky action" in event.channel_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "unauthorized",
        "bot_actor",
        "unknown_chat",
        "missing_thread",
        "unknown_message",
        "wrong_bot",
        "ambiguous_legacy_bot",
        "no_delta",
    ],
)
async def test_invalid_reaction_updates_fail_closed(monkeypatch, tmp_path, case):
    adapter_kwargs = {
        "authorized": case != "unauthorized",
        "thread_id": None if case == "missing_thread" else "77",
        "sender_id": None if case == "ambiguous_legacy_bot" else "111",
    }
    adapter, runner = _make_reaction_adapter(monkeypatch, tmp_path, **adapter_kwargs)
    update_kwargs = {
        "old": ["👍"] if case == "no_delta" else [],
        "new": ["👍"],
        "is_bot": case == "bot_actor",
        "chat_id": "-200" if case == "unknown_chat" else "-100",
    }
    update = _update(**update_kwargs)
    if case == "unknown_message":
        update.message_reaction.message_id = 901
    elif case == "wrong_bot":
        adapter._bot = SimpleNamespace(id=222)
    elif case == "ambiguous_legacy_bot":
        runner.adapters = {Platform.TELEGRAM: adapter}
        runner._profile_adapters = {"family": {Platform.TELEGRAM: object()}}

    await adapter._handle_message_reaction(update)

    assert getattr(adapter.handle_message, "await_count", 0) == 0
    runner._set_pending_turn_sidecar_notes.assert_not_called()
    if case != "unauthorized":
        runner._is_user_authorized.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_reaction_update_id_starts_only_one_turn(monkeypatch, tmp_path):
    adapter, runner = _make_reaction_adapter(monkeypatch, tmp_path)
    update = _update(old=[], new=["👍"], update_id=456)

    await adapter._handle_message_reaction(update)
    await adapter._handle_message_reaction(update)

    assert adapter.handle_message.await_count == 1
    assert runner._is_user_authorized.call_count == 2


def test_conflicting_profile_index_routes_are_ignored(monkeypatch, tmp_path):
    from gateway import rich_sent_store

    current_path = tmp_path / "current" / "state" / "rich_sent_index.json"
    base_home = tmp_path / "base"
    secondary_path = (
        base_home / "profiles" / "secondary" / "state" / "rich_sent_index.json"
    )
    monkeypatch.setattr(rich_sent_store, "_store_path", lambda: str(current_path))
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: base_home,
    )
    current_path.parent.mkdir(parents=True)
    current_path.write_text(
        json.dumps({"-100:900": {"t": "one", "ts": 100, "thread_id": "77"}}),
        encoding="utf-8",
    )
    secondary_path.parent.mkdir(parents=True)
    secondary_path.write_text(
        json.dumps({"-100:900": {"t": "two", "ts": 200, "thread_id": "88"}}),
        encoding="utf-8",
    )

    assert rich_sent_store.lookup("-100", "900") == "one"
    local_entry = rich_sent_store.lookup_entry("-100", "900")
    assert local_entry is not None
    assert local_entry["thread_id"] == "77"
    assert rich_sent_store.lookup_entry("-100", "900", all_profiles=True) is None
    secondary_path.write_text(current_path.read_text(), encoding="utf-8")
    assert rich_sent_store.lookup_entry("-100", "900", all_profiles=True) is None


def test_sent_index_survives_restart(tmp_path):
    import os
    import subprocess
    import sys

    env = {**os.environ, "HERMES_HOME": str(tmp_path)}
    subprocess.run(
        [sys.executable, "-c", "from gateway.rich_sent_store import record; record('c','m','text',thread_id='1',sender_id='7')"],
        check=True,
        env=env,
    )
    result = subprocess.run(
        [sys.executable, "-c", "from gateway.rich_sent_store import lookup_entry; print(lookup_entry('c','m'))"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert "'thread_id': '1'" in result.stdout
    assert "'sender_id': '7'" in result.stdout


def test_sent_index_serializes_concurrent_writes_and_bad_timestamps(
    monkeypatch, tmp_path
):
    from concurrent.futures import ThreadPoolExecutor
    from gateway import rich_sent_store

    path = tmp_path / "state" / "rich_sent_index.json"
    monkeypatch.setattr(rich_sent_store, "_store_path", lambda: str(path))
    monkeypatch.setattr(rich_sent_store, "_MAX_ENTRIES", 20)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"bad:x": {"t": "old", "ts": None}}))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: rich_sent_store.record("c", i, str(i)), range(20)))
    data = json.loads(path.read_text())
    assert len(data) == 20
    assert "bad:x" not in data
    assert all(f"c:{i}" in data for i in range(20))


@pytest.mark.asyncio
async def test_secondary_profile_and_enum_chat_type_route_to_owner(
    monkeypatch, tmp_path
):
    adapter, _ = _make_reaction_adapter(monkeypatch, tmp_path)
    current_path = tmp_path / "state" / "rich_sent_index.json"
    current_path.unlink()
    secondary = (
        tmp_path / "base" / "profiles" / "secondary" / "state"
        / "rich_sent_index.json"
    )
    secondary.parent.mkdir(parents=True)
    secondary.write_text(
        json.dumps({"-100:900": {"t": "answer", "ts": 1, "sender_id": "111"}}),
        encoding="utf-8",
    )
    update = _update(old=[], new=["👍"])
    update.message_reaction.chat.is_forum = False
    update.message_reaction.chat.type = SimpleNamespace(value="private")

    await adapter._handle_message_reaction(update)

    event = getattr(adapter.handle_message, "await_args").args[0]
    assert event.source.profile == "secondary"
    assert event.source.chat_type == "dm"


@pytest.mark.parametrize(("thread_id", "expected"), [(None, "900"), ("77", None)])
def test_reaction_reply_anchor_uses_message_for_root_but_topic_for_forum(
    thread_id, expected
):
    from gateway.platforms.base import MessageEvent, _reply_anchor_for_event

    event = MessageEvent(
        text="[Telegram reaction: added 👍]",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100",
            chat_type="group",
            user_id="42",
            thread_id=thread_id,
        ),
        message_id="900",
        reply_to_message_id="900",
        metadata={"deferred_followup_event": True},
    )

    assert _reply_anchor_for_event(event) == expected


def test_sent_result_ignores_non_forum_reply_anchor_thread_id():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter._record_sent_message = Mock()
    result = SimpleNamespace(
        message_id=900,
        message_thread_id=123,
        is_topic_message=False,
        chat=SimpleNamespace(is_forum=False),
        text="answer",
        caption=None,
    )

    adapter._record_sent_result(
        "-100",
        result,
        effective_thread_id=None,
    )

    assert adapter._record_sent_message.call_args.kwargs["effective_thread_id"] is None


@pytest.mark.asyncio
async def test_live_general_topic_final_send_then_reaction_routes_immediate_turn(
    monkeypatch, tmp_path
):
    """End-to-end regression for the live General-topic reaction failure.

    Drive the production ``send()`` path exactly as the gateway does for a
    final reply in a forum's General topic (logical thread id "1"): the
    transport request must omit ``message_thread_id`` and Telegram's success
    response omits it as well. The sent index must still record the logical
    topic so the user's reaction — which carries no topic identity — routes
    to an immediate MessageEvent in thread "1" instead of being fail-closed
    dropped.
    """
    from gateway import rich_sent_store
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setattr(
        rich_sent_store,
        "_store_path",
        lambda: str(tmp_path / "state" / "rich_sent_index.json"),
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root",
        lambda: tmp_path / "base",
    )

    adapter = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="fake-token",
            extra={"rich_messages": False, "inbound_reactions": True},
        )
    )
    bot = MagicMock()
    bot.id = 111
    bot.send_message = AsyncMock(
        return_value=SimpleNamespace(message_id=700, message_thread_id=None)
    )
    bot.send_chat_action = AsyncMock()
    adapter._bot = bot

    send_result = await adapter.send(
        "-100",
        "The final answer.",
        metadata={"thread_id": "1", "notify": True},
    )

    assert send_result.success is True
    # Live transport constraint: the send really omitted the General topic id.
    assert bot.send_message.await_args.kwargs["message_thread_id"] is None
    entry = rich_sent_store.lookup_entry("-100", "700")
    assert entry is not None
    assert entry["thread_id"] == "1"
    assert entry["sender_id"] == "111"

    runner = SimpleNamespace(
        _is_user_authorized=Mock(return_value=True),
        _session_key_for_source=Mock(return_value="telegram:42:1"),
        _profile_adapters={},
    )
    adapter.gateway_runner = runner
    adapter.handle_message = AsyncMock()
    runner._authorization_adapter = lambda platform, profile=None: adapter

    update = _update(old=[], new=["👍"])
    update.message_reaction.message_id = 700
    await adapter._handle_message_reaction(update)

    adapter.handle_message.assert_awaited_once()
    event = getattr(adapter.handle_message, "await_args").args[0]
    assert event.source.thread_id == "1"
    assert event.reply_to_message_id == "700"
    assert event.message_id == "700"
    assert event.metadata["deferred_followup_event"] is True


@pytest.mark.asyncio
async def test_rich_and_overflow_sends_keep_general_topic_ownership(
    monkeypatch, tmp_path
):
    from gateway import rich_sent_store
    from gateway.config import PlatformConfig
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setattr(
        rich_sent_store,
        "_store_path",
        lambda: str(tmp_path / "state" / "rich_sent_index.json"),
    )
    monkeypatch.setattr(
        "hermes_constants.get_default_hermes_root", lambda: tmp_path / "base"
    )

    rich = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="fake-token",
            extra={"rich_messages": True, "inbound_reactions": True},
        )
    )
    rich._bot = MagicMock(
        id=111,
        do_api_request=AsyncMock(
            return_value=SimpleNamespace(message_id=701, message_thread_id=None)
        ),
        send_chat_action=AsyncMock(),
    )
    content = "## Results\n\n| Case | Status |\n|---|---|\n| rich | ok |"
    assert (
        await rich.send(
            "-100", content, metadata={"thread_id": "1", "notify": True}
        )
    ).success
    call = rich._bot.do_api_request.call_args
    assert call.args[0] == "sendRichMessage"
    assert "message_thread_id" not in call.kwargs["api_kwargs"]
    assert rich_sent_store.lookup_entry("-100", "701")["thread_id"] == "1"

    overflow = TelegramAdapter(
        PlatformConfig(
            enabled=True,
            token="fake-token",
            extra={"rich_messages": False, "inbound_reactions": True},
        )
    )
    overflow._bot = MagicMock(
        id=111,
        edit_message_text=AsyncMock(return_value=SimpleNamespace(message_id=500)),
        send_message=AsyncMock(
            return_value=SimpleNamespace(message_id=501, message_thread_id=None)
        ),
    )
    assert (
        await overflow._edit_overflow_split(
            "-100",
            "500",
            "word " * 1200,
            finalize=True,
            metadata={"thread_id": "1", "notify": True},
        )
    ).success
    assert rich_sent_store.lookup_entry("-100", "500")["thread_id"] == "1"
    assert rich_sent_store.lookup_entry("-100", "501")["thread_id"] == "1"