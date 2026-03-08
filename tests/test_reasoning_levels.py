from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.reasoning import is_complex_reasoning_task, resolve_reasoning_effort
from nanobot.bus.events import InboundMessage
from nanobot.config.schema import Config
from nanobot.providers.base import LLMResponse, ToolCallRequest
from nanobot.session.manager import Session


def _make_loop(*, restart_callback=None):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "openai/gpt-5"
    workspace = MagicMock()
    workspace.__truediv__ = MagicMock(return_value=MagicMock())

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as mock_sub_mgr:
        mock_sub_mgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=workspace,
            restart_callback=restart_callback,
        )
    return loop


class _ContextStub:
    def build_messages(
        self,
        history,
        current_message,
        skill_names=None,
        media=None,
        channel=None,
        chat_id=None,
    ):
        return [
            {"role": "system", "content": "system"},
            {"role": "user", "content": current_message},
        ]

    def add_assistant_message(
        self,
        messages,
        content,
        tool_calls=None,
        reasoning_content=None,
        thinking_blocks=None,
    ):
        entry = {"role": "assistant", "content": content}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        messages.append(entry)
        return messages

    def add_tool_result(self, messages, tool_call_id, tool_name, result):
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "name": tool_name, "content": result})
        return messages


def _make_loop_with_session(session: Session, *, restart_callback=None):
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "openai/gpt-5"
    provider.chat = AsyncMock()
    workspace = Path(".")
    sessions = MagicMock()
    sessions.get_or_create.return_value = session

    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager", return_value=sessions), \
         patch("nanobot.agent.loop.SubagentManager") as mock_sub_mgr:
        mock_sub_mgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        loop = AgentLoop(
            bus=bus,
            provider=provider,
            workspace=workspace,
            restart_callback=restart_callback,
        )

    loop.context = _ContextStub()
    loop.tools = SimpleNamespace(
        get=lambda _name: None,
        get_definitions=lambda: [],
        execute=AsyncMock(return_value="ok"),
    )
    return loop, provider, sessions


def test_resolve_reasoning_effort_adaptive_low_for_simple_task() -> None:
    assert resolve_reasoning_effort("adaptive", task_text="解释一下 TCP 和 UDP 的区别", model="openai/gpt-5") == "low"


def test_resolve_reasoning_effort_adaptive_high_for_complex_task() -> None:
    task = "请 debug 这个问题。\n1. 看 `nanobot/agent/loop.py`\n2. 分析 ./tests/test_reasoning_levels.py"
    assert resolve_reasoning_effort("adaptive", task_text=task, model="openai/gpt-5") == "high"
    assert is_complex_reasoning_task(task) is True


def test_resolve_reasoning_effort_minimal_degrades_for_unsupported_model() -> None:
    assert resolve_reasoning_effort("minimal", task_text="hello", model="anthropic/claude-sonnet-4-5") == "low"


@pytest.mark.asyncio
async def test_think_command_updates_runtime_and_persists_config() -> None:
    loop = _make_loop()
    config = Config()

    with patch("nanobot.agent.loop.load_config", return_value=config), \
         patch("nanobot.agent.loop.save_config") as mock_save:
        response = await loop._process_message(
            InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/think adaptive")
        )

    assert response.content == "Default reasoning level set to `adaptive`.\nAdaptive mode: simple tasks use low, complex tasks use high."
    assert loop.reasoning_effort == "adaptive"
    assert loop.subagents.reasoning_effort == "adaptive"
    assert config.agents.defaults.reasoning_effort == "adaptive"
    mock_save.assert_called_once_with(config)


@pytest.mark.asyncio
async def test_think_off_persists_none() -> None:
    loop = _make_loop()
    config = Config()

    with patch("nanobot.agent.loop.load_config", return_value=config), \
         patch("nanobot.agent.loop.save_config"):
        response = await loop._process_message(
            InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/think off")
        )

    assert response.content == "Default reasoning level set to `off`."
    assert loop.reasoning_effort is None
    assert loop.subagents.reasoning_effort is None
    assert config.agents.defaults.reasoning_effort is None


@pytest.mark.asyncio
async def test_think_invalid_value_returns_usage() -> None:
    loop = _make_loop()

    response = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/think turbo")
    )

    assert response.content == "Usage: /think [off|minimal|low|medium|high|adaptive]"


@pytest.mark.asyncio
async def test_help_mentions_think() -> None:
    loop = _make_loop()

    response = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/help")
    )

    assert "/think — Set default reasoning level" in response.content
    assert "/status — Show current session status" in response.content
    assert "/compact — Compact current session context" in response.content
    assert "/restart — Restart the gateway" in response.content


@pytest.mark.asyncio
async def test_status_empty_session_shows_defaults() -> None:
    session = Session(key="cli:c1")
    loop, _, _ = _make_loop_with_session(session)

    response = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/status")
    )

    assert response.content == (
        "🧠 Model: openai/gpt-5\n"
        "🧮 Tokens: 0 in / 0 out\n"
        "🗄️ Cache: 0% hit · 0 cached, 0 new\n"
        "📚 Context: 0/200k (0%) · 🧹 Compactions: 0\n"
        "⚙️ Think: off"
    )


@pytest.mark.asyncio
async def test_status_with_args_returns_usage() -> None:
    session = Session(key="cli:c1")
    loop, _, _ = _make_loop_with_session(session)

    response = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/status now")
    )

    assert response.content == "Usage: /status"


@pytest.mark.asyncio
async def test_status_think_uses_current_runtime_setting_over_snapshot() -> None:
    session = Session(key="cli:c1")
    session.metadata["last_turn_status"] = {
        "model": "openai/gpt-5",
        "configured_reasoning": "off",
        "effective_reasoning": "off",
        "prompt_tokens_total": 100,
        "completion_tokens_total": 20,
        "cached_tokens_total": 0,
        "new_prompt_tokens_total": 100,
        "context_peak_tokens": 100,
        "context_window_tokens": 200000,
    }
    loop, _, _ = _make_loop_with_session(session)
    loop.reasoning_effort = "adaptive"
    loop.subagents.reasoning_effort = "adaptive"

    response = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/status")
    )

    assert response.content.endswith("⚙️ Think: adaptive")


@pytest.mark.asyncio
async def test_status_aggregates_multi_call_turn_and_formats_adaptive() -> None:
    session = Session(key="cli:c1")
    loop, provider, sessions = _make_loop_with_session(session)
    loop.reasoning_effort = "adaptive"
    loop.subagents.reasoning_effort = "adaptive"
    provider.chat.side_effect = [
        LLMResponse(
            content="planning",
            tool_calls=[ToolCallRequest(id="tc1", name="read_file", arguments={"path": "a.py"})],
            usage={"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500, "cached_tokens": 200},
        ),
        LLMResponse(
            content="done",
            usage={"prompt_tokens": 1800, "completion_tokens": 500, "total_tokens": 2300, "cached_tokens": 100},
        ),
    ]

    response = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="请 debug 这个问题")
    )

    assert response.content == "done"
    status = session.metadata["last_turn_status"]
    assert status["prompt_tokens_total"] == 3000
    assert status["completion_tokens_total"] == 800
    assert status["cached_tokens_total"] == 300
    assert status["new_prompt_tokens_total"] == 2700
    assert status["context_peak_tokens"] == 1800
    assert status["effective_reasoning"] == "high"
    sessions.save.assert_called()

    status_response = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/status")
    )
    assert status_response.content == (
        "🧠 Model: openai/gpt-5\n"
        "🧮 Tokens: 3k in / 800 out\n"
        "🗄️ Cache: 10% hit · 300 cached, 2.7k new\n"
        "📚 Context: 1.8k/200k (0%) · 🧹 Compactions: 0\n"
        "⚙️ Think: adaptive -> high"
    )


@pytest.mark.asyncio
async def test_failed_turn_does_not_overwrite_last_status() -> None:
    session = Session(key="cli:c1")
    session.metadata["last_turn_status"] = {
        "model": "openai/gpt-5",
        "configured_reasoning": "adaptive",
        "effective_reasoning": "high",
        "prompt_tokens_total": 100,
        "completion_tokens_total": 20,
        "cached_tokens_total": 10,
        "new_prompt_tokens_total": 90,
        "context_peak_tokens": 100,
        "context_window_tokens": 200000,
    }
    loop, provider, _ = _make_loop_with_session(session)
    provider.chat.return_value = LLMResponse(
        content="Error calling LLM: boom",
        finish_reason="error",
        usage={"prompt_tokens": 999, "completion_tokens": 1, "total_tokens": 1000},
    )

    response = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="broken")
    )

    assert response.content == "Error calling LLM: boom"
    assert session.metadata["last_turn_status"]["prompt_tokens_total"] == 100


def test_session_clear_resets_metadata() -> None:
    session = Session(key="cli:c1", metadata={"last_turn_status": {"model": "x"}, "compactions": 3})

    session.clear()

    assert session.metadata == {}


@pytest.mark.asyncio
async def test_consolidate_memory_increments_compactions_only_on_success() -> None:
    session = Session(key="cli:c1")
    loop, _, _ = _make_loop_with_session(session)

    with patch("nanobot.agent.loop.MemoryStore") as mock_store:
        mock_store.return_value.consolidate = AsyncMock(return_value=True)
        assert await loop._consolidate_memory(session) is True
        assert session.metadata["compactions"] == 1

        mock_store.return_value.consolidate = AsyncMock(return_value=False)
        assert await loop._consolidate_memory(session) is False
        assert session.metadata["compactions"] == 1


@pytest.mark.asyncio
async def test_compact_command_compacts_session_and_returns_success() -> None:
    session = Session(key="cli:c1")
    for i in range(60):
        session.add_message("user", f"msg{i}")
    loop, _, sessions = _make_loop_with_session(session)

    with patch.object(loop, "_consolidate_memory", AsyncMock(return_value=True)) as mock_compact:
        response = await loop._process_message(
            InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/compact")
        )

    assert response.content == "compact success"
    mock_compact.assert_awaited_once_with(session, instructions=None)
    sessions.save.assert_called_once_with(session)


@pytest.mark.asyncio
async def test_compact_command_passes_instructions() -> None:
    session = Session(key="cli:c1")
    for i in range(60):
        session.add_message("user", f"msg{i}")
    loop, _, _ = _make_loop_with_session(session)

    with patch.object(loop, "_consolidate_memory", AsyncMock(return_value=True)) as mock_compact:
        response = await loop._process_message(
            InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/compact 保留 TODO 与决策")
        )

    assert response.content == "compact success"
    mock_compact.assert_awaited_once_with(session, instructions="保留 TODO 与决策")


@pytest.mark.asyncio
async def test_compact_command_returns_noop_when_nothing_to_compact() -> None:
    session = Session(key="cli:c1")
    for i in range(10):
        session.add_message("user", f"msg{i}")
    loop, _, sessions = _make_loop_with_session(session)

    with patch.object(loop, "_consolidate_memory", AsyncMock(return_value=True)) as mock_compact:
        response = await loop._process_message(
            InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/compact")
        )

    assert response.content == "nothing to compact"
    mock_compact.assert_not_awaited()
    sessions.save.assert_not_called()


@pytest.mark.asyncio
async def test_compact_command_returns_in_progress_when_session_is_compacting() -> None:
    session = Session(key="cli:c1")
    for i in range(60):
        session.add_message("user", f"msg{i}")
    loop, _, sessions = _make_loop_with_session(session)
    loop._consolidating.add(session.key)

    with patch.object(loop, "_consolidate_memory", AsyncMock(return_value=True)) as mock_compact:
        response = await loop._process_message(
            InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/compact")
        )

    assert response.content == "compact in progress"
    mock_compact.assert_not_awaited()
    sessions.save.assert_not_called()


@pytest.mark.asyncio
async def test_compact_command_returns_failed_when_consolidation_fails() -> None:
    session = Session(key="cli:c1")
    for i in range(60):
        session.add_message("user", f"msg{i}")
    session.last_consolidated = 5
    original_last_consolidated = session.last_consolidated
    loop, _, sessions = _make_loop_with_session(session)

    with patch.object(loop, "_consolidate_memory", AsyncMock(return_value=False)) as mock_compact:
        response = await loop._process_message(
            InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/compact")
        )

    assert response.content == "compact failed"
    assert session.last_consolidated == original_last_consolidated
    mock_compact.assert_awaited_once_with(session, instructions=None)
    sessions.save.assert_not_called()


@pytest.mark.asyncio
async def test_restart_requires_gateway_mode() -> None:
    loop = _make_loop()

    response = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/restart")
    )

    assert response.content == "Restart is only available in gateway mode."


@pytest.mark.asyncio
async def test_restart_command_requests_gateway_restart() -> None:
    calls: list[str] = []

    def _restart() -> bool:
        calls.append("restart")
        return True

    loop = _make_loop(restart_callback=_restart)

    response = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/restart")
    )

    assert response.content == "Restarting gateway..."
    assert calls == ["restart"]


@pytest.mark.asyncio
async def test_restart_command_rejects_duplicate_restart() -> None:
    loop = _make_loop(restart_callback=lambda: False)

    response = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/restart")
    )

    assert response.content == "Gateway restart already in progress."


@pytest.mark.asyncio
async def test_restart_with_args_returns_usage() -> None:
    loop = _make_loop(restart_callback=lambda: True)

    response = await loop._process_message(
        InboundMessage(channel="cli", sender_id="u1", chat_id="c1", content="/restart now")
    )

    assert response.content == "Usage: /restart"
