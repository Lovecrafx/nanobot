"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import re
import weakref
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.reasoning import (
    VALID_REASONING_LEVELS,
    is_valid_reasoning_level,
    normalize_reasoning_level,
    resolve_reasoning_effort,
)
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.config.loader import load_config, save_config
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 500
    _STATUS_KEY = "last_turn_status"
    _COMPACTIONS_KEY = "compactions"
    _CONTEXT_WINDOW_TOKENS = 200_000
    _IMAGE_TOKEN_COST = 256
    _HELP_TEXT = (
        "🐈 nanobot commands:\n"
        "/new — Start a new conversation\n"
        "/stop — Stop the current task\n"
        "/help — Show available commands\n"
        "/think — Set default reasoning level\n"
        "/status — Show current session status\n"
        "/restart — Restart the gateway"
    )

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        reasoning_effort: str | None = None,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        restart_callback: Callable[[], Awaitable[bool] | bool] | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        self.channels_config = channels_config
        self.restart_callback = restart_callback
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.reasoning_effort = reasoning_effort
        self.brave_api_key = brave_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=reasoning_effort,
            brave_api_key=brave_api_key,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._consolidating: set[str] = set()  # Session keys with consolidation in progress
        self._consolidation_tasks: set[asyncio.Task] = set()  # Strong refs to in-flight tasks
        self._consolidation_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._processing_lock = asyncio.Lock()
        self._register_default_tools()

    @staticmethod
    def _format_reasoning_level(level: str | None) -> str:
        """Format config/runtime reasoning level for human-readable responses."""
        return normalize_reasoning_level(level) or "off"

    @staticmethod
    def _think_usage() -> str:
        """Return `/think` usage text."""
        return f"Usage: /think [{'|'.join(VALID_REASONING_LEVELS)}]"

    def _resolve_reasoning_effort(self, task_text: str | None) -> str | None:
        """Resolve the configured reasoning level for a concrete task."""
        return resolve_reasoning_effort(self.reasoning_effort, task_text=task_text, model=self.model)

    @staticmethod
    def _status_usage() -> str:
        """Return `/status` usage text."""
        return "Usage: /status"

    @staticmethod
    def _restart_usage() -> str:
        """Return `/restart` usage text."""
        return "Usage: /restart"

    def _set_default_reasoning_effort(self, level: str | None) -> str:
        """Persist and apply the instance-level default reasoning effort."""
        normalized = normalize_reasoning_level(level)
        config = load_config()
        config.agents.defaults.reasoning_effort = normalized
        save_config(config)
        self.reasoning_effort = normalized
        self.subagents.reasoning_effort = normalized
        return self._format_reasoning_level(normalized)

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    @staticmethod
    def _usage_int(usage: dict[str, Any], key: str, default: int = 0) -> int:
        """Read an integer usage field with sane fallback."""
        value = usage.get(key, default)
        return value if isinstance(value, int) and value >= 0 else default

    @classmethod
    def _estimate_text_tokens(cls, text: str | None) -> int:
        """Estimate token count using a simple chars/4 heuristic."""
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))

    @classmethod
    def _estimate_content_tokens(cls, content: Any) -> int:
        """Estimate token count for provider message content."""
        if isinstance(content, str):
            return cls._estimate_text_tokens(content)
        if isinstance(content, list):
            return sum(cls._estimate_content_tokens(item) for item in content)
        if isinstance(content, dict):
            item_type = content.get("type")
            if item_type in {"text", "input_text", "output_text"}:
                return cls._estimate_text_tokens(content.get("text"))
            if item_type in {"image_url", "input_image"}:
                return cls._IMAGE_TOKEN_COST
            if isinstance(content.get("text"), str):
                return cls._estimate_text_tokens(content["text"])
            return cls._estimate_text_tokens(json.dumps(content, ensure_ascii=False))
        return 0

    @classmethod
    def _estimate_message_tokens(cls, message: dict[str, Any]) -> int:
        """Estimate token count for a single request message."""
        total = cls._estimate_content_tokens(message.get("content"))
        for tool_call in message.get("tool_calls") or []:
            total += cls._estimate_text_tokens(json.dumps(tool_call, ensure_ascii=False))
        return total

    @classmethod
    def _estimate_messages_tokens(cls, messages: list[dict[str, Any]]) -> int:
        """Estimate token count for provider input messages."""
        return sum(cls._estimate_message_tokens(message) for message in messages)

    @classmethod
    def _estimate_response_output_tokens(cls, response) -> int:
        """Estimate token count for assistant output when provider usage is missing."""
        total = cls._estimate_content_tokens(response.content)
        for tool_call in response.tool_calls:
            payload = {
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
            total += cls._estimate_text_tokens(json.dumps(payload, ensure_ascii=False))
        return total

    @classmethod
    def _compact_number(cls, value: int) -> str:
        """Format integer counters as compact human-readable units."""
        value = max(0, int(value))
        if value < 1000:
            return str(value)
        if value < 1_000_000:
            return f"{value / 1000:.1f}".rstrip("0").rstrip(".") + "k"
        return f"{value / 1_000_000:.1f}".rstrip("0").rstrip(".") + "m"

    def _make_turn_stats(self, resolved_reasoning: str | None) -> dict[str, Any]:
        """Initialize aggregated turn stats for a user turn."""
        return {
            "model": self.model,
            "configured_reasoning": self._format_reasoning_level(self.reasoning_effort),
            "effective_reasoning": self._format_reasoning_level(resolved_reasoning),
            "prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "cached_tokens_total": 0,
            "new_prompt_tokens_total": 0,
            "context_peak_tokens": 0,
            "context_window_tokens": self._CONTEXT_WINDOW_TOKENS,
            "completed": False,
        }

    def _record_call_stats(self, turn_stats: dict[str, Any], messages: list[dict[str, Any]], response) -> None:
        """Aggregate one provider call into the current turn stats."""
        usage = response.usage or {}
        estimated_prompt = self._estimate_messages_tokens(messages)
        estimated_completion = self._estimate_response_output_tokens(response)
        prompt_tokens = self._usage_int(usage, "prompt_tokens", estimated_prompt)
        completion_tokens = self._usage_int(usage, "completion_tokens", estimated_completion)
        cached_tokens = min(prompt_tokens, self._usage_int(usage, "cached_tokens", 0))
        new_tokens = max(prompt_tokens - cached_tokens, 0)

        turn_stats["prompt_tokens_total"] += prompt_tokens
        turn_stats["completion_tokens_total"] += completion_tokens
        turn_stats["cached_tokens_total"] += cached_tokens
        turn_stats["new_prompt_tokens_total"] += new_tokens
        turn_stats["context_peak_tokens"] = max(turn_stats["context_peak_tokens"], prompt_tokens)

    def _save_turn_status(self, session: Session, turn_stats: dict[str, Any]) -> None:
        """Persist the most recent successful turn status into session metadata."""
        snapshot = dict(turn_stats)
        snapshot.pop("completed", None)
        session.metadata[self._STATUS_KEY] = snapshot

    def _compaction_count(self, session: Session) -> int:
        """Return the session compaction count."""
        value = session.metadata.get(self._COMPACTIONS_KEY, 0)
        return value if isinstance(value, int) and value >= 0 else 0

    def _format_status(self, session: Session) -> str:
        """Render the current session status summary."""
        snapshot = session.metadata.get(self._STATUS_KEY)
        status = snapshot if isinstance(snapshot, dict) else {}
        prompt_tokens = self._usage_int(status, "prompt_tokens_total", 0)
        completion_tokens = self._usage_int(status, "completion_tokens_total", 0)
        cached_tokens = self._usage_int(status, "cached_tokens_total", 0)
        new_tokens = self._usage_int(status, "new_prompt_tokens_total", 0)
        context_tokens = self._usage_int(status, "context_peak_tokens", 0)
        context_window = max(self._usage_int(status, "context_window_tokens", self._CONTEXT_WINDOW_TOKENS), 1)
        cache_hit = (cached_tokens * 100) // prompt_tokens if prompt_tokens else 0
        context_pct = min(100, (context_tokens * 100) // context_window) if context_tokens else 0

        configured = self._format_reasoning_level(self.reasoning_effort)
        snapshot_configured = status.get("configured_reasoning") if isinstance(status.get("configured_reasoning"), str) else None
        effective = status.get("effective_reasoning") if isinstance(status.get("effective_reasoning"), str) else configured
        think = configured
        if configured == "adaptive" and snapshot_configured == "adaptive" and effective in {"low", "high"}:
            think = f"adaptive -> {effective}"

        model = status.get("model") if isinstance(status.get("model"), str) else self.model
        return "\n".join([
            f"🧠 Model: {model}",
            f"🧮 Tokens: {self._compact_number(prompt_tokens)} in / {self._compact_number(completion_tokens)} out",
            f"🗄️ Cache: {cache_hit}% hit · {self._compact_number(cached_tokens)} cached, {self._compact_number(new_tokens)} new",
            f"📚 Context: {self._compact_number(context_tokens)}/{self._compact_number(context_window)} ({context_pct}%) · 🧹 Compactions: {self._compaction_count(session)}",
            f"⚙️ Think: {think}",
        ])

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        task_text: str | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict], dict[str, Any]]:
        """Run the agent iteration loop. Returns (final_content, tools_used, messages, turn_stats)."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        resolved_reasoning = self._resolve_reasoning_effort(task_text)
        turn_stats = self._make_turn_stats(resolved_reasoning)

        while iteration < self.max_iterations:
            iteration += 1

            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=resolved_reasoning,
            )
            self._record_call_stats(turn_stats, messages, response)

            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                turn_stats["completed"] = True
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )
            turn_stats["completed"] = True

        return final_content, tools_used, messages, turn_stats

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"⏹ Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Close MCP connections."""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=self.memory_window)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            final_content, _, all_msgs, turn_stats = await self._run_agent_loop(messages, task_text=msg.content)
            self._save_turn(session, all_msgs, 1 + len(history))
            if turn_stats.get("completed"):
                self._save_turn_status(session, turn_stats)
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        raw_cmd = msg.content.strip()
        parts = raw_cmd.split(maxsplit=1)
        cmd = parts[0].lower().split("@", 1)[0] if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd == "/new":
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated:]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)

            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content=self._HELP_TEXT)
        if cmd == "/think":
            if not arg or not is_valid_reasoning_level(arg):
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=self._think_usage())
            active = self._set_default_reasoning_effort(arg)
            extra = "\nAdaptive mode: simple tasks use low, complex tasks use high." if active == "adaptive" else ""
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"Default reasoning level set to `{active}`.{extra}",
            )
        if cmd == "/status":
            if arg:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=self._status_usage())
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=self._format_status(session))
        if cmd == "/restart":
            if arg:
                return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content=self._restart_usage())
            if self.restart_callback is None:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Restart is only available in gateway mode.",
                )
            requested = self.restart_callback()
            if inspect.isawaitable(requested):
                requested = await requested
            if not requested:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="Gateway restart already in progress.",
                )
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="Restarting gateway...",
            )

        unconsolidated = len(session.messages) - session.last_consolidated
        if (unconsolidated >= self.memory_window and session.key not in self._consolidating):
            self._consolidating.add(session.key)
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        await self._consolidate_memory(session)
                finally:
                    self._consolidating.discard(session.key)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=self.memory_window)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, _, all_msgs, turn_stats = await self._run_agent_loop(
            initial_messages, task_text=msg.content, on_progress=on_progress or _bus_progress,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        if turn_stats.get("completed"):
            self._save_turn_status(session, turn_stats)
        self.sessions.save(session)

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue  # Strip runtime context from multimodal messages
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """Delegate to MemoryStore.consolidate(). Returns True on success."""
        ok = await MemoryStore(self.workspace).consolidate(
            session, self.provider, self.model,
            archive_all=archive_all, memory_window=self.memory_window,
        )
        if ok:
            session.metadata[self._COMPACTIONS_KEY] = self._compaction_count(session) + 1
        return ok

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
