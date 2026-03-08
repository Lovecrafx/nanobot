from types import SimpleNamespace

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.telegram import TelegramChannel
from nanobot.config.schema import TelegramConfig


class _FakeHTTPXRequest:
    instances: list["_FakeHTTPXRequest"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.__class__.instances.append(self)


class _FakeUpdater:
    def __init__(self, on_start_polling) -> None:
        self._on_start_polling = on_start_polling
        self.kwargs = None

    async def start_polling(self, **kwargs) -> None:
        self.kwargs = kwargs
        self._on_start_polling()


class _FakeBot:
    def __init__(self) -> None:
        self.sent_messages: list[dict] = []

    async def get_me(self):
        return SimpleNamespace(username="nanobot_test")

    async def set_my_commands(self, commands) -> None:
        self.commands = commands

    async def send_message(self, **kwargs) -> None:
        self.sent_messages.append(kwargs)


class _FakeApp:
    def __init__(self, on_start_polling) -> None:
        self.bot = _FakeBot()
        self.updater = _FakeUpdater(on_start_polling)
        self.handlers = []
        self.error_handlers = []

    def add_error_handler(self, handler) -> None:
        self.error_handlers.append(handler)

    def add_handler(self, handler) -> None:
        self.handlers.append(handler)

    async def initialize(self) -> None:
        pass

    async def start(self) -> None:
        pass


class _FakeBuilder:
    def __init__(self, app: _FakeApp) -> None:
        self.app = app
        self.token_value = None
        self.request_value = None
        self.get_updates_request_value = None

    def token(self, token: str):
        self.token_value = token
        return self

    def request(self, request):
        self.request_value = request
        return self

    def get_updates_request(self, request):
        self.get_updates_request_value = request
        return self

    def proxy(self, _proxy):
        raise AssertionError("builder.proxy should not be called when request is set")

    def get_updates_proxy(self, _proxy):
        raise AssertionError("builder.get_updates_proxy should not be called when request is set")

    def build(self):
        return self.app


@pytest.mark.asyncio
async def test_start_uses_request_proxy_without_builder_proxy(monkeypatch) -> None:
    config = TelegramConfig(
        enabled=True,
        token="123:abc",
        allow_from=["*"],
        proxy="http://127.0.0.1:7890",
    )
    bus = MessageBus()
    channel = TelegramChannel(config, bus)
    app = _FakeApp(lambda: setattr(channel, "_running", False))
    builder = _FakeBuilder(app)

    monkeypatch.setattr("nanobot.channels.telegram.HTTPXRequest", _FakeHTTPXRequest)
    monkeypatch.setattr(
        "nanobot.channels.telegram.Application",
        SimpleNamespace(builder=lambda: builder),
    )

    await channel.start()

    assert len(_FakeHTTPXRequest.instances) == 1
    assert _FakeHTTPXRequest.instances[0].kwargs["proxy"] == config.proxy
    assert builder.request_value is _FakeHTTPXRequest.instances[0]
    assert builder.get_updates_request_value is _FakeHTTPXRequest.instances[0]
    assert [cmd.command for cmd in app.bot.commands] == ["start", "new", "stop", "help", "think", "status", "restart"]
    assert app.updater.kwargs["allowed_updates"] == ["message", "callback_query"]


def test_derive_topic_session_key_uses_thread_id() -> None:
    message = SimpleNamespace(
        chat=SimpleNamespace(type="supergroup"),
        chat_id=-100123,
        message_thread_id=42,
    )

    assert TelegramChannel._derive_topic_session_key(message) == "telegram:-100123:topic:42"


def test_get_extension_falls_back_to_original_filename() -> None:
    channel = TelegramChannel(TelegramConfig(), MessageBus())

    assert channel._get_extension("file", None, "report.pdf") == ".pdf"
    assert channel._get_extension("file", None, "archive.tar.gz") == ".tar.gz"


def test_is_allowed_accepts_legacy_telegram_id_username_formats() -> None:
    channel = TelegramChannel(TelegramConfig(allow_from=["12345", "alice", "67890|bob"]), MessageBus())

    assert channel.is_allowed("12345|carol") is True
    assert channel.is_allowed("99999|alice") is True
    assert channel.is_allowed("67890|bob") is True


def test_is_allowed_rejects_invalid_legacy_telegram_sender_shapes() -> None:
    channel = TelegramChannel(TelegramConfig(allow_from=["alice"]), MessageBus())

    assert channel.is_allowed("attacker|alice|extra") is False
    assert channel.is_allowed("not-a-number|alice") is False


@pytest.mark.asyncio
async def test_send_progress_keeps_message_in_topic() -> None:
    config = TelegramConfig(enabled=True, token="123:abc", allow_from=["*"])
    channel = TelegramChannel(config, MessageBus())
    channel._app = _FakeApp(lambda: None)

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="hello",
            metadata={"_progress": True, "message_thread_id": 42},
        )
    )

    assert channel._app.bot.sent_messages[0]["message_thread_id"] == 42


@pytest.mark.asyncio
async def test_send_reply_infers_topic_from_message_id_cache() -> None:
    config = TelegramConfig(enabled=True, token="123:abc", allow_from=["*"], reply_to_message=True)
    channel = TelegramChannel(config, MessageBus())
    channel._app = _FakeApp(lambda: None)
    channel._message_threads[("123", 10)] = 42

    await channel.send(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="hello",
            metadata={"message_id": 10},
        )
    )

    assert channel._app.bot.sent_messages[0]["message_thread_id"] == 42
    assert channel._app.bot.sent_messages[0]["reply_parameters"].message_id == 10


@pytest.mark.asyncio
async def test_on_think_without_args_shows_inline_keyboard() -> None:
    channel = TelegramChannel(TelegramConfig(allow_from=["*"]), MessageBus())
    calls: list[dict] = []

    async def _reply_text(text: str, reply_markup=None) -> None:
        calls.append({"text": text, "reply_markup": reply_markup})

    update = SimpleNamespace(
        message=SimpleNamespace(reply_text=_reply_text),
        effective_user=SimpleNamespace(id=123),
    )
    context = SimpleNamespace(args=[])

    await channel._on_think(update, context)

    assert calls[0]["text"] == "Choose the default reasoning level:"
    rows = calls[0]["reply_markup"].inline_keyboard
    assert [[btn.text for btn in row] for row in rows] == [
        ["off", "minimal", "low"],
        ["medium", "high", "adaptive"],
    ]
    assert [[btn.callback_data for btn in row] for row in rows] == [
        ["think:off", "think:minimal", "think:low"],
        ["think:medium", "think:high", "think:adaptive"],
    ]


@pytest.mark.asyncio
async def test_on_think_with_args_forwards_command(monkeypatch) -> None:
    channel = TelegramChannel(TelegramConfig(allow_from=["*"]), MessageBus())
    forwarded: list[tuple] = []

    async def _forward(update, context) -> None:
        forwarded.append((update, context))

    monkeypatch.setattr(channel, "_forward_command", _forward)
    update = SimpleNamespace(message=object(), effective_user=SimpleNamespace(id=123))
    context = SimpleNamespace(args=["high"])

    await channel._on_think(update, context)

    assert forwarded == [(update, context)]


@pytest.mark.asyncio
async def test_on_think_callback_forwards_internal_command(monkeypatch) -> None:
    channel = TelegramChannel(TelegramConfig(allow_from=["*"]), MessageBus())
    handled: list[dict] = []
    typing: list[str] = []

    async def _handle_message(**kwargs) -> None:
        handled.append(kwargs)

    monkeypatch.setattr(channel, "_handle_message", _handle_message)
    monkeypatch.setattr(channel, "_start_typing", lambda chat_id: typing.append(chat_id))

    answered: list[bool] = []
    message = SimpleNamespace(
        chat_id=123,
        message_id=77,
        chat=SimpleNamespace(type="private", is_forum=False),
        message_thread_id=None,
    )
    callback_query = SimpleNamespace(
        data="think:adaptive",
        message=message,
        answer=lambda: answered.append(True),
    )
    update = SimpleNamespace(
        callback_query=callback_query,
        effective_user=SimpleNamespace(id=456, username="alice", first_name="Alice"),
    )

    async def _answer() -> None:
        answered.append(True)

    callback_query.answer = _answer

    await channel._on_think_callback(update, SimpleNamespace())

    assert answered == [True]
    assert typing == ["123"]
    assert handled == [{
        "sender_id": "456|alice",
        "chat_id": "123",
        "content": "/think adaptive",
        "metadata": {
            "message_id": 77,
            "user_id": 456,
            "username": "alice",
            "first_name": "Alice",
            "is_group": False,
            "message_thread_id": None,
            "is_forum": False,
        },
        "session_key": None,
    }]


@pytest.mark.asyncio
async def test_help_mentions_status() -> None:
    channel = TelegramChannel(TelegramConfig(allow_from=["*"]), MessageBus())
    replies: list[str] = []

    async def _reply_text(text: str, reply_markup=None) -> None:
        replies.append(text)

    update = SimpleNamespace(message=SimpleNamespace(reply_text=_reply_text))

    await channel._on_help(update, SimpleNamespace())

    assert "/status — Show current session status" in replies[0]
