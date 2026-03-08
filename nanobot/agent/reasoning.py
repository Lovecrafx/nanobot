"""Reasoning effort helpers."""

from __future__ import annotations

import re

VALID_REASONING_LEVELS = ("off", "minimal", "low", "medium", "high", "adaptive")
_COMPLEXITY_KEYWORDS = (
    "debug",
    "fix",
    "refactor",
    "implement",
    "design",
    "architecture",
    "migration",
    "test",
    "review",
    "analyze",
    "analysis",
    "plan",
)
_PATH_RE = re.compile(r"(^|[\s(])(?:\.{1,2}/|/|~\/|\w[\w.-]*/)\S+")
_FILE_RE = re.compile(r"\b[\w.-]+\.(?:py|ts|tsx|js|jsx|json|yaml|yml|toml|md|txt|sh|go|rs|java|kt|swift|c|cc|cpp|h|hpp)\b")
_COMMAND_RE = re.compile(
    r"`[^`]+`|\$\s+\S+|(?:^|\n)\s{0,3}(?:git|npm|pnpm|yarn|uv|pytest|python|node|cargo|go|make)\b",
    re.IGNORECASE,
)
_LIST_RE = re.compile(r"(?m)^\s*(?:\d+\.\s+|[-*]\s+)")
_MODEL_MINIMAL_RE = re.compile(r"(gpt-5|(?:^|[/._-])(o1|o3|o4)(?:$|[/._-]))", re.IGNORECASE)


def normalize_reasoning_level(level: str | None) -> str | None:
    """Normalize configured/user-provided reasoning levels."""
    if level is None:
        return None
    clean = level.strip().lower()
    if not clean or clean == "off":
        return None
    if clean in VALID_REASONING_LEVELS:
        return clean
    return None


def is_valid_reasoning_level(level: str) -> bool:
    """Return True when *level* is accepted by `/think`."""
    return level.strip().lower() in VALID_REASONING_LEVELS


def supports_minimal_reasoning(model: str | None) -> bool:
    """Best-effort check for models likely to support `minimal` reasoning."""
    if not model:
        return False
    return bool(_MODEL_MINIMAL_RE.search(model))


def is_complex_reasoning_task(text: str | None) -> bool:
    """Heuristic for deciding whether a task deserves `high` effort."""
    if not text:
        return False

    body = text.strip()
    lowered = body.lower()
    if "```" in body or _PATH_RE.search(body) or _FILE_RE.search(body) or _COMMAND_RE.search(body):
        return True
    if any(keyword in lowered for keyword in _COMPLEXITY_KEYWORDS):
        return True
    if len(body) >= 160 or body.count("\n") >= 2:
        return True
    if len(_LIST_RE.findall(body)) >= 2:
        return True
    if sum(lowered.count(token) for token in (" and ", " then ", " also ", "同时", "并且", "另外")) >= 2:
        return True
    return False


def resolve_reasoning_effort(
    configured_level: str | None,
    *,
    task_text: str | None = None,
    model: str | None = None,
) -> str | None:
    """Resolve configured level to the concrete provider-facing value."""
    level = normalize_reasoning_level(configured_level)
    if level is None:
        return None
    if level == "adaptive":
        return "high" if is_complex_reasoning_task(task_text) else "low"
    if level == "minimal":
        return "minimal" if supports_minimal_reasoning(model) else "low"
    return level
