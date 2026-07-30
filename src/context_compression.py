"""Opt-in, local extractive compression for long RAG conversations."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CompressionConfig:
    enabled: bool = False
    max_history_chars: int = 2000
    max_chunk_chars: int = 1200
    max_context_chars: int = 5000

    @classmethod
    def from_env(cls) -> "CompressionConfig":
        return cls(
            enabled=_env_bool("CONTEXT_COMPRESSION", False),
            max_history_chars=int(os.getenv("CONTEXT_COMPRESSION_HISTORY_CHARS", "2000")),
            max_chunk_chars=int(os.getenv("CONTEXT_COMPRESSION_CHUNK_CHARS", "1200")),
            max_context_chars=int(os.getenv("CONTEXT_COMPRESSION_MAX_CHARS", "5000")),
        )


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _truncate(text: str, limit: int, keep_tail: bool = False) -> str:
    text = _compact(text)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    marker = "..."
    if limit <= len(marker):
        return text[-limit:] if keep_tail else text[:limit]
    available = max(0, limit - len(marker))
    if keep_tail:
        return marker + text[-available:]
    return text[:available].rstrip() + marker


def _history_pairs(history: Sequence[Any], current_message: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in history or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user, assistant = _compact(item[0]), _compact(item[1])
        elif isinstance(item, dict):
            user = _compact(item.get("user") or item.get("message"))
            assistant = _compact(item.get("assistant") or item.get("response"))
        else:
            continue
        if user == _compact(current_message) and not assistant:
            continue
        if user or assistant:
            pairs.append((user, assistant))
    return pairs


def build_retrieval_query(
    message: str,
    history: Sequence[Any],
    config: CompressionConfig,
) -> str:
    """Build a bounded retrieval query; disabled mode preserves existing behavior."""
    if not config.enabled:
        return message

    lines: List[str] = []
    for user, assistant in _history_pairs(history, message):
        if user:
            lines.append(f"User: {user}")
        if assistant:
            lines.append(f"Assistant: {assistant}")
    history_text = _truncate("\n".join(lines), config.max_history_chars, keep_tail=True)
    if not history_text:
        return _compact(message)
    return f"Recent conversation:\n{history_text}\nCurrent question:\n{_compact(message)}"


def format_retrieved_context(
    chunks: Iterable[Any],
    config: CompressionConfig,
) -> str:
    """Format ranked chunks and optionally bound their total prompt footprint."""
    if not config.enabled:
        return "\n\n".join(
            f"From {chunk.filename}:\n{chunk.content}" for chunk in chunks
        )

    sections: List[str] = []
    used = 0
    for chunk in chunks:
        content = _truncate(chunk.content, config.max_chunk_chars)
        section = f"From {chunk.filename}:\n{content}"
        remaining = config.max_context_chars - used
        if remaining <= 0:
            break
        if len(section) > remaining:
            section = _truncate(section, remaining)
        if section:
            sections.append(section)
            used += len(section) + 2
    return "\n\n".join(sections)


__all__ = ["CompressionConfig", "build_retrieval_query", "format_retrieved_context"]
