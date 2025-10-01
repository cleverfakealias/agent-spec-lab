"""Utilities for instantiating OpenAI chat models."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0


def get_openai_llm(**kwargs: Any) -> ChatOpenAI:
    """Return a configured :class:`~langchain_openai.ChatOpenAI` instance."""

    params: dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
    }
    params.update(kwargs)
    return ChatOpenAI(**params)


__all__ = ["get_openai_llm", "DEFAULT_MODEL", "DEFAULT_TEMPERATURE"]
