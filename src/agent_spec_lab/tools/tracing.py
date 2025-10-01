"""Helpers for LangSmith tracing integration."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from langsmith import Client


def get_langsmith_project(default: str = "agent-spec-lab") -> str:
    """Return the LangSmith project name to use for traces."""

    return default


@contextmanager
def start_tracing(run_name: str, project_name: str | None = None) -> Iterator[None]:
    """Context manager for recording LangSmith traces.

    This sets up environment variables for LangSmith tracing.
    The actual tracing is handled automatically by LangChain when
    LANGCHAIN_TRACING_V2=true is set.
    """

    # Set up environment variables for tracing
    original_tracing = os.environ.get("LANGCHAIN_TRACING_V2")
    original_project = os.environ.get("LANGCHAIN_PROJECT")

    try:
        # Enable tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = project_name or get_langsmith_project()

        # Initialize client to ensure it's available
        Client()

        yield

    finally:
        # Restore original environment variables
        if original_tracing is not None:
            os.environ["LANGCHAIN_TRACING_V2"] = original_tracing
        else:
            os.environ.pop("LANGCHAIN_TRACING_V2", None)

        if original_project is not None:
            os.environ["LANGCHAIN_PROJECT"] = original_project
        else:
            os.environ.pop("LANGCHAIN_PROJECT", None)


__all__ = ["start_tracing", "get_langsmith_project"]
