"""Helpers for LangSmith tracing integration."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from langsmith import Client
from langsmith.utils import tracing_context


def get_langsmith_project(default: str = "agent-spec-lab") -> str:
    """Return the LangSmith project name to use for traces."""

    return default


@contextmanager
def start_tracing(run_name: str, project_name: Optional[str] = None) -> Iterator[None]:
    """Context manager for recording LangSmith traces."""

    client = Client()
    with tracing_context(client=client, project_name=project_name or get_langsmith_project(), run_name=run_name):
        yield


__all__ = ["start_tracing", "get_langsmith_project"]
