"""Agent state definitions for the FAQ assistant."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Pydantic state shared between LangGraph nodes."""

    question: str
    context: list[str] = Field(default_factory=list)
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)

    model_config = {
        "frozen": False,
        "arbitrary_types_allowed": True,
    }


__all__ = ["AgentState"]
