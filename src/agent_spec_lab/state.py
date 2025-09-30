"""Agent state definitions for the FAQ assistant."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Pydantic state shared between LangGraph nodes."""

    question: str
    context: List[str] = Field(default_factory=list)
    answer: Optional[str] = None
    citations: List[str] = Field(default_factory=list)

    model_config = {
        "frozen": False,
        "arbitrary_types_allowed": True,
    }


__all__ = ["AgentState"]
