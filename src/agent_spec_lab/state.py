"""Agent state definitions for the FAQ assistant."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Pydantic state shared between LangGraph nodes.

    Following LangGraph best practices:
    - Flat structure for consistent dict access patterns
    - Grouped related fields with clear naming
    - Minimal complexity while maintaining functionality
    """

    # Core fields - always present
    question: str
    context: list[str] = Field(default_factory=list)
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)

    # Question processing - flat but grouped by naming
    original_question: str | None = None
    was_reformulated: bool = False
    question_type: str | None = None  # "feature", "troubleshooting", "general", etc.
    processing_steps: list[str] = Field(default_factory=list)

    # Confidence assessment - flat but grouped by naming
    confidence_score: int | None = None  # 1-10 scale
    confidence_reason: str | None = None
    needs_clarification: bool = False
    uncertainty_type: str | None = None  # "ambiguous", "off_topic", "insufficient_context"

    # Response metadata - flat but grouped by naming
    is_fallback_response: bool = False
    response_strategy: str | None = None  # "direct", "reformulated", "uncertainty_handled"
    response_explanation: str | None = None  # For uncertainty or fallback responses

    # Conversation context
    conversation_history: str | None = None

    model_config = {
        "frozen": False,
        "arbitrary_types_allowed": True,
    }


__all__ = ["AgentState"]
