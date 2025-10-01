"""Query reformulation node for improving question clarity."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agent_spec_lab.state import AgentState
from agent_spec_lab.tools.logging import StructuredLogger, performance_timer, trace_node

_REFORMULATION_PROMPT = """
You are helping to reformulate user questions to be clearer and more specific 
for a FAQ system about a Python agent framework.

Original question: {question}

If the question is already clear and specific, return it unchanged.
If the question is vague, unclear, or uses ambiguous terms, reformulate it to be more specific.

Examples:
- "How do I use this?" → "How do I install and run the agent-spec-lab FAQ system?"
- "It's broken" → "What are common troubleshooting steps when the agent doesn't work?"
- "What about nodes?" → "How do I add custom nodes to the LangGraph agent?"

Return only the reformulated question, nothing else.
""".strip()


def create_reformulator_node(llm: BaseChatModel) -> Callable[[AgentState], AgentState]:
    """Create a node that reformulates unclear questions."""

    prompt = ChatPromptTemplate.from_messages([("human", _REFORMULATION_PROMPT)])
    logger = StructuredLogger("reformulator")

    @trace_node("reformulator")
    def reformulate(state: AgentState) -> AgentState:
        # Only reformulate if the question is genuinely vague or uses unclear references
        unclear_indicators = ["this", "it", "that", "here", "there"]

        # Check for vague questions that need context
        has_unclear_refs = any(
            indicator in state.question.lower() for indicator in unclear_indicators
        )
        is_very_vague = state.question.lower().strip() in [
            "help",
            "how?",
            "what?",
            "why?",
            "how do i use this?",
            "what is this?",
        ]

        # Don't reformulate clear questions, even if short
        clear_question_patterns = [
            "what is",
            "how does",
            "where is",
            "when does",
            "why does",
            "can i",
            "should i",
            "will it",
            "is there",
        ]
        is_clear_question = any(
            pattern in state.question.lower() for pattern in clear_question_patterns
        )

        if (has_unclear_refs or is_very_vague) and not is_clear_question:
            logger.info(
                "Question requires reformulation",
                state=state,
                has_unclear_refs=has_unclear_refs,
                is_very_vague=is_very_vague,
            )

            messages = prompt.format_messages(question=state.question)

            with performance_timer("llm_reformulation", logger):
                response = llm.invoke(messages)

            reformulated_question = getattr(response, "content", str(response)).strip()

            logger.info(
                "Question reformulated successfully",
                state=state,
                original_length=len(state.question),
                reformulated_length=len(reformulated_question),
            )

            processing_steps = list(state.processing_steps)
            processing_steps.append("reformulated")
            return state.model_copy(
                update={
                    "original_question": state.question,
                    "question": reformulated_question,
                    "was_reformulated": True,
                    "processing_steps": processing_steps,
                }
            )
        else:
            logger.info(
                "Question is clear, no reformulation needed",
                state=state,
                is_clear_question=is_clear_question,
            )
            return state.model_copy(
                update={"original_question": state.question, "was_reformulated": False}
            )

    return reformulate


__all__ = ["create_reformulator_node"]
