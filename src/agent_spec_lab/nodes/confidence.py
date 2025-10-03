"""Confidence scoring and uncertainty detection node."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agent_spec_lab.state import AgentState
from agent_spec_lab.tools.logging import StructuredLogger, performance_timer, trace_node

_CONFIDENCE_PROMPT = """
Analyze how well the provided context answers the user's question.

Question: {question}

Context:
{context}

Rate the confidence on a scale of 1-10 where:
- 1-3: The context doesn't contain relevant information
- 4-6: The context partially answers the question but lacks details
- 7-8: The context mostly answers the question with minor gaps
- 9-10: The context completely and clearly answers the question

Also provide a brief reason for your confidence score.

Respond in this exact format:
Score: [number]
Reason: [brief explanation]
""".strip()


def create_confidence_node(llm: BaseChatModel) -> Callable[[AgentState], AgentState]:
    """Create a node that assesses confidence in the retrieved context."""

    prompt = ChatPromptTemplate.from_messages([("human", _CONFIDENCE_PROMPT)])
    logger = StructuredLogger("confidence")

    @trace_node("confidence")
    def assess_confidence(state: AgentState) -> AgentState:
        context = "\n\n".join(state.context) if state.context else "No context provided."

        logger.info(
            "Assessing confidence in retrieved context",
            state=state,
            context_available=bool(state.context),
            context_snippets=len(state.context),
        )

        messages = prompt.format_messages(question=state.question, context=context)

        with performance_timer("llm_confidence_assessment", logger):
            response = llm.invoke(messages)

        content = getattr(response, "content", str(response))

        # Parse confidence score
        confidence_score = 5  # Default
        confidence_reason = "Unable to parse confidence assessment"

        try:
            lines = content.split("\n")
            for line in lines:
                if line.startswith("Score:"):
                    confidence_score = int(line.split(":")[1].strip())
                elif line.startswith("Reason:"):
                    confidence_reason = line.split(":", 1)[1].strip()
        except (ValueError, IndexError):
            logger.warning(
                "Failed to parse confidence response",
                state=state,
                raw_response=content,
            )

        needs_clarification = confidence_score <= 4

        logger.info(
            "Confidence assessment completed",
            state=state,
            confidence_score=confidence_score,
            needs_clarification=needs_clarification,
            confidence_reason=confidence_reason,
        )

        return state.model_copy(
            update={
                "confidence_score": confidence_score,
                "confidence_reason": confidence_reason,
                "needs_clarification": needs_clarification,
            }
        )

    return assess_confidence


__all__ = ["create_confidence_node"]
