"""Document expansion node for simulating external knowledge retrieval."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agent_spec_lab.state import AgentState

_EXPANSION_PROMPT = """
The user asked a question that may require information beyond the basic FAQ.

Question: {question}
Available FAQ context: {context}

Generate additional helpful information that would typically come from:
- GitHub documentation
- API reference
- Community discussions
- Best practices guides

Provide 2-3 short paragraphs of relevant additional context that would help answer
the question more completely.
Focus on practical, actionable information.
""".strip()


def create_expansion_node(llm: BaseChatModel) -> Callable[[AgentState], AgentState]:
    """Create a node that expands available context with simulated external sources."""

    prompt = ChatPromptTemplate.from_messages([("human", _EXPANSION_PROMPT)])

    def expand_context(state: AgentState) -> AgentState:
        # Only expand if we have low confidence or specific question types
        should_expand = (
            state.confidence_score and state.confidence_score <= 6
        ) or state.question_type in ["feature", "troubleshooting"]

        if not should_expand:
            return state

        context = "\n\n".join(state.context) if state.context else "No FAQ context available."
        messages = prompt.format_messages(question=state.question, context=context)
        response = llm.invoke(messages)
        expanded_content = getattr(response, "content", str(response))

        # Add expanded context
        enhanced_context = list(state.context)
        enhanced_context.append(f"[Expanded Context]\n{expanded_content}")

        enhanced_citations = list(state.citations)
        enhanced_citations.append("external-knowledge-base")

        processing_steps = list(state.processing_steps)
        processing_steps.append("expanded")
        return state.model_copy(
            update={
                "context": enhanced_context,
                "citations": enhanced_citations,
                "processing_steps": processing_steps,
            }
        )

    return expand_context


__all__ = ["create_expansion_node"]
