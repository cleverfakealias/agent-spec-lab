"""LLM answering node for the FAQ graph."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agent_spec_lab.state import AgentState

_SYSTEM_PROMPT = """
You are a helpful assistant that answers product FAQ questions.
Use the provided context snippets from the knowledge base.
Quote directly from the context when possible and keep answers concise.
If the answer is not present in the context, acknowledge the gap.
""".strip()

_USER_PROMPT = """
Question: {question}

Context:
{context}
""".strip()


def create_answer_node(llm: BaseChatModel) -> Callable[[AgentState], AgentState]:
    """Create a node that calls an OpenAI chat model to craft an answer."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("human", _USER_PROMPT),
        ]
    )

    def answer(state: AgentState) -> AgentState:
        context = "\n\n".join(state.context) if state.context else "No relevant context provided."
        messages = prompt.format_messages(question=state.question, context=context)
        response = llm.invoke(messages)
        content = getattr(response, "content", str(response))
        return state.model_copy(update={"answer": content})

    return answer


__all__ = ["create_answer_node"]
