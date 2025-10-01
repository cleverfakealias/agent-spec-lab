"""LangGraph builder for the FAQ answering agent."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from agent_spec_lab.nodes.answer import create_answer_node
from agent_spec_lab.nodes.retriever import create_retrieve_node
from agent_spec_lab.state import AgentState


def build_faq_graph(documents: Sequence[Document], llm: BaseChatModel) -> Any:
    """Compile and return the FAQ LangGraph."""

    graph = StateGraph(AgentState)

    # Convert functions to RunnableLambda explicitly for better type compatibility
    retrieve_runnable = RunnableLambda(create_retrieve_node(documents))
    answer_runnable = RunnableLambda(create_answer_node(llm))

    graph.add_node("retrieve", retrieve_runnable)
    graph.add_node("answer", answer_runnable)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


__all__ = ["build_faq_graph"]
