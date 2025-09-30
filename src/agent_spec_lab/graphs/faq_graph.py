"""LangGraph builder for the FAQ answering agent."""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph

from agent_spec_lab.nodes.answer import create_answer_node
from agent_spec_lab.nodes.retriever import create_retrieve_node
from agent_spec_lab.state import AgentState


def build_faq_graph(documents: Sequence[Document], llm: BaseChatModel):
    """Compile and return the FAQ LangGraph."""

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", create_retrieve_node(documents))
    graph.add_node("answer", create_answer_node(llm))

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


__all__ = ["build_faq_graph"]
