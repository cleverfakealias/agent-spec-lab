"""Enhanced LangGraph with conditional routing and multiple node types."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_spec_lab.nodes.answer import create_answer_node
from agent_spec_lab.nodes.classifier import create_classifier_node
from agent_spec_lab.nodes.confidence import create_confidence_node
from agent_spec_lab.nodes.fallback import create_fallback_node
from agent_spec_lab.nodes.reformulator import create_reformulator_node
from agent_spec_lab.nodes.retriever import create_retrieve_node
from agent_spec_lab.state import AgentState


def should_use_fallback(state: AgentState) -> str:
    """Routing function to decide between regular answer and fallback."""
    if state.needs_clarification:
        return "fallback"
    return "answer"


def build_enhanced_faq_graph(
    documents: Sequence[Document], llm: BaseChatModel
) -> CompiledStateGraph:
    """Build an enhanced FAQ graph with conditional routing and multiple node types."""
    graph = StateGraph(AgentState)

    # Create all node functions
    reformulate_runnable = RunnableLambda(create_reformulator_node(llm))
    classify_runnable = RunnableLambda(create_classifier_node(llm))
    retrieve_runnable = RunnableLambda(create_retrieve_node(documents))
    confidence_runnable = RunnableLambda(create_confidence_node(llm))
    answer_runnable = RunnableLambda(create_answer_node(llm))
    fallback_runnable = RunnableLambda(create_fallback_node(llm))

    # Add all nodes
    graph.add_node("reformulate", reformulate_runnable)
    graph.add_node("classify", classify_runnable)
    graph.add_node("retrieve", retrieve_runnable)
    graph.add_node("assess_confidence", confidence_runnable)
    graph.add_node("answer", answer_runnable)
    graph.add_node("fallback", fallback_runnable)

    # Set up the flow
    graph.set_entry_point("reformulate")
    graph.add_edge("reformulate", "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "assess_confidence")

    # Conditional routing based on confidence
    graph.add_conditional_edges(
        "assess_confidence", should_use_fallback, {"answer": "answer", "fallback": "fallback"}
    )

    # Both paths end
    graph.add_edge("answer", END)
    graph.add_edge("fallback", END)

    return graph.compile()


__all__ = ["build_enhanced_faq_graph"]
