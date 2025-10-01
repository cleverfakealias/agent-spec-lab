"""Intelligent FAQ graph with advanced uncertainty handling."""

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
from agent_spec_lab.nodes.uncertainty import (
    create_uncertainty_handler_node,
    should_handle_uncertainty,
)
from agent_spec_lab.state import AgentState


def route_based_on_confidence(state: AgentState) -> str:
    """Enhanced routing that considers uncertainty indicators."""

    # Check if we should use uncertainty handling
    if should_handle_uncertainty(state):
        return "handle_uncertainty"

    # If confidence is medium, use regular fallback
    confidence = state.confidence_score or 5
    if confidence <= 7 and confidence > 4:
        return "fallback"

    # High confidence - proceed with regular answer
    return "answer"


def build_intelligent_faq_graph(
    documents: Sequence[Document], llm: BaseChatModel
) -> CompiledStateGraph:
    """Build an intelligent FAQ graph with sophisticated uncertainty handling."""

    graph = StateGraph(AgentState)

    # Create all node functions
    reformulate_runnable = RunnableLambda(create_reformulator_node(llm))
    classify_runnable = RunnableLambda(create_classifier_node(llm))
    retrieve_runnable = RunnableLambda(create_retrieve_node(documents))
    confidence_runnable = RunnableLambda(create_confidence_node(llm))
    answer_runnable = RunnableLambda(create_answer_node(llm))
    fallback_runnable = RunnableLambda(create_fallback_node(llm))
    uncertainty_runnable = RunnableLambda(create_uncertainty_handler_node(llm))

    # Add all nodes
    graph.add_node("reformulate", reformulate_runnable)
    graph.add_node("classify", classify_runnable)
    graph.add_node("retrieve", retrieve_runnable)
    graph.add_node("assess_confidence", confidence_runnable)
    graph.add_node("answer", answer_runnable)
    graph.add_node("fallback", fallback_runnable)
    graph.add_node("handle_uncertainty", uncertainty_runnable)

    # Set up the enhanced flow
    graph.set_entry_point("reformulate")
    graph.add_edge("reformulate", "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "assess_confidence")

    # Intelligent routing based on confidence and uncertainty indicators
    graph.add_conditional_edges(
        "assess_confidence",
        route_based_on_confidence,
        {"answer": "answer", "fallback": "fallback", "handle_uncertainty": "handle_uncertainty"},
    )

    # All paths end
    graph.add_edge("answer", END)
    graph.add_edge("fallback", END)
    graph.add_edge("handle_uncertainty", END)

    return graph.compile()


__all__ = ["build_intelligent_faq_graph"]
