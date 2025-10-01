"""Tests for the FAQ LangGraph."""

from __future__ import annotations

from agent_spec_lab.graphs.faq_graph import build_faq_graph
from agent_spec_lab.state import AgentState


def test_graph_invocation_returns_answer(faq_documents, fake_llm) -> None:
    graph = build_faq_graph(faq_documents, fake_llm)
    result = graph.invoke(AgentState(question="How do I install the agent?"))

    # Following LangGraph best practices: compiled graphs return dictionaries
    assert isinstance(result, dict)
    assert result["answer"] is not None
    assert "Test answer based on" in result["answer"]
    assert result["context"]
    assert result["citations"]

    # Verify we can still convert to AgentState when validation is needed
    validated_result = AgentState(**result)
    assert isinstance(validated_result, AgentState)
    assert validated_result.answer is not None
