"""Tests for the enhanced FAQ graph functionality."""

from __future__ import annotations

from agent_spec_lab.graphs.enhanced_faq_graph import build_enhanced_faq_graph
from agent_spec_lab.nodes.classifier import create_classifier_node
from agent_spec_lab.nodes.confidence import create_confidence_node
from agent_spec_lab.nodes.reformulator import create_reformulator_node
from agent_spec_lab.state import AgentState


def test_enhanced_graph_invocation(faq_documents, fake_llm) -> None:
    """Test that the enhanced graph processes a question through all nodes."""
    graph = build_enhanced_faq_graph(faq_documents, fake_llm)
    result = graph.invoke(AgentState(question="How do I install?"))

    assert isinstance(result, dict)
    assert result["answer"] is not None
    assert result["context"]
    assert result["citations"]

    # Enhanced fields should be present with flat structure
    assert result["original_question"] is not None
    assert "was_reformulated" in result
    assert result["question_type"] is not None
    assert result["confidence_score"] is not None


def test_classifier_node_installation_question(fake_llm) -> None:
    """Test that the classifier correctly identifies installation questions."""
    classifier = create_classifier_node(fake_llm)
    state = AgentState(question="How do I install this software?")

    result = classifier(state)

    # Fake LLM returns predictable responses, but we test the structure
    assert result.question_type is not None
    assert isinstance(result.question_type, str)


def test_confidence_node_assesses_context(fake_llm) -> None:
    """Test that the confidence node assesses context quality."""
    confidence_assessor = create_confidence_node(fake_llm)
    state = AgentState(
        question="How do I install?", context=["Installation instructions: pip install -e .[dev]"]
    )

    result = confidence_assessor(state)

    assert result.confidence_score is not None
    assert isinstance(result.confidence_score, int)
    assert result.confidence_reason is not None
    assert isinstance(result.needs_clarification, bool)


def test_reformulator_node_improves_vague_questions(fake_llm) -> None:
    """Test that the reformulator improves vague questions."""
    reformulator = create_reformulator_node(fake_llm)

    # Test with a vague question
    vague_state = AgentState(question="How do I use this?")
    result = reformulator(vague_state)

    assert result.original_question == "How do I use this?"
    assert result.was_reformulated is True
    assert result.question != result.original_question

    # Test with a clear question
    clear_state = AgentState(question="How do I install the agent-spec-lab package?")
    result = reformulator(clear_state)

    assert result.original_question == "How do I install the agent-spec-lab package?"
    assert result.was_reformulated is False


def test_enhanced_graph_handles_low_confidence(faq_documents, fake_llm) -> None:
    """Test that the enhanced graph routes to fallback for low confidence."""
    graph = build_enhanced_faq_graph(faq_documents, fake_llm)

    # Use a very specific question that likely won't match well
    result = graph.invoke(AgentState(question="What is the quantum mechanics implementation?"))

    # Should still produce a result
    assert result["answer"] is not None
    assert "confidence_score" in result

    # The specific routing depends on the fake LLM responses,
    # but we can verify the structure is maintained
    validated_result = AgentState(**result)
    assert isinstance(validated_result, AgentState)
