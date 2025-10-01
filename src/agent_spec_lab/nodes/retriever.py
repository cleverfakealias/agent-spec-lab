"""Retriever node for sourcing FAQ context snippets."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from difflib import SequenceMatcher

from langchain_core.documents import Document

from agent_spec_lab.state import AgentState


def _score_document(question: str, document: Document) -> float:
    """Return a basic similarity score between the question and a document."""

    return SequenceMatcher(None, question.lower(), document.page_content.lower()).ratio()


def create_retrieve_node(
    documents: Sequence[Document], top_k: int = 3
) -> Callable[[AgentState], AgentState]:
    """Create a LangGraph node that ranks FAQ documents by similarity.

    Parameters
    ----------
    documents:
        An iterable of :class:`~langchain_core.documents.Document` instances that will
        be ranked for relevance.
    top_k:
        The maximum number of context snippets to keep on the agent state.
    """

    docs: list[Document] = list(documents)

    def retrieve(state: AgentState) -> AgentState:
        ranked = sorted(
            docs,
            key=lambda document: _score_document(state.question, document),
            reverse=True,
        )
        chosen = ranked[:top_k]
        return state.model_copy(
            update={
                "context": [doc.page_content for doc in chosen],
                "citations": [str(doc.metadata.get("source", "unknown")) for doc in chosen],
            }
        )

    return retrieve


__all__ = ["create_retrieve_node"]
