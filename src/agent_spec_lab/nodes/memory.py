"""Conversation history and memory management node."""

from __future__ import annotations

from collections.abc import Callable

from agent_spec_lab.state import AgentState


class ConversationMemory:
    """Simple in-memory conversation storage."""

    def __init__(self, max_history: int = 5):
        self.history: list[dict[str, str]] = []
        self.max_history = max_history

    def add_exchange(self, question: str, answer: str) -> None:
        """Add a question-answer pair to history."""
        self.history.append({"question": question, "answer": answer})
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context(self) -> str:
        """Get formatted conversation history."""
        if not self.history:
            return "No previous conversation."

        context_lines = ["Previous conversation:"]
        for i, exchange in enumerate(self.history[-3:], 1):
            context_lines.append(f"{i}. Q: {exchange['question']}")
            context_lines.append(f"   A: {exchange['answer'][:100]}...")

        return "\n".join(context_lines)


# Global memory instance (in real apps, use dependency injection)
_conversation_memory = ConversationMemory()


def create_memory_node() -> Callable[[AgentState], AgentState]:
    """Create a node that adds conversation context."""

    def add_memory_context(state: AgentState) -> AgentState:
        conversation_context = _conversation_memory.get_context()

        # Add conversation context to the state
        return state.model_copy(update={"conversation_history": conversation_context})

    return add_memory_context


def create_memory_update_node() -> Callable[[AgentState], AgentState]:
    """Create a node that updates conversation memory after answering."""

    def update_memory(state: AgentState) -> AgentState:
        if state.answer:
            _conversation_memory.add_exchange(state.question, state.answer)
        return state

    return update_memory


__all__ = ["create_memory_node", "create_memory_update_node", "ConversationMemory"]
