"""Question classification node for intelligent routing."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from agent_spec_lab.state import AgentState


class QuestionType(str, Enum):
    """Types of questions the agent can handle."""

    INSTALLATION = "installation"
    TROUBLESHOOTING = "troubleshooting"
    FEATURE = "feature"
    GENERAL = "general"


_CLASSIFICATION_PROMPT = """
Classify this question into one of these categories:
- installation: Questions about installing, setting up, or getting started
- troubleshooting: Questions about fixing problems, errors, or issues
- feature: Questions about how to use specific features or functionality  
- general: General questions that don't fit other categories

Question: {question}

Respond with only the category name (installation, troubleshooting, feature, or general).
""".strip()


def create_classifier_node(llm: BaseChatModel) -> Callable[[AgentState], AgentState]:
    """Create a node that classifies the user's question type."""

    prompt = ChatPromptTemplate.from_messages([("human", _CLASSIFICATION_PROMPT)])

    def classify(state: AgentState) -> AgentState:
        messages = prompt.format_messages(question=state.question)
        response = llm.invoke(messages)
        content = getattr(response, "content", str(response)).strip().lower()

        # Validate and default to general if invalid
        try:
            question_type = QuestionType(content)
        except ValueError:
            question_type = QuestionType.GENERAL

        return state.model_copy(update={"question_type": question_type.value})

    return classify


__all__ = ["create_classifier_node", "QuestionType"]
