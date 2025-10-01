"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent_spec_lab.tools.faq_loader import load_faq_documents


@pytest.fixture(scope="session")
def faq_documents() -> list[Document]:
    """Load the sample FAQ documents."""

    return load_faq_documents(Path("data/faq"))


class FakeChatModel(BaseChatModel):
    """Simple deterministic chat model for testing."""

    model_config = {"arbitrary_types_allowed": True}

    def _generate(
        self,
        messages: Sequence[BaseMessage] | Iterable[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        message_list = list(messages)
        last_message = message_list[-1]
        response = f"Test answer based on: {last_message.content}"[:200]
        generation = ChatGeneration(message=AIMessage(content=response))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "fake-chat"


@pytest.fixture()
def fake_llm() -> FakeChatModel:
    return FakeChatModel()
