"""Command-line entry point for the FAQ agent."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from agent_spec_lab.graphs.enhanced_faq_graph import build_enhanced_faq_graph
from agent_spec_lab.state import AgentState
from agent_spec_lab.tools.faq_loader import load_faq_documents
from agent_spec_lab.tools.openai import get_openai_llm
from agent_spec_lab.tools.tracing import start_tracing


def pre_screen_question(question: str) -> tuple[bool, str | None]:
    """Pre-screen questions for obvious harmful content before sending to LLM.

    Returns:
        (is_safe, rejection_reason): If is_safe is False, provides rejection reason.
    """
    question_lower = question.lower()

    # Check for obviously harmful keywords
    harmful_patterns = [
        "hack",
        "crack",
        "exploit",
        "attack",
        "malware",
        "virus",
        "steal",
        "fraud",
        "scam",
        "illegal",
        "bomb",
        "weapon",
        "hurt",
        "kill",
        "suicide",
        "self-harm",
    ]

    for pattern in harmful_patterns:
        if pattern in question_lower:
            return False, f"I can't provide assistance with requests related to '{pattern}'"

    # Check for completely empty or nonsensical input
    if not question.strip() or len(question.strip()) < 2:
        return False, "Please provide a clear question about the agent-spec-lab framework"

    return True, None


app = typer.Typer(help="Interact with the agent-spec-lab FAQ assistant.")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question to ask the FAQ assistant.")],
    faq_dir: Annotated[
        Path | None, typer.Option(help="Path to a directory of markdown FAQ files.")
    ] = None,
    model: Annotated[str | None, typer.Option(help="Override the OpenAI chat model name.")] = None,
) -> None:
    """Ask the LangGraph agent a question."""

    load_dotenv()

    # Pre-screen the question before any LLM calls
    is_safe, rejection_reason = pre_screen_question(question)
    if not is_safe:
        typer.echo(f"Answer:\n{rejection_reason}")
        typer.echo("\nFor questions about the agent-spec-lab framework, I'm here to help!")
        return

    documents = load_faq_documents(faq_dir)
    llm_kwargs = {"model": model} if model else {}
    llm = get_openai_llm(**llm_kwargs)
    graph = build_enhanced_faq_graph(documents, llm)  # Use enhanced graph with fallback
    state = AgentState(question=question)

    with start_tracing(run_name="faq-query"):
        result = graph.invoke(state)

    # Following LangGraph best practices: work with dict results
    typer.echo("Answer:\n" + (result.get("answer") or "No answer generated."))
    if result.get("citations"):
        typer.echo("\nCitations:")
        for citation in result.get("citations", []):
            typer.echo(f" - {citation}")


def main() -> None:
    """Entrypoint used by ``python -m agent_spec_lab.cli``."""

    app()


if __name__ == "__main__":
    main()
