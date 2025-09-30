"""Command-line entry point for the FAQ agent."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

from agent_spec_lab.graphs.faq_graph import build_faq_graph
from agent_spec_lab.state import AgentState
from agent_spec_lab.tools.faq_loader import load_faq_documents
from agent_spec_lab.tools.openai import get_openai_llm
from agent_spec_lab.tools.tracing import start_tracing

app = typer.Typer(help="Interact with the agent-spec-lab FAQ assistant.")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask the FAQ assistant."),
    faq_dir: Optional[Path] = typer.Option(None, help="Path to a directory of markdown FAQ files."),
    model: Optional[str] = typer.Option(None, help="Override the OpenAI chat model name."),
) -> None:
    """Ask the LangGraph agent a question."""

    load_dotenv()
    documents = load_faq_documents(faq_dir)
    llm_kwargs = {"model": model} if model else {}
    llm = get_openai_llm(**llm_kwargs)
    graph = build_faq_graph(documents, llm)
    state = AgentState(question=question)

    with start_tracing(run_name="faq-query"):
        result = graph.invoke(state)

    typer.echo("Answer:\n" + (result.answer or "No answer generated."))
    if result.citations:
        typer.echo("\nCitations:")
        for citation in result.citations:
            typer.echo(f" - {citation}")


def main() -> None:
    """Entrypoint used by ``python -m agent_spec_lab.cli``."""

    app()


if __name__ == "__main__":
    main()
