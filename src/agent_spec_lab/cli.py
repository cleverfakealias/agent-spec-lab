"""Command-line entry point for the FAQ agent."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from agent_spec_lab.graphs.enhanced_faq_graph import build_enhanced_faq_graph
from agent_spec_lab.state import AgentState
from agent_spec_lab.tools.faq_loader import load_faq_documents
from agent_spec_lab.tools.langsmith_utils import check_langsmith_configuration, get_trace_url
from agent_spec_lab.tools.logging import StructuredLogger, ensure_correlation_id
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

    # Initialize CLI logger
    cli_logger = StructuredLogger("cli")
    start_time = time.time()

    load_dotenv()

    # Create initial state with correlation ID
    state = AgentState(question=question, start_time=start_time)
    state = ensure_correlation_id(state)

    # Check LangSmith configuration
    langsmith_config = check_langsmith_configuration()

    cli_logger.info(
        "Starting FAQ query processing",
        state=state,
        faq_dir=str(faq_dir) if faq_dir else "default",
        model=model or "default",
        langsmith_config=langsmith_config,
    )

    # Show LangSmith status to user if tracing is enabled
    if langsmith_config["tracing_enabled"]:
        if langsmith_config["langsmith_accessible"]:
            typer.echo(
                f"🔍 LangSmith tracing enabled - Project: {langsmith_config['project_name']}"
            )
        else:
            typer.echo("⚠️  LangSmith tracing enabled but not accessible - check your configuration")
    else:
        typer.echo("ℹ️  LangSmith tracing disabled - set LANGCHAIN_TRACING_V2=true to enable")

    # Pre-screen the question before any LLM calls
    is_safe, rejection_reason = pre_screen_question(question)
    if not is_safe:
        cli_logger.warning(
            "Question rejected by pre-screening",
            state=state,
            rejection_reason=rejection_reason,
        )
        typer.echo(f"Answer:\n{rejection_reason}")
        typer.echo("\nFor questions about the agent-spec-lab framework, I'm here to help!")
        return

    try:
        documents = load_faq_documents(faq_dir)
        cli_logger.info(
            "Documents loaded successfully",
            state=state,
            document_count=len(documents),
        )

        llm_kwargs = {"model": model} if model else {}
        llm = get_openai_llm(**llm_kwargs)
        graph = build_enhanced_faq_graph(documents, llm)  # Use enhanced graph with fallback

        with start_tracing(run_name=f"faq-query-{state.correlation_id}"):
            result = graph.invoke(state)

        processing_time = time.time() - start_time

        cli_logger.info(
            "FAQ query completed successfully",
            state=state,
            processing_time_ms=processing_time * 1000,
            has_answer=bool(result.get("answer")),
            confidence_score=result.get("confidence_score"),
            is_fallback=result.get("is_fallback_response", False),
            citations_count=len(result.get("citations", [])),
        )

        # Show trace URL if available
        trace_url = get_trace_url()
        if trace_url and langsmith_config["tracing_enabled"]:
            typer.echo(f"\n🔗 View trace: {trace_url}")

        # Following LangGraph best practices: work with dict results
        typer.echo("Answer:\n" + (result.get("answer") or "No answer generated."))
        if result.get("citations"):
            typer.echo("\nCitations:")
            for citation in result.get("citations", []):
                typer.echo(f" - {citation}")

    except Exception as e:
        processing_time = time.time() - start_time
        cli_logger.error(
            "FAQ query failed with error",
            state=state,
            error_type=type(e).__name__,
            error_message=str(e),
            processing_time_ms=processing_time * 1000,
        )
        typer.echo(f"Error processing question: {str(e)}")
        raise


def main() -> None:
    """Entrypoint used by ``python -m agent_spec_lab.cli``."""

    app()


if __name__ == "__main__":
    main()
