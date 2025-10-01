"""Tests for the CLI interface.

This test suite ensures complete isolation from external services:
- No real OpenAI API calls are made (all LLM interactions mocked)
- No real LangSmith tracing calls are made (tracing context mocked)
- No real file system dependencies beyond temp directories
- All graph building and invocation is mocked
- Tests can run without any API keys or environment variables

The tests verify:
1. Basic CLI functionality (help, arguments, options)
2. Error handling (missing args, empty directories, etc.)
3. Environment handling (dotenv loading)
4. Integration behavior (mocked but realistic workflow)
5. Idempotency (same inputs produce same outputs)
6. Complete isolation (no external API dependencies)
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from agent_spec_lab.cli import app


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_faq_dir() -> Generator[Path]:
    """Create a temporary directory with sample FAQ files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        faq_dir = Path(temp_dir) / "faq"
        faq_dir.mkdir()

        # Create sample FAQ files
        getting_started = faq_dir / "getting_started.md"
        getting_started.write_text(
            """# Getting Started FAQ

## How do I install the agent?
To install the agent-spec-lab project, create a Python 3.11 virtual environment,
install the package in editable mode with `pip install -e .[dev]`, and copy the
`.env.example` file to `.env`.

## Where does the agent find knowledge?
The agent reads markdown files stored in the `data/faq` directory. Each file is
automatically split by level-2 headings (##) into separate document chunks.
"""
        )

        troubleshooting = faq_dir / "troubleshooting.md"
        troubleshooting.write_text(
            """# Troubleshooting FAQ

## What if the agent cannot answer a question?
If the context snippets do not contain enough information to answer the question,
the agent will respond that it cannot provide a helpful answer based on the available
context.

## How do I enable LangSmith tracing?
Set `LANGCHAIN_TRACING_V2=true` and provide `LANGCHAIN_API_KEY` along with optional
`LANGCHAIN_PROJECT` values in your `.env` file.
"""
        )

        yield faq_dir


class TestCLIBasicFunctionality:
    """Test basic CLI functionality without external API calls."""

    def test_cli_help(self, cli_runner: CliRunner) -> None:
        """Test that CLI help works."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Ask the LangGraph agent a question" in result.stdout
        assert "Question to ask the FAQ assistant" in result.stdout

    def test_cli_with_question_argument(self, cli_runner: CliRunner, temp_faq_dir: Path) -> None:
        """Test CLI with a question argument and custom FAQ directory."""
        # Mock ALL external dependencies to avoid any API calls
        with (
            patch("agent_spec_lab.cli.load_dotenv") as mock_dotenv,
            patch("agent_spec_lab.cli.load_faq_documents") as mock_load_docs,
            patch("agent_spec_lab.cli.get_openai_llm") as mock_llm,
            patch("agent_spec_lab.cli.build_enhanced_faq_graph") as mock_build_graph,
            patch("agent_spec_lab.cli.start_tracing") as mock_tracing,
        ):
            # Mock all dependencies
            mock_load_docs.return_value = []
            mock_llm.return_value = MagicMock()

            # Mock the graph and its invoke method to return dict (LangGraph best practice)
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {
                "question": "How do I install?",
                "context": ["## How do I install the agent?\nTo install..."],
                "answer": "Test answer about installation",
                "citations": ["getting_started.md#section-1"],
            }
            mock_build_graph.return_value = mock_graph

            # Configure the tracing context manager
            mock_tracing.return_value.__enter__ = MagicMock(return_value=None)
            mock_tracing.return_value.__exit__ = MagicMock(return_value=None)

            result = cli_runner.invoke(app, ["How do I install?", "--faq-dir", str(temp_faq_dir)])

            assert result.exit_code == 0
            assert "Answer:" in result.stdout
            assert "Test answer about installation" in result.stdout
            assert "Citations:" in result.stdout
            assert "getting_started.md#section-1" in result.stdout

            # Verify no real APIs were called
            mock_dotenv.assert_called_once()
            mock_load_docs.assert_called_once_with(temp_faq_dir)
            mock_llm.assert_called_once()
            mock_build_graph.assert_called_once()
            mock_graph.invoke.assert_called_once()

    def test_cli_with_model_option(self, cli_runner: CliRunner, temp_faq_dir: Path) -> None:
        """Test CLI with custom model option."""
        with (
            patch("agent_spec_lab.cli.load_faq_documents") as mock_load_docs,
            patch("agent_spec_lab.cli.get_openai_llm") as mock_llm,
            patch("agent_spec_lab.cli.build_enhanced_faq_graph") as mock_build_graph,
            patch("agent_spec_lab.cli.start_tracing") as mock_tracing,
        ):
            # Mock all dependencies
            mock_load_docs.return_value = []
            mock_llm.return_value = MagicMock()

            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {
                "question": "Test question",
                "context": [],
                "answer": "Test answer",
                "citations": [],
            }
            mock_build_graph.return_value = mock_graph

            mock_tracing.return_value.__enter__ = MagicMock(return_value=None)
            mock_tracing.return_value.__exit__ = MagicMock(return_value=None)

            result = cli_runner.invoke(
                app, ["Test question", "--faq-dir", str(temp_faq_dir), "--model", "gpt-4"]
            )

            # Verify the model was passed to get_openai_llm
            mock_llm.assert_called_once_with(model="gpt-4")
            assert result.exit_code == 0

    def test_cli_without_faq_dir_uses_default(self, cli_runner: CliRunner) -> None:
        """Test CLI without faq-dir option uses default."""
        with (
            patch("agent_spec_lab.cli.load_faq_documents") as mock_loader,
            patch("agent_spec_lab.cli.get_openai_llm") as mock_llm,
            patch("agent_spec_lab.cli.build_enhanced_faq_graph") as mock_build_graph,
            patch("agent_spec_lab.cli.start_tracing") as mock_tracing,
        ):
            # Mock all dependencies
            mock_loader.return_value = []
            mock_llm.return_value = MagicMock()

            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {
                "question": "Test",
                "context": [],
                "answer": "Test answer",
                "citations": [],
            }
            mock_build_graph.return_value = mock_graph

            mock_tracing.return_value.__enter__ = MagicMock(return_value=None)
            mock_tracing.return_value.__exit__ = MagicMock(return_value=None)

            result = cli_runner.invoke(app, ["Test question"])

            # Verify load_faq_documents was called with None (default)
            mock_loader.assert_called_once_with(None)
            assert result.exit_code == 0


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""

    def test_cli_missing_question_argument(self, cli_runner: CliRunner) -> None:
        """Test CLI fails gracefully when question argument is missing."""
        result = cli_runner.invoke(app, [])
        assert result.exit_code != 0
        # Check for error in stderr since that's where typer puts error messages
        assert "Missing argument" in result.stderr or result.exit_code == 2

    def test_cli_handles_empty_faq_directory(self, cli_runner: CliRunner) -> None:
        """Test CLI handles empty FAQ directory gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_faq_dir = Path(temp_dir) / "empty_faq"
            empty_faq_dir.mkdir()

            with (
                patch("agent_spec_lab.cli.load_faq_documents") as mock_load_docs,
                patch("agent_spec_lab.cli.get_openai_llm") as mock_llm,
                patch("agent_spec_lab.cli.build_enhanced_faq_graph") as mock_build_graph,
                patch("agent_spec_lab.cli.start_tracing") as mock_tracing,
            ):
                # Mock all dependencies - simulate empty FAQ directory
                mock_load_docs.return_value = []  # Empty documents
                mock_llm.return_value = MagicMock()

                mock_graph = MagicMock()
                mock_graph.invoke.return_value = {
                    "question": "Test",
                    "context": [],
                    "answer": None,  # No answer case
                    "citations": [],
                }
                mock_build_graph.return_value = mock_graph

                mock_tracing.return_value.__enter__ = MagicMock(return_value=None)
                mock_tracing.return_value.__exit__ = MagicMock(return_value=None)

                result = cli_runner.invoke(app, ["Test question", "--faq-dir", str(empty_faq_dir)])

                assert result.exit_code == 0
                assert "No answer generated." in result.stdout

    def test_cli_handles_nonexistent_faq_directory(self, cli_runner: CliRunner) -> None:
        """Test CLI handles nonexistent FAQ directory."""
        nonexistent_dir = Path("/nonexistent/faq/dir")

        with (
            patch("agent_spec_lab.cli.load_faq_documents") as mock_load_docs,
            patch("agent_spec_lab.cli.get_openai_llm") as mock_llm,
            patch("agent_spec_lab.cli.build_enhanced_faq_graph") as mock_build_graph,
            patch("agent_spec_lab.cli.start_tracing") as mock_tracing,
        ):
            # Mock the load_faq_documents to return empty list for nonexistent directory
            mock_load_docs.return_value = []
            mock_llm.return_value = MagicMock()

            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {
                "question": "Test",
                "context": [],
                "answer": "Test answer",
                "citations": [],
            }
            mock_build_graph.return_value = mock_graph

            mock_tracing.return_value.__enter__ = MagicMock(return_value=None)
            mock_tracing.return_value.__exit__ = MagicMock(return_value=None)

            result = cli_runner.invoke(app, ["Test question", "--faq-dir", str(nonexistent_dir)])

            # Should handle gracefully with mocked dependencies
            assert result.exit_code == 0
            mock_load_docs.assert_called_once_with(nonexistent_dir)


class TestCLIIntegration:
    """Integration tests with mocked dependencies - no real API calls."""

    def test_cli_loads_real_faq_files(self, cli_runner: CliRunner, temp_faq_dir: Path) -> None:
        """Test CLI with FAQ file loading simulation but all APIs mocked."""
        with (
            patch("agent_spec_lab.cli.load_faq_documents") as mock_load_docs,
            patch("agent_spec_lab.cli.get_openai_llm") as mock_llm,
            patch("agent_spec_lab.cli.build_enhanced_faq_graph") as mock_build_graph,
            patch("agent_spec_lab.cli.start_tracing") as mock_tracing,
        ):
            # Simulate loading real FAQ files (but don't actually do it)
            from langchain_core.documents import Document

            mock_load_docs.return_value = [
                Document(
                    page_content="## How do I install the agent?\nTo install...",
                    metadata={"source": "getting_started.md#section-1"},
                )
            ]

            mock_llm.return_value = MagicMock()

            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {
                "question": "How do I install the agent?",
                "context": ["## How do I install the agent?\nTo install..."],
                "answer": "Installation requires Python 3.11",
                "citations": ["getting_started.md#section-1"],
            }
            mock_build_graph.return_value = mock_graph

            mock_tracing.return_value.__enter__ = MagicMock(return_value=None)
            mock_tracing.return_value.__exit__ = MagicMock(return_value=None)

            result = cli_runner.invoke(
                app, ["How do I install the agent?", "--faq-dir", str(temp_faq_dir)]
            )

            assert result.exit_code == 0
            assert "Answer:" in result.stdout
            assert "Installation requires Python 3.11" in result.stdout

            # Verify all mocks were called - no real APIs
            mock_load_docs.assert_called_once_with(temp_faq_dir)
            mock_llm.assert_called_once()
            mock_build_graph.assert_called_once()
            mock_graph.invoke.assert_called_once()


class TestCLIEnvironmentHandling:
    """Test CLI environment variable handling."""

    def test_cli_loads_dotenv(self, cli_runner: CliRunner, temp_faq_dir: Path) -> None:
        """Test that CLI loads environment variables from .env file."""
        with (
            patch("agent_spec_lab.cli.load_dotenv") as mock_load_dotenv,
            patch("agent_spec_lab.cli.load_faq_documents") as mock_load_docs,
            patch("agent_spec_lab.cli.get_openai_llm") as mock_llm,
            patch("agent_spec_lab.cli.build_enhanced_faq_graph") as mock_build_graph,
            patch("agent_spec_lab.cli.start_tracing") as mock_tracing,
        ):
            # Mock all dependencies
            mock_load_docs.return_value = []
            mock_llm.return_value = MagicMock()

            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {
                "question": "Test",
                "context": [],
                "answer": "Test answer",
                "citations": [],
            }
            mock_build_graph.return_value = mock_graph

            mock_tracing.return_value.__enter__ = MagicMock(return_value=None)
            mock_tracing.return_value.__exit__ = MagicMock(return_value=None)

            result = cli_runner.invoke(app, ["Test question", "--faq-dir", str(temp_faq_dir)])

            # Verify load_dotenv was called
            mock_load_dotenv.assert_called_once()
            assert result.exit_code == 0


class TestCLIIsolation:
    """Test that CLI tests are completely isolated from external services."""

    def test_no_real_api_calls_made(self, cli_runner: CliRunner, temp_faq_dir: Path) -> None:
        """Verify that no real API calls are made during testing."""
        # Temporarily remove any API keys from environment
        original_openai_key = os.environ.get("OPENAI_API_KEY")
        original_langchain_key = os.environ.get("LANGCHAIN_API_KEY")

        try:
            # Remove API keys to ensure no real calls
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            if "LANGCHAIN_API_KEY" in os.environ:
                del os.environ["LANGCHAIN_API_KEY"]

            with (
                patch("agent_spec_lab.cli.load_dotenv") as mock_dotenv,
                patch("agent_spec_lab.cli.load_faq_documents") as mock_load_docs,
                patch("agent_spec_lab.cli.get_openai_llm") as mock_llm,
                patch("agent_spec_lab.cli.build_enhanced_faq_graph") as mock_build_graph,
                patch("agent_spec_lab.cli.start_tracing") as mock_tracing,
            ):
                # Mock all dependencies
                mock_load_docs.return_value = []
                mock_llm.return_value = MagicMock()

                mock_graph = MagicMock()
                mock_graph.invoke.return_value = {
                    "question": "Test",
                    "context": [],
                    "answer": "Mocked answer - no real API called",
                    "citations": [],
                }
                mock_build_graph.return_value = mock_graph

                mock_tracing.return_value.__enter__ = MagicMock(return_value=None)
                mock_tracing.return_value.__exit__ = MagicMock(return_value=None)

                # This should work even without API keys because everything is mocked
                result = cli_runner.invoke(app, ["Test question", "--faq-dir", str(temp_faq_dir)])

                assert result.exit_code == 0
                assert "Mocked answer - no real API called" in result.stdout

                # Verify all our mocks were called instead of real services
                mock_dotenv.assert_called_once()
                mock_load_docs.assert_called_once()
                mock_llm.assert_called_once()
                mock_build_graph.assert_called_once()

        finally:
            # Restore original API keys
            if original_openai_key is not None:
                os.environ["OPENAI_API_KEY"] = original_openai_key
            if original_langchain_key is not None:
                os.environ["LANGCHAIN_API_KEY"] = original_langchain_key


class TestCLIIdempotency:
    """Test that CLI operations are idempotent."""

    def test_cli_same_question_same_result(self, cli_runner: CliRunner, temp_faq_dir: Path) -> None:
        """Test that asking the same question multiple times produces consistent results."""
        question = "How do I install the agent?"

        with (
            patch("agent_spec_lab.cli.load_faq_documents") as mock_load_docs,
            patch("agent_spec_lab.cli.get_openai_llm") as mock_llm,
            patch("agent_spec_lab.cli.build_enhanced_faq_graph") as mock_build_graph,
            patch("agent_spec_lab.cli.start_tracing") as mock_tracing,
        ):
            # Configure deterministic mock responses
            mock_load_docs.return_value = []
            mock_llm.return_value = MagicMock()

            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {
                "question": question,
                "context": ["## How to install"],
                "answer": "Deterministic installation answer",
                "citations": ["getting_started.md#section-1"],  # Deterministic order
            }
            mock_build_graph.return_value = mock_graph

            mock_tracing.return_value.__enter__ = MagicMock(return_value=None)
            mock_tracing.return_value.__exit__ = MagicMock(return_value=None)

            # Run the same command multiple times
            results = []
            for _ in range(3):
                result = cli_runner.invoke(app, [question, "--faq-dir", str(temp_faq_dir)])
                assert result.exit_code == 0
                results.append(result.stdout)

            # All results should be identical (idempotent)
            assert len(set(results)) == 1, "CLI results should be deterministic/idempotent"

            # Verify consistent behavior - same calls made each time
            assert mock_load_docs.call_count == 3
            assert mock_llm.call_count == 3
            assert mock_build_graph.call_count == 3

    def test_cli_different_faq_dirs_different_results(self, cli_runner: CliRunner) -> None:
        """Test that different FAQ directories produce different results (when appropriate)."""
        with tempfile.TemporaryDirectory() as temp_dir1, tempfile.TemporaryDirectory() as temp_dir2:
            # Create two different FAQ directories paths (don't actually create files)
            faq_dir1 = Path(temp_dir1) / "faq1"
            faq_dir2 = Path(temp_dir2) / "faq2"

            with (
                patch("agent_spec_lab.cli.load_faq_documents") as mock_load_docs,
                patch("agent_spec_lab.cli.get_openai_llm") as mock_llm,
                patch("agent_spec_lab.cli.build_enhanced_faq_graph") as mock_build_graph,
                patch("agent_spec_lab.cli.start_tracing") as mock_tracing,
            ):
                mock_llm.return_value = MagicMock()
                mock_tracing.return_value.__enter__ = MagicMock(return_value=None)
                mock_tracing.return_value.__exit__ = MagicMock(return_value=None)

                # Mock different responses based on directory
                def mock_load_docs_side_effect(faq_dir):
                    if faq_dir == faq_dir1:
                        return [{"source": "dir1_content"}]
                    elif faq_dir == faq_dir2:
                        return [{"source": "dir2_content"}]
                    return []

                mock_load_docs.side_effect = mock_load_docs_side_effect

                def mock_build_graph_side_effect(docs, llm):
                    mock_graph = MagicMock()
                    if docs and docs[0].get("source") == "dir1_content":
                        mock_graph.invoke.return_value = {
                            "question": "Question A",
                            "answer": "Answer from directory 1",
                            "citations": ["dir1.md"],
                        }
                    elif docs and docs[0].get("source") == "dir2_content":
                        mock_graph.invoke.return_value = {
                            "question": "Question A",
                            "answer": "Answer from directory 2",
                            "citations": ["dir2.md"],
                        }
                    else:
                        mock_graph.invoke.return_value = {
                            "question": "Question A",
                            "answer": "Generic answer",
                            "citations": [],
                        }
                    return mock_graph

                mock_build_graph.side_effect = mock_build_graph_side_effect

                # Test with first directory
                result1 = cli_runner.invoke(app, ["Question A", "--faq-dir", str(faq_dir1)])

                # Test with second directory
                result2 = cli_runner.invoke(app, ["Question A", "--faq-dir", str(faq_dir2)])

                assert result1.exit_code == 0
                assert result2.exit_code == 0

                # Verify different directories were used
                mock_load_docs.assert_any_call(faq_dir1)
                mock_load_docs.assert_any_call(faq_dir2)

                # Results should reflect different directory usage
                assert "Answer from directory 1" in result1.stdout
                assert "Answer from directory 2" in result2.stdout
