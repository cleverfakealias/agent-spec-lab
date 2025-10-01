"""Tests for FAQ document loading utilities."""

from __future__ import annotations

from pathlib import Path

from agent_spec_lab.tools.faq_loader import load_faq_documents


def test_load_faq_documents(tmp_path: Path) -> None:
    markdown = """# FAQ\n\n## Q1\nA1\n\n## Q2\nA2"""
    (tmp_path / "sample.md").write_text(markdown, encoding="utf-8")

    documents = load_faq_documents(tmp_path)

    assert len(documents) == 3  # Title + 2 sections
    assert documents[0].metadata["source"] == "sample.md#section-0"
    assert documents[0].page_content == "# FAQ"  # First section is the title
    assert documents[1].metadata["source"] == "sample.md#section-1"
    assert "Q1" in documents[1].page_content
