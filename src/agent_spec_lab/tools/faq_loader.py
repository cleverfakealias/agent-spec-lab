"""Utilities for loading local FAQ markdown files."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

DEFAULT_FAQ_DIR = Path(__file__).resolve().parents[2] / "data" / "faq"


def _split_markdown_sections(markdown: str) -> list[str]:
    """Split a markdown document into sections based on level-2 headings."""

    sections: list[str] = []
    current: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [section for section in sections if section]


def load_faq_documents(directory: Path | None = None) -> list[Document]:
    """Load FAQ markdown files into LangChain documents."""

    directory = directory or DEFAULT_FAQ_DIR
    documents: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        sections = _split_markdown_sections(content) or [content]
        for index, section in enumerate(sections):
            documents.append(
                Document(
                    page_content=section,
                    metadata={"source": f"{path.name}#section-{index}"},
                )
            )
    return documents


__all__ = ["load_faq_documents", "DEFAULT_FAQ_DIR"]
