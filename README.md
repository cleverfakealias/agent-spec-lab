# Agent Spec Lab

Agent Spec Lab is a modular Python 3.13 project that demonstrates how to build a
LangGraph-powered agent capable of answering frequently asked questions (FAQ)
from local markdown files. The project is designed for spec-driven development
and multi-agent expansion.

## Features

- 🧠 **LangGraph agent** composed of retrieval and answer nodes.
- 📄 **Markdown knowledge base** stored in `data/faq/`.
- 🤖 **OpenAI** chat model integration for answer generation.
- 📊 **LangSmith tracing** utilities for observability.
- 🧱 **Typed Pydantic state** shared across the graph.
- 🧪 Testing, linting, formatting, and type-checking via GitHub Actions CI.

## Project Layout

```
agent-spec-lab/
├── src/agent_spec_lab/
│   ├── cli.py            # Typer CLI entry point
│   ├── state.py          # Shared Pydantic state model
│   ├── graphs/           # LangGraph builders
│   ├── nodes/            # LangGraph node factories
│   └── tools/            # Utilities (FAQ loader, OpenAI, LangSmith)
├── data/faq/             # Sample FAQ markdown knowledge base
├── tests/                # Pytest suite
├── pyproject.toml        # Project and tooling configuration
├── pre-commit-config.yaml
└── .github/workflows/ci.yml
```

## Getting Started

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and populate the required keys.

```bash
cp .env.example .env
```

At minimum you need:

- `OPENAI_API_KEY`
- `LANGCHAIN_TRACING_V2` (set to `true` to enable LangSmith tracing)
- `LANGCHAIN_API_KEY`

### 3. Run the CLI

```bash
python -m agent_spec_lab.cli ask "How do I install the agent?"
```

The command loads markdown documents from `data/faq/`, builds the LangGraph
agent, and prints the generated answer along with citations.

## Development Tasks

### Formatting and linting

```bash
ruff check .
ruff format --check .
black --check .
```

### Type checking

```bash
mypy src tests
```

### Running tests

```bash
pytest
```

## Continuous Integration

GitHub Actions executes formatting (ruff + black), linting, type checking, and
pytest on each push. The workflow definition lives in
`.github/workflows/ci.yml`.

## License

This project is licensed under the terms of the MIT license. See `LICENSE` for
more information.
