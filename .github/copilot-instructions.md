# Agent Spec Lab - AI Coding Instructions

## Architecture Overview

This is a **LangGraph-powered FAQ agent** built with Python 3.13 that demonstrates modular agent architecture. The system follows a **strict separation of concerns**:

- **State Management**: Single `AgentState` Pydantic model (`src/agent_spec_lab/state.py`) shared across all nodes
- **Graph Assembly**: `build_faq_graph()` in `src/agent_spec_lab/graphs/faq_graph.py` wires retrieval → answer nodes
- **Node Factories**: Functions in `src/agent_spec_lab/nodes/` return configured LangGraph node callables
- **Tools**: Utility modules in `src/agent_spec_lab/tools/` for OpenAI, document loading, and LangSmith tracing

## Key Patterns

### Node Creation Pattern
Nodes are **factory functions** that return state transformers:
```python
def create_retrieve_node(documents: Sequence[Document], top_k: int = 3) -> Callable[[AgentState], AgentState]:
    def retrieve(state: AgentState) -> AgentState:
        # Transform state immutably using state.model_copy(update={...})
        return state.model_copy(update={"context": [...], "citations": [...]})
    return retrieve
```

### State Updates
Always use `state.model_copy(update={...})` for immutable state updates, never direct assignment.

### Document Processing
FAQ documents are **automatically split by level-2 headings** (`## `) into sections, each becoming a separate LangChain Document with metadata `{"source": "filename.md#section-N"}`.

### Retrieval Strategy
Uses **basic SequenceMatcher similarity** (not embeddings) between question and document content. Simple but effective for small FAQ datasets.

## Development Workflow

### Setup Commands
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -U pip && pip install -e .[dev]
cp .env.example .env    # Configure OPENAI_API_KEY, LANGCHAIN_* vars
```

### CLI Usage
```bash
python -m agent_spec_lab.cli ask "How do I install the agent?"
# Loads data/faq/*.md, builds graph, returns answer + citations
```

### Code Quality (CI enforced)
```bash
ruff check . && ruff format --check . && black --check .
mypy src tests
pytest
```

## Testing Strategy

- **Fixtures**: `tests/conftest.py` provides `faq_documents` and `fake_llm` fixtures
- **FakeChatModel**: Deterministic test model that echoes input for predictable testing
- **Test Focus**: Graph invocation and state transformations, not LLM behavior

## Environment Variables

Required: `OPENAI_API_KEY`, `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`
The `start_tracing()` context manager handles LangSmith environment setup automatically.

## Extension Points

- Add new nodes by creating factory functions in `src/agent_spec_lab/nodes/`
- Extend state by adding fields to `AgentState` 
- Replace retrieval with vector search by swapping `create_retrieve_node()`
- Add new document types by extending `load_faq_documents()`