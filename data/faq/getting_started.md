# Getting Started FAQ

## How do I install the agent?
To install the agent-spec-lab project, create a Python 3.11 virtual environment,
install the package in editable mode with `pip install -e .[dev]`, and copy the
`.env.example` file to `.env`.

## Where does the agent find knowledge?
The agent reads markdown files stored in the `data/faq/` directory. You can add
more files to extend the knowledge base without changing the code.
