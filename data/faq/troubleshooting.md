# Troubleshooting FAQ

## What if the agent cannot answer a question?
If the context snippets do not contain the answer, the agent will explain that
the information is unavailable. Consider adding a new markdown section that
covers the topic.

## How do I enable LangSmith tracing?
Set `LANGCHAIN_TRACING_V2=true` and provide `LANGCHAIN_API_KEY` and optional
`LANGCHAIN_PROJECT` values in your `.env` file. Traces will appear in the
configured LangSmith project.
