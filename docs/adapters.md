# Trace adapters

The Gate operates on a native `Trace` object. Adapters convert
external formats into that object.

Built-in adapters:

- **OpenAI-style** (`openai.py`) — function/tool calls in OpenAI
  chat-completion format.
- **OpenTelemetry** (`otel.py`) — GenAI spans in OTLP format.
- **LangGraph** (`langgraph.py`) — events and threads from a
  LangGraph run.
- **LiteLLM** (`litellm.py`) — LiteLLM callback logs.

Auto-detection lives in `adapters/auto.py`. Pasting any of the
above at `fewword-ai.fly.dev/analyze` or passing to `trajeval
analyze` works without telling the tool which format it is.

## Writing a new adapter

Subclass `adapters.base.TraceAdapter`. Implement `detect()`
(returning a confidence score for a given payload) and `parse()`
(returning a `Trace`). Register the class at the bottom of the
file; `auto_detect()` picks it up through the ordering in
`adapters/auto.py`.
