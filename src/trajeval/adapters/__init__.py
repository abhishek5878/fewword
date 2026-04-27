"""TrajEval adapter layer — convert user-native trace formats into TrajEval Traces.

This module is the ingestion side of the OTel integration story.
``sdk/otel.py`` exports TrajEval Traces *to* OTel spans.
These adapters import *from* user-native formats into TrajEval Traces.

Supported sources
-----------------
- OpenAI messages array (``adapters.openai``)
- OpenTelemetry GenAI semantic convention spans (``adapters.otel``)
- LangGraph astream_events JSONL (``adapters.langgraph``)

Each adapter returns an :class:`~trajeval.adapters.base.AdapterResult`
containing a normalized :class:`~trajeval.sdk.models.Trace` and an
:class:`~trajeval.adapters.base.AdapterCapabilities` manifest that declares
which TrajEval analysis modules can run given the fields that were recoverable
from the source format.

Layer rule: this module is part of the SDK layer. Zero imports from
``trajeval.backend``.
"""

from trajeval.adapters.auto import auto_detect
from trajeval.adapters.base import AdapterCapabilities, AdapterResult
from trajeval.adapters.langgraph import from_langgraph_events, from_langgraph_jsonl
from trajeval.adapters.openai import from_openai_messages
from trajeval.adapters.otel import from_otel_spans

__all__ = [
    "AdapterCapabilities",
    "AdapterResult",
    "auto_detect",
    "from_langgraph_events",
    "from_langgraph_jsonl",
    "from_openai_messages",
    "from_otel_spans",
]
