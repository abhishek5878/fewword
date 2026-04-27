"""Unit tests for auto-detect trace format (A7)."""

from __future__ import annotations

import pytest

from trajeval.adapters.auto import auto_detect

# ---------------------------------------------------------------------------
# Native TrajEval format
# ---------------------------------------------------------------------------


def test_detect_native_trajeval() -> None:
    payload = {
        "trace_id": "t1",
        "agent_id": "agent",
        "version_hash": "v1",
        "started_at": "2024-01-01T00:00:00Z",
        "completed_at": "2024-01-01T00:00:00Z",
        "nodes": [
            {
                "node_id": "n0",
                "node_type": "tool_call",
                "tool_name": "search",
                "tool_input": {},
                "tool_output": {"results": ["a"]},
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ],
        "edges": [],
    }
    result = auto_detect(payload)
    assert result.trace.trace_id == "t1"
    assert len(result.trace.nodes) == 1
    assert result.trace.nodes[0].tool_name == "search"


# ---------------------------------------------------------------------------
# OpenAI messages (list of dicts with "role")
# ---------------------------------------------------------------------------


def test_detect_openai_messages_list() -> None:
    messages = [
        {"role": "user", "content": "Book a flight"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search_flights",
                        "arguments": '{"dest": "SFO"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"flights": []}',
        },
    ]
    result = auto_detect(messages, agent_id="booking")
    assert result.trace.agent_id == "booking"
    tool_nodes = [
        n for n in result.trace.nodes if n.node_type == "tool_call"
    ]
    assert len(tool_nodes) >= 1


def test_detect_openai_messages_wrapped_dict() -> None:
    payload = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
    }
    result = auto_detect(payload, agent_id="chat")
    assert result.trace.agent_id == "chat"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_error_on_empty_list() -> None:
    with pytest.raises(ValueError, match="empty list"):
        auto_detect([])


def test_error_on_unrecognized_dict() -> None:
    with pytest.raises(ValueError, match="unrecognized dict"):
        auto_detect({"foo": "bar", "baz": 123})


def test_error_on_non_dict_non_list() -> None:
    with pytest.raises(ValueError, match="expected dict or list"):
        auto_detect("not json")  # type: ignore[arg-type]


def test_error_on_unrecognized_list() -> None:
    with pytest.raises(ValueError, match="unrecognized list"):
        auto_detect([{"x": 1}, {"y": 2}])


# ---------------------------------------------------------------------------
# Integration: auto-detect via CLI
# ---------------------------------------------------------------------------


def test_cli_run_auto_detects_native_format() -> None:
    """trajeval run should auto-detect native TrajEval format."""
    import json
    import os
    import tempfile

    import yaml

    from trajeval.cli import main

    trace_data = {
        "trace_id": "auto-t",
        "agent_id": "a",
        "version_hash": "v",
        "started_at": "2024-01-01T00:00:00Z",
        "completed_at": "2024-01-01T00:00:00Z",
        "nodes": [
            {
                "node_id": "n0",
                "node_type": "tool_call",
                "tool_name": "search",
                "tool_input": {},
                "tool_output": {},
                "timestamp": "2024-01-01T00:00:00Z",
            }
        ],
        "edges": [],
    }
    fd, tpath = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(tpath, "w") as f:
        json.dump(trace_data, f)

    with tempfile.NamedTemporaryFile(
        suffix=".yml", mode="w", delete=False
    ) as cf:
        yaml.dump({"max_retries": 3}, cf)
        cpath = cf.name

    try:
        rc = main(["run", tpath, "--config", cpath])
        assert rc == 0
    finally:
        os.unlink(tpath)
        os.unlink(cpath)
