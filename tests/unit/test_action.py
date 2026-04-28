"""Unit tests for the TrajEval Action — the boring version that ships."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from trajeval.action import ActionConfig, ActionResult, load_config, run_checks
from trajeval.sdk.models import Trace, TraceEdge, TraceNode

_TS = "2024-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(
    node_id: str,
    tool_name: str,
    *,
    tool_input: dict[str, object] | None = None,
    tool_output: dict[str, object] | None = None,
    cost: float = 0.0,
) -> TraceNode:
    return TraceNode(
        node_id=node_id,
        node_type="tool_call",
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_output=tool_output or {},
        cost_usd=cost,
        timestamp=_TS,
    )


def _trace(
    nodes: list[TraceNode],
    *,
    total_cost: float = 0.0,
) -> Trace:
    edges = [
        TraceEdge(
            source=nodes[i].node_id,
            target=nodes[i + 1].node_id,
            edge_type="sequential",
        )
        for i in range(len(nodes) - 1)
    ]
    return Trace(
        trace_id="action-test",
        agent_id="agent",
        version_hash="v1",
        started_at=_TS,
        completed_at=_TS,
        nodes=nodes,
        edges=edges,
        total_cost_usd=total_cost,
    )


def _write_config(data: dict[str, object]) -> Path:
    import yaml

    with tempfile.NamedTemporaryFile(
        suffix=".yml", mode="w", delete=False
    ) as fd:
        yaml.dump(data, fd)
    return Path(fd.name)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_parses_yaml() -> None:
    path = _write_config({
        "banned_tools": ["delete_user"],
        "required_tools": ["validate"],
        "cost_budget_usd": 1.50,
        "max_retries": 5,
        "schemas": {"search": {"type": "object"}},
    })
    cfg = load_config(path)
    assert cfg.banned_tools == ["delete_user"]
    assert cfg.required_tools == ["validate"]
    assert cfg.cost_budget_usd == 1.50
    assert cfg.max_retries == 5
    assert "search" in cfg.schemas
    path.unlink()


def test_load_config_defaults() -> None:
    path = _write_config({})
    cfg = load_config(path)
    assert cfg.banned_tools == []
    assert cfg.required_tools == []
    assert cfg.cost_budget_usd is None
    assert cfg.max_retries == 3
    assert cfg.schemas == {}
    path.unlink()


# ---------------------------------------------------------------------------
# run_checks — all pass
# ---------------------------------------------------------------------------


def test_all_checks_pass() -> None:
    trace = _trace([
        _node("n0", "search", tool_output={"results": ["a"]}),
        _node("n1", "validate"),
        _node("n2", "book", tool_output={"id": "b1"}),
    ])
    config = ActionConfig(
        banned_tools=["delete_user"],
        required_tools=["search", "validate"],
        max_retries=3,
    )
    result = run_checks(trace, config)
    assert result.all_passed
    assert result.failed == 0
    assert result.passed >= 3  # retry + banned + required


# ---------------------------------------------------------------------------
# run_checks — individual failures
# ---------------------------------------------------------------------------


def test_retry_storm_detected() -> None:
    nodes = [_node(f"n{i}", "search", tool_input={"q": "same"}) for i in range(5)]
    trace = _trace(nodes)
    config = ActionConfig(max_retries=3)
    result = run_checks(trace, config)
    failed = [c for c in result.checks if not c.passed]
    assert len(failed) == 1
    assert failed[0].name == "retry_storm"


def test_cost_budget_exceeded() -> None:
    trace = _trace(
        [_node("n0", "search", cost=5.0)],
        total_cost=5.0,
    )
    config = ActionConfig(cost_budget_usd=2.0)
    result = run_checks(trace, config)
    failed = [c for c in result.checks if not c.passed]
    assert any(c.name == "cost_budget" for c in failed)


def test_banned_tool_called() -> None:
    trace = _trace([_node("n0", "delete_user")])
    config = ActionConfig(banned_tools=["delete_user"])
    result = run_checks(trace, config)
    failed = [c for c in result.checks if not c.passed]
    assert any("delete_user" in c.name for c in failed)


def test_required_tool_missing() -> None:
    trace = _trace([_node("n0", "search")])
    config = ActionConfig(required_tools=["search", "validate"])
    result = run_checks(trace, config)
    failed = [c for c in result.checks if not c.passed]
    assert any(c.name == "required_tools" for c in failed)


def test_schema_violation() -> None:
    trace = _trace([
        _node("n0", "search", tool_output={"results": "not_array"}),
    ])
    config = ActionConfig(
        schemas={
            "search": {
                "type": "object",
                "properties": {"results": {"type": "array"}},
            }
        }
    )
    result = run_checks(trace, config)
    failed = [c for c in result.checks if not c.passed]
    assert any(c.name == "schema_validation" for c in failed)


# ---------------------------------------------------------------------------
# ActionResult methods
# ---------------------------------------------------------------------------


def test_summary_all_pass() -> None:
    result = ActionResult(
        trace_id="t",
        checks=[
            CheckResult(name="a", passed=True, message="ok"),
            CheckResult(name="b", passed=True, message="ok"),
        ],
    )
    assert "PASS" in result.summary
    assert "2/2" in result.summary


def test_summary_with_failure() -> None:
    result = ActionResult(
        trace_id="t",
        checks=[
            CheckResult(name="a", passed=True, message="ok"),
            CheckResult(name="b", passed=False, message="bad stuff"),
        ],
    )
    assert "FAIL" in result.summary
    assert "1/2" in result.summary
    assert "bad stuff" in result.summary


def test_to_json_roundtrip() -> None:
    result = ActionResult(
        trace_id="t",
        checks=[CheckResult(name="a", passed=True, message="ok")],
    )
    parsed = json.loads(result.to_json())
    assert parsed["all_passed"] is True
    assert parsed["checks"][0]["name"] == "a"


# ---------------------------------------------------------------------------
# cost_budget_usd=None skips cost check
# ---------------------------------------------------------------------------


def test_no_cost_check_when_budget_is_none() -> None:
    trace = _trace(
        [_node("n0", "search", cost=100.0)],
        total_cost=100.0,
    )
    config = ActionConfig()  # cost_budget_usd=None
    result = run_checks(trace, config)
    assert all(c.name != "cost_budget" for c in result.checks)


# ---------------------------------------------------------------------------
# CLI: trajeval run
# ---------------------------------------------------------------------------


def test_cli_run_all_pass() -> None:
    import os

    from trajeval.cli import main

    trace = _trace([
        _node("n0", "search", tool_output={"results": ["a"]}),
        _node("n1", "validate"),
    ])
    # Write trace
    fd, tpath = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(tpath, "w") as f:
        f.write(trace.model_dump_json())
    # Write config
    cpath = _write_config({
        "banned_tools": ["delete_user"],
        "required_tools": ["search"],
        "max_retries": 3,
    })
    try:
        rc = main(["run", tpath, "--config", str(cpath)])
        assert rc == 0
    finally:
        os.unlink(tpath)
        cpath.unlink()


def test_cli_run_failure_exits_one() -> None:
    import os

    from trajeval.cli import main

    trace = _trace([_node("n0", "delete_user")])
    fd, tpath = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(tpath, "w") as f:
        f.write(trace.model_dump_json())
    cpath = _write_config({"banned_tools": ["delete_user"]})
    try:
        rc = main(["run", tpath, "--config", str(cpath)])
        assert rc == 1
    finally:
        os.unlink(tpath)
        cpath.unlink()


# Need this import for test_summary_* tests
from trajeval.action import CheckResult  # noqa: E402

# ---------------------------------------------------------------------------
# tools.<name>.{requires, postcondition} via load_config
# ---------------------------------------------------------------------------


def test_load_config_parses_tools_section() -> None:
    path = _write_config(
        {
            "tools": {
                "cancel_reservation": {
                    "requires": [
                        "state.reservation_known == true",
                        "state.reservation_cabin != basic_economy",
                    ],
                    "postcondition": {
                        "returns": {"success": "bool"},
                        "state_updates": {"reservation_status": "cancelled"},
                    },
                }
            }
        }
    )
    cfg = load_config(path)
    assert "cancel_reservation" in cfg.tools
    entry = cfg.tools["cancel_reservation"]
    assert len(entry.requires) == 2
    assert entry.requires[0].var == "reservation_known"
    assert entry.postcondition is not None
    assert entry.postcondition.returns == {"success": "bool"}
    assert entry.postcondition.state_updates == {
        "reservation_status": "cancelled"
    }
    path.unlink()


def test_load_config_tools_default_empty() -> None:
    path = _write_config({"banned_tools": ["x"]})
    cfg = load_config(path)
    assert cfg.tools == {}
    path.unlink()


def test_load_config_rejects_malformed_tools_predicate() -> None:
    path = _write_config(
        {"tools": {"cancel": {"requires": ["nope == true"]}}}
    )
    with pytest.raises(ValueError, match="state.<var>"):
        load_config(path)
    path.unlink()


# ---------------------------------------------------------------------------
# Postcondition replay with typed symbolic state
# ---------------------------------------------------------------------------


def _tools_config(**raw: object) -> ActionConfig:
    """Build an ActionConfig with a tools section by round-tripping through YAML."""
    path = _write_config({"tools": raw})
    cfg = load_config(path)
    path.unlink()
    return cfg


def _named(check_name: str, result: ActionResult) -> CheckResult:
    return next(c for c in result.checks if c.name == check_name)


def test_replay_no_tools_emits_no_replay_checks() -> None:
    trace = _trace([_node("n0", "anything", tool_output={"x": 1})])
    cfg = ActionConfig()
    result = run_checks(trace, cfg)
    assert all(
        not c.name.startswith(("precondition[", "postcondition_schema["))
        for c in result.checks
    )


def test_replay_precondition_fails_when_state_unset() -> None:
    """`requires: state.X == true` with X never set — precondition fails."""
    cfg = _tools_config(
        cancel_reservation={"requires": ["state.reservation_known == true"]}
    )
    trace = _trace([_node("n0", "cancel_reservation")])
    result = run_checks(trace, cfg)
    check = _named("precondition[cancel_reservation]", result)
    assert check.passed is False
    assert "n0" in check.message
    assert "reservation_known" in check.message


def test_replay_postcondition_sets_state_for_downstream_precondition() -> None:
    """A postcondition that sets state.X enables a later requires: state.X."""
    cfg = _tools_config(
        get_reservation_details={
            "postcondition": {
                "returns": {"reservation_id": "str"},
                "state_updates": {"reservation_known": True},
            }
        },
        cancel_reservation={"requires": ["state.reservation_known == true"]},
    )
    trace = _trace(
        [
            _node(
                "n0",
                "get_reservation_details",
                tool_output={"reservation_id": "R-123"},
            ),
            _node("n1", "cancel_reservation"),
        ]
    )
    result = run_checks(trace, cfg)
    assert _named("precondition[cancel_reservation]", result).passed is True
    schema = _named("postcondition_schema[get_reservation_details]", result)
    assert schema.passed is True


def test_replay_postcondition_schema_fail_blocks_state_mutation() -> None:
    """A bad tool_output fails the schema AND prevents state mutation —
    so downstream preconditions also fail."""
    cfg = _tools_config(
        get_reservation_details={
            "postcondition": {
                "returns": {"reservation_id": "str"},
                "state_updates": {"reservation_known": True},
            }
        },
        cancel_reservation={"requires": ["state.reservation_known == true"]},
    )
    trace = _trace(
        [
            _node(
                "n0",
                "get_reservation_details",
                tool_output={"reservation_id": 42},  # int, schema says str
            ),
            _node("n1", "cancel_reservation"),
        ]
    )
    result = run_checks(trace, cfg)
    schema = _named("postcondition_schema[get_reservation_details]", result)
    pre = _named("precondition[cancel_reservation]", result)
    assert schema.passed is False
    assert "reservation_id" in schema.message
    assert pre.passed is False, "state must NOT mutate when schema fails"


def test_replay_template_var_resolves_from_output() -> None:
    """state_updates: {x: '{cabin}'} resolves against tool_output['cabin']."""
    cfg = _tools_config(
        get_reservation_details={
            "postcondition": {
                "returns": {"cabin": "str"},
                "state_updates": {"reservation_cabin": "{cabin}"},
            }
        },
        cancel_reservation={
            "requires": ["state.reservation_cabin != basic_economy"]
        },
    )
    # case A — cabin is economy → cancel allowed
    trace_ok = _trace(
        [
            _node(
                "n0", "get_reservation_details", tool_output={"cabin": "economy"}
            ),
            _node("n1", "cancel_reservation"),
        ]
    )
    assert _named(
        "precondition[cancel_reservation]", run_checks(trace_ok, cfg)
    ).passed is True

    # case B — cabin is basic_economy → cancel blocked
    trace_bad = _trace(
        [
            _node(
                "n0",
                "get_reservation_details",
                tool_output={"cabin": "basic_economy"},
            ),
            _node("n1", "cancel_reservation"),
        ]
    )
    assert _named(
        "precondition[cancel_reservation]", run_checks(trace_bad, cfg)
    ).passed is False


def test_replay_multiple_violations_collapse_to_one_check() -> None:
    """Existing pattern: one CheckResult per (tool, kind) even with N violations."""
    cfg = _tools_config(
        write_x={"requires": ["state.ready == true"]},
    )
    trace = _trace(
        [
            _node("n0", "write_x"),
            _node("n1", "write_x"),
            _node("n2", "write_x"),
        ]
    )
    result = run_checks(trace, cfg)
    pres = [c for c in result.checks if c.name == "precondition[write_x]"]
    assert len(pres) == 1
    assert pres[0].passed is False
    assert all(f"n{i}" in pres[0].message for i in range(3))


def test_replay_tool_with_no_tools_match_skipped() -> None:
    """Tools not in config.tools never produce replay checks."""
    cfg = _tools_config(only_this={"requires": ["state.x == true"]})
    trace = _trace([_node("n0", "something_else")])
    result = run_checks(trace, cfg)
    assert all(
        not c.name.startswith(("precondition[", "postcondition_schema["))
        for c in result.checks
    )


def test_replay_postcondition_with_only_state_updates_no_schema_check() -> None:
    """Empty `returns` means no schema CheckResult is emitted; state still mutates."""
    cfg = _tools_config(
        seed={
            "postcondition": {"state_updates": {"started": True}}
        },
        gated={"requires": ["state.started == true"]},
    )
    trace = _trace(
        [_node("n0", "seed", tool_output={}), _node("n1", "gated")]
    )
    result = run_checks(trace, cfg)
    schema_checks = [
        c
        for c in result.checks
        if c.name == "postcondition_schema[seed]"
    ]
    assert schema_checks == []
    assert _named("precondition[gated]", result).passed is True


def test_replay_precondition_in_op_with_list_literal() -> None:
    cfg = _tools_config(
        seed={
            "postcondition": {
                "returns": {"cabin": "str"},
                "state_updates": {"cabin": "{cabin}"},
            }
        },
        write={"requires": ["state.cabin in [economy, premium, business]"]},
    )
    trace_ok = _trace(
        [
            _node("n0", "seed", tool_output={"cabin": "economy"}),
            _node("n1", "write"),
        ]
    )
    trace_bad = _trace(
        [
            _node("n0", "seed", tool_output={"cabin": "basic_economy"}),
            _node("n1", "write"),
        ]
    )
    assert _named("precondition[write]", run_checks(trace_ok, cfg)).passed is True
    assert (
        _named("precondition[write]", run_checks(trace_bad, cfg)).passed is False
    )


def test_replay_postcondition_unknown_type_in_returns_emits_error() -> None:
    cfg = _tools_config(
        x={"postcondition": {"returns": {"id": "uuid"}}}  # 'uuid' is not in type map
    )
    trace = _trace([_node("n0", "x", tool_output={"id": "abc"})])
    result = run_checks(trace, cfg)
    schema = _named("postcondition_schema[x]", result)
    assert schema.passed is False
    assert "unknown type" in schema.message
