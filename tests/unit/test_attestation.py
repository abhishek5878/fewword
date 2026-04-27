"""Unit tests for trajeval.attestation."""

from __future__ import annotations

import tempfile
from pathlib import Path

from trajeval.attestation import (
    AttestationReceipt,
    Ledger,
    attest,
    contract_digest,
    sign_receipt,
    trace_digest,
    verify_receipt,
)

KEY = b"test-signing-key-not-for-production"


def test_trace_digest_is_deterministic():
    a = trace_digest({"trace_id": "t1", "nodes": [1, 2, 3]})
    b = trace_digest({"nodes": [1, 2, 3], "trace_id": "t1"})
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_trace_digest_changes_on_any_field():
    a = trace_digest({"trace_id": "t1", "nodes": [1, 2, 3]})
    b = trace_digest({"trace_id": "t1", "nodes": [1, 2, 4]})
    assert a != b


def test_contract_digest_is_independent_of_trace_digest():
    t = {"trace_id": "t1"}
    c = {"banned_tools": ["drop_database"]}
    assert trace_digest(t) != contract_digest(c)


def test_sign_and_verify_roundtrip():
    r = sign_receipt(
        trace_id="t1",
        trace_dict={"a": 1},
        contract_dict={"b": 2},
        verdict="ALLOW",
        reasons=[],
        key=KEY,
    )
    assert verify_receipt(r, key=KEY)


def test_verify_fails_on_wrong_key():
    r = sign_receipt(
        trace_id="t1",
        trace_dict={"a": 1},
        contract_dict={"b": 2},
        verdict="ALLOW",
        key=KEY,
    )
    assert not verify_receipt(r, key=b"different-key")


def test_verify_fails_on_tampered_verdict():
    r = sign_receipt(
        trace_id="t1",
        trace_dict={"a": 1},
        contract_dict={"b": 2},
        verdict="ALLOW",
        key=KEY,
    )
    tampered = AttestationReceipt(
        trace_id=r.trace_id,
        trace_digest=r.trace_digest,
        contract_digest=r.contract_digest,
        verdict="BLOCK",  # CHANGED
        reasons=r.reasons,
        timestamp=r.timestamp,
        prev_signature=r.prev_signature,
        signature=r.signature,
    )
    assert not verify_receipt(tampered, key=KEY)


def test_ledger_append_and_read():
    with tempfile.TemporaryDirectory() as d:
        ledger_path = Path(d) / "ledger.jsonl"
        ledger = Ledger(ledger_path)
        r1 = sign_receipt(
            trace_id="t1",
            trace_dict={"a": 1},
            contract_dict={"b": 2},
            verdict="ALLOW",
            key=KEY,
        )
        ledger.append(r1)
        all_receipts = ledger.read_all()
        assert len(all_receipts) == 1
        assert all_receipts[0].signature == r1.signature


def test_ledger_chain_integrity():
    with tempfile.TemporaryDirectory() as d:
        ledger_path = Path(d) / "ledger.jsonl"
        r1 = attest(
            trace_id="t1",
            trace_dict={"a": 1},
            contract_dict={"b": 2},
            verdict="ALLOW",
            ledger_path=ledger_path,
            key=KEY,
        )
        r2 = attest(
            trace_id="t2",
            trace_dict={"a": 2},
            contract_dict={"b": 2},
            verdict="BLOCK",
            reasons=["test"],
            ledger_path=ledger_path,
            key=KEY,
        )
        assert r2.prev_signature == r1.signature
        assert Ledger(ledger_path).verify_chain(key=KEY)


def test_ledger_chain_breaks_on_removal():
    """If a record is removed, the chain check must fail."""
    with tempfile.TemporaryDirectory() as d:
        ledger_path = Path(d) / "ledger.jsonl"
        attest(
            trace_id="t1",
            trace_dict={"a": 1},
            contract_dict={"b": 2},
            verdict="ALLOW",
            ledger_path=ledger_path,
            key=KEY,
        )
        attest(
            trace_id="t2",
            trace_dict={"a": 2},
            contract_dict={"b": 2},
            verdict="BLOCK",
            ledger_path=ledger_path,
            key=KEY,
        )
        attest(
            trace_id="t3",
            trace_dict={"a": 3},
            contract_dict={"b": 2},
            verdict="ALLOW",
            ledger_path=ledger_path,
            key=KEY,
        )
        # Remove the middle record
        lines = ledger_path.read_text().splitlines()
        assert len(lines) == 3
        ledger_path.write_text(lines[0] + "\n" + lines[2] + "\n")
        assert not Ledger(ledger_path).verify_chain(key=KEY)
