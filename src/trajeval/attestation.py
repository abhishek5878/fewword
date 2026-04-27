"""Cryptographic attestation for fewwords verification results.

When `fewwords.check()` produces a verdict (ALLOW or BLOCK with reasons),
:func:`attest` signs a receipt of that decision, producing an audit trail
that is regulator-readable and tamper-evident.

Receipts are HMAC-SHA256 signed with a configurable key. The
:class:`Ledger` class provides an append-only JSONL store with chain
integrity, each receipt's signature includes the previous receipt's
signature, so removing any record breaks the chain.

Layer rule: this module imports only stdlib + sdk.models. No FastAPI,
no httpx, no networkx. Ships in the public bundle.

Example::

    from trajeval.attestation import attest

    receipt = attest(
        trace_id="abc-123",
        trace_dict=trace_obj.model_dump(),
        contract_dict=config.model_dump(),
        verdict="BLOCK",
        reasons=["banned:drop_database"],
        ledger_path=".fewwords/ledger.jsonl",
    )
    print(receipt.signature)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_KEY_ENV = "FEWWORDS_SIGNING_KEY"
_DEV_KEY_FALLBACK = b"fewwords-dev-key-DO-NOT-USE-IN-PRODUCTION"


@dataclass(frozen=True)
class AttestationReceipt:
    """A signed verdict on an agent action.

    Tamper-evident: changing any field invalidates the signature.
    Chain-evident: each receipt includes the previous receipt's
    signature, so removing any record breaks the chain integrity check.
    """

    trace_id: str
    trace_digest: str
    contract_digest: str
    verdict: str
    reasons: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    prev_signature: str = ""
    signature: str = ""


def _canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization for hashing and signing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def trace_digest(trace_dict: dict[str, Any]) -> str:
    """SHA-256 over the canonical-JSON form of a trace dict."""
    return hashlib.sha256(_canonical_json(trace_dict).encode("utf-8")).hexdigest()


def contract_digest(contract_dict: dict[str, Any]) -> str:
    """SHA-256 over the canonical-JSON form of a contract dict."""
    return hashlib.sha256(_canonical_json(contract_dict).encode("utf-8")).hexdigest()


def _resolve_key(key: bytes | str | None) -> bytes:
    """Resolve the HMAC signing key from arg, env, or development fallback."""
    if key is not None:
        return key.encode("utf-8") if isinstance(key, str) else key
    env_key = os.environ.get(DEFAULT_KEY_ENV)
    if env_key:
        return env_key.encode("utf-8")
    return _DEV_KEY_FALLBACK


def _sign_payload(payload: dict[str, Any], key: bytes) -> str:
    """HMAC-SHA256 signature over the canonical JSON payload."""
    return hmac.new(key, _canonical_json(payload).encode("utf-8"), hashlib.sha256).hexdigest()


def sign_receipt(
    *,
    trace_id: str,
    trace_dict: dict[str, Any],
    contract_dict: dict[str, Any],
    verdict: str,
    reasons: list[str] | None = None,
    prev_signature: str = "",
    timestamp: float | None = None,
    key: bytes | str | None = None,
) -> AttestationReceipt:
    """Construct a signed :class:`AttestationReceipt`."""
    body = {
        "trace_id": trace_id,
        "trace_digest": trace_digest(trace_dict),
        "contract_digest": contract_digest(contract_dict),
        "verdict": verdict,
        "reasons": list(reasons or []),
        "timestamp": timestamp if timestamp is not None else time.time(),
        "prev_signature": prev_signature,
    }
    sig = _sign_payload(body, _resolve_key(key))
    return AttestationReceipt(**body, signature=sig)


def verify_receipt(receipt: AttestationReceipt, key: bytes | str | None = None) -> bool:
    """Recompute the signature and compare in constant time."""
    body = {k: v for k, v in asdict(receipt).items() if k != "signature"}
    expected = _sign_payload(body, _resolve_key(key))
    return hmac.compare_digest(expected, receipt.signature)


@dataclass
class Ledger:
    """Append-only JSONL ledger of attestation receipts.

    Each receipt's prev_signature points at the previous receipt's
    signature; :meth:`verify_chain` walks the file and checks both
    signature integrity and chain integrity.
    """

    path: Path

    def __post_init__(self) -> None:
        if isinstance(self.path, str):
            self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, receipt: AttestationReceipt) -> None:
        with open(self.path, "a") as f:
            f.write(_canonical_json(asdict(receipt)) + "\n")

    def read_all(self) -> list[AttestationReceipt]:
        if not self.path.exists():
            return []
        out: list[AttestationReceipt] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(AttestationReceipt(**json.loads(line)))
        return out

    def latest_signature(self) -> str:
        receipts = self.read_all()
        return receipts[-1].signature if receipts else ""

    def verify_chain(self, key: bytes | str | None = None) -> bool:
        prev = ""
        for r in self.read_all():
            if r.prev_signature != prev:
                return False
            if not verify_receipt(r, key):
                return False
            prev = r.signature
        return True


def attest(
    *,
    trace_id: str,
    trace_dict: dict[str, Any],
    contract_dict: dict[str, Any],
    verdict: str,
    reasons: list[str] | None = None,
    ledger_path: Path | str | None = None,
    key: bytes | str | None = None,
) -> AttestationReceipt:
    """Sign a verdict and (optionally) append it to a ledger.

    The top-level public attestation API. Use after :func:`fewwords.check`
    or :func:`run_checks` to produce a tamper-evident audit record.
    """
    prev_sig = ""
    if ledger_path is not None:
        prev_sig = Ledger(Path(ledger_path)).latest_signature()

    receipt = sign_receipt(
        trace_id=trace_id,
        trace_dict=trace_dict,
        contract_dict=contract_dict,
        verdict=verdict,
        reasons=reasons,
        prev_signature=prev_sig,
        key=key,
    )

    if ledger_path is not None:
        Ledger(Path(ledger_path)).append(receipt)

    return receipt


__all__ = [
    "AttestationReceipt",
    "Ledger",
    "attest",
    "contract_digest",
    "sign_receipt",
    "trace_digest",
    "verify_receipt",
]
