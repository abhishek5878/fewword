"""fewwords, the safety net for AI agents in production.

`pip install fewwords` ships this package as the user-facing entry point.
The internal Python package is `trajeval` (preserved for backward
compatibility); this module re-exports it so `from fewwords import ...`
works as expected.

Both `import fewwords` and `import trajeval` are supported. So are both
the `fewwords` and `trajeval` console scripts.

The brand is fewwords. The philosophy is in the name: fewer words, plain
rules, no LLM grading the LLM.

    "Me think, why waste time say lot word, when few word do trick."
    (Kevin Malone, The Office)
"""

from __future__ import annotations

from trajeval import (
    end_session,
    get_callback,
    get_traces,
    init,
    reset,
)
from trajeval.attestation import (
    AttestationReceipt,
    Ledger,
    attest,
    sign_receipt,
    verify_receipt,
)

__all__ = [
    "AttestationReceipt",
    "Ledger",
    "attest",
    "end_session",
    "get_callback",
    "get_traces",
    "init",
    "reset",
    "sign_receipt",
    "verify_receipt",
]
