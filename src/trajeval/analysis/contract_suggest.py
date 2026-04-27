"""Contract pack suggestion — recommend which vertical pack fits a trace.

When a developer pastes a trace, this module inspects the tool names
and suggests the best-matching contract pack + a mapping from the
pack's generic tool names to the trace's actual tool names.

Usage::

    from trajeval.analysis.contract_suggest import suggest_pack

    suggestion = suggest_pack(trace)
    print(f"Best match: {suggestion.pack_name}")
    print(f"Confidence: {suggestion.confidence:.0%}")
    for generic, actual in suggestion.tool_mapping.items():
        print(f"  {generic} → {actual}")
"""

from __future__ import annotations

from dataclasses import dataclass

from trajeval.sdk.models import Trace


@dataclass(frozen=True)
class PackSuggestion:
    """Suggested contract pack + tool name mapping."""

    pack_name: str
    confidence: float
    tool_mapping: dict[str, str | None]
    unmapped_tools: list[str]
    pack_path: str

    @property
    def is_confident(self) -> bool:
        """True only if the suggestion is safe to recommend.

        We require at least 50% confidence AND at least half the
        trace's tools to map into the pack. Low-confidence
        suggestions should fall back to ``contracts/generic.yml``
        instead of pushing the user toward a wrong vertical.
        """
        total_tools = len(self.tool_mapping) + len(self.unmapped_tools)
        if total_tools == 0:
            return False
        mapped_frac = len(self.tool_mapping) / total_tools
        return self.confidence >= 0.5 and mapped_frac >= 0.5


# Each pack: (name, path, keyword → generic_tool_name mapping)
_PACKS: list[tuple[str, str, dict[str, str]]] = [
    ("code_agents", "contracts/code_agents.yml", {
        "read": "read_file", "edit": "edit_file", "write": "edit_file",
        "test": "run_tests", "lint": "run_linter", "commit": "git_commit",
        "grep": "grep_search", "search": "grep_search",
        "execute": "execute_code", "python": "execute_code",
        "drop": "drop_database", "delete_db": "drop_database",
    }),
    ("healthcare", "contracts/healthcare.yml", {
        "patient": "lookup_patient", "verify": "verify_patient_identity",
        "allerg": "check_allergies", "drug": "check_drug_interactions",
        "prescri": "prescribe_medication", "symptom": "check_symptoms",
        "appointment": "schedule_appointment", "insurance": "check_insurance",
        "medical": "lookup_patient", "diagnos": "check_symptoms",
    }),
    ("financial", "contracts/financial.yml", {
        "trade": "place_order", "order": "place_order", "buy": "place_order",
        "sell": "place_order", "quote": "get_quote", "price": "get_quote",
        "balance": "check_balance", "portfolio": "get_portfolio",
        "verify": "verify_identity", "identity": "verify_identity",
        "margin": "margin_trade", "compliance": "check_compliance",
    }),
    ("support", "contracts/support.yml", {
        "ticket": "lookup_ticket", "customer": "lookup_customer",
        "refund": "process_refund", "verify": "verify_identity",
        "escalat": "escalate_to_human", "knowledge": "search_knowledge_base",
        "survey": "send_survey", "account": "lookup_account",
    }),
    ("legal", "contracts/legal.yml", {
        "precedent": "search_precedent", "case": "search_precedent",
        "draft": "draft_document", "document": "draft_document",
        "conflict": "check_conflicts", "court": "file_court_document",
        "privilege": "waive_privilege", "flag_for_review": "flag_for_review",
    }),
]


def suggest_pack(trace: Trace) -> PackSuggestion:
    """Suggest the best-matching contract pack for a trace."""
    tool_names = {
        n.tool_name.lower()
        for n in trace.nodes
        if n.node_type == "tool_call" and n.tool_name
    }
    if not tool_names:
        return PackSuggestion(
            pack_name="code_agents",
            confidence=0.0,
            tool_mapping={},
            unmapped_tools=[],
            pack_path="contracts/code_agents.yml",
        )

    best_pack = ""
    best_score = 0.0
    best_mapping: dict[str, str | None] = {}
    best_path = ""

    for pack_name, pack_path, keywords in _PACKS:
        mapping: dict[str, str | None] = {}
        hits = 0

        for tool in tool_names:
            for keyword, generic in keywords.items():
                if keyword in tool:
                    mapping[generic] = tool
                    hits += 1
                    break

        score = hits / max(len(tool_names), 1)
        if score > best_score:
            best_score = score
            best_pack = pack_name
            best_mapping = mapping
            best_path = pack_path

    unmapped = [
        t for t in sorted(tool_names)
        if t not in best_mapping.values()
    ]

    return PackSuggestion(
        pack_name=best_pack,
        confidence=min(0.95, best_score),
        tool_mapping=best_mapping,
        unmapped_tools=unmapped,
        pack_path=best_path,
    )
