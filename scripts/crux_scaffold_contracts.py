#!/usr/bin/env python3
"""
Scaffold 250 sub-contract YAMLs from the CRUX master contract.

Deterministic, idempotent: re-running overwrites only files whose
`status: draft` — never touches enriched contracts.

Output: contracts/crux-{ID}-v1.yaml, one per story.
"""
from __future__ import annotations
import sys
import pathlib
import subprocess
import json

ROOT = pathlib.Path(__file__).resolve().parent.parent
MASTER = ROOT / "contracts" / "crux-competitive-research-ux-v1.yaml"
CONTRACTS_DIR = ROOT / "contracts"

CATEGORY_NAMES = {
    "A": "Model Intake & Discovery",
    "B": "Inspection & Debugging",
    "C": "Serving & Chat UX",
    "D": "Inference & Generation",
    "E": "Training & Fine-tuning",
    "F": "Tensor Ops & Low-level",
    "G": "Format Conversion",
    "H": "Data & Datasets",
    "I": "Observability & Metrics",
    "J": "Vision-Language (OpenCLIP/OpenCLAW)",
    "K": "Ecosystem & Integration",
}

STATUS_BADGE = {
    "supported": "active",
    "partial": "draft",
    "missing": "draft",
    "unclear": "draft",
}

PRIORITY = {5: "critical", 4: "high", 3: "medium", 2: "low", 1: "low"}


def read_master() -> list[dict]:
    out = subprocess.check_output(["yq", "-o", "json", ".stories", str(MASTER)])
    return json.loads(out)


def make_contract(story: dict) -> str:
    sid: str = story["id"]  # e.g. CRUX-A-01
    parts = sid.split("-")
    category = parts[1]
    title = story["title"]
    competitor = story["competitor"]
    demand = int(story["demand_score"])
    status = story["status"]
    contract_status = STATUS_BADGE[status]
    prio = PRIORITY[demand]
    cat_name = CATEGORY_NAMES.get(category, category)

    openclip_note = ""
    if story.get("interpretation") == "pending":
        openclip_note = (
            '\n  openclaw_interpretation: pending-user-confirmation'
            '\n  # Category J assumes OpenCLIP until user resolves OpenCLAW (see master §10).'
        )

    return f"""# {sid} — {title}
# Auto-scaffolded from crux-competitive-research-ux-v1.yaml (DRAFT).
# DO NOT promote to ACTIVE until: (1) evidence collected, (2) falsification
# body reflects competitor's canonical CLI, (3) apr surface mapped or gap
# tracked as a work ticket (see master §12).

metadata:
  id: {sid}
  version: "1.0.0-draft"
  created: "2026-04-18"
  author: PAIML Engineering
  registry: true
  status: {contract_status}
  parent_contracts:
  - crux-competitive-research-ux-v1
  category: "{category} — {cat_name}"
  competitor: {competitor}
  demand_score: {demand}  # 1..5, {prio} priority as a work ticket
  intake_status: {status}
  description: >
    {title}. Root-cause workflow extracted from {competitor} UX — see master
    subspec §5.{category} and §2 Five Whys methodology for rationale. This
    draft contract exists to satisfy the Iron Rule "no contract → no user
    story"; the falsification body below is a placeholder and MUST be
    replaced with the competitor's canonical CLI transcript before promotion.{openclip_note}

equations:
  placeholder_contract:
    formula: |
      DRAFT — user workflow "{title}"
      must produce observable parity with {competitor}'s canonical verb
      on the golden input defined in evidence/crux/{competitor}/.
    domain: user invocation
    codomain: apr CLI output
    invariants:
    - "TBD — replace with actual competitor parity invariants"
    - "TBD — replace with falsifiable output contract"

falsification_tests:
- id: FALSIFY-{sid}-001
  rule: placeholder
  prediction: "apr surface for '{title}' matches {competitor} canonical behavior"
  test: |
    # TBD — collect {competitor} transcript in evidence/crux/{competitor}/
    # and write a shell test that compares apr output to the golden.
    echo "TODO: falsification body for {sid}" >&2
    exit 2  # non-zero = placeholder not yet implemented
  if_fails: "draft contract — falsification body not yet authored"

proof_obligations:
- type: invariant
  property: "TBD placeholder — replace with concrete obligation before promotion"

verification_summary:
  total_obligations: 1
  proven: 0
  tested: 0
  status: draft

pmat_work_tracking:
  # Managed by master contract §12. If intake_status == "missing", a ticket
  # tagged `crux-{sid.lower()}` is auto-created by scripts/crux_bulk_pmat_work.sh.
  ticket_tag: crux-{sid.lower()}
  priority: {prio}
  auto_created: {"true" if status == "missing" else "false"}
"""


def main() -> int:
    stories = read_master()
    assert len(stories) == 250, f"expected 250 stories, got {len(stories)}"

    created = 0
    skipped = 0
    for story in stories:
        sid = story["id"]
        path = CONTRACTS_DIR / f"crux-{sid}-v1.yaml".lower().replace("crux-crux-", "crux-")
        # Canonical: contracts/crux-A-01-v1.yaml (preserve case on ID segment)
        path = CONTRACTS_DIR / f"crux-{sid.replace('CRUX-', '')}-v1.yaml"

        if path.exists():
            # Idempotent: only overwrite if still in draft state
            first = path.read_text().splitlines()[0] if path.stat().st_size else ""
            if "DRAFT" not in path.read_text()[:600].upper() and "draft" not in path.read_text()[:600]:
                skipped += 1
                continue

        path.write_text(make_contract(story))
        created += 1

    print(f"scaffolded {created} contracts, skipped {skipped} (non-draft)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
