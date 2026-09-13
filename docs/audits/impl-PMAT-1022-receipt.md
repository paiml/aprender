---
status: complete
ticket: PMAT-1022
row: DEC-D-10
decision: D-10
epic: 2873
kind: decision
decided_by: noahgift via driver prompt 2026-09-06
record: https://github.com/paiml/aprender/issues/2873#issuecomment-5558719920
model: claude-fable-5-1 (orchestrator, recording only)
tokens_used: 0 (a recorded decision, no implementation)
wall_clock_s: 0
turns: 1
---
# decision receipt — DEC-D-10 (PMAT-1022): D-10

**Decision (verbatim from the driver prompt of 2026-09-06, recorded as a comment on #2873):** spec §0 default per target from S0-16: five nightly targets with cli; cuda only where R-5 runs S0-14 for that target under D-9; each target feature set printed into the release manifest beside its sha256.

- The comment is the record; this receipt exists so the DAG row's status is DERIVED from a receipt like every other row (G-11, scripts/lib/dag_status.py) and `scripts/check_receipt_complete.sh --dag` finds it.
- Rows this decision unblocks are named in the comment and in the DAG's `blockers`.
- Verdict: DONE (recorded; nothing to implement in this row).
