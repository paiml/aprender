---
status: complete
ticket: PMAT-1018
row: DEC-D-2
decision: D-2
epic: 2873
kind: decision
decided_by: noahgift via driver prompt 2026-09-06
record: https://github.com/paiml/aprender/issues/2873#issuecomment-5558717537
model: claude-fable-5-1 (orchestrator, recording only)
tokens_used: 0 (a recorded decision, no implementation)
wall_clock_s: 0
turns: 1
---
# decision receipt — DEC-D-2 (PMAT-1018): D-2

**Decision (verbatim from the driver prompt of 2026-09-06, recorded as a comment on #2873):** spec §0 default — the Unsloth throughput concession is not assumed revoked; T-3 ships REPORTING and arms only on P-5 PASS.

- The comment is the record; this receipt exists so the DAG row's status is DERIVED from a receipt like every other row (G-11, scripts/lib/dag_status.py) and `scripts/check_receipt_complete.sh --dag` finds it.
- Rows this decision unblocks are named in the comment and in the DAG's `blockers`.
- Verdict: DONE (recorded; nothing to implement in this row).
