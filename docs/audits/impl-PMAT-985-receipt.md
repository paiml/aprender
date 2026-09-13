---
status: complete
ticket: PMAT-985
row: G-2
decision: D-5
epic: 2873
kind: decision
decided_by: noahgift via driver prompt 2026-09-06
record: https://github.com/paiml/aprender/issues/2873#issuecomment-5558718271
model: claude-fable-5-1 (orchestrator, recording only)
tokens_used: 0 (a recorded decision, no implementation)
wall_clock_s: 0
turns: 1
---
# decision receipt — G-2 (PMAT-985): D-5

**Decision (verbatim from the driver prompt of 2026-09-06, recorded as a comment on #2873):** the root facade keeps [lib] name = "aprender"; the aprender-core rename is queued for 0.67 (renames, behind DEC-D-8); G-1 carries an allow-list line citing the comment.

- The comment is the record; this receipt exists so the DAG row's status is DERIVED from a receipt like every other row (G-11, scripts/lib/dag_status.py) and `scripts/check_receipt_complete.sh --dag` finds it.
- Rows this decision unblocks are named in the comment and in the DAG's `blockers`.
- Verdict: DONE (recorded; nothing to implement in this row).
