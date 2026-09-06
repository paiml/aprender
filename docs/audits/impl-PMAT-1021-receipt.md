---
status: complete
ticket: PMAT-1021
row: DEC-D-9
decision: D-9
epic: 2873
kind: decision
decided_by: noahgift via driver prompt 2026-09-06
record: https://github.com/paiml/aprender/issues/2873#issuecomment-5558719666
model: claude-fable-5-1 (orchestrator, recording only)
tokens_used: 0 (a recorded decision, no implementation)
wall_clock_s: 0
turns: 1
---
# decision receipt — DEC-D-9 (PMAT-1021): D-9

**Decision (verbatim from the driver prompt of 2026-09-06, recorded as a comment on #2873):** cuda joins default features iff S0-14 PASSES on all four hosts AND scripts/run_clean_room.sh exits 0; today NOT satisfied (S0-14 on intel and mini only; the clean-room script does not exist) ⇒ default = ["cli"], --features cuda documented; R-2 may flip it only with both command outputs pasted into a new decision comment.

- The comment is the record; this receipt exists so the DAG row's status is DERIVED from a receipt like every other row (G-11, scripts/lib/dag_status.py) and `scripts/check_receipt_complete.sh --dag` finds it.
- Rows this decision unblocks are named in the comment and in the DAG's `blockers`.
- Verdict: DONE (recorded; nothing to implement in this row).
