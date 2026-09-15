# Archived 2026-09-02 — superseded by `docs/specifications/PP-LLAMA-001-MASTER.md`

These documents governed, or attempted to govern, inference performance before
`PP-LLAMA-001`. They are archived, not deleted: superseded material that still reads as
current is the condition the master exists to end.

| document | why it is here |
|---|---|
| `performance-parity-llama.cpp.md` | the 1194-line **reviewed draft** the master rewrote. It was introduced by PR #2845 and never reached `main`; this pull request lands the master and archives the draft in one change. Its assets are the `PP-nn` invariant IDs, the §5 protocol text and the §11 superseded-document list (now `evidence/parity/LEDGER.md` §13). Everything else is superseded, and the audit of it is `docs/audits/parity-spec-audit-2026-09-02.md` |
| `benchmarking-gate-spec.md` | a `DRAFT FOR REVIEW` gate specification that **republished the withdrawn comparator table** the master's §2.1 refuses to quote. Superseded by `PP-LLAMA-001-MASTER.md` §6 (invariants), §7 (the gate) and `scripts/perf-matrix.yaml` (every threshold) |

**Retained in `docs/specifications/`, deliberately:** `APR-PERF-GATE-001-status-review.md`.
It is the effectiveness review that justifies the supersession and is not itself superseded.

The 2026-09-01 archiving event is `docs/archive/perf-2026-09-01/README.md`.
