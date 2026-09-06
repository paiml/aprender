---
status: complete
merged: "PR #2987 squash 65680cdf8, 2026-09-06T02:37:35Z, through the merge queue (queue CI run 34005252966: ci / gate + workspace-test green)"
ticket: PMAT-1056
row: C0-7
issue: 2984
epic: 2873
model: "orchestrator claude-fable-5-1 (direct)"
tokens_used: "orchestrator [U] (not instrumented)"
wall_clock_s: "[U] (not instrumented); 4 orchestrator bash calls for P_1–P_4"
---
# impl-PMAT-1056 — C0-7 · receipt terminal marker + check_receipt_complete.sh (#2984; driver STEP A6)

## Identity
ticket PMAT-1056 · kind code · branch `agent/C0-7` (worktree, claim held) · base `132ccda56` · `discover.json` at `$XDG_RUNTIME_DIR/paiml-implement/discover-C0-7.json` (`gate_cmd_fallback=true`; no Rust) · quorum `--quorum never` (review-only row) · K̂ = 184 (`basis=docs/audits/impl-estimates.jsonl:L1-L7`).

## What lands
- `scripts/check_receipt_complete.sh` — `<receipt>` mode (0 complete · 1 partial / no marker / `*.tmp` torn · 2 usage), `--dag [<yaml>]` mode (every row with `status: complete` has a receipt whose leading front matter says `status: complete`; a tracked `*.tmp` receipt is refused; a missing DAG is exit 2), `--selftest` (13 rows, both polarities, incl. the registered mutation: a row marked complete over a partial receipt → RED).
- `ci.yml` `guard-runner-labels`: case table, then `--dag`.
- `contracts/apr-impl-receipt-v1.yaml` (`kind: pattern`).
- The write discipline (`.tmp` then `mv`) is used by every receipt this run wrote (PMAT-1054, PMAT-980, PMAT-987, this one).

## Verification (orchestrator)
| check | result |
|---|---|
| `bash scripts/check_receipt_complete.sh --selftest` | rc 0, `13/13 rows` |
| `bashrs lint scripts/check_receipt_complete.sh` | 0 errors |
| existing receipts under the rule | `impl-PMAT-929-receipt.md` → no marker → partial (legacy; RED only if the DAG claims it complete — it does not) |
| `--dag` on the plan branch's DAG (untracked copy) | PASS (no row is complete yet) |
| `check_guards_are_wired.sh` · `check_workflow_env_defined.sh` | rc 0 |
| `pv validate contracts/apr-impl-receipt-v1.yaml` · `check_contract_test_binding.sh` · `check_contract_enforcement.sh` · `pv lint` | rc 0 · rc 0 · rc 0 · 0 errors |

## Mutation (RED, restored GREEN)
DAG row C0-5 set to `status: complete` while `docs/audits/impl-PMAT-1054-receipt.md` says `status: partial` (both untracked copies) → `--dag` rc 1: `FAIL C0-5 (PMAT-1054): the DAG says status: complete but docs/audits/impl-PMAT-1054-receipt.md is partial`. Restored → rc 0.

## Gaps
- Merges after #2981 (the DAG) in the STEP B order; until then the `--dag` step is exit 2 on a branch without the file.
- Receipt for this PR: advisory, not produced (driver A1).

## Merge evidence (orchestrator, after the fact)
- Merged as `65680cdf8` (#2987, the STEP B batch G-6 → G-4 → SPEC-1.6 → C0-7) at 2026-09-06T02:37:35Z through the merge queue; queue CI run 34005252966: `ci / gate` and `workspace-test` green; PR run 34003610208: every gate leg green, including the `guard-runner-labels` steps this row adds.
- Every A_i was re-run by the orchestrator on the batch tip before the push (PR body, "Acceptance commands" table) and the row's registered mutation shown RED then restored GREEN there.

## Verdict
DONE (`status: complete`).
