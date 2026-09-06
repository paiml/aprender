---
status: partial
ticket: PMAT-1062
row: G-11
epic: 2873
branch: agent/G-11
pr: opened after PR-A (#3011) merges — the only PR in the queue until then (driver PHASE 1)
model: claude-fable-5-1 (orchestrator, direct) · review quorum on agy lanes recorded below when run
tokens_used: orchestrator [U] (not exposed to the orchestrator)
wall_clock_s: 2100 (basis=session clock, G-11 claim to the receipt commit; [U] precision)
turns: 9 (orchestrator turns on this ticket, counted from the transcript)
---
# impl receipt — PMAT-1062 (PP-066 row G-11): shared-file write contention

## Identity
- kind: code · branch `agent/G-11` off `origin/main` 027ed889d · this receipt is the second commit
- depends on PR-A (#3011, PMAT-1059): `scripts/check_row_pr_write_set.sh` names its base through `scripts/lib/resolve_base.sh`, which PR-A adds. Until #3011 merges the worktree carries an UNTRACKED copy for local runs; the branch is rebased onto main after the merge and the copy becomes the tracked file.
- write set: `scripts/lib/dag_status.py` (new), `scripts/render_dag.py`, `scripts/lib/dag_invariants.py`, `scripts/check_dag_invariants.sh`, `scripts/check_row_pr_write_set.sh` (new), `scripts/check_readme_claims.sh`, `contracts/apr-row-pr-write-set-v1.yaml` (new), `.github/workflows/ci.yml` (four steps in guard-runner-labels), this receipt. **No DAG, roadmap, spec or README edit** — the rule this row installs, obeyed by its own PR (the README lags by one contract, which the new ratchet permits).

## Why (five whys, short)
Eight armed row PRs each hand-edited the DAG status, the roadmap or a README count → every merge made the other seven DIRTY → each needed a rebuild through a ~1 PR/hour queue → because completion was recorded in three shared files no row owns → because nothing derived it from the artifact the row does own (its receipt). The fix moves the record to the receipt and forbids the shared writes mechanically.

## Plan and routing (all direct; the row is `quorum: review-only`, so a 3-lane agy review of the diff precedes the PR)
| phase | content | A_i | result |
|---|---|---|---|
| P1 | `dag_status.py`: status derived from the receipt marker (same rule as `check_receipt_complete.sh`); `render_dag.py` prints it; `dag_invariants.py` D7 refuses a disagreeing typed status; past-expiry reads the derived status | `bash scripts/check_dag_invariants.sh --selftest` · `python3 scripts/render_dag.py --check` | 15/15 (D7 rows 12–15) · byte-identical, 91 rows |
| P2 | `check_row_pr_write_set.sh`: row PR = `agent/<id>` with `<id>` a DAG row; forbidden = the DAG, the roadmap, the spec, README count lines; non-row branches say so; merge_group/push REPORT; missing DAG = exit 2 | `bash scripts/check_row_pr_write_set.sh --self-test` | 11/11 |
| P3 | `check_readme_claims.sh`: `compare_count()` lag allowed, overstatement RED, `--exact` for the orchestrator, self-contradiction RED; `README_PATH` fixture; `--self-test` | `bash scripts/check_readme_claims.sh --self-test` · live | 7/7 · PASS ×5 |
| P4 | ci.yml: README case table before its live step; write-set case table + live after the DAG invariants | `bash scripts/check_guards_are_wired.sh` | PASS (ratcheted) |
| P5 | contract `apr-row-pr-write-set-v1.yaml` (kind: pattern; WS-OB-001..004 ↔ WS-F-001..004) | `pv validate` (via `scripts/pv_bin.sh`) | valid |

K̂ [U] (second receipt of the guard class; PMAT-1059 is the first; a basis needs three).

## Mutations observed RED → GREEN
| mutation | RED | GREEN |
|---|---|---|
| a row branch appends to `pp-066-dag.yaml` (write-set row 2, the registered mutation) | `FAIL … writes the shared file docs/specifications/pp-066-dag.yaml` rc=1 | crate-code-only diff (row 1) PASS |
| a row branch bumps `1812` → `1813` in README (row 5) | RED naming the line | README prose edit (row 6) PASS |
| typed `status: complete` with no receipt (D7 row 12) | rc=1 naming the row and the receipt path | typed complete over a complete receipt (row 13) PASS; no typed status + complete receipt + past expiry → not expired (row 15) |
| README overstates the crate count by one (README row 5) | `FAIL … the README may lag, never overstate` | lag by one (row 3) PASS with `lags at N` reported; `--exact` RED (row 4) |
| the merge_group / push shapes (rows 9–10) | — | REPORT line asserted present (a silent exit 0 is RED in the table) |

## Verification (claimed vs re-run by the orchestrator)
No worker or lane ran on this row before this commit; every number above is the orchestrator's own run (`.pr/G-11-verify.log`). Also PASS: `check_shell_lint_ratchet.sh`, `check_sourced_libs_option_neutral.sh`, `check_no_claim_literals.sh`, `check_receipt_complete.sh --dag`. The live write-set run on the local worktree refused its base by name (HEAD was the origin/main tip in a shallow clone: "never the tree against itself") — the PR-A rule working as designed; with `--base origin/main` it reports `names no DAG row` until the orchestrator commit adds row G-11.

## Dispatch ledger
none yet (direct). The pre-PR 3-lane review is appended below by the orchestrator after it runs.

## Gaps
- Orchestrator docs commit (agent/pp-066-spec): DAG row G-11 (+ G-10b, G-10c, U-1, S-0), the P-1.1/P-1.2 lane amendment (D-11), the C0-3 → U-1 edge, the roadmap mint of PMAT-1061..1064, the re-rendered §5.0 block, README counts regenerated and verified with `--exact`.
- Rebase onto main after #3011; the untracked `scripts/lib/resolve_base.sh` copy then becomes tracked.
- `check_receipt_complete.sh --dag` still reads typed `status: complete` rows; it stays (redundant with D7, both must hold) — the orchestrator may drop the typed keys once every consumer derives.

## Verdict
PARTIAL — awaiting the review lanes, the rebase onto main after #3011, `ci / gate` + `workspace-test`, and the merge.
