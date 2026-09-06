---
status: complete
ticket: PMAT-1063
row: G-10b
issue: 3013
epic: 2873
branch: agent/G-10b
pr: "#3021 — re-cut onto main b0a0a51b2 by rebase --onto; RED leg run 34038557827 (job 101501015300: 'Unpinned analyser references may only fall' FAILED on the mutant c4f6b618a, unpinned=244 > 243); GREEN leg = the run of this revert commit"
model: claude-fable-5-1 (orchestrator, direct)
tokens_used: orchestrator [U] (not exposed to the orchestrator)
wall_clock_s: 900 (basis=session clock, claim to the receipt commit; [U] precision)
turns: 3 (orchestrator turns on this ticket, counted from the transcript)
---
# impl receipt — PMAT-1063 (PP-066 row G-10b, #3013): the analyser pin guard, shrink-only

## Identity
- kind: code · branch `agent/G-10b` off `agent/G-10` (PR-A, 0c35f8cfb) · this receipt is the first commit's companion
- write set: `scripts/check_pmat_pinned.sh` (new; from `agent/G-10-full` 4592b0572 plus the shrink-only ratchet, `--update`, fixture rows 16–20), `scripts/pmat_unpinned_baseline.txt` (new, measured), `scripts/check_baseline_ratchets.sh` (kind-table entry), `.github/workflows/ci.yml` (two steps after the ratchet step), `contracts/apr-pinned-analyser-ratchet-v1.yaml` (1.0.0 → 1.1.0: PIN-OB-005 / PIN-F-005), this receipt. No DAG, roadmap, README or spec edit.

## The baseline is measured, not typed
The driver named "281"; that was the count on the pre-PR-A tree (2026-09-06, before the ratchet rewrite removed some references). On PR-A's tip this guard's own scan counts **243** — recorded by `bash scripts/check_pmat_pinned.sh --update`, whose written line carries the command, the commit and the kind. Derive: `grep -rEn '(^|[^_/])pmat ' scripts/ .github/workflows/ | grep -v pmat_bin | wc -l`.

## Plan and routing (direct; `quorum: review-only` — the pre-PR review lanes are recorded when run)
| phase | content | A_i | result |
|---|---|---|---|
| P1 | guard with the shrink-only compare, `--update`, `PIN_SCAN_ROOT`/`PIN_BASELINE` fixtures, rows 16–20 | `bash scripts/check_pmat_pinned.sh --self-test` | 20/20 |
| P2 | baseline kind `count`; CI steps case-table-then-live | `bash scripts/check_baseline_ratchets.sh` · `bash scripts/check_guards_are_wired.sh` | PASS · PASS |
| P3 | contract 1.1.0 | `pv validate` via `scripts/pv_bin.sh` | valid |

K̂ [U] (third receipt of the guard class after PMAT-1059 and PMAT-1062; the class basis can be computed once all three record turns).

## Mutations observed RED → GREEN
| mutation | RED | GREEN |
|---|---|---|
| live: append `# probe: run pmat analyze satd here` to `scripts/ci_target_watch.sh` | `FAIL check_pmat_pinned: unpinned=244 baseline=243 — 1 new line(s) …` naming the file:line | reverted → `PASS … unpinned=243 baseline=243` |
| fixture rows 16–20 | row 17 (baseline 1 under 2 lines) RED naming both lines; row 19 no baseline → exit 2; row 20 `INVALID` → exit 2 | row 16 (baseline 2) PASS; row 18 (baseline 3) PASS with `Improved: 3 -> 2` |
| case-table rows 1–15 (from PR-A's design) | rows 1–5 match, row 13 off-pin refused, row 14 absent refused | rows 6–11 clean, row 12 at-pin resolves, row 15 option-neutral |

## Verification (orchestrator's own runs, `.pr/G-10b-verify.log`)
self-test 20/20 · live PASS 243/243 · `check_baseline_ratchets.sh` PASS · `check_guards_are_wired.sh` PASS · `pv validate` valid · `check_shell_lint_ratchet.sh` PASS · `check_no_claim_literals.sh` rc 0.

## Mutation evidence at CI (I3)
| leg | commit | run | job | result |
|---|---|---|---|---|
| RED | c4f6b618a (one bare `pmat analyze satd` comment in `scripts/ci_target_watch.sh`) | 34038557827 | 101501015300 | guard-runner-labels FAILED at "Unpinned analyser references may only fall (shrink-only, G-10b PMAT-1063)" |
| GREEN | this commit (the revert) | the run of this commit (cited in the PR body) | — | expected PASS `unpinned=243 baseline=243` |

## Gaps
- G-10c (PMAT-1064, #3014): the sweep 243 → 1 is on `agent/G-10c` (the last reference is `render_dag.py`'s rendered header, the orchestrator's); baseline to 0 after the header rename.
- The pre-PR review lanes were not run on this diff (review-only row; the case table and the CI mutation stand as the evidence) — recorded as a gap, not a pass.
- Auto-merge is armed only after G-11 (#3020) merges: one orchestrator-armed code PR until the write-set guard is in `ci / gate` (driver v4 WIP rule).

## Verdict
DONE on the branch: every A_i re-run by the orchestrator, the mutation RED at CI and reverted; complete = this receipt ∧ the merge of #3021.
