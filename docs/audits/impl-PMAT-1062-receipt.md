---
status: partial
ticket: PMAT-1062
row: G-11
epic: 2873
branch: agent/G-11
pr: opened after PR-A (#3011) merges — the only PR in the queue until then (driver PHASE 1)
model: claude-fable-5-1 (orchestrator, direct) · review quorum on agy lanes recorded below when run
tokens_used: 82736 (review delegate, measured by the harness) + orchestrator [U] (not exposed to the orchestrator)
wall_clock_s: 4500 (basis=session clock, G-11 claim to the fold commit; [U] precision)
turns: 16 (orchestrator turns on this ticket, counted from the transcript)
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
| P1 | `dag_status.py`: status derived from the receipt marker (same rule as `check_receipt_complete.sh`); `render_dag.py` prints it; `dag_invariants.py` D7 refuses a disagreeing typed status; past-expiry reads the derived status | `bash scripts/check_dag_invariants.sh --selftest` · `python3 scripts/render_dag.py --check` | 16/16 (D7 rows 12–16) · byte-identical, 91 rows |
| P2 | `check_row_pr_write_set.sh`: the DAG and the spec orchestrator-only on every branch; a row PR (`agent/<id>`, `<id>` a DAG row) also not the roadmap or a README count line; `--no-renames`, DAG read at the base; non-row branches say so; merge_group/push REPORT; DAG missing at the base = exit 2 | `bash scripts/check_row_pr_write_set.sh --self-test` | 14/14 (after the fold) |
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
| dispatch | agent | lane | width | agy conversations | child | turns | note |
|---|---|---|---|---|---|---|---|
| P6 review | paiml-agy-delegate `a6945921a70850177` (opus) | quorum, mode plan | 3 | 6b26a481-2126-442f-b5cf-91d8e1f7c5b2 · 0991a3bb-08da-491d-b410-c3113f4d02c4 · 89d65a03-6ba6-4b52-8cb9-44ded95a6d49 | 0 | 1 per lane | first launch killed at ~117 s by the harness timeout (3 conversations discarded); NO lane ran the four required commands — every `measured` tag in the lane files is a claim |

slots used 1/3 · denials 0 · I-3: attempted=1 denied=0 running_peak=1 slots=3. Lane files: `/run/user/1000/paiml-implement/agy/G-11-review/lane-{1,2,3}.json`.

## Review quorum: 3/3 implement-with-changes — every finding re-verified by the orchestrator, then folded
| Q | lane finding | my verification | disposition |
|---|---|---|---|
| Q1 | a PR from a branch not named `agent/<id>` walks through (3/3) | correct as written | **folded**: the DAG and the spec are orchestrator-only on EVERY branch (allow-list `agent/pp-066-*`, `agent/pr-triage*`); the roadmap/README-count rule stays row-only because other work mints tickets in the roadmap; rows 9–10 added (a `fix/` branch writing the DAG → RED; `agent/<not-a-row>` writing the spec → RED) |
| Q1 | `git diff --name-only` hides a rename's source path (2/3) | correct: rename detection collapses old→new | **folded**: `--no-renames`; the DAG is read at the BASE commit so a renamed/deleted DAG is a write, not ENV; row 11 (rename → RED naming the file) |
| Q1 | the README count regex is escaped by >2 words | true, and by design: the regex IS check_readme_claims.sh's extractor — a line it does not read is not a claim | comment added; no change |
| Q2 | the pull_request run is not required (lane 2) vs it is (lane 3) | **lane 3 is right**: ruleset "Green Main" (13878864) requires the context `gate`; the local `gate` job (ci.yml) `needs: [ci, workspace-test, mutants, guard-runner-labels]`; branch protection adds `ci / gate` + `workspace-test` | REPORT-and-exit-0 on merge_group/push stands, and the header now cites the two mechanisms |
| Q2 | a row PR could delete the step from ci.yml (lane 1) | true of every guard; `scripts/check_guards_are_wired.sh` names an unwired guard RED (class #2878) | noted in the header; no change |
| Q3 | CRLF: bash keeps `\r` (no front matter), python universal newlines drop it (3/3) | reproduced | **folded**: `open(..., newline="")`; D7 row 16 (a CRLF receipt, typed complete → RED); parity checked by hand: bash `none`, python `none` |
| Q3 | awk strips quotes anywhere, python only at the edges (lane 3) | reproduced (`co"mpl"ete`) | **folded**: `re.sub(r'[\s"\']', '', v)` |
| Q3 | render_dag drift on a receipt flip is a Catch-22 for a row PR | correct reading; the discipline was implicit | **folded**: the drift message states it — a row PR ships `status: partial` and never edits the spec; the orchestrator flips and re-renders in one commit |
| Q4 | `cli_command_count` bypassed `compare_count` (3/3) | correct | **folded**: routed through `compare_count` (lag reported) |
| Q4 | the exact behaviour survives in `crates/aprender-core/tests/readme_contract.rs` (lane 2) | **correct and load-bearing**: FALSIFY-README-005/007 asserted equality and ride `workspace-test`, so a lagging README would have failed every row PR | **folded**: both tests now assert claimed ≤ measured (`number_before()`), lag stated in the message; `cargo test -p aprender-core --test readme_contract` → 15 passed |
| Q4 | claimed==measured==0 passes vacuously (lane 1) vs cannot (lane 3) | lane 3 is right: the measurement functions exit 1 on an empty/zero count before any compare | no change |
| Q5 | `resolve_base.sh` untracked here → live step exit 2 in CI | correct until the rebase onto main after PR-A (#3011) | documented; the rebase is a gap below |
| Q5 | README case table AFTER its live step (lane 3) | correct: I appended the self-test after the live run line | **folded**: one `run:` block, case table first |
| Q6 | obligations 1:1, every `test:` real | holds | — |

## Verification after the fold (orchestrator's own runs, `.pr/G-11-verify2.log`)
`check_row_pr_write_set.sh --self-test` 14/14 · `check_dag_invariants.sh --selftest` 16/16 · `check_readme_claims.sh --self-test` 7/7 · live README PASS ×5 · `render_dag.py --check` byte-identical · `check_guards_are_wired.sh` PASS · `check_shell_lint_ratchet.sh` PASS · `check_sourced_libs_option_neutral.sh` OK · `cargo test -p aprender-core --test readme_contract` 15 passed (own target dir, pinned cargo path) · clippy on that target: see the follow-up commit if it changed anything.

## Gaps
- Orchestrator docs commit (agent/pp-066-spec): DAG row G-11 (+ G-10b, G-10c, U-1, S-0), the P-1.1/P-1.2 lane amendment (D-11), the C0-3 → U-1 edge, the roadmap mint of PMAT-1061..1064, the re-rendered §5.0 block, README counts regenerated and verified with `--exact`.
- Rebase onto main after #3011; the untracked `scripts/lib/resolve_base.sh` copy then becomes tracked.
- `check_receipt_complete.sh --dag` still reads typed `status: complete` rows; it stays (redundant with D7, both must hold) — the orchestrator may drop the typed keys once every consumer derives.

## Verdict
PARTIAL — awaiting the review lanes, the rebase onto main after #3011, `ci / gate` + `workspace-test`, and the merge.
