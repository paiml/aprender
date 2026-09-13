---
status: complete
ticket: PMAT-1059
row: G-10
issue: 2999
epic: 2873
branch: agent/G-10
pr: "#3011 — merged 2026-09-06T13:51:59Z as b0a0a51b2 (squash); guard-runner-labels and workspace-test green on the PR run 34029601098 and the merge-queue run 34034085658"
model: claude-fable-5-1 (orchestrator) · paiml-agy-delegate on opus · agy 1.1.27 quorum lanes ×3 (mode plan)
tokens_used: 64665 (delegate, measured by the harness) + orchestrator [U] (not exposed to the orchestrator)
wall_clock_s: 2700 (basis=session clock, PHASE 1 start to the receipt commit; [U] precision)
turns: 15 (orchestrator turns on this ticket in the session of 2026-09-06, counted from the transcript)
---
# impl receipt — PMAT-1059 (PP-066 row G-10, #2999): the shipped-path ratchet runs under ONE pinned analyser

## Identity
- kind: code · branch `agent/G-10` · code HEAD `b504c3a30` (this receipt is the next commit) · base `origin/main` 027ed889d
- discover.json sha256 (first 16): `9cfe24381fc8d4cc` · gate_cmd / required_check: `cargo test --workspace ci / gate,workspace-test`
- write set: `scripts/pmat_bin.sh`, `scripts/lib/resolve_base.sh`, `scripts/check_hardcoded_paths.sh`, `scripts/check_roadmap_diff_additive.sh`, `scripts/lib_baseline_ratchet.sh`, `scripts/hardcoded_path_shipped_baseline.txt`, `scripts/shell_lint_baseline.txt`, `contracts/apr-pinned-analyser-ratchet-v1.yaml`, `.github/workflows/ci.yml` (one job's steps), `README.md` (contract count, until G-11 makes it ≥), this receipt. No DAG, roadmap, or book edits.
- split per the driver of 2026-09-06: this is PR-A. The pin guard (`check_pmat_pinned.sh`, 281 shrink-only) is G-10b; the sweep of the 281 references is G-10c. Both are preserved on `agent/G-10-full` (5f8f28f19, pushed) and are re-cut from there onto main after PR-A merges.

## Root cause (cited, not inferred)
| run | shape | analyser line | ratchet |
|---|---|---|---|
| 34011762858 (main @ 027ed889d) | push | `pmat: /home/noah/.cargo/bin/pmat (pmat 3.31.0)` | `SKIPPED (proven): pmat 3.31.0 cannot run this analysis.` → success |
| 34018449576 (#3008) | pull_request | `pmat: /home/noah/.cargo/bin/pmat (pmat 3.37.0)` | `ARMED and blocking` → `FAIL: shipped machine-specific paths grew 277 -> 317.` |

Derive: `gh api repos/paiml/aprender/actions/jobs/<101428716226|101446481475>/logs | grep -E 'pmat: |SKIPPED|ARMED|FAIL: shipped'`. The baseline (277) named no instrument; 3.37.0 and 3.38.0 both count 317 on the unchanged tree. Main was vacuously green; every PR after the runner's flip is red for a defect none introduced.

## Plan and routing
| phase | content | A_i | route |
|---|---|---|---|
| P1 | resolver (`pmat_bin.sh`), stamped baseline, differential ratchet, fixtures R1–R8, contract | `bash scripts/check_hardcoded_paths.sh --self-test` | direct (Fable) |
| P2 | CI wiring (step reads the pin; push shape deepened by one) | `bash scripts/check_guards_are_wired.sh` | direct |
| P3 | review-only quorum on the diff | delegate receipt | quorum:agy ×3 |
| P4 | fold of the quorum + live differential and live mutation | `bash scripts/check_hardcoded_paths.sh --full-if-capable` (delta 0), +1 literal (delta +1 RED) | direct |

K̂ [U] (first receipt of the ratchet-guard class; three are needed before a basis exists).

## Dispatch ledger
| dispatch | agent | lane | width | agy conversations | child | turns | note |
|---|---|---|---|---|---|---|---|
| P3 | paiml-agy-delegate `abea1754afcc039b4` (opus) | quorum, mode plan | 3 | a65b3d4e-9e2c-4ef3-9390-338fe57cf58f · 44040040-9f0b-4c49-9f1f-79fafc820fb0 · 05a77382-8e34-4887-a5e2-67ab6ce7b188 | 0 | 1 per lane | every lane returned in one turn: NO lane ran the three required commands; every `measured` tag in the lane files is a claim |

slots used 1/3 · denials 0 (this session's hook log) · I-3: attempted=1 denied=0 running_peak=1 slots=3. Lane files: `/run/user/1000/paiml-implement/agy/G-10-review/lane-{1,2,3}.json`.

## Quorum verdict and disposition (3/3 do-not-implement-as-written; every finding re-verified by the orchestrator)
| # | lane finding | my verification | disposition |
|---|---|---|---|
| Q4 | the differential passes vacuously when merge-base(origin/main, HEAD) is HEAD (push shape) | **CONFIRMED** from the main push run's own log: G-6's guard printed `base=027ed889d (merge-base(origin/main, HEAD)) head=HEAD` — the base IS the pushed commit | fixed: `resolve_base` names the tip's first parent on the push shape and refuses when it is not fetched; `check_hardcoded_paths.sh` also refuses a base equal to HEAD; ci.yml deepens the depth-1 checkout by one on push; rows R9–R11 and G-6 rows 16–17 |
| Q4 (lane 1) | GIT_DIR/GIT_WORK_TREE leak makes the base scan enumerate HEAD | not reproduced (actions/checkout sets neither; the scan `cd`s into the base worktree) | hygiene added anyway: `env -u GIT_DIR -u GIT_WORK_TREE -u GIT_INDEX_FILE` around the analyser |
| Q5 | `resolve_base` refuses a single-parent head that is not the origin/main tip, so a shallow push is refused | correct reading of the code; the push shape is now the first-parent rule above; a single-parent non-tip head is not one of ci.yml's three shapes (pull_request merge ref, merge_group squash head, push to main) | documented in the resolver header; no change |
| Q6 | four obligations collapsed into two tests | correct | PIN-OB-001..004 now map 1:1 onto PIN-F-001..004 (`pv validate`: valid) |
| Q1 (lane 1 dissent) | `lib_baseline_ratchet.sh` compares the baseline file's stored `count:` across commits regardless of instrument, so a re-baseline that rises is impossible | correct: that is the shrink-only "no raise" ratchet on the stored number, not a measurement compare; it does block PMAT-1061 (277 → 317 under the pin) as written | out of PR-A's scope and recorded as PMAT-1061's precondition: the kind table needs a "stamped series" rule with its own case table before the re-baseline can land |
| Q2, Q3 | INVALID is never coerced; the resolver refuses off-pin and leaks no option | re-run: rows R5/R8 PASS; `bash -c` / `zsh -c` sourcing resolve the pin; an off-pin override is refused naming 3.38.0 | no change |
| mine | R7 "no base" had passed by an unbound-variable death: `PROG=x . file` does not outlive the `.` builtin, so `$PROG` was unset inside `resolve_base` under `set -u` and bash exited 1 before the refusal | reproduced with a two-line fixture (`in f: UNSET`) | plain assignment before sourcing; `resolve_base` defaults PROG; R7 now asserts the refusal text |
| mine | `comm: input is not in sorted order` on the live mutation (sort under LC_ALL=C, comm under the user locale) | reproduced in the first live run | `LC_ALL=C comm` |

## Verification (claimed vs re-run by the orchestrator)
| check | delegate/worker claim | my run | rc |
|---|---|---|---|
| `bash scripts/check_hardcoded_paths.sh --self-test` | not run by any lane | SELF-TEST PASSED (16 rows: 5 contract rows + R1–R11) | 0 |
| `bash scripts/check_roadmap_diff_additive.sh --self-test` | — | 17/17 rows | 0 |
| `bash scripts/check_guards_are_wired.sh` | not run | PASS (ratcheted) | 0 |
| `bash -c '. scripts/pmat_bin.sh && echo $PMAT $PMAT_VERSION'` | not run | `/home/noah/.local/pmat/3.37.0/bin/pmat 3.37.0` (also under zsh) | 0 |
| `bash scripts/check_hardcoded_paths.sh --full-if-capable` (HEAD vs merge-base 027ed889d, pin 3.37.0) | — | `REPORT BASELINE-INVALID` → `differential: base 027ed889d = 317 shipped; HEAD = 317 shipped; delta +0` → PASS | 0 |
| same, with `pub const MUTATION_PROBE: &str = "/home/probe/models/x.gguf";` appended to `crates/apr-cli/src/main.rs` | — | `delta +1` → `FAIL ... crates/apr-cli/src/main.rs\|/home/probe/models/x.gguf` (reverted) | 1 |
| `pv validate contracts/apr-pinned-analyser-ratchet-v1.yaml` (via `scripts/pv_bin.sh`) | — | Contract is valid | 0 |
| `bash scripts/check_baseline_ratchets.sh` · `check_sourced_libs_option_neutral.sh` · `check_shell_lint_ratchet.sh` (9 → 8 recorded) · `check_readme_claims.sh` · `check_no_claim_literals.sh` | — | PASS each | 0 |
| `bashrs lint scripts/pmat_bin.sh` | — | 9 SC diagnostics remain (SC1012 on printf formats, info class); the shell-lint ratchet is the gate and passes | — |

## Jidoka
- PROG-sourcing death (same repo, blocking): fixed in this PR; five whys end at "a sourced library read a variable its caller set only for the duration of the `.` builtin, and the fixture row had no wanted text so a death read as a refusal". Both closed: plain assignment, defaulted PROG, R7 asserts the text.

## Estimates
K̂ [U] · actual turns 15 · basis=first-run[U] (class: ratchet-guard). Appended to `docs/audits/impl-estimates.jsonl` by the orchestrator docs commit after merge, not here (write set).

## Gaps (each with the artifact that closes it)
- G-10b — `scripts/check_pmat_pinned.sh` + `pmat_unpinned_baseline.txt` (281, shrink-only) + the two CI steps; extends this contract (PIN-OB-005). From `agent/G-10-full` commit 4592b0572.
- G-10c — the sweep 281 → 0 (45 files, `agent/G-10-full` 5f8f28f19).
- PMAT-1061 — the stamped re-baseline under the pin, with the "stamped series" rule in the baseline kind table (the Q1 dissent).
- fleet pin: `paiml/infra machines/intel/forjar.yaml` `stack-tool-pmat` pins 3.37.0 (lines 765–785, PMAT-231) — verified by reading the file; no infra issue needed.
- `present` / `pr-review-receipt` are judged from the base after merge (C0-6 class), non-required.

## Verdict
DONE — #3011 merged (b0a0a51b2). The marker was flipped by the orchestrator docs commit (driver v3 rule); from G-11 on, the receipt says complete inside the PR before auto-merge is armed (driver v4: nothing is written after a merge) — recorded as a kaizen line.
