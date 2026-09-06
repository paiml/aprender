---
status: partial
ticket: PMAT-1064
row: G-10c
issue: 3014
epic: 2873
branch: agent/G-10c
pr: opened after G-10b merges; re-cut onto main by diff-apply (the base is agent/G-10b)
model: claude-fable-5-1 (orchestrator, direct)
tokens_used: orchestrator [U] (not exposed to the orchestrator)
wall_clock_s: 600 (basis=session clock; [U] precision)
turns: 3 (orchestrator turns on this ticket, counted from the transcript)
---
# impl receipt — PMAT-1064 (PP-066 row G-10c, #3014): the analyser reference sweep

## Identity
- kind: code · branch `agent/G-10c` off `agent/G-10b` (21b11c7e2) · the sweep is `agent/G-10-full` 5f8f28f19 cherry-picked (35 files applied cleanly) minus `scripts/render_dag.py`.
- write set: 5 workflows (`binary-release`, `book`, `ci`, `coverage-nightly`, `qwen-story-daily`), 29 scripts under `scripts/`, `scripts/pmat_unpinned_baseline.txt` (243 → 1, recorded by `--update`), this receipt. No DAG, roadmap, README or spec edit.

## Why render_dag.py is NOT in this sweep
Its table header string `| … | issue | pmat | status |` is the one remaining match (prose, counted by design). Changing it changes the rendered §5.0 block, which only the orchestrator may write (G-11). The orchestrator docs commit renames the header cell to `pmat_id` and re-renders; a later `--update` records 0. The baseline is therefore **1, measured**, not 0 typed.

## Verification (orchestrator's own runs, `.pr/G-10c-verify.log`)
`check_pmat_pinned.sh --self-test` 20/20 · live `unpinned=1 baseline=1` · `check_verifier_pinning.sh` PASS · `check_complexity_ratchet.sh --selftest` 3/3 · `check_hardcoded_paths.sh --self-test` PASS · `check_roadmap_diff_additive.sh --self-test` 17/17 · `check_dag_invariants.sh --selftest` 11/11 · `check_publish_safety.sh --self-test` 9/9 · `check_cargo_install_private_root.sh --self-test` PASS · `check_pr_review_wiring.sh` PASS · `check_roadmap_completion_is_cited.sh` PASS · `check_no_tracked_ignored_files.sh` PASS · `check_sourced_libs_option_neutral.sh` 9 libs OK · `check_shell_lint_ratchet.sh` PASS · `check_guards_are_wired.sh` PASS · `check_baseline_ratchets.sh` PASS · the five touched workflows parse as YAML. `check_pr_review_receipt.sh` has no `--self-test` flag (its case table is `scripts/mutate-guard.sh`; not re-run here — a gap).

## Mutation
Restore one bare reference (e.g. `pmat comply check` in `scripts/ci.sh`) → `FAIL … unpinned=2 baseline=1` naming the line (the G-10b guard); revert → PASS. To be shown with both CI run ids in the PR body (I3).

## Gaps
- The orchestrator header rename (`pmat` → `pmat_id`) + re-render, then `--update` → 0.
- `scripts/mutate-guard.sh` run on the swept `check_pr_review_receipt.sh` (233 mutants) before the PR.
- The pre-PR review lanes; the re-cut onto main after G-10b.

## Verdict
PARTIAL.
