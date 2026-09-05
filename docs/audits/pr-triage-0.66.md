# PR triage into 0.66 — 29 dispositions (epic #2873, ticket #2986)

Generated 2026-09-05 against `origin/main` cdc0acb99. Source of truth: `docs/audits/pr-triage-0.66.yaml` (every number carries its command). Producer is never the gate: `ci / gate` on the merge queue decides; this file records.

## Counts

| disposition | n |
|---|---|
| MERGE-0.66 | 13 |
| MERGE-NOW | 7 |
| CARRY-0.67 | 3 |
| HOLD | 2 |
| DRIVER | 2 |
| SPLIT | 1 |
| STOP | 1 |

## Table

| PR | age d | behind | mergeable@0 | gate@0 red | rows (issue→row / file∩) | rebase | disposition | q | action |
|---|---|---|---|---|---|---|---|---|---|
| #2575 | 14 | 63 | MERGEABLE/BEHIND | gate, workspace-test | R-2 | CLEAN | **MERGE-0.66:R-2** | 20 | rebased clean -> push -> arm |
| #2635 | 13 | 58 | CONFLICTING/DIRTY | green | C0-1, C0-5, G-1, G-4, G-6, P-0.6 | CONFLICT | **MERGE-0.66:C0-1** | 14 | rebase attempt -> CONFLICT -> STOP |
| #2638 | 13 | 57 | CONFLICTING/DIRTY | gate, workspace-test | R-0, T-2 | CONFLICT | **MERGE-0.66:R-0** | 12 | rebase attempt -> CONFLICT -> STOP |
| #2659 | 12 | 14 | UNKNOWN/UNKNOWN | gate, workspace-test, pr-review-receipt | B-M2, G-5 | not | **HOLD:#2985** | — | labelled pp-066/hold, commented; no branch edit |
| #2666 | 12 | 101 | CONFLICTING/DIRTY | green | — | CONFLICT | **MERGE-NOW** | 7 | rebase attempt -> CONFLICT -> STOP |
| #2711 | 8 | 37 | CONFLICTING/DIRTY | green | G-8, P-0.6, R-0, SPEC-1.6 | CONFLICT | **MERGE-0.66:SPEC-1.6** | 5 | rebase attempt -> CONFLICT -> STOP |
| #2720 | 8 | 44 | CONFLICTING/DIRTY | gate, workspace-test | — | CONFLICT | **MERGE-NOW** | 8 | rebase attempt -> CONFLICT -> STOP |
| #2738 | 8 | 42 | CONFLICTING/DIRTY | gate, guard-runner-labels | C0-1, C0-5, G-1, G-4, G-6, P-0.6 | CONFLICT | **SPLIT** | 21 | rebase attempt (--rebase-merges) -> CONFLICT; sub-branch scan: feat/m2-alloc CLEAN, fix/make-targets-always-green CLEAN, feat/m3-cuda-ci CONFLICT (zram-core gpu/mod.rs modify/delete), feat/m5-serve-paths CONFLICT (hardcoded_path_shipped_baseline.txt), PERF-033 cherry-pick 4a356c3b5 CONFLICT (llama_bin.sh llama_pin.toml) |
| #2741 | 8 | 42 | CONFLICTING/DIRTY | green | G-9, I-24, P-0.3 | CONFLICT | **MERGE-0.66:I-24** | 18 | rebase attempt (--rebase-merges) -> CONFLICT -> STOP |
| #2773 | 7 | 38 | CONFLICTING/DIRTY | green | C0-1, C0-5, G-1, G-4, G-6, G-8, I-24, P-0.6, S-2 | CONFLICT | **MERGE-0.66:I-24** | 19 | rebase attempt (--rebase-merges): .gitignore both-appended -> union; second stop CONFLICT -> STOP |
| #2793 | 6 | 15 | MERGEABLE/UNSTABLE | pr-review-receipt | — | CLEAN | **MERGE-NOW** | 6 | rebased clean -> push; NOT armed |
| #2794 | 6 | 30 | UNKNOWN/UNKNOWN | green | C0-1 | not | **STOP:author** | — | review comment posted (7 findings; 2 blocking once rebased: aprender-core version pin 0.64.0 vs workspace 0.65.2; no CI has ever run on the branch) |
| #2800 | 5 | 28 | MERGEABLE/BEHIND | gate, workspace-test | W-G | not | **CARRY-0.67:W-G** | — | converted to draft, labelled pp-066/carry-0.67, commented with the row id |
| #2801 | 5 | 31 | MERGEABLE/BEHIND | ci / gate, ci / security, gate, guard-runner-labels, mutants, workspace-test | W-G | not | **CARRY-0.67:W-G** | — | converted to draft, labelled pp-066/carry-0.67, commented with the row id |
| #2802 | 5 | 28 | MERGEABLE/BEHIND | gate, workspace-test | — | CLEAN | **MERGE-NOW** | 3 | rebased clean -> push -> arm (docs) |
| #2803 | 5 | 28 | CONFLICTING/DIRTY | gate, workspace-test | C0-1, C0-5, G-1, G-4, G-6, P-0.6, R-0 | CONFLICT | **MERGE-0.66:R-0** | 10 | rebase attempt -> CONFLICT -> STOP |
| #2807 | 5 | 28 | CONFLICTING/DIRTY | gate, guard-runner-labels, workspace-test | B-S1, C0-1 | CLEAN(lockfile-regen) | **MERGE-NOW** | 1 | rebase: Cargo.lock conflict resolved mechanically (main's lock + `cargo update -p wasmtime --precise 47.0.4`, 43 lock lines; `cargo metadata --locked` rc=0; `cargo deny check advisories` rc=0) -> pushed 7104c4e45 → pushed 7104c4e45 gate=pending |
| #2811 | 5 | 28 | CONFLICTING/DIRTY | gate, guard-runner-labels, workspace-test | C0-1, C0-5, G-1, G-4, G-6, P-0.6 | CONFLICT | **MERGE-0.66:C0-1** | 15 | rebase attempt -> CONFLICT -> STOP |
| #2820 | 5 | 21 | CONFLICTING/DIRTY | gate, workspace-test, pr-review-receipt | C0-4 | CONFLICT | **MERGE-0.66:C0-4** | 13 | rebase attempt -> CONFLICT -> STOP |
| #2821 | 5 | 28 | CONFLICTING/DIRTY | gate, guard-runner-labels | C0-1, C0-5, G-1, G-4, G-6, P-0.6 | CONFLICT | **MERGE-0.66:C0-1** | 16 | rebase attempt -> CONFLICT -> STOP |
| #2825 | 5 | 21 | CONFLICTING/DIRTY | gate, guard-runner-labels, workspace-test, pr-review-receipt | G-5, R-0, R-2, T-2 | CONFLICT | **MERGE-0.66:R-0** | 11 | rebase attempt -> CONFLICT -> STOP |
| #2836 | 4 | 18 | CONFLICTING/DIRTY | pr-review-receipt | C0-1, C0-5, G-1, G-4, G-6, P-0.6 | CONFLICT | **HOLD:#2985** | 22 | rebase attempt -> CONFLICT -> STOP |
| #2838 | 4 | 15 | MERGEABLE/BEHIND | ci / gate, ci / lint, gate, guard-runner-labels, workspace-test, pr-review-receipt | C0-1, C0-5, G-1, G-4, G-6, P-0.6, W-G | not | **CARRY-0.67:W-G** | — | converted to draft, labelled pp-066/carry-0.67, commented with the row id |
| #2847 | 3 | 15 | MERGEABLE/BEHIND | ci / gate, ci / lint, gate, guard-runner-labels, pr-review-present, pr-review-sign, pr-review-receipt, workspace-test | C0-1, C0-5, G-1, G-4, G-6, P-0.6 | CONFLICT | **MERGE-0.66:C0-1** | 17 | rebase attempt -> CONFLICT -> STOP |
| #2848 | 3 | 15 | MERGEABLE/BEHIND | gate, guard-runner-labels, pr-review-present, pr-review-receipt, workspace-test | — | CLEAN | **MERGE-NOW** | 2 | rebased clean -> push -> arm |
| #2858 | 2 | 12 | CONFLICTING/DIRTY | ci / gate, ci / security, gate, pr-review-present, workspace-test, pr-review-receipt | I-1 | CONFLICT | **MERGE-NOW** | 9 | rebase attempt -> CONFLICT -> STOP |
| #2871 | 0 | 3 | MERGEABLE/UNSTABLE | pr-review-receipt | — | CLEAN | **MERGE-0.66:SPEC-1.6** | 4 | rebased clean -> push -> arm (docs) |
| #2981 | 0 | 1 | MERGEABLE/BEHIND | gate, pr-review-present, pr-review-receipt | C0-4, G-4, G-6, G-8, G-9, I-1, I-24, I-25, bar-gating | not | **DRIVER** | — | none |
| #2985 | 0 | 1 | MERGEABLE/UNSTABLE | green | C0-1, C0-5, G-1, G-4, G-6, P-0.6 | not | **DRIVER** | — | none |

## STOP list (operator decisions)

- **#2635** — conflict: scripts/check_cascade_covers_all_crates.sh; workflow-touching
- **#2638** — conflict: serve/handlers.rs device/mod.rs README.md; duplicates #2825 — operator picks one; adds workflow gpu-vulkan.yml
- **#2659** — author app/dependabot (not noahgift); workflow-touching; needs @dependabot rebase + operator decision
- **#2666** — conflict: scripts/benchmark-matrix.sh (main rewrote the script: 363 changed lines since merge-base)
- **#2711** — conflict: serve/handler_gpu_completion.rs api/router.rs
- **#2720** — conflict: beat-speed-nightly.yml; main runs the lane on clean-room, PR moves it to perf-solo — runner switch is escalate-class
- **#2738** — conflict: zram-core gpu/mod.rs (modify/delete) hardcoded_path_shipped_baseline.txt llama_bin.sh llama_pin.toml; 3 of 5 sub-tickets also conflict
- **#2741** — conflict: cbtop_get_cpu_memory.rs (inside the batch's own p3-warmup merge replay, commit 5a29ac364)
- **#2773** — conflict: check_bench_protocol.sh check_comparator_flags.sh lib/parity_block.py llama_bin.sh llama_pin.toml parity_host_receipt.sh
- **#2793** — workflow-touching (silicon-nightly.yml): arming needs the operator
- **#2794** — author guyernest
- **#2803** — conflict: test_llm_band.rs parity_host_receipt.sh; workflow-touching
- **#2811** — conflict: ci.yml; superseded in INTENT by main's PMAT-742 (workspace-test timeout 150 job / 110 step, comment cites #2811) but not in diff (PR proposes 110 / 55) — operator: close or renumber
- **#2820** — conflict: scripts/perf_gate.sh
- **#2821** — conflict: test_llm_band.rs prompts-w1.jsonl llm/band.rs perf_gate/receipt.rs llama_pin.toml; depends on #2820
- **#2825** — conflict: handler_gpu_completion.rs serve/mod.rs device/mod.rs parity_host_receipt.sh; duplicates #2638's PMAT-778 fix
- **#2836** — conflict: ci.yml scripts/check_pr_review_counts.sh; the receipt gate is now base-owned, rung 1 needs re-siting (design decision)
- **#2847** — conflict: ci.yml tests/pr-review.bats; workflow-touching
- **#2858** — conflict: docs/audits/impl-estimates.jsonl (mechanical) BUT the PR's roadmap.yaml hunk re-mints PMAT-746..768 and 17 ids (750,751,753-767) already exist on main as other tickets; roadmap.yaml is outside the triage write surface

## Method

- STEP 0 evidence per PR: `gh pr view … --json`, `gh pr checks`, `git rev-list --count`, a three-way `git merge-tree` scan (git 2.34 has no `--write-tree`), linked issues from title/body (`#nnnn`), issue→row through `docs/specifications/pp-066-dag.yaml` (`issues` lists on rows), file∩ against the first-wave P1 file lists in `docs/audits/pp-066-plan.md` plus the path tokens in each DAG row.
- STEP 1 one disposition per PR; MERGE-0.66 queue position = earliest expiry among intersecting rows.
- STEP 2 one rebase attempt per PR in a detached worktree; batches with `--rebase-merges`. Only two mechanical resolutions were applied and both are named in the entry: #2807 (lockfile regenerated from main's Cargo.lock with `cargo update -p wasmtime --precise 47.0.4`, verified by `cargo metadata --locked` and `cargo deny check advisories`) and #2773 (`.gitignore` both-appended → union; the replay then stopped on six other files and was aborted).
- STEP 3 arming only for MERGE-* with gate green and mergeable == MERGEABLE, one code PR at a time; workflow-touching PRs are pushed but not armed (STOP list).
