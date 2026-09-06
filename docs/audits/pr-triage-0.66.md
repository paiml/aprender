# PR triage into 0.66 — 29 dispositions (epic #2873, ticket #2986)

Generated 2026-09-05 against `origin/main` cdc0acb99. Source of truth: `docs/audits/pr-triage-0.66.yaml` (every number carries its command). Producer is never the gate: `ci / gate` on the merge queue decides; this file records.

## Counts

| disposition | n |
|---|---|
| PRIOR-ART-CLOSE | 12 |
| MERGE-NOW | 5 |
| CARRY-0.67 | 3 |
| MERGE-0.66 | 2 |
| HOLD | 2 |
| DRIVER | 2 |
| SPLIT | 1 |
| STOP | 1 |
| CLOSE | 1 |

## Table

| PR | age d | behind | mergeable@0 | gate@0 red | rows (issue→row / file∩) | rebase | disposition | q | action |
|---|---|---|---|---|---|---|---|---|---|
| #2575 | 14 | 63 | MERGEABLE/BEHIND | gate, workspace-test | R-2 | CLEAN | **MERGE-0.66:R-2** | 20 | rebased clean -> armed on the ORIGINAL head per operator (no push tonight; refs/triage/2575 holds the clean rebase locally); the merge queue builds the merge group → pushed None gate=queue decides |
| #2635 | 13 | 58 | CONFLICTING/DIRTY | green | C0-1, C0-5, G-1, G-4, G-6, P-0.6 | CONFLICT | **PRIOR-ART-CLOSE:#2992** | 14 | rebase attempt -> CONFLICT -> STOP |
| #2638 | 13 | 57 | CONFLICTING/DIRTY | gate, workspace-test | R-0, T-2 | CONFLICT | **PRIOR-ART-CLOSE:R-0** | 12 | rebase attempt -> CONFLICT -> STOP |
| #2659 | 12 | 14 | UNKNOWN/UNKNOWN | gate, workspace-test, pr-review-receipt | B-M2, G-5 | not | **HOLD:#2985** | — | labelled pp-066/hold, commented; no branch edit |
| #2666 | 12 | 101 | CONFLICTING/DIRTY | green | — | CONFLICT | **PRIOR-ART-CLOSE:#2988** | 7 | rebase attempt -> CONFLICT -> STOP |
| #2711 | 8 | 37 | CONFLICTING/DIRTY | green | G-8, P-0.6, R-0, SPEC-1.6 | CONFLICT | **PRIOR-ART-CLOSE:#2989** | 5 | rebase attempt -> CONFLICT -> STOP |
| #2720 | 8 | 44 | CONFLICTING/DIRTY | gate, workspace-test | — | CONFLICT | **HOLD:DEC-D-3** | 8 | labelled pp-066/hold, commented "held on D-3 decision"; operator: leave untouched. Triage recommendation (not applied): carry to 0.67 under #2873 dated to DEC-D-3 expiry 2026-09-12 — the perf-solo move is downstream of D-3 (CI home for non-CUDA lanes on intel), conflicts with main's clean-room runner, and split-per-repo is an infra#338 fleet change |
| #2738 | 8 | 42 | CONFLICTING/DIRTY | gate, guard-runner-labels | C0-1, C0-5, G-1, G-4, G-6, P-0.6 | CONFLICT | **SPLIT:closed** | 21 | rebase attempt (--rebase-merges) -> CONFLICT; sub-branch scan: feat/m2-alloc CLEAN, fix/make-targets-always-green CLEAN, feat/m3-cuda-ci CONFLICT (zram-core gpu/mod.rs modify/delete), feat/m5-serve-paths CONFLICT (hardcoded_path_shipped_baseline.txt), PERF-033 cherry-pick 4a356c3b5 CONFLICT (llama_bin.sh llama_pin.toml) |
| #2741 | 8 | 42 | CONFLICTING/DIRTY | green | G-9, I-24, P-0.3 | CONFLICT | **PRIOR-ART-CLOSE:I-24** | 18 | rebase attempt (--rebase-merges) -> CONFLICT -> STOP |
| #2773 | 7 | 38 | CONFLICTING/DIRTY | green | C0-1, C0-5, G-1, G-4, G-6, G-8, I-24, P-0.6, S-2 | CONFLICT | **PRIOR-ART-CLOSE:G-8** | 19 | rebase attempt (--rebase-merges): .gitignore both-appended -> union; second stop CONFLICT -> STOP |
| #2793 | 6 | 15 | MERGEABLE/UNSTABLE | pr-review-receipt | — | CLEAN | **MERGE-NOW** | 6 | rebased clean (refs/triage/2793 local); NOT pushed, NOT armed tonight (workflow file silicon-nightly.yml; operator arms) |
| #2794 | 6 | 30 | UNKNOWN/UNKNOWN | green | C0-1 | not | **STOP:author** | — | review comment posted (7 findings; 2 blocking once rebased: aprender-core version pin 0.64.0 vs workspace 0.65.2; no CI has ever run on the branch) |
| #2800 | 5 | 28 | MERGEABLE/BEHIND | gate, workspace-test | W-G | not | **CARRY-0.67:W-G** | — | converted to draft, labelled pp-066/carry-0.67, commented with the row id |
| #2801 | 5 | 31 | MERGEABLE/BEHIND | ci / gate, ci / security, gate, guard-runner-labels, mutants, workspace-test | W-G | not | **CARRY-0.67:W-G** | — | converted to draft, labelled pp-066/carry-0.67, commented with the row id |
| #2802 | 5 | 28 | MERGEABLE/BEHIND | gate, workspace-test | — | CLEAN | **MERGE-NOW** | 3 | rebased clean -> armed on the ORIGINAL head per operator (no push tonight; refs/triage/2802 holds the clean rebase locally); the merge queue builds the merge group (docs) → pushed None gate=queue decides |
| #2803 | 5 | 28 | CONFLICTING/DIRTY | gate, workspace-test | C0-1, C0-5, G-1, G-4, G-6, P-0.6, R-0 | CONFLICT | **PRIOR-ART-CLOSE:R-0** | 10 | rebase attempt -> CONFLICT -> STOP |
| #2807 | 5 | 28 | CONFLICTING/DIRTY | gate, guard-runner-labels, workspace-test | B-S1, C0-1 | CLEAN(lockfile-regen) | **MERGE-NOW** | 1 | rebase: Cargo.lock conflict resolved mechanically (main's lock + `cargo update -p wasmtime --precise 47.0.4`, 43 lock lines; `cargo metadata --locked` rc=0; `cargo deny check advisories` rc=0) -> pushed 7104c4e45 → pushed 7104c4e45 gate=pending |
| #2811 | 5 | 28 | CONFLICTING/DIRTY | gate, guard-runner-labels, workspace-test | C0-1, C0-5, G-1, G-4, G-6, P-0.6 | CONFLICT | **CLOSE:464c168e6** | 15 | rebase attempt -> CONFLICT -> STOP |
| #2820 | 5 | 21 | CONFLICTING/DIRTY | gate, workspace-test, pr-review-receipt | C0-4 | CONFLICT | **PRIOR-ART-CLOSE:#2990** | 13 | rebase attempt -> CONFLICT -> STOP |
| #2821 | 5 | 28 | CONFLICTING/DIRTY | gate, guard-runner-labels | C0-1, C0-5, G-1, G-4, G-6, P-0.6 | CONFLICT | **PRIOR-ART-CLOSE:#2991** | 16 | rebase attempt -> CONFLICT -> STOP |
| #2825 | 5 | 21 | CONFLICTING/DIRTY | gate, guard-runner-labels, workspace-test, pr-review-receipt | G-5, R-0, R-2, T-2 | CONFLICT | **PRIOR-ART-CLOSE:R-0** | 11 | rebase attempt -> CONFLICT -> STOP |
| #2836 | 4 | 18 | CONFLICTING/DIRTY | pr-review-receipt | C0-1, C0-5, G-1, G-4, G-6, P-0.6 | CONFLICT | **PRIOR-ART-CLOSE:#2997** | 22 | rebase attempt -> CONFLICT -> STOP |
| #2838 | 4 | 15 | MERGEABLE/BEHIND | ci / gate, ci / lint, gate, guard-runner-labels, workspace-test, pr-review-receipt | C0-1, C0-5, G-1, G-4, G-6, P-0.6, W-G | not | **CARRY-0.67:W-G** | — | converted to draft, labelled pp-066/carry-0.67, commented with the row id |
| #2847 | 3 | 15 | MERGEABLE/BEHIND | ci / gate, ci / lint, gate, guard-runner-labels, pr-review-present, pr-review-sign, pr-review-receipt, workspace-test | C0-1, C0-5, G-1, G-4, G-6, P-0.6 | CONFLICT | **PRIOR-ART-CLOSE:#2993** | 17 | rebase attempt -> CONFLICT -> STOP |
| #2848 | 3 | 15 | MERGEABLE/BEHIND | gate, guard-runner-labels, pr-review-present, pr-review-receipt, workspace-test | — | CLEAN | **MERGE-NOW** | 2 | rebased clean -> armed on the ORIGINAL head per operator (no push tonight; refs/triage/2848 holds the clean rebase locally); the merge queue builds the merge group → pushed None gate=queue decides |
| #2858 | 2 | 12 | CONFLICTING/DIRTY | ci / gate, ci / security, gate, pr-review-present, workspace-test, pr-review-receipt | I-1 | REBUILT(docs-only) | **MERGE-NOW** | 7 | STEP 2b: roadmap.yaml restored from origin/main on the branch (b6f1dcbf7); the rebase then stopped on the original commits' roadmap hunk, so the branch was REBUILT as one docs-only commit on main (receipt.md verbatim + the PR's two impl-estimates lines appended after main's; no merge) -> undraft -> arm with the docs batch → pushed 91f58b0d9 gate=pending |
| #2871 | 0 | 3 | MERGEABLE/UNSTABLE | pr-review-receipt | — | CLEAN | **MERGE-0.66:SPEC-1.6** | 4 | rebased clean -> armed on the ORIGINAL head per operator (no push tonight; refs/triage/2871 holds the clean rebase locally); the merge queue builds the merge group (docs) → pushed None gate=queue decides |
| #2981 | 0 | 1 | MERGEABLE/BEHIND | gate, pr-review-present, pr-review-receipt | C0-4, G-4, G-6, G-8, G-9, I-1, I-24, I-25, bar-gating | not | **DRIVER** | — | none |
| #2985 | 0 | 1 | MERGEABLE/UNSTABLE | green | C0-1, C0-5, G-1, G-4, G-6, P-0.6 | not | **DRIVER** | — | none |

## End-state assertions

- `open_prs_below_2981_by_noahgift`: **11 (expected 7 after drain)** — open now: 2575,2720,2793,2800,2801,2802,2807,2838,2848,2858,2871; pending merges [2807, 2848, 2802, 2871, 2575, 2858]; terminal marker `state: awaiting-drain`
- `open_prs_below_2981_all_authors`: 2871(noahgift),2858(noahgift),2848(noahgift),2838(noahgift),2807(noahgift),2802(noahgift),2801(noahgift),2800(noahgift),2794(guyernest),2793(noahgift),2720(noahgift),2659(app/dependabot),2575(noahgift) — #2659 (dependabot) and #2794 (guyernest) are outside the write surface; Noah replies
- `prior_art_branches`: expected 17, actual 17 → **PASS** (`git ls-remote --heads origin 'prior-art/*' | wc -l   # == git branch -r | grep -c prior-art/ after fetch`)
- `semantic_merges`: expected 0, actual 0 → **PASS** (``)

## STOP list (operator decisions)

- **#2659** — author app/dependabot (not noahgift); workflow-touching; needs @dependabot rebase + operator decision
- **#2720** — held on D-3 decision (DEC-D-3 #2934); no branch edits
- **#2793** — workflow-touching (silicon-nightly.yml): arming needs the operator
- **#2794** — author guyernest

## Method

- STEP 0 evidence per PR: `gh pr view … --json`, `gh pr checks`, `git rev-list --count`, a three-way `git merge-tree` scan (git 2.34 has no `--write-tree`), linked issues from title/body (`#nnnn`), issue→row through `docs/specifications/pp-066-dag.yaml` (`issues` lists on rows), file∩ against the first-wave P1 file lists in `docs/audits/pp-066-plan.md` plus the path tokens in each DAG row.
- STEP 1 one disposition per PR; MERGE-0.66 queue position = earliest expiry among intersecting rows.
- STEP 2 one rebase attempt per PR in a detached worktree; batches with `--rebase-merges`. Only two mechanical resolutions were applied and both are named in the entry: #2807 (lockfile regenerated from main's Cargo.lock with `cargo update -p wasmtime --precise 47.0.4`, verified by `cargo metadata --locked` and `cargo deny check advisories`) and #2773 (`.gitignore` both-appended → union; the replay then stopped on six other files and was aborted).
- STEP 3 arming only for MERGE-* with gate green and mergeable == MERGEABLE, one code PR at a time; workflow-touching PRs are pushed but not armed (STOP list).
