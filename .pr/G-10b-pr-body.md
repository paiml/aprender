PP-066 DAG row **G-10b** · ticket **PMAT-1063** · Closes #3013 · refs #2999 · epic #2873. Receipt: `docs/audits/impl-PMAT-1063-receipt.md`. Follows PR-A (#3011, merged b0a0a51b2).

**What lands.** `scripts/check_pmat_pinned.sh` — the operator assertion as a **shrink-only** guard: `grep -rEn '(^|[^_/])pmat ' scripts/ .github/workflows/ | grep -v pmat_bin`, counted against `scripts/pmat_unpinned_baseline.txt`. The baseline is **243, measured by the guard itself at the commit it names** (`--update` writes the command and the sha into the file; the "281" of the driver was the pre-PR-A count). A count above the baseline is RED naming every line; below it is an improvement to record; a missing or `INVALID` baseline is ENV (exit 2), never a pass. Kind-table entry (`count`) in `check_baseline_ratchets.sh`; two CI steps in `guard-runner-labels` (case table, then live); contract `apr-pinned-analyser-ratchet-v1` 1.0.0 → **1.1.0** (PIN-OB-005 / PIN-F-005). The sweep to 0 is G-10c (#3014).

**Mutation evidence (I3) — on this branch, never in the queue.**
| leg | commit | what | run |
|---|---|---|---|
| RED | `c4f6b618a` mutant: one bare `pmat analyze satd` comment appended to `scripts/ci_target_watch.sh` | `FAIL check_pmat_pinned: unpinned=244 baseline=243 — 1 new line(s) …` naming the line | _run id filled in after CI reports_ |
| GREEN | the revert (next commit) | `PASS … unpinned=243 baseline=243` | _run id filled in_ |

Case table `bash scripts/check_pmat_pinned.sh --self-test` → **20/20**: rows 1–11 the spellings (five match, six sanctioned do not), rows 12–15 the resolver (at-pin resolves; off-pin and absent refused; option-neutral), rows 16–20 the ratchet on a fixture tree (baseline 2 PASS · baseline 1 RED naming both lines · baseline 3 PASS + improvement · no baseline ENV 2 · `INVALID` ENV 2).

**Acceptance (re-run by the orchestrator on the re-cut base b0a0a51b2 — `.pr/G-10b-verify.log`):** self-test 20/20 · live `unpinned=243 baseline=243` · `check_baseline_ratchets.sh` PASS · `check_guards_are_wired.sh` PASS · `pv validate` valid · `check_shell_lint_ratchet.sh` PASS · `check_no_claim_literals.sh` rc 0.
**Write set:** the guard, the baseline, the kind-table line, two ci.yml steps, the contract, the receipt. No DAG/roadmap/README/spec edit.
