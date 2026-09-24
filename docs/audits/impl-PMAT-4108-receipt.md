# PMAT-4108 receipt: guard_tree.sh vacuous PASS (#4108)

- ticket PMAT-4108 (from #4108), kind code, branch `fix/4108-guard-tree-zero-checks` off `origin/main` 49fe19c28; orchestrator opus-5-5.
- **Defect (measured, lambda 2026-09-23):** two concurrent `make gate` runs made guard_tree's mktemp scratch dir vanish mid-run. The tally loop read nothing from the missing `$PLAN`, and the run printed `0 checks, 0 failed` and exited 0, so `make gate` was green on nothing. What removed the dir is not diagnosed here; the gate now fails closed whatever the cause.
- **Fix:** the plan is counted in memory as it is written (`planned`). The tally counts the guards it accounts for (`accounted`). `accounted != planned` fails as `guard_tree [plan]`, naming both numbers and the dir. `total == 0` fails as `guard_tree [vacuous]`. `--list`, `--dry-run` and `--internal-run-one` return before the check (review lane C enumerated the callers: ci.yml:1096, Makefile gate, predict_merge.sh:157, all `--no-cargo`; none can legitimately select zero).
- **Must-RED rows** (`scripts/tests/guard_tree_test.sh`, run by ci.yml:1105):
  - 28: a guard deletes `$GUARD_TREE_RUN_DIR` (the incident) → rc 1, named reason.
  - 29: its mutant without the check → `0 checks, 0 failed`, exit 0 (the incident reproduced).
  - 30: an empty universe → rc 1 vacuous; 31: its mutant → exit 0.
  - 32: the plan truncated in place → rc 1, naming 2 planned / 0 accounted.
  - `vmutant_of` verifies its own mutant: a drifted anchor gives rc 1 (623 vs 653 lines), so rows 29/31 cannot pass on a truncated script.
- **Verification (orchestrator re-runs):** case table 34/34; `make gate` solo on the final tree gave 92 checks, 0 failed, exit 0 (gate-reduce sha256 972c2f8b…); bashrs findings unchanged vs main (33 / 15).
- **Quorum:** agy 429 (every family), so per the operator fallback rule 3 Claude Code lanes on sonnet-5, `degraded: same-family`, author opus-5-5. A PASS (verified by hand that truncation is also caught; minor: add row 32, done). B PASS (verified row 28 deterministic across GUARD_TREE_JOBS=1/8/16; major-latent: row 31 unguarded mutant, fixed). C PASS (no collateral damage; guard_tree_job_test 4/4, guard_tree_parallel_test 6/6).
- Verdict: DONE to the quorum receipt. Not armed (batching).
- **Re-review at b75088a2e (rows 33-37, universe check, --dry-run count):** agy ph7 gemini-3.8-flash-high PASS + gemini-3.7-flash-high PASS, plus the Claude Code haiku-4-5 lane PASS (case table 40/40 re-run), in the operator's 2 agy + 1 haiku shape. Record: `docs/audits/quorum-PMAT-4108.json`. The ph5 dissent (`|| exit 1` on the universe) was declined with a measurement: grep -L and xargs exit 1/123 on correct runs.
