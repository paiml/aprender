Receipt for paiml/aprender#4045 on branch feat/4045-release-gate-normal, base = feat/4040-nightly @ f07e330b4 (the #4037/#4040/#4051 stack, quorumed separately). The diff also carries f19a3ee6d (the crux-smoke judge skipping *.meta.json) via the merge.

- **M4 (normal release gate)**
  - `ladder.release_gate` (from 0.70.0) + `--scope release`: CRUX smoke on the release binary AND the admitted nightly.
  - Rows: 5 smoke rows; release-e2e-green; e2e-smoke-red-nightly-green (mutant smoke-folded-out); `--scope release` without `--nightly` declines (mutant smoke-alone); release-before-from, release-unrecorded.
- **M6 (docs)**
  - APR-RELEASE-001 §14 (the runbook).
  - pre-release skill Gate 13.
  - The CLAUDE.md text is a PROPOSAL file only.
- **M8 (shift-left)**
  - release_gate_classes (69 derived gates, 50 real / 19 bookkeeping after the Sonnet round (`contracts`) and the Fable round (`coverage`, `pv-lint`, `pv-contracts`) made them real; 13 rows / 11 mutants).
  - candidate_watch.sh, including G-ONT as the REAL row `g-ont:complete` (fail-closed when unconfigured) and autofix proposals; rows and mutants are in check_release_shift_left.sh.
  - check_no_shadowed_repo_skill.sh (5 rows / 3 mutants).
  - autopilot watch_gate/run_preflight: the publish re-reads the watch, aged by its own timestamp, the newest chosen by that timestamp, bound across the squash merge by tree equality (check_publish_reads_watch.sh, 9 rows / 7 mutants).
  - bookkeeping_autofix.sh + autofix_invariants.py (check_bookkeeping_autofix.sh, 9 rows / 7 mutants).
- **G-ONT**
  - check_ont_complete.sh: 9 rows / 8 mutants; live RED as designed.
  - REPORT-only in CI with no `--infra`.

All new guards are cargo-free, so guard_tree runs them on every PR. check_guards_are_wired PASSes. check_model_ladder.sh --self-test: 155/0. bashrs: 0 errors on the new scripts.

Pre-existing, not from this diff: check_release_models_t1 and check_baseline_ratchets exit non-zero on the #4046 base too.

**Not done here, stated (the degraded Fable round on ae15411aa):**
- `--scope release` has no caller in the publish path yet, so Phase 2 is retired by design only (#4117).
- A BOOKKEEPING class does not yet reach the publish; dogfood/R5 still stop on it (#4118, a design choice).
- G-ONT now REDs any spec row that has no probe file, and its last line names the pinned infra commit.
