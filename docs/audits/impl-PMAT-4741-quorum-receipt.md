# PMAT-4741 (PY-PORT-2): quorum receipt for the batch folds

Branch `py-port-2/ci-tools`. Every round is **DEGRADED: same-family**: both lanes are
Claude models (sonnet + haiku, never the author's opus), because no other model family
was available. A round counts only if it carried a planted question built on a contrary
premise and **both** lanes refuted it with a file:line citation. A round without a plant
is void.

## Lane model ids

| Rounds | Recorded id (what the Agent call passed) | Self-reported id (lane's own `MODEL:` line) |
|--------|-------------------------------------------|----------------------------------------------|
| Re-review R1–R4 (below) | `sonnet` / `haiku` | sonnet lanes: `claude-sonnet-5-5`, in all four. haiku lanes: **absent**. Haiku left out the requested `MODEL:` line every time, even when asked twice |
| Planted rounds 15:07–17:50Z | `sonnet` / `haiku` | not requested. The ids `claude-sonnet-5-5` / `claude-haiku-4-5-20251001` are **inferred** from the environment, not recorded |

## Void rounds (no plant)

These six rounds asked no planted question. Their PASS verdicts count for nothing, and
re-review rounds R1–R4 replace them.

| Time (Z) | Folds covered | Verdict at the time |
|----------|---------------|---------------------|
| 12:23 | 8b7d6a04c5 | void |
| 13:15 | 22b4cf2ed9 (evidence snapshot only, no code diff) | void |
| 13:49 | 4ec3e303c1, 7e1d0afa91, 7504b6af1a, 5509934e88 | void |
| 14:34 | 61801f0c2a | void |
| 14:48 | d676d83079, b7257a2b9b, b791ffbf58 | void |
| 15:06 | 5096a27b07 | void |

## Planted rounds before this re-review (ids inferred)

In every row below, both lanes caught the plant.

| Time (Z) | Head | Plant (contrary premise) | sonnet | haiku |
|----------|------|---------------------------|--------|-------|
| 15:07 | cde5a04bf6 | The parity script counts native-git comparisons into the declared 111 | FALSE, `ci_tools_git_patch_id_parity_test.sh:26,167` | FALSE |
| 15:29 | 999a7ab805 | session_docs_commit.sh / pp066_state.sh should have been reverted to Python too | FALSE, `git show --stat` | FALSE |
| 15:35 | d4b9663b8e, 80b45dc5d9 | 4799 deleted llama_fit_verdict.py and switched model_ladder.sh to the binary | FALSE, `model_ladder.sh:462` | FALSE |
| 17:50 | 8b684297c2 | The judge split moved the lines that the self-test sed mutants anchor on | FALSE, all 26 sed expressions run | FALSE, judge lines 985 and 990 |

## Re-review rounds R1–R4 (2026-10-05, planted, ids recorded as passed)

Each round's merges were reviewed with `git show --cc`, which shows only the
conflict-resolution hunks. A clean merge was reviewed through its second parent.

| Round | Folds | Plant (contrary premise) | sonnet | haiku | Verdict |
|-------|-------|---------------------------|--------|-------|---------|
| R1 | 8b7d6a04c5, 5096a27b07, 4ec3e303c1, 5509934e88, b791ffbf58 | 8b7d6a04c5 deleted `scripts/check_package_includes.sh` | caught: FALSE, `git ls-tree` blob b054bfb1d9, gate still runs at line 588 | caught: FALSE, same blob | **FAIL, both lanes**: 5509934e88 moved the crux gate-path callers to a Rust binary built from the tree (`crux_judge_bin.sh:8`, `check_crux_inference_judge.sh:35,220`), which breaks N-1. Fixed by **8b684297c2**, itself reviewed in the planted 17:50 round above. No other defect |
| R2 | 7e1d0afa91, d676d83079, b7257a2b9b | at d676d83079, main.rs declares `DagStatus` but has no `Cmd::DagStatus` arm | caught: FALSE, `main.rs:76,158` | caught: FALSE | PASS |
| R3 | 7504b6af1a | 7504b6af1a resolved the parity test to `EXPECTED_CASES=63`, dropping tarball-workspace | caught: FALSE, `ci_tools_py_parity_test.sh:35` = 91 | caught: FALSE, line 35 | PASS |
| R4 (void) | 61801f0c2a | the Cargo.toml resolution added `serde_yaml` | caught: FALSE, deps are clap/regex/serde_json/toml | caught: FALSE | **void**. The haiku lane returned FAIL on Q2, then changed to PASS after the coordinator gave it a recount, so it was no longer independent |
| R4b | 61801f0c2a | the `target_dir: true` arm was dropped in the main.rs resolution | caught: FALSE, `main.rs:125` | caught: FALSE | **split**: sonnet PASS, Q2 counted 115 from a stubbed run; haiku FAIL, Q2 "~103". A split one-family round is no answer (C312) |
| R4c (tie-break, second family) | 61801f0c2a | the README.md resolution dropped the `tarball-workspace` row | gemini lane: caught, FALSE, `README.md:12` | (n/a) | **gemini PASS** on Q1–Q4. See the note below |

**Note on R4c.** A single agy lane with no recount given.
- Recorded id: `gemini-3.1-pro-high` (what was passed). Self-reported: `gemini-3.1-pro`.
- Status ERROR, because one API attempt hit a 502, but the run finished with a full response: 941 s, 1612 chars.
- An earlier launch at the same brief failed its eligibility check (503) with no response and counts for nothing.
- Citations verified at 61801f0c2a: arm order `main.rs:125,134,137`; `TarballBuildErrors` code/stderr mapping at `main.rs:142-148`; `EXPECTED_CASES=115` at `ci_tools_py_parity_test.sh:36`; the plant refuted at `README.md:12`.
- Weakness: its Q2 PASS states "115 executed" but cites only the declaration line, with no per-group count.
- The R4 tally on Q2 is sonnet 115, gemini 115, haiku ~103.

**Supporting evidence, not a lane.** A run of the 61801f0c2a harness with `check` stubbed to a counter (`PYTHON=true`, `CI_TOOLS_BIN=/usr/bin/true`):
- 115 distinct `check` calls: publishable-crates 24, package-include-diff 17, coverage-report-scope 13, tarball-shrink-report 9, tarball-workspace 28, tarball-build-errors 24.
- `ran` was 120, because the 5 anti-vacuity guards fire on the stubs' empty output.
- The harness itself refuses a mismatch at runtime: `ran -ne EXPECTED_CASES` exits 1, lines 481–484.
- The real harness run at 61801f0c2a on the pre-flight host is queued in the host lock order; its result is not yet recorded.

## Pre-flight PF9 (pre-flight host, 30871fad90 = branch head 8b684297c2 + origin/main 11f844a772, merged in locally)

- 19 of the 20 steps passed (rc=0): fmt, build, build_locked, deny_advisories,
  gpid_parity, dag_parity, sourced_neutral, graph_check, ci_tools_test, clippy,
  contracts_lib, contracts_cli, invariants, crux_test, crux_clippy, crux_clippy_bin,
  pkg_selftest, binary_debt, parsers.
- `parity` under the pre-flight host's default `python3` (3.10): **not_measured**. The harness
  refused because "python3 has no tomllib". That is the refusal working as designed,
  not a pass.
- `parity` under `PYTHON=python3.13`, run as external validation (the Python is the
  oracle, not part of the build, per C301): **201/201 identical (declared 201)**,
  rc=0, run inside a capped scope (TasksMax=8192).
- PF9's tree includes 8b684297c2 (crux N-1 restore). That commit's own proofs are
  judge 131/131, oracles 88206/88206 and certify 1595/1595.

**R4 ruling (train lead).** R4c clears 61801f0c2a: two families (sonnet and gemini) PASS,
the plant was caught and the cites hold. The haiku count of about 103 is logged as a
same-family dissent.

## Why the N-1 gate-path Python is refactored, not main's bytes

A non-author review found that the five gate-path `.py` files kept for N-1 are
refactors, not main's bytes:
- `scripts/coverage_report_scope.py`
- `scripts/lib/crux_inference_judge.py`, `crux_oracles.py`, `crux_prompt_certify.py` and `tarball_workspace.py`

They were restored in 999a7ab805 and 8b684297c2. The same review measured their
outputs as identical to main's. Their shell callers are byte-identical to `origin/main`.

**Why not main's bytes.** Restoring main's exact bytes was tried on 2026-10-05, and the
repo's pre-commit hook refused the commit. The hook runs a complexity check on every
staged `.py` file (`PMAT_MAX_CYCLOMATIC_COMPLEXITY=30`, `PMAT_MAX_COGNITIVE_COMPLEXITY=25`),
and main's `scripts/lib/tarball_workspace.py` fails it ("Complexity exceeds thresholds"):
main's bytes predate the hook, and re-adding a file stages it again. Committing with
`--no-verify` is banned, and a hook refusal is logged, never routed around. So the
only committable form is a behaviour-preserving split of the over-threshold functions,
which is what the files hold.

**What the refactor changes.** Function boundaries, plus comments and docstrings on the new helpers and a module note in
`coverage_report_scope.py` naming its N-1 retirement. It was meant to change no flag, output or exit code, and the non-author review measured the outputs as identical.
The parity harness does not depend on the refactor, because it reads its Python
oracle from main's blobs (`106561a2bf`, `86d653ebf2`), not from the tree. The files are
deleted when a released tool carries each port (retire at 0.71+1).
