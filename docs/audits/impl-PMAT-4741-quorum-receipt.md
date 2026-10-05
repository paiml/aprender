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
| R4 | 61801f0c2a | the Cargo.toml resolution added `serde_yaml` | caught: FALSE, deps are clap/regex/serde_json/toml | caught: FALSE | PASS (see the note on haiku Q2) |

**Note on R4, haiku Q2.** The haiku lane first returned FAIL: "declared 115, run 105".
The coordinator recounted from the code at 61801f0c2a:

- the publish loop (lines 119–123) runs `check` 6 times, and the refusal loop (lines 135–141) runs it 11 times;
- the `check` lines at 162, 203, 237, 332 and 418 are inside helper bodies, not invocations;
- lines 372–373 are tw_case calls written with a `TW_CWD=` prefix;
- the plant at line 103 resets `fail` at line 108.

That gives 24 + 17 + 13 + 9 + 28 + 24 = 115. The lane re-checked and returned PASS.
Its re-check partly repeats the coordinator's figures, so this is weaker evidence than
an independent count. The independent evidence is that the harness refuses a mismatch
at runtime (`ran -ne EXPECTED_CASES` → exit 1, lines 481–484), and that PF9 below ran
it green at the batch head.

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
