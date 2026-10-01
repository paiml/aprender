---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 12
subsystem: infra
tags: [laya, aprender-decide, claude-md, binding-registry, contract-audit, ci, justfile, close-out, pv]
outcome: complete

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-16 gate_pass (08-GATE-RUN-EVIDENCE.json) and the 08-18 FINAL live record, outcome deployed-passed (08-LIVE-DEPLOY-EVIDENCE.json, final: true)"
provides:
  - "CLAUDE.md Decision-Model Classification row plus the third deliberate EXCEPTION paragraph (Phase 8 D-16). Every claim in it is licensed by the two records"
  - "All 44 Phase 8 binding rows implemented, with signatures copied from source; accepted_region_cold implemented from the deployed-passed record"
  - "Strict make contract-audit-phase8: refuses any BIND- line, resolves every row to a definition site, PHASE8_LIVE_EXEMPT empty"
  - "CI integration step runs -p aprender-decide --test ui and -p aprender-mcp-decide --test e2e_stdio (user decision, option A narrowed)"
  - "just laya-verify-suite: the Python-parity / real-weights surface in one local-only command"
  - "The contract dependency cycle decide-apr-v1 -> laya-finetune-gate-v1 -> laya-parity-v1 -> decide-apr-v1, present since 08-01, is removed"
  - "deferred-items.md Phase 8 close-out plus D-ITEM-08-12-A..D"
affects: [phase-8 verification, ci workspace-test, contract graph, future decide-model plans]

actuals:
  tokens: 23538    # chars/4 over git diff 363a26fec..7d5c65f4b (91010 chars) + the uncommitted deferred-items diff (3145), SUMMARY excluded
  tasks: 3         # Task 1 decided by the user, Task 2 committed by the previous executor, Task 3 this continuation
  commits: 4       # MEASURED: git rev-list --count 363a26fec..HEAD before this SUMMARY's docs commit
plan_head_before: 363a26fecd71c2cbf6c7b83dd1be072acf0c9f8a

tech-stack:
  added: []
  patterns:
    - "Env-gated parity targets stay out of CI behind their own flags and get ONE local recipe that arms them all; in that recipe an armed leg that prints SKIP: is a failure"
    - "A base-commit control that shares the main target dir poisons it: cargo judges path-crate freshness by mtime, so the fresh worktree's artifacts look newer than HEAD's sources. Use a separate target dir, or touch the tracked sources afterwards"

key-files:
  created:
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-12-SUMMARY.md
  modified:
    - CLAUDE.md
    - Makefile
    - contracts/aprender/binding.yaml
    - contracts/decide-tool-boundary-v1.yaml
    - contracts/laya-finetune-gate-v1.yaml
    - contracts/decide-apr-v1.yaml
    - contracts/laya-parity-v1.yaml
    - crates/aprender-decide/src/artifact/determinism.rs
    - .github/workflows/ci.yml
    - justfile
    - scripts/laya_train/README.md
    - crates/aprender-decide/README.md
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/deferred-items.md

key-decisions:
  - "CI checkpoint answered by the user (2026-09-27), option A narrowed. Only the two pure-Rust targets were added to ci.yml. make contract-audit-phase8 stays tier3-only, since make in the CI image is unconfirmed. The four env-gated parity targets and the Python self-tests are local-only, via just laya-verify-suite"
  - "laya-parity-v1 no longer depends_on decide-apr-v1. The edge runs the other way, and the back-edge made pv certify / verify-pipeline refuse the whole contract set. pv diff calls the edit identical, so there is no version bump"
  - "FULL was measured twice. First the exact CI command, which is fail-fast under the ci profile and stopped after one test. Then the same command with --no-fail-fast, to enumerate every failure. Each failure was controlled on base 59d9ed07f, except 15 ENOSPC ones (D-ITEM-08-12-D)"

patterns-established:
  - "Local-only verification surfaces are named in deferred-items as open-by-design, with the recipe and the obligation to run it before touching the covered code"

requirements-completed: [D-15, D-16, D-17, D-18]

coverage:
  - id: D1
    description: "CLAUDE.md Decision-Model Classification row and the Phase 8 D-16 exception paragraph. Only real paths; claims follow gate_pass (with the 1.4.0 in-distribution claim scope and the shift probe) and deployed-passed"
    requirement: "D-16"
    verification:
      - kind: integration
        ref: "cargo test -p aprender-core --test monorepo_invariants --test readme_contract -> 11 + 15 passed (FALSIFY-DOCS-CLAUDE-001)"
        status: pass
      - kind: other
        ref: "08-12-PLAN Task 2 verify 3 (records vs bindings, TOOL-009 and CLAUDE.md claims), re-run by the continuation -> 'bindings, TOOL-009 and CLAUDE.md claims follow the records', rc 0"
        status: pass
    human_judgment: false
  - id: D2
    description: "All 44 Phase 8 binding rows implemented with source signatures; strict contract-audit-phase8 refuses any BIND- line and resolves every row; 7 induced-failure / positive controls behave"
    requirement: "D-17"
    verification:
      - kind: other
        ref: "make contract-audit-phase8 -> rc 0, 4 'Total equations:', 0 BIND- lines, 0 EXEMPT lines, 'resolved 44 Phase 8 binding rows to definition sites' (before and after the laya-parity fix)"
        status: pass
      - kind: other
        ref: "scratchpad controls.sh: c0 rc 0, c1 rename rc 2, c2 other row partial + exemption rc 2, c3 live row partial + exemption rc 0 (EXEMPT line), c4 live row pending + exemption rc 2, c5 bad exemption rc 2, c6 live row partial without exemption rc 2"
        status: pass
    human_judgment: false
  - id: D3
    description: "accepted_region_cold implemented from the FINAL deployed-passed record; FALSIFY-DECIDE-TOOL-009 names the live harness and evidence path; PHASE8_LIVE_EXEMPT empty"
    requirement: "D-18"
    verification:
      - kind: other
        ref: "08-12-PLAN Task 2 verify 3 -> 'gate outcome gate_pass | final live outcome deployed-passed', non-implemented Phase 8 rows: []"
        status: pass
    human_judgment: false
  - id: D4
    description: "ci.yml integration step gains exactly cargo test -p aprender-decide --test ui and cargo test -p aprender-mcp-decide --test e2e_stdio; YAML parses; no other workflow change"
    requirement: "D-15"
    verification:
      - kind: other
        ref: "python3 yaml.safe_load(.github/workflows/ci.yml) OK; git diff -U0: 1 line changed; greps: ui 1, e2e_stdio 1, laya_parity/fail_closed_vectors/demo_run/python_records/make contract-audit-phase8 0"
        status: pass
      - kind: integration
        ref: "cargo test -p aprender-decide --test ui (1 passed, 33 s warm) and cargo test -p aprender-mcp-decide --test e2e_stdio (2 passed, 12 s, real-model leg SKIP)"
        status: pass
    human_judgment: true
    rationale: "The CI-side result of the edited line is not yet observed: nothing was pushed in this plan. The first PR run's integration step is the evidence"
  - id: D5
    description: "just laya-verify-suite runs the Python self-tests and the four env-gated parity targets ARMED; documented in scripts/laya_train/README.md and crates/aprender-decide/README.md"
    requirement: "D-15"
    verification:
      - kind: e2e
        ref: "rtk proxy just laya-verify-suite -> LAYA VERIFY SUITE OK, rc 0, 903 s (the only SKIP was the optional ladder rung)"
        status: pass
    human_judgment: false
  - id: D6
    description: "Contract dependency cycle removed (laya-parity-v1 depends_on: []); the three aprender-contracts-cli real-contract tests that failed at every Phase 8 commit now pass"
    verification:
      - kind: unit
        ref: "cargo nextest run --profile ci --workspace --lib (3 excl.) --no-fail-fast -E <7 contracts tests> -> 7 run: 3 passed (the -cli three), 4 failed (the base-identical aprender-contracts four)"
        status: pass
    human_judgment: false
  - id: D7
    description: "FULL CI-equivalent evidence with base-commit controls"
    verification:
      - kind: other
        ref: "FULL / controls sections of this SUMMARY: 84 of 87 controlled failures fail identically on 59d9ed07f, 3 were a Phase 8 regression (fixed, D6); examples failure identical on base"
        status: pass
    human_judgment: true
    rationale: "15 aprender-qa-runner failures (ENOSPC) could not be controlled at 4 GiB free (D-ITEM-08-12-D). They are attributed by evidence only: the crate is untouched by Phase 8, and the error is the host's disk"
  - id: D8
    description: "deferred-items.md: Phase 8 close-out section plus D-ITEM-08-12-A (verify.rs split), -B (parity suite local-only), -C (audit not a CI gate), -D (uncontrolled ENOSPC)"
    requirement: "D-18"
    verification:
      - kind: other
        ref: "grep 'Phase 8 close-out', 'D-ITEM-08-13-A', 'D-ITEM-08-01-A', 'D-ITEM-08-12-B', 'D-ITEM-08-12-C', 'D-ITEM-08-12-D' in deferred-items.md"
        status: pass
    human_judgment: false

duration: 86min
completed: 2026-09-28
status: complete
---

# Phase 8 Plan 12: Close-out Summary

**This close-out documents the D-16 decision-model exception in CLAUDE.md and binds all 44 Phase 8 equations to their real code, with `accepted_region_cold` implemented from the deployed-passed record. `contract-audit-phase8` now refuses any unbound or partial row. CI runs the two pure-Rust targets; the Python-parity surface runs locally through `just laya-verify-suite`. It also fixes a contract dependency cycle, present since 08-01, that would have turned `workspace-test` red.**

## Performance

- **Duration:** about 86 min in total. Task 2 ran in the previous executor, from about 23:18Z; this continuation ran 23:50:54Z to 00:43:30Z (52 min).
- **Started:** 2026-09-27T23:18Z (approximate; the plan start after 08-18's STATE write)
- **Completed:** 2026-09-28T00:43:30Z
- **Tasks:** 3 of 3
  - Task 1: decision checkpoint, answered by the user.
  - Task 2: `f1226a4d5`.
  - Task 3: `57e18b541`, `57423ef6b`, `7d5c65f4b`.
- **Files modified:** 13 across the plan's commits, plus deferred-items.md in the docs commit.

## Accomplishments

- **D-16 in CLAUDE.md** (Task 2). The new row and paragraph cite only real paths. They claim one TweetEval stance model passed the declared gate, and they carry its claim scope:
  - in-distribution calibration only, not robustness to shifted input;
  - the eval set was chosen after the SemEval test split failed (spike 027);
  - the 0.1896 shift-probe ECE is stated beside it.

  The paragraph also says the model is live on pmcp.run at 3 GB, with the thin cold margin stated. `readme_contract` passes.
- **Bindings** (Task 2). All 44 Phase 8 rows are `implemented`, with signatures copied from source (`Decider`, `ArtifactError`, `declared_len: Option<u64>`). `accepted_region_cold` is implemented from the FINAL record, outcome `deployed-passed`. FALSIFY-DECIDE-TOOL-009 names `just laya-deploy-verify …` and an `evidence:` path.
- **Audit** (Task 2). `contract-audit-phase8` now has the phase6 shape and adds row-to-definition resolution. It resolves 44 rows and admits a single `PHASE8_LIVE_EXEMPT`, which is empty.
- **CI** (Task 3, user decision). The integration step gains exactly `-p aprender-decide --test ui` and `-p aprender-mcp-decide --test e2e_stdio`.
- **Local suite** (Task 3). `just laya-verify-suite` runs:
  - `metrics.py`, `data.py` and `gate.py --selftest`;
  - `laya_parity`, `fail_closed_vectors` and `demo_run`, ARMED;
  - the lifecycle kept to a temp dir, then `python_records` against it.

  It ran once: `LAYA VERIFY SUITE OK`.
- **A real Phase 8 regression, found and fixed** (Task 3 FULL). The 08-01 dependency edge `laya-parity-v1 → decide-apr-v1` closed a cycle, and `pv certify` / `pv verify-pipeline` refused the contract set. The three `aprender-contracts-cli` tests failed at HEAD and passed on the base. Only the unfiltered workspace lib run exercises them.

## Task Commits

1. **Task 1: CI checkpoint** (decision, user). There is no commit; the ci.yml edit it authorized is part of Task 3.
2. **Task 2: CLAUDE.md D-16, bindings, strict audit, close-out section**: `f1226a4d5` (feat), by the previous executor.
3. **Task 3: CI edit, local suite, FULL with controls**:
   - `57e18b541` (chore): ci.yml.
   - `57423ef6b` (feat): justfile, the READMEs and deferred items.
   - `7d5c65f4b` (fix): the contract cycle.

**Plan metadata:** this docs commit (SUMMARY, deferred-items D-ITEM-08-12-D, STATE, ROADMAP, REQUIREMENTS).

## Files Created/Modified

- `CLAUDE.md`: the Decision-Model Classification row and the "third deliberate EXCEPTION (Phase 8 D-16)" paragraphs.
- `contracts/aprender/binding.yaml`: 44 Phase 8 rows implemented, with source signatures.
- `contracts/decide-tool-boundary-v1.yaml` (3.0.0): FALSIFY-DECIDE-TOOL-009 names the live harness and evidence.
- `contracts/laya-finetune-gate-v1.yaml` (3.0.0): `why_quantized` and the `seed_selection_median` invariant now state the rank_key rule 08-15 implemented. No value moved.
- `contracts/decide-apr-v1.yaml`: FALSIFY-DECIDE-APR-003's ON leg now runs on `--features serde-preserve-order`.
- `crates/aprender-decide/src/artifact/determinism.rs`: its doc comment follows that change.
- `contracts/laya-parity-v1.yaml`: `depends_on: []`, which removes the cycle.
- `Makefile`: the strict `contract-audit-phase8`, and `PHASE8_LIVE_EXEMPT :=` (empty).
- `.github/workflows/ci.yml`: the two pure-Rust targets on the integration line.
- `justfile`: `laya_model_dir` and `laya-verify-suite`; the Laya section header records the CI decision.
- `scripts/laya_train/README.md` and `crates/aprender-decide/README.md`: document the recipe and why the suite is local-only.
- `deferred-items.md`: the Phase 8 close-out section, plus "From plan 08-12" with D-ITEM-08-12-B, -C and -D.

## CI checkpoint (Task 1): decision and measured costs

The user's answer (2026-09-27): *"Option A. Regarding the make question, I'm not sure as the Aprender is designed for Rust, and our addition of converting Python algorithm and verifying that it give the same results is something that I've added only recently. Not sure that we need to have it as part of the CI, and we can bypass them or protect them with a specific flag."*

What was applied, with the local wall times measured on warm artifacts, `CARGO_INCREMENTAL=0`, on this aarch64 box:

| Item | Where it runs now | Measured |
|------|-------------------|----------|
| `cargo test -p aprender-decide --test ui` | CI integration step | 33 s (trybuild 20.8 s), 1 passed |
| `cargo test -p aprender-mcp-decide --test e2e_stdio` | CI integration step | 12 s, 2 passed, `SKIP: APR_MCP_E2E_DECIDE_MODEL not set` (real-model leg) |
| `make contract-audit-phase8` | local / tier3 only (D-ITEM-08-12-C) | 8 s |
| `laya_parity`, `fail_closed_vectors`, `demo_run`, `python_records` + Python self-tests + lifecycle | local only, `just laya-verify-suite` (D-ITEM-08-12-B) | 903 s together, armed |

## `just laya-verify-suite`: the recorded run (2026-09-27, rc 0, 903 s)

Run through `rtk proxy` so the tests' `println!` lines survive. The key lines:

- `METRICS SELFTEST OK (5 frozen cases replayed within 1e-06)`, `DATA SELFTEST OK`, `GATE SELFTEST OK`. The gate self-test's run-dir recompute ran on both vectors (`max |d| = 0 (tol 1e-05)`), with no SKIP.
- **laya_parity** (5.8 s): `ids 14/14; argmax 14/14; max |dp| 3.841e-6 (bar 1e-5); max |dlogit| 2.146e-5 (bar 1e-4); truncated rows 1; ARCH aarch64`. It printed `SKIP ladder rung: LAYA_LADDER_BIN not set`; that is the optional oracle dump, not the leg.
- **fail_closed_vectors** (387 s): `FAIL-CLOSED VECTORS REFUSED 2/2`.
  - `d0f4e40d` refused `RescoreDrift which=fine_tuned row=59 max_abs=4.667e-5 bound=1e-5` on both pack and verify.
  - `3d4b91da` refused `GateFailed clauses=[ece_post]` (margin 0.1056, ece_post 0.2224) on both.
- **demo_run** (419 s): `DEMO OUTCOME gate_pass decided on the exact bytes`: sha256 `24a44d7e…`, shipped_seed 17, ece_post 0.04425, margin 0.2217, rescore_max_abs 1.13e-5 within the noise-derived bound 3.34e-5.
- **lifecycle**: `LIFECYCLE OK`. The run was kept to a temp dir.
- **python_records**: `NOISE which=fine_tuned rust=4.2993464954843574e-8 python=4.2993464954843574e-8`, `NOISE which=zero_shot rust=3.1062774519252656e-8 python=3.1062774519252656e-8`, `MEDIAN rust=23 python=23`, `SHIFT probe recomputed`.
- `LAYA VERIFY SUITE OK`

## Task 2 results (previous executor, re-verified by this continuation)

- The two outcomes, quoted from the records: `gate outcome gate_pass | final live outcome deployed-passed`. The CLAUDE.md paragraph was read against both: the gate-pass claim carries the 1.4.0 claim scope and the spike 027 history, and the live claim is licensed by `deployed-passed`.
- `readme_contract` passes 15 tests and `monorepo_invariants` 11. `grep -c 'Phase 8 D-16' CLAUDE.md` = 1 and `grep -c 'Decision-Model Classification' CLAUDE.md` = 1.
- `make contract-audit-phase8`: rc 0, 4 `Total equations:`, 0 BIND- lines, 0 EXEMPT lines, 44 rows resolved. `make contract-validate` rc 0.
- The strict-binding guard is VACUOUS (D-ITEM-08-01-A). On the lifted copy: `Resolved 684 test references; 44 dangling across 15 contract(s)`, and the only FAIL lines are the two pre-existing contracts, chronos-bolt-parity-v1 (9) and setfit-encoder-conformance-v1 (8).
- **Induced-failure controls**, re-run on the committed tree by this continuation (each mutation reverted with `git checkout -- contracts/aprender/binding.yaml`):

| Control | Mutation | rc | Verdict |
|---------|----------|----|---------|
| c0 | none (baseline, exemption empty) | 0 | 44 rows resolved, no finding |
| c1 | `Admission::try_admit` renamed in its row | 2 | `RESOLVE- … no definition site for fn 'try_admit_renamed'` |
| c2 | exemption set + `classify_admission` partial | 2 | its BIND-002 is refused; only the live equation is admissible |
| c3 | exemption set + `accepted_region_cold` partial | 0 | `EXEMPT (live, see deferred-items.md): … BIND-002 …` (positive control) |
| c4 | exemption set + `accepted_region_cold` pending | 2 | BIND-004 refused; only BIND-002 is admitted |
| c5 | `PHASE8_LIVE_EXEMPT=decide-apr-v1.yaml:private_mint` | 2 | `It may hold only decide-tool-boundary-v1.yaml:accepted_region_cold` |
| c6 | `accepted_region_cold` partial, no exemption | 2 | BIND-002 refused (the deployed-passed posture) |

## Task 3 evidence

### FOCUSED

| Check | Result |
|-------|--------|
| Phase 8 nextest filter (`package(aprender-decide) \| package(aprender-mcp-decide) \| package(aprender-mcp-decide-lambda) \| test(/modernbert/)`, ci profile) | rc 0, `Summary [   5.042s] 185 tests run: 185 passed, 81905 skipped` |
| `-p aprender-decide --test ui --test laya_parity --test fail_closed_vectors --test demo_run --test python_records`, LAYA_* unset | rc 0, 5 × `test result: ok`, 4 SKIP lines (the env-gated four, as CI would see them) |
| `-p aprender-mcp-decide --test e2e_stdio` | rc 0, 2 passed (real-model leg SKIP) |
| `-p aprender-core --test monorepo_invariants --test readme_contract` | rc 0, 11 + 15 passed |
| `make contract-audit-phase8` / `make contract-validate` | rc 0 / rc 0 (also after the laya-parity fix) |
| strict-binding guard | VACUOUS → lifted copy: only the two pre-existing contracts dangle (D-ITEM-08-01-A) |
| `just laya-verify-suite` (armed) | rc 0, `LAYA VERIFY SUITE OK` (above) |

### FULL

The exact CI command, `cargo nextest run --profile ci --workspace --lib --exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute`, gave rc 100. Its verbatim Summary line:

```
     Summary [  31.368s] 7921/81942 tests run: 7920 passed, 1 failed, 148 skipped
```

The `ci` profile is `fail-fast = true` (`.config/nextest.toml`). It stopped after its first failure, `aprender-cgp analysis::roofline::tests::test_empirical_flops_positive` ("Single-core FLOPS 0.8 GFLOP/s suspiciously low", 3 tries), and did not run 74,021 tests. To enumerate everything, the same command was re-run with `--no-fail-fast` (an added flag, not what CI runs). It gave rc 100, and its verbatim Summary line:

```
     Summary [ 428.777s] 81942 tests run: 81840 passed (12 slow, 2 flaky), 102 failed, 148 skipped
```

`cargo build --examples --workspace --exclude aprender-profile --keep-going` gave rc 101. The only failure is `aprender-serve` example `bench_simd_dot`: 8 errors (`std::arch::x86_64`, `ymm0-2`, `avx2`/`fma` target features), an x86_64-only example on this aarch64 host.

**Base-commit controls** used base `59d9ed07f`, the last commit before any Phase 8 code, in a detached worktree that has since been removed:

- **87 of the 102 failures** ran through the same `--workspace --no-fail-fast` command, with `-E` selecting exactly those tests. The base Summary line: `Summary [  34.162s] 87 tests run: 3 passed, 84 failed, 46732 skipped`.
  - **84 fail on the base.** The first failure message is identical for 83 of them. The 84th, `lint_passes_on_real_contracts`, has the identical `PROVABILITY-001: Kernel contract has no kani_harnesses (spectral-indices-v1)` error; only the contract count differs (1740 vs 1744, the four Phase 8 contracts).
  - The 84 by crate:
    - `aprender-serve` 51: debug-mode `attempt to multiply with overflow`.
    - `aprender-train` 21: `gpu::ledger` / `guard` / `wait` on a host with no NVIDIA GPU.
    - `aprender-contracts` 4: D-ITEM-08-01-A, plus the chronos/setfit dangling baselines.
    - `aprender-zram-core` 2: NEON lz4 at `pattern_len=16`.
    - `aprender-cgp` 2: roofline FLOPS and `/proc/meminfo`.
    - `aprender-orchestrate` 2: `apr` not on PATH.
    - `aprender-mcp` 1: ETXTBSY semantics.
    - `aprender-core` 1: the setfit golden hash, build-graph dependent (D-ITEM-05-17-A).
  - **3 pass on the base, so they were a Phase 8 regression:** `aprender-contracts-cli` `certify_on_real_contracts`, `verify_pipeline_on_real_contracts` and `verify_pipeline_json_on_real_contracts`. The error was `ERROR: 1 cycle(s) detected in dependency graph — cannot verify pipeline / cycle: decide-apr-v1 → laya-finetune-gate-v1 → laya-parity-v1`. **Fixed in `7d5c65f4b`.** Re-run at HEAD with the same command, filtered to the 7 contracts tests: `7 tests run: 3 passed, 4 failed`. The 3 are the fixed `-cli` tests; the 4 are the base-identical `aprender-contracts` ones.
- **15 of the 102 were NOT controlled:** `aprender-qa-runner` (14 `dimensional_check_tests_tokenizer_dtype`, 1 `layout_contract::tests`), each failing with `No space left on device`. The first full-workspace control was killed by its disk watchdog at 1 GiB free, and running these multi-GiB tests again at 4 GiB free was unsafe. They are attributed by evidence only: Phase 8 changed 0 files under `crates/aprender-qa-runner`, and the error is the host's disk. Recorded as D-ITEM-08-12-D.
- **Examples:** `cargo check -p aprender-serve --example bench_simd_dot` on the base, in its own target dir, gave rc 101 with the same 8 errors. Phase 8 changed 0 files under `crates/aprender-serve`.
- The STATE-listed SC2 invariance red (`aprender-forecast`) did not fail in this run.

### DOCUMENTED EXCLUSIONS

| Exclusion used locally | CI uses it? | Reason |
|------------------------|-------------|--------|
| `--exclude aprender-profile` on `cargo build --examples --workspace` | no | Darwin: `crates/aprender-profile/src/main.rs:6` is a `compile_error!` off Linux (STATE, pre-existing) |
| `--no-fail-fast` on the second FULL run | no | the `ci` profile is fail-fast; the flag only enumerates failures |
| the four env-gated parity targets and the Python self-tests | not in CI, by the user's decision | they need the base snapshot and gitignored run dirs; they run in `just laya-verify-suite` |
| **CI-side result** | pending | not observed in this plan: nothing was pushed. The first PR run of the edited integration line and `workspace-test` is that evidence |

## Decisions Made

- The CI edit was narrowed exactly as the user asked (above). No `make` in CI, and no Python step.
- `just laya-verify-suite` treats an armed leg that prints `SKIP:` as a failure. A SKIP there means the leg measured nothing. A missing snapshot is refused with exit 2, not skipped.
- `laya-parity-v1` declares `depends_on: []`. The dependency goes consumer → producer, the same as the gate contract and the artifact contract, which both depend on parity.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Contract dependency cycle among three Phase 8 contracts**
- **Found during:** Task 3 FULL, confirmed by the base control.
- **Issue:** since 08-01, `laya-parity-v1` declared `depends_on: [decide-apr-v1]`. `decide-apr-v1` depends on the gate and parity contracts, and the gate depends on parity, so `pv` found a cycle. As a result, `pv certify` / `pv verify-pipeline` refused the real contract set, and 3 `aprender-contracts-cli` tests failed at every Phase 8 commit. CI's `workspace-test` would have been red on push.
- **Fix:** removed the back-edge (`depends_on: []`, with a comment giving the direction and the evidence).
- **Files modified:** `contracts/laya-parity-v1.yaml`.
- **Verification:**
  - `pv validate` passes, and `pv diff` reports "Contracts are identical", so no bump applies.
  - The 3 tests pass under the same `--workspace` nextest command.
  - `make contract-audit-phase8` and `make contract-validate` pass.
- **Committed in:** `7d5c65f4b`.

### Adaptations of Task 3 to the narrowed decision

**2. [Scope - user decision] ci.yml carries 2 of the 7 planned additions.** The acceptance greps were adapted:
- `aprender-decide --test ui` = 1 and `aprender-mcp-decide --test e2e_stdio` = 1, as the plan asks.
- `laya_parity`, `fail_closed_vectors`, `demo_run`, `python_records` and `make contract-audit-phase8` each = 0. That is the intended result now, not a failure.
- `git diff .github/workflows/ci.yml` is exactly one changed line.

**3. [Scope - user decision] The decline-style deferred item is written for the parts left out.** It covers the four targets, the Python self-tests and the audit:
- D-ITEM-08-12-B: the parity suite is local-only by design.
- D-ITEM-08-12-C: the audit is not a CI gate until `make` in the image is confirmed.

  Task 3 also gained `just laya-verify-suite`, at the user's request ("protect them with a specific flag").

**4. [Measurement] FULL is reported twice.** The exact CI command is fail-fast under the `ci` profile, so its Summary line covers only 7921 of 81942 tests. A `--no-fail-fast` re-run of the same command enumerates all 102 failures. Both Summary lines are quoted verbatim above.

**5. [Measurement] 15 of the 102 FULL failures are not base-controlled** (ENOSPC, D-ITEM-08-12-D). The plan requires a control for every failure. This one could not be run safely at the host's free space, and it is recorded as open rather than claimed.

### Carried from Task 2 (previous executor, `f1226a4d5`)

**6. [Rule 1] The FALSIFY-DECIDE-APR-003 ON leg was swapped, not APR-005.** The brief said APR-005; the preserve_order ON leg is APR-003. It now runs on the test-only `--features serde-preserve-order`, not on feature unification through `-p aprender-mcp-setfit`, with `determinism.rs` doc updated. `pv diff` reports identical.

**7. [pv-suggested bumps] Two contracts went to 3.0.0.**
- `decide-tool-boundary-v1`: TOOL-009's live harness plus evidence, and the invariant / qa_gate prose follows the record. `pv diff` classifies it major.
- `laya-finetune-gate-v1`: the `why_quantized` text and the `seed_selection_median` invariant now state the rank_key rule 08-15 implemented. No value moved; `pv diff` classifies any invariant edit major.

**8. [Rule 1] Rebinds to the enforcing functions.** Some rows were rebound from the function their 08-01 guess named to the function that actually enforces the equation, as found in source. That is what row-to-definition resolution requires.

**9. [Record] D-ITEM-08-07-A recorded closed.** The bootstrap handler ran under the real Lambda runtime in 08-17/08-18, including its failed-load path.

**10. [New item] D-ITEM-08-12-A:** `crates/aprender-decide/src/verify.rs` is 2454 lines. A split is left to a refactor plan, which must re-run `make contract-audit-phase8`.

---

**Total deviations:** 1 auto-fixed bug in this continuation (Rule 1), 4 adaptations to the user's narrowed decision and to measurement limits, and 5 carried from Task 2.
**Impact on plan:** the cycle fix was required for the FULL run's regression rule. It is metadata-only and does not move a bar. The CI scope follows the user's explicit decision.

## Issues Encountered

- **Shared-target contamination (resolved).** The base control shared the main `target/` dir. Cargo keys workspace path crates by workspace-relative path, so both trees share one artifact set, but it judges freshness by source mtime. The fresh worktree's artifacts therefore looked newer than HEAD's sources, and HEAD's `aprender-decide` compiled against a base-built `aprender-core` (`E0425 MIN_INDEX_ENTRY_BYTES`).
  - Resolved by `touch`-ing the 10,349 tracked `.rs` files (content unchanged), which forced a rebuild from HEAD.
  - Every FOCUSED/FULL result above predates the control, or was re-run after the rebuild. The examples control used a separate target dir.
- **Disk.** Free space went 24 → 10 → 1 GiB over the plan. The first full-workspace control was killed by its own watchdog, and its killed test processes leaked 22 `.tmp*` TempDirs, **6.0 GiB in `$TMPDIR`**, born at or after 17:27:31 local. Removing them was refused by the permission classifier, so they are still on disk (D-ITEM-08-12-D); about 4 GiB is free now. `ctl-target` (425 MB, scratchpad) was removed, and so was the control worktree.
- **`.pv/` side effects.** `pv` runs and the contracts tests rewrote `.pv/contracts.idx`, `.pv/contracts.idx.mtime` and `.pv/lint-previous.json`. Each was restored with `git checkout -- <file>`; none is committed.
- **rtk filtering.** The hook filtered `make`'s output even when it was redirected to a file. Audit results were re-captured through `rtk proxy`.

## User Setup Required

None. There is no external service configuration.

- To reclaim disk, remove the 22 leaked `.tmp*` dirs in `$TMPDIR` born at or after 17:27:31 on 2026-09-27 (6.0 GiB).
- Run `just laya-verify-suite` before touching `aprender_decide::{pack,verify,laya}`, `scripts/laya_train` or `contracts/laya-*.yaml`.

## Next Phase Readiness

- Plan 18 of 18 is complete; the phase is ready for verification.
- Open work is recorded in the close-out section:
  - D-ITEM-08-12-B / -C / -D;
  - D-ITEM-08-01-A;
  - D-ITEM-08-13-A (x86_64);
  - D-ITEM-08-17-C / -D / -E;
  - D-ITEM-08-18-A (the endpoint is RUNNING with auth off);
  - D-ITEM-08-10-A / -B.
- The CI-side result of the edited integration line and of `workspace-test` (now free of the cycle) is the first thing to read when this branch is pushed.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-28*

## Self-Check: PASSED

- FOUND: every key file (ci.yml, justfile, both READMEs, laya-parity-v1.yaml, CLAUDE.md, Makefile, binding.yaml, deferred-items.md).
- FOUND: commits f1226a4d5, 57e18b541, 57423ef6b, 7d5c65f4b.
- Plan-level verification was re-run: Task 2 verify 3 rc 0, the audit, contract-validate, readme_contract, and the Task 3 FOCUSED set.
