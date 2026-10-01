---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 09
subsystem: decision-model-verification
tags: [laya, aprender-decide, verify, gate, fail-closed, rescore, parity, ece, macro-f1, pack, option-a]

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-05 packer (PackInputs::from_run_dir, write_decide_apr, Decider::load_bytes/load_path); 08-08 the two fail-closed demo vectors and laya-finetune-gate-v1 1.2.0 demo block; debug session laya-rescore-drift (RoPE inv_freq fix 8e55e0bed)"
provides:
  - "aprender_decide::verify (VerifyPolicy, verify_run, pack_for_serving, verify_path, fixture_bytes, pack_fixture, check_variant, check_base, check_inputs, check_split, validate_probs, rescore, recompute_metrics, check_gate, VerifyError with exit_code)"
  - "aprender_decide::pack::load_checkpoint_for_scoring (back-office zero-shot scorer, never a Decider)"
  - "examples/pack_laya.rs pack / verify / inspect / pack-fixture; just laya-pack / laya-verify / laya-inspect / laya-pack-fixture (policy read from the contracts at run time, no override)"
  - "tests/laya_parity.rs full_model_reproduces_spike_025_fixture (LAYA_MODEL_DIR-gated)"
  - "tests/fail_closed_vectors.rs demo_vectors_are_refused_fail_closed (LAYA_MODEL_DIR + LAYA_FAIL_CLOSED_VECTORS=1)"
  - "FALSIFY-LAYA-GATE-010 (per-vector refusal); laya-finetune-gate-v1 1.3.0; concrete test: bindings for GATE-001/002/003/005/008, DECIDE-APR-012, LAYA-PARITY-001/003/005"
  - "models/decide/selftest/laya_tiny.apr (gitignored synthetic-fixture test artifact, golden 37d65159…, refused by verify) for 08-10"
affects: [08-10, 08-11, 08-12, laya-parity-v1, laya-finetune-gate-v1, decide-apr-v1, calibration spike]

actuals:
  tokens: 45250    # chars/4 over the realized plan diff aad9a7951..HEAD for crates/aprender-decide, contracts, justfile, Cargo.lock, 08-09-PLAN.md, deferred-items.md (181001 chars; the debug session's aprender-core layer.rs change excluded)
  tasks: 3
  commits: 9       # MEASURED: git rev-list --count aad9a7951..HEAD before this SUMMARY commit. It includes the three debug-session commits (8e55e0bed, eee452b2a, f3e9c8087) that landed between the halt and this continuation
plan_head_before: aad9a795107b88c1c5375fc1f953ae3a8eb58f5d

tech-stack:
  added: []
  patterns:
    - "The gate is decided on metrics RECOMPUTED in Rust from probability files that were first validated row by row and re-scored in Rust"
    - "pack_for_serving writes only after verification, atomically (temp file in the target dir + rename)"
    - "VerifyError::exit_code(): 3 for GateFailed, 2 for every other refusal (the scripts/laya_train convention)"
    - "A fail-closed vector carries a DECLARED refusal keyed by its full recipe_id; an undeclared refusal fails the test (the stop rule, as code)"
    - "Lib tests that landed GREEN because the code came first are proven by mutating each guarded check (11/11 killed)"

key-files:
  created:
    - crates/aprender-decide/src/verify.rs
    - crates/aprender-decide/src/verify/tests.rs
    - crates/aprender-decide/examples/pack_laya.rs
    - crates/aprender-decide/tests/laya_parity.rs
    - crates/aprender-decide/tests/fail_closed_vectors.rs
  modified:
    - crates/aprender-decide/src/pack.rs
    - crates/aprender-decide/src/lib.rs
    - crates/aprender-decide/Cargo.toml
    - Cargo.lock
    - justfile
    - contracts/laya-finetune-gate-v1.yaml
    - contracts/laya-parity-v1.yaml
    - contracts/decide-apr-v1.yaml
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-09-PLAN.md
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/deferred-items.md

key-decisions:
  - "USER DECISION option A (2026-09-26): pack_rescore_probs_abs stays 1e-5. early_stopping (3d4b91da) demonstrates GateFailed[ece_post], exit 3. fixed_epochs (d0f4e40d) is refused with RescoreDrift which=fine_tuned, exit 2, nothing written. FALSIFY-LAYA-GATE-010 states the per-vector refusal. No tolerance, threshold or demo value moved."
  - "USER APPROVAL: the stdio server's real-model leg is DEFERRED (D-ITEM-08-09-A); no artifact pack_laya verify accepts exists."
  - "USER APPROVAL: FALSIFY-LAYA-GATE-010 added; laya-finetune-gate-v1 1.2.0 -> 1.3.0 as pv diff suggests (minor). pv diff reports decide-apr-v1 and laya-parity-v1 identical (test: lines only), so they stay 1.0.0."
  - "Task 3's test is named demo_vectors_are_refused_fail_closed, not ..._on_the_ece_clause, because one vector is no longer refused on that clause."
  - "demo.fail_closed_rule prose was NOT edited (the plan forbids moving demo values). It still says both vectors fail on ece_post; D-ITEM-08-09-C records it for a user call."

requirements-completed: [D-06, D-07, D-11, D-12, D-17, D-19]

coverage:
  - id: D1
    description: "Rust verifier accept path on the tiny fixture: a production-variant copy under a TEST-ONLY permissive policy verifies, both re-scores within 1e-5, argmax 9/9, the recomputed ece_post matches the report"
    requirement: D-07
    verification:
      - kind: unit
        ref: "crates/aprender-decide/src/verify/tests.rs#tiny_verify_roundtrip"
        status: pass
    human_judgment: false
  - id: D2
    description: "Every verify refusal on the tiny fixture (26 verify::tests incl. both review forgeries, margin-only clause set, write-nothing, house ECE, fixture-only writer + golden, verify_path on the exact file); 11 check mutants killed"
    requirement: D-07
    verification:
      - kind: unit
        ref: "cargo test -p aprender-decide --lib verify:: -> 26 passed"
        status: pass
    human_judgment: false
  - id: D3
    description: "early_stopping fail-closed vector refused by just laya-pack on the recomputed ece_post clause alone, argmax 280/280, nothing written"
    requirement: D-19
    verification:
      - kind: integration
        ref: "just laya-pack models/decide/tweet-stance-16 data/decide/tweet-stance-16 <base> models/decide/fail-closed-check.apr -> rc 3, REFUSED GateFailed clauses=[ece_post]"
        status: pass
    human_judgment: false
  - id: D4
    description: "fixed_epochs fail-closed vector refused by just laya-pack with its documented refusal (option A), nothing written"
    requirement: D-19
    verification:
      - kind: integration
        ref: "just laya-pack models/decide/tweet-stance-16-fixed-epochs ... -> rc 2, REFUSED RescoreDrift which=fine_tuned row=59"
        status: pass
    human_judgment: false
  - id: D5
    description: "CLI controls: pack-fixture reproduces golden 37d65159…; laya-verify refuses the synthetic artifact (exit 2); laya-inspect is identity-only"
    requirement: D-11
    verification:
      - kind: integration
        ref: "just laya-pack-fixture / laya-verify / laya-inspect on crates/aprender-decide/tests/fixtures/laya_tiny"
        status: pass
    human_judgment: false
  - id: D6
    description: "Full-model parity on the English root against the spike-025 fixture (ids 14/14, argmax 14/14, max |dp| 3.841e-6, 512-token row truncated, 32 ladder blocks within bars)"
    requirement: D-17
    verification:
      - kind: integration
        ref: "LAYA_MODEL_DIR=<snapshot> LAYA_LADDER_BIN=<ladder> cargo test -p aprender-decide --release --test laya_parity full_model_reproduces_spike_025_fixture"
        status: pass
    human_judgment: false
  - id: D7
    description: "Both vectors refused by pack AND by verify on the exact bytes, each with its documented refusal (FALSIFY-LAYA-GATE-010), 2/2"
    requirement: D-07
    verification:
      - kind: integration
        ref: "LAYA_FAIL_CLOSED_VECTORS=1 LAYA_MODEL_DIR=<snapshot> cargo test -p aprender-decide --release --test fail_closed_vectors demo_vectors_are_refused_fail_closed"
        status: pass
    human_judgment: false
  - id: D8
    description: "Stdio server real-model leg (E2E-DECIDE-PMCP-001 real) — deferred as D-ITEM-08-09-A"
    verification: []
    human_judgment: true
    rationale: "Not run by design: no artifact pack_laya verify accepts exists (option 3). The user approved the deferral; the verifier should confirm D-ITEM-08-09-A's coverage argument."

duration: 45min
completed: 2026-09-27
status: complete
---

# Phase 8 Plan 09: Rust Gate Verifier and Fail-Closed Pack Summary

**`just laya-pack` and `pack_laya verify` decide the D-07 gate in Rust. The metrics are recomputed from probability files that are first re-scored row by row from the packed bytes and from the declared base. Both D-19 demo runs are refused and nothing is written. early_stopping refuses on the recomputed ece_post clause alone (exit 3). fixed_epochs refuses on its fine-tuned re-score (exit 2), as user decision option A documents. The real English root reproduces spike 025 through the `.apr` path: ids 14/14, max |dp| 3.841e-6.**

## Performance

- **Duration:** 45 min total. The first session took 21 min and halted at the tracer. This continuation took 24 min, from 2026-09-27T03:20:39Z to 03:44:51Z.
- **Tasks:** 3 of 3.
- **Files:** 5 created, 10 modified (see key-files).

## Resume from the halt

- The first execution stopped at the Task 1 stop rule (d4980e1b9). The early_stopping vector was refused with `RescoreDrift which=zero_shot row=49 1.028e-5`, not with `GateFailed[ece_post]`.
- Debug session `.planning/debug/resolved/laya-rescore-drift.md` found two causes:
  - **A port defect, now fixed.** Rust single-rounded RoPE `inv_freq` from f64, where torch double-rounds in f32. The fix is 8e55e0bed, pinned by `rope_inv_freq_is_torch_bitwise`.
  - **The bar sits at the fp32 noise floor.** On real checkpoints, the 1e-5 re-score bar is inside fp32 rounding noise.
- The user chose **option A**. The bar stays. early_stopping demonstrates the gate path, and fixed_epochs is recorded as a RescoreDrift refusal.
- This continuation re-ran the tracer on HEAD. The plan's expectations for fixed_epochs were amended in `1ed27cc95` (plan-only commit).

## Measured outcomes

### Task 1 tracer: early_stopping vector (re-run on HEAD, 142 s)

```
REFUSED GateFailed clauses=[ece_post] zs_macro_f1=0.34021732211112976 ft_macro_f1=0.445831298828125 margin=0.10561397671699524 ece_post=0.22240783274173737 rescore_max_abs=0.000006735324859619141 zs_rescore_max_abs=0.0000068247318267822266 argmax=280/280 packed_sha256=e397228192960328e974c657baad37daa3401f31beabfec07fb71995e387e217 (nothing written)
error: Recipe `laya-pack` failed with exit code 3
```

| metric | recomputed in Rust | reported (gate-report.json) |
|---|---|---|
| zs macro-F1 | 0.3402173 | 0.3402 |
| ft macro-F1 | 0.4458313 | 0.4458 |
| margin | 0.1056140 (passes, >= 0.05) | 0.1056 |
| ece_post | 0.2224078 (FAILS, > 0.10) | 0.2224 |

- The re-score maxima are 6.74e-6 for the fine-tuned model and 6.82e-6 for zero-shot. Both are within 1e-5, and argmax agrees on 280/280.
- `models/decide/fail-closed-check.apr` is absent, and the `models/decide` listing is byte-identical before and after.

### Task 2 control (d): fixed_epochs vector (85 s)

```
REFUSED RescoreDrift which=fine_tuned row=59 max_abs=0.00004667043685913086 (nothing written)
error: Recipe `laya-pack` failed with exit code 2
```

- This is option A's documented refusal, and nothing was written.
- The gate is not reached. The report records ece_post 0.3773, T clamped at 5.0 and margin 0.1299, and the first session's read-only diagnostic recomputed the same verdict (ece_post-only FAIL).

### Task 2 CLI controls (a)-(c)

```
PACKED-FIXTURE models/decide/selftest/laya_tiny.apr sha256=37d65159b2be0fa091aa840cd56c1a84b73c0bcd9e2df5906d1f8218f5448561 variant=synthetic-fixture (not deployable)
REFUSED SyntheticNotDeployable recipe variant "synthetic-fixture" is not deployable (nothing written)      # just laya-verify, exit 2
{"artifact_sha256":"37d65159…","base":"laya-tiny-synthetic@fixtures","embedded_gate":{"ece_post":0.008166154225667355,"margin":0.0,"pass":false},"labels":["shipping","billing","account"],"method":"laya","recipe_id":"7478c6f2…","schema_version":1,"variant":"synthetic-fixture"}   # just laya-inspect: no eligibility/policy field
```

### Task 3: full-model parity (release, aarch64 Apple M4, 14.7 s test time, load 2.26 s)

```
ids 14/14; argmax 14/14; max |dp| 3.841e-6 (bar 1e-5); max |dlogit| 2.146e-5 (bar 1e-4); truncated rows 1; ARCH aarch64
ladder emb: max_abs 9.537e-7 ... ladder layer27: max_abs 9.961e-2 rel_rms 2.145e-6 ... ladder final: max_abs 9.155e-5 ... ladder head1: rel_rms 6.384e-7
ladder: 32 blocks within bars
```

- The 512-token row reports `truncated == true` (D-12). The fixture sha256 `ebf9d94e…` equals the value laya-parity-v1 records.
- The largest per-row |dp| is 3.84e-6 (the 512-token billing row). Spike 025 measured 3.80e-6 on the same fixture.
- The ladder `final` block has little headroom: 9.155e-5 against `final_norm_abs` 1e-4, only 1.09x. See D-ITEM-08-09-B.

### Task 3: both vectors, by pack and by verify on the exact bytes (release, 413 s)

```
VECTOR d0f4e40d pack: REFUSED RescoreDrift which=fine_tuned row=59 max_abs=0.00004667043685913086
VECTOR d0f4e40d verify: REFUSED RescoreDrift which=fine_tuned row=59 max_abs=0.00004667043685913086
VECTOR 3d4b91da pack: REFUSED GateFailed clauses=[ece_post] ... argmax=280/280 packed_sha256=e397228192960328e974c657baad37daa3401f31beabfec07fb71995e387e217
VECTOR 3d4b91da verify: REFUSED GateFailed clauses=[ece_post] ... argmax=280/280 packed_sha256=e397228192960328e974c657baad37daa3401f31beabfec07fb71995e387e217
FAIL-CLOSED VECTORS REFUSED 2/2 (413 s, ARCH aarch64)
```

- For early_stopping, the verify-side sha equals the pack-side sha and the sha of the written scratch file.
- For fixed_epochs, the error carries no sha, so the test compares the row and bitwise max_abs across the two paths instead.
- The scratch TempDir is removed, and the `models/decide` listing is unchanged.

## Accomplishments

- **`aprender_decide::verify`** is the D-07 policy in a CI-tested library, in this order:
  - variant, then base (contract and on disk), then input hashes;
  - `check_split`, which mirrors `data.py`;
  - `validate_probs`;
  - the two 280-row re-scores;
  - the house macro-F1 and ECE;
  - the recomputed gate.
- **`pack_laya` subcommands**, each a thin wrapper that holds no policy:
  - `pack` writes only on accept.
  - `verify` is the only eligibility check, run on the exact file.
  - `inspect` reports identity only.
  - `pack-fixture` writes synthetic-fixture artifacts only.
  - The policy is read from the contracts at run time. There is no argument or environment override.
- **26 `verify::tests`** cover every plan-named `VerifyError`. `argmax_drift`, `verify_path_accepts_exact_file` and `verify_path_manifest_mismatch` go beyond the plan's list.
- **Real-weights evidence without a serving artifact:** full-model parity, plus both vectors refused on both paths (FALSIFY-LAYA-GATE-010).

## Task Commits

1. **Task 1 (tracer): verifier + `pack` + `just laya-pack`**: `d7031c319` (feat). The halt docs are `d4980e1b9` and `4ff5b02e4`. The tracer verify re-ran on HEAD and passed, with no code change needed after the 8e55e0bed RoPE fix.
2. **Plan amendment (option A):** `1ed27cc95` (docs).
3. **Task 2: subcommands, 25 new tests, bindings:** `9bdaebb95` (feat).
4. **Task 3: parity + vectors tests, GATE-010, deferrals:** `42bf7c47f` (test).

The debug session's own commits (8e55e0bed fix, eee452b2a, f3e9c8087) fall inside the measured range but are not this plan's.

## Verification run

| check | result |
|---|---|
| T1 verify 1: `verify::tests::tiny_verify_roundtrip` | PASS |
| T1 verify 2: `just laya-pack` early_stopping | PASS: rc 3, `clauses=[ece_post]`, argmax 280/280, ece_post within 1e-5 of 0.2224, margin >= 0.05, nothing written |
| T2 verify 1: `cargo test -p aprender-decide --lib verify::` | PASS: 26 ok (>= 23), including all 8 named tests |
| T2 verify 2: CLI controls (a)-(d) | PASS: golden sha; verify exit 2 SyntheticNotDeployable; inspect identity-only; fixed_epochs rc 2 `RescoreDrift which=fine_tuned`, nothing written |
| T2 verify 3: pv validate x3, strict binding, clippy | PASS: 0 errors each. Guard VACUOUS; the lifted copy resolved 658 refs, and only the two pre-existing contracts FAIL. `cargo clippy -p aprender-decide --all-targets --no-deps -- -D warnings` rc 0 |
| T3 verify 1: armed parity | PASS: `ids 14/14`, 1 passed, no SKIP |
| T3 verify 2: armed vectors | PASS: 2/2, all four VECTOR lines, listing unchanged |
| T3 verify 3: unarmed SKIPs, pv validate, greps, strict binding | PASS: both print SKIP and pass. The lifted copy resolved 661 refs. A mutated GATE-010 name was flagged dangling, so the resolver discriminates |
| Mutation check of the Task 2 lib tests | 11/11 mutants of guarded checks killed by their named test (variant, threshold, reported-metric, pass agreement, ece clause, split overlap, conflicting labels, slice group, rescore tol, fixture-only writer, write-after-verify order) |
| `cargo test -p aprender-decide --lib` | 93 passed, 0 failed |
| `cargo fmt -p aprender-decide -- --check` | rc 0 |
| acceptance greps | `expected_calibration_error_top_label` 3, `Average::Macro` 2 (verify.rs); `pack_rescore_probs_abs` 2, `model_safetensors_sha256` 2, `gate_policy_ok` 0 (pack_laya.rs) |

## Decisions Made

- The option A and approval decisions are recorded in `key-decisions`.
- **Per-vector expectations live in the test, keyed by the full recipe_id.** The contract's vector strings carry no refusal field, and parsing FALSIFY prose would be fragile. A contract vector with no entry fails the test, so a new vector cannot pass without a declared refusal.
- **The parity test builds the model through `pack::load_checkpoint_for_scoring`.** That is the verifier's own F16 `.apr` path (OPS-03), used instead of a second loader. It binds record 0's `department` choice task, because the rows carry their own questions.
- **The ladder rung checks head0 and head1 against `per_layer_rel_rms`.** The contract names no separate head bar.
- **`json!` object literals were replaced by an explicit `Map` builder.** `json!` expands to `unwrap`, which this workspace bans.
- **Earlier choices carried over from the first session:**
  - `VerifyPolicy.calibration_slice_min_per_class` is read from the contract.
  - `PackInputs` carries the probability-file bytes.
  - `check_gate` returns the failed clauses.
  - `rescore` takes a classify closure.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `PackError::ReportHashMismatch` fires before `check_inputs`** (first session, `d7031c319`)
- It is mapped to `InputHashMismatch` naming the file. `input_hash_mismatch` asserts `file: eval_jsonl`.

**2. [Rule 2 - Missing critical] The slice check needs the contract's per-class minimum** (first session, `d7031c319`)

**3. [Rule 3 - Blocking] clippy `disallowed_methods` on `serde_json::json!` objects** (Task 2, `9bdaebb95`)
- **Issue:** the object form expands to `.unwrap()`.
- **Fix:** an `obj()` Map builder in the example, and explicit Maps in the tests.

**4. [User decision - option A] fixed_epochs refuses with RescoreDrift, not `clauses=[ece_post]`** (plan amended in `1ed27cc95`)
- **Plan changes:** Task 2 (d), Task 3's test and its name, GATE-010, the verifies and the criteria now assert the per-vector refusal.
- **Verification:** the stop rule still applies to any other refusal. The test panics with `STOP RULE: an undocumented refusal`.

**5. [Scope] Three tests beyond the plan's 22 behaviours** (Task 2)
- The new tests are `argmax_drift`, `verify_path_accepts_exact_file` and `verify_path_manifest_mismatch`.
- `ArgmaxDrift` is a plan-named variant with no behaviour test, and Task 3 relies on `verify_path`.

**6. [TDD note] Task 2's lib tests landed GREEN.**
- `tdd="true"`, but the library was already written in Task 1, so there was no RED phase.
- In place of RED, each test's discriminating power was shown by mutating its guarded check: 11/11 were killed.
- `workflow.tdd_mode` is false, so the TDD gate is advisory.

---

**Total deviations:** 3 auto-fixed (2 blocking, 1 missing critical), 1 user-decided scope change, 1 scope addition and 1 TDD note. **Impact:** the plan is complete. The fixed_epochs refusal differs from the original expectation by decision, and it is still fail-closed.

## Issues Encountered

- The rtk hook rewrites `cargo test`, `git diff` and `grep -h` output. Test results were read through `rtk proxy`, and the diff size was measured with `/usr/bin/git`.
- `cargo fmt` reflowed the two new test files AFTER the armed release runs compiled them. The change is formatting only, so the measured runs used semantically identical source. The unarmed runs and clippy used the formatted files.
- The main checkout contains `models/decide/tweet-stance-16-var/`, which this plan did not create. It was not touched.

## Known Stubs

None.

## Threat Flags

None. No new surface outside the plan's threat model. `verify` / `inspect` / `pack-fixture` are the planned T-08-09-06/08 mitigations.

## Deferred (deferred-items.md "From plan 08-09")

- **D-ITEM-08-09-A:** the stdio real-model leg is deferred. It lists what covers the gap, what re-arms the leg, and two alternatives that each need a user decision.
- **D-ITEM-08-09-B:** the 1e-5 bar is fragile (about 1.5x headroom; a 1-ULP codegen change consumed it once; x86_64 never measured), and the ladder `final` block has 1.09x headroom. This is queued in `.planning/todos/pending/spike-laya-calibration-slice-and-temperature-cap.md` (section "Added 2026-09-26"). Not acted on.
- **D-ITEM-08-09-C:** `demo.fail_closed_rule` prose still says both vectors fail on ece_post. It needs a one-line user call.
- **D-ITEM-08-09-D:** D-ITEM-08-01-A was re-measured: 658 refs, then 661, with no Phase 8 contract dangling.

## User Setup Required

None.

## Next Phase Readiness

- **08-10:**
  - `models/decide/selftest/laya_tiny.apr` exists (gitignored, golden `37d65159…`), and `just laya-verify` refuses it.
  - `just laya-verify` is the eligibility check the deploy recipes use. It exits 0 only on accept.
  - The accept path runs in CI on the tiny fixture only.
- **D-18 deploy stays deferred.** No real artifact is eligible. The first one needs the calibration spike and then a declared run that passes. Per D-ITEM-08-09-B, such a run may still hit the 1e-5 re-score bar on a high-temperature checkpoint, so the spike must settle the bar question before that run is read.
- **08-12:** `laya_parity` and `fail_closed_vectors` are dark in CI: they compile and SKIP. Wiring them is 08-12's decision.

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-27*

## Self-Check: PASSED

- Created files exist: verify.rs, verify/tests.rs, examples/pack_laya.rs, tests/laya_parity.rs, tests/fail_closed_vectors.rs.
- Commits d7031c319, 1ed27cc95, 9bdaebb95 and 42bf7c47f are in history. None of the task commits deletes a tracked file.
- `models/decide/fail-closed-check.apr` is absent. The vector run dirs were not modified: only the scratch TempDirs and the gitignored `models/decide/selftest/` were written.
