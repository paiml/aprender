---
phase: 03-faithful-two-stage-trainer-and-head
plan: 10
subsystem: training
tags: [setfit, trn-01, trn-06, safe-03, trybuild, compile-fail, cross-process, reproducibility, mutation, complexity, closing-audit]

requires:
  - phase: 03-09
    provides: SelectionLock / CanonicalTestToken / ValidationEvaluation — the three private-field types cases 3, 6 and 7 attack
  - phase: 03-08
    provides: the eleven public reproducibility accessors on SetFitRun<ArtifactReloadedAndVerified>, the sealed SetFitCodec, SerdeJsonCodec
  - phase: 03-06
    provides: the calibrated-regime threshold table, without which tune_encoder refuses the fixture run and no digest exists to compare
  - phase: 03-05
    provides: the recorded consumed-pair and batch-boundary digests, absorbed AT consumption; the conformance-fixtures dev-dependency edge
  - phase: 03-03
    provides: the twelve-knob SetFitTrainConfig and the setfit feature on aprender-train
  - phase: 03-02
    provides: tests/gemm_thread_determinism.rs — the current_exe + exact-test-name subprocess pattern reused here
  - phase: 02
    provides: PreparedDataset::from_labeled_rows, FewShotSelector, PairSampler, LabeledPair; the trybuild harness pattern from aprender-contrastive-data
provides:
  - crates/aprender-train/tests/ui.rs + seven compile-fail cases — TRN-01/TRN-07/SAFE-03's legality claims as rustc diagnostics with committed snapshots
  - crates/aprender-train/tests/setfit_repro.rs — the FIRST out-of-crate consumer of the whole SetFit lifecycle; in-process, cross-process and recorded-vs-expected gates
  - SetFitRun::batch_boundary_digest() — the recorded boundary digest's first public reader
  - Makefile setfit-repro-inproc (tier2), setfit-repro-crossproc (tier3), gemm-thread-determinism (tier3)
  - the Phase 3 surface under the project's cyclomatic ceiling of 10 (four functions decomposed)
affects:
  - Phase 4 (OPS): tests/setfit_repro.rs is the worked example of driving the lifecycle from outside the crate — the shape an apr subcommand needs
  - Phase 5 (EVAL): TRN-06 is now a measured cross-process claim, so an eval report may cite it rather than assert it

tech-stack:
  added:
    - trybuild (dev-dependency on aprender-train; already a workspace dep)
  patterns:
    - "One illegal expression per compile-fail case: rustc's privacy pass runs AFTER type checking, so two claims that fail in different passes cannot share a snapshot — the earlier pass wins and the other claim vanishes from its own evidence"
    - "The harness is #![cfg(feature = ...)]-gated so a default build runs ZERO cases instead of N passing ones for the wrong reason"
    - "Reproducibility and correctness are SEPARATE tests with separate failure messages: a fused check cannot tell reproducibly-wrong from correct"
    - "Prove the blindness rather than asserting it: one perturbation makes the in-process gate green and the cross-process gate red on the same tree"
    - "Decomposing for a complexity ceiling can MOVE the complexity instead of dissolving it — re-measure the extracted helper, do not assume the caller's drop is the whole story"

key-files:
  created:
    - crates/aprender-train/tests/ui.rs
    - crates/aprender-train/tests/ui/setfit_direct_state_construction.rs (+ .stderr)
    - crates/aprender-train/tests/ui/setfit_fit_head_before_tune.rs (+ .stderr)
    - crates/aprender-train/tests/ui/setfit_token_without_lock.rs (+ .stderr)
    - crates/aprender-train/tests/ui/setfit_pairs_into_fit_head.rs (+ .stderr)
    - crates/aprender-train/tests/ui/setfit_probe_claims_setfit.rs (+ .stderr)
    - crates/aprender-train/tests/ui/setfit_metric_value_asserted.rs (+ .stderr)
    - crates/aprender-train/tests/ui/setfit_external_codec_impl.rs (+ .stderr)
    - crates/aprender-train/tests/setfit_repro.rs
  modified:
    - crates/aprender-train/Cargo.toml
    - crates/aprender-train/src/train/setfit/mod.rs
    - crates/aprender-train/src/train/setfit/tune.rs
    - crates/aprender-train/src/train/setfit/verify.rs
    - crates/aprender-train/src/train/setfit/verify_tests.rs
    - crates/aprender-core/src/classification/multinomial.rs
    - Makefile
    - .planning/REQUIREMENTS.md

key-decisions:
  - "Cases 1 and 6 carry ONE illegal expression each, not two: rustc's privacy pass runs after typeck, so the first draft's E0616/E0599 aborted the compile before E0451 was emitted and the blessed snapshots contained only the weaker half — the structural claim was silently absent from its own evidence"
  - "The trybuild harness is feature-gated: without the gate a default build finds seven cases that fail because train::setfit does not exist and reports SEVEN PASSING compile-fail tests, which is the vacuous proof the suite exists to rule out (controlled: 0 cases without the feature, 7 with)"
  - "batch_boundary_digest() was ADDED to the public accessor block rather than digesting batch_boundaries() in the test: a digest computed from an accessor's output is a digest of a DESCRIPTION, and the composite hash's whole claim is that it compares RECORDED execution"
  - "The composite comparison is over RECORDED digests and the replay check is a SEPARATE test, because a run that consumed the wrong order reproduces it perfectly in every process forever — cross-process equality is structurally silent about correctness"
  - "The acceptance criterion's own `grep -c 'tee'` is UNFIT and was measured so: it matched the word guaran-tee in two failure messages and reported a violation in recipes containing no pipe. The durable pattern is pipe-aware (`\\| *tee`); the prose was reworded so the naive form also reads clean, and the Makefile records why"
  - "cargo-mutants could not be run to completion and this is reported as a SHORTFALL, not waived: --in-place is mutually exclusive with --jobs in 25.3.1, and the plan's mandated --timeout 20 kills the BASELINE at 20.05 s because aprender-core's test binary has not finished LINKING by then"
  - "The 24 known-red aprender-train names are diffed, not eyeballed: the workspace run's gpu:: failures are byte-identical to known-red-baseline.md (`diff` rc=0)"

patterns-established:
  - "A tooling limit is reported with the exact refusal message and the exact elapsed-vs-budget numbers, so the next plan fixes the invocation instead of rediscovering the wall"
  - "A decomposition claims behaviour preservation only with a before/after test count on the same filter (7629 -> 7629)"
  - "Running a test suite can DELETE tracked files (insta .snap.new); the post-run porcelain check catches it and `git checkout -- <specific paths>` restores it"

requirements-completed: [TRN-01, TRN-02, TRN-03, TRN-04, TRN-06, SAFE-03]

metrics:
  duration: ~4h30m
  completed: 2026-08-11
---

# Phase 3 Plan 10: Verification Spine — Compile-Fail Legality and Cross-Process Reproducibility Summary

TRN-01's legality is now seven rustc diagnostics with reviewed snapshots, TRN-06's reproducibility is a measured cross-process hash equality at two proven-different rayon pool sizes over RECORDED execution digests, and the phase's closing audit is booked with its two real shortfalls stated as shortfalls.

## Commits

| # | Hash | Subject |
|---|------|---------|
| 1 | `b7503075a` | seven compile-fail proofs that the illegal lifecycle is inexpressible |
| 2 | `09af9ecaa` | make the two-clean-runs claim cross-process, and separate reproducible from correct |
| 3 | `c04674039` | clear the cyclomatic ceiling on the Phase 3 surface, measured not asserted |

## Task 1 — The seven compile-fail proofs (`b7503075a`)

| Case | Diagnostic | Names |
|------|-----------|-------|
| `setfit_direct_state_construction` | E0451 six private fields | `SetFitRun` |
| `setfit_fit_head_before_tune` | E0599 method not found | `SetFitRun<Prepared>`, `fit_head` |
| `setfit_token_without_lock` | E0451 four private fields | `CanonicalTestToken` |
| `setfit_pairs_into_fit_head` | E0061 takes 0 args, 1 supplied | `fit_head`, `Vec<LabeledPair>` |
| `setfit_probe_claims_setfit` | E0277 no `From<FrozenProbeRun>` | `FrozenProbeRun`, `SetFitRun<EncoderTuned>` |
| `setfit_metric_value_asserted` | E0451 six private fields | `ValidationEvaluation` |
| `setfit_external_codec_impl` | E0277 unsatisfied supertrait | `SetFitCodec`, `verify::sealed::Sealed` |

`cargo test -p aprender-train --test ui --features setfit` → **rc=0, 7 compile-fail cases**.

**The named-types rule, mechanised.** Every one of the seven `.stderr` files matches
`SetFitRun|fit_head|CanonicalTestToken|FrozenProbeRun|ValidationEvaluation|SetFitCodec|Sealed`
(measured per file with `awk`; `grep` is rewritten by the `rtk` hook into an uncountable summary).
Case 7's snapshot names the private `Sealed` explicitly and rustc even spells out the mechanism:
`` `SetFitCodec` is a "sealed trait", because to implement it you also need to implement
`entrenar::train::setfit::verify::sealed::Sealed`, which is not accessible ``.

**Two cases were CORRECTED rather than blessed — the plan's most substantive finding here.**
Cases 1 and 6 each first attempted TWO illegal expressions (a struct literal expecting E0451 plus a
private-field read / a `::new(value)` call). Only ONE diagnostic reached each snapshot, and it was
the weaker one: **rustc's privacy pass runs after type checking**, so a typeck error (E0616, E0599)
aborts the compile before E0451 is ever emitted. Both snapshots would have shipped WITHOUT the
structural claim — "the fields are private, so no literal works" — that the case exists to make.
Each case now carries the single load-bearing attempt with the pass-ordering measurement in its
header. This is the vacuous-snapshot failure mode (T-3-31) arriving in a form the plan did not
anticipate: not a syntax error, but a *correct* diagnostic for the *other* half of the claim.

**Non-vacuity of the snapshot check itself, measured.** Rewriting `verify::sealed::Sealed` to
`verify::sealed::INDUCED_RED_PROBE` in one `.stderr` gives **rc=101**; restoring the file returns it
to byte-identical (`sha256 56fb8892...`, compared against the pre-probe copy).

**Vacuity control on the harness.** `cargo test -p aprender-train --test ui` (no `setfit`) →
**rc=0 with 0 tests**. `#![cfg(feature = "setfit")]` is what makes that 0 rather than 7 passing
compile-fail tests against an absent module.

**Public-API-only, verified.** 0 occurrences of `pub(crate)`, `doc(hidden)`, `test_fixtures` or
`for_tests` across all seven case files. All 14 files tracked
(`git ls-files --others --exclude-standard crates/aprender-train/tests/ui/` empty; `git ls-files`
counts 14).

## Task 2 — The two-clean-runs gates (`09af9ecaa`)

`crates/aprender-train/tests/setfit_repro.rs` is the **first out-of-crate consumer of the whole
SetFit lifecycle**: `from_labeled_rows -> FewShotSelector::select -> SetFitTrainConfig::new ->
prepare -> tune_encoder -> fit_head -> verify_artifact(&SerdeJsonCodec)`, built entirely from public
API, at the calibrated cell (`root_seed 1`, 8 shots, 1 epoch, batch 4, budget 12).

**The fixture was reached through an existing `pub fn` and NO production backdoor was added.**
`SetFitMiniLm::from_slice_fixture` is `pub` but `#[cfg(feature = "conformance-fixtures")]`
(aprender-core `setfit/mod.rs:305`) — its *visibility* was never the question, its *reachability*
was. 03-05 already solved it with a **dev-dependency** edge (`aprender-train/Cargo.toml`:
`aprender = { path = ..., features = ["conformance-fixtures"] }`), which enables the feature for
test builds only; `make setfit-feature-matrix` traverses `-e normal`, so the shipped dependency set
is unchanged. Nothing was widened for this file. The one accessor added —
`batch_boundary_digest()` — is a public read-only accessor in the counted block, because the
recorded boundary digest had **no public reader at all**; the alternative was digesting
`batch_boundaries()`'s output, i.e. a digest of a description.

### The three gates

| Test | rc | What it claims |
|------|----|----------------|
| `setfit_repro_in_process_two_runs_agree` | 0 | D-16's fast half — structurally blind |
| `setfit_repro_cross_process` | 0 | AUTHORITATIVE — two fresh processes, pools 1 and 3 |
| `setfit_repro_recorded_matches_expected_replay` | 0 | the run consumed the INTENDED order |
| `setfit_repro_child` | 0 | no-op unless `SETFIT_REPRO_CHILD=1` |

`cargo test -p aprender-train --test setfit_repro --features setfit` → **rc=0, 4 passed**.

**Mechanism engaged, observed not requested.** With `--nocapture`:

```
child RAYON_NUM_THREADS=1 -> THREADS=1 ARTIFACT=08ed601d1298dc04775da57e32344186a8e7035f090c90d8d6eb0b61308eabb6
child RAYON_NUM_THREADS=3 -> THREADS=3 ARTIFACT=08ed601d1298dc04775da57e32344186a8e7035f090c90d8d6eb0b61308eabb6
```

`THREADS` is `rayon::current_num_threads()` inside the child — what rayon BUILT, not what the env
var asked for (CLAUDE.md rule 2). The parent asserts the observed values differ **before** it
compares any hash, so an agreement cannot be one pool size measured twice. `POOL_SIZES` is the
literal `[1, 3]`; the only `current_num_threads()` call in the file is the child's report line
(measured: 1 occurrence, line 326) — never a spawn size derived from the host.

**Ten components compared**, all from public accessors, all RECORDED: `SELECTION` (selection
semantic hash), `PAIRORDER` (recorded consumed-pair digest), `BATCHES` (count:recorded
boundary digest), `STEPS`, `LOSSTRACE`, `EVIDENCE`, `LEDGER`, `REGISTRY`, `ARTIFACT`,
`PREDICTIONS` (SHA-256 over the reloaded model's probe answers, ids/labels length-prefixed and
floats as LE bit patterns).

**The separate replay check.** `setfit_repro_recorded_matches_expected_replay` recomputes the
expected pair order from the public `epoch_pair_order(seed, epoch, n)`, the batch windows from the
consecutive-window rule, and the endpoints from a freshly built `PairSampler`, absorbing them in
the field order 03-05 contracted — then compares against the RECORDED digests. Its doc comment
states why it is separate: cross-process equality proves the run is reproducible and is silent
about whether the reproduced execution was the intended one.

### Induced-red evidence, five observations

| # | Perturbation | Target | rc | Failure observed |
|---|--------------|--------|----|------------------|
| A | `POOL_SIZES = [1, 1]` | `setfit-repro-crossproc` | **2** | "did not run at two DISTINCT pool sizes" branch |
| B | probe digest absorbs `current_num_threads()` | `setfit-repro-crossproc` | **2** | `probe predictions: THREADS=1 gave b8629841… but THREADS=3 gave c7988a99… — the pipeline DEPENDS ON THE RAYON POOL SIZE` |
| B′ | **same tree as B** | `setfit-repro-inproc` | **0** | **nothing — the in-process gate cannot see it** |
| C | per-call `AtomicU64` in the probe digest | `setfit-repro-inproc` | **2** (recipe rc=101) | `probe predictions` |
| D | `chunk.iter().rev()` in the replay | `--test setfit_repro recorded_matches` | **101** | "the run is a faithful function of the WRONG consumption order (cross-process equality cannot see this)" |
| E | `hash_f32` absorbs `pool_threads()` | `gemm-thread-determinism` | **2** | `shape 80x384x384: THREADS=1 gave 5f657198… but THREADS=2 gave de7ab795…` |

**Observation B′ is D-16's rationale as a number.** The same defect, on the same tree, is invisible
to the in-process gate (rc=0) and caught by the cross-process gate (rc=2). D-16 argued the
in-process form is "structurally blind"; this measures it. Every perturbation was reverted and each
file confirmed byte-identical by `shasum -a 256` (`setfit_repro.rs` = `b74761a1…`,
`gemm_thread_determinism.rs` = `e4656661…`).

### Make targets and the tier split (W-10)

| Target | Tier | Standalone rc |
|--------|------|---------------|
| `setfit-repro-inproc` | **tier2** | 0 |
| `setfit-repro-crossproc` | **tier3** | 0 |
| `gemm-thread-determinism` | **tier3** (deferred from 03-02, now wired) | 0 |

D-16 splits ONE gate across two tiers and both halves are wired; a tier3-only wiring would have
left the fast signal in a file no tier invokes.

**Every recipe reads `$?` on the line after the redirect. Case table, re-run after the fix:**

| Recipe | lines | `tee` substring | `\| *tee` | direct `rc=$$?` |
|--------|-------|-----------------|-----------|-----------------|
| `setfit-repro-inproc` | 14 | 0 | 0 | 1 |
| `setfit-repro-crossproc` | 15 | 0 | 0 | 1 |
| `gemm-thread-determinism` | 13 | 0 | 0 | 1 |

**The plan's own criterion regex was wrong, and it was measured rather than reasoned about.** The
first run of `awk '/^setfit-repro-crossproc:/,/^$/' | grep -c 'tee'` returned **1** for two of the
three recipes. Both hits were the substring inside the word **"guaran-tee"** in a failure message —
recipes that contain no pipe at all. This is CLAUDE.md rule 7 exactly: the pattern was caught by
re-running its case table, not by re-reading it. Both messages now say "claim" instead, so the naive
form also reads clean, and the Makefile records that the pattern to REUSE is the pipe-aware one.

**tier2 caveat, recorded not papered over.** tier2 as a whole is red on arm64 from pre-existing
clippy errors (D-ITEM-02) and its headline `cargo test --lib` step runs zero tests (D-ITEM-03).
Neither is this gate's status: `setfit-repro-inproc` was verified STANDALONE with its rc captured
directly (rc=0), and its recipe says so in the failure message it prints.

## Task 3 — Closing audit, complexity, and two honest shortfalls

### Free-disk headroom (W-05)

| When | Available | Capacity |
|------|-----------|----------|
| Before | **105 GiB** | 89% |
| After | **74 GiB** | 92% |

Above the 60 GiB threshold throughout; no `cargo clean` was needed and no ENOSPC occurred. The
worktree began with no `target/` at all, so a gitignored `.cargo/config.toml` points `target-dir` at
the main checkout's existing tree — registry artifacts are reused and workspace members rebuild
under their own metadata hashes.

### Closing audit — every rc captured directly, no pipes

| Check | Command | rc | Result |
|-------|---------|----|--------|
| train lib | `cargo test -p aprender-train --lib --features setfit` | **101** | 7839 passed, **24 failed** — see known-red |
| train lib (filtered) | `… -- --skip gpu:: --skip prune::snapshot_tests` | **0** | **7629 passed**, 15 ignored |
| core lib | `cargo test -p aprender-core --lib --features setfit` | **0** | **14285 passed**, 2 ignored |
| ui | `cargo test -p aprender-train --test ui --features setfit` | **0** | 7 compile-fail cases |
| repro | `cargo test -p aprender-train --test setfit_repro --features setfit` | **0** | 4 passed |
| gemm | `cargo test -p aprender-core --test gemm_thread_determinism` | **0** | 2 passed |
| FULL SUITE | `cargo test --workspace --lib --exclude aprender-profile --no-fail-fast` | **101** | **175 failures in 8 crates** — see below |
| clippy train | `cargo clippy -p aprender-train --lib --features setfit --no-deps -- -D warnings` | **0** | |
| clippy core | `cargo clippy -p aprender-core --lib --features setfit --no-deps -- -D warnings` | **101** | 1 pre-existing arm64 error — see below |
| workspace check | `cargo check --workspace --exclude aprender-profile` | **0** | |
| contract audit | `make contract-audit-phase3` | **0** | 5/5 and 16/16 equations bound and implemented |
| feature matrix | `make setfit-feature-matrix` | **0** | PASSED |
| repro (tier3) | `make setfit-repro-crossproc` | **0** | |
| gemm (tier3) | `make gemm-thread-determinism` | **0** | |
| repro (tier2) | `make setfit-repro-inproc` | **0** | |
| pv lint | `pv lint contracts/multinomial-head-v1.yaml` | **0** | 0 errors, 0 warnings, PASS |
| pv lint | `pv lint contracts/setfit-train-lifecycle-v1.yaml` | **0** | 0 errors, 0 warnings, PASS |
| fmt | `cargo fmt --check -p aprender-train -p aprender-core` | **0** | |
| PMAT SATD | `pmat analyze satd` × 3 scopes | **0**,**0**,**0** | **0 violations** in all three |
| PMAT complexity | `pmat analyze complexity --max-cyclomatic 10` × 3 scopes | **0** | **0 errors** after the decomposition |
| known-red package | `cargo package -p aprender-train --no-verify` | **101** | expected — see below |

### The full suite cannot exit 0 on this host, and it is not this plan's doing

`cargo test --workspace --lib --exclude aprender-profile --no-fail-fast` — VALIDATION.md's declared
full suite — reports **175 failing tests across 8 crates**:

| Package | passed | failed | dominant prefix |
|---------|--------|--------|-----------------|
| aprender-gpu | 2468 | **94** | `driver::` (90) |
| aprender-serve | 15389 | **51** | `apr_transformer::` (49) |
| **aprender-train** | 7612 | **21** | `gpu::` (21) |
| aprender-orchestrate | 6542 | 3 | |
| aprender-cgp | 114 | 2 | `/proc/meminfo`, a load-sensitive FLOPS floor |
| aprender-zram-core | 377 | 2 | `lz4::` |
| aprender-compute | 3398 | 1 | `blis::` |
| aprender-test-lib | 6203 | 1 | |

**Control.** `git diff --name-only <base> HEAD` lists 21 files, ALL under `crates/aprender-train/`
plus `Makefile`, `Cargo.lock` and (after Task 3) `crates/aprender-core/src/classification/
multinomial.rs`. Seven of the eight failing crates are untouched. For the eighth, the 21
`aprender-train` failures were **diffed, not eyeballed**, against `known-red-baseline.md`'s 21
`gpu::` names: `diff` **rc=0, identical**. (`prune::snapshot_tests` does not appear because the
workspace run builds without `--features setfit` and selects a different set.)

So: 154 of 175 failures are in crates this phase does not own, and the remaining 21 are the
pre-established baseline. **The plan's "full suite exits 0" criterion is not satisfiable on this
host and is reported as a shortfall, not as a pass.** The suite was RUN, which is what review fix 5
asked for, and the result is now on the record with per-crate attribution — the first time this
phase has that number.

Two of the aprender-cgp failures deserve naming because they will recur:
`profilers::system::tests::test_read_system_memory_total_mb` panics on
`/proc/meminfo MemTotal should be readable` (there is no `/proc` on Darwin), and
`analysis::roofline::tests::test_empirical_flops_positive` panics on
`Single-core FLOPS 0.8 GFLOP/s suspiciously low` — a wall-clock threshold measured while this
session was compiling, i.e. load-sensitive by construction.

### The one core clippy error, re-measured against an existing deferred item

`cargo clippy -p aprender-core --lib --features setfit -- -D warnings` fails with exactly one
error: `unreachable expression` at `crates/aprender-core/src/demo/reliable/performance.rs:126`.
`cpu_backend_name()` returns unconditionally inside `#[cfg(target_arch = "aarch64")]`, so the
trailing `"Scalar".to_string()` is dead on arm64 and live on the x64 CI. `--no-deps` does not help
because the error is in aprender-core itself.

This is **already a recorded deferred item** with its own two-sided control (03-08 measured
`git diff … -- crates/aprender-core/src/demo/` as EMPTY). My independent control agrees: this plan's
only aprender-core change is `classification/multinomial.rs`. Not fixed — out of scope under the
executor's scope boundary, and it belongs to whoever owns `demo/reliable/`.

### Complexity: four functions decomposed, ceiling cleared (`c04674039`)

`pmat analyze complexity --max-cyclomatic 10` over the three Phase 3 scopes reported four breaches.
The plan says decompose rather than record an exception, so they were decomposed:

| Function | Before | After | Split into |
|----------|--------|-------|-----------|
| `validate_evidence` (tune.rs) | **20** | 10 | `worst_failing_gated_parameter` + `gated_row_failure` |
| `validate_fit_inputs` (multinomial.rs) | **20** | — | shape / values / class-indices / regularization rungs |
| `compare_probes` (verify.rs) | **17** | — | row-counts / strings / embeddings / probabilities rungs |
| `run_batch` (tune.rs) | **13** | 9 | `clear_grads_then_tape`, `open_batch`, `pull_batch_pairs` |

Final: **all three scopes report 0 errors and Max Cyclomatic 0**; SATD is 0 in all three.

**The first cut of `validate_evidence` was wrong in the instructive way, and the code records it.**
It moved the caller from 20 to 10 while the extracted helper landed at **12** — the complexity had
MOVED, not dissolved, which is what decomposition-by-cut-and-paste produces. Only re-measuring the
*helper* caught it; the caller's drop looked like success. The second cut takes the per-row
predicate out of the accumulation loop and reduces both.

Every split is order-preserving and the order is documented where it is now decided, because in each
case the sequence is behaviour:

- `validate_fit_inputs` — several of the sixteen FALSIFY inputs are invalid on more than one rung (a
  ragged row of NaNs is both `RaggedRow` and `NanFeature`), so reordering changes WHICH error an
  input that is still correctly rejected reports.
- `compare_probes` — the row-count rung must stay first: every rung below walks with `zip`, which
  stops at the shorter side, so a desynchronised probe would pass on a PARTIAL comparison.
- `run_batch` — stages were lifted WHOLE, not sliced. `(a)`-`(m)` is a pinned order the order-pin
  test asserts on; fragmenting a pinned sequence makes it harder to audit, which is the opposite of
  what the ceiling is for.

**Behaviour control:** the same filtered command reports **7629 passed before and 7629 passed
after**; `cargo test -p aprender-core --lib multinomial` 62 passed; `tune_` 47, `evidence_` 20,
`negative_` 41, all rc=0.

### SHORTFALL 1 — the scoped mutation run did not complete

**The ≥85% adjusted-score criterion is NOT met. It is reported as a shortfall, not waived** — the
plan explicitly forbids waiving it by narrative, and no score is claimed.

**Mutant inventory (measured, `cargo mutants --list`):**

| Scope | Mutants |
|-------|---------|
| `crates/aprender-train/src/train/setfit/**` | **921** |
| `crates/aprender-core/src/classification/multinomial.rs` | **181** |
| `crates/aprender-core/src/setfit/dropout_rng.rs` | **79** |
| **Total** | **1181** |

**Two tooling blockers, both measured with the exact message:**

1. **`--in-place` is mutually exclusive with `--jobs` in cargo-mutants 25.3.1:**
   `error: the argument '--in-place' cannot be used with '--jobs <JOBS>'`. The plan mandates
   `--in-place` (from 02-08); parallelism therefore requires the tree-copy mode.
2. **The plan's mandated `--timeout 20` kills the BASELINE before any mutant runs.** Both attempts
   ended with `*** result: Timeout` and
   `ERROR cargo test failed in an unmutated tree, so no mutants were tested`. The debug log gives
   the exact numbers: `elapsed=20.050001083s → outcome=Timeout`. `--build-timeout 1800` does not
   help, because the phase being killed is the TEST phase and aprender-core's 14285-test binary has
   not finished **linking** inside 20 s.

   This is CLAUDE.md rule 4 in miniature: `--timeout 20` was measured on
   `aprender-contrastive-data` at 02-08, and **extending its scope to aprender-core required
   re-measuring in the new scope**. The old measurement did not transfer.

**Projected wall time, from this session's own measurements:** aprender-core lib build 53 s +
lib test run 82.7 s = **~136 s per mutant** → 1181 × 136 s ≈ **44.6 hours** single-job. That exceeds
CLAUDE.md's ">1 hr compute" escalation threshold by more than an order of magnitude, so it is
surfaced rather than attempted.

**What unblocks it:** a larger `--timeout` (≥120 s for aprender-core, to cover the link), tree-copy
mode with `-j` for parallelism, and a compute budget decision — this is a dedicated run, not a tail
step of a plan. The `mutants.out/` artifacts are gitignored (`.gitignore:26`).

**Source integrity after `--in-place`:** the tree is clean (`rtk proxy git status --porcelain` shows
only `.planning/REQUIREMENTS.md`) and `cargo test -p aprender-core --lib --features setfit
dropout_rng` still reports 15 passed, so no mutation was left behind.

### SHORTFALL 2 — `make coverage` deferred

Not run. It is the single heaviest command in the phase (`cargo llvm-cov` across the workspace,
with the mold linker disabled) and it needs the cargo lock that the mutation attempts and the
full-suite run held for the remainder of the budget. **What unblocks it:** the same dedicated
compute window as the mutation run; the two should be scheduled together since both need an
uncontended target dir. `COV_FLOOR` remains 88% and the last measured figure is the 88.78% CLAUDE.md
records from 2026-07-29.

Per CLAUDE.md's coverage/contract co-evolution rule, no coverage work was done, so no contract
obligations were weakened either — the two contracts touched by this phase both `pv lint` clean.

### Known-red re-measured

`cargo package -p aprender-train --no-verify` → **rc=101**:

```
error: failed to prepare local package for uploading
Caused by:
  no matching package named `aprender-contrastive-data` found
  location searched: crates.io index
  required by package `aprender-train v0.63.0`
```

Expected (Pitfall 9): `--no-verify` skips the packaged-crate BUILD, not the manifest resolution that
rewrites the path dep into a registry dep. Unchanged by this plan.

## Requirements

**Booked complete — six:**

| ID | Evidence |
|----|----------|
| **TRN-01** | the lifecycle is driven end-to-end from OUTSIDE the crate (`tests/setfit_repro.rs`), and every illegal stage transition is a compile error with a reviewed snapshot |
| **TRN-02** | with the explicit `max_length` qualifier below |
| **TRN-03** | the evidence gate fires inside `tune_encoder` on the path the out-of-crate run takes; `evidence_` 20 + `tune_` 47 + `negative_` 41 green |
| **TRN-04** | `cargo test -p aprender-core --lib multinomial` 62 passed; core lib 14285 passed; `pv lint contracts/multinomial-head-v1.yaml` PASS |
| **TRN-06** | cross-process hash equality at proven-different pool sizes over RECORDED digests, plus the separate intended-order replay check |
| **SAFE-03** | `tests/ui/setfit_probe_claims_setfit.rs` pins E0277 from outside the crate; `SetFitRun`'s constructors are private so no out-of-crate conversion can be written either |

**TRN-02 carries an explicit qualifier, written into REQUIREMENTS.md itself (W-09):**
**`max_length` is VALIDATED, not configurable — the only accepted value is the tokenizer's pinned
256.** `MiniLmTokenizer` hard-truncates at `MAX_SEQUENCE_LENGTH = 256` (tokenizer.rs:52) and
`encode_texts` takes no length parameter, so knob 6 accepts 256 and rejects everything else with
`MaxLengthNotSupported`. ROADMAP criterion 1 ("invalid … length … configuration fails before
training begins") IS satisfied; the requirement read as "choose a value" is not. Eleven of the
twelve knobs are genuinely configurable. Booked with the qualifier rather than silently as fully
met, and not left unbooked as though nothing was delivered.

**TRN-07 is left UNCHECKED, deliberately.** Its NEGATIVE half is now proven from outside the crate
at compile time (`setfit_token_without_lock` E0451, `setfit_metric_value_asserted` E0451) and the
mechanics are densely tested in-crate (73 `lock_` + 45 `evaluate_`). What is missing is the
POSITIVE "a user can" tier that 03-09 explicitly handed to this plan: **no out-of-crate caller and
no `apr` surface exercises `create_selection_lock -> mint_test_token ->
CanonicalTestAccess::grant`**, so nothing yet demonstrates a user REACHING the lock.
`tests/setfit_repro.rs` drives the lifecycle but stops at `verify_artifact`. Checking the box would
put a claim in the traceability table that the shipped surface does not support — the policy Phase 2
applied to DATA-01..06. The precise gap is written into the REQUIREMENTS.md line so the next plan
inherits a specification, not a mystery.

## Deviations from Plan

**1. [Rule 2 — Missing critical functionality] `batch_boundary_digest()` added to `mod.rs`.**
- **Found during:** Task 2. The plan's composite hash requires the RECORDED batch-boundary digest;
  03-08 exposed `batch_boundaries()` (the triples) but no reader for the digest.
- **Fix:** a public read-only accessor in the counted block, per the plan's own instruction
  ("that is an 03-08 defect — add the accessor in mod.rs, never a test-support backdoor"). The
  exhaustiveness guard moves 11 → 12 and exercises it through a shared reference;
  `verify_pair_order_digest_is_the_recorded_one` gains an assertion that it MOVES the recorded value
  rather than restating it.
- **Files:** `mod.rs`, `verify_tests.rs`. **Commit:** `09af9ecaa`.

**2. [Rule 1 — Bug in my own first draft] Cases 1 and 6 were rebuilt, not re-blessed.**
- **Found during:** Task 1 snapshot review. See Task 1 above — rustc's pass ordering silently
  dropped the structural half of each claim.
- **Files:** the two case files and their snapshots. **Commit:** `b7503075a`.

**3. [Plan instruction] Four functions decomposed for the complexity ceiling.**
- The plan's Task 3 `<files>` names only `Makefile` and `REQUIREMENTS.md`, but its action text says
  "if anything still breaches 10, name the function and decompose it here". Three files were
  therefore modified: `tune.rs`, `verify.rs`, `multinomial.rs`. **Commit:** `c04674039`.

**4. [Measured, corrected] The plan's `grep -c 'tee'` acceptance criterion is unfit.**
- It matched "guaran-tee" and reported violations in pipe-free recipes. The messages were reworded
  and the durable pipe-aware pattern recorded in the Makefile. **Commit:** `09af9ecaa`.

**5. [Forced by tooling] `--in-place` + `-j` and `--timeout 20` do not work as the plan assumes.**
- See SHORTFALL 1. Both attempted invocations are recorded verbatim with their refusal messages
  and elapsed-vs-budget numbers.

**6. [Environment] Running the test suite DELETED three tracked files.**
- `crates/aprender-train/src/prune/snapshots/*.snap.new` (three insta files, tracked in HEAD) were
  removed from the working tree by the `prune::snapshot_tests` run, and `pv lint` rewrote
  `.pv/contracts.idx`, `.pv/contracts.idx.mtime` and `.pv/lint-previous.json`. All six were restored
  with `git checkout -- <specific paths>` and the porcelain check re-run clean. Recorded because a
  plan that only ran `git status` at commit time would have committed three deletions it did not
  intend — which is what the post-commit deletion check exists to catch.
- No `git clean`, no blanket reset, and no `git stash` was used at any point.

## Environment Notes

- **`git commit` needs `-c commit.gpgsign=false` on this host.** `commit.gpgsign=true` with
  `gpg.program=/opt/homebrew/bin/gpg`; every recent commit is unsigned (`%G?` = `N`). Hooks ran
  normally; `--no-verify` was NOT used for any of the three commits.
- **The `rtk` hook rewrites `grep`, `git status`, `cargo clippy`, `cargo test` and `make` output**
  into a summary form and writes THAT to a redirect target. Every count here was taken with `awk`
  (unhooked) and every raw log read through `rtk proxy` or the hook's own tee log under
  `~/Library/Application Support/rtk/tee/`.
- **Long-running background jobs are killed between turns.** Three attempts at the full workspace
  suite were truncated mid-run; the completing run used `nohup … & disown` and was polled.
- `CARGO_INCREMENTAL=0` for every cargo invocation, per STATE.md's ENOSPC mitigation.
- No `apr` binary was invoked, so `scripts/apr_bin.sh` pinning did not arise.

## Known Stubs

None. Every file added is a test or a Make target that runs real code: the seven cases compile
against the real public API, the repro gates run the real pipeline over the real MiniLM slice, and
the three targets invoke real test binaries. No placeholder text, no hardcoded empty collection, no
component wired to mock data.

## Threat Flags

None. Every file created or modified is test code, in-process library code, or a Makefile recipe.
No network endpoint, no new deserialization surface reachable from untrusted input, no filesystem
path derived from caller input, and no schema change at a trust boundary. The one production change
(`batch_boundary_digest`) is a read-only accessor returning a `&str` the run already recorded.

## What the Next Plan Inherits

- **TRN-07's positive tier is the one specified gap.** An out-of-crate or `apr`-level path through
  `create_selection_lock -> mint_test_token -> CanonicalTestAccess::grant` is all that stands
  between the current evidence and an honest checkbox. `tests/setfit_repro.rs` is the worked
  example to extend — it already holds a `SetFitRun<ArtifactReloadedAndVerified>`.
- **The mutation run is a dedicated compute item**, with its blockers now diagnosed rather than
  discovered: `--timeout` ≥ 120 s for aprender-core, tree-copy mode for `-j`, 1181 mutants,
  ~44.6 h single-job.
- **`make coverage` should be scheduled in the same window** — both need an uncontended target dir.
- **The full workspace suite has 175 pre-existing failures across 8 crates**, now attributed
  per-crate. Any future plan that treats "full suite exits 0" as a gate needs that number in front
  of it first; the `aprender-train` slice of it is exactly the 21 baseline `gpu::` names.
- **`tests/ui/` is now a live snapshot suite.** A toolchain bump can reword a diagnostic and turn it
  red with no behaviour change; the re-baseline command and the named-types review rule are in
  `tests/ui.rs`'s module docs.

## Self-Check: PASSED

Every claimed file exists on disk (`tests/ui.rs` 3.2K, `tests/setfit_repro.rs` 27.1K,
`03-10-SUMMARY.md` 36.5K) and `crates/aprender-train/tests/ui/` holds exactly **14 files, all 14
tracked by git**. All four claimed commits resolve in `git log --all`: `b7503075a`, `09af9ecaa`,
`c04674039`, `fcea3276d`.

All three Make targets are defined AND wired inside the tier they are claimed to be in, verified by
line number rather than by reading: `tier2:` at 205, `tier3:` at 285, `setfit-repro-inproc` invoked
at **272** (inside tier2), `setfit-repro-crossproc` and `gemm-thread-determinism` at **336-337**
(inside tier3).

REQUIREMENTS.md reads back as claimed: TRN-01/02/03/04/05/06 and SAFE-03 `[x]`, **TRN-07 `[ ]`**,
and the traceability table agrees row for row including TRN-02's qualifier text.

No commit in this plan deleted a tracked file — `git diff --diff-filter=D HEAD~1 HEAD` was empty
after each of the four. The working tree is clean.
