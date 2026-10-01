---
phase: 03-faithful-two-stage-trainer-and-head
verified: 2026-08-12T01:28:03Z
status: passed
human_verification_resolved: 2026-08-14T22:35:57Z
human_verification_outcomes: |
  All five human_needed items adjudicated; full evidence in 03-HUMAN-UAT.md (status: complete).
  1. Mutation budget: authorized. dropout_rng.rs closed at 75/79 caught, 100% adjusted
     (test commit 075a318b7, kill confirmed by scoped re-run 16/16). Remaining scopes
     (setfit dir 948 + multinomial.rs 186, ~44h) deferred to GitHub CI post-PR by human
     decision 2026-08-14.
  2. Coverage: authorized and attempted; NOT MEASURABLE against the floor on this host
     (best data 57/58 binaries, 73%, denominator not comparable). COV_FLOOR
     unpinned-denominator gate defect recorded for its own ticket. No floor claim either way.
  3. Feature-closure: human chose (b) — D-ITEM-05 fixed in d7b65a116; the literal must-have
     `cargo check -p aprender-train --no-default-features --features setfit` now exits 0 and
     the feature-matrix leg (a) enforces the plain green check.
  4. CR-01 test-surface guarding: option (a) landed in both halves — a844f6a98 (tier3),
     52357404b (CI). Human confirmed 2026-08-14.
  5. CR-02/03/04 scope: fixed inside Phase 3 (51db85ace, 0158a758d, 1038f6414), RED-proven.
score: 5/5 roadmap success criteria verified; 47/50 plan must-have truths verified (52/55 total)
overrides_applied: 0
re_verification:
  previous_status: none
  previous_score: none
  note: "Initial verification. No prior 03-VERIFICATION.md existed."
human_verification:
  - test: "Authorize (or decline) a dedicated compute budget for the scoped cargo-mutants run over the Phase 3 surface, then run it and record the ADJUSTED score."
    expected: "Adjusted mutation score >= 85% after excluding only proven-equivalent re-run survivors (plan 03-10 must-have 5). Requires --timeout >= 120 for aprender-core (the 14285-test binary has not finished LINKING at 20s) and cargo-mutants tree-copy mode instead of --in-place, because 25.3.1 refuses --in-place with --jobs."
    why_human: "Projected ~44.6h single-job. CLAUDE.md places any non-lambda-vector compute spend >1hr behind explicit human authorization, and the verifier was instructed not to start the run. The criterion is UNMEASURED, not failed by implementation — no score exists to judge. Verifier independently reproduced blocker (1): cargo-mutants 25.3.1 emits `error: the argument '--in-place' cannot be used with '--jobs <JOBS>'`."
  - test: "Authorize the `make coverage` run on an uncontended target dir and confirm the measured line coverage against the enforced floor."
    expected: "Line coverage >= COV_FLOOR (88%) with the Phase 3 surface included; plan 03-10 must-have 6 lists coverage in the closing audit (VALIDATION.md task 3-10-03)."
    why_human: "`cargo llvm-cov` across the workspace with the mold linker disabled is the phase's second-heaviest command and needs the same cargo lock the mutation attempts held. Same >1hr compute authorization gate. Verifier was instructed not to start it."
  - test: "Decide whether the substituted feature-closure gate is accepted, or whether `monitor::tui` must be gated inside this phase."
    expected: "Either (a) accept `make setfit-feature-matrix` leg (a)'s two-sided diagnostic diff as satisfying plan 03-03 must-have 1, recording an override; or (b) require the D-ITEM-05 fix (gate `crates/aprender-train/src/monitor/mod.rs:45`'s unconditional `pub mod tui;` on `feature = \"tui\"` plus its re-exports) so the plain `cargo check -p aprender-train --no-default-features --features setfit` can exit 0 as the plan literally asked."
    why_human: "The literal must-have is measured RED (rc=101, 8 errors) and the cause is a pre-existing defect in a module outside Phase 3's declared file set. The property Phase 3 owns (setfit does not leak into the minimal build) IS verified. Whether the substitution is accepted or the out-of-scope fix is pulled in is a scope decision, not a measurement."
deferred:
  - truth: "TRN-07 POSITIVE tier — a user REACHES the lock: an out-of-crate or `apr` caller exercising create_selection_lock -> mint_test_token -> CanonicalTestAccess::grant"
    addressed_in: "Phase 4 and Phase 5"
    evidence: "Phase 4 success criterion 3: 'A Rust caller and an `apr` user can complete the CPU train -> APR -> inspect -> eval -> predict lifecycle through stable fallible APIs and machine-readable output.' Phase 5 success criterion 3: 'A user receives one machine-readable row per method/shot/seed run containing dataset/model revisions, selection lock, artifact hash, encoder-update evidence, ...' — 'selection lock' appears verbatim. Phase 5 criterion 4 requires post-test-selected cells to invalidate the report, which is the lock's consumer. ROADMAP criterion 5 for THIS phase asks only that test access remain BLOCKED until a lock record exists, and the blocking half is verified below."
---

# Phase 3: Faithful Two-Stage Trainer and Head — Verification Report

**Phase Goal:** Users can reproducibly tune the encoder and then fit one stable multiclass head on
each unique tuned embedding exactly once, with SetFit identity and test access enforced by lifecycle
evidence.

**Verified:** 2026-08-12T01:28:03Z
**Status:** passed (human_needed items resolved 2026-08-14 — see frontmatter `human_verification_outcomes` and 03-HUMAN-UAT.md)
**Re-verification:** No — initial verification
**Tree verified:** `bd54801e0` / root tree `a2f7f91b0`, working tree clean (`rtk proxy git status --porcelain` -> only `?? .serena/`)

---

## Goal Achievement

### Observable Truths — ROADMAP Success Criteria (the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Only the legal `Prepared -> EncoderTuned -> HeadFitted -> ArtifactReloadedAndVerified` transitions can run, and invalid learning / batching / warmup / clipping / length / pairing / freeze / regularization / seed / device configuration fails before training begins | VERIFIED | Typestate is real, not documentary: `mod sealed { pub trait Sealed {} }` (mod.rs:87), four markers, `impl LifecycleState` per marker, `SetFitRun<S>` with six private fields + `PhantomData<S>` (mod.rs:488). `prepare()` is the only door (mod.rs:559); `into_parts()` is `pub(crate)` so no caller can extract-tune-reinsert. Each transition is on exactly one state impl block: `tune_encoder` on `Prepared` (625), `fit_head` on `EncoderTuned` (700), `verify_artifact` on `HeadFitted` (768). Compiler-proven from OUTSIDE the crate: `setfit_direct_state_construction` pins E0451 on all six private fields; `setfit_fit_head_before_tune` pins E0599 `no method named fit_head found for struct SetFitRun<Prepared>`. Config: all 12 knobs validated in `SetFitTrainConfig::new` (config.rs:450-465) with knob-naming typed errors; serde REDIRECTED through `#[serde(into=…, try_from = "SetFitTrainConfigWire")]` (config.rs:176) so a JSON payload runs the same validators; device grammar at construction + `UnsupportedDeviceForPhase3` at `prepare`; freeze zero-match rejected in `preflight` -> `apply_freeze` (aprender-core setfit/mod.rs:560) BEFORE the parameter snapshot and baseline encode, i.e. before any gradient step. Measured: `config_` 411 passed rc=0; `--test ui` 7/7 cases rc=0 |
| 2 | A run cannot identify itself as SetFit or export a model until named encoder gradients, parameter deltas, embedding deltas and pair-loss behavior pass; frozen probes / centroids / non-updating baselines remain explicitly labeled | VERIFIED | `validate_evidence` is THE single gate and it runs INSIDE the transition (tune.rs:1095, `#[contract("setfit-train-lifecycle-v1", equation="evidence_gate")]`). Five ordered rungs: uncalibrated regime refused BEFORE any comparison; empty trainable set; all-ungated trainable set (closes the "freeze everything except the gradient-free biases" hole); per-parameter finite-grad + strict-movement + class epsilon; run-level embedding-delta floor last. `summary.verdict` is stamped `Fail` pessimistically so a forgotten path cannot emit a passing summary. Export is downstream of `HeadFitted` which is downstream of `EncoderTuned`, so there is no ordering in which an unproven run serializes. Thresholds are MEASURED, not chosen — verifier re-ran `calibration_matrix_epsilon_basis` (rc=0, 40.9s) and every frozen epsilon sits inside its independently re-measured two-sided band: embedding 1.1e-5 in [4.206e-6, 1.198e-5]; layer_norm_weight 2.9e-6 in [0, 2.930e-6]; layer_norm_bias 1.1e-5 in [1.548e-6, 1.161e-5]; projection_weight 1.8e-5 in [4.092e-6, 1.875e-5]; projection_bias 8.3e-6 in [1.460e-6, 8.394e-6]; attention_key_bias null/ungated, matching the run's own `EPS-BELOW-NOISE` flag. SAFE-03: `FROZEN_PROBE_KIND = "frozen_linear_probe"` (baseline.rs:34), no `From`/`Into` to any `SetFitRun<*>`, compile-proven by `setfit_probe_claims_setfit` (E0277) and reinforced by the private-field E0451 proof. Measured: `evidence_` 20, `negative_` 41, `thresholds_` 6, `baseline_` 9, `evidence_gate` 1 — all rc=0 |
| 3 | After encoder tuning, dropout is disabled and each unique selected row is encoded exactly once in evaluation/no-gradient mode before one deterministic L2-regularized multinomial softmax head is fit for any ordered `K >= 2`; pair multiplicity cannot reweight the head data | VERIFIED | `head_dataset` calls `encoder.set_training(false)` BEFORE anything is encoded (head_input.rs:217), then `encode_once` runs the whole loop inside `autograd::no_grad` and `detach()`s before storing (head_input.rs:275-296). Mechanism PROVEN engaged, not asserted: an `EncodeWitness` records `training_observed`, `requires_grad_observed` and the tape delta, and `built.witness().require_isolated()?` is a PRODUCTION gate (head_input.rs:229), not a test. Encode-exactly-once is proven by ledger, not by count: the ledger entry and the encoder input come from the same slice of one `rows` vector built from `selection.ordered_ids()`, and `head_input_a_batch_larger_than_the_row_count_still_encodes_every_row_once` compares SORTED MULTISETS; `head_input_rows_from_distinct_texts_have_distinct_embeddings` defeats the "one row encoded 24 times" defect that every count check would pass. Pair multiplicity is INEXPRESSIBLE: `pub fn fit_head(self)` takes no pair parameter and `SetFitRun` has no field one could live in — compile-proven by `setfit_pairs_into_fit_head` (E0061 arity, naming the real signature). Lambda resolves ONCE against unique rows: `resolve_lambda(&head_regularization, input.n())` (head_input.rs:451) with `SklearnEquivalentC{c:1.0}` over 24 rows pinned to exactly 1/48. Adversarial control `negative.rs` builds the pair-weighted fitter at an IDENTICAL lambda and shows it moves the coefficients. Measured: `head_input_` 13, `fit_head` 7, `pair_weight` 3 — all rc=0 |
| 4 | The shared binary/multiclass head reports explicit convergence or typed failure, finite logits and probabilities summing to one, stable ordered-label semantics, and reference-matching regularization behavior | VERIFIED | `MultinomialLogisticRegression` (multinomial.rs:863) covers K>=2 including binary — the pre-existing binary `LogisticRegression` is deliberately untouched (D-01) and the multinomial head is the shared one. Every `ConvergenceStatus` is matched exhaustively and `MaxIterations` / `Stalled` / `NumericalError` become typed `HeadFitError`s (1114-1135) — a non-convergence is an error, never sklearn's warning. Logits accumulate in f64 and a non-finite one is `HeadFitError::NonFiniteLogit{row, class}` (1182), not an Inf that propagates. Probabilities asserted finite and summing to 1 within 1e-6 at K=2 (`k2_binary_boundary_fit_predict_proba_and_labels`) and K=3; exact ties break to the lowest label index, with the test first proving the fit really landed on exactly-zero parameters so the tie is real. Regularization matches the reference: `lambda = 1/(2*c*n_rows)` (multinomial.rs:116) — D-04 as amended — pinned by the frozen sklearn fixture, a factor-2 falsification at n=24, and `falsify_multinomial_001_sklearn_optimum_is_stationary_only_at_the_halved_lambda`. Fits in f64 via `LbfgsF64`, stores f32; `HeadFitReport` carries no `Duration`. `pub struct LBFGS` still has ZERO generic parameters (lbfgs.rs:618). Measured: `multinomial` 62, `multinomial_contract` 8, `lbfgs` 68, `optim::` 516 — all rc=0 |
| 5 | Two clean CPU runs reproduce selected IDs, ordered pairs and batches, step count, declared loss trace, semantic hashes and predictions; canonical test access stays blocked until canonical-validation selection emits a selection-lock record | VERIFIED | Re-run independently by the verifier with rc captured directly (not through a pipe): `make setfit-repro-crossproc` **rc=0** — "cross-process: THREADS differed and all ten components agreed"; `make gemm-thread-determinism` **rc=0**; `make setfit-repro-inproc` **rc=0**; `cargo test -p aprender-train --test setfit_repro --features setfit` 4 passed rc=0. The ten components map 1:1 to the criterion (setfit_repro.rs:280): selection semantic hash, pair-order digest, batch boundaries+digest, step count, loss-trace hash, evidence-table hash, encode-ledger hash, parameter-registry hash, artifact hash, probe-prediction digest over LE BIT PATTERNS. The digests are RECORDED, not recomputed: accessors return `evidence.passed.table().consumed_pair_digest` (mod.rs:850) and the loop absorbs them AT CONSUMPTION (`absorb_batch_digests` once per pair inside the pull loop, `absorb_boundary_digest` once at batch open — tune.rs:540-612). Reproducible is separated from CORRECT: `setfit_repro_recorded_matches_expected_replay` independently recomputes the order from the public `epoch_pair_order` + a fresh `PairSampler` and compares against the recorded digests, with explicit non-vacuity asserts (`n_pairs > 0`, boundaries non-empty, replay step count == recorded step count). The GEMM control is mechanism-engaged per CLAUDE.md rule 2 — verifier observed `PARTITIONS=1 / 2 / 2` at pool sizes 1/2/3 with every hash still matching, printing `FALSIFIED: 80x384x384 was partitioned [1, 2, 2] ways`; the harness would print `SKIPPED-WITH-EVIDENCE` and NOT claim a falsification had the partitioning never moved. Test-access blocking: `CanonicalTestGrant` needs a `CanonicalTestToken` + the model + `&Split<Test>` and re-checks identity at grant time; the token has no public constructor and can only come from `SelectionLock::mint_test_token`, which calls `verify_integrity()` FIRST and then reads `model.artifact_hash()` off the OBJECT (lock.rs:419-441) — no `[u8;32]` parameter exists. Compile-proven from outside the crate: `setfit_token_without_lock` (E0451). Measured: `lock_` 77, `evaluate_` 45 — rc=0 |

**Score: 5/5 roadmap success criteria VERIFIED.**

### Plan-Level Must-Have Truths — the three that are not plain VERIFIED

The other 47 of 50 plan must-have truths across plans 03-01…03-10 resolve to VERIFIED on the
evidence in the tables above and below. These three do not:

| Plan | Must-Have Truth | Status | Evidence |
|------|-----------------|--------|----------|
| 03-10 | "cargo-mutants scoped to the new Phase 3 code reports an ADJUSTED score >= 85% after excluding proven-equivalent mutants; narrative justification alone does not waive the threshold" | UNCERTAIN — unmeasured (WARNING) | NOT RUN, and correctly NOT waived by narrative — no score is claimed anywhere. Verifier independently reproduced tooling blocker (1): `cargo mutants --version` -> `cargo-mutants 25.3.1`; `cargo mutants --in-place --jobs 2 -f … --list` -> `error: the argument '--in-place' cannot be used with '--jobs <JOBS>'`. Blocker (2) (`--timeout 20` kills the BASELINE because aprender-core's 14285-test binary has not finished linking in 20s) was reproduced by the orchestrator. ~44.6h projected single-job exceeds the >1hr human-authorization threshold, and the verifier was instructed not to start it. **Caveat on the SUMMARY:** verifier could NOT reproduce the claimed 1181-mutant inventory — `cargo mutants --list` returned EMPTY output on this host for every `-f` glob form tried (`crates/aprender-train/src/train/setfit/`, `**/train/setfit/**`, `**/classification/multinomial.rs`), rc=0. The inventory number in 03-10-SUMMARY is therefore unconfirmed. It does not change the verdict — the must-have is unmeasured either way |
| 03-10 | "The closing audit runs the VALIDATION.md full suite and the scoped clippy gates" — the coverage leg of it (`make coverage`, VALIDATION.md task 3-10-03) | UNCERTAIN — unmeasured (WARNING) | The full suite and scoped clippy legs ARE verified below. `make coverage` was not run by the executor and not run by the verifier (same >1hr compute gate, and it needs the uncontended target dir). No coverage number is claimed for the Phase 3 surface |
| 03-03 | "cargo check -p aprender-train --no-default-features --features setfit exits 0 AND the maximal CPU-safe feature combination with setfit also exits 0" | UNCERTAIN — substituted (WARNING) | The first half is measured **RED**. Verifier observed leg (a) of `make setfit-feature-matrix` report `control rc=101, setfit rc=101` on this tree. Cause is D-ITEM-05: `crates/aprender-train/src/monitor/mod.rs:45` declares `pub mod tui;` UNCONDITIONALLY while `presentar-terminal` is gated behind `feature = "tui"` — 8 errors, all in `src/monitor/tui/{app,dashboard}.rs`, in a module outside Phase 3's declared file set. What WAS delivered instead is defensible and non-vacuous: leg (a) is a two-sided DIFF that requires diagnostics to exist when rc != 0 (`if ctl_rc != 0 && ! -s control.errs -> FAIL`), requires the exit statuses to match, and diffs the two diagnostic streams byte-for-byte — plus a two-sided `cargo tree` check that both proves the absence in the default build AND proves `--features setfit` actually pulls `aprender-contrastive-data`/`aprender-rand`/`tokenizers` in, so the absence check cannot pass vacuously. The SECOND half of the must-have (maximal CPU-safe combination) IS green. Verifier measured `make setfit-feature-matrix` rc=0 overall |

---

## Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | TRN-07's POSITIVE "a user can" tier — an out-of-crate or `apr` caller exercising `create_selection_lock -> mint_test_token -> CanonicalTestAccess::grant` | Phase 4 and Phase 5 | Verifier confirms the orchestrator's reading is CORRECT, and adds a precision: the API **is** publicly reachable (`entrenar::train::setfit::lock` is `pub mod` under `pub mod setfit` under `pub mod train`; `create_selection_lock`, `mint_test_token`, `CanonicalTestToken::grant` and `SelectionCandidate::from_evaluation` are all `pub`) — what is missing is an out-of-crate EXERCISE. Measured: `grep -rn --include="*.rs"` for `create_selection_lock|mint_test_token|CanonicalTestGrant` across `crates/` and `src/` returns exactly 3 files — `lock_tests.rs` (14, in-crate `#[cfg(test)]`), `lock.rs` (10, definitions), and `tests/ui/setfit_token_without_lock.rs` (1, the compile-fail NEGATIVE). No `apr-cli` reference to `SetFitRun` exists at all. Phase 4 SC3 delivers the `apr` lifecycle surface; Phase 5 SC3 names "selection lock" verbatim as a required field of every benchmark row. ROADMAP criterion 5 for THIS phase asks only that access remain BLOCKED until a lock exists, and that half is VERIFIED |

TRN-07 correctly remains `[ ]` in REQUIREMENTS.md. Checking it would put a claim in the traceability
table the shipped surface does not yet demonstrate.

---

## Required Artifacts

Levels: 1 exists · 2 substantive · 3 wired · 4 data flows.

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/aprender-core/src/optim/lbfgs.rs` | Private float-generic core; non-generic public `LBFGS` (f32); separate `LbfgsF64` | VERIFIED | `pub struct LBFGS` at 618 with zero generic params; `pub struct LbfgsF64` at 766. Consumed by `multinomial.rs:1106` (`LbfgsF64::new`) |
| `crates/aprender-core/src/optim/tests_lbfgs_contract.rs` | f64 twins, four-channel non-finite matrix in both widths, f32 golden trajectory | VERIFIED | `lbfgs` filter 68 passed rc=0 |
| `contracts/lbfgs-kernel-v1.yaml` | Version-bumped, declaring the f64 entry point and the non-finite obligation | VERIFIED | `pv validate` rc=0, "0 error(s), 0 warning(s)". `pv diff` against the materialized `e2dee4be9` revision: "v1.0.0 -> v1.1.0, Suggested bump: minor", additions only (`+ nonfinite_input_status`, `+ FALSIFY-LB-007/008/009`) — additive as the must-have required |
| `crates/aprender-core/src/setfit/dropout_rng.rs` | Counter-based Philox mask source with forward-ordinal coordinate, SHA-256 domain derivation | VERIFIED | `dropout_rng` 15 passed rc=0. Wired: `tune_the_two_branches_use_distinct_forward_ordinals` observes ordinals `2*s` and `2*s+1` at runtime |
| `crates/aprender-core/tests/gemm_thread_determinism.rs` | Subprocess thread-count falsification at FIXED pool sizes | VERIFIED | rc=0, 2 tests. Mechanism-engaged: PARTITIONS 1/2/2 observed differing |
| `crates/aprender-compute/src/blis/parallel.rs` | Read-only accessor extraction only; behavior UNCHANGED | VERIFIED | Diff reviewed line-by-line: `gemm_m_partitions` is a verbatim lift of the FLOP ladder, `HeijunkaScheduler::default()`, the `min(max_threads)` cap and `partition_m`; `gemm_blis_parallel` now calls it. The CONTINGENT partitioner behavior change was correctly NOT taken — `contracts/gemm-partition-determinism-v1.yaml` does not exist, consistent with the gate being green |
| `crates/aprender-train/src/train/setfit/config.rs` | `SetFitTrainConfig` + `ResolvedSetFitConfig`, 12 knobs, serde via `TryFrom` | VERIFIED | 53.9K, `#[serde(try_from = "SetFitTrainConfigWire")]` at 176, `TryFrom` impl at 782 calling the same `new()`. `config_` 411 passed |
| `crates/aprender-train/src/train/setfit/mod.rs` | Sealed `LifecycleState`, four markers, `SetFitRun<Prepared>` + `prepare()` | VERIFIED | 85.2K. See truth 1 |
| `crates/aprender-train/src/train/setfit/reduce.rs` | Fixed-order f64-accumulating reductions (D-13) | VERIFIED | `reduce_` 15 passed. Wired: `evaluate.rs::accuracy` routes through `reduce::mean_in_index_order` rather than counting in a `usize` |
| `crates/aprender-train/src/train/setfit/epoch.rs` | Philox epoch-shuffle under tag `apr-setfit-train-v1` | VERIFIED | `epoch_` 42 passed. Wired: `tune_with_probes` calls `epoch_pair_order(root_seed, epoch, n_pairs)`; the independent replay test calls the same public fn |
| `crates/aprender-train/src/optim/scheduler/warmup_linear_decay.rs` | Linear warmup then linear decay to 0, HF reference | VERIFIED | `warmup_linear` 13 passed. Wired into `TuneCtx.scheduler`; `tune_step_order_is_pinned` observes `steps[0].applied_lr == 0.0` and `!= peak`, plus a non-vacuity assert that the rate later leaves warmup |
| `crates/aprender-core/src/classification/multinomial.rs` | `MultinomialLogisticRegression` + `HeadFitError` + `HeadFitReport` | VERIFIED | See truth 4 |
| `contracts/multinomial-head-v1.yaml` | Both objective conventions expanded, gradient obligation, falsification tests | VERIFIED | `pv validate` rc=0. Binding audit: 5 equations, 5 bound, 5 implemented, 0 partial, 0 unimplemented |
| `scripts/gen_multinomial_sklearn_fixture.py` | PEP 723 self-pinning generator asserting `n_iter_ < max_iter` | VERIFIED | 10.0K present; generated constants marked DO-NOT-HAND-EDIT in `tests_multinomial_contract.rs:319` |
| `crates/aprender-train/src/train/setfit/tune.rs` | `run_tuning` with in-band execution digests | VERIFIED | 71.4K. `tune_` 47 passed. Digests absorbed at consumption (540, 586), finalized at 899-900 |
| `crates/aprender-train/src/train/setfit/evidence.rs` | `UpdateEvidence` + `EvidenceSummary` + canonical bytes + binding table hash | VERIFIED | 82.4K. `evidence_` 20, `negative_` 41 passed |
| `crates/aprender-train/src/train/setfit/test_fixtures.rs` | Deterministic encoder + Selection + pair config, synthetic text only | VERIFIED | 31.6K. `fixture_` 12 passed. Fixture is a REAL slice of the pinned MiniLM (real pretrained weights, synthetic only in DIMENSIONS) — the distinction the calibration argument depends on |
| `contracts/setfit-train-lifecycle-v1.yaml` | Per-class thresholds + calibration regime, endpoint k/margin, RNG derivations, digest definitions | VERIFIED | 1411 lines, v2.0.0. `pv validate` rc=0. Binding audit: **16 equations, 16 bound, 16 implemented, 0 partial, 0 unimplemented** |
| `crates/aprender-train/src/train/setfit/thresholds.rs` | Single Rust threshold source with a YAML-PARSING test | VERIFIED | `include_str!("../../../../../contracts/setfit-train-lifecycle-v1.yaml")` at 49 (compile-time — a missing file is a build error, not a skipped test). `thresholds_match_the_contract` DESERIALIZES into typed structs and compares per class per field, with a non-vacuity assert FIRST (`frozen_thresholds.len() == ParameterClass::ALL.len()`). Not a substring search. `thresholds_` 6 passed |
| `crates/aprender-train/src/train/setfit/baseline.rs` | `FrozenProbeRun` — never claims SetFit | VERIFIED | `FROZEN_PROBE_KIND = "frozen_linear_probe"` at 34; binds `linear-probe-classifier-v1` at 146. `baseline_` 9 passed |
| `contracts/linear-probe-classifier-v1.yaml` | In `PHASE3_CONTRACTS`, reachable by the scoped audit | VERIFIED | `pv validate` rc=0; in `PHASE3_CONTRACTS` (Makefile:1236); binding audit 1/1 bound and implemented |
| `crates/aprender-train/src/train/setfit/head_input.rs` | Encode-once embedding matrix with an encode ledger | VERIFIED | 39.6K. See truth 3 |
| `crates/aprender-train/src/train/setfit/negative.rs` | In-band pair-weighted fitter that must FAIL its gate | VERIFIED | 17.8K, `mod negative` is PRIVATE (mod.rs:68) so it is not a shipped door. `pair_weight` 3 passed rc=0 |
| `crates/aprender-train/src/train/setfit/bundle.rs` | Complete deterministic state, canonical wire form, bounded deserialization | VERIFIED | 31.6K. All 19 fields of the contract's `SetFitBundle` formula present and in order (317-356): schema_version, format_id, architecture, tokenizer_bytes_hex, pooling, normalization, l2_epsilon, truncation_max_sequence_length, padding_mode, max_length, root_seed, tensors, head_weights_hex, head_intercepts_hex, head_n_features, ordered_labels, requested_config, resolved_config, evidence. `bundle_` 32 passed |
| `crates/aprender-train/src/train/setfit/verify.rs` | Sealed `SetFitCodec` + `SerdeJsonCodec` + trusted `verify_artifact` policy | VERIFIED | Seal proven from OUTSIDE the crate: `setfit_external_codec_impl.stderr` pins E0277 `the trait bound MyCodec: verify::sealed::Sealed is not satisfied` and rustc itself explains `SetFitCodec is a "sealed trait"`. The drop is STRUCTURAL — `run_verify_policy` takes encoder and head BY VALUE, so no binding to the pre-close model survives. `verify_` 43 passed |
| `crates/aprender-train/src/train/setfit/evaluate.rs` | `ValidationEvaluation` + trusted evaluator; no public constructor taking a metric value | VERIFIED | `evaluate_validation` (249) computes the metric itself and reads `artifact_hash` off the run. Compile-proven: `setfit_metric_value_asserted.stderr` E0451 on all six private fields incl. `value`. Compatibility-selection is non-constructible BY TYPE — `validation()` lives only inside `impl PreparedDataset<Canonical>` (prepared.rs:187-368), and `impl PreparedDataset<Compatibility>` starts at 368. `evaluate_` 45 passed |
| `crates/aprender-train/src/train/setfit/lock.rs` | Candidate-committing, rule-APPLYING `SelectionLock` + token + grant | VERIFIED | 38.3K. `from_candidates` is `pub(super)` so no public constructor takes provenance strings; `create_selection_lock` refuses a candidate set that omits the creating run; `forge_chosen_index_for_tests` is `#[cfg(test)] pub(super)` — the forgery door is not in a shipped build. `lock_` 77 passed |
| `crates/aprender-train/tests/ui.rs` + 7 `tests/ui/*.rs` + 7 `.stderr` | trybuild harness, non-vacuous | VERIFIED | rc=0 and, with `--nocapture`, all SEVEN named cases observed executing individually — a glob matching zero files would have passed silently. Harness is `#![cfg(feature = "setfit")]` for exactly that reason. Every `.stderr` names real crate types/visibility: E0451 ×3, E0277 ×2, E0599, E0061. None is a syntax error or unresolved import |
| `crates/aprender-train/tests/setfit_repro.rs` | In-process two-run equality + subprocess child mode | VERIFIED | 27.1K, 4 tests rc=0. Exercises the full public chain `prepare -> tune_encoder -> fit_head -> verify_artifact` from OUTSIDE the crate |
| `Makefile` | `setfit-repro-crossproc` + `gemm-thread-determinism` wired into tier3 | VERIFIED | Wiring confirmed at the surface where the decision is made (CLAUDE.md rule 6): `tier3` body invokes `contract-validate`, `contract-audit-phase2`, `contract-audit-phase3`, `setfit-repro-crossproc`, `gemm-thread-determinism`, `setfit-feature-matrix` (Makefile:314-338); `tier2` invokes `setfit-repro-inproc` (272). Both recipes read `$$?` on the line AFTER the redirect and contain no `tee` |

No artifact is MISSING, STUB, ORPHANED or HOLLOW.

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `optim/lbfgs.rs` | `contracts/lbfgs-kernel-v1.yaml` | `#[contract]` on the two-loop recursion | WIRED | `pv validate` rc=0; `pv diff` minor |
| `setfit/encoder.rs` | `setfit/dropout_rng.rs` | mask draws keyed by forward ordinal | WIRED | Runtime-observed distinct ordinals per branch |
| `aprender-train/Cargo.toml` | `aprender-contrastive-data` | optional dep under `setfit` | WIRED | Two-sided `cargo tree`: absent from the default build, PRESENT under `--features setfit` (so the absence check is non-vacuous) |
| `setfit/config.rs` | `train/device.rs` | `resolve_device` reused in place | WIRED | `validate_device_grammar` delegates; `CudaNotAvailable` deliberately deferred to `prepare()` |
| `classification/multinomial.rs` | `optim/lbfgs.rs` | `LbfgsF64` entry point | WIRED | `LbfgsF64::new(max_iter, tol, history_size)` at 1106 |
| `Makefile` | `contracts/multinomial-head-v1.yaml` | `PHASE3_CONTRACTS` + blocking `contract-audit-phase3` in tier3 | WIRED | Verifier ran `make contract-audit-phase3` rc=0 |
| `setfit/tune.rs` | pair replay | `PairSampler::pair_at` over the epoch permutation, each ordinal fed to the in-band digest | WIRED | `sampler.pair_at(o)` inside the pull loop (540), digest absorbed in the SAME iteration |
| `setfit/tune.rs` | `SetFitMiniLm` | `set_forward_ordinal(2*step+branch)` + train-mode forward + `pair_cosine_mse` | WIRED | Order pin test observes the ordinals and the graph-side gradients |
| `setfit/evidence.rs` | `setfit/reduce.rs` | every norm/mean through fixed-order reductions | WIRED | `in_index_order` family; `reduce_` 15 passed |
| `setfit/tune.rs` | `contracts/setfit-train-lifecycle-v1.yaml` | `#[contract(…, equation="evidence_gate")]` on the single gate fn | WIRED | tune.rs:1093; binding audit 16/16 |
| `setfit/baseline.rs` | `contracts/linear-probe-classifier-v1.yaml` | binding.yaml entry with real module path | WIRED | Audit 1/1 implemented |
| `setfit/mod.rs` | `classification/multinomial.rs` | `fit_head` fits on the encode-once matrix | WIRED | `head_input::fit_on_selection` -> `MultinomialLogisticRegression` |
| `setfit/head_input.rs` | `Selection` | `examples()`/`ordered_ids()` in deterministic order, each id appended to the ledger | WIRED | Same-slice construction; multiset test |
| `setfit/verify.rs` | `setfit/mod.rs` | `verify_artifact(self, codec)` consuming HeadFitted, minting ArtifactReloadedAndVerified | WIRED | mod.rs:768 |
| `setfit/bundle.rs` | `aprender-core setfit/mod.rs` | `SetFitMiniLm::from_bundle_parts` rebuilding a working encoder from bytes | WIRED | `tokenizer_bytes` 4 passed; `bundle_` 32 passed |
| `setfit/evaluate.rs` | `Split<Validation>` | evaluator reaches the split via `validation()` / `validation_witness()`; witness fingerprint committed | WIRED | Fingerprints compared on BOTH sides before a single row is encoded |
| `setfit/lock.rs` | `setfit/verify.rs` | `mint_test_token` reading the run's own `artifact_hash()` | WIRED | lock.rs:429 `let observed = model.artifact_hash();` |
| `Makefile` | `tests/setfit_repro.rs` | tier3 target spawning two processes and comparing hash lines | WIRED | rc=0, THREADS proven to differ |
| `tests/ui.rs` | `tests/ui/*.rs` | trybuild `compile_fail` glob | WIRED | 7/7 cases observed executing |

No link is NOT_WIRED or PARTIAL.

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `mod.rs` reproducibility accessors | `consumed_pair_digest`, `batch_boundary_digest`, `batch_boundary_list`, `loss_trace_hash`, `parameter_registry_hash`, `step_count` | `evidence.passed.table()` — populated by the tuning loop at consumption | Yes — and provably NOT recomputed from configuration | FLOWING |
| `HeadFittedEvidence` | `head`, `report`, `effective_lambda`, `ordered_labels`, `encode_ledger`, `encode_call_count` | `head_input::fit_on_selection` over the encode-once matrix | Yes | FLOWING |
| `SetFitBundle` | 19 fields | Live run parts, `pub(crate)` assembly owned by the verify transition | Yes — reload re-serializes and requires byte-equality to what was hashed, so the reloaded value is provably a function of the bytes | FLOWING |
| `ValidationEvaluation.value` | `value` | `accuracy` / `macro_f1` computed inside `evaluate_validation` from `dataset.validation().rows()` | Yes — no caller-supplied f64 path exists (E0451) | FLOWING |
| `SelectionLock` | candidate history, chosen index, fingerprints, ledger hash | `from_candidates` APPLIES the rule; provenance read off the run | Yes — no `chosen` parameter exists | FLOWING |
| `Thresholds::frozen()` | per-class eps / scale_floor / sparse / gated | Hardcoded Rust constants, cross-checked against the parsed contract by test | Yes — and the numbers were independently re-derived by the verifier's own `calibration_matrix_epsilon_basis` run | FLOWING |

No HOLLOW or DISCONNECTED artifact.

---

## Behavioral Spot-Checks

All run by the verifier on this tree with rc captured directly (`cmd > log 2>&1; rc=$?`), never
through a pipe (CLAUDE.md rule 1). `rtk proxy` used so the hook did not rewrite counts.

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase 3 trainer surface, 18 scoped filters | `cargo test -p aprender-train --lib --features setfit <filter>` for `config_ reduce_ epoch_ warmup_linear fixture_ tune_ evidence_ evidence_gate negative_ thresholds_ baseline_ head_input_ fit_head pair_weight bundle_ verify_ evaluate_ lock_` | rc=0 on all 18; 837 tests passed, 0 failed | PASS |
| aprender-core Phase 3 surface | same for `optim:: lbfgs multinomial multinomial_contract dropout_rng tokenizer_bytes setfit::` | rc=0 on all 7; 775 passed, 0 failed | PASS |
| Compile-fail lifecycle proofs | `cargo test -p aprender-train --test ui --features setfit` | rc=0; with `--nocapture`, 7/7 named cases observed | PASS |
| Cross-process reproducibility | `cargo test -p aprender-train --test setfit_repro --features setfit` | rc=0, 4 passed | PASS |
| GEMM determinism harness | `cargo test -p aprender-core --test gemm_thread_determinism` | rc=0, 2 passed; `FALSIFIED: … partitioned [1, 2, 2] ways` | PASS |
| Epsilon calibration basis (ignored test) | `cargo test -p aprender-train --lib --features setfit calibration_matrix -- --ignored --nocapture` | rc=0, 40.88s; every frozen epsilon inside its measured two-sided band | PASS |
| aprender-train full lib suite | `cargo test -p aprender-train --lib --features setfit` | rc=101, **7839 passed / 24 failed**. The 24 names are byte-for-byte the `known-red-baseline.md` list (21 `gpu::` + 3 `prune::snapshot_tests`). **Zero regressions** | PASS (against baseline) |
| aprender-core full lib suite | `cargo test -p aprender-core --lib --features setfit` | rc=0, **14285 passed / 0 failed** | PASS |
| aprender-compute lib suite | `cargo test -p aprender-compute --lib` | rc=101, 3320 passed / 1 failed: `brick::tests::profiler::test_brick_profiler_reset_v2` | PASS (flake, see Anti-Patterns) |
| Scoped clippy, aprender-train | `cargo clippy -p aprender-train --lib --features setfit --no-deps -- -D warnings` | rc=0 | PASS |
| Scoped clippy, aprender-train incl. tests | `… --lib --tests --features setfit --no-deps -- -D warnings` | rc=0 | PASS |
| Scoped clippy, aprender-core | `cargo clippy -p aprender-core --lib --features setfit --no-deps -- -D warnings` | rc=101, ONE finding: `demo/reliable/performance.rs:126 unreachable expression` — exactly D-ITEM-07, arm64-only, in a file absent from the phase diff | PASS (pre-existing) |
| Debt markers in phase-modified files | `grep -n -E 'TBD\|FIXME\|XXX'` over all 59 changed files | zero matches | PASS |
| Source integrity after the `--in-place` mutation attempts | `rtk proxy git status --porcelain` | only `?? .serena/` — no mutation left behind in source | PASS |

---

## Probe Execution

Phase 3 declares no `scripts/*/tests/probe-*.sh`; its runnable gates are Make targets. All were
run by the verifier standalone with rc captured directly.

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| Cross-process two-clean-runs (AUTHORITATIVE, D-16) | `make setfit-repro-crossproc` | rc=0 — "cross-process: THREADS differed and all ten components agreed" | PASS |
| In-process two-run equality (D-16 fast half) | `make setfit-repro-inproc` | rc=0 — "in-process: every composite component agreed" | PASS |
| GEMM pool-size determinism (D-13) | `make gemm-thread-determinism` | rc=0 — "identical hashes at pool sizes 1, 2 and 3", partitioning observed to differ | PASS |
| Phase 3 contract binding audit (BLOCKING, tier3) | `make contract-audit-phase3` | rc=0 — 22 equations across 3 contracts, all bound and implemented, 0 gaps | PASS |
| setfit feature isolation (D-05/D-06) | `make setfit-feature-matrix` | rc=0 — leg (a) diagnostics byte-identical with and without setfit; two-sided `cargo tree` both directions | PASS (see WARNING on the substituted leg) |
| Contract schema validation | `pv validate` on `lbfgs-kernel-v1`, `multinomial-head-v1`, `setfit-train-lifecycle-v1`, `linear-probe-classifier-v1` | rc=0 each, "0 error(s), 0 warning(s)" | PASS |
| Contract semver drift | `pv diff /tmp/lbfgs-old.yaml contracts/lbfgs-kernel-v1.yaml` (old revision materialized with `git show` first, per CLAUDE.md) | rc=0 — "v1.0.0 -> v1.1.0, Suggested bump: minor", additions only | PASS |
| Scoped mutation run | `cargo mutants … -f <3 scopes>` | NOT RUN — see human_verification item 1 | MISSING (compute authorization) |
| Coverage floor | `make coverage` | NOT RUN — see human_verification item 2 | MISSING (compute authorization) |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| TRN-01 | 03-03, 03-08, 03-10 | Typed SetFit lifecycle, four legal stages | SATISFIED | Truth 1. Sealed typestate + 4 compile-fail proofs |
| TRN-02 | 03-03, 03-10 | Configure and validate 12 knobs before training | SATISFIED (qualified) | Truth 1. The REQUIREMENTS.md qualifier is HONEST and independently confirmed: `MAX_SEQUENCE_LENGTH: usize = 256` (tokenizer.rs:52) and `pub fn encode_texts(&self, texts: &[&str])` (setfit/mod.rs:479) takes no length parameter, so `max_length` is validated-not-configurable. ROADMAP criterion 1's "invalid … length … configuration fails before training begins" IS satisfied via `MaxLengthNotSupported` |
| TRN-03 | 03-05, 03-06, 03-10 | Proof of gradients/deltas/pair-loss before a run may claim SetFit or export | SATISFIED | Truth 2. Gate inside the transition; thresholds independently re-measured |
| TRN-04 | 03-01, 03-04, 03-10 | One deterministic L2-regularized multinomial softmax head, K>=2 | SATISFIED | Truth 4 |
| TRN-05 | 03-07, 03-10 | Head fit exactly once per unique row in eval/no-grad; multiplicity cannot reweight | SATISFIED | Truth 3 |
| TRN-06 | 03-02, 03-05, 03-08, 03-10 | Two clean CPU runs reproduce ten components | SATISFIED | Truth 5. Cross-process rc=0 re-run by verifier; recorded-vs-replay separated |
| TRN-07 | 03-09, 03-10 | Validation-only selection with a selection-lock record before test access | PARTIAL — correctly `[ ]` | Blocking half VERIFIED (truth 5). Positive "a user can reach the lock" tier DEFERRED to Phase 4/5 with specific roadmap evidence. Mechanics fully tested in-crate (77 `lock_` + 45 `evaluate_`) and the negatives are compile-proven from outside the crate |
| SAFE-03 | 03-06, 03-10 | A frozen probe / centroid cannot be labeled SetFit | SATISFIED | Truth 2. Two independent compiler barriers: no `From<FrozenProbeRun>` (E0277) AND `SetFitRun`'s fields private so no out-of-crate conversion can be written (E0451) |

**No ORPHANED requirements.** REQUIREMENTS.md maps exactly TRN-01…TRN-07 + SAFE-03 to Phase 3, and
all eight appear in plan `requirements:` frontmatter (03-10 claims all eight for the closing audit).

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | `TBD` / `FIXME` / `XXX` in any of the 59 phase-modified files | — | NONE FOUND. Debt-marker gate passes |
| `tests/ui/setfit_direct_state_construction.rs`, `setfit_external_codec_impl.rs` | 31-36, 34 | `unimplemented!()` | Info | Legitimate: these are trybuild compile-fail programs that never compile, let alone run. `unimplemented!()` has type `!` so it type-checks every field and leaves the E0451/E0277 diagnostic as the only error |
| `multinomial.rs:940`, `nn/transformer/mod.rs:308`, `setfit/mod.rs:33/1433/1457/1463`, `contracts/setfit-train-lifecycle-v1.yaml:348` | — | the word "placeholder" | Info | Every occurrence is prose or an assertion REQUIRING the absence of a placeholder (e.g. `assert!(!id.contains('?'), "a recorded id must never be a placeholder")`). No placeholder implementation |
| `crates/aprender-compute/src/brick/tests/profiler.rs:301` | 301 | `assert!(profiler.total_ns() > 0)` timing assertion | Info | Measured FLAKE, not a regression. Varied per CLAUDE.md rule 5: 4 isolated runs -> pass, FAIL, pass, pass; single-threaded -> pass. The file is absent from the phase diff, and `blis/parallel.rs` (the only aprender-compute file this phase touched) is an extraction with no behavior change |
| `crates/aprender-train/src/monitor/{mod.rs,tui/}` | mod.rs:45 | `pub mod tui;` declared unconditionally while its only dep is feature-gated | Warning | D-ITEM-05. Makes `cargo check -p aprender-train --no-default-features` rc=101 with 8 errors, which is why plan 03-03's must-have 1 could not be met as written. Pre-existing, outside Phase 3's file set, measured with byte-identical diagnostics with and without `setfit`. Escalated as human decision 3 |
| `crates/aprender-core/src/demo/reliable/performance.rs` | 126 | unreachable expression on aarch64 | Info | D-ITEM-07. Pre-existing, absent from the phase diff, x86 CI does not compile that arm |

**Zero blocker anti-patterns.**

---

## Human Verification Required

No planner-deferred `<verify><human-check>` blocks exist — all 29 tasks carry `<automated>` verify
and 03-VALIDATION.md records "All phase behaviors have automated verification". The three items
below are ESCALATIONS the verifier cannot resolve within its authority.

### 1. Authorize (or decline) the scoped mutation run

**Test:** Authorize a dedicated compute budget, then run the scoped `cargo-mutants` job over
`crates/aprender-train/src/train/setfit/**`, `crates/aprender-core/src/classification/multinomial.rs`
and `crates/aprender-core/src/setfit/dropout_rng.rs`, and record the ADJUSTED score.
**Expected:** Adjusted score >= 85% after excluding only proven-equivalent re-run survivors.
Requires two departures from the plan's literal command, both measured: `--timeout >= 120` for
aprender-core (its 14285-test binary has not finished LINKING at 20s), and tree-copy mode with
`-j` instead of `--in-place` (25.3.1 refuses the combination).
**Why human:** ~44.6h single-job projection exceeds CLAUDE.md's >1hr compute authorization
threshold, and the verifier was explicitly instructed not to start it. The criterion is
**unmeasured**, not failed by implementation. Note: the verifier could not reproduce the
SUMMARY's 1181-mutant inventory — `cargo mutants --list` returned empty output for every `-f`
glob form attempted on this host — so that number should be re-established as part of the run.

### 2. Authorize (or decline) `make coverage`

**Test:** Run `make coverage` on an uncontended target dir and compare against the enforced floor.
**Expected:** Line coverage >= COV_FLOOR (88%) with the Phase 3 surface included.
**Why human:** Same compute authorization gate. The phase added ~600KB of source across
`train/setfit/` with 837 passing scoped tests plus 14285 green aprender-core tests, so a coverage
regression is unlikely — but no number is claimed and none was measured.

### 3. Decide the feature-closure substitution

**Test:** Either accept `make setfit-feature-matrix` leg (a)'s two-sided diagnostic diff as
satisfying plan 03-03 must-have 1 (recording an override), or require the D-ITEM-05 fix in-phase —
gate `crates/aprender-train/src/monitor/mod.rs:45`'s unconditional `pub mod tui;` on
`feature = "tui"` together with its re-exports, then replace the diff leg with the plain check.
**Expected:** Either a recorded override, or `cargo check -p aprender-train --no-default-features
--features setfit` exiting 0.
**Why human:** A scope decision, not a measurement. The literal must-have is measured RED; the
property Phase 3 actually owns is verified by a substitute gate that was itself falsified before
being trusted.

**If this deviation is accepted, add to this file's frontmatter:**

```yaml
overrides:
  - must_have: "cargo check -p aprender-train --no-default-features --features setfit exits 0"
    reason: "Blocked by D-ITEM-05, a pre-existing unconditional `pub mod tui;` in src/monitor/ outside Phase 3's file set. Replaced by make setfit-feature-matrix leg (a), a two-sided diagnostic diff that verifies the property Phase 3 owns (setfit does not leak into the minimal build) without importing an unrelated red. Tracked with a fix direction in deferred-items.md."
    accepted_by: "{name}"
    accepted_at: "{ISO timestamp}"
```

---

## Gaps Summary

**There are no gaps in the phase goal.** All five ROADMAP success criteria are achieved on real,
executed code, and the verifier could not falsify any of them.

What was specifically hunted for and not found:

- **Tautological gates.** The three most suspicious candidates all survive scrutiny. (a) The
  cross-process reproducibility claim could have been a statement about the config file — it is not,
  because the accessors return `table().consumed_pair_digest`, absorbed in the same loop iteration
  that draws each pair, and a SEPARATE test recomputes the expected order from the public protocol
  and compares. (b) The GEMM determinism gate could have passed by never entering the hazard window
  — it does not, because it reports PARTITIONS alongside the hashes and the verifier observed
  1/2/2 across pool sizes 1/2/3; had the partitioning not moved, the harness prints
  `SKIPPED-WITH-EVIDENCE` and explicitly refuses to claim a falsification. (c) The trybuild suite
  could have passed with a glob matching zero files — it does not; all seven cases were observed
  executing by name, and the harness is `#![cfg(feature = "setfit")]` precisely so a default build
  runs zero cases rather than seven fake ones.
- **Snapshots pinning a weaker claim than advertised.** All seven `.stderr` files name real crate
  types, methods or visibility (E0451 ×3 on private fields, E0277 ×2 on trait bounds incl. rustc's
  own "sealed trait" note, E0599 method-not-found, E0061 arity). None is a syntax error, misspelling
  or unresolved import.
- **Thresholds chosen and then blessed.** The frozen per-class epsilons were re-derived from
  scratch by the verifier's own `calibration_matrix_epsilon_basis` run and every one sits inside
  its measured two-sided band. `thresholds_match_the_contract` deserializes the YAML into typed
  structs (not a substring search) with a non-vacuity assert first, and the contract is loaded by
  `include_str!` so a missing file is a build error rather than a skipped test.
- **Count-only proofs of encode-once.** The ledger comparison is a sorted MULTISET, and a companion
  test requires distinct texts to produce distinct embeddings — which is what defeats the
  "one row encoded 24 times" defect that every count check passes.
- **Regressions hidden behind known reds.** The 24 `aprender-train` failures are byte-for-byte the
  documented baseline list; `aprender-core` is 14285/0; the single `aprender-compute` failure was
  varied four ways and is a timing flake in a file this phase did not touch.

**Three items are open, and none of them is an implementation defect.** Two are unmeasured
verification-confidence criteria (adjusted mutation score, coverage floor) blocked behind a compute
budget only the developer may authorize — reported as shortfalls by the executor and, critically,
**not waived by narrative**: no score is claimed anywhere. The third is a scope decision about a
pre-existing defect in a module outside this phase's file set.

**TRN-07 remains correctly unchecked.** The verifier confirms the orchestrator's reading and refines
it: the positive path IS publicly reachable in the library API, but it is EXERCISED only in-crate,
and the "a user can" tier lands naturally in Phase 4 (the `apr` lifecycle surface) and Phase 5
(where "selection lock" is a named field of every benchmark row). ROADMAP criterion 5 asks only that
test access remain BLOCKED until a lock exists, and that half is verified by typestate plus an
out-of-crate compile-fail proof.

**Recommendation:** the phase goal is achieved. Resolve the three escalations, then mark Phase 3
complete. Do not treat the mutation and coverage shortfalls as blockers on Phase 4 unless the
developer decides the confidence floor must be established before dependent work proceeds — they
are measurement debt against an implementation that is otherwise densely and non-vacuously proven.

---

_Verified: 2026-08-12T01:28:03Z_
_Verifier: Claude (gsd-verifier)_
