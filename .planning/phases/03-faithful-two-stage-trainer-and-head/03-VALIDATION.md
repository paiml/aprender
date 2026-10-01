---
phase: 3
slug: faithful-two-stage-trainer-and-head
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-08-09
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Populated from 03-RESEARCH.md "Validation Architecture" + the 29 tasks in plans 03-01…03-10.
> Revision 2 (phase-3 cross-AI review): 03-08's lock work became plan 03-09 and the closing
> plan shifted to 03-10 / wave 7, so task IDs 3-09-01…03 now denote the SelectionLock tasks
> and 3-10-01…03 the closing gates. Task count 26 -> 29; every task still carries `<automated>`.
> Revision 3 (final plan-checker pass): task IDs and counts UNCHANGED (29). The calibration
> matrix moved from a non-compiling out-of-crate `--test setfit_calibration` target to an
> `#[ignore]`-marked in-crate lib test (N-01); clippy legs corrected to the crates each task
> actually edits (N-06); tier2 gained `setfit-repro-inproc` per D-16 (W-10).
> Revision 2b (plan-checker pass): task IDs and counts UNCHANGED. Commands changed only by
> ADDITION — a scoped clippy leg on every plan's final task (W-08), the new
> the calibration matrix split out of the `evidence_` filter (W-11 — its placement was corrected in revision 3), and PMAT
> complexity/SATD plus `make coverage` in the closing audit (W-07).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (libtest) + proptest 1 + trybuild 1 + cargo-mutants (scoped) — all pre-existing (Phase 2 used each) |
| **Config file** | per-crate `Cargo.toml`; `.clippy.toml` (unwrap ban); `.pmat-gates.toml` |
| **Quick run command** | `cargo test -p aprender-train --lib --features setfit` AND `cargo test -p aprender-core --lib --features setfit` |
| **Full suite command** | `cargo test --workspace --lib --exclude aprender-profile` (Darwin form, STATE.md) |
| **Estimated runtime** | seconds-scale for every scoped `--lib` filter; end-to-end is Rust-compile-bound (~1–3 min scoped, `CARGO_INCREMENTAL=0` per STATE.md ENOSPC mitigation). **Two runs are deliberately NOT seconds-scale and are kept OFF the default filters** (revision 3): the `#[ignore]`-marked `calibration_matrix_epsilon_basis` lib test runs ≥12 complete tuning passes over the real-weight MiniLM slice and executes only under `-- --ignored`, and 03-10 T3's mutation + coverage run is the phase's heaviest. 03-05 T3 records the measured wall time of both `evidence_` and `setfit_calibration`; if the `evidence_` filter exceeds 30 s the split has failed and the matrix has leaked back onto the default path. |

**Command form (contracted — CLAUDE.md Verification Discipline rule 1, shipped broken twice: #2336, #2360):**
Every `<automated>` verify command in every plan runs with DIRECT rc capture, never through a pipe:

```
cmd > /tmp/p3-NNtM.log 2>&1; rc=$?; tail -3 /tmp/p3-NNtM.log; exit $rc
```

Multi-leg commands chain with `[ $rc -eq 0 ] || exit $rc` between legs. The table below shows the
underlying invocations; each executes in this rc-capture form (log names `/tmp/p3-<plan>t<task>[a|b|c].log`
are unique per plan/task so parallel-wave runs cannot collide). Phase 2 lesson 02-06 also applies
verbatim: every contract falsification command carries `--lib` (or a concrete `--test` target) —
bare filter forms pass vacuously.

---

## Sampling Rate

- **After every task commit:** Run that task's scoped `<automated>` command (rc-capture form). Every plan's FINAL task chains a scoped `cargo clippy … -- -D warnings` leg (revision 2 / W-08): `.clippy.toml` bans `unwrap()` and workspace lints set `pedantic = "warn"`, so deferring clippy to wave 7 would mean re-touching modules across all seven waves.
- **After every plan wave:** Run both quick run commands + `cargo check -p aprender-train --no-default-features --features setfit` (feature-closure leg; aprender-core leg for waves touching it). **Wave 2 additionally runs `cargo test -p aprender-train --lib --features setfit calibration_matrix -- --ignored --nocapture`** — the calibration matrix is an `#[ignore]`-marked IN-CRATE lib test, so it never runs on a developer's default `evidence_` filter (revision 3 / N-01; revision 2's out-of-crate `--test setfit_calibration` target could not compile against `pub(crate)` `run_tuning` and `#[cfg(test)]` `calibration_variants`).
- **Before `/gsd:verify-work`:** Full suite green (`cargo test --workspace --lib --exclude aprender-profile`) + scoped clippy on aprender-train and aprender-core with `--features setfit -- -D warnings` + the four Phase 3 tier3 targets standalone with rc recorded: `contract-audit-phase3`, `setfit-repro-crossproc`, `gemm-thread-determinism`, `setfit-feature-matrix`
- **Max feedback latency:** seconds once compiled; no watch modes anywhere (compile time is the floor, accepted for a Rust workspace)

---

## Per-Task Verification Map

*Commands shown are the underlying invocations; all run in the rc-capture form defined above.*

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 01 | 1 | TRN-04 | T-3-43 / T-3-48 | branch PRE-EXISTS (checked, never created); f32 golden trajectory frozen as bit patterns before the widening; public LBFGS stays non-generic | unit + source assertion | `cargo test -p aprender-core --lib optim::` + `cargo clippy -p aprender-core --lib -- -D warnings` | ✅ existing optim suite (golden test authored in-task) | ⬜ pending |
| 3-01-02 | 01 | 1 | TRN-04 | T-3-01 / T-3-02 | four-channel x two-width non-finite matrix returns NumericalError, never panics; contract drift visible via pv diff vs materialized old revision | unit (falsify f64 twins + non-finite matrix) + pv | `cargo test -p aprender-core --lib lbfgs` + `pv validate contracts/lbfgs-kernel-v1.yaml` + `cargo clippy -p aprender-core --lib -- -D warnings` | ✅ tests_lbfgs_contract.rs (twins authored in-task, tests-first) | ⬜ pending |
| 3-02-01 | 02 | 1 | TRN-06 | T-3-04 / T-3-49 | StdRng path removed from SetFit route; forward-ordinal coordinate present with typed overflow; feature closure holds | unit | `cargo test -p aprender-core --lib --features setfit setfit::` + `cargo check -p aprender-core --no-default-features` | ✅ Ph1 setfit suite (mode tests must survive) | ⬜ pending |
| 3-02-02 | 02 | 1 | TRN-06 | T-3-06 / T-3-07 / T-3-40 | branch-independence Hamming band (A vs B at one step); exact floor(p*2^64) threshold golden; p=1-1e-40 typed error | unit + golden (tdd) | `cargo test -p aprender-core --lib --features setfit dropout_rng` | in-task (tests-first) | ⬜ pending |
| 3-02-03 | 02 | 1 | TRN-06 | T-3-05 / T-3-45 | thread-count independence MEASURED at FIXED pools 1/2/3 with a PARTITIONS mechanism-engaged assert; contingent compute fix ships contracted | integration (subprocess) | `cargo test -p aprender-core --test gemm_thread_determinism` + `cargo clippy -p aprender-core --lib --tests --features setfit -- -D warnings` + `cargo clippy -p aprender-compute --lib -- -D warnings` | in-task | ⬜ pending |
| 3-03-01 | 03 | 1 | TRN-01 | T-3-09 / T-3-43 | states unforgeable (sealed trait, private fields, PhantomData); run holds its own dataset; maximal CPU-safe feature leg green with a no-setfit control | feature-closure check + source assertion | `cargo check -p aprender-train --no-default-features --features setfit` + default leg + `make setfit-feature-matrix` | ✅ compiler | ⬜ pending |
| 3-03-02 | 03 | 1 | TRN-02 | T-3-08 / T-3-36 / T-3-50 | all 12 knobs fail closed with knob-naming errors AND the same validation runs on the serde path (try_from wire type); resolved device never deserializable | unit (FALSIFY case tables + invalid-payload deserialization, tdd) | `cargo test -p aprender-train --lib --features setfit config_` | in-task (tests-first) | ⬜ pending |
| 3-03-03 | 03 | 1 | TRN-02 / TRN-06 | T-3-10 / T-3-11 / T-3-51 | epoch order pure fn of (seed, epoch) under own tag; warmup uses ceil per HF reference with a ceil-vs-round divergence test; LR at step 0 is exactly 0.0 | unit + independent golden | `cargo test -p aprender-train --lib --features setfit reduce_` / `epoch_` / `--lib warmup_linear` + `cargo clippy -p aprender-train --lib --features setfit -- -D warnings` | in-task | ⬜ pending |
| 3-04-01 | 04 | 2 | TRN-04 | T-3-12 / T-3-15 / T-3-52 / T-3-53 | 16 enumerated invalid inputs typed-rejected pre-solve; f64 logit accumulation with typed NonFiniteLogit; intercept gauge asserted centered; Duration-free report | unit (tdd) | `cargo test -p aprender-core --lib multinomial` | in-task (tests-first) | ⬜ pending |
| 3-04-02 | 04 | 2 | TRN-04 | T-3-46 / T-3-13 / T-3-14 | central-difference gradient suite over {K=2,K=3}x{λ=0,λ>0} incl. intercept exclusion; factor-2 lambda drift caught; PEP 723 self-pinning fixture provably converged | gradient suite + fixture falsification pair (tdd) | `cargo test -p aprender-core --lib multinomial_contract` | in-task | ⬜ pending |
| 3-04-03 | 04 | 2 | TRN-04 | T-3-13 | blocking contract audit proven falsifiable (induced red recorded); analytic_gradient equation bound | pv + make gate | `pv validate contracts/multinomial-head-v1.yaml` + `make contract-audit-phase3` + `cargo clippy -p aprender-core --lib -- -D warnings` | in-task (contract authored here) | ⬜ pending |
| 3-05-01 | 05 | 2 | TRN-03 / TRN-06 | T-3-18 | synthetic-text-only fixture (grep gate: no TweetEval content); calibration_variants matrix available | unit | `cargo test -p aprender-train --lib --features setfit fixture_` | in-task | ⬜ pending |
| 3-05-02 | 05 | 2 | TRN-03 / TRN-06 | T-3-39 / T-3-38 / T-3-54 / T-3-55 / T-3-19 | pinned step order (zero_grad + scheduled LR before step) with a skip-zero_grad negative; digests absorbed AT CONSUMPTION with a reversed-consumption negative; registry hash asserted per step; device/max_length consumed | unit (order pin + two negatives) | `cargo test -p aprender-train --lib --features setfit tune_` | in-task | ⬜ pending |
| 3-05-03 | 05 | 2 | TRN-03 | T-3-16 / T-3-17 / T-3-42 | relative delta finite at zero init (floored, support-restricted denominator); per-class classification fail-closed; calibration matrix separation across ≥3 seeds x ≥2 boundary configs | unit (hash-binding + zero-init) + `#[ignore]`-gated in-crate calibration test | `cargo test -p aprender-train --lib --features setfit evidence_` + `cargo test -p aprender-train --lib --features setfit calibration_matrix -- --ignored --nocapture` + `cargo clippy -p aprender-train --lib --features setfit -- -D warnings` + `cargo clippy -p aprender-core --lib --features setfit -- -D warnings` | in-task | ⬜ pending |
| 3-06-01 | 06 | 3 | TRN-03 | T-3-21 / T-3-42 | per-class epsilon/scale floors, k, margin AND the calibration regime frozen in a pv-validated contract BEFORE any judgment; linear-probe contract added to PHASE3_CONTRACTS | pv | `pv validate contracts/setfit-train-lifecycle-v1.yaml` | in-task (contract authored here) | ⬜ pending |
| 3-06-02 | 06 | 3 | TRN-03 / SAFE-03 | T-3-20 / T-3-22 / T-3-42 / T-3-56 | gate inside the transition; frozen, 1e-30-LR and uncalibrated-regime negatives red in every cargo test; failure carries the full evidence table; Rust constants PARSED from the YAML | unit (negative/control/mirror + threshold provenance, tdd) | `cargo test -p aprender-train --lib --features setfit evidence_gate` + `negative_` + `thresholds_` | in-task (negatives RED-observed first) | ⬜ pending |
| 3-06-03 | 06 | 3 | SAFE-03 | T-3-20 / T-3-57 | FrozenProbeRun never claims SetFit (kind string + no-conversion grep gate + no graph built); its bindings proven reachable by the scoped audit | unit + make gate (induced red) | `cargo test -p aprender-train --lib --features setfit baseline_` + `make contract-audit-phase3` + `cargo clippy -p aprender-train --lib --features setfit -- -D warnings` | in-task | ⬜ pending |
| 3-07-01 | 07 | 4 | TRN-05 | T-3-44 / T-3-24 / T-3-58 | encode ledger multiset equals selection ids exactly once (row count alone insufficient); no_grad/detach mechanism proven via graph_tape_len and grad snapshots; label order from label_names() | unit (tdd) | `cargo test -p aprender-train --lib --features setfit head_input_` | in-task (tests-first) | ⬜ pending |
| 3-07-02 | 07 | 4 | TRN-05 | T-3-25 | one resolve_lambda against unique rows (exact 1/48 pin) + pair-budget invariance of the fitted weights | unit (tdd) | `cargo test -p aprender-train --lib --features setfit fit_head` | in-task | ⬜ pending |
| 3-07-03 | 07 | 4 | TRN-05 | T-3-23 | pair-multiplicity adversary red in every cargo test at a SHARED lambda (unconfounded control); inexpressible via fit_head | in-band negative/control/mirror | `cargo test -p aprender-train --lib --features setfit pair_weight` + `cargo clippy -p aprender-train --lib --features setfit -- -D warnings` | in-task | ⬜ pending |
| 3-08-01 | 08 | 5 | TRN-01 | T-3-26 / T-3-41 / T-3-59 | bundle rebuilds a BIT-IDENTICAL encoder from bytes (tokenizer bytes + architecture retained); four bounded-deserialization limits; f32 exactness (subnormal, -0.0) | unit | `cargo test -p aprender-core --lib --features setfit tokenizer_bytes` + `cargo test -p aprender-train --lib --features setfit bundle_` | in-task | ⬜ pending |
| 3-08-02 | 08 | 5 | TRN-01 / TRN-06 | T-3-35 / T-3-38 | codec is a sealed three-method pure codec (no hash/compare/tolerance); EchoCodec negative; accessors return RECORDED digests (perturbed-consumption companion) | unit (tdd) | `cargo test -p aprender-train --lib --features setfit verify_` | in-task | ⬜ pending |
| 3-08-03 | 08 | 5 | TRN-01 | T-3-26 / T-3-59 | reload/bundle/limit obligations contracted incl. what a codec may NOT do | pv + make gate | `pv validate contracts/setfit-train-lifecycle-v1.yaml` + `make contract-audit-phase3` + `cargo clippy -p aprender-train --lib --features setfit -- -D warnings` + `cargo clippy -p aprender-core --lib --features setfit -- -D warnings` | in-task | ⬜ pending |
| 3-09-01 | 09 | 6 | TRN-07 | T-3-37 / T-3-29 | ValidationEvaluation produced ONLY by a trusted evaluator taking the verified run + Split<Validation>; no API accepts a metric value; split fingerprint committed | unit (tdd) | `cargo test -p aprender-train --lib --features setfit evaluate_` | in-task (tests-first) | ⬜ pending |
| 3-09-02 | 09 | 6 | TRN-07 | T-3-27 / T-3-28 / T-3-34 / T-3-60 / T-3-61 | mint_test_token takes the run OBJECT (no [u8;32]); lock commits full candidate history and applies the rule; lock-then-tune-then-test yields StaleLock; grant re-checks identity | unit (tdd) | `cargo test -p aprender-train --lib --features setfit lock_` | in-task | ⬜ pending |
| 3-09-03 | 09 | 6 | TRN-07 | T-3-27 / T-3-37 | evaluation/lock/token obligations contracted; audit proven to reach the new bindings (induced red) | pv + make gate | `pv validate contracts/setfit-train-lifecycle-v1.yaml` + `make contract-audit-phase3` + `cargo clippy -p aprender-train --lib --features setfit -- -D warnings` | in-task | ⬜ pending |
| 3-10-01 | 10 | 7 | TRN-01 / TRN-07 | T-3-31 | seven compile-fail proofs incl. asserted-metric-value and external-codec-impl; each .stderr names a real type/method and is git-tracked | trybuild compile-fail | `cargo test -p aprender-train --test ui --features setfit` | in-task | ⬜ pending |
| 3-10-02 | 10 | 7 | TRN-06 | T-3-30 / T-3-32 / T-3-38 | cross-process hash equality over RECORDED digests at FIXED pools 1/3, child spawned by exact test name, THREADS-differ assert; SEPARATE recorded-vs-expected-replay test; rc captured directly in both Make targets | integration (subprocess) + make gates | `cargo test -p aprender-train --test setfit_repro --features setfit` + `make setfit-repro-crossproc` + `make setfit-repro-inproc` | in-task (public accessors from 03-08 only) | ⬜ pending |
| 3-10-03 | 10 | 7 | TRN-01…07 + SAFE-03 (closing audit) | T-3-33 / T-3-62 | ADJUSTED mutation score ≥85% after excluding only proven-equivalent re-run survivors; full workspace suite + scoped clippy + PMAT complexity/SATD + coverage floor in the audit; TRN-02 booked with the max_length validated-not-configurable qualifier; free-disk headroom checked before the heavy run | scoped cargo-mutants + full-suite audit | `cargo mutants --no-times --timeout 20 --in-place -f <scope>` (3 scopes) + `cargo test --workspace --lib --exclude aprender-profile` + scoped clippy + `pmat analyze complexity` / `pmat analyze satd` (3 scopes) + `make coverage` + full closing-audit list in 03-10 T3 | ✅ tooling exists | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*
*"in-task" = the test file/module is authored inside the same task that implements the behavior (TDD tasks write tests first); no task depends on a test file another plan was supposed to scaffold.*

**Sampling continuity:** every one of the 29 tasks carries an `<automated>` verify — there is no
run of even two consecutive tasks without automated feedback.

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. No Wave 0 scaffold plan is needed:

- **Framework present:** libtest (cargo test), proptest 1 and trybuild 1 are workspace deps, cargo-mutants is installed — Phase 2 exercised all of them.
- **Tests authored inline:** every task writes its own tests in the same task (tdd tasks write them first); no plan references a test file that some earlier scaffold was supposed to create, so there are no `MISSING` `<automated>` entries.
- **The one genuinely missing surface** — the `crates/aprender-train/src/train/setfit/` module tree and the aprender-train `setfit` feature — is created by plan 03-03 Task 1 in wave 1, before any command that depends on it runs; wave ordering enforces the dependency.
- RESEARCH's "Wave 0 Gaps" list maps onto scheduled plans, not a Wave 0: module tree + feature → 03-03 T1; f64 L-BFGS widening → 03-01; GEMM falsification harness → 03-02 T3; contracts + `PHASE3_CONTRACTS` + blocking audit target → 03-04 T3 (grown in 03-06/03-08/03-09); sklearn pinned fixture → 03-04 T2; trybuild ui cases → 03-10 T1. The bytes-reconstruction surface `SetFitMiniLm::from_bundle_parts` and the `graph_tape_len` observation accessor are also new aprender-core surfaces, created in 03-08 T1 and 03-05 T2 respectively, each before the wave that consumes it.

---

## Manual-Only Verifications

All phase behaviors have automated verification. (SUMMARY evidence recording — e.g. the epsilon
basis numbers, induced-red observations, mutation table — is documentation of automated runs,
not manual testing.)

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies (29/29 have `<automated>`; zero Wave 0 dependencies)
- [x] Sampling continuity: no 3 consecutive tasks without automated verify (density is 29/29)
- [x] Wave 0 covers all MISSING references (none exist — see Wave 0 section)
- [x] No watch-mode flags
- [x] Feedback latency: seconds-scale test execution once compiled; compile-bound end-to-end accepted for Rust (no faster path exists; no watch modes)
- [x] `nyquist_compliant: true` set in frontmatter
- [x] All `<automated>` commands use direct rc capture — no `| tail` pipe forms remain (revision 1 fix; CLAUDE.md Verification Discipline rule 1)

**Approval:** approved 2026-08-09 (revision 3 — N-01 test-placement correction, N-06 clippy
targeting, W-10 tier2 wiring; the 29-row map and every task ID are unchanged). Earlier: (revision 2b — clippy-per-plan, calibration-target split, and
PMAT/coverage legs added; task IDs and the 29-row map are unchanged). Earlier: (revision 1 — populated alongside the rc-capture rewrite so the
contract records the corrected command forms). Revision 2, 2026-08-09 — re-synced to the 29-task /
10-plan / 7-wave structure produced by the cross-AI review replan; every command retains the direct
rc-capture form and the full-suite + scoped-clippy legs were added to the pre-verify sampling row.
