---
phase: 06-native-time-series-forecasting-stack
plan: 04
subsystem: forecasting
tags: [neuralprophet, autograd, adamw, huber, ar-net, parity, contracts, pv, time-series, rust]

requires:
  - phase: 06-native-time-series-forecasting-stack
    provides: "plan 06-01 landed crates/aprender-forecast (the door, the dispatch site, the refusing `neuralprophet` stub), the 17 byte-verified oracle fixtures including np_oracle_peyton.json, test_support's load_json/read_csv/equation_tolerance/constant_u64, and the MEASURED [profile.dev.package.aprender-forecast] opt-level = 3 decision this plan's wall is measured under"
  - phase: 06-native-time-series-forecasting-stack
    provides: "plan 06-03 established the contract shape to copy (kind: kernel on the setfit-apr-v1 template), the contract-read bar pattern with the name written in full at each call site, and the observed-induced-RED discipline for a qa_gate falsification"
provides:
  - "crates/aprender-forecast/src/np.rs — the NeuralProphet-lite port (20 public symbols) on aprender's f32 autograd, verbatim from sources/004 modulo three clippy/rustc edits"
  - "The real `\"neuralprophet\"` arm inside forecast::forecast, replacing 06-01's refusing tracer stub (REVIEW-06-06) — `model: neuralprophet` now dispatches end to end"
  - "contracts/neuralprophet-parity-v1.yaml — 9 equations (4 carrying float_tolerance), 10 proof obligations, 10 FALSIFY-NP tests, 2 Kani harnesses, qa_gate F-FORECAST-NP-001; pv validate 0 errors, pv status 9/10/10/2"
  - "9 CI-reachable `np::parity` --lib tests: the SC3 accuracy bars and the five D-10 training-rule invariants, every numeric bar contract-read"
  - "4 `forecast::tests::neuralprophet_*` tests: dispatch + band + empty tape, and the three D-11 refusals asserted at the Validation variant"
  - "An OBSERVED induced-RED for F-FORECAST-NP-001: two bars tightened in the YAML alone turn exactly 2 of 9 tests red, each quoting the new bar; a byte-identical revert restores 9/0"
  - "The in-tree seed-42 measurements (lag-free 365-day MAE 0.4463, AR-Net one-step 0.2458) recorded beside the spike's seed-7 numbers"
affects: [06-05, 06-06, 06-07, 06-08, 06-09]

actuals:
  tokens: 22480
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "A ported file names its own bans in prose but never writes the banned identifier, so the D-10 ban stays a file-wide grep rather than a reviewer's memory"
    - "Where the oracle publishes a number that is reproducible (data prep, the auto formulas) bar it EXACTLY; where it publishes only an outcome of an unreproducible procedure (NP's lr-finder, its torch RNG) bar a one-sided threshold instead of a parity residual"
    - "A sweep is an ordered list, so it lives in the contract as a comma-separated string and is read at test time — the same D-15 rule the scalar tolerances follow"
    - "Extending a guard to a NEW file re-proves it there: clippy's reach into np.rs was demonstrated with a RED-turning mutation inside np.rs, not inherited from 06-01's proof on the crate"

key-files:
  created:
    - contracts/neuralprophet-parity-v1.yaml
    - crates/aprender-forecast/src/np.rs
  modified:
    - crates/aprender-forecast/src/forecast.rs
    - crates/aprender-forecast/src/lib.rs

key-decisions:
  - "Gradient connectivity is asserted through `aprender::autograd::get_grad(id)`, not `Tensor::grad()`: backward() accumulates into the tape's grad map and `Optimizer::step_with_params` reads it through get_grad, so get_grad is the ground truth for 'can this loss train anything'. `Tensor::grad()` is a separate field the training path never consults."
  - "The contract's lr sweeps are read by a `constant_lr_sweep` helper LOCAL to np.rs's parity module rather than by a new `test_support` reader, so the plan's declared files_modified set stays honest and 06-05 does not inherit a merge point in test_support.rs"
  - "Two falsification tests beyond the plan's seven (fourier_features_are_phased_on_1900, lr_selection_is_by_train_loss) because PROVABILITY-001 requires falsification_tests >= proof_obligations and the D-10 rules genuinely warranted ten obligations — each new test is runnable, not a declaration"
  - "huber_grad_min_abs is a strict `> 0.0` CONNECTIVITY threshold and is documented as NOT induced-RED-able by tightening: raising it would test gradient MAGNITUDE, a different and seed-dependent property. Its negative is a code mutation, and the contract says so rather than prescribing an impossible experiment (06-03 Deviation 2's lesson)"
  - "No value-parity bar against Python's own MAE. NeuralProphet's lr-finder, torch RNG and epoch schedule are not reproducible from the oracle, so a residual bar there would be barring seed luck; the reproducible half (data prep + the two auto formulas) is barred exactly instead"

patterns-established:
  - "Contract-read bars with the contract name written in full at every call site (4 equation_tolerance + 8 constant_u64 sites), so `grep -c` is a real link check"
  - "Induced-RED observed before the claim is made: 2 bars tightened in YAML alone -> 7 passed / 2 failed quoting the new bars -> byte-identical revert -> 9 passed / 0 failed"
  - "Every clippy/rustc edit to ported code is named in the source, the commit and the SUMMARY with the lint that forced it (the D-08 audit trail 06-01 started)"

requirements-completed: [SC3]

coverage:
  - id: D1
    description: "`model: neuralprophet` dispatches through the verbatim spike arm end to end — a 120-point daily series returns a strictly-ordered residual band, a selected_lr from the contract's sweep, and leaves the autograd tape empty"
    requirement: SC3
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#tests::neuralprophet_arm_dispatches_and_returns_a_band"
        status: pass
      - kind: other
        ref: "grep 'ported in plan 06-04' crates/aprender-forecast/src/forecast.rs — no match; the 06-01 tracer stub sentence is gone"
        status: pass
    human_judgment: false
  - id: D2
    description: "NeuralProphet-lite data preparation matches the committed NeuralProphet 0.9.0 oracle: soft-normalisation shift/scale, t0/span, the 11-segment changepoints_t within 1e-9, and auto batch/epochs exactly 64/80"
    requirement: SC3
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/np.rs#parity::data_prep_matches_np_oracle"
        status: pass
    human_judgment: false
  - id: D3
    description: "The lag-free model reaches 365-day-ahead holdout MAE <= 0.47 on Peyton Manning with the lr selected by TRAIN loss from the contract's {0.01, 0.03, 0.1} sweep (measured 0.4463; Python NeuralProphet 0.4611)"
    requirement: SC3
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/np.rs#parity::lag_free_365_day_mae_within_contract"
        status: pass
    human_judgment: false
  - id: D4
    description: "With n_lags = 30 and hidden [32] the AR-Net's one-step test MAE beats the naive one-step MAE recorded in the oracle (0.2458 vs 0.3461)"
    requirement: SC3
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/np.rs#parity::ar_net_30_lags_beats_naive_one_step"
        status: pass
    human_judgment: false
  - id: D5
    description: "The D-10 training rules are invariant tests: graph-connected Huber, strict mini-batches over the whole supported range, an empty tape after train() and after the predict helpers, the 1900 Fourier epoch, and lr selection by train loss"
    verification:
      - kind: unit
        ref: "np::parity::{weighted_huber_is_graph_connected, auto_batch_is_mini_batch_over_grid, full_batch_is_not_what_auto_batch_returns, train_clears_the_tape, fourier_features_are_phased_on_1900, lr_selection_is_by_train_loss} — 6 tests"
        status: pass
    human_judgment: false
  - id: D6
    description: "Every bar is read from contracts/neuralprophet-parity-v1.yaml at test time (D-15); the contract validates with 0 errors and is not hollow"
    verification:
      - kind: other
        ref: "pv validate contracts/neuralprophet-parity-v1.yaml (rc=0, `0 error(s), 0 warning(s)`); pv status — Equations 9 / Proof obligations 10 / Falsification tests 10 / Kani harnesses 2, no count 0"
        status: pass
      - kind: other
        ref: "OBSERVED induced-RED: lag_free_holdout_mae 0.47->0.40 and ar_net_vs_naive_margin 0.0->0.20 in the YAML alone -> 7 passed / 2 failed, each quoting the new bar; byte-identical `git checkout` -> 9 passed / 0 failed (/tmp/p06-04-induced-red.log, /tmp/p06-04-restored.log)"
        status: pass
      - kind: other
        ref: "grep -c 'equation_tolerance(\"neuralprophet-parity-v1\"' np.rs = 4; 'constant_u64(\"neuralprophet-parity-v1\"' = 8; `! grep -E '0\\.47[^0-9]' np.rs` clean"
        status: pass
    human_judgment: false
  - id: D7
    description: "The three NeuralProphet-specific door refusals (freq other than D, n_lags > 365, n_lags at or above the series span) are Validation-class with fix-naming messages"
    verification:
      - kind: unit
        ref: "forecast::tests::{neuralprophet_refuses_non_daily_freq, neuralprophet_refuses_n_lags_above_365, neuralprophet_refuses_n_lags_at_or_above_span} — each matches on ForecastError::Validation, never Internal"
        status: pass
    human_judgment: false
  - id: D8
    description: "The residual-sd band is honest about what it is: the response's diagnostics say it is not NeuralProphet's quantile regression, and no coverage bar is claimed"
    verification:
      - kind: unit
        ref: "forecast::tests::neuralprophet_arm_dispatches_and_returns_a_band asserts diagnostics.band contains \"not NeuralProphet's quantile regression\""
        status: pass
    human_judgment: true
    rationale: "The test proves the SENTENCE is present, not that the band is adequate. Nominal 80% coverage was measured at 0.60-0.69 on rolling origins in spike 006 (D-16), and no test in this phase asserts empirical coverage. A human should confirm that shipping a demonstrably under-covering band with a disclaimer — rather than deferring the model until quantile regression lands — is acceptable before the phase ships."

duration: 23min
completed: 2026-09-05
status: complete
---

# Phase 06 Plan 04: NeuralProphet-Lite End to End Summary

**`model: neuralprophet` now runs the real spike arm instead of refusing: the port meets NeuralProphet 0.9.0's own data preparation to 1e-16, forecasts Peyton Manning 365 days ahead at MAE 0.4463 against Python's 0.4611, and beats the oracle's naive one-step baseline with a 30-lag AR-Net (0.2458 vs 0.3461) — with all five D-10 training rules pinned as invariant tests whose bars were proven to come from the contract by turning two of them red with a one-line YAML edit.**

## Performance

- **Duration:** 23 min
- **Started:** 2026-09-06T05:40:52Z
- **Completed:** 2026-09-06T06:04:02Z
- **Tasks:** 3
- **Files created/modified:** 4 (2 created, 2 modified)

## Accomplishments

- **`crates/aprender-forecast/src/np.rs`** — the NeuralProphet-lite port: NP trend on 11 segments,
  Fourier seasonality on days since 1900-01-01 (sin block then cos block, NP order), an optional
  AR-Net on stationarised lags, the graph-connected weighted Huber, AdamW with a three-phase
  one-cycle cosine schedule, and NP's auto batch/epoch formulas. Verbatim from
  `sources/004` modulo three lint-forced edits and one import path.
- **The tracer's one stub is gone.** `forecast.rs`'s `"neuralprophet"` arm is now the spike's, and
  `grep 'ported in plan 06-04'` returns nothing. REVIEW-06-06's scope move is closed: NeuralProphet
  was implemented once, in the same plan as its oracle, rather than twice.
- **`contracts/neuralprophet-parity-v1.yaml`** — 9 equations (4 numeric bars + 5 D-10 invariants),
  10 proof obligations, 10 `FALSIFY-NP-0NN` tests each naming its `cargo test` filter, 2 Kani
  harnesses under the DECLARED-NOT-EXECUTED house comment, and `qa_gate F-FORECAST-NP-001`.
  `pv validate` 0 errors; `pv status` 9 / 10 / 10 / 2 — no count is 0.
- **SC3 is met, and by more than the spike projected.** Lag-free 365-day-ahead holdout MAE **0.4463**
  against the 0.47 bar and Python NeuralProphet's 0.4611. AR-Net (30 lags, hidden [32]) one-step MAE
  **0.2458** against the oracle's naive 0.3461 — a 0.10 margin, and worth noting that *Python
  NeuralProphet's own AR-Net (0.3933) does not beat naive on this split*.
- **The D-10 rules are falsifiable, not documented.** Six invariant tests: the Huber is
  graph-connected (every parameter has a non-zero gradient after `backward()`), `auto_batch` is a
  strict mini-batch across 206 grid points spanning the whole supported range, the tape is empty
  after `train()` and after `predict_ts`, the Fourier epoch is 1900 and is distinguishable from the
  Unix epoch, and the door's reported `selected_lr` equals an independent argmin over the contract's
  sweep by TRAIN loss.
- **The contract link was proven, not asserted.** Tightening two tolerances in the YAML — no Rust
  touched — turned exactly the two accuracy tests red, each quoting the new bar; a byte-identical
  revert restored 9/0.

## Task Commits

1. **Task 1: port np.rs verbatim, fill the refusing stub, assert dispatch + the three refusals** — `35bd8859d` (feat)
2. **Task 2: freeze the SC3 bars and the D-10 rules in contracts/neuralprophet-parity-v1.yaml** — `7eb64f524` (feat)
3. **Task 3: the np::parity ladder, contract-read** — `dbee2d1c5` (test)

## Files Created/Modified

| File | What it does |
|---|---|
| `crates/aprender-forecast/src/np.rs` | **NEW, 1163 lines.** The port (`Rng`, `NpSeason`, `np_auto_seasonalities`, `season_dim`, `fourier_feats`, `trend_feats`, `sample_weight`, `NpData`, `NpModel`, `weighted_huber`, `one_cycle_lr`, `auto_batch`, `auto_epochs`, `TrainConfig`, `TrainLog`, `Rows`, `rows_for`, `train`, `predict_ts`, `predict_trend`, `predict_ar_1step`, `predict_ar_recursive`) plus `#[cfg(test)] mod parity` |
| `contracts/neuralprophet-parity-v1.yaml` | **NEW, 355 lines.** `data_prep_vs_oracle_abs` 1e-9, `lag_free_holdout_mae` 0.47, `ar_net_vs_naive_margin` 0.0, `huber_grad_min_abs` 0.0; invariants `fourier_epoch_1900`, `mini_batch_never_full`, `lr_selected_by_train_loss`, `tape_cleared_per_step`, `huber_built_from_ops`; `constants` (both lr sweeps, the 320 epoch cap, the mini-batch clamp, NP's own 64/80, the 1900 epoch) |
| `crates/aprender-forecast/src/forecast.rs` | The verbatim `"neuralprophet"` arm replacing the stub, `use crate::np;`, and a new `#[cfg(test)] mod tests` with the dispatch smoke test and the three D-11 refusals |
| `crates/aprender-forecast/src/lib.rs` | `pub mod np;` and the module doc-comment line that promised it |

## Measured results

### Data preparation vs the NeuralProphet 0.9.0 oracle (bar `data_prep_vs_oracle_abs` = 1.0e-9)

| Quantity | Rust | NeuralProphet 0.9.0 | Status |
|---|---|---|---|
| soft-normalisation shift (min of observed train y) | 5.26269018890489 | `"5.26269018890489"` | within 1e-9 |
| soft-normalisation scale (q95 − min) | 4.582609246193553 | `"4.582609246193553"` | within 1e-9 |
| time origin t0 | 2007-12-10 | `"2007-12-10 00:00:00"` | **exact integer day** |
| train span | 2596 d | `"2596 days 00:00:00"` | within 1e-9 |
| `changepoints_t` (11 entries) | 0.0 … 0.7273 | same | max abs diff within 1e-9 |
| `auto_batch(2540)` | 64 | 64 | **exact integer** |
| `auto_epochs(2540)` | 80 | 80 | **exact integer** |

The two auto formulas are asserted as integer equality, not within a tolerance — they are
reproducible arithmetic, so a tolerance there would be slack with no purpose.

### Accuracy (SC3)

| Rung | Bar | Rust in-tree (seed 42) | Rust in spike (seed 7) | Python NeuralProphet 0.9.0 | Baseline |
|---|---|---|---|---|---|
| lag-free 365-day-ahead MAE | ≤ 0.47 | **0.4463** (lr 0.1, train loss 0.01498) | 0.4510 | 0.4611 (over 363 of the 365 rows) | last-year mean 0.6894 |
| AR-Net(30, [32]) one-step MAE | < naive − 0.0 | **0.2458** (lr 0.1, train loss 0.00829) | 0.2509 | 0.3933 | naive 0.3461 |

The seed-42 numbers are the ones the ladder produces; the spike's seed-7 numbers and its seed
1/2/3 ablation (0.4549 / 0.4515 / 0.4508) are kept in the contract because the seed-to-seed spread
is what sets the headroom under the bar — ~0.009, with the bar ~0.015 above the worst value ever
observed.

**One asymmetry worth stating plainly:** Python NeuralProphet's own AR-Net scores 0.3933 on this
split, which is *worse* than the naive 0.3461 baseline it is compared against. The Rust AR-Net
beats both. The contract records this rather than presenting the comparison as a parity claim.

### The D-10 invariants

| Rule | Test | What it observed |
|---|---|---|
| Huber graph-connected | `weighted_huber_is_graph_connected` | every parameter has `Some(grad)` with an entry `> 0.0` after `backward()` |
| Never full-batch | `auto_batch_is_mini_batch_over_grid` | 206 grid points, n = 64 … 19983 step 97: `auto_batch(n) < n` and inside [8, 2048] everywhere |
| NP's own batch on Peyton | `full_batch_is_not_what_auto_batch_returns` | `auto_batch(2540) == 64` |
| Tape hygiene | `train_clears_the_tape`, and the dispatch smoke test | `graph_tape_len() == 0` after `train()`, after `predict_ts`, and after a completed door call |
| Fourier epoch 1900 | `fourier_features_are_phased_on_1900` | sin block then cos block reproduced from `day − days_from_civil(1900,1,1)`, and distinguishable from a Unix-epoch phasing |
| lr by TRAIN loss | `lr_selection_is_by_train_loss` | the door's `diagnostics.selected_lr` and `final_train_loss` equal an independent argmin over the contract's sweep |

### Ladder timing (RESEARCH Pitfall 9 / F10, D-ITEM-06-03-a)

Root-manifest profile state = **1** (`grep -c '^\[profile\.dev\.package\.aprender-forecast\]' Cargo.toml`),
exactly as 06-01 Task 2 decided; **this plan edited no root manifest** (`git diff 1dc7e20af..HEAD --stat -- Cargo.toml Cargo.lock` is empty).

| Measurement | Value |
|---|---|
| `cargo test -p aprender-forecast --lib np::parity` wall | 46 s |
| of which cargo's `Finished test profile in` | 12.22 s |
| of which test execution (`finished in`) | 34.44 s |
| the same 9 tests run straight from `target/debug/deps/aprender_forecast-*` | 36.65 s |
| whole-crate `cargo test -p aprender-forecast --lib` (60 tests) | 34.63 s of execution |

The ~12 s is the always-rebuild D-ITEM-06-03-a already logged against `aprender-compute` /
`aprender-core` build scripts; subtracting it, the plan's own cost is the 34 s of AdamW steps.
That is the price of running 5 real fits over 2540–2567 samples (3 lag-free × 3200 steps, 2 AR ×
3280 steps) with **`aprender-core`'s autograd compiled at debug opt-level** — the profile override
covers `aprender-forecast` only. Well inside nextest's 1200 s kill, and no manifest change is
proposed here: the ladder is one invocation, and Wave 3 has no reason to pay for a wider override.

## Clippy audit trail on the ported code (D-08)

Against `all + pedantic + -D warnings`, the ported `np.rs` needed **three** mechanical edits and one
import-path change. Nothing else in 384 lines of spike source moved.

| # | Lint | Where | Resolution |
|---|---|---|---|
| 1 | `unused_parens` (rustc) | `train`, `flat_map(\|&i\| (i - l..i))` | dropped the parentheses |
| 2 | `unused_parens` (rustc) | `predict_ar_1step`, same expression | dropped the parentheses |
| 3 | `clippy::int_plus_one` | `predict_ar_recursive`, `while hist_days[len-1] + 1 <= day` | `while hist_days[len-1] < day` — identical over integers |
| — | not a lint | `use crate::prophet::days_from_civil` | `use crate::dates::days_from_civil` (06-01 moved the civil-date helpers into their own module) |

Two further edits were made in the **arm** (`forecast.rs`), both forced by workspace lints rather
than by choice:

| # | Lint | Resolution |
|---|---|---|
| 5 | `clippy::manual_is_none_or` (the `map_or(true, ..)` form) | `best.as_ref().is_none_or(\|b\| fl < b.0)` |
| 6 | `clippy::inconsistent_struct_constructor` (a workspace `warn`) | `yhat_lower` / `yhat_upper` bound to locals BEFORE the struct literal so the literal stays in declaration order; the spike could write them inline only because it moved `yhat` last |

**No file-scope or crate-wide `allow` was added.** `np.rs` carries none at all; `forecast.rs` keeps
06-01's single `#![allow(clippy::disallowed_methods)]` for the `serde_json::json!` expansion.

**Clippy's reach into the NEW file was proven, not inherited** (CLAUDE.md Verification Discipline #4
— extending a guard's scope requires re-mutating in the new scope). Injecting a
`clippy::needless_bool` violation into `np.rs` turned
`cargo clippy -p aprender-forecast --all-targets --no-deps -- -D warnings` red (`rc=101`, cited at
`crates/aprender-forecast/src/np.rs:99`); a byte-identical restore turned it green (`rc=0`).
06-01's proof was on `lib.rs`, and a proof on one file is not a proof on another.

## Decisions Made

**Gradient connectivity is read through `get_grad(id)`, not `Tensor::grad()`.** The plan's
`<behavior>` says "every parameter's `grad()` is `Some`". In this tree `backward()` accumulates into
the tape's gradient map, and `Optimizer::update_param` retrieves it with
`aprender::autograd::get_grad(param.id())` (`crates/aprender-core/src/nn/optim/mod.rs:302`); the
`Tensor::grad` field is a separate slot the training path never consults. Asserting on the field
would have tested something no optimiser reads — the exact shape of mistake this test exists to
catch. The assertion uses `get_grad`, which is also what spike-002's own connectivity probe used.

**The lr sweeps are read by a helper local to `np.rs`'s parity module.** `test_support` carries
`constant_u64` only, and a sweep is an ordered list rather than a scalar. Adding
`constant_str_list` there would have modified a file outside the plan's `files_modified` set and
created a merge point with 06-05, which lands in the same crate in Wave 3. The local
`constant_lr_sweep` reads the same `constants:` block from the same contract, so D-15 is satisfied
without widening the blast radius.

**Ten proof obligations, so ten falsification tests.** `pv validate` enforces
`falsification_tests >= proof_obligations` (PROVABILITY-001), and the first draft failed at 8 vs
10. The fix was two more *runnable* tests (`fourier_features_are_phased_on_1900`,
`lr_selection_is_by_train_loss`) rather than deleting obligations — the D-10 rules genuinely warrant
ten, and a contract that drops an obligation to satisfy an arithmetic check is a weaker contract.

**`huber_grad_min_abs` is documented as NOT induced-RED-able by tightening.** It is a strict
`> 0.0` connectivity threshold; raising it above zero would test gradient *magnitude*, which is
seed-dependent and not what the bar means. The contract states this explicitly and names the code
mutation (route the loss through a detached implementation) as its negative instead. This is
06-03 Deviation 2's lesson applied in advance rather than discovered after: a contract that
prescribes an unrunnable falsification is a contract nobody will falsify.

**No value-parity bar against Python's MAE.** NeuralProphet's lr-finder, its torch RNG and its epoch
schedule cannot be reproduced from the committed oracle, so a residual bar against 0.4611 would be
barring seed luck. The reproducible half — the data preparation and the two auto-formulas — is
barred exactly (to 1e-9 and to integer equality); the accuracy claims are one-sided thresholds
against numbers the oracle published.

**The oracle scores 363 rows where the ladder scores 365, and this is recorded rather than
corrected for.** NeuralProphet drops two rows its own windowing cannot cover. Aligning to 363 would
mean choosing which two Rust rows to discard, which is a decision the oracle does not license. The
assertion message prints both row counts.

## Deviations from Plan

### 1. [Rule 3 - Blocking] The contract needed two more falsification tests than the plan enumerated

- **Found during:** Task 2, first `pv validate` run.
- **Issue:** `PROVABILITY-001: falsification_tests (8) < proof_obligations (10)`. The plan asks for
  "`falsification_tests` FALSIFY-NP-001.. each naming `cargo test ... np::parity::<fn>`" over its
  seven named tests, and for `proof_obligations` ">= 1 per equation" over nine equations — the two
  requirements are not simultaneously satisfiable at seven tests.
- **Fix:** two additional obligations already existed for the Fourier epoch and the lr-selection
  rule; they gained `FALSIFY-NP-009` and `FALSIFY-NP-010`, each naming a NEW runnable test
  (`fourier_features_are_phased_on_1900`, `lr_selection_is_by_train_loss`) written in Task 3.
  Neither is a declaration: the first checks the sin/cos blocks against a 1900-phased recomputation
  and against a Unix-epoch alternative, and the second drives the shipped door and compares its
  reported `selected_lr` / `final_train_loss` to an independent argmin.
- **Effect on the plan's numbers:** 9 `np::parity` tests rather than the plan's 7. The `<verify>`
  bar is `>= 7` and the acceptance criteria name six specific functions — all present.
- **Verification:** `pv validate` 0 errors; `ok. 9 passed; 0 failed; 0 ignored`.
- **Committed in:** `7eb64f524` (contract) and `dbee2d1c5` (tests).

### 2. [Rule 1 - Bug, self-inflicted] The D-10 ban was written into the file it bans, and caught by the plan's own gate

- **Found during:** Task 1, running the `<verify>` negative greps.
- **Issue:** `np.rs`'s module docs and `weighted_huber`'s doc comment named core's detached loss
  type by identifier, to explain why the function exists. The plan's acceptance state is a
  **file-wide** `! grep -q 'SmoothL1Loss'`, and the plan's own `planner-region-allow` note says the
  ban is satisfiable precisely because no task writes that identifier into either source file. The
  gate went red on prose.
- **Fix:** both comments reworded to describe the type ("core's own smooth-L1 (Huber) loss in
  `aprender::nn::loss`") and to say explicitly that it is not named so the ban stays a file-wide
  grep. The explanation is preserved in full; only the greppable token is gone.
- **Why it is recorded rather than quietly fixed:** a ban that a doc comment can trip is a ban whose
  scope was chosen carelessly, and the alternative — weakening the grep to exclude comments — would
  have made a real `use aprender::nn::loss::…` line inside a comment-heavy region invisible. The
  gate is right; the prose was wrong.
- **Verification:** `grep -c SmoothL1Loss` is 0 in both `np.rs` and `forecast.rs`.
- **Committed in:** `35bd8859d`.

### 3. [Rule 1 - Bug] The contract's bars were read through a `CONTRACT` constant, defeating the link grep

- **Found during:** Task 3, checking the acceptance criteria after the first green run.
- **Issue:** the parity module wrote `equation_tolerance(CONTRACT, "…")` with
  `const CONTRACT: &str = "neuralprophet-parity-v1"`. That reads the right file at runtime, but
  `grep -c 'equation_tolerance("neuralprophet-parity-v1"'` returned **0** — the acceptance criterion
  demands `>= 3`, and 06-03 established the reason: the criterion is a *static link check*, and a
  constant makes the link invisible to it.
- **Fix:** the contract name is written in full at all 12 call sites (4 `equation_tolerance`,
  8 `constant_u64`) and the constant deleted. Byte-for-byte the same reads; now greppable.
- **Verification:** 4 and 8 respectively; the full ladder re-ran green afterwards.
- **Committed in:** `dbee2d1c5`.

### 4. [Rule 1 - Bug] The contract quoted only the spike's numbers as "measured"

- **Found during:** Task 3, after the first full ladder run.
- **Issue:** the contract (written in Task 2, before any in-tree fit had run) said "Rust measured
  0.4510" and "Rust measured 0.2509". Those are spike-002's seed-7 numbers. The ladder runs at seed
  42 and produces 0.4463 and 0.2458. A number in a contract is indistinguishable from an in-tree
  measurement unless it says otherwise — the same failure 06-01 recorded as its Deviation 6.
- **Fix:** the description, both equation `invariants` blocks and both falsification `prediction`
  texts now carry the in-tree seed-42 numbers labelled as such, with the spike numbers kept beside
  them because the seed spread is what justifies the headroom.
- **Committed in:** `dbee2d1c5`.

### 5. [Improvement on plan] The qa_gate falsification records the negative that was OBSERVED

- **Found during:** Task 3, running the induced-RED control.
- **What changed:** the `qa_gate.falsification` text was written in Task 2 as a *prescription*
  ("set the tolerance to 0.40 …"). After running it, the text was rewritten to the experiment
  actually performed, with the two failure messages quoted verbatim and the restore result stated.
- **Why:** 06-03 Deviation 2 found a prescribed falsification that could not be performed at all.
  Recording only what was run makes that class of error impossible to reintroduce.
- **Committed in:** `dbee2d1c5`.

---

**Total deviations:** 5 — 1 Rule 3 blocking (the PROVABILITY-001 arithmetic), 3 Rule 1 bugs (the
self-tripped D-10 grep, the constant that defeated the link check, the un-labelled spike numbers),
1 improvement to a contract text after observing the experiment it describes.
**Impact on plan:** No scope creep and no loosened bar. Two tests were ADDED beyond the plan's
seven, both runnable; every named acceptance criterion is met; the root manifest and `Cargo.lock`
are untouched.

## Gates run

| Gate | Result |
|---|---|
| `cargo test -p aprender-forecast --lib forecast::` | **ok. 4 passed; 0 failed**, 4 `forecast::tests::neuralprophet_` lines |
| `cargo test -p aprender-forecast --lib np::parity` | **ok. 9 passed; 0 failed; 0 ignored**, 9 `np::parity::` lines |
| `cargo test -p aprender-forecast --lib` | **ok. 60 passed; 0 failed; 0 ignored** (47 before this plan) |
| `pv validate contracts/neuralprophet-parity-v1.yaml` | **rc=0**, `0 error(s), 0 warning(s)` |
| `pv status contracts/neuralprophet-parity-v1.yaml` | Equations **9**, obligations **10**, falsification tests **10**, Kani harnesses **2**, qa_gate `F-FORECAST-NP-001` — no count is 0 |
| Induced-RED through the contract alone (0.47→0.40, 0.0→0.20) | **7 passed / 2 failed**, each quoting the new bar; byte-identical revert → **9 passed / 0 failed** |
| `cargo clippy -p aprender-forecast -p aprender-mcp-forecast --all-targets --no-deps -- -D warnings` | **rc=0**; reach into `np.rs` proven by a `needless_bool` probe (rc=101, `np.rs:99`) and a byte-identical revert (rc=0) |
| `cargo fmt --all -- --check` | **rc=0** |
| `cargo check --workspace --exclude aprender-profile` | **rc=0**, 0 errors |
| Stub sentence `ported in plan 06-04` | **gone** from `forecast.rs` |
| `SmoothL1Loss` in `np.rs` / `forecast.rs` | **0 / 0** |
| `0.47` literal in `np.rs` | **absent** — the MAE bar is contract-read |
| Root-manifest / lockfile guard (`1dc7e20af..HEAD`) | both diffs empty; profile state = 1 |
| Task 1 / 2 / 3 acceptance criteria | **17/17**, **11/11**, **13/13** |

`--no-deps` and `--exclude aprender-profile` are 06-01's recorded, measured scopings
(D-ITEM-06-01-b and the Darwin `compile_error!`), not this plan loosening a gate.

## Issues Encountered

- **The `rtk` Bash hook rewrites `cargo test` and strips both the `test result:` line and `eprintln!`
  output**, exactly as 06-01 and 06-03 recorded. Every verification command here ran through
  `rtk proxy cargo …`, and the per-rung measured numbers were captured by invoking
  `target/debug/deps/aprender_forecast-*` directly with `--nocapture --test-threads=1`.
- **`.pv/contracts.idx` did NOT go dirty this session**, unlike 06-03's run. `pv validate` and
  `pv status` left the tracked index untouched here, so nothing needed restoring. Recorded because
  06-03's SUMMARY warns re-runners to expect the opposite; the behaviour is evidently not
  deterministic across runs.
- **The parity ladder costs 34 s of execution against 1.7 s for the Prophet ladder.** The reason is
  not the tests but the profile: `aprender-core`'s autograd — where every AdamW step actually runs —
  is compiled at debug opt-level, since 06-01's override is scoped to `aprender-forecast`. Left
  as-is deliberately (see the timing section); flagged for anyone tempted to widen the override.

## Known Stubs

**None introduced by this plan, and one REMOVED.** 06-01's `model: "neuralprophet"` refusing stub —
the only stub the phase carried — is gone, replaced by the real arm. Every value the response
returns is computed: `yhat` from `predict_ts` or `predict_ar_recursive`, `trend` from
`predict_trend`, the band from the fitted residual standard deviation, and every `diagnostics` key
from the `TrainLog` the fit produced.

Two things are deliberately NOT asserted, and are recorded rather than hidden:

- **Empirical band coverage.** The band is residual-sd based; nominal 80 % covered 0.60–0.69 on
  rolling origins (spike 006, D-16). The response says in words that this is not NeuralProphet's
  quantile regression, and a test asserts that sentence is present — but no test asserts coverage,
  and the contract explicitly forbids adding a coverage bar until quantile regression ships.
  Surfaced for human sign-off as `coverage.D8`.
- **Value parity against Python's MAE.** Barred as a one-sided threshold, not a residual — see
  Decisions.

## Threat Flags

None. This plan adds one YAML contract, one library module and test code. `Cargo.lock` is
byte-identical to the plan-start commit (T-06-SC satisfied by construction, verified rather than
assumed), no network surface is introduced, and no file-access pattern beyond reading the committed
fixtures and the contract that 06-01's `test_support` already read.

The two registered threats this plan owns are both mitigated and now observable:

- **T-06-02 (DoS via `n_lags`)** — `n_lags <= 365`, `n_lags < the training grid span` and
  `epochs = auto_epochs(n).min(320)` are all enforced at the door, and the first two now have
  Validation-class tests (`neuralprophet_refuses_n_lags_above_365`,
  `neuralprophet_refuses_n_lags_at_or_above_span`). `MAX_POINTS` is 06-01's and unchanged.
- **T-06-12 (tape tampering)** — `clear_graph()` after every step is pinned by
  `train_clears_the_tape` and by the door-level `graph_tape_len() == 0` assertion in the dispatch
  smoke test. The `spawn_blocking` isolation and the pool equality test remain 06-06's.

## User Setup Required

None — no external service configuration, no new dependency, no credentials.

## Next Phase Readiness

**Ready for Wave 3 (06-05).** What it inherits:

- **`crates/aprender-forecast/src/lib.rs`'s `pub mod` block now carries `np`**, immediately above
  where 06-05 appends `bolt`, `safetensors` and `chronos`. This is the serialisation REVIEW-06-06
  accounted for by moving 06-05 to Wave 3; the block is otherwise untouched.
- **`contracts/neuralprophet-parity-v1.yaml` is a second worked example of the contract shape** —
  and specifically the first one carrying *invariant* equations that are not tolerances, plus a
  `constants:` entry read as a comma-separated list. A Chronos contract needing a quantile list can
  copy `constant_lr_sweep`.
- **The root `Cargo.toml` remains closed for this phase**, now verified across two waves.
- **One number to watch:** the parity ladder's 34 s is 20× the Prophet ladder's, and it is
  `aprender-core`'s debug autograd, not the tests. If 06-05's Chronos forward runs through the same
  autograd in tests, the phase's total `--lib` wall will grow super-linearly. Measure before
  assuming; do not widen the profile override without a measured wall (06-01's rule).

---
*Phase: 06-native-time-series-forecasting-stack*
*Completed: 2026-09-05*

## Self-Check: PASSED

All 4 `key-files` entries exist on disk (`contracts/neuralprophet-parity-v1.yaml`,
`crates/aprender-forecast/src/np.rs`, `crates/aprender-forecast/src/forecast.rs`,
`crates/aprender-forecast/src/lib.rs`), and all three task commits (`35bd8859d`, `7eb64f524`,
`dbee2d1c5`) are present in `git log --all`.
