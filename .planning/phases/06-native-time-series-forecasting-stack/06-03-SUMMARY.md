---
phase: 06-native-time-series-forecasting-stack
plan: 03
subsystem: testing
tags: [prophet, parity, contracts, provable-contracts, pv, tolerances, lbfgs, time-series, rust]

requires:
  - phase: 06-native-time-series-forecasting-stack
    provides: "plan 06-01 landed crates/aprender-forecast (the Prophet 1.4.0 port), the 17 byte-verified oracle fixtures, test_support's equation_tolerance/constant_u64 readers, the single rung-2 Peyton test carrying the last 1e-9 literal, and the MEASURED [profile.dev.package.aprender-forecast] opt-level = 3 decision this plan's ladder wall is measured under"
provides:
  - "contracts/prophet-parity-v1.yaml — 12 equations, 11 float_tolerance bars and a constants map, on the setfit-apr-v1 shape (kind: kernel); pv validate 0 errors, pv status 12/12/2"
  - "32 CI-reachable `prophet::parity` --lib tests over ALL SEVEN committed Prophet 1.4.0 fixtures, one test per rung per fixture"
  - "D-15 closed: the tracer's last tolerance literal is gone; 11 equation_tolerance + 2 constant_u64 call sites, guarded by a region-scoped literal scan that was PROVEN to fire"
  - "D-09 diagnostics asserted on every fixture: rounds <= 8, status Stalled|Converged, budget_hit false"
  - "The measured warm ladder wall (14 s via cargo / 1.73 s of test execution) and the finding that 06-01's 91 s projection was a per-invocation-rebuild artefact"
  - "One observed induced-RED for the F-FORECAST-PROPHET-001 qa_gate (7 tests red under a contract-only tolerance change, byte-identical revert restores green)"
affects: [06-04, 06-05, 06-06, 06-08, 06-09]

actuals:
  tokens: 19120
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Every numeric bar a parity test asserts is READ from contracts/<name>.yaml at test time; the contract name is written at each call site so a static grep proves the link"
    - "A guard is only a guard once it has been seen RED: the literal scan, the clippy gate and the contract-read link were each proven by a mutation and a byte-identical revert"
    - "Where the oracle publishes no number, write NO test rather than a Rust-vs-Rust one, and record the asymmetry in the contract"
    - "Expensive per-fixture work (the MAP fit, the Python-params forecast) is memoised in a per-fixture OnceLock so N rungs cost one fit, not N"

key-files:
  created:
    - contracts/prophet-parity-v1.yaml
  modified:
    - crates/aprender-forecast/src/prophet.rs
    - .planning/phases/06-native-time-series-forecasting-stack/deferred-items.md

key-decisions:
  - "objective_at_python_map binds the THREE spike-001 fixtures only — the four spike-003 files publish no log_posterior_at_map_unnormalized, so a seventh test would have compared Rust to Rust; the contract states the asymmetry and the spike-003 fixtures are held by fitted_objective_slack plus the whole Python-params-through-Rust chain instead"
  - "The predict path carries BOTH bars: D-04's absolute 1e-10 on the two Peyton fixtures AND 1e-14 x y_scale on all seven, because an absolute 1e-10 is FALSE on retail_sales (y_scale 518253, measured 2.33e-10 absolute = 4.5e-16 relative)"
  - "air_passengers and retail_sales future yhat is RECORDED, never barred: no committed control band exists, so an epsilon there would bar optimiser luck rather than parity"
  - "The qa_gate's falsification recipe was rewritten to the experiment actually OBSERVED (band_width_rel 0.02 -> 0.0001), because fitted_objective_slack cannot be induced-RED by tightening a positive value — every fixture lands BELOW Python"
  - "Fits and Python-params forecasts are memoised per fixture (OnceLock), so the ladder runs 7 fits for 14 fit-dependent rungs"
  - "Root Cargo.toml NOT touched: the profile table is 06-01 Task 2's decision, and this plan only measures under it"

patterns-established:
  - "Contract-read bars: `equation_tolerance(\"<contract>\", \"<equation>\")` written in full at each site, so `grep -c` is a real link check, not a proxy"
  - "Region-scoped anti-literal gates refuse to pass vacuously: the gate first asserts it can LOCATE `^mod parity`, then scans"
  - "Two-sided control on every gate touched: mutate to RED, revert byte-identical, observe GREEN"

requirements-completed: [SC2]

coverage:
  - id: D1
    description: "contracts/prophet-parity-v1.yaml exists on the setfit-apr-v1 shape (kind: kernel), passes pv validate with 0 errors, and is NOT hollow — pv status reports 12 proof obligations, 12 falsification tests and 2 Kani harnesses"
    requirement: SC2
    verification:
      - kind: other
        ref: "cargo run -p aprender-contracts-cli --bin pv -- validate contracts/prophet-parity-v1.yaml (rc=0, `0 error(s), 0 warning(s)`) — /tmp/p06-03-t1a.log"
        status: pass
      - kind: other
        ref: "pv status contracts/prophet-parity-v1.yaml — Equations 12 / Proof obligations 12 / Falsification tests 12 / Kani harnesses 2 / QA gate F-FORECAST-PROPHET-001 — /tmp/p06-03-t1b.log"
        status: pass
    human_judgment: false
  - id: D2
    description: "The Prophet port reproduces Python Prophet 1.4.0 on all seven committed fixtures as --lib tests CI's nextest leg runs without a workflow edit"
    requirement: SC2
    verification:
      - kind: unit
        ref: "cargo test -p aprender-forecast --lib prophet::parity — `test result: ok. 32 passed; 0 failed; 0 ignored`, 32 `test prophet::parity::` lines"
        status: pass
      - kind: unit
        ref: "cargo test -p aprender-forecast --lib — `ok. 47 passed; 0 failed; 0 ignored` (the whole crate, no regression against 06-01's 16)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Every tolerance the tests assert is read at test time from contracts/prophet-parity-v1.yaml and never duplicated as a literal (D-15); the tracer's one 1e-9 literal is gone"
    verification:
      - kind: other
        ref: "grep -c 'equation_tolerance(\"prophet-parity-v1\"' prophet.rs = 11; grep -c 'constant_u64(\"prophet-parity-v1\"' = 2"
        status: pass
      - kind: other
        ref: "region-scoped scan of `mod parity` for 1e-9|1.0e-9|1e-10|1.0e-10|0.5|0.02|0.0079|0.05 finds nothing; PROVEN to fire by injecting `const MUTATION_PROBE: f64 = 0.02;` (gate went RED) and reverting byte-identically (GREEN)"
        status: pass
      - kind: other
        ref: "induced-RED through the contract ALONE: band_width_rel 0.02 -> 0.0001 turns 7 band tests red (25 passed / 7 failed) quoting `over the contract bar 1e-4`; byte-identical revert restores 32 passed / 0 failed — /tmp/p06-03-induced-red.log, /tmp/p06-03-restored.log"
        status: pass
    human_judgment: false
  - id: D4
    description: "D-09 is observable in diagnostics: every fit reports rounds <= 8, a Stalled-or-Converged status, and budget_hit false on all seven fixtures"
    verification:
      - kind: unit
        ref: "prophet::parity::<fixture>_fit_objective_and_forecast (7 tests) — measured rounds 3..8, every status Stalled, budget_hit false on all seven"
        status: pass
    human_judgment: false
  - id: D5
    description: "The seven-fixture ladder's warm debug-profile wall is MEASURED and recorded together with the root-manifest profile state 06-01 Task 2 decided; this plan edits no root manifest"
    verification:
      - kind: other
        ref: "two warm `cargo test -p aprender-forecast --lib prophet::parity` runs: 14 s wall each, `Finished test profile in 12.2 s`, `finished in 1.73 s`/`1.78 s`; the same 32 tests run straight from target/debug/deps: 1766-1791 ms. Profile state grep -c '^[profile.dev.package.aprender-forecast]' Cargo.toml = 1"
        status: pass
      - kind: other
        ref: "git diff --stat HEAD -- Cargo.toml and git diff --stat <plan-start SHA>..HEAD -- Cargo.toml are both empty"
        status: pass
    human_judgment: false
  - id: D6
    description: "The band-width reference for the three spike-001 fixtures is DERIVED from the fixture's own forecast band (they publish no uncertainty block), with Python's parameters through Rust predict on the Rust side so no optimiser disagreement can leak into a band verdict"
    verification:
      - kind: unit
        ref: "prophet::parity::{peyton_manning,air_passengers,retail_sales}_band_widths_within_contract — measured 0.0368 %/1.1102 %, 0.5295 %/0.0302 %, 0.1665 %/0.3920 % (history/future) against the 2 % bar"
        status: pass
    human_judgment: true
    rationale: "The derived reference is this plan's own construction, not an oracle Python published: nobody outside this session has confirmed that `mean(yhat_upper - yhat_lower)` over the fixture's future rows is the right stand-in for Prophet's `future_band_mean_width`. The spike-003 cross-check is strong circumstantial evidence (peyton_manning's derived future reference is 1.3044 against peyton_default's published 1.2936 on the same series and the same Prophet defaults — a 0.8 % difference explained by Python's own sampling seed), but a human should confirm the substitution before the phase ships."

duration: 24min
completed: 2026-09-05
status: complete
---

# Phase 06 Plan 03: Prophet Parity Ladder Summary

**All seven Python Prophet 1.4.0 oracles now hold the port to a frozen, contract-read bar across five rungs — 32 `--lib` tests where the tracer had one — and every tolerance was proven to come from `contracts/prophet-parity-v1.yaml` by turning seven of them red with a one-line YAML edit and no Rust change at all.**

## Performance

- **Duration:** 24 min
- **Started:** 2026-09-06T05:10:34Z
- **Completed:** 2026-09-06T05:34:04Z
- **Tasks:** 2
- **Files created/modified:** 3 (1 created, 2 modified)

## Accomplishments

- **`contracts/prophet-parity-v1.yaml`** — 12 equations (11 carrying a `float_tolerance`), 12 proof obligations, 12 `FALSIFY-PROPHET-0NN` tests each naming the exact `cargo test` filter that discharges it, 2 Kani harnesses under the verbatim DECLARED-NOT-EXECUTED house comment, the `F-FORECAST-PROPHET-001` qa_gate, and a top-level `constants:` map. Templated on `setfit-apr-v1.yaml`, so `kind: kernel` makes PROVABILITY-001 fire — **never** the hollow `neon-blis-v1.yaml` shape that validates while reporting 0/0/0 (RESEARCH F4).
- **The ladder is complete** — 32 tests, one per rung per fixture, `ok. 32 passed; 0 failed; 0 ignored`. Data prep is exact (`0.00e0`) on all seven. Python's MAP reproduces Python's `-lp` to **exactly 0.0** on Peyton Manning and to 5.7e-14 / 2.3e-13 on air/retail. Python's parameters through Rust `predict` land at 2.7e-15 to 2.3e-10 absolute, i.e. **<= 4.5e-16 relative to `y_scale` everywhere**.
- **Every fit lands BELOW Python.** With the D-09 restart loop, the slack is negative on all seven fixtures (-0.0095 to -1.7265) against a one-sided bar of +0.5. Both Peyton forecasts sit at 0.0020 inside Prophet's own 0.0079 Newton-vs-L-BFGS band.
- **D-15 is closed.** The tracer's single `1e-9` literal is gone: 11 `equation_tolerance("prophet-parity-v1", ...)` sites and 2 `constant_u64(..., "max_rounds")` sites, with a region-scoped scan of `mod parity` refusing any tolerance literal — and that scan was **proven to fire** rather than assumed.
- **The qa_gate's induced negative was OBSERVED, not prescribed.** Tightening `band_width_rel` from 0.02 to 0.0001 in the YAML alone — no Rust touched — turned all seven band tests red, each quoting the new bar. A byte-identical revert restored 32/0.
- **The feedback-latency concern 06-01 left open is closed as a measurement artefact.** The ladder's warm wall is **14 s** via cargo and **1.73 s** of actual test execution, not the projected 91 s.

## Task Commits

1. **Task 1: `contracts/prophet-parity-v1.yaml` on the setfit-apr-v1 shape, proven not hollow** — `44809d8c4` (feat)
2. **Task 2: the full seven-fixture ladder as `--lib` tests reading the contract** — `d6b12ae1e` (test)

## Files Created/Modified

| File | What it does |
|---|---|
| `contracts/prophet-parity-v1.yaml` | **NEW.** The frozen correctness bar: `data_prep_exact`, `objective_at_python_map_abs`, `predict_path_abs_peyton`, `predict_path_rel_yscale`, `fitted_objective_slack`, `future_yhat_band_peyton_abs`, `components_via_python_params_abs`, `components_rebuild_yhat_abs`, `band_width_rel`, `band_width_last30_rel`, `trend_band_width_rel`, `fit_recipe`; `constants` (max_rounds 8, max_iters_per_round 2000, fit_budget_secs 15, lbfgs_memory 20) |
| `crates/aprender-forecast/src/prophet.rs` | `mod parity` grew from 1 test to 32: the `Ladder` helper (`spec_of` rebuilds the Spec from the fixture alone), per-fixture `OnceLock` caches for the fit and the Python-params forecast, five rung bodies, 31 named test wrappers and the bounded date-grid test |
| `.planning/.../deferred-items.md` | D-ITEM-06-03-a: every cargo invocation in this workspace pays a ~12 s forced rebuild |

## Measured results

### Rung 1 — data preparation (bar `data_prep_exact` = **0.0**, exact)

`t`, `y_scaled`, `changepoints_t` and the `X` first/last 3 rows are **`0.00e0` on all seven fixtures**, with 25 changepoints everywhere and K = 20 / 26 / 30 as the fixture declares. Seasonality lists, `prior_scales`, `y_scale` and `t_scale_days` match; on the four spike-003 fixtures the `columns` names AND order, `s_a`, `s_m` and (logistic) `history.cap_scaled` match too.

### Rung 2 — objective at Python's MAP (bar `objective_at_python_map_abs` = 1.0e-9)

| Fixture | `f_rust(theta_py)` | Python `-lp` | residual |
|---|---|---|---|
| peyton_manning | -8004.797952932482 | -8004.797952932482 | **0e0** (bit-identical) |
| air_passengers | -401.979951646592 | -401.979951646592 | 5.68e-14 |
| retail_sales | -1020.946963584151 | -1020.946963584150 | 2.27e-13 |

The four spike-003 fixtures publish no `-lp` and therefore carry no rung-2 test — see Deviation 1.

### Rung 3 — Python's parameters through Rust `predict`

| Fixture | max\|d yhat\| | max\|d trend\| | y_scale | relative bar | worst component | rebuild |
|---|---|---|---|---|---|---|
| peyton_manning | 5.33e-15 | 3.55e-15 | 12.8467 | 1.28e-13 | yearly 5.6e-16 | 0.0e0 |
| air_passengers | 2.27e-13 | 2.27e-13 | 622 | 6.22e-12 | yearly 3.2e-14 | 0.0e0 |
| retail_sales | 2.33e-10 | 1.16e-10 | 518253 | 5.18e-9 | yearly **2.9e-11** | 0.0e0 |
| peyton_default | 5.33e-15 | 3.55e-15 | 12.8467 | 1.28e-13 | additive_terms 6.7e-16 | 0.0e0 |
| peyton_holidays | 7.11e-15 | 7.11e-15 | 12.8467 | 1.28e-13 | additive_terms 6.7e-16 | 0.0e0 |
| wp_log_r_logistic | 2.66e-15 | 2.66e-15 | 9.0575 | 9.06e-14 | additive_terms 2.2e-16 | 0.0e0 |
| air_multiplicative | 3.41e-13 | 1.71e-13 | 622 | 6.22e-12 | yearly 1.1e-16 | 0.0e0 |

Both Peyton fixtures additionally clear D-04's ABSOLUTE `predict_path_abs_peyton` bar (1.0e-10) by five orders of magnitude. `trend*(1+multiplicative_terms)+additive_terms` reconstructs `yhat` to **exactly 0.0** on every row of every fixture. **Tightest margin on the whole ladder:** `retail_sales`'s `yearly` component at 2.9e-11 against the 1.0e-10 `components_via_python_params_abs` bar — 3.4x, and a consequence of that fixture's `y_scale` of 518253, since additive components are on the original scale.

### Rung 4 — the Rust fit (bar `fitted_objective_slack` = +0.5, one-sided)

| Fixture | `f_rust` | `f_python` | delta | rounds | iters | evals | status | hist max\|d yhat\| | future max\|d yhat\| |
|---|---|---|---|---|---|---|---|---|---|
| peyton_manning | -8004.9253 | -8004.7980 | **-0.1273** | 5 | 1312 | 11261 | Stalled | 0.0044 | **0.0020** (bar 0.0079) |
| air_passengers | -401.9895 | -401.9800 | **-0.0095** | 4 | 470 | 4154 | Stalled | 0.9042 | 0.2853 _(unbarred)_ |
| retail_sales | -1022.6735 | -1020.9470 | **-1.7265** | 8 | 2569 | 18403 | Stalled | 1991.0220 | 2529.5531 _(unbarred)_ |
| peyton_default | -8004.9253 | -8004.7980 | **-0.1273** | 5 | 1312 | 11261 | Stalled | 0.0044 | **0.0020** (bar 0.0079) |
| peyton_holidays | -8155.7557 | -8155.2205 | **-0.5352** | 3 | 1193 | 4577 | Stalled | 0.0230 | 0.0207 _(unbarred)_ |
| wp_log_r_logistic | -9019.8558 | -9019.6702 | **-0.1857** | 6 | 667 | 1764 | Stalled | 0.0034 | 0.0012 _(unbarred)_ |
| air_multiplicative | -503.4387 | -503.3958 | **-0.0429** | 5 | 449 | 1398 | Stalled | 0.8074 | 26.2487 _(unbarred)_ |

**The measured future max\|d yhat\| for the two unbarred monthly fixtures the plan asked for: `air_passengers` 0.2853 and `retail_sales` 2529.5531.** Both are far better than the spike's single-round numbers (35.38 and 1702.21) because the D-09 restart loop reaches a materially better basin — on retail it beats Python's own MAP by 1.73 objective units. The contract records why no bar is placed there.

**D-09 diagnostics on all seven:** `rounds` 3-8 (all `<= max_rounds` = 8), every `status` = `Stalled` (the expected terminal state for exact L1 on `delta`), `budget_hit` = **false** everywhere. `retail_sales` sits exactly AT the 8-round cap and is the one to watch: it is the only fixture whose restart loop never stopped improving by more than the 1e-6 relative threshold.

### Rung 5 — 80 % band widths (bars: 2 % history/future/last-30, 5 % trend)

| Fixture | reference | history | future | last-30 | future trend |
|---|---|---|---|---|---|
| peyton_manning | derived from the fixture band | 0.0368 % | **1.1102 %** | — | — |
| air_passengers | derived | 0.5295 % | 0.0302 % | — | — |
| retail_sales | derived | 0.1665 % | 0.3920 % | — | — |
| peyton_default | `uncertainty.*` | 0.0267 % | 0.2017 % | 0.2902 % | **3.9073 %** |
| peyton_holidays | `uncertainty.*` | 0.1944 % | 0.0065 % | 0.6660 % | 0.8701 % |
| wp_log_r_logistic | `uncertainty.*` | 0.0813 % | 0.4012 % | 1.4033 % | 0.3734 % |
| air_multiplicative | `uncertainty.*` | 0.3598 % | 0.2684 % | 0.0207 % | 0.9067 % |

The two thinnest margins are `peyton_manning`'s future band at 1.11 % (bar 2 %) and `peyton_default`'s future TREND band at 3.91 % (bar 5 %) — the two the contract already singles out, and both consistent with the ~1.2 % seed-to-seed variance the spike measured.

### Ladder timing (RESEARCH Pitfall 9 / F10)

**`debug parity ladder wall: 14 s`**, measured under **root-manifest profile state = 1** (`[profile.dev.package.aprender-forecast] opt-level = 3` present, exactly as 06-01 Task 2 decided; this plan edited no manifest, confirmed in the working tree AND across every commit since the plan-start SHA `639c2d84b`).

Two consecutive warm invocations, `CARGO_INCREMENTAL=0`, no source edit between them:

| Measurement | run 1 | run 2 |
|---|---|---|
| `cargo test -p aprender-forecast --lib prophet::parity` wall | 14 s | 14 s |
| of which cargo's `Finished test profile in` | 12.21 s | 12.20 s |
| of which test execution (`finished in`) | 1.73 s | 1.78 s |
| the same 32 tests run straight from `target/debug/deps/aprender_forecast-*` | 1791 ms | 1766 ms |

**Well under the 60 s target, and the reason is worth recording rather than celebrating.** 06-01 Task 2 projected `7 x 13 s = 91 s` from the wall of one `cargo test -p aprender-mcp-forecast --lib e2e` invocation. Re-measured this session, that proxy is **11.73 s of forced rebuild + 1.64 s of test** — so the projection multiplied a CONSTANT per-invocation build cost seven times. The real ladder is ONE invocation. The profile override is still doing real work (the fits are the 1.7 s, not the 12 s), but the "~91 s against a 60 s target" concern is closed as a **measurement artefact**, not as an optimisation win. Logged as `D-ITEM-06-03-a` for 06-09 because the ~12 s always-rebuild belongs to `aprender-compute`/`aprender-core` build scripts this plan does not touch.

## Decisions Made

**Rung 2 binds three fixtures, not seven — and that is a property of the fixtures.** See Deviation 1.

**Both predict-path bars ship.** D-04 states the predict path as an absolute 1e-10 on Peyton. That literal is kept and asserted on the two Peyton fixtures, but it is FALSE as a general bar: `retail_sales` reproduces Python's `yhat` to 2.33e-10 ABSOLUTE at a `y_scale` of 518253, which is 4.5e-16 RELATIVE — tighter than Peyton in every sense that matters. So `predict_path_rel_yscale` (1e-14 x `y_scale`) is the general bar and `predict_path_abs_peyton` is the dataset-specific one.

**air/retail future `yhat` is recorded and not barred.** The yearly Fourier block is near-unidentified on monthly data; Prophet's own optimisers disagree by far more than any epsilon worth writing down. A bar there would bar optimiser luck. The tests compute the number, print it, and assert only that it is finite.

**Fits are memoised.** `fit_objective_and_forecast` and (on spike-003 fixtures) `band_widths_within_contract` both need the MAP fit, and the plan asked for 7 fits rather than 11. A per-fixture `OnceLock` gives exactly that without serialising the whole module: the second caller blocks on the first rather than duplicating a 2-3 s Peyton fit.

**The band reference for spike-001 fixtures is derived, and the derivation is stated in the contract.** Those three files carry no `uncertainty` block, so the Python reference is `mean(forecast.yhat_upper - forecast.yhat_lower)` over the future rows and, separately, the history rows; the Rust side is Python's parameters through Rust `predict` (fit-independent), so air/retail's unbarred optimiser disagreement cannot leak into a band verdict. Flagged for human confirmation in `coverage.D6`.

## Deviations from Plan

### 1. [Rule 3 - Blocking] `objective_at_python_map` binds THREE fixtures, not seven — 32 tests instead of 36

- **Found during:** Task 2, reading the fixtures before writing the rung.
- **Issue:** the plan asks for `<f>_objective_at_python_map` on all seven fixtures, "as the tracer's test". The tracer's test compares against `log_posterior_at_map_unnormalized`. **The four spike-003 fixtures do not have that key** — `peyton_default`, `peyton_holidays`, `wp_log_R_logistic` and `air_multiplicative` publish `columns`/`s_a`/`s_m`/`uncertainty` and no Python `-lp`. (Verified directly: `json.load(...).get('log_posterior_at_map_unnormalized')` is `None` on all four.) There is no oracle to compare against there.
- **Fix:** the rung is written for the three fixtures that publish the number. A seventh test on the other four would have compared the Rust objective to itself and passed unconditionally — theatre, and worse than an absent test because it would look like coverage. The contract's description, the `objective_at_python_map_abs` invariants and `FALSIFY-PROPHET-002` all state the asymmetry explicitly, and the spike-003 fixtures are held instead by `fitted_objective_slack` (whose Python reference there is the Rust objective at Python's MAP — legitimate precisely BECAUSE rung 2 proves that quantity is Python's objective wherever Python publishes it) plus the full rung-3 chain.
- **Effect on the plan's numbers:** 32 tests rather than the plan's projected 36. The `<verify>` bar is `>= 30` and the acceptance criteria name nine specific functions, none of them a spike-003 objective test — all met.
- **Files modified:** `crates/aprender-forecast/src/prophet.rs`, `contracts/prophet-parity-v1.yaml`
- **Verification:** `ok. 32 passed; 0 failed; 0 ignored`; 32 `test prophet::parity::` lines.
- **Committed in:** `d6b12ae1e`

### 2. [Rule 1 - Bug] The contract's qa_gate prescribed a falsification that cannot be performed

- **Found during:** Task 2, running the induced-RED control the qa_gate demands (T-06-11).
- **Issue:** the `qa_gate.falsification` text (written in Task 1, before any fit had been run) said to "tighten `fitted_objective_slack` from 0.5 to a value below the measured slack". Once measured, **every fixture lands BELOW Python** — the slacks are -0.0095 to -1.7265 — so no smaller POSITIVE tolerance can turn that test red. The prescribed recipe was impossible.
- **Fix:** the qa_gate now records the induced negative actually **observed** — `band_width_rel` 0.02 -> 0.0001, changing only YAML — and notes that the `fitted_objective_slack` negative must be a NEGATIVE slack (e.g. -1.0), not a smaller positive one. A contract that prescribes an unrunnable falsification is a contract nobody will falsify.
- **Verification:** `pv validate` still 0 errors / 0 warnings; `pv status` unchanged at 12/12/2.
- **Committed in:** `d6b12ae1e`

### 3. [Rule 1 - Bug, self-inflicted] The parity module was first installed at the wrong line offset

- **Found during:** Task 2, installing the new `mod parity`.
- **Issue:** the line numbers used to truncate `prophet.rs` were read off a paginated tool transcript whose own header lines shifted every number by 9, so `head -n 872` cut nine lines INTO the old module and the result had two `mod parity` blocks.
- **Fix:** caught immediately by `grep -n '^mod parity'` returning two matches (865 and 874) instead of one, reverted with `git checkout -- crates/aprender-forecast/src/prophet.rs` and redone at the verified offset 863 (confirmed by reading lines 860-866 of the real file, not a transcript). **Nothing was committed in the broken state**, and the region gate that surfaced it is the same one the plan requires.
- **Recorded because** the general rule — never take a line number from a rendering of a file when you can ask the file — is exactly the kind of thing CLAUDE.md's Verification Discipline exists for.

### 4. [Out of scope - logged, not fixed] `.pv/contracts.idx` is regenerated by every `pv` run

- Running `pv validate` / `pv status` rewrites the tracked 4 MB single-line `.pv/contracts.idx`, `.pv/contracts.idx.mtime` and `.pv/lint-previous.json`. Git history shows that index is refreshed in dedicated `chore(pv): rebuild contracts.idx` commits, not alongside each new contract, so the three files were restored to HEAD (`git checkout -- .pv/...`) rather than committed. Anyone re-running this plan's `<verify>` blocks will see the same three files go dirty.

### 5. [Out of scope - logged, not fixed] Every cargo invocation forces a ~12 s rebuild

- Logged as `D-ITEM-06-03-a` (see the timing section). Owner: 06-09.

---

**Total deviations:** 5 — 1 Rule 3 blocking (the missing oracle key), 2 Rule 1 bugs (an impossible contract falsification; a self-inflicted bad line offset, caught pre-commit), 2 out-of-scope findings logged rather than fixed.
**Impact on plan:** No scope creep and no loosened bar. One rung binds fewer fixtures than the plan projected because the oracle does not publish the number it would need — stated in the contract rather than papered over with a self-comparison.

## Gates run

| Gate | Result |
|---|---|
| `pv validate contracts/prophet-parity-v1.yaml` | **rc=0**, `0 error(s), 0 warning(s)` |
| `pv status contracts/prophet-parity-v1.yaml` | Equations **12**, obligations **12**, falsification tests **12**, Kani harnesses **2**, qa_gate `F-FORECAST-PROPHET-001` — no count is 0 (RESEARCH F4) |
| `cargo test -p aprender-forecast --lib prophet::parity` | **ok. 32 passed; 0 failed; 0 ignored**, 32 `test prophet::parity::` lines |
| `cargo test -p aprender-forecast --lib` | **ok. 47 passed; 0 failed; 0 ignored** (16 before this plan) |
| Region-scoped tolerance-literal scan of `mod parity` | clean; **proven to fire** by an injected `0.02` and reverted byte-identically |
| Induced-RED through the contract alone (`band_width_rel` 0.02 -> 0.0001) | **25 passed / 7 failed**, each failure quoting `over the contract bar 1e-4`; byte-identical revert -> **32 passed / 0 failed** |
| `cargo clippy -p aprender-forecast --all-targets --no-deps -- -D warnings` | **rc=0**; engagement proven by a `needless_bool` probe (rc=101, lint named) and a byte-identical revert (rc=0) |
| `cargo fmt --all -- --check` | **rc=0** |
| `cargo check --workspace --exclude aprender-profile` | **rc=0**, 0 errors |
| Root-manifest guard (working tree AND `639c2d84b..HEAD`) | both empty; profile state = 1 |
| Task 1 acceptance criteria | **5/5** |
| Task 2 acceptance criteria | **6/6** |

`--no-deps` and `--exclude aprender-profile` are 06-01's recorded, measured scopings (D-ITEM-06-01-b and the Darwin `compile_error!`), not this plan loosening a gate.

## Issues Encountered

- **The `rtk` Bash hook rewrites `cargo test` and strips both the `test result:` line and `println!` output**, exactly as 06-01 recorded. Every verification command here was run through `rtk proxy cargo ...`, and the per-rung diagnostics were captured by invoking `target/debug/deps/aprender_forecast-*` directly with `--nocapture --test-threads=1`.
- **`cargo test`'s reported wall is dominated by a rebuild that is not this crate's** — see D-ITEM-06-03-a. Any future timing claim in this workspace should separate `Finished ... in` from `finished in`.

## Known Stubs

**None introduced by this plan.** No hardcoded empty value, no placeholder, and no test that can pass without loading its fixture (`load_json` `expect`s, and every rung asserts against fixture data). The pre-existing `model: "neuralprophet"` refusing stub from 06-01 is untouched and still belongs to plan 06-04.

Two things are deliberately NOT asserted, and are recorded rather than hidden:
- `air_passengers` / `retail_sales` future `yhat` (no committed control band — the contract explains why; the numbers are 0.2853 and 2529.5531).
- `objective_at_python_map` on the four spike-003 fixtures (no oracle key — Deviation 1).

## Threat Flags

None. This plan adds one YAML contract and test code; it introduces no network surface, no new dependency (`Cargo.lock` untouched — T-06-SC satisfied by construction), and no new file-access pattern beyond reading fixtures and the contract that 06-01's `test_support` already read.

T-06-10 (tampering with tolerance values) is mitigated as planned: the tolerances live in ONE `pv`-validated file, the tests read them, and a change is a `pv diff`-visible edit. T-06-11 (a parity claim never falsified) is mitigated with an **observed** induced-RED recorded above, not a promised one.

## User Setup Required

None.

## Next Phase Readiness

**Ready for 06-04** (the NeuralProphet port), which is the other Wave 2 plan and shares no file with this one.

What 06-04 and later plans inherit:

- **`contracts/prophet-parity-v1.yaml` is the shape to copy** for `np-parity` / `chronos-parity`: `kind: kernel`, one equation per bar with `float_tolerance`, obligations >= 1 per equation, falsification tests >= obligations each naming its `cargo test` filter, the DECLARED-NOT-EXECUTED Kani comment, and a top-level `constants:` map that `constant_u64` reads. `pv status` is the check that it is not hollow.
- **The rung pattern**: a `Ladder` built from the fixture alone, per-fixture `OnceLock` memoisation for anything expensive, and one named `#[test]` wrapper per fixture per rung so a failure names both.
- **The root `Cargo.toml` remains closed** for this phase.
- **One number to watch:** `retail_sales` uses all 8 restart rounds. If a later change to `fit.rs` slows convergence, that fixture is the first to trip `rounds <= max_rounds`.

---
*Phase: 06-native-time-series-forecasting-stack*
*Completed: 2026-09-05*

## Self-Check: PASSED

All `key-files` entries exist on disk (`contracts/prophet-parity-v1.yaml`,
`crates/aprender-forecast/src/prophet.rs`, the phase `deferred-items.md`), and both task
commits (`44809d8c4`, `d6b12ae1e`) are present in `git log --all`.
