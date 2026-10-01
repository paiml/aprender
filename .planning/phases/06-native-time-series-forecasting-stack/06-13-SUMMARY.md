---
phase: 06-native-time-series-forecasting-stack
plan: 13
subsystem: api
tags: [rust, prophet, forecasting, poisson, numerical-stability, uncertainty-intervals, provable-contracts, falsification]

requires:
  - phase: 06-native-time-series-forecasting-stack
    provides: "the Prophet parity ladder and its contract-read tolerances (06-03); the D-15 mutation-control idiom for proving a bar is contract-owned (06-11 Task 2, 06-12 Task 2); the binding-row + `contract-audit-phase6` resolution convention (06-12)"
provides:
  - "`prophet::poisson(rng, lambda)` — the changepoint-count sampler EXTRACTED from `predict`'s logistic uncertainty arm, so its numerical domain is assertable without a full logistic forecast"
  - "A valid-domain branch: Knuth's product method at or below `POISSON_NORMAL_BRANCH_LAMBDA` (30.0), `lambda + sqrt(lambda) * N(0,1)` above it — mean and variance both exactly lambda"
  - "`prophet::sampler::poisson_mean_tracks_lambda_across_its_whole_domain` — a six-lambda falsification test OBSERVED RED first (745.13 at lambda 900, 745.45 at 2839) and green after"
  - "`prophet::sampler::wp_log_r_logistic_fixture_lambda_is_far_below_the_branch_threshold` — the parity argument as an ENFORCED measurement (lambda 3.1239) rather than a SUMMARY claim"
  - "`equations.poisson_sampler_domain` + a `type: invariant` proof obligation + `FALSIFY-PROPHET-013` in prophet-parity-v1.yaml, version 1.0.0 -> 1.1.0 on `pv diff`'s own suggestion"
  - "A `binding.yaml` row that RESOLVES to `aprender_forecast::prophet::poisson` (Phase 6 resolved rows 57 -> 58)"
  - "`LOGISTIC BAND WALL:` — an `#[ignore]` release harness walling both reachable logistic extremes against SC1's 2 s bar"
affects: [gsd-verify-work, prophet-parity, forecast-tool-boundary]

actuals:
  tokens: 5319
  tasks: 3
  commits: 4
plan_head_before: a56ef5df0e5710762ce6cc296494ddf5f4ec9f7a

tech-stack:
  added: []
  patterns:
    - "EXTRACT TO ASSERT: a numerical defect buried inside a hot loop is not assertable at all; lifting it to a named free function (behaviour byte-for-byte unchanged, parity re-run to prove it) is the step that makes falsification possible"
    - "Print the whole sweep BEFORE asserting any of it: a print-and-assert loop aborts at the first failing input and hides the SHAPE of the failure at the larger ones — six points made 'saturates at 745' visible where one would only have shown 'wrong at 900'"
    - "Turn the parity argument into an enforced test: the claim 'no rung crosses the threshold' ships as `wp_log_r_logistic_fixture_lambda_is_far_below_the_branch_threshold`, so a future fixture change goes RED instead of silently invalidating the argument"
    - "Measure the BEFORE side of a behavioural change by a reversible control (threshold to `f64::INFINITY`, `cmp`-verified restore), not by reasoning about what the old code would have done"

key-files:
  created: []
  modified:
    - crates/aprender-forecast/src/prophet.rs
    - contracts/prophet-parity-v1.yaml
    - contracts/aprender/binding.yaml

key-decisions:
  - "Threshold 30.0, the value the verifier named, KEPT because the measurement allowed it: the only logistic parity fixture reaches lambda 3.1239, about a tenth of the threshold, so no rung can take the new branch. The threshold did NOT have to be raised"
  - "Outcome taken: the VALID SAMPLER BRANCH, not a refusal and not a reported clamp. The other two would have kept the caller from getting the bands SC1 promises; the regime is legal input, so refusing it narrows the product to work around our own arithmetic"
  - "Normal approximation rather than a PTRS port of numpy's `np.random.poisson`: this sampler never matched numpy stream-for-stream (the RNG is this file's seeded xorshift, not MT19937), and the bar it feeds (`band_width_rel`) is a RELATIVE bar on mean band width — a Monte-Carlo estimate that depends on the count's mean and variance, both exactly lambda for `lambda + sqrt(lambda) * N(0,1)`"
  - "float_tolerance 0.01, not the 0.02 the transient const used: 3.16 sigma at the noisiest point in the sweep (lambda 5) and >= 7.62 sigma everywhere else, against a pre-fix failure of 17.2 % — a 17x margin over the defect and real headroom over the noise"
  - "The tolerance entered the contract EXACTLY ONCE, in Task 3. Tasks 1-2 read a file-local const and Task 1 pinned both contract files clean with `git status --porcelain`; no placeholder key was ever written to get past the missing-key panic"
  - "No per-site `#[allow(clippy::cast_...)]` was needed: `cast_precision_loss`, `cast_possible_truncation`, `cast_sign_loss` and `cast_lossless` are ALREADY allowed at the workspace level. Nothing was widened — the plan anticipated a per-site allow that the existing configuration made unnecessary"
  - "The `logistic_band_wall` harness takes `LOGISTIC_BENCH_POINTS` / `LOGISTIC_BENCH_HORIZON` so BOTH reachable extremes could be walled. The 100-point case is only a 1.24x count increase (745 -> ~922); the 10-point floor is the ~3.8x case the threat register named, and measuring only the first would have understated the cost"

patterns-established:
  - "A behavioural before/after is measured with a one-line reversible control and a `cmp`-verified restore, so the BEFORE number comes from running the old behaviour rather than from arguing about it"
  - "A claim written into a contract gets recomputed before the SUMMARY is written — the sigma-headroom figure shipped wrong by 6x and was caught by re-deriving it, not by re-reading it"

requirements-completed: [SC1]

coverage:
  - id: D1
    description: "The changepoint-count sampler is a NAMED, DIRECTLY TESTABLE free function — `prophet::poisson(rng, lambda)` — extracted from `predict`'s logistic uncertainty arm with the algorithm byte-for-byte unchanged, and the extraction proven behaviour-preserving by an unmoved parity ladder"
    requirement: "SC1"
    verification:
      - kind: unit
        ref: "cargo test -p aprender-forecast --lib prophet::parity — 32 passed / 0 failed, identical before (a56ef5df0) and after the extraction"
        status: pass
    human_judgment: false
  - id: D2
    description: "The RED side was OBSERVED, not asserted: with the unmodified Knuth sampler the six-lambda sweep reported 745.1324 at lambda 900 and 745.4496 at lambda 2839, and the test panicked naming the lambda, the mean and the bar — with BOTH contract files pinned clean by `git status --porcelain`"
    requirement: "SC1"
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/prophet.rs#prophet::sampler::poisson_mean_tracks_lambda_across_its_whole_domain (RED run at commit 596b08e56, rc=101, all six POISSON MEAN lines quoted verbatim below)"
        status: pass
    human_judgment: false
  - id: D3
    description: "The sampler's mean tracks lambda across its whole reachable domain after the branch: 900 -> 900.0384 (rel 4.3e-5) and 2839 -> 2839.5755 (rel 2.0e-4), with lambda 5 and 29 byte-identical to the RED run so the Knuth path is provably undisturbed"
    requirement: "SC1"
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/prophet.rs#prophet::sampler::poisson_mean_tracks_lambda_across_its_whole_domain (2 passed / 0 failed)"
        status: pass
    human_judgment: false
  - id: D4
    description: "No parity rung moved, and the reason is a MEASUREMENT: the only logistic fixture reaches lambda 3.1239 (25 changepoints x (t_max 1.124957 - 1)), about a tenth of the 30.0 branch threshold, and that measurement now ships as an enforced test rather than a SUMMARY claim"
    requirement: "SC1"
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/prophet.rs#prophet::sampler::wp_log_r_logistic_fixture_lambda_is_far_below_the_branch_threshold"
        status: pass
      - kind: unit
        ref: "cargo test -p aprender-forecast --lib prophet::parity — exactly 32 passed; 0 failed, after the branch"
        status: pass
    human_judgment: false
  - id: D5
    description: "The bar is CONTRACT-OWNED and the swap was proven mechanically, not by inspection: `equations.poisson_sampler_domain.float_tolerance` set to 1.0e-9 made the shipped test FAIL naming `lambda=5.0 ... > bar=1e-9`, and the restored 0.01 made it pass"
    requirement: "SC1"
    verification:
      - kind: unit
        ref: "cargo test -p aprender-forecast --lib prophet::sampler under a mutated contract value — rc=101, assertion message quoted verbatim below; restored, rc=0"
        status: pass
      - kind: other
        ref: "pv validate contracts/prophet-parity-v1.yaml — 0 error(s), 0 warning(s); pv status — Equations 13 / Proof obligations 13 / Falsification tests 13 (was 12/12/12)"
        status: pass
    human_judgment: false
  - id: D6
    description: "The new equation binds to a REAL definition site: `make contract-audit-phase6` reports rc=0, zero BIND-, zero RESOLVE- over four contracts, and 58 resolved Phase 6 binding rows (was 57 at 06-12)"
    requirement: "SC1"
    verification:
      - kind: other
        ref: "make contract-audit-phase6 — rc=0, BIND- 0, RESOLVE- 0, four `Total equations:` summaries, `resolved 58 Phase 6 binding rows`"
        status: pass
      - kind: other
        ref: "make contract-validate — rc=0, `Contract validation passed`"
        status: pass
    human_judgment: false
  - id: D7
    description: "The cost of the correctness fix is MEASURED at both reachable logistic extremes on a release build, against SC1's 2 s bar: 100 points 0.225 -> 0.233 s, 10 points 0.144 -> 0.205 s. Neither crosses the bar; the mean band width at the 100-point case widened 33.7674 -> 34.8853"
    requirement: "SC1"
    verification:
      - kind: other
        ref: "cargo test --release -p aprender-forecast --lib prophet::sampler::logistic_band_wall -- --ignored --nocapture, at LOGISTIC_BENCH_POINTS in {100, 10}, before (threshold at f64::INFINITY) and after — four LOGISTIC BAND WALL lines quoted below"
        status: pass
    human_judgment: false
  - id: D8
    description: "The band-width change at the 10-point `MIN_POINTS` floor is negligible (2.0553 -> 2.0559) even though its count truncation was the LARGEST — whether that regime's bands are adequate for a user is not something this plan measured"
    verification: []
    human_judgment: true
    rationale: "The correctness defect is closed at both extremes — the sampler is inside its domain at every reachable lambda, which is what SC1's `yhat_lower`/`yhat_upper` promise required. But a 10-point history extrapolated 3650 days is a degenerate request whose bands are dominated by the fit, not by the changepoint simulation, and no automated bar says whether such a forecast should be SERVED at all rather than refused at the door. That is a product judgment about `MIN_POINTS` vs `MAX_HORIZON`, not a numerical one, and this plan deliberately did not take it."

duration: 20 min
completed: 2026-09-07
status: complete
---

# Phase 06 Plan 13: Close the logistic-band sampler's domain gap Summary

**The logistic uncertainty path's changepoint-count draw was running past f64's exp-underflow point and saturating near 745 regardless of lambda — measured at 745.1324 for lambda 900 and 745.4496 for lambda 2839 — so long-horizon logistic bands came back narrower than the `interval_width` they advertised; the sampler is now an extracted, directly testable `poisson()` with a normal-approximation branch above lambda 30, the RED was observed BEFORE the fix at six lambdas with both contract files pinned clean, and the bar lives in `equations.poisson_sampler_domain` with the D-15 swap proven by mutating the contract value to 1e-9 and watching the shipped test go red.**

## Performance

- **Duration:** 20 min
- **Started:** 2026-09-07T04:02:15Z
- **Completed:** 2026-09-07T04:22:09Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- **The defect is now ASSERTABLE.** Before this plan the only path to the sampler was a full logistic forecast, which is exactly why 32 parity rungs passed over a wrong path. `poisson(rng, lambda)` is a named free function and the extraction was proven behaviour-preserving by an unmoved ladder.
- **The RED was OBSERVED, at six lambdas, with the contract untouched.** 745.1324 at lambda 900 (rel 17.2 %) and 745.4496 at lambda 2839 (rel 73.7 %), while 5 / 29 / 31 / 100 all tracked. That shape — *tracks up to ~745, then saturates* — is the diagnosis, and only a multi-point sweep shows it.
- **The premise was a code reading; this is the first direct measurement of it, and it CONFIRMED the premise.** The verifier predicted saturation near 745 at both large lambdas. The measurement agrees to four significant figures.
- **The parity argument is a number, and now an enforced test.** The only logistic fixture reaches lambda **3.1239**, about a tenth of the 30.0 threshold. `wp_log_r_logistic_fixture_lambda_is_far_below_the_branch_threshold` goes red if a future fixture change ever invalidates that.
- **The bar is contract-owned and the swap was PROVEN.** `float_tolerance` at 1e-9 turned the shipped test red naming `lambda=5.0 ... > bar=1e-9`; restored, green. A test whose verdict does not move when the contract moves is still reading a literal.
- **The cost of the fix was measured at BOTH reachable extremes**, not just the one the plan named — 0.233 s and 0.205 s against SC1's 2 s bar.

## Task Commits

1. **Task 1: extract the sampler and observe it RED across its domain** — `596b08e56` (test)
2. **Task 2: add the valid-domain branch, observe GREEN, prove no rung moved** — `b865acae9` (fix)
3. **Task 3: contract identity — equation, obligation, FALSIFY-013, binding row, D-15 swap** — `f8bb1e608` (feat)
4. **Deviation fix: correct the sigma-headroom claim in the new invariant** — `b166573a8` (fix)

## Files Created/Modified

- `crates/aprender-forecast/src/prophet.rs` — `POISSON_NORMAL_BRANCH_LAMBDA`, the extracted `poisson()` with its branch and its two doc sections, `mod sampler` (three tests), and the `predict` call site
- `contracts/prophet-parity-v1.yaml` — `equations.poisson_sampler_domain`, one `type: invariant` proof obligation, `FALSIFY-PROPHET-013`, version 1.0.0 -> 1.1.0
- `contracts/aprender/binding.yaml` — one row: `poisson_sampler_domain` -> `aprender_forecast::prophet::poisson`

---

## THE MEASURED FIXTURE LAMBDA (Task 1) — why the ladder was silent

`lambda = changepoints_t.len() * (t_max - 1)`, and both factors come from the fixture's own
published data:

| factor | value | source |
|---|---|---|
| `changepoints_t.len()` | **25** | `wp_log_R_logistic_prophet140.json` `changepoints_t` |
| `start` | 2008-01-01 | fixture `start` |
| `t_scale_days` | 2921.0 | fixture `t_scale_days` |
| last forecast `ds` | 2016-12-30 | last element of `forecast.ds` |
| `t_max` | (2016-12-30 - 2008-01-01) / 2921 = **1.124957** | derived |
| **lambda** | 25 x (1.124957 - 1) = **3.1239** | derived |

```
FIXTURE LAMBDA: fixture=wp_log_R_logistic n_changepoints=25 t_max=1.124957 lambda=3.1239
```

**3.1239 is far BELOW 30**, so every one of the 32 parity rungs takes the unchanged Knuth path
and cannot move because of this branch. Reading these two factors off the fixture is a legitimate
measurement of what the ladder runs on: `wp_log_r_logistic_data_prep_exact` already asserts that
Rust's `make_design` reproduces `changepoints_t` **exactly**, so the number the fixture publishes
and the number the ladder computes are the same number.

The ladder's silence here is a **coverage fact**, and it is now recorded in the contract's own
invariants so it is not mistaken for evidence.

## THE SIX-LAMBDA SWEEP — BEFORE and AFTER, verbatim

N = 20 000 draws per lambda at a fixed seed. Both runs printed all six lines before asserting any
of them, deliberately: a print-and-assert loop would have aborted at lambda 900 and never shown
2839.

| lambda | BEFORE (Knuth everywhere) | rel | AFTER (branch at 30) | rel | path |
|---|---|---|---|---|---|
| 5.0 | 5.0076 | 0.001510 | **5.0076** | 0.001510 | Knuth — *byte-identical* |
| 29.0 | 28.9383 | 0.002126 | **28.9383** | 0.002126 | Knuth — *byte-identical* |
| 31.0 | 31.0402 | 0.001297 | 31.0664 | 0.002144 | normal |
| 100.0 | 99.9995 | 0.000005 | 100.0811 | 0.000811 | normal |
| **900.0** | **745.1324** | **0.172075** | **900.0384** | 0.000043 | normal |
| **2839.0** | **745.4496** | **0.737425** | **2839.5755** | 0.000203 | normal |

RED run, verbatim (commit `596b08e56`, `rc=101`):

```
POISSON MEAN: lambda=5.0 n=20000 mean=5.0076 rel=0.001510
POISSON MEAN: lambda=29.0 n=20000 mean=28.9383 rel=0.002126
POISSON MEAN: lambda=31.0 n=20000 mean=31.0402 rel=0.001297
POISSON MEAN: lambda=100.0 n=20000 mean=99.9995 rel=0.000005
POISSON MEAN: lambda=900.0 n=20000 mean=745.1324 rel=0.172075
POISSON MEAN: lambda=2839.0 n=20000 mean=745.4496 rel=0.737425
thread 'prophet::sampler::poisson_mean_tracks_lambda_across_its_whole_domain' panicked at
crates/aprender-forecast/src/prophet.rs:958:13:
poisson sampler outside its domain at lambda=900.0: mean=745.1324 rel=0.172075 > bar=2e-2
```

GREEN run, verbatim (commit `b865acae9`, `rc=0`):

```
POISSON MEAN: lambda=5.0 n=20000 mean=5.0076 rel=0.001510
POISSON MEAN: lambda=29.0 n=20000 mean=28.9383 rel=0.002126
POISSON MEAN: lambda=31.0 n=20000 mean=31.0664 rel=0.002144
POISSON MEAN: lambda=100.0 n=20000 mean=100.0811 rel=0.000811
POISSON MEAN: lambda=900.0 n=20000 mean=900.0384 rel=0.000043
POISSON MEAN: lambda=2839.0 n=20000 mean=2839.5755 rel=0.000203
test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 97 filtered out
```

**The two Knuth-path lambdas are byte-identical across the two runs.** That is not decoration: it
is the direct evidence that the branch left the sub-threshold path untouched, which is the same
claim the unmoved parity ladder makes from the other direction.

**Did the measurement contradict the premise? No.** The verifier's expectation — 900 and 2839 both
reporting a mean near 745, 5 / 29 / 31 / 100 tracking — is exactly what was measured. The gap's
premise was a code reading; this is the first direct measurement of it, and it holds.

### Why sampling noise is not the explanation

At lambda = 900 the standard error of the mean over N = 20 000 draws is
**sqrt(900 / 20 000) = 0.2121**, i.e. **0.0236 % of lambda**. The observed shortfall was
**17.2 %** — about **730 standard errors**. At lambda = 2839 the SE is 0.3768 (0.0133 % of lambda)
against a 73.7 % shortfall, **5 500 standard errors**. No sample size dispute survives that gap;
20 000 was chosen so the arithmetic would be this one-sided.

## THE BRANCH (Task 2) — threshold, shape, and the outcome taken

```rust
pub const POISSON_NORMAL_BRANCH_LAMBDA: f64 = 30.0;

pub fn poisson(rng: &mut Rng, lambda: f64) -> usize {
    if lambda > POISSON_NORMAL_BRANCH_LAMBDA {
        return (lambda + lambda.sqrt() * rng.normal()).round().max(0.0) as usize;
    }
    // Knuth's product method, unchanged
```

**Threshold: 30.0, kept, and the measurement is what allowed keeping it.** It is the value the
verifier named, and the measured fixture lambda (3.1239) sits about a tenth of the way to it —
so it did NOT have to be raised, and the ladder confirms that at **exactly 32 passed / 0 failed**
before and after.

**Outcome taken: the VALID SAMPLER BRANCH** — the first of the three the plan allowed, not an
explicit refusal and not a reported clamp. The regime is legal input inside every door bound; a
refusal would narrow the product to work around our own arithmetic, and a clamp reported in
`diagnostics` would still hand back a band that is not the band the caller asked for. Correcting
the draw is the only outcome that makes `yhat_lower` / `yhat_upper` mean what SC1 says they mean.

**Normal approximation, not a PTRS port**, and the code comment records why: this sampler never
matched `np.random.poisson` stream-for-stream (the RNG is this file's own seeded xorshift, not
MT19937), the bar it feeds is `band_width_rel` — a RELATIVE bar on mean band width, stated in the
contract as a Monte-Carlo estimate — and both the mean and the variance of
`lambda + sqrt(lambda) * N(0,1)` are exactly lambda, which is what a band-width estimate depends
on. Above lambda 30 the Poisson's skew is already <= 0.18.

## THE COST OF THE FIX (Task 2 step 4) — both reachable extremes, release profile

The BEFORE side is a **measurement, not a recollection**: `POISSON_NORMAL_BRANCH_LAMBDA` was
temporarily set to `f64::INFINITY` (making every lambda take the Knuth path), both cases were
walled, and the file was restored from a copy and verified byte-identical with `cmp`.

| case | lambda | BEFORE total_s | AFTER total_s | BEFORE band width | AFTER band width | vs SC1 2 s |
|---|---|---|---|---|---|---|
| 100 pts, horizon 3650, logistic | ~922 | **0.225** | **0.233** | 33.7674 | **34.8853** | 8.6x under |
| 10 pts (`MIN_POINTS`), horizon 3650 | ~2839 | **0.144** | **0.205** | 2.0553 | 2.0559 | 9.8x under |

```
LOGISTIC BAND WALL: points=100 horizon=3650 growth=logistic total_s=0.225 fit_s=0.066 predict_s=0.158 mean_band_width=33.7674 profile=release
LOGISTIC BAND WALL: points=100 horizon=3650 growth=logistic total_s=0.233 fit_s=0.076 predict_s=0.157 mean_band_width=34.8853 profile=release
LOGISTIC BAND WALL: points=10 horizon=3650 growth=logistic total_s=0.144 fit_s=0.001 predict_s=0.142 mean_band_width=2.0553 profile=release
LOGISTIC BAND WALL: points=10 horizon=3650 growth=logistic total_s=0.205 fit_s=0.001 predict_s=0.203 mean_band_width=2.0559 profile=release
```

**Neither wall crosses the 2 s bar, so T-06-30 did not materialise.** The increase is bounded
exactly as the linearity argument predicted: the 100-point case is only a 1.24x count increase
(745 -> ~922) and costs +3.6 %; the 10-point floor is the ~3.8x case and costs +42 % of a very
small number.

**The band widened where the truncation mattered and barely moved where it did not.** At the
100-point case the mean band width rose 33.7674 -> 34.8853 (+3.3 %) — that widening IS the
correctness fix, visible at the response level. At the 10-point floor it moved 2.0553 -> 2.0559,
because with 9 days of history the fitted deltas are tiny and the band is dominated by the fit
rather than by the changepoint simulation. Recorded rather than smoothed over: the fix is a
correctness fix at both extremes, but only one of them shows it in the output.

## CONTRACT IDENTITY AND THE D-15 SWAP (Task 3)

**`float_tolerance = 0.01`**, tightened from the transient `const REL_TOLERANCE: f64 = 0.02;` that
Tasks 1-2 read. The arithmetic that makes it a bar rather than a fudge:

| lambda | SE = sqrt(lambda/N) | SE relative | headroom at 0.01 |
|---|---|---|---|
| 5 | 0.015811 | 0.3162 % | **3.16 sigma** (the tightest point) |
| 29 | 0.038079 | 0.1313 % | 7.62 sigma |
| 31 | 0.039370 | 0.1270 % | 7.87 sigma |
| 100 | 0.070711 | 0.0707 % | 14.14 sigma |
| 900 | 0.212132 | 0.0236 % | 42.43 sigma |
| 2839 | 0.376763 | 0.0133 % | 75.35 sigma |

The measured post-fix maximum is **0.214 %** (lambda 31) and the pre-fix failure at lambda 900 is
**17.2 %** — **17x the bar**. So 0.01 has real headroom over the noise at the noisiest point and
is still an order of magnitude inside the defect it catches.

**THE D-15 SWAP, PROVEN MECHANICALLY.** The file-local const was deleted and the single assertion
site now reads `equation_tolerance("prophet-parity-v1", "poisson_sampler_domain")`. With
`equations.poisson_sampler_domain.float_tolerance` mutated to `1.0e-9`:

```
$ cargo test -p aprender-forecast --lib prophet::sampler          # contract value 1.0e-9
thread 'prophet::sampler::poisson_mean_tracks_lambda_across_its_whole_domain' panicked at
crates/aprender-forecast/src/prophet.rs:989:13:
poisson sampler outside its domain at lambda=5.0: mean=5.0076 rel=0.001510 > bar=1e-9
test result: FAILED. 1 passed; 1 failed; 1 ignored; 0 measured; 97 filtered out
rc=101
```

restored to `0.01`:

```
$ cargo test -p aprender-forecast --lib prophet::sampler          # contract value 0.01
test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 97 filtered out
rc=0
```

**The verdict moved with the contract value**, which is the only mechanical proof that the bar is
no longer a literal. `git status --porcelain -- contracts/prophet-parity-v1.yaml` is empty as of
this plan's commits — the 1e-9 was restored, not left in.

### `pv diff` and the version

`pv diff` was given TWO FILESYSTEM PATHS (CLAUDE.md: never a git revision), the old one
materialised with `git show`:

```
Contract diff: v1.0.0 → v1.0.0
Suggested bump: minor

  equations:
    + poisson_sampler_domain
  proof_obligations:
    + invariant:The changepoint-count sampler operates INSIDE its numerical domain for every lambda
      the forecast door can produce, not merely for the lambdas the parity ladder happens to reach
  falsification_tests:
    + FALSIFY-PROPHET-013
```

Applied verbatim: **`prophet-parity-v1.yaml` 1.0.0 -> 1.1.0**.

| measure | before | after |
|---|---|---|
| `pv status` Equations / Obligations / Falsification tests | 12 / 12 / 12 | **13 / 13 / 13** |
| `float_tolerance` KEYS in the file (`^    float_tolerance:`) | 11 | **12** (exactly +1) |
| `make contract-audit-phase6` resolved Phase 6 binding rows | 57 (06-12) | **58** |

The contract's invariants cite the numbers the verifier measured — `745.13`, `922` and `2839` all
appear in the file — and record that Python Prophet uses `np.random.poisson`, so this is a parity
defect and not a design difference.

## Verification Results

| check | result |
|---|---|
| `cargo test -p aprender-forecast --lib prophet::sampler -- --nocapture` | rc=0, **6** `POISSON MEAN:` lines, `2 passed; 0 failed; 1 ignored` |
| `cargo test -p aprender-forecast --lib prophet::parity` (before extraction) | rc=0, **32 passed; 0 failed** |
| `cargo test -p aprender-forecast --lib prophet::parity` (after extraction) | rc=0, **32 passed; 0 failed** |
| `cargo test -p aprender-forecast --lib prophet::parity` (after the branch) | rc=0, **32 passed; 0 failed** |
| `cargo test -p aprender-forecast --lib` | rc=0, **91 passed; 0 failed; 9 ignored** |
| `pv validate contracts/prophet-parity-v1.yaml` | rc=0, **0 error(s), 0 warning(s)** |
| `make contract-validate` | rc=0, `Contract validation passed` |
| `make contract-audit-phase6` | rc=0, **0** `BIND-`, **0** `RESOLVE-`, **4** `Total equations:` summaries, `resolved 58 Phase 6 binding rows` |
| `cargo clippy -p aprender-forecast --all-targets --no-deps -- -D warnings` | rc=0 |
| `cargo fmt --all -- --check` | rc=0 |

Every command was run through `rtk proxy` and its exit status captured with `rc=$?` on the command
itself, never through a pipe (CLAUDE.md Verification Discipline rule 1). The `make contract-audit-phase6`
banner and the `Total equations:` count are the specific outputs the `rtk` Bash hook was previously
measured to truncate, which is why they were read from a proxied run.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] A sigma-headroom figure in the new contract invariant was wrong by ~6x**

- **Found during:** SUMMARY authoring, re-deriving the numbers before writing them down
- **Issue:** The `poisson_sampler_domain` invariant claimed 1 % is "~3.2 sigma [at lambda 5] and >= 46 sigma at every other point". Recomputed: the floor among the other five lambdas is **7.62 sigma** (lambda 29), not 46. The 46 was an unchecked estimate that would have shipped as a contract assertion.
- **Fix:** Replaced the claim with the per-lambda figures (3.16 / 7.62 / 7.87 / 14.1 / 42.4 / 75.4). No tolerance changed; the bar is still 0.01 and still has 3.16 sigma at the tightest point.
- **Files modified:** `contracts/prophet-parity-v1.yaml`
- **Verification:** `pv validate` 0 error(s); `prophet::sampler` 2 passed / 0 failed; `make contract-audit-phase6` rc=0, 58 rows
- **Committed in:** `b166573a8`

**2. [Rule 2 - Missing Critical] The parity argument shipped as an ENFORCED test, not only as a SUMMARY sentence**

- **Found during:** Task 1 (measuring the fixture lambda)
- **Issue:** The plan asked for the fixture lambda to be measured by a *temporary probe* and recorded in the SUMMARY. That leaves the load-bearing claim — "no parity rung crosses the branch threshold" — unenforced: a future fixture regeneration or an `n_changepoints` change could quietly push the ladder onto the new branch, and nothing would notice.
- **Fix:** Made the probe a permanent test, `wp_log_r_logistic_fixture_lambda_is_far_below_the_branch_threshold`, asserting `lambda < POISSON_NORMAL_BRANCH_LAMBDA`. It prints `FIXTURE LAMBDA:` for the record and goes red if the argument ever stops holding.
- **Files modified:** `crates/aprender-forecast/src/prophet.rs`
- **Verification:** passes at lambda 3.1239; the assertion references the same const the branch uses, so raising the threshold cannot silently invalidate it
- **Committed in:** `596b08e56` (Task 1), retargeted to the named const in `b865acae9` (Task 2)

**3. [Rule 2 - Missing Critical] The wall harness was parameterised so BOTH reachable extremes could be measured**

- **Found during:** Task 2 step 4
- **Issue:** The plan named one worst case (100 points / horizon 3650). Measured, that case is only a **1.24x** count increase (745 -> ~922) — but the threat register's T-06-30 is about the **~3.8x** case, which is the 10-point `MIN_POINTS` floor at lambda ~2839. Measuring only the 100-point case would have reported the smaller multiple as if it were the worst.
- **Fix:** `logistic_band_wall` reads `LOGISTIC_BENCH_POINTS` / `LOGISTIC_BENCH_HORIZON` (defaults 100 / 3650), matching the `design_cost::holiday_design_wall` convention, and both extremes were walled before and after.
- **Files modified:** `crates/aprender-forecast/src/prophet.rs`
- **Verification:** four `LOGISTIC BAND WALL:` lines, quoted above; both after-walls under 0.25 s against a 2 s bar
- **Committed in:** `b865acae9`

---

**Total deviations:** 3 auto-fixed (1 bug, 2 missing critical)
**Impact on plan:** No scope creep. Deviation 1 corrects a false number this plan itself introduced; deviations 2 and 3 both strengthen evidence the plan already required, by making a recorded claim enforceable and by measuring the extreme the threat register actually names. No tolerance was loosened, no parity bar moved, and no prohibited file was touched.

### Prohibitions — all four honoured

| prohibition | evidence |
|---|---|
| Never loosen a parity tolerance | No parity tolerance changed. `prophet::parity` is 32 passed / 0 failed at all three measurement points, and the threshold did not need raising (fixture lambda 3.1239 vs threshold 30.0) |
| Never leave the out-of-domain behaviour silent | The **valid sampler branch** was taken — the first of the three accepted outcomes. Nothing is clamped and nothing is silently narrowed |
| The shipped test never hardcodes the tolerance | The transient `const REL_TOLERANCE` is deleted (`grep -c REL_TOLERANCE` = 0); the single assertion site reads `equation_tolerance("prophet-parity-v1", "poisson_sampler_domain")`, proven by the 1e-9 mutation control |
| Never write the contract from Task 1 or 2 | `git status --porcelain -- contracts/prophet-parity-v1.yaml contracts/aprender/binding.yaml` printed `[]` at the end of Task 1, and Tasks 1-2 committed only `crates/aprender-forecast/src/prophet.rs`. The tolerance key entered the file exactly once, in `f8bb1e608` |
| Scope fence (no `aprender-compute`, no workflows, no fixtures) | The whole plan touched three files: `prophet.rs`, `prophet-parity-v1.yaml`, `binding.yaml` |

### A note on the anticipated clippy allow

The plan expected the numeric cast in the new branch to need a per-site
`#[allow(clippy::cast_..., reason = "...")]`. It did not: `cast_precision_loss`,
`cast_possible_truncation`, `cast_sign_loss` and `cast_lossless` are **already** allowed at the
workspace level in the root `Cargo.toml` (`[workspace.lints.clippy]`, pre-existing ML-cast
allows). Nothing was widened — the prohibition against widening a workspace lint is satisfied by
not having touched it. `cargo clippy -p aprender-forecast --all-targets --no-deps -- -D warnings`
is rc=0.

## Issues Encountered

- **`rtk` output filtering, as every prior executor in this phase measured.** Every verification
  command was run under `rtk proxy` and redirected to a log file whose contents were then read, so
  no `test result:` line or `resolved N binding rows` banner was judged from a filtered stream.
- **`grep -c 'fn poisson'` returns 2, not the 1 the acceptance criterion literally asks for.** The
  extra match is the *test function's own name*
  (`fn poisson_mean_tracks_lambda_across_its_whole_domain`). The criterion's intent — exactly one
  definition of `poisson` — is met: `grep -c '^pub fn poisson'` is **1**. Recorded rather than
  silently reinterpreted.
- **`grep -c 'float_tolerance'` in the contract went 25 -> 29, not +1.** Three of the four extra
  lines are *prose references* to the key (in `formula`, in an invariant and in the obligation's
  `formal`), which is what a contract that explains its own bar looks like. The criterion's intent
  — the tolerance KEY appears once — is met exactly: `grep -c '^    float_tolerance:'` went
  **11 -> 12**.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **VERIFICATION gap 3 is closed on both `missing` bullets.** Bullet 1 (a branch above lambda ~30
  rather than silent degradation) is Task 2; bullet 2 (a falsification test asserting the mean at
  lambda 900 is within a few percent) is Tasks 1 and 3, observed red first and now contract-owned.
- **This is the last plan of phase 06.** The tree is green: `cargo test -p aprender-forecast --lib`
  91 passed / 0 failed, clippy rc=0, fmt rc=0, `make contract-validate` rc=0,
  `make contract-audit-phase6` rc=0 with 58 resolved rows and zero findings.
- **The three `human_verification` items remain OPEN and were NOT planned as automated work by
  this plan.** They are untouched: (1) the browser MCP handshake + band-chart rendering check on
  both demo pages; (2) the x86_64 `quantiles_abs_f32_nonaarch64` measurement for
  `chronos-bolt-parity-v1.yaml`; (3) the SC4-dark-in-CI decision. This plan touched no surface
  belonging to any of them.
- **One judgment item is added, not closed** (coverage D8): whether a 10-point history extrapolated
  3650 days should be *served at all* rather than refused at the door. The sampler is now correct
  in that regime; whether the regime should be admitted is a `MIN_POINTS` vs `MAX_HORIZON` product
  decision that no test can make.

---
*Phase: 06-native-time-series-forecasting-stack*
*Completed: 2026-09-07*

## Self-Check: PASSED

All three modified files exist on disk; all five commits (`596b08e56`, `b865acae9`, `f8bb1e608`,
`b166573a8`, `22b2c6a6f`) are present in `git log --oneline --all`.
