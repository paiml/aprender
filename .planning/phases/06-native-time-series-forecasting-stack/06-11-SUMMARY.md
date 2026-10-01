---
phase: 06-native-time-series-forecasting-stack
plan: 11
subsystem: api
tags: [rust, prophet, forecasting, dos, resource-exhaustion, input-validation, provable-contracts, benchmarking]

requires:
  - phase: 06-native-time-series-forecasting-stack
    provides: "the forecast door (06-01..06-05), the SC1 forecast-bench recipe and POOL SPEEDUP printing convention (06-08), the cap-is-logistic-only door refusal this one sits beside (06-10)"
provides:
  - "A MEASURED attribution of the 16.113 s in-bounds wall that REFUTES the code-read hypothesis: the design build is 13 ms of it, the fit is 99.8%"
  - "`prophet::holiday_day_sets` + an O(1) `HashSet<i64>` membership rewrite of `feature_row`, with every parity number unmoved"
  - "`MAX_HOLIDAY_DESIGN_COST` (50 000) and `MAX_HOLIDAY_DATES_TOTAL` (10 000), contract-owned and enforced at THE door before `make_design`"
  - "`constants.fit_max_holiday_design_cost` / `constants.fit_max_holiday_dates_total` in forecast-tool-boundary-v1.yaml, asserted equal by a re-mutated `cost_bounds_match_contract`"
  - "`just forecast-holiday-bench` — the host-gated release measurement recipe, defaulted to the worst ACCEPTED shape so 06-12's 2 s bar has something to bite on"
  - "The measured finding that NO payload statistic bounds the wall — the fit's iteration count is data-dependent, so these bounds cap WORK, not WALL"
affects: [06-12, 06-13, forecast-tool-boundary, gsd-verify-work]

actuals:
  tokens: 6157
  tasks: 2
  commits: 3
plan_head_before: 699de27a0a3986b6fce13ed72ce96959634f26d6

tech-stack:
  added: []
  patterns:
    - "Attribute before fixing: an ignored release-profile harness prints ONE machine-parsable line splitting the wall into design/fit/predict from the response's own fit_seconds/predict_seconds, at several scaling points and against a control, BEFORE any code is changed"
    - "Bound the PRODUCT at the door, on the statistic the measurements support rather than the one the code reading suggests — and say in the constant's doc comment which of WORK and WALL it actually bounds"
    - "Re-mutate an extended guard in its NEW scope: perturb each newly-added contract value by one and observe the guard failing while NAMING that key"

key-files:
  created: []
  modified:
    - crates/aprender-forecast/src/prophet.rs
    - crates/aprender-forecast/src/types.rs
    - crates/aprender-forecast/src/forecast.rs
    - contracts/forecast-tool-boundary-v1.yaml
    - justfile

key-decisions:
  - "The verifier's attribution of the 16.113 s wall to `feature_row` is REFUTED by measurement: the design build is 0.013 s of 16.081 s (0.08%) and the FIT is 16.051 s (99.8%). Reported as a finding rather than quietly working around"
  - "Bounded `(points + horizon) x holiday_columns` rather than the verifier's literal triple product, as `<bound_statistic_decision>` anticipated — two configurations with near-equal triple products (4.00e7 vs 4.56e7) measured 4.3x apart in wall, and the ordering follows cells"
  - "MAX_HOLIDAY_DESIGN_COST = 50 000: the largest round value whose three at-the-bound compositions all clear 2 s (1.174 / 1.692 / 0.106 s). 100 000 fails two of the three (2.771 / 2.305 s)"
  - "MAX_HOLIDAY_DATES_TOTAL = 10 000: 588x the largest total any committed fixture sends (17), and 0.115 ms of set construction against the 50 ms bar"
  - "Plain multiply plus an overflow-impossibility comment, NOT `saturating_mul` — all three factors are already refused above, so the product is at most 23 650 000 and `saturating_mul` would obscure that rather than add safety"
  - "The bound caps WORK, not WALL, and the SUMMARY says so: a 25 000-cell request (half the bound) reproducibly walls at 4.2 s because the fit's iteration count is data-dependent"
  - "No hand semver bump: `pv diff` reports `Contracts are identical` because `constants:` is a dropped top-level key. Recorded rather than worked around; 06-12's equation is what will make it contract-visible"

patterns-established:
  - "Attribution harness first, fix second: build the measurement before touching the code, so the fix is aimed by numbers rather than by a code reading"
  - "The bench recipe's DEFAULTS are the worst shape the door still accepts, so a later wall-clock bar is asserted against the configuration that actually stresses it"

requirements-completed: [SC1]

coverage:
  - id: D1
    description: "The 16.113 s in-bounds wall is ATTRIBUTED by measurement rather than inferred: reproduced at 16.081 s on this release host and split into design / fit / predict at five configurations plus a no-holiday control, showing the design build is 0.08% of it and the fit 99.8%"
    requirement: "SC1"
    verification:
      - kind: other
        ref: "just forecast-holiday-bench {1000 61 84 365 | 2000 121 84 365 | 3000 181 84 365 | 20000 1000 1 365} + just forecast-bench (release, aarch64) — five HOLIDAY DESIGN WALL lines recorded below"
        status: pass
    human_judgment: false
  - id: D2
    description: "Holiday membership is an O(1) prebuilt `HashSet<i64>` lookup built once per `make_design` / `predict` call, and no parity number moved"
    requirement: "SC1"
    verification:
      - kind: unit
        ref: "cargo test -p aprender-forecast --lib prophet::parity (32 passed / 0 failed, identical before and after the rewrite)"
        status: pass
    human_judgment: false
  - id: D3
    description: "The choice of bounding statistic is settled by measurement, not argument: two configurations with near-equal triple products measure 4.3x apart in wall and the ordering follows `(points + horizon) x holiday_columns`"
    requirement: "SC1"
    verification:
      - kind: other
        ref: "just forecast-holiday-bench 20000 1000 1 365 (triple 4.00e7, cells 2.04e7, 70.089 s) vs 3000 181 84 365 (triple 4.56e7, cells 6.09e5, 16.216 s)"
        status: pass
    human_judgment: false
  - id: D4
    description: "The product of the three separately-bounded factors is bounded at THE door, before `make_design`: an in-bounds spec whose product is not is refused, a spec exactly at the bound is accepted, and holidays whose date total exceeds the aggregate bound are refused"
    requirement: "SC1"
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::an_in_bounds_holiday_spec_whose_product_is_not_is_refused"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::a_holiday_spec_just_under_the_design_cost_bound_is_accepted"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::holidays_carrying_more_dates_than_the_total_bound_are_refused"
        status: pass
      - kind: other
        ref: "just forecast-holiday-bench 3000 181 84 365 — the exact 16.081 s configuration is now refused with `609065 design feature cells ... exceeds max_holiday_design_cost 50000`"
        status: pass
    human_judgment: false
  - id: D5
    description: "Both new bounds are contract-owned, and the extended guard was proven to bite in its NEW scope: each new contract key perturbed by one, observed failing while naming that key, then reverted byte-exactly"
    requirement: "SC1"
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/types.rs#types::tests::cost_bounds_match_contract (1 passed; both re-mutation failure messages quoted verbatim below)"
        status: pass
      - kind: other
        ref: "cargo run --release -p aprender-contracts-cli --bin pv -- validate contracts/forecast-tool-boundary-v1.yaml (0 error(s))"
        status: pass
    human_judgment: false
  - id: D6
    description: "The no-holiday SC1 path is untouched — with no holidays both new products are 0, so neither refusal can fire"
    requirement: "SC1"
    verification:
      - kind: other
        ref: "just forecast-bench — ROUND TRIP OK: 0.214 s < 2.0 s (SC1)"
        status: pass
    human_judgment: false
  - id: D7
    description: "SC1's 2 s bar for a HOLIDAY-CARRYING request is still not guaranteed, and the measurements say why: the fit's iteration count is data-dependent, so a 25 000-cell request (half the bound) reproducibly walls at 4.2 s"
    verification: []
    human_judgment: true
    rationale: "This is a finding, not a deliverable — it says the gap the verifier opened cannot be fully closed by any payload bound, and the residual wall-clock exposure is FIT_BUDGET_SECS (15 s), a separate already-owned knob. A human must decide whether SC1's 2 s bar applies to holiday-carrying requests at all, whether FIT_BUDGET_SECS should drop, or whether the bound should tighten further at the cost of refusing ordinary multi-holiday usage. No test can make that call."

duration: 1h 21m
completed: 2026-09-07
status: complete
---

# Phase 06 Plan 11: Bound the holiday design cost at THE door Summary

**The 16.113 s in-bounds wall was measured rather than inferred — and the measurement refuted the code reading: the design build is 13 ms of it and the FIT is 99.8% — so the fix is an O(1) `HashSet` membership rewrite that moves no parity number, plus two contract-owned product bounds (`MAX_HOLIDAY_DESIGN_COST` 50 000, `MAX_HOLIDAY_DATES_TOTAL` 10 000) enforced at the door before `make_design`, with the values derived from twenty release-profile measurements.**

## Performance

- **Duration:** 1h 21m
- **Started:** 2026-09-07T02:07:00Z
- **Completed:** 2026-09-07T03:28:16Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- **The wall is ATTRIBUTED, not assumed.** A release-profile harness reproduced the verifier's 16.113 s at **16.081 s** and split it with the response's own `fit_seconds` / `predict_seconds`. `other_s` — design build plus validation — is **0.013 s**. The verifier attributed the wall to `feature_row` from a code reading; the numbers say the FIT is 99.8% of it.
- **Holiday membership is O(1).** `prophet::holiday_day_sets` builds one `HashSet<i64>` per holiday once per `make_design` / `predict`, and `feature_row` answers the identical predicate (`d + off == day` iff `d == day - off`) by lookup. `prophet::parity` is **32 passed / 0 failed** on both sides.
- **The product is bounded at THE door.** `MAX_HOLIDAY_DESIGN_COST` bounds `(points + horizon) x holiday_columns` and `MAX_HOLIDAY_DATES_TOTAL` bounds `sum(len(dates))`, both refused after the holiday loop and before `make_design`. The exact configuration that measured 16.081 s is now refused.
- **The bounds are contract-owned and the guard was re-mutated in its new scope.** Each new key perturbed by one, observed failing while naming that key, reverted byte-exactly.
- **The honest limit is recorded.** No payload statistic bounds the wall — the fit's iteration count is data-dependent, reproducibly. These constants cap WORK. Stated in the constant's own doc comment, not just here.

## Task Commits

1. **Task 1: attribute the wall by measurement, make membership O(1)** — `d83fc008f` (perf)
2. **Task 2: bound the product at the door, contract-owned, guard re-mutated** — `475e2f9ab` (feat)
3. **Deviation fix: point the bench at the worst ACCEPTED shape** — `42a3ffbf0` (fix)

## Files Created/Modified

- `crates/aprender-forecast/src/prophet.rs` — `holiday_day_sets`, the `feature_row` signature change and its two hoisted call sites, and `mod design_cost` (the ignored release-profile harness)
- `crates/aprender-forecast/src/types.rs` — `MAX_HOLIDAY_DESIGN_COST`, `MAX_HOLIDAY_DATES_TOTAL`, two new rows in `cost_bounds_match_contract`, and a correction to the now-false `MAX_HOLIDAY_DATES` doc comment
- `crates/aprender-forecast/src/forecast.rs` — `holiday_dates_total` accumulation, the two door refusals, the `holiday_args` helper and three new door tests
- `contracts/forecast-tool-boundary-v1.yaml` — `constants.fit_max_holiday_design_cost`, `constants.fit_max_holiday_dates_total`, and the comment sentence naming the product they close
- `justfile` — `forecast-holiday-bench`

---

## THE ATTRIBUTION (Task 1) — five configurations, release, aarch64

`total_s` from an `Instant` around `forecast()`; `fit_s` and `predict_s` read off the response's public fields; `other_s = total_s - fit_s - predict_s` is the design-build-plus-validation share. `cells = (points + horizon) x columns`; `triple = points x columns x dates_total`.

### BEFORE the membership rewrite

| # | config (points/cols/dates) | cells | triple | total_s | fit_s | predict_s | **other_s** |
|---|---|---|---|---|---|---|---|
| a | 1000 / 61 / 84 | 83 265 | 5.12e6 | 2.405 | 2.387 | 0.016 | **0.002** |
| b | 2000 / 121 / 84 | 286 165 | 2.03e7 | 7.930 | 7.907 | 0.016 | **0.006** |
| c | 3000 / 181 / 84 | 609 065 | 4.56e7 | **16.081** | 16.051 | 0.017 | **0.013** |
| d | 3000 / **0 holidays** (control, `just forecast-bench`) | 0 | 0 | **0.216** | 0.201 | 0.015 | — |
| e | 20000 / 1000 / 2 | 20 365 000 | 4.00e7 | **69.640** | 69.590 | 0.018 | **0.032** |

### AFTER the membership rewrite

| # | config | total_s | fit_s | predict_s | **other_s** | vs before |
|---|---|---|---|---|---|---|
| a | 1000 / 61 / 84 | 2.435 | 2.419 | 0.015 | **0.001** | 0.002 -> 0.001 |
| b | 2000 / 121 / 84 | 7.851 | 7.832 | 0.016 | **0.003** | 0.006 -> 0.003 |
| c | 3000 / 181 / 84 | 16.216 | 16.194 | 0.016 | **0.006** | 0.013 -> 0.006 |
| d | 3000 / 0 holidays | 0.212 | 0.201 | 0.015 | — | 0.216 -> 0.212 |
| e | 20000 / 1000 / 2 | 70.089 | 69.941 | 0.019 | **0.128** | 0.032 -> **0.128** |

### Which of design build or fit dominated the 3 000 x 181 x 84 wall — by the numbers

**The FIT dominated, overwhelmingly.** At 3 000 x 181 x 84 the wall is 16.081 s, of which `fit_s` is 16.051 s (**99.81%**) and the entire design build plus every validation check is `other_s` = **0.013 s** (**0.08%**). The verifier's `missing` bullet 2 attributed the wall to `feature_row`'s per-row linear scan, read from the source. That attribution is **refuted by measurement**: removing the scan entirely moved the wall from 16.081 s to 16.216 s — i.e. not at all, within run-to-run noise — because the scan was never more than 13 ms of it. The no-holiday control at the same point count (0.216 s) shows the 74x blow-up is real; it just lives in the fit, whose per-iteration cost is `O(rows x K)` over the design matrix the holiday columns widen.

The rewrite is still right and was still made: it is a strict algorithmic improvement (0.013 s -> 0.006 s at config c, 2.2x) and it removes a `dates`-shaped multiplier from the inner loop. It is simply **not** what made the request slow. Honest cost of the rewrite, also recorded: at config e the sets are built over only 2 dates and then probed 2.04e7 times, so `other_s` went **0.032 -> 0.128 s** — a `HashSet` probe is slower than a 1-element linear scan. That is 0.18% of that configuration's wall and does not change any conclusion, but it is a real regression on the degenerate shape and is not hidden here.

### The statistic test — the two shapes side by side

| shape | cells `(P+H)xC` | triple `PxCxD` | wall (after) |
|---|---|---|---|
| large-cells / small-triple: 20000 / 1000 / 2 | **20 365 000** | 4.00e7 | **70.089 s** |
| small-cells / large-triple: 3000 / 181 / 84 | **609 065** | 4.56e7 | **16.216 s** |

**These CONFIRM `<bound_statistic_decision>`.** The two triple products are within 14% of each other (4.00e7 vs 4.56e7) — and the second is the LARGER — yet the walls are **4.3x apart** with the ordering following `cells`, not `triple`. A single threshold on the triple product would therefore have to sit below 4.00e7 to refuse the 70 s request, which also refuses the 16 s one, and above 4.56e7 to accept the cheap shapes, which accepts the 70 s one. The statistic cannot separate them. `MAX_HOLIDAY_DESIGN_COST` is therefore on `(points + horizon) x holiday_columns`, and `MAX_HOLIDAY_DATES_TOTAL` closes the third factor additively — both of the verifier's factors bounded, on the statistic each is monotone in.

`FIT_BUDGET_SECS` (15 s) is confirmed as no substitute, and for a stronger reason than the plan assumed. It is blind to `make_design` as expected — but it is also blind to a long ROUND: config e walled at **70.089 s** against a 15 s cooperative round-boundary budget, a **4.7x overshoot inside the fit**. Lowering `FIT_BUDGET_SECS` could not have caught this.

---

## THE DERIVATION OF THE TWO CONSTANTS (Task 2)

### `MAX_HOLIDAY_DESIGN_COST = 50 000` — three at-the-bound compositions

Per CLAUDE.md rule 6, the value is not fixed from one measurement. Three configurations composing the SAME product differently, each measured end to end on the release host:

| composition | points x horizon x columns | cells | wall | under 2 s? |
|---|---|---|---|---|
| many rows / few columns | 9500 + 500 rows x 5 cols | 50 000 | **1.174 s** | yes |
| balanced | 800 + 200 rows x 50 cols | 50 000 | **1.692 s** | yes |
| few rows / many columns | 50 + 50 rows x 500 cols | 50 000 | **0.106 s** | yes |

**100 000 was tested and REJECTED** — two of the same three compositions exceed the bar:

| composition | cells | wall | under 2 s? |
|---|---|---|---|
| balanced (800+200 x 100) | 100 000 | **2.771 s** | **no** |
| many rows / few columns (19700+300 x 5) | 100 000 | **2.305 s** | **no** |
| few rows / many columns (150+50 x 500) | 100 000 | 0.339 s | yes |

So 50 000 is the largest round value satisfying the plan's derivation rule. The structural maximum is `(20 000 + 3 650) x 1 000` = **23 650 000**, so the bound is a 473x reduction — decidedly not vacuous.

### The bound caps WORK, not WALL — and this is measured, not conceded

While deriving the value, a sub-bound configuration was found that exceeds 2 s. Rule 6 again: it was re-run and varied before being reported.

| config | cells | wall |
|---|---|---|
| 4700 pts / 5 cols / horizon 300 | 25 000 (**half the bound**) | **4.202 s**, re-run **4.222 s**, re-run **4.358 s** |
| 4700 pts / **1** col / horizon 300 | 5 000 | 0.487 s |
| 4700 pts / **0** holidays / horizon 300 | 0 | 0.925 s |
| 20000 pts / 1 col / horizon 365 | 20 365 | 2.087 s |

The wall is **not monotone in anything** — 4 700 points with 1 holiday column is *faster* (0.487 s) than the same series with none (0.925 s), and 5 columns is 8.6x slower than either. The cause is the L-BFGS iteration count, which is data-dependent and which no function of `(points, columns, dates)` predicts. **Therefore no payload bound can guarantee a sub-2-second wall**, and this SUMMARY does not claim one. What `MAX_HOLIDAY_DESIGN_COST` guarantees is the arithmetic ceiling per iteration, which is exactly what turns the ~85-minute in-bounds request the verifier extrapolated into a refusal. The residual wall-clock exposure is `FIT_BUDGET_SECS`, a separate already-owned knob — see the open item in `coverage: D7`.

### `MAX_HOLIDAY_DATES_TOTAL = 10 000`

**Largest total any committed fixture, example or test sends: 17.** Derived, not assumed — a script parsed every `crates/aprender-forecast/tests/fixtures/*.json` and grouped Prophet's long-form holidays frame by name, plus a regex sweep of every literal date array in `forecast.rs`, `prophet.rs`, `aprender-mcp-forecast/src/lib.rs` and the demo page:

```
fixture peyton_holidays_prophet140.json grouped: {'playoff': 14, 'superbowl': 3} TOTAL dates = 17
crates/aprender-forecast/src/forecast.rs: literal dates array with 1 entries
crates/aprender-mcp-forecast/src/lib.rs: literal dates array with 7 entries
crates/aprender-mcp-forecast/static/index.html: 1 date literals total
```

**Set-construction cost, measured** (`rustc -O`, two sets per request — one in `make_design`, one in `predict`):

```
SET CONSTRUCTION: total_dates=17        two_sets_ms=0.001
SET CONSTRUCTION: total_dates=10000     two_sets_ms=0.115
SET CONSTRUCTION: total_dates=100000    two_sets_ms=1.486
SET CONSTRUCTION: total_dates=1000000   two_sets_ms=18.049
```

**Chosen: 10 000** — 588x the largest committed use, 0.115 ms of construction (435x under the 50 ms bar), and a 100x reduction from the structural maximum of 1 000 000 (1 000 holidays x 1 000 dates). Recorded honestly: the structural maximum ITSELF measures 18.049 ms, i.e. under the 50 ms bar, so this constant is not load-bearing for *set construction*. It is load-bearing for the door's `parse_date` loop and the ~10 MB payload 1 000 000 date strings would require, and it closes the verifier's third factor, which was otherwise unbounded in aggregate.

### Plain multiply, not `saturating_mul`

`design_cells = (ds.len() + args.horizon) * holiday_columns` is a **plain multiply with an overflow-impossibility comment**. All three factors are already refused above it — `ds.len() <= MAX_POINTS` (20 000), `args.horizon <= MAX_HORIZON` (3 650), `holiday_columns <= MAX_HOLIDAY_COLUMNS` (1 000, refused *inside* the loop) — so the product is at most 23 650 000, six orders of magnitude below `usize::MAX`. `saturating_mul` would signal that the factors might be unbounded, which is the opposite of what the surrounding code establishes.

---

## THE RE-MUTATION (CLAUDE.md Verification Discipline rule 4)

Extending `cost_bounds_match_contract` from five rows to seven extends its SCOPE; the old four-factor proof does not transfer. Each new contract value was perturbed by one, the guard OBSERVED failing while naming that key, then reverted and observed passing. Both failure messages verbatim:

**Mutation 1 — `fit_max_holiday_design_cost: 50000 -> 50001`:**

```
---- types::tests::cost_bounds_match_contract stdout ----

thread 'types::tests::cost_bounds_match_contract' (94171361) panicked at crates/aprender-forecast/src/types.rs:265:13:
assertion `left == right` failed: types::MAX_HOLIDAY_DESIGN_COST must equal constants.fit_max_holiday_design_cost in forecast-tool-boundary-v1
  left: 50000
 right: 50001

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 93 filtered out; finished in 0.00s
```

**Mutation 2 — `fit_max_holiday_dates_total: 10000 -> 10001`:**

```
---- types::tests::cost_bounds_match_contract stdout ----

thread 'types::tests::cost_bounds_match_contract' (94173099) panicked at crates/aprender-forecast/src/types.rs:265:13:
assertion `left == right` failed: types::MAX_HOLIDAY_DATES_TOTAL must equal constants.fit_max_holiday_dates_total in forecast-tool-boundary-v1
  left: 10000
 right: 10001

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 93 filtered out; finished in 0.00s
```

Each message names the **new** key, not a neighbouring one. After each revert, `cmp` against a pre-mutation copy reported **byte-identical**, and the guard returned `ok. 1 passed; 0 failed`.

---

## THE RED SIDE, OBSERVED (Task 2 step 6)

All three door tests were written and run BEFORE the door checks existed. Because they are three separate `#[test]` fns, no failure masks another:

```
running 3 tests
test forecast::tests::holidays_carrying_more_dates_than_the_total_bound_are_refused ... FAILED
test forecast::tests::a_holiday_spec_just_under_the_design_cost_bound_is_accepted ... ok
test forecast::tests::an_in_bounds_holiday_spec_whose_product_is_not_is_refused ... FAILED

failures:

---- forecast::tests::holidays_carrying_more_dates_than_the_total_bound_are_refused stdout ----

thread 'forecast::tests::holidays_carrying_more_dates_than_the_total_bound_are_refused' (94176271) panicked at crates/aprender-forecast/src/forecast.rs:565:22:
expected a Validation refusal, got Ok("prophet")

---- forecast::tests::an_in_bounds_holiday_spec_whose_product_is_not_is_refused stdout ----

thread 'forecast::tests::an_in_bounds_holiday_spec_whose_product_is_not_is_refused' (94176270) panicked at crates/aprender-forecast/src/forecast.rs:565:22:
expected a Validation refusal, got Ok("prophet")

failures:
    forecast::tests::an_in_bounds_holiday_spec_whose_product_is_not_is_refused
    forecast::tests::holidays_carrying_more_dates_than_the_total_bound_are_refused

test result: FAILED. 1 passed; 2 failed; 0 ignored; 0 measured; 94 filtered out; finished in 2.76s
```

`got Ok("prophet")` is the load-bearing detail: the door did not merely fail to refuse — it **accepted the request and fitted it**. The near-miss positive control passed in the same RED run, which is what proves it is a control and not a co-moving assertion.

After the checks:

```
running 3 tests
test forecast::tests::an_in_bounds_holiday_spec_whose_product_is_not_is_refused ... ok
test forecast::tests::holidays_carrying_more_dates_than_the_total_bound_are_refused ... ok
test forecast::tests::a_holiday_spec_just_under_the_design_cost_bound_is_accepted ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 94 filtered out; finished in 0.13s
```

### The bound bites end to end, on the verifier's own configuration

`just forecast-holiday-bench 3000 181 84 365` — the exact request that walled at 16.081 s — now exits 101 at the door:

```
thread 'prophet::design_cost::holiday_design_wall' panicked at crates/aprender-forecast/src/prophet.rs:1887:50:
the bench configuration must be accepted: Validation("holidays expand to 609065 design feature cells ((points + horizon) x holiday_columns = (3000 + 365) x 181), which exceeds max_holiday_design_cost 50000; reduce the holiday windows, the number of holidays, the history length or the horizon")
```

The message names the observed cell count, how it was computed, the bound and the fix.

---

## Measured before / after

| Measurement | Before | After |
|---|---|---|
| `cargo test -p aprender-forecast --lib` | 86 passed, 0 failed, 7 ignored | **89 passed, 0 failed, 8 ignored** |
| `... --lib prophet::parity` | **32 passed, 0 failed** | **32 passed, 0 failed — UNMOVED** |
| `... --lib forecast::tests::` | 10 passed, 0 failed | **13 passed, 0 failed** |
| `... --lib types::tests::cost_bounds_match_contract` | 1 passed (5 rows) | 1 passed (**7 rows**) |
| `cargo test -p aprender-mcp-forecast --lib` | 30 passed, 0 failed | 30 passed, 0 failed |
| `just forecast-bench` (no-holiday SC1) | 0.216 s | **0.214 s — unchanged** |
| `just forecast-holiday-bench` (default) | 3000/181/84, 16.081 s | **800/50/84/200 at the bound, 1.701 s** |
| `pv validate forecast-tool-boundary-v1.yaml` | 0 error(s) | 0 error(s) |
| `constants:` keys | 4 fit cost bounds | **6 fit cost bounds** |

The `+1 ignored` is the new measurement test appearing as a COUNTED ignore (D-18 discipline — never a println-and-return, never a silent absence). The `+3 forecast::tests::` are the three new door tests.

## `pv diff` and semver

Run per CLAUDE.md against two materialised filesystem paths:

```
$ git show HEAD:contracts/forecast-tool-boundary-v1.yaml > /tmp/ftb-old.yaml
$ cargo run --release -p aprender-contracts-cli --bin pv -- diff /tmp/ftb-old.yaml contracts/forecast-tool-boundary-v1.yaml
Contracts are identical.
```

**No bump applied.** `constants:` is a top-level key the contract parser deliberately drops (the file says so at its own lines 59-63), so `pv diff` compares the `Contract` struct and sees nothing. Recorded rather than worked around, per CLAUDE.md's "never work around `pv`": the behaviour DID narrow — a holiday-carrying request that was accepted (slowly) is now refused — and 06-12's equation plus falsification test are what will make that visible to `pv diff` and earn the bump. This is the same blind spot 06-10 recorded from the other side (it saw additions but not the narrowing); here it sees neither.

## Decisions Made

See `key-decisions` in the frontmatter. The two that a reader should not miss:

1. **The verifier's attribution was wrong and is reported as wrong.** The plan's own `<bound_statistic_decision>` invited the measurement to overrule it and it did. Nothing here quietly implements the plan's hypothesis while measuring something else.
2. **The bound caps WORK, not WALL.** Stating otherwise would have been the easy close, and it would have been false — a 25 000-cell request walls at 4.2 s, reproducibly, three runs.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Every plan `<verify>` grep was unreachable through the rtk hook**

- **Found during:** Task 1 (baseline run)
- **Issue:** The environment's rtk hook rewrites bare `cargo` / `just` invocations and replaces libtest's `test result:` line with a one-line summary. Every `<verify>` block in this plan greps `test result: ok\. N passed; 0 failed`; through the hook that matches nothing on a passing run, so the gate could only ever report failure — or, with an `rc`-only check, report a green that measured nothing.
- **Fix:** Ran every verification as `rtk proxy <cmd>`. Same resolution 06-10 reached one wave earlier; the plan's own prompt flagged it.
- **Files modified:** none (execution-harness only)
- **Verification:** `rtk proxy cargo test -p aprender-forecast --lib` printed the real libtest line.
- **Committed in:** n/a

**2. [Rule 3 - Blocking] Task 2's bound broke Task 1's own `<verify>` command**

- **Found during:** Task 2 close-out
- **Issue:** `just forecast-holiday-bench` defaulted to 3 000 / 181 / 84 — the verifier's configuration. After Task 2 the door refuses it (609 065 > 50 000), so the harness's `.expect("the bench configuration must be accepted")` panicked and the plan-level `<verification>` line "`just forecast-holiday-bench` — exactly one `HOLIDAY DESIGN WALL:` line" could not pass on a correct build.
- **Fix:** Defaulted the recipe to **800 / 50 / 84 / 200** — exactly 50 000 cells, and the slowest of the three at-the-bound compositions (1.701 s). This is strictly better than restoring the old default: 06-12's 2 s bar must be asserted against the worst shape the door still ACCEPTS, not against one it refuses. The old configuration remains reachable as an explicit argument, where it now demonstrates the refusal.
- **Files modified:** `justfile`
- **Verification:** `just forecast-holiday-bench` -> rc 0, exactly one `HOLIDAY DESIGN WALL:` line, `profile=release`, `cells=50000`, `total_s=1.701`.
- **Committed in:** `42a3ffbf0`

**3. [Rule 1 - Bug] A doc comment made false by Task 1 was corrected**

- **Found during:** Task 2
- **Issue:** `types.rs` `MAX_HOLIDAY_DATES`' doc comment read "`prophet::feature_row` scans this list per row per holiday column, so it is a second multiplier on the design build." After Task 1's rewrite that is no longer true, and it is exactly the sentence a future reader would use to reason about the cost.
- **Fix:** Rewritten to say the scan WAS a multiplier, that membership is now a prebuilt `HashSet` lookup, and that the aggregate is bounded by `MAX_HOLIDAY_DATES_TOTAL`.
- **Files modified:** `crates/aprender-forecast/src/types.rs`
- **Verification:** `cargo doc` path unchanged; the claim now matches `prophet::feature_row`'s implementation.
- **Committed in:** `475e2f9ab`

**4. [Documentation - measurement correction] The plan's causal hypothesis is refuted, not implemented**

- **Found during:** Task 1 step 3
- **Issue:** The plan's objective, threat register T-06-22 and the verifier's `missing` bullet 2 all attribute the 16.113 s wall to `feature_row`'s per-row linear scan. The measurement shows the scan is 0.013 s of it (0.08%).
- **Fix:** No code change beyond what the plan already required. The rewrite was made anyway (it is a real improvement and the plan's `must_haves` require it), and the attribution is reported as the plan's own `<bound_statistic_decision>` instructed — "the measurement decides".
- **Files modified:** none beyond plan scope
- **Verification:** the five-configuration table above; `other_s` never exceeds 0.2% of any wall.
- **Committed in:** `d83fc008f` (recorded in the commit body)

---

**Total deviations:** 3 auto-fixed (2 blocking, 1 bug) + 1 documented measurement correction with no extra code change.
**Impact on plan:** No scope creep — every file touched is in the plan's `files_modified`. Deviation 1 was required for any verification here to measure anything. Deviation 2 keeps the plan's own verification command runnable and improves what 06-12 will assert against. Deviation 4 is the plan working as designed: it told the executor to measure before fixing, and the measurement changed the answer.

## Prohibitions honoured

- **Never assert a wall-clock bar inside a libtest assertion** — `holiday_design_wall` asserts only that the call succeeded and `yhat.len() == horizon`. The bar lives in the host-gated recipe (and lands there in 06-12). VERIFIED by reading the test and by its passing in the default `--lib` run as an ignored test.
- **Never change a parity number** — `prophet::parity` is 32 passed / 0 failed before and after. VERIFIED by test, both sides.
- **Never bound with the literal triple product** — the measurements were taken and they CONFIRM the plan's arithmetic (two near-equal triple products, 4.3x apart in wall). The bound is on `(points + horizon) x holiday_columns`. VERIFIED by the statistic-test table.
- **Do not touch `crates/aprender-compute`, `.github/workflows/*.yml`, or any parity fixture** — `git status --porcelain | grep -E '\.github/workflows/|crates/aprender-compute/'` returned nothing at every commit. VERIFIED.

## Human verification items — ALL THREE REMAIN OPEN

None of the three items `06-VERIFICATION.md` routed to `human_verification` was touched, closed, or planned as automated work:

1. **Drive `initialize` -> `tools/list` -> `tools/call` on both demo pages in a browser and confirm a forecast is charted — OPEN.** This plan does not touch either demo page. The page's holiday field can now produce a new refusal, but nothing here asserts browser rendering.
2. **Measure `quantiles_abs_f32_nonaarch64` on an x86_64 host — OPEN.** Untouched.
3. **Decide whether SC4's Chronos ladder staying dark in CI is acceptable — OPEN.** Untouched.

A **fourth** item is opened by this plan and is recorded as `coverage: D7`: SC1's 2 s bar cannot be guaranteed for holiday-carrying requests by any payload bound, and a human must decide between tightening the bound (which would refuse ordinary multi-holiday usage — 15 country holidays with +-1 day windows on a 3 000-point series is ~151 000 cells, 3x over), lowering `FIT_BUDGET_SECS`, or scoping SC1's bar to the no-holiday shape it was measured on.

## Issues Encountered

- **The recipe's own default became unreachable** after Task 2 — see deviation 2. Caught by re-running the plan-level verification at close-out rather than trusting the Task 1 run, which is why the close-out re-run exists.
- **A shell loop clobbered `$1` while sweeping configurations**, producing two spurious `exit 127` recipe failures. Diagnosed immediately (the log files did not exist, so nothing had been measured), replaced with a `while read` script. No measurement in this SUMMARY came from that run.

## Known Stubs

None. No stub, placeholder, TODO, FIXME or hardcoded empty value was introduced. Scanned across all five modified files.

## Threat Flags

None. The change removes attack surface (T-06-22, T-06-23, T-06-25 mitigated at the door; T-06-24's contract claim moved closer to true) and adds none: no new endpoint, no new auth path, no file access, no schema change at a trust boundary. **T-06-SC not applicable** — `std::collections::HashSet` is standard library; no package-manager install ran and no dependency entered the graph.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Gap 2's `missing` bullets 1 and 2 are closed here. Bullet 3 (an e2e case proving an over-cost request is refused through a live server) is 06-12's, as the plan scoped it — the e2e suite lives in `crates/aprender-mcp-forecast/src/lib.rs`, which this plan does not touch.
- **06-12 inherits three things it should not re-derive:** the recipe now defaults to the worst ACCEPTED shape (800/50/84/200, 1.701 s), so its 2 s bar has the right target; the two refusal message substrings `max_holiday_design_cost` and `max_holiday_dates_total` are pinned by unit tests and ready to assert through the server; and the contract-visible identity (equation + proof obligation + falsification test) is still missing, which is why `pv diff` currently reports `Contracts are identical`.
- **06-12 must read `coverage: D7` before writing its bar.** A 2 s assertion on this recipe is honest at the at-the-bound default and would be false as a general SC1 claim for holiday-carrying requests.
- Blockers: none.

## Self-Check: PASSED

Files claimed modified — all present on disk:

```
FOUND: crates/aprender-forecast/src/prophet.rs
FOUND: crates/aprender-forecast/src/types.rs
FOUND: crates/aprender-forecast/src/forecast.rs
FOUND: contracts/forecast-tool-boundary-v1.yaml
FOUND: justfile
```

Commits claimed — all present in history:

```
FOUND: d83fc008f
FOUND: 475e2f9ab
FOUND: 42a3ffbf0
```

Plan-level `<verification>` block, re-run at close-out:

| Check | Result |
|---|---|
| `cargo test -p aprender-forecast --lib prophet::parity` | `ok. 32 passed; 0 failed` (before AND after the rewrite) |
| `cargo test -p aprender-forecast --lib` | `ok. 89 passed; 0 failed; 8 ignored` |
| `cargo test -p aprender-forecast --lib types::tests::cost_bounds_match_contract` | `ok. 1 passed; 0 failed`, and OBSERVED failing under a one-off perturbation of each new key |
| `just forecast-holiday-bench` | one `HOLIDAY DESIGN WALL:` line, `profile=release`, `cells=50000`, `total_s=1.701` |
| `just forecast-bench` | `ROUND TRIP OK: 0.214 s < 2.0 s (SC1)` |
| `pv validate contracts/forecast-tool-boundary-v1.yaml` | `0 error(s), 0 warning(s) / Contract is valid.` |
| `cargo test -p aprender-mcp-forecast --lib` | `ok. 30 passed; 0 failed` |
| `cargo clippy -p aprender-forecast -p aprender-mcp-forecast --all-targets --no-deps -- -D warnings` | rc=0 |
| `cargo fmt --all -- --check` | rc=0 |

Acceptance criteria greps, re-run:

```
prophet.rs HashSet: 3 (>=3)          prophet.rs holiday_day_sets(: 3 (>=3)
prophet.rs fn holiday_day_sets: 1    prophet.rs HOLIDAY DESIGN WALL:: 1
types.rs MAX_HOLIDAY_DESIGN_COST: 5 (>=3)    forecast.rs: 3 (>=2)
types.rs MAX_HOLIDAY_DATES_TOTAL: 5 (>=3)    forecast.rs: 3 (>=2)
contract fit_max_holiday_design_cost: 2 (>=1, inside constants:)
contract fit_max_holiday_dates_total: 2 (>=1, inside constants:)
scope fence: no .github/workflows/ or crates/aprender-compute/ file touched
```

Commit count MEASURED, not narrated: `git rev-list --count 699de27a0..HEAD` = **3**.

---
*Phase: 06-native-time-series-forecasting-stack*
*Completed: 2026-09-07*
