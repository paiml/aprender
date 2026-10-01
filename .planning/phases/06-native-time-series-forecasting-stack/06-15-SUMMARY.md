---
phase: 06-native-time-series-forecasting-stack
plan: 15
subsystem: api
tags: [forecasting, prophet, neuralprophet, dos-bounds, provable-contracts, mcp, rust]

requires:
  - phase: 06-native-time-series-forecasting-stack
    provides: "06-14's door_surface enumeration (C-07 and C-08 are its findings), its unbounded_pending_06_15 markers, its DECLARED-RED types::tests::no_cost_axis_is_pending, and the straddling-e2e-pair + raised-constant RED template from 06-12"
provides:
  - "types::MAX_HOLIDAY_NAME_LEN (200 bytes) + constants.fit_max_holiday_name_len — cost axis C-07, the one caller-settable field whose door_surface.knobs entry read `enforced_by: NOTHING`"
  - "types::MAX_NP_TRAIN_COST (15 000 000) + constants.fit_max_np_train_cost — cost axis C-08, the one axis with no budget of ANY kind on its path"
  - "np::train_cost, np::n_training_samples, np::door_epochs, np::door_lr_sweep, np::request_train_cost — ONE implementation of the price, which the door also uses to CONFIGURE the sweep"
  - "the aggregate holiday-date ceiling enforced INSIDE the holiday loop (WR-03), with the post-loop exact-total refusal retained"
  - "np::wall::np_train_wall — an #[ignore]d release harness with 7 modes printing a machine-parsable NP TRAIN WALL: line carrying profile=, train_cost=, outcome= and total_s="
  - "forecast::wr03_wall::wr03_aggregate_dates_wall — the same for WR-03's avoided work"
  - "door_surface.cost_axes with NO pending marker: types::tests::no_cost_axis_is_pending goes RED -> GREEN, earned"
affects: [06-16, 06-17]

actuals:
  tokens: 26600
  tasks: 3
  commits: 8
plan_head_before: 73ca9cb32250196fb83f56700070a91f6344f7a1

tech-stack:
  added: []
  patterns:
    - "Close a declared-red window by REPLACING what made it red, never by editing the assertion — and quote both outputs side by side"
    - "Let the measurement REJECT a candidate bound: 20 000 000 measured 2.089 s, over the bar, and was discarded before 15 000 000 was measured"
    - "When the at-the-bound positive control is unaffordable in the always-run suite, MEASURE what it would cost (45.619 s debug) and say so, then pin the boundary on the door's own named comparison instead"
    - "Name the door's price ONCE and build the door's own configuration out of the same functions, so the price is the work"

key-files:
  created: []
  modified:
    - crates/aprender-forecast/src/forecast.rs
    - crates/aprender-forecast/src/types.rs
    - crates/aprender-forecast/src/np.rs
    - crates/aprender-mcp-forecast/src/lib.rs
    - contracts/forecast-tool-boundary-v1.yaml
    - contracts/aprender/binding.yaml

key-decisions:
  - "MAX_NP_TRAIN_COST = 15 000 000 was chosen by measurement AND a first candidate was rejected by it: 20 000 000 measured 2.089 s on the long-history composition, over SC1's 2 s bar"
  - "15 000 000 is additionally the SMALLEST round value that does not refuse np::parity's own Peyton n_lags=30 geometry (14 552 640) — 14 000 000 would have passed the walls and refused the request the ladder proves correctness on"
  - "C-08 is bounded, NOT recorded measured_at_structural_maximum: its structural maximum is 47.924 s on release, 24x the bar, so recording that wall as a closure would have recorded a breach as a bound"
  - "MAX_HOLIDAY_NAME_LEN is on BYTES, not chars, and the multi-byte UTF-8 case is the only test in the suite a chars().count() rewrite turns red"
  - "The holiday-name check is FIRST in the loop, which also bounds every OTHER refusal message in that loop — each of them formats h.name back to the caller"
  - "The post-loop holiday_dates_total refusal is KEPT: the in-loop one has seen only part of the list, so its count is a RUNNING total and its message says so"
  - "The at-the-bound positive control for C-08 lives in the #[ignore]d release harness because at the bound ONE request costs 45.619 s on a debug profile — measured, not assumed"

patterns-established:
  - "Pattern: a position-discriminating test — cross the ceiling mid-loop and give the LATER items input a different gate rejects, so the message names which check ran first"
  - "Pattern: a refusal that names the index and the LENGTH and never the value, so an error message cannot re-materialise the bytes the bound refuses"
  - "Pattern: a rejected bound candidate kept as a named harness mode, so its rejection is reproducible rather than only quoted"

requirements-completed: [SC1]

coverage:
  - id: D1
    description: "WR-03: the aggregate holiday-date ceiling is enforced INSIDE the holiday loop, proven by POSITION and not by presence, with the exact-total post-loop refusal retained"
    requirement: SC1
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::the_aggregate_dates_refusal_fires_inside_the_holiday_loop"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::holidays_carrying_exactly_the_total_bound_are_accepted"
        status: pass
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::refuses_holiday_dates_total_before_parsing_the_rest"
        status: pass
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::refuses_holiday_dates_total_over_bound"
        status: pass
    human_judgment: false
  - id: D2
    description: "C-07 closed: holidays[].name is bounded in BYTES at the door, first in the holiday loop, with the refusal naming the index and the length and never the name"
    requirement: SC1
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::a_holiday_name_over_the_length_bound_is_refused"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::a_holiday_name_at_the_length_bound_is_accepted"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::a_holiday_name_whose_char_count_fits_but_whose_byte_length_does_not_is_refused"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::the_oversized_name_refusal_names_the_length_and_not_the_name"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::an_empty_holiday_name_is_not_refused_by_the_length_bound"
        status: pass
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::refuses_holiday_name_over_length_bound"
        status: pass
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::accepts_holiday_name_at_length_bound"
        status: pass
    human_judgment: false
  - id: D3
    description: "C-08 closed: the unbudgeted NeuralProphet training path is MEASURED at its structural maximum (47.924 s release) and bounded at the door on a value the measurement chose"
    requirement: SC1
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::a_neuralprophet_request_over_the_train_cost_bound_is_refused"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::a_neuralprophet_request_under_the_train_cost_bound_is_accepted"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::the_np_parity_ladder_geometry_prices_under_the_train_cost_bound"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::the_np_train_cost_bound_is_exclusive_not_inclusive"
        status: pass
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::refuses_np_train_cost_over_bound"
        status: pass
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::accepts_np_train_cost_under_bound"
        status: pass
      - kind: other
        ref: "NP_WALL_MODE=structural_max cargo test --release -p aprender-forecast --lib np_train_wall -- --ignored --nocapture (outcome=refused, total_s=0.001, profile=release)"
        status: pass
    human_judgment: false
  - id: D4
    description: "The door prices a NeuralProphet request at exactly the work it then configures the sweep to spend, so the bound is not evadable by drift"
    requirement: SC1
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::the_door_prices_a_request_at_exactly_the_work_it_configures"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/np.rs#np::train debug_assert_eq!(n, n_training_samples(d, l))"
        status: pass
    human_judgment: false
  - id: D5
    description: "The class invariant's second half is now TRUE rather than pending: door_surface.cost_axes carries no pending marker and 06-14's declared-red test is green, earned"
    requirement: SC1
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/types.rs#types::tests::no_cost_axis_is_pending"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/types.rs#types::tests::every_cost_axis_names_a_real_bound"
        status: pass
      - kind: unit
        ref: "cargo test -p aprender-forecast --lib (NO --skip): 112 passed; 0 failed; 11 ignored"
        status: pass
    human_judgment: false
  - id: D6
    description: "Both new bounds are contract-owned and machine-mirrored, with the mirror proven by mutating the YAML value alone"
    requirement: SC1
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/types.rs#types::tests::cost_bounds_match_contract"
        status: pass
      - kind: other
        ref: "pv validate contracts/forecast-tool-boundary-v1.yaml (0 error(s), 0 warning(s)); make contract-audit-phase6 rc=0, 62 rows resolved, zero real BIND-/RESOLVE- findings"
        status: pass
    human_judgment: false
  - id: D7
    description: "WR-03's avoided work is MEASURED (0.012000 s -> 0.000212 s on release for the review's exact 1 000 x 1 000 payload), not estimated"
    verification:
      - kind: other
        ref: "cargo test --release -p aprender-forecast --lib wr03_aggregate_dates_wall -- --ignored --nocapture"
        status: pass
    human_judgment: false

duration: 3h 5m
completed: 2026-09-07
status: complete
---

# Phase 6 Plan 15: Close the two unbounded cost axes Summary

**`holidays[].name` bounded at 200 bytes and the unbudgeted NeuralProphet training path bounded at 15 000 000 units of work — the latter chosen by a measurement that first REJECTED 20 000 000 — closing 06-14's declared-red window by replacing both pending markers, so `cargo test -p aprender-forecast --lib` is green with no `--skip` for the first time since wave 12.**

## Performance

- **Duration:** ~3h 5m
- **Tasks:** 3 (two TDD, one auto)
- **Commits:** 8 (measured: `git rev-list --count 73ca9cb32..HEAD` at this SUMMARY's own commit) — 6 production commits plus the SUMMARY commit and this finalising amend
- **Files modified:** 6

## Accomplishments

- **The declared-red window is CLOSED, and earned.** 06-14 shipped `types::tests::no_cost_axis_is_pending` failing on purpose, naming C-07 and C-08. This plan turned it green by replacing both `unbounded_pending_06_15` markers with real bounds — not by deleting, weakening, `#[ignore]`-ing, `#[should_panic]`-ing or CI-filtering it.
- **C-07 closed by a bound**, because it never had a structural maximum to measure.
- **C-08 closed by a bound the MEASUREMENT chose**, after that same measurement rejected the first candidate.
- **WR-03's check is where its own comment always claimed it was**, proven by position and not by presence, with the saving measured rather than argued.
- **Every new bound has an observed RED side through the live streamable-HTTP server**, and every contract mirror has an observed YAML-only mutation.

## Task Commits

1. **Task 1 RED: the aggregate refusal fires after the loop** — `3d0235850` (test)
2. **Task 1 GREEN: refuse the aggregate holiday-date total INSIDE the loop** — `d5ea9758c` (feat)
3. **Task 2 RED: holidays[].name has no enforcement at all** — `b7cbe0e64` (test)
4. **Task 2 GREEN: bound holidays[].name in bytes at the door** — `170053b9f` (feat)
5. **Task 3: measure the unbudgeted NeuralProphet path and bound it** — `8a9d9232f` (feat)
6. **Task 1 step (5): measure the work WR-03's in-loop refusal avoids** — `8ab2faf86` (perf)

## TDD Gate Compliance

`workflow.tdd_mode` is `false` in `.planning/config.json`, but Tasks 1 and 2 carry `tdd="true"` and were executed through the full RED → GREEN cycle. Gate commits, matched anchored at the commit-scope position:

| Gate | Required | Present | Commits |
|------|----------|---------|---------|
| RED (`test(06-15):`) | yes | **2** | `3d0235850`, `b7cbe0e64` |
| GREEN (`feat(06-15):`) | yes | **3** | `d5ea9758c`, `170053b9f`, `8a9d9232f` |
| REFACTOR (`refactor(06-15):`) | no | 0 | none needed — neither GREEN left an obvious cleanup |

Both REDs were verified as INTENTIONAL, not merely nonzero-exit, via `gsd-tools check tdd-red-evidence`: verdict **`RED_EVIDENCE_OK`**, reason `target_test_failed`, in both cases. See "Deviations" for the translation step that was required to get there.

## WR-03 — the aggregate refusal now fires inside the loop

### The three line numbers (the positional claim a grep count cannot make)

In `crates/aprender-forecast/src/forecast.rs`, at `8ab2faf86`:

| what | line |
|---|---|
| the holiday loop opens (`for h in args.holidays…`) | **191** |
| **first** `max_holiday_name_len` (the C-07 refusal, first in the loop) | **210** |
| **first** `max_holiday_dates_total` (the in-loop aggregate refusal) | **245** |
| the retained post-loop `max_holiday_dates_total` refusal | 291 |
| **first** `max_holiday_design_cost` | **306** |
| the `make_design(` call | **371** |

245 < 306 < 371, and 191 < 210 < 245 < 291. `grep -c 'max_holiday_dates_total'` is **2**.

### The RED observation, before the move, verbatim

```
thread 'forecast::tests::the_aggregate_dates_refusal_fires_inside_the_holiday_loop'
panicked at crates/aprender-forecast/src/forecast.rs:785:17:
a DATE-SHAPE refusal means the loop kept parsing holidays after the door already had the
information to refuse — the aggregate check is still AFTER the loop;
got "bad date \"2020-1-01\": want YYYY-MM-DD"
```

and through the transport:

```
thread 'e2e::refuses_holiday_dates_total_before_parsing_the_rest' panicked at :474:9:
the refusal must name the fix ("max_holiday_dates_total"); got:
{"jsonrpc":"2.0","id":2,"error":{"code":-32603,
 "message":"Validation error: bad date \"2020-1-01\": want YYYY-MM-DD"}}
```

**Why that is a POSITION test and not a presence test.** The running sum crosses `MAX_HOLIDAY_DATES_TOTAL` at the ELEVENTH holiday, and holidays 12–15 each carry `"2020-1-01"` — nine bytes, which `parse_date`'s shape gate rejects (not its calendar-range gate, which is a different message). With the check after the loop those later holidays are parsed first and the caller receives the date-shape refusal. The message that comes back therefore names which check ran first.

### `refuses_holiday_dates_total_over_bound` — needle unchanged

Its eleven-holiday payload now crosses the ceiling at holiday eleven and is answered by the **in-loop** message rather than the post-loop one. Its needle `max_holiday_dates_total` is carried by both messages, deliberately, so **the needle was not changed and did not need to be**. It still passes.

### The saving, MEASURED (Task 1 step 5, taken because it was cheap)

`forecast::wr03_wall::wr03_aggregate_dates_wall`, `#[ignore]`d, release, the review's exact trigger — 1 000 holidays with zero windows (so `holiday_columns` reaches only 1 000 and never trips), 1 000 dates each, 1 000 000 dates in one request:

```
WR03 AGGREGATE DATES WALL: holidays=1000 dates_per_holiday=1000 dates_sent=1000000
    outcome=refused total_s=0.012000 profile=release   <- in-loop refusal neutralised
WR03 AGGREGATE DATES WALL: holidays=1000 dates_per_holiday=1000 dates_sent=1000000
    outcome=refused total_s=0.000212 profile=release   <- in-loop refusal present
```

~57x, i.e. **~11.8 ms and ~8 MB of `Vec<i64>`** no longer spent on a request already known to be refused. That **confirms** the review's own characterisation rather than inflating it: WR-03 is a hygiene defect roughly 1:1 with the ~11 MB payload, not a second CR-01. The BEFORE side is reproducible — delete the in-loop refusal and re-run.

## C-07 — `holidays[].name`, the field with no enforcement at all

### The committed maximum, and the arithmetic

The longest holiday name in **any** committed fixture, example or test is **9 bytes** — `superbowl`, from Prophet 1.4.0's own canonical `peyton_holidays` frame (`crates/aprender-forecast/tests/fixtures/peyton_holidays_prophet140.json`, 17 entries), whose other label is `playoff` (7 bytes). Derived by scanning every `*.json` fixture's `holidays[].holiday` field.

`MAX_HOLIDAY_NAME_LEN = 200` is **22x** that. The amplification it bounds, read from `prophet::columns` rather than taken from the plan's table: each design column gets **two** owned `String`s from one payload occurrence — the column `name` (`format!("{}_delim_{}{}", h.name, sign, off.abs())`) and the `component` (`h.name.clone()`) — and then `hcols.sort_by(|a, b| a.name.cmp(&b.name))` makes the name the key of an `O(C log C)` byte-wise comparison sort; `prophet::predict`'s component-name dedup compares it again.

| | bounded (200 B) | unbounded (1 MB name) |
|---|---|---|
| owned `String` at `MAX_HOLIDAY_COLUMNS` = 1 000 | 1 000 × 2 × 200 B = **~400 KB** | 1 000 × 2 × 1 MB = **~2 GB** |
| sort comparisons | ~10 000 × ≤ 200 B | ~10 000 × 1 MB |

The bound was derived from **the amplification ratio and the committed maximum**, not from a wall — stated explicitly because this axis has no wall to take: there is no structural maximum (below).

### Why a bound and never a measurement

`no_structural_maximum: true` was 06-14's recorded reading and it held: `crates/aprender-mcp-forecast/src/lib.rs`'s router construction (`http_app` / `pooled_app`) applies no `DefaultBodyLimit`, no `max_body` and no content-length layer, and the stdio transport has no framing cap. Any `measured_seconds` written for this axis would have measured an arbitrarily **chosen** name length.

### Bytes, not chars — and what that case proves

`a_holiday_name_whose_char_count_fits_but_whose_byte_length_does_not_is_refused` sends 101 `é` — **101 characters (under the bound)** and **202 bytes (over it)**. It is the only test in the suite that a rewrite of `h.name.len()` to `h.name.chars().count()` turns red, and the reason is in a comment beside the check so a later reader does not "fix" it. `String::len` is bytes, and bytes are what the two clones per column and the byte-wise comparison actually cost.

### The RED observation

**Before the door check existed at all**, through the live server, the amplification appeared in the reply itself:

```
thread 'e2e::refuses_holiday_name_over_length_bound' panicked at :473:9:
must be REFUSED, never defaulted; got: {"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text",
"text":"{\"model\":\"prophet\", ... \"components\":{\"weekly\":[...],
\"nnnnnnnn…(201 n's)…nnn\":[0.0,0.0,0.0,0.0,0.0,0.0,0.0], ...}"}],"isError":false}}
```

**And under the raised-constant protocol**, with the payload PINNED at 201 bytes so raising the bound could not raise the request with it:

| step | result |
|---|---|
| `fit_max_holiday_name_len` and `MAX_HOLIDAY_NAME_LEN` raised to 100 000, payload pinned at 201 B | server **ACCEPTED**, full-shape forecast returned (`test result: FAILED. 0 passed; 1 failed`) |
| both restored to 200, same pinned payload | **refused** (`test result: ok. 1 passed; 0 failed`) |

> The first attempt at this observation was **vacuous** and was caught: the e2e test derives its payload from `MAX_HOLIDAY_NAME_LEN + 1`, so raising the constant raised the request with it and the test passed under the raised bound. Pinning the payload is what made the observation mean anything.

### The mirror mutation, YAML alone, no Rust touched

```
  fit_max_holiday_name_len: 201
→ thread 'types::tests::cost_bounds_match_contract' panicked at types.rs:398:13:
  assertion `left == right` failed: types::MAX_HOLIDAY_NAME_LEN must equal
  constants.fit_max_holiday_name_len in forecast-tool-boundary-v1
    left: 200
→ restored → test result: ok. 1 passed; 0 failed
```

### The refusal does not echo the name (T-06-38)

`the_oversized_name_refusal_names_the_length_and_not_the_name` sends a name built from a distinctive token and asserts the message contains `index 0`, the observed byte count and `max_holiday_name_len`, and does **not** contain the token. The assertion is non-vacuous by construction: every *other* refusal in the same loop *does* interpolate `h.name`, which is exactly why the length check is **first** in the loop — those messages can now only ever carry a bounded name.

## C-08 — the NeuralProphet training path, which had no budget of any kind

### The seven release walls

`np::wall::np_train_wall`, `#[ignore]`d, `cargo test --release -p aprender-forecast --lib np_train_wall -- --ignored --nocapture`:

```
NP TRAIN WALL: mode=lag_free               points=20000 span_days=20000 n_lags=0   horizon=3650 train_cost=3000000    outcome=accepted total_s=0.508  profile=release
NP TRAIN WALL: mode=mid_range              points=2000  span_days=2000  n_lags=30  horizon=365  train_cost=10992600   outcome=accepted total_s=1.106  profile=release
NP TRAIN WALL: mode=at_bound_short_history points=2000  span_days=2000  n_lags=41  horizon=365  train_cost=14810040   outcome=accepted total_s=1.402  profile=release
NP TRAIN WALL: mode=at_bound_mid_history   points=10000 span_days=10000 n_lags=11  horizon=3650 train_cost=14384160   outcome=accepted total_s=1.541  profile=release
NP TRAIN WALL: mode=at_bound_long_history  points=20000 span_days=20000 n_lags=6   horizon=3650 train_cost=13995800   outcome=accepted total_s=1.642  profile=release
NP TRAIN WALL: mode=structural_max         points=20000 span_days=20000 n_lags=365 horizon=3650 train_cost=718641000  outcome=accepted total_s=47.924 profile=release   <- BEFORE the bound
NP TRAIN WALL: mode=structural_max         points=20000 span_days=20000 n_lags=365 horizon=3650 train_cost=718641000  outcome=refused  total_s=0.001  profile=release   <- AFTER the bound
```

plus, during derivation and now reproducible as `NP_WALL_MODE=rejected_candidate_20m`:

```
NP TRAIN WALL: mode=at_bound_long_history  points=20000 span_days=20000 n_lags=9   horizon=3650 train_cost=19991000   outcome=accepted total_s=2.089  profile=release
```

### WHICH BRANCH OF STEP (3) WAS TAKEN, AND THE NUMBER THAT DECIDED IT

**Branch (3a): a bound was added.** The number that decided it is **47.924 s** — the worst LEGAL request (20 000 contiguous daily points so the span is also at `MAX_SPAN_DAYS` and `n_train_grid` is at its ceiling, `n_lags` at its ceiling of 365, `horizon` at `MAX_HORIZON`, `freq: "D"`, the only frequency this arm accepts), on a release build, against SC1's 2 s bar. That is **24x over**. `measured_at_structural_maximum` was therefore never an available disposition: recording that wall as a closure would have recorded a breach as a bound.

### The VALUE, and the candidate the measurement rejected

**A first candidate of 20 000 000 was discarded by its own wall**: the long-history composition at 19 991 000 measured **2.089 s**, over the bar. `MAX_NP_TRAIN_COST = 15 000 000` is the value, and three compositions differing in **every** factor the proxy multiplies all clear the bar at it (1.642 / 1.541 / 1.402 s, worst 1.642 s).

It is additionally **the smallest round value that does not refuse the parity ladder's own geometry.** `np::parity` proves the model correct on Peyton Manning (2 905 rows over a 2 964-day span) at `n_lags` 0 and 30; through the door those price at **697 200** and **14 552 640** — the second is 97% of the bound. 14 000 000, which the walls would equally have allowed, would refuse the very request the ladder validates. `the_np_parity_ladder_geometry_prices_under_the_train_cost_bound` pins that and turns red the moment the bound is lowered past it.

**The lag-free arm can never reach this bound at all**: its cost is `3 × auto_epochs(n) × n × 1`, whose maximum over the legal range is 3 000 000, measured at 0.508 s. C-08 is entirely about the lagged arm.

### One implementation of the price

`np::request_train_cost` is built from `np::door_lr_sweep`, `np::door_epochs` and `np::n_training_samples` — and the door now builds its `TrainConfig` out of those same three functions instead of literals, so the price **is** the work. `np::train` additionally carries `debug_assert_eq!(n, n_training_samples(d, l))`. `grep -n 'fn train_cost' crates/aprender-forecast/src/np.rs` prints exactly one line (**461**), and `train_cost` is referenced from both `np.rs` (in `request_train_cost` and in `TrainLog`) and `forecast.rs` (via `request_train_cost`, and directly in `the_np_parity_ladder_geometry_prices_under_the_train_cost_bound`).

### The RED observation, through the live server

| step | result |
|---|---|
| `fit_max_np_train_cost` and `MAX_NP_TRAIN_COST` raised to 100 000 000 000 | server **ACCEPTED**, full-shape neuralprophet forecast, **276.93 s for ONE request** on a debug build |
| both restored to 15 000 000 | **refused in 0.01 s** |

### The mirror mutation, YAML alone

```
  fit_max_np_train_cost: 15000001
→ assertion `left == right` failed: types::MAX_NP_TRAIN_COST must equal
  constants.fit_max_np_train_cost in forecast-tool-boundary-v1
    left: 15000000
   right: 15000001
→ restored → test result: ok. 1 passed; 0 failed
```

### The post-condition the whole disposition rests on

```
NP TRAIN WALL: mode=structural_max ... train_cost=718641000 outcome=refused total_s=0.001 profile=release
NP TRAIN WALL REFUSAL: this neuralprophet request buys 718641000 units of training work
  (learning-rate sweep 2 x epochs 50 x samples 19635 x (n_lags + 1) 366), which exceeds
  max_np_train_cost 15000000; reduce n_lags, shorten the history, or narrow the series span
```

The worst legal request is **refused at the door**, not merely fast.

### Why the at-the-bound positive control is not in the always-run suite

Because the cost was measured rather than guessed: at the bound **one request costs 45.619 s on a debug profile** (`NP_WALL_MODE=at_bound_short_history`, `profile=debug`), and this crate's suite runs in debug. Paying that per `cargo test` would be a ~60x regression on a 0.76 s suite. The always-run controls are therefore:

- `the_np_parity_ladder_geometry_prices_under_the_train_cost_bound` — pure arithmetic at **97% of the bound**, the exact over-refusal risk the plan names;
- `a_neuralprophet_request_under_the_train_cost_bound_is_accepted` — a real lagged request through the door returning the full shape;
- `the_np_train_cost_bound_is_exclusive_not_inclusive` — pins the door's own named comparison `np_train_cost_is_over` at the boundary (`>` and not `>=`), which no runnable near-miss reaches;
- `e2e::accepts_np_train_cost_under_bound` — the same through the transport.

The at-the-bound near misses (93–99% of the bound) are the three release walls above.

## The class invariant's second half: RED → GREEN, earned

**06-14's output** (declared red, from its SUMMARY):

```
thread 'types::tests::no_cost_axis_is_pending' panicked at crates/aprender-forecast/src/types.rs:603:9:
2 cost axis/axes are still UNBOUNDED and are held open by this assertion: C-07 (marker
unbounded_pending_06_15, owed by plan 06-15); C-08 (marker unbounded_pending_06_15, owed by plan
06-15). ... Land 06-15.

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 108 filtered out
```

**After Task 2 alone** (C-07's marker replaced, C-08's still there — the intermediate state, quoted because it shows the test naming exactly what remained):

```
1 cost axis/axes are still UNBOUNDED and are held open by this assertion: C-08 (marker
unbounded_pending_06_15, owed by plan 06-15).
```

**This plan's output:**

```
$ cargo test -p aprender-forecast --lib -- types::tests::every_cost_axis_names_a_real_bound \
      types::tests::no_cost_axis_is_pending --nocapture
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 120 filtered out
```

**And the crate's whole `--lib` suite, with NO `--skip`:**

```
$ cargo test -p aprender-forecast --lib
test result: ok. 112 passed; 0 failed; 11 ignored; 0 measured; 0 filtered out
```

06-14 needed `-- --skip no_cost_axis_is_pending` to reach 99 passed. This plan needs nothing. The window is closed by replacement, exactly as `no_cost_axis_is_pending`'s own doc comment demanded; the assertion body is byte-identical and only its doc comment moved from "why it is red" to "what closed it".

## Verification

| gate | result |
|---|---|
| `cargo test -p aprender-forecast --lib` (no `--skip`) | **112 passed; 0 failed; 11 ignored** |
| `cargo test -p aprender-mcp-forecast --lib` | **40 passed; 0 failed** |
| `cargo test -p aprender-mcp-forecast --lib e2e::` | **37 passed; 0 failed** |
| `cargo test -p aprender-mcp-forecast --test e2e_stdio` | **1 passed; 0 failed** (dark in CI; run deliberately) |
| `np::parity` | **9 passed; 0 failed** — unchanged |
| `prophet::parity` | **32 passed; 0 failed** — unchanged (06-14 recorded 32/0) |
| `pv validate contracts/forecast-tool-boundary-v1.yaml` | **0 error(s), 0 warning(s)** |
| `pv status` | v1.4.0 · Equations **22** · Proof obligations **25** · Falsification tests **25** |
| `pv equations` | lists `holiday_name_cost_bounded` and `np_train_cost_bounded` |
| `make contract-audit-phase6` | **rc=0**, 4 `Total equations:` summaries, **62** rows resolved, **0** real `BIND-`/`RESOLVE-` findings |
| `cargo clippy -p aprender-forecast -p aprender-mcp-forecast --all-targets --no-deps -- -D warnings` | **rc=0** |
| `cargo fmt --all -- --check` | **rc=0** |
| release NP walls with `profile=release` | **7** (≥ 3 required) |

### Contract version and `pv diff`

`pv diff` takes two filesystem paths, never a git revision, so the old side was materialised first: `git show 73ca9cb32:contracts/forecast-tool-boundary-v1.yaml > /tmp/…`.

```
Contract diff: v1.3.0 → v1.3.0
Suggested bump: minor
  equations:            + holiday_name_cost_bounded  + np_train_cost_bounded
  proof_obligations:    + 5
  falsification_tests:  + FALSIFY-BOUNDARY-021 … 025
```

**Version bumped 1.3.0 → 1.4.0** on that suggestion. Both bounds narrow the advertised tool boundary (a long holiday label, and an extreme-but-previously-legal NeuralProphet geometry, are now refused), which is client-visible and reversible — a minor bump, not a patch.

### No parity fixture or tolerance was touched

`git diff 73ca9cb32 -- crates/aprender-forecast/tests/fixtures/ crates/aprender-mcp-forecast/fixtures/ contracts/{prophet,neuralprophet,chronos-bolt}-parity-v1.yaml` is **empty**.

The plan asked for `git diff --stat ce3e5a8ea..HEAD` on the same paths. That base predates 06-13, and the diff there is **non-empty** — `contracts/prophet-parity-v1.yaml` gains 29 lines. Inspected rather than assumed: it is 06-13's **new** `poisson_sampler_domain` equation with its own `float_tolerance: 0.01`, plus a version bump 1.0.0 → 1.1.0. **No pre-existing numeric tolerance was changed** and no fixture moved, at either base.

## Decisions Made

Recorded in the frontmatter `key-decisions`. The two that decided the shape of this plan:

1. **The measurement rejected the first candidate bound.** 20 000 000 measured 2.089 s. Keeping it would have shipped a "bound" that permits a request over the bar it exists to enforce.
2. **The at-the-bound near-miss for C-08 is release-only, for a measured reason (45.619 s debug), and the boundary is pinned on the door's own comparison instead.** The alternative — a near-miss an order of magnitude below the bound, presented as if it straddled it — is the kind of control that passes for a bound that does not exist.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Four of the plan's `<verify>` commands pass two test filters positionally, which `cargo test` rejects**

- **Found during:** Task 2 (first hit), then Tasks 2 and 3
- **Issue:** `cargo test -p aprender-forecast --lib types::tests forecast::tests` exits 1 with `error: unexpected argument 'forecast::tests' found`. `cargo test` takes **one** `[TESTNAME]` filter; multiple filters must follow `--`. Affects T2a, T2b and T3b as written. (This is the same class the prior wave flagged: 06-14's own plan shipped four defective verify commands.)
- **Fix:** Ran the corrected form, `cargo test -p <crate> --lib -- <filter1> <filter2>`, everywhere. No test or source change was needed.
- **Verification:** The corrected form runs and discriminates — it produced the Task 2 RED (`29 passed; 3 failed`) and the Task 3 GREEN (`2 passed`).
- **Committed in:** no code change; recorded here.

**2. [Rule 3 - Blocking] The `rtk` Bash hook rewrites libtest output even through a redirect, so `--nocapture` `println!` lines never reach the log the verify gates grep**

- **Found during:** Task 3 (the `lag_free` wall printed nothing)
- **Issue:** The plan's wall gates grep `^NP TRAIN WALL: ` out of a redirected log. Under the hook, the redirected file contains `cargo test: 1 passed, 116 filtered out` instead of libtest's own output, so the gate is unsatisfiable — the same defect 06-14 recorded, reproduced here on the `println!` path rather than the summary path.
- **Fix:** Ran every wall measurement and every log-grepping gate through `rtk proxy`, per CLAUDE.md's documented escape hatch.
- **Verification:** `rtk proxy env NP_WALL_MODE=lag_free cargo test --release …` produced the `NP TRAIN WALL:` line; the same command without `rtk proxy` did not. Confirmed on both.
- **Committed in:** no code change; recorded here.

**3. [Rule 3 - Blocking] The plan's `make contract-audit-phase6` gate requires `grep -c 'BIND-' == 0`, which is unsatisfiable**

- **Found during:** Task 3
- **Issue:** The target's own **success** line is `Phase 6 binding audit: 4 contract(s) audited, zero BIND- findings`, which contains the literal `BIND-`. `grep -c 'BIND-'` on a passing run returns **1**. (06-14 recorded this; the plan reintroduced the naive form.)
- **Fix:** Used the refined pattern `grep -E '(^|[[:space:]])(BIND|RESOLVE)-[0-9]' | grep -v 'zero BIND- findings'`, and **proved it with a must-match / must-not-match case table** (CLAUDE.md rule 7) rather than by reading it: success line alone → 0; a synthetic `BIND-001 …` finding → 1; a synthetic `RESOLVE-002 …` finding → 1; the real log → 0, while naive `grep -c 'BIND-'` → 1.
- **Verification:** case table above, plus `make contract-audit-phase6` rc=0.
- **Committed in:** no code change; recorded here.

**4. [Rule 1 - Bug] The first C-07 RED-side observation was VACUOUS and was caught before it was believed**

- **Found during:** Task 2
- **Issue:** `e2e::refuses_holiday_name_over_length_bound` derives its payload from `MAX_HOLIDAY_NAME_LEN + 1`. Raising the constant to observe the accepted side raised the **request** with it, so the test passed under the raised bound and "proved" nothing. Reported as `test result: ok` — a green that meant the opposite of what it looked like (CLAUDE.md rule 1: when a result looks good, check how it was measured).
- **Fix:** Pinned the payload at a literal 201 bytes for the duration of the observation, re-ran under the raised bound (**ACCEPTED**), restored the constant (**refused**), then restored the payload to its constant-derived form.
- **Verification:** both runs recorded above; the working tree was confirmed clean of all three temporary mutations afterwards.
- **Committed in:** no code change; recorded here.

**5. [Rule 2 - Missing Critical] `gsd-tools check tdd-red-evidence` cannot classify a Rust RED without translation**

- **Found during:** Task 1
- **Issue:** The checker requires `{command, targetTest, exitCode, output}` and parses `output` as node:test TAP (`# tests`/`# pass`/`# fail`, `not ok N - name`). A raw libtest log yields `INVALID_RED` with reason `invalid_record` — so the #3770 intentional-RED gate could not be satisfied at all. (06-14 recorded this; it is a tooling gap, not a project defect.)
- **Fix:** Wrote a **faithful mechanical translator** (`/tmp/p06-15-libtest2tap.py`) that derives the failing test names from libtest's own `thread '<name>' … panicked at` lines and the counts from libtest's own `test result:` summary line. Nothing is invented; a run with no `test result:` line is a hard error rather than an assumed pass.
- **Verification:** verdict **`RED_EVIDENCE_OK`**, reason `target_test_failed`, for both Task 1 (`tests 1, pass 0, fail 1`) and Task 2 (`tests 32, pass 29, fail 3`).
- **Committed in:** no code change (the translator is a scratch tool, not a repo artifact); recorded here.

**6. [Rule 2 - Missing Critical] The plan's three new proof obligations exceeded its one falsification test, which `pv validate` rejects**

- **Found during:** Task 2
- **Issue:** `PROVABILITY-001: falsification_tests (21) < proof_obligations (23)`. The plan specified one `holiday_name_cost_bounded` equation with "its `proof_obligations` entry", but the deliverables genuinely carry three distinct claims (the bound; the refusal not echoing the name; the aggregate check's position).
- **Fix:** Split into **FALSIFY-BOUNDARY-021 / 022 / 023**, one per obligation, and added the test 022 promises — `the_oversized_name_refusal_names_the_length_and_not_the_name`. Task 3 added obligations and FALSIFY-024/025 in matched pairs from the start.
- **Verification:** `pv validate` → `0 error(s), 0 warning(s)`; `pv status` → 25 obligations, 25 falsification tests.
- **Committed in:** `170053b9f`, `8a9d9232f`

**7. [Rule 2 - Missing Critical] The declared-red narration in `types.rs` and the contract went stale the moment the window closed**

- **Found during:** Task 3
- **Issue:** `no_cost_axis_is_pending`'s doc comment opened "**THIS TEST IS RED ON PURPOSE, FROM THE CLOSE OF PLAN 06-14 UNTIL 06-15 LANDS**", and three places in `forecast-tool-boundary-v1.yaml` asserted the same. All were false after Task 3, and a doc that confidently states a falsehood about a guard is worse than no doc.
- **Fix:** Rewrote the narration to record what the window WAS, quote 06-14's failing output, name what closed it, and restate that the forbidden repair is still forbidden and a new pending axis is *supposed* to turn it red again. **The assertion body is byte-identical** — only the comment changed.
- **Verification:** `no_cost_axis_is_pending` still passes and still fails on a pending marker (proven earlier in Task 2, where it named C-08 alone).
- **Committed in:** `8a9d9232f`

---

**Total deviations:** 7 auto-fixed (4 blocking, 3 missing-critical, of which one was a bug in this plan's own evidence).
**Impact on plan:** Four are defects in the plan's own verify commands or in shared tooling and changed no behaviour. Two strengthened the contract and the docs. One (#4) prevented a false RED-side claim from being recorded as evidence. No scope creep; the scope fence (`aprender-compute`, `.github/workflows/*.yml`, `aprender-serve`) was not touched.

## Prohibitions — all seven honoured

| # | Prohibition | Status |
|---|---|---|
| 1 | Never remove the post-loop `holiday_dates_total` refusal | **Kept**; `grep -c` is 2, and its message still reports the exact total |
| 2 | Never add an NP cost bound without measuring first | **7 release walls taken first**; the bound came from them, and they rejected the first candidate |
| 3 | Never let the door and `np::train` compute the cost differently | **One implementation**: the door builds its `TrainConfig` from the same `door_lr_sweep` / `door_epochs` / `n_training_samples` the price uses; `train` `debug_assert`s its sample count |
| 4 | Never touch a parity fixture, tolerance or `*-parity-v1.yaml` number | **Untouched** (empty diff from the plan base); `np::parity` 9/0 and `prophet::parity` 32/0 unchanged |
| 5 | Never close 06-14's red window by editing 06-14's test | **Closed by replacing both markers**; the assertion body is byte-identical, no `#[ignore]`, no `should_panic`, no CI filter |
| 6 | Do not touch `aprender-compute`, `.github/workflows/*.yml`, `aprender-serve` | **Untouched** — the six modified files are exactly the plan's `files_modified` |

## Review Dispositions Ledger

| ID | Severity | Disposition | Status after 06-15 |
|---|---|---|---|
| CR-01 | Critical | INCORPORATED | Closed by 06-14; untouched here |
| WR-01 | Warning | INCORPORATED, SPLIT | 06-14 half closed; 06-17 owes the sweep point |
| WR-02 | Warning | INCORPORATED | Owed by 06-17 |
| **WR-03** | **Warning** | **INCORPORATED** | **CLOSED by 06-15 T1** — in-loop refusal, position proven, saving measured |
| WR-04 | Warning | INCORPORATED | Owed by 06-16 |
| IN-01 | Info | INCORPORATED | Owed by 06-16 |
| IN-02 | Info | INCORPORATED, NARROWED | Owed by 06-17 |
| IN-03 | Info | INCORPORATED | Closed by 06-14 |
| IN-04 | Info | INCORPORATED IN PART | Owed by 06-17 |

Cost axes: **C-07 CLOSED** (T2), **C-08 CLOSED** (T3). No axis carries a pending marker.

## Known Stubs

None. No hardcoded empty value, placeholder string, `TODO`, `FIXME` or unwired component was introduced.

Two `#[ignore]`d harnesses were added — `np::wall::np_train_wall` and `forecast::wr03_wall::wr03_aggregate_dates_wall`. These are **measurement** harnesses, not skipped assertions, and follow the pattern 06-14 established with `prophet::sampler::logistic_band_wall`: a wall-clock number belongs in a deliberate release run, never in the correctness suite (REVIEW-06-04). Every correctness claim in this plan is carried by an always-run test.

## Threat Flags

None. The three DoS threats this plan owns (T-06-35 `holidays[].name`, T-06-36 `np::train`, T-06-37 the late aggregate refusal) and the one information-disclosure threat (T-06-38, the refusal message) are all **mitigated** as their register entries specify. No new network endpoint, auth path, file access pattern or trust-boundary schema change was introduced; no dependency entered the graph, so T-06-SC's package-legitimacy gate was never engaged.

## Issues Encountered

- **The wall for the structural maximum is 47.924 s**, so the C-08 measurement was not free in wall-clock terms. Run once, before the bound; the post-bound re-run is 0.001 s because the request is now refused.
- **The RED-side observation for C-08 took 276.93 s** on a debug build, because the raised bound let the request actually train. That duration is itself the evidence.

## Next Phase Readiness

- **The branch can now be pushed.** `cargo test -p aprender-forecast --lib` is green with no `--skip`, so `make tier3` (`cargo test --all`) and CI's `workspace-test` are no longer red on this crate. 06-14's push-sequencing constraint (land waves 12 and 13 in ONE push) is satisfied by this plan landing.
- **06-16** owns WR-04, IN-01 and the SC1 sweep gate. Its NP arm sweep case is what `FALSIFY-BOUNDARY-024`'s `test:` field names, and `np::wall::np_train_wall`'s `NP TRAIN WALL:` token is available to it (the permanent `SC1 WALL:` token remains 06-16's to create).
- **06-17** owns WR-01's tight sub-threshold point, WR-02, IN-02 and IN-04.
- No blocker. `contracts/forecast-tool-boundary-v1.yaml` is at **1.4.0** with 22 equations, 25 obligations and 25 falsification tests; the Phase 6 binding block is at **62** rows, header updated.

---
*Phase: 06-native-time-series-forecasting-stack*
*Completed: 2026-09-07*

## Self-Check: PASSED

- All 7 files named in `key-files.modified` plus this SUMMARY exist on disk.
- All 6 production commit hashes (`3d0235850`, `d5ea9758c`, `b7cbe0e64`, `170053b9f`, `8a9d9232f`, `8ab2faf86`) and the SUMMARY commit (`23fe8ab0f`) resolve in `git log --oneline --all`; the eighth is this finalising commit, which cannot cite its own hash.
- `MAX_HOLIDAY_NAME_LEN` and `MAX_NP_TRAIN_COST` are declared in `types.rs`; `fit_max_holiday_name_len` and `fit_max_np_train_cost` are present in the contract; `grep -c 'fn train_cost'` in `np.rs` is exactly **1**.
- `grep -n 'bound: unbounded_pending' contracts/forecast-tool-boundary-v1.yaml` returns **nothing** — no `cost_axes` entry carries a pending marker. The four remaining `unbounded_pending` hits are prose describing the marker vocabulary and the closed window, in the `door_surface:` header comment, `door_surface_is_complete`'s formula and invariant, and `FALSIFY-BOUNDARY-019`.
- Every plan-level `<verification>` gate was re-run at `8ab2faf86` and is recorded in the Verification table above.
