---
phase: 06-native-time-series-forecasting-stack
plan: 12
subsystem: api
tags: [rust, mcp, streamable-http, prophet, forecasting, dos, resource-exhaustion, provable-contracts, just, benchmarking]

requires:
  - phase: 06-native-time-series-forecasting-stack
    provides: "the two product bounds and the `forecast-holiday-bench` measurement recipe (06-11); the e2e refusal-case shape and FALSIFY-BOUNDARY-017 (06-10); the live streamable-HTTP e2e suite and `refused` helper (06-06)"
provides:
  - "`e2e::refuses_holiday_design_cost_over_bound` / `e2e::accepts_holiday_design_cost_just_under_bound` — the design-cost bound asserted THROUGH a live streamable-HTTP server, two-sided and ONE history point apart at the boundary"
  - "`e2e::refuses_holiday_dates_total_over_bound` — the third factor, previously never reached through the transport"
  - "`equations.holiday_design_cost_bounded` + a `type: bound` proof obligation + `FALSIFY-BOUNDARY-018` in forecast-tool-boundary-v1.yaml, version bumped 1.1.0 -> 1.2.0 on `pv diff`'s own suggestion"
  - "A `binding.yaml` row that RESOLVES to `aprender_forecast::forecast::forecast` (Phase 6 resolved rows 56 -> 57)"
  - "`just forecast-holiday-bench` as a GATE: token-parsed `total_s`, a `profile=release` check, a 2.0 s SC1 bar, and three failure paths OBSERVED red"
affects: [06-13, forecast-tool-boundary, gsd-verify-work]

actuals:
  tokens: 4528
  tasks: 3
  commits: 3
plan_head_before: 2c774ee56ed1370d8f8653bf411fdd046a01ca96

tech-stack:
  added: []
  patterns:
    - "Two-sided boundary control at ONE STEP: the refusal case and its accept-side twin differ by a single history point and straddle the bound, so the pair proves the bound is where the contract says it is rather than an order of magnitude away"
    - "Assert a bound at EVERY layer that can bypass it: constants (contract mirror test), library door (two-sided), transport (the same two sides through a live server)"
    - "A recipe-level bar states in its own header what it CLAIMS and what it does not, and cites the open human decision it deliberately does not close"

key-files:
  created: []
  modified:
    - crates/aprender-mcp-forecast/src/lib.rs
    - contracts/forecast-tool-boundary-v1.yaml
    - contracts/aprender/binding.yaml
    - justfile

key-decisions:
  - "The OVER/UNDER pair is at the BOUNDARY, not merely across it: 62+7 rows x 731 columns = 50 439 cells vs 61+7 x 731 = 49 708, against MAX_HOLIDAY_DESIGN_COST 50 000. One history point separates refusal from acceptance"
  - "The OVER geometry uses the widest LEGAL holiday window (±365 = 731 columns) with a single date, so the whole excess comes from the product and every individual bound is satisfied with room to spare"
  - "The third case (`max_holiday_dates_total`) WAS needed — no e2e case reached that factor before. Eleven holidays of 1 000 dates each: each individually legal, the aggregate 11 000 > 10 000"
  - "The RED observation needed TWO temporary perturbations, both recorded: raising the constant alone was caught by the test's own geometry pre-assert, which had to be removed as well to reach the door and observe the server accept and fit"
  - "The recipe's DEFAULTS were NOT changed — 06-11's deviation 2 already pointed them at the at-the-bound geometry (800+200) x 50 = 50 000 cells. Re-deriving them would have been churn"
  - "The 2 s bar is scoped in the recipe header to the worst ACCEPTED shape, explicitly NOT a general SC1 guarantee for holiday-carrying requests. WINDOWS.md entry 7 / 06-11 coverage D7 stays OPEN and this plan does not pick one of its options"
  - "`pv diff` suggested `minor`; applied as 1.1.0 -> 1.2.0. 06-11 could get no bump because `constants:` is a dropped top-level key — the equation is exactly what made the narrowing contract-visible, as 06-11 predicted"

patterns-established:
  - "Induce the guard's OWN failure path, not just the surrounding command's: the recipe was observed failing on an over-bound geometry (the cargo path), on a temporarily lowered bar (the comparison path) and on a temporarily broken `profile=` pattern (the provenance path)"

requirements-completed: [SC1]

coverage:
  - id: D1
    description: "An in-bounds-except-for-the-product holiday request is REFUSED through a live streamable-HTTP server with a validation-class error naming `max_holiday_design_cost`"
    requirement: "SC1"
    verification:
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::refuses_holiday_design_cost_over_bound"
        status: pass
    human_judgment: false
  - id: D2
    description: "A near-miss request ONE history point under the bound is ACCEPTED through the same server and returns the full promised response shape (ds/yhat/yhat_lower/yhat_upper/trend at horizon length, banded, plus components and diagnostics)"
    requirement: "SC1"
    verification:
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::accepts_holiday_design_cost_just_under_bound"
        status: pass
    human_judgment: false
  - id: D3
    description: "The aggregate holiday-dates bound is reached through the transport for the first time: eleven individually-legal holidays whose date total exceeds `max_holiday_dates_total`"
    requirement: "SC1"
    verification:
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::refuses_holiday_dates_total_over_bound"
        status: pass
    human_judgment: false
  - id: D4
    description: "The bar was shown to turn RED, not merely added: with `MAX_HOLIDAY_DESIGN_COST` raised to 60 000 the server ACCEPTED and FITTED the over-bound request and returned a full forecast, and `types::tests::cost_bounds_match_contract` went red naming the key; both perturbations restored byte-identically"
    requirement: "SC1"
    verification:
      - kind: other
        ref: "cargo test -p aprender-mcp-forecast --lib e2e::refuses_holiday_design_cost_over_bound under a raised constant — RC=101, panic text and reply body quoted verbatim below"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/types.rs#types::tests::cost_bounds_match_contract (red under the raise, green after restore, git diff empty)"
        status: pass
    human_judgment: false
  - id: D5
    description: "`forecast-tool-boundary-v1.yaml` OWNS the rule: `holiday_design_cost_bounded` with a `type: bound` proof obligation and `FALSIFY-BOUNDARY-018` naming all five falsifying assertions, so the HARD-ceiling claim in `fit_server_bounds`' third invariant is backed by a constant, a check and a test"
    requirement: "SC1"
    verification:
      - kind: other
        ref: "pv validate contracts/forecast-tool-boundary-v1.yaml — 0 error(s), 0 warning(s), Contract is valid"
        status: pass
      - kind: other
        ref: "pv status contracts/forecast-tool-boundary-v1.yaml — Equations 18 / Proof obligations 18 / Falsification tests 18 (was 17/17/17)"
        status: pass
    human_judgment: false
  - id: D6
    description: "The new equation binds to a REAL definition site, not merely to a registry entry: `make contract-audit-phase6` reports rc=0, zero BIND-, zero RESOLVE- over four contracts"
    requirement: "SC1"
    verification:
      - kind: other
        ref: "make contract-audit-phase6 — rc=0, BIND- 0, RESOLVE- 0, four `Total equations:` summaries, `resolved 57 Phase 6 binding rows` (was 56)"
        status: pass
    human_judgment: false
  - id: D7
    description: "`just forecast-holiday-bench` is a GATE, not a printout: it token-parses `total_s`, requires `profile=release`, asserts < 2.0 s on the at-the-bound geometry, captures rc from a redirect, and fails hard when no measurement line was produced"
    requirement: "SC1"
    verification:
      - kind: other
        ref: "just forecast-holiday-bench — rc=0, one HOLIDAY DESIGN WALL line, HOLIDAY DESIGN OK: 1.712 s < 2.0 s (SC1), profile=release"
        status: pass
      - kind: other
        ref: "three induced RED runs: over-bound geometry (rc 101), a 0.5 s bar (rc 1, named the measured 1.695 s), a broken profile= pattern (rc 1) — all restored byte-identically"
        status: pass
    human_judgment: false
  - id: D8
    description: "SC1's 2 s bar is STILL not a general guarantee for holiday-carrying requests — this plan scopes the claim to the at-the-bound default rather than closing 06-11's D7"
    verification: []
    human_judgment: true
    rationale: "Carried forward unchanged from 06-11 coverage D7 / WINDOWS.md entry 7. A human must still decide between tightening MAX_HOLIDAY_DESIGN_COST (which refuses ordinary multi-holiday usage), lowering FIT_BUDGET_SECS, or scoping SC1's bar to the no-holiday shape it was measured on. This plan deliberately picked none of the three; it recorded the scope in the recipe header instead."

duration: 18 min
completed: 2026-09-07
status: complete
---

# Phase 06 Plan 12: Prove the design-cost bar through the server Summary

**The holiday design-cost bound is now asserted at every layer that could bypass it — a two-sided e2e pair one history point apart at the boundary (50 439 cells refused, 49 708 accepted with a full response shape), a contract equation with its obligation and `FALSIFY-BOUNDARY-018` naming all five falsifying assertions, a binding row that RESOLVES, and a `just` recipe that fails on a missed 2 s bar — with the refusal OBSERVED accepting-and-fitting under a raised constant and the recipe OBSERVED failing on three separate paths.**

## Performance

- **Duration:** 18 min
- **Started:** 2026-09-07T03:36:28Z
- **Completed:** 2026-09-07T03:54:08Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- **The bar is asserted THROUGH THE SERVER, two-sided, at the boundary.** `refuses_holiday_design_cost_over_bound` and `accepts_holiday_design_cost_just_under_bound` differ by ONE history point and straddle `MAX_HOLIDAY_DESIGN_COST`. A control an order of magnitude away would prove only that a huge request is refused.
- **The third factor is reached through the transport for the first time.** `refuses_holiday_dates_total_over_bound` — eleven individually-legal holidays, aggregate 11 000 > 10 000. This case WAS needed; no e2e case had ever touched `max_holiday_dates_total`.
- **The refusal was OBSERVED failing.** With the constant raised to 60 000 the server **accepted the request and returned a full forecast** — quoted verbatim below.
- **The contract now OWNS the rule.** `holiday_design_cost_bounded` + a proof obligation + `FALSIFY-BOUNDARY-018`; `pv status` 17/17/17 -> 18/18/18; `pv diff` suggested `minor` and the version moved 1.1.0 -> 1.2.0. 06-11 predicted exactly this: its `constants:`-only change was invisible to `pv diff`, and the equation is what made the narrowing visible.
- **The recipe is a gate whose failure paths were all seen.** Not one induced red but three: the cargo path, the comparison path, and the provenance (`profile=`) path.

## Task Commits

1. **Task 1: assert the design-cost bar through the server, RED side observed** — `265181e44` (test)
2. **Task 2: contract identity — equation, obligation, FALSIFY-018, binding row** — `e6ca94d97` (feat)
3. **Task 3: turn `just forecast-holiday-bench` into an SC1 gate** — `1632f8a78` (feat)

## Files Created/Modified

- `crates/aprender-mcp-forecast/src/lib.rs` — `WIDE_HOLIDAY_COLUMNS`, `wide_holiday()`, and the three new `#[tokio::test]` e2e cases
- `contracts/forecast-tool-boundary-v1.yaml` — `equations.holiday_design_cost_bounded`, one `type: bound` proof obligation, `FALSIFY-BOUNDARY-018`, version 1.1.0 -> 1.2.0
- `contracts/aprender/binding.yaml` — one row: `holiday_design_cost_bounded` -> `aprender_forecast::forecast::forecast`
- `justfile` — the `forecast-holiday-bench` assertion block and the header paragraph scoping what the bar claims

---

## THE TWO GEOMETRIES (Task 1) — derived from the constant, not from a plan

`MAX_HOLIDAY_DESIGN_COST` read out of `crates/aprender-forecast/src/types.rs:82` = **50 000**.
`MAX_HOLIDAY_DATES_TOTAL` = **10 000**.

Both cases use ONE holiday at the widest legal window — `lower_window = -365`, `upper_window = 365`, so `columns = 365 - (-365) + 1 =` **731** — carrying a SINGLE date. All the excess therefore comes from the product; no other factor is anywhere near its ceiling.

| case | points | horizon | rows = points+horizon | lower/upper | columns | **cells = rows x columns** | vs bound 50 000 |
|---|---|---|---|---|---|---|---|
| **OVER** (`refuses_holiday_design_cost_over_bound`) | 62 | 7 | 69 | -365 / +365 | 731 | **50 439** | **+439 — strictly ABOVE** |
| **UNDER** (`accepts_holiday_design_cost_just_under_bound`) | 61 | 7 | 68 | -365 / +365 | 731 | **49 708** | **-292 — strictly BELOW** |

**One history point separates them.** That is the two-sided control the must-have asks for: at the boundary, not an order of magnitude away.

### Every individual bound is satisfied by the OVER case

| factor | observed | bound | satisfied? |
|---|---|---|---|
| points | 62 | `MIN_POINTS` 10 .. `MAX_POINTS` 20 000 | yes |
| horizon | 7 | 1 .. `MAX_HORIZON` 3 650 | yes |
| ds span | 61 days (62 consecutive days from 2020-01-01) | `MAX_SPAN_DAYS` 20 000 | yes |
| \|lower_window\| | 365 | `MAX_HOLIDAY_WINDOW` 365 | yes (at the ceiling, not over) |
| upper_window | 365 | `MAX_HOLIDAY_WINDOW` 365 | yes (at the ceiling, not over) |
| holiday columns | 731 | `MAX_HOLIDAY_COLUMNS` 1 000 | yes |
| dates in the one holiday | 1 | `MAX_HOLIDAY_DATES` 1 000 | yes |
| dates total | 1 | `MAX_HOLIDAY_DATES_TOTAL` 10 000 | yes |
| **(points + horizon) x columns** | **50 439** | **`MAX_HOLIDAY_DESIGN_COST` 50 000** | **NO — the only violation** |

A case that tripped a neighbouring bound would have proven nothing about this one, so the tests carry their own runtime geometry assertions against `aprender_forecast::types::MAX_HOLIDAY_DESIGN_COST` — if a future editor lowers the constant, the geometry claim goes red rather than silently becoming a different test.

### The near-miss asserts the FULL promised shape, not merely "did not error"

`accepts_holiday_design_cost_just_under_bound` asserts, by name:

- `assert_shared_shape(&out, 7)` — `ds`, `yhat`, `yhat_lower`, `yhat_upper` and `trend` each present with **length == horizon (7)**, every `yhat` finite, and `yhat_lower <= yhat <= yhat_upper` row by row, plus a finite `predict_seconds`
- `out.get("components").is_some_and(|c| c.is_object())`
- `out.get("diagnostics").is_some_and(|d| d.is_object())`

### The third case WAS needed

`grep` of the e2e suite before this plan: **no case reached `max_holiday_dates_total`**. The 24 pre-existing `refuses_*` cases cover single factors and the two 06-10 cap cases; the aggregate date bound had a library test (06-11) and no transport test. `refuses_holiday_dates_total_over_bound` sends **11 holidays x 1 000 dates = 11 000 > 10 000**, each holiday individually at exactly `MAX_HOLIDAY_DATES` and contributing 1 column each (11 total, far under 1 000), so only the aggregate is over.

---

## THE RED SIDE, OBSERVED (Task 1 step 3) — verbatim

### Perturbation 1 — the constant

`crates/aprender-forecast/src/types.rs:82`, `MAX_HOLIDAY_DESIGN_COST: usize = 50_000` -> **`60_000`** (above the OVER case's 50 439).

**First run: the test's OWN geometry pre-assert caught it before the request was sent.** Recorded rather than hidden — it is itself evidence the geometry guard is live:

```
thread 'e2e::refuses_holiday_design_cost_over_bound' (94256513) panicked at crates/aprender-mcp-forecast/src/lib.rs:816:9:
the OVER geometry must exceed the bound it is testing

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 32 filtered out; finished in 0.00s
```

### Perturbation 2 — the geometry pre-assert removed, so the request reaches the door

RC=101. `refused`'s own assertion text, and the reply body carrying a forecast where a refusal was required:

```
thread 'e2e::refuses_holiday_design_cost_over_bound' (94257941) panicked at crates/aprender-mcp-forecast/src/lib.rs:473:9:
must be REFUSED, never defaulted; got: {"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"{\"model\":\"prophet\",\"freq\":\"D\",\"n_history\":62,\"fit_seconds\":0.121906208,\"predict_seconds\":0.00047925,\"ds\":[\"2020-03-03\",\"2020-03-04\",\"2020-03-05\",\"2020-03-06\",\"2020-03-07\",\"2020-03-08\",\"2020-03-09\"],\"yhat\":[11.585374106923329,12.369943467329891,13.14732129899966,13.355890452929582,12.853928048125837,12.0373556001613,11.538886145220914],\"yhat_lower\":[...],\"yhat_upper\":[...],\"trend\":[...],\"components\":{\"weekly\":[...],\"anchor\":[0.0,0.0,0.0,0.0,0.0,0.0,0.0],\"holidays\":[0.0,...],\"additive_terms\":[...

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 32 filtered out; finished in 0.13s
```

`"fit_seconds":0.121906208` and a seven-element `yhat` are the load-bearing details: **the server did not merely fail to refuse — it accepted the request, built the 50 439-cell design, ran the fit and returned a forecast.**

### The accompanying red — 06-11's contract binding is live

The same perturbation turned `types::tests::cost_bounds_match_contract` red, exactly as the plan anticipated:

```
thread 'types::tests::cost_bounds_match_contract' (94259616) panicked at crates/aprender-forecast/src/types.rs:265:13:
assertion `left == right` failed: types::MAX_HOLIDAY_DESIGN_COST must equal constants.fit_max_holiday_design_cost in forecast-tool-boundary-v1
  left: 60000
 right: 50000

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 96 filtered out; finished in 0.00s
```

### Restored — both perturbations, byte-exact

```
types.rs byte-identical
lib.rs byte-identical
types.rs git diff: [0 lines]
```

Both green after restore:

```
types::tests::cost_bounds_match_contract   ->  test result: ok. 1 passed; 0 failed
cargo test -p aprender-mcp-forecast --lib  ->  test result: ok. 33 passed; 0 failed
```

`33` = the 30 that stood after 06-10/06-11, plus this plan's 3. The plan's `>= 32` floor holds and nothing was replaced.

---

## THE CONTRACT IDENTITY (Task 2)

### `pv diff` — the suggested bump, verbatim

```
$ git show HEAD:contracts/forecast-tool-boundary-v1.yaml > /tmp/ftb-old-12.yaml
$ cargo run --release -p aprender-contracts-cli --bin pv -- diff /tmp/ftb-old-12.yaml contracts/forecast-tool-boundary-v1.yaml
Contract diff: v1.1.0 → v1.1.0
Suggested bump: minor

  equations:
    + holiday_design_cost_bounded
  proof_obligations:
    + bound:The work one holiday-carrying request can buy is bounded by a constant this file names, not merely by four factors checked in isolation
  falsification_tests:
    + FALSIFY-BOUNDARY-018
```

**Version before: `1.1.0`. Version after: `1.2.0`.** Applied exactly the bump `pv` suggested, no hand-judgement.

This closes the loop 06-11 left open. 06-11 recorded `Contracts are identical` because `constants:` is a top-level key the parser drops, so its behavioural narrowing was invisible to `pv diff`. The equation is precisely what made it visible — as 06-11's SUMMARY predicted it would be.

### Which statistic the equation declares bounded — and it MATCHES what 06-11 measured

The equation declares **`(points + horizon) x holiday_columns`** bounded — the design cell count — and closes the third factor additively with `fit_max_holiday_dates_total`. That is exactly the statistic 06-11 chose, and 06-11's measurements are the reason: 20 000 x 1 000 x 2 has triple 4.00e7 and cells 2.04e7 and walls at 70.089 s; 3 000 x 181 x 84 has the **larger** triple 4.56e7 but only 6.09e5 cells and walls at 16.216 s. No single threshold on the triple both refuses the 70 s request and accepts the cheap shapes. **No disagreement to report** — the equation's invariant states the ordering follows cells, which is what was measured.

### `pv status` counts

| | before | after |
|---|---|---|
| Equations | 17 | **18** |
| Proof obligations | 17 | **18** |
| Falsification tests | 17 | **18** |

### The audit resolves the new row to a real definition site

| | before | after |
|---|---|---|
| `resolved N Phase 6 binding rows` | **56** (06-10's recorded number) | **57** |
| `BIND-` findings | 0 | **0** |
| `RESOLVE-` findings | 0 | **0** |
| `Total equations:` summaries | 4 | **4** |

`make contract-audit-phase6` rc=0. The row is `contract: forecast-tool-boundary-v1.yaml` / `equation: holiday_design_cost_bounded` / `module_path: aprender_forecast::forecast` / `function: forecast`, which resolves because `fn forecast` has a definition site in `crates/aprender-forecast/src/forecast.rs`.

### `FALSIFY-BOUNDARY-018` names all five assertions

`cost_bounds_match_contract` (constants layer) · `an_in_bounds_holiday_spec_whose_product_is_not_is_refused` and `a_holiday_spec_just_under_the_design_cost_bound_is_accepted` (library door, two-sided) · `refuses_holiday_design_cost_over_bound` and `accepts_holiday_design_cost_just_under_bound` (server, the same two sides). Verified by grep — each string occurs exactly once in the contract file.

The equation's invariants carry the measured breach (`16.113`, `8.926`, `2.664` all present in the file), state that the check runs before `make_design`, and state that `FIT_BUDGET_SECS` is structurally blind to it (`FIT_BUDGET_SECS` now occurs 4 times in the file, up from 2).

---

## THE GATE (Task 3)

### Defaults — unchanged, and deliberately so

06-11's deviation 2 already pointed the recipe at the at-the-bound geometry. Re-deriving it would have been churn:

`points=800`, `horizon=200`, `columns=50` -> `(800 + 200) x 50 =` **50 000 cells** = exactly `MAX_HOLIDAY_DESIGN_COST`. This is the worst request the door still accepts, and the slowest of the three at-the-bound compositions 06-11 measured.

### The passing run

```
HOLIDAY DESIGN WALL: points=800 columns=50 dates=84 holidays=1 horizon=200 cells=50000 triple=3360000 total_s=1.712 fit_s=1.703 predict_s=0.008 other_s=0.001 arch=aarch64 profile=release
  HOLIDAY DESIGN OK: 1.712 s < 2.0 s (SC1)
```

rc=0, exactly one `HOLIDAY DESIGN WALL:` line, `profile=release`. Three runs across the session measured 1.705 / 1.695 / 1.712 s.

### The parse — BY TOKEN, quoted

```bash
total=$(printf '%s\n' "$line" \
    | awk '{ for (i = 1; i <= NF; i++) if ($i ~ /^total_s=/) { sub(/^total_s=/, "", $i); print $i; exit } }')
```

A field scan for a `^total_s=` prefix, not a column index — the printed field order is not load-bearing, so reordering the measurement line cannot silently move what the bar reads.

### `rc` is captured from a redirect, never through a pipe

The recipe's cargo invocation keeps 06-11's `set +e` / `cargo ... > "$LOG" 2>&1` / `rc=$?` / `set -e` block. A token scan of the recipe body finds **1** `rc=$?` and **no** `tee`. This is CLAUDE.md Verification Discipline rule 1, and the defect it prevents shipped twice in this repo (#2336, #2360), both times making a fail-the-job step unreachable.

### THREE failure paths, each OBSERVED red

The plan asked for one. A bar with three failure paths of which one was tested would be two-thirds theatre, so all three were induced.

**1. Over-bound geometry — the cargo path.** `just forecast-holiday-bench 3000 181 84 365` (the verifier's original configuration, now refused at the door):

```
RC=101
FAIL: the holiday design bench exited 101 - log target/p06-forecast-holiday-bench.log
error: Recipe `forecast-holiday-bench` failed with exit code 101
```

with, in the log:

```
thread 'prophet::design_cost::holiday_design_wall' panicked at crates/aprender-forecast/src/prophet.rs:1887:50:
the bench configuration must be accepted: Validation("holidays expand to 609065 design feature cells ((points + horizon) x holiday_columns = (3000 + 365) x 181), which exceeds max_holiday_design_cost 50000; reduce the holiday windows, the number of holidays, the history length or the horizon")
```

The recipe fails LOUDLY rather than passing on a missing measurement line.

**2. The 2.0 comparison itself.** Bar temporarily lowered to `0.5`:

```
RC=1
HOLIDAY DESIGN WALL: ... total_s=1.695 ... profile=release
FAIL: 1.695 s is at or above the 0.5 s SC1 bar, measured on the
      at-the-bound geometry points=800 columns=50
      dates=84 horizon=200. line: HOLIDAY DESIGN WALL: ...
error: Recipe `forecast-holiday-bench` failed with exit code 1
```

The message names the measured number, the geometry and SC1.

**3. The `profile=release` provenance guard.** Pattern temporarily changed to `*profile=nonesuch*`:

```
RC=1
FAIL: the wall was not measured on a release build (profile= is not
      release), so it is not the SC1 bar. line: HOLIDAY DESIGN WALL: ... profile=release
error: Recipe `forecast-holiday-bench` failed with exit code 1
```

All three perturbations restored — `cmp` against a pre-perturbation copy reported the justfile **byte-identical** after each.

### `bashrs lint` — RUN, 0 errors

`bashrs 6.66.3` is available on this host. The recipe body was extracted (with `{{...}}` substituted for shell variables, which is what produces the `SC2154` warnings — `just` supplies those values, not the shell):

```
Summary: 0 error(s), 6 warning(s), 5 info(s)
```

All six warnings are `SC2154` on the four `just` parameters, i.e. extraction artefacts. Of the infos, `SC1012` flags `printf '%s\n'` — correct usage, and the same idiom the pre-existing `forecast-pool-ratio` recipe already uses — and `SC2098` misreads the `*profile=release*` glob pattern as an assignment. Nothing actionable.

### Host-gated, as required

`grep` across `.github/workflows/*.yml` and `Makefile`: **no reference to `forecast-holiday-bench`**. It keeps exactly the status `forecast-bench` and `forecast-pool-ratio` have. No CI workflow was edited (an explicit human escalation per CLAUDE.md).

---

## What the 2 s bar CLAIMS — and what it does not

Recorded in the recipe's own header, not only here:

> It asserts that the WORST holiday-carrying request the door still accepts — the at-the-bound default geometry — walls under 2 s on this release host. It is **NOT** a general SC1 guarantee for every accepted holiday request.

06-11 measured that no payload statistic bounds the wall: the L-BFGS iteration count is data-dependent, so a 4 700-point / 5-column request is 25 000 cells (half the bound) and reproducibly walls at ~4.2 s. `MAX_HOLIDAY_DESIGN_COST` caps **WORK**, not **WALL**. **WINDOWS.md entry 7 / 06-11 coverage D7 remains OPEN**: this plan did not tighten the bound, did not lower `FIT_BUDGET_SECS`, and did not rescope SC1. It scoped its own claim and left the decision to the human.

## Decisions Made

See `key-decisions` in the frontmatter. The two a reader should not miss:

1. **The RED observation needed two perturbations, and both are reported.** Raising the constant alone was intercepted by the test's own geometry pre-assert. Quietly deleting that assert and reporting only the second red would have hidden the fact that the first red measured something different from what the must-have asked for.
2. **The 2 s bar is scoped, not asserted as general SC1.** 06-11's measurements make the general claim false, and the recipe header says so.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Every `<verify>` grep and the `make` audit were unreachable through the rtk hook**

- **Found during:** Task 1 (baseline run)
- **Issue:** The environment's rtk hook rewrites bare `cargo` / `just` / `make` invocations and collapses their output. Every `<verify>` block here greps `test result: ok. N passed; 0 failed`; through the hook that matches nothing on a passing run. Worse for Task 2: `make contract-audit-phase6 > log 2>&1` produced a **51-line** log with **2** `Total equations:` lines against the real **104**-line / **4**-line output, so the plan's `-eq 4` check would have failed on a correct build and the `resolved N binding rows` banner was absent entirely.
- **Fix:** Ran every verification as `rtk proxy <cmd>`. Same resolution 06-10 and 06-11 reached; the prompt flagged it for tests but the `make` case is new evidence — the hook truncates non-cargo commands too.
- **Files modified:** none (execution-harness only)
- **Verification:** `rtk proxy make contract-audit-phase6` produced 104 lines, 4 `Total equations:` and the `resolved 57 Phase 6 binding rows` banner.
- **Committed in:** n/a

**2. [Rule 2 - Missing Critical] The RED observation required removing the test's own geometry pre-assert**

- **Found during:** Task 1 step 3
- **Issue:** The plan's step 3 says to raise `MAX_HOLIDAY_DESIGN_COST` and observe the case fail "with `refused`'s own assertion text and a reply body carrying a forecast". The case's runtime geometry guard (`(ds.len() + 7) * WIDE_HOLIDAY_COLUMNS > MAX_HOLIDAY_DESIGN_COST`) fires FIRST under a raised constant, so the observed red was the guard's text, not the door's.
- **Fix:** Recorded that first red (it is real evidence the guard is live), then removed the pre-assert as a SECOND temporary perturbation to reach the door and capture the required evidence. Both perturbations restored and `cmp`-verified byte-identical.
- **Files modified:** `crates/aprender-forecast/src/types.rs`, `crates/aprender-mcp-forecast/src/lib.rs` — both restored within the task
- **Verification:** `git diff crates/aprender-forecast/src/types.rs` is 0 lines; both files `cmp`-identical to their pre-perturbation copies; both tests green.
- **Committed in:** `265181e44` (the restored state)

**3. [Rule 2 - Missing Critical] Two additional failure paths of the new gate were induced, beyond the one the plan required**

- **Found during:** Task 3
- **Issue:** The plan required observing the recipe fail on an over-bound geometry. That exercises the **cargo-exit** path only — it leaves the `total_s >= 2.0` comparison and the `profile=release` guard, i.e. the two assertions this task actually ADDS, never seen failing. CLAUDE.md rule 7 ("guard regexes ship a case table") and rule 4 ("extending a guard's scope requires re-mutating in the new scope") both bite here.
- **Fix:** Induced both: the bar lowered to 0.5 s, and the `profile=` case pattern broken. Both observed failing with their own messages; both restored byte-identically.
- **Files modified:** `justfile` (restored within the task)
- **Verification:** `cmp /tmp/justfile-pre-red justfile` reported identical after each; `just forecast-holiday-bench` rc=0 afterwards.
- **Committed in:** `1632f8a78` (the restored state)

**4. [Documentation - scope] The recipe's defaults were NOT changed, contrary to the plan's Task 3 action text**

- **Found during:** Task 3
- **Issue:** The plan says to "change the DEFAULTS to the at-the-bound geometry". 06-11's deviation 2 had already done exactly that — `points=800 columns=50 dates=84 horizon=200` is `(800 + 200) x 50 = 50 000` cells, exactly the bound.
- **Fix:** None needed. Verified the arithmetic and left the defaults alone rather than churning them.
- **Files modified:** none
- **Verification:** the measured line reports `cells=50000`, which equals `MAX_HOLIDAY_DESIGN_COST`.
- **Committed in:** n/a

---

**Total deviations:** 3 auto-fixed (1 blocking, 2 missing-critical) + 1 documented scope correction with no code change.
**Impact on plan:** No scope creep — every file touched is in the plan's `files_modified`. Deviation 1 was required for any verification here to measure anything; deviation 2 was required for the RED evidence the must-have demands; deviation 3 strengthens the gate the plan asked for; deviation 4 avoids churn.

## Prohibitions honoured

- **Never assert the wall inside a libtest assertion** — `prophet::design_cost::holiday_design_wall` still asserts only that the call succeeded and `yhat.len() == horizon`. The 2 s bar lives entirely in the host-gated `just` recipe. VERIFIED by reading the test and by the recipe being where the comparison is.
- **Never read `$?` through a pipe in the recipe** — 1 `rc=$?`, captured after a redirect; no `tee` anywhere in the recipe. VERIFIED by token scan.
- **Never write a yq/python/bash script that re-implements `pv`** — `pv validate`, `pv status` and `pv diff` were used directly and their output quoted. The only Python used was for COUNTING lines in an already-produced log (because rtk truncated `grep`'s view), never for parsing or validating a contract. VERIFIED by judgment.
- **Do not touch `crates/aprender-compute`, `.github/workflows/*.yml`, or any parity fixture** — `git status` at every commit showed only the four planned files. VERIFIED.

## Human verification items — ALL THREE REMAIN OPEN

Untouched, unclosed, not planned as automated work:

1. **Drive `initialize` -> `tools/list` -> `tools/call` on both demo pages in a browser and confirm a forecast is charted — OPEN.** This plan touches no demo page.
2. **Measure `quantiles_abs_f32_nonaarch64` on an x86_64 host — OPEN.** Untouched.
3. **Decide whether SC4's Chronos ladder staying dark in CI is acceptable — OPEN.** Untouched.

And the fourth, opened by 06-11 and **still open after this plan**: whether SC1's 2 s bar applies to holiday-carrying requests at all (WINDOWS.md entry 7 / 06-11 coverage D7). This plan scoped its claim to the at-the-bound default and picked none of that item's three options.

## Issues Encountered

- **The rtk hook truncated `make` output**, not just cargo's. The base run of `make contract-audit-phase6` produced a 51-line log with 2 `Total equations:` lines and no `resolved N binding rows` banner — a check written against that output would have failed on a correct build. Diagnosed by counting lines in the log file rather than trusting the terminal view, and resolved with `rtk proxy` (deviation 1).
- **`cargo fmt` reshaped one new assertion** after Task 1's first write; re-run and re-verified before committing.

## Known Stubs

None. A scan of every added line across all four modified files for `TODO`, `FIXME`, `HACK`, `unimplemented!`, `todo!`, `placeholder`, `coming soon` and `not available` returned nothing.

## Threat Flags

None. The change adds no endpoint, no auth path, no file access and no schema change at a trust boundary — it adds two test cases, one contract block, one binding row and a recipe assertion, all of which narrow or observe existing surface. **T-06-SC not applicable:** no package-manager install ran and no dependency entered the graph.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **VERIFICATION gap 2 is closed on all three `missing` bullets** across 06-11 and this plan: bullets 1 and 2 by 06-11 (the constant and the attribution), bullet 3 by this plan's Task 1 (an e2e case proving an over-cost request is refused, observed RED). The contract half of bullet 1 is this plan's Task 2.
- `contracts/forecast-tool-boundary-v1.yaml`'s HARD-ceiling claim is now backed by a constant, a door check, a two-sided library pair, a two-sided e2e pair, a contract equation with a proof obligation, and a host-gated wall assertion.
- **06-13 inherits:** `FALSIFY-BOUNDARY-018` is taken, so the next id is 019; the contract is at version 1.2.0; `pv status` reads 18/18/18; the Phase 6 binding rows resolve at 57.
- Blockers: none. One open human decision carried forward (WINDOWS.md entry 7).

## Self-Check: PASSED

Files claimed modified — all present on disk:

```
FOUND: crates/aprender-mcp-forecast/src/lib.rs
FOUND: contracts/forecast-tool-boundary-v1.yaml
FOUND: contracts/aprender/binding.yaml
FOUND: justfile
```

Commits claimed — all present in history:

```
FOUND: 265181e44
FOUND: e6ca94d97
FOUND: 1632f8a78
```

Plan-level `<verification>` block, re-run at close-out:

| Check | Result |
|---|---|
| `cargo test -p aprender-mcp-forecast --lib` | `ok. 33 passed; 0 failed` (>= 32 floor) |
| `cargo test -p aprender-forecast --lib types::tests::cost_bounds_match_contract` | `ok. 1 passed; 0 failed` |
| `cargo test -p aprender-forecast --lib` | `ok. 89 passed; 0 failed; 8 ignored` (unmoved from 06-11) |
| `pv validate contracts/forecast-tool-boundary-v1.yaml` | `0 error(s), 0 warning(s) / Contract is valid.` |
| `pv status contracts/forecast-tool-boundary-v1.yaml` | `Equations: 18 / Proof obligations: 18 / Falsification tests: 18` |
| `make contract-audit-phase6` | rc=0, BIND- 0, RESOLVE- 0, 4 `Total equations:`, `resolved 57 Phase 6 binding rows` |
| `just forecast-holiday-bench` | rc=0, one WALL line, `HOLIDAY DESIGN OK: 1.712 s < 2.0 s (SC1)`, `profile=release`; OBSERVED failing on three induced paths |
| `just forecast-bench` | `ROUND TRIP OK: 0.212 s < 2.0 s (SC1)` — unchanged |
| `cargo clippy -p aprender-mcp-forecast -p aprender-forecast --all-targets --no-deps -- -D warnings` | rc=0 |
| `cargo fmt --all -- --check` | rc=0 |

Acceptance-criteria greps, re-run:

```
lib.rs  async fn refuses_holiday_design_cost_over_bound        -> line 807 (exactly 1)
lib.rs  async fn accepts_holiday_design_cost_just_under_bound  -> line 832 (exactly 1)
lib.rs  async fn refuses_holiday_dates_total_over_bound        -> line 874 (exactly 1)
contract  holiday_design_cost_bounded: 1     binding  equation: holiday_design_cost_bounded: 1
contract  16.113: 1   8.926: 1   2.664: 1    FALSIFY-BOUNDARY-018: 1
contract  FIT_BUDGET_SECS: 4 (>= 2)
contract  cost_bounds_match_contract: 3, and each of the four other named assertions: 1
justfile  HOLIDAY DESIGN OK: 1 (>= 1);  rc=$? in the recipe body: 1 (>= 1)
scope fence: no .github/workflows/*.yml or Makefile reference to forecast-holiday-bench
```

Commit count MEASURED, not narrated: `git rev-list --count 2c774ee56..HEAD` = **3**.

---
*Phase: 06-native-time-series-forecasting-stack*
*Completed: 2026-09-07*
