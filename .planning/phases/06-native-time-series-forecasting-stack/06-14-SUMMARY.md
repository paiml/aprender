---
phase: 06-native-time-series-forecasting-stack
plan: 14
subsystem: api
tags: [forecasting, prophet, dos-bounds, provable-contracts, mcp, rust, schemars]

requires:
  - phase: 06-native-time-series-forecasting-stack
    provides: "06-13's corrected Poisson sampler (whose correct fix un-bounded the simulated changepoint count — CR-01), 06-11's MAX_HOLIDAY_DESIGN_COST derivation template, 06-12's straddling-e2e-pair + raised-constant RED template"
provides:
  - "door_surface: in contracts/forecast-tool-boundary-v1.yaml — 16 caller-settable knobs and 14 cost axes, enumerated by inspection and MACHINE-CHECKED in both directions"
  - "types::MAX_LOGISTIC_CHANGEPOINT_LAMBDA (f64, 20 000) + constants.fit_max_logistic_changepoint_lambda, refused at the door before make_design (CR-01, cost axis C-06)"
  - "prophet::changepoint_count — the ONE implementation of the effective changepoint count, read by both the door and make_design"
  - "test_support::constant_f64 and contract_value"
  - "constants.poisson_normal_branch_lambda — WR-01's contract mirror for the sampler branch threshold"
  - "types::tests::no_cost_axis_is_pending — a DECLARED-RED guard holding C-07 and C-08 open until 06-15"
affects: [06-15, 06-16, 06-17]

actuals:
  tokens: 21474
  tasks: 3
  commits: 5
plan_head_before: 05497fdd45d17efb9973444262d8c0ec9147f9ec

tech-stack:
  added: []
  patterns:
    - "Enumerate the whole surface, then close one instance of it — a checked enumeration instead of a fourth point fix"
    - "Split a completeness check in two so a known-open item is a persistent RED test rather than a prose note"
    - "Derive a checked list from schemars::schema_for! (the same generator that produces the advertised schema) rather than hand-keeping it"

key-files:
  created: []
  modified:
    - crates/aprender-forecast/src/prophet.rs
    - crates/aprender-forecast/src/types.rs
    - crates/aprender-forecast/src/forecast.rs
    - crates/aprender-forecast/src/test_support.rs
    - crates/aprender-mcp-forecast/src/lib.rs
    - contracts/forecast-tool-boundary-v1.yaml
    - contracts/aprender/binding.yaml

key-decisions:
  - "The bound value 20 000 was DERIVED from three at-the-bound release measurements, not chosen; all three clear the 2 s SC1 bar with 2.8x worst-case headroom"
  - "C-07 carries no_structural_maximum: true — it can be closed only by a bound, never by a measurement, because holidays[].name has no structural maximum on either transport"
  - "no_cost_axis_is_pending SHIPS RED rather than shipping a dishonest measured_at_structural_maximum for C-07"
  - "changepoint_geometry is a private single arithmetic site; changepoint_count is the public LENGTH reading of it, so the door and make_design cannot drift"
  - "C-05's bound recorded as fit_max_horizon with an explicit note, correcting the plan's 'NONE' — the allocation IS bounded; what is not is the SPAN that count buys, which routes through C-06"

patterns-established:
  - "Pattern: a completeness test whose failure message prints the symmetric difference in BOTH directions, so a failure names the field rather than a count"
  - "Pattern: a declared-red test carrying its own doc comment explaining why it is red, what it turns red, and which repair is forbidden"
  - "Pattern: an in-test assert! proving the geometry is on the side the test name claims, so a constant change cannot leave a test named 'over' sitting under the bound"

requirements-completed: [SC1]

coverage:
  - id: D1
    description: "The door's caller-settable surface (16 knobs) and cost surface (14 axes) are enumerated in contracts/forecast-tool-boundary-v1.yaml door_surface: and machine-checked in both directions"
    requirement: SC1
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/types.rs#types::tests::every_request_knob_is_enumerated"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/types.rs#types::tests::every_cost_axis_names_a_real_bound"
        status: pass
      - kind: unit
        ref: "cargo test -p aprender-forecast --lib -- types::tests::every_request_knob_is_enumerated types::tests::every_cost_axis_names_a_real_bound types::tests::pool_default_matches_contract (3 passed)"
        status: pass
    human_judgment: false
  - id: D2
    description: "CR-01 closed: a logistic request whose Poisson mean exceeds a measured, contract-owned bound is refused at the door before make_design, with the near-miss and the linear arm still accepted"
    requirement: SC1
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::a_logistic_request_over_the_changepoint_lambda_bound_is_refused"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::a_logistic_request_just_under_the_changepoint_lambda_bound_is_accepted"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::the_linear_arm_at_the_same_geometry_is_still_accepted"
        status: pass
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::refuses_logistic_changepoint_lambda_over_bound"
        status: pass
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::accepts_logistic_changepoint_lambda_just_under_bound"
        status: pass
    human_judgment: false
  - id: D3
    description: "The door and make_design cannot disagree about the effective changepoint count, so the bound is not evadable"
    requirement: SC1
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/prophet.rs#prophet::changepoints::changepoint_count_equals_the_design_it_describes"
        status: pass
      - kind: unit
        ref: "cargo test -p aprender-forecast --lib prophet::parity (32 passed; 0 failed, before and after)"
        status: pass
    human_judgment: false
  - id: D4
    description: "WR-01: prophet::POISSON_NORMAL_BRANCH_LAMBDA is contract-owned, with the mirror proven by mutating the YAML value alone"
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/types.rs#types::tests::cost_bounds_match_contract"
        status: pass
    human_judgment: false
  - id: D5
    description: "IN-03: cost_bounds_match_contract no longer states a drifting count and no longer carries DEFAULT_POOL, which has its own assertion"
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/types.rs#types::tests::pool_default_matches_contract"
        status: pass
    human_judgment: false
  - id: D6
    description: "The bound value is justified by three at-the-bound release wall measurements (freq D / W / MS), all under the 2 s SC1 bar"
    requirement: SC1
    verification:
      - kind: other
        ref: "LOGISTIC_BENCH_FREQ={D,W,MS} cargo test --release -p aprender-forecast --lib prophet::sampler::logistic_band_wall -- --ignored --nocapture (0.244 / 0.711 / 0.582 s, all profile=release)"
        status: pass
    human_judgment: false
  - id: D7
    description: "C-07 and C-08 are held OPEN by a persistent red test naming both, rather than reduced to a note"
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/types.rs#types::tests::no_cost_axis_is_pending — EXPECTED RED; exit 101, '1 failed', names C-07 and C-08"
        status: fail
    human_judgment: true
    rationale: "This deliverable's success condition is a FAILING test, which no auto-pass rule can express. A human must confirm the red is the designed one — the pending-marker assertion naming C-07 and C-08 — and not an unrelated regression, and must confirm the push-sequencing mitigation (land waves 12 and 13 in one push) before this branch reaches CI."

duration: 2h 5m
completed: 2026-09-07
status: complete
---

# Phase 06 Plan 14: Enumerate the Door Surface and Close CR-01 Through It — Summary

**`door_surface:` lands 16 machine-checked knobs and 14 cost axes in the tool-boundary contract, and CR-01 closes as the first instance of it: a logistic request whose Poisson mean exceeds a measured `MAX_LOGISTIC_CHANGEPOINT_LAMBDA` (20 000) is refused at the door before `make_design`, proven red-side through a live server at 2.579 s.**

---

## ⚠️ READ FIRST — THIS PLAN SHIPS A DELIBERATELY FAILING TEST

**`types::tests::no_cost_axis_is_pending` is RED right now, on purpose.** It is not broken, not flaky, and not an oversight. It is a real failing assertion naming the two cost axes this plan enumerated and did **not** close — **C-07** (`holidays[].name` byte amplification, owed by 06-15 T2) and **C-08** (unbudgeted NeuralProphet training work, owed by 06-15 T3) — and it stays red until **plan 06-15** replaces both pending markers with real bounds.

**What this turns red, verified by reading the gates rather than assumed:**

| Gate | Effect during the window |
|---|---|
| git hooks (`.git/hooks/` has no `pre-commit`, only `.sample` files) | unaffected — committing wave 12 is not blocked |
| `make tier1` (fmt / clippy / check) | unaffected |
| `make tier2` (its `cargo test --lib` runs at the workspace root, the near-empty facade) | unaffected |
| `make tier3` (`cargo test --all`, `Makefile:313`) | **RED for the whole window** — this is the documented pre-push gate |
| CI `workspace-test` (`cargo nextest … --workspace --lib`, which does **not** exclude `aprender-forecast`) | **RED for the whole window** |

**And `workspace-test` is a REQUIRED status check on a protected `main`** (CLAUDE.md, Git Workflow). A red `workspace-test` does not just look untidy — it blocks the PR outright.

**THE MITIGATION IS PUSH SEQUENCING, NOT A CI EDIT: land waves 12 and 13 in ONE push.** Commit 06-14 and 06-15 separately, as normal, but do **not** `git push` between them. The red window then never reaches a CI runner while the local red still does its whole job. There is no `--skip` escape on the CI side: `-- --skip no_cost_axis_is_pending` is a libtest idiom, `cargo nextest` can only express it as `-E 'not test(no_cost_axis_is_pending)'`, and applying that would require editing `.github/workflows/ci.yml` — fenced off by every plan in this round and a human check-in per CLAUDE.md.

**THE FORBIDDEN REPAIR.** Deleting, weakening, `#[ignore]`-ing, `#[should_panic]`-ing or CI-filtering this assertion restores precisely the guard-that-cannot-fail contradiction it exists to remove — the IN-01 class this whole round is closing. **The only legitimate repair is 06-15 landing.** If the window must close sooner for an unrelated reason, that is a human decision to re-sequence the round, not a licence to edit the assertion.

**What forces 06-15 if someone stops here:** the local red, which is a stronger forcing function than a note. `cargo test -p aprender-forecast --lib` and `make tier3` both fail on the developer's own machine, before any push is possible. The branch cannot be pushed green, so stopping after wave 12 is not a quiet partial success — it is a branch that will not go anywhere.

**The verbatim failing output:**

```
thread 'types::tests::no_cost_axis_is_pending' panicked at crates/aprender-forecast/src/types.rs:603:9:
2 cost axis/axes are still UNBOUNDED and are held open by this assertion: C-07 (marker
unbounded_pending_06_15, owed by plan 06-15); C-08 (marker unbounded_pending_06_15, owed by plan
06-15). This test is RED ON PURPOSE from the close of wave 12 (plan 06-14) until wave 13 (plan
06-15) replaces both pending markers with real bounds. Do NOT repair it by deleting, weakening,
ignoring or CI-filtering it — that reintroduces the guard-that-cannot-fail class this round exists
to end. Land 06-15.

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 108 filtered out
```

**The crate's `--lib` suite is RED on exactly that one test and nothing else:**
`cargo test -p aprender-forecast --lib -- --skip no_cost_axis_is_pending` → **99 passed; 0 failed; 9 ignored; 1 filtered out**.

---

## Performance

- **Duration:** ~2h 5m
- **Tasks:** 3 (one tracer, one TDD, one auto)
- **Commits:** 5
- **Files modified:** 7

## Accomplishments

- **The class, not the probe.** `door_surface:` enumerates every caller-settable knob (16) and every cost axis (14) with the arm that reads it, the enforcement that covers it, where it is spent, and whether that is outside `fit::FIT_BUDGET_SECS`. **13 of the 14 axes are outside it** — the structural reason every bound this phase has added had to go at the door.
- **CR-01 closed as an instance of that enumeration** (axis C-06), at three layers: constant mirror, two-sided library control, live-server straddling pair — with the refusal **observed red** under a raised constant.
- **The bound is derived from measurement**, at three compositions, on a release build.
- **WR-01 and IN-03 closed**, each with an observed red side.
- **C-07 and C-08 are held open by a test, not a note.**

## Task Commits

1. **Task 1 (pure refactor, committed separately so the behaviour-preserving claim is bisectable)** — `e1b441944` (refactor)
2. **Task 1: enumerate the door surface, close CR-01 through it** — `bf5bb2e71` (feat)
3. **Task 2 RED: the class invariant as three tests + the WR-01 mirror** — `258d77f9a` (test)
4. **Task 2 GREEN: contract-own the branch threshold and the invariant** — `3184b1402` (feat)
5. **Task 3: prove the bound through a live server; the contract owns it** — `0433c4502` (feat)

## The re-derived enumeration, and every delta from the plan's table

The plan's `<door_surface_enumeration>` was a hypothesis. It was re-derived by reading `forecast.rs`, `types.rs`, `prophet.rs`, `dates.rs`, `np.rs` and `fit.rs` after a `pmat query` semantic pass (CLAUDE.md Code Search Policy — `pmat query "forecast door validation bound refusal"` and `pmat query "uncertainty simulation sample loop"`, each hit then confirmed by reading the source).

**The knob count confirms exactly: 16 (12 `ForecastArgs` + 4 `HolidayArg`).** The axis count confirms at 14.

**Deltas found by inspection — the source won in each case:**

| # | Plan's table said | Inspection found | Recorded as |
|---|---|---|---|
| 1 | C-05 `bound:` **NONE** — feeds C-06 | The *allocation* IS bounded: `len(fut) == horizon <= fit_max_horizon`. What is unbounded is the **span** that count buys (30.44x more days on `MS` than `D`), whose only cost consumer is C-06 | `bound: fit_max_horizon` plus an explicit `note:` routing the amplification to C-06. Writing `NONE` would have been wrong in the direction that matters — it would imply an open axis where the open thing is the *ratio*, which C-06 now closes |
| 2 | C-04 formula `(points + horizon) * holiday_columns` | `predict` additionally runs `comp_of` **once per distinct component name**, so the cells are re-swept `(1 + distinct_components)` times. `distinct_components <= fit_max_holiday_columns`, so the product stays bounded (≤ 5e7 float ops at the ceiling) — but the multiplier **was not part of 06-11's derivation** | A `note:` on C-04. Not a new axis and not a live defect: the bound still holds arithmetically. Recorded rather than left implicit |
| 3 | C-07 spent in `prophet::columns` | Also spent in `prophet::predict`'s component-name dedup (`names.contains` is O(K × \|names\|) string comparisons over arbitrary-length names) | C-07's `spent_in` names **both** sites. This *strengthens* C-07 rather than adding a 15th axis |

**Axes the enumeration found that `06-REVIEW.md` did NOT point at.** The review raised exactly one cost axis (C-06, as CR-01). The enumeration additionally names and dispositions **C-05, C-09, C-10, C-11, C-12, C-13 and C-14** — the future-grid span with its frequency multiplier, the linear/flat uncertainty simulation, the percentile sorts, the seasonality design cells, the NP recursive AR predict, the response serialization, and the L-BFGS fit. None was in any review finding. C-11 and C-14 also record facts that were previously implicit: `auto_seasonalities` is **not** caller-settable (its 34-column ceiling is a property of the code), and C-14 is the **only** axis inside `FIT_BUDGET_SECS`.

**No unbounded axis was found that this plan does not already route to 06-15**, so the plan's STOP-and-report condition **did not trigger**. C-07 and C-08 are the only two open axes and both were already routed.

## The bound's derivation — three at-the-bound release walls

`LOGISTIC_BENCH_FREQ={D,W,MS} LOGISTIC_BENCH_POINTS=33 LOGISTIC_BENCH_HORIZON=… cargo test --release -p aprender-forecast --lib prophet::sampler::logistic_band_wall -- --ignored --nocapture`

```
LOGISTIC BAND WALL: points=33 horizon=3650 freq=D  growth=logistic lambda=2851.6  total_s=0.244 fit_s=0.029 predict_s=0.215 mean_band_width=45.5102 profile=release
LOGISTIC BAND WALL: points=33 horizon=3650 freq=W  growth=logistic lambda=19960.9 total_s=0.711 fit_s=0.030 predict_s=0.680 mean_band_width=49.3714 profile=release
LOGISTIC BAND WALL: points=33 horizon=840  freq=MS growth=logistic lambda=19974.2 total_s=0.582 fit_s=0.029 predict_s=0.552 mean_band_width=49.3891 profile=release
```

All three clear the 2 s SC1 bar; the worst is **0.711 s**, ~2.8x headroom. The verify gate **parses** each `total_s` and compares it after a shape check rather than counting a `profile=release` marker — a marker count would pass at 5.0 s per composition, which is exactly how a bound value gets set from walls that are over the bar. `checked=3 bad=0`.

The cost is almost entirely `predict_s`, which is the half `FIT_BUDGET_SECS` structurally cannot cover.

**`freq: "D"` cannot reach this bound at all.** At 33 points its largest attainable lambda is 2 851.6, so its row is that frequency's **own structural maximum** rather than an at-the-bound point. This is a fact about the frequency multiplier, not a gap: `MS` buys 30.44x the span of `D` for the same legal horizon, which is precisely why the bound must be on the PRODUCT and not on the horizon. **Recorded as a delta** — the plan's acceptance criterion assumed three at-the-bound rows.

**The structural maximum is 86 790**, reached at 33 daily points / horizon 3650 / `MS` — *exactly* the review's measured configuration.

## The refusal precedes the design build

- First occurrence of `max_logistic_changepoint_lambda` in `crates/aprender-forecast/src/forecast.rs`: **line 323**
- The `make_design(&ds, &args.y, &spec)` call: **line 329**

323 < 329. The refusal is before the design build, which is the only place it *can* be: the cost is spent in `predict`, which `forecast` calls after the fit returns with no budget at all.

`grep -nE 'fn changepoint_count\(' crates/aprender-forecast/src/prophet.rs` → exactly one line, **268**. (An unanchored `grep -n 'fn changepoint_count'` prints two lines, because the plan-mandated test name `changepoint_count_equals_the_design_it_describes` contains the string — a naming artefact, not a second implementation.) `grep -c 'changepoint_count(' …` → **5**, ≥ 2 as required. The single arithmetic site is the private `changepoint_geometry` at line **247**.

## The three mutation controls — all observed RED, then restored

**(a) One `door_surface.knobs` entry DELETED (the `cap` entry), YAML only:**
```
  MISSING from the contract (a field exists with no entry — nothing has reasoned about its cost
  or its enforcement): [("ForecastArgs", "cap")]
  PHANTOM in the contract (an entry names a field that exists on neither struct — the
  enumeration has stopped describing the code): []
test result: FAILED. 0 passed; 1 failed
```
Restored → `test result: ok. 1 passed; 0 failed`.

**(b) A PHANTOM entry added for `temperature`, a field on neither struct:**
```
  MISSING from the contract (…): []
  PHANTOM in the contract (…): [("ForecastArgs", "temperature")]
test result: FAILED. 0 passed; 1 failed
```
Restored → `test result: ok. 1 passed; 0 failed`.

This is the direction that lets an enumeration accumulate entries for fields that were **removed** — the way an enumeration silently stops describing the code while still looking complete. Set equality gives both directions structurally, but a direction nobody mutated is a direction nobody has tested.

**(c) `poisson_normal_branch_lambda` changed to 31 in the YAML ALONE, touching no Rust:**
```
thread 'types::tests::cost_bounds_match_contract' panicked at crates/aprender-forecast/src/types.rs:319:13:
types::prophet::POISSON_NORMAL_BRANCH_LAMBDA (30) must equal
constants.poisson_normal_branch_lambda (31) in forecast-tool-boundary-v1
test result: FAILED. 0 passed; 1 failed
```
Restored → `test result: ok. 1 passed; 0 failed`, and `cmp` confirms the revert is **byte-identical** to the pre-mutation file.

## The RED side, through the live streamable-HTTP server

With `types::MAX_LOGISTIC_CHANGEPOINT_LAMBDA` and `constants.fit_max_logistic_changepoint_lambda` both temporarily raised to 200 000 (past the 86 790 structural maximum), a temporary probe drove the review's exact geometry through `serve()`:

```
REDSIDE: bound=200000 lambda=86790.6 accepted=true wall_s=2.579
test result: ok. 1 passed; 0 failed
```

**The server ACCEPTED the request and returned a full-shape forecast, walling at 2.579 s** — reproducing `06-REVIEW.md`'s 2.334 s claim on this host (same order; this run is release, in-process, and includes transport). Restored to 20 000, the same request is refused and both e2e cases pass (`2 passed`).

Note on how the observation had to be made: the first attempt tripped the e2e test's **own** in-test geometry guard (`the OVER geometry must exceed the bound it is testing: lambda=86790.6 vs 200000`) — which is that guard working exactly as designed, since it exists to stop a test named "over" from silently sitting under a moved bound. The probe was therefore a separate temporary test, removed before commit (`grep -c 'tmp_redside' …` → 0).

## `prophet::parity` before and after

`test result: ok. 32 passed; 0 failed` at the plan base `05497fdd4`, again after the `changepoint_count` extraction, and again at the close of the plan. `git diff --stat 05497fdd4` shows **no change** to `contracts/{prophet,neuralprophet,chronos-bolt}-parity-v1.yaml` or to anything under `crates/aprender-forecast/tests/fixtures/`.

## C-07 has no structural maximum — the read that establishes it

`crates/aprender-mcp-forecast/src/lib.rs`'s router construction was **read**, not assumed:

- `http_app` (line 77) builds `axum::Router::new()` with three `get` routes and `.nest("/mcp", mcp)`. There is **no `DefaultBodyLimit` layer, no `max_body`, and no content-length layer** anywhere in it.
- `pooled_app` (line 140) composes K `http_app`s behind a round-robin fallback and adds no limit of its own.
- The stdio transport has **no framing cap at all**.

`holidays[].name` is therefore an unbounded `String` with no structural maximum, and any `measured_seconds` written for C-07 would measure an **arbitrarily chosen** name length rather than a maximum. That is why its `cost_axes` entry carries `no_structural_maximum: true`, recording that `measured_at_structural_maximum` is not an available disposition for it, and why only a real bound (06-15 T2) can close it.

*(Caveat recorded for 06-15: if `pmcp`'s own router internally applies axum's default 2 MB `Bytes` limit, the HTTP surface would have an implicit cap — but the **stdio** surface provably does not, and both share one door, so the door still has no structural maximum. 06-15 should bound the field rather than rely on any transport limit.)*

## Contract version

`pv diff` run on **two real filesystem paths** (`git show 05497fdd4:contracts/forecast-tool-boundary-v1.yaml > /tmp/ftb-old.yaml`, then `pv diff /tmp/ftb-old.yaml contracts/…`), never a git revision:

```
Contract diff: v1.2.0 → v1.2.0
Suggested bump: minor
  equations:
    + door_surface_is_complete
    + logistic_changepoint_cost_bounded
  proof_obligations:
    + bound:The Poisson mean the logistic uncertainty simulation draws is bounded …
    + completeness:The door surface is fully enumerated …
  falsification_tests:
    + FALSIFY-BOUNDARY-019
    + FALSIFY-BOUNDARY-020
```

Applied: **`metadata.version` 1.2.0 → 1.3.0**. `pv status` counts moved 18 → **20** equations, 18 → **20** proof obligations, 18 → **20** falsification tests.

## Review Dispositions Ledger — this plan's three findings

| ID | Severity | Disposition | Status at close of 06-14 |
|---|---|---|---|
| **CR-01** | Critical | INCORPORATED | **CLOSED** — enumeration + `MAX_LOGISTIC_CHANGEPOINT_LAMBDA` (T1), red-side e2e + `logistic_changepoint_cost_bounded` (T3). The sweep gate that would have caught it remains 06-16 |
| **WR-01** | Warning | INCORPORATED, SPLIT | **OWNERSHIP HALF CLOSED** — `poisson_normal_branch_lambda` is contract-owned with an observed red side. The *test-strength* half (a tight sub-threshold sweep point the normal branch actually fails, so lowering the constant can no longer leave the suite green) remains **06-17 T1** |
| **IN-03** | Info | INCORPORATED | **CLOSED** — the doc no longer states a drifting count, and `DEFAULT_POOL` moved to `pool_default_matches_contract` |

The other six findings (WR-02, WR-03, WR-04, IN-01, IN-02, IN-04) are unchanged and land in 06-15, 06-16 and 06-17 exactly as the ledger records. None was silently dropped.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] The `rtk` hook rewrites cargo test output, so every verify command's grep was unsatisfiable**

- **Found during:** Task 1 (baseline parity run, before any edit)
- **Issue:** This box runs an `rtk` Bash hook that summarises `cargo test` output — even through a shell redirect to a file. `test result: ok. 32 passed; 0 failed…` becomes `cargo test: 32 passed, 68 filtered out`. Every one of the plan's verify commands greps for the libtest literal, so all of them would have failed against passing runs — or, worse, a `grep -c '… 0 failed'` returning 0 could have been read as evidence.
- **Fix:** Every `cargo test` in this plan's verification was run as `rtk proxy cargo test …`, which bypasses the filter and emits raw libtest output. No assertion was weakened to fit the filtered format.
- **Verification:** `rtk proxy cargo test -p aprender-forecast --lib prophet::parity` → `test result: ok. 32 passed; 0 failed; …`, the exact literal the gate wants.

**2. [Rule 3 - Blocking] The wall-measurement verify command would have panicked on `MS`**

- **Found during:** Task 1 step (6)
- **Issue:** The plan's loop sets only `LOGISTIC_BENCH_FREQ`, leaving `LOGISTIC_BENCH_POINTS=100` / `LOGISTIC_BENCH_HORIZON=3650`. At 100 points and horizon 3650, `MS` gives lambda **28 052** — *over* the very 20 000 bound being measured. The bench does `.expect("the bench configuration must be accepted")`, so the door would refuse and the test would panic. The `D` row would also have measured lambda 922, not an at-the-bound value.
- **Fix:** Each composition sets `LOGISTIC_BENCH_POINTS=33` (the tightest legal history that still earns all 25 changepoints) and a per-freq horizon — D 3650, W 3650, MS 840 — so `W` and `MS` sit within 0.2% of the bound and `D` sits at its own maximum.
- **Verification:** Three `LOGISTIC BAND WALL:` lines, `checked=3 bad=0`, all `profile=release`, all under 2.000 s.

**3. [Rule 3 - Blocking] `pv status` prints counts, not equation names**

- **Found during:** Task 2 step (5)
- **Issue:** The plan's gate is `pv status … | grep -q 'door_surface_is_complete'`. `pv status` emits the description block and a count summary; it never prints equation names, so the gate could not pass however correct the contract was.
- **Fix:** Used `pv equations contracts/forecast-tool-boundary-v1.yaml`, which names them — still dogfooded `pv` per CLAUDE.md, no bash workaround. `pv status`'s count is retained as corroboration.
- **Verification:** `pv equations` prints `door_surface_is_complete` and `logistic_changepoint_cost_bounded`; `pv status` counts moved 18 → 20 on all three axes, confirming both are parsed *as equations* at the right nesting level.

**4. [Rule 3 - Blocking] The audit gate `grep -c 'BIND-' == 0` is unsatisfiable by construction**

- **Found during:** Task 3 step (4)
- **Issue:** `make contract-audit-phase6`'s **success** line is `Phase 6 binding audit: 4 contract(s) audited, zero BIND- findings` — which contains the literal `BIND-`. The gate therefore reports 1 on a perfectly clean audit. This is CLAUDE.md Verification Discipline rule 7 exactly: the guard regex was never run against a case table.
- **Fix:** Refined to `grep -E '(^|[[:space:]])(BIND|RESOLVE)-[0-9]' | grep -v 'zero BIND- findings'`, and shipped a must-match / must-not-match case table with it (`BIND-001 …` MATCH, `RESOLVE-014 …` MATCH, the success line NO-MATCH, `no BIND- findings were produced` NO-MATCH).
- **Verification:** `real BIND-/RESOLVE- FINDINGS: 0`, `Total equations summaries: 4`, `rc=0`, `resolved 60 Phase 6 binding rows`.

**5. [Rule 1 - Bug] The Phase 6 binding header's row count was already stale at HEAD**

- **Found during:** Task 3 step (4)
- **Issue:** `contracts/aprender/binding.yaml:1610` claimed "55 rows"; the block actually held **58** before this plan touched it. The count had drifted by 3 and nothing checked it — the same drift class as IN-03.
- **Fix:** Corrected to the true post-change total, **60** (58 pre-existing + 2 added). Recorded here rather than silently rewritten, because the pre-existing drift is itself a finding.
- **Verification:** `awk 'NR>1610' … | grep -c '^- contract:'` → 60, matching the header.

**6. [Rule 1 - Bug] `check tdd-red-evidence` parses node:test TAP and cannot read Rust libtest**

- **Found during:** Task 2 RED gate
- **Issue:** The verb's `parseNodeTestSummary` matches `^# tests N` / `^# pass N` / `^# fail N` and `tapFailedTestNames` matches `^not ok N - name`. Rust libtest emits none of these, so a genuine Rust RED classifies as `INVALID_RED (zero_tests_discovered)` — the gate would be skipped rather than satisfied on every Rust project. (Its record schema is also camelCase — `targetTest` / `exitCode` — which the plan's prose does not state.)
- **Fix:** Translated the libtest run mechanically into the TAP shape the verb parses — same test names, same per-test verdicts, same totals — and kept the raw libtest output in the record as `rawLibtestOutput` so the translation is auditable. The gate then classified the *real* run rather than being bypassed.
- **Verification:** `RED_EVIDENCE_OK`, reason `target_test_failed`, `exit_code 101`, `tests 10 / pass 8 / fail 2`, target `types::tests::cost_bounds_match_contract`.

**7. [Rule 1 - Bug] Two numbers in draft artifacts were pre-measurement estimates**

- **Found during:** Task 1 step (6) and Task 3 step (1)
- **Issue:** (a) `MAX_LOGISTIC_CHANGEPOINT_LAMBDA`'s doc-comment wall table was drafted with *projected* walls (0.088 / 0.635 / 0.190 s) before the release run. (b) The e2e comment repeated the review's "1 132 bytes" as though this test's payload had been measured at it.
- **Fix:** (a) Replaced with the measured values (0.244 / 0.711 / 0.582 s) plus `predict_s` and `mean_band_width`. (b) Measured this payload's `arguments` object at **1 138 bytes** and said so, attributing 1 132 to the review's own probe.
- **Verification:** The doc table now matches the `LOGISTIC BAND WALL:` lines quoted above verbatim; the byte count was measured by serialising the exact payload.

---

**Total deviations:** 7 auto-fixed (4 blocking [Rule 3], 3 bugs [Rule 1]).
**Impact on plan:** No scope creep. Four of the seven are defects in the plan's own *verify commands* — each would have reported a false result (three unsatisfiable-by-construction greps and one that would have panicked). Two are pre-existing drift/tooling defects found by doing the work. One is self-correction of numbers that had not yet been measured. Every plan objective was met as written.

## Issues Encountered

- **`set -- $SPEC` does not word-split in zsh**, so the first wall loop passed `"D 3650"` as a frequency. This surfaced as `unsupported freq "D 3650": use D, W or MS` — which incidentally confirmed that `logistic_band_wall` refuses an unrecognised frequency rather than silently coercing it to `"D"`. Re-run as three explicit invocations.
- **`cargo test a b c` is not multi-filter syntax**; libtest needs `-- a b c`. The plan's multi-name verify commands were adjusted accordingly, and the `N passed` count was checked each time precisely to rule out the zero-match trap.

## Known Stubs / Open Items

Two cost axes are **deliberately open**, enumerated, and held open by a failing test rather than by a note. Both are recorded in `.planning/WINDOWS.md`.

| Axis | File | What is open | Owed by |
|---|---|---|---|
| **C-07** | `contracts/forecast-tool-boundary-v1.yaml` (`door_surface.cost_axes`) | `holidays[].name` byte amplification: `holiday_columns * 2 * len(name)` plus an `O(C log C)` string sort and `predict`'s `O(C * distinct_components)` dedup. `bound: unbounded_pending_06_15`, `no_structural_maximum: true` | **06-15 T2** |
| **C-08** | `contracts/forecast-tool-boundary-v1.yaml` (`door_surface.cost_axes`) | NeuralProphet training work `n_lrs * epochs * n_samples * (n_lags + 1)`, on a path with **no budget of any kind** | **06-15 T3** |

These are not stubs in the "unwired UI" sense — they are honestly-dispositioned open axes that the plan explicitly scoped out, and the mechanism holding them open (`no_cost_axis_is_pending`) is the plan's own deliverable D7.

## Threat Flags

None. No new network endpoint, auth path, file-access pattern or schema change at a trust boundary was introduced. The plan **narrows** the existing trust boundary (T-06-31 mitigated, T-06-32 structurally mitigated, T-06-33 mitigated). No package-manager install ran, so `RESEARCH.md`'s Package Legitimacy Audit was not engaged.

## TDD Gate Compliance

Plan task 2 carried `tdd="true"`. All gates present and in order:

| Gate | Commit | Evidence |
|---|---|---|
| **RED** | `258d77f9a` `test(06-14)` | `RED_EVIDENCE_OK` / `target_test_failed`, exit 101, target `types::tests::cost_bounds_match_contract` failing on `contract forecast-tool-boundary-v1 must define constants.poisson_normal_branch_lambda` — an assertion about the planned behaviour, with 8 of 10 cases passing (not a discovery, syntax or fixture error) |
| **GREEN** | `3184b1402` `feat(06-14)` | `types::tests -- --skip no_cost_axis_is_pending` → 9 passed, 0 failed |
| **REFACTOR** | `e1b441944` `refactor(06-14)` | The `changepoint_count` extraction, committed separately and proven behaviour-preserving by `prophet::parity` 32/0 before and after |

No violations.

## Next Phase Readiness

**06-15 is not optional and must land before this branch is pushed.** It owes: `MAX_HOLIDAY_NAME_LEN` + `constants.fit_max_holiday_name_len` (C-07, T2), the NP-training cost bound and its key (C-08, T3), the WR-03 aggregate-dates refusal moved into the holiday loop (T1), and the two `equations` entries `holiday_name_cost_bounded` / `np_train_cost_bounded`. Replacing both `unbounded_pending_06_15` markers with real bounds is what turns `no_cost_axis_is_pending` green and unblocks `make tier3` and CI `workspace-test`.

Everything 06-15/06-16/06-17 needs is in place: `door_surface.cost_axes` already carries C-07 and C-08 with their formulas, their spend sites and their owed-by plans; `test_support::constant_f64` exists for real-valued bounds; `every_cost_axis_names_a_real_bound` will accept the new keys the moment they appear in `constants:`; and `LOGISTIC_BENCH_FREQ` is on the path 06-16 folds into its `sc1_wall` sweep.

**Concern for 06-16:** the `awk … (v + 0 < 2.0)` numeric coercion this plan used in its wall gate is the IN-01 pattern 06-16 T2 is chartered to replace with a shared numeric-shape validator at all five bar sites. This plan's gate does perform a shape check before the comparison, but it is a sixth site written in the same style, and 06-16 should fold it in.

---
*Phase: 06-native-time-series-forecasting-stack*
*Completed: 2026-09-07*

## Self-Check: PASSED

All 7 modified source/contract files exist on disk; all 5 task commits resolve in `git log`.
`commits: 5` is MEASURED (`git rev-list --count 05497fdd45d17efb9973444262d8c0ec9147f9ec..HEAD`),
not narrated. `actuals.tokens: 21474` is chars/4 over the realized diff (85 897 chars) — the plan
estimated 78 000, a 3.6x over-estimate recorded unrounded so it calibrates future estimates
honestly rather than flatteringly.
