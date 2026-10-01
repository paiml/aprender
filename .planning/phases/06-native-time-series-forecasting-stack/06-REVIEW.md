---
phase: 06
phase_name: native-time-series-forecasting-stack
reviewed: 2026-09-07T19:10:00Z
depth: standard
scope: incremental
diff_base: e1b441944
supersedes: 06-REVIEW.md @ ba383f6c1 (CR-01, WR-01..WR-04, IN-01..IN-04)
files_reviewed: 17
files_reviewed_list:
  - contracts/aprender/binding.yaml
  - contracts/forecast-tool-boundary-v1.yaml
  - contracts/prophet-parity-v1.yaml
  - crates/aprender-forecast/README.md
  - crates/aprender-forecast/src/bolt.rs
  - crates/aprender-forecast/src/forecast.rs
  - crates/aprender-forecast/src/lib.rs
  - crates/aprender-forecast/src/np.rs
  - crates/aprender-forecast/src/prophet.rs
  - crates/aprender-forecast/src/sc1_wall.rs
  - crates/aprender-forecast/src/test_support.rs
  - crates/aprender-forecast/src/types.rs
  - crates/aprender-mcp-forecast/src/lib.rs
  - justfile
  - Makefile
  - scripts/assert_measurement_under.sh
  - scripts/check_assert_measurement_under_cases.sh
findings:
  critical: 1
  warning: 9
  info: 4
  total: 14
critical: 1
warning: 9
info: 4
status: issues_found
---

# Phase 06 — Incremental Code Review (gap-closure round 3: plans 06-14 … 06-17)

**Reviewed:** 2026-09-07
**Depth:** standard, incremental over `e1b441944^..f1cb0dc13` (30 commits, +3 790 / −233 across 17 files)
**Status:** issues_found — **1 Critical, 9 Warning, 4 Info**

## Summary

The four things the brief asked me to attack hardest hold up in three cases out of four, and
I checked them by running the code rather than by reading the summaries:

- **The five bar sites really are converted, and `assert_measurement_under.sh` really does
  reject what `v + 0` let through.** I drove it with twelve probes outside its own table
  (`nan`, a trailing space, an embedded newline, `1.2.3` in the BAR slot, `0000000`,
  `999999999999999999999999999`): every malformed token is refused with a distinct message,
  and the 23-row table runs 23/23. `bashrs lint` is clean on both scripts (0 errors,
  0 warnings). Neither script is sourced, so the option-neutrality rule does not apply.
- **The door bounds agree in all three places.** `cost_bounds_match_contract`,
  `bounds_match_contract`, `pool_default_matches_contract` and `chronos_bounds_match_contract`
  pass; `forecast.rs` imports the constants from `types.rs` rather than re-declaring them, so
  there is no third copy to drift.
- **The lambda bound is computed the way `predict` computes it.** The door's
  `t_scale_days` / `future_span_days` / `changepoint_count(ds.len(), &spec)` reproduce
  `d.t_scale_days`, `t[n-1]` and `d.changepoints_t.len()` exactly; `changepoint_geometry` is
  the single arithmetic site and `changepoint_count` returns the LENGTH (`n_cp.max(1)`), which
  is what `make_design`'s `vec![0.0]` branch actually produces. I ran the accepted near-miss
  geometry across 300 seeds: no panic, no non-finite band value.
- **The sampler's three bars are real bars.** I re-derived the sigma arithmetic in the
  contract independently: relative SE of the sample variance is `sqrt(1/λ + 2)/sqrt(N)`
  (0.624 % at λ=3, N=60 000 → 8.02σ at the 0.05 bar), and the zero-mass window
  `[0.1983, 0.2478]` at N=60 000 is correct, as is the claim that it is empty at N=20 000.
  `checked_zero_mass == 2` is a genuine non-vacuity assertion.
- **`every_request_knob_is_enumerated` is genuinely two-directional** for the structs it is
  given, and `no_cost_axis_is_pending` went red-then-green across 06-14/06-15 as claimed.

What is wrong is the fourth item and the shape of the class closure.

**06-15's `MAX_NP_TRAIN_COST` shipped without anyone computing what it refuses.** Every number
in its derivation is at or above the bound — the structural maximum (718 M), a rejected 20 M
candidate, three at-the-bound compositions, and one parity geometry at 97 % of it. Nobody asked
what an ordinary request costs. I asked, through the door, on release: **20 000 daily points
with `n_lags: 7` is refused** (CR-01), as is 10 000 × 14 and 5 000 × 30 — while 2 905 × 30 is
accepted and completes in 1.3 s, well under the 2 s bar the bound cites as its derivation.

And the class closure is half a closure. `door_surface.knobs` is derived from
`schemars::schema_for!` and checked in both directions — that half is real. `cost_axes` is
hand-kept, and **nothing derives it from code**: `every_cost_axis_names_a_real_bound` and
`no_cost_axis_is_pending` both read only the contract, so a new cost axis is exactly as
invisible as C-06 was (WR-03). CR-01 from the previous round would not have been caught by
either test — `growth`, `freq`, `horizon` and `ds` all had knobs entries the whole time. The
contract's own claim that "MISSING (a field exists with no entry) is how CR-01's cost axis got
in" is not what happened.

Two other things this round asserts are checkably false: that neither transport caps the
request body (WR-02 — pmcp 2.19.3 enforces 4 MiB and I read the enforcement), and that the
`feature_row` / `safetensors::load` changes are "breaking for external callers" of a crate
whose manifest says `publish = false` (WR-09).

---

## Critical Issues

### CR-01: `MAX_NP_TRAIN_COST` refuses in-spec NeuralProphet requests that clear the 2 s bar it was derived from — including 20 000 points at `n_lags: 7`

**File:** `crates/aprender-forecast/src/types.rs:258` (`MAX_NP_TRAIN_COST = 15_000_000`),
enforced at `crates/aprender-forecast/src/forecast.rs:452-464`;
`contracts/forecast-tool-boundary-v1.yaml:170` (`fit_max_np_train_cost`)

**Issue.**
The bound was derived from four kinds of evidence and every one of them sits **at or above**
the bound: the structural maximum (718 641 000 → 47.924 s), a rejected 20 000 000 candidate
(19 991 000 → 2.089 s), three `at_bound_*` compositions (13.9 M–14.8 M → 1.402–1.642 s), and
`np::parity`'s Peyton geometry at 14 552 640 — **97.0 % of the bound**. No evidence was
gathered below the bound, so the *accepted region* was never characterised, and it is not
recorded in `MAX_NP_TRAIN_COST`'s doc, in `np_train_cost_bounded`'s invariants, or in any test.
The always-run positive control is `np_args(120, 7)` with `n_lags = 7` — 120 points, which
prices at 0.06 M, four hundred times inside the bound.

Measured through the door (`aprender_forecast::forecast`), release build, `freq: "D"`,
contiguous daily history from 2000-01-01. Accepted rows report their wall; refused rows report
the cost the door printed:

| points | `n_lags` 0 | 1 | 7 | 14 | 30 | 60 |
|---|---|---|---|---|---|---|
| 365 | ok 0.1 s | ok | ok | ok | ok | ok 0.5 s |
| 1 095 | ok | ok | ok | ok | ok | ok 1.3 s |
| 2 905 | ok | ok | ok | ok | **ok 1.3 s** (14 552 640) | **REFUSED** 27 767 200 |
| 5 000 | ok | ok | ok 0.7 s | ok 1.1 s | **REFUSED** 21 569 800 | REFUSED 42 187 600 |
| 10 000 | ok | ok | ok 1.3 s | **REFUSED** 17 974 800 | REFUSED 37 088 400 | REFUSED 72 760 800 |
| 20 000 | ok 0.5 s | ok 1.0 s | **REFUSED** 15 994 400 | REFUSED 29 979 000 | REFUSED 61 907 000 | REFUSED 121 634 000 |

Three things this makes concrete:

1. **The advertised knob is unreachable.** `ForecastArgs.n_lags` is documented as
   "NeuralProphet only: number of autoregressive lags" and range-checked at `n_lags <= 365`
   (`forecast.rs:427-429`); `door_surface.knobs` records that range. At the advertised maximum
   history (`fit_max_points: 20000`) the reachable range is `n_lags <= 6`. A week of AR lags on
   a long daily series is the single most idiomatic NeuralProphet configuration and it is
   refused by **6.6 %** (15 994 400 against 15 000 000).
2. **The refusals are not required by the bar the bound cites.** The refused 20 000 × 7 request
   is 10 % cheaper by the proxy than the 17.97 M 10 000 × 14 request, and the nearest measured
   neighbours below it complete in 1.0–1.3 s. The plan's own rejected candidate puts the 2 s
   crossing near 19 M–20 M, so 15 M discards roughly a quarter of the region the SC1 bar would
   have allowed, and the discarded band is exactly where real requests live.
3. **The value was chosen to keep one geometry green, not to admit a region.**
   `the_np_parity_ladder_geometry_prices_under_the_train_cost_bound` pins the ladder at 97 % of
   the bound. That test proves the bound does not refuse the ladder; it says nothing about the
   3 % of headroom above it, and 3 % is all there is.

The proxy is also not monotone in wall: `n_lags + 1` multiplies the cost linearly while the
per-sample work at `n_lags = 7` with `ar_layers: vec![32]` is dominated by the 32-wide AR head,
not by the lag count — which is why 20 000 × 1 (4.0 M) walls at 1.0 s while 20 000 × 7 (16.0 M,
4× the proxy) is refused rather than measured.

**Fix.** Two changes, and the first is the one that must land:

```rust
// forecast.rs / np.rs — characterise the ACCEPTED region and pin it, the way
// the_np_parity_ladder_geometry_prices_under_the_train_cost_bound pins the ladder.
#[test]
fn the_advertised_np_geometry_is_accepted_by_the_train_cost_bound() {
    // Pure arithmetic on the door's own pricing functions — costs nothing to run.
    for (points, span, n_lags) in [
        (20_000usize, 20_000usize, 7usize),   // the advertised maximum history, a week of lags
        (10_000, 10_000, 14),
        (5_000, 5_000, 30),
    ] {
        let n_samples = span - n_lags;
        let epochs = np::door_epochs(points, n_samples, n_lags);
        let cost = np::train_cost(n_samples, epochs, n_lags)
            * np::door_lr_sweep(n_lags).len() as u64;
        assert!(
            cost <= MAX_NP_TRAIN_COST,
            "{points} points at n_lags={n_lags} prices at {cost}, which the bound \
             {MAX_NP_TRAIN_COST} REFUSES — the advertised n_lags range is unreachable there"
        );
    }
}
```

Then raise the bound to the value that test and the measured 2 s crossing jointly permit
(the `rejected_candidate_20m` mode measured 2.089 s at 19 991 000, so ~18 000 000 is the
largest round value with measured headroom), **or** narrow the advertised range: if
`n_lags <= 6` at 20 000 points is the intended contract, `knobs.n_lags.enforced_by` and the
`ForecastArgs.n_lags` doc comment must say so, and the refusal message must name the reachable
`n_lags` for the history the caller sent rather than only the four factors. Either way the
accepted region has to be written down and asserted; today it is neither.

---

## Warnings

### WR-01: the post-loop aggregate-dates refusal is unreachable dead code, and three artifacts state that it fires

**File:** `crates/aprender-forecast/src/forecast.rs:283-295`; the in-loop refusal it is dead
behind is at `forecast.rs:236-250`

**Issue.** `holiday_dates_total` is written in exactly one place — `forecast.rs:241`, inside
the holiday loop — and the very next statement (line 242) returns `Err` if the running sum
exceeds `MAX_HOLIDAY_DATES_TOTAL`. There is no `continue` in the loop body. Therefore, on every
path that reaches line 289, `holiday_dates_total <= MAX_HOLIDAY_DATES_TOTAL` holds, and the
post-loop `if holiday_dates_total > MAX_HOLIDAY_DATES_TOTAL` is **never true for any input**.

Its own comment (lines 284-288) concedes only partial unreachability — "Unreachable for a
request whose sum crosses the ceiling mid-loop" — but *every* request whose total crosses the
ceiling crosses it mid-loop, because the loop visits every holiday. The comment then makes a
claim that cannot be exercised: "this one has seen every holiday, so it is the only one that
can honestly report the request's EXACT total." That message is unreachable, and so is the only
justification for keeping the block. `contracts/forecast-tool-boundary-v1.yaml:235`
(`knobs.dates.enforced_by`) and `cost_axes` C-03 repeat the claim, so the contract describes a
refusal the code cannot produce.

The WR-03 position test (`the_aggregate_dates_refusal_fires_inside_the_holiday_loop`,
`forecast.rs:831-878`) is correct and does discriminate position — it asserts on the message,
not on the constant. It simply cannot see that the second block is dead, because the message it
requires (`max_holiday_dates_total`) is present in both.

**Fix.** Delete lines 283-295 and correct the two contract claims, **or** — if an exact-total
message is genuinely wanted — make it reachable by hoisting the sum out of the loop:

```rust
// before the loop, so the exact total is known and the partial-sum caveat disappears
let holiday_dates_total: usize = args
    .holidays
    .as_deref()
    .unwrap_or(&[])
    .iter()
    .map(|h| h.dates.len())
    .sum();
if holiday_dates_total > MAX_HOLIDAY_DATES_TOTAL { /* the exact-total refusal, reachable */ }
```
`h.dates.len()` is O(1) per holiday and reads no string, so this is cheaper than the in-loop
accumulation it replaces and preserves WR-03's "refuse before parsing anything" property.

### WR-02: "neither transport caps the request body" is false — pmcp 2.19.3 enforces a 4 MiB limit, and that claim is load-bearing for C-07's disposition

**Files:** `crates/aprender-forecast/src/types.rs:188-194`;
`contracts/forecast-tool-boundary-v1.yaml:303` (C-07 `note`), `:401`
(`holiday_name_cost_bounded` invariant), `:457` (`door_surface_is_complete` invariant);
`contracts/aprender/binding.yaml` (`holiday_name_cost_bounded` notes)

**Issue.** All four places state, as the reason C-07 has *no structural maximum* and could only
be closed by a chosen bound:

> `crates/aprender-mcp-forecast/src/lib.rs`'s router construction (`http_app` / `pooled_app`)
> applies no `DefaultBodyLimit`, no `max_body` and no content-length layer, and the stdio
> transport has no framing cap, so any wall written here would describe an arbitrarily CHOSEN
> name length rather than a maximum.

The first half is wrong, and the mechanism is in the dependency the code configures rather than
in the file that was read. `http_app` (`aprender-mcp-forecast/src/lib.rs:83-88`) builds
`RouterConfig { server_config: StreamableHttpServerConfig::stateless(), .. }`, and in
pmcp 2.19.3 `stateless()` sets `max_request_bytes: DEFAULT_MAX_REQUEST_BYTES` = **4 MiB**
(`pmcp-2.19.3/src/server/streamable_http_server.rs:471`,
`pmcp-2.19.3/src/server/limits.rs:46`). The streamable-HTTP handler enforces it before any JSON
parsing: `read_body_with_limit(body, state.config.max_request_bytes)` →
`axum::body::to_bytes(body, max_bytes)` → HTTP 413
(`streamable_http_server.rs:3565-3577`, `:4571`). pmcp additionally ships
`max_tool_args_bytes` (1 MiB default).

So on the shipped HTTP transport there **is** a structural maximum for `holidays[].name`
(~4 MiB), `measured_at_structural_maximum` **was** an available disposition, and the sentence
"any wall written here would describe an arbitrarily CHOSEN name length" is false. This does not
make the 200-byte bound wrong — it is well derived from the 9-byte committed maximum — but it
does mean a load-bearing justification, recorded in the D-15 source-of-truth contract and
repeated in two Rust doc comments and a binding note, was argued from the absence of a
mechanism in one file rather than proved against the transport (CLAUDE.md rule 2).

**Fix.** Correct all four sites to say what is true: the HTTP transport caps the body at
pmcp's 4 MiB `max_request_bytes` (and tool args at 1 MiB), the stdio transport does not, and
`fit_max_holiday_name_len` is a bound rather than a measurement because the 4 MiB structural
maximum is three orders of magnitude past anything a holiday label needs — not because no
maximum exists. While there, consider asserting the pmcp limit rather than inheriting it:
`constants.max_request_bytes` + `StreamableHttpServerConfig { max_request_bytes: .., .. }`
makes it a value this repository owns instead of a dependency default that can move under it.

### WR-03: `cost_axes` completeness is not machine-checked — the half of the class invariant that would have caught CR-01 is the hand-kept half

**Files:** `crates/aprender-forecast/src/types.rs:596-618` (`enumerated_axes`), `:635-668`
(`every_cost_axis_names_a_real_bound`), `:703-726` (`no_cost_axis_is_pending`);
`contracts/forecast-tool-boundary-v1.yaml:449` (`door_surface_is_complete.formula`)

**Issue.** The knobs half is genuinely structural: `schema_knobs()` derives the field set from
`schemars::schema_for!` — the same generator that produces the advertised tool schema — and
`every_request_knob_is_enumerated` asserts set equality, so neither MISSING nor PHANTOM can
survive. That is real and it is the good half.

The cost-axis half has no such derivation. `enumerated_axes()` reads
`door_surface.cost_axes` out of the YAML and nothing else. Both axis tests iterate that list:

- `every_cost_axis_names_a_real_bound` checks that each listed axis names a key that exists in
  `constants:`.
- `no_cost_axis_is_pending` checks that no listed axis carries an `unbounded_pending_*` marker.

Neither can observe an axis that **is not listed**. There is no `schema_for!` analogue for
"cost axes present in `prophet.rs` / `np.rs` / `forecast.rs`", so a new cost axis is exactly as
invisible today as C-06 was before the previous round. The formula in the contract confirms this
by omission: it states `knobs(door_surface) == properties(...)` — an equality — but only
`forall a in cost_axes(door_surface): a.bound in keys(constants) or ...` — a per-entry
predicate, never a completeness claim.

This is checkable against the round's own motivating example. The contract asserts (`:457`):

> MISSING (a field exists with no entry) is how CR-01's cost axis got in

That is not what happened. CR-01's axis (C-06, `changepoint_count(len(ds)) * (t_max - 1)`) is a
function of `growth`, `freq`, `horizon` and the *spacing* of `ds` — four knobs that all existed
and would all have carried entries. `every_request_knob_is_enumerated` would have been green
before CR-01 and is green after. The test that would have had to catch it is a cost-axis
completeness check, and that is the one test the round did not build.

**Fix.** The honest short-term move is to stop claiming what is not checked: change
`door_surface_is_complete`'s invariant text and `types.rs:513-519`'s module comment to say the
knobs half is derived and the axes half is an inspected list, and record what would make the
axes half structural. Candidates that are actually derivable and worth costing:

- assert that every `cost_axes[].spent_in` names a symbol that exists (the Makefile's
  `contract-audit-phase6` resolver already does file-scoped symbol resolution for
  `binding.yaml`; the same walk over `spent_in` is a small extension), which at least makes an
  axis pointing at deleted code turn red;
- pin the axis **count** with an assertion carrying a written rationale, so adding a bound
  without adding its axis is a deliberate edit rather than an omission;
- require every `constants.fit_max_*` key to be named by at least one `cost_axes[].bound` — the
  inverse direction, which is derivable today and would catch "a bound landed with no axis".

### WR-04: `door_surface` describes one of the two doors this contract bounds; the Chronos door has zero knobs and zero cost axes

**Files:** `crates/aprender-forecast/src/types.rs:542-563` (`schema_knobs`);
`contracts/forecast-tool-boundary-v1.yaml:145` ("THE DOOR'S WHOLE SURFACE"), `:179-243`
(16 knobs, all `owner: ForecastArgs` or `owner: HolidayArg`), `:265-357` (C-01…C-14, all
Prophet/NeuralProphet); `crates/aprender-forecast/src/chronos.rs:127-143` (`ChronosArgs`)

**Issue.** `forecast-tool-boundary-v1.yaml` is the boundary contract for **both** thin servers.
Its `constants:` block carries an `aprender-mcp-chronos` section (`chronos_min_points`,
`chronos_max_points`, `chronos_max_horizon`, `chronos_native_horizon`), it defines
`chronos_server_bounds` (`:462`) and the `allow_long_horizon` gate (`:563`), and
`types::tests::chronos_bounds_match_contract` mirrors three of those constants.

`door_surface` covers none of it. `schema_knobs()` feeds exactly two structs into the set
equality — `ForecastArgs` and `HolidayArg` — so `ChronosArgs`'s five caller-settable fields
(`ds`, `y`, `horizon`, `freq`, `allow_long_horizon`) have no knobs entries and no
`enforced_by` prose, and `cost_axes` has no Chronos axis at all. Nothing in the file scopes
`door_surface` to one server; the header says "THE DOOR'S WHOLE SURFACE" directly under a
`constants:` block containing four `chronos_*` keys, and the axis IDs run C-01…C-14 with no gap
or note marking the Chronos door as out of scope.

The consequence is the same shape as WR-03, one level up: a new `ChronosArgs` field with
`enforced_by: NOTHING` — the exact condition 06-14's enumeration was built to surface, and the
one that found C-07 — is invisible to all three completeness tests.

**Fix.** Either extend `schema_knobs()` with `("ChronosArgs", schema_for!(ChronosArgs))` and add
the five knobs entries plus the Chronos cost axes (the rolling long-horizon path at
`ceil(horizon / chronos_native_horizon)` forward passes is the obvious one), or add an explicit,
tested scoping key — e.g. `door_surface.owner: aprender_forecast::forecast` — and a matching
`chronos_door_surface` block, so the omission is a stated boundary rather than a silent one.

### WR-05: `normal_quantile`'s doc comment and `#[must_use]` were orphaned onto `np_train_cost_is_over`

**File:** `crates/aprender-forecast/src/forecast.rs:576-597`

**Issue.** Plan 06-15 (commit `8a9d9232f`) inserted `np_train_cost_is_over` **between**
`normal_quantile`'s doc block and `normal_quantile` itself. The result compiles, and it is
wrong in three ways:

```rust
/// Acklam's inverse normal CDF (enough for band z-scores).     // 576  ← meant for
/// ... the clamp / interval_width = 0.9999999999999999 story ...//        normal_quantile
#[must_use]                                                      // 584  ← meant for
/// The door's C-08 comparison, named ONCE ...                   // 585     normal_quantile
fn np_train_cost_is_over(cost: u64) -> bool { ... }              // 593  ← receives all three
pub fn normal_quantile(p: f64) -> f64 { ... }                    // 597  ← now undocumented
```

- `np_train_cost_is_over`'s rustdoc now opens with "Acklam's inverse normal CDF (enough for
  band z-scores)" followed by the entire NaN-band explanation, which has nothing to do with it.
- `pub fn normal_quantile` — a public item — has no doc comment at all, and the clamp rationale
  that `the_widest_accepted_interval_still_yields_a_finite_z` exists to protect is no longer
  attached to the function it protects.
- `#[must_use]` moved to the private predicate. (Harmless there; absent where it was intended.)

The doc block also contains `[tests::the_np_train_cost_bound_is_exclusive_not_inclusive]`, an
intra-doc link into a `#[cfg(test)]` module, which does not resolve under `cargo doc`.

**Fix.** Move `np_train_cost_is_over` (and its own doc) above line 576 or below
`normal_quantile`, so each doc block and `#[must_use]` sits on its intended item; change the
intra-doc link to plain backticks since the target is test-only.

### WR-06: the "ONE numeric bar check every wall-clock gate calls" is not on every wall-clock gate, and one comparison is still on the old coercion

**Files:** `scripts/assert_measurement_under.sh:2-3` and `:15`;
`justfile:656` (`chronos-coldstart`), `justfile:740` (`forecast-pool-ratio`)

**Issue.** Three things, and the first is the CLAUDE.md rule 5 one:

1. **`chronos-coldstart` is a wall-clock SC4 gate with a 150 ms bar and it does not use the
   validator.** `justfile:656` is `if [ "$med" -ge 150 ]`. It happens to fail closed today
   because the `sed` at line 651 only emits digits and `[ -ge ]` errors on anything else — but
   that is the accident the validator exists to replace, and it is a fifth wall-clock decision
   surface the round enumerated as four converted plus one new. The script's own header says it
   is "the ONE numeric bar check every wall-clock gate in this repository calls"; that sentence
   is false as written.
2. **`justfile:740` still runs the old coercion.**
   `awk -v a="$ratio" -v b="$best" 'BEGIN { exit (a + 0 > b + 0) ? 0 : 1 }'` is the best-of-three
   *selector*. The `sed` above it accepts `[0-9][0-9.]*`, so `1.2.3` survives extraction, is
   compared as `1.2`, and is then handed to the validator (which correctly refuses it). The gate
   fails closed, but the class the round set out to remove is still present in the file, three
   lines above a comment explaining why it was removed everywhere else.
3. **The header's count is off.** "Five bar sites read a measurement through `awk -v v=...`" —
   four did (`chronos-bench`, `forecast-bench`, `forecast-pool-ratio`,
   `forecast-holiday-bench`); the fifth (`forecast-sc1-sweep`) is new in this round and never
   carried the old form. `check_assert_measurement_under_cases.sh:22` says "the four
   pre-existing bar sites" and is correct, so the two scripts disagree.

**Fix.** Route `chronos-coldstart` through
`bash scripts/assert_measurement_under.sh under "$med" 150 "COLD START (SC4)"` and re-mutate it
*there* (CLAUDE.md rule 4 — the old proof does not transfer); replace the `justfile:740`
selector with a `max` helper that shape-checks both operands, or extract the ratio with a regex
that cannot admit `1.2.3`; correct the two counts in the script header.

### WR-07: `just forecast-sc1-sweep` is called "THE SC1 GATE" but runs on no automatic surface, and the in-suite run it ships asserts no bar

**Files:** `justfile:907`; `crates/aprender-forecast/src/sc1_wall.rs:407-463`; `Makefile`
(no target references it); `.github/workflows/ci.yml:289`

**Issue.** `grep` over `.github/`, `Makefile` and `scripts/` finds **no** reference to
`forecast-sc1-sweep`, `forecast-holiday-bench`, `forecast-bench`, `forecast-pool-ratio` or
`chronos-bench`. `make tier3` gained `contract-audit-phase6` this round but no wall-clock gate.
So the only path that ever executes `assert_measurement_under.sh` — and therefore the only path
that ever runs `check_assert_measurement_under_cases.sh`, whose whole point is that it runs
"on every gate invocation and not only when somebody remembers" — is a human typing
`just forecast-sc1-sweep`.

What *does* run automatically is `sc1_wall::sc1_wall_sweep`, which CI reaches through
`cargo nextest run --profile ci --workspace --lib`. That run is a debug build, so
`sc1_wall.rs:446-450` returns before the 2 s assertion. Its only live assertion is the 120 s CI
budget — explicitly "NOT the SC1 bar" — and `.config/nextest.toml` sets `retries = 2`, so even
that is retried away. Net effect: the SC1 bar this round built a swept gate for is asserted by
nothing that runs without a human.

This is not new to this round — the other four recipes were already manual — but the round
promotes one of them to "THE SC1 GATE" and binds it in `contracts/aprender/binding.yaml` as
`sc1_wall_swept … status: implemented`, which reads as coverage.

**Fix.** Wire `forecast-sc1-sweep` (and `chronos-bench`) into a tier or a scheduled workflow, the
way `toolchain-ceiling.yml` runs the clippy ceiling daily — a release-profile wall-clock sweep
is too slow for per-PR CI but is exactly a nightly job. If it stays manual, say so in the recipe
header and in the binding `notes`, so `status: implemented` is not read as `status: running`.

### WR-08: the swept matrix still hard-codes the points axis; the slowest compositions anyone has measured are outside it

**Files:** `crates/aprender-forecast/src/sc1_wall.rs:409` (`SC1_SWEEP_POINTS`, default 33),
`justfile:907` (`points="33"`); `crates/aprender-forecast/src/sc1_wall.rs:39-51`

**Issue.** WR-04 of the previous round diagnosed three benches each "hard-coded to the geometry
it was born from". The replacement sweeps `freq × growth × holidays` — three axes — and pins
`points` at 33 in both the harness default and the recipe default. The recipe's own header says
"shrinking it for the in-suite run shrinks the POINTS and the HORIZON, never an axis", which is
true of the in-suite run and equally true of the **gate**: the gate ships the same 33.

I ran the sweep. Every Prophet composition is a 33-point history, and the worst is 0.890 s:

```
SC1 WALL: model=prophet freq=W growth=logistic holidays=at_bound points=33 horizon=3650 ... total_s=0.890
SC1 WALL: model=prophet freq=MS growth=logistic holidays=at_bound points=33 horizon=841  ... total_s=0.703
SC1 SWEEP: compositions=19 elapsed_s=7.252 profile=debug
```

The two slowest holiday compositions on record are outside this matrix: the
`forecast-holiday-bench` default (800 points × 50 columns, **1.692 s**) and the accepted
4 700-point / 5-column request that `types.rs:79` records at **4.2 s**. At `points = 33` the
holiday rows resolve to `columns = 50000/3683 = 13`, i.e. a many-horizon-rows / few-columns
shape; the many-history-rows shape — the one that produced both slow numbers — is never swept.

`forecast-holiday-bench` still covers 800 points and `forecast-bench` covers 3 000, so the
coverage exists across three recipes. But the sweep is the one billed as the gate and the one
bound in the contract, and its shipped defaults do not reach the region where the wall is known
to be worst.

**Fix.** Make `points` an axis of the gate: sweep `{33, 800, 3000}` in the recipe (the in-suite
run can keep 33 for the budget), or set the recipe default to the composition
`forecast-holiday-bench` measured slowest and keep 33 only as the in-suite shape check. Update
the module header, which currently argues the default matrix "is already at the bound" on the
strength of the design-cost bound alone — the design-cost bound is on *cells*, and cells are the
statistic `types.rs:76-81` says does not predict wall.

### WR-09: the new README "Public API changes" section describes breakage that cannot exist — the crate is `publish = false`

**Files:** `crates/aprender-forecast/README.md:99-113`;
`crates/aprender-forecast/Cargo.toml:14-15`

**Issue.** The section states that `prophet::feature_row`'s signature change and
`safetensors::load`'s removal are "**breaking for external callers** of this crate" in "the
0.63.0 line", and defers the record to "the repository-root `CHANGELOG.md`".

`crates/aprender-forecast/Cargo.toml:15` is `publish = false`, with the comment "not a published
API yet". `git show v0.63.0:crates/aprender-forecast/Cargo.toml` does not exist — the crate was
created in this phase and has never been released. There are no external callers, `feature_row`
and `safetensors::load` have never been public outside this workspace, and no CHANGELOG entry
was added.

This originates in the previous review's IN-02/IN-04, which asserted "breaking for external
callers of this 0.63.0 crate" without checking the manifest. 06-17 escalated the unverified
claim into a shipped README section instead of refuting it — the failure mode
`superpowers:receiving-code-review` exists to prevent. The cost is not cosmetic: a future
maintainer reading this table will treat `feature_row`'s shape as frozen and route around it.
The same claim is repeated in `bolt.rs:284-285` ("a breaking public-API change to a published
crate") as the reason `transpose` was not widened to `Result` (IN-03).

**Fix.** Replace the section with an accurate one: the crate is unpublished (`publish = false`),
both changes are internal, and the note that matters is the *design* one — build `hol_sets` with
`holiday_day_sets` from the same `Spec`. Keep the index-safety paragraph, which is correct and
useful. Re-open `transpose`'s `Result` question on its merits rather than on a semver constraint
that does not apply.

---

## Info

### IN-01: `holiday_design_wall`'s in-code defaults are a geometry the door refuses, so running it without the recipe panics

**File:** `crates/aprender-forecast/src/prophet.rs:2219-2222`

The harness defaults are `points=3000, columns=181, dates=84, horizon=365` — the verifier's
original geometry, which `justfile:816-818` correctly notes "is now REFUSED at the door". It
prices at `(3000 + 365) × 181 = 609 065` design cells against `MAX_HOLIDAY_DESIGN_COST =
50 000`, so `time_accepted` (`sc1_wall.rs:135-140`) panics with "the composition must be
ACCEPTED by the door". The only non-panicking entry point is `just forecast-holiday-bench`,
whose own defaults (`800/50/84/200`) differ from the code's. Point the code defaults at the
recipe's values so the two cannot drift and the documented `cargo test --release --ignored`
invocation works.

### IN-02: two of the three dispositions `every_cost_axis_names_a_real_bound` validates are reached by no row

**File:** `crates/aprender-forecast/src/types.rs:643-659`

No `cost_axes` entry currently carries `bound: measured_at_structural_maximum` or a
`bound: unbounded_pending_*` marker (C-08 carries `measured_seconds: 47.924` but its `bound:` is
`fit_max_np_train_cost`, so the `measured_seconds < 2.0` branch does not run for it). Both
branches are therefore dead in the shipped tree: the `measured_seconds`-under-2.0 check and the
pending skip have never executed against real data and will first run on whatever future axis
uses them. Worth one synthetic row in a `#[test]` that drives `enumerated_axes`-shaped input
through the same predicate, so the dispositions the contract advertises are exercised before
someone relies on them.

### IN-03: three checkable claims in `bolt::transpose`'s new comment are off

**File:** `crates/aprender-forecast/src/bolt.rs:275-291`

The `debug_assert_eq!` itself is a real improvement over the prose-only `expect`. Its comment
says: (a) "all thirteen in-crate call sites" — `grep` finds **11** (`bolt.rs` only; nothing in
`chronos.rs` or `aprender-mcp-chronos`); (b) "a breaking public-API change to a **published**
crate" — `publish = false`, see WR-09; (c) "a caller that violates the invariant is named at
ITS OWN call site with both numbers" — `debug_assert_eq!` panics *inside* `transpose` and the
panic location is `bolt.rs:287`; only a backtrace names the caller. Also note the assert sits
after `let mut t = vec![0.0f32; w.len()]`, so the allocation happens first. Trim the claims to
what the code does.

### IN-04: the sweep prints a lambda on rows that never pay it, 4.3× over the door's own bound

**File:** `crates/aprender-forecast/src/sc1_wall.rs:212-216`, `:342-353`

`max_legal_horizon` clamps the horizon only when `growth == "logistic"`, but `measure` prints
`lambda=` unconditionally. The gate's primary artifact therefore contains lines like:

```
SC1 WALL: model=prophet freq=MS growth=linear holidays=none points=33 horizon=3650 lambda=86789.8 ... total_s=0.187
```

86 789.8 is the structural maximum of the axis and 4.3× `MAX_LOGISTIC_CHANGEPOINT_LAMBDA`, on a
line the gate reports as passing. It is correct — the linear arm never calls `poisson` — but a
number that large next to `OK` on the gate's own log is the sentence a future reader will quote
out of context. Print `lambda=n_a` on the non-logistic rows, or suffix it `lambda=86789.8(unpaid)`.

---

## Verification notes

Everything below was run against this working tree; no repository source was modified.

- `cargo test -p aprender-forecast --lib -- types:: forecast::tests sampler::` →
  **40 passed, 0 failed**.
- `cargo test -p aprender-forecast --lib -- sc1_wall:: --nocapture` → 19 compositions,
  `elapsed_s=7.252`, `profile=debug` (the module header's 7.19 s reproduces).
- `bash scripts/check_assert_measurement_under_cases.sh` → `TABLE OK: 23/23`.
- `bashrs lint` on both new scripts → **0 errors, 0 warnings** (3 and 7 info respectively).
- `pv validate` on `contracts/forecast-tool-boundary-v1.yaml` and
  `contracts/prophet-parity-v1.yaml` → `0 error(s), 0 warning(s)` each.
- `awk 'NR>1610' contracts/aprender/binding.yaml | grep -c '^- contract:'` → **63**, split
  23 / 18 / 13 / 9 exactly as the header claims.
- **CR-01** was measured through `aprender_forecast::forecast` on a release build from a
  throwaway crate outside the repository that depends on `crates/aprender-forecast` by path,
  driving the (points × n_lags) grid in the table. Refusal costs are the door's own printed
  numbers; accepted walls are `Instant::now()` around the call at `horizon: 1`.
- **WR-02** was verified by reading pmcp 2.19.3 in the cargo registry:
  `src/server/limits.rs:46` (`DEFAULT_MAX_REQUEST_BYTES = 4 * 1024 * 1024`),
  `src/server/streamable_http_server.rs:471` (`stateless()` sets it) and `:3565-3577`, `:4571`
  (`read_body_with_limit` → `axum::body::to_bytes` → 413).
- **WR-01**'s unreachability is by inspection and is total: `holiday_dates_total` has exactly
  one write site (`forecast.rs:241`) and is compared to the ceiling on the next statement, with
  no `continue` in the loop.
- I probed the accepted logistic near-miss geometry (33 points, `freq: "MS"`, `horizon: 840`,
  `growth: "logistic"`) across 300 seeds looking for a `partial_cmp(...).expect("f")` panic or a
  non-finite band from `logistic_gammas` at the now-27×-larger simulated changepoint count:
  **0 panics, 0 non-finite values**. Recording the negative so it is not re-litigated.

---

_Reviewed: 2026-09-07_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard, incremental over `e1b441944^..f1cb0dc13`_
