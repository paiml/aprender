---
phase: 06-native-time-series-forecasting-stack
plan: 06
subsystem: api
tags: [mcp, pmcp, axum, tower, streamable-http, stdio, concurrency, router-pool, contracts, pv, validation, rust]

requires:
  - phase: 06-native-time-series-forecasting-stack
    provides: "plan 06-01 landed crates/aprender-mcp-forecast (build_server, http_app, the in-process e2e Client harness), the strict parse_date, test_support::constant_u64 and the measured opt-level 3 profile"
  - phase: 06-native-time-series-forecasting-stack
    provides: "plan 06-04 filled the real `neuralprophet` match arm, so Task 2's NP happy path has something to dispatch to"
  - phase: 06-native-time-series-forecasting-stack
    provides: "plan 06-05 landed crates/aprender-forecast/src/chronos.rs with CHRONOS_MIN_POINTS / CHRONOS_MAX_POINTS / CHRONOS_MAX_HORIZON and the weights-free validate() seam the boundary contract pins"
  - phase: 06-native-time-series-forecasting-stack
    provides: "plan 06-03 established the contract shape (kind: kernel on the setfit-apr-v1 template) and the contract-read bar pattern with the contract name written in full at every call site"
  - phase: 04-setfit-mcp
    provides: "crates/aprender-mcp-setfit/tests/e2e_stdio.rs — the spawned-binary stdio harness copied whole here"
provides:
  - "contracts/forecast-tool-boundary-v1.yaml — the ONE contract both forecasting servers' bounds, refusal rules and shared response shape are held to (16 equations, 16 obligations, 16 FALSIFY-BOUNDARY tests, 3 declared Kani harnesses, qa_gate F-FORECAST-BOUNDARY-001)"
  - "types::tests::bounds_match_contract / chronos_bounds_match_contract — six Rust constants asserted EQUAL to the contract, read at test time"
  - "21 e2e::refuses_* cases + 4 happy paths driven through a live streamable-HTTP server (SC1)"
  - "aprender_mcp_forecast::pooled_app(K, name, version) — K independent pmcp routers behind a round-robin fallback (D-12)"
  - "CLI flag --http PORT [--pool K], default 8, usage error -> exit 2"
  - "pool_equality: 8/8 and 16/16 bit-identical responses under load (SC5 correctness half) and the machine-parsable POOL SPEEDUP line 06-08 greps"
  - "crates/aprender-mcp-forecast/tests/e2e_stdio.rs — the spawned-binary stdio round trip, ungated, dark in CI pending 06-08"
  - "The MEASURED pool control pair: 1.002x with one router vs 2.070x with eight, same host, same requests"
affects: [06-07, 06-08, 06-09]

actuals:
  tokens: 26525
  tasks: 3
  commits: 4

tech-stack:
  added: []
  patterns:
    - "A contract claim is SPLIT BY KIND when its halves are falsifiable by different instruments: a correctness claim gets a unit test in CI, a benchmark claim gets a host-gated recipe. Mixing them makes the correctness suite flaky and the benchmark unenforced."
    - "A refusal test asserts the class MARKER the SDK actually ships, not the class the API name implies — read the wire, then pin what is there."
    - "A speed-up is reported with a CONTROL, never alone: the same test with the mechanism disabled is what turns a number into evidence."
    - "One #[tokio::test] per refusal case: a failing case names ONE cause because it perturbs the valid argument set by exactly one field."

key-files:
  created:
    - contracts/forecast-tool-boundary-v1.yaml
    - crates/aprender-mcp-forecast/tests/e2e_stdio.rs
  modified:
    - crates/aprender-forecast/src/types.rs
    - crates/aprender-mcp-forecast/src/lib.rs
    - crates/aprender-mcp-forecast/src/main.rs
    - crates/aprender-mcp-forecast/README.md
    - README.md
    - .planning/phases/06-native-time-series-forecasting-stack/deferred-items.md

key-decisions:
  - "The validation error CODE was read off the wire, not assumed: pmcp 2.19.3 emits -32603 (the JSON-RPC INTERNAL error code) for Error::validation, because error_code() returns None for both Validation and Internal. The real class discriminator is the thiserror-rendered message prefix, so the suite pins BOTH -32603 and \"Validation error: \". The contract was corrected to say this rather than the code-only claim it was drafted with."
  - "REVIEW-06-04 implemented exactly as specified: zero timing assertions and zero cfg!(target_arch) gates inside mod pool_equality. The ratio is printed once, with arch/profile/workers/cpus provenance, and `just forecast-pool-ratio` (06-08) owns the >= 2.0 bar."
  - "The pool's effect was proven with a CONTROL run, not asserted: POOL=1 measured 1.002x and POOL=8 measured 2.070x on the same host with the same eight requests. Equality (8/8, max|d yhat| 0.0) held in BOTH configurations, which is what separates the correctness claim from the speed claim."
  - "Refusal cases use a 60-point synthetic series rather than Peyton Manning: every refusal fires at the door before any design is built, so the series only has to be valid in the dimensions the case is not perturbing, and each request stays small."
  - "clippy gated with --no-deps (the D-ITEM-06-01-b pre-existing condition), engagement re-proven in THIS crate with a needless_bool mutation rather than inherited from 06-01."

patterns-established:
  - "Contract-read bounds: the contract is the source and the Rust constant is the mirror, proven by editing ONLY the YAML and watching the named test go red"
  - "Assertion-engagement proof for a test suite: break the production mapping (Validation -> internal) and count how many cases go red; the ones that stay green tell you which path they actually exercise"
  - "A pooled service's equality claim is asserted every run; its throughput claim is measured every run and asserted only on a gated host"

requirements-completed: [SC1, SC5]

coverage:
  - id: D1
    description: "contracts/forecast-tool-boundary-v1.yaml pins BOTH servers' bounds, the date/freq/interval/cap/constant-y rules, the strict schema, the shared response shape, the pool claims and the per-request seed; it validates with 0 errors and every pv status count is non-zero"
    requirement: SC1
    verification:
      - kind: other
        ref: "pv validate contracts/forecast-tool-boundary-v1.yaml -> `0 error(s), 0 warning(s)` / `Contract is valid.`"
        status: pass
      - kind: other
        ref: "pv status -> References 5, Equations 16, Proof obligations 16, Falsification tests 16, Kani harnesses 3, QA gate F-FORECAST-BOUNDARY-001"
        status: pass
    human_judgment: false
  - id: D2
    description: "The fit server's three bounds and the Chronos door's three bounds are asserted EQUAL to the contract constants, read at test time"
    requirement: SC1
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/types.rs#tests::bounds_match_contract"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/types.rs#tests::chronos_bounds_match_contract"
        status: pass
      - kind: other
        ref: "induced RED from the YAML alone: fit_max_horizon 3650->3651 and chronos_max_horizon 1024->1025 each turn their test red quoting the value it read; byte-identical revert restores 2 passed / 0 failed"
        status: pass
    human_judgment: false
  - id: D3
    description: "Every malformed input SC1 lists is refused through a live streamable-HTTP server as a validation-class error naming the fix — 21 cases, including the four REVIEW-06-U2 malformed-date shapes"
    requirement: SC1
    verification:
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::refuses_* (21 cases; `cargo test -p aprender-mcp-forecast --lib e2e::` -> 25 passed, 0 failed, 0 ignored)"
        status: pass
      - kind: other
        ref: "engagement proof: routing ForecastError::Validation to pmcp::Error::internal turns 20 of 21 red quoting `got: Internal error: ...`"
        status: pass
    human_judgment: false
  - id: D4
    description: "The advertised tool schema has additionalProperties: false and requires exactly ds, y, horizon, with freq/model/n_lags/seed/holidays advertised as optional"
    requirement: SC1
    verification:
      - kind: unit
        ref: "crates/aprender-mcp-forecast/src/lib.rs#tests::the_tool_schema_is_strict_and_requires_ds_y_horizon"
        status: pass
    human_judgment: false
  - id: D5
    description: "The shared response shape holds across both models and across MS+multiplicative and logistic+holidays configurations"
    requirement: SC1
    verification:
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::neuralprophet_lag_free_happy_path"
        status: pass
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::prophet_ms_multiplicative_happy_path"
        status: pass
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::prophet_logistic_with_holidays_happy_path"
        status: pass
    human_judgment: false
  - id: D6
    description: "Eight (and sixteen) concurrent requests against the pooled streamable-HTTP app return responses bit-identical to their sequential results"
    requirement: SC5
    verification:
      - kind: unit
        ref: "crates/aprender-mcp-forecast/src/lib.rs#pool_equality::eight_concurrent_requests_are_bit_identical_to_sequential (8/8, max|d yhat| 0.0)"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-forecast/src/lib.rs#pool_equality::sixteen_concurrent_neuralprophet_fits_stay_identical (16/16, 0 errors)"
        status: pass
    human_judgment: false
  - id: D7
    description: "--pool K pools K pmcp routers behind a round-robin fallback (default 8) and the stdio path is unaffected"
    requirement: SC5
    verification:
      - kind: unit
        ref: "crates/aprender-mcp-forecast/src/main.rs#tests (3 tests: defaults, either order, usage errors refuse rather than default)"
        status: pass
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/tests/e2e_stdio.rs#the_thin_server_forecasts_over_live_stdio (stdio still default, unaffected by the pool)"
        status: pass
    human_judgment: false
  - id: D8
    description: "The pool actually engages: the measured speed-up is 1.002x with one router and 2.070x with eight, same host, same requests"
    requirement: SC5
    verification:
      - kind: other
        ref: "control pair, POOL const flipped 8->1->8: `POOL SPEEDUP: 1.002x seq=20041ms conc=19997ms` vs `POOL SPEEDUP: 2.070x seq=20256ms conc=9784ms`, both arch=aarch64 profile=debug workers=4 cpus=14"
        status: pass
    human_judgment: false
  - id: D9
    description: "The >= 2.0 speed-up BAR itself is gated to a release aarch64 host and is not asserted here (REVIEW-06-04)"
    requirement: SC5
    verification: []
    human_judgment: true
    rationale: "By design this plan measures and prints the ratio and asserts nothing about it; `just forecast-pool-ratio` in plan 06-08 is what applies the >= 2.0 best-of-3 bar. Until that recipe exists, no automated gate holds the speed-up claim — a human should confirm at 06-08 that the recipe greps the exact POOL SPEEDUP shape this plan emits."
  - id: D10
    description: "The spawned binary answers initialize -> tools/list (one tool, strict schema) -> tools/call over stdio and refuses an unknown key by name"
    requirement: SC1
    verification:
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/tests/e2e_stdio.rs#the_thin_server_forecasts_over_live_stdio (1 passed, 0 ignored)"
        status: pass
    human_judgment: true
    rationale: "The test passes locally and is unconditional, but it is DARK IN CI: .github/workflows/ci.yml's explicit --test target line does not list e2e_stdio, and this plan is forbidden from editing that file. A human must add it at the 06-08 check-in, or the stdio transport stays unguarded on every CI run."

duration: 49min
completed: 2026-09-06
status: complete
---

# Phase 06 Plan 06: Tool Boundary, Refusal Set and Router Pool Summary

**Both forecasting servers are now held to one `pv`-validated boundary contract whose six bounds the Rust constants are asserted EQUAL to; 21 malformed inputs are refused as validation-class errors through a live streamable-HTTP server; and `--pool K` turns eight concurrent fits from a measured 1.002x into a measured 2.070x while every response stays bit-identical to its sequential twin.**

## Performance

- **Duration:** 49 min
- **Started:** 2026-09-06T13:46:19Z
- **Completed:** 2026-09-06T14:35:00Z
- **Tasks:** 3
- **Files created/modified:** 8 (2 created, 6 modified)

## Accomplishments

- **`contracts/forecast-tool-boundary-v1.yaml`** — the one document a client can read to know what *either* forecasting endpoint accepts. 16 equations (both servers' bounds, date validity and the ten-ASCII-byte shape, the closed `D/W/MS` allowlist, the open-unit `interval_width`, logistic-cap, constant-`y`, unknown-field, unknown-enum, the Chronos horizon gate and its warning, the shared response shape, refusal classification, both pool claims, the per-request seed), 16 proof obligations, 16 `FALSIFY-BOUNDARY` tests, 3 declared-not-executed Kani harnesses, `qa_gate: F-FORECAST-BOUNDARY-001`. `pv validate`: **0 errors, 0 warnings**.
- **Six bounds turned from documentation into assertions.** `types::tests::bounds_match_contract` and `chronos_bounds_match_contract` read `constants.*` out of the YAML at test time via `test_support::constant_u64`, with the contract name written in full at every call site.
- **21 `e2e::refuses_*` cases** (the plan asked for ≥ 18), each its own `#[tokio::test]` against a live in-process streamable-HTTP server, each perturbing a valid argument set by exactly one field. Includes the four REVIEW-06-U2 malformed-date shapes — trailing content (`"2008-02-01garbage"`, the exact string the spike's prefix-take used to accept), empty, leading whitespace, full-width non-ASCII digits.
- **Three new happy paths** (four total with the 06-01 tracer): the first e2e proof that `model: "neuralprophet"` dispatches over the wire since 06-04 replaced 06-01's refusing stub; Prophet with `freq: MS` + multiplicative seasonality on Air Passengers (every future `ds` lands on a month start); Prophet with logistic growth, a valid cap and a named `superbowl` holiday that becomes a named component.
- **`pooled_app` and `--pool K`.** K independent pmcp routers behind an `axum` fallback dispatching round-robin on an `AtomicUsize` via `tower::ServiceExt::oneshot`. Default 8, `--pool 1` restores pre-pool behaviour, a usage error exits 2 rather than defaulting.
- **Equality under load, and a control for the speed-up.** 8/8 and 16/16 bit-identical signatures with `max|Δ yhat|` exactly `0.0` — asserted. The ratio — printed, never asserted — with a control run that turns it into evidence.
- **`tests/e2e_stdio.rs`** — the SetFit harness against the real spawned binary, with the environment gate *removed* because the fit server needs no model. It always runs; nothing about it can silently report "0 ignored" while proving nothing.

## Task Commits

1. **Task 1: the boundary contract + the six contract-read bound assertions** — `5de8b7728` (test)
2. **Task 2: the full D-11 refusal set, strict schema, three happy paths** — `d0a63573a` (test)
3. **Task 3: router pool, `--pool K`, equality under load, spawned-binary stdio e2e** — `c5608982e` (feat)
4. **`--pool` sizing docs (T-06-02) + two stale README claims corrected** — `14eceff67` (docs)

## The measurement the plan asked for — reported with a control

The plan's success criterion was that the pool's effect be **measured and reported**, not asserted. It is, and the number that makes it evidence is the control rather than the headline.

Same host, same test, same eight requests, same profile. **Only the `POOL` constant changed:**

| Configuration | Printed line | Reading |
|---|---|---|
| `POOL = 1` (one router) | `POOL SPEEDUP: 1.002x seq=20041ms conc=19997ms arch=aarch64 profile=debug workers=4 cpus=14` | Eight concurrent fits take **exactly as long as running them one after another** — no concurrency at all |
| `POOL = 8` (the default) | `POOL SPEEDUP: 2.070x seq=20256ms conc=9784ms arch=aarch64 profile=debug workers=4 cpus=14` | The same eight fits finish in **less than half** the sequential wall |

The `1.002x` row is the point. Without it, `2.070x` is a number that could have come from anything; with it, the router mutex is demonstrably what the pool removes, in-tree, and spike 010's `1.0x → 3.9x` finding is reproduced here rather than quoted. **Equality held in BOTH configurations** (8/8 identical, `max|Δ yhat| = 0.0`), which is exactly why the two claims are separated: the correctness claim is independent of whether the pool helps.

The line committed to `main` is the verbatim `POOL = 8` run in the plan's own `<verify>` block:

```
POOL SPEEDUP: 2.097x seq=20179ms conc=9622ms arch=aarch64 profile=debug workers=4 cpus=14
```

**The ratio bar itself is asserted only by `just forecast-pool-ratio` in plan 06-08** (REVIEW-06-04). There is no `assert!(speedup ...)` and no `cfg!(target_arch)` gate anywhere in `mod pool_equality` — verified mechanically:
`sed -n '/^mod pool_equality/,$p' … | grep -v '^\s*//' | grep -c 'assert!(speedup'` → **0**, over a 260-line region, with **0** `target_arch = "aarch64"` occurrences.

Three notes for 06-08, all measured here:

- **This host is `arch=aarch64` but `profile=debug` and `cpus=14`.** 2.07x on debug already clears the 2.0 bar, but 06-08's release build is the one the bar was written for.
- **Sibling-test contention is negligible.** Running `pool_equality` (both tests, libtest's default parallelism) printed `2.037x`; running the eight-wide test *alone* printed `2.070x`. `--test-threads=1` is therefore not required for the recipe, though it costs nothing.
- **The line format is the contract.** `just forecast-pool-ratio` must grep `^POOL SPEEDUP: [0-9]+\.[0-9]{3}x seq=[0-9]+ms conc=[0-9]+ms arch=[a-z0-9_]+ profile=(debug|release) workers=4 cpus=[0-9]+$`. Exactly one such line is emitted per run — the sixteen-wide stress test deliberately prints none.

## The pinned validation error code — read off the wire, and not what it looks like

The plan said to read the validation error code off one failing reply and pin it. Doing so produced a finding worth stating plainly, because pinning the number alone would have shipped a test that proves nothing.

A live refusal comes back as:

```json
{"jsonrpc":"2.0","id":2,"error":{"code":-32603,"message":"Validation error: need at least 10 points, got 5"}}
```

`-32603` is the JSON-RPC **internal error** code. In pmcp 2.19.3, `Error::error_code()` returns `None` for both the `Validation` and the `Internal` variant (only `Protocol` carries an explicit code), so the transport falls back to `-32603` for *either*. **The numeric code cannot distinguish a caller's fault from a server's fault on this SDK version.** What can is the message prefix `thiserror` renders from the variant: `"Validation error: "` versus `"Internal error: "`.

So `refused()` pins **both**, and the contract's `refusals_are_validation_errors` invariant and `FALSIFY-BOUNDARY-008` were rewritten to record this rather than the code-only claim they were drafted with (see Deviations, Rule 1 #1).

**Engagement proven, not assumed.** Rewriting `map_error` to route `ForecastError::Validation` to `pmcp::Error::internal` turns **20 of the 21** refusal cases red, each quoting `got: Internal error: …`. The one that stays green is `refuses_unknown_field` — pmcp refuses an unknown key during argument deserialization and renders its own `"Validation error: Invalid arguments for tool 'forecast': unknown field \`bogus\`…"`, never reaching `map_error`. That asymmetry is now documented in the contract too: it tells a future reader exactly which path each case exercises.

## Contract-read bounds: induced RED from the YAML alone

The claim "the contract is the source and the code is the mirror" is only worth making if the code actually reads the contract. Changing **one line of YAML and no Rust**:

| Edit (contract only) | Result |
|---|---|
| `constants.fit_max_horizon: 3650 → 3651` | `types::tests::bounds_match_contract` FAILED — `types::MAX_HORIZON must equal constants.fit_max_horizon in forecast-tool-boundary-v1` |
| `constants.chronos_max_horizon: 1024 → 1025` | `types::tests::chronos_bounds_match_contract` FAILED — same shape, naming `CHRONOS_MAX_HORIZON` |

A byte-identical revert restores `ok. 2 passed; 0 failed` in both cases.

## Files Created/Modified

| File | What it does |
|---|---|
| `contracts/forecast-tool-boundary-v1.yaml` | **NEW.** The boundary both servers are held to: 16 equations, 16 obligations, 16 `FALSIFY-BOUNDARY` tests, 3 Kani declarations, `F-FORECAST-BOUNDARY-001`, and a top-level `constants:` block read at test time |
| `crates/aprender-mcp-forecast/tests/e2e_stdio.rs` | **NEW.** The spawned-binary stdio round trip: `initialize` → `notifications/initialized` → `tools/list` → `tools/call` → unknown-key probe. No environment gate |
| `crates/aprender-forecast/src/types.rs` | `bounds_match_contract` + `chronos_bounds_match_contract` |
| `crates/aprender-mcp-forecast/src/lib.rs` | `pooled_app`; the 21 refusal cases and 3 happy paths in `mod e2e`; `mod tests` (strict schema); `mod pool_equality`; `Client::call_id` and `client_for` so the pool module reuses the harness |
| `crates/aprender-mcp-forecast/src/main.rs` | `parse_http_args` (`--http [PORT] [--pool K]`, either order, refuse-never-default), `serve_http(port, pool)`, a pool-aware stderr banner, 3 parser unit tests |
| `crates/aprender-mcp-forecast/README.md` | The "Sizing `--pool`" section (T-06-02 mitigation); the stale NeuralProphet-stub sentence corrected |
| `README.md` | Contract count 1789 → 1790 |
| `.planning/…/deferred-items.md` | D-ITEM-06-01-a closed; D-ITEM-06-06-a (`.pv/` cache files) logged |

## Decisions Made

**The two pool claims are split by INSTRUMENT, not by strictness.** `pool_equality_under_load` is falsified by a unit test that runs in CI on every host; `pool_speedup_host_gated` is falsified by a benchmark recipe on one gated host. The reason is not that the ratio matters less — it is that a wall-clock ratio measured under four Tokio workers and eight heterogeneous blocking fits moves with CPU throttling *independently of the router serialisation the pool exists to remove*. A flaky bar inside the correctness suite would train readers to ignore the suite, including the equality failure that would actually matter. Both reviewers reached this independently (REVIEW-06-04); the contract's `qa_gate.checks` list separates them explicitly and marks the last line `HOST-GATED, NOT IN CI`.

**Refusal cases use a 60-point synthetic series, not Peyton Manning.** Every refusal fires at the door before any design is built, so the series only needs to be valid in the dimensions a given case is not perturbing. Keeping requests small keeps 21 cases at ~1.8 s wall for the whole `e2e::` suite.

**One `#[tokio::test]` per case, each perturbing `valid_args()` by exactly one field.** A batched refusal test that loops over cases reports "some refusal was wrong"; these report which one, and nothing else can be the cause.

**`--pool` parsing refuses rather than defaults.** `--pool` with no value, `--pool many`, `not-a-port` and `70000` all exit 2 with a usage line. This mirrors the tool boundary's own D-11 rule at the CLI: a knob you set is never silently ignored.

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 1 — Bug] The contract asserted something false about the validation error code; corrected before the plan finished.**
- **Found during:** Task 2, reading the code off a live reply as the plan instructed.
- **Issue:** Task 1's `refusals_are_validation_errors` invariant and `FALSIFY-BOUNDARY-008` were drafted saying the e2e suite "pins the numeric JSON-RPC error code pmcp emits for `Error::validation`, so no refusal reaches a client as an internal error." Measured, that sentence is false: the code IS the internal-error code (`-32603`) for both variants, so pinning it proves nothing about the class.
- **Fix:** Both were rewritten to record the measured behaviour — the code is `-32603` for either variant, the discriminator is the `thiserror` message prefix, and the suite pins both. `FALSIFY-BOUNDARY-008`'s prediction now also records the OBSERVED red (20 of 21) and names the one case that stays green and why.
- **Files modified:** `contracts/forecast-tool-boundary-v1.yaml`
- **Verification:** `pv validate` still 0 errors; the described mutation reproduced exactly as written.
- **Committed in:** `d0a63573a`

**2. [Rule 1 — Bug] `crates/aprender-mcp-forecast/README.md` still said `model: "neuralprophet"` refuses.**
- **Found during:** Task 3, reading the README while adding the `--pool` section.
- **Issue:** The sentence "`model: "neuralprophet"` currently refuses with a message naming plan 06-04, which ports it" was true when 06-01 wrote it and false since 06-04 landed. This plan's `neuralprophet_lag_free_happy_path` is the e2e proof it is false, so leaving it would have shipped a README contradicted by a test in the same crate.
- **Fix:** Replaced with what the arm does (AR-Net over `n_lags`, `freq: D` only, residual-sd band with `diagnostics.band` saying so), plus a paragraph on refuse-never-default and a link to the boundary contract.
- **Committed in:** `14eceff67`

**3. [Rule 1 — Bug] Root `README.md` contract count 1789 → 1790.**
- **Found during:** Task 3, running the `readme_contract` drift gate.
- **Issue:** Adding `contracts/forecast-tool-boundary-v1.yaml` moved `find contracts/ -name '*.yaml' | wc -l` to 1790 while the claims table still said 1789, turning `FALSIFY-README-007` red on this plan's edit. Per the 06-01 precedent (fix the count you moved), closing it is this plan's responsibility.
- **Fix:** One-line correction. `cargo test -p aprender-core --test readme_contract` is now **15 passed / 0 failed** — the last of the three failures 06-01 logged as D-ITEM-06-01-a is closed.
- **Committed in:** `14eceff67`

**4. [Rule 2 — Missing critical] The T-06-02 mitigation's README half was not in the plan's `files_modified`.**
- **Found during:** Task 3, checking the plan's `<threat_model>` against what had actually been built.
- **Issue:** T-06-02's mitigation column requires "README documents sizing K to the blocking-thread budget", but `crates/aprender-mcp-forecast/README.md` is not in the plan's `files_modified` list. A registered mitigation with no implementation is an unmitigated threat with a tick next to it.
- **Fix:** Added a "Sizing `--pool`" section: K is the number of in-flight fits so size it to the CPU/blocking budget; the HARD bounds are `MAX_POINTS`/`MAX_HORIZON`/the per-round iteration cap while `FIT_BUDGET_SECS` is a *cooperative* round-boundary budget and not a wall-clock cap (REVIEW-06-U1's wording carried through to user-facing docs); the measured 1.002x/2.070x control pair; and the note that responses do not depend on load.
- **Committed in:** `14eceff67`

**5. [Rule 3 — Blocking] The plan's clippy `<verify>` command cannot pass for any crate in this tree; scoped with `--no-deps`.**
- **Found during:** Task 3.
- **Issue:** Pre-existing and already logged as D-ITEM-06-01-b: `-D warnings` reaches path dependencies on toolchain 1.93.0 and fails inside `aprender-compute` for untouched crates.
- **Fix:** `--no-deps`, as 06-01 through 06-05 all did. **Engagement re-proven in THIS crate rather than inherited:** inserting a `clippy::needless_bool` violation into `pooled_app` turned the command red (`rc=101`, `needless_bool` cited by name at the 1.93.0 docs URL); removing it turned it green. Extending a guard's scope requires re-mutating in the new scope (CLAUDE.md Verification Discipline #4).

**6. [Rule 2 — Missing critical] Three refusal cases beyond the plan's enumerated set.**
- **Found during:** Task 2.
- **Issue:** The plan's list yields 18 cases. Three D-11 refusals the contract names had no e2e case: the `ds`/`y` length mismatch (`"ds has"`, which the plan's own `<interfaces>` block lists as a needle), duplicate `ds`, and horizon 0 as distinct from horizon 3651 (both share a message but not a cause).
- **Fix:** Added `refuses_length_mismatch`, `refuses_duplicate_ds`, `refuses_horizon_zero` — 21 cases total. Cheap, and they close the gap between the contract's claims and the suite's coverage.
- **Committed in:** `d0a63573a`

---

**Total deviations:** 6 — 3 × Rule 1 (bug: a false contract claim, a false README claim, a stale count), 2 × Rule 2 (missing critical: the T-06-02 README mitigation, three uncovered refusals), 1 × Rule 3 (blocking: the pre-existing clippy scope).
**Impact on plan:** No scope creep. Every deviation either corrected something this plan itself made false, or closed a gap between a claim in this plan's own artifacts and the evidence behind it.

## Verification results

| Gate | Result |
|---|---|
| `pv validate contracts/forecast-tool-boundary-v1.yaml` | **0 error(s), 0 warning(s)** — `Contract is valid.` |
| `pv status` (non-hollow) | References 5, **Equations 16, Proof obligations 16, Falsification tests 16, Kani harnesses 3**, QA gate `F-FORECAST-BOUNDARY-001` |
| `cargo test -p aprender-forecast --lib bounds_match_contract` | **ok. 2 passed; 0 failed** |
| `cargo test -p aprender-mcp-forecast --lib e2e::` | **ok. 25 passed; 0 failed; 0 ignored** — 21 `refuses_`, 4 `happy_path` |
| `cargo test -p aprender-mcp-forecast --lib the_tool_schema_is_strict` | **ok. 1 passed** |
| `cargo test -p aprender-mcp-forecast --lib pool_equality -- --nocapture` | **ok. 2 passed; 0 failed**; exactly **1** well-formed `POOL SPEEDUP:` line |
| `cargo test -p aprender-mcp-forecast --test e2e_stdio` | **ok. 1 passed; 0 failed; 0 ignored** |
| `cargo test -p aprender-mcp-forecast --bins` | **ok. 3 passed; 0 failed** (the `--pool` argv parser) |
| `cargo test -p aprender-forecast -p aprender-mcp-forecast` (all targets) | **79 + 28 + 3 + 1 + 1 passed, 0 failed** (7 ignored are 06-05's `cfg(chronos_weights)` skips) |
| `cargo clippy -p aprender-mcp-forecast --all-targets --no-deps -- -D warnings` | **rc=0** (engagement re-proven in this crate by a `needless_bool` mutation) |
| `cargo fmt --all -- --check` | **rc=0** |
| `cargo test -p aprender-core --test readme_contract` | **15 passed; 0 failed** (was 3 failing after 06-01) |
| `.github/workflows` untouched | **0 changed lines** in the working tree AND **0** across `edbf96130..HEAD` |
| `git check-ignore` on `tests/e2e_stdio.rs` | **exit 1** — tracked, not silently ignored |

### Workspace regression: zero new failures

`cargo nextest run --profile ci --no-fail-fast --workspace --lib --exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute`:

```
Summary [354.062s] 81556 tests run: 81475 passed (4 slow, 1 flaky, 3 leaky), 81 failed, 136 skipped
```

**81 failed — byte-identical to the pinned darwin baseline**, and identically distributed: 51 `aprender-serve`, 21 `aprender-train` (gpu), 2 `aprender-zram-core`, 2 `aprender-orchestrate`, 2 `aprender-cgp`, 1 each `aprender-mcp` / `aprender-core` / `aprender-contracts`. **Zero failures in `aprender-forecast` or `aprender-mcp-forecast`.**

(The first attempt used the plain `ci` profile, which fails fast and stopped at 7 917/81 556 on the pre-existing `aprender-cgp` roofline test. `--no-fail-fast` is what produces a comparable denominator — a run that stops early cannot be compared to an 81-failure baseline, which is the whole reason to re-run rather than to reason about it.)

## Issues Encountered

- **The `rtk` hook rewrites `cargo test` and strips the `test result:` line**, exactly as 06-01 recorded, so every `<verify>` grep on that line was run through `rtk proxy cargo test …`. Anyone re-running these blocks verbatim on a box with the hook active will hit it. It is a harness artifact, not a code defect.
- **`pv` is not on `PATH`**; `target/debug/pv` was already built and was used directly.
- **The `ci` nextest profile fails fast**, which makes it useless for a baseline comparison. See above.

## Known Stubs

None. Every symbol this plan advertises is implemented and exercised: `pooled_app` dispatches across K real routers (proven by the `POOL = 1` control giving 1.002x and `POOL = 8` giving 2.070x on the same requests), `--pool` reaches `pooled_app` from `main`, all 21 refusals and 4 happy paths run against live servers, and `tests/e2e_stdio.rs` drives the actual built binary with no environment gate that could let it pass vacuously.

Two things are *incomplete by design and by plan*, and neither is a stub:

- **`just forecast-pool-ratio` does not exist yet** — it is plan 06-08's deliverable. This plan's job was to emit the line it will parse, which it does, in exactly the shape 06-08's grep expects.
- **`tests/e2e_stdio.rs` is dark in CI** — `.github/workflows/ci.yml`'s explicit `--test` target line does not list it, and this plan is forbidden from editing that file. Adding it is 06-08's human check-in. The test itself is complete, unconditional and green locally.

## Threat Flags

None. No file created or modified here introduces security-relevant surface outside the plan's `<threat_model>`. `pooled_app` is T-06-02/T-06-03 as registered: each of the K routers is built by the same `build_server` and carries the same `AllowedOrigins::localhost()`, and the pool binds nothing itself — `main` still binds `127.0.0.1` only. No new dependency entered the graph (`tower` was already a workspace dependency and already in this crate's manifest, so `Cargo.lock` is untouched — T-06-SC holds and was checked, not assumed).

## Pre-existing gate state (NOT regressions)

- **81 workspace `--lib` failures on this darwin host** — the pinned baseline, reproduced exactly. See above.
- **`cargo clippy -- -D warnings` unusable unscoped** — D-ITEM-06-01-b.
- **`.pv/contracts.idx`, `.pv/contracts.idx.mtime`, `.pv/lint-previous.json` dirty in the working tree** — pv's local index cache, already modified before this plan's first command. Logged as D-ITEM-06-06-a and deliberately not committed.

## User Setup Required

None — no external service configuration. `aprender-mcp-forecast` needs no model file, no credentials and no network access.

## Next Phase Readiness

**Ready for 06-07** (the Chronos server), which inherits everything it needs:

- `contracts/forecast-tool-boundary-v1.yaml` already carries `chronos_server_bounds`, `horizon_gate_with_warning` and the Chronos half of `shared_response_shape`, so 06-07 writes assertions against an existing contract rather than authoring one. `chronos_native_horizon: 64` is in `constants:` waiting for 06-07 to assert it against the shipped config fixture — deliberately *not* asserted here, because it is a property of the weights rather than of this crate.
- `FALSIFY-BOUNDARY-014` names the Chronos refusal cases 06-05 already ships, so the contract's Chronos claims are falsifiable today, on an unarmed host.
- The e2e patterns (`serve()`, `refused()`, `assert_shared_shape`) are in `mod e2e` and are the template for the Chronos server's own suite.

**For 06-08**, three concrete inputs are in the "measurement" section above: the exact `POOL SPEEDUP` grep shape, the finding that sibling-test contention is negligible (2.037x parallel vs 2.070x isolated, so `--test-threads=1` is optional), and the debug-profile baseline of ~2.07x that its release run should beat.

**One thing needs a human, not a plan.** `tests/e2e_stdio.rs` is green and unconditional but invisible to CI until someone adds it to `.github/workflows/ci.yml`'s `--test` target line. Until then the stdio transport — the one MCP clients actually spawn — has no CI guard, and a regression in it would ship.

---
*Phase: 06-native-time-series-forecasting-stack*
*Completed: 2026-09-06*

## Self-Check: PASSED

All 8 `key-files` entries exist on disk; all four task commits
(`5de8b7728`, `d0a63573a`, `c5608982e`, `14eceff67`) are present in `git log --all`.
