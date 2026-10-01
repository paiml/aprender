---
phase: 06-native-time-series-forecasting-stack
plan: 10
subsystem: api
tags: [rust, mcp, prophet, forecasting, input-validation, provable-contracts, pv]

requires:
  - phase: 06-native-time-series-forecasting-stack
    provides: "the forecast door (06-01..06-05), the e2e refusal suite and pinned REFUSAL_CODE (06-06), contract-audit-phase6 (06-09)"
provides:
  - "A `cap` on any NON-logistic growth arm — explicit linear, explicit flat, or the DEFAULTED arm with no growth key — is refused at THE door with `cap is logistic-only; set growth to \"logistic\"`"
  - "`spec.cap` is bound from `checked_cap`, set only inside the logistic branch, so a Spec carrying a cap the door did not validate is structurally unreachable"
  - "Two `e2e::refuses_cap_*` cases proving the bar through a live streamable-HTTP server"
  - "`cap_is_logistic_only` equation + proof obligation + FALSIFY-BOUNDARY-017 in forecast-tool-boundary-v1.yaml, with a resolving binding row"
  - "`COVERAGE.md` — the reasoned no-external-API declaration for Phase 6"
affects: [06-11, 06-12, 06-13, forecast-tool-boundary, gsd-verify-work]

actuals:
  tokens: 59886
  tasks: 3
  commits: 3
plan_head_before: 5b91ce2ad261c5e58b1791d2600f6a3c637d200f

tech-stack:
  added: []
  patterns:
    - "Cross-ARM refusal (not just cross-MODEL): an option scoped to one enum variant of one model refuses on the other variants, including the DEFAULTED one"
    - "Structural validation: the validated value is re-bound inside the branch that checked it, so the unvalidated path cannot reach the spec"

key-files:
  created:
    - .planning/phases/06-native-time-series-forecasting-stack/COVERAGE.md
  modified:
    - crates/aprender-forecast/src/forecast.rs
    - crates/aprender-mcp-forecast/src/lib.rs
    - contracts/forecast-tool-boundary-v1.yaml
    - contracts/aprender/binding.yaml

key-decisions:
  - "Applied the MINOR bump `pv diff` itself suggested (1.0.0 -> 1.1.0), while recording that the change narrows the accepted-input surface — `pv diff` sees only additions and cannot see a refusal tightening"
  - "Asserted four refusing shapes plus a positive control in the unit test, not the two the verifier named — one failing input is an anecdote (CLAUDE.md rule 6)"
  - "Ran every verification through `rtk proxy`: the rtk hook rewrites bare `cargo`/`make` and replaces libtest's `test result:` line with a summary, which would make every plan `<verify>` grep fail vacuously"

patterns-established:
  - "RED-side observation for an already-committed door check: temporarily disarm the check, run the new e2e cases, record the failing reply body, restore from a byte-exact backup and prove restoration with an empty `git diff`"

requirements-completed: [SC1]

coverage:
  - id: D1
    description: "A `cap` on the explicit-linear, explicit-flat or DEFAULTED growth arm is refused at THE door with `cap is logistic-only; set growth to \"logistic\"`, and a cap below max(y) on a non-logistic arm refuses for being off-arm rather than sneaking through the logistic bound"
    requirement: "SC1"
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::an_option_belonging_to_the_other_model_is_refused_not_dropped"
        status: pass
    human_judgment: false
  - id: D2
    description: "The refusal does not over-refuse: `growth: logistic` with a finite cap above max(y) still fits and returns a 7-point band, and the three logistic-arm messages plus the older cross-MODEL `cap is prophet-only` all keep their own distinct texts"
    requirement: "SC1"
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/forecast.rs#forecast::tests::an_option_belonging_to_the_other_model_is_refused_not_dropped (logistic positive control + `cap is prophet-only` pin)"
        status: pass
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::prophet_logistic_with_holidays_happy_path"
        status: pass
    human_judgment: false
  - id: D3
    description: "The bar is asserted THROUGH a live streamable-HTTP server, as every other SC1 refusal is: a validation-class refusal carrying the pinned REFUSAL_CODE and VALIDATION_PREFIX and naming `logistic-only`"
    requirement: "SC1"
    verification:
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::refuses_cap_without_logistic_growth"
        status: pass
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::refuses_cap_with_explicit_linear_growth"
        status: pass
    human_judgment: false
  - id: D4
    description: "The numerics are UNMOVED: the change is at the door, before `make_design`, so no parity rung can shift"
    verification:
      - kind: unit
        ref: "cargo test -p aprender-forecast --lib prophet::parity (32 passed, identical to the pre-change count)"
        status: pass
    human_judgment: false
  - id: D5
    description: "The rule has a contract identity that cannot be deleted silently: equation `cap_is_logistic_only`, an invariant proof obligation, FALSIFY-BOUNDARY-017 naming its four falsifying assertions, and a binding row that both audits and RESOLVES"
    verification:
      - kind: other
        ref: "cargo run -p aprender-contracts-cli --bin pv -- validate contracts/forecast-tool-boundary-v1.yaml (0 error(s))"
        status: pass
      - kind: other
        ref: "make contract-audit-phase6 (rc=0, zero BIND-, zero RESOLVE-, 4 `Total equations:` summaries, resolved 56 rows)"
        status: pass
    human_judgment: false
  - id: D6
    description: "Phase 6's `COVERAGE.md` declares, with a reason, that the phase integrates no external API, so the api-coverage seal gate has a decided answer rather than a re-run detector"
    verification:
      - kind: other
        ref: "grep -qE '^No external API integration: .{40,}\\.$' COVERAGE.md && grep -c '^|' == 0"
        status: pass
    human_judgment: false
  - id: D7
    description: "A page user who types a cap on the linear arm now receives a refusal naming the fix instead of a silently-linear forecast, rendered in the demo page"
    verification: []
    human_judgment: true
    rationale: "The demo page needed no edit and nothing here asserts browser rendering. The 06-VERIFICATION human item — drive initialize -> tools/list -> tools/call on both demo pages in a browser and confirm a forecast is charted — REMAINS OPEN and was deliberately not planned as automated work."

duration: 23 min
completed: 2026-09-07
status: complete
---

# Phase 06 Plan 10: Close VERIFICATION gap 1 — `cap` is logistic-only Summary

**A `cap` sent on any non-logistic growth arm is now a validation refusal naming the fix, structurally rather than by a guard, proven RED-first at the library door and through a live streamable-HTTP server, and pinned as `cap_is_logistic_only` / FALSIFY-BOUNDARY-017.**

## Performance

- **Duration:** 23 min
- **Started:** 2026-09-07T02:18:00Z
- **Completed:** 2026-09-07T02:41:13Z
- **Tasks:** 3
- **Files modified:** 5 (4 modified, 1 created)

## Accomplishments

- **The refusal exists at THE door.** `crates/aprender-forecast/src/forecast.rs` now refuses `cap` on the explicit-`linear`, explicit-`flat` and DEFAULTED growth arms with `cap is logistic-only; set growth to "logistic"`, in the same shape as the neighbouring cross-MODEL refusals. The check runs AFTER the growth enum is parsed, so an unknown growth string still refuses first with its own message.
- **It is structural, not merely guarded.** `spec.cap` is now assigned from `checked_cap`, a local set only at the END of the logistic branch after both the finiteness and `cap > max(y)` checks pass. There is no longer any path that puts a `Some` into `spec.cap` without validating it.
- **The bar is asserted through the server.** Two new `e2e::refuses_cap_*` cases drive a live streamable-HTTP router and observe a validation-class refusal carrying the pinned `REFUSAL_CODE` and `VALIDATION_PREFIX`.
- **The rule has a contract identity.** `forecast-tool-boundary-v1.yaml` gained `cap_is_logistic_only`, an invariant proof obligation and FALSIFY-BOUNDARY-017; `binding.yaml` gained a row that both audits and RESOLVES to a real definition site.
- **The phase's `COVERAGE.md` declaration landed**, so the api-coverage seal gate has a decided answer.

## Task Commits

1. **Task 1 (TRACER): refuse `cap` off the logistic growth arm at THE door** — `9d6eb639e` (feat)
2. **Task 2: assert the bar THROUGH the server — two e2e refusal cases** — `e9a93dbb8` (test)
3. **Task 3: contract identity + COVERAGE.md** — `d5d713571` (docs)

## Files Created/Modified

- `crates/aprender-forecast/src/forecast.rs` — the cross-GROWTH cap refusal, the `checked_cap` structural binding, and six new assertions inside `an_option_belonging_to_the_other_model_is_refused_not_dropped`
- `crates/aprender-mcp-forecast/src/lib.rs` — `e2e::refuses_cap_without_logistic_growth`, `e2e::refuses_cap_with_explicit_linear_growth`
- `contracts/forecast-tool-boundary-v1.yaml` — equation `cap_is_logistic_only`, one proof obligation, FALSIFY-BOUNDARY-017, version 1.0.0 -> 1.1.0
- `contracts/aprender/binding.yaml` — the `cap_is_logistic_only` row (`aprender_forecast::forecast` / `forecast`)
- `.planning/phases/06-native-time-series-forecasting-stack/COVERAGE.md` — the reasoned no-external-API declaration (created)

## The RED side, OBSERVED (CLAUDE.md Verification Discipline rule 4)

### Unit cases — written and run BEFORE the door check existed

Route: tests first, fix second. Because all cases live in one `#[test]` fn the first failure masks the rest, so a temporary four-test scratch harness (deleted before the GREEN commit; `grep -c scratch` == 0 in the committed file) witnessed each shape individually.

The composite test, verbatim:

```
running 1 test
test forecast::tests::an_option_belonging_to_the_other_model_is_refused_not_dropped ... FAILED

---- forecast::tests::an_option_belonging_to_the_other_model_is_refused_not_dropped stdout ----

thread 'forecast::tests::an_option_belonging_to_the_other_model_is_refused_not_dropped' (93995713) panicked at crates/aprender-forecast/src/forecast.rs:549:22:
expected a Validation refusal, got Ok("prophet")

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 92 filtered out; finished in 0.04s
```

Per-shape, verbatim (temporary harness):

```
running 4 tests
test forecast::tests::scratch_red_explicit_flat_cap ... FAILED
test forecast::tests::scratch_red_explicit_linear_cap ... FAILED
test forecast::tests::scratch_red_bare_cap ... FAILED
test forecast::tests::scratch_red_linear_cap_below_max_y ... FAILED

---- forecast::tests::scratch_red_explicit_flat_cap stdout ----
panicked at crates/aprender-forecast/src/forecast.rs:549:22:
expected a Validation refusal, got Ok("prophet")

---- forecast::tests::scratch_red_explicit_linear_cap stdout ----
panicked at crates/aprender-forecast/src/forecast.rs:549:22:
expected a Validation refusal, got Ok("prophet")

---- forecast::tests::scratch_red_bare_cap stdout ----
panicked at crates/aprender-forecast/src/forecast.rs:549:22:
expected a Validation refusal, got Ok("prophet")

---- forecast::tests::scratch_red_linear_cap_below_max_y stdout ----
panicked at crates/aprender-forecast/src/forecast.rs:549:22:
expected a Validation refusal, got Ok("prophet")

test result: FAILED. 0 passed; 4 failed; 0 ignored; 0 measured; 93 filtered out; finished in 0.05s
```

Post-fix:

```
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 92 filtered out; finished in 0.01s
```

### e2e cases — RED observation route: door check TEMPORARILY DISARMED

Task 1 was already committed when Task 2 began, so the plan's second route was used: the new `if growth != Growth::Logistic && args.cap.is_some()` branch was guarded with `if false && ...`, the two cases were run, and the file was restored from a byte-exact backup (`git diff crates/aprender-forecast/src/forecast.rs` was EMPTY after restoration, and `grep -c 'if false &&'` == 0).

Both cases failed, and the reply bodies re-confirm the verifier's measurement independently — the two replies are byte-identical in `yhat` and `diagnostics` despite one carrying `growth: "linear"` and one carrying no growth key at all:

```
running 2 tests
test e2e::refuses_cap_with_explicit_linear_growth ... FAILED
test e2e::refuses_cap_without_logistic_growth ... FAILED

---- e2e::refuses_cap_with_explicit_linear_growth stdout ----
thread 'e2e::refuses_cap_with_explicit_linear_growth' (94014794) panicked at crates/aprender-mcp-forecast/src/lib.rs:473:9:
must be REFUSED, never defaulted; got: {"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"{\"model\":\"prophet\",\"freq\":\"D\",\"n_history\":60,\"fit_seconds\":0.025993708,\"predict_seconds\":0.000436958,\"ds\":[\"2020-03-01\",...],\"yhat\":[12.566119986155195,12.07507220953749,12.318166300743732,13.149998276501645,13.9818284061225,14.224925067987055,13.733884900006261],...,\"diagnostics\":{\"growth\":\"Linear\",\"seasonality_mode\":\"Additive\",...}}"}],"isError":false}}

---- e2e::refuses_cap_without_logistic_growth stdout ----
thread 'e2e::refuses_cap_without_logistic_growth' (94014795) panicked at crates/aprender-mcp-forecast/src/lib.rs:473:9:
must be REFUSED, never defaulted; got: {"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"{\"model\":\"prophet\",\"freq\":\"D\",\"n_history\":60,\"fit_seconds\":0.025950334,\"predict_seconds\":0.000460541,\"ds\":[\"2020-03-01\",...],\"yhat\":[12.566119986155195,12.07507220953749,12.318166300743732,13.149998276501645,13.9818284061225,14.224925067987055,13.733884900006261],...,\"diagnostics\":{\"growth\":\"Linear\",\"seasonality_mode\":\"Additive\",...}}"}],"isError":false}}

test result: FAILED. 0 passed; 2 failed; 0 ignored; 0 measured; 28 filtered out; finished in 0.04s
```

Post-fix (door check restored):

```
test e2e::refuses_cap_without_logistic_growth ... ok
test e2e::refuses_cap_with_explicit_linear_growth ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 28 filtered out; finished in 0.01s
```

## Measured before / after

| Measurement | Before | After |
|---|---|---|
| `cargo test -p aprender-forecast --lib` | 86 passed, 0 failed, 7 ignored | 86 passed, 0 failed, 7 ignored |
| `... --lib forecast::tests::` | 10 passed, 0 failed | 10 passed, 0 failed |
| `... --lib prophet::parity` | **32 passed, 0 failed** | **32 passed, 0 failed — UNMOVED** |
| `cargo test -p aprender-mcp-forecast --lib` | 28 passed, 0 failed | **30 passed, 0 failed** |
| `... --lib e2e::refuses_` | **21** passed, 0 failed | **23** passed, 0 failed |
| `... --test e2e_stdio` | 1 passed | 1 passed |
| `pv status` equations / obligations / falsifications | 16 / 16 / 16 | **17 / 17 / 17** |
| `make contract-audit-phase6` resolved rows | **55** | **56** |
| `forecast-tool-boundary-v1.yaml` `metadata.version` | 1.0.0 | **1.1.0** |

The `--lib` and `forecast::tests::` counts are unchanged BY CONSTRUCTION: the six new library assertions were added inside an existing `#[test]` fn, per the verifier's own `missing` bullet 2 ("extend `an_option_belonging_to_the_other_model_is_refused_not_dropped`"). The evidence that they run is the RED observation above, not a count delta.

`prophet::parity` at 32 both sides is the load-bearing number: it proves the change stayed at the door and never reached `make_design`.

## The four probe shapes now refused

| Probe | Before (measured by the verifier at HEAD `ce3e5a8ea`) | After |
|---|---|---|
| explicit `growth: "linear"` + `cap: 100.0` | accepted, `yhat_max` 35.8994 | refused: `cap is logistic-only; set growth to "logistic"` |
| explicit `growth: "flat"` + `cap: 100.0` | accepted, same byte-identical diagnostics | refused, same message |
| DEFAULTED arm (no `growth` key) + `cap: 100.0` | accepted, same byte-identical diagnostics | refused, same message |
| `growth: "linear"` + `cap: 0.5` (BELOW max(y) = 28.9) | accepted, same byte-identical diagnostics | refused, same message |

**The cross-MODEL refusal is unchanged.** `model: "neuralprophet"` + `cap` still refuses with the older `cap is prophet-only; set model to "prophet"` — pinned by a new assertion in the same test (`refusal(&np_cap, "cap is prophet-only")`), not merely read off the source. The three logistic-arm messages (`logistic growth needs cap`, `cap must be a finite number`, `cap {cap} must exceed max(y) = {y_max}`) are byte-untouched, and their e2e cases (`refuses_logistic_without_cap`, `refuses_logistic_cap_below_max_y`) still pass.

**Positive control:** `growth: "logistic"` with a cap computed as `max(y) + 1.0` from the args themselves (never a hardcoded number) still fits and returns a 7-point band, both in the unit test and through the server via `e2e::prophet_logistic_with_holidays_happy_path` — which passed in the same run (`test result: ok. 1 passed; 0 failed`).

## `pv diff` and the semver bump

Run per CLAUDE.md against two materialised filesystem paths, never a git revision:

```
$ git show HEAD:contracts/forecast-tool-boundary-v1.yaml > /tmp/ftb-old.yaml
$ cargo run --release -p aprender-contracts-cli --bin pv -- diff /tmp/ftb-old.yaml contracts/forecast-tool-boundary-v1.yaml
Contract diff: v1.0.0 → v1.0.0
Suggested bump: minor

  equations:
    + cap_is_logistic_only
  proof_obligations:
    + invariant:A cap is accepted only on the logistic growth arm; on every other arm — explicit linear, explicit flat, or the defaulted arm carrying no growth key — it is a validation refusal, never an accepted-and-dropped knob
  falsification_tests:
    + FALSIFY-BOUNDARY-017
```

**Bump applied: minor, 1.0.0 -> 1.1.0**, exactly as suggested.

Noted for the record without overriding the tool: `pv diff` compares contract STRUCTURE and sees only three additions, so it cannot see that the underlying door narrowed its accepted-input surface. A client that was silently sending `cap` on the linear arm now receives an error, which by strict API-compatibility reasoning is a breaking change. The plan's `<reversibility rating="costly">` takes that deliberately — D-11 and FALSIFY-BOUNDARY-005's own `if_fails` text already promised this refusal, so the change makes a published promise true rather than making a new one. If a future run decides the boundary's semver should track behaviour rather than contract structure, that is a `pv` enhancement, not a hand-edit here.

## Decisions Made

- **Applied the `pv`-suggested MINOR bump** rather than hand-choosing MAJOR. The dogfooded tool is the source of truth for this file (CLAUDE.md: never work around `pv` with a script or a judgement call); the narrowing observation is recorded above instead of silently overriding it.
- **Asserted four refusing shapes, not the two the verifier named.** CLAUDE.md rule 6: one failing input is an anecdote. The bare-cap case alone would still pass if the check were keyed on the ABSENCE of a `growth` key; the explicit-linear case alone would still pass if it were keyed on the PRESENCE of one. `flat` and `cap-below-max(y)` close the remaining two arms the verifier measured.
- **Every verification ran through `rtk proxy`.** See deviation 1.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Every plan `<verify>` grep was unreachable through the rtk hook**

- **Found during:** Task 1 (first baseline run)
- **Issue:** The environment's rtk hook rewrites a bare `cargo test` / `make` invocation and replaces libtest's raw output with a one-line summary (`cargo test: 86 passed, 7 ignored (1 suite, 33.13s)`). Every `<verify>` block in this plan greps for `test result: ok\. [1-9][0-9]* passed; 0 failed`, and every `<fails_when>` warns that a missing line must FAIL. Through the hook, `grep -E 'test result:'` matched nothing on a passing run — the gate could only ever report failure, and a naive `test $rc -eq 0` alone would have been exactly the "green that measured nothing" the plan's own `<fails_when>` texts guard against.
- **Fix:** Ran every verification command as `rtk proxy <cmd>`, which executes the raw command unfiltered. Confirmed by re-running the baseline and observing the real libtest line.
- **Files modified:** none (execution-harness only)
- **Verification:** `rtk proxy cargo test -p aprender-forecast --lib` printed `test result: ok. 86 passed; 0 failed; 7 ignored; ...`; `rtk proxy make contract-audit-phase6` printed the four `Total equations:` summaries and the `resolved N Phase 6 binding rows` banner that the hooked form had stripped.
- **Committed in:** n/a (no source change)

**2. [Rule 2 - Missing critical] Two extra refusing shapes and a cross-MODEL pin added to the unit test**

- **Found during:** Task 1
- **Issue:** The plan's `<action>` named only the explicit-linear and bare-cap cases, but its own `<acceptance_criteria>` require the SUMMARY to state FOUR refused probe shapes and to confirm `model: "neuralprophet"` + cap still refuses with the older `cap is prophet-only` message. Satisfying those by reading the source would be an assertion, not a measurement.
- **Fix:** Added `flat + cap`, `linear + cap 0.5 (below max(y))` and a `refusal(&np_cap, "cap is prophet-only")` pin to the same test. All three were witnessed RED where applicable (the two new refusing shapes appear in the per-shape harness output above).
- **Files modified:** `crates/aprender-forecast/src/forecast.rs`
- **Verification:** `cargo test -p aprender-forecast --lib forecast::tests::an_option_belonging -- --exact` reports `1 passed; 0 failed`
- **Committed in:** `9d6eb639e`

**3. [Documentation - measurement drift] The plan's `e2e::refuses_` baseline was stale**

- **Found during:** Task 2
- **Issue:** The plan states "22 existing" `e2e::refuses_*` cases and sets the acceptance bar at "at least 24 (22 existing + 2)". The MEASURED baseline is **21**, so the post-change count is **23**, not 24.
- **Fix:** No code change. Per CLAUDE.md's "Derive the numbers; do not quote them" — the command wins over a number written in a document. The criterion's actual intent (two cases ADDED, none replaced) is met and independently corroborated by the `--lib` total moving 28 -> 30, which is the plan's own `<verify>` gate (`>= 30`) and which passed.
- **Files modified:** none
- **Verification:** `rtk proxy cargo test -p aprender-mcp-forecast --lib e2e::refuses_` -> `23 passed; 0 failed`; `--lib` -> `30 passed; 0 failed`
- **Committed in:** n/a

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 missing-critical) + 1 documented measurement drift with no code change.
**Impact on plan:** No scope creep. Deviation 1 was required for ANY verification in this plan to measure anything. Deviation 2 strengthens the assertions the plan's own acceptance criteria demand. Deviation 3 is a stale number in the plan text, corrected by measurement.

## Prohibitions honoured

- **Never widen the refusal to the logistic arm** — the positive control (`cap = max(y) + 1.0`) fits and returns a 7-point band; `e2e::prophet_logistic_with_holidays_happy_path` passes. VERIFIED by test.
- **Never change the two existing logistic refusal messages** — `logistic growth needs cap` and `cap {cap} must exceed max(y) = {y_max}` are byte-untouched; `refuses_logistic_without_cap` and `refuses_logistic_cap_below_max_y` still pass. VERIFIED by test.
- **Do not touch `crates/aprender-compute`, `.github/workflows/*.yml`, or any parity fixture** — `git status --porcelain | grep -E '\.github/workflows/|crates/aprender-compute/'` returned nothing at every commit; `prophet::parity` is 32/32 on both sides. VERIFIED.
- **Do not plan or perform the three human_verification items** — none was touched. See below.

## Human verification items — ALL THREE REMAIN OPEN

None of the three items `06-VERIFICATION.md` routed to `human_verification` was closed, and none was planned as automated work:

1. **Drive `initialize` -> `tools/list` -> `tools/call` on both demo pages in a browser and confirm a forecast is charted — OPEN.** `crates/aprender-mcp-forecast/static/index.html` sends `cap` only when the field is non-empty and already labels it `Cap (logistic)`; `growth` is always sent from a `<select>`. So after this plan a page user who types a cap on the linear arm receives a refusal naming the fix instead of a silently-linear forecast — the intended behaviour, and **the page needed no edit**. Nothing here asserts browser rendering.
2. **Measure `quantiles_abs_f32_nonaarch64` on an x86_64 host — OPEN.** Untouched by this plan.
3. **Decide whether SC4's Chronos ladder staying dark in CI is acceptable — OPEN.** Untouched by this plan.

## Issues Encountered

- The rtk output filter (deviation 1) — resolved by `rtk proxy`, and the resolution is itself the reason every number in this SUMMARY is a real libtest line rather than a summarised one.
- Removing the temporary RED harness with a Python slice initially also removed the closing brace of `mod tests`; caught immediately by inspecting the file tail (`od -c`) before any test run, restored, and confirmed by a clean `git diff --stat` and a green compile. No commit ever contained the broken state.

## Known Stubs

None. No stub, placeholder, TODO or hardcoded empty value was introduced by this plan.

## Threat Flags

None. The change removes surface (a caller-supplied parameter that was accepted and ignored, T-06-19 / T-06-20) and adds none: no new endpoint, no new auth path, no file access, no schema change at a trust boundary, and no new dependency (T-06-SC not applicable — this plan runs no package-manager install).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- VERIFICATION gap 1 is closed on all three of its `missing` bullets: the refusal (Task 1), the extended unit test (Task 1), and the e2e assertion through the server (Task 2, two cases rather than one).
- The door path is re-proven end to end before the heavier gap-2 work in 06-11/06-12 lands on the same files. `forecast.rs` and `lib.rs` are both green and lint-clean, so those plans start from a known-good base.
- Blockers: none. The three human_verification items above remain open and are the human's to close.

## Self-Check: PASSED

Files claimed created/modified — all present on disk:
```
FOUND: crates/aprender-forecast/src/forecast.rs
FOUND: crates/aprender-mcp-forecast/src/lib.rs
FOUND: contracts/forecast-tool-boundary-v1.yaml
FOUND: contracts/aprender/binding.yaml
FOUND: .planning/phases/06-native-time-series-forecasting-stack/COVERAGE.md
```

Commits claimed — all present in history:
```
FOUND: 9d6eb639e
FOUND: e9a93dbb8
FOUND: d5d713571
```

Plan-level `<verification>` block, re-run at close-out:

| Check | Result |
|---|---|
| `cargo test -p aprender-forecast --lib` | `ok. 86 passed; 0 failed; 7 ignored` |
| `cargo test -p aprender-forecast --lib prophet::parity` | `ok. 32 passed; 0 failed` |
| `cargo test -p aprender-mcp-forecast --lib` | `ok. 30 passed; 0 failed` |
| `cargo test -p aprender-mcp-forecast --lib e2e::refuses_` | `ok. 23 passed; 0 failed` |
| `cargo test -p aprender-mcp-forecast --test e2e_stdio` | `ok. 1 passed; 0 failed` |
| `pv validate contracts/forecast-tool-boundary-v1.yaml` | `0 error(s), 0 warning(s) / Contract is valid.` |
| `pv status contracts/forecast-tool-boundary-v1.yaml` | `eq=17 ob=17 ft=17` (all risen from 16) |
| `make contract-audit-phase6` | rc=0, `BIND=0 RESOLVE=0`, 4 `Total equations:` summaries, `resolved 56 Phase 6 binding rows` |
| `cargo clippy -p aprender-forecast -p aprender-mcp-forecast --all-targets --no-deps -- -D warnings` | rc=0 |
| `cargo fmt --all -- --check` | rc=0 |
| `COVERAGE.md` shape gate | `COVERAGE OK` (declaration line >= 40 chars + period, zero table rows) |

---
*Phase: 06-native-time-series-forecasting-stack*
*Completed: 2026-09-07*
