---
phase: 06-native-time-series-forecasting-stack
plan: 01
subsystem: forecasting
tags: [prophet, time-series, mcp, pmcp, lbfgs, parity, axum, streamable-http, rust]

requires:
  - phase: 04-setfit-mcp
    provides: "the thin single-purpose MCP server template (crates/aprender-mcp-setfit) — manifest shape, build_server, stderr-only main, file-scope disallowed_methods allow"
provides:
  - "crates/aprender-forecast — a workspace member carrying the Prophet 1.4.0 port (dates, types, fit, forecast, prophet) behind ONE stateless forecast() door"
  - "crates/aprender-mcp-forecast — a thin pmcp 2.19.3 server: one `forecast` tool over stdio and in-process streamable-HTTP, with the spike demo page"
  - "D-04 rung 2 green: the Rust objective at Python Prophet 1.4.0's MAP on Peyton Manning, residual EXACTLY 0.0"
  - "17 oracle fixtures committed in-crate, byte-verified against the spike originals, with a provenance README"
  - "test_support: fixture / CSV / contract readers that plans 06-03..06-06 build their ladders on"
  - "The measured [profile.dev.package.aprender-forecast] opt-level = 3 decision — taken in Wave 1 so no Wave 2 plan edits the root manifest"
affects: [06-03, 06-04, 06-05, 06-06, 06-07, 06-08, 06-09]

actuals:
  tokens: 27200
  tasks: 2
  commits: 2

tech-stack:
  added: ["pmcp 2.19.3 (streamable-http, schema-generation)", "schemars 1.0", "axum 0.8 (workspace)", "tower 0.5 (workspace)", "reqwest 0.12 rustls-tls (dev)", "serde_yaml 0.9 (dev)"]
  patterns:
    - "The library is THE door: every bound, default and refusal lives in aprender_forecast::forecast; the MCP transport re-checks nothing (OPS-03)"
    - "Oracle fixtures live in-crate and are `expect`ed, never skipped — a parity test that silently does not run proves nothing"
    - "Tolerances are read from contracts, not written as test literals (test_support::equation_tolerance); this plan writes the last literal, 06-03 removes it"
    - "Profile overrides are decided on a MEASURED wall in the wave that owns the manifest, never opportunistically in a later plan"

key-files:
  created:
    - crates/aprender-forecast/Cargo.toml
    - crates/aprender-forecast/README.md
    - crates/aprender-forecast/src/lib.rs
    - crates/aprender-forecast/src/dates.rs
    - crates/aprender-forecast/src/types.rs
    - crates/aprender-forecast/src/fit.rs
    - crates/aprender-forecast/src/forecast.rs
    - crates/aprender-forecast/src/prophet.rs
    - crates/aprender-forecast/src/test_support.rs
    - crates/aprender-forecast/tests/fixtures/README.md
    - crates/aprender-mcp-forecast/Cargo.toml
    - crates/aprender-mcp-forecast/README.md
    - crates/aprender-mcp-forecast/src/lib.rs
    - crates/aprender-mcp-forecast/src/main.rs
    - crates/aprender-mcp-forecast/static/index.html
    - .planning/phases/06-native-time-series-forecasting-stack/deferred-items.md
  modified:
    - Cargo.toml
    - Cargo.lock
    - README.md

key-decisions:
  - "Fixtures COPIED into crates/aprender-forecast/tests/fixtures/ (17 files, 5.9 MB), byte-verified with cmp against .planning/spikes/*/fixtures/ — RESEARCH Open Question 2 resolved as recommended"
  - "Lambda wrapper crates DEFERRED (RESEARCH Open Question 3): each adds a bootstrap [[bin]] to the already-red FALSIFY-MONO-011 count, and http_app/stateless() ship so the wrapper is a short copy later"
  - "[profile.dev.package.aprender-forecast] opt-level = 3 ADDED on a measured wall: 64 s before -> 7x ladder projects 448 s > 60 s. ONE dev table, no test twin — the reviewers' non-inheritance claim was refuted and re-refuted in-tree"
  - "clippy gated with --no-deps: `-D warnings` reaches path dependencies on this toolchain and fails identically for untouched pre-existing crates; --no-deps is what the criterion was trying to measure, and its engagement was proven by a RED-turning mutation"
  - "model: neuralprophet left as a declared REFUSING stub naming plan 06-04 (REVIEW-06-06 scope cut) — the arm, the args and the dispatch site are already final"

patterns-established:
  - "Tracer discipline: one thin path committed through every seam (manifests, pedantic lints, module layout, pmcp axum router, fixture location) before any horizontal expansion"
  - "Recorded D-08 deviations: a ported line may only change for clippy, a banned method, or rustfmt — anything else is named in the source, in the commit and in the SUMMARY with its review ID"
  - "Prove the mechanism engaged, never label a run by intent: every gate in this plan was confirmed with a RED-turning probe or a control run against untouched code"

requirements-completed: [SC1, SC2]

coverage:
  - id: D1
    description: "aprender-forecast is a workspace member whose Prophet 1.4.0 port builds and lints clean under the workspace lints"
    requirement: SC2
    verification:
      - kind: unit
        ref: "cargo test -p aprender-forecast --lib (16 passed; 0 failed)"
        status: pass
      - kind: other
        ref: "cargo clippy -p aprender-forecast -p aprender-mcp-forecast --all-targets --no-deps -- -D warnings (rc=0); cargo fmt --all -- --check (rc=0)"
        status: pass
      - kind: other
        ref: "cargo metadata --no-deps reports 85 workspace packages (83 + 2)"
        status: pass
    human_judgment: false
  - id: D2
    description: "A client POSTs one JSON-RPC tools/call {forecast, {ds,y,horizon}} to the in-process streamable-HTTP app and receives ds/yhat/yhat_lower/yhat_upper/trend/components/fit_seconds/predict_seconds/diagnostics with no fit -> artifact -> forecast round-trip"
    requirement: SC1
    verification:
      - kind: e2e
        ref: "crates/aprender-mcp-forecast/src/lib.rs#e2e::forecast_prophet_happy_path_over_streamable_http"
        status: pass
    human_judgment: false
  - id: D3
    description: "The same binary serves the same one tool over stdio via Server::run_stdio, with human text on stderr only"
    requirement: SC1
    verification:
      - kind: manual_procedural
        ref: "piped initialize + tools/list into target/debug/aprender-mcp-forecast: both answered on stdout, only the `serving `forecast` on stdio` banner on stderr"
        status: pass
    human_judgment: true
    rationale: "Proven by a manual run this session, but NO committed test asserts the stdio transport — the plan's e2e covers streamable-HTTP only. A human (or a later plan) should decide whether stdio warrants its own regression test before the phase ships."
  - id: D4
    description: "D-04 rung 2: the Prophet port evaluates the Stan objective at Python Prophet 1.4.0's MAP on peyton_manning_prophet140.json within 1e-9"
    requirement: SC2
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/prophet.rs#parity::peyton_objective_at_python_map_within_1e9"
        status: pass
    human_judgment: false
  - id: D5
    description: "17 oracle fixtures committed in-crate, byte-identical to the spike originals, none gitignored, with a provenance README"
    verification:
      - kind: other
        ref: "cmp loop over tests/fixtures/*.{json,csv} vs .planning/spikes/*/fixtures/ — 17 byte-verified; git check-ignore exits 1"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/test_support.rs#tests::every_committed_fixture_is_readable"
        status: pass
    human_judgment: false
  - id: D6
    description: "parse_date accepts EXACTLY ten ASCII bytes of YYYY-MM-DD and refuses trailing content, whitespace, empty, non-ASCII and short strings (REVIEW-06-U2); future_days covers D/W/MS and refuses H by name (D-17)"
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/dates.rs#tests (9 tests, all `test dates::`)"
        status: pass
    human_judgment: false
  - id: D7
    description: "The root-manifest profile decision is taken on a MEASURED debug fit wall in Wave 1, and the override is proven to reach test builds"
    verification:
      - kind: other
        ref: "timed `cargo test -p aprender-mcp-forecast --lib e2e`: 64 s before / 13 s after; clean-build `cargo test -p aprender-forecast --lib -v | grep -o 'opt-level=[0-9]'` reports only opt-level=3, on the aprender_forecast unit"
        status: pass
    human_judgment: false
  - id: D8
    description: "Both crate READMEs exist with the paiml/aprender monorepo link and the D-16 empirical-coverage sentence; TOOL_DESCRIPTION carries it too"
    verification:
      - kind: integration
        ref: "cargo test -p aprender-core --test readme_contract test_every_readme_links_monorepo — both Phase 6 crates pass (the one remaining name is the pre-existing aprender-mcp-setfit)"
        status: pass
      - kind: other
        ref: "grep 0.60 in both READMEs and in TOOL_DESCRIPTION"
        status: pass
    human_judgment: false
  - id: D9
    description: "model: neuralprophet is a declared, REFUSING stub naming plan 06-04, with the arm, args and dispatch site already final"
    verification: []
    human_judgment: true
    rationale: "The stub's refusal is not asserted by any committed test in this plan — the NP refusal cases were moved to 06-04/06-06 with the port. A human should confirm the stub is acceptable as a Wave 1 functionality gap before the phase ships."

duration: 37min
completed: 2026-09-05
status: complete
---

# Phase 06 Plan 01: Forecast Tracer Summary

**One JSON-RPC `forecast` call now crosses tool schema → validation door → Prophet design → core L-BFGS → predict → response, with the Peyton Manning rung-2 objective matching Python Prophet 1.4.0's MAP to a residual of exactly 0.0.**

## Performance

- **Duration:** 37 min
- **Started:** 2026-09-06T03:01:12Z
- **Completed:** 2026-09-06T03:38:34Z
- **Tasks:** 2 (1 tracer + 1 auto)
- **Files created/modified:** 40 (23 authored, 17 copied oracle fixtures)

## Accomplishments

- **`crates/aprender-forecast`** — the Prophet 1.4.0 port as a workspace member: `dates` (civil-date arithmetic, no calendar library), `types` (the strict tool boundary), `fit` (the D-09 L-BFGS recipe verbatim), `forecast` (THE door), `prophet` (520 lines ported verbatim minus its private date copies). Builds and lints clean under the workspace's `all + pedantic` lints.
- **`crates/aprender-mcp-forecast`** — a thin pmcp 2.19.3 server on the `aprender-mcp-setfit` template: one `forecast` tool, stdio by default, `--http` serving the spike demo page plus stateless streamable-HTTP at `/mcp`, `--bench` for timing. It re-implements no numerics and re-checks no bound.
- **The tracer round trip passes** — `forecast_prophet_happy_path_over_streamable_http` drives a live in-process server through `initialize` → `tools/list` → `tools/call` on the 2 905-point Peyton Manning series, 365 days ahead, and checks the band brackets `yhat` pointwise, the named components, the trend length, finite timings and `diagnostics.lbfgs.rounds >= 1`.
- **D-04 rung 2 is green, and tight** — the residual is not merely under 1e-9, it is **exactly 0.0**: `f_rust = -8004.797952932482076` against the fixture's `log_posterior_at_map_unnormalized = 8004.797952932482076`, bit-identical.
- **17 oracle fixtures committed in-crate**, each `cmp`-verified byte-identical to its spike original, with a provenance table naming the generating environment (Prophet 1.4.0 / NeuralProphet 0.9.0 / chronos-forecasting 2.3.1) and the regeneration command.
- **The root-manifest profile decision is measured and closed in Wave 1**, so plans 06-02 through 06-09 read a stable `Cargo.toml`.

## Task Commits

1. **Task 1: End-to-end `forecast` tracer (both crates, one e2e happy path, stdio)** — `725d5c7b2` (feat)
2. **Task 2: 17 fixtures, test_support, Peyton rung 2, both READMEs, measured profile decision** — `13f785820` (feat)

## Files Created/Modified

| File | What it does |
|---|---|
| `crates/aprender-forecast/src/dates.rs` | `days_from_civil` / `civil_from_days` / `parse_ymd` / `parse_date` (strict) / `format_ymd` / `future_days` (D-17), plus 9 refusal tests |
| `crates/aprender-forecast/src/types.rs` | `ForecastArgs` (`deny_unknown_fields`, `JsonSchema`), `HolidayArg`, `ForecastResponse`, `ForecastError`, `MIN_POINTS`/`MAX_POINTS`/`MAX_HORIZON` |
| `crates/aprender-forecast/src/fit.rs` | x-keyed value+grad `Cached`, `FitInfo`, `MAX_ITERS_PER_ROUND`, `FIT_BUDGET_SECS`, `fit_prophet` (D-09) |
| `crates/aprender-forecast/src/forecast.rs` | `pub fn forecast` — THE validation door + prophet/neuralprophet dispatch; `normal_quantile` |
| `crates/aprender-forecast/src/prophet.rs` | The port: `Spec`, `columns`, `make_design`, `Model` objective + analytic gradient, `predict`, `Rng`, plus `mod parity` |
| `crates/aprender-forecast/src/test_support.rs` | `fixture_path`, `load_json`, `read_csv`, `contract_path`, `equation_tolerance`, `constant_u64` |
| `crates/aprender-forecast/tests/fixtures/` | 13 JSON oracles + 4 CSVs + provenance README |
| `crates/aprender-mcp-forecast/src/lib.rs` | `TOOL_NAME`, `TOOL_DESCRIPTION`, `build_server`, `http_app`, `map_error`, `mod e2e` |
| `crates/aprender-mcp-forecast/src/main.rs` | stdio default, `--http PORT`, `--bench` — stderr only on every protocol path |
| `crates/aprender-mcp-forecast/static/index.html` | The spike demo page, byte-identical; itself an MCP client |
| `Cargo.toml` | Two `members` lines + the measured `[profile.dev.package.aprender-forecast]` table |
| `README.md` | Workspace crate count 82 → 85 |

## Decisions Made

**Fixture location — COPY (RESEARCH Open Question 2).** The 17 files live in
`crates/aprender-forecast/tests/fixtures/` (5.9 MB), not referenced from `.planning/spikes/`.
Referencing would make the parity ladder depend on a planning directory that a consumer of the
crate would not have, and would let a spike edit silently move a tolerance. The `cmp` loop in
Task 2's `<verify>` re-establishes byte-identity on every run, so drift is red, not quiet.

**Lambda wrapper crates — DEFERRED (RESEARCH Open Question 3).** Each adds a `bootstrap`
`[[bin]]` to the already-failing `FALSIFY-MONO-011` allowlist count, and cargo-pmcp's `*-lambda`
discovery is ambiguous. `http_app` and `StreamableHttpServerConfig::stateless()` ship now, so the
wrapper is a short copy when someone wants it. Filed for 06-09 as D-ITEM-06-01.

**One `dev` profile table, not two — and this was re-proven, not merely inherited.** Both cross-AI
reviewers filed this as HIGH/CRITICAL: `[profile.dev.package.X]` supposedly does not reach
`cargo test`, so a `[profile.test.package.aprender-forecast]` twin was demanded. The plan carried
the orchestrator's refutation; this executor did not take it on trust. On a clean build,
`cargo test -p aprender-forecast --lib -v 2>&1 | grep -o 'opt-level=[0-9]' | sort -u` emits
**exactly one value, `opt-level=3`**, and the unit carrying it is `--crate-name aprender_forecast`.
`profile.test` inherits `dev` package overrides. The twin was not added.

**clippy scoped with `--no-deps`.** See the deviations section — this was a measured pre-existing
condition, not a loosened gate.

**`model: neuralprophet` stays a refusing stub.** REVIEW-06-06's scope cut holds: the match arm,
the `ForecastArgs` field set and the dispatch site are already in final form, so 06-04 Task 1 fills
a body without moving a seam.

## The measured profile decision (RESEARCH Pitfall 9 / F10)

The rule the plan set: add the override iff `7 × single-fit wall > 60 s`. Both walls are the
warm-build wall of `cargo test -p aprender-mcp-forecast --lib e2e`, which is exactly one real
Peyton fit (2 905 points, up to 8 L-BFGS rounds), timed with `date +%s` around the command.

- **debug e2e wall (before): 64 s** → 7 × 64 = **448 s**, far above the 60 s feedback-latency
  target and nextest's slow threshold. The rule fires.
- **debug e2e wall (after): 13 s** → 7 × 13 = **91 s** — a **4.9×** improvement.

91 s still exceeds 60 s on this proxy, and the manifest comment says so rather than rounding the
story: 7 × the LARGEST fit is a deliberate upper bound, since the 06-03 ladder's other six
fixtures are much smaller than Peyton's. 06-03 Task 2 re-measures the real ladder and edits no
manifest.

**Mechanism-engaged proof** (CLAUDE.md Verification Discipline #2): on a clean
`cargo clean -p aprender-forecast` build, the observed `opt-level` set for
`cargo test -p aprender-forecast --lib -v` is `['opt-level=3']`, carried by the
`aprender_forecast` crate unit. Only the `dev` table was written.

## Clippy audit trail on the ported code (D-08)

The spike sources were written for a standalone crate with no workspace lints. Against
`all + pedantic + -D warnings`, the ported code needed **three** mechanical edits — far fewer than
RESEARCH Pitfall 1 anticipated, because the workspace already allows the ML-shaped pedantic lints
(`float_cmp`, `too_many_lines`, `needless_range_loop`, `many_single_char_names`, `cast_*`,
`unreadable_literal`, `items_after_statements`, `type_complexity`).

| # | Lint | Where | Resolution |
|---|---|---|---|
| 1 | `unused_mut` (rustc) | `prophet.rs` `let mut comp_of = ...` | dropped the `mut` |
| 2 | `clippy::manual_is_some_and` (via the `map_or(false, ..)` form) | `fit.rs` `Cached::ensure` | `is_some_and` — the plan named this one in advance |
| 3 | `clippy::double_must_use` | `aprender-mcp-forecast::http_app` | removed the redundant `#[must_use]` (`axum::Router` is already `must_use`) |

Two file-scope `#![allow(clippy::disallowed_methods)]` were added, in `types.rs` and
`forecast.rs`, each with the `aprender-mcp-setfit/src/lib.rs:29-32` precedent quoted in a comment:
the `JsonSchema` derive and `serde_json::json!` expand to `.unwrap()` at file scope, where a
narrower allow cannot reach. `test_support.rs` carries a module-scope `#![allow(dead_code)]` with a
comment naming the plans (06-03, 06-05, 06-06) that will call `contract_path`,
`equation_tolerance` and `constant_u64`. **No crate-wide allow was added anywhere.**

## Deviations from Plan

### Recorded D-08 deviations (both planned, both carried out as written)

**1. [Planned — REVIEW-06-U2] `parse_date` strict ten-ASCII-byte shape check.**
- **Found during:** Task 1 (`dates.rs`), carried from the plan.
- **Issue:** the spike's `s.get(..10)` prefix take never inspected what followed, so
  `"2024-01-01garbage"` and `"2024-01-01T00:00:00"` parsed as valid dates. D-11 says the tool
  boundary refuses, never defaults.
- **Fix:** the shape is checked before splitting — `len() == 10`, `is_ascii()`, `-` at bytes 4 and
  7, the other eight `is_ascii_digit()`. The message text is unchanged (`bad date {s:?}: want
  YYYY-MM-DD`), so no 06-06 needle moves, and the calendar round-trip check is untouched, so
  `"2008-02-30"` still refuses with `not a calendar date`.
- **Cannot move a parity number:** every committed fixture and CSV carries a bare `YYYY-MM-DD`.
- **Verification:** 9 `dates::` tests, including trailing content, leading/trailing whitespace,
  empty, non-ASCII prefix and 9-byte cases.
- **Committed in:** `725d5c7b2`.

**2. [Planned — REVIEW-06-U1] `FIT_BUDGET_SECS` doc comment reworded; code unchanged.**
- **Found during:** Task 1 (`fit.rs`), carried from the plan.
- **Issue:** the 15 s budget was described as a cap. It is inspected once per COMPLETED L-BFGS
  round, after the improvement check, so one round of up to 2 000 iterations can overrun it.
- **Fix:** the doc comment now says the budget is *cooperative*, that `budget_hit` reports a late
  round boundary rather than a 15 s cut-off, and that the HARD bounds are `MAX_ITERS_PER_ROUND`,
  `MAX_POINTS` and `MAX_HORIZON`. `grep -c 't0.elapsed()' fit.rs` is **1** — the recipe is
  byte-for-byte the D-09 one.
- **Committed in:** `725d5c7b2`.

### Auto-fixed issues

**3. [Rule 3 — Blocking] The plan's clippy `<verify>` command cannot pass for ANY crate in this tree; scoped it with `--no-deps`.**
- **Found during:** Task 1, first clippy run.
- **Issue:** `cargo clippy -p <crate> --all-targets -- -D warnings` propagates `-D warnings` to
  **path dependencies** on this toolchain (1.93.0), so it failed with 18 errors inside
  `crates/aprender-compute` (unused imports, unreachable expressions, `dead_code`, unused
  variables) — none of them in code this plan touches.
- **Proved pre-existing rather than assumed:** the identical command against the untouched
  `aprender-mcp-setfit` fails the same way (`rc=101`, the same 18 `aprender-compute` findings), and
  `-p aprender-serve` fails inside `aprender-present-terminal`. The condition is workspace-wide and
  predates Phase 6.
- **Fix:** gated both Phase 6 crates with `--no-deps`, which scopes the lint to the primary
  packages — the thing the criterion was actually trying to measure. **Engagement proven, not
  assumed:** inserting a `clippy::needless_bool` violation into `aprender-forecast/src/lib.rs` turned
  the `--no-deps` command red (`rc=101`, `needless_bool` cited by name); removing it turned it green.
- **The 18 `aprender-compute` findings were NOT fixed** — out of scope, logged to
  `deferred-items.md` as D-ITEM-06-01-b.

**4. [Rule 3 — Blocking] `README.md` workspace crate count corrected 82 → 85.**
- **Found during:** Task 2, running the `readme_contract` drift gate.
- **Issue:** adding two workspace members moved `cargo metadata --no-deps` to 85 while the README
  claims table still said 82. `FALSIFY-README-005` was *already* red pre-plan (82 vs 83); this plan
  widened the gap, so closing it is this plan's responsibility.
- **Fix:** one-line correction. `test_readme_crate_count_matches_workspace` now passes.
- **The other three `readme_contract` failures were NOT fixed** — the contract count (1778 vs 1786,
  no contract added here), two missing crate READMEs, and the `aprender-mcp-setfit` missing link all
  predate Phase 6 and belong to crates this plan does not touch. Logged as D-ITEM-06-01-a. Net effect
  of this plan on that test: **4 failures → 3**.

**5. [Rule 1 — Bug] `tests/fixtures/README.md` line-ending note corrected to what was measured.**
- **Found during:** Task 2, writing the provenance README.
- **Issue:** the plan's `read_csv` note said `air_passengers.csv` has CR-only line endings. Measured:
  all four committed CSVs are LF-terminated; two carry `"`-quoted fields and two do not.
- **Fix:** the README records the measured state, and `read_csv` strips a trailing `\r` regardless so
  a CRLF checkout on another platform cannot silently change a parsed value.
- **Committed in:** `13f785820`.

**6. [Rule 1 — Bug, self-inflicted] A fabricated "after" timing was written into the manifest comment and corrected before commit.**
- **Found during:** Task 2, adding the profile table.
- **Issue:** the comment was drafted with a placeholder `after (opt-level 3): 12 s` *before* the
  post-override run existed. A number in a manifest comment is indistinguishable from a measured one.
- **Fix:** measured immediately (13 s), corrected the comment, and added the sentence explaining that
  `7 ×` the largest fit is an upper bound rather than a ladder prediction. Recorded here because
  CLAUDE.md's Verification Discipline is the reason it was caught, and a silent correction would
  teach nothing.

---

**Total deviations:** 6 — 2 planned D-08 deviations carried out as specified, 4 auto-fixed
(2 × Rule 3 blocking, 2 × Rule 1 bug).
**Impact on plan:** No scope creep. Two verify commands were scoped (`--no-deps`) or repaired
(README count) so they measure what they were written to measure; three out-of-scope pre-existing
defects were logged rather than fixed.

## Verification results

| Gate | Result |
|---|---|
| `cargo test -p aprender-mcp-forecast --lib e2e` | **ok. 1 passed; 0 failed** (50.9 s test time before the profile decision) |
| `cargo test -p aprender-forecast --lib` | **ok. 16 passed; 0 failed** |
| `cargo test -p aprender-forecast --lib prophet::parity` | **ok. 1 passed; 0 failed** — residual exactly 0.0 |
| `cargo test -p aprender-forecast --lib dates::` | **ok. 9 passed; 0 failed**, 9 `test dates::` lines |
| `cargo clippy -p aprender-forecast -p aprender-mcp-forecast --all-targets --no-deps -- -D warnings` | **rc=0** (engagement proven by RED-turning mutation) |
| `cargo fmt --all -- --check` | **rc=0** |
| `cargo check --workspace --exclude aprender-profile` | **rc=0**, 0 errors |
| 17-fixture `cmp` loop vs spike originals | **17 byte-verified** |
| `git check-ignore -v` on every new asset | **exit 1** — nothing silently untracked |
| Task 1 acceptance criteria | **39/39** |
| Task 2 acceptance criteria | **15/15** |

**`Cargo.lock` gained only the two workspace members' own `[[package]]` entries — no new registry
package, no `[patch]`** (threat T-06-SC satisfied and measured, not assumed:
`git diff Cargo.lock | grep '^+name = '` lists exactly `aprender-forecast` and
`aprender-mcp-forecast`).

**Tracer feedback gate:** Task 1 is `type="tracer"`. Interactive run,
`workflow.human_verify_mode = end-of-phase`, and the task's `<verify>` carries only `<automated>`
blocks, so per the checkpoints precedence chain the gate re-ran the full verify end-to-end rather
than raising a checkpoint. All four blocks passed; expansion proceeded to Task 2.

## Issues Encountered

- **The `rtk` hook rewrites `cargo test` and strips the `test result:` line**, so the plan's
  `grep -E 'test result: ok\. N passed; 0 failed'` could never match — a harness artifact, not a code
  defect. Resolved by running the verification commands through `rtk proxy cargo test ...`, which
  emits raw output. Anyone re-running this plan's `<verify>` blocks verbatim on a box with the hook
  active will hit the same thing.
- **`cargo clippy ... -- -D warnings` is unusable unscoped in this workspace** — see deviation 3.

## Known Stubs

| Stub | File | Why it is intentional, and who resolves it |
|---|---|---|
| `model: "neuralprophet"` returns `ForecastError::Validation("model neuralprophet is ported in plan 06-04")` | `crates/aprender-forecast/src/forecast.rs` (the `"neuralprophet"` match arm) | REVIEW-06-06 deliberately cut the NP port out of the tracer. This is a **functionality** gap, never an architectural one: the match arm, the `ForecastArgs` field set and the dispatch site are already final, so **plan 06-04 Task 1** replaces one arm body. The plan's `<success_criteria>` names this stub explicitly as a required outcome. |

No other stub exists: no hardcoded empty return flows to a response, no placeholder text, and every
component the server advertises is computed by the port.

## Threat Flags

None. No file created or modified here introduces security-relevant surface outside the plan's
`<threat_model>`. The one new network surface (`--http`, `/`, `/sample/*`, `/mcp`) is T-06-03 as
registered: loopback bind, `AllowedOrigins::localhost()`, documented in the README as a dev surface.

## Pre-existing gate state (NOT regressions)

- **`FALSIFY-MONO-011` reports `aprender-mcp-forecast`** alongside the four SetFit crates that
  already violated it (`aprender-mcp-setfit`, `-setfit-lambda`, `-setfit-train`,
  `aprender-setfit-train-lambda`). The plan predicted exactly this: the allowlist is shrink-only and
  raising it is **plan 06-02's blocking human decision**. Expected, recorded, not a regression.
- **`readme_contract` fails 3/15** after this plan (4/15 before it). See D-ITEM-06-01-a.
- **`aprender-profile`** remains a Linux-only `compile_error` on Darwin, hence the
  `--exclude aprender-profile` in the workspace check. Pre-existing, recorded in STATE.md.

## User Setup Required

None — no external service configuration. `aprender-mcp-forecast` needs no model file, no
credentials and no network access; the fit runs on data the caller supplies.

## Next Phase Readiness

**Ready for 06-02** (the `FALSIFY-MONO-011` human decision), which is the one thing standing between
this phase and a green `monorepo_invariants`.

Wave 2 inherits a stable base:

- The root `Cargo.toml` is **closed for this phase** — the members and the one profile table are in,
  measured, and no Wave 2 plan needs to touch it.
- `test_support` is in place, so 06-03's ladder writes assertions, not readers, and can replace this
  plan's single `1e-9` literal with `equation_tolerance` the moment `contracts/` carries the equation.
- All 13 JSON oracles are committed, including the NeuralProphet and Chronos ones that 06-04, 06-05
  and 06-07 need — no later plan pays the fixture-copy cost or re-litigates the location.
- `crates/aprender-forecast/src/lib.rs`'s `pub mod` block is the file 06-04 (`np`) and 06-05
  (`bolt`, `safetensors`, `chronos`) both append to — the serialisation REVIEW-06-06 already
  accounted for by moving 06-05 to Wave 3.

**One concern for the phase, not for the next plan:** the debug-profile ladder still projects ~91 s
against a 60 s target on the pessimistic upper bound. 06-03 Task 2 should re-measure the real
seven-fixture ladder before anyone concludes the feedback-latency target is met.

---
*Phase: 06-native-time-series-forecasting-stack*
*Completed: 2026-09-05*

## Self-Check: PASSED

All 17 `key-files.created` entries exist on disk; all three commits
(`725d5c7b2`, `13f785820`, `5e971ea3e`) are present in `git log --all`.
