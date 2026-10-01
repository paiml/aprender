---
phase: 06-native-time-series-forecasting-stack
plan: 08
subsystem: testing
tags: [just, benchmark, evidence, chronos-bolt, prophet, neuralprophet, mase, rolling-origin, mcp, ci, aarch64, neon]

requires:
  - phase: 06-06
    provides: "the router pool (`pooled_app`, `--pool K`), `pool_equality`'s printed `POOL SPEEDUP:` line, and the measured 1.002x/2.070x control pair this plan re-measures on release"
  - phase: 06-07
    provides: "aprender-mcp-chronos, its CHRONOS_EMBED_DIR build, and the --coldstart / --bench entry points the recipes wrap"
  - phase: 06-05
    provides: "the verify-always `fetch-chronos-tiny` recipe, `chronos_abs`, and the arch-keyed provisional f32 bar"
provides:
  - "Seven `just` recipes that turn SC1/SC4/SC5 into self-failing gates: chronos-embed-build, chronos-gate, chronos-bench, chronos-coldstart, forecast-bench, forecast-pool-ratio, mase-rolling-origin"
  - "`just chronos-gate` — the phase's embedded-weights gate (D-18 clause 2, local form): unconditional weight re-verification, then both armed suites, failing unless each reports 0 ignored and >= 1 passed"
  - "`just forecast-pool-ratio` — the recipe (not any unit test) that asserts the SC5 >= 2.0 speed-up, best of three"
  - "crates/aprender-forecast/examples/mase_rolling_origin.rs — the D-16 harness as a compiled example, on CI's `cargo build --examples` line"
  - "06-EVIDENCE.md — every host-gated bar MEASURED on aarch64 release with command, host, profile, N and log path"
  - "06-ci-chronos-step.patch — the ci.yml proposal, verified with `git apply --check`, unapplied"
affects: [06-09, phase-verification, release-gates]

actuals:
  tokens: 18506
  tasks: 3
  commits: 4
plan_head_before: 59a65367bfb0bf74b46827bc28a0ef15a0059a73
# MEASURED, not narrated. `git rev-list --count 59a65367b..HEAD` = 2 at SUMMARY-write time
# (adc8a560a, 1e2c410b9), 3 after the SUMMARY commit (298baebd9), and 4 at the boundary this
# line is committed on — the STATE.md/ROADMAP.md commit that immediately follows. The
# BOUNDARY is stated because a count of its own commit cannot be measured before it exists;
# naming the measurement point is what keeps the number falsifiable (#3968). Re-derive with
# `git rev-list --count 59a65367b..<the docs(06-08) state+roadmap commit>`.
# `tokens` is estimateTokens scale: 74 026 chars over `git diff 59a65367b..HEAD` / 4. The plan
# estimated 65 000 against 18 506 actual — a 3.5x over-estimate. The plan's `confidence: low`
# was right about the uncertainty and wrong about its direction: two of the three tasks were
# measurement, which costs wall-clock (compute) rather than diff.

tech-stack:
  added: []
  patterns:
    - "Host-gated evidence recipes: a bar that only holds on one arch/profile lives in a `just` recipe that states the arch, states the bar, and exits non-zero with the number — not in a test that CI would run on a host where the bar is meaningless"
    - "rc=$? on its own line, always: every recipe brackets its measured command with `set +e` / `rc=$?` / `set -e` so `set -e` cannot abort before the capture and no status is ever read through a pipe"
    - "Retries for noise, never for assertions: forecast-pool-ratio retries a wall-clock ratio up to 3x but fails the whole recipe on the first non-zero cargo exit, because that is a correctness failure"
    - "A ported harness proves itself by reproducing the source measurement, not by running: mase_rolling_origin matches spike-006 to 3 dp on all 17 windows through the shipped crate API"
    - "Evidence carries its own falsifiers: 06-EVIDENCE.md §7 lists what the numbers do NOT establish, and records the uncommitted working-tree delta's sha256 beside the commit hash"

key-files:
  created:
    - crates/aprender-forecast/examples/mase_rolling_origin.rs
    - .planning/phases/06-native-time-series-forecasting-stack/06-EVIDENCE.md
    - .planning/phases/06-native-time-series-forecasting-stack/06-ci-chronos-step.patch
  modified:
    - justfile
    - .planning/phases/06-native-time-series-forecasting-stack/deferred-items.md

key-decisions:
  - "CI decision measure-x86-first: neither ci.yml hunk is applied in this phase; the gating work is one x86_64 `just chronos-gate` run that turns the PROVISIONAL quantiles_abs_f32_nonaarch64 bar into a measurement, and only then is there evidence to wire a CI leg onto"
  - "Hunk (a) has TWO unprovisioned prerequisites, not one: the weights mount AND a way for `uv run --with huggingface_hub --with safetensors --with numpy` to resolve inside the network-less clean-room container. Surfaced before the decision rather than discovered in a red run"
  - "The SC5 ratio bar lives in `just forecast-pool-ratio` and in no unit test (REVIEW-06-04). Verified in-tree: `assert!(speedup` appears exactly once in aprender-mcp-forecast/src/lib.rs, inside the doc comment explaining its absence"
  - "`chronos-gate` calls `fetch-chronos-tiny` unconditionally (REVIEW-06-03) and echoes its sha256 output into the gate's own stdout, so the pins it verified are part of the evidence rather than an assertion about it"
  - "forecast-bench's asserted number is the SYNTHETIC 3 000-point series (0.212 s); the real 2 905-point Peyton fits are 1.5-3x slower (0.31-0.65 s) and are recorded as a separate row rather than the flattering one being reported alone"
  - "5.162x is not a contradiction of 06-06's 2.070x: release + real Peyton vs debug + synthetic. Both are stated with their conditions; 06-06's 1.002x single-router control is what makes either number evidence"
  - "The 18 pre-existing `cargo clippy -- -D warnings` failures in aprender-compute were NOT fixed (scope boundary) but were proven pre-existing by a control run and logged, because three of them are aarch64-conditional and therefore structurally invisible to the X64 CI"

patterns-established:
  - "Control before cause: a failing gate on new code was diagnosed by re-running it with the new file removed, and the identical 18-finding set proved the failure pre-existing rather than introduced"
  - "Lint what has no lint gate: `just` recipe bodies are not under scripts/, so they were extracted to temp .sh files and run through `bashrs lint` (0 errors) rather than left unlinted"
  - "Name the measurement point: every count and every number in the evidence file is stated with the boundary that produced it, so a later re-run can disagree with it"

requirements-completed: [SC1, SC4, SC5]

coverage:
  - id: D1
    description: "SC1 — a 3 000-point-class daily Prophet round trip completes in under 2 s on aarch64 release"
    requirement: SC1
    verification:
      - kind: integration
        ref: "just forecast-bench (target/p06-forecast-bench.log) — 0.212/0.212/0.213 s over N=3, bar 2.0 s"
        status: pass
      - kind: integration
        ref: "just mase-rolling-origin (target/p06-mase-rolling-origin.log) — real 2 200-2 815-point Peyton windows, Prophet 0.31/0.49/0.65 s over N=5"
        status: pass
    human_judgment: false
  - id: D2
    description: "SC4 — tiny-f16 Chronos forward at 2 048 context under 100 ms"
    requirement: SC4
    verification:
      - kind: integration
        ref: "just chronos-bench (target/p06-chronos-bench.log) — 18.0/18.1/18.2 ms over N=3 on the D-14 production routing row of the `weights F16` section"
        status: pass
    human_judgment: false
  - id: D3
    description: "SC4 — the embedded tiny-f16 release binary stays under 30 MB"
    requirement: SC4
    verification:
      - kind: integration
        ref: "just chronos-embed-build (target/p06-chronos-embed-build.log) — wc -c = 24 671 472 bytes, bar 30 000 000, byte-identical across N=5 builds"
        status: pass
    human_judgment: false
  - id: D4
    description: "SC4 — median exec to first forecast reply over stdio under 150 ms"
    requirement: SC4
    verification:
      - kind: e2e
        ref: "just chronos-coldstart 5 (target/p06-chronos-coldstart.log) — 52/53/91 ms, median 53 ms over N=5 spawned stdio servers"
        status: pass
    human_judgment: false
  - id: D5
    description: "SC5 — 8 concurrent requests complete in under half the sequential wall, and every response is bit-identical to its sequential result"
    requirement: SC5
    verification:
      - kind: integration
        ref: "just forecast-pool-ratio (target/p06-forecast-pool-ratio-{1,2,3}.log) — 5.146/5.150/5.162, best 5.162x vs the 2.0 bar, profile=release arch=aarch64"
        status: pass
      - kind: unit
        ref: "crates/aprender-mcp-forecast/src/lib.rs#pool_equality::eight_concurrent_requests_are_bit_identical_to_sequential — 2 passed; 0 failed; 0 ignored on all three attempts"
        status: pass
    human_judgment: false
  - id: D6
    description: "D-18 clause 2 (local form) — `just chronos-gate` re-verifies the pinned weights unconditionally, then runs both armed Chronos suites and fails unless each reports 0 ignored"
    verification:
      - kind: integration
        ref: "just chronos-gate (target/p06-chronos-gate-{weights,forecast,server}.log) — CHRONOS GATE: PASS; 8 passed/0 ignored and 9 passed/0 ignored; all three sha256 pins matched with nothing downloaded"
        status: pass
    human_judgment: false
  - id: D7
    description: "D-16 — the rolling-origin MASE/coverage/WQL3 harness ships as a compiled example with naive + seasonal-naive baselines and a 1-64 vs 65+ horizon slice"
    verification:
      - kind: integration
        ref: "cargo build --examples -p aprender-forecast (rc 0) and just mase-rolling-origin — reproduces spike-006's summary table to 3 dp on all 17 windows; oracle cross-check max |dMAE| = 7.8e-3"
        status: pass
    human_judgment: false
  - id: D8
    description: "The ci.yml change is proposed, not applied: a clean-applying patch exists and .github/ is untouched in the working tree and in every commit of this plan"
    verification:
      - kind: integration
        ref: "git apply --check .planning/phases/06-native-time-series-forecasting-stack/06-ci-chronos-step.patch (rc 0); git diff --stat HEAD -- .github and git diff --stat 59a65367b..HEAD -- .github both empty"
        status: pass
    human_judgment: false
  - id: D9
    description: "Both demo pages perform initialize -> tools/list -> tools/call and chart a forecast; the chronos page shows the allow_long_horizon refusal at horizon 365 without the flag and a warning with it"
    verification:
      - kind: e2e
        ref: "GET http://127.0.0.1:8787/ and :8788/ -> HTTP 200 with `tools/call` in the body; POST /mcp initialize -> serverInfo for both; live tools/call on :8788 returned a 12-step forecast with quantiles"
        status: pass
    human_judgment: true
    rationale: "The executor proved both servers are up and answer MCP, but whether the page CHARTS a legible forecast with a band, shows fit_seconds/predict_seconds in its event log, and renders the refusal vs the warning distinguishably is a visual judgement no assertion here makes. Both servers are left running for the end-of-phase human check."

duration: ~32 min
completed: 2026-09-06
status: complete
---

# Phase 06 Plan 08: Host-Gated Evidence, the D-16 MASE Harness, and the CI Decision Summary

**Seven self-failing `just` recipes that turn SC1/SC4/SC5 into measured numbers on this aarch64 release box — all six bars pass with 2.6x to 9x margin — plus the spike-006 rolling-origin harness as a compiled example that reproduces the spike's MASE table to three decimal places, and the ci.yml edit written as a clean-applying patch the human deferred behind one missing x86_64 measurement.**

## Performance

- **Duration:** ~32 min of executor wall time (approximate: the first tool call was not timestamped; the two task commits are `2026-09-06T22:56:58Z` and `2026-09-06T23:11:04Z`, and the run began ~12 min before the first). Excludes the human checkpoint wait.
- **Started:** ~2026-09-06T22:44Z
- **Completed:** 2026-09-06T23:16Z
- **Tasks:** 3 (2 executed, 1 decision)
- **Files created/modified:** 5 (3 created, 2 modified), plus this SUMMARY and the state files

## Accomplishments

- **All six host-gated bars MEASURED and passing on aarch64 release** at commit `adc8a560a`, each with command, host, profile, N and a `target/` log path in `06-EVIDENCE.md`. Not one number is carried over from the spikes; the spike value sits in its own column beside the measurement and never decides a pass.

  | gate | bar | measured (min/median/max, N) | margin |
  |---|---|---|---|
  | `chronos-gate` | rc 0, `0 ignored` in both summaries | 8 passed/0 ignored, 9 passed/0 ignored | PASS |
  | `chronos-embed-build` | < 30 000 000 bytes | 24 671 472 (N=5, byte-identical) | 82 % of bar |
  | `chronos-bench` | < 100 ms | 18.0 / 18.1 / 18.2 ms (N=3) | 5.5x under |
  | `chronos-coldstart 5` | < 150 ms median | 52 / **53** / 91 ms (N=5) | 2.8x under |
  | `forecast-bench` | < 2 s | 0.212 / 0.212 / 0.213 s (N=3) | 9.4x under |
  | `forecast-pool-ratio` | >= 2.0 best-of-3 | 5.146 / 5.150 / **5.162** (N=3) | 2.6x over |

- **`just chronos-gate` is the phase's embedded-weights gate and it cannot pass vacuously.** It calls `fetch-chronos-tiny` *unconditionally* (REVIEW-06-03), echoes its sha256 output into the gate's own stdout, and requires `0 ignored` **and** at least one passing test in *both* summaries. On this run all three pins matched with nothing downloaded — which is exactly the failing direction REVIEW-06-03 cared about, and exactly what a CI weights mount would exercise.

- **The SC5 bar moved to where it belongs.** `just forecast-pool-ratio` asserts `>= 2.0` best-of-three; `pool_equality` asserts only bit-identical responses and prints the ratio (REVIEW-06-04). Verified rather than assumed: `assert!(speedup` occurs exactly once in `aprender-mcp-forecast/src/lib.rs`, inside the doc comment that explains why there is no such assertion. Zero occurrences in code.

- **The D-16 harness ships as a compiled example and validated the port while doing it.** `mase_rolling_origin` reproduces spike-006's summary table — naive 2.017, seasonal naive 1.693, Prophet 1.094, NP-lite 1.029, Chronos-tiny 1.238 mean MASE — **identical to three decimal places on all 17 windows**, through `aprender_forecast::{fit, np, prophet, chronos}` instead of the spike's standalone modules. Oracle cross-check: max |MAE(Rust) − MAE(Python)| = 7.8e-3 over 17 forecasts. The only column that differs is speed (Chronos 3.0 s vs 5.9 s).

- **The CI edit was proposed, never applied.** `.github/` is provably untouched — `git diff --stat HEAD -- .github` and `git diff --stat 59a65367b..HEAD -- .github` are both empty. The patch applies cleanly and carries a decision header stating both hunks, both unprovisioned runner prerequisites, and the quantified x86_64 risk.

- **Two out-of-scope findings surfaced rather than swallowed** — see Issues Encountered.

## Task Commits

1. **Task 1: the just recipes + the MASE rolling-origin example + the ci.yml proposal as a patch** — `adc8a560a` (feat)
2. **Task 2: measure every host-gated bar on aarch64 release; record 06-EVIDENCE.md; start both demo servers** — `1e2c410b9` (docs)
3. **Task 3: DECISION — the ci.yml embedded-weights leg and e2e_stdio target** — no commit; `gate="blocking-human"`, resolved by the human, recorded below

**Plan metadata:** this commit (docs: complete plan)

## Files Created/Modified

- `justfile` — +7 recipes under a "Phase 6 host-gated evidence recipes" header that states why these bars are aarch64-release-only and what CI asserts instead. 10 `rc=$?` captures, none after a pipe.
- `crates/aprender-forecast/examples/mase_rolling_origin.rs` — 4 series x 17 rolling origins x 5 models: MASE (period 7/12), cov80, width/σ, WQL3, the 1-64 vs 65+ horizon slice, `naive` and `seasonal naive` rows on every window, and the per-window cross-check against `tests/fixtures/chronos_holdout_oracle.json`. Chronos rows run only when `CHRONOS_MODEL_DIR` is set and say so when they do not. The header documents the measured routing rule and states that `model: auto` is deferred.
- `.planning/phases/06-native-time-series-forecasting-stack/06-EVIDENCE.md` — provenance, the measured table, the SC5 attempt lines verbatim, the gate's sha256 output, mechanism proofs, the D-16 table beside spike-006's, and §7 "What these numbers do NOT establish".
- `.planning/phases/06-native-time-series-forecasting-stack/06-ci-chronos-step.patch` — the two proposed hunks with a decision header. `git apply --check` rc 0.
- `.planning/phases/06-native-time-series-forecasting-stack/deferred-items.md` — five entries added (items 1-5 below).

## The CI decision

CI decision: measure-x86-first
CI decision (verbatim): CI decision: measure-x86-first

**No ci.yml edit happens in this plan, and none is scheduled for 06-09 by default.** The rationale, recorded here so 06-09 and the phase verifier inherit it rather than re-deriving it:

1. **Neither hunk is applied.** Hunk (a) has two prerequisites nobody has provisioned: a runner-local weights mount (`/srv/models` in the diff is a placeholder), and a way for `uv run --with huggingface_hub --with safetensors --with numpy` to resolve inside the network-less `sovereign-ci:stable` clean-room — a warm uv-cache mount or those packages baked into the image. Wiring it blind produces a step that dies at the weight-hash check before running a single test: a red run that proves nothing.

2. **The gating work is the missing x86_64 measurement, and it is this phase's own open item, not new scope.** Run `just chronos-gate` ONCE on an x86_64 Linux host with the weights (lambda-vector qualifies and is pre-authorized compute per CLAUDE.md). Read the printed `f32_quantile_bar` line and the measured max|delta| from `peyton_ladder_matches_oracle_f32`, then tighten `quantiles_abs_f32_nonaarch64` in `contracts/chronos-bolt-parity-v1.yaml` from the PROVISIONAL-UNMEASURED `5.0e-6` to measurement + margin, as a `pv diff`-visible contract edit. That single run discharges REVIEW-06-02 and windows-ledger entry #4 together, and only then is there evidence to wire a CI leg onto.

3. **If no x86_64 host is reachable within this phase, this collapses to `defer`** — by this plan's own option table. It must then be recorded as `defer` explicitly, with both hunks preserved in `06-ci-chronos-step.patch` and **D-18 clause 2 named as an open CI gap** evidenced locally by `just chronos-gate` and `06-EVIDENCE.md` §4 only. It must not become a silent drop.

The exact open-item text for 06-09 is item 4 in `deferred-items.md`. Hunk (b) (`cargo test -p aprender-mcp-forecast --test e2e_stdio` on the single Integration-tests line) needs no runner provisioning and can be taken independently at any time; it was not split out here because the decision was to gate on the measurement first.

## Decisions Made

Beyond the CI decision above:

- **The synthetic and the real SC1 numbers are both recorded.** The recipe's asserted bar uses `--bench 1000 3000`'s generated series (0.212 s). Real 2 905-point Peyton fits are 1.5-3x slower (0.31-0.65 s across five windows). Reporting only the synthetic number would have been the flattering half of a two-sided measurement.
- **The coldstart's 91 ms cold-cache first spawn is kept beside the 53 ms median.** That first number is what a Lambda cold start actually sees; a median alone would have hidden it.
- **5.162x is stated as a different measurement from 06-06's 2.070x, not as an improvement.** Release + real Peyton vs debug + synthetic. 06-06's `1.002x` single-router control is what makes either number evidence rather than an assertion.
- **The evidence file records the uncommitted working-tree delta's sha256 beside the commit hash.** A commit hash alone would have described code that was not what ran.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] The justfile header comment tripped the plan's own anti-pattern guard**

- **Found during:** Task 1 (justfile acceptance criteria)
- **Issue:** The header comment explaining the "never read `$?` through a pipe" rule spelled the anti-pattern out literally, and the plan's acceptance grep `! grep -Eq '\| *(grep|tail|awk)[^;]*; *rc=\$\?' justfile` matched the *documentation of the defect* as though it were the defect.
- **Fix:** Reworded the comment to describe the anti-pattern rather than spell it, with a note saying why. The rule is still documented; the guard is no longer self-tripping.
- **Files modified:** `justfile`
- **Verification:** guard re-run — no match; `rc=$?` count still 10.
- **Committed in:** `adc8a560a`

**2. [Rule 1 - Bug] bashrs found four constructs that made the recipes harder to lint than to read**

- **Found during:** Task 1 (`bashrs lint` on the extracted recipe bodies, per CLAUDE.md's bashrs-not-shellcheck rule)
- **Issue:** 15 SC1100 "unicode dash" errors from em dashes inside shell strings; SC1020 from `${bytes//[[:space:]]/}` parsed as an unterminated `[` test; SC1099 from `${pair##*|}`; SC2065 from `->` inside an echo string.
- **Fix:** em dashes inside recipe bodies replaced with `-`; `bytes=$(( $(wc -c < ...) ))` (which also proves the value is an integer); the `label|log` pair loop replaced with a named `check_summary` function taking two arguments (clearer, and it also removed two SC2125s and a PERF002); `->` written as `to`.
- **Files modified:** `justfile`
- **Verification:** `bashrs lint` on all seven extracted bodies: **0 errors** (was 16). Remaining warnings are demonstrable false positives on assignments (SC2046/SC2047/SC2198) plus one intended in-loop command substitution (PERF002).
- **Committed in:** `adc8a560a`

### Accepted (not fixed — scope boundary)

**3. The plan's Task 1 clippy verify cannot pass on this host, for reasons in a different crate**

`cargo clippy -p aprender-forecast --all-targets -- -D warnings` exits 101 with **18 errors, all in `crates/aprender-compute/` and none in the crate being linted**. Not auto-fixed: the deviation scope boundary is explicit that pre-existing lint failures in unrelated files are out of scope.

Proven pre-existing rather than assumed, with two controls:
- the identical command with `mase_rolling_origin.rs` removed from the tree produces the **identical 18-finding set** (only emission order differs);
- `cargo clippy -p aprender-forecast --example mase_rolling_origin -- -D warnings` reports the same 18, again none in `aprender-forecast`.

Mechanism: a trailing `-- -D warnings` becomes `CLIPPY_ARGS`, which clippy applies to every locally-compiled crate, not just the selected package. **The example itself is clippy-clean** — that is what the two controls establish. Logged as `deferred-items.md` item 2 with the full finding list, because three of the 18 are `unreachable expression` after aarch64-only `return`s and are therefore structurally invisible to the X64 CI — the same shape as CLAUDE.md #2370.

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug), 1 accepted out-of-scope.
**Impact on plan:** No scope creep. Both auto-fixes hardened the artifact the plan asked for. The accepted item is a pre-existing workspace condition that the plan's verify surfaced.

## Issues Encountered

**1. `cargo clippy -- -D warnings` is red workspace-wide on aarch64 macOS (pre-existing).** 18 findings in `aprender-compute`: three `unreachable expression` after aarch64-gated `return`s, nine `dead_code`, three `unused_imports`, two `unused_variables`. Consequence worth stating plainly: **`make tier1` / `make tier2` cannot pass on an aarch64 macOS dev box today**, and because three of the findings are arch-conditional, the X64 CI's green says nothing about them. Full list in `deferred-items.md` item 2. Whoever picks this up should fix `aprender-compute` rather than add `#[allow]`s, and re-run on both arches.

**2. Substantial uncommitted phase-06 work was already in the tree (pre-existing).** At this plan's start the working tree carried 18 uncommitted paths (14 source/doc + 3 `.pv/` artifacts, 510 insertions / 164 deletions) concentrated in exactly the crates this plan measured — a `MAX_SPAN_DAYS` bound, cross-model option refusals, a `MAX_POOL` ceiling. They were present in the session-start `git status`, so they are neither this plan's nor 06-07's; `git show HEAD:crates/aprender-forecast/src/forecast.rs` has no `MAX_SPAN_DAYS`.

06-08 did not commit them (out of scope) but **every measurement in `06-EVIDENCE.md` ran against them**, which is why that file records the delta's sha256 (`bd42a46dda0f0c660cc450b1c97d50f2c017d9111a6164a73272793b7dbcb2b6`, source-only, 59 848 bytes) beside the commit hash. **Decision: investigate, then commit — 06-09 must establish that delta's provenance and intent BEFORE committing it**, so the evidence numbers become reproducible from a real commit rather than from a commit-plus-delta. The delta digest is the honest record until then, not a substitute for landing the work. Recorded as `deferred-items.md` item 5.

**3. `FALSIFY-BOUNDARY-011` is now unblocked.** `deferred-items.md` previously deferred it because "`just forecast-pool-ratio` **does not exist**". It exists and passes. Re-arming it is a contract-side edit to `contracts/forecast-tool-boundary-v1.yaml` — outside this plan's `files_modified` and in a file already carrying uncommitted edits, so it is recorded as item 1 for 06-09, with the caveat that the bar is host-gated and the contract must say so.

## User Setup Required

None — no external service configuration required.

The two demo servers are **left running deliberately** for the end-of-phase human check and must not be shut down:

- `aprender-mcp-forecast --http 8787 --pool 8` → http://127.0.0.1:8787/
- `aprender-mcp-chronos --http 8788` (embedded tiny-f16, banner confirms `F16, embedded`) → http://127.0.0.1:8788/

Both return HTTP 200 with `tools/call` in the body and answer MCP `initialize` over `/mcp`; a live `tools/call` on :8788 returned a 12-step forecast with quantiles. What remains for the human is the visual judgement in coverage entry D9.

## Next Phase Readiness

**Ready for 06-09.** It inherits five explicit open items, all in `deferred-items.md`:

1. Re-arm `FALSIFY-BOUNDARY-011` in `contracts/forecast-tool-boundary-v1.yaml`, noting the host gate.
2. `aprender-compute`'s 18 clippy findings — a real defect, arch-conditional, invisible to CI.
3. / 5. Establish the provenance of the uncommitted delta, **then** commit it, so `06-EVIDENCE.md` becomes reproducible from a commit alone.
4. The CI item: one x86_64 `just chronos-gate` run → tighten `quantiles_abs_f32_nonaarch64` → then decide on the patch. If no x86_64 host is reachable this phase, record `defer` explicitly and name D-18 clause 2 as an open CI gap.

**No blockers.** SC1, SC4 and SC5 are measured and passing; D-16 ships; D-18 clause 2 holds locally with recorded evidence and is a named, dated CI gap rather than an assumption.

## Self-Check: PASSED

- `crates/aprender-forecast/examples/mase_rolling_origin.rs` — FOUND
- `.planning/phases/06-native-time-series-forecasting-stack/06-EVIDENCE.md` — FOUND
- `.planning/phases/06-native-time-series-forecasting-stack/06-ci-chronos-step.patch` — FOUND
- commit `adc8a560a` — FOUND
- commit `1e2c410b9` — FOUND
- `git apply --check` on the patch — rc 0
- `git diff --stat HEAD -- .github` and `git diff --stat 59a65367b..HEAD -- .github` — both empty
- `just --list` — rc 0, all seven recipes present
- `grep -c 'rc=$?' justfile` — 10 (>= 6); no `rc=$?` after a pipe
- `cargo build --examples -p aprender-forecast` — rc 0
- all six bars re-checked against their recipe logs under `target/`

---
*Phase: 06-native-time-series-forecasting-stack*
*Completed: 2026-09-06*
