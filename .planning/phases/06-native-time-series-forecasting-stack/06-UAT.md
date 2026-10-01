---
status: complete
phase: 06-native-time-series-forecasting-stack
source: [06-VERIFICATION.md]
started: 2026-09-07T20:16:59Z
updated: 2026-09-07T23:37:23Z
---

## Current Test

[testing complete]

## Tests

### 1. Demo pages: MCP handshake + chart rendering
expected: Both pages complete the MCP handshake same-origin under /mcp and render a band chart.
result: pass
first_result: issue (FIXED AND RE-CONFIRMED IN THE SAME SESSION)
reported: |
  "Validation error: n_lags is neuralprophet-only; set model to \"neuralprophet\""
  "Validation error: growth is prophet-only; set model to \"prophet\""
severity: blocker
resolved_by: "demo-page arm-scoped args fix (this session); user confirmed both models render"
resolved_at: "2026-09-07"
tested_on: http://localhost:8770 (built from HEAD 22e85dd3f; the servers already running on
  8765/8766 were v0.0.0 spike binaries from Sep 4-5 and 8787/8788 were pre-round-3 Sep 6
  builds — 8787 ACCEPTED a 201-byte holiday name that HEAD refuses, so none of them was
  testing phase-06 code)
how: `cargo run -p aprender-mcp-forecast -- --http 8770` and `cargo run -p aprender-mcp-chronos -- --http 8771`, open each demo page, drive initialize -> tools/list -> tools/call. CHECK FOR STALE LISTENERS FIRST (`lsof -nP -iTCP:PORT -sTCP:LISTEN`) — four stale servers were occupying the documented ports and answered `initialize` with HTTP 200.
why_human: CARRIED FORWARD, still open. 06-08 declared this a human end-of-phase check (visual rendering + a real browser MCP client). No automated test covers the pages; re-confirmed at HEAD.

### 2. Chronos x86_64 parity measurement replaces the PROVISIONAL tolerance
expected: A measured max|delta| replaces the PROVISIONAL 5.0e-6 headroom value in contracts/chronos-bolt-parity-v1.yaml.
result: pass
measured: |
  Run on AWS EC2 x86_64, Amazon Linux 2023, 2026-09-07. All 8 tests passed.
    f32 quantile bar: chronos-bolt-parity-v1.quantiles_abs_f32_nonaarch64 = 5e-6 (ARCH=x86_64)
    peyton: quantiles_64 max|delta| 9.5367e-7 against bar 5e-6
  The printed ARCH=x86_64 proves the non-aarch64 branch ENGAGED rather than was assumed
  (CLAUDE.md rule 2).
outcome: |
  Bar tightened 5.0e-6 -> 2.0e-6, version 1.0.0 -> 2.0.0 on `pv diff`'s MAJOR verdict.
  `pv validate` rc=0; local aarch64 ladder re-run 8/8 (its own 1.0e-6 bar untouched);
  `make contract-audit-phase6` rc=0, 63 rows, zero BIND-.
  THE 5.0e-6 PREMISE WAS FALSIFIED: the headroom existed for a different GEMM accumulation
  order on x86_64, but the delta is IDENTICAL to aarch64's spike-005/007 value (9.54e-7)
  because 9.5367431640625e-7 is exactly 2^-20 — a quantization quantum of the f32 output,
  not accumulated error. A quantum does not vary with accumulation order.
  NOT NARROWED TO THE MEASUREMENT (1.0e-6 would also have passed at 95.4%): one run on one
  instance type is one sample, and fitting a bound from one side is the CR-01 error.
incidental_finding: |
  The same run put `probe five_points` at 3.815e-6 against its 4.0258e-6 bar — 94.8% — and
  `probe tiny_len_130_rollout` at 80.6%, while the peyton bar this item is about ran at 19.1%.
  If an x86_64 microarchitecture ever moves these numbers, five_points fails FIRST and this
  equation fails LAST. Nobody asked about five_points; it is the real margin risk in the ladder.
how: |
  On an x86_64 host, from the repo root:
    just chronos-gate            # fetches+verifies pinned weights, then runs BOTH armed suites
  or, to see the printed deltas directly:
    just fetch-chronos-tiny
    CHRONOS_MODEL_DIR="$PWD/models/chronos-bolt-tiny/f32" \
      cargo test -p aprender-forecast --lib -- --nocapture bolt::parity chronos::parity
  Then read the printed max|delta| and tighten `quantiles_abs_f32_nonaarch64` in
  contracts/chronos-bolt-parity-v1.yaml in a `pv diff`-visible edit.
  CORRECTED 2026-09-07: the command originally written here passed TWO positional filters
  before `--` and fails with `error: unexpected argument 'chronos::parity' found` (verified).
  cargo test takes ONE positional; both filters must follow `--`. The justfile had it right.
why_human: CARRIED FORWARD, still open. This box is aarch64 (arm64); every CI runner is X64 with no Chronos leg. D-ITEM-06-04 / REVIEW-06-02.

### 3. Decide whether SC4's Chronos ladder staying DARK in CI is acceptable
expected: An explicit decision recorded against D-ITEM-06-03.
result: DECIDED 2026-09-07 by Guy — ACCEPT DARK, and say so explicitly.
decision: |
  SC4's Chronos parity ladder stays a MANUAL local gate (`just chronos-gate`). It is not
  wired into `.github/workflows/ci.yml` and is not expected to be.

  The obligation that comes with accepting this: the wording everywhere must say MANUAL out
  loud, so a reader never mistakes a recorded green for a running gate. `status: implemented`
  in binding.yaml means "the gate exists and passes when run", NOT "the gate runs on every
  push". That distinction is the whole content of this decision — an unstated dark gate is
  indistinguishable from a gate nobody noticed stopped running.
follow_on: |
  - Record against D-ITEM-06-03.
  - Amend the `chronos-gate` recipe header and the binding.yaml row wording to state MANUAL.
  - Any future claim that SC4 is "enforced" must name the manual command, not CI.
how: Decide whether the ci.yml embedded-weights leg should now be applied, or the ladder stays a manual local gate.
why_human: CARRIED FORWARD, still open. `.github/workflows/ci.yml` contains ZERO `chronos` or `forecast` matches; every SC4 parity claim rests on a manual local gate.

### 4. Decide the disposition of CR-01 (MAX_NP_TRAIN_COST over-refuses in-spec requests)
expected: A bound whose VALUE is resolved per deployment envelope, with the accepted region written down and asserted.
result: DECIDED 2026-09-07 by Guy — option (c), neither (a) nor (b).
decision: |
  "Make the limits such as CR-01 more flexible as some algorithms might have bigger
  requirements."

  This supersedes the two options previously offered (raise to ~18M, or narrow the
  advertised n_lags). Both assumed one hard constant was correct and only its value was
  in question. The correct frame is that there is no single right number, because the
  product ships four deployment envelopes that differ by orders of magnitude:
  CloudFlare Workers WASM (tightest, possibly below our requirements), AWS Lambda,
  Docker on GCP/Azure, and customer-hosted pmcp.run.

  FLEXIBLE IS NOT UNBOUNDED. Round 3's whole thesis — every cost axis enumerated, every
  bound enforced at the door, no axis pending — is preserved. What changes is only where
  the VALUE comes from: from a hard `pub const` to a resolved limits profile with today's
  constants as the default. The invariant strengthens rather than weakens:
    before: "cost axis C-08 is bounded at 15_000_000"
    after:  "cost axis C-08 is always bounded; its value comes from the resolved profile;
             no profile can disable a bound or set it above the tier's structural maximum"
follow_on: |
  Converts into a gap-closure/next-phase plan, not a one-line constant edit:
    - the 13 `pub const MAX_*` in types.rs become a `DoorLimits` policy struct with a
      `Default` equal to today's values
    - contracts/forecast-tool-boundary-v1.yaml `constants:` becomes the DEFAULT profile;
      `types::tests::cost_bounds_match_contract` keeps pinning the default
    - named tier profiles whose ceilings come from each envelope's real structural maximum
    - the accepted region gains a test at every profile — the gap CR-01 actually exposed

### 5. Decide whether the forecast/chronos gates get an automatic surface
expected: Either the wiring lands, or the recipe headers and binding.yaml's `sc1_wall_swept ... status: implemented` note say explicitly that the gate is MANUAL, so `implemented` is not read as `running`.
how: Decide whether `just forecast-sc1-sweep` (and chronos-bench / chronos-coldstart / forecast-pool-ratio / forecast-holiday-bench) belong in a `make` tier or a scheduled workflow shaped like toolchain-ceiling.yml.
why_human: Every round-3 plan fences off .github/workflows/*.yml and states that CI wiring is a human decision. No reference to forecast-sc1-sweep exists in .github/, Makefile or scripts/. The in-suite sc1_wall_sweep that CI does run is a debug build and returns before the 2 s assertion by design.
result: DECIDED 2026-09-07 by Guy — wire into `make tier3`.
decision: |
  `just forecast-sc1-sweep` now runs as `make forecast-sc1-gate` inside tier3 (commit 6ce4ff7d8),
  wired per the Makefile's own evidence convention rather than by appending a line: standalone
  first with rc captured off the command (rc=0, 25 s warm, 19 compositions), then a failure
  INDUCED and OBSERVED — the bar lowered 2.0 -> 1.0 made the neuralprophet row fail at 1.579 s
  while every prophet composition (0.156-0.843 s) still PASSED, and the target exited 2. Bar
  restored, re-run green.
  NOT wired into CI: item 3 keeps Phase 6's release-profile gates manual/pre-push. The gate
  FAILS LOUDLY if `just` is absent rather than skipping.
  Recorded margin: the worst composition measured 1.420 s then 1.579 s on the SAME geometry —
  11.2% run-to-run spread, 21-29% headroom under the 2 s bar.

## Summary

total: 5
passed: 2
decided: 3
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

- truth: "The forecast demo page completes initialize -> tools/list -> tools/call and renders a band chart"
  status: resolved
  resolved_by: "crates/aprender-mcp-forecast/static/index.html — run() builds args per arm; syncModelKnobs() disables off-arm controls"
  resolved_at: "2026-09-07"
  verified: "curl on :8770 (HEAD build) returns a banded forecast for BOTH prophet and neuralprophet; user confirmed the page renders"
  correction: "The original diagnosis named only `growth` for the neuralprophet arm. TWO prophet-only fields were sent unconditionally — `growth` AND `seasonality_mode`. The door returns on the first offender, so the message only ever showed one."
  still_open: "Door policy (D-11) is NOT settled by this fix: should an off-arm option carrying its own DEFAULT (n_lags:0, growth:\"linear\") be refused, or only a non-neutral value? Deferred to Phase 7 — it bears on any client that serializes a full settings object." 
  reason: |
    User reported: 'Validation error: n_lags is neuralprophet-only; set model to "neuralprophet"'
    and 'Validation error: growth is prophet-only; set model to "prophet"'. Reproduced by curl
    against a HEAD build on :8770 for BOTH model values — the page cannot complete a single
    tools/call for either model.
  severity: blocker
  test: 1
  root_cause: |
    The page and the door disagree, and the disagreement is total rather than partial.
    `crates/aprender-mcp-forecast/static/index.html:49` builds `args` UNCONDITIONALLY, always
    including both `growth` (prophet-only) and `n_lags` (neuralprophet-only). The door
    (`crates/aprender-forecast/src/forecast.rs:112-129`) refuses an off-arm option on PRESENCE
    (`is_some()`), by design (D-11: never give the caller a plausible answer to a question they
    did not ask). So model=prophet is refused for carrying `n_lags`, and model=neuralprophet is
    refused for carrying `growth`. There is no selection that works.
    TIMELINE: the page was written 2026-09-05 in 06-01 (725d5c7b2) and has NEVER been modified
    since; the arm-scoping refusals landed 2026-09-06 in 06-09 (1d170f383) and were extended in
    06-10 (9d6eb639e). The page has therefore been broken for four plans and three gap-closure
    rounds, undetected because no automated test drives it — which 06-VERIFICATION.md states
    explicitly ("No automated test covers the pages").
    OPEN DESIGN QUESTION, not a settled page bug: the door refuses `n_lags: 0` and
    `growth: "linear"` — each field's NEUTRAL value, which is what the page's untouched form
    controls emit. Refusing a field carrying its own default is the same over-refusal class as
    CR-01 (a bound whose ACCEPTED region was never exercised). Fixing only the page leaves every
    other client that sends a fully-populated payload with defaults hitting the same wall.
  artifacts:
    - path: "crates/aprender-mcp-forecast/static/index.html"
      issue: "line 49 sends both growth and n_lags regardless of the selected model"
    - path: "crates/aprender-forecast/src/forecast.rs"
      issue: "lines 112-129 refuse an off-arm option on presence, including at its default value"
    - path: "crates/aprender-mcp-chronos/static/index.html"
      issue: "not yet tested — chronos server still building at time of report"
  missing:
    - "Decide door policy: refuse on presence (status quo), or refuse only when the off-arm option is set to a NON-default value"
    - "Send only the selected model's applicable knobs from the demo page"
    - "An automated test that drives each demo page's tools/call payload, so this cannot regress silently again"
  debug_session: ""
