---
phase: 06-native-time-series-forecasting-stack
plan: 17
subsystem: testing
tags: [forecasting, prophet, poisson-sampler, falsification, provable-contracts, wr-01, wr-02, in-02, in-04, semver]

requires:
  - phase: 06-14
    provides: "constants.poisson_normal_branch_lambda (WR-01's ownership half) and test_support::constant_f64 — the contract mirror this plan makes DETECTABLE"
  - phase: 06-15
    provides: "the closed declared-red window, so this plan's gates run against a crate that is green with no --skip"
  - phase: 06-16
    provides: "just forecast-sc1-sweep and scripts/assert_measurement_under.sh — two of the twelve gates in this plan's closing sweep"
provides:
  - "test_support::equation_float — a contract reader keyed on the KEY as well as the equation, so one equation can own more than one bar"
  - "prophet::sampler::poisson_mean_and_variance_track_lambda_across_its_whole_domain — the sampler's domain check now bars its SPREAD and its LOW TAIL, not only its centre"
  - "equations.poisson_sampler_domain.variance_tolerance (0.05) and .zero_mass_tolerance (0.22), with the sigma arithmetic for all three bars at all seven sweep points written into the contract"
  - "FALSIFY-PROPHET-014 — the two implementations that pass FALSIFY-PROPHET-013 and fail this"
  - "a TOTAL feature_row holiday lookup (hol_sets.get) — a mismatched slice is a miss, not an out-of-bounds panic in library code"
  - "a debug_assert_eq! making bolt::transpose's expect-safety claim checkable, proved non-vacuous by mutation"
  - "crates/aprender-forecast/README.md ## Public API changes — the two breaking 0.63.0 surface changes"
  - "the round's closing evidence sweep: twelve gates, every rc read from a redirect"
affects: [phase-6-verification, ship]

actuals:
  tokens: 14434
  tasks: 3
  commits: 7
plan_head_before: 4ef2b5e5b747c01338e126a98e05fdeabaf172cd

tech-stack:
  added: []
  patterns:
    - "A falsification test must bar every statistic its own codomain promises — one statistic is not a domain check"
    - "Never set a statistical bar without writing down its sigma AT EVERY POINT THE BAR IS APPLIED; adding a sweep point weakens every bar at that point"
    - "When a defect gap and a noise floor do not leave a window for a single shared bar, raise N rather than widen the bar"
    - "Make an unchecked safety claim CHECKABLE (debug_assert) rather than widening a published API to Result"

key-files:
  created: []
  modified:
    - crates/aprender-forecast/src/prophet.rs
    - crates/aprender-forecast/src/test_support.rs
    - crates/aprender-forecast/src/bolt.rs
    - crates/aprender-forecast/README.md
    - contracts/prophet-parity-v1.yaml
    - contracts/aprender/binding.yaml

key-decisions:
  - "N raised 20 000 -> 60 000 because the two new bars are JOINTLY INFEASIBLE at 20 000: the zero-mass window [0.3434, 0.2478] is empty there. The sample size, not the bar, was the thing to move"
  - "The review's suggested 1-2 % variance bar is REFUSED, not widened: 2 % is 1.91 sigma at N = 20 000, and raising N to 80 000 to rescue it still gives only 3.70 sigma at the lambda = 3 this plan adds"
  - "variance_tolerance 0.05 (weakest 8.02 sigma) and zero_mass_tolerance 0.22 (weakest 4.44 sigma); the weakest bar SHIPPED is the pre-existing MEAN bar at lambda = 3 (4.24 sigma), which is the second reason N had to rise"
  - "feature_row's signature is NOT reverted — the lookup is made total instead. Reverting re-opens 06-11's measured design-build fix, and the review records the hazard as pre-existing"
  - "bolt::transpose keeps its `expect` and does NOT become `Result`: a breaking public-API change for a condition every in-crate caller already satisfies"
  - "prophet-parity-v1 bumped 1.1.0 -> 2.0.0 on `pv diff`'s own MAJOR verdict — strengthening an equation narrows what conforms, and the zero-variance stub is exactly a thing that conformed before and does not now"

patterns-established:
  - "Pattern: observe the RED by substituting the defect implementation, and record which bars fired and which did NOT — the juxtaposition is the finding, not the failure"
  - "Pattern: a non-vacuity assertion pinning the number of points a conditional bar actually ran at, so N/LAMBDAS/threshold cannot quietly stop it running"

requirements-completed: [SC1]

coverage:
  - id: D1
    description: "WR-02 closed: the sampler's falsification test bars its VARIANCE with a contract-owned tolerance, so a zero-variance stub no longer passes the test that names itself the sampler's domain check"
    requirement: SC1
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/prophet.rs#prophet::sampler::poisson_mean_and_variance_track_lambda_across_its_whole_domain"
        status: pass
      - kind: unit
        ref: "OBSERVED RED: poisson stubbed to `lambda.round() as usize` -> exit 101, variance rel=1.000000 at all seven lambdas, mean rel=0.000000 at all seven; restored -> exit 0 (.tdd-evidence/06-17-t1-red.json, RED_EVIDENCE_OK)"
        status: pass
    human_judgment: false
  - id: D2
    description: "WR-01's second half closed: the sweep can now DETECT POISSON_NORMAL_BRANCH_LAMBDA being lowered, via the zero mass at the sub-threshold lambdas"
    requirement: SC1
    verification:
      - kind: unit
        ref: "OBSERVED RED: threshold + its contract mirror lowered to 3.2 -> exit 101, zero-mass rel=2.284878 at lambda 5 (p_zero 0.022133 vs exact 0.006738) while every mean stayed inside 0.01 and the variance bar stayed green; both restored byte-identically (cmp clean)"
        status: pass
      - kind: unit
        ref: "cargo test -p aprender-forecast --lib -- prophet::sampler types::tests::cost_bounds_match_contract (3 passed, 0 failed after restore)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Every shipped bar is at least 4 standard errors from its own sampling noise AT EVERY POINT IT IS APPLIED, with the arithmetic in the contract rather than only in this SUMMARY"
    requirement: SC1
    verification:
      - kind: unit
        ref: "contracts/prophet-parity-v1.yaml equations.poisson_sampler_domain.invariants (six invariants carrying the per-point sigma for all three bars); weakest shipped bar is the MEAN at lambda 3, 4.24 sigma"
        status: pass
      - kind: unit
        ref: "pv validate contracts/prophet-parity-v1.yaml -> 0 error(s), 0 warning(s)"
        status: pass
    human_judgment: true
    rationale: "The arithmetic was re-derived independently and agrees with the plan's to three digits, and the tests pass — but whether 4 sigma is the RIGHT floor, and whether 0.05 / 0.22 are the right points inside their feasible windows, is a judgment about how much sampling risk this gate should carry. A human should sign off on the floor, not just on the arithmetic."
  - id: D4
    description: "IN-02 closed as narrowed: feature_row's holiday lookup is total, so a mismatched slice is a miss rather than an out-of-bounds panic in library code, and the signature is NOT reverted"
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/prophet.rs#prophet::feature_row_totality::a_short_hol_sets_is_a_miss_and_never_an_out_of_bounds_panic"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/prophet.rs#prophet::feature_row_totality::a_correctly_built_hol_sets_still_hits_exactly_the_same_days"
        status: pass
      - kind: unit
        ref: "PRE-CHANGE PANIC OBSERVED: `index out of bounds: the len is 0 but the index is 0` at prophet.rs:205:19, exit 101"
        status: pass
      - kind: unit
        ref: "cargo test -p aprender-forecast --lib prophet::parity (32 passed; 0 failed, before and after)"
        status: pass
    human_judgment: false
  - id: D5
    description: "IN-04's two substantive items closed: bolt::transpose's expect-safety claim is checked by a debug_assert_eq!, and both breaking 0.63.0 surface changes are documented where a consumer would look"
    verification:
      - kind: unit
        ref: "crates/aprender-forecast/src/bolt.rs:287 debug_assert_eq!(w.len(), out * inp, ..) immediately before trueno::blis::transpose"
        status: pass
      - kind: unit
        ref: "PROVED NON-VACUOUS: assert mutated to `out * inp + 1` -> 2 always-run tests red naming bolt.rs:287 (single_row_routing_is_dot8_at_production_defaults, multi_row_routing_is_gemm_blis_at_production_defaults); restored -> 115 passed / 0 failed"
        status: pass
      - kind: unit
        ref: "grep -c 'Public API changes' crates/aprender-forecast/README.md -> 1"
        status: pass
    human_judgment: false
  - id: D6
    description: "The round is closed on evidence: twelve gates green in ONE recorded sweep, every rc read from a redirect and never through a pipe"
    requirement: SC1
    verification:
      - kind: integration
        ref: "the twelve-row gate table below; every rc 0, no gate recorded as a known issue"
        status: pass
    human_judgment: false
  - id: D7
    description: "All nine 06-REVIEW.md findings resolve to a symbol, file or contract key IN THE TREE, each with a command that proves it; the two rejected items carry their rationale"
    verification:
      - kind: integration
        ref: "the nine-row ledger below, each row verified against the tree rather than against the table"
        status: pass
    human_judgment: true
    rationale: "Each row resolves to a real symbol and a passing command, which is mechanical. Whether a finding is genuinely CLOSED — as opposed to merely having something in the tree that gestures at it — is the judgment three prior rounds got wrong, and it is the judgment this round is asking to be trusted on. A human should read the ledger, not just its green."
  - id: D8
    description: "No parity rung moved across the round and no fixture or existing tolerance was edited"
    verification:
      - kind: unit
        ref: "prophet::parity 32 passed / 0 failed; np::parity 9 passed / 0 failed (unmoved from 06-15's recorded 9)"
        status: pass
      - kind: unit
        ref: "git diff ce3e5a8ea..HEAD -- crates/aprender-forecast/tests/fixtures/ crates/aprender-mcp-forecast/fixtures/ -> EMPTY; the only deleted line in contracts/prophet-parity-v1.yaml across the whole review range is `version: 1.0.0`"
        status: pass
    human_judgment: false

duration: 31 min
completed: 2026-09-07
status: complete
---

# Phase 6 Plan 17: One statistic is not a domain check Summary

**The sampler's falsification test now bars its VARIANCE and its LOW-TAIL ZERO MASS as well as its mean — both contract-owned, both at least 4 sigma from their own sampling noise — and the two implementations the review demonstrated (a zero-variance stub, and the branch threshold lowered to 3.2) have each been OBSERVED turning it red, closing gap-closure round 3 with twelve gates green in one recorded sweep.**

---

## ⚠️ READ FIRST — THE ROUND'S CENTRAL GATE IS WIRED TO NOTHING

**`just forecast-sc1-sweep` is invoked by no CI job, no `make` tier and no git hook.** It is the gate 06-16 built to close WR-04, it is the only thing in this repository that sweeps the axis CR-01 actually lived on, and it was observed failing on the exact CR-01 configuration — and today nothing runs it but a human typing it.

That is one step away from the very posture WR-04 exists to fix. WR-04's finding was not "the harness is wrong"; it was that `logistic_band_wall` was `#[ignore]`d with no recipe and no bar, so **nothing could reach it**. A recipe nobody invokes is the same defect with a different spelling. The Makefile says so in its own words, three lines above where this would go: *"a target outside the tiers is a target that stops being run."*

All four round-3 plans fence off `.github/workflows/*.yml`, and CLAUDE.md requires a human check-in before modifying CI — so **not wiring it was correct**, and this is a decision to take, not a defect to report.

**The recommended wiring, so the decision is one approval and not a research task:**

Add one line to `tier3` in the `Makefile`, immediately after **line 380** (`@$(MAKE) contract-audit-phase6`), joining the run of gates that already carry the "a target outside the tiers stops being run" comment block:

```make
	@echo "Sweeping the SC1 wall across freq x growth x holiday (WR-04)..."
	@just forecast-sc1-sweep
```

**The cost, measured rather than estimated: 24 s wall on a warm release build** (19 compositions, `rc=0`), against tier3's stated 1-5 minute budget. Cold, it additionally pays one release build of `aprender-forecast`.

Follow the protocol the neighbouring lines document and that this repository has been burned for skipping: run it standalone first with the status captured directly (never through a pipe), then **induce a failure, observe it, and revert it** before wiring — 06-16 already has the induction recipe (raise `MAX_LOGISTIC_CHANGEPOINT_LAMBDA` and its mirror to `200_000`, set `SC1_SWEEP_HORIZON=3650`, watch the two `MS` rows wall at 2.4 s).

`tier3` is the pre-push gate, not CI, so this does not touch `.github/workflows/` at all and needs no workflow edit. If the 24 s is judged too expensive for every push, the alternative is a scheduled workflow like `toolchain-ceiling.yml` — but that is a second decision, and the `tier3` line is the cheap one.

---

## Performance

- **Duration:** 31 min
- **Started:** 2026-09-07T18:03:55Z
- **Completed:** 2026-09-07T18:34:32Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- **WR-02 closed.** The variance bar refuses the zero-variance stub, which previously passed outright at every sweep point. Observed, not argued.
- **WR-01's second half closed.** The zero-mass bar at the sub-threshold lambdas detects the branch threshold being lowered — the one thing the mean, and as it turns out the variance too, cannot see.
- **The sample size was the thing to move, not the bar.** The two new bars are jointly infeasible at N = 20 000; the plan's arithmetic was re-derived independently and agrees to three digits.
- **IN-02 and IN-04's two substantive items closed** without reverting a measured fix or widening a published API.
- **Twelve gates green in one recorded sweep**, and all nine review findings resolved against the tree.

## Task Commits

1. **Task 1 RED** — `0222fe60d` (test) — bar the sampler's variance and zero mass; `equation_float`; the two contract keys
2. **Task 1 GREEN** — `93b019ef8` (feat) — the contract owns both bars, with their sigma arithmetic; FALSIFY-PROPHET-014
3. **Task 2** — `65364694e` (fix) — total `feature_row` lookup, `bolt::transpose` debug_assert, README API notes
4. **Task 3** — `b27f66330` (docs) — binding row count reconciled by counting; prophet-parity 1.1.0 -> 2.0.0

**Plan metadata:** the `docs(06-17)` commit carrying this file, STATE.md and ROADMAP.md.

## TDD Gate Compliance

`workflow.tdd_mode` is `false` in `.planning/config.json`, but Task 1 carries `tdd="true"` and was executed through the full RED -> GREEN cycle.

| Gate | Commit | Evidence |
|---|---|---|
| **RED** | `0222fe60d` `test(06-17)` | `RED_EVIDENCE_OK` / `target_test_failed`, exit 101, target `prophet::sampler::poisson_mean_and_variance_track_lambda_across_its_whole_domain`, 1 of 2 tests failing on a variance assertion about the planned behaviour |
| **GREEN** | `93b019ef8` `feat(06-17)` | sampler 2 passed / 0 failed; `prophet::parity` 32 / 0 |
| **REFACTOR** | — | none needed |

Record committed at `.tdd-evidence/06-17-t1-red.json`, using the libtest -> node-test-TAP projection 06-14/06-15/06-16 established (the classifier parses node-test TAP; Rust emits libtest). Counts are read off libtest's own `test result:` line by the script, never asserted by hand; a run with no `test result:` line is a hard error rather than an assumed pass.

**A note on the SHAPE of this RED, stated rather than glossed.** This task strengthens a TEST; the sampler under it was already correct. So the RED could not be "the implementation is missing" — it had to be "the DETECTION is missing", which is demonstrated by substituting the defect implementation. Recorded honestly: the pre-strengthening run against the stub (`rc=0`, `rel=0.000000` at all six lambdas — the defect) is in `/tmp/p06-17-stub-vs-old-test.log`, and the post-strengthening run against the same stub (`rc=101`) is the RED record. A reader should know this is a mutation-based RED, not a missing-implementation RED.

## The seven-point sweep — the shipped GREEN run

`cargo test -p aprender-forecast --lib prophet::sampler -- --nocapture`, `rc=0`:

```
POISSON MEAN: lambda=3.0 n=60000 mean=2.9975 rel=0.000839 (bar 1e-2)
POISSON VAR:  lambda=3.0 n=60000 var=3.0008 rel=0.000276 (bar 5e-2)
POISSON ZERO: lambda=3.0 CHECKED expected_zeros=2987.2 (>= 30) observed=2951 p_zero=0.049183 exact=0.049787 rel=0.012126 (bar 2.2e-1)
POISSON MEAN: lambda=5.0 n=60000 mean=4.9965 rel=0.000693 (bar 1e-2)
POISSON VAR:  lambda=5.0 n=60000 var=4.9964 rel=0.000712 (bar 5e-2)
POISSON ZERO: lambda=5.0 CHECKED expected_zeros=404.3 (>= 30) observed=406 p_zero=0.006767 exact=0.006738 rel=0.004262 (bar 2.2e-1)
POISSON MEAN: lambda=29.0 n=60000 mean=29.0105 rel=0.000362 (bar 1e-2)
POISSON VAR:  lambda=29.0 n=60000 var=29.2184 rel=0.007531 (bar 5e-2)
POISSON ZERO: lambda=29.0 SKIPPED expected_zeros=1.526e-8 (< 30, too few to measure) observed=0 exact=2.544e-13
POISSON MEAN: lambda=31.0 n=60000 mean=31.0512 rel=0.001653 (bar 1e-2)
POISSON VAR:  lambda=31.0 n=60000 var=31.0899 rel=0.002900 (bar 5e-2)
POISSON ZERO: lambda=31.0 SKIPPED expected_zeros=2.065e-9 (< 30, too few to measure) observed=0 exact=3.442e-14
POISSON MEAN: lambda=100.0 n=60000 mean=100.0106 rel=0.000106 (bar 1e-2)
POISSON VAR:  lambda=100.0 n=60000 var=98.8695 rel=0.011305 (bar 5e-2)
POISSON ZERO: lambda=100.0 SKIPPED expected_zeros=2.232e-39 (< 30, too few to measure) observed=0 exact=3.720e-44
POISSON MEAN: lambda=900.0 n=60000 mean=900.1335 rel=0.000148 (bar 1e-2)
POISSON VAR:  lambda=900.0 n=60000 var=897.6760 rel=0.002582 (bar 5e-2)
POISSON ZERO: lambda=900.0 SKIPPED expected_zeros=0.000e0 (< 30, too few to measure) observed=0 exact=0.000e0
POISSON MEAN: lambda=2839.0 n=60000 mean=2839.1862 rel=0.000066 (bar 1e-2)
POISSON VAR:  lambda=2839.0 n=60000 var=2848.5028 rel=0.003347 (bar 5e-2)
POISSON ZERO: lambda=2839.0 SKIPPED expected_zeros=0.000e0 (< 30, too few to measure) observed=0 exact=0.000e0

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 121 filtered out
```

**Margins.** Worst mean `rel` 0.001653 (lambda 31, 6.0x margin to the bar); worst variance `rel` 0.011305 (lambda 100, 4.4x); worst zero-mass `rel` 0.012126 (lambda 3, 18x). Every SKIPPED point prints its expected zero count and the reason, and a non-vacuity assertion pins the CHECKED count at exactly 2 — so `N`, `LAMBDAS` or `MIN_EXPECTED_ZEROS` cannot move and quietly leave the zero-mass bar running nowhere.

## The sigma arithmetic — re-derived, and it agrees

The plan instructed: *"Re-derive all six numbers before writing them; if the re-derivation disagrees with this paragraph, the re-derivation wins and the SUMMARY records the delta."* **Re-derived independently. It agrees to three significant figures at every point; there is no delta to record.** The derivations:

| Bar | Relative standard error | At N = 60 000 |
|---|---|---|
| Mean | `sqrt(1 / (lambda * N))` | 0.2357 % at lambda 3 |
| Variance | `sqrt(1/lambda + 2) / sqrt(N)` (Poisson `mu4 = lambda + 3 lambda^2`) | 0.6236 % at lambda 3, 0.6055 % at 5 |
| Zero mass | `sqrt((1 - p) / (N p))`, `p = exp(-lambda)` | 1.7835 % at lambda 3, 4.9567 % at 5 |

**Every shipped bar, at every point it is applied:**

| lambda | mean bar 0.01 | variance bar 0.05 | zero-mass bar 0.22 |
|---|---|---|---|
| 3 | **4.24 sigma** ← weakest shipped | 8.02 sigma | 12.34 sigma |
| 5 | 5.48 | 8.26 | **4.44 sigma** |
| 29 | 13.2 | 8.59 | (skipped) |
| 31 | 13.6 | 8.59 | (skipped) |
| 100 | 24.5 | 8.64 | (skipped) |
| 900 | 73.5 | 8.66 | (skipped) |
| 2839 | 130.5 | 8.66 | (skipped) |

The **weakest bar in the whole sweep is the pre-existing MEAN bar at the newly added lambda = 3, at 4.24 sigma** — not either of the new ones. That is the plan's point about adding a sweep point weakening every bar at that point, and it is the second reason N had to rise: the same bar at the same point is **2.45 sigma at N = 20 000**, below the 4-sigma floor.

### The review's suggested 1-2 % variance bar: REFUSED, with both reasons

The review wrote that "a 1-2 % bar is real rather than a fudge". It is not, and the refusal is recorded rather than the bar quietly widened:

1. **2 % is 1.91 sigma at N = 20 000** — a flake, not a bar. It would go red on sampling noise roughly one run in seventeen.
2. **The raise-N escape does not rescue it.** At N = 80 000, a 2 % bar is still only **3.70 sigma** at the lambda = 3 this plan adds. The 1-2 % branch does not hold at any reasonable N once the sweep includes the regime the parity ladder actually runs in.

Shipped instead: **`variance_tolerance: 0.05`**, whose weakest point is 8.02 sigma, against a zero-variance stub sitting at a relative deviation of **1.0 — twenty times the bar**. `round()` inflates a normal-branch variance by ~1/12 (0.27 % at lambda 31, smaller above), negligible against 5 %; the Knuth branch draws exact integers and carries no such term.

### The zero-mass window, and why N = 20 000 could not hold it

One shared key is pinned from both sides:

- **Noise floor:** at least 4 sigma above the worst point's noise -> `4 x 0.049567` = **0.1983** (lambda 5).
- **Discrimination ceiling:** at most half the SMALLER defect gap -> `0.4956 / 2` = **0.2478** (lambda 3).

`zero_mass_tolerance: 0.22` sits inside `[0.1983, 0.2478]` — 4.44 sigma at lambda 5, 12.34 sigma at lambda 3.

**At N = 20 000 the same two constraints are 0.3434 and 0.2478, and they do not overlap.** No single shared key there is both non-flaky and discriminating. That infeasibility — not the size of the defect gap — is what forced N to 60 000. The plan's own earlier draft proposed 0.15 at N = 20 000 and mis-stated it as "roughly 2.6 sigma"; it is **1.75 sigma**, below the plan's own written prohibition. Recorded because the plan asked for it to be, and because it is a good illustration of why the arithmetic is written down rather than eyeballed.

## RED observation (a) — the zero-variance stub

`prophet::poisson` temporarily replaced with `lambda.round() as usize`. **This is the precise statement of WR-02: the mean bar passes at all seven while the other two fail.**

```
POISSON MEAN: lambda=3.0 n=60000 mean=3.0000 rel=0.000000 (bar 1e-2)
POISSON VAR:  lambda=3.0 n=60000 var=0.0000 rel=1.000000 (bar 5e-2)
POISSON ZERO: lambda=3.0 CHECKED expected_zeros=2987.2 (>= 30) observed=0 p_zero=0.000000 exact=0.049787 rel=1.000000 (bar 2.2e-1)
POISSON MEAN: lambda=5.0 n=60000 mean=5.0000 rel=0.000000 (bar 1e-2)
POISSON VAR:  lambda=5.0 n=60000 var=0.0000 rel=1.000000 (bar 5e-2)
POISSON ZERO: lambda=5.0 CHECKED expected_zeros=404.3 (>= 30) observed=0 p_zero=0.000000 exact=0.006738 rel=1.000000 (bar 2.2e-1)
POISSON MEAN: lambda=29.0 ... rel=0.000000    POISSON VAR: lambda=29.0 ... rel=1.000000
POISSON MEAN: lambda=31.0 ... rel=0.000000    POISSON VAR: lambda=31.0 ... rel=1.000000
POISSON MEAN: lambda=100.0 ... rel=0.000000   POISSON VAR: lambda=100.0 ... rel=1.000000
POISSON MEAN: lambda=900.0 ... rel=0.000000   POISSON VAR: lambda=900.0 ... rel=1.000000
POISSON MEAN: lambda=2839.0 ... rel=0.000000  POISSON VAR: lambda=2839.0 ... rel=1.000000

thread 'prophet::sampler::poisson_mean_and_variance_track_lambda_across_its_whole_domain'
panicked at crates/aprender-forecast/src/prophet.rs:1160:13:
poisson sampler VARIANCE outside its domain at lambda=3.0: var=0.0000 rel=1.000000 > bar=5e-2.
The codomain is a count whose mean AND VARIANCE are both lambda; a sampler with the right
centre and the wrong spread returns yhat_lower / yhat_upper narrower than the interval_width
it advertises

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 121 filtered out
```

**And the same stub against the PRE-strengthening test, which is the defect itself** (`rc=0`, all six lambdas, `poisson_mean_tracks_lambda_across_its_whole_domain` **passing**):

```
POISSON MEAN: lambda=5.0 n=20000 mean=5.0000 rel=0.000000
POISSON MEAN: lambda=29.0 n=20000 mean=29.0000 rel=0.000000
POISSON MEAN: lambda=31.0 n=20000 mean=31.0000 rel=0.000000
POISSON MEAN: lambda=100.0 n=20000 mean=100.0000 rel=0.000000
POISSON MEAN: lambda=900.0 n=20000 mean=900.0000 rel=0.000000
POISSON MEAN: lambda=2839.0 n=20000 mean=2839.0000 rel=0.000000
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 121 filtered out
```

Restored; `grep -c 'TEMPORARY WR-02 STUB'` -> **0**.

## RED observation (b) — `POISSON_NORMAL_BRANCH_LAMBDA` lowered to 3.2

Both the Rust constant and its contract mirror `constants.poisson_normal_branch_lambda` lowered together (mirror included, or 06-14's `cost_bounds_match_contract` fires first and masks the point):

```
POISSON MEAN: lambda=3.0 n=60000 mean=2.9975 rel=0.000839 (bar 1e-2)
POISSON VAR:  lambda=3.0 n=60000 var=3.0008 rel=0.000276 (bar 5e-2)
POISSON ZERO: lambda=3.0 CHECKED expected_zeros=2987.2 observed=2951 p_zero=0.049183 exact=0.049787 rel=0.012126 (bar 2.2e-1)
POISSON MEAN: lambda=5.0 n=60000 mean=5.0146 rel=0.002923 (bar 1e-2)
POISSON VAR:  lambda=5.0 n=60000 var=4.9970 rel=0.000596 (bar 5e-2)
POISSON ZERO: lambda=5.0 CHECKED expected_zeros=404.3 observed=1328 p_zero=0.022133 exact=0.006738 rel=2.284878 (bar 2.2e-1)
POISSON MEAN: lambda=29.0 ... rel=0.000091      POISSON MEAN: lambda=31.0 ... rel=0.001653
POISSON MEAN: lambda=100.0 ... rel=0.000106     POISSON MEAN: lambda=900.0 ... rel=0.000148
POISSON MEAN: lambda=2839.0 ... rel=0.000066

thread 'prophet::sampler::poisson_mean_and_variance_track_lambda_across_its_whole_domain'
panicked at crates/aprender-forecast/src/prophet.rs:1172:13:
poisson sampler ZERO MASS outside its domain at lambda=5.0: p_zero=0.022133 against an exact
exp(-lambda)=0.006738 (rel=2.284878 > bar=2.2e-1, expected_zeros=404.3). The low tail is where
the clamped normal approximation is visibly wrong and the exact sampler is not, so this is the
bar that notices POISSON_NORMAL_BRANCH_LAMBDA being lowered

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 121 filtered out
```

**Three things worth reading off this run rather than leaving implicit:**

- **The observation matches the analytic prediction to three digits.** Predicted `p_zero` at lambda 5 under the clamped normal: `Phi((0.5 - 5)/sqrt(5))` = `Phi(-2.01246)` = 0.02209. Observed: **0.022133**. Predicted relative deviation 2.278; observed **2.284878**.
- **Every MEAN stayed inside the 0.01 bar** — reproducing the review's six-row table and then catching what it could not. This is WR-01's claim, confirmed and then closed.
- **The VARIANCE bar ALSO stayed green** (`rel=0.000596` at lambda 5). So the variance bar does *not* detect a lowered threshold, and the zero-mass bar does *not* detect a zero-variance stub any more sharply than the variance bar does. **Neither new bar subsumes the other; both were needed.** That is the single most useful fact this plan measured, and it was not in the plan.
- **lambda = 3.0 is unchanged** (3.0 < 3.2, still Knuth), which is the control: the discriminator fires exactly at the point that crossed the branch and nowhere else.

Both restored. `cmp` confirms `contracts/forecast-tool-boundary-v1.yaml` is **byte-identical** to its pre-mutation state; the restored run is `3 passed; 0 failed`.

## IN-02 — the `feature_row` panic, before and after

**Before** (`hol_sets[hi]`), the totality test against unmodified code, `rc=101`:

```
thread 'prophet::feature_row_totality::a_short_hol_sets_is_a_miss_and_never_an_out_of_bounds_panic'
panicked at crates/aprender-forecast/src/prophet.rs:205:19:
index out of bounds: the len is 0 but the index is 0
test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 123 filtered out
```

**After** (`hol_sets.get(hi).is_some_and(..)`): the call RETURNS with the holiday entries zero, for both the empty slice and the off-by-one slice. `test result: ok. 34 passed; 0 failed` (32 parity + 2 new).

**The signature is NOT reverted**, and the second test is why that is safe: `a_correctly_built_hol_sets_still_hits_exactly_the_same_days` pins the shipped path at the row level, and it **passed against the pre-change indexing code too** — so it is a genuine before/after control, not a post-hoc rationalisation. `prophet::parity` is 32 / 0 either way.

Stated in the code and in the README so a later reader is not misled: a mismatched slice was a panic and is now a zero column, and **neither is correct output** for a caller who built the slice wrong. What the change buys is that a library does not abort the caller's process over it.

## IN-04 — the two substantive items

**`bolt::transpose`.** A `debug_assert_eq!(w.len(), out * inp, ..)` now sits immediately before the `trueno::blis::transpose(..).expect(..)` call. The `expect` is kept — the assert is the check, the expect is the message if a release build is wrong anyway — and the return type is deliberately **not** widened to `Result`: that is a breaking public-API change to a published crate for a condition every one of the thirteen in-crate call sites already satisfies.

**Proved non-vacuous rather than assumed** (CLAUDE.md rule 4/5 — a guard that is never reached is theater). Mutated to `out * inp + 1`:

```
thread 'bolt::tests::single_row_routing_is_dot8_at_production_defaults' panicked at bolt.rs:287:5
thread 'bolt::tests::multi_row_routing_is_gemm_blis_at_production_defaults' panicked at bolt.rs:287:5
test result: FAILED. 113 passed; 2 failed; 10 ignored
```

Two always-run tests reach it. Restored: **115 passed; 0 failed; 10 ignored**. The assert does **not** fire on the real code, which means the `expect` message's prose claim was true — had it fired, that would have been a real finding to report rather than an assert to weaken.

**`safetensors::load`.** Confirmed removed and unreferenced by search, not trust:

- `pmat query "safetensors load from path" --limit 10` (CLAUDE.md Code Search Policy) — every hit is a *different* symbol in a *different* crate (`chronos::load_model_from_dir`, `aprender-qa-gen::load_golden_from_path`, `aprender-core`'s `load_safetensors` / `load_model` / `load_from_safetensors`, `kmeans_impl::load_safetensors`). None is `aprender_forecast::safetensors::load`.
- Confirmed by reading: `git show ce3e5a8ea:crates/aprender-forecast/src/safetensors.rs | grep -n 'pub fn load'` -> **line 128** at the review base; at HEAD the module declares only `load_bytes` (line 35).
- `grep -rn 'safetensors::load\b' crates/ src/` (excluding the differently-named symbols) -> **none**.

**No caller has appeared since the review.** The deletion remains correct as dead code, and it remains a breaking removal from a published surface — which is now written down.

## The closing evidence sweep — twelve gates, one run

Every `rc` read from a redirect, never through a pipe (CLAUDE.md rule 1 — this exact mistake shipped twice in this repository). Run on the final tree, after every edit.

| # | Gate | rc | Decisive line |
|---|---|----|---|
| 1 | `cargo test -p aprender-forecast --lib` | **0** | `test result: ok. 115 passed; 0 failed; 10 ignored; 0 measured; 0 filtered out` (no `--skip`) |
| 2 | `cargo test -p aprender-mcp-forecast --lib` | **0** | `test result: ok. 40 passed; 0 failed` |
| 3 | `cargo test -p aprender-mcp-forecast --test e2e_stdio` | **0** | `test result: ok. 1 passed; 0 failed` (dark in CI; run deliberately) |
| 4 | `cargo test --release -p aprender-forecast --lib sc1_wall::` | **0** | `test result: ok. 1 passed; 0 failed`; 19 `SC1 WALL:` lines, all `profile=release` |
| 5 | `just forecast-sc1-sweep` | **0** | `SC1 SWEEP OK: 19 compositions, every one under the 2.0 s SC1 bar` |
| 6 | `just forecast-bench` | **0** | `ROUND TRIP OK: 0.211 s < 2.0 s (SC1)` |
| 7 | `just forecast-holiday-bench` | **0** | `HOLIDAY DESIGN OK: 1.684 s < 2.0 s (SC1)` |
| 8 | `bash scripts/check_assert_measurement_under_cases.sh` | **0** | `TABLE OK: 23/23 rows behaved as tabled` |
| 9 | `cargo clippy -p aprender-forecast -p aprender-mcp-forecast --all-targets --no-deps -- -D warnings` | **0** | clean on both named packages |
| 10 | `cargo fmt --all -- --check` | **0** | no diff |
| 11 | `pv validate` x 4 Phase 6 contracts | **0** | **4** x `0 error(s), 0 warning(s)` |
| 12 | `make contract-audit-phase6` | **0** | 4 `Total equations:` summaries; `resolved 63 Phase 6 binding rows`; **0** real `BIND-`/`RESOLVE-` findings |

**Worst SC1 composition: the NeuralProphet row at 1.412 s**; worst Prophet composition `freq=W growth=logistic holidays=at_bound` at **0.755 s**. Both under the 2 s bar with the same shape 06-16 recorded.

**Parity, unmoved:** `prophet::parity` **32 passed / 0 failed**; `np::parity` **9 passed / 0 failed**, exactly 06-15's recorded pre-round count.

### The `BIND-` pattern, proved with a case table rather than read

The plan's own gate is `test "$(grep -c 'BIND-' log)" -eq 0`. **That is unsatisfiable by construction** — the success line is `Phase 6 binding audit: 4 contract(s) audited, zero BIND- findings`, which contains the literal. It measured **1** on this clean run. This is the fourth appearance of this exact defect in this round (06-14, 06-15, 06-16, and now the 06-17 plan text). The refined pattern `grep -E '(^|[[:space:]])(BIND|RESOLVE)-[0-9]' | grep -v 'zero BIND- findings'` was shipped with a **9-row must-match / must-not-match table that was RUN** (CLAUDE.md rule 7), using the shapes the target actually emits:

```
MUST_MATCH:  [ERROR] BIND-001: Equation '...' has no binding entry          -> 1
             [WARN] BIND-004: Equation '...' is pending implementation      -> 1
             RESOLVE-001 prophet-parity-v1.yaml ... (no definition site)    -> 1
             (an indented finding line)                                      -> 1
MUST_NOT_MATCH: the success line                                             -> 0
             'no BIND- findings were produced'                               -> 0
             'Phase 6 source resolution: resolved 63 ... rows'               -> 0
             prose mentioning the BIND- vocabulary                           -> 0
             'why zero BIND- findings is not that proof'                     -> 0

PATTERN CASE TABLE: 9/9 rows behaved as tabled
REAL LOG: refined=0   naive 'grep -c BIND-'=1
```

A first draft of the table had a MUST_MATCH row `RESOLVE- 007` (with a space) that the pattern correctly missed. The row was **wrong, not the pattern** — the target emits no such shape — so the row was corrected against the Makefile's real output rather than the pattern loosened to accept an invented one.

## Review Dispositions Ledger — all nine, verified AGAINST THE TREE

Each row names the symbol, file or contract key that closes it, and a command whose output proves it. **A row that could not be resolved would be REOPENED, not re-asserted.** All nine resolved.

| ID | Disposition | What in the tree closes it | Command that proves it |
|---|---|---|---|
| **CR-01** | INCORPORATED (06-14 T1/T3, 06-16 T1) | `types::MAX_LOGISTIC_CHANGEPOINT_LAMBDA = 20_000.0` (`types.rs:168`); `prophet::changepoint_count` (exactly one definition); the door refusal at `forecast.rs:366`, before `make_design`; `sc1_wall` observed failing on the CR-01 configuration | `grep -n 'pub const MAX_LOGISTIC_CHANGEPOINT_LAMBDA' crates/aprender-forecast/src/types.rs`; `just forecast-sc1-sweep` -> `SC1 SWEEP OK` |
| **WR-01** | INCORPORATED, SPLIT — both halves now closed | *Ownership* (06-14): `constants.poisson_normal_branch_lambda: 30` mirrored by `cost_bounds_match_contract`. *Detection* (**this plan**): `zero_mass_tolerance` + the zero-mass assertion | `grep -n 'poisson_normal_branch_lambda' contracts/forecast-tool-boundary-v1.yaml`; threshold lowered to 3.2 -> exit 101 (recorded above) |
| **WR-02** | INCORPORATED — **closed here** | `equations.poisson_sampler_domain.variance_tolerance: 0.05`; the variance assertion in `poisson_mean_and_variance_track_lambda_across_its_whole_domain` | `cargo test -p aprender-forecast --lib prophet::sampler` -> 2 passed; stub -> exit 101 (recorded above) |
| **WR-03** | INCORPORATED (06-15 T1) | the in-loop `max_holiday_dates_total` refusal at `forecast.rs:246`, with the post-loop exact-total refusal retained at `:292` (`grep -c` = 2, position proven) | `grep -n 'max_holiday_dates_total' crates/aprender-forecast/src/forecast.rs` -> 246 then 292 |
| **WR-04** | INCORPORATED (06-16) | `crates/aprender-forecast/src/sc1_wall.rs` (19 compositions, no `#[ignore]`); `justfile:907 forecast-sc1-sweep`; `logistic_band_wall` deleted | `just forecast-sc1-sweep` -> `SC1 SWEEP OK: 19 compositions` |
| **IN-01** | INCORPORATED (06-16) | `scripts/assert_measurement_under.sh` behind all five wall-clock bars; `scripts/check_assert_measurement_under_cases.sh` run before every gate measurement | `bash scripts/check_assert_measurement_under_cases.sh` -> `TABLE OK: 23/23` |
| **IN-02** | INCORPORATED, NARROWED — **closed here** | `hol_sets.get(hi).is_some_and(..)` at `prophet.rs:218`; `mod feature_row_totality` (2 tests); the README API note. Signature deliberately NOT reverted | `grep -n 'hol_sets.get(' crates/aprender-forecast/src/prophet.rs`; `cargo test -p aprender-forecast --lib feature_row_totality` |
| **IN-03** | INCORPORATED (06-14) | `types::tests::pool_default_matches_contract` (`types.rs:481`); the drifting doc count corrected | `grep -n 'fn pool_default_matches_contract' crates/aprender-forecast/src/types.rs` |
| **IN-04** | IN PART, REST REJECTED — **the accepted part closed here** | `debug_assert_eq!` at `bolt.rs:287`; `## Public API changes` at `README.md:99` | `grep -n 'debug_assert_eq!' crates/aprender-forecast/src/bolt.rs`; `grep -c 'Public API changes' crates/aprender-forecast/README.md` |

### The two REJECTED items inside IN-04, with rationale (recorded so they cannot be re-raised as new)

1. **"Five source files changed in the reviewed range fall outside the stated 9-file scope."** **REJECTED, no action.** This is an observation about the REVIEW's own scope statement, not a defect in the tree. The five files are legitimately part of the diff; the review read them and reported what it found. Nothing in the code is wrong because a header said "9".
2. **The `dates.rs` `parse_date` rewrite.** **REJECTED, no action** — because the reviewer examined it and found it **SOUND**: the shape gate proves ten ASCII bytes with `-` at positions 4 and 7 and digits elsewhere before the byte fold runs, so the removed error arm really was unreachable, and the 4-digit year keeps the day count far from any `i64` edge. There is nothing to change. Recorded here only so a later reader does not mistake "mentioned in a finding" for "outstanding".

## The prophet-parity scope-fence DEVIATION — recorded, deliberate, and narrow

The round's fence says a refusal added at the door must not move a parity fixture. **This plan does edit `contracts/prophet-parity-v1.yaml`.** That is a deliberate, plan-authorised, narrow deviation and not the fence being ignored:

- The edits are **ADDITIVE keys** (`variance_tolerance`, `zero_mass_tolerance`) plus invariant prose, a widened `formal`, and a new `FALSIFY-PROPHET-014`.
- They land on **`poisson_sampler_domain`, which is 06-13's SAMPLER-DOMAIN equation — not a parity rung.** It bars the sampler against its own mathematical law, not against a Python oracle.
- **`float_tolerance: 0.01` is byte-identical.**
- **No fixture** under `crates/aprender-forecast/tests/fixtures/` or `crates/aprender-mcp-forecast/fixtures/` is touched.
- **Strengthening a falsification test is the opposite of loosening a parity tolerance.** The fence exists to stop a bar being widened to admit code that was failing it; this edit narrows what conforms.

**Proven, at the review's own pinned base:**

```
$ git diff --stat ce3e5a8ea..HEAD -- crates/aprender-forecast/tests/fixtures/ crates/aprender-mcp-forecast/fixtures/
(empty)
$ git diff --stat ce3e5a8ea..HEAD -- contracts/neuralprophet-parity-v1.yaml contracts/chronos-bolt-parity-v1.yaml
(empty)
$ git diff ce3e5a8ea..HEAD -- contracts/prophet-parity-v1.yaml | grep '^-' | grep -v '^---'
-  version: 1.0.0
```

**The ONLY deleted line in `prophet-parity-v1.yaml` across the entire review range is the version number.** Every `*tolerance:` line in that diff is a `+`. No tolerance has ever been deleted or changed; `float_tolerance: 0.01` appears as an addition only because 06-13 introduced the whole equation after the review base.

### Contract version: 1.1.0 -> 2.0.0, on `pv diff`'s verdict

`pv diff` run on **two real filesystem paths** (`git show 4ef2b5e5b:contracts/prophet-parity-v1.yaml > /tmp/...`, never a git revision — `pv diff` takes paths):

```
Contract diff: v1.1.0 → v1.1.0
Suggested bump: major
  equations:
    ~ poisson_sampler_domain: formula changed
    ~ poisson_sampler_domain: invariants changed
  proof_obligations:
    + invariant:The changepoint-count sampler operates INSIDE its numerical domain — in its SPREAD and its LOW TAIL ...
    - invariant:The changepoint-count sampler operates INSIDE its numerical domain for every lambda ...
  falsification_tests:
    + FALSIFY-PROPHET-014
```

**The MAJOR verdict was followed rather than argued down**, and it is right: the equation now demands strictly more of any implementation, so something that conformed to v1.1.0 need not conform now. The zero-variance stub is precisely such a thing — which is the entire point of the plan. Nothing in the tree pins this version (checked), so the bump is safe.

## The binding row count — counted, not incremented

Four round-3 plans each added rows and each bumped the Phase 6 header by hand. That is the same by-hand drift that left the Phase 5 header wrong by 3 with nothing checking it (06-14 deviation 5). **Counted for real:**

```
$ awk 'NR>1610' contracts/aprender/binding.yaml | grep -c '^- contract:'
63
```

split **23** `forecast-tool-boundary-v1` / **18** `chronos-bolt-parity-v1` / **13** `prophet-parity-v1` / **9** `neuralprophet-parity-v1` = 63.

The header already said 63, so this is a **confirmation, not a correction** — but it is now recorded *as a count*, with the reproducing command in the header comment, so the number stops being four sequential guesses that happened to land right. **06-17 added no row**: rows are per EQUATION, and this plan strengthened an existing equation rather than declaring a new one. `make contract-audit-phase6` after the reconciliation: rc 0, 63 rows resolved.

## What the round bounded, what it closed by measurement, and what is still OPEN

**Bounded at the door, by a named contract-owned constant:**

| Axis | Constant | Contract key | Plan |
|---|---|---|---|
| C-06 logistic changepoint simulation | `MAX_LOGISTIC_CHANGEPOINT_LAMBDA` = 20 000 | `fit_max_logistic_changepoint_lambda` | 06-14 |
| C-07 `holidays[].name` byte amplification | `MAX_HOLIDAY_NAME_LEN` = 200 **bytes** | `fit_max_holiday_name_len` | 06-15 |
| C-08 NeuralProphet training work | `MAX_NP_TRAIN_COST` = 15 000 000 | `fit_max_np_train_cost` | 06-15 |
| (pre-round) holiday design cost / dates total / columns / points / horizon / span | `MAX_HOLIDAY_DESIGN_COST`, `MAX_HOLIDAY_DATES_TOTAL`, `MAX_HOLIDAY_COLUMNS`, `MAX_POINTS`, `MAX_HORIZON`, `MAX_SPAN_DAYS` | mirrored, all | 06-11/06-12 |

**Closed by a recorded MEASUREMENT rather than a bound, with the number:**

- The whole SC1 surface: **19 compositions, worst 1.412 s** against a 2 s bar (`just forecast-sc1-sweep`).
- C-08's structural maximum: **47.924 s on release** — 24x the bar, which is exactly why it was closed by a bound and *not* recorded as `measured_at_structural_maximum`.
- WR-03's avoided work: **0.012000 s -> 0.000212 s** (~57x) on the review's own 1 000 x 1 000 payload.
- CR-01's regression, reproduced: **2.443 s** at `freq=MS growth=logistic` with the bound raised, against `freq=D` at **0.219 s** in the same run.

**The class-invariant tests that make this checkable rather than asserted — all three green:**

```
test types::tests::no_cost_axis_is_pending ... ok
test types::tests::every_cost_axis_names_a_real_bound ... ok
test types::tests::every_request_knob_is_enumerated ... ok
test result: ok. 3 passed; 0 failed
```

`grep -c 'bound: unbounded_pending' contracts/forecast-tool-boundary-v1.yaml` -> **0**. `no_cost_axis_is_pending` is the one 06-14 shipped **deliberately RED** and 06-15 cleared by replacing both markers; it has stayed green through 06-16 and 06-17.

**STILL OPEN — not closed by this round:**

1. **`just forecast-sc1-sweep` is wired to nothing.** The round's own central gate. See the top of this document; the recommendation is one Makefile line and a measured 24 s.
2. **The three `human_verification` items from `06-VERIFICATION.md`**, all untouched by round 3:
   - the browser MCP handshake and chart rendering on both demo pages;
   - the x86_64 `quantiles_abs_f32_nonaarch64` measurement (**D-ITEM-06-04**);
   - the decision on SC4's Chronos ladder staying dark in CI (**D-ITEM-06-03**).
3. **D-ITEM-06-16** — `chronos-coldstart`'s 150 ms bar is safe only by its upstream `sed` parse, not by itself; a token that makes its integer test *error* takes the not-taken branch and the recipe prints `COLD START OK`. Logged in `WINDOWS.md`; surfaces at ship time.
4. **The root `CHANGELOG.md`** does not yet carry the two breaking 0.63.0 surface changes. They are in the crate README; the release-time capture is the follow-up (this plan deliberately did not edit 172 KB of shared history).

## The residual risk — what would falsify "the class is closed"

Round 3 closed every axis the enumeration could find. **The claim that the door-bound CLASS is closed would be falsified by: a caller-settable knob, or a caller-driven cost axis, that exists in the code and is absent from `door_surface`.**

Two tests would go red if such a thing were added **through the structs**:

- `types::tests::every_request_knob_is_enumerated` — derived from `schemars::schema_for!`, the same generator that produces the advertised schema, and it fails in **both** directions (a field with no entry, and an entry naming no field). Adding a field to `ForecastArgs` or `HolidayArg` without a `door_surface.knobs` entry turns it red.
- `types::tests::every_cost_axis_names_a_real_bound` — every `cost_axes` entry must name a bound that exists in `constants:`.

**The one way a new cost axis could still slip in unnoticed:** an axis introduced *inside* `prophet.rs` or `np.rs` that **no `ForecastArgs` field names** — work whose size is driven by an existing knob in a new way, so no new field appears and no `door_surface` entry is owed. `every_request_knob_is_enumerated` cannot see it, because no knob changed. `every_cost_axis_names_a_real_bound` cannot see it, because nobody wrote the entry it would check.

**The only guard against that is the `sc1_wall` sweep noticing the wall move** — which is a measurement, not an invariant, and which (see item 1 above) **currently runs only when a human types it.** That is the residual risk, and it is the strongest argument for wiring the sweep into `tier3`: it is the sole remaining detector for the one gap the completeness tests structurally cannot cover.

Stating this is more useful than declaring victory. Three prior rounds each believed they had closed the class.

## Decisions Made

Recorded in the frontmatter `key-decisions`. The three that shaped the plan:

1. **The sample size was the thing to move, not the bar.** When the noise floor and the discrimination ceiling left no window at N = 20 000, the answer was 60 000 draws, not a looser tolerance. A bar chosen to fit the available N is a bar chosen by convenience.
2. **The review's suggested tolerance was refused with arithmetic, not overridden with taste.** Both escape routes (2 % at N = 20 000, and 2 % at N = 80 000) were computed and both fail the 4-sigma floor.
3. **Neither new bar subsumes the other**, measured: the variance bar is blind to a lowered threshold and the zero-mass bar is not sharper than the variance bar against a zero-variance stub. Shipping only one would have closed only one finding.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `git diff` through the Bash tool is rewritten by the `rtk` hook, and the scope-fence check silently returned NOTHING**

- **Found during:** Task 3, the `ce3e5a8ea..HEAD` fence verification
- **Issue:** `git diff ce3e5a8ea..HEAD -- contracts/prophet-parity-v1.yaml | grep -E '^[-+].*tolerance'` printed **nothing at all** — which reads as "no tolerance line was touched" and would have been recorded as the fence holding. It is not: the `rtk` Bash hook summarises `git diff`, so the grep was searching a summary that contains no diff lines. The known-good `rtk proxy` escape had been applied to every `cargo test` in this plan but not to `git diff`. **This is the highest-severity defect found in this plan, because it produces a confident FALSE GREEN on the exact claim the round's scope fence rests on.**
- **Fix:** Re-ran as `rtk proxy git diff ...`. The real answer is stronger than the silent one: the only deleted line in the whole range is `version: 1.0.0`, and all three `*tolerance:` lines are additions.
- **Verification:** both forms run side by side; the filtered form yields 0 matching lines, the `rtk proxy` form yields the full diff.
- **Files modified:** none (verification method only)

**2. [Rule 1 - Bug] Backticks inside a double-quoted `git commit -m` were command-substituted, deleting text from a commit message**

- **Found during:** Task 3
- **Issue:** `git commit -m "... \`pv diff\` on two real paths ..."` — backticks inside double quotes are command substitution in `zsh`/`sh`. The shell tried to run `pv diff`, printed `command not found: pv`, and committed the message with the phrase **removed**: "CONTRACT VERSION.  on two real filesystem paths". The commit succeeded, so nothing failed loudly.
- **Fix:** Amended with `git commit --amend -F <file>`, writing the message via a heredoc-created file so no shell interpretation occurs. Verified the restored text by reading the committed message back.
- **Note:** the two earlier commits that contain backticks (`93b019ef8`, `65364694e`) escaped them as `` \` `` and were checked — both are intact. `0222fe60d` contains no backticks. Only the one commit was damaged, and it was repaired.
- **Files modified:** none (commit message only)

**3. [Rule 1 - Bug] The plan's `make contract-audit-phase6` gate is unsatisfiable by construction — for the fourth time this round**

- **Found during:** Task 3
- **Issue:** `test "$(grep -c 'BIND-' log)" -eq 0`. The target's SUCCESS line contains the literal `BIND-`, so a clean run measures 1. 06-14, 06-15 and 06-16 each recorded this and each fixed it locally; the 06-17 plan text reintroduced the naive form a fourth time.
- **Fix:** Used the refined pattern and shipped a **9-row must-match/must-not-match table that was RUN** (recorded above), built from the shapes the Makefile actually emits rather than invented ones.
- **Files modified:** none (verification method only)
- **Recorded as a class, not an instance:** four occurrences in four consecutive plans is not four mistakes, it is one missing artifact. The fix that would end it is a shared `scripts/assert_no_audit_findings.sh` with its own case table, exactly as 06-16 did for the wall-clock bars. Not taken here (it is a new script in a plan whose scope is four review findings), and logged below as a follow-up.

**4. [Rule 1 - Bug] The plan's stated `equation_tolerance` call-site count is stale**

- **Found during:** Task 1
- **Issue:** The plan and the RED commit message both say `equation_tolerance` "has eleven call sites". Measured: **25** (`grep -rn 'equation_tolerance(' crates/aprender-forecast/src/ | grep -v 'fn equation_tolerance' | wc -l`). The plan's number predates the parity ladder growing.
- **Fix:** No code change — the *decision* the number supports (do not generalise `equation_tolerance` away) is only strengthened by the true figure. Recorded here so the SUMMARY does not repeat a number the tree contradicts.
- **Verification:** the `git diff` of `test_support.rs` against the plan base contains **zero deleted lines**, so `equation_tolerance` and all its call sites are provably untouched.

**5. [Rule 2 - Missing Critical] The `debug_assert` had to be proved REACHED, not merely added**

- **Found during:** Task 2
- **Issue:** The plan asks for a `debug_assert_eq!` and for `cargo test ... bolt::` to be green. Green is exactly what a `debug_assert` on a never-exercised path also produces. `cargo test --lib bolt::` runs only 5 tests with 4 ignored, so "it passed" was not evidence the assert had ever executed.
- **Fix:** Mutated the assert to `out * inp + 1` and ran the **whole crate suite**: two always-run tests go red naming `bolt.rs:287`. Restored and re-ran green. The guard is reached; the claim it checks is true.
- **Files modified:** none beyond the assert itself (mutation reverted)

**6. [Rule 2 - Missing Critical] A non-vacuity assertion for the conditional zero-mass bar**

- **Found during:** Task 1
- **Issue:** The zero-mass bar runs only where `N * exp(-lambda) >= 30`. Nothing in the plan prevents a later change to `N`, `LAMBDAS` or that minimum from reducing the checked set to **zero** — leaving a green test that asserts nothing about the low tail, which is the vacuous-guard class this entire round exists to end.
- **Fix:** Added `assert_eq!(checked_zero_mass, 2, ...)` with a message naming the three constants whose movement would cause it. Beyond the plan's text, which asked only that skips be printed.
- **Verification:** present in the shipped test; the sweep prints CHECKED/SKIPPED with the expected count for all seven points.

---

**Total deviations:** 6 auto-fixed (4 bugs, 2 missing-critical).
**Impact on plan:** No scope creep. **Three of the six (1, 3, 4) are defects in this plan's own verification method or stated facts** — and deviation 1 would have recorded a false green on the round's central scope-fence claim, which is precisely the failure mode CLAUDE.md rule 1 exists for. Two (5, 6) strengthen guards the plan specified but did not require to be proved non-vacuous. Every plan objective was met as written.

## Issues Encountered

- **The `rtk` hook's reach is wider than this round had assumed.** Prior plans documented it for `cargo test` summary lines and `println!` output. It also rewrites `git diff`. The working rule is now: **any command whose output a gate greps must go through `rtk proxy`**, not merely `cargo test`.
- **Shell metacharacters in commit messages.** Backticks are natural when writing about code and are command substitution inside double quotes. Commit messages of any length should be written via `-F file`.

## Known Stubs

None. No hardcoded empty value, placeholder string, `TODO` or `FIXME` was introduced, and no test was skipped or `#[ignore]`d — the ignored count is unchanged at 10 and the crate suite runs with **no `--skip`**.

Two temporary mutations were made and both are provably reverted: the zero-variance `poisson` stub (`grep -c 'TEMPORARY WR-02 STUB'` -> 0) and the lowered `POISSON_NORMAL_BRANCH_LAMBDA` with its contract mirror (`cmp` on `forecast-tool-boundary-v1.yaml` -> byte-identical). The `bolt.rs` assert mutation was restored from a pre-mutation copy.

## Follow-ups logged (not taken here)

- **`scripts/assert_no_audit_findings.sh`** with its own case table, to end the four-times-repeated `grep -c 'BIND-'` defect at its class rather than per plan.
- **The root `CHANGELOG.md`** entry for the two breaking 0.63.0 surface changes, at release time.

## Threat Flags

None. This plan adds no request-handling code and no network, auth, file-access or schema surface. All six `mitigate` dispositions in its own register are discharged: **T-06-43** (the mean-only falsification test) by the variance bar with its stub observed red; **T-06-44** (`POISSON_NORMAL_BRANCH_LAMBDA` undetectable) by the zero-mass bar with the lowered threshold observed red; **T-06-45** (`feature_row`'s out-of-bounds panic) by the total lookup with the pre-change panic recorded; **T-06-46** (`bolt::transpose`'s unchecked claim) by a `debug_assert` proved reached; **T-06-47** (the round's own completion claim) by the twelve-gate sweep, the tree-verified ledger and the residual-risk statement above. **T-06-SC** was never engaged: no package-manager install ran and no dependency entered the graph.

## Next Phase Readiness

- **Gap-closure round 3 is complete.** All nine `06-REVIEW.md` findings are dispositioned and every one resolves to something in the tree with a command that proves it.
- **The branch is green.** `cargo test -p aprender-forecast --lib` passes with no `--skip`; `prophet::parity` 32/0 and `np::parity` 9/0 are unmoved across the whole round.
- **The one consequential decision this round leaves for a human** is wiring `just forecast-sc1-sweep` into `tier3` — one line, 24 s measured, recommended at the top of this document. Until it is taken, the round's central gate runs only on request, and it is the sole detector for the one residual path by which a new cost axis could enter unnoticed.
- `06-VERIFICATION.md` is **STALE** — it predates 06-10..06-13, which closed all three of its gaps. Its three `human_verification` items remain genuinely open.

---
*Phase: 06-native-time-series-forecasting-stack*
*Completed: 2026-09-07*

## Self-Check: PASSED

- All six files in `key-files.modified` exist on disk, as does `.tdd-evidence/06-17-t1-red.json` and this SUMMARY.
- All four production commits resolve in `git log --all`: `0222fe60d`, `93b019ef8`, `65364694e`, `b27f66330`.
- `commits: 7` is **MEASURED**, not narrated, and was **reconciled in place** rather than left stale (06-15/06-16 precedent): `git rev-list --count 4ef2b5e5b747c01338e126a98e05fdeabaf172cd..HEAD` was **4** at the four production commits, **5** at the SUMMARY commit `93064ac92` (which also carries STATE.md and ROADMAP.md), **6** after the `WINDOWS.md` ledger commit `53801f380`, and **7** counting this reconciliation commit itself — which is what makes the number terminal and checkable. A frontmatter count that disagrees with the ledger is exactly the narrated-vs-measured failure this field exists to prevent, so it was corrected rather than rounded to the first value written. `plan_head_before` is recorded so the count is checkable with the same instrument.
- `actuals.tokens: 14434` is chars/4 over the realized diff (57 735 chars) against the plan's estimate of 70 000 — a **4.85x over-estimate**, recorded unrounded rather than flattered, so it calibrates future estimates honestly.
- Every plan-level `<verification>` command was re-run on the final tree and is recorded in the twelve-row gate table above.
