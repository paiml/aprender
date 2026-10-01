---
phase: 06-native-time-series-forecasting-stack
plan: 16
subsystem: testing
tags: [forecasting, prophet, neuralprophet, sc1, benchmark-gate, bashrs, provable-contracts, wr-04, in-01]

requires:
  - phase: 06-14
    provides: "MAX_LOGISTIC_CHANGEPOINT_LAMBDA, prophet::changepoint_count, the door_surface enumeration, and the LOGISTIC_BENCH_FREQ selector this plan then deleted along with its harness"
  - phase: 06-15
    provides: "MAX_NP_TRAIN_COST = 15_000_000 and its three at-the-bound compositions, which the sweep's NeuralProphet row is widened to"
provides:
  - "crates/aprender-forecast/src/sc1_wall.rs — ONE parameterized SC1 sweep over freq x growth x holiday shape, not #[ignore]d, with a release-only 2 s bar and a measured in-suite budget"
  - "max_legal_horizon — the per-composition horizon derivation, bisected through the same lambda arithmetic the door uses"
  - "scripts/assert_measurement_under.sh — the ONE shape-checking numeric bar behind all five wall-clock gates, both directions explicit"
  - "scripts/check_assert_measurement_under_cases.sh — its 23-row must-match/must-not-match table, RUN before every gate measurement"
  - "just forecast-sc1-sweep — the gate, observed failing on the CR-01 configuration"
  - "contracts/forecast-tool-boundary-v1.yaml v1.5.0 — equations.sc1_wall_swept, its proof obligation, FALSIFY-BOUNDARY-026"
  - "a repaired contract-audit-phase6 justfile resolver that can resolve a PARAMETERIZED recipe"
affects: [06-17, phase-6-verification, ship]

actuals:
  tokens: 20460
  tasks: 3
  commits: 9
plan_head_before: bfc3aeb4f91d24e2dd9c34a5a133fd1c4a458827

tech-stack:
  added: []
  patterns:
    - "Cross-product bench: shrink by points/horizon, NEVER by an axis"
    - "One shared numeric-bar validator with an explicit direction argument and a case table run on every gate invocation"
    - "Derive a composition's legal geometry from the bound, through the door's own arithmetic, rather than restating the multipliers"

key-files:
  created:
    - crates/aprender-forecast/src/sc1_wall.rs
    - scripts/assert_measurement_under.sh
    - scripts/check_assert_measurement_under_cases.sh
  modified:
    - crates/aprender-forecast/src/lib.rs
    - crates/aprender-forecast/src/prophet.rs
    - crates/aprender-forecast/src/types.rs
    - justfile
    - Makefile
    - contracts/forecast-tool-boundary-v1.yaml
    - contracts/aprender/binding.yaml

key-decisions:
  - "The DEFAULT in-suite matrix keeps the at-the-bound Prophet geometry rather than the reduced one the plan budgeted for, because 7.19 s of debug test time against a 60 s budget is measured headroom and because a cap below ~841 stops exercising max_legal_horizon on MS — leaving the CR-01 axis derivation unrun in the suite"
  - "logistic_band_wall DELETED (no recipe, no caller, no bar); holiday_design_wall FOLDED onto the shared builder (its recipe is cited by 06-EVIDENCE.md and by the contract)"
  - "The validator carries NO character class anywhere: bashrs 6.66.3 mis-parses a bracket as a `[ ]` test in globs, parameter expansions and single-quoted EREs alike, and it reported that in a minimal reproducer while MISSING it inside the longer file"
  - "The two excluded comparison sites use bash's integer test, which REFUSES a non-numeric token (measured rc=2) rather than coercing it — so they are structurally not the IN-01 class"
  - "contract-audit-phase6's justfile resolver pattern was widened to match a parameterized recipe, with a case table over old and new and a re-mutation in its own scope"

patterns-established:
  - "Tracer-style gate evidence: a gate is not shipped until it has been OBSERVED failing on the specific defect it exists for"
  - "Every guard pattern change ships a must-match/must-not-match table that is RUN, and every converted guard site is re-mutated in its own scope"

requirements-completed: [SC1]

coverage:
  - id: D1
    description: "ONE SC1 gate sweeps freq {D,W,MS} x growth {linear,logistic,flat} x holiday {none, at-the-design-cost-bound} plus the NeuralProphet row, at the tightest legal history span, on release, with a 2 s bar"
    requirement: SC1
    verification:
      - kind: integration
        ref: "just forecast-sc1-sweep"
        status: pass
      - kind: unit
        ref: "crates/aprender-forecast/src/sc1_wall.rs#sc1_wall_sweep"
        status: pass
    human_judgment: false
  - id: D2
    description: "The gate was OBSERVED FAILING on the CR-01 configuration: with the lambda bound raised past its structural maximum, freq=MS growth=logistic walls at 2.443 s / 2.536 s and the sweep goes red naming both, while freq=D passes at 0.219 s"
    requirement: SC1
    verification:
      - kind: integration
        ref: "SC1_SWEEP_HORIZON=3650 cargo test --release -p aprender-forecast --lib sc1_wall:: with MAX_LOGISTIC_CHANGEPOINT_LAMBDA=200_000 (exit 101, both MS rows named), then restored (exit 0)"
        status: pass
    human_judgment: false
  - id: D3
    description: "No wall-clock bar in the repository can pass on a measurement it could not parse; the guard proving that is exercised by a 23-row case table on every gate run"
    verification:
      - kind: unit
        ref: "bash scripts/check_assert_measurement_under_cases.sh (23/23)"
        status: pass
      - kind: integration
        ref: "five re-mutations: just chronos-bench / forecast-bench / forecast-pool-ratio / forecast-holiday-bench / forecast-sc1-sweep each observed failing on a garbage token and restored"
        status: pass
    human_judgment: false
  - id: D4
    description: "One composition builder, not three: logistic_band_wall deleted, holiday_design_wall folded, exactly one `fn holidays_for` in the crate"
    verification:
      - kind: unit
        ref: "grep -rn 'fn holidays_for' crates/aprender-forecast/src/ -> sc1_wall.rs:85 only"
        status: pass
      - kind: integration
        ref: "just forecast-holiday-bench (HOLIDAY DESIGN OK 1.690 s) and just forecast-bench (ROUND TRIP OK 0.214 s) still pass with their original OK tokens"
        status: pass
    human_judgment: false
  - id: D5
    description: "The contract owns the swept surface as a BENCHMARK claim falsified by a recipe (sc1_wall_swept, FALSIFY-BOUNDARY-026, binding row, v1.4.0 -> v1.5.0)"
    verification:
      - kind: unit
        ref: "pv validate contracts/forecast-tool-boundary-v1.yaml (0 errors) + make contract-audit-phase6 (rc 0, 63 rows resolved, zero findings)"
        status: pass
    human_judgment: false
  - id: D6
    description: "The forecast-holiday-bench comment no longer asserts a superlative its own next paragraph contradicts"
    verification:
      - kind: unit
        ref: "grep -c 'WORST SHAPE THE DOOR' justfile -> 0; the block now states the slowest-of-three-measured claim with all three numbers"
        status: pass
    human_judgment: true
    rationale: "The grep proves the sentence changed; whether the replacement claim is the RIGHT one to make about the three measurements is an editorial judgment a human should sign off on."

duration: 96 min
completed: 2026-09-07
status: complete
---

# Phase 6 Plan 16: One swept SC1 gate, and a bar that can fail Summary

**Three geometry-hard-coded benches replaced by one freq x growth x holiday cross product with a release-only 2 s bar — observed failing on the exact CR-01 configuration the phase's previous benches could not see — plus a shared shape-checking validator behind all five wall-clock bars whose 23-row case table runs before every gate measurement.**

## Performance

- **Duration:** 96 min
- **Started:** 2026-09-07T16:20:00Z (approx; ledger base `bfc3aeb4f`)
- **Completed:** 2026-09-07T17:56:25Z
- **Tasks:** 3
- **Files modified:** 14 (3 created, 11 modified)

## Accomplishments

- **WR-04 closed.** `crates/aprender-forecast/src/sc1_wall.rs` sweeps 18 Prophet compositions plus a NeuralProphet row, with no `#[ignore]`, a real 2 s bar on release, and `lambda=` / `cells=` / `band_width=` on every line so a slow composition says WHICH axis it is loaded on.
- **The gate was observed failing on CR-01.** Not argued — run, quoted below.
- **IN-01 closed at all five sites**, each re-mutated in its own scope. Four had never been mutated before.
- **One composition builder, not three.** `logistic_band_wall` deleted, `holiday_design_wall` folded, one `fn holidays_for`.
- **A gate defect the enumeration found and fixed**: `contract-audit-phase6` could not resolve a *parameterized* `just` recipe.

## Task Commits

1. **Task 1 RED** — `255c6c143` (test) — the sweep's MS x logistic composition refused at MAX_HORIZON
2. **Task 1 GREEN** — `ef24b0d5a` (feat) — derive each composition's legal horizon from the lambda bound
3. **Task 1 REFACTOR** — `774b4e57e` (refactor) — name which run is the gate; budget the in-suite cost by measurement
4. **Task 2 RED** — `ed98296a8` (test) — the case table the old awk coercion fails
5. **Task 2 GREEN** — `0aff470a6` (feat) — one shape-checking validator behind all five bars
6. **Task 3** — `a347d61e2` (feat) — one builder, one swept gate, and the contract owns it

**Plan metadata:** see the `docs(06-16)` commit that carries this file.

## TDD Gate Compliance

| Task | RED | GREEN | REFACTOR | Status |
|------|-----|-------|----------|--------|
| 1 | `255c6c143` | `ef24b0d5a` | `774b4e57e` | Pass |
| 2 | `ed98296a8` | `0aff470a6` | — (none needed) | Pass |
| 3 | n/a (`type="auto"`, no `tdd`) | `a347d61e2` | — | n/a |

Both RED phases were persisted and machine-verified with `gsd_run check tdd-red-evidence`, both returning `RED_EVIDENCE_OK` / `target_test_failed`. Records are committed at `.tdd-evidence/06-16-t1-red.json` and `06-16-t2-red.json`.

The classifier parses node-test TAP, and Rust libtest does not emit TAP. The records therefore carry a **faithful mechanical projection** of the libtest run — one TAP entry per test, with `# tests` / `# pass` / `# fail` read off libtest's own `test result:` line by `awk`, never asserted by hand. The projection script is recorded in the SUMMARY history; nothing in the record is a number I chose.

## The CR-01 regression observation (the evidence, not the claim)

`MAX_LOGISTIC_CHANGEPOINT_LAMBDA` and its contract mirror `constants.fit_max_logistic_changepoint_lambda` raised from `20_000` to `200_000` — past the 86 789.8 structural maximum, so the review's configuration is accepted again. Release build, `SC1_SWEEP_HORIZON=3650`:

```
SC1 WALL: model=prophet freq=D  growth=logistic holidays=none     points=33 horizon=3650 lambda=2851.6  total_s=0.219 predict_s=0.213 profile=release
SC1 WALL: model=prophet freq=D  growth=logistic holidays=at_bound points=33 horizon=3650 lambda=2851.6  total_s=0.294 predict_s=0.214 profile=release
SC1 WALL: model=prophet freq=W  growth=logistic holidays=none     points=33 horizon=3650 lambda=19960.9 total_s=0.646 predict_s=0.640 profile=release
SC1 WALL: model=prophet freq=W  growth=logistic holidays=at_bound points=33 horizon=3650 lambda=19960.9 total_s=0.723 predict_s=0.644 profile=release
SC1 WALL: model=prophet freq=MS growth=logistic holidays=none     points=33 horizon=3650 lambda=86789.8 total_s=2.443 predict_s=2.437 profile=release
SC1 WALL: model=prophet freq=MS growth=logistic holidays=at_bound points=33 horizon=3650 lambda=86789.8 total_s=2.536 predict_s=2.457 profile=release

thread 'sc1_wall::sc1_wall_sweep' panicked at crates/aprender-forecast/src/sc1_wall.rs:427:5:
2 of 19 compositions are at or above the 2.0 s SC1 bar: [model=prophet freq=MS growth=logistic
holidays=none points=33 horizon=3650] 2.443 s; [model=prophet freq=MS growth=logistic
holidays=at_bound points=33 horizon=3650] 2.536 s

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 123 filtered out; finished in 8.47s
```

`2.443 s` against the review's independently measured `2.334 s` on the same configuration, on the same host. `freq=D` passes at `0.219 s` in the SAME run — the control that pins the mechanism to the frequency multiplier and not to the host being slow that minute (CLAUDE.md rule 2).

Both constants restored byte-identically (`git diff` empty), same command:

```
SC1 WALL: model=prophet freq=D  growth=logistic holidays=none     points=33 horizon=3650 lambda=2851.6  total_s=0.221 profile=release
SC1 WALL: model=prophet freq=D  growth=logistic holidays=at_bound points=33 horizon=3650 lambda=2851.6  total_s=0.298 profile=release
SC1 WALL: model=prophet freq=W  growth=logistic holidays=none     points=33 horizon=3650 lambda=19960.9 total_s=0.702 profile=release
SC1 WALL: model=prophet freq=W  growth=logistic holidays=at_bound points=33 horizon=3650 lambda=19960.9 total_s=0.776 profile=release
SC1 WALL: model=prophet freq=MS growth=logistic holidays=none     points=33 horizon=841  lambda=19996.1 total_s=0.600 profile=release
SC1 WALL: model=prophet freq=MS growth=logistic holidays=at_bound points=33 horizon=841  lambda=19996.1 total_s=0.625 profile=release

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 123 filtered out; finished in 4.84s
```

The `MS` rows move from horizon 3650 to 841 because with the bound restored `max_legal_horizon` DERIVES the widest horizon that composition can legally ask for. That derivation is what the Task 1 RED was about.

## The full gate matrix (`just forecast-sc1-sweep`, release, aarch64)

```
freq growth    holidays  points horizon  lambda    cells      total_s  band_width
D    linear    none      33     3650     2851.6    0          0.159    0.0181
D    linear    at_bound  33     3650     2851.6    47879      0.150    363.4545
D    logistic  none      33     3650     2851.6    0          0.216    19.7402
D    logistic  at_bound  33     3650     2851.6    47879      0.292    19.3173
D    flat      none      33     3650     2851.6    0          0.088    0.2415
D    flat      at_bound  33     3650     2851.6    47879      0.092    0.1382
W    linear    none      33     3650     19960.9   0          0.154    0.1481
W    linear    at_bound  33     3650     19960.9   47879      0.151    2971.5728
W    logistic  none      33     3650     19960.9   0          0.639    21.0681
W    logistic  at_bound  33     3650     19960.9   47879      0.713    21.0032
W    flat      none      33     3650     19960.9   0          0.087    0.2415
W    flat      at_bound  33     3650     19960.9   47879      0.107    0.1382
MS   linear    none      33     3650     86789.8   0          0.155    0.6440
MS   linear    at_bound  33     3650     86789.8   47879      0.151    12920.7517
MS   logistic  none      33      841     19996.1   0          0.539    21.0713
MS   logistic  at_bound  33      841     19996.1   49818      0.553    20.6701
MS   flat      none      33     3650     86789.8   0          0.089    0.2415
MS   flat      at_bound  33     3650     86789.8   47879      0.093    0.1382
neuralprophet (freq=D, n_lags=41, points=2000, horizon=365, train_cost 14810040)  1.414   0.0744

  SC1 SWEEP OK: 19 compositions, every one under the 2.0 s SC1 bar
```

Worst composition: the NeuralProphet row at **1.414 s**, matching 06-15's independently recorded 1.402 s for the same `at_bound_short_history` geometry. Worst Prophet composition: `freq=W growth=logistic holidays=at_bound` at **0.713 s**, lambda 19 960.9.

Two facts worth reading off this table rather than leaving implicit:

- **`lambda` is printed even where the door does not bound it.** The `linear` and `flat` rows show `lambda=86789.8` on `MS` — that is the mean the logistic simulation *would* draw, and those arms never enter it, which is precisely why the door's bound is logistic-only. Printing it makes the asymmetry visible instead of inferable.
- **`band_width` on the non-logistic holiday rows is large** (2971 on `W linear at_bound`, 12920 on `MS linear at_bound`). That is the linear trend's own extrapolation over a 25 582- / 111 123-day future span, not a defect: those rows are measuring COST, and the width is an artifact of extrapolating a fitted slope that far. The logistic rows, which are the ones the band claim is about, sit at 19.7 (`D`, lambda 2851) versus 21.0 (`W`/`MS`, lambda ~20 000) — the band still widens with lambda, so the bound refuses cost rather than truncating the simulation.

## The naive-validator run (IN-01, which rows the old coercion let through)

The 23-row table was written FIRST and driven against the inline `awk` coercion extracted verbatim from `justfile:620/:680/:742/:844`. Verbatim, committed at `.tdd-evidence/06-16-t2-naive-validator-run.txt`:

```
MUST_FAIL - malformed, or at/over the bar. Every one of these printed OK
            under the inline 'awk -v v=$x { exit (v + 0 < BAR) }' coercion:
  BAD   under    []             bar=2.0    wanted MUST_FAIL, got MUST_PASS (rc=0)
  BAD   under    [abc]          bar=2.0    wanted MUST_FAIL, got MUST_PASS (rc=0)
  ok    under    [2.0]          bar=2.0    MUST_FAIL
  ok    under    [2.5]          bar=2.0    MUST_FAIL
  BAD   under    [1.2.3]        bar=2.0    wanted MUST_FAIL, got MUST_PASS (rc=0)
  BAD   under    [.]            bar=2.0    wanted MUST_FAIL, got MUST_PASS (rc=0)
  BAD   under    [-1]           bar=2.0    wanted MUST_FAIL, got MUST_PASS (rc=0)
  ok    under    [1e3]          bar=2.0    MUST_FAIL
  BAD   under    [1 2]          bar=2.0    wanted MUST_FAIL, got MUST_PASS (rc=0)
  BAD   under    [1.5<CR>]      bar=2.0    wanted MUST_FAIL, got MUST_PASS (rc=0)

FAIL: 7 of 23 rows behaved the other way.
```

**Exactly seven MUST_FAIL rows passed under the old coercion**: the empty token, `abc`, `1.2.3`, a lone `.`, `-1`, `1 2`, and a trailing-carriage-return `1.5`.

`1e3` is the instructive near-miss and is worth stating because it is the one that would have bitten next: the old coercion refused it **by accident**, reading it as 1000, which happens to be over a 2.0 bar. At `chronos-bench`'s 100 ms bar it would have been refused too, but at any bar above 1000 — a bar expressed in milliseconds, which this repository already has — the same token would have sailed through. The shape check refuses it for the right reason.

After the validator: **23/23 rows behave as tabled**, and `just forecast-sc1-sweep` runs the table before it measures anything.

## The five re-mutations (CLAUDE.md rule 4 — the old proof does not transfer)

Each site was fed a literal `abc` in place of its parsed measurement, the recipe was run and observed FAILING and naming the token, and the site was restored and re-run green. **Four of the five are pre-existing sites that had never had this done.**

| # | Site | Bar | Direction | Mutated run | Restored run |
|---|------|-----|-----------|-------------|--------------|
| 1 | `chronos-bench` (was `:620`) | 100 **ms** SC4 | `under` | rc=1, `FAIL "tiny-f16 forward (SC4)": ... token: "abc"` — note it fired even though the real measurement (18.7 ms) was comfortably under the bar | rc=0, `FORWARD OK: 18.7 ms < 100 ms` |
| 2 | `forecast-bench` (was `:680`) | 2.0 s SC1 | `under` | rc=1, token `"abc"` named; real value 0.218 s | rc=0, `ROUND TRIP OK: 0.216 s` |
| 3 | `forecast-pool-ratio` (was `:742`) | 2.0x SC5 | **`atleast`** | rc=1, token `"abc"` named; real best 5.511x | rc=0, `POOL SPEEDUP OK: best 5.191x >= 2.0` |
| 4 | `forecast-holiday-bench` (was `:844`, the instance the review named) | 2.0 s SC1 | `under` | rc=1, token `"abc"` named; real value 1.716 s | rc=0, `HOLIDAY DESIGN OK: 1.702 s` |
| 5 | `forecast-sc1-sweep` (new) | 2.0 s SC1 | `under` | rc=1 on the FIRST composition, naming it: `FAIL "SC1 model=prophet freq=D growth=linear holidays=none"` | rc=0, `SC1 SWEEP OK: 19 compositions` |

Site 3 is why the mode is a required argument rather than a default: its bar is `best >= 2.0`, the opposite direction from the other four, and a mechanical find-and-replace would have inverted it silently. The validator refuses an unknown mode rather than falling through to one direction, and that refusal is itself a tabled row.

## Step (4): every wall-clock bar in the repository, enumerated

`pmat query "wall clock bar assertion benchmark" --limit 10` plus a read of every `just` recipe that compares a measurement. The pmat sweep surfaced only criterion benches, a profiler's `wall_time` accessor and a CGP analysis helper — nothing that gates a release. Every decision-surface bar is in the `justfile`:

| Site | What it bars | Status |
|------|--------------|--------|
| `justfile:625` `chronos-bench` | 100 ms SC4 forward | **CONVERTED**, re-mutated |
| `justfile:686` `forecast-bench` | 2.0 s SC1 round trip | **CONVERTED**, re-mutated |
| `justfile:751` `forecast-pool-ratio` | 2.0x SC5 pool speedup (`atleast`) | **CONVERTED**, re-mutated |
| `justfile:869` `forecast-holiday-bench` | 2.0 s SC1 holiday design | **CONVERTED**, re-mutated |
| `justfile:960` `forecast-sc1-sweep` | 2.0 s SC1, per composition | **CONVERTED** from birth, re-mutated |
| `justfile:656` `chronos-coldstart` | 150 ms SC4 median | **Deliberately excluded** — see below |
| `justfile:498` `chronos-embed-build` | 30 000 000-byte binary ceiling | **Deliberately excluded** — not wall-clock, and integer-shaped |
| `justfile:740` `forecast-pool-ratio` inner `awk a > b` | nothing — it is a `max()` SELECTION over three attempts | **Not a bar.** The bar is site 3 |
| `justfile:580` `passed -lt 1` | a libtest non-vacuity count | **Not a measurement bar** |

**Why the two exclusions are not the IN-01 class, measured rather than asserted.** Both use bash's integer test, which refuses a non-numeric operand instead of coercing it:

```
  token=[]     -> refused rc=2  [: : integer expression expected
  token=[abc]  -> refused rc=2  [: abc: integer expression expected
  token=[1.5]  -> refused rc=2  [: 1.5: integer expression expected
  token=[150]  -> rc=0 (at the bar)
  token=[149]  -> rc=1 (under the bar)
```

That is the opposite of `awk`'s `+ 0`. **But there is a residual and it is recorded rather than glossed:** `chronos-coldstart` writes `if [ "$med" -ge 150 ]; then FAIL; fi`, so a token that makes the test *error* takes the not-taken branch and the recipe goes on to print `COLD START OK`. `set -euo pipefail` does not catch it — a command in an `if` condition is exempt from errexit. The hole is closed today only because `med` is extracted by a `sed` pattern that can emit nothing but digits, and an empty `med` is caught by an explicit guard two lines above. The bar is safe by its upstream parse, not by itself. Logged as **D-ITEM-06-16** in `deferred-items.md` and appended to `.planning/WINDOWS.md`.

## Folding the harnesses: what was chosen and why

- **`prophet::sampler::logistic_band_wall` — DELETED.** It was `#[ignore]`d, had no `just` recipe, asserted no bar, and had **no caller at all**, so unlike `holiday_design_wall` there was no entry point to preserve — only a second geometry to stop maintaining. Its whole geometry (33 points at the tightest legal daily span, logistic, per-frequency widest legal horizon) is now three rows of the sweep, which runs with **no flag**, asserts the bar, and prints `band_width=` exactly as it did. A signpost comment stands where it was. `types.rs`'s citation of it is repointed at the sweep and now carries **both** columns so the two harnesses can be seen agreeing:

  | freq | points | horizon | lambda | total_s (06-14 harness) | total_s (06-16 sweep) |
  |------|--------|---------|--------|-------------------------|-----------------------|
  | `D` | 33 | 3 650 | 2 851.6 | 0.244 s | 0.224 s |
  | `W` | 33 | 3 650 | 19 960.9 | 0.711 s | 0.714 s |
  | `MS` | 33 | 840 / 841 | 19 974.2 / 19 996.1 | 0.582 s | 0.620 s |

  The `MS` horizon differs by one step because the sweep DERIVES it from the bound rather than being told it.

- **`prophet::design_cost::holiday_design_wall` — FOLDED, not deleted.** `just forecast-holiday-bench` is cited by `06-EVIDENCE.md` and by the contract, and its caller-facing behaviour is unchanged. It now builds nothing of its own: the series (`tight_daily_series`), the splitter (`holidays_for`), the accept-and-time step (`time_accepted`) and the `profile=` token (`profile_token`) all come from `sc1_wall`. A `debug_assert_eq!` pins that the shared series builder still starts at the epoch its recorded numbers were measured against.

`grep -rn 'fn holidays_for' crates/aprender-forecast/src/` prints exactly one line: **`crates/aprender-forecast/src/sc1_wall.rs:85`**.

## `prophet::parity` before and after

**32 tests before the fold, 32 after**, all passing. Crate suite: **113 passed, 0 failed** on both runs, with **no `--skip`**. Ignored count 11 → 10, the difference being exactly the deleted `logistic_band_wall`. The declared-red window 06-14 opened and 06-15 closed stays closed.

## Contract

- `equations.sc1_wall_swept` — a BENCHMARK claim falsified by a recipe and never by a libtest assertion, following the split `pool_speedup_host_gated` already makes.
- A `bound` proof obligation.
- `FALSIFY-BOUNDARY-026`, whose `test:` is `just forecast-sc1-sweep`, recording the observed-red run.
- `contracts/aprender/binding.yaml`: a row bound to the **recipe** (the third such row); Phase 6 header 62 → 63 rows, and the "Two rows bind to `justfile` recipes" note corrected to three.
- `pv diff /tmp/ftb-old-16.yaml contracts/forecast-tool-boundary-v1.yaml` (two real filesystem paths) suggested **`minor`**. Applied: **v1.4.0 → v1.5.0**.
- `pv validate`: `0 error(s), 0 warning(s)`. `make contract-audit-phase6`: rc 0, 4 `Total equations:` summaries, 63 rows resolved, zero findings.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `contract-audit-phase6` could not resolve a PARAMETERIZED `just` recipe**
- **Found during:** Task 3, adding the binding row
- **Issue:** The resolver used `pattern="^$name[[:space:]]*:"`, which matches `forecast-pool-ratio:` but not `forecast-sc1-sweep points="33" ...:`. The audit failed with `RESOLVE- ... (no definition site in justfile)` for a recipe that plainly exists. Latent since the gate was written — invisible because this is the first plan to bind a parameterized recipe, and it would equally have failed for `forecast-holiday-bench`.
- **Fix:** `pattern="^$name([[:space:]][^:]*)?:"`, with a must-match/must-not-match table driven over the OLD and NEW patterns together. The table shows the change strictly widens MUST_MATCH (two parameterized recipes now resolve) with no MUST_NOT_MATCH regression — `forecast-sc1-sweep-other:`, `other-forecast-sc1-sweep:`, `forecast-sc1-sweepX:`, an indented occurrence and a comment mention all still miss.
- **Re-mutated in its own scope (CLAUDE.md rule 4):** renaming the recipe to `forecast-sc1-sweep-RENAMED` makes the audit exit 2 with `RESOLVE- forecast-tool-boundary-v1.yaml sc1_wall_swept justfile::forecast-sc1-sweep`; restoring it returns rc 0.
- **Files modified:** `Makefile`
- **Commit:** `a347d61e2`

**2. [Rule 1 - Bug] The plan's own `<verify>` for Task 3 (t3e) cannot be satisfied**
- **Found during:** Task 3
- **Issue:** `test "$(grep -c 'BIND-' log)" -eq 0`. The target's SUCCESS line is `Phase 6 binding audit: 4 contract(s) audited, zero BIND- findings` — it contains the literal `BIND-`, so the count is 1 on a passing run and the check can never be satisfied. This is the third defective `<verify>` class this round (four in 06-14, three in 06-15).
- **Fix:** used `grep -cE '^BIND-|^RESOLVE-'`, and ran a must-match/must-not-match table proving it distinguishes findings from the success line: a `BIND-001 ...` line and a `RESOLVE- ...` line MATCH; the success line and a prose line mentioning `BIND-` MISS. Measured 0 on the passing run and 1 on the mutated run.
- **Files modified:** none (verification method only)

**3. [Rule 2 - Missing Critical] The validator must carry no character class at all**
- **Found during:** Task 2
- **Issue:** The plan quotes the review's suggested form, `case "$total" in ''|*[!0-9.]*|*.*.*)`. `bashrs` 6.66.3 reports that bracket as `SC1020: Missing space before closing ] in test expression` plus `SC1140`. The parameter-expansion cousin `${tok//[0-9.]/}` and even a bracket inside a **single-quoted** ERE handed to `grep` are flagged the same way. Worse, the finding did **not** surface when the construct sat inside the full validator, only in a five-line minimal reproducer — so the "clean" lint on the real file was a false green.
- **Fix:** the shape check consumes the token one byte at a time with explicit `0 | 1 | ... | 9` alternation and a bracket-free `*.*.*` glob for the dot count. Both scripts are **0 errors / 0 warnings** under `bashrs lint <file>`, which is the invocation `make lint-scripts` itself uses.
- **Note recorded rather than swallowed:** `bashrs lint a b` (multi-file) and `bashrs lint a` (single-file) give DIFFERENT verdicts for the same file — multi-file mode reports `SC2046: Quote this to prevent word splitting` on plain arithmetic expansions like `$((checked+1))`. The plan's `<verify>` used the multi-file form; I verified with the single-file form the repository's own gate uses, and stated why.
- **Files modified:** `scripts/assert_measurement_under.sh`
- **Commit:** `0aff470a6`

**4. [Rule 2 - Missing Critical] The DEFAULT in-suite matrix keeps the at-the-bound Prophet geometry**
- **Found during:** Task 1, step (5b)
- **Issue:** The plan budgets for a REDUCED default geometry with the gate widening it. Measurement did not support that for the Prophet half: the full 18-composition cross product at the tightest legal history span costs **7.19 s of test time (21 s including compile)** on debug against a **60 s** budget. Worse, reducing it would have been actively harmful: `"MS"` only clamps when the requested cap exceeds ~841, so any default below that stops exercising `max_legal_horizon` entirely — leaving the CR-01 axis derivation, the exact code the Task 1 RED found missing, unrun in the suite.
- **Fix:** the Prophet default stays at the bound; only the NeuralProphet row defaults small, because *its* at-the-bound composition is measured at 45.6 s on debug (06-15) and alone would blow the budget. The module doc states this in plain words rather than repeating the plan's "far from any bound" phrasing, which would have been false of the shipped code — the same superlative-that-contradicts-its-own-evidence defect Task 3 fixes one file over.
- **Files modified:** `crates/aprender-forecast/src/sc1_wall.rs`
- **Commit:** `774b4e57e`

**5. [Rule 3 - Blocking] The plan's AC grep uses `\s`, which BSD grep does not support**
- **Found during:** Task 2
- **Issue:** `grep -v '^\s*#' justfile | grep -c 'assert_measurement_under.sh'` errors on macOS BSD grep. This is the same class as the repository's own known `include!()`-guard-vacuous-on-macOS lesson.
- **Fix:** used the portable POSIX class `grep -vE '^[[:space:]]*#'`. Both forms happen to return 5 here (BSD grep treats `\s` as a literal `s`, and no line starts with `s#`), but the literal form errors and could not be relied on.
- **Files modified:** none (verification method only)

---

**Total deviations:** 5 auto-fixed (2 bugs, 2 missing-critical, 1 blocking).
**Impact on plan:** No scope creep. Two are verification-method corrections that would otherwise have produced false greens — the class this whole plan exists to close. One (deviation 1) is a real latent gate defect the plan's own step-(4) enumeration was designed to surface, and it was fixed with the same table-plus-re-mutation discipline the plan demands of the validator.

## Review Dispositions Ledger

| ID | Severity | Disposition | Status after this plan |
|---|---|---|---|
| CR-01 | Critical | INCORPORATED | Bound landed 06-14; **the gate that would have caught it landed here, and was observed failing on it** |
| WR-01 | Warning | INCORPORATED, SPLIT | 06-14 done; 06-17 open |
| WR-02 | Warning | INCORPORATED | 06-17 open |
| WR-03 | Warning | INCORPORATED | 06-15 done |
| **WR-04** | Warning | INCORPORATED | **CLOSED here** — one swept gate, one composition builder, superlative reworded |
| **IN-01** | Info | INCORPORATED | **CLOSED here** — shared validator at all five sites, case table run on every gate invocation, all five re-mutated |
| IN-02 | Info | INCORPORATED, NARROWED | 06-17 open |
| IN-03 | Info | INCORPORATED | 06-14 done |
| IN-04 | Info | IN PART, REST REJECTED | 06-17 open |

## Known Stubs

None. No hardcoded empty value, placeholder string, TODO or FIXME was introduced; no test was skipped or `#[ignore]`d by this plan (the ignored count went DOWN by one).

## Threat Flags

None. This plan adds no request-handling code and no network, auth, file-access or schema surface. Its three new files are a test-only module and two executed shell scripts invoked only by `just` recipes. `T-06-39` (the blind gate surface) and `T-06-40` (the spoofable bar) are the two `mitigate` dispositions in the plan's own register and both are discharged above; `T-06-41` (the repudiable superlative) is discharged by the reworded comment; `T-06-42` is `accept` by design and the accepted design is what shipped. No package-manager install occurred (`T-06-SC`).

## Issues Encountered

- **`bashrs` gave two different verdicts for the same file.** Resolved by writing a construct no mode disagrees about, and by verifying with the invocation the repository's own `make lint-scripts` uses. Recorded in deviation 3 rather than left as a footnote, because "the linter passed" was momentarily true and wrong.
- **The TDD RED classifier expects node-test TAP and Rust emits libtest.** Resolved by a mechanical projection whose counts are read off libtest's own summary line by `awk`. Both records verify `RED_EVIDENCE_OK`.

## Next Phase Readiness

- Wave 15 (`06-17`) is unblocked: it depends on `06-16` and touches `prophet.rs`'s sampler module and `poisson_sampler_domain`, neither of which this plan's deletion region overlaps (`logistic_band_wall` was the last item in `mod sampler`).
- `just forecast-sc1-sweep` is **not wired into CI or any `make` tier** — deliberately. The plan's scope fence forbids touching `.github/workflows/*.yml`, and wiring a ~40 s release gate into `tier3` is a human decision. That is the one obvious follow-up.
- `D-ITEM-06-16` (`chronos-coldstart`'s residual) is open and logged in `WINDOWS.md`; it will surface at ship time.

---
*Phase: 06-native-time-series-forecasting-stack*
*Completed: 2026-09-07*

## Self-Check: PASSED

- All four created files exist on disk (`sc1_wall.rs`, both scripts, this SUMMARY).
- All seven commits resolve in `git log --all`: `255c6c143`, `ef24b0d5a`, `774b4e57e`, `ed98296a8`, `0aff470a6`, `a347d61e2`, `2807dcb12`.
- `commits: 9` in the frontmatter is MEASURED from the on-disk plan ledger, not narrated. It was 7 when this SUMMARY was first written, then 8 once `497a9d141` (`docs(06-16)`: STATE + ROADMAP) landed, and 9 counting this reconciliation commit itself — which is what makes the number terminal and checkable: `git rev-list --count bfc3aeb4f..<this commit>` = 9. Reconciled in place rather than left stale, following 06-15's `reconcile actuals.commits` precedent; a frontmatter count that disagrees with the ledger is exactly the narrated-vs-measured failure this field exists to prevent.
- Every plan-level `<verification>` command re-run and logged in the sections above.
