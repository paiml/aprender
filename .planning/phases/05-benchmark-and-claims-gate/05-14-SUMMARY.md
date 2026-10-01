---
phase: 05-benchmark-and-claims-gate
plan: 14
subsystem: setfit-calibration-derivation
tags: [d-18, epsilon-basis, fail-closed, calibration, contracts, verification-discipline]
status: complete
requires:
  - "05-01's 12 committed calibration passes and its production_calibration_matrix harness"
  - "thresholds.rs Thresholds::table_for / RegimeThresholds::of(class).gated"
provides:
  - "epsilon_basis() — the single window-rule site, fail-closed on collapse"
  - "typed BasisCoverage (Complete | Provisional{missing}) so a provisional basis cannot be read as freezable"
  - "the shared basis renderer both matrices call, with declared-ungated row annotations"
  - "REGIME TABLE FOR THE DERIVED REGIME: resolved|absent — the gated-set lookup's own answer, readable off any run"
  - "a committed-store digest guard that asserts the PAIR COUNT before any agreement check"
affects:
  - "05-03 — its central claim (every gated class has a legal epsilon) is now machine-checked by running the derivation"
tech-stack:
  added: []
  patterns:
    - "verdict and report as separate surfaces: a declared exclusion moves the verdict and never removes the row"
    - "the required set READ from the regime's frozen table, never from a local allowlist/constant/env override"
    - "report written to its destination BEFORE the refusal panics"
    - "state-dependent controls DERIVE their expected status from the mechanism's own report line, not from a text proxy"
key-files:
  created:
    - .planning/phases/05-benchmark-and-claims-gate/05-14-controls.md
  modified:
    - crates/aprender-train/src/train/setfit/evidence.rs
decisions:
  - "epsilon_basis lives in evidence.rs's #[cfg(test)] module, beside the two matrices that are its only callers — production gate code never derives a basis, so shipping it in the lib would add dead code to the published crate"
  - "the refusal is Box<CollapsedWindows>: it deliberately carries the whole basis so the report can still be rendered on refusal, which makes it the larger variant (clippy::result_large_err)"
  - "the DURABLE real-data guard derives under a synthetic architecture no table can ever cover, and asserts that None explicitly — 05-01's MEASURED aggregates with a label chosen so the guard cannot flip"
  - "every test carries epsilon_basis in its name PREFIXED (evidence_...), so the verify block's filter matches while `grep -c 'fn epsilon_basis'` still finds exactly one function"
  - "both verify blocks run cargo test through `rtk proxy`: the rtk hook strips the `test result:` line they grep for, measured with a control"
metrics:
  duration: ~1h45m
  completed: 2026-08-17
actuals:
  tokens: 17900
  tasks: 2
  commits: 2
---

# Phase 5 Plan 14: Fail-Closed Epsilon Window Verdict Summary

`epsilon_basis()` is now the single site that applies the epsilon window rule, returning a typed
collapse refusal instead of a line of prose beside `rc=0` — proven by a behaviour delta on the
identical four-cell combine 05-01 recorded exiting `rc=0`, which now exits `rc=101` and names all
five collapsed classes.

## What Was Built

D-18 said an empty epsilon basis could coexist with a green calibration run. It could: separation
was ASSERTED (`ctrl_max < real_min`) while `supports_margin` was only REPORTED, so 05-01's
four-cell combine printed `EMPTY` five times and exited zero. A green run was therefore not
evidence that an epsilon basis existed — in exactly the surface F-10 exists to protect.

The fix is structural, not an assertion bolted on. The window rule existed as inline arithmetic in
TWO places (the fixture matrix's basis table and the production matrix's), which is why 05-01 had
to fix its `eps/noise` reporting defect twice to keep it from surviving in a sibling report. It is
now one function:

```
epsilon_basis(regime_id, aggregates, measured_cells, full_matrix_cells)
    -> Result<EpsilonBasis, Box<CollapsedWindows>>
```

Five properties, each of which was the point of a `<behavior>` bullet:

1. **One row per class, one strict comparison.** `grep -v '^[[:space:]]*//' … | grep -c 'lower <
   upper'` returns **1** where it returned **2**. Each row carries the four measured aggregates the
   report already printed, the derived bounds, and a per-class status that distinguishes a legal
   window from an empty one and carries the exceed factor for the empty case. `eps/noise` stays
   suppressed to `n/a` for an empty window exactly as 05-01 left it.
2. **A typed coverage value.** `Complete` or `Provisional { measured, total, missing }`, with the
   missing-cell derivation MOVED out of the production matrix's PROVISIONAL banner rather than
   reimplemented. Provisional coverage is NOT a refusal — 05-01's per-pass resumable workflow
   depends on a partial measurement staying legal, and the monotonicity argument (within a fixed
   rule `lower` is monotone non-decreasing and `upper` monotone non-increasing in the measured
   cells) is what makes that sound while a collapse is not.
3. **The required set is READ from the regime.** `Thresholds::frozen().table_for(regime_id)`; a
   class is required exactly when the resolved table gates it. No allowlist, no local constant, no
   environment override. Where no table resolves — the production regime's state today — EVERY
   class is required, so a class the contract has never recorded cannot be exempted by omission.
4. **One renderer both matrices call.** The PROVISIONAL banner, the `WINDOWS THAT DO NOT EXIST`
   block and its explanatory paragraph are preserved verbatim. One thing was ADDED: a `gating`
   column reading `required` or `declared-ungated`, plus the same annotation inside the
   empty-windows block, so an exclusion is visible in the artifact rather than inferable only from
   the absence of a failure.
5. **Both call sites assert the verdict — after writing their report.** A derivation whose numbers
   cannot be read is worse than one that fails loudly.

One line was added to the derivation report, which Task 2's control keys on and which is worth
having anyway: `REGIME TABLE FOR THE DERIVED REGIME: resolved for <id> — gated classes: [...]` or
`... absent — no frozen table covers <id>, so EVERY class is required`. Printing the lookup's own
answer makes the declared-ungated annotations traceable to a mechanism, and lets any later control
read the verdict's precondition off the run instead of proxying it from source or contract text.

The module doc now states the RESIDUAL: this makes an empty basis unable to coexist with a green
run, and does NOT make a non-empty basis sufficient. A legal window can still be too thin to be
worth freezing, and coverage can still be provisional. Both are judgement inputs the report now
surfaces as typed values rather than prose.

## Tasks Completed

| Task | Name | Commit | Files |
| ---- | ---- | ------ | ----- |
| 1 | One fail-closed window-rule site, falsified two-sided in the default suite | `e890cccde` | `crates/aprender-train/src/train/setfit/evidence.rs` |
| 2 | Prove the guard in its real scope — RED on the committed 12 passes, GREEN on the fixture regime | `c4fd5e63e` | `.planning/phases/05-benchmark-and-claims-gate/05-14-controls.md` |

## The evidence, and which of it is durable

Full quoted logs and tables: `05-14-controls.md`. The distinction below is the one a later reader
most needs.

**DURABLE — re-runnable forever, no contract dependency, cannot flip on what 05-03 lands:**

- The nine tests filtered by `epsilon_basis`, all in a plain
  `cargo test -p aprender-train --lib --features setfit`: no `#[ignore]`, no env gate, no 86.7 MB
  checkout, no training. `test result: ok. 9 passed` (floor 8, one per `<behavior>` bullet, read
  from the run's own count).
- Of those, `evidence_epsilon_basis_real_four_cell_aggregates_refuse_under_no_resolved_table` is
  the permanent real-data guard: 05-01's MEASURED four-cell aggregates as literals, derived under
  an architecture component that is not a real encoder fingerprint and never will be. It asserts
  `table_for(...).is_none()` explicitly, so if that id ever DID resolve the guard fails loudly
  rather than silently ceasing to be red. It reproduces all five exceed factors 05-01 published.

**STATE-DEPENDENT — a control on today's state, deliberately not durable:**

- The `--ignored` four-cell combine. Its verify block DERIVES the expected status from the
  derivation's own `REGIME TABLE FOR THE DERIVED REGIME:` line — the mechanism that actually
  decides RED vs GREEN — never from a contract- or source-text proxy. A text proxy would
  false-flip on the option-F halt branch, where 05-03 still lands MEASURED/COVERED prose naming
  all ten contracted seeds while no production table exists, and the block would then fail
  forever: the very defect this restructuring avoids. The block also fails loudly if the line is
  absent, so a naming drift cannot silently pick a default branch. It was RUN as one script and
  exited 0 having taken the `absent -> expecting RED` branch.

### RED control — behaviour delta on identical input

| | 05-01 (recorded) | 05-14 (measured) |
|---|---|---|
| exit status | **`rc=0`** | **`rc=101`** |
| `PASSES RUN` | 12 | 12 |
| cells loaded | 4 | 4 |
| classes printing `EMPTY` | 5 | 5, same numbers |
| verdict | none | typed refusal naming all five with factors |

Exceed factors reproduce FINDING 1 to the digit: `embedding` 8.56x, `layer_norm_bias` 13.84x,
`projection_weight` 8.54x, `projection_bias` 31.94x, `attention_key_bias` 5.41x;
`layer_norm_weight` is again the sole survivor. Status captured on its own line, never through a
pipe. **No SKIP occurred in either direction** — the pinned checkout is present, and the log shows
twelve timing rows and a full basis table, so the refusal is over the real basis rather than an
early abort.

### GREEN control — the fixture matrix, `rc=0`

Its table RESOLVES, its five gated classes are legal, and `attention_key_bias` appears carrying
the `declared-ungated` annotation with the resolved table's gated set printed on the line above
it. **Neither documented refusal branch fired**; the predicate was not adjusted and nothing was
loosened to obtain the green.

Stated so it is not over-read: in the fixture regime `attention_key_bias`'s window happens to
EXIST, so this run does not exercise the declaration CHANGING a verdict. That two-sided
demonstration is
`evidence_epsilon_basis_declared_ungated_moves_the_verdict_without_removing_the_row` in the
default suite, which places one identical empty window on the declared-ungated class (success, row
retained and annotated) and on a gated class (refusal), and additionally derives the ungated case
under a no-table regime (refusal).

### Falsification by induced mutation

The tests could not have an executable RED phase in the ordinary sense — they cannot compile until
the interface they consume exists (the produced-before-consumed constraint 02-09 Task 3 hit).
Rather than manufacture a RED after the fact, the guard was falsified by three induced mutations,
each run and each reverted:

| mutation | effect | tests turned RED |
|---|---|---|
| `lower < upper` -> `lower <= upper` | a touching bound accepted as a window | 1 (`..._exactly_equal_bounds_refuse_because_the_window_is_strict`) |
| `None => true` -> `None => false` in the required-set read | exemption by omission | 3 (`..._without_a_resolved_table...`, `..._declared_ungated...`, `..._real_four_cell_aggregates...`) |
| renderer skips rows where `!required` | verdict and report collapse into one surface | 1 (`..._declared_ungated_moves_the_verdict_without_removing_the_row`) |

All three reverted before the Task 1 commit; the file at `e890cccde` is the unmutated form.

### Isolation and integrity

| check | result |
|---|---|
| `setfit::` lib suite | `322 passed; 0 failed; 3 ignored` — 05-02 baseline **313** plus this plan's **9**, exactly |
| committed store digests | `12 of 12 pairs verified`; pair count asserted BEFORE any agreement check |
| `git status --short` on the calibration store | empty |
| serialization surface | `git diff -U0 … \| grep '^-' \| grep -c 'serde\|to_canonical_bytes'` = **0** |
| `fn epsilon_basis` definitions | **1** |
| strict-window comparisons | **1** (was 2) |
| clippy (`--all-targets`, setfit) | zero findings in `evidence.rs` |
| `cargo check` (non-test build) | zero new `aprender-train` warnings |
| `rustfmt --check` on `evidence.rs` | clean |

`UpdateEvidence`, its serde attributes and `to_canonical_bytes` are untouched, so 05-01's
cross-process bit-identity proof — the property that licenses combining per-condition passes —
still holds over the same bytes.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] Both verify blocks grep for a line this host does not emit**

- **Found during:** Task 1, taking the pre-change baseline.
- **Issue:** the `rtk` hook rewrites `cargo test` output into a one-line summary
  (`cargo test: 313 passed, 3 ignored, …`). Measured, not assumed: `grep -c "test result"` over a
  captured log of a run that exited **0** returned **0** — there is no
  `test result: ok. N passed` line anywhere in the file. Both verify blocks assert exactly that
  line and feed the count floor from it, so they would have failed on a GREEN run and the floor
  would have read 0. This is the same artifact family as 05-01's `git log --oneline` rewrite that
  reported seven existing commits as MISSING.
- **Fix:** every `cargo test` in both verify blocks runs through `rtk proxy`, which yields the raw
  libtest stream with the `test result:` line intact. Same commands, same assertions, unrewritten
  stream. This follows the standing Phase 2 ruling that `git status --porcelain` assertions must
  go through `rtk proxy` for the same reason; the same applies to `git status`/`git diff`/`git
  log` used in the checks above.
- **Files modified:** none (verification procedure only).
- **Commit:** documented in `05-14-controls.md` at `c4fd5e63e`.

**2. [Rule 3 — Blocking] `clippy::result_large_err` on the typed refusal**

- **Found during:** Task 1 clippy gate.
- **Issue:** `CollapsedWindows` deliberately carries the whole basis (so the report can still be
  rendered on refusal), which makes the `Err` variant large enough to trip
  `clippy::result_large_err` — the one finding in this file.
- **Fix:** `Result<EpsilonBasis, Box<CollapsedWindows>>`. Boxed rather than trimmed: dropping the
  basis from the refusal would defeat the "write the report first" property the plan requires.
- **Files modified:** `crates/aprender-train/src/train/setfit/evidence.rs`.
- **Commit:** `e890cccde`.

### Deliberate departures from the plan text

**Task 2 produced no source change, and that is the honest outcome.** The plan scoped
`evidence.rs` to Task 2 for the two documented repair branches (fix the fixture regime lookup if
it does not resolve; report and HALT if a gated class refuses). Neither fired: the fixture regime
resolved and its five gated classes are legal. Task 2's deliverable is therefore the recorded
controls, committed as `05-14-controls.md` so the task still lands atomically and the RED/GREEN
numbers sit beside 05-01's `rc=0` where the next reader will look.

**`epsilon_basis` lives in the `#[cfg(test)]` module.** The plan says "one function in
`evidence.rs`", and both call sites are the two `#[ignore]`d calibration matrices, which are test
items. Production gate code never derives a basis — it reads a FROZEN epsilon out of the contract
table — so placing the derivation in the shipped lib would add dead code to the published crate
and require a `#[cfg_attr(not(test), allow(dead_code))]` to keep the non-test build clean. The
acceptance criterion it exists to serve ("zero new warnings in the non-test build") is met.

**Test names are prefixed `evidence_epsilon_basis_…` rather than bare `epsilon_basis_…`.** Both
of the plan's own criteria have to hold at once: every test carries `epsilon_basis` in its name
(so the filter matches), AND `grep -c 'fn epsilon_basis'` returns 1. A test named
`fn epsilon_basis_refuses_…` literally contains the string `fn epsilon_basis` and would have made
that count 9. The prefix satisfies both. The convention is stated in a comment at the head of the
test block so it does not decay.

## Out-of-scope discovery (NOT fixed)

`crates/aprender-train/src/train/setfit/apr_reload.rs:331` is unformatted at HEAD — `cargo fmt -p
aprender-train -- --check` reports a diff there that is PRE-EXISTING (the file is untouched by
this plan; the working tree contained only `evidence.rs`). Left alone per the scope boundary;
logged to the phase's `deferred-items.md`. The plan's criterion is scoped "for this file", and
`rustfmt --check` on `evidence.rs` is clean.

## Requirements

`EVAL-02` is advanced but NOT claimed complete by this plan — it is the phase's benchmark/claims
requirement and 05-03 and the benchmark plans carry the rest. No requirement checkbox was ticked.

## Known Stubs

None. Every symbol this plan introduces is reached by a default-suite test, and the one
state-dependent control is named as such rather than presented as a permanent guard.

## What 05-03 inherits

- Its central claim — "every class this regime gates has a legal epsilon under the chosen rule" —
  is now a machine-checked property of the derivation RUN, not something read off a report.
- The window arithmetic is a single named site (`WINDOW_SAFETY_FACTOR` and the two lines above the
  strict comparison). 05-03 edits those; the nine tests here must still hold afterwards, and
  05-03 Task 1 re-runs the `grep -c 'lower < upper'` == 1 check so the invariant is held across
  both plans rather than only at this one's end.
- The four-cell combine is RED today for a recorded reason. When 05-03 lands a production table
  the same invocation consults it and the same verify block asserts GREEN — red-before /
  green-after on identical input, with the reason written down in the contract.
- The only mechanism that can exempt a class from needing an epsilon is a recorded contract table.
  There is no knob here to reach for.

## Self-Check: PASSED

| claim | check | result |
|---|---|---|
| `crates/aprender-train/src/train/setfit/evidence.rs` modified | file present, `fn epsilon_basis` count | FOUND, 1 |
| `.planning/phases/05-benchmark-and-claims-gate/05-14-controls.md` created | file present | FOUND |
| commit `e890cccde` | `git log --oneline` | FOUND |
| commit `c4fd5e63e` | `git log --oneline` | FOUND |
| 9 tests pass under the `epsilon_basis` filter | run's own count | FOUND, `9 passed` |
| RED control `rc != 0` | status on its own line | FOUND, `rc=101` |
| GREEN control `rc == 0` | status on its own line | FOUND, `rc=0` |
