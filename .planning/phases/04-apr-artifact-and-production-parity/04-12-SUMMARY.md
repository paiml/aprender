---
phase: 04-apr-artifact-and-production-parity
plan: 12
status: PARTIAL
subsystem: training
tags: [setfit, ops-01, public-api-boundary, integration-test, finding, f-10]

# Dependency graph
requires:
  - phase: 04-04
    provides: "ClassifyRequestDocument + VerifiedSetFitModel::classify + ClassifyResponse accessors — present, and still UNREACHABLE from out-of-crate because nothing can construct the receiver"
  - phase: 04-05
    provides: "AprCodec (setfit-apr-v1) — present and driven from out-of-crate; it REFUSES the only trainable encoder (F-10)"
  - phase: 04-17
    provides: "SetFitRun::<ArtifactReloadedAndVerified>::into_artifact_bytes — the save rung's door. CONFIRMED reachable and correct out-of-crate; OPS-01-F1 is closed"
provides:
  - "crates/aprender-train/tests/setfit_apr_lifecycle.rs — the OPS-01 boundary test: the reachable rungs proven, the unreachable ones measured and named"
  - "The out-of-crate witness for 04-17's G1 door (its own test runs --lib and cannot make this claim)"
  - "F-10 kept EXECUTABLE at the OPS-01 tier, over THREE independent routes to APR bytes"
  - "The compiler-proof that 04-05's in-crate remedy for F-10 (encoder+head substitution) is E0451 out-of-crate"
affects: [04-06, 04-07, 04-08, 04-10, 04-11, 04-16, phase-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "A self-scanning source guard stays FILTER-FREE by keeping the needle out of the file entirely — including out of path literals, which is why the control file is opened with a constructed path instead of include_str!"
    - "A blocked lifecycle rung is recorded as a PASSING test asserting the TYPED refusal, never as an #[ignore]d test that would fail if run"
    - "Two independent routes to the same refusal is what separates 'this door is broken' from 'this input cannot work'"

key-files:
  created:
    - crates/aprender-train/tests/setfit_apr_lifecycle.rs
  modified: []

key-decisions:
  - "DID NOT manufacture an artifact by hand-building a SetFitArtifactView and calling core's public write_setfit_apr with a synthetic APR-capable encoder. It compiles and it would have satisfied the plan's `.embed(`/`.classify(` acceptance criteria — while the model classified would not be the one the train rung produced. OPS-01 is a claim about the JOIN between the rungs."
  - "DID NOT write an #[ignore]d full-chain test. The phase's convention (04-06/04-17) is that an ignored test still PASSES under `-- --ignored`; a test that cannot pass would be a landmine for 04-10's gates."
  - "Wrote the blocked rungs as PASSING tests that assert the typed refusal, so the day F-10 closes they turn RED and name this module — rather than a comment that ages silently."
  - "Every guard was FALSIFIED before being trusted: four mutations, four reds. A green that has never been shown able to go red proves nothing."

requirements-completed: []

# Metrics
duration: ~1h25m
completed: 2026-08-15
---

# Phase 04 Plan 12: OPS-01 Public-API Lifecycle — PARTIAL (F-10)

**The blocker I reported last time is genuinely gone. 04-17's `into_artifact_bytes` works
out-of-crate and I proved it by re-hashing. The chain still cannot complete, and the cause
is a DIFFERENT, older blocker that this phase has known about since 04-05: F-10. No encoder
in this repository can both pass `tune_encoder`'s calibration gate and compute the
`setfit-apr-v1` contract's six probes, so `load_setfit_apr` has no bytes and `embed`,
`classify` and `doc_view` have no receiver.**

`crates/aprender-train/tests/setfit_apr_lifecycle.rs` exists, compiles, and passes 5/5. It
proves what is reachable and it keeps the blocker executable. **OPS-01 is NOT complete and
must not be marked complete.**

## Task Commits

| Task | Deliverable | Commit |
| ---- | ----------- | ------ |
| 1 | `setfit_apr_lifecycle.rs` — 613 lines, 5 tests, OPS-01 boundary + F-10 executable | `a5f1c5787` |

## What the chain looks like today, rung by rung

| rung | reachable out-of-crate | evidence |
|---|---|---|
| train | **yes** | `prepare -> tune_encoder -> fit_head`, all public doors |
| save (bytes) | **yes**, since 04-17 | 1,824,298 bytes, re-hashed to the recorded digest |
| save (as `setfit-apr-v1`) | **NO — F-10** | typed `ProbeComputation{probe:"probe_unicode"}` on three routes |
| load | **NO** | no admissible input exists |
| embed / classify / inspect | **NO** | all three are methods on `VerifiedSetFitModel`, whose ONLY constructor is `load_setfit_apr` |

## OPS-01-F1 is CLOSED — measured, not assumed

My 04-12-BLOCKED report said the verified run retained a hash and a LENGTH and dropped the
bytes. 04-17 fixed it. Confirmed here from a different crate:

```
into_artifact_bytes() -> 1824298 bytes
  format_id                        = setfit-serde-json-v1
  run.artifact_hash()              = 28c6bea1d8845e75a8dc719dc1b42a5596cb705af208f87c1e9e82f37bd045a2
  sha256(returned bytes)           = 28c6bea1d8845e75a8dc719dc1b42a5596cb705af208f87c1e9e82f37bd045a2   [EQUAL]
  VerifyReport::artifact_bytes()   = 1824298                                                            [EQUAL to bytes.len()]
```

The re-hash is the whole point. A non-emptiness check would pass for a re-serialization —
the exact substitute the door exists to make unnecessary. The length equality is the second
half: before 04-17 the length was ALL that survived `run_verify_policy`, and this asserts the
number and the payload are now the same object.

**04-17's own test makes the same claim but runs `--lib`, so it cannot distinguish "the door
exists" from "a downstream caller can reach it".** This file is a different crate. That is
the new fact.

## THE BLOCKER — F-10, and why it is terminal at this tier

Two facts, jointly fatal. Both measured on this tree, from out-of-crate.

### (1) Exactly one encoder can be trained, and it is the phase-3 slice

`tune_encoder` renders the run's own regime coordinates and requires the frozen thresholds'
calibrated set to COVER them. Varying two INDEPENDENT coordinates, the refusal publishes the
whole set:

```
seed varied:  UncalibratedRegime {
                observed:   "minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=2|cells=s8e1b4",
                calibrated: ["minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1,42,7|cells=s16e2b8,s8e1b4"] }
cell varied:  UncalibratedRegime {
                observed:   "minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1|cells=s8e1b8",
                calibrated: [ ...same one entry... ] }
```

One entry. Its architecture component is compared for **exact equality**
(`thresholds.rs`, `RegimeCoordinates::covers`), and the doc says why in as many words: "an
epsilon measured on a 2-layer/64-hidden/97-vocab slice is not evidence about a
6-layer/384-hidden/30522-vocab model."

`SetFitMiniLm::from_bundle_parts` is `pub`, so a synthetic APR-capable encoder is
CONSTRUCTIBLE out here. It just cannot be TRAINED.

Two coordinates rather than one, deliberately: a single variation would leave open that the
gate fired for an unrelated reason. Each observed id names its own varied coordinate.

### (2) That slice provably cannot carry a `setfit-apr-v1` artifact

Measured fingerprint: `minilm-slice-h64-l2-a2-i256-v97`. Two structural gaps: it is a 97-row
**vocabulary closure**, and it declares 64 position rows against a 256-token truncation
probe. **Three independent routes to APR bytes, one refusal each:**

| # | Route | Result |
|---|---|---|
| 1 | `SetFitRun::<HeadFitted>::verify_artifact(&AprCodec)` — the shipped door | `Codec(Artifact{ ProbeComputation{ probe:"probe_unicode", reason:"VocabOutOfSlice(canonical id 5915 is outside the slice closure)" }})` |
| 2 | Public codec seam: run's real bytes -> `SerdeJsonCodec::deserialize` -> `SetFitBundle` -> `AprCodec::serialize` | **identical** `ProbeComputation{probe:"probe_unicode"}` |
| 3 | `aprender::setfit::write_setfit_apr` — core's public writer, which routes 1 and 2 both end in | same, by construction: it RECOMPUTES the six probes from the view's own tensors |

Route 2 matters on its own terms. It is a legitimate public composition my previous report
did not find — `SetFitCodec::deserialize` is `pub` and returns a `pub SetFitBundle`, so a
caller CAN now obtain a bundle from a run (via 04-17's bytes) without touching
`from_run_parts`, which was last time's E0624. It bypasses the verify policy entirely and
still dies at the same probe. **That is what proves the cause is the ENCODER's vocabulary
rather than any particular door** (CLAUDE.md verification discipline 6).

### (3) 04-05's remedy is not available here — E0451

The phase's F-10 note instructs 04-12 to "reuse 04-05's substitution shape": destructure the
run and swap in an APR-capable encoder and head. That shape is compiler-closed out-of-crate.
A probe asking both doors at once (`tests/zz_probe_0412b.rs`, created, checked, **deleted**):

```
error[E0451]: fields `encoder`, `dataset`, `selection`, `config`, `evidence` and `_state`
              of struct `SetFitRun` are private
error[E0451]: fields `passed`, `head`, `report`, `effective_lambda`, `ordered_labels`,
              `encode_ledger` and `encode_call_count` of struct `HeadFittedEvidence` are private
```

**This is the correction the phase needs:** the F-10 note's action item for 04-12 assumed a
remedy that exists only at the `--lib` tier. There is no out-of-crate workaround, which is
exactly why OPS-01 is the requirement that F-10 blocks hardest.

### (4) No committed fixture short-circuits it either

`crates/aprender-core/tests/fixtures/setfit/slice_model.apr` (447,108 bytes) is a plain Bert
`.apr`: `load_setfit_apr` refuses it with `NotASetFitArtifact { model_type: "Bert" }`. There
is no `setfit-apr-v1` artifact committed anywhere in this repository.

## What I deliberately did NOT do

**I did not manufacture an artifact.** `SetFitArtifactView`'s fields are public and
`write_setfit_apr` is public, so a synthetic APR-capable encoder built through
`from_bundle_parts` would have produced real `setfit-apr-v1` bytes, and this file could then
have called `.embed(` and `.classify(` and satisfied the plan's acceptance criteria to the
letter. The model classified would not have been the one the `train` rung produced. OPS-01 is
a claim about the JOIN between the rungs; a green light meaning "a test-local writer
round-trips" while reading as "OPS-01 holds" is the exact false green this phase's discipline
exists to reject.

**I did not reach for a `pub(crate)` door**, and no `src/` file was touched — the plan's hard
constraint, honoured.

**I did not write an `#[ignore]`d full-chain test.** The phase's convention (04-06's surviving
ignored test, re-confirmed by 04-17) is that an ignored test still PASSES under `--ignored`.
A full-chain test would fail, and would be a landmine for 04-10's gates.

## What the file DOES assert — 5 tests, all passing

| Test | Claim |
|---|---|
| `lifecycle_train_then_save_hands_the_caller_the_hashed_artifact_bytes` | train -> save works; the returned bytes RE-HASH to the recorded digest and match the published length |
| `lifecycle_the_apr_save_rung_is_refused_by_the_only_calibrated_encoder` | both structural gaps + the typed `probe_unicode` refusal on TWO routes |
| `lifecycle_no_second_encoder_can_reach_the_save_rung` | the regime gate is fail-closed on two coordinates; the calibrated set has exactly ONE entry and it still names the slice |
| `lifecycle_the_load_rung_requires_setfit_apr_v1_bytes` | TWO non-APR inputs refused at TWO DIFFERENT rungs (3 and 4) — a single input could not distinguish "wrong input" from "refuses everything" |
| `lifecycle_source_names_no_command_line_crate` | T-04-36's import half, with a positive control |

Each blocked-rung test panics with an explicit instruction if the refusal ever stops
happening — e.g. *"the slice fixture produced a setfit-apr-v1 artifact — F-10 is CLOSED.
Restore the full OPS-01 chain in this file: `load_setfit_apr -> embed -> classify ->
doc_view`."* The finding cannot go stale silently.

### The self-scan is filter-free by construction

Neither spelling of the CLI crate's name appears anywhere in the file — not in code, not in
comments, and **not in a path literal**, which is why the control file is opened with a
runtime-assembled path rather than `include_str!` (a literal path would itself be an
occurrence). Both needles are built at runtime from pieces. A comment-filter would have been
the part of such a guard that goes wrong silently (CLAUDE.md discipline 7); keeping the file
clean is what removes the need for one.

## Every guard was FALSIFIED before being trusted

| # | Mutation | Result |
|---|---|---|
| 1 | inject `// apr_cli` into the source | **RED** — `left: 1, right: 0`, rc=101 |
| 2 | expect `probe_ascii` instead of `probe_unicode` | **RED** — rc=101, message shows the real `probe_unicode` value |
| 3 | re-hash `&bytes[1..]` instead of `&bytes` | **RED** — rc=101, the RE-SERIALIZATION assertion fires |
| 4 | expect `ContainerIntegrity` where `NotASetFitArtifact` is correct | **RED** — rc=101 |

All four reverted; `git status --short` clean but for the new file before staging.

## T-04-36 — dependency direction, RE-MEASURED with its positive control

| Command | `apr-cli` occurrences | grep rc | tree lines |
|---|---|---|---|
| `cargo tree -p aprender-core -e normal` | **0** | 1 | 121 |
| `cargo tree -p aprender-train -e normal` | **0** | 1 | 633 |
| `cargo tree -p aprender-train -e normal --features setfit` | **0** | 1 | 710 |
| `cargo tree -p aprender -e normal` (**POSITIVE CONTROL**) | **1** | 0 | 1611 |

The control is what makes the three zeros measurements rather than theater: the root facade
genuinely depends on the CLI crate and the same counter returns 1 there. The `--features
setfit` row is included because that feature is where this phase's new edges appear — checking
only default features would not cover the surface where the decision is made.

Measured through `rtk proxy` throughout: the rtk hook rewrites `wc` and `grep` output, and a
plain `wc -l < file` reported **0 lines for a 5,733-byte file** during this run. That is
CLAUDE.md's F-11 hazard (a rendered view captured instead of the bytes) hitting a counter
rather than a file write.

## Verification — status captured directly, never through a pipe

Every command ran as `cmd > log 2>&1; echo "rc=$?"`.

```
$ cargo test -p aprender-train --features setfit --test setfit_apr_lifecycle
rc=0     5 passed; 0 failed; 0 ignored

$ cargo test -p aprender-train --lib --features setfit
rc=101   7891 passed; 24 failed; 15 ignored      (identical to 04-17's post-plan numbers)

$ cargo fmt -p aprender-train -- --check                                  rc=0
$ cargo check -p aprender-train --test setfit_apr_lifecycle  (feature OFF) rc=0
$ cargo clippy -p aprender-train --features setfit --test setfit_apr_lifecycle --no-deps
rc=0     0 diagnostics naming setfit_apr_lifecycle.rs
```

**The clippy run's mechanism was PROVEN engaged, not assumed.** "Zero diagnostics naming my
file" is indistinguishable from "my file was never linted". Injecting
`let _clippy_probe = format!("{}", "x");` produced `warning: useless use of format!` naming
`setfit_apr_lifecycle.rs` (1 hit); reverted. `--no-deps` is mandatory here (F-03): without it
the run exits on `aprender-compute`'s pre-existing arm64 debt.

### The 24 red tests are the known-red baseline — DIFFED, not eyeballed

```
observed_count=24   baseline_count=24
DIFF_RC=0           IDENTICAL: observed failure set == known-red baseline
CONTROL_RC=1        (a sentinel name appended -> the diff reports a difference)
```

**Zero regressions.** The control matters: a comparison that could only ever report equality
would pass on a tree where the extraction produced an empty set. Names are the 21 `gpu::` +
3 `prune::snapshot_tests` from
`.planning/phases/03-faithful-two-stage-trainer-and-head/known-red-baseline.md`.

## Plan acceptance criteria — what was and was not met

| Criterion | Status |
|---|---|
| `cargo test ... --test setfit_apr_lifecycle` exits 0 with >= 1 test passed | **MET** — 5 passed |
| `grep -c "apr_cli\|apr-cli"` on the test == 0 | **MET** — and asserted in-test with a positive control |
| Both `cargo tree` boundary counts are 0, recorded in SUMMARY | **MET** — plus a third row and a positive control |
| Artifact `contains: "load_setfit_apr"` | **MET** — the loader is called, and refuses |
| Every asserted response value read through an accessor, no struct field access | **MET vacuously** — there is no response to read |
| `grep -c "\.embed("` >= 1 | **NOT MET** — no receiver exists |
| `grep -c "\.classify("` >= 1 | **NOT MET** — no receiver exists |
| key_link -> `VerifiedSetFitModel::classify` via `\.classify\(` | **NOT MET** |
| must_have: full chain train -> save -> load -> embed -> classify -> inspect | **NOT MET** |
| must_have: classify goes through the shared `ClassifyRequestDocument` | **NOT MET** |
| must_have: zero `apr-cli` dependencies (cargo tree) | **MET** |

`requirements-completed: []`. **OPS-01 must not be checked off in REQUIREMENTS.md.**

## Deviations from Plan

### 1. [Rule 3 — blocking] The plan's chain could not be executed; the file asserts the boundary instead

The plan's `<action>` prescribes a single test walking the whole chain. Rungs 3-6 have no
reachable input. Rather than stop with nothing (my previous run's outcome, correct then
because the file would not have compiled), the file now proves rungs 1-2 and records rungs
3-6 as measured, typed refusals. The plan's HARD CONSTRAINT was honoured: no `src/` file was
edited and no `pub(crate)` door was used.

### 2. [Rule 2 — missing critical] The self-scan needed a two-sided control and a non-vacuity check the plan did not specify

The plan asks for a `grep`-based source assertion. As specified it would have been a counter
that could only return zero. Added: a positive control against the CLI crate's own manifest,
and a non-vacuity assertion that `include_str!` actually read this file (a wrong or empty
path would otherwise report two clean zeros).

### 3. Temporary probe files created and deleted, never committed

`tests/zz_probe_0412.rs` (six runtime measurements) and `tests/zz_probe_0412b.rs` (the E0451
compile probe). Both deleted; `git status --short` confirms.

## Known Stubs

**None.** No stub, no placeholder, no `#[ignore]`, no test that cannot pass. The unreachable
rungs are absent, not faked — and their absence is asserted by tests that will turn red the
day it stops being justified.

## Threat Flags

None. No new network endpoint, no auth path, no schema at a trust boundary, no new package.
The file is an integration test that reads two files (the fixture directory and the control
manifest) and writes nothing.

## Notes for later plans and for the orchestrator

- **The F-10 action item for 04-12 in `04-ORCHESTRATOR-NOTES.md` is unsatisfiable and should
  be corrected.** It says "Reuse 04-05's substitution shape." That shape is `E0451` from
  out-of-crate (proof above). There is no OPS-01-tier workaround.
- **OPS-01 is now blocked on exactly one thing: `CALIBRATED_REGIMES` covering an encoder that
  can compute the contract's six probes.** That is a calibration run plus a deliberate edit to
  `contracts/setfit-train-lifecycle-v1.yaml` (D-10(c)) — a Phase 5 item, never an inline
  change. When it lands, this file's four blocked-rung panics fire and hand their author the
  instructions to restore the chain.
- **04-10.** This test target is cheap (5.5 s) and green; it can join the default gate as-is.
  Its `lifecycle_source_names_no_command_line_crate` overlaps your planned
  `setfit-api-boundary` Make gate — the durable version should still be yours, but note the
  in-test one carries a positive control that a `grep | wc -l` in a Makefile would not.
- **04-11.** Do not mark OPS-01 complete. The dependency-direction half holds and is measured;
  the usability half does not.
- **04-16.** Unchanged for you: your blocker is the one 04-17 closed, and this plan touched no
  `src/` file.

## Self-Check: PASSED

Files claimed, checked on disk:

```
FOUND: crates/aprender-train/tests/setfit_apr_lifecycle.rs   (613 lines)
DELETED (as claimed): crates/aprender-train/tests/zz_probe_0412.rs, zz_probe_0412b.rs
```

Commit claimed, checked in the log: `a5f1c5787`.

| Assertion | Criterion | Observed |
| --- | --- | --- |
| `git diff --diff-filter=D --name-only HEAD~1 HEAD` | empty | **empty** |
| files changed by this plan | exactly 1, the plan's named file | **1** |
| `src/` files touched | 0 | **0** |
| `STATE.md` / `ROADMAP.md` modified | no | **not touched** — the orchestrator owns them |
| scoped test target | green, counted | **5 passed, rc=0** |
| known-red failure names | identical to baseline | **DIFF_RC=0**, 24 names, CONTROL_RC=1 |
| each guard shown able to fail | 4 mutations | **4 reds**, all reverted |
| clippy actually linted the file | mechanism proven | **injected lint fired**, then reverted |

---
*Phase: 04-apr-artifact-and-production-parity*
*Plan: 12 — PARTIAL. OPS-01 is NOT satisfied; the remaining blocker is F-10, which is not this plan's to close.*
*Completed: 2026-08-15*
