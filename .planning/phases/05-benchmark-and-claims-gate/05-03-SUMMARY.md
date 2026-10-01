---
phase: 05-benchmark-and-claims-gate
plan: 03
subsystem: training
tags: [setfit, calibration-regime, epsilon-basis, contract-gate, d-04, f-10-unblock]

requires:
  - phase: 05-benchmark-and-claims-gate
    provides: "05-01's 12 banked calibration passes — the measurement bank every candidate was derived from"
  - phase: 05-benchmark-and-claims-gate
    provides: "05-02's per-regime RegimeThresholds / table_for restructure and the deliberate sole() tripwire"
  - phase: 05-benchmark-and-claims-gate
    provides: "05-14's fail-closed epsilon_basis derivation, which this plan turns from RED to GREEN"
provides:
  - "The PRODUCTION calibrated regime — all-MiniLM-L6-v2, 10 contracted seeds x 4 cell labels — frozen on the f32 rounding-noise lower bound at a human-chosen 10x factor"
  - "F-10 UNBLOCKED at commit a63bb130b: a production-encoder run can now resolve a threshold table instead of hitting UncalibratedRegime"
  - "05-03-epsilon-basis-decision.md — the candidate lower-bound tables derived by running the shipped combine, with the L1/L3 refutations"
  - "Thresholds::sole() and its regime-less wrappers REMOVED — every threshold read now names its regime"
affects: [05-07, 05-09, 05-11, 05-12, 05-13]

tech-stack:
  added: []
  patterns:
    - "a candidate rule's verdict is obtained by CALLING the single fail-closed derivation with that rule, never by open-coding a second comparison"
    - "the bound is cited, the FACTOR is chosen — and the contract records which is which, printing the bare-condition window beside the chosen one"
    - "a provisional basis says so beside the frozen values, not only in the memo"

key-files:
  created:
    - .planning/phases/05-benchmark-and-claims-gate/05-03-epsilon-basis-decision.md
    - .planning/phases/05-benchmark-and-claims-gate/05-03-prepared-edit.patch
  modified:
    - contracts/setfit-train-lifecycle-v1.yaml
    - crates/aprender-train/src/train/setfit/thresholds.rs
    - crates/aprender-train/src/train/setfit/evidence.rs
    - crates/aprender-train/src/train/setfit/mod.rs

key-decisions:
  - "D-04 selection, verbatim: option 'A — L2, freeze now (provisional)', factor '10x noise_floor'. The executor prepared the edit; the human chose the rule at a blocking checkpoint after seeing all three candidate bounds and both factor columns."
  - "The candidate tables were derived by RUNNING the shipped combine (rc=101, the expected fail-closed refusal), never by hand arithmetic — threat T-05-03-05 names hand-derivation as the tampering vector this plan exists to prevent."
  - "epsilon_basis was WIDENED to take a named LowerBound rather than gaining a sibling function: a second function would have made `grep -c 'fn epsilon_basis'` return 2, and open-coding four candidate comparisons would have destroyed 05-14's one-comparison invariant by construction."
  - "The 10x factor is recorded in the contract as CHOSEN, not cited. The contract states a clearance CONDITION and records 49x-315x as measured, never a `lower = k x floor` formula."
  - "PROVISIONAL status is stated beside the frozen values in the contract and in the Rust table's doc comment, not only in the memo — a 4-of-6-cell basis must not read as a finished matrix."
  - "05-12's SetFit compute is NOT pre-authorized. Deferred to wave 7 by explicit human instruction; nothing in the contract, code, memo or this SUMMARY records it as approved."

patterns-established:
  - "LowerBound as a named, passed rule: the fixture regime keeps the contracted near-null bound while the production regime uses the selected one, so two independent calibrations coexist without either inheriting the other's derivation"
  - "State-dependent guards migrate by asserting the state CHANGE positively rather than by quietly disappearing"

requirements-completed: [EVAL-02]

coverage:
  - id: D1
    description: "The production all-MiniLM-L6-v2 regime is calibrated: a benchmark run resolves a measured threshold table instead of UncalibratedRegime"
    requirement: "EVAL-02"
    verification:
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/thresholds.rs#production_envelope_is_calibrated"
        status: pass
      - kind: unit
        ref: "crates/aprender-train/src/train/setfit/thresholds.rs#thresholds_match_the_contract"
        status: pass
    human_judgment: false
  - id: D2
    description: "Every frozen production epsilon is PROVEN legal by running 05-14's fail-closed derivation — the four-cell combine that was RED is GREEN"
    verification:
      - kind: integration
        ref: "APRENDER_CALIBRATION_COMBINE=\"s8:13,s8:31,s8:53,s64:13\" cargo test --release -p aprender-train --lib --features setfit production_calibration -- --ignored (rc 101 -> 0)"
        status: pass
    human_judgment: false
  - id: D3
    description: "The contract edit is additive apart from named corrections: the fixture entry and its entire frozen_thresholds block are byte-untouched"
    verification:
      - kind: other
        ref: "rtk proxy git diff -U0 HEAD~2 HEAD -- contracts/setfit-train-lifecycle-v1.yaml (fixture entry deletions 0, frozen_thresholds field deletions 0)"
        status: pass
      - kind: other
        ref: "pv validate contracts/setfit-train-lifecycle-v1.yaml (0 errors); pv diff -> major, v2.0.0 -> v3.0.0"
        status: pass
    human_judgment: false
  - id: D4
    description: "The 05-02 tripwire is MIGRATED not silenced: sole() deleted, all 11 regime-less reads route through table_for"
    verification:
      - kind: other
        ref: "grep -c 'fn sole' thresholds.rs = 0; grep -c 'frozen.of(' evidence.rs = 0"
        status: pass
      - kind: unit
        ref: "cargo test -p aprender-train --lib --features setfit setfit:: (323 passed, 0 failed)"
        status: pass
    human_judgment: false
  - id: D5
    description: "The lower bound that binds, the chosen factor, the weakened claim, the provisional basis, and the D-03/L3 refutations are recorded in the contract at the strength the evidence supports"
    verification: []
    human_judgment: true
    rationale: "Whether contract prose states a weakening honestly and at the right strength is the judgement the D-04 checkpoint existed to make. It was reviewed and approved by a human; no test can assert that the wording is candid."
  - id: D6
    description: "attention_key_bias's disposition is re-derived on a measured margin, with the refuted sentences corrected and the exact-arithmetic sentence kept"
    verification:
      - kind: other
        ref: "byte-identity control: the two refuted sentences appear in deletion hunks; THE MECHANISM invariant's exact-arithmetic sentence appears in none"
        status: pass
    human_judgment: true
    rationale: "That the replacement justification is adequate — rather than merely different — is a judgement about evidentiary standards, made by the human at D-04."

duration: ~3h20m
completed: 2026-08-17

actuals:
  tokens: 46000
  tasks: 3
  commits: 3

status: complete
---

# Phase 5 Plan 03: Epsilon Basis Decision Summary

**The production all-MiniLM-L6-v2 regime is calibrated and F-10 is unblocked — frozen on the
contract's f32 rounding-noise lower bound at a human-chosen 10x factor, after the 10x/10x rule
D-03 assumed was measured collapsing for five of six parameter classes at 1536 optimizer steps.**

## Performance

- **Duration:** ~3h 20m (across a checkpoint halt and re-dispatch)
- **Tasks:** 3 of 3
- **Files modified:** 4 (+ 2 planning artifacts created)

## Accomplishments

- **F-10 unblocked at `a63bb130b`.** A production-encoder run now resolves a measured threshold
  table instead of failing closed with `UncalibratedRegime`. This is the milestone keystone every
  Phase 5 benchmark cell was blocked on.
- **D-03's premise refuted on the record, with numbers, in the contract itself** — five of six
  classes have no legal window at the production envelope, and the collapse holds within `s64`
  alone (39.7 against a rule needing > 100).
- **The replacement rule was chosen by a human at a blocking checkpoint**, from three candidates
  derived by running the shipped combine, with two of them refuted by arithmetic before the
  choice was offered.
- **05-14's fail-closed derivation went RED → GREEN on identical input**, so the frozen basis is
  proven legal by a run rather than asserted from a report.
- **05-02's `sole()` tripwire was migrated, not silenced:** all eleven regime-less threshold reads
  now name their regime, and the panicking accessor is gone rather than left armed.

## Task Commits

1. **Task 1 — derive candidates, commit the memo, prepare the edit** — `2f76df22f` (docs; memo only, no gate byte)
2. **Checkpoint halt record + prepared-edit artifact** — `d756cfc67` (docs)
3. **Task 3 — the approved gate edit** — `a63bb130b` (feat; contract + thresholds.rs + evidence.rs + mod.rs)

`git log --oneline` order: memo → halt record → gate commit. The halt record sits between them
because the D-04 checkpoint genuinely stopped execution; it is left in history rather than
squashed, since the ceremony is the point.

## The D-04 decision, recorded verbatim

> Option: **"A — L2, freeze now (provisional)"**
> Factor: **"10x noise_floor"**
> 05-12 compute pre-authorization: **"Defer — ask me at wave 7"**

The human was presented with all three candidate lower bounds, both factor columns, the
claim-given-up statement at full strength, the L3 arithmetic refutation, the
`attention_key_bias` disposition with its three separately-labelled margins, the coverage
arithmetic for the two unmeasured cells, `pv diff` + bump, the byte-identity control, and 05-14's
derivation result. The executor prepared the edit; it did not choose the rule.

**05-12's SetFit compute is NOT pre-authorized.** It is deferred to wave 7. Nothing in the
contract, the code, the memo or this SUMMARY records it as approved, and 05-12 must ask
separately.

## What was frozen

| class | frozen eps | over 10x its noise floor | over the bare floor |
|---|---|---|---|
| `embedding` | 1.8e-4 | 10.1x | 101x |
| `layer_norm_weight` | 1.8e-5 | 30.2x | 302x |
| `layer_norm_bias` | 7.1e-5 | 119x | 1191x |
| `projection_weight` | 1.2e-4 | 201x | 2013x |
| `projection_bias` | 3.4e-5 | 57.0x | 570x |
| `attention_key_bias` | **null — ungated** | — | — |

Run-level `embedding_delta_floor` **2.6e-4**, from the smallest embedding MEDIAN across measured
cells (2.611e-3) over ten, rounded down — read off a run, not hand-computed.

Each value is `best_real / 10` rounded DOWN to two significant figures: the same upper edge and
the same rounding rule as the fixture derivation. **The change of lower bound does not touch the
upper edge** — only which values are LEGAL changed, not how they are computed.

**The bare-floor clearance band 101x–2013x is comparable to the fixture's recorded 49x–315x**,
which is the sense in which the contracted clearance condition is satisfied at least as well
here as where it was first recorded.

## What the gate gives up, recorded in the contract

**The gate no longer refuses a genuinely-executed but ~2000x-underpowered contrastive run.** A run
at `encoder_lr = 1e-8` moves `projection_bias` by `1.101e-4` at s64, and the frozen `3.4e-5` sits
below that, so such a run passes.

**What is retained, with its margin:** all three of the adversaries the contract NAMES — a frozen
encoder, a centroid baseline, and a 1e-30-LR run — produce exactly `0.000e0` at 1536 steps
(measured: `ctrl_max = 0.000e0` for all six classes in every cell) and stay refused with the
widest possible margin. The 1e-8 near-null run was never a named adversary; 03-05 introduced it
to obtain a non-degenerate lower bound because the 1e-30 control bounds epsilon from above only.

The contract states the new claim explicitly rather than substituting one bound for the other
silently: *the encoder moved beyond f32 rounding noise by at least an order of magnitude, and a
run that does not train produces exactly zero.*

## The bound is cited; the FACTOR is chosen

Recorded that way in three places (contract invariant, Rust doc comment, memo). The contract
states a clearance CONDITION — no parameter may satisfy its threshold by rounding alone — and
records 49x–315x as the clearance the fixture epsilons *happened to have*. There is no
`lower = k x floor` formula anywhere in it. The 10x is this plan's choice, made because it mirrors
the near-null leg's own factor and is strictly stricter than bare clearance, and the contract
prints the bare-condition window beside the chosen one so a reader can see what the factor bought.

**Narrowed precedent, stated at the strength it holds:** the noise floor was ALREADY this
contract's operative lower bound for one class — the fixture derivation table records
`layer_norm_weight` with `worst 1e-8` = `0.000e0` and `lower edge` = `0.000e0`, so its near-null
leg was degenerate and the clearance condition was doing the work. That does not mean the whole
rule pre-existed, and it does not make the multiplier inherited.

## The basis is PROVISIONAL, and says so where the numbers are

Four of six boundary cells measured (`s8:{13,31,53}`, `s64:13`); **`s64:31` and `s64:53` were
never run**. Under this lower bound an unmeasured cell CAN still narrow a window by raising the
worst noise floor — which is why it is stated beside the frozen values in the contract and in the
Rust table's doc comment, not only in the memo.

The factor that would close each window is its width: `attention_key_bias` 1.51x (ungated, so it
cannot change the verdict), `embedding` 10.19x, and 31.72x–206.56x for the rest. Noise floors
have been stable across four cells spanning a 64x range in optimizer steps (`5.960e-8` for every
dense class in every cell; `1.741e-6`–`1.779e-6` for embedding).

**A measured result whose absence would have been invisible:** the four-cell noise floors were
re-derived from the run rather than carried forward from the s8 half, and they COINCIDE —
`s64:13` did not raise any class's floor. Folding the s64 cell in moved `best_real` but left
`lower` alone.

`PROSPECTIVE VALIDATION (s16 seed 41, s32 seed 29): NOT RUN — compute budget`, recorded in the
contract as that literal rather than as a claim.

## Refutations on the record

**D-03's premise (L1):** five of six classes EMPTY, exceed factors 8.56x / 13.84x / 8.54x /
31.94x / 5.41x, reproducing 05-01 FINDING 1 to the digit; the collapse holds within `s64` alone
at 39.7 against a rule needing > 100. The cause is the near-null leg, not the control: the 1e-30
control writes back bit-identical weights even at 1536 steps.

**L3 (relaxing the factors), by arithmetic rather than by taste:** largest admissible near-null
factor **0.313**, largest admissible factor PRODUCT **3.131** against the contracted 100, binding
class `projection_bias`. Below 1 the margin is INVERTED — epsilon would sit under the near-null
delta it must exceed, so a near-null run would PASS. Both figures were recomputed from the run.

## `attention_key_bias`

Ungated, now on a MEASURED margin rather than the refuted gradient-free argument. Three
quantities kept separate, because conflating them credits the epsilon with a margin it lacks:

| quantity | value | the other five |
|---|---|---|
| raw separation `best_real / noise_floor` | 151x | 1019x – 20654x |
| the candidate epsilon's clearance `upper / noise_floor` | 15.12x | 101x – 2065x |
| window WIDTH under the chosen bound | 1.51x | 10.19x – 206.56x |

**Corrected per D-17:** "those deltas are f32 cancellation residue" and the unqualified "softmax
shift-invariance makes dL/db_k exactly zero" — both refuted by `grad_norm_max` 8.084e-10 (s8) /
7.298e-10 (s64) and by deltas scaling with the learning rate. **Kept:** `dL/db_k = 0` in EXACT
arithmetic, verbatim in THE MECHANISM invariant. The physics explains why the gradient is tiny;
it does not establish that it is zero.

The apparent D-17 collision on `15.1 / 10 = 1.51` is disarmed in the contract and the memo: D-17
forbids reading `upper / noise_floor` as a MARGIN for an epsilon with no legal window. Under this
bound the window is non-empty, so `upper` IS a legal epsilon and `upper / lower` is a width —
arithmetic on two measured quantities.

## Verification at the COMMITTED state

Every status captured directly (`cmd > /tmp/out.log 2>&1; rc=$?`), never through a pipe.

| Check | Command | Result |
|---|---|---|
| commit shape | `git log -1 --name-only --format=` | **4 files** — contract, `thresholds.rs`, `evidence.rs`, `mod.rs`; memo NOT included |
| contract valid | `cargo run --release -p aprender-contracts-cli --bin pv -- validate ...` | **rc=0, 0 error(s), 0 warning(s)** |
| version bump | `pv diff /tmp/t3-old.yaml contracts/...` (old materialized with `git show HEAD~2:`) | **`v2.0.0 → v3.0.0`, suggested bump: major** — and the file declares 3.0.0 |
| **05-14 derivation GREEN** | four-cell combine, identical input | **rc=0** (was `rc=101`) |
| — and it states its reason | `/tmp/t3-green.txt` | `REGIME TABLE ...: resolved`; `Every class above has a non-empty window`; `PASSES RUN: 12` |
| byte-identity: fixture entry | `rtk proxy git diff -U0 HEAD~2 HEAD -- <contract>` | **0** deleted `calibrated_regimes` fixture lines |
| byte-identity: fixture table | same | **0** deleted `frozen_thresholds` field lines (20 deletions total, all amended prose + header) |
| 40-cell envelope | `cargo test ... production_envelope_is_calibrated` | **rc=0, 1 passed** |
| `UncalibratedRegime` negatives | `cargo test ... regime` | **rc=0, 15 passed** |
| 05-14's guards | `cargo test ... epsilon_basis` | **rc=0, 9 passed** |
| **full `setfit::` suite** | `cargo test ... setfit::` | **rc=0, `323 passed; 0 failed; 3 ignored`** |
| binding audit | `make contract-audit-phase4` | **rc=0**, 15 bound / 15 implemented, `No binding gaps found` |
| — `BIND-` count | counted in command substitution, never via a redirected file | **0** anchored, **0** unanchored (case-table control: the anchored pattern is not silently blind) |
| clippy | `cargo clippy --release ... --all-targets` | **rc=0, 0 findings** |
| rustfmt | `rustfmt --check` on the three Rust files | clean (only pre-existing `apr_reload.rs:331`, reached via `mod` traversal) |

**The RED → GREEN arc, quoted at three points** so it is legible without re-deriving it:

| state | four-cell combine | rc |
|---|---|---|
| 05-01 recorded (pre-05-14) | identical input | `0` — the D-18 defect: green with five empty windows |
| after 05-14, before this plan | identical input | `101` |
| **after this commit** | identical input | **`0`** — for a reason the contract now records |

**The per-regime association loop is live and BITES.** Proven by induced mutation, not asserted:
perturbing the production `embedding` epsilon `1.8e-4` → `1.9e-4` turned
`thresholds_match_the_contract` RED, naming the production regime and the class. Reverted; suite
back to green.

**`production_envelope_is_calibrated` is two-sided:** all 40 production ids (rendered through
`RegimeCoordinates::render_run`, the run's own grammar) resolve the PRODUCTION table; a fixture id
still resolves the FIXTURE table; and three out-of-envelope coordinates still resolve nothing — so
the entry is a measured envelope, not an architecture-wide permit.

## For 05-07: an expectation, not a result

`apr setfit train` on the production encoder is now **expected** to reach exit 0 — the regime gate
that returned `UncalibratedRegime` at rung 4 of 04-15's ladder now resolves a table. **This plan
did not run that ladder and does not claim it passes.** Proving it end-to-end is 05-07's
obligation, and the unblock commit to reference is **`a63bb130b`**.

## Deviations from Plan

### 1. [Rule 2 — Missing critical] The run-level floor was the one frozen number the report did not derive

- **Found during:** Task 1, deriving the recommended table.
- **Issue:** the contract derives `embedding_delta_floor` from the smallest embedding MEDIAN, but
  the report printed only `EMBEDDING DELTA MIN`. The floor would have had to be hand-computed —
  the exact hand-derivation T-05-03-05 exists to prevent.
- **Fix:** one accumulator and one report line (`EMBEDDING DELTA MEDIAN MIN ...: 2.611e-3`).
- **Commit:** `a63bb130b`.

### 2. [Rule 3 — Blocking] `epsilon_basis` widened rather than duplicated

- **Issue:** four candidate emptiness determinations were needed while 05-14 pins the comparison
  count at one, and a helper named `epsilon_basis_with` would have made
  `grep -c 'fn epsilon_basis'` return 2 — failing the plan's own criterion.
- **Fix:** widened the single function with a named `LowerBound`; every call site passes an
  explicit rule. Both counts remain **1**.
- **Commit:** `a63bb130b`.

### 3. [Rule 3 — Blocking] Two of 05-14's state-dependent tests asserted the production regime is uncalibrated

- **Issue:** one asserted `table_for(PRODUCTION_REGIME_TODAY).is_none()`, the other
  `calibrated.len() == 1`. Both true only until this plan landed.
- **Fix:** the first now derives under `UNCALIBRATABLE_REGIME` (red by construction — which is why
  05-14's own doc said the DURABLE test uses it) and asserts the state change **positively**, so
  the transition is recorded by a test rather than by a test's disappearance; a control was added
  proving the production table's *declaration* is what moves the verdict. The second asserts 2 and
  gained a check that no calibrated entry carries the `minilm-full-` rendering — architecture is
  matched for equality, never by family.
- **Commit:** `a63bb130b`.

### 4. [Scope boundary, then explicitly authorized] `mod.rs` was edited in this commit

- **Issue:** `mod.rs:1690` asserted `calibrated.len() == 1`. `mod.rs` is outside the plan's
  `files_modified` and a parallel executor (05-05) was reported as touching it.
- **Handling:** at the checkpoint I reported it as a deviation with the exact patch rather than
  editing it. The coordinator verified independently that 05-05 touches `mod.rs` only at line 43
  (a `pub mod bench_row;` declaration), that 05-05 had completed, and explicitly authorized the
  one-line change as part of this commit.
- **Consequence:** the plan's Task 3 verify block asserts the commit names EXACTLY three files;
  it names **four**. That literal is superseded by the coordinator's instruction, and the fourth
  file is a single assertion literal, not a gate change.
- **Commit:** `a63bb130b`.

### 5. [Rule 3 — Blocking] The prepared edit was preserved as a committed patch across the checkpoint

- **Issue:** D-04 requires the gate edit to stay uncommitted until approved, but this executor
  runs in a worktree the orchestrator force-removes on return — an uncommitted edit is destroyed.
- **Fix:** `05-03-prepared-edit.patch`, committed at `d756cfc67`. A patch file is not a gate
  change: nothing reads it, no `include_str!` parses it, no test consults it.
- **NOW SUPERSEDED.** The edit it records is committed at `a63bb130b`. The file is retained as the
  checkpoint's audit trail — **do not `git apply` it**; it would conflict with the committed
  state. It was left in place rather than deleted because the coordinator's Task 3 instruction
  scoped this commit to the approved edit, the `mod.rs` line and the SUMMARY.

### 6. [Measurement discipline] Two measurements were re-taken because the first was rewritten

- The `rtk` hook rewrites `git diff` into a lossy prose summary. The first byte-identity control
  therefore reported **19** deletion lines; re-run through `rtk proxy`, the true count is **20**
  (the hidden line was `-  version: 2.0.0`, which is in the metadata header and within the
  permitted set, so the conclusion held — but the first measurement could not have supported it).
- The same rewrite made `git diff > patch` produce a **non-applicable** file. Caught by
  `git apply --check --reverse` returning 128, fixed by regenerating through `rtk proxy`,
  re-checked rc=0.
- Both are the CLAUDE.md rule-1/rule-8 family, and both were caught by verifying the measurement
  rather than the result.

---

**Total deviations:** 5 auto-handled (1 missing-critical, 3 blocking, 1 measurement) + 1 scope
escalation resolved by explicit authorization.
**Impact:** no scope creep. Every change was required for correctness or by the plan's own
criteria; the one out-of-scope edit was escalated at the checkpoint and authorized before being
made.

## Issues Encountered

The plan was re-dispatched after a prior executor halted on a host ENOSPC blocker before Task 1.
That halt left no partial state to reconcile — the 12 banked calibration passes were intact, so no
measurement compute had to be re-spent. This run began from Task 1 unchanged. Disk stayed
comfortable throughout (136 GiB free at the end); `CARGO_INCREMENTAL=0` was exported on every
build.

## Known Stubs

None. Every symbol introduced is reached by a default-suite test, and the one report-only renderer
is exercised by the combine run quoted above.

## Deferred / out of scope (NOT fixed)

- **`mod.rs:1692` and `:1720` use `calibrated[0]`**, which is position-dependent now that the list
  has two entries. They are still CORRECT (the fixture entry is first) and were left alone: the
  coordinator scoped this commit to the one `mod.rs` line. Worth converting to
  `calibrated.iter().any(...)` in a plan that already owns the file.
- **`crates/aprender-train/src/train/setfit/apr_reload.rs:331`** remains unformatted at HEAD —
  pre-existing, untouched here, already logged by 05-02 and 05-14. `rustfmt` was run on the files
  this plan owns rather than `cargo fmt -p aprender-train`, so the pre-existing diff was not
  silently absorbed into this plan's changes.

## Threat Flags

None. No network endpoint, auth path, file-access pattern or trust-boundary schema was introduced.
The register is discharged rather than extended:

- **T-05-03-01** (gate loosening) — full-list string equality over the 2-entry list preserved;
  `len()` literal exactly 2; human checkpoint before any commit.
- **T-05-03-02** (fixture semantics) — byte-identity control with zero fixture deletions; the
  per-regime association test now compares the fixture block too, so drift is red.
- **T-05-03-03** (spoofing) — architecture component byte-copied from a measured run id; the
  envelope test renders all 40 ids through the run's own grammar; the provenance note forbids
  prefix aliasing and the negatives prove `minilm-full-` still resolves nothing.
- **T-05-03-04** (repudiation) — `pv diff` + bump recorded in the header; one commit; the
  selection recorded verbatim.
- **T-05-03-05** (a rule chosen to close the table) — every candidate derived by RUNNING the
  shipped combine; L3's admissible factors printed so relaxation is refuted by arithmetic; the
  selection was the human's; the frozen basis makes 05-14's derivation return the success value.
- **T-05-03-06** (a weakened claim recorded as unchanged) — the contract states the weakened claim
  explicitly, naming which adversaries remain refused and which run would now pass.
- **T-05-03-07** (exemption by omission) — `attention_key_bias`'s exclusion is a recorded contract
  declaration with measured numbers; with no table resolved every class is required, proven by a
  default-suite test.

## Next Phase Readiness

**F-10 is unblocked at `a63bb130b`.** 05-07, 05-12 and (transitively) 05-09, 05-11, 05-13 are
released to run.

Two things the next plans must carry rather than rediscover:

1. **05-07** owns proving the `apr setfit train → inspect → eval → predict` ladder reaches exit 0.
   This plan states that as an expectation only.
2. **05-12 must ask for its SetFit compute separately** — it is explicitly NOT pre-authorized, and
   the human deferred the question to wave 7. 05-12 also runs cells SEQUENTIALLY; parallel cell
   processes would corrupt EVAL-05's resource measurements, so a wall-clock overrun is a
   checkpoint, never a reason to parallelize.

The provisional basis is the one standing caveat: if a later reviewer wants the asterisk removed,
`s64:31` and `s64:53` are ~5.6 h of retryable compute and the store is already set up to bank them
without redoing anything.

## Self-Check: PASSED

| Claim | Check | Result |
|---|---|---|
| `05-03-epsilon-basis-decision.md` exists | `test -f` | FOUND |
| `05-03-prepared-edit.patch` exists | `test -f` | FOUND |
| commit `2f76df22f` (memo) | `git log` | FOUND |
| commit `d756cfc67` (halt record) | `git log` | FOUND |
| commit `a63bb130b` (gate edit) | `git log` | FOUND |
| gate commit names 4 files, memo excluded | `git log -1 --name-only --format=` | CONFIRMED |
| contract at committed state validates | `pv validate` | rc=0, 0 errors |
| 05-14 derivation GREEN at committed state | four-cell combine | rc=0 |
| `setfit::` suite | `rtk proxy cargo test` | 323 passed, 0 failed |
| exactly one top-level `status:` in this frontmatter | `grep -c '^status:'` | 1 |
