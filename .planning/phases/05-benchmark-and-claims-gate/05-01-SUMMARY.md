---
phase: 05-benchmark-and-claims-gate
plan: 01
subsystem: testing
tags: [setfit, minilm, calibration, evidence-gate, contract-gate, compute-budget]

requires:
  - phase: 03-faithful-two-stage-trainer-and-head
    provides: "UpdateEvidence / ParameterClass / rounding_noise_floor and the fixture calibration matrix whose mechanics this harness replicates at production scale"
  - phase: 01-differentiable-minilm-conformance
    provides: "SetFitMiniLm::from_pretrained_dir and the pinned 86.7 MB all-MiniLM-L6-v2 checkout (revision 1110a243), plus the APRENDER_MINILM_DIR env-gate pattern"
provides:
  - "production_calibration_matrix: an #[ignore]d, env-gated in-crate harness that measures the production encoder's per-class relative-delta distributions with probe / prospective / boundary-matrix modes"
  - "Frozen production hyperparameters epochs=1, batch=16, read from the pinned setfit 1.1.3 uv environment with command and stdout recorded"
  - "The production calibration regime id, RENDERED by production code and quoted verbatim: minilm-slice-h384-l6-a12-i1536-v30522@1110a243|seeds=13|cells=s8e1b16"
  - "A measured compute projection for the 18-pass boundary matrix, and the corrected s64/s8 weighting (64x, not the plan's ~8x)"
  - "Proof that debug and release profiles produce BIT-IDENTICAL calibration measurements"
affects: [05-02, 05-03, 05-07, 05-12]

actuals:
  tokens: 154000
  tasks: 1        # Task 1 complete; Task 2 partial (12 of 18); Task 3 not completable as specified
  commits: 15

tech-stack:
  added: []
  patterns:
    - "Env-gated, #[ignore]d in-crate measurement harness that skips with a named remedy when the fetched checkout is absent"
    - "Mode selection by environment variable (probe / prospective / full matrix) so an expensive matrix can be projected before it is committed to"

key-files:
  created:
    - .planning/phases/05-benchmark-and-claims-gate/05-01-calibration-measurements.md
  modified:
    - crates/aprender-train/src/train/setfit/evidence.rs

key-decisions:
  - "The production regime id is rendered by calibration_regime_id from the run's own coordinates and printed, never composed in the harness (T-05-01-01)"
  - "The production corpus is regenerated at 64 rows per class rather than reusing the 16-row fixture pool, because an s64 selection cannot be drawn from a 16-row pool"
  - "The pair budget is left to the contracted closed form (PairConfig::budget = None) because that is what a production benchmark cell will run"
  - "Halted at the CLAUDE.md 60-minute compute check-in rather than starting an 8.3-hour matrix unilaterally"

patterns-established:
  - "Measure the profile question instead of assuming it: a release cross-check proved debug/release measurement equivalence, converting 'release would be faster' from an assumption into a fact that also licenses release for the real run"
  - "Refute a plan's own projection arithmetic with the measured step count rather than inheriting it"

requirements-completed: []

coverage:
  - id: D1
    description: "production_calibration_matrix harness exists, is #[ignore]d, env-gated on APRENDER_MINILM_DIR, and runs green in probe mode against the pinned production checkout"
    requirement: EVAL-02
    verification:
      - kind: integration
        ref: "CARGO_INCREMENTAL=0 APRENDER_CALIBRATION_PROBE=1 cargo test -p aprender-train --lib --features setfit production_calibration -- --ignored --nocapture"
        status: pass
    human_judgment: false
  - id: D2
    description: "Production epochs/batch frozen from the pinned setfit 1.1.3 environment before any calibration pass, with the literal command and stdout recorded"
    requirement: EVAL-02
    verification:
      - kind: integration
        ref: "cd scripts/setfit_fixtures && uv run python -c \"from setfit import TrainingArguments; a = TrainingArguments(); print(a.num_epochs, a.batch_size, a.body_learning_rate)\""
        status: pass
      - kind: unit
        ref: "production_config asserts reference.epochs()==1 and reference.batch_size()==16"
        status: pass
    human_judgment: false
  - id: D3
    description: "No gate, threshold, tune or contract file modified by this plan"
    requirement: EVAL-02
    verification:
      - kind: unit
        ref: "CARGO_INCREMENTAL=0 cargo test -p aprender-train --lib --features setfit thresholds (27 passed)"
        status: pass
      - kind: other
        ref: "git diff --name-only -> crates/aprender-train/src/train/setfit/evidence.rs only"
        status: pass
    human_judgment: false
  - id: D4
    description: "Compute go/no-go for the 18-pass boundary matrix (8.29 h release / 285.8 h debug, both over the CLAUDE.md 60-minute check-in threshold) — RESOLVED: human chose Option B, full matrix on lambda-vector"
    verification:
      - kind: other
        ref: "coordinator message 2026-08-17: Option B, full contracted matrix, trims C1/C2/C3 and option D rejected"
        status: pass
    human_judgment: true
    rationale: "CLAUDE.md requires a human check-in BEFORE compute spend over 1 hour on non-lambda-vector hosts. Resolved by explicit human decision."
  - id: D5
    description: "Dispatch of the boundary matrix to lambda-vector — BLOCKED, host unreachable; superseded by Option A (run locally, authorized)"
    verification:
      - kind: integration
        ref: "ssh -o BatchMode=yes -o ConnectTimeout=5 lambda-vector hostname -> rc=255, 'Could not resolve hostname'; ping/dscacheutil/known_hosts/ssh-config/tailscale all negative"
        status: fail
    human_judgment: true
    rationale: "No dispatch path existed. Superseded by an explicit human authorization to run the full matrix locally."
  - id: D6
    description: "s8 half of the boundary matrix: {s8} x seeds {13,31,53} x {real, control, near-null}, 9 passes, separation ctrl_max < real_min holding for every gated class in every cell"
    requirement: EVAL-02
    verification:
      - kind: integration
        ref: "APRENDER_CALIBRATION_CELLS=s8:13,s8:31,s8:53 cargo test --release ... production_calibration -- --ignored --nocapture -> rc=0, 9 passes, 439.4 s, STATUS: COMPLETE"
        status: pass
    human_judgment: false
  - id: D7
    description: "Harness made crash-persistent (per-cell report flush with PARTIAL/COMPLETE banner) and resumable (APRENDER_CALIBRATION_CELLS chunking with full 3 conditions)"
    verification:
      - kind: integration
        ref: "one-cell chunk rc=0 249 s; s8 half chunk rc=0 542 s; both reports carry STATUS: COMPLETE"
        status: pass
    human_judgment: false
  - id: D8
    description: "s64 half of the boundary matrix (9 passes, ~7.81 h re-projected from measured s8 cost) — NOT RUN, halted for a decision after the first attempt was killed externally at pass 9 of 18"
    verification: []
    human_judgment: true
    rationale: "Standing instruction on partway failure: report what completed, what failed and the measured cost, and halt rather than silently restart or reduce scope. A single ~7.8 h unattended invocation also exceeds the session's background-task lifetime, so the execution shape (per-seed chunks vs one relaunch vs out-of-session) is a decision, not an executor default."
  - id: D9
    description: "gradient_free re-derivation for attention_key_bias — PRELIMINARY verdict on 3 s8 seeds: the class moves and its movement scales with the learning rate, so the fixture's bit-exactness justification does not survive production scale; a window still exists but its margin is the narrowest of six classes"
    requirement: EVAL-02
    verification:
      - kind: integration
        ref: "s8 half report: real_min 1.714e-7/1.721e-7/1.801e-7 vs near-null max 3.105e-11/5.162e-11/4.430e-11, grad_norm_max 8.084e-10, noise floor 1.133e-9, eps/noise 15.1"
        status: pass
    human_judgment: true
    rationale: "Three seeds at one shot count is one cell. The s64 cells run 64x the optimizer steps and accumulated residual is exactly what could move real_min. CLAUDE.md rule 6 -- not a final verdict until the s64 half lands, and the verdict changes 05-03's contract edit."

duration: 175min
completed: 2026-08-17
status: partial
---

# Phase 5 Plan 01: Production Calibration Measurements Summary

**The s8 half of the production boundary matrix is measured across three seeds with full
controls and clean separation, and it already shows that `attention_key_bias` — the class the
fixture leaves ungated as gradient-free — moves in proportion to the learning rate at production
scale; the s64 half (~7.81 h) is unrun after the first attempt was killed externally at pass 9
of 18, and is halted for a decision on execution shape.**

## Performance

- **Duration:** ~10 h wall clock (12 banked tuning passes ≈ 3.6 h of `run_tuning`; one killed
  chunk ~56 min; harness work, the determinism proof and analysis the remainder)
- **Tasks completed:** 1 of 3 — Task 2 PARTIAL (12 of 18 passes), Task 3 NOT completable as
  specified (the ε windows do not exist for 5 of 6 classes)
- **Commits:** 15

## STATUS: PARTIAL — 12 of 18 passes; stopped by human decision, not by failure

**Measurement was stopped deliberately, and stopping was correct.** The remaining six passes
cannot change the verdict: `lower = 10 × worst_nnull` can only **rise** as more near-null passes
land, and `upper = best_real / 10` can only **fall** as more real passes land, so every window can
only narrow further. Spending another ~5.5 h to confirm a determined outcome was rejected in favour
of closing out with the result in hand.

**All 12 passes are committed as reviewable artifacts** in
`.planning/phases/05-benchmark-and-claims-gate/calibration-store/` (evidence + metadata per pass,
each digest independently reproducible with `shasum -a 256`), so `s64:31` and `s64:53` can be
measured later without redoing any of this.

**On the plan's own terms this is a success.** 05-01 front-loaded the measurement precisely so F-10
would fail early if it was going to — and it did, at 12 passes rather than after ε had been frozen
and built upon. Four real defects surfaced along the way, none of which a plan assuming its rule
worked would have found: the false-green test child that ran zero tests and exited 0; the
unconditional ε-basis header inviting a freeze from half the matrix; the fail-closed hole letting
`rc=0` coexist with an empty ε basis; and the refuted gradient-free premise for
`attention_key_bias`.

**The six findings 05-03 inherits** are written up in
`05-01-calibration-measurements.md` → *PLAN CLOSEOUT*: (1) the 10×/10× rule does not survive
production step counts for 5 of 6 classes; (2) separation is **intact** — this is not "calibration
is impossible"; (3) `COMBINE` exits `rc=0` with an empty ε basis, a fail-closed hole 05-03 must
close; (4) `attention_key_bias`'s "receives no gradient" justification is refuted by measurement;
(5) the superlinear near-null growth **hypothesis**, explicitly untested; (6) the cross-label check
**passed**, which is what makes (1) a result about step count rather than two different models.

## Earlier status (superseded, kept for the record): s8 half measured, s64 half awaiting a decision

**Current state (supersedes the two earlier halt states below, which are kept for the record).**
The compute gate was resolved twice: first Option B (lambda-vector), which proved unreachable;
then **Option A — run the full matrix locally in release, ~8.29 h explicitly authorized**.

**PHASE-LEVEL FINDING: the 10× ε window rule collapses at s64 for five of six classes.**
`s64:13:near-null` is banked (12 of 18 passes), and it moved the lower bound far more than the s8
half predicted. `attention_key_bias` near-null went from **3.105e-11 (s8) to 9.278e-9 (s64) — 299×**,
so its window is `lower = 9.278e-8` vs `upper = 1.714e-8`: **empty, lower exceeding upper by 5.4×**.
The `eps/noise = 15.1` figure must **not** be carried forward — that column is computed whether or
not a window exists, so when `supports_margin` is `false` it divides an illegal ε by the noise floor
and is meaningless, not merely smaller. Only `layer_norm_weight` retains a window. The collapse is
real within s64 alone (`real/near-null = 39.7`, rule needs `> 100`), not an artifact of cross-cell
mixing. **Separation is unaffected** — `ctrl_max = 0.000e0 < real_min` for all six classes,
`rc=0` — so real training remains cleanly distinguishable from none; what fails is freezing ε by
the current rule. That is a 05-03 decision, not a number to pick here.

**Pre-registered cross-label check: PASSES.** A genuinely cross-half combine
(`s8:13,s8:31,s8:53,s64:13`, 12 passes, both labels in one invocation) evaluated s64 ids against an
architecture component taken from an s8 id. `minilm-slice-h384-l6-a12-i1536-v30522@1110a243` is
**byte-identical across `cells=s8e1b16` and `cells=s64e1b16`** — both halves measured the same
22M-parameter production encoder, which is what makes the finding above a result about step count
rather than two different models.

**`s64:13:control` banked — 11 of 18 passes.** `evidence_sha256=9c969aa9…c704`, `steps=1536`,
`wall_clock=3348.9s`, regime byte-identical to the cell's real pass. Two findings: the survivable
window is **at least 57.5 min** (the earlier 55.8 min was a death observation, i.e. a floor, not a
ceiling — a 55.82 min pass completed comfortably); and the **control leg does not erode at s64** —
at 1536 steps the 1e-30 control still writes back bit-identical weights (max `relative_delta` 0.0,
0 rows moved, 0 support), so `ctrl_max` stays `0.000e0` and only the near-null leg and `best_real`
can still move `attention_key_bias`'s 15.1 margin.

**D′ is now VALIDATED END-TO-END at s8.** The nine s8 passes were re-banked (28.7 min, all
`rc=0`) and `APRENDER_CALIBRATION_COMBINE="s8:13,s8:31,s8:53"` ran the separation assertion over
the persisted tables with no training. **Every measured digit is identical to the in-process s8
run** — all 18 per-cell rows × 9 columns, all 6 ε-basis rows × 9 columns, every binding row, and
`attention_key_bias eps/noise = 1.51e1` unchanged. The only difference is wall clock (501.4 s vs
439.4 s), which is metadata and excluded from the evidence file by design. The combine path is
the equivalence it was argued to be, so the remaining eight s64 passes are licensed. The store
holds **10 of 18 passes**, all committed.

**D′ is implemented, its precondition is PROVEN, and the first s64 pass is banked.** Cross-process
determinism was proven before any s64 compute was spent: two fresh processes running the same
s8 pass persisted **bit-identical** evidence (`5838b3d2…57dc`, 48 555 bytes) while disagreeing by
12.35 s of wall clock — confirmed both by the in-tree test
`cross_process_determinism_of_persisted_evidence` and independently at the shell. Then
`s64:13:real` ran to completion and was persisted:
`s64e1b16-seed13-real.evidence.json`, `evidence_sha256=23e8b60d…0665`, `steps=1536`,
`wall_clock=3327.1s`. **1 of 9 s64 passes banked; STATUS: PARTIAL** — one condition supports no
separation assertion. Note the margin: 55.45 min against a ~55.8 min window, ~20 s of headroom,
so expect some passes to need retrying — which under D′ costs one pass, not a cell.

**A′ chunk 1 (`s64:13`) was attempted and killed at 55.8 min with 1 of 3 passes done and no
tables written.** It did yield the decisive measurement: an s64 pass costs **3 246.2 s
(54.1 min)** at `steps=1536` (the closed form confirmed at the far end of the envelope), against
a **measured ~55.8 min survivable unattended window**. An s64 *cell* is 2.71 h and is atomic for
the separation assertion, so **no in-session chunking at cell granularity can complete one** —
A′ as dispatched is refuted by measurement, and re-dispatching `s64:31` unchanged would burn
another ~56 min for nothing. Options are in the measurements file; the recommendation is **D′**
(per-condition persistence, making a chunk one retryable 54-min pass) or **C′** (run
out-of-session).

Task 2 is **9 of 18 passes complete**:

- The `s8e1b16` half is **measured across all three seeds with the full three-condition
  treatment** (`rc=0`, 439.4 s of tuning, `STATUS: COMPLETE`). Separation `ctrl_max < real_min`
  holds for every gated class in every s8 cell, asserted in-harness.
- The `s64e1b16` half is **not run**. Re-projected from the measured s8 cost: ~52 min per pass,
  **~7.81 h** for the nine passes.

The first unchunked full-matrix attempt was **killed externally at pass 9 of 18** (~40 min in).
Not a defect — no panic, no assertion failure, no compile error; the wrapper never reached its
epilogue, which is the signature of a terminated process rather than an exited one. It cost the
per-class tables for those nine passes because the harness only wrote its report at the end.
**That weakness is now fixed and the s8 half was re-measured.** Per the standing instruction, I
did not silently restart the whole matrix or reduce scope; the s64 decision is below.

### The highest-value result so far: `attention_key_bias` does not behave as the fixture assumes

Consistent across all three s8 seeds, the class the fixture leaves **ungated** on the
gradient-free argument (`dL/db_k = 0` by softmax shift-invariance) **moves, and its movement
scales with the learning rate**: `grad_norm_max ≈ 8.1e-10`, near-null (1e-8) max ~4e-11 vs real
(2e-5) min ~1.7e-7 — a ~3600× delta ratio for a 2000× learning-rate ratio. A parameter whose
delta tracks the learning rate is being trained, not held fixed. In `f32` the softmax
shift-invariance is only approximate, so a residual gradient at the 1e-10 level is the expected
numerical consequence rather than a bug.

It is not vacuous either: it separates real from near-null by ~3300× and sits ~150× above its
own rounding-noise floor, so a window `[5.162e-10, 1.714e-8]` exists. But its `eps/noise` margin
is **15.1 — the narrowest of the six classes by an order of magnitude** (next narrowest:
embedding at 102).

**Status: PRELIMINARY.** Three seeds at one shot count is three samples of one cell, and the s64
cells run 64× the optimizer steps — accumulated residual is exactly what could move `real_min`.
CLAUDE.md rule 6 applies: not a verdict until the s64 half is measured. **If it holds, it is a
phase-level finding for 05-03**: the *justification* recorded in the contract cannot remain
"this parameter receives no gradient", because at production scale it demonstrably does.

---

## Earlier halt state (superseded, kept for the record): dispatch target unreachable

Two sequential stops, only the second still open.

**Stop 1 (RESOLVED).** Plan 05-01 Task 1 step (4) is an explicit conditional halt: if the
projected boundary-matrix wall-clock exceeds 60 minutes, stop and wait for a human go/no-go. The
projection is **8.29 h** (release) / **285.8 h** (debug), so execution stopped and reported.
The human chose **Option B — run the full contracted matrix (`{s8, s64}` × 3 seeds × 3
conditions, 18 passes) on lambda-vector**, which is pre-authorized for compute; trims C1/C2/C3
and option D were explicitly rejected because they weaken the D-02 margin argument.

**Stop 2 (OPEN).** *lambda-vector is not reachable from this executor.* `ssh -o BatchMode=yes
lambda-vector` returns **rc=255, "Could not resolve hostname"**; DNS, `dscacheutil`,
`known_hosts`, `~/.ssh/config` (one entry, `rvsc`, an AWS EC2 host), tailscale, and the repo's
`scripts/` are all negative — the three `scripts/` mentions of lambda-vector are prose about its
disk layout and GPU, not a dispatch path. The full evidence table is in the measurements file.

**The matrix was NOT run locally.** That fallback was explicitly forbidden, and correctly so: it
would have spent the 8.29 h the human's decision redirected and turned a pre-authorized spend
into an unauthorized one.

**Tasks 2 and 3 remain not started, the boundary matrix was not run, and no ε has been derived.**
Nothing in this plan edits a contract, a threshold table, `tune.rs`, or any gate.

## Accomplishments

### 1. `production_calibration_matrix` — the harness (Task 1, step 2)

Added to `crates/aprender-train/src/train/setfit/evidence.rs` beside
`calibration_matrix_epsilon_basis`, replicating its mechanics verbatim on the production
encoder: real (2e-5) / control (1e-30) / near-null (1e-8) conditions, per-class real
min/median/max, ctrl max, near-null max + moved flags, `rounding_noise_floor`, support fraction,
binding parameter name, and the per-cell `control_max < real_min` separation assertion.

- `#[ignore]`d, and env-gated on `APRENDER_MINILM_DIR` (default
  `~/.cache/aprender/minilm-l6-v2-1110a243`), printing a SKIP that names the env var and the
  fetch script when the checkout is absent — the `full_weight_parity.rs` pattern.
- Three modes: `APRENDER_CALIBRATION_PROBE=1` (one cell, REAL only, for timing),
  `APRENDER_CALIBRATION_PROSPECTIVE="s16:41,s32:29"` (named cells, REAL only, for Task 3's
  post-freeze validation), and the default full boundary matrix.
- Measurement needs **no gate widening**: the `UncalibratedRegime` refusal lives in
  `validate_evidence` at judgement time, not in `run_tuning` / `from_tune_output`.
- The regime id is produced by calling the production `calibration_regime_id` on the run's own
  encoder / selection / config, then asserted to contain the expected cell label. The harness
  never composes the string (T-05-01-01), and it prints every rendered id verbatim while
  asserting all passes share one architecture component.

### 2. Frozen production hyperparameters (Task 1, step 1)

From the hash-locked `uv` environment, not from documentation:

```text
$ cd scripts/setfit_fixtures && uv run python -c \
    "from setfit import TrainingArguments; a = TrainingArguments(); \
     print(a.num_epochs, a.batch_size, a.body_learning_rate)"
(1, 16) (16, 2) (2e-05, 1e-05)
```

Each tuple is `(body, head)`; the contrastive body member is the one `SetFitTrainConfig`
configures. **epochs = 1, batch = 16**, encoder lr 2e-5. Frozen before any calibration pass, as
the plan's must-have truth requires. `production_config` asserts the in-repo
`REFERENCE_EPOCHS` / `REFERENCE_BATCH_SIZE` agree, so a future divergence between the pinned
Python env and the Rust recipe turns the harness red instead of quietly measuring cells that
production runs never enter.

Consequence: every cell label this milestone can enumerate is `s{shots}e1b16`, so D-02's
`cells=` component is `s8e1b16,s16e1b16,s32e1b16,s64e1b16`.

### 3. The timed probe and the rendered regime id (Task 1, step 3)

Green, `rc=0`, 1811 s wall clock, 24 optimizer steps. Quoted verbatim from `/tmp/probe.log`:

```text
CALIBRATION REGIME: minilm-slice-h384-l6-a12-i1536-v30522@1110a243
minilm-slice-h384-l6-a12-i1536-v30522@1110a243|seeds=13|cells=s8e1b16
```

The `h384-l6-a12-i1536-v30522` dimensions prove the **production** encoder was loaded, not the
97-token slice; the `minilm-slice-` prefix is hardcoded in `architecture_fingerprint` and
renders for the full model too (F-R2, as researched).

### 4. Two findings that change the plan's own arithmetic

**(a) The plan's `~8x` s64 weighting is wrong; it is `64x`.** The plan projected s64 at ~8x s8
because rows scale 8x. But the contracted default pair budget is `2·max(pos_cap, neg_cap)` and
`neg_cap = 3n²` dominates, so the budget — and the step count — is **quadratic** in shots:

| cell | budget | steps @ b16, e1 | vs s8 |
|---|---|---|---|
| `s8e1b16`  | 384    | 24 (measured)   | 1x  |
| `s16e1b16` | 1536   | 96              | 4x  |
| `s32e1b16` | 6144   | 384             | 16x |
| `s64e1b16` | 24576  | 1536            | 64x |

This is the difference between a ~6-hour projection and a ~12-day one, which is why it was
surfaced rather than silently applied.

**(b) Debug and release produce BIT-IDENTICAL measurements.** The release cross-check was run
because "release would fix the compute problem" is an assumption, and CLAUDE.md verification
rule 2 forbids labelling a run by intent. Release is 34.5x faster (2.125 s/step vs 73.3 s/step)
and **every printed relative delta, binding parameter, `delta_norm`, `init_norm`, `grad_norm_max`
and noise floor agrees to the last printed digit**. So the profile is not a degree of freedom in
the calibration: choosing release for the compute budget does not change what is measured. This
matters for 05-03 — and release is arguably the *more* faithful profile, since a user's
`apr setfit train` is a release binary.

### 5. A preliminary observation carried into Task 2 (not a conclusion)

`attention_key_bias`, the class the fixture leaves **ungated** on the gradient-free argument
(`dL/db_k = 0` by softmax shift-invariance), is not bit-frozen at production scale:
`grad_norm_max = 8.084e-10`, `real_min relative_delta = 1.714e-7`, ~150x its own
`rounding_noise_floor` of `1.133e-9`. Whether that is f32 reduction residue or a real gradient
cannot be decided without the 1e-30 and 1e-8 controls — which is exactly what Task 2's boundary
matrix runs. Recorded now so the Task 2 re-derivation answers a question that was already open
rather than one invented after seeing its own result.

## The decision that was made, and what now blocks acting on it

The human chose **Option B — the full contracted matrix on lambda-vector**, which is
pre-authorized for compute so the >1 hr check-in is *removed rather than waived*. Trims C1/C2/C3
and option D were explicitly rejected for weakening the D-02 margin argument.

**That decision cannot be executed from here: lambda-vector does not resolve.** Unblocking needs
**one** of:

1. A dispatch path from this host (a `~/.ssh/config` entry, a resolvable name or IP, or an
   overlay network), after which a fresh executor resumes at Task 2; or
2. A human running the two recorded commands directly on lambda-vector and returning
   `$SETFIT_PRODUCTION_CALIBRATION_REPORT` (default `$TMPDIR/setfit-production-calibration.txt`)
   plus `/tmp/matrix.log`. Task 2's transcription and Task 3's derivation then proceed with
   nothing re-run.

**A caveat for whoever dispatches it.** lambda-vector is a GPU host, but this measurement is
CPU-bound by construction — `production_config` sets `device: "cpu"` and the SetFit contrastive
trainer has no GPU path. Option B's benefit is *authorization*, not speed, exactly as the human's
rationale said. The 8.29 h figure is this Apple-silicon box; **lambda-vector's wall-clock is
unknown** and must be re-projected there with the same one-cell probe before the full matrix is
launched. Carrying 8.29 h over as if it measured that host would be the "label a run by intent"
error CLAUDE.md rule 2 forbids.

The original options table, for the record:

| # | Option | Cost | Evidence cost |
|---|---|---|---|
| A | Full matrix locally, **release** | 8.29 h | None — the plan as written, at the measured price |
| B | Move to **lambda-vector** (pre-authorized) | 8.29 h scaled by that host | None; also removes the check-in requirement |
| C1 | Trim boundary to `{s8, s16}` | ~38 min (under gate) | Weakest — s32/s64 covered by an unobserved 16x/64x extrapolation |
| C2 | Trim boundary to `{s8, s32}` | ~2.17 h | Moderate; still over the gate |
| C3 | Keep `{s8, s64}`, one seed | ~2.76 h | Loses cross-seed spread, which is what makes the window a window |
| D | Pin a small explicit pair budget | ~38 min | **Flagged, not offered as equal** — the cell label does not record the budget, so cells would be labelled `s64e1b16` while having trained 64x less |

Recommendation as given at the time (non-binding): **B, else A.** Options C1–C3 buy time by
weakening the margin argument in exactly the dimension the cross-AI review already flagged as
this plan's soft spot (six measured cells as an engineering margin for the other 34). **The
human chose B.**

### Reachability evidence (mechanism, not intent — CLAUDE.md rule 2)

Executing host read from the machine: `hostname` → `MacBook-Pro-7.local`;
`uname -a` → `Darwin … RELEASE_ARM64_T6041 arm64`.

| Probe | Result |
|---|---|
| `ping -c 1 lambda-vector` | `cannot resolve lambda-vector: Unknown host` |
| `dscacheutil -q host -a name lambda-vector` | no records |
| `ssh -o BatchMode=yes -o ConnectTimeout=5 lambda-vector hostname` | **rc=255**, `Could not resolve hostname` |
| `grep -i '^Host ' ~/.ssh/config` | one entry, `rvsc` (AWS EC2 eu-west-1) — not lambda-vector |
| `grep -ci lambda ~/.ssh/known_hosts` | `0` |
| `command -v tailscale` | not installed |
| `grep -rIn ssh scripts/` | no dispatch path; the three lambda-vector mentions are prose about its disk layout and GPU |
| `memory/feedback_compute_pre_authorized.md` (named by CLAUDE.md) | not present |

Every status was read directly (`cmd > log 2>&1; rc=$?`), never through a pipe (rule 1).

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 3 - Blocking] `full_weight_parity.rs` is not where the plan says it is**
- **Found during:** Task 1, `read_first`
- **Issue:** The plan cites `crates/aprender-train/tests/full_weight_parity.rs` for the
  `APRENDER_MINILM_DIR` env-gate pattern. No such file exists.
- **Fix:** Read the real one at
  `crates/aprender-core/tests/setfit_conformance/full_weight_parity.rs:54` and replicated its
  resolver exactly. A doc comment on `production_checkout_dir` records the correct path so the
  next reader is not sent to the same dead end.
- **Files modified:** `crates/aprender-train/src/train/setfit/evidence.rs`
- **Commit:** `0763656a8`

**2. [Rule 3 - Blocking] The fixture corpus cannot express an `s64` cell**
- **Found during:** Task 1, harness construction
- **Issue:** `fx::TRAIN_PER_CLASS` is 16 — "exactly the largest shot count the FIXTURE matrix
  asks for". `FewShotSelector` cannot draw 64 rows per class from a 16-row pool, so reusing
  `fx::synthetic_dataset` would have made the s64 boundary cell unrunnable.
- **Fix:** Added a production corpus generator (64 distinct rows per class from an 8x8
  modifier/object grid, with disjoint held-out validation/test material) built through the same
  `from_labeled_rows` ingest ladder. Text stays entirely synthetic (T-3-18); the production
  encoder's full 30522-token vocabulary removes the slice's vocabulary constraint.
- **Files modified:** `crates/aprender-train/src/train/setfit/evidence.rs`
- **Commit:** `0763656a8`

**3. [Scope boundary] `cargo fmt -p aprender-train` reformatted an untouched file**
- **Found during:** Task 1 verification
- **Issue:** `cargo fmt` also rewrote `crates/aprender-train/src/train/setfit/apr_reload.rs`
  (4 insertions, 5 deletions), which this plan does not touch — a pre-existing formatting drift.
- **Fix:** Reverted it (`git checkout -- apr_reload.rs`) rather than absorbing an unrelated
  change into this commit. It is logged below as a deferred item, not fixed here.
- **Commit:** n/a (reverted)

### Additions beyond the plan

**4. Dispatch-reachability probe for lambda-vector (after the checkpoint resolved)**
- **Why:** The approved option names a specific host. Reporting "ran on lambda-vector" from
  intent is exactly the CLAUDE.md rule 2 failure, so reachability was probed at the mechanism
  level (DNS, directory service, non-interactive SSH, ssh-config, known_hosts, overlay network,
  in-repo dispatch scripts) before any attempt to run. All seven paths were negative.
- **Outcome:** Halted per the coordinator's explicit instruction rather than falling back to a
  local run, which would have spent the 8.29 h the decision redirected and converted a
  pre-authorized spend into an unauthorized one.

**5. Release-profile cross-check probe**
- **Why:** The debug projection (285.8 h) was far enough over the gate that the checkpoint's
  options depended on whether release closes the gap. "Release would fix it" is an assumption;
  CLAUDE.md verification rule 2 says prove the mechanism. The measurement cost ~4 minutes and
  produced two results the checkpoint needs: the real 8.29 h figure, and the debug/release
  bit-identity that licenses running the matrix in release at all.

## Deferred Issues

- `crates/aprender-train/src/train/setfit/apr_reload.rs` is not `cargo fmt`-clean at HEAD
  (4 insertions / 5 deletions). Pre-existing, unrelated to this plan, left untouched.

## Known Stubs

None. The harness is a complete, running measurement.

**The ε derivation is ABSENT BY RESULT, not stubbed.** Task 3's per-class ε windows are not
delivered because the measurement showed they do not exist for 5 of 6 classes — see Finding 1. No
placeholder ε was written, and the report now prints `n/a` and `EMPTY` for those classes rather than
a number that would look like a margin. The two cells `s64:31` and `s64:53` are *unmeasured* and the
report names them as missing in its `PROVISIONAL` banner; nothing treats them as passing.

## Threat Flags

None. This plan installs nothing, reads only the already-materialized pinned checkout through
`SetFitMiniLm::from_pretrained_dir` (which verifies the tokenizer SHA-256 against the pin), and
writes only a test harness and a planning document.

## Verification

| Check | Command | Result |
|---|---|---|
| Probe green (debug) | `CARGO_INCREMENTAL=0 APRENDER_CALIBRATION_PROBE=1 cargo test -p aprender-train --lib --features setfit production_calibration -- --ignored --nocapture` | `rc=0`, 1 passed, 1811 s |
| Probe green (release) | same with `--release` | `rc=0`, 1 passed, 220 s |
| Gate files untouched | `cargo test -p aprender-train --lib --features setfit thresholds` | `rc=0`, 27 passed |
| Diff scope | `git diff --name-only` | `crates/aprender-train/src/train/setfit/evidence.rs` only |
| Formatting | `cargo fmt -p aprender-train -- --check` | `rc=0` |
| Lints | `cargo clippy -p aprender-train --lib --features setfit --all-targets` | `rc=0` |
| D′ cross-process determinism | `cargo test --release ... cross_process_determinism -- --ignored` | `rc=0`, 1 passed, 97 s; two child processes, **bit-identical** evidence `5838b3d2…57dc` |
| D′ determinism, independent of the test | two invocations + `shasum -a 256`, separate stores | identical digest, `BIT-IDENTICAL: yes`, 48 555 bytes each |
| D′ combine == in-process (s8) | `COMBINE="s8:13,s8:31,s8:53"` vs the in-process s8 report | `rc=0`; **every measured digit identical** (18 rows × 9 cols, 6 ε rows × 9 cols, all binding rows); only wall clock differs |
| s64 cell separation | `COMBINE="s64:13"` | `rc=0`; `ctrl_max = 0.000e0 < real_min` for all 6 classes |
| Cross-label architecture component | `COMBINE="s8:13,s8:31,s8:53,s64:13"` (12 passes, both labels) | `rc=0`; `minilm-slice-h384-l6-a12-i1536-v30522@1110a243` **byte-identical** across `s8e1b16` and `s64e1b16` |
| Empty-window reporting fix | same combine, after the fix | `eps/noise` = `n/a` and `window` = `EMPTY` for the 5 classes; explicit block names each and the factor by which `lower` exceeds `upper` |
| Persisted store integrity | `shasum -a 256` on each committed `*.evidence.json` | every digest matches its `meta.json` `evidence_sha256` |

## Self-Check: PASSED

| claim | verified |
|---|---|
| measurements file, SUMMARY, `evidence.rs` exist | 3 of 3 FOUND |
| store holds 12 passes | 12 `*.evidence.json` + 12 `*.meta.json` |
| every persisted digest matches its recorded `evidence_sha256` | **12 of 12 OK, 0 mismatches** |
| all 7 plan commits exist | 7 of 7 FOUND |
| the `1.51e1` trap is gone from the regenerated report | absent; 5 classes marked `EMPTY` |

**One methodological note on the self-check itself.** Its first commit-existence check used
`git log --oneline --all | grep "^<hash>"` and reported **all seven commits MISSING** — a false
negative, because this shell environment rewrites `git log --oneline` output (the same artifact made
an earlier `ls` of a populated directory print `(empty)`). Re-verified with `git log --format=%h`,
which is stable: all seven are present. Recorded because the failure is directional-blind — a check
that can report MISSING for a commit that exists can equally report FOUND for one that does not, so
the *method*, not just the result, had to be corrected. This is CLAUDE.md rule 1's family: when a
result looks wrong, check how it was measured.

## Next Steps — all for 05-03; this plan is closed

1. ~~Human go/no-go on compute~~ — **DONE.** 2. ~~s8 half~~ — **DONE**, 9 passes. 3. ~~Execution
   shape for s64~~ — **DONE**, D′ built, proven and validated end-to-end. 4. ~~Cross-label
   architecture check~~ — **DONE, PASSED.**
5. **05-03 must decide what replaces the 10×/10× rule.** Five of six classes have no legal ε at the
   measured boundary (Finding 1). Either the safety factors are revisited against this evidence, or
   the near-null condition's role in setting the lower bound is. **Not by picking a number that
   makes the table close.**
6. **05-03 must close the fail-closed hole** (Finding 3): an empty window must become a hard
   failure in whatever surface freezes ε. Today `rc=0` coexists with five empty windows, so a green
   run is not evidence that an ε basis exists.
7. **05-03 must rewrite `attention_key_bias`'s justification** (Finding 4). "Receives no gradient,
   `dL/db_k = 0`" is refuted by measurement. Note the class now has **no window either**, so
   "ungated because it cannot move" and "gated at a frozen ε" are *both* unavailable on this
   evidence — the decision has to be made explicitly and recorded with the measurements that force
   it.
8. **Optional, cheap:** measure `s64:31` and `s64:53` (six passes, ~5.5 h) if 05-03 wants the
   boundary fully covered. It cannot change Finding 1 — every window can only narrow — so it is
   confirmation, not evidence. The store makes it resumable at any time.
9. **Optional:** test the Finding 5 hypothesis by measuring near-null at an intermediate step count
   (e.g. s16 or s32). Worth doing only if the mechanism matters to the replacement rule; the
   hypothesis is *not* needed to act on Findings 1–4.

**The `attention_key_bias` verdict is the item to watch.** It is the open question from the
probe and 05-03's contract edit depends on it: the fixture leaves that class **ungated** on the
gradient-free argument (`dL/db_k = 0` by softmax shift-invariance), but at production scale it
shows `grad_norm_max = 8.084e-10` and `real_min relative_delta = 1.714e-7`, ~150× its own
`rounding_noise_floor` of `1.133e-9`. Task 2's 1e-30 and 1e-8 controls decide it. If the
gradient-free argument does **not** survive production scale, that changes the proposed regime
entry and must be surfaced loudly, not absorbed.

**Note for 05-03:** the `cells=` component of the proposed entry is already determined by the
frozen E/B — `s8e1b16,s16e1b16,s32e1b16,s64e1b16` — and the architecture component is already
observed as `minilm-slice-h384-l6-a12-i1536-v30522@1110a243`. Only the ε table is missing.

## Self-Check: PASSED

- `crates/aprender-train/src/train/setfit/evidence.rs` — FOUND (modified, contains
  `fn production_calibration_matrix`, `#[ignore`, `APRENDER_MINILM_DIR`)
- `.planning/phases/05-benchmark-and-claims-gate/05-01-calibration-measurements.md` — FOUND
- Commit `0763656a8` — FOUND in `git log`
