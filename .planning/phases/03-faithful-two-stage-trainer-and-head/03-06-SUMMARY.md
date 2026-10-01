---
phase: 03-faithful-two-stage-trainer-and-head
plan: 06
subsystem: training
tags: [setfit, evidence-gate, typestate, contract, calibration, safe-03, trn-03, softmax-invariance]

requires:
  - phase: 03-faithful-two-stage-trainer-and-head
    plan: 05
    provides: "run_tuning, UpdateEvidence/EvidenceSummary, classify_parameter, the 6-cell calibration matrix and the relative-delta formula"
  - phase: 03-faithful-two-stage-trainer-and-head
    plan: 03
    provides: "SetFitRun<Prepared> + prepare(), the 12-knob ResolvedSetFitConfig, epoch_pair_order"
  - phase: 03-faithful-two-stage-trainer-and-head
    plan: 04
    provides: "MultinomialLogisticRegression + multinomial-head-v1.yaml, the PHASE3_CONTRACTS Makefile block"
provides:
  - "contracts/setfit-train-lifecycle-v1.yaml — the phase's frozen numbers: 5 gated epsilons, scale floors, the embedding floor, the calibrated regime, both RNG derivations, the execution digests and the 12-knob provenance"
  - "ParameterClass::AttentionKeyBias — the analytically gradient-free class, split on a mechanism and excluded from the verdict"
  - "train::setfit::thresholds — the single Rust threshold source, PARSED from the contract by a test"
  - "SetFitRun::tune_encoder() — the public transition that mints EncoderTuned only through the gate"
  - "tune::validate_evidence — THE one validating function; tune::PassedEvidence carrying the complete record"
  - "SetFitTrainError::{UncalibratedRegime, NoTrainableParameters, NoTestifyingParameters, EvidenceRejected}"
  - "train::setfit::baseline::FrozenProbeRun — the SAFE-03 baseline, contract-bound and never SetFit-labeled"
  - "A NEAR-NULL (1e-8) calibration condition — the real lower bound 03-05 recorded the matrix as lacking"
  - "BertSentenceEncoder/SetFitMiniLm::architecture_fingerprint() — what makes the regime a measurement rather than a label"
affects: [03-07, 03-08, 03-10, 05]

tech-stack:
  patterns:
    - "Split a parameter class on a MECHANISM (softmax shift-invariance), not on a failing margin, and pin the mechanism with a test that runs in every cargo test"
    - "Rounding-noise floor as the falsifiable form of 'this threshold sits at f32 resolution': (EPSILON/2)*init_norm/denom, computed from fields the row already records"
    - "A second, non-null control (1e-8) to supply a lower bound a null control (1e-30, exactly zero) cannot"
    - "Pessimistic verdict stamping: set Fail first so every early return leaves Fail, and only the success path promotes to Pass"
    - "Assemble a source-assertion needle from parts so the grep cannot match its own assertion"

key-files:
  created:
    - contracts/setfit-train-lifecycle-v1.yaml
    - crates/aprender-train/src/train/setfit/thresholds.rs
    - crates/aprender-train/src/train/setfit/baseline.rs
  modified:
    - crates/aprender-train/src/train/setfit/evidence.rs
    - crates/aprender-train/src/train/setfit/tune.rs
    - crates/aprender-train/src/train/setfit/mod.rs
    - crates/aprender-train/src/train/setfit/test_fixtures.rs
    - crates/aprender-core/src/setfit/mod.rs
    - crates/aprender-core/src/setfit/encoder.rs
    - contracts/aprender/binding.yaml
    - Makefile

key-decisions:
  - "03-06: the attention key bias is ANALYTICALLY gradient-free, not merely slow. Softmax is invariant to a constant shift of its inputs, so dL/db_k = 0 in exact arithmetic. Measured grad_norm_max 2.290e-10 against 8.007e-3 for the query weight and 3.444e-3 for the query bias in the same block. This finding is what caused DECISION 1 to be withdrawn and replaced with the class split"
  - "03-06: attention_key_bias carries NO epsilon AND NO movement predicate. An epsilon would sit at 0.49x its own rounding-noise floor, and ||dTheta|| > 0 passes on 1e-8 runs while its element support fraction on REAL runs is 0.9219-0.9922 rather than the 1.0000 every real-gradient class shows. It cannot testify in either direction"
  - "03-06: NoTestifyingParameters is a distinct typed error from NoTrainableParameters. Excluding a class from the gate opens a hole -- freeze everything checked, leave only what is not -- that an empty-set check on the TRAINABLE set does not close"
  - "03-06: the 1e-30 negative never exercises the epsilon. Its deltas are bit-for-bit zero so the strict predicate catches it first; deleting the threshold comparison left every negative GREEN. Only a table in which everything MOVED and everything fell short isolates the comparison"
  - "03-06: architecture_fingerprint() is production surface, not test support. A gate that fails closed outside its calibration regime must be able to OBSERVE the architecture it judges, or the regime is a caller-supplied label"
  - "03-06: pair_loss_endpoint is omitted from the armed set and the omission is written INTO the contract with its numbers, so its absence reads as an evidenced decision rather than an oversight"

patterns-established:
  - "When a flag's PREDICATE does not test the claim its PROSE makes, add the quantity the prose is about rather than trusting either"
  - "Induce the failure the plan suggests AND check it actually moves the gate -- 'I induced something the gate does not check' is indistinguishable from 'I saw red' afterwards"

requirements-completed: [TRN-03, SAFE-03]

duration: ~6h
completed: 2026-08-09
---

# Phase 3 Plan 06: Arm the SetFit-Identity Gate Summary

**`SetFitRun<EncoderTuned>` is now unmintable without passing evidence measured against five per-class epsilons frozen in a `pv`-valid contract before any comparison ran — and the one class that could not support a threshold was split out on a measured mechanism (its gradient is analytically zero) rather than by narrowing a margin, with its exclusion, the omitted endpoint statistic, and an unconfirmed reference assumption all written into the contract as evidenced decisions.**

## Performance

- **Duration:** ~6h · **Tasks:** 3 of 3 · **Files created:** 3 · **Files modified:** 8

## Task Commits

| Task | Name | Commit | Type |
|---|---|---|---|
| — | Rounding-noise floor instrumentation (escalation evidence) | `581ba157a` | test |
| — | AttentionKeyBias split + 1e-8 near-null control | `27cd6e3ec` | feat |
| 1 | Freeze the thresholds in setfit-train-lifecycle-v1 | `371dc5a45` | docs |
| 2 | Arm `tune_encoder()` | `79e988ccd` | feat |
| 3 | `FrozenProbeRun` baseline | `fe13735e8` | feat |

---

## The escalation, and what it found

03-05 flagged `projection_bias` as unfreezable because `min/10` "sits close to f32 resolution". **Its predicate tested `median/min > 100`** — a spread statistic that says nothing about resolution. I added the quantity the claim is actually about: the largest relative delta pure representation rounding can produce, `(f32::EPSILON/2) · init_norm / max(denom, s_class)`, which is rigorous (half-ULP rounding gives `|dx_i| ≤ (EPS/2)|x_i|`) and computed from fields the evidence row already records.

| class | eps at 10x margin | rounding-noise floor | eps/noise |
|---|---|---|---|
| embedding | 1.198e-5 | 9.398e-8 | 127x |
| layer_norm_weight | 2.930e-6 | 5.960e-8 | 49x |
| layer_norm_bias | 1.161e-5 | 5.290e-8 | 220x |
| projection_weight | 1.875e-5 | 5.960e-8 | 315x |
| projection_bias (pre-split) | 3.361e-10 | 5.960e-8 | **0.0056x** |

Then the mechanism, which is what changed the decision: `encoder.layer.1.attention.self.key.bias` has `grad_norm_max` **2.290e-10** against **8.007e-3** for the query weight in the same block. Softmax is invariant to a constant shift of its inputs, so the key bias contributes the same amount to every key's pre-softmax logit for a given query and **∂L/∂b_k = 0 in exact arithmetic**. The 2.29e-10 is f32 cancellation residue.

Widening was measured, not extrapolated (seeds 7/42, rungs 3/12/48/192/768 steps): the key bias reaches 1.001e-6 at 768 steps, so eps would clear the floor by only **1.68x**, at ~40 min per calibration plus **~10 min added to every default `cargo test`** (the control, mirror and 1e-30 negatives must run in-regime). A 3x margin needed ~1300 steps ≈ 68 min, over the stated cap. That, plus the mechanism, was escalated.

**The human withdrew DECISION 1 and chose option D (split).** Both deviations are recorded below.

---

## Deviations from Plan

### DEVIATION 1 — DECISION 1 superseded: split, do not widen (human-directed)

The orchestrator's original DECISION 1 mandated widening the matrix. My measurements showed widening amplifies rounding residue of an analytically-zero gradient. **The human withdrew that instruction and directed option D.** The matrix was NOT widened; it remains 6 cells at 3–4 steps and 27s (now 42s with the third condition).

`ParameterClass::AttentionKeyBias` is a sixth class matching `*.attention.self.key.bias` and nothing else. The boundary is tested from both sides: the key WEIGHT stays `ProjectionWeight` (`W_k x` is not constant in x) and the query/value biases stay `ProjectionBias` (`(q+b_q)·k_j` varies with j; the value bias adds a constant the downstream layers see).

**Freeing `projection_bias` of the key bias made it freezable** — it is now bound by `encoder.layer.0.attention.self.query.bias` (`grad_norm_max` 3.444e-3, a real gradient) at `best_real` 8.394e-5, `eps/noise` **141x**. No STOP condition was hit.

**What the gradient-free class asserts: NOTHING.** Determined by measurement, as directed:

- **No epsilon.** Even in isolation its 10x-margin epsilon is 3.361e-10 against its own noise floor of 6.880e-10 — ratio **0.49**.
- **No movement predicate.** It PASSES on runs that are not training (under the 1e-8 control key biases moved in 5 of 6 cells, worst 9.095e-13), and it is unreliable on runs that ARE (element support fraction 0.9219–0.9922 across real cells, **never the 1.0000 that every real-gradient class shows** — whether a given key bias moves is decided by whether its cancellation residue rounded to zero). It cannot testify in either direction, so arming it would risk rejecting legitimate runs for a reason unrelated to training. The same reasoning that killed `pair_loss_endpoint`, applied for consistency rather than selectively.

The contract states plainly that **these parameters cannot serve as encoder-update evidence**, and `NoTestifyingParameters` closes the hole the exclusion would otherwise open.

### DEVIATION 2 — DECISION 2: `pair_loss_endpoint` not armed (human-directed)

Omitted from the armed set as briefed. The `pair_loss_endpoint` equation records the measured cross-seed spread **0.4933**, the sign flip, and the k=1 / lr-0.0-at-step-0 reasoning, plus an explicit "this is not permission to re-arm a looser version" clause. **Plan acceptance criteria adjusted:** the plan's must-have naming "pair-loss endpoints" as a gate condition, and its Task 1 criterion requiring "a numeric k and a numeric margin", are NOT met and deliberately so. `k` is recorded in the evidence; no margin exists. Armed gates are three, not four: per-class relative delta, gradient finiteness, embedding delta.

### DEVIATION 3 — line 208 amended: three → six eps values (human-confirmed)

The plan's criterion required "exactly THREE numeric eps values". 03-05 had already superseded that premise (its key-decision: *"five parameter classes rather than three"*). The contract freezes **six class entries — five with a numeric epsilon, one with `null`**. The human confirmed the amendment.

### DEVIATION 4 — [Rule 2] `architecture_fingerprint()` added to the encoder

`SetFitMiniLm` exposed no way to observe its architecture (`encoder()` is behind `conformance-fixtures`, absent from production builds). A gate that fails closed outside its calibration regime must be able to observe what it is judging, or the regime is a caller-supplied label rather than a measurement. Added a read-only accessor on `BertSentenceEncoder` and forwarded it — shape numbers only, already implied by every published parameter. Not behind a test feature: this is production behaviour.

### DEVIATION 5 — [Rule 2] `NoTestifyingParameters` added

Not in the plan. Excluding a class from the gate creates a hole an empty-TRAINABLE-set check does not close: freeze everything gated, leave only key biases, and a loop over the gated set passes vacuously by iterating over nothing. Typed error + negative test.

### DEVIATION 6 — [Rule 2] a third calibration condition (1e-8 near-null)

03-05 recorded that the 1e-30 control is exactly 0.000e0 everywhere, so the matrix bounded epsilon **from above only**. Freezing a two-sided margin against a one-sided measurement would have made the lower edge decorative. The 1e-8 condition supplies a real lower bound; every frozen epsilon sits at least 2.8x above 10x that control.

### DEVIATION 7 — [Rule 1] the plan's own audit induction does not work

Task 3's criterion asks to observe `make contract-audit-phase3` "go red with a deliberately bogus binding status". **Measured: it does not.** `status: pending` yields `[WARN] BIND-004 ... is pending implementation` at **rc 0**, and a bogus `module_path` yields nothing at all. The audit's blocking condition is entry PRESENCE (BIND-001), not correctness. The real induced-red arrived naturally and is stronger — see below.

---

## Guards falsified before being trusted

| Guard | Induced mutation | Observed |
|---|---|---|
| `evidence_attention_key_bias_is_gradient_free_relative_to_its_own_block` | classify the QUERY bias as gradient-free instead | **RED** — "the key bias gradient (3.446e-3) is not >=1e4x below the smallest ordinary projection-bias gradient (1.824e-10)", measuring the same 1.9e7x separation from the other side of the classifier. Reverted. |
| `thresholds_match_the_contract` | `projection_bias` 8.3e-6 → 8.3e-7, **Rust side only** | **RED** naming both values and the T-3-21 reason. Reverted. |
| `contract-audit-phase3` reaching linear-probe | add the contract to PHASE3_CONTRACTS | **RED** — `[ERROR] BIND-001: Equation 'linear_probe' has no binding entry`, green once bound. This is the evidence the scoped audit reaches that file. |
| `baseline_never_constructs_a_setfit_run` | none needed | **RED on its own search string** — fourth self-reference trap this phase. Needle now assembled from parts. |

### The falsification that changed the test set

Deleting the `relative_delta > eps` comparison outright **left every negative GREEN**:

- the 1e-30 run has bit-for-bit zero deltas, so the strict movement predicate catches it before the threshold is consulted;
- even a real 1e-8 run contains parameters that did not move (`layer_norm_weight` has non-movers, measured), so it too is rejected without the epsilon deciding anything.

**Every frozen threshold in the contract could have been any number and the suite would not have noticed.** A second attempt asserting properties of the 1e-8 *table* also failed to kill the mutation, because it tested the table rather than the gate's decision. The negative that works builds a table where **every gated parameter moved with finite gradients and every relative delta was scaled below its class epsilon**, and asserts rejection. Under the mutation it goes red (the gate falls through to the run-level floor, eps 2.7e-5 instead of the class's 1.1e-5).

---

## The frozen numbers

| class | eps | scale floor | gated | worst 1e-8 | best real | eps/noise |
|---|---|---|---|---|---|---|
| embedding | 1.1e-5 | 1.0 | yes | 4.206e-7 | 1.198e-4 | 127x |
| layer_norm_weight | 2.9e-6 | 1.0 | yes | 0.000e0 | 2.930e-5 | 49x |
| layer_norm_bias | 1.1e-5 | 1.0 | yes | 1.548e-7 | 1.161e-4 | 220x |
| projection_weight | 1.8e-5 | 1.0 | yes | 4.092e-7 | 1.875e-4 | 315x |
| projection_bias | 8.3e-6 | 1.0 | yes | 1.460e-7 | 8.394e-5 | 141x |
| attention_key_bias | **none** | 1.0 | **no** | 9.095e-13 | 3.361e-9 | 0.49x |

Embedding delta floor **2.7e-5** (smallest class median across cells 2.748e-4, upper edge rounded down). Each epsilon is the upper edge rounded **DOWN** to two significant figures — rounding up would erode the 10x false-rejection margin.

`calibration_regime_id`: `minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1,42,7|cells=s16e2b8,s8e1b4` — **exactly one** fingerprint. Phase 5's production-encoder cells are blocked until someone calibrates there and makes a deliberate contract edit (D-10(c)).

**A1 is UNCONFIRMED.** `import setfit` fails offline and the pinned fixtures record `sampler.py` only. The defaults are **"reference-cited"**; the stronger word is absent from `knob_defaults` and a grep asserts it. The contract carries the confirmation recipe.

**Reference cross-check: skipped, deliberately.** The coordinator offered huggingface/setfit as optional. It could not settle either open question: reference SetFit does not classify parameters at all (it calls the HF trainer), and the gradient-free key bias is a property of dot-product attention rather than of SetFit. The one thing it could settle — A1 — needs the pinned environment executed, not the source read, which is exactly what is unavailable offline.

---

## Measured verification

All via `rtk proxy`, `CARGO_INCREMENTAL=0`.

| Command | rc | Result |
|---|---|---|
| `pv validate contracts/setfit-train-lifecycle-v1.yaml` | **0** | Contract is valid |
| `cargo test … --features setfit evidence_gate` | **0** | 1 passed |
| `cargo test … --features setfit negative_` | **0** | 41 passed |
| `cargo test … --features setfit thresholds_` | **0** | 6 passed |
| `cargo test … --features setfit baseline_` | **0** | 9 passed |
| `cargo test … --features setfit evidence_` / `tune_` | **0** | 17 / 46 passed |
| `cargo test … calibration_matrix -- --ignored` | **0** | 1 passed, 41.92s, 18 passes |
| `make contract-audit-phase3` | **0** | every equation bound |
| `cargo fmt -p aprender-train -- --check` | **0** | |
| `cargo test -p aprender-train --lib --features setfit` | 101 | **7718 passed, 24 failed** |

**Zero regressions.** The 24 failures are byte-identical to `known-red-baseline.md` (21 `gpu::` + 3 `prune::snapshot_tests`); set difference computed both ways is empty.

**Clippy (D-ITEM-02), two-sided:** `--features setfit` → 20 errors; without → 22. All in `aprender-compute`. Diagnostics citing `crates/aprender-train/src/train/setfit/`: **0** in both legs.

**`pv lint <single-file>` is VACUOUS, confirmed:** rc 0, `Result: PASS`, and `0 contracts` on every gate. `pv validate` is the real single-file gate — it found the PROVABILITY-001 error that `lint` missed entirely. The contract's falsification entries cite `validate`.

---

## Known Stubs

None. `calibration_regime_id` derives the non-calibrated branch's id descriptively (`{fingerprint}|seeds=?|cells=?`) — that string is never compared against anything except the calibrated set it deliberately fails to match, so the `?` placeholders are not a stub but the point.

## Threat Flags

None. `architecture_fingerprint()` is the only new public surface; it exposes shape numbers already implied by every published parameter, no tensor, weight or tokenizer material.

Mitigations applied: T-3-20 (gate inside the transition, three typed unpassable states, distinct `FrozenProbeRun`), T-3-21 (epsilons in the pv-validated contract, Rust constants PARSED from it, one-sided edit observed RED), T-3-22 (negative + control + mirror, and the mutation that proved the original set could not falsify the epsilon), T-3-42 (`calibration_regime` + `UncalibratedRegime` first in `validate_evidence`), T-3-56 (`EvidenceRejected` carries the complete table; a test walks it), T-3-57 (linear-probe in PHASE3_CONTRACTS with a real induced red).

## Issues Encountered

- **`make contract-audit-phase3` checks entry presence, not correctness.** Neither a bogus status nor a bogus module path turns it red. Worth a future ticket; recorded here so the next executor does not mistake a warning for a gate.
- **`binding.yaml` entries must be inserted before `critical_path:`** — the file documents this at line 870 and my first append landed after it, parsing into the wrong list. The documented trap is real.

## Next Phase Readiness

- **03-07** gets `PassedEvidence` owning the complete `UpdateEvidence` + `EvidenceSummary`; `HeadFittedEvidence` should be a NAMED STRUCT extending it, not a 2-tuple.
- **03-08** gets every run-level field it needs on the carrier chain: both digests, `batch_boundary_list`, `parameter_registry_hash`, `step_count`, endpoint means and `k`, embedding aggregates, `pre_clip_norm_max`, `calibration_regime_id`.
- **Phase 5** is **BLOCKED by design** on the production encoder: its fingerprint is not calibrated and `tune_encoder` returns `UncalibratedRegime`. Unblocking requires a calibration run there plus a deliberate contract edit — not a code change and not something a Phase 5 executor may do inline.
- `STATE.md`, `ROADMAP.md` and `REQUIREMENTS.md` were **not** touched, per the worktree protocol.

---
*Phase: 03-faithful-two-stage-trainer-and-head · Plan: 06 · Completed: 2026-08-09*

## Self-Check: PASSED

Files verified present on disk:

- `contracts/setfit-train-lifecycle-v1.yaml` — FOUND (47.1K)
- `crates/aprender-train/src/train/setfit/thresholds.rs` — FOUND (12.4K)
- `crates/aprender-train/src/train/setfit/baseline.rs` — FOUND (11.1K)
- `.planning/phases/03-faithful-two-stage-trainer-and-head/03-06-SUMMARY.md` — FOUND (20.5K)

Commits verified in `git log`:

- `581ba157a` — FOUND (rounding-noise instrumentation)
- `27cd6e3ec` — FOUND (class split + near-null control)
- `371dc5a45` — FOUND (Task 1)
- `79e988ccd` — FOUND (Task 2)
- `fe13735e8` — FOUND (Task 3)

`STATE.md`, `ROADMAP.md` and `REQUIREMENTS.md` are NOT in the diff against base
`0cfc33de8` — the orchestrator owns those writes. Zero file deletions across the range.
