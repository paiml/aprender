---
phase: 03-faithful-two-stage-trainer-and-head
plan: 05
subsystem: training
tags: [setfit, contrastive, adamw, autograd, determinism, sha256, evidence, calibration, serde]

requires:
  - phase: 03-faithful-two-stage-trainer-and-head
    plan: 02
    provides: "setfit::dropout_rng keyed on (root_seed, site, forward ordinal, element) and SetFitMiniLm::set_forward_ordinal — D-15's branch coordinate"
  - phase: 03-faithful-two-stage-trainer-and-head
    plan: 03
    provides: "SetFitRun<Prepared> + prepare(), the 12-knob ResolvedSetFitConfig, WarmupLinearDecayLR + warmup_steps_from_ratio, reduce.rs fixed-order reductions, epoch_pair_order"
provides:
  - "aprender::autograd::graph_tape_len() — the tape-growth / no-graph observation accessor the crate had no way to expose"
  - "train::setfit::tune::run_tuning — the full stage-one loop, pub(crate) and record-only"
  - "TuneOutput: initial-tensor snapshot, per-parameter grad/delta/support measurements, loss trace + hash, in-band consumed-pair and batch-boundary digests, the readable boundary list, step observations, before/after eval embeddings"
  - "train::setfit::evidence — classify_parameter (5 classes, fail-closed), the support-restricted floored relative delta, UpdateEvidence + EvidenceSummary with a binding table hash"
  - "train::setfit::test_fixtures — the cfg(test) deterministic real-slice / synthetic-text fixture and the calibration matrix's source"
  - "A MEASURED per-class relative-delta distribution across 3 seeds x 2 boundary configurations x (real, 1e-30 control) = 12 runs — 03-06's epsilon basis"
affects: [03-06, 03-07, 03-08, 03-10]

tech-stack:
  added:
    - "aprender-core dev-dependency with `conformance-fixtures` — enables the slice constructor for TEST builds only, no production surface widened"
  patterns:
    - "In-band digest absorption: hash what the loop CONSUMES, at the moment it consumes it, so a run that consumed the wrong order cannot reproduce a correct-looking digest"
    - "Two-crate autograd bridge: an explicit, index-aligned mirror that lets a loop REUSE a reference optimizer across incompatible Tensor types instead of growing a second one"
    - "Floored, support-restricted ratio: a positive contracted floor removes the zero-init NaN/inf pair, and restricting a sparse denominator to the delta's support makes the ratio dimension-invariant and therefore transferable"
    - "Probes with private fields and #[cfg(test)]-only constructors, so a negative test can disable a correctness step that production cannot"
    - "Report-to-file from an #[ignore]d measurement test, because the repo's cargo wrapper drops --nocapture output"

key-files:
  created:
    - crates/aprender-train/src/train/setfit/test_fixtures.rs
    - crates/aprender-train/src/train/setfit/tune.rs
    - crates/aprender-train/src/train/setfit/evidence.rs
  modified:
    - crates/aprender-core/src/autograd/mod.rs
    - crates/aprender-train/src/train/setfit/mod.rs
    - crates/aprender-train/Cargo.toml
    - .planning/phases/03-faithful-two-stage-trainer-and-head/deferred-items.md

key-decisions:
  - "03-05: the plan's context claim `aprender-train's Tensor is the re-exported aprender-core autograd Tensor (lib.rs:139)` is FALSE. Measured: lib.rs:139 re-exports aprender-train's OWN autograd::tensor::Tensor (an Array1<f32> with an Rc<RefCell> grad), which is unrelated to aprender::autograd::Tensor (a Vector<f32> + shape on a thread-local tape). AdamW::step_refs and clip_grad_norm_refs are therefore not directly applicable to encoder parameters; ParamBridge is the explicit adapter that keeps the plan's REUSE instruction true"
  - "03-05: gradients do not live on the parameter tensors. ComputationGraph::backward writes into the graph's own registry copies, so param.grad() is None after a successful backward and the gradient is reached with autograd::get_grad(param.id()). A loop reading param.grad() would clip nothing and step nothing while reporting a falling loss"
  - "03-05: the plan's predicted gradient-accumulation failure does NOT occur in this engine — register_tensor replaces a leaf's entry every forward and a previous step's sub-tape has no seeded output gradient, so backward skips it. What step (b) actually guarantees is `the tape is empty when THIS step's forward begins`, which step (l) cannot provide at step 0. Two probes now separate the two properties instead of one probe measuring neither"
  - "03-05: preflight takes (device, max_length, freeze_policy) rather than &ResolvedSetFitConfig, because a ResolvedSetFitConfig is only mintable by resolve() which probes the host — on a CPU-only machine the CUDA rejection would otherwise have had no reachable test, and the alternative was a test-only constructor on the validated config type"
  - "03-05: the calibration matrix stays INSIDE the crate as an #[ignore]d lib test. run_tuning is pub(crate), calibration_variants is #[cfg(test)], and a ResolvedSetFitConfig only comes from prepare() — no out-of-crate integration target could have compiled against any of the three, and every widening that would fix that is forbidden elsewhere in this phase"
  - "03-05: five parameter classes rather than three. Weight and bias movement differ by up to 5 orders of magnitude inside the same block (measured), so a class spanning both would have its epsilon set by whichever member moves least"
  - "03-05: from_slice_fixture is gated on aprender-core's `conformance-fixtures`, NOT on `setfit`. A dev-dependency enables it for test builds only; adding it to the `setfit` feature would put a test-only constructor and two conformance accessors on every production build"

patterns-established:
  - "Falsify the digest gate by making the digest recomputed: the reversed-consumption test is only meaningful if the same test goes red when absorption drifts out of the pull loop, and it was induced and observed"
  - "Assert a DELTA when the absolute is owned by someone else: the baseline-encode tape check measures growth across the encode and asserts the entry value is non-zero, so it cannot become a vacuous test of a loader property it does not own"
  - "Name what binds each bound: a calibration report that gives a class minimum without naming the parameter that set it tells 03-06 the number but not what to widen"

requirements-completed: []

duration: ~5h40m
completed: 2026-08-09
---

# Phase 3 Plan 05: Stage One — Tuning Loop, Recorded Execution and the Epsilon Basis Summary

**The SetFit contrastive stage now runs end-to-end on a real-weight MiniLM slice with a pinned optimizer step order, hashes the pairs and batch boundaries it ACTUALLY consumed as it consumes them, and reports a canonical hash-bound evidence table whose per-class relative-delta distribution was measured across 12 complete runs — giving plan 03-06 a real basis to freeze epsilon from, plus one class flagged as unfreezable at the usual margin.**

## Performance

- **Duration:** ~5h40m
- **Tasks:** 3 of 3
- **Files created:** 3 · **Files modified:** 4

## Task Commits

| Task | Name | Commit | Type |
|---|---|---|---|
| 1 | Deterministic trainer test fixture | `d32eb431c` | test |
| 2 | `run_tuning` — step order + in-band digests | `6f948de52` | feat |
| 3 | Evidence table + relative delta + calibration matrix | `623d529da` | feat |
| — | Exact snapshot accounting + D-ITEM-06 | `b312a4498` | test |

---

## The fixture: what is REAL and what is SYNTHETIC

The plan asked for this distinction to be stated precisely, because the calibration's transfer
argument turns on it and the review challenged it directly.

**Resolved path:** `APRENDER_SETFIT_FIXTURES` when it names a directory, otherwise
`crates/aprender-train/../aprender-core/tests/fixtures/setfit` — i.e.
`crates/aprender-core/tests/fixtures/setfit`, the same directory `aprender-core`'s own slice
tests read.

**Slice dimensions**, read from `slice_config.json` and asserted by
`fixture_slice_encoder_has_the_recorded_dimensions`:

| Field | Slice | Upstream |
|---|---|---|
| `hidden` | 64 | 384 |
| `num_layers` | 2 | 6 |
| `heads` | 2 | 12 |
| `head_dim` | 32 | 32 |
| `intermediate` | 256 | 1536 |
| `vocab` | 97 | 30522 |
| `positions` | 64 | 512 |
| `type_vocab_size` | 2 | 2 |
| `hidden_act` | gelu | gelu |
| `layer_norm_eps` | 1e-12 | 1e-12 |
| `source_revision` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | — |

**The encoder's WEIGHT VALUES are the real pretrained `all-MiniLM-L6-v2` values** for the rows,
columns and layers retained. Only the DIMENSIONS are reduced. It is not a randomly initialized
toy, and that is the whole reason a relative-delta distribution measured here is argued to
transfer: a random init would have a different gradient scale and a different initial-norm
scale, and the epsilon 03-06 freezes would be calibrated against neither.

**The TEXT is entirely synthetic.** No corpus content of any kind is committed.
`grep -c -iE 'abortion|tweet|stance' test_fixtures.rs` → **0**.

**A vocabulary constraint the plan did not anticipate.** The slice carries 97 token rows and
`BertSentenceEncoder` maps canonical ids through `vocab_remap.json`, returning
`SetFitError::VocabOutOfSlice` for anything outside it. The corpus therefore cannot spell
arbitrary words — **not even digits**, which are absent from the slice, so the plan's suggested
`"class-a sample 0"` would have failed. Every sentence is assembled from the retained token
strings (`the`, `quick`, `brown`, `cat`, `sat`, `over`, `mat`, `.` and so on), and
`fixture_every_row_encodes_within_the_slice_vocabulary` encodes all 48 train rows plus both
held-out splits so the constraint is enforced rather than trusted.

**Corpus:** 3 classes x 16 distinct train rows (48), 1 validation and 1 test row per class,
built through the real `PreparedDataset::from_labeled_rows` ingest ladder. Each class has its
own subject and verb, so same-class pairs are genuinely more similar than cross-class pairs and
the contrastive objective has real signal.

**Calibration matrix source:** `calibration_variants()` yields **6** cells — seeds `{1, 7, 42}`
x boundary configurations `{(shots 8, epochs 1, batch 4, budget 12), (shots 16, epochs 2,
batch 8, budget 16)}`. The two configurations differ in shots AND epochs AND batch size
(asserted), so the second genuinely straddles an epoch boundary and doubles the selected rows.

Explicit pair budgets are used because Phase 2's DEFAULT closed form is **384** pairs at 8 shots
and **1536** at 16 — a training run, not a test.

---

## Epsilon basis — the measured calibration matrix

Produced by `calibration_matrix_epsilon_basis`, 12 complete `run_tuning` passes (6 real +
6 controls at `encoder_lr = 1e-30`, each control differing from its cell in the learning rate
and in nothing else — asserted by `fixture_control_config_differs_only_in_the_learning_rate`).

**`calibration_regime_id`** (proposed, and what the evidence records):

```
minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1,42,7|cells=s16e2b8,s8e1b4
```

### Per class, per cell — `relative_delta`

| seed | cell | class | real min | real median | real max | control max | support frac | all moved |
|---|---|---|---|---|---|---|---|---|
| 1 | s8e1b4 | embedding | 2.093e-4 | 3.061e-4 | 5.917e-4 | 0.000e0 | 0.3196 | true |
| 1 | s8e1b4 | layer_norm_weight | 3.707e-5 | 3.919e-5 | 4.062e-5 | 0.000e0 | 1.0000 | true |
| 1 | s8e1b4 | layer_norm_bias | 2.056e-4 | 2.063e-4 | 2.129e-4 | 0.000e0 | 1.0000 | true |
| 1 | s8e1b4 | projection_weight | 2.006e-4 | 3.549e-4 | 5.571e-4 | 0.000e0 | 1.0000 | true |
| 1 | s8e1b4 | projection_bias | **5.037e-9** | 2.006e-4 | 2.178e-4 | 0.000e0 | 1.0000 | true |
| 1 | s16e2b8 | embedding | 2.862e-4 | 4.261e-4 | 8.281e-4 | 0.000e0 | 0.3299 | true |
| 1 | s16e2b8 | layer_norm_weight | 4.956e-5 | 5.389e-5 | 5.561e-5 | 0.000e0 | 1.0000 | true |
| 1 | s16e2b8 | layer_norm_bias | 2.831e-4 | 2.901e-4 | 2.974e-4 | 0.000e0 | 1.0000 | true |
| 1 | s16e2b8 | projection_weight | 2.877e-4 | 4.861e-4 | 7.726e-4 | 0.000e0 | 1.0000 | true |
| 1 | s16e2b8 | projection_bias | **5.195e-9** | 2.804e-4 | 2.954e-4 | 0.000e0 | 1.0000 | true |
| 7 | s8e1b4 | embedding | 1.198e-4 | 2.748e-4 | 4.162e-4 | 0.000e0 | 0.3402 | true |
| 7 | s8e1b4 | layer_norm_weight | 2.930e-5 | 3.202e-5 | 3.548e-5 | 0.000e0 | 1.0000 | true |
| 7 | s8e1b4 | layer_norm_bias | 1.161e-4 | 1.178e-4 | 1.231e-4 | 0.000e0 | 1.0000 | true |
| 7 | s8e1b4 | projection_weight | 1.875e-4 | 2.969e-4 | 4.628e-4 | 0.000e0 | 1.0000 | true |
| 7 | s8e1b4 | projection_bias | **3.361e-9** | 1.162e-4 | 1.294e-4 | 0.000e0 | 1.0000 | true |
| 7 | s16e2b8 | embedding | 2.804e-4 | 4.129e-4 | 8.034e-4 | 0.000e0 | 0.3505 | true |
| 7 | s16e2b8 | layer_norm_weight | 4.900e-5 | 5.211e-5 | 5.517e-5 | 0.000e0 | 1.0000 | true |
| 7 | s16e2b8 | layer_norm_bias | 2.761e-4 | 2.840e-4 | 2.892e-4 | 0.000e0 | 1.0000 | true |
| 7 | s16e2b8 | projection_weight | 2.626e-4 | 4.628e-4 | 7.517e-4 | 0.000e0 | 1.0000 | true |
| 7 | s16e2b8 | projection_bias | **5.936e-9** | 2.690e-4 | 2.820e-4 | 0.000e0 | 1.0000 | true |
| 42 | s8e1b4 | embedding | 2.041e-4 | 2.992e-4 | 5.894e-4 | 0.000e0 | 0.3402 | true |
| 42 | s8e1b4 | layer_norm_weight | 3.711e-5 | 3.885e-5 | 4.075e-5 | 0.000e0 | 1.0000 | true |
| 42 | s8e1b4 | layer_norm_bias | 2.023e-4 | 2.085e-4 | 2.134e-4 | 0.000e0 | 1.0000 | true |
| 42 | s8e1b4 | projection_weight | 1.982e-4 | 3.490e-4 | 5.584e-4 | 0.000e0 | 1.0000 | true |
| 42 | s8e1b4 | projection_bias | **5.885e-9** | 2.007e-4 | 2.065e-4 | 0.000e0 | 1.0000 | true |
| 42 | s16e2b8 | embedding | 2.820e-4 | 4.397e-4 | 8.357e-4 | 0.000e0 | 0.3299 | true |
| 42 | s16e2b8 | layer_norm_weight | 5.281e-5 | 5.431e-5 | 5.791e-5 | 0.000e0 | 1.0000 | true |
| 42 | s16e2b8 | layer_norm_bias | 2.812e-4 | 3.016e-4 | 3.046e-4 | 0.000e0 | 1.0000 | true |
| 42 | s16e2b8 | projection_weight | 2.852e-4 | 4.988e-4 | 8.059e-4 | 0.000e0 | 1.0000 | true |
| 42 | s16e2b8 | projection_bias | **4.410e-9** | 2.851e-4 | 3.001e-4 | 0.000e0 | 1.0000 | true |

**Separation holds in all 30 (cell x class) combinations** — the assertion is inside the test,
per class and per cell, and it was falsified (below).

### Cross-cell aggregate — what 03-06 freezes from

| class | worst control | best real | 10x lower bound | 10x upper bound | supports margin | median / min |
|---|---|---|---|---|---|---|
| embedding | 0.000e0 | 1.198e-4 | 0.000e0 | 1.198e-5 | yes | 3.7 |
| layer_norm_weight | 0.000e0 | 2.930e-5 | 0.000e0 | 2.930e-6 | yes | 1.9 |
| layer_norm_bias | 0.000e0 | 1.161e-4 | 0.000e0 | 1.161e-5 | yes | 2.6 |
| projection_weight | 0.000e0 | 1.875e-4 | 0.000e0 | 1.875e-5 | yes | 2.7 |
| projection_bias | 0.000e0 | **3.361e-9** | 0.000e0 | **3.361e-10** | yes | **8.5e4** |

**What binds each class's lower edge** (all at seed 7, cell s8e1b4 — the slowest cell):

| class | binding parameter |
|---|---|
| embedding | `embeddings.token_type_embeddings.weight` |
| layer_norm_weight | `encoder.layer.1.output.LayerNorm.weight` |
| layer_norm_bias | `embeddings.LayerNorm.bias` |
| projection_weight | `encoder.layer.0.attention.self.query.weight` |
| projection_bias | `encoder.layer.1.attention.self.key.bias` |

### The control is EXACTLY zero, and that is a finding rather than a formality

Every 1e-30 cell reports `relative_delta = 0.000e0` for every class. The mechanism is
measurable and expected: AdamW's normalized update has magnitude ~`lr` per element per step, so
a 1e-30 learning rate produces per-element updates ~1e-30 against parameters of order 1e-2 to
1e0, whose `f32` ULP is ~1e-9 — the addition is a no-op at every element, and the delta is bit-
for-bit zero. The control also fails the strict `||dTheta|| > 0` predicate for every parameter,
which is the second discriminator recorded alongside the ratio.

**Consequence for 03-06:** the lower bound is not "10x above the noise floor", it is "anything
above zero". The matrix bounds epsilon from ABOVE (best real / 10) and establishes that a null
learning rate produces exactly no movement at `f32` resolution. If 03-06 wants a tighter lower
bound it needs a control at a small but representable learning rate (1e-8 would be a natural
choice); that would be a matrix extension, not a narrowing, and is flagged here rather than
silently assumed away.

### FLAG — `projection_bias` cannot be frozen at the usual margin

The report emits this automatically:

```
WIDE-SPREAD projection_bias: median 2.851e-4 is 8.5e4x its own class minimum 3.361e-9;
a single per-class epsilon at min/10 = 3.361e-10 sits close to f32 resolution
```

The class minimum is set by `encoder.layer.1.attention.self.key.bias` in every cell, four to
five orders of magnitude below the class median. This is explicable rather than anomalous — a
uniform shift of every key contributes almost nothing to a softmax over query-key dot products,
so the key bias receives an unusually small gradient — but it means **one epsilon over the whole
`projection_bias` class is set by the key bias and is therefore near float noise.** Three
options for 03-06, none of which is "narrow the margin quietly":

1. Split `projection_bias` into `attention_key_bias` and the rest, and freeze two epsilons.
2. Exclude the key biases from the per-class gate and rely on the strict `||dTheta|| > 0`
   predicate for them (they DO move — `all_moved` is true in every cell).
3. Widen the matrix (more steps per cell) until the key bias's movement clears noise, and
   re-measure.

### Endpoint means and their cross-seed spread

| seed | cell | k | first_k | last_k | delta |
|---|---|---|---|---|---|
| 1 | s8e1b4 | 1 | 0.128356501 | 0.244817525 | **+0.116461024** |
| 1 | s16e2b8 | 2 | 0.277520135 | 0.271580085 | −0.005940050 |
| 7 | s8e1b4 | 1 | 0.018987009 | 0.203567699 | **+0.184580689** |
| 7 | s16e2b8 | 2 | 0.263382450 | 0.253794916 | −0.009587534 |
| 42 | s8e1b4 | 1 | 0.445803940 | 0.137069106 | −0.308734834 |
| 42 | s16e2b8 | 2 | 0.258881509 | 0.286387399 | **+0.027505890** |

**Cross-seed spread of `(last_k − first_k)`: min −0.3087, max +0.1846, range 0.4933.**

This is exactly the review's concern made concrete. **The endpoint statistic is NOT usable as a
convergence gate at this scale.** Four of six cells show the loss going UP between endpoints,
and the sign flips with the seed. At 3 steps the window is `k = 1`, i.e. a single loss value at
each end, and step 0 runs at learning rate 0.0 by reference construction — so `first_k` is the
untrained loss on one arbitrary batch. Any gate on `last_k < first_k` calibrated on one trace
would reject correct runs on other seeds. 03-05 computes the statistic and records it; it makes
no judgment, and 03-06 should not arm one on this evidence.

---

## Measured verification

All commands run with `CARGO_INCREMENTAL=0`. Every load-bearing measurement was taken through
`rtk proxy`, because the repo's `cargo` hook summarizes output and silently reshapes both
`--nocapture` lines and clippy diagnostic streams (03-03's finding, re-confirmed here).

| Command | rc | Result |
|---|---|---|
| `cargo test -p aprender-train --lib --features setfit fixture_` | **0** | 10 passed, **0.75 s** |
| `cargo test -p aprender-train --lib --features setfit tune_` | **0** | 46 passed, **14.53 s** |
| `cargo test -p aprender-train --lib --features setfit evidence_` | **0** | 15 passed, 0 ignored, **3.11 s** |
| `cargo test … calibration_matrix -- --ignored` | **0** | 1 passed, **28.14 s** |
| `cargo test -p aprender-core --lib autograd` | **0** | 246 passed |
| `cargo test -p aprender-core --lib --features conformance-fixtures setfit::` | **0** | 179 passed |
| `cargo test -p aprender-train --lib --features setfit -- --test-threads=1` | 101 | 7702 passed, **24 failed**, 15 ignored — see below |
| `cargo fmt -p aprender-train -p aprender-core -- --check` | **0** | |

**W-11 satisfied:** the `evidence_` filter is **3.11 s**, far under the 30 s ceiling, and the
calibration matrix at **28.14 s** is off it entirely. `evidence_` reports `0 ignored` — the
matrix is filtered out by name, not skipped after being counted.

### Zero regressions against the known-red baseline

The 24 `aprender-train --lib` failures are **byte-identical** to
`known-red-baseline.md`'s list — 21 `gpu::` plus 3 `prune::snapshot_tests`:

```
diff -u <baseline names> <observed names>   ->   rc=0, no output
```

No new failure. 7702 passed.

### Source assertions

| Assertion | Result |
|---|---|
| `awk '/fn run_batch/,/^}/' tune.rs \| awk '/for .*pair_at\|while .*pair_at/,0' \| grep -c 'absorb_batch_digests'` | **1** (criterion: ≥ 1) |
| `awk '/fn run_tuning/,/^}/' tune.rs \| grep -c 'absorb_batch_digests'` | **0** |
| `tune.rs` contains `clip_grad_norm_refs` / `step_refs` / `zero_grad_` / `clear_graph` / `warmup_steps_from_ratio` | all present, as CALLS |
| `grep -rc 'par_iter' train/setfit/` on NON-COMMENT lines | **0** (see deviation 6) |
| `grep -c 'pub fn run_tuning' tune.rs` | **0** — it is `pub(crate)` |
| `test_fixtures` module gate | `#[cfg(test)]`, unchanged |
| `grep -c 'pub fn graph_tape_len' autograd/mod.rs` | **1** |
| `grep -c -E 'Duration\|Instant\|SystemTime\|chrono' evidence.rs` | **0** |
| `grep -c -i 'non-forgeable' evidence.rs` | **0** |
| `evidence.rs` `deny_unknown_fields` occurrences | **4** (both wire structs + both summary structs) |
| `UpdateEvidence` carries `batch_boundary_list` | yes — W-06's recorded source for 03-08 |
| `evidence.rs` uses `BTreeMap` for the per-parameter table | yes |
| `grep -c -iE 'abortion\|tweet\|stance' test_fixtures.rs` | **0** (see deviation 7) |
| `calibration_variants()` cell count | **6** (criterion: ≥ 6) |

### Known reds — measured, controlled, NOT caused by this plan

**1. Clippy (D-ITEM-02, re-measured for both crates this plan touches).**

| Leg | rc | Errors | Citing this plan's files |
|---|---|---|---|
| `cargo clippy -p aprender-train --lib --features setfit -- -D warnings` | 101 | 19, all `aprender-compute` | **0** |
| …the same WITHOUT `--features setfit` (control) | 101 | 19 | **0** |
| `diff -u` of the two error streams | **0** | byte-identical | |
| `cargo clippy -p aprender-core --lib --features setfit -- -D warnings` (N-06) | 101 | 20, all `aprender-compute` | **0** |

Diagnostics citing `crates/aprender-train/` or `crates/aprender-core/src`: **0** in every leg.
`autograd/mod.rs` specifically: **0**. Unchanged from 03-02's and 03-03's measurements.

**2. Minimal build (D-ITEM-05).** `cargo check -p aprender-train --no-default-features
--features setfit` → rc **101**; the control without `setfit` → rc **101**; the two error
streams are **byte-identical** (25 lines each, `diff` rc 0). `setfit` contributes zero, as
03-03 established.

### D-05 feature closure still holds after the dev-dependency

The new `aprender-core` dev-dependency with `conformance-fixtures` does NOT widen the shipped
dependency set, because the D-05 guards traverse `-e normal` and cargo does not follow dev edges
there. Measured both ways:

| Leg | Result |
|---|---|
| `cargo tree -p aprender-train -e normal` matches `aprender-contrastive-data\|aprender-rand\|tokenizers` | **0** |
| `cargo tree -p aprender-train --features setfit -e normal` | contrastive-data **1**, rand **3**, tokenizers **1** |

The absence half and the presence half both hold, so the absence is not vacuous.

---

## Guards falsified before being trusted

Neither new gate was believed on the strength of being green. Each had its failure mode induced,
observed and reverted, with the tree confirmed byte-identical afterwards (`git diff --stat`
empty).

| Guard | Induced mutation | Observed |
|---|---|---|
| `tune_digest_is_recorded_not_recomputed` | absorb `ordinals[position]` (the CONFIGURED order) instead of the drawn `pair_ordinal`, i.e. move the digest from consumption to configuration | **RED.** `assertion left != right failed` with the two 32-byte digests printed IDENTICAL — the exact false-green the in-band design removes. Reverted; tree byte-identical. |
| `calibration_matrix_epsilon_basis` separation | `CONTROL_LR: 1e-30 -> 2e-5` (the real rate), so the "control" is no longer null | **RED** on the first cell: `seed 1 cell s8e1b4 class embedding: the 1e-30 control's max relative delta (5.917e-4) is not below the real run's min (2.093e-4)`. Reverted. |

Two further gates turned red on their own honest first draft and are recorded as findings rather
than as guards that always worked — see deviations 2, 3 and 7.

---

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] The plan's Tensor-compatibility claim is false; `ParamBridge` added**

- **Found during:** Task 2, before writing any loop code.
- **Issue:** The plan's context states *"aprender-train's Tensor is the re-exported aprender-core
  autograd Tensor (lib.rs:139) — compatible with `trainable_parameters_mut`."* Measured: `lib.rs:139`
  is `pub use autograd::{backward, Context, Tensor};` where `autograd` is **aprender-train's own**
  module. `aprender_train::autograd::tensor::Tensor` wraps an `Array1<f32>` with an
  `Rc<RefCell<Option<Array1<f32>>>>` gradient and an `Rc<dyn BackwardOp>`;
  `aprender::autograd::Tensor` carries a `Vector<f32>` plus a shape and is differentiated through a
  thread-local tape. They are unrelated types from two unrelated autograd engines, so
  `AdamW::step_refs(&mut [&mut Tensor])` and `clip_grad_norm_refs` do not accept encoder parameters
  and the plan's Task 2 does not typecheck as written.
- **Fix:** `ParamBridge` — one `aprender-train` tensor per trainable parameter, allocated once
  pre-loop and index-aligned with `trainable_parameters_mut()`. Per step it loads current values and
  graph-side gradients in, `clip_grad_norm_refs` and `AdamW::step_refs` run on it unchanged, and the
  updated values are stored back. The plan's REUSE instruction and its acceptance criteria are
  satisfied literally; the alternative (`aprender::nn::optim::AdamW`, which is native to the core
  Tensor) would have required abandoning `clip_grad_norm_refs`, `step_refs` and
  `WarmupLinearDecayLR::apply` all three.
- **Cost, stated:** the bridge is a fifth full copy of the trainable parameters. See the memory
  accounting below.
- **Commit:** `6f948de52`

**2. [Rule 1 - Bug] Gradients are not on the parameter tensors**

- **Found during:** Task 2.
- **Issue:** `ComputationGraph::backward` writes gradients into the graph's own registry copies
  (`autograd/graph.rs:145`), not onto the caller's tensors. `param.grad()` on an encoder parameter
  is `None` immediately after a successful backward. A loop reading `param.grad()` — which is what
  both `clip_grad_norm_refs` and `AdamW::step_refs` do natively — would have clipped nothing and
  stepped nothing while the loss trace looked entirely plausible. This is PF-001 in a new costume,
  and it is precisely the class of defect this phase exists to close.
- **Fix:** every gradient read goes through `autograd::get_grad(param.id())`.
  `tune_gradients_reach_the_parameters_through_the_graph` pins the mechanism from both sides:
  tensor-side count **0**, graph-side count **> 0**, after a real backward.
- **Commit:** `6f948de52`

**3. [Rule 1 - Plan defect] The predicted gradient-accumulation failure does not occur**

- **Found during:** Task 2, when the plan's negative test failed for the wrong reason.
- **Issue:** The plan's Review fix 1 states that without `zero_grad_`, *"step N's gradient would be
  the sum of steps 0..N"*, and its acceptance criterion requires a negative asserting the second
  batch's gradient norm DIFFERS. Measured: it does not. `register_tensor` REPLACES a leaf's entry on
  every forward (`autograd/graph.rs:67`), and a previous step's sub-tape has no seeded output
  gradient so `backward` skips it entirely. Removing only step (b) is additionally unobservable past
  step 0, because step (l) still leaves an empty tape — the first draft of the test failed with
  `[24, 0, 0]`, which is the probe measuring nothing.
- **Fix:** two probes for two distinct properties, each with a measured effect.
  `SKIP_STEP_TOP_CLEAR` shows that step (b) is what makes *"the tape is empty when THIS step's
  forward begins"* true — with it removed, step 0 inherits **24** foreign operations from the model
  loader, and the test asserts that count equals what the pre-loop encode reported.
  `SKIP_ALL_GRAPH_CLEARS` shows monotone tape growth across steps and a stale graph-side gradient at
  a later step's top (T-3-55). The plan's stated property is replaced by the two that are true, and
  both are asserted, not argued.
- **Commit:** `6f948de52`

**4. [Rule 3 - Blocking] `from_slice_fixture` is not reachable under `setfit`**

- **Found during:** Task 1. The plan asked to *"confirm it is reachable cross-crate under the setfit
  feature"* — it is not.
- **Issue:** `SetFitMiniLm::from_slice_fixture` is `#[cfg(feature = "conformance-fixtures")]`
  (`aprender-core/src/setfit/mod.rs:237`). The plan's *"it is `pub`, so cross-crate reuse needs no
  new door"* is true of its VISIBILITY and false of its REACHABILITY. Without it there is no
  network-free constructor for a real-weight encoder and Task 1 cannot exist.
- **Fix:** a `[dev-dependencies]` entry on `aprender-core` with `features = ["conformance-fixtures"]`.
  This is the only mechanism that enables a dependency's feature for TEST builds alone — cargo
  forbids optional dev-dependencies, and adding it to the `setfit` feature would put a test-only
  constructor plus the `encoder()`/`tokenize()` conformance accessors on every production build,
  which is the test-support backdoor 03-10's criteria reject. Measured not to widen the shipped
  dependency set: both D-05 tree legs still hold.
- **Files:** `crates/aprender-train/Cargo.toml` (not in `files_modified`, mechanically required)
- **Commit:** `d32eb431c`

**5. [Rule 1 - Plan defect] `preflight` takes three values, not a `&ResolvedSetFitConfig`**

- **Found during:** Task 2, writing the two device/`max_length` negatives the plan requires.
- **Issue:** A `ResolvedSetFitConfig` is only mintable by `resolve()`, which probes the host. On a
  CPU-only machine there is no way to construct a CUDA-RESOLVED configuration, so the
  `UnsupportedDeviceForPhase3` rejection would have had no reachable test — the plan asks for a test
  that its own type design makes unwritable.
- **Fix:** `preflight(encoder, device, requested_max_length, freeze_policy)`. Both negatives are now
  direct and neither needs a test-only constructor on the validated config type, which would have
  been a wider door than the thing it tests. Each negative carries a CONTROL asserting the same call
  with valid inputs succeeds, so neither is "preflight always fails".
- **Commit:** `6f948de52`

**6. [Rule 1 - Plan defect] The `par_iter` criterion is unsatisfiable as literally written**

- **Issue:** `grep -rc 'par_iter' train/setfit/ | grep -v ':0$'` prints
  `reduce.rs:2` and `tune.rs:1`. All three are DOC COMMENTS explaining why `par_iter` is absent —
  the same trap 03-03 recorded for this exact criterion (its deviation 6). No conforming code can
  satisfy the literal form while also documenting the decision the plan requires documenting.
- **Fix:** measured on NON-COMMENT lines: **0** occurrences. The criterion's substance holds.

**7. [Rule 1 - Bug] The provenance gate turned red on its own prose**

- **Found during:** Task 1.
- **Issue:** `grep -c -iE 'abortion|tweet|stance' test_fixtures.rs` returned **1** — the module doc
  named the source corpus while explaining that no corpus text is committed. Third occurrence of
  this pattern in the phase (03-02 twice, 03-03 once).
- **Fix:** the paragraph now DESCRIBES the forbidden terms instead of quoting them, and says why in
  place so the next reader does not reintroduce it. Re-measured: **0**. The red observation IS the
  falsification evidence for this gate — it was seen failing before it was seen passing.
- **Commit:** `d32eb431c`

**8. [Rule 1 - Bug] The baseline-encode assertion was an absolute where only a delta is owned**

- **Found during:** Task 2.
- **Issue:** `tune_baseline_encode_records_no_operations` asserted `tape_len == 0` after the
  `no_grad` encode and failed with `24`. Investigation (below) showed the 24 entries come from the
  MODEL LOADER, not from the encode. The absolute form was a test of `from_slice_fixture` wearing
  this function's name.
- **Fix:** assert the DELTA across the encode (`before == after`) plus a non-vacuity assertion that
  `before > 0`, with a doc comment stating that if the loader is ever fixed this test becomes
  vacuous and must be tightened. The absolute `no_grad` claim is pinned where it belongs, in
  `aprender::autograd::tests::autograd_graph_tape_len_stays_zero_under_no_grad`, which carries its
  own recording control.
- **Commit:** `6f948de52`

**9. [Rule 3 - Blocking] The calibration report cannot be written to `target/`**

- **Issue:** the first run of `calibration_matrix_epsilon_basis` panicked with `NotFound` writing
  `target/setfit-calibration-matrix.txt`. `.cargo/config.toml` redirects the target directory, so it
  does not exist relative to the test's cwd.
- **Fix:** `SETFIT_CALIBRATION_REPORT` when set, otherwise `std::env::temp_dir()`. A file is written
  at all because the repo's `cargo` wrapper drops `--nocapture` output entirely, and a calibration
  whose numbers cannot be read is a calibration that did not happen.
- **Commit:** `623d529da`

**10. [Rule 1 - Bug] The zero-init test asserted the wrong failure direction**

- **Issue:** the first draft asserted that the un-floored `||dTheta|| / ||theta_init||` fails a
  threshold comparison at zero init. Measured: with `delta > 0` it is `+inf`, which is GREATER than
  every threshold — the un-floored form ACCEPTS, it does not reject.
- **Fix:** the test now demonstrates BOTH failure modes, which fail in opposite directions:
  `+inf` accepts any movement from zero however small, and `NaN` (from `0/0`) is rejected because
  `NaN > eps` is false. The module doc named only one; both are now recorded.
- **Commit:** `623d529da`

**11. [Rule 2 - Missing critical] `ParameterRegistryMoved` is a hard error, not an observation**

- **Issue:** the plan says step (h) "asserts" the registry hash. A recorded boolean would let a
  reordered registry proceed to the optimizer step, where AdamW's POSITIONAL moment state would pair
  every moment with the wrong parameter — silently wrong rather than loudly broken (T-3-54).
- **Fix:** a typed `SetFitTrainError::ParameterRegistryMoved` returned before `step_refs` runs, plus
  the boolean recorded per step so the evidence shows the check happened. The registry hash is
  length-prefixed, so `["ab","c"]` and `["a","bc"]` cannot collide — asserted by
  `tune_registry_hash_is_length_prefixed`.
- **Commit:** `6f948de52`

---

**Total deviations:** 11 auto-fixed — 3 blocking, 3 plan-defect corrections, 4 bugs, 1
missing-critical. No scope creep: every change is inside this plan's declared subsystem, and the
three blocking items were mechanically required for the plan's own instructions to compile or its
own tests to be writable.

---

## The step order, and how it is pinned

```
(a) scheduler.get_lr() -> adamw.set_lr        (b) zero_grad_ on every param, then clear_graph
(c) set_forward_ordinal(2s+0), forward A      (d) set_forward_ordinal(2s+1), forward B
(e) pair_cosine_mse                           (f) backward
(g) per-name PRE-clip grad norms (reduce.rs)  (h) registry hash assertion
(i) clip_grad_norm_refs                       (j) adamw.step_refs
(k) scheduler.step()                          (l) clear_graph
(m) push loss
```

The pin is **behavioural**, which is strictly stronger than reading the source. `tune_step_order_is_pinned`
asserts, on a 3-step run and from observations the loop recorded as it ran:

- every trainable parameter's gradient is absent at the top of every step, on **both** sides
  (tensor-side 0 AND graph-side 0);
- `graph_tape_len()` is **0** at the top of every step;
- step 0's applied learning rate is **exactly 0.0** and is **not** `config.encoder_lr` — reference
  fidelity, since HF's `LambdaLR` lambda(0) is 0 (03-03 pinned this and warned against "fixing" it);
- and, so the previous assertion is not satisfied by a scheduler stuck at zero, at least one later
  step has a positive rate.

Step count on every cell is exactly `epochs * ceil(n_pairs / batch_size)`, asserted over all six
variants. `warmup_steps` comes from 03-03's single `warmup_steps_from_ratio` — never re-derived.

## In-band digests

`consumed_pair_digest` absorbs, once per pair AT THE DRAW, inside `run_batch`'s pair-pull loop:
`epoch:u32 || batch_index:u32 || position_in_batch:u32 || pair_ordinal:u64 || a:u32 || b:u32 ||
target.to_bits():u32`, all little-endian. `batch_boundary_digest` absorbs, once per batch at batch
open: `epoch:u32 || batch_index:u32 || global_step:u64 || batch_start_ordinal:u64 || batch_len:u32`.

The readable `Vec<(epoch, start_ordinal, len)>` survives into `UpdateEvidence.batch_boundary_list`
(W-06) — 03-08's `batch_boundaries()` accessor has a RECORDED source and does not have to recompute.

## Reproducibility observed

- **In-process (TRN-06's first signal):** two `run_tuning` calls from identical inputs produce equal
  loss traces, equal loss-trace hashes, equal `consumed_pair_digest`, equal `batch_boundary_digest`,
  equal boundary lists, equal step counts, equal per-parameter measurements and equal final
  embeddings. Non-vacuity asserted: the trace has 3 finite entries.
- **Cross-process (a bonus, not required by this plan):** the calibration report — 30 rows of
  measured relative deltas plus 6 endpoint rows — is **byte-identical** across two separate
  invocations of the test binary (`diff` rc 0).

## Memory accounting, stated rather than discovered

Peak tuning memory is **five** copies of the trainable parameters, not four: the live parameters,
the initial snapshot, `ParamBridge`'s mirror, and AdamW's two moment buffers. The bridge is the copy
an earlier count would have missed; it is a real cost of reusing the reference optimizer across two
`Tensor` types rather than reimplementing it, and it is named rather than absorbed into a round
number.

| | Pinned slice (measured, asserted) | `all-MiniLM-L6-v2` (projected) |
|---|---|---|
| Trainable parameters | 37 tensors, **110,528** elements | ~22.7 M elements |
| One copy | **442,112 bytes** | ~90.8 MB |
| Peak (5 copies) | ~2.16 MB | **~454 MB** |

## Observed support fractions

`delta_support_fraction` is **1.0000** for every dense class in every cell — a batch that touches a
block touches all of it, which is exactly why only the embedding class is support-restricted. For
the sparse class it is **0.3196 – 0.3505** across the matrix: a few-shot batch touches roughly a
third of the 97-row slice vocabulary. On the production 30522-row vocabulary the same absolute row
count is a fraction ~300x smaller, which is the regime the support-restricted denominator exists to
make comparable.

---

## Known Stubs

None. Every type this plan ships is fully implemented for its declared scope.
`Verdict` has exactly one arm (`Unjudged`) and `epsilon_used` is always `None` — that is the plan's
explicit record-only contract, not a stub: a `Pass`/`Fail` arm shipped now would invite a comparison
against a threshold that does not exist until 03-06 freezes it. `EncoderTuned` still has no
`LifecycleState` impl, deliberately: `run_tuning` is `pub(crate)` and mints no state transition,
which is this plan's stated caveat.

## Threat Flags

None. Every file this plan touches is inside the declared `<threat_model>` surface: no new network
endpoint, no new schema at a trust boundary beyond the evidence wire form (T-3-16 covers it), and
the only filesystem write is the `#[ignore]`d calibration report into a temp path.

Mitigations applied: T-3-16 (binding hash + mutation test), T-3-17 (reduce.rs-only aggregation,
`BTreeMap` order, wall-clock-free structs, two-run bitwise test), T-3-18 (synthetic-text-only
fixture with a grep gate that was observed failing before it was observed passing), T-3-19
(`apply_freeze` before the registry hash, the snapshot AND the baseline encode; zero-match test),
T-3-38 (in-band absorption, falsified by induced recomputation), T-3-39 (behavioural step-order pin
plus two clearing negatives), T-3-42 (6-cell matrix, per-class distributions, support-restricted
denominator, `calibration_regime_id` recorded in the evidence), T-3-54 (typed
`ParameterRegistryMoved` before every step, length-prefixed hash), T-3-55 (`clear_graph` per step,
`graph_tape_len` asserted 0 at every step top, growth demonstrated when removed).

## Issues Encountered

- **`rtk` reshapes measurement output.** `cargo clippy`'s diagnostic stream arrives as a summary
  ("20 errors, 10 warnings") rather than the raw stream, which made a `grep -A1 -E '^error'`
  two-sided diff silently compare two EMPTY files — a vacuous pass. Caught by the line-count check
  before the diff. Every load-bearing measurement in this summary was re-taken through `rtk proxy`.
  Extends 03-03's finding from `grep`/`diff`/`wc` to `cargo clippy`'s and `cargo test`'s output
  shape.
- **`println!` from a test does not survive the wrapper either**, which is why the calibration report
  is written to a file. An early scratch probe had to encode its measurements into a panic message to
  be readable at all.
- **The full single-threaded suite takes 1115 s.** Budget for it; it is the only way to compare
  against the known-red baseline without parallelism artifacts.

## Next Phase Readiness

**03-06** has everything it needs to freeze epsilon: five per-class distributions across six cells
with their controls, the parameter that binds each class's lower edge by name, the proposed
`calibration_regime_id`, and one explicit FLAG (`projection_bias`) with three options and the numbers
behind each. It must also implement `LifecycleState for EncoderTuned` and mint the public
`tune_encoder` transition around `run_tuning`. **It should NOT arm a gate on the endpoint means** —
the cross-seed spread is 0.49 and the sign flips.

**03-07** gets the same fixture door and the same `no_grad` + `set_training(false)` encode pattern,
plus `graph_tape_len()` for its "no graph was built" assertion.

**03-08** gets `batch_boundary_list` as the RECORDED source for `batch_boundaries()`, the consumed
`max_length`, both digests, the registry hash and the canonical evidence bytes.

**03-10** gets a `pub(crate)` `run_tuning` and a `#[cfg(test)]` `test_fixtures` — neither visibility
was widened, and the calibration lives inside the crate behind `#[ignore]` precisely so it did not
have to be.

**Concerns for the orchestrator:**

1. **D-ITEM-06 is new** — the MiniLM slice loader leaves 24 operations on the autograd tape at LOAD
   time. Not a correctness bug today (the loop clears before every forward) but a live trap and a
   small permanent allocation for inference callers. Logged with its fix direction and the test that
   will turn red when it is fixed.
2. **`projection_bias` cannot be frozen at the 10x margin** — flagged with numbers, three options, no
   silent narrowing.
3. **The 1e-30 control is exactly zero** in every cell, so the matrix bounds epsilon from above only.
   A control at ~1e-8 would give a real lower bound if 03-06 wants one.
4. This plan did **not** touch `STATE.md` or `ROADMAP.md`, per the worktree protocol.
   `deferred-items.md` was appended to (D-ITEM-06); if 03-04 also appended, the conflict is an
   append-order resolution.

## Requirements

`requirements-completed: []` — **deliberately empty.** This plan's frontmatter declares TRN-03 and
TRN-06. TRN-03 ("the encoder demonstrably moved, per the SetFit-identity gate") is not complete
until 03-06 arms the gate against a frozen epsilon; this plan CAPTURES the evidence and explicitly
makes no judgment, which is its own stated caveat. TRN-06 is also claimed by 03-08 and 03-10 and
this plan delivers only the in-process half — there is no cross-process comparison here.
Checking either box now would put a false claim in the traceability table.
`REQUIREMENTS.md` is byte-unchanged.

---
*Phase: 03-faithful-two-stage-trainer-and-head*
*Plan: 05*
*Completed: 2026-08-09*

## Self-Check: PASSED

Files verified present on disk:

- `crates/aprender-train/src/train/setfit/test_fixtures.rs` — FOUND (19.9K)
- `crates/aprender-train/src/train/setfit/tune.rs` — FOUND (56.3K)
- `crates/aprender-train/src/train/setfit/evidence.rs` — FOUND (49.8K)
- `.planning/phases/03-faithful-two-stage-trainer-and-head/03-05-SUMMARY.md` — FOUND (43.3K)

Commits verified in `git log`:

- `d32eb431c` — FOUND (Task 1)
- `6f948de52` — FOUND (Task 2)
- `623d529da` — FOUND (Task 3)
- `b312a4498` — FOUND (snapshot accounting + D-ITEM-06)

Diff against base `66ef41495` touches exactly eight files — this plan's five declared
`files_modified`, plus `crates/aprender-train/Cargo.toml` (deviation 4, mechanically required),
`deferred-items.md` (D-ITEM-06) and this SUMMARY. **`STATE.md`, `ROADMAP.md` and
`REQUIREMENTS.md` are NOT in the diff** — the orchestrator owns those writes. Nothing owned by
the parallel plan 03-04 (`classification/`, `contracts/`, `binding.yaml`, `Makefile`) was
touched. **Zero file deletions** across the whole range. `git status --short` clean.
