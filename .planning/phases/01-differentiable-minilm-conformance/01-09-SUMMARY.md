---
phase: 01-differentiable-minilm-conformance
plan: 09
subsystem: autograd
tags: [autograd, attention-mask, broadcast, gelu, erf, numerics, conformance]
requires:
  - contract:setfit-encoder-conformance-v1
provides:
  - op:apply_additive_mask
  - op:gelu_exact
  - gradfn:GeluExactBackward
  - fn:batuta_common::math::erf_precise
  - fn:batuta_common::math::erfc_precise
affects:
  - crates/aprender-core/src/nn/transformer/
  - crates/aprender-core/src/autograd/
  - crates/aprender-common/src/math.rs
tech-stack:
  added: []
  patterns:
    - "constant-expand-then-autograd-add (PMAT-922) instead of a new broadcast backward"
    - "right-aligned numpy/torch broadcasting with debug_assert on non-broadcastable dims"
    - "independently derived in-test oracle (series + continued fraction) vs production (Cody)"
    - "erfc-form evaluation to avoid negative-tail catastrophic cancellation"
key-files:
  created:
    - crates/aprender-core/src/nn/transformer/tests_attention_mask_broadcast.rs
    - crates/aprender-core/src/autograd/ops/tests_gelu_exact_backward.rs
  modified:
    - crates/aprender-core/src/nn/transformer/positional_encoding.rs
    - crates/aprender-core/src/nn/transformer/tests.rs
    - crates/aprender-core/src/autograd/ops/activation.rs
    - crates/aprender-core/src/autograd/grad_fn.rs
    - crates/aprender-core/src/autograd/tests_matmul_backward.rs
    - crates/aprender-common/src/math.rs
decisions:
  - "add_mask repaired by expanding the mask to a CONSTANT of the scores' shape and applying it via scores.add(&expanded) — reuses the contract-covered AddBackward, adds no new severable path"
  - "non-broadcastable mask dims: debug_assert! + deterministic release clamp; explicitly NOT 'return scores unmodified', which would silently disable masking (T-1-18)"
  - "MEASURED: max |gelu_exact - gelu| over [-6,6] is 4.7325e-4, so the plan's >1e-3 differential assertion is unsatisfiable against correct code; asserted >3e-4 instead"
  - "A&S erf ruled INSUFFICIENT by measurement (129 f32 ulps of local value at x=-2.67); added Cody erf_precise/erfc_precise to aprender-common additively, existing erf untouched"
  - "gelu_exact evaluated as 0.5*x*erfc(-x/sqrt(2)) — algebraically identical to the contract formula, numerically stable in the negative tail"
  - "Task 2 RED was proven by measurement but NOT captured as a separate commit — recorded honestly rather than claimed"
metrics:
  duration: ~2h
  completed: 2026-08-08
  tasks: 2
  commits: 3
  tests_added: 24
---

# Phase 1 Plan 09: Broadcast Attention Mask Repair and Exact erf GELU Summary

Two confirmed defects upstream of every ENC-03 parity gate are closed: the attention mask now
broadcasts correctly *and* keeps its autograd edge at every shape combination, and an exact erf
GELU exists as a seventh differentiable primitive with an independently derived accuracy oracle.

## What Was Built

### 1. `add_mask` repair (amendment A-02, equation `apply_additive_mask`)

The old body applied the mask via `scores.add(mask)` — which records a graph edge — **only** when
shapes matched exactly, and otherwise fell through to a `.zip()` over the two data slices.

**The defect was worse than the plan described.** The plan predicted silent truncation ("writes the
mask over only the first `B*S` elements"). In fact `Tensor::from_vec` asserts
`data.len() == shape.product()` (`autograd/tensor.rs:111`), so the `.zip()` produced `B*S` values
against a `B*H*T*S` shape and the call **panicked outright**:

```
Data length 10 doesn't match shape [2, 4, 3, 5] (expected 120)
```

The masked attention path was therefore not merely wrong at realistic shapes — it was
**unreachable**, and no test in the repo exercised it (`scaled_dot_product_attention` is the sole
caller, and no existing test passes `Some(mask)` to it). That is why a defect this loud survived.

The repair expands the mask into a constant of exactly the scores' shape by right-aligned
numpy/torch broadcasting, then applies it with `scores.add(&expanded)`. Because the expanded tensor
matches shape exactly, the existing `AddBackward` records the edge — the PMAT-922 pattern, reusing
an already-contracted backward rather than adding a new severable code path.

### 2. `Tensor::gelu_exact` (amendment A-03, equation `gelu_exact`)

Exact erf GELU with `GeluExactBackward` (`Phi(x) + x*phi(x)`), computed in f64 and narrowed at
store time. `Tensor::gelu` is **unmodified** — `git diff` on `activation.rs` shows additions only,
no removed lines. Both methods now document which HF `hidden_act` they correspond to.

### 3. `erf_precise` / `erfc_precise` in `aprender-common` (Cody rational Chebyshev)

Added **additively**; the existing A&S `erf` is untouched for its current stats callers.

## The Two Measurements That Changed the Plan

The plan required measuring rather than assuming on both points. Both measurements came back
against the plan's provisional figures.

### A&S erf is INSUFFICIENT — and the reason is cancellation, not the error bound

| Quantity | Measured |
|---|---|
| A&S-based `gelu_exact` max **absolute** deviation | 4.77e-7 (at x=4.81) = **1.00 f32 ulp** |
| A&S-based deviation **relative to the local value** | **129 f32 ulps** at x = -2.67 |
| Cody-based deviation vs independent oracle | **2.3528e-7** at x=5.35 = **0.49 f32 ulps** |

Judged on absolute error alone A&S looks acceptable (1 ulp). It is not, and the plan's ">2 ulps of
the corresponding value" rule is what catches it: at x = -2.67, `1 + erf(x/sqrt(2))` is the sum of
two near-equal magnitudes leaving ~0.0077 out of ~1.0, so a 1.5e-7 *absolute* erf error becomes a
~2e-5 *relative* output error. That is a systematic, fixed-sign bias that compounds across six FFN
layers rather than averaging out.

Two fixes were required, not one:
1. **Accuracy** — `erf_precise`/`erfc_precise` (Cody CALERF, ~1e-15).
2. **Formulation** — evaluate `0.5*x*erfc(-x/sqrt(2))` rather than `0.5*x*(1 + erf(x/sqrt(2)))`.
   These are algebraically identical (`1 + erf(t) == erfc(-t)`) but only the erfc form avoids the
   cancellation. A high-accuracy erf alone would *not* have fixed the negative tail.

The result — 0.49 ulps — means the implementation is correctly rounded in f32.

### The plan's `>1e-3` differential assertion was unsatisfiable

| Quantity | Measured |
|---|---|
| max \|gelu_exact - gelu\| over [-6,6] | **4.7325e-4** at x = -2.7 |

The plan's acceptance criterion asked to assert `max |gelu_exact - gelu| > 1e-3`. The true maximum
is 4.73e-4, so that assertion **fails against a correct implementation** — the same class of error
01-01 caught in the ENC-04 gradient gate. The test asserts `> 3e-4`, which sits below the measured
maximum and ~3 orders of magnitude above f32 noise, so it cannot be satisfied by rounding while
still failing loudly if `gelu_exact` is ever routed back to the tanh form.

**No contract tolerance was edited** (`contracts/setfit-encoder-conformance-v1.yaml` is untouched —
01-04 owns that table). The contract's prose "differ by ~1e-3 near |x|≈2" is within an order of
magnitude of 4.73e-4; the precise figure is recorded here so 01-04 can pin the tolerance table
against a measured number rather than an estimate.

## What Makes These Gates Non-Tautological

**The oracle is independently derived.** Production uses Cody's rational Chebyshev approximation;
the in-test oracle uses a completely different derivation (Maclaurin series for |t| <= 2, Laplace
continued fraction beyond). The oracle is *itself* verified against known high-precision erf values
to < 1e-14, because an oracle that is wrong is worse than none — its measured agreement with libm
across a dense grid is 1.1e-15.

**The differential guard is two-sided.** One test proves the two functions differ; a second proves
`gelu_exact` is the one that is *right* (its worst oracle deviation is >100x smaller). Differing
alone would also be satisfied by a *newly broken* implementation.

**Point values cannot distinguish the two functions.** At x=1 the exact and tanh forms agree to
1.5e-4, and both round to `0.8413` at 4 digits — a spot-check passes on the wrong function. Only a
scan over the divergence region separates them, which is why the differential test scans a 241-point
grid rather than asserting at a few points.

**The mask tests attack indexing directly.** Distinct per-`(b,s)` mask values with distinct scores;
a zero-score test asserting each batch block equals its own mask row (wrong-axis catcher); a
head/query uniformity test; and `T != S` (3 vs 7), the shape a square-mask assumption gets wrong.

**The mask backward is a numeric check, not `is_some`.** The mask is a constant, so
`d(sum(c * masked))/d(scores) == c` exactly — asserted elementwise against the coefficient vector.

## Tasks and Commits

| Task | Gate | Commit | Result |
|------|------|--------|--------|
| 1 — `add_mask` | RED | `697317a27` | **7 failed / 1 passed** (panic in `Tensor::from_vec`) |
| 1 — `add_mask` | GREEN | `f0b095402` | 8 passed |
| 2 — `gelu_exact` | RED | *(not committed — see below)* | **8 failed / 4 passed** |
| 2 — `gelu_exact` | GREEN | `0735ad7a6` | 12 passed |

### TDD Gate Compliance

**Task 1 has the full `test(...)` -> `fix(...)` commit pair.** GREEN used `fix` rather than `feat`
because it repairs a confirmed defect; the commit-type table assigns `fix` to bug fixes.

**Task 2's RED was proven by measurement but NOT captured as a separate commit.** This is a real gap
against the TDD protocol, recorded rather than glossed. The RED state was a stub
`pub fn gelu_exact(&self) -> Tensor { self.gelu() }` — deliberately the exact tampering threat
T-1-19 describes — and it produced **8 failed / 4 passed**, with each discriminating test failing
for its own reason:

| Test | RED failure |
|---|---|
| `..._matches_the_independent_f64_oracle...` | deviates by 4.7325e-4 at x=-2.7 |
| `..._is_a_different_function_from_the_tanh_gelu` | differ by only 0.0000e0 |
| `..._relative_accuracy_holds_in_the_negative_tail` | rel error 2.172e-3 at x=-1.5 |
| `..._tracks_the_oracle_more_closely_than_the_tanh...` | worst 4.732e-4 vs 4.732e-4 |
| `..._records_a_gelu_exact_backward_edge` | left `"GeluBackward"`, right `"GeluExactBackward"` |
| `..._backward_matches_the_closed_form_derivative` | -0.01158 vs -0.01195 at x=-3 |
| `..._matches_known_point_values` | 0.841192 vs 0.8413447 |
| `..._propagates_non_finite_input...` | finite element wrong under the stub |

A stub RED is stronger than a compile-error RED — every assertion is proven reachable and
meaningful — and it keeps the branch building at every commit, which matters because the
orchestrator merges this worktree with others.

## Verification

Every exit code below was captured with `cmd > file 2>&1; rc=$?`, never read through a pipe
(CLAUDE.md rule 1).

| Check | Result |
|-------|--------|
| `cargo test -p aprender-core --lib attention_mask_broadcast_` | 8 passed (RED: 7 failed / 1 passed) |
| `cargo test -p aprender-core --lib gelu_exact_` | 12 passed (RED: 8 failed / 4 passed) |
| `cargo test -p aprender-core --lib transformer` | 177 passed |
| `cargo test -p aprender-core --lib test_all_backward_names` | 1 passed |
| `cargo test -p aprender-core --lib autograd` | 197 passed |
| `cargo test -p aprender-core --lib` (full) | **14063 passed, 2 ignored, 0 failed** (+20 net) |
| `cargo test -p aprender-common --lib precise` | 4 passed |
| `cargo test -p aprender-common --doc` | 29 passed |
| `cargo clippy -p aprender-common --lib --tests -- -D warnings` | exit 0 |
| `cargo clippy -p aprender-core --lib --tests` | exit 0, **0 findings in any changed file** |
| `cargo fmt -p aprender-core -p aprender-common -- --check` | exit 0 |
| `cargo build -p aprender-core --no-default-features` | exit 0 (both changes ungated) |
| `git check-ignore -v` on both new `#[path]` files | exit 1 (not ignored) |

**Binary/target pinning (CLAUDE.md rule 3 + 8).** An early run of
`cargo test -p batuta-common --lib erf` reported "5 passed" and exit 0 — while testing the
**crates.io** `batuta-common@0.1.0`, not the in-tree source. `batuta-common` is a dependency alias
for the package `aprender-common`, and a same-named registry crate exists. The four new tests had
not been compiled at all. Re-run correctly as `-p aprender-common`. Logged as deferred item **D7**;
every `aprender-common` result above uses the correct selector.

### Acceptance-criteria source assertions

| Criterion | Evidence |
|---|---|
| No `.zip(` over `scores.data()`/`mask.data()` | only remaining match is the doc comment describing the old defect |
| Broadcast path uses an autograd-aware op | `scores.add(&expanded)`, not a bare `Tensor::from_vec` of sums |
| `gelu_exact` body has no `tanh` / `0.044715` | 0 matches |
| `gelu_exact` body uses erf | 1 match (`erfc_precise`) |
| `Tensor::gelu` unmodified apart from docs | `git diff activation.rs` has **no removed lines** |
| Contract tolerance table untouched | `contracts/` absent from `git status` |

## Deviations from Plan

### 1. [Rule 1 - Correctness] Differential threshold 3e-4, not the plan's 1e-3

Measured max divergence is 4.7325e-4. Asserting `>1e-3` would fail the phase on correct code. Fully
documented in the test body with the measured number so the threshold is not mistaken for arbitrary.

### 2. [Plan-authorized] `crates/aprender-common/src/math.rs` modified

Not in the plan's `files_modified` list, but the plan explicitly pre-authorized it: *"implement
`erf_precise(f64) -> f64` ... additively in `crates/aprender-common/src/math.rs` ... Adding that
function is IN SCOPE for this task."* The measurement triggered that branch. Existing `erf` and
`erf_f32` are byte-identical; four accuracy tests added.

### 3. [Rule 2 - Correctness] `erfc_precise` added alongside `erf_precise`

The plan named only `erf_precise`. A high-accuracy *erf* alone does not fix the negative tail —
the cancellation is in the `1 + erf(...)` formulation, not only in erf's error. `erfc_precise` is
required to make `gelu_exact` accurate where it matters, and is what the implementation calls.

### 4. [Rule 3 - Blocking] Clippy lint conflict on the Cody coefficient tables

`excessive_precision` (9x) then `inconsistent_digit_grouping` (11x) fired on the transcribed
constants — the two lints cannot both be satisfied here, since the integer and fractional parts have
different digit counts per row. Resolved with plain ungrouped literals plus a justified
`#[allow(clippy::unreadable_literal)]` per table. Literals are the shortest round-tripping f64 forms
of Cody's published values, so the stored bit patterns are exactly the published constants, and the
table stays diffable against its source.

### 5. New private helper `broadcast_mask_to` in `positional_encoding.rs`

The plan asked for changes "confined to `add_mask` and its doc comment". The broadcast logic is a
separate private function for testability and to keep `add_mask` under complexity limits. It is
called only by `add_mask`.

### 6. Non-broadcastable-shape behavior (recorded per plan request)

`debug_assert!` on both mask rank and per-dimension broadcastability (fires in every test and debug
build), and in release the mask index is clamped into range — deterministic and panic-free. It
deliberately does **not** "return scores unmodified", which would silently disable masking and let
padded keys contribute to attention (threat T-1-18).

### 7. Task 2 RED not committed separately

See TDD Gate Compliance above. Recorded as a gap, with the measured RED evidence, rather than
claimed as compliant.

## Known Stubs

None. The Task 2 RED stub was replaced in its GREEN commit; a scan of both new files and all
modified files for `RED stub` / `TODO` / `FIXME` / `unimplemented` / `todo!` returns nothing. A
temporary probe file used to verify a `Tensor::gelu` reading was removed before commit — the working
tree is clean.

## Threat Flags

None. This plan adds no network endpoint, auth path, file-access pattern, or schema change, and no
new packages (T-1-SC held: `erf_precise`/`erfc_precise` are in-tree additions to an existing
non-optional dependency).

The three `mitigate` dispositions in the plan's register are discharged:

| Threat | Mitigation delivered |
|---|---|
| T-1-18 (info disclosure via unmasked keys) | correct right-aligned broadcast, elementwise-verified at B>1/H>1/T!=S; non-broadcastable dims never disable masking |
| T-1-05 (severed graph in masked attention) | requires_grad + grad_fn + exact numeric backward asserted on the BROADCAST path, not just the equal-shape path |
| T-1-19 (tanh substituted for exact GELU) | two-sided differential guard + independently derived f64 oracle |

## For the Orchestrator

- `STATE.md` and `ROADMAP.md` were **not** touched (worktree mode).
- `REQUIREMENTS.md` was **not** touched. ENC-03 is *not* complete: this plan lands two repaired
  primitives on its path, not the requirement itself (the encoder path is 01-06, fixtures 01-04).
  Marking it here would be labelling by intent rather than evidence.
- New deferred items **D7** (`-p batuta-common` tests a registry crate — false-green hazard) and
  **D8** (`Tensor::gelu` verified correct, recorded to prevent a future false alarm) appended to
  `deferred-items.md` under a new `## From plan 01-09` section; D1–D6 untouched.
- **For 01-04's tolerance table:** the FFN activation tolerance should be pinned against the
  measured figures here — exact-vs-tanh divergence **4.7325e-4** (max, at x=-2.7), and `gelu_exact`
  accuracy **2.3528e-7** (0.49 f32 ulps). Do not widen tolerance to absorb the former.
- **For 01-06:** the encoder FFN must call `Tensor::gelu_exact`, never `Tensor::gelu`.

## Self-Check: PASSED

Both created files verified present on disk. All 3 commits verified present in
`git log e5c2c0c5f..HEAD`. `contracts/` confirmed unmodified. Working tree clean apart from this
SUMMARY and `deferred-items.md`.
