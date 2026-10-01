---
phase: 01-differentiable-minilm-conformance
plan: 03
subsystem: autograd
tags: [autograd, normalize, cosine, mse, epsilon-clamp, piecewise-derivative, batched-graph, spike]
requires:
  - contract:setfit-encoder-conformance-v1
  - op:embedding_gather
  - op:additive_attention_mask
  - op:masked_mean_pool
  - op:apply_additive_mask
  - op:gelu_exact
provides:
  - op:l2_normalize_rows
  - op:cosine_similarity_rows
  - op:mse_loss
  - gradfn:L2NormalizeRowsBackward
  - gradfn:CosineSimilarityBackward
  - gradfn:MseBackward
  - variant:autograd::OpError::InvalidEpsilon
  - test:batched_graph_spike
affects:
  - crates/aprender-core/src/autograd/
  - crates/aprender-core/tests/
tech-stack:
  added: []
  patterns:
    - "piecewise clamp derivative: store the RAW norm, re-take the forward's branch in the backward"
    - "f64 accumulation of sums-of-squares so a 1e-20 component does not underflow the clamp decision"
    - "error payload stored as IEEE-754 BITS so the enum keeps Eq and a NaN case stays assertable"
    - "two-sided mutation measurement per branch, recorded rather than committed"
    - "aggregate-per-component gradient gate + two-sided near-zero exemption for analytically-zero tensors"
key-files:
  created:
    - crates/aprender-core/src/autograd/ops/normalize.rs
    - crates/aprender-core/src/autograd/ops/similarity.rs
    - crates/aprender-core/src/autograd/ops/tests_normalize_backward.rs
    - crates/aprender-core/src/autograd/ops/tests_similarity_backward.rs
    - crates/aprender-core/tests/batched_graph_spike.rs
  modified:
    - crates/aprender-core/src/autograd/grad_fn.rs
    - crates/aprender-core/src/autograd/mod.rs
    - crates/aprender-core/src/autograd/ops/mod.rs
    - crates/aprender-core/src/autograd/ops/op_error.rs
    - crates/aprender-core/src/autograd/tests_matmul_backward.rs
decisions:
  - "epsilon clamp derivative implemented PIECEWISE; the backward stores the raw norm because a clamped output carries no record of which side it came from"
  - "branch boundary n == eps assigned to the CONSTANT branch (projected form requires the strict n > eps), documented on op, backward and tests"
  - "OpError::InvalidEpsilon carries eps as u32 BITS: an f32 field would forbid the enum's Eq derive AND make the NaN rejection test unsatisfiable (NaN != NaN)"
  - "cosine denominator clamps each FACTOR independently per the plan, not the PRODUCT per the contract prose; the two coincide across the whole non-degenerate domain — logged as D12 for 01-04/01-08 to reconcile in YAML"
  - "mse_loss requires rank-1 pred and scans the TARGET (untrusted labels) but not pred (computed intermediate), following masked_mean_pool's documented rule"
  - "SPIKE RESULT: batch>1 gradient flow through the repaired masked attention path WORKS — 36/36 parameter tensors, all 7 ENC-04 components, padding invariance exactly 0.0"
  - "spike padding-invariance gate kept at the plan's 1e-5 despite a measured 0.0, because SIMD reduction order is architecture-dependent and bit-exactness would assert a property of the host"
metrics:
  duration: ~3h
  completed: 2026-08-08
  tasks: 3
  commits: 5
  tests_added: 53
---

# Phase 1 Plan 03: Remaining Primitives and the Batched-Graph Spike Summary

The last three D-03 primitives landed with per-element finite-difference gradchecks on **both**
sides of their epsilon clamps, and the phase's highest technical unknown — whether gradient flows
end to end at `batch > 1` through a **masked** attention graph — is resolved: **it does.**

## The spike result, stated plainly

**Batch > 1 gradient flow through the repaired `add_mask` broadcast path works.** This is a new
result, not a confirmation. Plan 01-09 established that the masked path was *unreachable* before its
repair (`Tensor::from_vec` asserts on length, so the truncating `.zip()` panicked at every
realistic shape) and that no test in the repository had ever passed a mask to
`scaled_dot_product_attention`. `crates/aprender-core/tests/batched_graph_spike.rs` is the first
execution of that path end to end, and the first backward taken off it.

Measured, on a 2-layer / hidden-16 / 2-head / seq-9 / batch-2 chain with mixed valid lengths
(5 and 9):

| Quantity | Measured |
|---|---|
| Named parameter tensors receiving a **finite** gradient | **36 of 36** |
| ENC-04 component aggregate gradient L2 (7 components) | **3.05e-1 … 8.75e-1** (gate: > 1e-9) |
| `k_proj.bias` max abs gradient (layers 0, 1) | **9.09e-10**, **1.05e-9** |
| `q_proj.bias` gradient L2 (the two-sided comparison) | **3.04e-2** — a ~7-order separation |
| Padding invariance: solo vs padded-batch row 0 | **max delta 0.0, exactly** |
| Mask effect on row 0 (has padding) | **4.86e-1** |
| Mask effect on row 1 (fully valid) | **0.0, exactly** |

### What this does NOT prove — read before relying on it

These limits are not hedging; they are the boundary of the evidence.

1. **Synthetic seeded weights, not real MiniLM.** This is a graph-flow proof. Numerical parity
   against real weights is D-09's job and lands in 01-06/01-08. Nothing here says the *values* are
   right, only that gradient arrives everywhere it must, finite, with the right structure.
2. **Miniature scale.** 2 layers, hidden 16, 2 heads of 8, seq 9, batch 2. Defects that only appear
   at hidden 384 / 6 layers / seq 128 are outside this evidence.
3. **`dropout_p == 0`.** The spike deliberately avoids the unseedable attention-probs dropout. A5 is
   confirmed (below) but not solved.
4. **Only the ABOVE-clamp branch of the new ops is exercised by the spike** (`eps = 1e-12` against
   unit-scale embeddings). The clamped branches are covered by the unit tests in Tasks 1 and 2, not
   by the spike.
5. **Per-row difference is measured with SEPARATE backward passes**, per the plan's revision. A
   single combined backward sums every row's contribution into one tensor and genuinely cannot
   expose them individually; "gradients differ across batch rows" is not measurable any other way.

### Why the spike is not theater — the falsification

Both mutations were applied to `add_mask`, measured, and reverted; `git diff --stat` on
`positional_encoding.rs` afterwards is empty, so the file is byte-identical to its committed state.

| Mutation | Result |
|---|---|
| Broadcast kept, edge severed (`Tensor::from_vec` instead of `scores.add`) | **3 passed / 4 failed** — `encoder.layer.0.attention.q_proj.weight` received NO gradient at batch 2 |
| Full pre-01-09 revert (truncating `.zip()`) | **0 passed / 7 failed** — `Data length 18 doesn't match shape [2, 2, 9, 9]` |

The first mutation is the important one: it keeps every shape correct and only removes the graph
edge, which is exactly the PMAT-913/922 failure mode. Four tests catch it, naming the tensor.

The three tests that survive the severed-edge mutation are the two forward-only ones (padding
invariance, mask-is-not-a-no-op) and the per-row embedding test, which reaches the token table
through the value/residual path. That is the correct behaviour, not a gap — they assert forward
properties, not attention-graph connectivity.

**Two further guards exist specifically because everything above would also pass with the mask
silently dropped.** `batched_graph_masking_changes_the_result_so_the_mask_is_not_a_no_op` asserts
the mask moves row 0's states by more than 1e-4 (measured 4.86e-1) *and* leaves the fully-valid row
1 completely alone (measured exactly 0.0) — the second half is the wrong-axis catcher.

### A5 (SDPA dropout seeding) — CONFIRMED, for plan 01-06

Recorded in the spike's doc comment with the citation, as the plan required:

```text
crates/aprender-core/src/nn/functional.rs:333
pub fn dropout(x: &Tensor, p: f32, training: bool) -> Tensor
```

`scaled_dot_product_attention` (`nn/transformer/mod.rs:66-70`) → `apply_dropout`
(`positional_encoding.rs:515`) → this function, which takes **no seed**. The internal
attention-probs dropout is therefore not seedable today. **Plan 01-06 should implement the seeded
hook as its primary path, not as a contingency.** No further investigation is needed.

## What Was Built

### `l2_normalize_rows` (Task 1) — 18 tests

`Result<Tensor, OpError>`, rank-2 only, explicit epsilon, validation before any arithmetic. The
derivative is piecewise and the implementation says so in three places (op doc, backward struct doc,
test module doc):

```text
n >  eps :  dy/dx = (I - y yᵀ) / n     # d = n depends on x
n <= eps :  dy/dx = I / eps            # d is a CONSTANT — no projection term
```

`L2NormalizeRowsBackward` stores the **raw** per-row norm and re-takes the identical `n > eps`
comparison. It cannot infer the branch from the output: a clamped row and an unclamped row are both
just rows of numbers.

Sums of squares accumulate in **f64**. In f32 a legitimate 1e-20 embedding component squares to
1e-40, which is subnormal and rounds toward zero — the row norm would collapse to 0 and the clamp
decision would be taken on a fabricated value. That is not hypothetical: the test
`l2_normalize_rows_stays_finite_at_extreme_underflow_below_the_clamp` drives exactly this input.

### `cosine_similarity_rows` + `mse_loss` (Task 2) — 28 tests

`CosineSimilarityBackward` returns gradients for **both** inputs and takes the two clamp branches
**independently** — `a`'s branch never changes `b`'s gradient. All four combinations are reachable
and all four are covered.

`mse_loss` returns `Tensor[1]` with an `MseBackward` edge. The target stays a plain `&[f32]`, so it
cannot receive gradient by *construction* rather than by convention — there is no id to record.
Neither `nn/loss.rs` nor `nn/self_supervised.rs` is imported or called (the sole textual match in
`similarity.rs` is the doc comment explaining why not).

## What Makes These Gates Non-Tautological

Every branch claim was **measured** by mutating the implementation and recording which tests turn
red. Mutations were applied, measured, and reverted; none were committed.

| Op | Mutation | Result |
|---|---|---|
| `l2_normalize_rows` | projected form everywhere (`if true`) | 13 passed / **5 failed** |
| `l2_normalize_rows` | clamped form everywhere (`if false`) | 14 passed / **4 failed** |
| `cosine_similarity_rows` | projected form everywhere, both inputs | 25 passed / **3 failed** |
| `cosine_similarity_rows` | `grad_b` zeroed (one-sided edge) | 21 passed / **7 failed** |

Each mutation reddens **exactly** the branch it breaks, and nothing else. The above-clamp tests are
correctly indifferent to the below-clamp mutation and vice versa — which is what makes the pair
evidence rather than coincidence. The three tests that catch projected-everywhere on cosine are
precisely the a-clamped, b-clamped and both-clamped cases.

Beyond the mutations:

- **Every clamped-branch test computes what the WRONG form would have produced and asserts the two
  differ.** Without that, a test could pass because the two forms happen to agree on the chosen
  input, proving nothing about which branch ran.
- **The boundary test uses `eps = 0.25`, a power of two**, so a one-hot row of exactly `eps` has
  `n == eps` bit-exactly in f32 and the branch comparison is not at the mercy of rounding. The two
  sides give visibly different answers there: clamped gives `c₀/eps`, projected gives ~0 because the
  projection annihilates the row direction.
- **A mixed-branch batch** puts row 0 above the clamp and row 1 below it *in one call*, so hoisting
  the clamp decision out of the row loop fails.
- **An orthogonality test** (`<dL/dx, x> == 0` above the clamp) is a structural property of the
  projected form that the clamped form does not have — an entirely independent discriminator from
  the FD checks.
- **The spike's `k_proj.bias` exemption is two-sided.** Asserting near-zero alone would also pass if
  the backward returned zeros for *every* bias. The test additionally asserts `q_proj.bias` and
  `v_proj.bias` carry real gradient (> 1e-4) and that `q_proj.bias` exceeds `k_proj.bias` by 1e4×
  (measured: ~2.3e7×).

## Tasks and Commits

| Task | Gate | Commit | Result |
|------|------|--------|--------|
| 1 — `l2_normalize_rows` | RED | `4667184c8` | **0 passed / 18 failed** |
| 1 — `l2_normalize_rows` | GREEN | `8d423d89e` | 18 passed |
| 2 — cosine + mse | RED | `98e36d057` | **0 passed / 28 failed** |
| 2 — cosine + mse | GREEN | `2fd433dca` | 28 passed |
| 3 — batched spike | — | `3475979ed` | 7 passed |

### TDD Gate Compliance

Both `tdd="true"` tasks show the required `test(...)` → `feat(...)` sequence in git log.

RED was produced by a **stub forward returning a fixed `ShapeOverflow`**, not by a compile error —
the same choice 01-01 and 01-09 made, for the same two reasons: every assertion is proven reachable
and meaningful rather than the file merely failing to compile, and the branch builds at every commit,
which matters because the orchestrator merges this worktree with others. `ShapeOverflow { dims: [] }`
was chosen specifically because **no test expects it**, so the RED count is a clean 0-passed in both
tasks rather than a mix.

Task 3 is `type="auto"` without `tdd="true"` (it writes a test, not a behaviour), so it carries no
RED/GREEN pair. Its falsification evidence is the two `add_mask` mutations above, which serve the
same purpose: proof that the test can fail.

## Verification

Every exit code below was captured with `cmd > file 2>&1; rc=$?`, never read through a pipe
(CLAUDE.md rule 1).

| Check | Result |
|-------|--------|
| `cargo test -p aprender-core --lib l2_normalize_rows` | 18 passed (RED: 0/18) |
| `cargo test -p aprender-core --lib tests_similarity_backward` | 28 passed (RED: 0/28) |
| `cargo test -p aprender-core --lib cosine_similarity_rows` | 19 passed |
| `cargo test -p aprender-core --lib mse_loss` | 22 passed — **see D13**, only 11 are this plan's |
| `cargo test -p aprender-core --lib test_all_backward_names` | 1 passed |
| `cargo test -p aprender-core --test batched_graph_spike` | **7 passed** |
| `cargo test -p aprender-core --lib` (full) | **14109 passed, 2 ignored, 0 failed** (+46) |
| `cargo clippy -p aprender-core --lib --tests` | exit 0, **0 findings in any changed file** |
| `cargo fmt -p aprender-core -- --check` | exit 0 |
| `cargo build -p aprender-core --no-default-features` | exit 0 (all three ops ungated) |
| `cargo check --workspace --exclude aprender-profile` | exit 0 (D5 form) |
| `git check-ignore -v` on all 5 new files | exit 1 for each (not ignored) |

**D2 applied:** the plan's `cargo clippy -p aprender-core -- -D warnings` was NOT used — it exits
101 from pre-existing `aprender-compute` findings regardless of this crate's state. The
`--lib --tests` form documented in D2 was used instead and reports zero findings in
`normalize.rs`, `similarity.rs`, `grad_fn.rs`, `op_error.rs`, `tests_matmul_backward.rs` or
`batched_graph_spike.rs`.

**D1 applied:** `scripts/check_include_files.sh` is vacuous on macOS, so the two new `include!()`
files and the three `#[path]` test files were verified by hand — `git check-ignore -v` exits 1 for
each and `git ls-files` lists all five.

### Source assertions from the acceptance criteria

| Criterion | Evidence |
|---|---|
| `l2_normalize_rows` returns `Result<Tensor, OpError>` | `normalize.rs` signature, not a bare `Tensor` |
| Piecewise formula documented on the backward struct | `grad_fn.rs` `L2NormalizeRowsBackward` doc block |
| Below-eps test names the branch explicitly | `l2_normalize_rows_backward_below_epsilon_clamp_is_identity_over_eps_not_the_projected_form` |
| `similarity.rs` imports nothing from `nn/loss.rs` / `nn/self_supervised.rs` | 1 grep match, in a doc comment saying why not |
| All five backward names registered | `test_all_backward_names` asserts each by name |
| Spike uses `gelu_exact`, not `gelu` | 1 call site, `.gelu_exact()`; zero `.gelu()` matches |
| Spike uses H > 1 and unequal per-row masks | `HEADS = 2`, `batch_a()` lengths 5 and 9 |
| Spike doc cites the A5 dropout signature | doc comment lines quoting `functional.rs:333` |
| `contracts/` untouched | `git diff --stat 124e61f44..HEAD -- contracts/` is empty |

## Deviations from Plan

### 1. [Deliberate departure] Cosine clamps each FACTOR, not the PRODUCT

The plan's `<interfaces>` block specifies `max(||a||, eps) * max(||b||, eps)` with four branch
combinations, and its acceptance criteria require dedicated a-clamped and b-clamped tests — both of
which presuppose per-factor clamping. The contract YAML (`:241`) writes the denominator as
`max(||a|| * ||b||, eps)`. **The plan was followed**; per-factor clamping is also what
`torch.nn.functional.cosine_similarity` implements.

The divergence is confined to the degenerate branch: wherever both norms exceed `eps` — the entire
non-degenerate domain — the two forms are *identical*, and the `|out| <= 1` invariant holds under
both. `contracts/` was deliberately not edited (01-04 owns the contract's next commit, and editing
it from a parallel worktree risks a merge conflict). Logged as **D12** with the recommended
resolution: reword the YAML, do not change the implementation.

### 2. [Rule 2 - Correctness] `OpError::InvalidEpsilon` stores BITS, not an `f32`

The plan's sketch was `InvalidEpsilon { eps }`. Implemented as `InvalidEpsilon { eps_bits: u32 }`
with an `OpError::epsilon()` accessor, for two independent reasons:

1. `OpError` derives `Eq`. An `f32` field forbids that derive for the whole enum, changing the API
   of seven pre-existing variants for the sake of one.
2. **`NaN != NaN` under `PartialEq`.** With an `f32` payload,
   `assert_eq!(err, OpError::InvalidEpsilon { eps: f32::NAN })` would be **unsatisfiable against a
   correct implementation** — for exactly the `NaN` input the variant exists to reject. That is the
   same class of self-defeating assertion the ENC-04 gradient gate and 01-09's `>1e-3` differential
   threshold were both rewritten to avoid. Caught at design time here rather than after the fact.

No existing variant was renamed.

### 3. [Deliberate departure] `mse_loss` requires rank-1 `pred`

The plan's interface declares `pred: [B]`. Enforced literally: a rank-2 `pred` returns
`ShapeMismatch { expected: [0], got }`. This is fail-closed in the phase's style, and it composes:
`cosine_similarity_rows` returns `[B]`, which is what 01-07's `pair_cosine_mse` will feed it. Noted
here because a future caller wanting `[B,1]` will hit it deliberately rather than by accident.

### 4. `mse_loss` scans the TARGET but not `pred`

Following the rule 01-01 documented on `masked_mean_pool`: the target is caller-supplied label data
and therefore untrusted, while `pred` is a computed graph intermediate. Rejecting a non-finite
activation mid-graph would convert a training-dynamics signal into a hard error at an arbitrary
layer. Documented on the function.

### 5. Spike parameter count corrected before commit

The first draft of the spike commit message claimed 42 named parameter tensors. The measured count
is **36** (2 embedding tables + 2 embedding-LayerNorm + 2 × 16 per layer). The commit was amended
before any further work. Recorded because an unchecked count in a commit message is precisely the
"labelling by intent" CLAUDE.md rule 2 warns about, and it was caught only by counting the probe
output rather than by re-reading the arithmetic.

### 6. `l2_normalize_rows` accumulates in f64 (not specified either way)

The plan did not state a precision. f32 accumulation would underflow the sum of squares for
components at 1e-20 and decide the clamp branch on a fabricated norm. Documented in the op body.

## Known Stubs

None. Both RED-phase stubs were replaced wholesale in their GREEN commits — `git grep` over the five
new files for `RED STUB`, `TODO`, `FIXME`, `unimplemented`, `todo!` returns nothing. The temporary
`probe_measurements` test used to capture the spike's margins was removed before the spike commit;
the working tree is clean apart from this SUMMARY and `deferred-items.md`.

## Threat Flags

None. This plan adds no network endpoint, auth path, file-access pattern, schema change, or package
(T-1-SC held: zero installs).

The three `mitigate` dispositions in the plan's register are discharged:

| Threat | Mitigation delivered |
|---|---|
| T-1-01 (DoS via degenerate input) | typed `OpError` on rank / shape / length / zero-dim / epsilon / non-finite input on all three ops; the piecewise backward keeps the clamped branch finite — measured at a 1e-20 row (output 1e-14, gradient `c/eps`, both finite) |
| T-1-05 (silent detachment at batch > 1) | spike asserts finite-per-tensor, non-zero-aggregate-per-component, and Q/K reachability in both layers; the severed-edge mutation fails 4 tests by name |
| T-1-16 (unsatisfiable gate wording) | per-tensor non-zero replaced by aggregate-per-component; the analytically-zero `k_proj.bias` is asserted NEAR zero with the softmax-shift-invariance proof in the message, plus a second-side assertion that the other biases are 4+ orders larger |

## For the Orchestrator

- `STATE.md`, `ROADMAP.md` and `REQUIREMENTS.md` were **not** touched (worktree mode; verified with
  `git diff --stat 124e61f44..HEAD --` on all three: empty).
- `contracts/` was **not** touched.
- ENC-03 and ENC-06 are **not** complete. This plan lands the last three primitives on their path,
  not the requirements: ENC-03 needs the shared encoder path with fixture-verified outputs (01-06 /
  01-04), ENC-06 needs the pair loss against frozen fixtures (01-07). Marking them here would be
  labelling by intent.
- New deferred items **D12** (contract-vs-implementation cosine denominator), **D13**
  (`--lib mse_loss` filter is not scoped to this plan's tests), **D14** (`--nocapture` output is
  swallowed in this environment) appended under a new `## From plan 01-03` section. D1–D11 untouched.
- **For 01-06:** A5 is settled — build the seeded attention-probs dropout hook as the primary path.
  The FFN must call `Tensor::gelu_exact`. `batched_graph_spike.rs` is the shape the real encoder
  should reproduce; if 01-06's encoder diverges from that chain, the divergence is worth stating.
- **For 01-07:** `mse_loss` takes a rank-1 `[B]` prediction and a `&[f32]` target;
  `cosine_similarity_rows` returns exactly that shape. The composition is gradchecked end to end in
  `mse_loss_of_cosine_similarity_rows_matches_central_finite_differences`.
- **For 01-04/01-08:** please resolve D12 in the contract YAML.

## Self-Check: PASSED

All 5 created files verified present on disk. All 5 commits verified present in
`git log 124e61f44..HEAD`. `contracts/`, `STATE.md`, `ROADMAP.md`, `REQUIREMENTS.md` confirmed
unmodified by diffstat. `positional_encoding.rs` confirmed byte-identical to its committed state
after the two falsification mutations (`git diff --stat` empty). Working tree clean apart from this
SUMMARY and `deferred-items.md`.
