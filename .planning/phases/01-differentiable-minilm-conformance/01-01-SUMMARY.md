---
phase: 01-differentiable-minilm-conformance
plan: 01
subsystem: autograd
tags: [autograd, contracts, setfit, embedding, masking, pooling, fail-closed]
requires: []
provides:
  - contract:setfit-encoder-conformance-v1
  - op:embedding_gather
  - op:additive_attention_mask
  - op:masked_mean_pool
  - type:autograd::OpError
  - const:autograd::NEG_MASK
  - gradfn:MaskedMeanPoolBackward
affects:
  - crates/aprender-core/src/autograd/
  - crates/aprender-core/src/generated_contracts.rs
tech-stack:
  added: []
  patterns:
    - "3-step autograd op: forward -> GradFn -> with_graph record"
    - "typed OpError enum, Display + Error, no thiserror (BertLoadError style)"
    - "central finite-difference gradcheck per element (D-04)"
    - "#[provable_contracts_macros::contract] binding (D-27)"
key-files:
  created:
    - contracts/setfit-encoder-conformance-v1.yaml
    - crates/aprender-core/src/autograd/ops/op_error.rs
    - crates/aprender-core/src/autograd/ops/embedding.rs
    - crates/aprender-core/src/autograd/ops/masking.rs
    - crates/aprender-core/src/autograd/ops/pooling.rs
    - crates/aprender-core/src/autograd/ops/tests_embedding_backward.rs
    - crates/aprender-core/src/autograd/ops/tests_masking.rs
    - crates/aprender-core/src/autograd/ops/tests_pooling_backward.rs
  modified:
    - crates/aprender-core/src/generated_contracts.rs
    - crates/aprender-core/src/autograd/mod.rs
    - crates/aprender-core/src/autograd/ops/mod.rs
    - crates/aprender-core/src/autograd/grad_fn.rs
    - crates/aprender-core/src/autograd/tests_matmul_backward.rs
decisions:
  - "tolerance fields OMITTED from the contract (schema Option<f64>) rather than set to 0.0 — pv accepted the omission, so 01-04 adds the table without removing a placeholder"
  - "contract preconditions asserted AFTER the typed guards, not at function entry — a debug_assert! at entry converts a fail-closed error into a debug panic on exactly the hostile inputs the op exists to reject"
  - "mse_loss preconditions/invariants chosen so codegen emits macros byte-identical to loss-functions-v1, making an unavoidable macro-name collision a semantic no-op"
  - "masked_mean_pool does NOT scan `hidden` for non-finite values — it is a computed graph intermediate, not untrusted file input; gradient finiteness is asserted at the ENC-04 gate instead"
  - "ENC-03 and ENC-06 deliberately NOT marked complete — this plan lands 3 primitives, not the requirements"
metrics:
  duration: ~1h
  completed: 2026-08-08
  tasks: 3
  commits: 5
  tests_added: 35
---

# Phase 1 Plan 01: Contract Skeleton and First Three Differentiable Primitives Summary

Three model-agnostic, ungated autograd ops (batched embedding gather, additive attention mask
builder, masked mean pooling) landed with per-element finite-difference gradchecks and fail-closed
typed errors, behind a pv-valid `setfit-encoder-conformance-v1` contract that declares all ten
phase equations up front.

## What Was Built

**`contracts/setfit-encoder-conformance-v1.yaml`** — 10 equations, 9 proof obligations, 14
falsification tests, 4 Kani harnesses, a qa_gate, and `depends_on` references to the nine existing
contracts (none edited). It is the single Phase 1 gate: no later plan needs to touch it except
01-04's tolerance commit.

The two obligations that changed shape versus the pre-review plan:

- **`gelu_exact`** declares the exact erf form `0.5·x·(1 + erf(x/√2))` and states explicitly that
  the tanh approximation is a *different function*, not an acceptable implementation. The
  activation-parity obligation reinforces this with a number: the two forms differ by ~1e-3 near
  |x|≈2, orders of magnitude above f32 round-trip noise, so the gate separates them instead of
  absorbing the gap into tolerance.
- **The ENC-04 gradient gate** is restated as *finite everywhere* + *non-zero AGGREGATE per
  contracted component* + a *two-sided, data-driven* `analytically_zero` exemption list. The prior
  "non-zero gradient on every trainable tensor" wording was unsatisfiable against a correct
  implementation and would have failed the phase on correct code. The obligation carries the proof:
  `attention.self.key.bias` adds the same constant to every key, so for a fixed query the term
  `q_i · b_k` is identical across all keys; softmax is invariant under a constant shift of a row's
  logits; therefore `dL/db_k = 0` exactly. Two-sidedness matters as much as the exemption — an
  unexpectedly *large* gradient on an exempt tensor is a gate failure, not a pass.

**Three ops** in `crates/aprender-core/src/autograd/ops/`, re-exported from `crate::autograd` and
deliberately **not** behind `#[cfg(feature = "setfit")]` (D-03), so the severed-graph debt they
retire is retired for every consumer:

| Op | Shape | Backward |
|----|-------|----------|
| `embedding_gather(weight, ids, batch, seq)` | `[V,H] → [B,S,H]` | existing `EmbeddingBackward`, flattened ids |
| `additive_attention_mask(mask, batch, seq)` | `[B*S] → [B,1,1,S]` | none — constant by contract |
| `masked_mean_pool(hidden, mask)` | `[B,S,H] → [B,H]` | new `MaskedMeanPoolBackward` |

**`OpError`** — eight variants (`OutOfVocabulary`, `ShapeMismatch`, `AllPaddingRow`,
`LengthMismatch`, `ZeroDimension`, `ShapeOverflow`, `NonBinaryMaskValue`, `NonFiniteInput`) with
`Display` + `Error`, no `thiserror`. Plans 01-03 and 01-09 extend it.

## What Makes the Gates Non-Tautological

Every op carries a **per-element central finite-difference** check, not an `is_some` assertion:

- `embedding_gather` gradchecks over the *weight table* (the only differentiable input), on a batch
  that reuses ids 0 and 2 so scatter-add accumulation is exercised inside the gradcheck itself. A
  separate test proves a token appearing twice accumulates exactly 2× the single-occurrence
  gradient — the overwriting-backward bug is invisible on a batch with distinct ids.
- `masked_mean_pool` gradchecks on a **mixed-length** batch (row counts 2 and 3). The
  uniform-denominator bug is invisible on a uniform-length batch and wrong on every real one, so
  three tests attack it directly: two rows of the same constant with different valid counts must
  both pool to that constant; the per-row gradient magnitudes must be 1/2 and 1/3; and padded
  positions must receive *exactly* 0.0.
- `masked_mean_pool_rejects_before_computing_anything` feeds a tensor of `f32::MAX` with an
  all-padding mask and asserts the error is `AllPaddingRow` — proving validation runs before any
  arithmetic could turn the poison into NaN.

Hostile input is proven, not assumed: OOV ids, `id == vocab_size` (the off-by-one boundary),
`checked_mul` overflow at *both* multiply steps, zero batch/seq/hidden/vocab, non-finite weights,
wrong rank, length mismatch, non-binary mask values, and all-padding rows each return the specific
typed variant.

`additive_attention_mask_is_a_constant_not_graph_connected` pins the contract's carve-out, so a
future "fix" cannot quietly attach a bogus grad_fn here and call the masking path graph-connected —
the real obligation lives on `apply_additive_mask` (01-09).

## Tasks and Commits

| Task | Gate | Commit | Result |
|------|------|--------|--------|
| 1 — contract + codegen | — | `5e45c0da4` | pv validate 0 errors / 0 warnings; `cargo check` 0 |
| 2 — gather + mask | RED | `98fd7a0fe` | 11 failed / 0 passed, 11 failed / 1 passed |
| 2 — gather + mask | GREEN | `cbe042161` | 11 ok, 12 ok |
| 3 — pooling | RED | `023ba226f` | 0 passed / 12 failed |
| 3 — pooling | GREEN | `2b6ea3a70` | 12 ok |

### TDD Gate Compliance

Both `tdd="true"` tasks show the required `test(...)` → `feat(...)` sequence in git log.

RED was produced by a **stub body returning a fixed `ShapeMismatch`**, not by a compile error. This
is a stronger RED — every assertion in every test is proven reachable and meaningful rather than the
file merely failing to compile — and it keeps the branch building at every commit, which matters
because the orchestrator merges this worktree with others. One test
(`additive_attention_mask_uses_a_finite_negative_constant`) passed at RED by design: it asserts only
properties of the `NEG_MASK` constant, which is API surface, not stubbed behaviour.

## Verification

| Check | Result |
|-------|--------|
| `pv validate contracts/setfit-encoder-conformance-v1.yaml` | exit 0, 0 errors 0 warnings |
| `cargo check -p aprender-core` | exit 0 |
| `cargo test -p aprender-core --lib embedding_gather` | 11 passed |
| `cargo test -p aprender-core --lib additive_attention_mask` | 12 passed |
| `cargo test -p aprender-core --lib masked_mean_pool` | 12 passed |
| `cargo test -p aprender-core --lib test_all_backward_names` | 1 passed |
| `cargo test -p aprender-core --lib` (full) | **14007 passed, 2 ignored, 0 failed** (baseline before this plan: 13972) |
| `cargo clippy -p aprender-core --lib --tests` | exit 0, **0 findings under `crates/aprender-core/src/autograd/`** |
| `cargo build -p aprender-core --no-default-features` | exit 0 (ops are ungated — no feature leak) |
| `git check-ignore -v` on all four new `include!()` files | exit 1 (not ignored); all tracked by `git ls-files` |
| `cargo test -p aprender-core --test <name> <one-filter>` form | exit 0 — the falsification harness command shape is valid |

**Measurement note (CLAUDE.md rule 1):** every exit code above was captured with
`cmd > file 2>&1; echo $?`, never read through a pipe.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `clippy::naive_bytecount` on the two mask-counting loops**
- **Found during:** Task 3 GREEN
- **Issue:** `mask[..].iter().filter(|&&m| m == 1).count()` on a `&[u8]` trips
  `clippy::naive_bytecount`, which `-D warnings` turns into a hard failure.
- **Fix:** Replaced with `fold(0usize, |acc, &m| acc + usize::from(m == 1))` in both
  `MaskedMeanPoolBackward::backward` and `masked_mean_pool`. This is not merely lint-appeasement:
  the explicit `m == 1` predicate means a stray non-binary value cannot inflate the divisor, and the
  divisor is the one place where being wrong is silent.
- **Files modified:** `crates/aprender-core/src/autograd/grad_fn.rs`, `.../ops/pooling.rs`
- **Commit:** `2b6ea3a70`

### Deliberate Departures

**2. Contract preconditions asserted after validation, not at function entry**

The plan's precedent (`models/qwen2/mod.rs:125`) calls `contract_pre_*!` as the first statement. Done
that way here it would be actively harmful: `contract_pre_embedding_gather!(ids)` expands to
`debug_assert!(ids.len() > 0, ..)`, which **panics in debug and test builds** on precisely the empty
input that `ZeroDimension` exists to reject — converting a fail-closed typed error into a panic on
the hostile-input path. The macros are therefore invoked *after* the typed guards, where the domain
is proven and the assertion cannot fire. Each call site carries a comment explaining why.

**3. `tolerance:` fields omitted entirely rather than set to `0.0`**

The plan asked to prefer omission if the schema permits, else `0.0`, and to record which pv accepted.
The schema types `tolerance` as `Option<f64>` with `#[serde(default)]`, so **omission was accepted**
— `pv validate` reports 0 errors and 0 warnings. Plan 01-04 therefore adds the tolerance table as
pure addition, with no placeholder to remove and no risk of a `0.0` being mistaken for a frozen value.

**4. ENC-03 and ENC-06 NOT marked complete in REQUIREMENTS.md**

The plan frontmatter lists `requirements: [ENC-03, ENC-06]`, and the standard flow would check them
off. That would be a false claim. ENC-03 requires *"one shared Transformer → masked mean pooling →
L2 normalization path with fixture-verified outputs"*; this plan delivers only the pooling
primitive — L2 normalization is 01-03, the encoder path is 01-06, the fixtures are 01-04. ENC-06
requires the tensor-valued pair loss matching frozen fixtures — 01-03 plus 01-07. Marking either
complete here would be labelling by intent rather than by evidence (CLAUDE.md verification rule 2).
`REQUIREMENTS.md` is left untouched; the plans that actually close these requirements should mark
them.

**5. Macro-name collision with `loss-functions-v1`**

`pv query` found that `loss-functions-v1` already declares an equation named `mse_loss`. `pv codegen`
derives macro names from the equation name alone, so both contracts emit `contract_pre_mse_loss!` and
`contract_inv_mse_loss!` into one `#[macro_use]` module, and the later (setfit) definition shadows the
earlier one crate-wide.

Renaming was not an option — the plan's acceptance criteria fix the ten equation names, and 01-03
annotates against `mse_loss`. Instead the preconditions (`predicted.len() > 0` primary, with a second
clause codegen skips as unbound) and the invariants (unicode `≥`/`=` only, no ASCII comparison
operators, no `result.`) were chosen so the **emitted macros are byte-identical** to the
`loss-functions-v1` ones. Verified at `generated_contracts.rs:20334` and `:29825`. Shadowing is a
semantic no-op. A YAML comment on the equation records the constraint for future editors, and the
underlying `pv` limitation is logged as deferred item D4.

**6. `masked_mean_pool` does not scan `hidden` for non-finite values**

`embedding_gather` does scan its weight table, because that table is loaded from an untrusted model
file. `hidden` is a computed graph intermediate produced by the encoder itself. Rejecting a
non-finite activation mid-graph would convert a training-dynamics signal into a hard error at an
arbitrary layer, and the threat register (T-1-01) does not list finiteness among the pooling
mitigations. Gradient finiteness is asserted where it is meaningful and total: the ENC-04 gate.
Documented on the function.

### Collateral: `generated_contracts.rs` regeneration

Running the plan's mandated regeneration produced a ~31k-line diff *before* this plan's contract was
counted — 2415 macros at base versus 2903 after, with 18 dropped. `pv codegen` had simply not been
re-run after a long run of contract edits.

Verified safe rather than assumed safe: all 18 dropped macros were confirmed to have **zero call
sites**; `generated_contracts` is a private module so nothing outside the crate can reference them;
`cargo check -p aprender-core` exits 0; and the full lib suite was run *before* any of this plan's
tests existed and came back **13972 passed / 0 failed**, so the regeneration alone broke nothing.
Logged as deferred item D3 — nothing in the repo currently detects this drift.

## Known Stubs

None. Both RED-phase stubs were replaced in their GREEN commits; a scan of the new files for
`RED stub` / `TODO` / `FIXME` / `unimplemented` / `todo!` returns nothing.

## Threat Flags

None. This plan adds no network endpoint, auth path, file-access pattern, or schema change. The one
trust boundary it touches — caller-supplied ids, masks and shapes at the public op entry points — is
already in the plan's threat register as T-1-01, and is mitigated exactly as specified: typed
`OpError` on OOV / length mismatch / all-padding row / zero dimension / non-binary mask value,
`checked_mul` before allocation, and no `assert!`/`panic!`/`unwrap()` on any public-reachable path.

## For the Orchestrator

- `STATE.md` and `ROADMAP.md` were **not** touched (worktree mode).
- `REQUIREMENTS.md` was **not** touched — see deviation 4 above; this is deliberate, not an omission.
- New out-of-scope findings are in
  `.planning/phases/01-differentiable-minilm-conformance/deferred-items.md` (D1–D4). D1 is the most
  actionable: `scripts/check_include_files.sh` uses `grep -P` and reports "0 include!() files" on
  macOS, so the CB-510 guard cannot fail there.

## Self-Check: PASSED

All 8 created files verified present on disk. All 5 commits verified present in
`git log e6dce92a0..HEAD`. Working tree clean apart from this SUMMARY and `deferred-items.md`.
