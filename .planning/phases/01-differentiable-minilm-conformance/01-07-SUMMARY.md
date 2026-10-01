---
phase: 01-differentiable-minilm-conformance
plan: 07
subsystem: setfit
tags: [setfit, loss, enc-06, freeze-groups, enc-04, d-08, d-20, d-21, d-22, seal, compile-probe]
requires:
  - type:BertSentenceEncoder
  - type:MiniLmTokenizer
  - type:MiniLmImport
  - type:SentenceBatch
  - type:SliceConfig
  - type:VocabRemap
  - type:SetFitError
  - op:cosine_similarity_rows
  - op:mse_loss
  - trait:Module-named_parameters
  - method:BertSentenceEncoder::forward_tokens_per_layer
  - fixture:setfit-slice-apr
provides:
  - fn:setfit::pair_cosine_mse
  - type:SetFitMiniLm
  - type:FreezeGroup
  - method:SetFitMiniLm::from_pretrained_dir
  - method:SetFitMiniLm::from_slice_fixture
  - method:SetFitMiniLm::encode_texts
  - method:SetFitMiniLm::encoder
  - method:SetFitMiniLm::tokenize
  - method:SetFitMiniLm::apply_freeze
  - method:SetFitMiniLm::clear_freeze
  - method:SetFitMiniLm::freeze_policy
  - method:SetFitMiniLm::trainable_parameters_mut
  - method:SetFitMiniLm::frozen_parameters
  - method:BertSentenceEncoder::num_layers
  - method:BertSentenceEncoder::tokenizer_sha256
  - variant:SetFitError::FreezeGroupInvalid
affects:
  - crates/aprender-core/src/setfit/
tech-stack:
  added: []
  patterns:
    - "validate-all -> normalize -> reset -> apply -> store: one code path yields replacement, idempotence and order/duplicate insensitivity instead of three special cases"
    - "freeze proven by an OPTIMIZER STEP over the same method the optimizer will use, two-sided on one parameter, rather than by a configuration round-trip"
    - "the objective is a two-call composition with no arithmetic of its own, asserted BITWISE against the explicit composition so a hand-rolled derivative cannot slip in"
    - "out-of-crate compile probes (negative + positive) as throwaway test targets, run for their exit code and diagnostic, then deleted"
    - "a source-scanning guard's must-match case table assembled at runtime, so a test corpus living inside the scanned directory cannot trip its own gate"
key-files:
  created:
    - crates/aprender-core/src/setfit/loss.rs
    - crates/aprender-core/src/setfit/loss_tests.rs
    - crates/aprender-core/src/setfit/model_tests.rs
  modified:
    - crates/aprender-core/src/setfit/mod.rs
    - crates/aprender-core/src/setfit/encoder.rs
    - crates/aprender-core/src/setfit/import.rs
    - crates/aprender-core/src/setfit/error.rs
decisions:
  - "the D-08 seal produces E0624, not the E0603 the plan predicted: these are pub(crate) METHODS on public types, not private items in a module path. Grepping for E0603 would have returned 0 against a perfectly sealed crate"
  - "the plan's declaration grep returns 8 matches on the pre-existing tree, all string literals inside 01-05/01-06's own seal assertions; scan scoped to non-test sources because a cfg(test) module is not in the library at all"
  - "freezing is load-bearing through EXCLUSION from trainable_parameters_mut, not through requires_grad: mutation D proved a Linear weight still receives gradient and still moves with the flag cleared"
  - "encoder::L2_EPS became pub(crate) so the pair objective clamps with the same constant the encoder normalized with, guarded by an equality test"
  - "SetFitError::FreezeGroupInvalid added rather than reusing BatchInvalid: an out-of-range layer and a zero-match group are configuration errors, and the tests must be able to tell them from a malformed batch"
  - "VocabRemapWire gated on conformance-fixtures like its sole constructor - the one dead-code finding that survived removing the allows is fixed by a targeted cfg, not by reinstating the allow"
metrics:
  duration: ~1h
  completed: 2026-08-08
  tasks: 2
  commits: 4
  tests_added: 56
requirements: [ENC-04, ENC-06]
---

# Phase 1 Plan 07: pair_cosine_mse + SetFitMiniLm Summary

The public Phase 1 API surface is closed. `pair_cosine_mse` is a graph-connected
`[1]` tensor built from exactly two 01-03 primitives, `SetFitMiniLm` is the sole
public way to obtain a paired tokenizer and encoder — **and rustc has been
observed saying so**, four times, from an out-of-crate position — and the
freeze partition is proven by what an optimizer step actually moves rather than
by what the policy reports.

## What Was Built

| File | Role |
|------|------|
| `setfit/loss.rs` | `pair_cosine_mse` (ENC-06) + `PAIR_COSINE_EPS` |
| `setfit/loss_tests.rs` | 22 tests, hand-computed |
| `setfit/mod.rs` | `SetFitMiniLm` (D-08) + `FreezeGroup` (D-22) + the partition |
| `setfit/model_tests.rs` | 34 tests over the real 2-layer/hidden-64 slice |

Plus `SetFitError::FreezeGroupInvalid`, two `BertSentenceEncoder` READ accessors
(`num_layers`, `tokenizer_sha256`), `encoder::L2_EPS` widened to `pub(crate)`,
and **both** `#![allow(dead_code)]` deleted.

## ENC-06: The Objective Is the Composition, Bitwise

`pair_cosine_mse` performs no arithmetic of its own. It validates, then calls
`cosine_similarity_rows` and `mse_loss`. That is asserted rather than described:
`pair_loss_equals_the_composition_of_the_two_primitives_bitwise` compares
`to_bits`, and `pair_loss_gradients_are_bitwise_those_of_the_explicit_composition`
does the same for both input gradients. A reimplementation that agreed to 1e-6
would still fail both — which is the point, because the epsilon-clamp branch
structure and the two backward edges took a dedicated plan (01-03) to get right
and must not acquire a second copy.

Numerics are hand-computed, not fixture-derived. Fixture parity for the pair
loss is 01-08's gate (`loss_pair.json`), and re-deriving the same numbers from
the same JSON one wave early would prove only that two readers of one file
agree. The 3-pair case:

| pair | za | zb | cos | label | sq. err |
|---|---|---|---|---|---|
| 0 | (3, 4) | (3, 4) | 25/(5·5) = 1 | 1.0 | 0 |
| 1 | (1, 0) | (0, 1) | 0/(1·1) = 0 | 0.0 | 0 |
| 2 | (1, 0) | (1, 1) | 1/(1·√2) = 0.70711 | 0.0 | 1/2 |

mean = **1/6**, measured 0.16666667.

**Validation order is asserted, not just written.** Shapes first, then label
length, finiteness, binary membership. The finiteness check is explicit and
precedes membership because `NaN != 0.0 && NaN != 1.0` is true — a
membership-only implementation rejects a NaN with a diagnosis that describes the
wrong problem. `pair_loss_rejects_a_nan_label_naming_non_finiteness_not_membership`
holds that line and mutation A (below) proves it can fail.

**D12 respected.** The implementation clamps each cosine factor independently
(01-03's form, torch's form). `contracts/setfit-encoder-conformance-v1.yaml:241`
still writes the product form. Neither was changed here — 01-08 owns that
reword — and the two coincide wherever both norms exceed eps.

## D-08: The Seal, Demonstrated by the Compiler

Three pieces of evidence, all three run.

### (a) Out-of-crate compile probe — the primary evidence

`crates/aprender-core/tests/zz_seal_probe.rs` called or named all four sealed
constructors from an out-of-crate position. Run with

```
cargo test -p aprender-core --features conformance-fixtures --test zz_seal_probe \
  > /tmp/seal_probe.log 2>&1; rc=$?
```

(status captured directly, never through a pipe — CLAUDE.md rule 1). **`rc=101`**,
four errors, verbatim:

```
error[E0624]: associated function `open` is private
   --> crates/aprender-core/tests/zz_seal_probe.rs:13:45
    |
 13 |     let _ = aprender::setfit::MiniLmImport::open(Path::new("/nonexistent"));
    |                                             ^^^^ private associated function
    |
   ::: crates/aprender-core/src/setfit/import.rs:398:5
    |
398 |     pub(crate) fn open(dir: &Path) -> Result<Self, SetFitError> {
    |     ----------------------------------------------------------- private associated function defined here

error[E0624]: associated function `from_bytes` is private
   --> crates/aprender-core/tests/zz_seal_probe.rs:14:48
    |
 14 |     let _ = aprender::setfit::MiniLmTokenizer::from_bytes(b"{}");
    |                                                ^^^^^^^^^^ private associated function
    |
   ::: crates/aprender-core/src/setfit/tokenizer.rs:211:5
    |
211 |     pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, SetFitError> {
    |     ------------------------------------------------------------------- private associated function defined here

error[E0624]: associated function `open_slice_fixture` is private
   --> crates/aprender-core/tests/zz_seal_probe.rs:20:45
    |
 20 |       let _ = aprender::setfit::MiniLmImport::open_slice_fixture;
    |                                               ^^^^^^^^^^^^^^^^^^ private associated function
    |
   ::: crates/aprender-core/src/setfit/import.rs:457:5
    |
457 | /     pub(crate) fn open_slice_fixture(

error[E0624]: associated function `from_import` is private
   --> crates/aprender-core/tests/zz_seal_probe.rs:21:52
    |
 21 |     let _ = aprender::setfit::BertSentenceEncoder::from_import;
    |                                                    ^^^^^^^^^^^ private associated function
    |
   ::: crates/aprender-core/src/setfit/encoder.rs:193:5
    |
193 |     pub(crate) fn from_import(import: &MiniLmImport, root_seed: u64) -> Result<Self, SetFitError> {
    |     --------------------------------------------------------------------------------------------- private associated function defined here
```

**The plan predicted E0603. It is E0624, and the difference is not cosmetic.**
E0603 is "this item is private" for something reached through a *module path*;
these are `pub(crate)` **methods on public types**, which rustc reports as
E0624. The acceptance criterion was to be measured by `grep -c E0603` — against
a perfectly sealed crate that returns **0**, which reads exactly like a broken
seal. A false negative on the phase's structural gate. Logged as **D41**.

`open_slice_fixture` and `from_import` use the value form rather than the call
form because their arguments (`SliceConfig`, `VocabRemap`, `MiniLmImport`) are
themselves unconstructible out of crate — which is the seal working. E0603/E0624
is raised during name resolution, so naming the function is sufficient.

Probe **deleted**; `git status` carries no `zz_seal_probe.rs`.

### (b) Declaration scan — and the plan's version does not hold

The plan's command, run verbatim at the base commit **before this plan changed
anything relevant**:

```
grep -rnE --include='*.rs' '^[^/]*\bpub fn (from_bytes|open|open_slice_fixture|from_import)\b' \
  crates/aprender-core/src/setfit/
```

returns **8 matches**, every one a string literal inside 01-05's and 01-06's own
seal assertions (`!src.contains("pub fn from_import(")` and siblings). The
`^[^/]*` prefix excludes `//` comments but has no notion of a string literal, and
the plan's nine-row case table has no string-literal row. Logged as **D40**.

Corrected to `--exclude='*_tests.rs'`, which returns **0** (`grep_rc=1`). The
justification is structural, not convenient: a `#[cfg(test)]` module is not
compiled into the library, so it cannot reopen the seal for an out-of-crate
consumer — and the compile probe is immune to the question entirely.

The plan's separate warning about crate-wide widening was **re-measured, not
repeated**: `crates/aprender-core/src/` yields 22 lines — the 14 legitimate
pre-existing declarations the plan lists (apr/mmap/bundle/onnx/gguf/hnsw
readers), exactly, plus the 8 string literals above.

**The case table was re-run, and the guard was turned red for real.** All nine
of the plan's rows are executed by `setfit_model_seal_scan_case_table` on every
test run, so the table is permanent rather than a one-off scratch file. Then the
real mutation: `pub(crate) fn open` -> `pub fn open` in `import.rs`.

| Check | Before mutation | Under mutation | After revert |
|---|---|---|---|
| shell scan (`--exclude='*_tests.rs'`) | `rc=1`, 0 lines | **`rc=0`**, `import.rs:398: pub fn open(...)` | `rc=1`, 0 lines |
| `setfit_model_the_lower_level_constructors_are_still_sealed` | pass | **fail**, naming `import.rs:398` | pass |

A wrinkle worth recording: a test file that ships must-MATCH rows *inside the
scanned directory* trips its own gate. It did, on the first run. The rows are now
assembled at runtime from a `PUB_FN` constant so the source text stays clean.

### (c) Re-export scan

```
grep -rnE --include='*.rs' '^[^/]*pub use .*(from_bytes|open_slice_fixture|from_import)\b' \
  crates/aprender-core/src/
```

`rc=1`, no match. Also permanent as
`setfit_model_no_sealed_constructor_is_re_exported_anywhere_in_the_crate`, which
walks every `.rs` file under `src/`.

### (d) Positive access probe — the mirror

The in-crate test `model.encoder().forward_tokens_per_layer(&model.tokenize(..))`
would compile identically if `forward_tokens_per_layer` were `pub(crate)`, so it
cannot prove out-of-crate reachability (CLAUDE.md rule 2). A throwaway
`crates/aprender-core/tests/zz_access_probe.rs` containing only that path, driven
through `SetFitMiniLm::from_slice_fixture`, exits **`rc=0`, 1 passed**. 01-08's
wave-6 access path is proven in wave 5. Probe deleted; `git status` clean.

## ENC-04: Freeze Proven by What Moves, Not by What Is Reported

**01-06's mutation F is the standard this had to meet.** A dropout site that was
constructed, seeded, mode-aware and reported — but never called — survived 41 of
42 tests. `freeze_policy()` echoing its argument and `frozen_parameters()`
listing the right names have the identical shape: both are satisfied by a policy
nothing consults.

So the load-bearing tests run a real backward off `pair_cosine_mse` and a real
SGD step built from `trainable_parameters_mut()` — **the same method 01-08's
AdamW parameter set comes from**, not a parallel reimplementation of freezing —
and compare `f32::to_bits`.

`setfit_model_the_same_parameter_moves_when_trainable_and_stays_when_frozen` is
the two-sided form and the one that isolates the freeze as the *cause*. Both runs
load the same fixture weights, run in eval mode, take the same objective and the
same step; the only difference is the policy. `embeddings.word_embeddings.weight`
must move in the first and be bitwise unchanged in the second, and the frozen
run's "after" must equal the free run's "before".

`setfit_model_the_pair_objective_backward_reaches_the_encoder_parameters` guards
the other side: at least 30 of 37 parameters must move under an all-trainable
policy. Without it, "the frozen ones did not move" is satisfied by an inert step.

### The finding: `requires_grad(false)` does not freeze anything

Mutation D made `trainable_parameters_mut()` ignore the policy while leaving
`apply_freeze`'s `requires_grad_(false)` in place. If the flag were sufficient,
nothing would change. Instead:

```
`encoder.layer.1.attention.self.query.weight` is frozen but its bits MOVED
across the optimizer step
```

The `Linear` weight still receives a gradient, because the ops consuming it
register the edge on their *input* requiring grad and then produce gradients for
both operands regardless of the weight's own flag. **Exclusion is the mechanism;
the flag is not.** It does protect `embeddings.word_embeddings.weight`, because
`embedding_gather` checks the weight's flag — which is exactly why a single-probe
test would have concluded the opposite. Logged as **D42**, with the consequence
spelled out for 01-08.

### The mapping

One function, `FreezeGroup::name_prefixes`, is the only place a prefix is
written:

| Group | Prefixes | Tensors on the slice |
|---|---|---|
| `Embeddings` | `embeddings.` | 5 |
| `LayerAttention(n)` | `encoder.layer.{n}.attention.self.`, `encoder.layer.{n}.attention.output.dense.` | 8 |
| `LayerFfn(n)` | `encoder.layer.{n}.intermediate.`, `encoder.layer.{n}.output.dense.` | 4 |
| `LayerNorm(n)` | `encoder.layer.{n}.attention.output.LayerNorm.`, `encoder.layer.{n}.output.LayerNorm.` | 4 |

5 + 2 × 16 = **37**, and `setfit_model_the_four_groups_cover_every_named_parameter`
asserts the union is the whole set with no gaps. Every prefix ends with `.`, so
`encoder.layer.1.` cannot address `encoder.layer.10.…` on a future model — a
property with its own test rather than a comment.

`apply_freeze` is **validate-all → normalize → reset → apply → store**. That
ordering is not stylistic: it delivers replacement, idempotence,
order-insensitivity and duplicate-tolerance from one code path, and it is what
makes "no partial application" true rather than approximately true. Mutations E
and F (below) each break exactly one of those and fail exactly one test.

## What Makes These Gates Non-Tautological

Seven mutations were applied, measured and reverted. `git diff` against the
committed state is clean afterwards.

| # | Mutation | Result |
|---|---|---|
| A | label membership checked BEFORE finiteness | **2 failed** — the NaN and Inf diagnoses |
| B | shape-equality check deleted | **1 failed** — the ordering test only; the plain shape test survives because `cosine_similarity_rows` catches it downstream |
| C | loss rebuilt with `Tensor::from_vec` (PMAT-913/922 sever) | **3 failed** — all three graph-connectivity gates |
| D | `trainable_parameters_mut` ignores the policy | **4 failed**, incl. the bitwise optimizer-step gate naming `encoder.layer.1.attention.self.query.weight` |
| E | `requires_grad` reset dropped from `apply_freeze` | **1 failed** — replacement semantics |
| F | validate-as-you-go instead of validate-all-first | **1 failed** — the policy-intact assertion |
| G | `LayerAttention` prefix widened to `attention.` | **3 failed** — the `attention.output.LayerNorm` boundary |
| seal | `pub(crate) fn open` -> `pub fn open` | shell scan **fired**, in-tree guard **fired** |

Mutation B is worth reading twice: the naive "reject a shape mismatch" test does
*not* catch it, because the downstream op rejects the same input. Only the
ordering test — which supplies a shape mismatch AND a bad label and requires the
shape to win — can see it. Without that test the "no compute on mismatched
inputs" claim would be about an unreachable branch.

Beyond the mutations:

- Every rejection test has a positive sibling. `pair_loss_accepts_both_binary_label_values`
  and `setfit_model_tokenize_stamps_this_models_own_tokenizer_hash` exist because
  a validator that rejects everything satisfies every rejection test.
- The foreign-tokenizer batch is a **real** second tokenizer, not a mutated hash
  field: the same vocabulary re-serialized with different byte formatting, so it
  tokenizes identically and differs only in its digest. That isolates the
  identity check as the cause.
- `pair_loss_is_one_for_antiparallel_embeddings_labelled_negative` exists so the
  two "≈1" cases cannot both be satisfied by a constant.
- The naming-drift guard is stated as a property over **every** valid group, not
  one example.

## Tasks and Commits

| Task | Gate | Commit | Result |
|------|------|--------|--------|
| 1 — `pair_cosine_mse` | RED | `2993e7034` | **3 passed / 19 failed** |
| 1 | GREEN | `653b68b5f` | 22 passed |
| 2 — `SetFitMiniLm` + `FreezeGroup` | RED | `f82823a4d` | **7 passed / 27 failed** |
| 2 | GREEN | `527f877bb` | 34 passed |

### TDD Gate Compliance

Both tasks show the required `test(...)` -> `feat(...)` sequence. Neither RED was
a compile error: Task 1's stub returns a fixed `SetFitError::RemapInvalid` that
no test expects, and Task 2's stubs return the same error from both constructors,
an empty prefix set from `name_prefixes`, and a policy-ignoring partition. Every
assertion is therefore proven reachable, and the branch builds at every commit —
which matters because the orchestrator merges this worktree with others.

**RED honesty was checked, not assumed.** Task 1's three RED-passing tests are
the two source assertions and the epsilon-equality check; each was re-run alone
and confirmed exit 0, and none asserts unimplemented behaviour. Task 2's seven
are the case table, the two seal scans, the three `mod.rs` source assertions and
the `FreezeGroup` ordering test — all pure structure. The 27 behavioural tests
were enumerated from the failure list and are exactly the 27 that touch a model
or a prefix set.

One RED-passing test was **strengthened** after measurement:
`setfit_model_every_freeze_prefix_ends_with_a_dot` passed against the empty-prefix
stub because its loop body never ran. A non-emptiness assertion was added and RED
re-taken; it then failed. That is the same class of vacuous-green D31 records for
01-06 and it was caught the same way — by asking why a test passed.

No REFACTOR commit was needed.

## Verification

Every exit code was captured with `cmd > file 2>&1; rc=$?` — never through a pipe
and never from a trailing `echo` (CLAUDE.md rule 1).

| Check | Result |
|-------|--------|
| `cargo test -p aprender-core --lib --features setfit pair_loss_` | 0 — **22 passed** (RED: 3/19) |
| `cargo test -p aprender-core --lib --features conformance-fixtures setfit_model_` | 0 — **34 passed** (RED: 7/27) |
| `cargo test -p aprender-core --lib --features conformance-fixtures` (full) | 0 — **14281 passed**, 2 ignored (+56 over 01-06's 14225) |
| `cargo test -p aprender-core --lib` (default features) | 0 — **14119 passed**, 2 ignored (unchanged; setfit is gated) |
| `cargo test -p aprender-core --features conformance-fixtures --test zz_seal_probe` | **101** — 4 × E0624, 0 × E0603 |
| `cargo test -p aprender-core --features conformance-fixtures --test zz_access_probe` | 0 — 1 passed |
| `cargo check -p aprender-core --no-default-features` | 0 |
| `cargo check -p aprender-core --features setfit` | 0 — **zero** dead-code findings under `src/setfit/` |
| `cargo check -p aprender-core --features conformance-fixtures` | 0 — same |
| `cargo check --workspace --exclude aprender-profile` (D5) | 0 |
| `cargo clippy -p aprender-core --lib --tests --features conformance-fixtures` (D2) | 0 — only the 3 pre-existing `import_tests.rs` findings |
| `cargo fmt -p aprender-core -- --check` | 0 |
| `cargo package -p aprender-core --list` | all 3 new sources present |
| `git check-ignore -v` on all 3 new files | 1 (none ignored) |

**Test-filter scoping (D13/D30) was applied before the fact, not after.**
`grep -rn "fn pair_loss_" crates/` and `grep -rn "fn setfit_model_" crates/`
were run *before* the prefixes were chosen and returned **0** pre-existing
matches each. The two filters select exactly 22 and exactly 34 tests, so the
counts above mean what they say.

**D2 applied**: the plan's `cargo clippy -p aprender-core --features setfit --
-D warnings` was not used — it exits 101 from pre-existing `aprender-compute`
findings regardless of this crate's state.

**D22 applied**: `--all-features` was not attempted (ALSA headers).

**D1 applied**: `scripts/check_include_files.sh` is vacuous on macOS. The three
new files use `#[path]`/`mod` rather than `include!()`, and were verified by hand
with `cargo package --list`, `git ls-files` and `git check-ignore`.

## D32 Closed, Measured

01-06 left both `#![allow(dead_code)]` in place and named 01-07 as the plan that
could remove them. Both are **gone**, and the removal was measured rather than
assumed.

| Build | `aprender-core` warnings at base | After |
|---|---|---|
| `--features setfit` | 2 (incl. `MiniLmTokenizer::from_bytes is never used`) | **1** (the pre-existing `demo/reliable/performance.rs` unreachable expression) |

`from_bytes` acquired a library caller — `SetFitMiniLm::from_pretrained_dir` —
which is the whole point of the bound type. `validate_pooling` and both
`from_json_bytes` constructors likewise.

**One finding did survive the removal, and it is reported rather than
suppressed**, as the plan instructed: `VocabRemapWire is never constructed` under
`--features setfit`. Cause: `VocabRemap::from_json_bytes` is
`#[cfg(feature = "conformance-fixtures")]` but its wire type was not, so a
`setfit`-only build compiled a struct with no constructor. Fixed by gating the
wire type exactly like its sole constructor — a targeted `cfg`, not an allow.
`src/setfit/` now reports zero dead-code findings in both feature builds.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] The plan's D-08 declaration grep does not hold on the current tree**

- **Found during:** Task 2, first run of the in-tree seal guard.
- **Issue:** the command the plan requires to "return no match" returns 8 —
  all string literals inside 01-05/01-06's own seal assertions. Verified with
  the literal shell command at the base commit, and the Rust reimplementation
  reproduces the same 8 lines exactly, which is also the evidence that the Rust
  predicate is faithful to the regex.
- **Fix:** scan scoped to non-test sources (`--exclude='*_tests.rs'`); the
  compile probe is the primary evidence and is unaffected.
- **Files modified:** `crates/aprender-core/src/setfit/model_tests.rs`
- **Commit:** `f82823a4d` / `527f877bb`
- **Logged as D40.**

**2. [Rule 1 - Bug] The plan's expected error code is wrong**

- **Found during:** Task 2, running the compile probe.
- **Issue:** the plan asserts the probe log "contains E0603". It contains
  **E0624** four times and E0603 zero times, because these are `pub(crate)`
  methods on public types rather than private items in a module path. Measuring
  the gate as specified would have reported a false negative on a correctly
  sealed crate.
- **Fix:** verbatim E0624 output recorded above; the correction is logged for
  01-08 as **D41**.
- **Commit:** `527f877bb`

**3. [Rule 2 - Missing Critical] `SetFitError::FreezeGroupInvalid` added**

- **Issue:** the plan requires a *typed* error for an out-of-range layer index
  and for the zero-match naming-drift guard, and requires tests to distinguish
  them from other failures. No existing variant fits: `BatchInvalid` is named
  for batch construction and is already used by `pair_cosine_mse` for label
  errors, so reusing it would make the freeze tests match on message text.
- **Fix:** one additive variant plus its `Display` arm.
- **Files modified:** `crates/aprender-core/src/setfit/error.rs`
- **Commit:** `f82823a4d`

**4. [Rule 3 - Blocking] `encoder.rs` gained two READ accessors**

`num_layers()` — freeze validation cannot reject `LayerAttention(7)` without
knowing the layer count, and `dims` is private. `tokenizer_sha256()` — the plan
requires a test asserting the encoder's hash equals the tokenizer's, and `mod.rs`
is an *ancestor* of `setfit::encoder`, so it cannot see the private field.
Both are read methods; the D-08 seal is about constructors and is untouched.

**5. [Rule 3 - Blocking] `encoder::L2_EPS` widened to `pub(crate)`**

The plan says the objective's epsilon is "the same explicit constant used by the
encoder normalize path". A second `1e-12` literal is exactly the drift the ENC-01
pin exists to prevent, so the constant is shared and
`pair_loss_epsilon_agrees_with_the_encoder_normalize_path` asserts the equality —
the same single-source-of-truth move 01-06 made for the two dropout
probabilities.

**6. [Rule 1 - Bug] `import.rs`: `VocabRemapWire` gated like its constructor**

The one dead-code finding that survived deleting the allows. See D32 above.

### Deliberate Departures

**7. The case table's must-match rows are assembled at runtime**

`model_tests.rs` lives under `src/setfit/`, the directory the declaration scan
walks, so contiguous `pub fn open` literals made the test corpus trip its own
gate — observed on the first run, not anticipated. The rows are built from a
`PUB_FN` constant; the table's meaning is unchanged and the source text stays
clean for both the Rust guard and the shell grep.

**8. `encoder()` and `tokenize()` are the only conformance-gated additions**

No conformance-gated public constructor was added, and
`setfit_model_exposes_exactly_two_public_constructors` asserts the set of
`pub fn from_*` in `mod.rs` is exactly `{from_pretrained_dir, from_slice_fixture}`.

**9. The B5 access-path test uses fixture text, not `"hello"`**

The plan's illustrative `model.tokenize(&["hello"])` fails: canonical id 7592 is
outside the 97-row slice closure and the encoder correctly returns
`VocabOutOfSlice` rather than zero-filling. Measured, then changed to a text from
the frozen pair fixture.

## Known Stubs

None. All four RED stubs were replaced by their GREEN commits. A scan of the
three new files for `RED STUB`, `TODO`, `FIXME`, `unimplemented`, `todo!`
returns nothing, and all seven falsification mutations were reverted (`git diff`
clean against the committed state).

## Threat Flags

None new. The plan's register is discharged:

| Threat | Mitigation delivered |
|---|---|
| T-1-13 (loss identity) | `pair_cosine_mse` never routes through the f32 utilities (source-asserted both spellings); annotated against its own contract equation; graph connectivity two-sided and mutation C fails 3 tests |
| T-1-14 (freeze misaddressing) | structured enum, no string DSL; exact prefix-set assertions incl. the LayerNorm boundary (mutation G fails 3); empty-match guard stated over every valid group |
| T-1-23 (partial freeze on error) | validate-all-then-apply; the rejected call leaves the policy AND every `requires_grad` flag byte-identical (asserted); mutation F fails it |
| T-1-21 (tokenizer/encoder mismatch) | STRUCTURAL: four E0624s recorded from an out-of-crate probe. DATA: `SentenceBatch` read-only out of crate. DEPTH: sha256 equality asserted through the bound type's own encode path with a REAL second tokenizer |
| T-1-29 (a guard that never fired) | case table executed as a test on every run; the real `pub fn open` mutation turned both the shell scan and the in-tree guard red; re-export scan clean and permanent; the compile probe depends on no regex being right |
| T-1-11 (label/group validation) | typed errors on non-finite labels, non-binary labels, length mismatch, shape mismatch and out-of-range layers, each with its own test |
| T-1-SC (third-party installs) | zero packages installed |

## For the Next Plans

- **01-08 must build the AdamW parameter set from
  `SetFitMiniLm::trainable_parameters_mut()`.** Building it from
  `encoder().named_parameters_mut()` and relying on `requires_grad` to skip the
  frozen ones will silently train frozen weights — mutation D measured exactly
  that, and no fixture parity gate can see it (D42).
- **01-08's access path is proven**: `encoder()` -> `forward_tokens_per_layer`
  and `tokenize()` compile and run from an out-of-crate position. Do not add a
  third accessor here or a new method on the encoder.
- **01-08 should assert `E0624`, not `E0603`**, if it re-runs the seal probe
  (D41). More robustly: non-zero exit plus `is private`.
- **01-08 owns the D12 reword** in `contracts/setfit-encoder-conformance-v1.yaml:241`
  (per-factor clamp, not product clamp). The implementation must NOT change.
- **`loss_pair.json` is untouched by this plan.** Its `cosine`
  `[0.876417816, 0.51780647]` and `mse` `0.141698048` are 01-08's parity gate;
  everything here is hand-computed so the two gates are independent.
- **Models load in EVAL mode** (01-06 departure 5, asserted here by
  `setfit_model_from_slice_fixture_returns_an_eval_mode_model`). Call
  `set_training(true)` before a training step and `set_training(false)` before
  any fixture comparison.
- Use the **module-path** filter, not a bare name prefix, when a prefix is a word
  the crate already uses (D30). `pair_loss_` and `setfit_model_` were both
  checked and are collision-free, so `--lib --features setfit pair_loss_` and
  `--lib --features conformance-fixtures setfit_model_` are exact.

## For the Orchestrator

- `STATE.md`, `ROADMAP.md` and `REQUIREMENTS.md` were **not** touched (worktree
  mode) — verified by `git diff --stat <base>..HEAD` on all three: empty.
- `contracts/` was **not** touched — same means, empty.
- `crates/aprender-core/src/models/bert/` was **not** touched (D-01) — same
  means, empty.
- ENC-06 is implemented and structurally proven here; its numerical parity
  against `loss_pair.json` is 01-08's gate. ENC-04's freeze mechanics are
  delivered and proven behaviourally; the controlled-step gate that consumes them
  is 01-08's. The orchestrator owns whether to mark either complete.
- New deferred items appended as **D40-D43** under a new `## From plan 01-07`
  section. D1-D14, D20-D23 and D30-D33 untouched. **D32 is closed** by this plan.

## Self-Check: PASSED

All 3 created source files verified present on disk, tracked by `git ls-files`,
not matched by `git check-ignore` (exit 1), and listed by
`cargo package -p aprender-core --list`. All 4 commits verified as **ancestors of
HEAD** with `git merge-base --is-ancestor` (exit 0 each) rather than by grepping
a `git log` the terminal proxy reformats — the failure 01-04 recorded. Both
throwaway probe files confirmed absent from `git status`. `STATE.md`,
`ROADMAP.md`, `REQUIREMENTS.md`, `contracts/` and `src/models/bert/` confirmed
unmodified by `git diff --stat <base>..HEAD` (empty). Working tree clean apart
from this SUMMARY and `deferred-items.md`.
