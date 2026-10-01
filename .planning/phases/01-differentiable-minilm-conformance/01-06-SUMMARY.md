---
phase: 01-differentiable-minilm-conformance
plan: 06
subsystem: setfit
tags: [setfit, encoder, enc-03, enc-05, d-01, d-08, d-15, d-18, dropout, seeded-rng, a5]
requires:
  - type:MiniLmImport
  - type:SentenceBatch
  - type:VocabRemap
  - type:SetFitError
  - op:embedding_gather
  - op:additive_attention_mask
  - op:apply_additive_mask
  - op:masked_mean_pool
  - op:l2_normalize_rows
  - op:gelu_exact
  - trait:Module-named_parameters
  - fixture:setfit-slice-apr
  - fixture:setfit-gradients-parameter-order
  - fixture:setfit-forward-per-layer
provides:
  - type:BertSentenceEncoder
  - method:BertSentenceEncoder::forward_tokens
  - method:BertSentenceEncoder::forward_tokens_per_layer
  - method:BertSentenceEncoder::encode
  - method:BertSentenceEncoder::max_seq
  - fn:MultiHeadAttention::with_attention_dropout_seed
  - fn:MultiHeadAttention::attention_dropout_seed
  - fn:MultiHeadAttention::dropout_p
  - fn:nn::transformer::apply_dropout_seeded
  - amendment:A5-seeded-attention-probs-dropout
affects:
  - crates/aprender-core/src/setfit/
  - crates/aprender-core/src/nn/transformer/
tech-stack:
  added: []
  patterns:
    - "one shared private forward returning (embeddings_out, layer_outputs); both public entry points delegate, proven by a to_bits equality test in eval mode"
    - "seeded-dropout hook added as a SEPARATE entry point rather than a seventh parameter, so ten existing call sites stay source- and bit-identical"
    - "per-call counter mixed into a site seed, so a reproducible stream still ADVANCES and does not replay one fixed mask"
    - "dropout-site introspection built by walking the ACTIVE modules, plus a per-site test that turns one site on against an eval-mode model to prove it is applied"
    - "HF name mapping by NAME, so a reordering upstream surfaces as an out-of-order parameter list rather than mislabelled tensors"
key-files:
  created:
    - crates/aprender-core/src/setfit/encoder.rs
    - crates/aprender-core/src/setfit/encoder_tests.rs
    - crates/aprender-core/src/nn/transformer/tests_seeded_attention_dropout.rs
  modified:
    - crates/aprender-core/src/setfit/mod.rs
    - crates/aprender-core/src/setfit/import.rs
    - crates/aprender-core/src/nn/transformer/mod.rs
    - crates/aprender-core/src/nn/transformer/positional_encoding.rs
    - crates/aprender-core/src/nn/transformer/tests.rs
decisions:
  - "the seeded SDPA hook is a SEPARATE 7-arg entry point with the 6-arg name delegating to it; adding the parameter in place needed ten call-site edits in attention_gqa.rs and tests_attention_contract.rs, and deriving the dropout from forward_qkv would have meant re-implementing attn_weights @ V"
  - "a per-call AtomicU64 counter is mixed into the site seed, and advances ONLY when the dropout actually fires, so eval passes between training steps cannot shift a reproducible run"
  - "from_import returns an EVAL-mode encoder, matching HF from_pretrained, which is also the mode the fixtures were generated in (D-16)"
  - "train()/eval() delegate to set_training rather than staying leaf-local: on a module whose point is dropout placement, a leaf-local eval() would silently make inference stochastic"
  - "the two pinned dropout probabilities became pub(crate) in import.rs so the encoder cannot hold a second unlinked copy of a pinned number; equality asserted at f32"
  - "added encoder_mode_every_site_is_actually_applied_in_the_forward (Rule 2): mutation F proved neither the site-count test nor the recursion test can see a site that is never called"
metrics:
  duration: ~3h
  completed: 2026-08-08
  tasks: 2
  commits: 4
  tests_added: 52
requirements: [ENC-03, ENC-05]
---

# Phase 1 Plan 06: BertSentenceEncoder Summary

The ENC-03 path exists and runs on the **real configuration**: one shared
`forward_layers` for both public entry points, HF-dotted names proven equal in
order to the frozen `parameter_order`, a fail-closed boundary inherited by every
caller, and — the last unknown of the phase — the attention-probs dropout is now
seedable, so ENC-05's mode contract is reproducible at all four HF-verified
sites.

## What Was Built

| File | Role |
|------|------|
| `setfit/encoder.rs` | `BertSentenceEncoder`: `forward_layers`, `forward_tokens`, `forward_tokens_per_layer`, `encode`, `max_seq`, `dropout_sites`, `Module` impl |
| `setfit/encoder_tests.rs` | 42 tests over the real 2-layer/hidden-64 slice |
| `nn/transformer/tests_seeded_attention_dropout.rs` | 10 ungated tests for the A5 hook |

Plus the additive `MultiHeadAttention` seed hook, `apply_dropout_seeded`, and
two `pub(crate)` visibility lines in `import.rs`.

## The Spike's Scope Limits, and What This Plan Actually Extended

01-03's batched-graph spike passed with five stated limits. Three of them are
discharged here and two are not; saying which is which matters more than the
green count.

| 01-03 limit | Status after 01-06 |
|---|---|
| Synthetic seeded weights, not real MiniLM | **Discharged.** Every test here runs on the frozen slice APR — real pinned weights, index-sliced. |
| Miniature scale (2 layers / hidden 16 / seq 9) | **Partly.** 2 layers / hidden 64 / seq 20 on the real slice. Hidden **384** and **6** layers are still unexercised; that is the full-pin path and it lands with 01-08's D-10 gated suite. |
| `dropout_p == 0` | **Discharged.** All four sites are placed at `p = 0.1` and each is proven APPLIED, not merely constructed. |
| Only the above-clamp branch of the new ops | Unchanged here. The clamped branches remain covered by 01-03's unit tests. |
| Per-row differences measured with separate backward passes | Unchanged; this plan takes one backward over the whole batch. |

Two things this plan is the **first** to run: masked attention at real shapes
with a real mask (01-09 established the masked path had been unreachable and
untested before its repair), and the attention-probs dropout with a seed.

## ENC-03: One Forward, Proven Structurally

`forward_layers` validates once, gathers the three embedding tables, and runs
the layer loop, pushing each output as it goes. `forward_tokens` pops the last;
`forward_tokens_per_layer` returns the pair. Collecting the intermediates is
**unconditional**, so the conformance build and the production build compute
identically — only the accessor is `cfg`-gated.

The proof is not the design note, it is
`encoder_forward_tokens_is_bitwise_identical_to_the_last_per_layer_output`,
which compares `f32::to_bits` elementwise **in eval mode**. Eval is mandatory
and the test says why: these are two separate calls, so in train mode the seeded
RNG advances between them and the comparison would fail for a reason unrelated
to divergence. Weakening it to a tolerance would destroy the one gate that makes
"one implementation" structural. A source assertion covers the other half —
`for layer in &self.layers {` occurs exactly once, inside `forward_layers`.

The FFN calls `gelu_exact`; a source assertion forbids `.gelu()` in the file.

## D-18: Names Proven In Order, Against an Array

`named_parameters()` returns the 37 HF dotted names verbatim, pooler excluded,
compared by `Vec<String>` equality against `gradients.json.parameter_order` — an
ordered **array**, not object keys, which carry no ordering guarantee.

The `MultiHeadAttention` local-to-HF mapping (`q_proj.weight` ->
`attention.self.query.weight`, `out_proj.*` -> `attention.output.dense.*`) is by
NAME, and an unrecognised local name is passed through verbatim. So a reordering
or renaming inside `MultiHeadAttention` fails the order gate loudly instead of
producing a plausible-looking but mislabelled tensor list.

## D-08 and the Boundary

`from_import` is `pub(crate)` — asserted by source grep, both directions
(`pub(crate) fn from_import(` present, bare `pub fn from_import(` absent). The
`tokenizer_sha256` check stays as defense in depth and lives inside
`forward_layers`, so **both** public entry points inherit it; a test asserts the
foreign-tokenizer rejection through `forward_tokens_per_layer` too.

`max_seq()` is `min(MAX_SEQUENCE_LENGTH, max_position_embeddings)` = **64** on
the slice, so a 100-token batch is rejected even though it is well under 256.
The test asserts the probe sits strictly between the two bounds, so it cannot
pass against a hardcoded `<= 256`.

Full matrix, each with its own typed error: foreign tokenizer, id/type/mask
length disagreement, zero batch or seq, oversize sequence, non-binary mask
value, all-padding row, token-type id out of range, canonical id outside the
slice closure, id beyond vocabulary on the full pin. `encoder_accepts_a_single_
sentence_batch` holds the other side of the line — a gate that rejects
everything would satisfy the matrix.

## A5 Discharged

`nn::functional::dropout(x, p, training)` has no seed parameter, so the
attention-probs site was not reproducible. The hook:

```rust
// positional_encoding.rs
pub(super) fn apply_dropout_seeded(x: &Tensor, p: f32, seed: Option<u64>) -> Tensor
```

`None` delegates to `apply_dropout` verbatim; `Some(seed)` routes through
`Dropout::with_seed` — the crate's audited seeded path, same PMAT-922
constant-mask `mul`, so the autograd edge is identical (asserted by
`mha_seeded_dropout_keeps_the_autograd_edge`).

**Shape chosen, as the plan asked be recorded.**
`scaled_dot_product_attention` keeps its 6-argument signature and delegates to a
new `scaled_dot_product_attention_seeded`. The alternatives were worse: adding
the parameter in place required editing ten call sites across `attention_gqa.rs`
and `tests_attention_contract.rs` — files this change has no business touching —
and deriving the dropout from `forward_qkv` would have meant re-implementing the
`attn_weights @ V` product, which the plan forbids. The split touches
`transformer/mod.rs` only.

**The stream advances.** A per-call `AtomicU64` is mixed into the seed, because
a seeded site replaying one fixed mask every step is reproducible and no longer
dropout. The counter advances **only when the dropout actually fires**, so
running inference between training steps cannot shift a reproducible run —
`encoder_mode_eval_passes_do_not_consume_the_dropout_stream` asserts exactly
that.

**Existing callers are unchanged, asserted not argued.** At `dropout_p == 0.0`
(the `MultiHeadAttention::new` default, and what `GroupedQueryAttention` and the
attention contract tests use) the dropout branch is not entered, so a seeded and
an unseeded module are compared bit-for-bit on identical weights and must agree.
An unseeded module with dropout on is separately asserted to remain
non-deterministic. `cargo test --lib transformer`: **187 passed** (177 before,
+10 new), zero regressions.

## ENC-05

`set_training` recurses into the embeddings dropout, the `MultiHeadAttention`
(which owns site 2 and propagates into its four projections), both per-layer
dropouts and the norms. `train()`/`eval()` delegate to it — the crate convention
leaves them leaf-local, but on a module whose entire point is dropout placement
a leaf-local `eval()` would make every "inference" run stochastic with no error
raised (logged as D33).

The gate reads each site's `training()` flag **directly** rather than inferring
from behaviour, and mutation E proves it fires (below). `snapshot_named` from
01-02 is reused — not copied — for the train -> eval -> train byte-identity
proof over all 37 tensors. A separate test asserts no seed or RNG state appears
in `named_parameters` (Pitfall 7).

`dropout_sites()` returns 7 names for the slice (`1 + 3 * layers`), built by
walking the **active modules** — a site that was never wired cannot appear.

## What Makes These Gates Non-Tautological

Six mutations were applied, measured, and reverted. `git diff` on the touched
files after reverting is clean.

| # | Mutation | Result |
|---|---|---|
| A | `forward_tokens` returns `embeddings_out` instead of the last layer | **4 failed** — the bitwise anti-divergence test plus all three gradient tests |
| B | embedding sum rebuilt with `Tensor::from_vec` (PMAT-913/922 sever) | **3 failed**, naming `embeddings.word_embeddings.weight` |
| C | `q_proj` left untranslated in `hf_attention_name` | **2 failed**, incl. the `parameter_order` gate |
| D | tokenizer identity check disabled inside `forward_layers` | **1 failed** — the through-BOTH-entry-points test |
| E | `set_training` stops recursing into `MultiHeadAttention` | **5 failed** — the per-site flag assertion and both eval-determinism tests |
| F | site 4 (FFN output dropout) constructed but never applied | **1 failed**, naming `encoder.layer.0.output.dropout` |

**Mutation F is the finding.** Before the gate it motivated,
`encoder_mode_every_site_is_actually_applied_in_the_forward`, mutation F passed
**41 of 42 tests**. `dropout_sites()` proves a site exists and is active; the
recursion test proves it follows the mode; neither can see a site that is never
called — and the fixture parity gates cannot either, because the fixtures were
generated in eval mode where dropout is inert. The new test turns exactly one
site on against an otherwise eval-mode encoder and requires the output to move.
Added under Rule 2.

Beyond the mutations:

- The seeded-determinism tests are **two-sided**: same seed must agree bitwise,
  a different seed must disagree, and train mode must differ from eval mode.
  Any one alone is satisfied by an encoder whose dropout never fires.
- The key-bias exemption is two-sided as in 01-03: the two analytically-zero
  biases named by `gradients.json.analytically_zero` are asserted near zero
  **and** the query/value biases are asserted `> 1e-4` with `q > 1e3 * k`.
- The remap test asserts the fixture batch actually carries canonical ids above
  the 97-row slice vocabulary, so a successful forward is itself the proof that
  the remap ran.

## Tasks and Commits

| Task | Gate | Commit | Result |
|------|------|--------|--------|
| 1 — encoder, one forward, names, boundary | RED | `5e01a8fdc` | **20 failed / 4 passed** |
| 1 | GREEN | `e87ca4796` | 25 passed |
| 2 — seeded dropout, encode, mode contract | RED | `2793d9495` | **4 failed / 39 passed** |
| 2 | GREEN | `ad5c0c37d` | 42 + 10 passed |

### TDD Gate Compliance

Both tasks show the required `test(...)` -> `feat(...)` sequence. Neither RED
was a compile error: Task 1's stub returns a fixed
`OpError::ShapeOverflow { dims: [] }` that no test expects, and Task 2's stub
accepts the seed and ignores it — so every assertion is proven reachable and the
branch builds at every commit, which matters because the orchestrator merges
this worktree with others.

**RED honesty was checked, not assumed.** Task 1's four RED-passing tests are
all source/shape assertions the stub deliberately satisfies (the D-08 seal, the
two forbidden imports, the conformance-gated `pub` declaration); none asserts
unimplemented behaviour. Task 2's 39 RED-passing tests were investigated
individually: the `encode()` pipeline and the ENC-05 mode assertions landed in
Task 1, and the two "stream advances" tests are satisfied by the ambient RNG —
they exist to catch the opposite defect, which only becomes reachable in GREEN.

**Task 2's RED was re-measured.** The first MHA determinism tests compared two
freshly constructed `MultiHeadAttention` instances, whose `Linear::new` weights
are random — they were measuring the initialiser, not the hook, and one of them
passed for the wrong reason. Both were rewritten against a `deterministic_mha`
helper and RED was taken again: the same two MHA tests fail. The invalid
measurement is logged as **D31**.

No REFACTOR commit was needed.

## Verification

Every exit code was captured with `cmd > file 2>&1; rc=$?` — never through a
pipe and never from a trailing `echo` (CLAUDE.md rule 1).

| Check | Result |
|-------|--------|
| `cargo test -p aprender-core --lib --features conformance-fixtures setfit::encoder::encoder_tests` | 0 — **42 passed** |
| `cargo test -p aprender-core --lib mha_seeded_dropout_` | 0 — **10 passed** |
| `cargo test -p aprender-core --lib transformer` | 0 — **187 passed** (177 + 10) |
| `cargo test -p aprender-core --lib --features conformance-fixtures` (full) | 0 — **14225 passed**, 2 ignored (+52) |
| `cargo test -p aprender-core --lib` (full, default features) | 0 — **14119 passed**, 2 ignored (+10) |
| `cargo check -p aprender-core --no-default-features` | 0 (the MHA hook is ungated) |
| `cargo check -p aprender-core --features setfit` | 0 |
| `cargo check --workspace --exclude aprender-profile` (D5) | 0 |
| `cargo clippy -p aprender-core --lib --tests --features conformance-fixtures` (D2) | 0 — **zero findings** in any file this plan created or modified |
| `cargo clippy -p aprender-core --lib --tests --features setfit` | 0 — only the 3 pre-existing `import_tests.rs` `.err().expect()` findings |
| `cargo fmt -p aprender-core -- --check` | 0 |
| `cargo package -p aprender-core --list` | all 3 new sources present |
| `git check-ignore -v` on all 3 new files | 1 each (not ignored); all 3 listed by `git ls-files` |
| `git diff <base>..HEAD -- crates/aprender-core/src/models/bert/` | **empty** (D-01 respected) |
| `git diff <base>..HEAD -- .../positional_encoding.rs` | **31 insertions, 0 deletions** — `add_mask` byte-identical, 01-09's repair untouched |

**D2 applied**: the plan's `cargo clippy -p aprender-core --features setfit --
-D warnings` was not used — it exits 101 from pre-existing `aprender-compute`
findings regardless of this crate's state. The `--lib --tests` form documented
in D2 was used and reports zero findings in `encoder.rs`, `encoder_tests.rs`,
`tests_seeded_attention_dropout.rs`, `transformer/mod.rs`,
`positional_encoding.rs` or `import.rs`.

**D1 applied**: `scripts/check_include_files.sh` is vacuous on macOS. The three
new files use `#[path]` / `mod` declarations rather than `include!()`, and were
verified directly with `cargo package --list`, `git ls-files` and
`git check-ignore`.

### Acceptance-criteria source assertions

| Criterion | Evidence |
|---|---|
| `from_import` is `pub(crate)`, no bare `pub fn` | `encoder_from_import_is_sealed_to_pub_crate` (both directions) |
| `pub fn forward_tokens_per_layer` occurs exactly once, conformance-gated | `encoder_forward_tokens_per_layer_is_public_and_conformance_gated` |
| Exactly ONE layer loop, inside `forward_layers` | `encoder_has_exactly_one_layer_loop` (count == 1 **and** position after `fn forward_layers`) |
| No bespoke `OpError` -> `SetFitError` conversion | `encoder_defines_no_competing_op_error_conversion` |
| FFN uses `gelu_exact`, no `.gelu()` | `encoder_uses_the_exact_erf_gelu` |
| No import of the asserting BERT embeddings path | `encoder_does_not_import_the_asserting_bert_embeddings_path` |
| `encode` adds no third forward path | `encoder_encode_adds_no_third_forward_path` (body contains `self.forward_tokens(`, not `self.layers`) |
| No tolerance literal referencing a fixture family | none present — parity is 01-08's, after D-14 |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added a per-site "is it actually applied" gate**

- **Found during:** Task 2, while falsifying the site-count test.
- **Issue:** `dropout_sites()` proves a site exists and is active, and the
  recursion test proves it follows the mode. Neither can detect a site that is
  constructed, mode-aware, and **never called** in the forward. Mutation F —
  removing the FFN output dropout from the layer loop — passed 41 of 42 tests.
  The fixture parity gates cannot catch it either, because the fixtures were
  generated in eval mode where dropout is inert (D-16).
- **Fix:** `encoder_mode_every_site_is_actually_applied_in_the_forward` turns
  exactly one site on against an otherwise eval-mode encoder and requires the
  output to move, for all 7 sites.
- **Files modified:** `crates/aprender-core/src/setfit/encoder_tests.rs`
- **Commit:** `ad5c0c37d`

**2. [Rule 1 - Bug] The first MHA determinism tests measured the weight initialiser**

- **Found during:** Task 2 GREEN.
- **Issue:** `MultiHeadAttention::new` builds four `Linear::new` projections,
  which draw **random** weights. Comparing two freshly built modules therefore
  compares two different models. `mha_seeded_dropout_same_seed_...` failed with
  `-0.1428478 vs -0.3084763` — far larger than any dropout mask — and its
  sibling `..._different_seeds_...` **passed for the wrong reason**. The Task 2
  RED measurement taken against that pair was invalid.
- **Fix:** a `deterministic_mha(dropout_p, seed)` helper that installs fixed
  weights before comparing, and **RED re-measured** against the corrected
  tests: the same two MHA tests fail.
- **Files modified:** `crates/aprender-core/src/nn/transformer/tests_seeded_attention_dropout.rs`
- **Commit:** `ad5c0c37d`
- **Generalisation logged as D31.**

**3. [Rule 3 - Blocking] `import.rs` touched beyond the plan's file list**

- **Issue:** the encoder needs the pinned dropout probability. Hardcoding `0.1`
  a second time creates exactly the drift the ENC-01 pin exists to prevent, and
  the two constants were module-private.
- **Fix:** `PINNED_HIDDEN_DROPOUT_PROB` and `PINNED_ATTENTION_DROPOUT_PROB`
  became `pub(crate)`, and
  `encoder_dropout_probability_agrees_with_the_enc01_pin` asserts equality at
  **f32** — the precision the model computes in; an f64 comparison would reject
  the pin's own value, the same narrowing rule 01-05 applied to
  `layer_norm_eps`. The only other change to `import.rs` is a comment recording
  the dead-code measurement below.
- **Files modified:** `crates/aprender-core/src/setfit/import.rs` (5 insertions, 2 deletions)
- **Commit:** `e87ca4796`

**4. [Rule 3 - Blocking] `nn/transformer/tests.rs` touched beyond the plan's file list**

One line, registering the new `#[path]` test module — the established pattern
01-09 used for `tests_attention_mask_broadcast.rs`.

### Deliberate Departures

**5. `from_import` returns an EVAL-mode encoder**

Not specified by the plan. HuggingFace `from_pretrained` calls `model.eval()`
before returning, and eval is the mode the frozen fixtures were generated in
(D-16), so this is the HF-faithful default and the one 01-08 needs. Training
callers flip it with `set_training(true)`. **01-07 must not assume train mode.**

**6. `train()` / `eval()` delegate to `set_training`**

The crate convention (01-02, D-17) leaves them leaf-local. Followed literally,
`encoder.eval()` would return an encoder with dropout still active — stochastic
"inference" with no error raised. Both spellings route through the one channel
here and a test asserts it. The crate-wide inconsistency is logged as **D33**;
this is one module opting out of it, not a fix.

**7. `#![allow(dead_code)]` added to `encoder.rs`; `import.rs`'s was NOT removed**

01-05 recorded that its allow "stops being needed the moment 01-06 wires the
encoder". Measured after wiring: the surface fell from ~15 findings to exactly
**three** (`VocabRemap::from_json_bytes`, `SliceConfig::from_json_bytes`,
`validate_pooling`), all still reachable only from tests and 01-07, so it is
still load-bearing. `encoder.rs` needs the same allow for the same structural
reason — `from_import` is `pub(crate)` under the D-08 seal with no non-test
caller — and a **targeted** version was tried first and measured to be
whack-a-mole: silencing `install_projection` and `EMBEDDINGS_DROPOUT_SITE`
moved the finding to `site_seed`. Logged as **D32**; 01-07 can delete both.

**8. `install_projection` is a free function, not an inline loop**

`q_proj_mut()`, `k_proj_mut()` and `v_proj_mut()` each borrow the whole
`MultiHeadAttention` mutably, so they cannot be collected into one iterable.

## Known Stubs

None. Both RED stubs were replaced by their GREEN commits. A scan of the three
new files for `RED STUB`, `RED RE-MEASUREMENT`, `TODO`, `FIXME`,
`unimplemented`, `todo!` returns nothing, and the six falsification mutations
were all reverted (`git diff` clean, and `positional_encoding.rs` vs the plan
base is 31 insertions / 0 deletions).

## Threat Flags

None new. The plan's register is discharged:

| Threat | Mitigation delivered |
|---|---|
| T-1-11 (DoS on hostile batch) | one boundary validation inside `forward_layers`, inherited by both entry points; `max_seq = min(256, max_position_embeddings)`; no `assert!`, `unwrap()` or panic on any reachable path |
| T-1-21 (batch/encoder tokenizer mismatch) | `from_import` `pub(crate)` (source-asserted both directions) alongside 01-05's sealed constructors and read-only `SentenceBatch`; sha256 equality retained as defense in depth and proven to run in the shared path by mutation D |
| T-1-12 (silent graph detachment) | real-slice mixed-length grad-flow test per named parameter with per-component aggregates; per-layer intermediates asserted graph-connected; mutation B fails 3 tests by tensor name |
| T-1-28 (divergent per-layer vs final paths) | one `forward_layers`; bitwise `to_bits` equality between the entry points; source assertion that the layer loop occurs once; mutation A fails 4 tests |
| T-1-19 (wrong activation) | `gelu_exact` called, `.gelu()` source-forbidden in the file |
| T-1-SC (third-party installs) | zero packages installed |

## For the Next Plans

- **01-07** builds `SetFitMiniLm` and is the first non-test caller of
  `BertSentenceEncoder::from_import` and of 01-05's sealed constructors. It
  should **delete `#![allow(dead_code)]` from both `setfit/encoder.rs` and
  `setfit/import.rs`** and re-run clippy rather than assume they became
  unnecessary (D32).
- **01-07** must expose a conformance-gated `SetFitMiniLm::encoder()` returning
  `&BertSentenceEncoder`, because that is how 01-08 reaches
  `forward_tokens_per_layer`.
- **01-07 must not assume train mode.** `from_import` returns an eval-mode
  encoder (departure 5).
- **01-08** consumes `forward_tokens_per_layer(&batch) -> (embeddings_out,
  Vec<layer_outputs>)`, which maps directly onto `forward_per_layer.json`'s
  `embeddings_out` / `layer_outputs` / `final_tokens` (`layer_outputs.last()` IS
  `final_tokens`). It exists, is `pub`, and is gated on `conformance-fixtures` —
  01-08 does not need to add it.
- **01-08's fixtures were generated in eval mode**, so call `set_training(false)`
  before any parity comparison. Note that the eval-mode fixtures cannot detect a
  missing dropout site; that gate lives here (deviation 1).
- Use the **module-path** test filter, not `encoder_` (D30): the name-prefix
  form selects 149 tests of which only 42 are this plan's.

## For the Orchestrator

- `STATE.md`, `ROADMAP.md` and `REQUIREMENTS.md` were **not** touched (worktree
  mode) — verified by `git diff --stat <base>..HEAD` on all three: empty.
- `contracts/` was **not** touched.
- ENC-03 and ENC-05 are implemented and proven structurally here, but numerical
  parity against the frozen fixtures is 01-08's gate. The orchestrator owns
  whether to mark them complete.
- New deferred items appended as **D30-D33** under a new `## From plan 01-06`
  section. D1-D14 and D20-D23 untouched. D1, D2, D5 and D13 were used as
  documented and not re-logged.

## Self-Check: PASSED

All 3 created source files and this SUMMARY verified present on disk and, for
the sources, tracked by `git ls-files` and not matched by `git check-ignore`
(exit 1 each). All 4 commits verified as **ancestors of HEAD** with
`git merge-base --is-ancestor` (exit 0 each) rather than by grepping a
reformatted `git log`, which is the failure 01-04 recorded. `STATE.md`,
`ROADMAP.md`, `REQUIREMENTS.md` and `contracts/` confirmed unmodified by
`git diff --stat <base>..HEAD` (empty). `crates/aprender-core/src/models/bert/`
confirmed unmodified by the same means. Working tree clean apart from this
SUMMARY and `deferred-items.md`.
