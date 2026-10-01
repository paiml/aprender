---
phase: 01-differentiable-minilm-conformance
verified: 2026-08-08T12:30:46Z
verified_at_commit: 4ef4a62ae82b07ba5c9179dbf24eadf23f249b5e
phase_base_commit: e6dce92a011f1bfa81a4e3c54a3cb8a1e410062b
status: gaps_found
score: 5/6 requirements satisfied (ENC-04 partial); 5/5 roadmap success criteria met
overrides_applied: 0
re_verification: null
gaps:
  - truth: "ENC-04 — the controlled optimizer step is proven correct against the frozen reference"
    status: partial
    severity: blocker
    reason: >
      OBLIG-ENC-04-POST-STEP-PARAMETER-PARITY is vacuous. Its tolerance
      (3.05175781e-05) is 1.526x the MAXIMUM per-element AdamW step-1 displacement
      (2.017474e-05, computed from the committed fixtures). Confirmed empirically by
      two independent mutations run by this verifier against the merged tree:
      M1 (decoupled weight decay deleted from AdamW::update_param) -> conformance
      suite STILL 25 passed / 0 failed, exit 0;
      M4 (beta1/beta2 hardcoded to 0.5/0.5, bias correction rewritten to match)
      -> conformance suite STILL 25 passed, exit 0, AND the pre-existing
      `--lib adamw` suite STILL 26 passed, exit 0.
      The harness's own scope statement at tests/setfit_conformance.rs:51
      ("Detected here: ... a wrong AdamW hyperparameter or decay coupling") is
      therefore false on both counts.
      Root cause reconfirmed at scripts/setfit_fixtures/generate_fixtures.py:557 —
      the optimizer family reuses `grad_delta` (the GRADIENT family's f32/f64
      delta); no f64 optimizer step is ever run, and the family floor (2^-15) alone
      exceeds the entire signal being gated.
    artifacts:
      - path: "crates/aprender-core/tests/setfit_conformance/tolerances_generated.rs:41"
        issue: "OPTIMIZER_STEP = 3.05175781e-5 — 1.526x the maximum displacement it must resolve"
      - path: "contracts/setfit-encoder-conformance-v1.yaml:520"
        issue: "OBLIG-ENC-04-POST-STEP-PARAMETER-PARITY tolerance is the gradient family's number, not a measured post-step number"
      - path: "scripts/setfit_fixtures/generate_fixtures.py:557"
        issue: "`max_abs_f32_f64_delta: grad_delta` — the optimizer family's f32/f64 round-trip noise was never measured"
      - path: "crates/aprender-core/tests/setfit_conformance.rs:51"
        issue: "documented detection scope claims AdamW hyperparameter and decay-coupling detection; both empirically disproven (M1, M4)"
      - path: "crates/aprender-core/tests/setfit_conformance/gradient_gate.rs:200-204"
        issue: "`assert!(wd > 0.0, ...the half that distinguishes it from Adam)` establishes the FIXTURE used decay, never that the Rust optimizer applied it"
    missing:
      - "Run an f64 AdamW step in generate_fixtures.py and derive `optimizer_step.max_abs_f32_f64_delta` from it; scale the family floor to the UPDATE magnitude (lr), not to |p|"
      - "Add the separation guard generate_fixtures.py already applies to the activation family: assert the post-step tolerance sits well BELOW the decay term it must separate"
      - "Harden assert_encoder_updates clause (f) from `delta > 0` to `delta ~= lr` for non-exempt tensors with max|g| >> eps — reference-free and cheap"
      - "Record the STRUCTURAL limitation: a single-step fixture is beta-independent under bias correction, so no tolerance change can make betas falsifiable. A 2+ step fixture, or a separate multi-step obligation, is required."
      - "Correct the scope claim at tests/setfit_conformance.rs:51 to match what the gates actually detect"
  - truth: "make tier2 runs the gradient, frozen, and detach-negative gates (plan 01-08 must-have)"
    status: failed
    severity: warning
    reason: >
      `make tier2` on the merged tree exits 2, dying at recipe line 188
      (`cargo clippy -- -D warnings`) on inherited lint debt in aprender-zram-core
      (3 errors) and aprender-compute (19 errors). GNU make 3.81 on this box ignores
      .ONESHELL:, so the recipe stops there and the Phase 1 lines at 210-213 NEVER
      EXECUTE — verified: the tier2 log contains no "Phase 1 SetFit" marker.
      INHERITED, not a phase regression: this phase changed 0 files in either crate
      and 0 lint-configuration files; its only root Cargo.toml change is an additive
      `tokenizers` workspace-dependency entry. Same for tier3, whose first line
      (`cargo test --all`) is blocked by 13 x E0063 in aprender-serve
      (`cargo check -p aprender-serve --tests` -> exit 101, independently confirmed).
      Already logged by the phase itself as D50.
    artifacts:
      - path: "Makefile:185-222"
        issue: "Phase 1 gates are wired at lines 210-213, behind a clippy line that is red for inherited reasons"
    missing:
      - "Scope tier2's clippy to the crates under change, or clean aprender-compute + aprender-present-terminal (separate ticket, outside this phase's blast radius)"
      - "Repair crates/aprender-serve/tests/driver_cpu.rs against the current GGUFConfig shape and exclude aprender-profile from `cargo test --all` on non-Linux, so tier3 can reach its contract-validate step"
deferred: []
---

# Phase 1: Differentiable MiniLM Conformance — Verification Report

**Phase Goal:** Developers have one contracted, graph-connected MiniLM sentence encoder whose
tokenizer, batched forward path, train/eval behavior, gradients, and controlled updates are proven
before any SetFit trainer is exposed.

**Verified:** 2026-08-08T12:30:46Z at `4ef4a62ae`
**Status:** `gaps_found`
**Re-verification:** No — initial verification

---

## How this verification was conducted

Every result below is a command this verifier ran against the merged tree, with the exit code
captured directly (`cmd > file 2>&1; rc=$?` — never through a pipe, CLAUDE.md rule 1). No claim is
carried over from a SUMMARY without independent re-execution.

**Four falsification mutations were applied to the shipped source and reverted** (working tree
confirmed byte-identical to baseline afterwards). A green gate proves nothing until it has been
shown to turn red; per CLAUDE.md rule 2, each positive result below is paired with the mutation
that kills it, or is explicitly reported as unfalsifiable.

| ID | Mutation | Conformance suite | Verdict |
|----|----------|-------------------|---------|
| **M1** | Delete decoupled weight decay from `AdamW::update_param` (`rm_sprop.rs:80`) | **25 passed, exit 0** | **SURVIVES this phase's gates** (caught only by the pre-existing `falsify_aw_001_decoupled…` lib test, exit 101) |
| **M2** | Halve the Adam update (`0.5 * self.lr`, `rm_sprop.rs:83`) | exit 101, 1 failed | **KILLED** — but by `loss_after` (1.688e-3 vs tol 7.63e-6), **not** by post-step parameter parity, which passed |
| **M3** | Uniform pooling denominator (`counts.push(seq)`, `pooling.rs:112`) | exit 101, **3 failed** | **KILLED** by pooled parity + both gradient gates |
| **M4** | Hardcode AdamW `beta1=beta2=0.5` incl. bias correction | **25 passed, exit 0** | **SURVIVES everything** — also 26 passed on the pre-existing `--lib adamw` suite |

---

## Goal Achievement

### ROADMAP Success Criteria

| # | Success Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Import the pinned revision; typed errors for unsupported architecture/tokenizer/pooling/normalization/config mutation | ✓ VERIFIED | `setfit/import.rs:398` validates config before any weight byte; `PINNED_REVISION:67`, `PINNED_TOKENIZER_SHA256:73`, `PINNED_ACTIVATION:85`. **24 distinct mutated-field rejection tests** in `import_tests.rs` (hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, vocab_size, max_position_embeddings, layer_norm_eps, type_vocab_size, pad_token_id, `gelu_new`/`gelu_pytorch_tanh`/`relu`, both dropout probs, relative position embeddings, non-BERT architecture, non-BERT model_type, CLS pooling, max pooling, module stack without Normalize, mutated max_seq_length, one flipped tokenizer byte, slice dimensions, missing/unparseable config). |
| 2 | Batch-tokenize once; single or mixed-length padded inputs through one `Transformer -> masked mean pooling -> L2 normalization` path; ordered token facts; fixture-matching, padding-invariant embeddings | ✓ VERIFIED | `tokenizer_parity_every_frozen_case_matches_exactly` compares **all 12 frozen cases** by exact integer equality (ids, type ids, mask, truncation facts) — corpus includes CJK, accented, mixed-case and over-length, exactly as `OBLIG-ENC-02-TOKENIZER-PARITY` requires. `forward_parity.rs` compares per-layer intermediates, pooled AND normalized stages, and padding invariance against torch fixtures. Falsifiability proven by **M3**. |
| 3 | Enumerate stable named parameter groups; switch full encoder train/eval; prove registered parameters do not change on mode flip | ✓ VERIFIED | `nn/module.rs:79` `named_parameters` (positional-fallback default), `:105` recursive `set_training`. `encoder_mode_every_site_is_actually_applied_in_the_forward` probes **all 7 dropout sites individually** for forward effect — the closer for 01-06's mutation F. `encoder_mode_parameters_are_byte_identical_across_train_eval_train` compares `f32::to_bits`. Rust traversal asserted `==` `gradients.json.parameter_order` in content AND order (37 tensors). |
| 4 | Every contracted trainable component receives a finite non-zero gradient and changes after one optimizer step; frozen components byte-identical; embeddings move loss-reducing | ⚠️ **PARTIAL** | Gradient half VERIFIED, freeze half VERIFIED, movement half VERIFIED; **post-step correctness NOT falsifiable**. See the ENC-04 breakdown below. |
| 5 | Cosine-MSE objective is a finite graph-connected tensor; forward values, gradients and all new gather/mask/pool/normalize primitives match frozen and finite-difference fixtures; deliberate detachment makes the gate fail | ✓ VERIFIED | `setfit/loss.rs:71` returns `Result<Tensor, SetFitError>` shaped `[1]`, composed from `cosine_similarity_rows` + `mse_loss` (never the `f32` `nn/loss.rs` helpers). Both stages compared (per-row cosine + scalar MSE). **`detach_negative_the_gate_rejects_a_detached_encoder` is a live negative that ran and produced `Err`** — with `detach_negative_the_connected_encoder_passes_the_same_call` as the mirror, and `detach_negative_uses_the_same_gate_helper_as_the_positive_suites` proving one shared implementation. |

**Score: 5/5 roadmap success criteria met** (SC4 met as literally worded; see the caveat).

---

## Requirements Verdicts (ENC-01 … ENC-06)

### ENC-01 — pinned import into a typed contract, typed errors on variants — **SATISFIED**

- `crates/aprender-core/src/setfit/import.rs:398` `MiniLmImport::open` — configuration validated
  **before** any weight byte is read, so a mutated checkout fails on the mutated field.
- `crates/aprender-core/src/setfit/error.rs:23` `pub enum SetFitError`.
- 51 tests in `import_tests.rs`, 24 of them mutated-field rejections (enumerated in SC1 above).
- **The real architecture was exercised, not only the slice.** This verifier re-ran the D-10
  full-weight suite against the pinned checkout at `~/.cache/aprender/minilm-l6-v2-1110a243`:
  `cargo test -p aprender-core --features setfit,conformance-fixtures,model-tests --test setfit_conformance -- --ignored full_weight_`
  → **exit 0, 2 passed** (6 layers, hidden 384, 30522-token vocabulary; `[3,384]` embeddings
  within `FULL_MODEL_REFERENCE`). Manifest digests match those recorded in 01-08's SUMMARY
  (`apr_sha256 980becb4…`, `source_safetensors_sha256 53aa5117…`).
- **The pass is proven to be a real run, not a skip** (CLAUDE.md rule 2):
  `APRENDER_MINILM_DIR=/nonexistent-verifier-probe … -- --ignored full_weight_` → **exit 101,
  0 passed, 2 failed**.

### ENC-02 — batch-tokenize once; ordered ids/type ids/masks/truncation facts/provenance — **SATISFIED**

- `setfit/tokenizer.rs:96` `SentenceBatch` — all fields `pub(crate)`, public read-only accessors
  (`input_ids`, `token_type_ids`, `attention_mask`, `batch`, `seq`, `truncation`, `provenance`,
  `tokenizer_sha256`); `sentence_batch_has_no_public_fields_and_no_mutable_accessors` holds the line.
- `tokenizer_parity_every_frozen_case_matches_exactly` — **exact integer equality, no tolerance**,
  over all 12 cases; also asserts every case was frozen at the same `MAX_SEQUENCE_LENGTH` and
  compares per-row `truncated` / `original_len`.
- `conformance_every_fixture_case_id_joins_the_corpus_of_record` proves the triple equality
  fixture-ids == corpus-of-record ids == what the Rust tokenizer produces from the recorded texts.
- `conformance_the_slice_batch_carries_canonical_ids_the_encoder_must_remap` proves the remap is a
  reachable branch, not a no-op (canonical ids above the 97-row slice vocabulary are present).

### ENC-03 — one shared batched `Transformer -> masked mean pool -> L2 norm` path, fixture-verified — **SATISFIED**

- `setfit/encoder.rs:377 forward_tokens`, `:405 forward_tokens_per_layer`, `:419 encode`.
- 01-09's repair is real: `nn/transformer/positional_encoding.rs:480 add_mask` now expands the mask
  to the scores' shape (`broadcast_mask_to:382`) and applies it with the autograd-aware
  `Tensor::add`, so `AddBackward` records the edge — replacing a `.zip()` fallback that both
  truncated and severed the tape. Reached from `nn/transformer/mod.rs:91`.
  `tests_attention_mask_broadcast.rs` (364 lines) covers B>1 / H>1 / T!=S value AND graph tests.
- `autograd/ops/activation.rs:183 gelu_exact`; the activation gate carries its own **live negative**
  (`conformance_activation_gate_rejects_the_tanh_approximation`) and asserts the measured
  exact-vs-tanh separation (4.734993e-04, ~106x tolerance) is comfortably above the epsilon —
  the plan's predicted `>1e-3` threshold was corrected to the measurement.
- `tests/batched_graph_spike.rs` → **exit 0, 7 passed**.
- **Falsifiability proven (M3):** a uniform pooling denominator turns pooled parity and both
  gradient gates RED. Notably padding-invariance stayed green under M3 — because L2 normalization
  is scale-invariant — which is precisely the failure mode
  `OBLIG-ENC-03-POOLED-EMBEDDING-PARITY` names as its reason for comparing *both* stages. The
  contract's stated rationale is empirically correct.

### ENC-04 — named parameters, freeze/trainable groups, finite non-zero gradients and parameter changes after a controlled step — **PARTIALLY SATISFIED**

Assessed in four parts, because they have different evidentiary standing.

| Part | Status | Evidence |
|---|---|---|
| (a) **Enumerate named encoder parameters** | ✓ SATISFIED | HF dotted names verbatim, pooler excluded; `gradient_gate_named_gradients_match_the_frozen_reference` asserts `names == gradients.json.parameter_order` in content **and order** (37 tensors), and that the counts agree. |
| (b) **Select frozen and trainable groups** | ✓ SATISFIED | `FreezeGroup` (`setfit/mod.rs:86`) with `apply_freeze` / `clear_freeze` / `freeze_policy`; `trainable_parameters_mut` / `frozen_parameters`. `frozen_gate.rs` proves the partitions are **disjoint and complete** against `named_parameters()`, that frozen tensors are absent from the optimizer's parameter set (the load-bearing mechanism per D42, not the `requires_grad` flag), and that frozen tensors are **bitwise** unchanged across a real AdamW step — with `assert!(!moved.is_empty())` blocking the vacuous "nothing moved" pass. `frozen_gate_never_compares_against_the_all_trainable_optimizer_fixture` guards the T-1-24 conflation. |
| (c) **Finite non-zero gradients per contracted component** | ✓ SATISFIED | `assert_encoder_updates` clauses (a)-(e) in `setfit_conformance.rs:509`, with the exemption set read as DATA from `gradients.json.analytically_zero[]` and enforced **two-sided**. `OBLIG-ENC-04-NAMED-GRADIENT-PARITY` is genuinely discriminating: measured from the fixtures, every non-exempt tensor's `max|g|` is **20x to 46,244x** the 3.05e-5 tolerance (global max 1.4113). The two exempt key biases measure 6.9e-11 / 9.5e-11 against a 6.129e-5 floor, and the gate additionally requires their **query.bias siblings** to exceed the floor so the exemption describes the key bias rather than the backward. Killed by M3. |
| (d) **"changes after a controlled optimizer step"** | ⚠️ **PARTIAL** | The *movement* clause (f) holds and is enforced per tensor and per component; the loss decreases to the **recorded** `loss_after` and M2 proved that assertion is live. But the *correctness* of the step against the frozen reference is **not falsifiable** — see below. |

**The unfalsifiable part, stated precisely.**

`OBLIG-ENC-04-POST-STEP-PARAMETER-PARITY` (tolerance `3.05175781e-05`) gates a signal it cannot
resolve. Computed directly from the committed `gradients.json` + `optimizer_step.json`
(`lr=2e-5, betas=[0.9,0.999], eps=1e-8, weight_decay=0.01`, 110,528 elements across 37 tensors):

| Quantity | Measured |
|---|---|
| Maximum per-element AdamW step-1 displacement | **2.017474e-05** |
| Maximum decoupled-decay-only displacement | 1.747607e-07 |
| `tol::OPTIMIZER_STEP` | **3.052e-05** = **1.526x the whole step**, **175x the decay term** |
| Decay's contribution to the loss delta (`lr·wd·Σ gᵢpᵢ`) | 7.33e-09 = **0.001x** the `LOSS_PAIR` tolerance |

Consequences, each verified by execution rather than argument:

1. **Missing decoupled weight decay is invisible to this phase** (M1: 25 passed, exit 0). It is
   caught only by the pre-existing `nn::optim::rm_sprop::tests::tests_adamw_contract::falsify_aw_001_decoupled…`
   lib test (exit 101) — i.e. by `adamw-kernel-v1`, not by `setfit-encoder-conformance-v1`.
2. **Wrong betas are invisible to everything** (M4: 25 passed conformance, 26 passed `--lib adamw`,
   both exit 0). This is **structural, not a tolerance problem**: at step 1 with bias correction,
   `m̂ = g` and `v̂ = g²` for any β₁, β₂, so a single-step fixture is beta-independent by
   construction. No tolerance edit can fix it; a multi-step fixture is required. **This finding is
   new — it is not covered by CR-01.**
3. **A halved learning rate passes post-step parity** (M2: the `assert_close` at
   `gradient_gate.rs:251` passed; the test failed only at `loss_after`, 1.688e-3 vs 7.63e-6).
   The `loss_after` assertion — 442.9x tolerance headroom — is the real backstop, and it is the
   only thing standing between this gate and CR-01's first two falsifications. CR-01's claim that
   `lr=1e-5` "PASSES" is therefore correct about the *obligation* and incorrect about the
   *enclosing test*.

**Root cause reconfirmed independently.** `scripts/setfit_fixtures/generate_fixtures.py:557` sets
`"max_abs_f32_f64_delta": grad_delta` for the optimizer family — `grad_delta` is computed at
`:507-509` from the *gradients*. No f64 optimizer step is run anywhere in the generator. Because
`FAMILY_REDUCTION_WIDTH["optimizer_step"] == FAMILY_REDUCTION_WIDTH["gradients"]`, the floor is
identical too, and the committed `tolerances_measured.json` entries for `gradients` and
`optimizer_step` are byte-identical — the tell. The activation family *does* carry a separation
guard (`generate_fixtures.py:320-324`); the optimizer family does not.

**A false scope claim ships with the harness.** `tests/setfit_conformance.rs:51` lists, under
"**Detected here:**", "a wrong AdamW hyperparameter or decay coupling". M1 and M4 disprove both
halves. The phase is otherwise unusually disciplined about stating what its gates cannot see
(the whole "NOT detected here" paragraph, and 01-08's SUMMARY); this one line is out of step with
that discipline and should be corrected, since downstream phases will read it as the gate's contract.

**Why this is PARTIAL and not FAILED.** ENC-04 as written in REQUIREMENTS.md asks for "finite
non-zero gradients **and parameter changes** for every contracted trainable component after a
controlled optimizer step". That is satisfied: gradients are finite, non-zero per component and
per non-exempt tensor and match the reference within a tolerance 20x-46,000x below the signal;
parameters demonstrably move; frozen ones demonstrably do not. What is not established is that the
step is the *right* step. The phase's own contract asserts the stronger property, and that
assertion cannot fail.

### ENC-05 — recursive train/eval switching with dropout, without changing registered parameters — **SATISFIED**

- `nn/module.rs:105` recursive `set_training` default plus semantic overrides on Linear, LayerNorm,
  Dropout, Sequential and MultiHeadAttention; `nn/tests_named_module.rs` (470 lines).
- `setfit/encoder.rs:802` `set_training`; `dropout_sites()` reports 7 sites (1 embeddings + 3 per
  layer x 2 layers), each with its own seeded stream.
- **`encoder_mode_every_site_is_actually_applied_in_the_forward`** turns each site on individually
  against an otherwise-eval encoder and requires the output to move. This is the direct closer for
  the 01-06 mutation-F class the task flagged (a constructed, seeded, mode-aware, never-called
  dropout site) and it is the reason that class is now covered.
- `encoder_mode_parameters_are_byte_identical_across_train_eval_train` uses `f32::to_bits`, over
  the same named traversal ENC-04 uses (`OBLIG-ENC-05-MODE-SWITCH-PARAMETER-INVARIANCE`).
- Eval-mode double-encode determinism is asserted in the same suite.

### ENC-06 — finite, tensor-valued, graph-connected cosine-similarity MSE pair loss matching frozen forward and gradient fixtures — **SATISFIED**

- `setfit/loss.rs:71` `pair_cosine_mse` → `Result<Tensor, SetFitError>`, shape asserted `[1]`.
  Composed from `cosine_similarity_rows` + `mse_loss` only; the module docs and a source assertion
  in `loss_tests.rs` forbid the `f32`-returning `nn/loss.rs` helpers (the PF-001 trap).
- Validation order is shapes → label length → **explicit finiteness** → binary membership, so a NaN
  label is diagnosed as non-finite rather than as "not in {0,1}".
- Forward parity compares **both** stages (per-row cosine vector and scalar MSE) at 7.63e-6, with a
  guard that every encoded row's norm exceeds 0.5 so the epsilon-clamp branch choice is immaterial.
- Gradient side: `loss.backward()` reaches encoder parameters (named-gradient parity, 37 tensors),
  and the **D-24 detach negative is a live red** proven in-run.

---

## Required Artifacts

| Artifact | Level 1 exists | Level 2 substantive | Level 3 wired | Level 4 data flows | Status |
|---|---|---|---|---|---|
| `contracts/setfit-encoder-conformance-v1.yaml` | ✓ 827 lines | ✓ 14 proof obligations, falsification tests, rationale | ✓ reached from `tier3 -> contract-validate` (Makefile:253, CONTRACTS:917) | ✓ `pv validate` exit 0, 0 errors / 0 warnings | ✓ VERIFIED |
| `autograd/ops/{embedding,masking,pooling,normalize,similarity,activation}.rs` | ✓ | ✓ typed `Result` returns, checked denominators, fail-closed OOV | ✓ consumed by `setfit/encoder.rs` and `setfit/loss.rs` | ✓ M3 proves the pooling path is load-bearing | ✓ VERIFIED |
| `nn/module.rs` (`named_parameters`, `set_training`) | ✓ 490 lines | ✓ default + 5 semantic overrides | ✓ used by encoder, freeze partition, all gates | ✓ order asserted against `parameter_order` | ✓ VERIFIED |
| `nn/transformer/positional_encoding.rs` (`add_mask`, `broadcast_mask_to`) | ✓ 651 lines | ✓ full broadcast + `AddBackward` edge | ✓ called at `mod.rs:91` | ✓ spike + broadcast tests green | ✓ VERIFIED |
| `setfit/{mod,error,tokenizer,import,encoder,loss}.rs` | ✓ 2,902 lines | ✓ | ✓ sealed constructors; `SetFitMiniLm` sole public entry | ✓ 162 lib tests green | ✓ VERIFIED |
| `tests/fixtures/setfit/*` (16 files) | ✓ | ✓ real-weight slice, 1.8 MB gradients, 1.6 MB optimizer step | ✓ loaded by harness through `read_fixture` | ⚠️ `optimizer_step` tolerance not derived from its own family | ⚠️ see gap 1 |
| `tests/setfit_conformance.rs` + 5 submodules | ✓ 2,220 lines | ✓ | ✓ | ✓ 25 passed / 1 ignored, exit 0 | ✓ VERIFIED |
| `tests/batched_graph_spike.rs` | ✓ 659 lines | ✓ | ✓ | ✓ 7 passed, exit 0 | ✓ VERIFIED |
| `scripts/setfit_fixtures/*` (uv.lock, generators) | ✓ | ✓ hash-locked env | ✓ | ⚠️ `:557` reuses the wrong family delta | ⚠️ see gap 1 |
| `Makefile` tier2/tier3 wiring | ✓ lines 210-213, 253-254 | ✓ | ⚠️ present but **unreachable** — tier2 dies at line 188 | ✗ Phase 1 markers absent from a real `make tier2` run | ⚠️ see gap 2 |

---

## Key Link Verification

| From | To | Via | Status | Evidence |
|---|---|---|---|---|
| harness | `SetFitMiniLm::tokenize` | the ONE batch builder | ✓ WIRED | `conformance_the_tokenizer_boundary_is_the_only_batch_source` — scans all 6 harness files, requires exactly 1 `.tokenize(` site inside `batch_from_case`, and 0 `SentenceBatch {` literals |
| harness | sealed constructors | must be unreachable | ✓ SEALED — **independently proven RED** | This verifier compiled an out-of-crate probe calling `MiniLmImport::open`, `MiniLmTokenizer::from_bytes`, `BertSentenceEncoder::from_import` from `tests/`: **exit 101, three `error[E0624]: associated function … is private`**. (E0624, not the plan's predicted E0603 — the shipped code documents this correction at `setfit_conformance.rs:39` / D41.) |
| `tolerances_generated.rs` | contract YAML | in-tree parser + sha256 + version agreement | ✓ WIRED | `conformance_tolerances_agree_with_the_contract` passes; contract sha256 `51fa2cb7…` matches; parsed with `setfit_contract_schema` (the in-tree crate, not the stale crates.io `provable-contracts` dev-dep — D53) |
| harness | hand-written epsilons | must not exist | ✓ ENFORCED | `conformance_no_hand_written_tolerance_literal_outside_the_generated_file` ships an 8-row must-match/must-not-match case table that is **re-run**, not re-read (CLAUDE.md rule 7) |
| `forward_parity.rs` | `forward_tokens_per_layer` | via `SetFitMiniLm::encoder()` | ✓ WIRED | per-layer parity green; failure messages name the layer index |
| `gradient_gate.rs` | `AdamW::step_with_params` | fixture hyperparameters | ✓ WIRED, ⚠️ weakly gated | see ENC-04(d) |
| `Makefile` tier3 | `pv validate` over the phase contract | `$(MAKE) contract-validate` → `CONTRACTS:917` | ✓ WIRED and GREEN in isolation | `make contract-validate` → **exit 0**, `contracts/setfit-encoder-conformance-v1.yaml … 0 error(s), 0 warning(s)`, 42 contracts clean |
| `Makefile` tier2 | Phase 1 gates | recipe lines 210-213 | ✗ **NOT REACHED** | `make tier2` → **exit 2** at line 188; no "Phase 1 SetFit" marker in the log |
| `aprender-core/Cargo.toml` | fixture files | root-anchored `/tokenizer.json` exclude | ✓ WIRED | `cargo package -p aprender-core --list` → exit 0, **all 16 fixture files present** (CB-510 class closed) |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Conformance suite green | `cargo test -p aprender-core --test setfit_conformance --features setfit,conformance-fixtures` | exit 0 — **25 passed, 0 failed, 1 ignored** | ✓ PASS |
| setfit library gates green | `cargo test -p aprender-core --lib --features setfit,conformance-fixtures setfit::` | exit 0 — **162 passed** | ✓ PASS |
| Batched graph spike | `cargo test -p aprender-core --test batched_graph_spike --features setfit` | exit 0 — **7 passed** | ✓ PASS |
| Contract validates | `make contract-validate` | exit 0 — setfit contract **0 errors / 0 warnings** | ✓ PASS |
| Feature isolation (D-06) | `make setfit-feature-matrix` | exit 0 — no `tokenizers` node in a no-default-features tree | ✓ PASS |
| Fixture integrity | `shasum -a 256 -c manifest.sha256` (in fixture dir) | exit 0 — **15/15 OK** | ✓ PASS |
| Fixtures survive packaging | `cargo package -p aprender-core --list` | exit 0 — 16/16 fixture paths present | ✓ PASS |
| **D-10 full-weight suite** | `cargo test … --features setfit,conformance-fixtures,model-tests … -- --ignored full_weight_` | exit 0 — **2 passed** on the real 6-layer/384-hidden pin | ✓ PASS |
| **D-10 negative control** | same, with `APRENDER_MINILM_DIR=/nonexistent-verifier-probe` | **exit 101 — 2 failed** | ✓ PASS (proves the run above is not a skip) |
| **D-08 seal negative control** | out-of-crate probe calling the three sealed constructors | **exit 101 — 3 × E0624** | ✓ PASS |
| `make tier2` | `make tier2` | **exit 2** at `cargo clippy -- -D warnings` (inherited) | ✗ FAIL (inherited) |
| tier3 blocker (inherited) | `cargo check -p aprender-serve --tests` | exit 101 — **13 × E0063** on `GGUFConfig` | ✗ FAIL (inherited) |

---

## Requirements Coverage

| Requirement | Source plans | Status | Evidence |
|---|---|---|---|
| ENC-01 | 01-04, 01-05 | ✓ SATISFIED | 24 mutated-field rejection tests; D-10 full-weight parity re-run green with negative control |
| ENC-02 | 01-04, 01-05 | ✓ SATISFIED | exact integer parity over 12 cases incl. CJK/accented/cased/over-length; read-only `SentenceBatch` |
| ENC-03 | 01-01, 01-03, 01-04, 01-06, 01-08, 01-09 | ✓ SATISFIED | per-layer + pooled + normalized + padding-invariance + activation parity; falsified by M3 |
| ENC-04 | 01-02, 01-04, 01-07, 01-08 | ⚠️ **PARTIAL** | (a)(b)(c) satisfied and falsifiable; (d) post-step correctness unfalsifiable — M1 and M4 survive |
| ENC-05 | 01-02, 01-04, 01-06, 01-08 | ✓ SATISFIED | 7-site forward-application probe + `to_bits` mode invariance |
| ENC-06 | 01-01, 01-03, 01-04, 01-07, 01-08 | ✓ SATISFIED | both-stage forward parity; live detach negative |

**Orphan check:** the union of `requirements:` across all 9 PLAN frontmatters is exactly
{ENC-01, ENC-02, ENC-03, ENC-04, ENC-05, ENC-06} — identical to the ROADMAP's Phase 1 mapping.
**No orphaned and no unclaimed requirements.**

**Traceability drift (Info):** `.planning/REQUIREMENTS.md` still lists ENC-01…ENC-06 as `Pending`
while `.planning/ROADMAP.md:20` marks Phase 1 complete. The traceability table should be updated
to reflect the verdicts above (ENC-01/02/03/05/06 satisfied, ENC-04 partial).

---

## Anti-Patterns Found

| Scope | Pattern | Result |
|---|---|---|
| All **62** code files changed between `e6dce92a0..HEAD` (`.rs`, `.py`, `.toml`, `.yaml`, `Makefile`) | `TBD` / `FIXME` / `XXX` | **0** — debt-marker gate clean |
| Same 62 files | `TODO` / `HACK` / `PLACEHOLDER` / `todo!` / `unimplemented!` | **0** |
| Phase blast radius | changes to `aprender-compute`, `aprender-zram-core`, `aprender-present-terminal`, `aprender-serve` | **0 files** — the tier failures are provably inherited |
| Phase blast radius | changes to `.github/workflows/`, `.clippy.toml`, workspace lint config | **0 files** — root `Cargo.toml` change is a single additive `tokenizers` dependency entry |

One documentation anti-pattern, reported under gap 1: an **overstated detection claim** at
`tests/setfit_conformance.rs:51`, falsified by M1 and M4.

---

## Additional unfalsifiable-check scan

The task asked for further instances of the "plan-specified check that cannot fail against correct
code" class. Beyond the five already recorded and CR-01, this verifier found:

1. **NEW — AdamW betas are structurally untestable by this fixture** (M4). Not a tolerance defect;
   a step-1 fixture with bias correction is beta-independent by construction. Unlike weight decay,
   there is **no** compensating workspace test. Folded into gap 1.
2. `assert_encoder_updates` clause (c) tests `l2(values) <= 0.0`, i.e. it can only catch an
   **exactly** zero gradient. A near-severed path with a 1e-30 gradient passes clause (c). This is
   defensible in context — the named-gradient parity gate at 20x-46,000x headroom is what actually
   constrains gradient values — but the clause alone is weaker than its prose suggests. **Info.**
3. `conformance_manifest_self_check` requires `checked >= 15` and the manifest contains exactly 15
   files. Correct today and correctly directional (removing a fixture fails), but it sits on the
   boundary with no margin. **Info.**
4. `conformance_tolerances_agree_with_the_contract` degrades to a 64-lowercase-hex-digit format
   check when `contracts/` is unreachable (packaged crate). Documented in-code and correctly
   scoped, not a defect. **Sound.**

Checks confirmed **genuinely falsifiable** by execution rather than inspection: pooled/normalized
parity, per-layer forward parity, named-gradient parity, `loss_after`, the detach negative, the
tanh-vs-exact activation negative, the D-08 seal, and the D-10 full-weight suite.

---

## Assessment of the two known shortfalls

**1. CR-01 — does it mean ENC-04 is satisfied, partial, or unsatisfied?**

**Partially satisfied**, and the split is clean. The gradient-parity portion and the freeze-policy
portion are sound *independently* and are falsifiable — verified by M3 killing gradient parity, by
the live detach negative, and by the frozen gate's disjoint/complete partition plus its bitwise
post-step identity check with a non-vacuity guard. The optimizer-step portion is where the failure
sits, and it is narrower than CR-01 states in one respect and broader in another:

- *Narrower:* a halved learning rate **is** caught, by `loss_after` (M2), which CR-01 does not
  credit. The obligation is vacuous; the enclosing test is not entirely so.
- *Broader:* CR-01 covers weight decay; **betas are also undetectable, and structurally so** (M4).
  That cannot be fixed by re-measuring the tolerance.

The requirement text ENC-04 ("finite non-zero gradients **and parameter changes** … after a
controlled optimizer step") is met. The phase's own stronger obligation, and its published
detection scope, are not.

**2. `make tier2` / `make tier3` RED — what does it mean for the "falsifiable gate" goal?**

The gates themselves are real, green and falsifiable — that is established above, independent of
any make target. What is lost is the *automatic* invocation: `make tier2` exits 2 before reaching
recipe lines 210-213, so nothing in the default developer pre-commit path exercises the Phase 1
gates today, and a future regression in `setfit/` would not surface until someone runs the two
`cargo test` commands by hand. That is a real erosion of the "gate" property (a gate that is not
reached is not a gate — the same argument 01-08 itself makes in the Makefile comment at line 189),
but the cause is entirely inherited: 0 files changed in the offending crates, 0 lint-config
changes, and the blockers reproduce as pure compile/lint errors in `aprender-compute`,
`aprender-zram-core` and `aprender-serve`. The wiring is correct and will start working the moment
those two tickets land. Classified **WARNING**, not blocker.

---

## Gaps Summary

The phase goal is substantively achieved. The encoder is contracted, graph-connected, tokenizer-
exact, fixture-parity-verified at both the slice and the real 6-layer/384-hidden pin, mode-correct
with every dropout site proven to be on the forward path, and its gradient gate is proven able to
turn red by a live detached-encoder negative and by three independent mutations run during this
verification. The D-08 seal was proven RED by an out-of-crate compile probe. Zero debt markers
across 62 changed files. Five of six requirements are fully satisfied and all five roadmap success
criteria are met as written.

Two gaps remain:

1. **(blocker)** The ENC-04 optimizer-step obligation cannot fail. Its tolerance is 1.526x the
   maximum displacement it gates; deleting decoupled weight decay (M1) and hardcoding the betas
   (M4) both leave the suite green. The betas case is structural and is a new finding beyond
   CR-01. The harness's own "Detected here" list at `setfit_conformance.rs:51` overstates its
   scope on both counts and will be read by Phase 3 as this gate's contract.
2. **(warning)** The tier wiring is correct but unreachable: `make tier2` dies on inherited clippy
   debt before the Phase 1 lines, so the gates run only when invoked directly.

Neither prevents Phase 2 (deterministic pair and data protocol) from starting — Phase 2 depends on
the tokenizer, the batch type and the encoder forward path, all of which are verified. Gap 1
**must** be closed before Phase 3, which is the phase that consumes the optimizer step and whose
TRN-03 requires "proof that named encoder gradients, parameter deltas, embedding deltas, and
pair-loss behavior passed before a run may identify itself as SetFit".

**Escalation — developer decision required.** Gap 1 can be closed by (a) fixing the generator and
regenerating the contract tolerance (with the `pv diff` semver bump), plus adding a multi-step
obligation for betas; or (b) accepting it as a documented deviation via a verification override,
on the grounds that `adamw-kernel-v1` already owns AdamW correctness and this phase's obligation
should be narrowed to movement rather than value parity. Option (b) still requires correcting the
false scope claim at `setfit_conformance.rs:51`.

---

_Verified: 2026-08-08T12:30:46Z_
_Verifier: Claude (gsd-verifier) — all results re-executed against `4ef4a62ae`; working tree
restored to baseline after four mutations_
