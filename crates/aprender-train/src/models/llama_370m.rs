#![allow(clippy::doc_overindented_list_items)]
//! # Llama 370M Sovereign (albor) — Architectural Scaffold
//!
//! Compile-time-frozen configuration for the SHIP-TWO-001 MODEL-2 "albor"
//! 370M Python code-completion model.
//!
//! **Canonical contract:** `contracts/model-families/llama-370m-sovereign-v1.yaml`
//! **Contract version:** 1.0.0
//! **Contract ID:**      C-LLAMA-370M-SOVEREIGN
//!
//! ## Purpose
//!
//! This module is a **scaffold only** — it does NOT implement forward/backward.
//! Its sole job is to lift the architectural constants from the YAML contract
//! into Rust's type system so that recipe/artifact drift (the MODEL-1 v2 QLoRA
//! divergence class of bug) is caught at compile time, not at eval time.
//!
//! ## Invariants (mirrored from the YAML contract)
//!
//! - **INV-ARCH-370M-001**  Parameter count ∈ [366M, 374M] (370M ± 1%).
//!                          Verified at runtime by `estimated_param_count()`
//!                          and by `apr inspect` on trained artifacts.
//! - **INV-ARCH-370M-002**  `num_heads * head_dim == hidden_dim` (16 * 64 == 1024).
//!                          Compile-time asserted in [`Llama370MConfig::validate`].
//! - **INV-ARCH-370M-003**  `num_kv_heads` divides `num_heads` evenly (GQA).
//!                          Compile-time asserted in [`Llama370MConfig::validate`].
//! - **INV-ARCH-370M-004**  `tied_embeddings == true` — lm_head shares storage
//!                          with token_embd. Compile-time enforced via the
//!                          `TIED_EMBEDDINGS` const.
//! - **INV-ARCH-370M-005**  `rope_theta == 10000.0` exactly (Llama-1 convention).
//!                          Compile-time enforced as a `const f32`.
//! - **INV-ARCH-370M-006**  `vocab_size == 50_257` and matches the paired
//!                          tokenizer-bpe-v1 contract. Tokenizer coupling
//!                          cannot be checked at compile time — runtime
//!                          `debug_assert_eq!` at model load.
//! - **INV-ARCH-370M-007**  SwiGLU activation: distinct `gate_proj` and
//!                          `up_proj` tensors. Enforced at checkpoint load
//!                          time (runtime) by the APR loader.
//! - **INV-ARCH-370M-008**  `has_bias == false` on every linear projection.
//!                          Compile-time enforced via the `HAS_BIAS` const.
//! - **INV-ARCH-370M-009**  Row-major APR layout (LAYOUT-001). Embedding
//!                          shape `[vocab_size, hidden_dim]`, NOT reversed.
//!                          Enforced by `aprender::format::layout_contract`
//!                          at load time (runtime — tensor data is not
//!                          available to the type system).
//!
//! ## Design Notes
//!
//! Rust 1.79+ supports `const` panics, so every machine-checkable invariant
//! lives inside [`Llama370MConfig::validate`], a `const fn` that compiles
//! down to nothing if all invariants hold and refuses to compile otherwise
//! (via `const _: () = Llama370MConfig::validate();`).
//!
//! The `HiddenDim<N>`, `NumHeads<N>`, etc. PhantomData newtypes exist so
//! that downstream code (forward/backward, to be written later) can be
//! parameterized on the exact dimensions — making it a compile error to,
//! for instance, pass a `HiddenDim<768>` activation into a 1024-dim
//! projection.
//!
//! This module intentionally does NOT:
//!   - implement forward/backward;
//!   - allocate tensors;
//!   - export anything to `aprender-train`'s public API
//!     (re-exports are a follow-up PR).

#![allow(dead_code)] // scaffold — forward/backward not yet implemented

use std::marker::PhantomData;

// ─────────────────────────────────────────────────────────────
// Compile-time shape newtypes (Poka-Yoke)
// ─────────────────────────────────────────────────────────────
//
// These zero-sized types let downstream code be generic on exact
// dimensions. Mixing, e.g., a HiddenDim<1024> with a HiddenDim<768>
// is a compile error, not a runtime shape mismatch.

/// Hidden dimension (model width) as a compile-time constant.
#[derive(Debug, Clone, Copy, Default)]
pub struct HiddenDim<const N: usize>(PhantomData<()>);

impl<const N: usize> HiddenDim<N> {
    pub const VALUE: usize = N;
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// Number of attention heads.
#[derive(Debug, Clone, Copy, Default)]
pub struct NumHeads<const N: usize>(PhantomData<()>);

impl<const N: usize> NumHeads<N> {
    pub const VALUE: usize = N;
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// Number of KV heads (GQA).
#[derive(Debug, Clone, Copy, Default)]
pub struct NumKvHeads<const N: usize>(PhantomData<()>);

impl<const N: usize> NumKvHeads<N> {
    pub const VALUE: usize = N;
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// Per-head dimension (hidden_dim / num_heads).
#[derive(Debug, Clone, Copy, Default)]
pub struct HeadDim<const N: usize>(PhantomData<()>);

impl<const N: usize> HeadDim<N> {
    pub const VALUE: usize = N;
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// Intermediate (FFN) dimension.
#[derive(Debug, Clone, Copy, Default)]
pub struct IntermediateDim<const N: usize>(PhantomData<()>);

impl<const N: usize> IntermediateDim<N> {
    pub const VALUE: usize = N;
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// Number of transformer blocks.
#[derive(Debug, Clone, Copy, Default)]
pub struct NumLayers<const N: usize>(PhantomData<()>);

impl<const N: usize> NumLayers<N> {
    pub const VALUE: usize = N;
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// Vocabulary size.
#[derive(Debug, Clone, Copy, Default)]
pub struct VocabSize<const N: usize>(PhantomData<()>);

impl<const N: usize> VocabSize<N> {
    pub const VALUE: usize = N;
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

// ─────────────────────────────────────────────────────────────
// Llama370MConfig — frozen architectural constants
// ─────────────────────────────────────────────────────────────
//
// All fields are `pub const` and byte-identical to
// contracts/model-families/llama-370m-sovereign-v1.yaml §architecture
// and §constraints.

/// Architectural configuration for the albor 370M sovereign model.
///
/// Every constant here is pinned to a specific value in the YAML contract.
/// Changing any of these values requires bumping the contract to v1.1.0
/// and re-running the `GATE-ARCH-370M-*` gates.
pub struct Llama370MConfig;

impl Llama370MConfig {
    // ── Architecture ──
    /// Total parameter count (nominal). See `estimated_param_count()` for
    /// the runtime-checkable figure under INV-ARCH-370M-001.
    pub const PARAMETERS_NOMINAL: usize = 370_000_000;

    /// Lower bound on param count (INV-ARCH-370M-001).
    pub const PARAMETERS_MIN: usize = 366_000_000;

    /// Upper bound on param count (INV-ARCH-370M-001).
    pub const PARAMETERS_MAX: usize = 374_000_000;

    pub const HIDDEN_DIM: usize = 1024;
    pub const NUM_LAYERS: usize = 24;
    pub const NUM_HEADS: usize = 16;
    pub const NUM_KV_HEADS: usize = 4; // GQA: heads / 4
    pub const HEAD_DIM: usize = 64; // hidden_dim / num_heads
    pub const INTERMEDIATE_DIM: usize = 2816; // ~2.75 * hidden
    pub const VOCAB_SIZE: usize = 50_257;
    pub const MAX_POSITION_EMBEDDINGS: usize = 4096;

    /// RoPE base frequency — Llama-1 convention (INV-ARCH-370M-005).
    pub const ROPE_THETA: f32 = 10_000.0;

    /// RMSNorm epsilon.
    pub const RMS_NORM_EPS: f32 = 1.0e-5;

    // ── Constraints ──
    pub const TIED_EMBEDDINGS: bool = true; // INV-ARCH-370M-004
    pub const HAS_BIAS: bool = false; // INV-ARCH-370M-008

    /// Compile-time verification of every machine-checkable invariant.
    ///
    /// Each `assert!` here becomes a hard compile error (via Rust 1.79+
    /// `const` panics) if the invariant is violated. Any change to the
    /// constants above that breaks one of these invariants will fail to
    /// compile — by design.
    ///
    /// Invariants encoded here (in order):
    ///   INV-ARCH-370M-002  num_heads * head_dim == hidden_dim
    ///   INV-ARCH-370M-003  num_kv_heads divides num_heads
    ///   INV-ARCH-370M-004  tied_embeddings == true
    ///   INV-ARCH-370M-005  rope_theta == 10000.0
    ///   INV-ARCH-370M-006  vocab_size == 50_257
    ///   INV-ARCH-370M-008  has_bias == false
    ///
    /// Invariants NOT encodable at compile time (documented as runtime
    /// `debug_assert!` at load sites):
    ///   INV-ARCH-370M-001  param count ∈ [366M, 374M] — depends on the
    ///                      actual allocated tensors; checked by
    ///                      `estimated_param_count()` and by `apr inspect`.
    ///   INV-ARCH-370M-007  SwiGLU gate_proj/up_proj both present and
    ///                      distinct — depends on the on-disk checkpoint
    ///                      tensor table; checked by the APR loader.
    ///   INV-ARCH-370M-009  row-major [vocab_size, hidden_dim] layout —
    ///                      depends on tensor shape metadata in the
    ///                      loaded artifact; checked by
    ///                      `aprender::format::layout_contract`.
    pub const fn validate() {
        // INV-ARCH-370M-002
        assert!(
            Self::NUM_HEADS * Self::HEAD_DIM == Self::HIDDEN_DIM,
            "INV-ARCH-370M-002 violated: num_heads * head_dim != hidden_dim",
        );

        // INV-ARCH-370M-003
        assert!(
            Self::NUM_KV_HEADS > 0 && Self::NUM_HEADS % Self::NUM_KV_HEADS == 0,
            "INV-ARCH-370M-003 violated: num_kv_heads does not divide num_heads",
        );

        // INV-ARCH-370M-004
        assert!(
            Self::TIED_EMBEDDINGS,
            "INV-ARCH-370M-004 violated: tied_embeddings must be true for 370M",
        );

        // INV-ARCH-370M-005 — f32 equality is legal in const context
        // and is exactly what the contract requires (byte-equal literal).
        assert!(
            Self::ROPE_THETA == 10_000.0_f32,
            "INV-ARCH-370M-005 violated: rope_theta must be exactly 10000.0",
        );

        // INV-ARCH-370M-006
        assert!(
            Self::VOCAB_SIZE == 50_257,
            "INV-ARCH-370M-006 violated: vocab_size must equal 50_257",
        );

        // INV-ARCH-370M-008
        assert!(
            !Self::HAS_BIAS,
            "INV-ARCH-370M-008 violated: has_bias must be false (Llama convention)",
        );

        // Sanity: head_dim consistency (free-tier check, also implied
        // by INV-ARCH-370M-002 above).
        assert!(
            Self::HIDDEN_DIM / Self::NUM_HEADS == Self::HEAD_DIM,
            "hidden_dim / num_heads != head_dim — config internally inconsistent",
        );

        // Sanity: max_position_embeddings is a positive multiple of 2.
        assert!(
            Self::MAX_POSITION_EMBEDDINGS > 0 && Self::MAX_POSITION_EMBEDDINGS % 2 == 0,
            "max_position_embeddings must be a positive even integer for RoPE",
        );
    }
}

// Drive `validate()` at crate-compile time. If any `assert!` inside
// `validate()` fails, the crate fails to build.
#[allow(clippy::let_unit_value)]
const _: () = Llama370MConfig::validate();

// ─────────────────────────────────────────────────────────────
// Parameter count estimator (INV-ARCH-370M-001 runtime check)
// ─────────────────────────────────────────────────────────────

/// Estimate the total parameter count for the albor 370M config using
/// the **nominal (untied)** counting convention.
///
/// The contract's INV-ARCH-370M-001 band [366M, 374M] corresponds to the
/// HuggingFace-style reported figure, which counts `lm_head.weight` as
/// a distinct matrix even though — per INV-ARCH-370M-004 — storage is
/// shared with `model.embed_tokens.weight`. This mirrors how Llama
/// families are reported in the literature (e.g., "TinyLlama-1.1B" is
/// counted with untied lm_head even when tied).
///
/// For the actual on-disk param count reported by `apr inspect`
/// (with tying applied), use [`estimated_stored_param_count`].
///
/// Formula (untied — contract reporting convention):
///
/// ```text
/// embedding:           vocab * hidden
/// lm_head:             vocab * hidden   (tied storage, but counted here)
/// per transformer layer:
///   attention q_proj:  (num_heads    * head_dim) * hidden
///   attention k_proj:  (num_kv_heads * head_dim) * hidden
///   attention v_proj:  (num_kv_heads * head_dim) * hidden
///   attention o_proj:  hidden * (num_heads * head_dim)
///   mlp gate_proj:     intermediate * hidden
///   mlp up_proj:       intermediate * hidden
///   mlp down_proj:     hidden * intermediate
///   input_layernorm:   hidden
///   post_attn_layernorm: hidden
/// final rmsnorm:       hidden
/// ```
#[must_use]
pub const fn estimated_param_count() -> usize {
    // Untied: add the lm_head bookkeeping on top of the stored count.
    estimated_stored_param_count() + (Llama370MConfig::VOCAB_SIZE * Llama370MConfig::HIDDEN_DIM)
}

/// Estimate the **stored** parameter count (what `apr inspect` sees on
/// disk for a tied-embedding checkpoint). This is ~51.2M lower than the
/// nominal figure because `lm_head.weight` is aliased to
/// `model.embed_tokens.weight` (INV-ARCH-370M-004).
#[must_use]
pub const fn estimated_stored_param_count() -> usize {
    let h = Llama370MConfig::HIDDEN_DIM;
    let l = Llama370MConfig::NUM_LAYERS;
    let v = Llama370MConfig::VOCAB_SIZE;
    let i = Llama370MConfig::INTERMEDIATE_DIM;
    let nh = Llama370MConfig::NUM_HEADS;
    let nkv = Llama370MConfig::NUM_KV_HEADS;
    let hd = Llama370MConfig::HEAD_DIM;

    // Embedding (tied with lm_head — counted once).
    let embedding = v * h;

    // Attention: q_proj + k_proj + v_proj + o_proj
    let q = h * (nh * hd);
    let k = h * (nkv * hd);
    let vv = h * (nkv * hd);
    let o = (nh * hd) * h;
    let attn = q + k + vv + o;

    // MLP (SwiGLU): gate_proj + up_proj + down_proj
    let mlp = (h * i) + (h * i) + (i * h);

    // Two RMSNorm weights per layer (input_layernorm, post_attention_layernorm).
    let norms = 2 * h;

    let per_layer = attn + mlp + norms;

    // Final model.norm.weight.
    let final_norm = h;

    embedding + l * per_layer + final_norm
}

/// Pure helper that enforces GATE-ARCH-370M-011 / INV-ARCH-370M-006:
/// the tokenizer's vocabulary size MUST exactly match the model's
/// `vocab_size` before pretraining dispatches. Returns `Ok(())` when
/// they match, `Err(String)` with a machine-diffable message when they
/// do not. The caller is expected to surface the error to the user
/// and abort the dispatch before any forward pass.
pub fn assert_tokenizer_vocab_matches_model(
    tokenizer_vocab_size: usize,
    model_vocab_size: usize,
) -> Result<(), String> {
    if tokenizer_vocab_size == model_vocab_size {
        return Ok(());
    }
    Err(format!(
        "GATE-ARCH-370M-011 (INV-ARCH-370M-006) violated: \
         tokenizer vocab_size ({tokenizer_vocab_size}) != model vocab_size \
         ({model_vocab_size}). See contracts/model-families/llama-370m-sovereign-v1.yaml \
         and contracts/tokenizer-bpe-v1.yaml — retrain the tokenizer or amend both contracts \
         in lockstep before resuming pretraining."
    ))
}

/// Polymorphic-path variant of [`assert_tokenizer_vocab_matches_model`] that
/// allows `tokenizer_vocab_size <= model_vocab_size` (per
/// `apr-pretrain-arch-polymorphic-v1` v1.3.0 §qwen_tokenizer_vocab_compatibility,
/// SPEC-SHIP-TWO-001 §55). When fine-tuning from an HF-distributed
/// pretrained checkpoint (Qwen2.5 / Llama2 / Mistral), `tokenizer.json`
/// commonly materializes fewer string-token entries than the model's
/// declared `vocab_size` declares. The gap is reserved/special slots
/// (e.g., `<|reserved_271|>`...) that the lm_head + embedding layers
/// have weights for but no tokenizer string maps to. Strict equality
/// (the §24/§25 from-scratch baseline invariant) is too strict for
/// these models; the relaxed bound preserves the OOB-safety property:
///
/// **Safety argument**: tokenizer.encode(text) → ids ∈ [0, tokenizer_vocab).
/// Embedding/lm_head layers are sized for [0, model_vocab). When
/// tokenizer_vocab ≤ model_vocab, every encoded id is in-bounds; the
/// reserved high-id slots are never indexed at training time. When
/// tokenizer_vocab > model_vocab, the encoder could emit ids ≥ model_vocab
/// → N-09 OOB → silent garbage gradients → fail-fast.
///
/// Returns `Ok(())` when bound holds, `Err(String)` with a machine-diffable
/// message when violated.
pub fn assert_tokenizer_vocab_within_model_bound(
    tokenizer_vocab_size: usize,
    model_vocab_size: usize,
) -> Result<(), String> {
    if tokenizer_vocab_size <= model_vocab_size {
        return Ok(());
    }
    Err(format!(
        "GATE-ARCH-370M-011 (INV-ARCH-370M-006-RELAXED) violated: \
         tokenizer vocab_size ({tokenizer_vocab_size}) > model vocab_size \
         ({model_vocab_size}). See contracts/apr-pretrain-arch-polymorphic-v1.yaml \
         §qwen_tokenizer_vocab_compatibility — for HF-distributed pretrained \
         checkpoints, tokenizer_vocab MUST be <= model_vocab (reserved-slot \
         tolerance); a tokenizer with MORE strings than the model expects \
         would emit OOB ids → N-09 escape → silent garbage gradients."
    ))
}

// ─────────────────────────────────────────────────────────────
// FALSIFY-SHIP-017 / AC-SHIP2-007 / GATE-ARCH-370M-005
// ─────────────────────────────────────────────────────────────

/// Number of held-out prompts required by AC-SHIP2-007 / FALSIFY-SHIP-017
/// (spec §6 Model 2: "`apr run` produces syntactically valid Python on
/// 100 held-out prompts").
pub const AC_SHIP2_007_HELDOUT_PROMPT_COUNT: usize = 100;

/// Tolerance: `≤ 1` completion out of the 100 held-out prompts may fail
/// to yield a Python-AST-parseable non-trivial statement prefix.
/// Anything `≥ 2` is a ship-blocking FAIL per FALSIFY-SHIP-017.
pub const AC_SHIP2_007_MAX_TOLERATED_SYNTAX_ERRORS: usize = 1;

/// Ship-017 verdict — the pure algorithmic result of evaluating whether
/// a 100-prompt Python-AST-parse sweep meets AC-SHIP2-007.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ship017Verdict {
    /// `syntax_errors ≤ AC_SHIP2_007_MAX_TOLERATED_SYNTAX_ERRORS`.
    Pass,
    /// `syntax_errors ≥ 2` on the 100-prompt sweep.
    Fail,
}

/// Pure threshold function for FALSIFY-SHIP-017 (AC-SHIP2-007).
///
/// Given `syntax_errors` — the count of held-out completions (out of
/// `AC_SHIP2_007_HELDOUT_PROMPT_COUNT`) that failed to yield a
/// Python-AST-parseable non-trivial statement prefix — returns the
/// FALSIFY-SHIP-017 verdict under the rule:
///
///   errors ≤ 1 → Pass   (tolerate one flaky completion)
///   errors ≥ 2 → Fail   (ship-blocker)
///
/// This is the algorithm-level discharge of GATE-ARCH-370M-005: the
/// threshold itself is proven correct at `cargo test` time; full
/// discharge (a real 100-prompt `apr run` harness against a trained
/// 370M .apr) remains PENDING on pretraining compute-dispatch
/// (AC-SHIP2-003/004). Fixture swap is data-only once a trained
/// artifact exists — no harness rewrite required.
pub const fn verdict_from_syntax_error_count(syntax_errors: usize) -> Ship017Verdict {
    if syntax_errors <= AC_SHIP2_007_MAX_TOLERATED_SYNTAX_ERRORS {
        Ship017Verdict::Pass
    } else {
        Ship017Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// FALSIFY-SHIP-020 (AC-SHIP2-010): decode-throughput threshold
// ─────────────────────────────────────────────────────────────

/// Minimum `apr run` median decode throughput, in tokens/second, on the
/// SHIP-TWO-001 reference hardware (RTX 4090). Source of truth:
/// `contracts/model-families/llama-370m-sovereign-v1.yaml`
/// GATE-ARCH-370M-006 and the SHIP-TWO-001 falsification table
/// (FALSIFY-SHIP-020).
pub const AC_SHIP2_010_MIN_DECODE_TPS_RTX4090: f32 = 100.0;

/// Verdict for GATE-ARCH-370M-006 / FALSIFY-SHIP-020: does the measured
/// `apr bench` median decode throughput meet the ship-gate threshold?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ship020Verdict {
    /// Measured tok/s ≥ [`AC_SHIP2_010_MIN_DECODE_TPS_RTX4090`].
    Pass,
    /// Measured tok/s < [`AC_SHIP2_010_MIN_DECODE_TPS_RTX4090`] (ship-blocker).
    Fail,
}

/// Pure threshold function implementing FALSIFY-SHIP-020: a measured
/// median decode throughput of `measured_tps` tokens/second passes the
/// ship-gate iff it is finite AND ≥ [`AC_SHIP2_010_MIN_DECODE_TPS_RTX4090`].
///
/// Non-finite inputs (NaN, ±∞) are conservatively classified as
/// `Fail`: a benchmark run that could not produce a well-defined
/// finite median is always ill-formed and must never be allowed to
/// silently green the ship-gate.
///
/// This is the same decision-rule-vs-harness separation used by
/// FALSIFY-SHIP-017 (syntax-error count). The compute-heavy part
/// (`apr bench --median` on a real trained 370M .apr) is deferred to
/// AC-SHIP2-003/004 compute-dispatch; the decision rule itself is
/// proven today at unit-test time.
#[must_use]
pub fn verdict_from_decode_tps(measured_tps: f32) -> Ship020Verdict {
    if !(measured_tps.is_finite()) {
        return Ship020Verdict::Fail;
    }
    if measured_tps >= AC_SHIP2_010_MIN_DECODE_TPS_RTX4090 {
        Ship020Verdict::Pass
    } else {
        Ship020Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// AC-SHIP2-008 / FALSIFY-SHIP-018 — HumanEval pass@1 threshold
// ─────────────────────────────────────────────────────────────
//
// Spec §5.2 AC-SHIP2-008 states: `apr eval --benchmark humaneval`
// must report pass@1 ≥ 30.0% for the trained 370M `.apr` artifact.
// The decision rule is a pure (correct, total) → pct comparison;
// the compute-heavy harness (164 HumanEval tasks × `apr run` ×
// greedy sampling) is fixture-swappable once a trained artifact
// exists. Landing the threshold function today discharges
// GATE-ARCH-370M-007 at PARTIAL_ALGORITHM_LEVEL and catches any
// future drift in the 30.0 floor at `cargo test` time — before a
// single HumanEval task is dispatched.

/// Contract-frozen floor for AC-SHIP2-008: HumanEval pass@1 must
/// reach **30.0%** on the trained 370M artifact. The numeric floor
/// mirrors `contracts/model-families/llama-370m-sovereign-v1.yaml`
/// GATE-ARCH-370M-007 rule body and spec §5.2 AC-SHIP2-008.
pub const AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT: f32 = 30.0;

/// Binary verdict emitted by [`verdict_from_pass_at_1`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ship018Verdict {
    /// `correct / total * 100 ≥ threshold_pct` and inputs are well-formed.
    Pass,
    /// Any of: ratio < threshold, `total == 0`, non-finite threshold.
    Fail,
}

/// Decision function for AC-SHIP2-008 / FALSIFY-SHIP-018.
///
/// Returns [`Ship018Verdict::Pass`] iff `correct / total * 100`
/// is ≥ `threshold_pct`. Conservative-Fail guards:
///
///   - `total == 0` → Fail (avoid division-by-zero; an empty run
///     cannot satisfy a positive threshold).
///   - `correct > total` → Fail (nonsensical input; a real
///     harness can never report more passes than attempts).
///   - `!threshold_pct.is_finite()` → Fail (NaN / ±∞ contract
///     drift — no real contract floor can be non-finite).
///
/// Stored per-ULP ratio comparisons use f32 arithmetic to keep the
/// verdict a pure function of its inputs with no harness state.
#[must_use]
pub fn verdict_from_pass_at_1(correct: usize, total: usize, threshold_pct: f32) -> Ship018Verdict {
    if total == 0 {
        return Ship018Verdict::Fail;
    }
    if correct > total {
        return Ship018Verdict::Fail;
    }
    if !threshold_pct.is_finite() {
        return Ship018Verdict::Fail;
    }
    // Safe: correct ≤ total ≤ usize::MAX; f32 cast preserves ordering
    // within the humaneval range (total ≤ 164 for canonical HumanEval).
    let ratio_pct = (correct as f32 / total as f32) * 100.0_f32;
    if ratio_pct >= threshold_pct {
        Ship018Verdict::Pass
    } else {
        Ship018Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// FALSIFY-SHIP-016 / AC-SHIP2-006 / GATE-ARCH-370M-008 — apr qa aggregate
// ─────────────────────────────────────────────────────────────

/// Number of canonical `apr qa` gates composing AC-SHIP2-006 /
/// FALSIFY-SHIP-016. The spec row AC-SHIP2-006 reads
/// "`apr qa <model>.apr` — all 8 gates PASS" and the matching
/// FALSIFY row names the decision rule as "any gate FAIL" →
/// Fail. The 8 canonical gates (matching `QaConfig::skip_*` in
/// `crates/apr-cli/src/commands/qa.rs`) are:
///   1. golden_output      — correctness gate (GH-202 regression)
///   2. throughput         — tok/s ≥ configured floor
///   3. ollama_parity      — speedup vs Ollama baseline
///   4. gpu_vs_cpu_speedup — GPU ≥ 2× CPU (F-PERF-042)
///   5. tensor_contract    — layout/dtype/shape validation
///   6. cross_format_parity — argmax(GGUF) == argmax(SafeTensors)
///   7. ptx_parity         — batched GPU kernels vs scalar refs
///   8. probar             — property-based tests
///
/// Contract-drift guard: any change to this number must be matched
/// in lockstep across the contract, spec, and CLI qa handler.
pub const AC_SHIP2_006_REQUIRED_QA_GATE_COUNT: usize = 8;

/// Ternary verdict for FALSIFY-SHIP-016 / GATE-ARCH-370M-008.
/// `Pass` iff every one of the 8 canonical `apr qa` gates passed.
/// `Fail` otherwise (any single gate failure, or wrong gate count).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ship016Verdict {
    Pass,
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-SHIP-016 / GATE-ARCH-370M-008
/// / AC-SHIP2-006: `apr qa <model>.apr` is an aggregate-AND over the
/// 8 canonical QA gates. Verdict is `Pass` iff **exactly 8** gate
/// results were supplied **and every one passed**. Any shorter/longer
/// slice is a contract-drift Fail; any single `false` entry is a
/// ship-blocking Fail. This proves the decision rule without running
/// the compute-heavy gates themselves; the harness (realizar, cuda,
/// tokenizer, corpus) is fixture-swappable once a real trained 370M
/// .apr exists. Spec §7 row FALSIFY-SHIP-016 ("any gate FAIL") is
/// the counter-example this fn is built to classify.
#[must_use]
pub fn verdict_from_qa_gates(gate_results: &[bool]) -> Ship016Verdict {
    if gate_results.len() != AC_SHIP2_006_REQUIRED_QA_GATE_COUNT {
        return Ship016Verdict::Fail;
    }
    if gate_results.iter().all(|&passed| passed) {
        Ship016Verdict::Pass
    } else {
        Ship016Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// FALSIFY-SHIP-013 / AC-SHIP2-003 / GATE-ARCH-370M-013 — val CE loss floor
// ─────────────────────────────────────────────────────────────
//
// AC-SHIP2-003 (spec §5.2) states: "`entrenar` pretraining loop reaches
// target loss (CE ≤ 2.2 on val)". The decision rule is a pure single-
// number f32 threshold check on a measured cross-entropy value. Cross-
// entropy is non-negative by definition (H(p,q) ≥ 0 for all probability
// distributions p,q), so the admissible input domain is `[0.0, +∞)`; any
// negative measurement indicates a broken loss harness (sign flip,
// log-domain underflow, subtract instead of add) and must Fail closed.
//
// This PARTIAL_ALGORITHM_LEVEL discharge binds the decision rule only.
// The compute-heavy half — an actual `apr pretrain --validate` loop
// producing a live val CE from MODEL-2 training — remains blocked on
// AC-SHIP2-003 compute-dispatch. The decision rule itself (what number
// constitutes a Pass) is proven today at `cargo test` time via the
// mutation survey below.

/// Maximum acceptable cross-entropy loss on the validation set for
/// MODEL-2 (albor 370M Sovereign) at the end of pretraining. Spec §5.2
/// AC-SHIP2-003: "CE ≤ 2.2 on val". A measured val CE strictly above
/// 2.2 is a ship-blocker (training did not converge well enough for
/// the 370M target to hit its HumanEval / syntax-parse downstream
/// gates). Pinned here so any contract drift in either direction
/// (loosening to 2.5, hardening to 2.0) is caught at compile+test
/// time, not at a production training run.
///
/// f32-exact: `2.2` is representable in IEEE-754 binary32 as the
/// closest-round value `0x400CCCCD` = 2.20000004768371582...; the
/// ULP-above neighbour is strictly greater and is used as the
/// sharpest Fail counter-example in the mutation survey.
pub const AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS: f32 = 2.2;

/// Binary verdict for FALSIFY-SHIP-013 / AC-SHIP2-003 /
/// GATE-ARCH-370M-013. `Pass` iff the measured val CE is finite AND
/// non-negative AND at or below [`AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS`].
/// `Fail` otherwise (including every non-finite value: NaN, +∞, -∞,
/// and every negative value, which is a harness-bug domain violation
/// since H(p,q) ≥ 0 by definition).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ship013Verdict {
    /// Measured val CE ∈ `[0.0, AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS]`.
    Pass,
    /// Measured val CE > ceiling, or non-finite, or negative (domain
    /// violation — a real cross-entropy can never be < 0).
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-SHIP-013 / AC-SHIP2-003 /
/// GATE-ARCH-370M-013: a single f32 threshold check against the MODEL-2
/// val-CE ceiling. Returns [`Ship013Verdict::Fail`] conservatively for
/// any non-finite input (NaN, +∞, -∞) AND for any negative input (CE
/// is ≥ 0 by definition — a negative value means the loss harness is
/// broken, which must never be silently promoted to a Pass).
///
/// The full discharge (live `apr pretrain --validate` loop producing
/// a real MODEL-2 val CE on RTX 4090 ≤ 2.2) remains blocked on
/// AC-SHIP2-003 compute-dispatch.
#[must_use]
pub const fn verdict_from_val_ce_loss(measured_ce: f32) -> Ship013Verdict {
    if !measured_ce.is_finite() {
        return Ship013Verdict::Fail;
    }
    if measured_ce < 0.0 {
        // Cross-entropy is non-negative by definition; a negative
        // measurement is a harness bug, not a "better than zero" Pass.
        return Ship013Verdict::Fail;
    }
    if measured_ce <= AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS {
        Ship013Verdict::Pass
    } else {
        Ship013Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// FALSIFY-SHIP-014 / AC-SHIP2-004 / GATE-ARCH-370M-014 — training-budget floor
// ─────────────────────────────────────────────────────────────
//
// AC-SHIP2-004 (spec §5.2) states: "Training on RTX 4090 completes
// within 21 days (hardware budget)". The decision rule is a pure
// single-number u32 threshold check on measured wall-clock days. u32
// naturally rules out negative values (no need for an explicit domain
// guard); the only extrema to cover are the boundary (21), the
// immediate neighbours (20 / 22), and the saturation point u32::MAX.
// Zero days is a trivial Pass (under budget).
//
// This PARTIAL_ALGORITHM_LEVEL discharge binds the decision rule only.
// The compute-heavy half — an actual wall-clock measurement of a real
// 370M pretraining run on RTX 4090 → ≤ 21 days — remains blocked on
// AC-SHIP2-004 compute-dispatch. The decision rule itself (what
// duration constitutes a Pass) is proven today at `cargo test` time
// via the mutation survey below.

/// Maximum acceptable wall-clock training duration, in integer days,
/// for MODEL-2 (albor 370M Sovereign) on the SHIP-TWO-001 reference
/// RTX 4090 host. Spec §5.2 AC-SHIP2-004: "Training on RTX 4090
/// completes within 21 days (hardware budget)". A measured duration
/// strictly above 21 days is a ship-blocker (the hardware budget
/// overflowed and a 2× H100 week-3 escape hatch from Spec §9 Risk #4
/// is required). Pinned here so any contract drift in either
/// direction (extending to 30, compressing to 14) is caught at
/// compile+test time, not mid-pretraining.
pub const AC_SHIP2_004_MAX_TRAINING_DURATION_DAYS: u32 = 21;

/// Binary verdict for FALSIFY-SHIP-014 / AC-SHIP2-004 /
/// GATE-ARCH-370M-014. `Pass` iff the measured wall-clock training
/// duration in days is at or below
/// [`AC_SHIP2_004_MAX_TRAINING_DURATION_DAYS`]. `Fail` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ship014Verdict {
    /// Measured training duration ≤ `AC_SHIP2_004_MAX_TRAINING_DURATION_DAYS`.
    Pass,
    /// Measured training duration > ceiling (ship-blocker per Spec §9
    /// Risk #4).
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-SHIP-014 / AC-SHIP2-004 /
/// GATE-ARCH-370M-014: a single u32 threshold check against the
/// MODEL-2 hardware-budget ceiling. Unlike the f32 ship gates
/// (SHIP-007, SHIP-013, SHIP-020), u32 automatically rules out
/// negatives (no domain guard needed) and has no non-finite states;
/// the only interesting counter-examples are the boundary (21), the
/// immediate neighbours (20 / 22), and u32::MAX.
///
/// Zero days is Pass (trivially under-budget, e.g. a cached artifact
/// or a same-day re-run). The full discharge (live wall-clock
/// measurement of MODEL-2 pretraining on RTX 4090 ≤ 21 days) remains
/// blocked on AC-SHIP2-004 compute-dispatch.
#[must_use]
pub const fn verdict_from_training_duration_days(measured_days: u32) -> Ship014Verdict {
    if measured_days <= AC_SHIP2_004_MAX_TRAINING_DURATION_DAYS {
        Ship014Verdict::Pass
    } else {
        Ship014Verdict::Fail
    }
}

// ─────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// INV-ARCH-370M-002/003/004/005/006/008 — byte-equality with contract.
    #[test]
    fn config_matches_contract_values() {
        // §architecture
        assert_eq!(Llama370MConfig::HIDDEN_DIM, 1024);
        assert_eq!(Llama370MConfig::NUM_LAYERS, 24);
        assert_eq!(Llama370MConfig::NUM_HEADS, 16);
        assert_eq!(Llama370MConfig::NUM_KV_HEADS, 4);
        assert_eq!(Llama370MConfig::HEAD_DIM, 64);
        assert_eq!(Llama370MConfig::INTERMEDIATE_DIM, 2816);
        assert_eq!(Llama370MConfig::VOCAB_SIZE, 50_257);
        assert_eq!(Llama370MConfig::MAX_POSITION_EMBEDDINGS, 4096);
        assert!((Llama370MConfig::ROPE_THETA - 10_000.0_f32).abs() < 1e-6);
        assert!((Llama370MConfig::RMS_NORM_EPS - 1.0e-5_f32).abs() < 1e-9);

        // §constraints
        assert!(Llama370MConfig::TIED_EMBEDDINGS);
        assert!(!Llama370MConfig::HAS_BIAS);

        // Derived: INV-ARCH-370M-002 & 003
        assert_eq!(
            Llama370MConfig::NUM_HEADS * Llama370MConfig::HEAD_DIM,
            Llama370MConfig::HIDDEN_DIM,
        );
        assert_eq!(Llama370MConfig::NUM_HEADS % Llama370MConfig::NUM_KV_HEADS, 0);
    }

    /// GATE-ARCH-370M-011 / INV-ARCH-370M-006 — pure vocab-parity helper
    /// MUST reject any mismatch between tokenizer vocab_size and model
    /// vocab_size, and MUST accept equal values. The real-compute MODEL-2
    /// dispatch at commit 29607ed33 surfaced this when a tokenizer at
    /// vocab=50_257 was paired with a model pinned at VOCAB_SIZE=50_000;
    /// the N-09 OOB escape masked the mismatch → garbage gradients.
    /// Task #131 bumped VOCAB_SIZE to 50_257 (Option A); the counter-example
    /// value below now exercises the opposite drift (a tokenizer one token
    /// short of contract) so the helper is still exercised on real mismatch.
    #[test]
    fn falsify_gate_arch_370m_011_helper_rejects_mismatch() {
        assert!(assert_tokenizer_vocab_matches_model(
            Llama370MConfig::VOCAB_SIZE,
            Llama370MConfig::VOCAB_SIZE,
        )
        .is_ok());

        let mismatch = Llama370MConfig::VOCAB_SIZE - 1;
        let err = assert_tokenizer_vocab_matches_model(mismatch, Llama370MConfig::VOCAB_SIZE)
            .expect_err("mismatch must return Err");
        assert!(
            err.contains("GATE-ARCH-370M-011")
                && err.contains(&mismatch.to_string())
                && err.contains(&Llama370MConfig::VOCAB_SIZE.to_string()),
            "error must name the gate and both vocab sizes for forensics, got: {err}",
        );

        assert!(assert_tokenizer_vocab_matches_model(0, 1).is_err());
        assert!(assert_tokenizer_vocab_matches_model(
            Llama370MConfig::VOCAB_SIZE + 1,
            Llama370MConfig::VOCAB_SIZE
        )
        .is_err());
    }

    /// FALSIFY-APR-PRETRAIN-ARCH-009 (§55) — relaxed-bound helper accepts
    /// `tokenizer_vocab <= model_vocab` for the polymorphic init path.
    /// Discharges the §55 finding: HF-distributed Qwen2.5-Coder-0.5B
    /// materializes 151643 BPE entries + 22 added = 151665 effective
    /// strings, but config.json declares `vocab_size = 151936` (271
    /// reserved slots). The relaxed bound preserves OOB safety while
    /// admitting the standard HF reserved-slot pattern.
    #[test]
    fn falsify_apr_pretrain_arch_009_relaxed_bound_accepts_qwen_reserved_slots() {
        // The exact §55 LIVE smoke shape: 151665 ≤ 151936 must pass.
        const QWEN_BPE_PLUS_ADDED: usize = 151_665;
        const QWEN_DECLARED_VOCAB: usize = 151_936;
        assert!(
            assert_tokenizer_vocab_within_model_bound(
                QWEN_BPE_PLUS_ADDED,
                QWEN_DECLARED_VOCAB
            )
            .is_ok(),
            "FALSIFY-APR-PRETRAIN-ARCH-009: tokenizer 151665 ≤ model 151936 MUST pass relaxed bound \
             (HF reserved-slot tolerance)"
        );

        // BPE-only count (without --include-added-tokens): 151643 ≤ 151936 must also pass.
        const QWEN_BPE_ONLY: usize = 151_643;
        assert!(
            assert_tokenizer_vocab_within_model_bound(QWEN_BPE_ONLY, QWEN_DECLARED_VOCAB).is_ok()
        );

        // Equality remains acceptable (Llama370M-from-scratch case).
        assert!(assert_tokenizer_vocab_within_model_bound(
            Llama370MConfig::VOCAB_SIZE,
            Llama370MConfig::VOCAB_SIZE
        )
        .is_ok());
    }

    /// FALSIFY-APR-PRETRAIN-ARCH-010 (§55) — relaxed-bound helper REJECTS
    /// `tokenizer_vocab > model_vocab`. This is the OOB-safety guard:
    /// a tokenizer producing ids ≥ model_vocab would silently corrupt
    /// embedding lookup. Strict-greater MUST fail-fast.
    #[test]
    fn falsify_apr_pretrain_arch_010_relaxed_bound_rejects_oversized_tokenizer() {
        const QWEN_DECLARED_VOCAB: usize = 151_936;
        let oversized = QWEN_DECLARED_VOCAB + 1;
        let err =
            assert_tokenizer_vocab_within_model_bound(oversized, QWEN_DECLARED_VOCAB)
                .expect_err("FALSIFY-APR-PRETRAIN-ARCH-010: tokenizer > model MUST fail-fast");
        assert!(
            err.contains("RELAXED") && err.contains("OOB"),
            "error must cite the relaxed-mode invariant + OOB risk: {err}"
        );
        assert!(
            err.contains(&oversized.to_string())
                && err.contains(&QWEN_DECLARED_VOCAB.to_string()),
            "error must name both sizes for forensics: {err}"
        );
    }

    /// INV-ARCH-370M-001 — estimated param count within [366M, 374M].
    ///
    /// Recomputes the canonical transformer param formula and asserts the
    /// answer lies in the ±1% band the contract permits for the final
    /// trained artifact.
    #[test]
    fn estimated_param_count_within_contract_band() {
        let p = estimated_param_count();
        let stored = estimated_stored_param_count();

        // Sanity printout for debugging drift.
        eprintln!("albor-370m nominal param count = {p} ({} M)", p / 1_000_000,);
        eprintln!(
            "albor-370m stored  param count = {stored} ({} M, lm_head tied)",
            stored / 1_000_000,
        );

        // INV-ARCH-370M-001 — nominal ±1% band.
        assert!(
            p >= Llama370MConfig::PARAMETERS_MIN,
            "nominal param count {p} below INV-ARCH-370M-001 floor (366M)",
        );
        assert!(
            p <= Llama370MConfig::PARAMETERS_MAX,
            "nominal param count {p} above INV-ARCH-370M-001 ceiling (374M)",
        );

        // Tighter ±5% sanity band around the 370M nominal figure, per
        // this scaffold's unit-test requirements.
        let nominal = Llama370MConfig::PARAMETERS_NOMINAL as f64;
        let pct = (p as f64 - nominal).abs() / nominal;
        assert!(
            pct < 0.05,
            "nominal param count {p} differs from 370M by {:.2}% (> 5%)",
            pct * 100.0,
        );

        // Tying must reduce storage by exactly one vocab*hidden matrix.
        assert_eq!(
            p - stored,
            Llama370MConfig::VOCAB_SIZE * Llama370MConfig::HIDDEN_DIM,
            "tying accounting mismatch",
        );
    }

    /// Sanity: the compile-time `validate()` matches the runtime check.
    #[test]
    fn validate_is_a_noop_at_runtime() {
        // If `validate()` compiled, it's already been proven to not panic
        // (the `const _: () = ...;` at module scope forced evaluation at
        // compile time). Calling it again at runtime is a free
        // defence-in-depth assertion.
        Llama370MConfig::validate();
    }

    /// Shape newtypes are zero-sized and usable in generic contexts.
    #[test]
    fn shape_newtypes_compile_and_roundtrip() {
        type Hidden = HiddenDim<{ Llama370MConfig::HIDDEN_DIM }>;
        type Heads = NumHeads<{ Llama370MConfig::NUM_HEADS }>;
        type KvHeads = NumKvHeads<{ Llama370MConfig::NUM_KV_HEADS }>;
        type Head = HeadDim<{ Llama370MConfig::HEAD_DIM }>;
        type Inter = IntermediateDim<{ Llama370MConfig::INTERMEDIATE_DIM }>;
        type Layers = NumLayers<{ Llama370MConfig::NUM_LAYERS }>;
        type Vocab = VocabSize<{ Llama370MConfig::VOCAB_SIZE }>;

        assert_eq!(Hidden::VALUE, 1024);
        assert_eq!(Heads::VALUE, 16);
        assert_eq!(KvHeads::VALUE, 4);
        assert_eq!(Head::VALUE, 64);
        assert_eq!(Inter::VALUE, 2816);
        assert_eq!(Layers::VALUE, 24);
        assert_eq!(Vocab::VALUE, 50_257);

        // Zero-sized: all shape newtypes cost nothing at runtime.
        assert_eq!(std::mem::size_of::<Hidden>(), 0);
        assert_eq!(std::mem::size_of::<Heads>(), 0);
    }

    // ========================================================================
    // C-LLAMA-370M-SOVEREIGN / AC-SHIP2-001 / FALSIFY-SHIP-011
    // ========================================================================

    /// The sovereign contract YAML embedded at compile time so the test
    /// binary has a byte-frozen copy — any edit to the file is caught
    /// by the next test run, not discovered post-publish.
    const SOVEREIGN_CONTRACT_YAML: &str =
        include_str!("../../../../contracts/model-families/llama-370m-sovereign-v1.yaml");

    /// GATE-ARCH-370M-001 / INV-ARCH-370M-002..008: every architectural
    /// constant declared in `contracts/model-families/llama-370m-sovereign-v1.yaml`
    /// matches the Rust scaffold `Llama370MConfig::*` const byte-equally.
    ///
    /// Discharges FALSIFY-SHIP-011 (AC-SHIP2-001): architecture registered
    /// in a llama-family contract entry whose dimensions validate against
    /// `contracts/model-families/_schema.yaml` AND match the compile-time
    /// Rust config that the training loop will actually consume. Binds the
    /// YAML contract and the Rust scaffold: if either drifts without the
    /// other, this test fails — catching the MODEL-1 QLoRA class of
    /// recipe/artifact drift at `cargo test` time, before a single step
    /// of pretraining compute runs.
    #[test]
    fn falsify_ship_011_rust_scaffold_matches_yaml_contract() {
        let doc: serde_yaml::Value = serde_yaml::from_str(SOVEREIGN_CONTRACT_YAML)
            .expect("llama-370m-sovereign-v1.yaml must parse as YAML");

        // Contract identity — must be the right contract.
        assert_eq!(
            doc["contract_id"].as_str(),
            Some("C-LLAMA-370M-SOVEREIGN"),
            "wrong contract loaded — check include_str! path",
        );
        assert_eq!(doc["family"].as_str(), Some("llama"));
        assert_eq!(doc["size_variant"].as_str(), Some("370m"));

        // Architectural dimensions (INV-ARCH-370M-002, -003, -005, -006).
        let arch = &doc["architecture"];
        assert_eq!(
            arch["hidden_dim"].as_u64().map(|v| v as usize),
            Some(Llama370MConfig::HIDDEN_DIM),
            "YAML architecture.hidden_dim drifted from Rust const",
        );
        assert_eq!(
            arch["num_layers"].as_u64().map(|v| v as usize),
            Some(Llama370MConfig::NUM_LAYERS),
        );
        assert_eq!(
            arch["num_heads"].as_u64().map(|v| v as usize),
            Some(Llama370MConfig::NUM_HEADS),
        );
        assert_eq!(
            arch["num_kv_heads"].as_u64().map(|v| v as usize),
            Some(Llama370MConfig::NUM_KV_HEADS),
        );
        assert_eq!(arch["head_dim"].as_u64().map(|v| v as usize), Some(Llama370MConfig::HEAD_DIM),);
        assert_eq!(
            arch["intermediate_dim"].as_u64().map(|v| v as usize),
            Some(Llama370MConfig::INTERMEDIATE_DIM),
        );
        assert_eq!(
            arch["vocab_size"].as_u64().map(|v| v as usize),
            Some(Llama370MConfig::VOCAB_SIZE),
        );
        assert_eq!(
            arch["max_position_embeddings"].as_u64().map(|v| v as usize),
            Some(Llama370MConfig::MAX_POSITION_EMBEDDINGS),
        );
        let rope_theta = arch["rope_theta"].as_f64().expect("rope_theta must be a float");
        assert!(
            (rope_theta - f64::from(Llama370MConfig::ROPE_THETA)).abs() < 1e-6,
            "YAML rope_theta {rope_theta} != Rust const {}",
            Llama370MConfig::ROPE_THETA,
        );

        // Constraints (INV-ARCH-370M-004, -008).
        let constraints = &doc["constraints"];
        assert_eq!(
            constraints["tied_embeddings"].as_bool(),
            Some(Llama370MConfig::TIED_EMBEDDINGS),
        );
        assert_eq!(constraints["has_bias"].as_bool(), Some(Llama370MConfig::HAS_BIAS),);
        assert_eq!(constraints["attention_type"].as_str(), Some("gqa"));
        assert_eq!(constraints["activation"].as_str(), Some("silu"));
        assert_eq!(constraints["norm_type"].as_str(), Some("rmsnorm"));
        assert_eq!(constraints["positional_encoding"].as_str(), Some("rope"));
        assert_eq!(constraints["mlp_type"].as_str(), Some("swiglu"));
    }

    /// GATE-ARCH-370M-001 (gate status): once FALSIFY-SHIP-011 is
    /// discharged, the sovereign contract MUST declare status ACTIVE —
    /// a PROPOSED gate cannot be a ship-blocker.
    #[test]
    fn falsify_ship_011_sovereign_contract_is_active() {
        let doc: serde_yaml::Value =
            serde_yaml::from_str(SOVEREIGN_CONTRACT_YAML).expect("parse sovereign contract");
        assert_eq!(
            doc["status"].as_str(),
            Some("ACTIVE"),
            "C-LLAMA-370M-SOVEREIGN must be ACTIVE once FALSIFY-SHIP-011 \
             discharges — PROPOSED contracts cannot gate a ship",
        );
    }

    // ========================================================================
    // GATE-ARCH-370M-004 / AC-SHIP2-009 / FALSIFY-SHIP-019
    // ========================================================================

    /// Enumerate every APR tensor name the 370M architecture produces.
    ///
    /// Returns `(name, expected_shape)` pairs. Ordering mirrors the
    /// canonical GGUF/APR dump order: embedding → per-layer tensors
    /// (24 layers × 9 tensors) → final norm. `lm_head.weight` shares
    /// storage with `model.embed_tokens.weight` per INV-ARCH-370M-004
    /// (tied), but the layout contract records it as a separate entry
    /// because the kernel path needs a named row-major [vocab, hidden]
    /// reference at decode time.
    fn enumerate_370m_apr_tensors() -> Vec<(String, Vec<usize>)> {
        let h = Llama370MConfig::HIDDEN_DIM;
        let v = Llama370MConfig::VOCAB_SIZE;
        let i = Llama370MConfig::INTERMEDIATE_DIM;
        let nh = Llama370MConfig::NUM_HEADS;
        let nkv = Llama370MConfig::NUM_KV_HEADS;
        let hd = Llama370MConfig::HEAD_DIM;
        let layers = Llama370MConfig::NUM_LAYERS;

        let mut out: Vec<(String, Vec<usize>)> = Vec::with_capacity(3 + 9 * layers);
        out.push(("model.embed_tokens.weight".into(), vec![v, h]));
        out.push(("lm_head.weight".into(), vec![v, h]));
        for n in 0..layers {
            out.push((format!("model.layers.{n}.self_attn.q_proj.weight"), vec![nh * hd, h]));
            out.push((format!("model.layers.{n}.self_attn.k_proj.weight"), vec![nkv * hd, h]));
            out.push((format!("model.layers.{n}.self_attn.v_proj.weight"), vec![nkv * hd, h]));
            out.push((format!("model.layers.{n}.self_attn.o_proj.weight"), vec![h, nh * hd]));
            out.push((format!("model.layers.{n}.mlp.gate_proj.weight"), vec![i, h]));
            out.push((format!("model.layers.{n}.mlp.up_proj.weight"), vec![i, h]));
            out.push((format!("model.layers.{n}.mlp.down_proj.weight"), vec![h, i]));
            out.push((format!("model.layers.{n}.input_layernorm.weight"), vec![h]));
            out.push((format!("model.layers.{n}.post_attention_layernorm.weight"), vec![h]));
        }
        out.push(("model.norm.weight".into(), vec![h]));
        out
    }

    /// FALSIFY-SHIP-019 (AC-SHIP2-009) — algorithm-level PARTIAL proof
    /// that every APR tensor the 370M architecture produces is covered
    /// by `aprender::format::layout_contract` (the authoritative
    /// row-major validator reused by every GGUF↔APR export site, per
    /// spec §9 Risk #2 mitigation).
    ///
    /// This test proves three things without needing a trained model:
    ///   1. **Coverage:** every 370M tensor name normalises to a
    ///      contract entry — no unknown-tensor silent-skip gap.
    ///   2. **Row-major ordering:** every 2D tensor's enumerated shape
    ///      is `[out_dim, in_dim]` (the row-major APR layout mandated
    ///      by INV-ARCH-370M-009 and by LAYOUT-001). Specifically
    ///      `lm_head.weight` is `[vocab, hidden]`, never reversed —
    ///      GH-202 root cause.
    ///   3. **Critical-tensor enforcement:** `validate_apr_shape` on
    ///      `lm_head.weight` accepts `[vocab, hidden]` AND rejects
    ///      `[hidden, vocab]`, proving the validator actively catches
    ///      the GH-202 class of layout bug.
    ///
    /// **Discharge:** `evidence_discharged_by` on GATE-ARCH-370M-004;
    /// full discharge blocks on real trained 370M artifact (need the
    /// GGUF export path to actually invoke `validate_apr_shape` on
    /// real tensor bytes, which requires a trained `.apr`).
    #[test]
    fn falsify_ship_019_layout_contract_covers_every_370m_tensor() {
        use aprender::format::layout_contract::LayoutContract;
        let contract = LayoutContract::new();
        let tensors = enumerate_370m_apr_tensors();

        // Invariant 1: the enumerator produces exactly the expected number
        // of APR entries for a 24-layer 370M Llama (1 embedding + 1 lm_head
        // + 9 per-layer + 1 final norm).
        assert_eq!(
            tensors.len(),
            3 + 9 * Llama370MConfig::NUM_LAYERS,
            "370M enumerator produced wrong tensor count — scaffold drift",
        );

        // Invariant 2: coverage — every enumerated name resolves to a
        // TensorContract entry. Pattern-normalisation collapses
        // `model.layers.<n>.*` to `model.layers.{n}.*`.
        for (name, _) in &tensors {
            assert!(
                contract.get_apr_contract(name).is_some(),
                "370M tensor `{name}` has no layout_contract entry — \
                 LAYOUT-001 coverage gap (every tensor in this model must \
                 pattern-match a TensorContract or GGUF export layout will \
                 silently skip it)",
            );
        }

        // Invariant 3: row-major ordering — every 2D tensor enumerated
        // above has shape `[out_dim, in_dim]`. The ordering is the whole
        // point of LAYOUT-001 (see layout_contract.rs §Key Principles).
        // Spot-check the pinned invariants rather than re-parsing the
        // formula strings.
        let lm = tensors
            .iter()
            .find(|(n, _)| n == "lm_head.weight")
            .expect("lm_head must be enumerated");
        assert_eq!(
            lm.1,
            vec![Llama370MConfig::VOCAB_SIZE, Llama370MConfig::HIDDEN_DIM],
            "lm_head.weight must be row-major [vocab, hidden] — GH-202 \
             root cause; reversed `[hidden, vocab]` produces [PAD] garbage",
        );
        let embed = tensors
            .iter()
            .find(|(n, _)| n == "model.embed_tokens.weight")
            .expect("embed_tokens must be enumerated");
        assert_eq!(
            embed.1,
            vec![Llama370MConfig::VOCAB_SIZE, Llama370MConfig::HIDDEN_DIM],
            "embed_tokens.weight must be row-major [vocab, hidden]",
        );
        // GQA: K/V projections are 4× smaller on the out_dim axis vs Q/O.
        let k0 = tensors
            .iter()
            .find(|(n, _)| n == "model.layers.0.self_attn.k_proj.weight")
            .expect("k_proj layer 0 must be enumerated");
        assert_eq!(
            k0.1,
            vec![
                Llama370MConfig::NUM_KV_HEADS * Llama370MConfig::HEAD_DIM,
                Llama370MConfig::HIDDEN_DIM,
            ],
            "k_proj must be row-major [kv_heads*head_dim, hidden] — GQA",
        );
        let q0 = tensors
            .iter()
            .find(|(n, _)| n == "model.layers.0.self_attn.q_proj.weight")
            .expect("q_proj layer 0 must be enumerated");
        assert_eq!(
            q0.1,
            vec![
                Llama370MConfig::NUM_HEADS * Llama370MConfig::HEAD_DIM,
                Llama370MConfig::HIDDEN_DIM,
            ],
            "q_proj must be row-major [heads*head_dim, hidden]",
        );

        // Invariant 4: `validate_apr_shape` actively enforces the critical
        // tensor. Correct shape passes, reversed shape fails — the
        // validator must catch the GH-202 class of bug, not just
        // silently accept.
        contract
            .validate_apr_shape(
                "lm_head.weight",
                &[Llama370MConfig::VOCAB_SIZE, Llama370MConfig::HIDDEN_DIM],
                Llama370MConfig::VOCAB_SIZE,
                Llama370MConfig::HIDDEN_DIM,
            )
            .expect("correct [vocab, hidden] lm_head must validate");
        let bad = contract.validate_apr_shape(
            "lm_head.weight",
            &[Llama370MConfig::HIDDEN_DIM, Llama370MConfig::VOCAB_SIZE],
            Llama370MConfig::VOCAB_SIZE,
            Llama370MConfig::HIDDEN_DIM,
        );
        assert!(
            bad.is_err(),
            "reversed [hidden, vocab] lm_head MUST be rejected by the \
             layout contract — this is GH-202 regression protection",
        );
    }

    /// GATE-ARCH-370M-004 wiring check: once FALSIFY-SHIP-019 has an
    /// algorithm-level PARTIAL discharge, the sovereign contract YAML
    /// MUST record `discharge_status: PARTIAL_ALGORITHM_LEVEL` +
    /// `evidence_discharged_by` + `full_discharge_blocks_on` on
    /// GATE-ARCH-370M-004. Any edit that drops those fields fails this
    /// test before the artifact ships.
    #[test]
    fn falsify_ship_019_gate_arch_370m_004_has_partial_discharge_marker() {
        let doc: serde_yaml::Value =
            serde_yaml::from_str(SOVEREIGN_CONTRACT_YAML).expect("parse sovereign contract");
        let gates =
            doc["gates"].as_sequence().expect("gates must be a sequence in sovereign contract");
        let gate = gates
            .iter()
            .find(|g| g["id"].as_str() == Some("GATE-ARCH-370M-004"))
            .expect("GATE-ARCH-370M-004 must exist in sovereign contract");

        assert_eq!(
            gate["falsification_id"].as_str(),
            Some("FALSIFY-SHIP-019"),
            "GATE-ARCH-370M-004 must bind FALSIFY-SHIP-019",
        );
        assert_eq!(
            gate["binds_to"].as_str(),
            Some("AC-SHIP2-009"),
            "GATE-ARCH-370M-004 must bind AC-SHIP2-009",
        );
        assert_eq!(
            gate["discharge_status"].as_str(),
            Some("PARTIAL_ALGORITHM_LEVEL"),
            "GATE-ARCH-370M-004 must advertise PARTIAL_ALGORITHM_LEVEL \
             (full discharge blocks on real trained 370M .apr)",
        );
        let evidence = gate["evidence_discharged_by"]
            .as_sequence()
            .expect("GATE-ARCH-370M-004 must have evidence_discharged_by");
        assert!(
            !evidence.is_empty(),
            "GATE-ARCH-370M-004 evidence_discharged_by must list \
             at least one test function or artifact",
        );
        assert!(
            gate["full_discharge_blocks_on"].as_str().is_some(),
            "PARTIAL gate must document full_discharge_blocks_on",
        );
        assert_eq!(
            gate["ship_blocking"].as_bool(),
            Some(true),
            "GATE-ARCH-370M-004 must advertise ship_blocking:true — the \
             gate's `verdict:pass` alone is insufficient green while \
             discharge_status == PARTIAL_ALGORITHM_LEVEL",
        );
    }

    // ========================================================================
    // GATE-ARCH-370M-005 / AC-SHIP2-007 / FALSIFY-SHIP-017
    // ========================================================================

    /// FALSIFY-SHIP-017 (AC-SHIP2-007) — algorithm-level proof of the
    /// Python-AST-parse threshold function.
    ///
    /// The full-discharge harness (100 held-out prompts × `apr run` ×
    /// Python AST parse) blocks on a real trained 370M .apr. But the
    /// *decision rule* — "≥ 2 syntax errors on 100 prompts is a
    /// ship-blocker, ≤ 1 tolerated" — is a pure integer threshold and
    /// can be proven correct today. Any edit to `verdict_from_syntax_error_count`
    /// that widens the tolerance (or flips the boundary) is caught here
    /// before the artifact ships.
    ///
    /// Covers four invariants:
    ///   1. **Zero errors** → Pass (the trivial unanimous-parse case).
    ///   2. **Exactly-one error** → Pass (the tolerance boundary; matches
    ///      the EX-06 harness' `tolerate ≤ 1 SyntaxError` rule and
    ///      spec-§6 FALSIFY-SHIP-017 wording `tolerate ≤1`).
    ///   3. **Exactly-two errors** → Fail (the ship-blocker boundary;
    ///      FALSIFY-SHIP-017 says `≥ 2 SyntaxError`).
    ///   4. **Monotonicity** — raising the error count can only worsen
    ///      the verdict (Pass → Fail is one-way). This rules out any
    ///      future threshold edit that accidentally promotes a high
    ///      error count back to Pass.
    #[test]
    fn falsify_ship_017_syntax_error_count_threshold_logic() {
        // (1) Zero errors — unanimous parse.
        assert_eq!(
            verdict_from_syntax_error_count(0),
            Ship017Verdict::Pass,
            "0 syntax errors must always Pass",
        );

        // (2) Tolerance boundary — 1 error still tolerated.
        assert_eq!(
            verdict_from_syntax_error_count(1),
            Ship017Verdict::Pass,
            "1 syntax error is the AC_SHIP2_007_MAX_TOLERATED_SYNTAX_ERRORS \
             boundary and must Pass",
        );

        // (3) Ship-blocker boundary — 2 errors flips to Fail.
        assert_eq!(
            verdict_from_syntax_error_count(2),
            Ship017Verdict::Fail,
            "2 syntax errors is the FALSIFY-SHIP-017 ship-blocker \
             boundary and must Fail",
        );

        // (3b) Pathological cases — high error counts must still Fail.
        assert_eq!(
            verdict_from_syntax_error_count(AC_SHIP2_007_HELDOUT_PROMPT_COUNT),
            Ship017Verdict::Fail,
            "all-errors must Fail — trivial sanity",
        );
        assert_eq!(
            verdict_from_syntax_error_count(AC_SHIP2_007_HELDOUT_PROMPT_COUNT / 2),
            Ship017Verdict::Fail,
            "50% errors on 100 prompts must Fail",
        );

        // (4) Monotonicity — once Fail, always Fail as errors rise.
        let mut last_was_fail = false;
        for errors in 0..=AC_SHIP2_007_HELDOUT_PROMPT_COUNT {
            let verdict = verdict_from_syntax_error_count(errors);
            if last_was_fail {
                assert_eq!(
                    verdict,
                    Ship017Verdict::Fail,
                    "monotonicity violation at errors={errors}: once Fail, \
                     more errors cannot return to Pass",
                );
            }
            if verdict == Ship017Verdict::Fail {
                last_was_fail = true;
            }
        }

        // Provenance sanity — the AC-SHIP2-007 prompt-count constant
        // matches the spec's 100-prompt harness size. If this ever drifts,
        // the threshold is still correct (it's count-agnostic) but the
        // GATE-ARCH-370M-005 evidence no longer matches the spec wording.
        assert_eq!(
            AC_SHIP2_007_HELDOUT_PROMPT_COUNT, 100,
            "AC-SHIP2-007 spec §6 pins the harness at 100 held-out prompts",
        );
        assert_eq!(
            AC_SHIP2_007_MAX_TOLERATED_SYNTAX_ERRORS, 1,
            "FALSIFY-SHIP-017 (spec §8.3 row) tolerates ≤ 1 SyntaxError",
        );
    }

    /// GATE-ARCH-370M-005 wiring check: once FALSIFY-SHIP-017 has an
    /// algorithm-level PARTIAL discharge, the sovereign contract YAML
    /// MUST record `discharge_status: PARTIAL_ALGORITHM_LEVEL` plus
    /// `evidence_discharged_by` plus `full_discharge_blocks_on` on
    /// GATE-ARCH-370M-005. Any edit that drops those fields fails this
    /// test before the artifact ships.
    #[test]
    fn falsify_ship_017_gate_arch_370m_005_has_partial_discharge_marker() {
        let doc: serde_yaml::Value =
            serde_yaml::from_str(SOVEREIGN_CONTRACT_YAML).expect("parse sovereign contract");
        let gates =
            doc["gates"].as_sequence().expect("gates must be a sequence in sovereign contract");
        let gate = gates
            .iter()
            .find(|g| g["id"].as_str() == Some("GATE-ARCH-370M-005"))
            .expect("GATE-ARCH-370M-005 must exist in sovereign contract");

        assert_eq!(
            gate["falsification_id"].as_str(),
            Some("FALSIFY-SHIP-017"),
            "GATE-ARCH-370M-005 must bind FALSIFY-SHIP-017",
        );
        assert_eq!(
            gate["binds_to"].as_str(),
            Some("AC-SHIP2-007"),
            "GATE-ARCH-370M-005 must bind AC-SHIP2-007",
        );
        assert_eq!(
            gate["discharge_status"].as_str(),
            Some("PARTIAL_ALGORITHM_LEVEL"),
            "GATE-ARCH-370M-005 must advertise PARTIAL_ALGORITHM_LEVEL \
             (full discharge blocks on real trained 370M .apr + 100-prompt \
             `apr run` harness)",
        );
        let evidence = gate["evidence_discharged_by"]
            .as_sequence()
            .expect("GATE-ARCH-370M-005 must have evidence_discharged_by");
        assert!(
            !evidence.is_empty(),
            "GATE-ARCH-370M-005 evidence_discharged_by must list \
             at least one test function or artifact",
        );
        assert!(
            gate["full_discharge_blocks_on"].as_str().is_some(),
            "PARTIAL gate must document full_discharge_blocks_on",
        );
        assert_eq!(
            gate["ship_blocking"].as_bool(),
            Some(true),
            "GATE-ARCH-370M-005 must advertise ship_blocking:true — the \
             gate's `verdict:pass` alone is insufficient green while \
             discharge_status == PARTIAL_ALGORITHM_LEVEL",
        );
    }

    /// FALSIFY-SHIP-020 / AC-SHIP2-010 — pure decode-throughput threshold
    /// proof. `apr bench --median` on a real trained 370M .apr is the
    /// compute-heavy harness; the decision rule itself (≥100 tok/s
    /// passes, <100 tok/s fails) is separable and proven here.
    ///
    /// Invariants covered:
    ///   1. Pass boundary: exactly 100.0 tok/s → Pass (contract floor).
    ///   2. Fail boundary: 99.999 tok/s → Fail (one ULP below floor).
    ///   3. Generous green: 120.0 and 500.0 tok/s → Pass.
    ///   4. Hard red: 0.0 and 50.0 tok/s → Fail.
    ///   5. Monotonicity: once Fail, all strictly lower tps stay Fail.
    ///   6. Degenerate inputs: NaN and ±∞ → Fail (no well-defined
    ///      median → no proof).
    ///   7. Provenance pinning: the const floor MUST be 100.0 — any
    ///      edit that loosens the threshold trips this test before the
    ///      contract can ship.
    #[test]
    fn falsify_ship_020_decode_tps_threshold_logic() {
        // Pass boundary
        assert_eq!(
            verdict_from_decode_tps(100.0),
            Ship020Verdict::Pass,
            "exactly 100.0 tok/s must Pass (contract floor)",
        );
        // Fail boundary (one f32 step below 100.0)
        let just_below = f32::from_bits(100.0_f32.to_bits() - 1);
        assert!(just_below < 100.0);
        assert_eq!(
            verdict_from_decode_tps(just_below),
            Ship020Verdict::Fail,
            "one ULP below 100.0 tok/s must Fail",
        );
        // Generous-green sanity
        assert_eq!(verdict_from_decode_tps(120.0), Ship020Verdict::Pass);
        assert_eq!(verdict_from_decode_tps(500.0), Ship020Verdict::Pass);
        // Hard-red sanity
        assert_eq!(verdict_from_decode_tps(0.0), Ship020Verdict::Fail);
        assert_eq!(verdict_from_decode_tps(50.0), Ship020Verdict::Fail);
        // Monotonicity sweep: once Fail, any strictly smaller tps stays Fail.
        let samples = [0.0_f32, 25.0, 50.0, 75.0, 99.0, 99.5, 99.99, 100.0, 150.0, 10_000.0];
        let mut seen_fail = false;
        for &t in &samples {
            let v = verdict_from_decode_tps(t);
            if v == Ship020Verdict::Fail {
                seen_fail = true;
            } else if seen_fail {
                // We saw Fail earlier in the monotonically-increasing sweep
                // and now see Pass — that's fine (Fail→Pass is the
                // allowed crossover at the threshold). Re-arm by
                // resetting the seen_fail flag so we only guard against
                // Pass→Fail regressions within the sweep.
                seen_fail = false;
            }
        }
        // Separate direct Pass→Fail regression guard: walk strictly
        // decreasing from a clear Pass; once Fail shows, it must stick.
        let decreasing = [10_000.0_f32, 500.0, 150.0, 100.0, 99.99, 99.0, 50.0, 25.0, 0.0];
        let mut locked_fail = false;
        for &t in &decreasing {
            let v = verdict_from_decode_tps(t);
            if v == Ship020Verdict::Fail {
                locked_fail = true;
            } else {
                assert!(
                    !locked_fail,
                    "monotonicity violated: tps={t} produced Pass after a \
                     lower-tps Fail was already observed",
                );
            }
        }
        // Degenerate inputs: NaN / ±∞ are all conservatively Fail.
        // A real `apr bench` run can NEVER produce a non-finite median;
        // if one appears, the harness itself is broken and we must
        // refuse to claim a ship-gate pass on a value we cannot
        // meaningfully compare against the threshold.
        assert_eq!(
            verdict_from_decode_tps(f32::NAN),
            Ship020Verdict::Fail,
            "NaN tps has no well-defined median and must Fail",
        );
        assert_eq!(verdict_from_decode_tps(f32::NEG_INFINITY), Ship020Verdict::Fail,);
        assert_eq!(
            verdict_from_decode_tps(f32::INFINITY),
            Ship020Verdict::Fail,
            "+∞ tok/s is ill-formed — a real `apr bench` median is \
             always a finite positive; treating +∞ as Pass would let \
             an instrumentation bug silently green the ship-gate",
        );
        // Provenance pinning — any edit that loosens the threshold
        // (say, to 60.0) would silently lower the ship bar. Trip here
        // before the contract can ship.
        assert!(
            (AC_SHIP2_010_MIN_DECODE_TPS_RTX4090 - 100.0_f32).abs() < f32::EPSILON,
            "AC_SHIP2_010_MIN_DECODE_TPS_RTX4090 must stay pinned to 100.0 \
             tok/s — see contracts/model-families/llama-370m-sovereign-v1.yaml \
             GATE-ARCH-370M-006",
        );
    }

    /// GATE-ARCH-370M-006 wiring check: the sovereign contract YAML
    /// MUST record `discharge_status: PARTIAL_ALGORITHM_LEVEL` +
    /// `evidence_discharged_by` + `full_discharge_blocks_on` +
    /// `ship_blocking: true` on GATE-ARCH-370M-006, and bind it to
    /// AC-SHIP2-010 / FALSIFY-SHIP-020. Any edit that drops those
    /// fields fails this test before the artifact ships.
    #[test]
    fn falsify_ship_020_gate_arch_370m_006_has_partial_discharge_marker() {
        let doc: serde_yaml::Value =
            serde_yaml::from_str(SOVEREIGN_CONTRACT_YAML).expect("parse sovereign contract");
        let gates =
            doc["gates"].as_sequence().expect("gates must be a sequence in sovereign contract");
        let gate = gates
            .iter()
            .find(|g| g["id"].as_str() == Some("GATE-ARCH-370M-006"))
            .expect("GATE-ARCH-370M-006 must exist in sovereign contract");

        assert_eq!(
            gate["falsification_id"].as_str(),
            Some("FALSIFY-SHIP-020"),
            "GATE-ARCH-370M-006 must bind FALSIFY-SHIP-020",
        );
        assert_eq!(
            gate["binds_to"].as_str(),
            Some("AC-SHIP2-010"),
            "GATE-ARCH-370M-006 must bind AC-SHIP2-010",
        );
        assert_eq!(
            gate["discharge_status"].as_str(),
            Some("PARTIAL_ALGORITHM_LEVEL"),
            "GATE-ARCH-370M-006 must advertise PARTIAL_ALGORITHM_LEVEL \
             (full discharge blocks on real trained 370M .apr + RTX 4090 \
             `apr bench` median run)",
        );
        let evidence = gate["evidence_discharged_by"]
            .as_sequence()
            .expect("GATE-ARCH-370M-006 must have evidence_discharged_by");
        assert!(
            !evidence.is_empty(),
            "GATE-ARCH-370M-006 evidence_discharged_by must list \
             at least one test function or artifact",
        );
        assert!(
            gate["full_discharge_blocks_on"].as_str().is_some(),
            "PARTIAL gate must document full_discharge_blocks_on",
        );
        assert_eq!(
            gate["ship_blocking"].as_bool(),
            Some(true),
            "GATE-ARCH-370M-006 must advertise ship_blocking:true — the \
             gate's `verdict:pass` alone is insufficient green while \
             discharge_status == PARTIAL_ALGORITHM_LEVEL",
        );
    }

    // ========================================================================
    // FALSIFY-SHIP-018 / AC-SHIP2-008 / GATE-ARCH-370M-007 — pass@1 threshold
    // ========================================================================

    /// Algorithm-level proof that `verdict_from_pass_at_1` enforces the
    /// spec §5.2 AC-SHIP2-008 rule "HumanEval pass@1 ≥ 30.0%" correctly:
    ///
    ///   1. Exactly-at-floor passes (30/100, 60/200, 1/1 at threshold 100).
    ///   2. One f32 ULP below the floor fails.
    ///   3. Generous-green (50/100, 164/164) passes.
    ///   4. Hard-red (0/100, 1/100) fails.
    ///   5. Monotonicity: sweeping `correct` upward from 0 with total fixed
    ///      at 164 (canonical HumanEval) never flips Pass → Fail.
    ///   6. Div-safety guard: `total == 0` always fails, even with 0 correct.
    ///   7. Sanity guard: `correct > total` always fails (impossible harness).
    ///   8. Non-finite threshold guard: NaN / ±∞ always fail.
    ///   9. Provenance: the contract floor const is pinned to 30.0 (edit
    ///      to the const without amending the contract is caught here).
    ///
    /// Full `apr eval --benchmark humaneval` discharge blocks on the trained
    /// 370M .apr from AC-SHIP2-003/004 compute-dispatch — fixture swap only.
    #[test]
    fn falsify_ship_018_humaneval_pass_at_1_threshold_logic() {
        // ── (1) Exactly-at-floor → Pass
        assert_eq!(
            verdict_from_pass_at_1(30, 100, AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT),
            Ship018Verdict::Pass,
            "30/100 = 30.0% must pass the 30.0 floor",
        );
        assert_eq!(
            verdict_from_pass_at_1(60, 200, AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT),
            Ship018Verdict::Pass,
            "60/200 = 30.0% must pass the 30.0 floor",
        );
        // HumanEval canonical N=164 at-floor: ⌈0.3 × 164⌉ = 50 → 50/164 = 30.49%
        assert_eq!(
            verdict_from_pass_at_1(50, 164, AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT),
            Ship018Verdict::Pass,
            "50/164 ≈ 30.49% must pass the 30.0 floor",
        );

        // ── (2a) Just below the floor → Fail.
        //
        // 49/164 ≈ 29.878% computes to a ratio strictly less than 30.0 in
        // f32 (see below) and must therefore fail against the 30.0 floor.
        // (Note: 30/100 in f32 rounds to ~30.000002, slightly *above* 30.0,
        // so "exactly-at-floor" must be tested with ratios that are exact
        // in f32 — see case (2b).)
        assert_eq!(
            verdict_from_pass_at_1(49, 164, AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT),
            Ship018Verdict::Fail,
            "49/164 ≈ 29.88% must fail the 30.0 floor",
        );
        assert_eq!(
            verdict_from_pass_at_1(29, 100, AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT),
            Ship018Verdict::Fail,
            "29/100 = 29.0% must fail the 30.0 floor",
        );

        // ── (2b) Inclusive-floor proof at an f32-exact ratio.
        //
        // 50/100 → exactly 50.0% in f32 (both operands integer-exact, the
        // quotient 0.5 is a power of two, and 0.5 × 100.0 = 50.0 exactly).
        // This lets us prove the comparison is `>=` (inclusive), not `>`:
        //   - Threshold = 50.0              → Pass (exact equality).
        //   - Threshold = 50.0 + one ULP    → Fail (strictly above ratio).
        //   - Threshold = 50.0 - one ULP    → Pass (strictly below ratio).
        let exact_50 = 50.0_f32;
        assert_eq!(50.0_f32 * 2.0_f32, 100.0_f32, "sanity: 50.0 is exact in f32");
        let fifty_plus_ulp = f32::from_bits(exact_50.to_bits() + 1);
        let fifty_minus_ulp = f32::from_bits(exact_50.to_bits() - 1);
        assert!(fifty_plus_ulp > exact_50, "sanity: +ULP is strictly above");
        assert!(fifty_minus_ulp < exact_50, "sanity: −ULP is strictly below");
        assert_eq!(
            verdict_from_pass_at_1(50, 100, exact_50),
            Ship018Verdict::Pass,
            "inclusive floor: 50.0% ≥ 50.0 must Pass (proves `>=`, not `>`)",
        );
        assert_eq!(
            verdict_from_pass_at_1(50, 100, fifty_plus_ulp),
            Ship018Verdict::Fail,
            "50/100 must fail when threshold is one ULP above 50.0",
        );
        assert_eq!(
            verdict_from_pass_at_1(50, 100, fifty_minus_ulp),
            Ship018Verdict::Pass,
            "50/100 must pass when threshold is one ULP below 50.0",
        );

        // ── (3) Generous-green
        assert_eq!(
            verdict_from_pass_at_1(82, 164, AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT),
            Ship018Verdict::Pass,
            "82/164 = 50% must pass",
        );
        assert_eq!(
            verdict_from_pass_at_1(164, 164, AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT),
            Ship018Verdict::Pass,
            "perfect score must pass",
        );

        // ── (4) Hard-red
        assert_eq!(
            verdict_from_pass_at_1(0, 164, AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT),
            Ship018Verdict::Fail,
            "zero-correct run must fail",
        );
        assert_eq!(
            verdict_from_pass_at_1(1, 164, AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT),
            Ship018Verdict::Fail,
            "1/164 ≈ 0.6% must fail",
        );

        // ── (5) Monotonicity sweep: correct ∈ 0..=164, total = 164
        //
        // Over an increasing `correct` axis the verdict is allowed to flip
        // Fail → Pass exactly once; it must never flip Pass → Fail.
        let total = 164usize;
        let mut already_passed = false;
        for correct in 0..=total {
            let v =
                verdict_from_pass_at_1(correct, total, AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT);
            match v {
                Ship018Verdict::Pass => {
                    already_passed = true;
                }
                Ship018Verdict::Fail => {
                    assert!(
                        !already_passed,
                        "monotonicity violated: correct={correct} reverted Pass→Fail",
                    );
                }
            }
        }

        // ── (6) Div-safety: total=0 must always fail
        assert_eq!(
            verdict_from_pass_at_1(0, 0, AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT),
            Ship018Verdict::Fail,
            "empty run (total=0) must fail — a positive floor is unsatisfiable",
        );
        assert_eq!(
            verdict_from_pass_at_1(0, 0, 0.0_f32),
            Ship018Verdict::Fail,
            "empty run must fail even with a zero threshold — the harness \
             is broken if it reports an empty denominator",
        );

        // ── (7) Sanity guard: correct > total
        assert_eq!(
            verdict_from_pass_at_1(165, 164, AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT),
            Ship018Verdict::Fail,
            "correct > total is a broken harness report; must fail closed",
        );

        // ── (8) Non-finite threshold → Fail
        assert_eq!(
            verdict_from_pass_at_1(164, 164, f32::NAN),
            Ship018Verdict::Fail,
            "NaN threshold must fail",
        );
        assert_eq!(
            verdict_from_pass_at_1(164, 164, f32::INFINITY),
            Ship018Verdict::Fail,
            "+∞ threshold must fail",
        );
        assert_eq!(
            verdict_from_pass_at_1(164, 164, f32::NEG_INFINITY),
            Ship018Verdict::Fail,
            "−∞ threshold must fail",
        );

        // ── (9) Provenance pin
        assert!(
            (AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT - 30.0_f32).abs() < f32::EPSILON,
            "contract floor drift: AC_SHIP2_008_MIN_HUMANEVAL_PASS_AT_1_PCT \
             must stay pinned to 30.0 (spec §5.2 AC-SHIP2-008)",
        );
    }

    /// Binds the YAML contract's `GATE-ARCH-370M-007` block to this test
    /// binary: any rename, removal, or discharge_status regression in
    /// `contracts/model-families/llama-370m-sovereign-v1.yaml` is caught
    /// at `cargo test` time.
    #[test]
    fn falsify_ship_018_gate_arch_370m_007_has_partial_discharge_marker() {
        let doc: serde_yaml::Value =
            serde_yaml::from_str(SOVEREIGN_CONTRACT_YAML).expect("parse sovereign contract");

        let gates = doc["gates"].as_sequence().expect("contract must have `gates:` sequence");

        let gate = gates
            .iter()
            .find(|g| g["id"].as_str() == Some("GATE-ARCH-370M-007"))
            .expect("GATE-ARCH-370M-007 (SHIP-018 humaneval pass@1) must be present");

        assert_eq!(
            gate["binds_to"].as_str(),
            Some("AC-SHIP2-008"),
            "GATE-ARCH-370M-007 must bind AC-SHIP2-008",
        );
        assert_eq!(
            gate["falsification_id"].as_str(),
            Some("FALSIFY-SHIP-018"),
            "GATE-ARCH-370M-007 must bind FALSIFY-SHIP-018",
        );
        assert_eq!(
            gate["discharge_status"].as_str(),
            Some("PARTIAL_ALGORITHM_LEVEL"),
            "GATE-ARCH-370M-007 must advertise PARTIAL_ALGORITHM_LEVEL — \
             full discharge blocks on trained 370M .apr + real apr eval",
        );
        let evidence = gate["evidence_discharged_by"]
            .as_sequence()
            .expect("GATE-ARCH-370M-007 must have evidence_discharged_by");
        assert!(
            !evidence.is_empty(),
            "GATE-ARCH-370M-007 evidence_discharged_by must list at least \
             one test function or const pin",
        );
        assert!(
            gate["full_discharge_blocks_on"].as_str().is_some(),
            "PARTIAL gate must document full_discharge_blocks_on",
        );
        assert_eq!(
            gate["ship_blocking"].as_bool(),
            Some(true),
            "GATE-ARCH-370M-007 must advertise ship_blocking:true — \
             verdict:pass alone is insufficient green while \
             discharge_status == PARTIAL_ALGORITHM_LEVEL",
        );
    }

    /// FALSIFY-SHIP-016 algorithm-level PARTIAL discharge: the decision
    /// rule of SHIP-016 — "`apr qa <model>.apr` passes iff all 8 gates
    /// PASS; any gate FAIL fails the ship" — is a pure aggregate-AND
    /// over a Boolean slice. This test proves the rule without running
    /// the compute-heavy gates themselves. The rule separates from the
    /// compute dependency: the gate-runner is fixture-swappable once
    /// a trained 370M .apr exists; the decision is proven today.
    #[test]
    fn falsify_ship_016_apr_qa_aggregate_and_logic() {
        // Section 1: canonical Pass — all 8 gates true → Pass.
        let all_pass = [true; AC_SHIP2_006_REQUIRED_QA_GATE_COUNT];
        assert_eq!(
            verdict_from_qa_gates(&all_pass),
            Ship016Verdict::Pass,
            "AC-SHIP2-006: all 8 gates PASS must yield Pass",
        );

        // Section 2: single-gate-fail must flip the aggregate to Fail —
        // this is the "any gate FAIL" counter-example from FALSIFY-SHIP-016.
        for flip_idx in 0..AC_SHIP2_006_REQUIRED_QA_GATE_COUNT {
            let mut gates = [true; AC_SHIP2_006_REQUIRED_QA_GATE_COUNT];
            gates[flip_idx] = false;
            assert_eq!(
                verdict_from_qa_gates(&gates),
                Ship016Verdict::Fail,
                "flipping gate index {flip_idx} from Pass to Fail must yield aggregate Fail \
                 — SHIP-016 is an AND, not a majority or threshold",
            );
        }

        // Section 3: canonical Fail — all 8 gates false → Fail.
        let all_fail = [false; AC_SHIP2_006_REQUIRED_QA_GATE_COUNT];
        assert_eq!(
            verdict_from_qa_gates(&all_fail),
            Ship016Verdict::Fail,
            "all 8 gates FAIL must yield Fail",
        );

        // Section 4: exhaustive 2^8 = 256-combination proof — the ONLY
        // input yielding Pass is the all-true vector; every other
        // combination of 8 bools must yield Fail.
        let mut pass_count = 0usize;
        let mut fail_count = 0usize;
        for mask in 0u32..(1u32 << AC_SHIP2_006_REQUIRED_QA_GATE_COUNT) {
            let gates: [bool; AC_SHIP2_006_REQUIRED_QA_GATE_COUNT] =
                std::array::from_fn(|i| (mask >> i) & 1 == 1);
            match verdict_from_qa_gates(&gates) {
                Ship016Verdict::Pass => {
                    pass_count += 1;
                    assert!(
                        gates.iter().all(|&p| p),
                        "Pass verdict must only occur when all 8 gates are true; \
                         got {gates:?} at mask {mask:#010b}",
                    );
                }
                Ship016Verdict::Fail => {
                    fail_count += 1;
                    assert!(
                        gates.iter().any(|&p| !p),
                        "Fail verdict must only occur when at least one gate is false; \
                         got {gates:?} at mask {mask:#010b}",
                    );
                }
            }
        }
        assert_eq!(pass_count, 1, "exactly one of 256 combos (all-true) yields Pass");
        assert_eq!(fail_count, 255, "the other 255 combos must yield Fail");

        // Section 5: monotonicity — adding a Pass to a mixed slice can
        // only move the verdict up (Fail→Pass) or keep it the same,
        // never downgrade Pass→Fail. Pair each combo with the combo
        // obtained by flipping one bit from false to true; assert
        // the verdict never regresses.
        for mask in 0u32..(1u32 << AC_SHIP2_006_REQUIRED_QA_GATE_COUNT) {
            let before: [bool; AC_SHIP2_006_REQUIRED_QA_GATE_COUNT] =
                std::array::from_fn(|i| (mask >> i) & 1 == 1);
            for flip_idx in 0..AC_SHIP2_006_REQUIRED_QA_GATE_COUNT {
                if before[flip_idx] {
                    continue;
                }
                let mut after = before;
                after[flip_idx] = true;
                let before_v = verdict_from_qa_gates(&before);
                let after_v = verdict_from_qa_gates(&after);
                assert!(
                    !(before_v == Ship016Verdict::Pass && after_v == Ship016Verdict::Fail),
                    "monotonicity violated: flipping gate {flip_idx} from false to true \
                     regressed Pass→Fail at mask {mask:#010b}",
                );
            }
        }

        // Section 6: contract-drift guards — wrong gate count must Fail
        // conservatively even when every supplied entry is true. This
        // prevents a silent green from an out-of-sync harness that
        // shipped 7 or 9 gates instead of the spec-mandated 8.
        assert_eq!(
            verdict_from_qa_gates(&[]),
            Ship016Verdict::Fail,
            "empty gate slice must Fail (contract drift)",
        );
        assert_eq!(
            verdict_from_qa_gates(&[true; 7]),
            Ship016Verdict::Fail,
            "7 gates (short by one) must Fail even when all true (contract drift)",
        );
        assert_eq!(
            verdict_from_qa_gates(&[true; 9]),
            Ship016Verdict::Fail,
            "9 gates (long by one) must Fail even when all true (contract drift)",
        );
        assert_eq!(
            verdict_from_qa_gates(&[true; 16]),
            Ship016Verdict::Fail,
            "double-wide gate slice must Fail (contract drift)",
        );

        // Section 7: provenance pin — the 8-gate count is the
        // contract number; drift on this constant fails lockstep-wise
        // with the spec amendment and `QaConfig` skip flags.
        assert_eq!(
            AC_SHIP2_006_REQUIRED_QA_GATE_COUNT, 8,
            "AC-SHIP2-006 is the 8-gate aggregate; any change requires \
             contract + spec + CLI skip-flag edits in lockstep",
        );
    }

    /// GATE-ARCH-370M-008 wiring check: once FALSIFY-SHIP-016 has an
    /// algorithm-level PARTIAL discharge, the sovereign contract YAML
    /// MUST record `discharge_status: PARTIAL_ALGORITHM_LEVEL` +
    /// `evidence_discharged_by` + `full_discharge_blocks_on` on
    /// GATE-ARCH-370M-008. Any edit that drops those fields fails this
    /// test before the artifact ships.
    #[test]
    fn falsify_ship_016_gate_arch_370m_008_has_partial_discharge_marker() {
        let doc: serde_yaml::Value =
            serde_yaml::from_str(SOVEREIGN_CONTRACT_YAML).expect("parse sovereign contract");
        let gates =
            doc["gates"].as_sequence().expect("gates must be a sequence in sovereign contract");
        let gate = gates
            .iter()
            .find(|g| g["id"].as_str() == Some("GATE-ARCH-370M-008"))
            .expect("GATE-ARCH-370M-008 must exist in sovereign contract");

        assert_eq!(
            gate["falsification_id"].as_str(),
            Some("FALSIFY-SHIP-016"),
            "GATE-ARCH-370M-008 must bind FALSIFY-SHIP-016",
        );
        assert_eq!(
            gate["binds_to"].as_str(),
            Some("AC-SHIP2-006"),
            "GATE-ARCH-370M-008 must bind AC-SHIP2-006",
        );
        assert_eq!(
            gate["discharge_status"].as_str(),
            Some("PARTIAL_ALGORITHM_LEVEL"),
            "GATE-ARCH-370M-008 must advertise PARTIAL_ALGORITHM_LEVEL \
             (full discharge blocks on real trained 370M .apr + `apr qa` harness)",
        );
        let evidence = gate["evidence_discharged_by"]
            .as_sequence()
            .expect("GATE-ARCH-370M-008 must have evidence_discharged_by");
        assert!(
            !evidence.is_empty(),
            "GATE-ARCH-370M-008 evidence_discharged_by must list \
             at least one test function or artifact",
        );
        assert!(
            gate["full_discharge_blocks_on"].as_str().is_some(),
            "PARTIAL gate must document full_discharge_blocks_on",
        );
        assert_eq!(
            gate["ship_blocking"].as_bool(),
            Some(true),
            "GATE-ARCH-370M-008 must advertise ship_blocking:true — the \
             gate's `verdict:pass` alone is insufficient green while \
             discharge_status == PARTIAL_ALGORITHM_LEVEL",
        );
    }

    // ========================================================================
    // FALSIFY-SHIP-013 / AC-SHIP2-003 / GATE-ARCH-370M-013 — val CE loss floor
    // ========================================================================

    /// FALSIFY-SHIP-013 algorithm-level PARTIAL discharge: prove the
    /// f32-threshold decision rule binding the measured MODEL-2 val CE
    /// to [`AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS`] = 2.2. Any edit
    /// that changes the constant, the comparison direction, the
    /// non-finite handling, the negative-domain guard, or the
    /// monotonicity must break this test before an `apr pretrain
    /// --validate` compute dispatch is launched.
    ///
    /// Mutation survey (7 sections):
    ///   1. Exact boundary 2.2 → Pass (inclusive floor, not strict <)
    ///   2. One-ULP above boundary → Fail; one-ULP below → Pass
    ///   3. Clear Pass band {0.0, 0.5, 1.0, 2.0, 2.199}
    ///   4. Clear Fail band {2.201, 3.0, 10.0, f32::MAX}
    ///   5. Non-finite {NaN, +∞, -∞} → Fail conservatively
    ///   6. Negative (domain violation, CE ≥ 0) → Fail conservatively
    ///   7. Provenance pin: the const stays byte-equal to 2.2_f32
    #[test]
    fn falsify_ship_013_val_ce_loss_threshold_logic() {
        // Section 1: exact boundary 2.2 → Pass. AC-SHIP2-003 says
        // "CE ≤ 2.2" (inclusive), so sim == threshold must Pass. A
        // regression that silently swaps `<=` to `<` would make the
        // ceiling unreachable.
        assert_eq!(
            verdict_from_val_ce_loss(AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS),
            Ship013Verdict::Pass,
            "val CE == 2.2 must Pass (inclusive floor, not strict <)",
        );
        assert_eq!(verdict_from_val_ce_loss(2.2), Ship013Verdict::Pass, "literal 2.2 must Pass",);

        // Section 2: ULP asymmetry around the boundary. f32 at 2.2 is
        // `0x400CCCCD` = 2.20000004768371582...; the next-representable
        // float up is `0x400CCCCE` (Fail), and the next below is
        // `0x400CCCCC` (Pass). Sharpest possible counter-examples.
        let one_ulp_above = f32::from_bits(AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS.to_bits() + 1);
        let one_ulp_below = f32::from_bits(AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS.to_bits() - 1);
        assert!(one_ulp_above > AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS);
        assert!(one_ulp_below < AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS);
        assert_eq!(
            verdict_from_val_ce_loss(one_ulp_above),
            Ship013Verdict::Fail,
            "one ULP above 2.2 must Fail (strictly above ceiling)",
        );
        assert_eq!(
            verdict_from_val_ce_loss(one_ulp_below),
            Ship013Verdict::Pass,
            "one ULP below 2.2 must Pass (still under ceiling)",
        );

        // Section 3: clear Pass band. 0.0 is the theoretical minimum
        // (perfect predictor); 2.199 is safely under 2.2.
        for ce in [0.0_f32, 0.5, 1.0, 2.0, 2.199] {
            assert_eq!(
                verdict_from_val_ce_loss(ce),
                Ship013Verdict::Pass,
                "val CE = {ce} must Pass (in clear Pass band)",
            );
        }

        // Section 4: clear Fail band. A poorly-converged 370M would
        // land here (a random 50K-vocab predictor is ≈ ln(50257) ≈
        // 10.82 nats, which must obviously Fail). f32::MAX is the
        // saturation sanity check.
        for ce in [2.201_f32, 3.0, 10.0, f32::MAX] {
            assert_eq!(
                verdict_from_val_ce_loss(ce),
                Ship013Verdict::Fail,
                "val CE = {ce} must Fail (above 2.2 ceiling)",
            );
        }

        // Section 5: non-finite inputs must Fail conservatively. A
        // loss-harness bug that produces NaN (log(0) or divide-by-zero
        // in softmax) or ±∞ (overflow in exp) must never silently
        // promote to Pass via NaN comparison semantics.
        assert_eq!(
            verdict_from_val_ce_loss(f32::NAN),
            Ship013Verdict::Fail,
            "NaN val CE must Fail conservatively",
        );
        assert_eq!(
            verdict_from_val_ce_loss(f32::INFINITY),
            Ship013Verdict::Fail,
            "+∞ val CE must Fail conservatively",
        );
        assert_eq!(
            verdict_from_val_ce_loss(f32::NEG_INFINITY),
            Ship013Verdict::Fail,
            "-∞ val CE must Fail conservatively",
        );

        // Section 6: negative CE is a domain violation. Cross-entropy
        // H(p,q) = -Σ p(x) log q(x) ≥ 0 for all probability
        // distributions p,q. A negative measurement indicates a sign
        // flip, log-domain underflow, or subtract-instead-of-add bug
        // in the loss harness, and must never be promoted to "better
        // than zero" Pass.
        for neg_ce in [-0.001_f32, -1.0, -f32::INFINITY] {
            assert_eq!(
                verdict_from_val_ce_loss(neg_ce),
                Ship013Verdict::Fail,
                "negative val CE = {neg_ce} must Fail (CE ≥ 0 by definition)",
            );
        }

        // Section 7: provenance pin — the 2.2 constant is load-bearing
        // and lockstepped with the spec. If AC-SHIP2-003 ever changes
        // the ceiling (relaxing to 2.5 for a smaller training-data
        // budget, tightening to 2.0 for a 500M jump), this const and
        // this test must move together.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(
                AC_SHIP2_003_MAX_VAL_CROSS_ENTROPY_LOSS, 2.2_f32,
                "MODEL-2 val CE ceiling is 2.2 \
                 (spec §5.2 AC-SHIP2-003; albor 370M Sovereign target)",
            );
        }
    }

    // ========================================================================
    // FALSIFY-SHIP-014 / AC-SHIP2-004 / GATE-ARCH-370M-014 — training budget
    // ========================================================================

    /// FALSIFY-SHIP-014 algorithm-level PARTIAL discharge: prove the
    /// u32-threshold decision rule binding the measured MODEL-2
    /// training wall-clock duration to
    /// [`AC_SHIP2_004_MAX_TRAINING_DURATION_DAYS`] = 21. Any edit that
    /// changes the constant, the comparison direction, or introduces
    /// a non-monotonic regression must break this test before a
    /// real-compute dispatch on the RTX 4090 host is launched.
    ///
    /// Mutation survey (6 sections):
    ///   1. Exact boundary 21 → Pass (inclusive ceiling)
    ///   2. Adjacent values: 20 → Pass; 22 → Fail
    ///   3. Clear Pass band {0, 1, 7, 14, 20, 21}
    ///   4. Clear Fail band {22, 30, 100, u32::MAX}
    ///   5. Monotonicity sweep 0..=42 — verdict flips exactly once
    ///      at the 21→22 transition and never flips back
    ///   6. Provenance pin: the const stays byte-equal to 21_u32
    #[test]
    fn falsify_ship_014_training_duration_threshold_logic() {
        // Section 1: exact boundary 21 → Pass. AC-SHIP2-004 says
        // "within 21 days" (inclusive), so 21 == threshold must Pass.
        // A regression that silently swaps `<=` to `<` would make the
        // ceiling unreachable.
        assert_eq!(
            verdict_from_training_duration_days(AC_SHIP2_004_MAX_TRAINING_DURATION_DAYS),
            Ship014Verdict::Pass,
            "21 days must Pass (inclusive ceiling, not strict <)",
        );
        assert_eq!(
            verdict_from_training_duration_days(21),
            Ship014Verdict::Pass,
            "literal 21 days must Pass",
        );

        // Section 2: adjacent values. 20 is the sharpest Pass
        // counter-example (one below ceiling); 22 is the sharpest
        // Fail counter-example (one above). u32 neighbours are
        // exact (no ULP noise) so these are both rock-solid.
        assert_eq!(
            verdict_from_training_duration_days(20),
            Ship014Verdict::Pass,
            "20 days must Pass (one day under ceiling)",
        );
        assert_eq!(
            verdict_from_training_duration_days(22),
            Ship014Verdict::Fail,
            "22 days must Fail (one day over ceiling; Spec §9 Risk #4 escape hatch)",
        );

        // Section 3: clear Pass band. 0 is trivial (same-day cache
        // hit, no actual training); 1/7/14 are smaller training runs;
        // 21 is the boundary.
        for days in [0_u32, 1, 7, 14, 20, 21] {
            assert_eq!(
                verdict_from_training_duration_days(days),
                Ship014Verdict::Pass,
                "{days} days must Pass (in clear Pass band)",
            );
        }

        // Section 4: clear Fail band. 22/30/100 are incremental
        // overruns (each invokes Spec §9 Risk #4 escape-hatch planning
        // — rent 2× H100 week 3); u32::MAX is the saturation sanity.
        for days in [22_u32, 30, 100, u32::MAX] {
            assert_eq!(
                verdict_from_training_duration_days(days),
                Ship014Verdict::Fail,
                "{days} days must Fail (above 21-day ceiling)",
            );
        }

        // Section 5: monotonicity sweep 0..=42 — verdict must flip
        // exactly once at the 21→22 transition and never flip back.
        // This is the complete classification over a 2× the ceiling
        // range; any mutation that introduces non-monotonicity (e.g.
        // a midlife-crisis carve-out at "after day 14 but before day
        // 21" or a modular-arithmetic bug) is trivially caught.
        let mut seen_fail = false;
        for days in 0..=42_u32 {
            let v = verdict_from_training_duration_days(days);
            match (v, seen_fail) {
                (Ship014Verdict::Pass, true) => {
                    panic!(
                        "monotonicity broken: day {days} flipped back to Pass \
                         after a previous Fail was observed",
                    );
                }
                (Ship014Verdict::Fail, _) => {
                    seen_fail = true;
                }
                _ => {}
            }
        }
        // And verify the transition is exactly at 21→22, not earlier
        // or later (rules out off-by-one mutants).
        assert_eq!(
            verdict_from_training_duration_days(21),
            Ship014Verdict::Pass,
            "sweep boundary: day 21 must Pass",
        );
        assert_eq!(
            verdict_from_training_duration_days(22),
            Ship014Verdict::Fail,
            "sweep boundary: day 22 must Fail",
        );

        // Section 6: provenance pin — the 21 constant is load-bearing
        // and lockstepped with the spec. If AC-SHIP2-004 ever changes
        // the ceiling (extending to 30 for a longer run, tightening
        // to 14 for a distilled student), this const and this test
        // must move together.
        assert_eq!(
            AC_SHIP2_004_MAX_TRAINING_DURATION_DAYS, 21_u32,
            "MODEL-2 training-budget ceiling is 21 days \
             (spec §5.2 AC-SHIP2-004; RTX 4090 hardware budget)",
        );
    }
}
