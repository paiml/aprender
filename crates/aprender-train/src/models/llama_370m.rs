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

// ─────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "llama_370m_tests.rs"]
mod tests;
