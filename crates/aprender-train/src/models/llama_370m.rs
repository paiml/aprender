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
//! - **INV-ARCH-370M-006**  `vocab_size == 50_000` and matches the paired
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
    pub const VOCAB_SIZE: usize = 50_000;
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
    ///   INV-ARCH-370M-006  vocab_size == 50_000
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
            Self::VOCAB_SIZE == 50_000,
            "INV-ARCH-370M-006 violated: vocab_size must equal 50_000",
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
        assert_eq!(Llama370MConfig::VOCAB_SIZE, 50_000);
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
    #[test]
    fn falsify_gate_arch_370m_011_helper_rejects_mismatch() {
        assert!(assert_tokenizer_vocab_matches_model(
            Llama370MConfig::VOCAB_SIZE,
            Llama370MConfig::VOCAB_SIZE,
        )
        .is_ok());

        let err = assert_tokenizer_vocab_matches_model(50_257, Llama370MConfig::VOCAB_SIZE)
            .expect_err("mismatch must return Err");
        assert!(
            err.contains("GATE-ARCH-370M-011") && err.contains("50257") && err.contains("50000"),
            "error must name the gate and both vocab sizes for forensics, got: {err}",
        );

        assert!(assert_tokenizer_vocab_matches_model(0, 1).is_err());
        assert!(assert_tokenizer_vocab_matches_model(
            Llama370MConfig::VOCAB_SIZE + 1,
            Llama370MConfig::VOCAB_SIZE
        )
        .is_err());
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
        assert_eq!(Vocab::VALUE, 50_000);

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
}
