//! Kernel-cache keys, as one source both the pre-warm and the runtime read.
//!
//! # Why this module exists (YOGA-NIGHTLY-001 R-2, paiml/infra)
//!
//! The Blackwell cascade — seven PRs, PMAT-698e..PMAT-698p — was one defect
//! restated: **`pre_warm_for_model` compiled kernels under keys the runtime never
//! asked for.** The forms it took are all in the git record:
//!
//! - `warm!` hardcoded the literal `"silu_forward"` as the key for every kernel,
//!   so eleven-plus modules collided on one `HashMap` entry and only the first
//!   was ever stored (PMAT-698j).
//! - The pre-warm key for RMSNorm omitted the `_eps{bits:08x}` suffix the
//!   runtime key carries (PMAT-698k).
//! - It then pre-warmed the wrong epsilon: `1e-5` (Llama) while the model in
//!   flight was Qwen2 at `1e-6` (PMAT-698n).
//! - RoPE was pre-warmed at `seq_len=1` only, while the corpus phase ran at 256
//!   (PMAT-698p).
//! - `batched_fused_residual_rmsnorm` had no pre-warm entry at all
//!   (FALSIFY-CUDA-FUSED-RMSNORM-DEADLOCK-001).
//!
//! Every one of those is the same sentence: **two `format!` strings, in two
//! files, with nothing tying them together.** The consequence was silent on
//! sm_89 — the kernel simply JIT-compiled on demand and SUCCEEDED, which is why
//! the sm_89 lane was green through all seven — and fatal on sm_121, where JIT
//! mid-forward poisons the stream.
//!
//! # What this module changes
//!
//! Nothing about the kernels. It moves the KEYS out of the two places that
//! formatted them independently and into constructors both sides call, so that:
//!
//! 1. A CPU-ONLY property test can assert `prewarm_keys ⊇ forward_runtime_keys`
//!    for a table of real model shapes. No GPU, no CUDA toolkit, no second
//!    machine, no cross-architecture transcript diff — it runs on intel, on
//!    every PR, in milliseconds. Any one of the five defects above fails it.
//! 2. `pre_warm_for_model` checks the keys it ACTUALLY warmed against
//!    [`prewarm_keys`] before returning, so the pure list cannot quietly drift
//!    from the `warm!` sequence it models. A list that models nothing is how
//!    this defect class survives a test.
//!
//! The runtime counter (`jit_compiles`, R-3) catches the same defect on
//! hardware, after the fact. This catches it in the type of place it was
//! written, before the merge. They are not redundant: the counter needs a GPU
//! and a model, and this needs neither, so this one runs on every PR.
//!
//! # The contract
//!
//! A key is a string the kernel-module cache is indexed by. Two kernels that
//! must not share a compiled module must not share a key, and one kernel asked
//! for twice must produce the same key both times. That is all a key is, and
//! this module is the only place in the crate allowed to build one.

use std::collections::BTreeSet;

/// The model shape a pre-warm and a forward pass are both a function of.
///
/// Sizes are `u32` because that is what the kernel constructors and the key
/// format strings take; `usize` at the call sites is narrowed once, here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelKeyShape {
    /// `hidden_size`.
    pub hidden: u32,
    /// `intermediate_size` (FFN width).
    pub intermediate: u32,
    /// Attention heads.
    pub num_heads: u32,
    /// Key/value heads (GQA); equal to `num_heads` for MHA.
    pub num_kv_heads: u32,
    /// Per-head dimension.
    pub head_dim: u32,
    /// The sequence length the cache is warmed for.
    pub max_seq_len: u32,
}

impl ModelKeyShape {
    /// Q/O projection width, which is NOT `hidden` on every model — Qwen3-4B is
    /// `h=2560`, `q_dim=4096`, and pre-warming under `h` there is a miss.
    #[must_use]
    pub const fn q_dim(&self) -> u32 {
        self.num_heads * self.head_dim
    }

    /// K/V projection width under GQA.
    #[must_use]
    pub const fn kv_dim(&self) -> u32 {
        self.num_kv_heads * self.head_dim
    }
}

/// The two RMSNorm epsilons in the fleet's model corpus, as raw bits.
///
/// The bit pattern IS the key material: `1e-6` (Qwen2/Qwen2.5) is `0x358637bd`
/// and `1e-5` (Llama/Mistral) is `0x3727c5ac`, and PMAT-698n was pre-warming
/// the second while running the first.
pub const QWEN2_RMS_EPS: f32 = 1.0e-6;
/// Llama/Mistral RMSNorm epsilon. See [`QWEN2_RMS_EPS`].
pub const LLAMA_RMS_EPS: f32 = 1.0e-5;
/// The RoPE theta the corpus uses (Qwen2/Qwen2.5).
pub const QWEN_ROPE_THETA: f32 = 1_000_000.0;

// ---------------------------------------------------------------------------
// The constructors. One per key FORM. Nothing else in this crate may format a
// cache key.
// ---------------------------------------------------------------------------

/// `batched_rmsnorm_fwd_{hidden}_eps{bits:08x}` (normalization.rs).
#[must_use]
pub fn batched_rmsnorm_fwd(hidden: u32, eps: f32) -> String {
    let bits = eps.to_bits();
    format!("batched_rmsnorm_fwd_{hidden}_eps{bits:08x}")
}

/// `batched_fused_residual_rmsnorm_{hidden}_eps{bits:08x}` (normalization.rs).
#[must_use]
pub fn batched_fused_residual_rmsnorm(hidden: u32, eps: f32) -> String {
    let bits = eps.to_bits();
    format!("batched_fused_residual_rmsnorm_{hidden}_eps{bits:08x}")
}

/// `batched_rope_neox_fwd_{heads}_{head_dim}_{seq}_th{bits:08x}` (normalization.rs).
#[must_use]
pub fn batched_rope_neox_fwd(heads: u32, head_dim: u32, seq_len: u32, theta: f32) -> String {
    let bits = theta.to_bits();
    format!("batched_rope_neox_fwd_{heads}_{head_dim}_{seq_len}_th{bits:08x}")
}

/// `gemm_forward_{m}_{k}_{n}` (matmul.rs). The PTX GEMM, used when cuBLAS is absent.
#[must_use]
pub fn gemm_forward(m: u32, k: u32, n: u32) -> String {
    format!("gemm_forward_{m}_{k}_{n}")
}

/// `batched_4d_gemm_{batch}_{heads}_{m}_{n}_{k}` (matmul.rs).
#[must_use]
pub fn batched_4d_gemm(batch: u32, heads: u32, m: u32, n: u32, k: u32) -> String {
    format!("batched_4d_gemm_{batch}_{heads}_{m}_{n}_{k}")
}

/// `nf4_gemm_forward_{k}_{n}` (matmul.rs).
///
/// M is deliberately absent: the PTX is shape-independent and including the
/// sequence length made every batch whose `seq_len != max_seq_len` miss
/// (trueno#184).
#[must_use]
pub fn nf4_gemm_forward(k: u32, n: u32) -> String {
    format!("nf4_gemm_forward_{k}_{n}")
}

/// `nf4_gemm_transpose_{n}_{k}` (matmul.rs) — QLoRA backward through frozen NF4.
#[must_use]
pub fn nf4_gemm_transpose(n: u32, k: u32) -> String {
    format!("nf4_gemm_transpose_{n}_{k}")
}

/// `fused_nf4_gate_up_{k}_{n}` (matmul.rs) — fused gate+up, and fused K+V under GQA.
#[must_use]
pub fn fused_nf4_gate_up(k: u32, n: u32) -> String {
    format!("fused_nf4_gate_up_{k}_{n}")
}

/// Shape-independent keys: one compiled module serves every dimension, so the
/// key is a literal. They are listed here rather than typed at each call site
/// so that a typo is a compile error at one place instead of a cache miss at
/// runtime.
pub mod fixed {
    /// Fused SwiGLU (matmul.rs).
    pub const FUSED_SWIGLU_FORWARD: &str = "fused_swiglu_forward";
    /// Residual add (elementwise.rs).
    pub const RESIDUAL_ADD_FORWARD: &str = "residual_add_forward";
    /// Element-wise multiply (elementwise.rs).
    pub const ELEMENTWISE_MUL_FORWARD: &str = "elementwise_mul_forward";
    /// Scale (elementwise.rs).
    pub const SCALE_FORWARD: &str = "scale_forward";
    /// Interleaved to batched (elementwise.rs).
    pub const INTERLEAVED_TO_BATCHED: &str = "interleaved_to_batched";
    /// Batched transpose (elementwise.rs).
    pub const BATCHED_TRANSPOSE: &str = "batched_transpose";
    /// Batched to interleaved (elementwise.rs).
    pub const BATCHED_TO_INTERLEAVED: &str = "batched_to_interleaved";
    /// Batched softmax (activations.rs).
    pub const BATCHED_SOFTMAX_FORWARD: &str = "batched_softmax_forward";
    /// SiLU (activations.rs).
    pub const SILU_FORWARD: &str = "silu_forward";
}

// ---------------------------------------------------------------------------
// The two sets.
// ---------------------------------------------------------------------------

/// What `pre_warm_for_model` compiles, as a function of everything it branches on.
///
/// `has_cublas` and `rope_seq_lens` are parameters rather than reads because
/// they ARE the branches: PMAT-700 skips the four PTX GEMMs when cuBLAS is
/// bound, and PMAT-698p pre-warms RoPE at both `1` and `APR_DISTILL_SMOKE_SEQ_LEN`.
/// A pure function of them is testable at both settings; a function that read
/// the environment would be testable at neither.
#[derive(Debug, Clone)]
pub struct PreWarmSpec {
    /// The model shape.
    pub shape: ModelKeyShape,
    /// Whether a cuBLAS handle is bound (PMAT-700: skips the PTX GEMM pre-warms).
    pub has_cublas: bool,
    /// The sequence lengths RoPE is warmed at (PMAT-698p).
    pub rope_seq_lens: Vec<u32>,
}

/// The NF4 quantised projections, forward and transposed-backward.
///
/// Split out of [`prewarm_keys`] rather than inlined: every branch here is an
/// `is_multiple_of(64)` block-size test or a GQA asymmetry, and all six of them
/// in one function is where the complexity gate (cyclomatic 30) drew the line.
/// The split is also the honest shape — these are the keys that exist only when
/// the weights are quantised.
fn nf4_keys(s: ModelKeyShape, out: &mut BTreeSet<String>) {
    let (h, i) = (s.hidden, s.intermediate);
    let q_dim = s.q_dim();
    let kv_h = s.kv_dim();
    if !h.is_multiple_of(64) {
        return;
    }
    let kv_distinct = kv_h != h && kv_h != q_dim && kv_h.is_multiple_of(64);

    out.insert(nf4_gemm_forward(h, q_dim));
    out.insert(nf4_gemm_transpose(q_dim, h));
    if q_dim != h {
        out.insert(nf4_gemm_forward(q_dim, h));
        out.insert(nf4_gemm_transpose(h, q_dim));
    }
    if kv_distinct {
        out.insert(nf4_gemm_forward(h, kv_h));
        out.insert(nf4_gemm_transpose(kv_h, h));
    }
    if i.is_multiple_of(64) {
        out.insert(nf4_gemm_forward(h, i));
        out.insert(nf4_gemm_forward(i, h));
        out.insert(nf4_gemm_transpose(i, h));
        out.insert(nf4_gemm_transpose(h, i));
        out.insert(fused_nf4_gate_up(h, i));
    }
    if kv_h.is_multiple_of(64) && kv_h != i {
        out.insert(fused_nf4_gate_up(h, kv_h));
    }
}

/// The kernels whose PTX is dimension-independent, so one module serves every
/// shape and the key is a literal.
fn fixed_keys(out: &mut BTreeSet<String>) {
    for k in [
        fixed::FUSED_SWIGLU_FORWARD,
        fixed::RESIDUAL_ADD_FORWARD,
        fixed::INTERLEAVED_TO_BATCHED,
        fixed::BATCHED_TRANSPOSE,
        fixed::SCALE_FORWARD,
        fixed::BATCHED_SOFTMAX_FORWARD,
        fixed::BATCHED_TO_INTERLEAVED,
        fixed::ELEMENTWISE_MUL_FORWARD,
        fixed::SILU_FORWARD,
    ] {
        out.insert(k.to_string());
    }
}

/// The three attention 4D GEMMs: Q@K^T, attn@V, and the backward grad_V^T.
fn attention_gemm_keys(s: ModelKeyShape, seq: u32, out: &mut BTreeSet<String>) {
    let (nh, hd) = (s.num_heads, s.head_dim);
    out.insert(batched_4d_gemm(1, nh, seq, seq, hd));
    out.insert(batched_4d_gemm(1, nh, seq, hd, seq));
    out.insert(batched_4d_gemm(1, nh, hd, seq, seq));
}

/// RoPE, at every warmed sequence length and for the KV head count under GQA.
fn rope_keys(s: ModelKeyShape, seq_lens: &[u32], out: &mut BTreeSet<String>) {
    let (nh, nkv, hd) = (s.num_heads, s.num_kv_heads, s.head_dim);
    for &seq in seq_lens {
        out.insert(batched_rope_neox_fwd(nh, hd, seq, QWEN_ROPE_THETA));
        if nkv != nh {
            out.insert(batched_rope_neox_fwd(nkv, hd, seq, QWEN_ROPE_THETA));
        }
    }
}

/// Every cache key `pre_warm_for_model` compiles under this spec, deduplicated
/// and ordered.
///
/// This mirrors the `warm!` sequence in `cache.rs`, and `pre_warm_for_model`
/// asserts the two agree before it returns — so "mirrors" is checked on
/// hardware, not asserted here in prose.
#[must_use]
pub fn prewarm_keys(spec: &PreWarmSpec) -> BTreeSet<String> {
    let s = spec.shape;
    let (h, i, seq) = (s.hidden, s.intermediate, s.max_seq_len);
    let kv_h = s.kv_dim();
    let mut out = BTreeSet::new();

    // 1 / 1b. RMSNorm and fused residual RMSNorm, at BOTH corpus epsilons.
    for eps in [QWEN2_RMS_EPS, LLAMA_RMS_EPS] {
        out.insert(batched_rmsnorm_fwd(h, eps));
        out.insert(batched_fused_residual_rmsnorm(h, eps));
    }

    // 2-5. PTX GEMMs, only when cuBLAS is NOT bound (PMAT-700).
    if !spec.has_cublas {
        out.insert(gemm_forward(seq, h, h));
        if kv_h != h {
            out.insert(gemm_forward(seq, h, kv_h));
        }
        out.insert(gemm_forward(seq, h, i));
        out.insert(gemm_forward(seq, i, h));
    }

    rope_keys(s, &spec.rope_seq_lens, &mut out);
    fixed_keys(&mut out);
    attention_gemm_keys(s, seq, &mut out);
    nf4_keys(s, &mut out);

    out
}

/// Every cache key an NF4 QLoRA forward+backward pass ASKS FOR on this shape.
///
/// Derived from the dispatch sites, not from the pre-warm — that independence is
/// the whole point. `seq_len` is a parameter because the runtime's RoPE key
/// carries the ACTUAL sequence length of the batch in flight, which is exactly
/// the axis PMAT-698p got wrong: pre-warm at 1, run at 256.
#[must_use]
pub fn forward_runtime_keys(shape: ModelKeyShape, seq_len: u32, rms_eps: f32) -> BTreeSet<String> {
    let s = shape;
    let (h, i, nh, nkv, hd) = (s.hidden, s.intermediate, s.num_heads, s.num_kv_heads, s.head_dim);
    let q_dim = s.q_dim();
    let kv_h = s.kv_dim();
    let mut out = BTreeSet::new();

    // Norms: the input norm and the fused post-attention norm, at the model's
    // OWN epsilon — not at a default.
    out.insert(batched_rmsnorm_fwd(h, rms_eps));
    out.insert(batched_fused_residual_rmsnorm(h, rms_eps));

    // Q/K/V/O and the FFN, through the NF4 path.
    if h.is_multiple_of(64) {
        out.insert(nf4_gemm_forward(h, q_dim));
        if q_dim != h {
            out.insert(nf4_gemm_forward(q_dim, h));
        }
        if kv_h != h && kv_h != q_dim && kv_h.is_multiple_of(64) {
            out.insert(nf4_gemm_forward(h, kv_h));
        }
        if i.is_multiple_of(64) {
            out.insert(nf4_gemm_forward(h, i));
            out.insert(nf4_gemm_forward(i, h));
        }
    }
    if h.is_multiple_of(64) && i.is_multiple_of(64) {
        out.insert(fused_nf4_gate_up(h, i));
    }

    // Attention, at the sequence length actually in flight.
    out.insert(batched_rope_neox_fwd(nh, hd, seq_len, QWEN_ROPE_THETA));
    if nkv != nh {
        out.insert(batched_rope_neox_fwd(nkv, hd, seq_len, QWEN_ROPE_THETA));
    }
    out.insert(fixed::INTERLEAVED_TO_BATCHED.to_string());
    out.insert(fixed::BATCHED_TRANSPOSE.to_string());
    out.insert(batched_4d_gemm(1, nh, seq_len, seq_len, hd));
    out.insert(fixed::SCALE_FORWARD.to_string());
    out.insert(fixed::BATCHED_SOFTMAX_FORWARD.to_string());
    out.insert(batched_4d_gemm(1, nh, seq_len, hd, seq_len));
    out.insert(fixed::BATCHED_TO_INTERLEAVED.to_string());

    // FFN activation and the residual.
    out.insert(fixed::FUSED_SWIGLU_FORWARD.to_string());
    out.insert(fixed::RESIDUAL_ADD_FORWARD.to_string());

    out
}

#[cfg(test)]
mod tests;
