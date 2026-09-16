//! Model-shape arithmetic for `contracts/qwen35-e2e-verification-v1.yaml`.
//!
//! Each public function here implements one *equation* of that contract and is
//! registered in `contracts/binding.yaml`, which is what `pv coverage` /
//! `pv proof-status` count as `impl=N/6` (#3347). The equations are stated in
//! the contract's own variable names, and those names are used verbatim below:
//!
//! | Equation | Formula | Function |
//! |----------|---------|----------|
//! | `model_parameter_count` | `P = V*d + L*(d_attn + d_ffn + d_norm) + d_final + V*d` | [`model_parameter_count`] |
//! | `flops_per_token` | `F ≈ 2*P` (+ the `O(S*d*L)` attention term) | [`flops_per_token`] |
//! | `memory_breakdown` | `M = M_weights + M_kv + M_activations` | [`memory_breakdown`] |
//! | `throughput_model` | `tok/s = min(bandwidth / bytes_per_token, compute / flops_per_token)` | [`throughput_model`] |
//! | `contract_composition` | `model_contract = compose(embedding, L × block, final_norm, unembed)` | [`contract_composition`] |
//!
//! The sixth equation, `verification_ladder`, is deliberately NOT implemented
//! here — see the module-level note at the bottom of this doc block.
//!
//! # Architecture comes in as data
//!
//! Nothing in this module bakes in a constant for any model. Every function
//! takes [`ModelSizeConfig`] / [`ModelConstraints`] (parsed from
//! `contracts/model-families/*.yaml`) or explicit per-layer data, so the same
//! code answers for Qwen3.5-9B, Qwen3.5-27B, or a two-layer test model.
//!
//! # Totality
//!
//! All integer arithmetic saturates. These functions are called from property
//! tests with arbitrary dimensions, and a release build would silently wrap
//! where a debug build panics; saturation makes both builds agree and keeps
//! every function total (no panic, no `unwrap`).
//!
//! # Why `verification_ladder` is absent
//!
//! `coverage(contract_set) = verified_obligations / total_obligations` is not
//! model arithmetic — it measures the contract corpus. Implementing it requires
//! deciding what makes an obligation "verified", and that decision is exactly
//! what issue #3347 reports as broken today (the `--table` L2 column ticks on
//! `idx < falsification_tests.len()`, an index comparison rather than an
//! obligation↔test link). Worse, the function would be self-referential: its
//! output is what `QE2E-BND-005` asserts equals `1.0`, and binding it would
//! move the very counter it measures. That is a semantics decision the contract
//! does not settle, so it is left unbound and `QE2E-BND-005` stays false.

use crate::format::layout_contract::block_sizes;
use crate::format::model_family::{MlpType, ModelConstraints, ModelSizeConfig};

// ============================================================================
// Equation: model_parameter_count
// P = V*d + L*(d_attn + d_ffn + d_norm) + d_final + V*d
// ============================================================================

/// Per-layer parameter counts — the `d_attn`, `d_ffn`, `d_norm` terms of the
/// `model_parameter_count` equation, for ONE layer.
///
/// A hybrid family (Qwen3.5 interleaves Gated DeltaNet layers with full
/// attention layers) has different `d_attn` per layer kind, which is why
/// [`model_parameter_count`] takes a slice of these rather than a single
/// per-layer number multiplied by `L`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LayerParams {
    /// `d_attn` — parameters in this layer's token-mixing block.
    pub d_attn: u64,
    /// `d_ffn` — parameters in this layer's feed-forward block.
    pub d_ffn: u64,
    /// `d_norm` — parameters in this layer's normalization weights.
    pub d_norm: u64,
}

impl LayerParams {
    /// `d_attn + d_ffn + d_norm` for this layer.
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.d_attn
            .saturating_add(self.d_ffn)
            .saturating_add(self.d_norm)
    }
}

/// The `model_parameter_count` equation, term by term, so a caller can see
/// which term dominates rather than only the total `P`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParameterBreakdown {
    /// `V*d` — input embedding matrix.
    pub embedding: u64,
    /// `L*(d_attn + d_ffn + d_norm)` — all decoder layers.
    pub layers: u64,
    /// `d_final` — final norm weights.
    pub final_norm: u64,
    /// The trailing `V*d` — the untied `lm_head`; `0` when embeddings are tied.
    pub unembedding: u64,
    /// `P` — the sum of the four terms above.
    pub total: u64,
}

/// `d_attn`, `d_ffn`, `d_norm` for one ordinary (softmax-attention) decoder
/// layer of a family described by `size` + `constraints`.
///
/// - `d_attn` = `d*(n_h*d_k) + 2*d*(n_kv*d_k) + (n_h*d_k)*d` (Q, K, V, O),
///   plus the four bias vectors when `constraints.has_bias`.
/// - `d_ffn` = `3*d*d_ff` for a gated MLP (SwiGLU/GeGLU), else `2*d*d_ff`.
/// - `d_norm` = `2*d` (input norm + post-attention norm).
///
/// This is the dense/GQA accounting. It is NOT the Gated DeltaNet accounting:
/// see [`model_parameter_count`] for what that needs.
#[must_use]
pub fn attention_layer_params(
    size: &ModelSizeConfig,
    constraints: &ModelConstraints,
) -> LayerParams {
    let d = size.hidden_dim as u64;
    let d_k = size.head_dim as u64;
    let q_dim = (size.num_heads as u64).saturating_mul(d_k);
    let kv_dim = (size.num_kv_heads as u64).saturating_mul(d_k);

    let projections = d
        .saturating_mul(q_dim)
        .saturating_add(d.saturating_mul(kv_dim).saturating_mul(2))
        .saturating_add(q_dim.saturating_mul(d));
    let biases = if constraints.has_bias {
        q_dim
            .saturating_add(kv_dim.saturating_mul(2))
            .saturating_add(d)
    } else {
        0
    };

    let d_ff = size.intermediate_dim as u64;
    let matrices = if matches!(constraints.mlp_type, MlpType::SwiGlu | MlpType::GatedMlp) {
        3
    } else {
        2
    };

    LayerParams {
        d_attn: projections.saturating_add(biases),
        d_ffn: d.saturating_mul(d_ff).saturating_mul(matrices),
        d_norm: d.saturating_mul(2),
    }
}

/// `L` copies of [`attention_layer_params`] — the per-layer input for a
/// homogeneous (non-hybrid) model of `size.num_layers` layers.
#[must_use]
pub fn uniform_layers(size: &ModelSizeConfig, constraints: &ModelConstraints) -> Vec<LayerParams> {
    vec![attention_layer_params(size, constraints); size.num_layers]
}

/// Equation `model_parameter_count`:
/// `P = V*d + L*(d_attn + d_ffn + d_norm) + d_final + V*d`.
///
/// `V` = `size.vocab_size`, `d` = `size.hidden_dim`, `d_final` = `d` (the final
/// RMSNorm weights), and the trailing `V*d` is the untied `lm_head` — present
/// exactly when `constraints.tied_embeddings` is false, which is what
/// `contracts/model-families/qwen3_5.yaml` declares.
///
/// `layers` supplies `(d_attn, d_ffn, d_norm)` per layer, so `L = layers.len()`.
/// [`uniform_layers`] builds it for a homogeneous model.
///
/// # Contract precondition
///
/// The contract states `num_layers > 0`. This function does not reject `L = 0`
/// — it returns the embedding-only count, which is the correct value of the
/// formula at `L = 0` — but a real model outside that precondition is a config
/// bug, not an arithmetic one.
///
/// # What this does NOT discharge
///
/// `QE2E-INV-001` wants `P(Qwen3.5-9B) ∈ [9.0B, 9.2B]`. Feeding this function
/// [`uniform_layers`] for the 9B variant yields ≈8.21B, because Qwen3.5 is
/// `hybrid_gated_deltanet`: three of every four layers are Gated DeltaNet, whose
/// `d_attn` covers conv, gate and state projections sized by `inner_size`,
/// `state_size`, `conv_kernel` and `group_count`. Those four keys exist in
/// `contracts/model-families/qwen3_5.yaml` but NOT in [`ModelConstraints`], so
/// the GDN `d_attn` cannot be derived from the config type as it stands. The
/// equation is implemented; the 9B invariant is not verified.
#[must_use]
pub fn model_parameter_count(
    size: &ModelSizeConfig,
    constraints: &ModelConstraints,
    layers: &[LayerParams],
) -> ParameterBreakdown {
    let v = size.vocab_size as u64;
    let d = size.hidden_dim as u64;

    let embedding = v.saturating_mul(d);
    let layer_total = layers
        .iter()
        .fold(0u64, |acc, l| acc.saturating_add(l.total()));
    let unembedding = if constraints.tied_embeddings {
        0
    } else {
        embedding
    };

    let total = embedding
        .saturating_add(layer_total)
        .saturating_add(d)
        .saturating_add(unembedding);

    ParameterBreakdown {
        embedding,
        layers: layer_total,
        final_norm: d,
        unembedding,
        total,
    }
}

// ============================================================================
// Equation: flops_per_token
// F ≈ 2*P (forward pass) for dense compute
// ============================================================================

/// The `flops_per_token` equation split into its two terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlopsPerToken {
    /// `2*P` — one multiply-accumulate per parameter per token.
    pub dense: u64,
    /// `2*L*S*d` — the attention score/value term, `O(S*d*L)`.
    pub attention: u64,
    /// `F` = `dense + attention`.
    pub total: u64,
}

/// Equation `flops_per_token`: `F ≈ 2*P` for dense compute, plus the
/// `O(S*d*L)` attention term the contract's `QE2E-BND-002` allows for
/// (`F <= 2*P + O(seq_len * d * L)`).
///
/// `P` is a parameter count (from [`model_parameter_count`]), `S` = `seq_len`
/// is the number of cached positions attended over, `d` = `hidden_dim`, and
/// `L` = the number of layers that run softmax attention. For a Gated DeltaNet
/// layer pass `attn_layers = 0` for that layer: its mixing cost is `O(d^2)` per
/// token and is already inside `2*P`, which is what the contract's third
/// invariant ("GDN FLOP component is O(d^2) per token — no quadratic") states.
///
/// `F` is linear in `P`, the contract's first invariant.
#[must_use]
pub fn flops_per_token(
    parameter_count: u64,
    seq_len: u64,
    hidden_dim: u64,
    attn_layers: u64,
) -> FlopsPerToken {
    let dense = parameter_count.saturating_mul(2);
    let attention = attn_layers
        .saturating_mul(seq_len)
        .saturating_mul(hidden_dim)
        .saturating_mul(2);
    FlopsPerToken {
        dense,
        attention,
        total: dense.saturating_add(attention),
    }
}

// ============================================================================
// Equation: memory_breakdown
// M = M_weights + M_kv + M_activations
// ============================================================================

/// Element storage precisions, ordered by bytes per element — the ordering
/// `QE2E-ORD-003` asserts: `M(Q4K) < M(Q5K) < M(Q6K) < M(F16) < M(F32)` at a
/// whole number of 256-element K-quant super-blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Precision {
    /// K-quant 4-bit: 144 bytes per 256 elements.
    Q4K,
    /// K-quant 5-bit: 176 bytes per 256 elements.
    Q5K,
    /// K-quant 6-bit: 210 bytes per 256 elements.
    Q6K,
    /// IEEE half: 2 bytes per element.
    F16,
    /// IEEE single: 4 bytes per element.
    F32,
}

impl Precision {
    /// Bytes needed to store `elements` values at this precision.
    ///
    /// K-quants encode whole 256-element super-blocks, so the count is rounded
    /// UP to a whole super-block — which is why `QE2E-ORD-003` carries the
    /// precondition `n % 256 == 0`: below one super-block the ordering is false
    /// (one 210-byte Q6K block exceeds `2n` F16 bytes for `n <= 105`).
    #[must_use]
    pub const fn bytes_for(self, elements: u64) -> u64 {
        let qk = block_sizes::QK_K as u64;
        match self {
            Self::Q4K => elements
                .div_ceil(qk)
                .saturating_mul(block_sizes::Q4_K as u64),
            Self::Q5K => elements
                .div_ceil(qk)
                .saturating_mul(block_sizes::Q5_K as u64),
            Self::Q6K => elements
                .div_ceil(qk)
                .saturating_mul(block_sizes::Q6_K as u64),
            Self::F16 => elements.saturating_mul(2),
            Self::F32 => elements.saturating_mul(4),
        }
    }
}

/// The inference working point that `memory_breakdown` is evaluated at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferencePlan {
    /// `S` — sequence length (cached positions).
    pub seq_len: u64,
    /// Batch size.
    pub batch_size: u64,
    /// Number of layers holding a KV cache. For a hybrid model this is the
    /// count of full-attention layers only (`kv-cache-sizing-v1`
    /// `hybrid_accounting`), not `L`.
    pub kv_layers: u64,
    /// Precision of the stored weights.
    pub weights: Precision,
    /// Precision of the KV cache.
    pub kv_cache: Precision,
    /// Precision of activations.
    pub activations: Precision,
}

/// The `memory_breakdown` equation, term by term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBreakdown {
    /// `M_weights` — `P` parameters at the weight precision.
    pub weights: u64,
    /// `M_kv` — `2 * kv_layers * n_kv * d_k * S` elements at KV precision.
    pub kv: u64,
    /// `M_activations` — `batch_size * S * d` elements at activation precision.
    pub activations: u64,
    /// `M` = `M_weights + M_kv + M_activations`.
    pub total: u64,
}

/// Equation `memory_breakdown`: `M = M_weights + M_kv + M_activations`, in bytes.
///
/// - `M_weights` = `plan.weights.bytes_for(P)` — depends on quantization, the
///   contract's first invariant.
/// - `M_kv` = `2 * kv_layers * n_kv * d_k * S` elements (the factor 2 is K and
///   V; this is `kv-cache-sizing-v1`'s `total_kv_memory`), so it is linear in
///   `S` — the contract's second invariant.
/// - `M_activations` = `batch_size * S * d` elements — the contract's third
///   invariant states this is the bound.
#[must_use]
pub fn memory_breakdown(
    size: &ModelSizeConfig,
    parameter_count: u64,
    plan: &InferencePlan,
) -> MemoryBreakdown {
    let d = size.hidden_dim as u64;
    let n_kv = size.num_kv_heads as u64;
    let d_k = size.head_dim as u64;

    let kv_elements = plan
        .kv_layers
        .saturating_mul(n_kv)
        .saturating_mul(d_k)
        .saturating_mul(plan.seq_len)
        .saturating_mul(2);
    let act_elements = plan
        .batch_size
        .saturating_mul(plan.seq_len)
        .saturating_mul(d);

    let weights = plan.weights.bytes_for(parameter_count);
    let kv = plan.kv_cache.bytes_for(kv_elements);
    let activations = plan.activations.bytes_for(act_elements);

    MemoryBreakdown {
        weights,
        kv,
        activations,
        total: weights.saturating_add(kv).saturating_add(activations),
    }
}

// ============================================================================
// Equation: throughput_model
// tok/s = min(bandwidth / bytes_per_token, compute / flops_per_token)
// ============================================================================

/// Which side of the roofline [`throughput_model`] landed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RooflineLimit {
    /// `bandwidth / bytes_per_token` was the smaller term.
    MemoryBound,
    /// `compute / flops_per_token` was the smaller term.
    ComputeBound,
}

/// The `throughput_model` equation with both terms and the binding one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Throughput {
    /// `bandwidth / bytes_per_token`, tokens per second.
    pub memory_bound: f64,
    /// `compute / flops_per_token`, tokens per second.
    pub compute_bound: f64,
    /// `tok/s` — the min of the two.
    pub tokens_per_second: f64,
    /// Which term the min came from.
    pub limit: RooflineLimit,
}

/// Equation `throughput_model`:
/// `tok/s = min(bandwidth / bytes_per_token, compute / flops_per_token)`.
///
/// `bandwidth` is bytes/s, `compute` is FLOP/s, `bytes_per_token` is the bytes
/// that must be read to emit one token (for memory-bound decode, essentially
/// `M_weights` from [`memory_breakdown`]), and `flops_per_token` is `F` from
/// [`flops_per_token`].
///
/// Monotone in bandwidth — `QE2E-MON-004`: `bw1 < bw2 → tok_s(bw1) <= tok_s(bw2)`,
/// because `min(·, c)` is monotone in its first argument.
///
/// A zero or negative denominator yields `0.0` for that term rather than an
/// infinity or a NaN, so the result is always a comparable number.
#[must_use]
pub fn throughput_model(
    bandwidth_bytes_per_s: f64,
    compute_flops_per_s: f64,
    bytes_per_token: f64,
    flops_per_token: f64,
) -> Throughput {
    let ratio = |numerator: f64, denominator: f64| {
        if denominator > 0.0 && numerator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    };
    let memory_bound = ratio(bandwidth_bytes_per_s, bytes_per_token);
    let compute_bound = ratio(compute_flops_per_s, flops_per_token);

    let (tokens_per_second, limit) = if memory_bound <= compute_bound {
        (memory_bound, RooflineLimit::MemoryBound)
    } else {
        (compute_bound, RooflineLimit::ComputeBound)
    };

    Throughput {
        memory_bound,
        compute_bound,
        tokens_per_second,
        limit,
    }
}

// ============================================================================
// Equation: contract_composition
// model_contract = compose(embedding, L × block, final_norm, unembed)
// ============================================================================

/// One component of the composed model contract, with the shape it maps from
/// and to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompositionStage {
    /// Component name: `embedding`, `block_{l}`, `final_norm`, `unembed`.
    pub component: String,
    /// Shape entering this component.
    pub input: Vec<usize>,
    /// Shape leaving this component.
    pub output: Vec<usize>,
}

impl CompositionStage {
    /// True when this stage leaves the shape unchanged — the residual-stream
    /// property `QE2E-INV-006` asserts of every block: `shape(block_l(x)) = shape(x)`.
    #[must_use]
    pub fn preserves_shape(&self) -> bool {
        self.input == self.output
    }
}

/// Equation `contract_composition`:
/// `model_contract = compose(embedding, L × block, final_norm, unembed)`.
///
/// Returns the composition as a shape pipeline for `seq_len` tokens:
/// `[S] --embedding--> [S, d] --block_l--> [S, d] (×L) --final_norm--> [S, d]
/// --unembed--> [S, V]`.
///
/// Every stage's `output` is the next stage's `input` (composition is
/// well-formed), every `block_*` and `final_norm` stage satisfies
/// [`CompositionStage::preserves_shape`] (`QE2E-INV-006`), and the last stage's
/// output is `[seq_len, V]` (`QE2E-CON-007`: tokens in → logits out).
#[must_use]
pub fn contract_composition(size: &ModelSizeConfig, seq_len: usize) -> Vec<CompositionStage> {
    let d = size.hidden_dim;
    let v = size.vocab_size;
    let residual = vec![seq_len, d];

    let mut stages = Vec::with_capacity(size.num_layers.saturating_add(3));
    stages.push(CompositionStage {
        component: "embedding".to_string(),
        input: vec![seq_len],
        output: residual.clone(),
    });
    for l in 0..size.num_layers {
        stages.push(CompositionStage {
            component: format!("block_{l}"),
            input: residual.clone(),
            output: residual.clone(),
        });
    }
    stages.push(CompositionStage {
        component: "final_norm".to_string(),
        input: residual.clone(),
        output: residual.clone(),
    });
    stages.push(CompositionStage {
        component: "unembed".to_string(),
        input: residual,
        output: vec![seq_len, v],
    });
    stages
}

/// True when `stages` compose: each stage's output shape is the next stage's
/// input shape. A composition that does not chain proves nothing about the
/// model it claims to describe.
#[must_use]
pub fn composition_is_well_formed(stages: &[CompositionStage]) -> bool {
    stages.windows(2).all(|w| w[0].output == w[1].input)
}

#[cfg(test)]
#[path = "model_arithmetic_tests.rs"]
mod model_arithmetic_tests;
