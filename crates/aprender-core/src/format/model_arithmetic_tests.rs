//! Tests for the `qwen35-e2e-verification-v1` equation implementations.
//!
//! Each equation gets one worked example (hand-computed, so the test fails if
//! the formula changes) and, where the contract states one, one property.

use super::*;
use crate::format::model_family::{Activation, AttentionType, NormType, PositionalEncoding};

/// A four-dimension toy model whose parameter count is small enough to verify
/// by hand: V=10, d=4, L=2, n_h=2, n_kv=1, d_k=2, d_ff=8.
fn toy_size() -> ModelSizeConfig {
    ModelSizeConfig {
        parameters: "toy".to_string(),
        hidden_dim: 4,
        num_layers: 2,
        num_heads: 2,
        num_kv_heads: 1,
        intermediate_dim: 8,
        vocab_size: 10,
        max_position_embeddings: 128,
        head_dim: 2,
        rope_theta: 10_000.0,
        norm_eps: 1e-6,
    }
}

/// Qwen3.5-9B, exactly as `contracts/model-families/qwen3_5.yaml` declares it.
fn qwen35_9b_size() -> ModelSizeConfig {
    ModelSizeConfig {
        parameters: "9B".to_string(),
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 16,
        num_kv_heads: 4,
        intermediate_dim: 12288,
        vocab_size: 248_320,
        max_position_embeddings: 262_144,
        head_dim: 256,
        rope_theta: 1_000_000.0,
        norm_eps: 1e-6,
    }
}

/// Qwen3.5 constraints: SwiGLU, RMSNorm, no bias, UNTIED embeddings.
fn qwen35_constraints() -> ModelConstraints {
    ModelConstraints {
        attention_type: AttentionType::HybridGatedDeltaNet,
        activation: Activation::Silu,
        norm_type: NormType::RmsNorm,
        has_bias: false,
        tied_embeddings: false,
        positional_encoding: PositionalEncoding::Rope,
        mlp_type: MlpType::SwiGlu,
        qk_norm: true,
    }
}

// ---------------------------------------------------------------------------
// Ground truth: a REAL Qwen3.5 file (#3346)
// ---------------------------------------------------------------------------
//
// Every number below was read out of `~/models/Qwen3.5-0.8B-Q4_K_M.gguf`
// (sha256 `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517`,
// GGUF v3, 320 tensors) by parsing the file header directly on 2026-09-16 —
// not from a model card, a memory, or the family descriptor. The descriptor
// `contracts/model-families/qwen3_5.yaml` declares only the 9b and 27b
// variants, and no Qwen3.5-9B file is on this box, so the 0.8B file is the
// only Qwen3.5 whose true tensor inventory can be MEASURED here. It is the
// oracle for the shape arithmetic: if the config-derived count and this
// inventory disagree, the arithmetic is wrong.
//
// Shapes are GGUF `ne` order (`[in, out]` for a 2-D weight); only the element
// COUNT matters for a parameter total, so the order is reproduced verbatim
// rather than transposed.

/// One Gated DeltaNet layer of `Qwen3.5-0.8B-Q4_K_M.gguf` (measured: `blk.0`,
/// identical for the 18 layers whose index is not `interval-1 mod interval`).
const QWEN35_0_8B_GDN_LAYER: &[(&str, &[usize])] = &[
    ("attn_gate.weight", &[1024, 2048]),
    ("attn_norm.weight", &[1024]),
    ("attn_qkv.weight", &[1024, 6144]),
    ("ffn_down.weight", &[3584, 1024]),
    ("ffn_gate.weight", &[1024, 3584]),
    ("ffn_up.weight", &[1024, 3584]),
    ("post_attention_norm.weight", &[1024]),
    ("ssm_a", &[16]),
    ("ssm_alpha.weight", &[1024, 16]),
    ("ssm_beta.weight", &[1024, 16]),
    ("ssm_conv1d.weight", &[4, 6144]),
    ("ssm_dt.bias", &[16]),
    ("ssm_norm.weight", &[128]),
    ("ssm_out.weight", &[2048, 1024]),
];

/// One full-attention layer of the same file (measured: `blk.3`, identical for
/// the 6 layers at indices 3, 7, 11, 15, 19, 23 — `full_attention_interval` 4).
///
/// Two shapes here are NOT what dense/GQA accounting predicts, and both are
/// facts of the file: `attn_q` is `[1024, 4096]` = `2 * num_heads * head_dim`
/// (Qwen3.5 gates the attention output, so the q projection emits the gate
/// alongside the query — `attn_output` is `[2048, 1024]`, confirming
/// `num_heads * head_dim` = 2048), and `attn_q_norm`/`attn_k_norm` are present
/// at `head_dim`.
const QWEN35_0_8B_ATTENTION_LAYER: &[(&str, &[usize])] = &[
    ("attn_k.weight", &[1024, 512]),
    ("attn_k_norm.weight", &[256]),
    ("attn_norm.weight", &[1024]),
    ("attn_output.weight", &[2048, 1024]),
    ("attn_q.weight", &[1024, 4096]),
    ("attn_q_norm.weight", &[256]),
    ("attn_v.weight", &[1024, 512]),
    ("ffn_down.weight", &[3584, 1024]),
    ("ffn_gate.weight", &[1024, 3584]),
    ("ffn_up.weight", &[1024, 3584]),
    ("post_attention_norm.weight", &[1024]),
];

/// The file's non-layer tensors. There is no `output.weight`: the 0.8B TIES
/// its unembedding to the embedding, unlike the 9b descriptor.
const QWEN35_0_8B_GLOBAL: &[(&str, &[usize])] = &[
    ("output_norm.weight", &[1024]),
    ("token_embd.weight", &[1024, 248_320]),
];

/// Number of GDN and full-attention layers in the measured file (24 blocks,
/// `qwen35.full_attention_interval` = 4).
const QWEN35_0_8B_GDN_LAYERS: u64 = 18;
const QWEN35_0_8B_ATTENTION_LAYERS: u64 = 6;

/// Total elements of a measured tensor list.
fn tensor_elements(tensors: &[(&str, &[usize])]) -> u64 {
    tensors
        .iter()
        .map(|(_, dims)| u64::try_from(dims.iter().product::<usize>()).unwrap_or(u64::MAX))
        .sum()
}

/// `Qwen3.5-0.8B` as the GGUF's own metadata keys describe it: `block_count`
/// 24, `embedding_length` 1024, `feed_forward_length` 3584,
/// `attention.head_count` 8, `attention.head_count_kv` 2, `attention.key_length`
/// 256, `rope.freq_base` 1e7, `attention.layer_norm_rms_epsilon` 1e-6, and a
/// 248320-token vocabulary (`token_embd.weight` is `[1024, 248320]`).
fn qwen35_0_8b_size() -> ModelSizeConfig {
    ModelSizeConfig {
        parameters: "0.8B".to_string(),
        hidden_dim: 1024,
        num_layers: 24,
        num_heads: 8,
        num_kv_heads: 2,
        intermediate_dim: 3584,
        vocab_size: 248_320,
        max_position_embeddings: 262_144,
        head_dim: 256,
        rope_theta: 10_000_000.0,
        norm_eps: 1e-6,
    }
}

/// The measured file total: `752,393,024` parameters.
const QWEN35_0_8B_MEASURED_TOTAL: u64 = 752_393_024;

#[test]
fn qwen35_0_8b_measured_inventory_sums_to_the_file_total() {
    // Tensor COUNT: 14 per GDN layer, 11 per attention layer, 2 global = 320,
    // which is what the GGUF header declares (`n_tensors`).
    let counted = QWEN35_0_8B_GDN_LAYER.len() * 18 + QWEN35_0_8B_ATTENTION_LAYER.len() * 6 + 2;
    assert_eq!(counted, 320, "GGUF header declares 320 tensors");

    assert_eq!(tensor_elements(QWEN35_0_8B_GDN_LAYER), 21_555_360);
    assert_eq!(tensor_elements(QWEN35_0_8B_ATTENTION_LAYER), 18_352_640);
    assert_eq!(tensor_elements(QWEN35_0_8B_GLOBAL), 254_280_704);

    let total = tensor_elements(QWEN35_0_8B_GLOBAL)
        + QWEN35_0_8B_GDN_LAYERS * tensor_elements(QWEN35_0_8B_GDN_LAYER)
        + QWEN35_0_8B_ATTENTION_LAYERS * tensor_elements(QWEN35_0_8B_ATTENTION_LAYER);
    assert_eq!(total, QWEN35_0_8B_MEASURED_TOTAL);
}

/// The defect of #3346, as a number rather than a claim: dense/GQA accounting
/// applied to a hybrid family under-counts a REAL file by 107,992,896
/// parameters — 14.4% of the model. Three quarters of the layers are Gated
/// DeltaNet, and none of their conv/gate/state tensors have a term here.
#[test]
fn dense_accounting_cannot_reproduce_the_measured_qwen35_0_8b_file() {
    let size = qwen35_0_8b_size();
    let mut constraints = qwen35_constraints();
    constraints.tied_embeddings = true; // measured: the file has no output.weight
    let layers = uniform_layers(&size, &constraints);
    let p = model_parameter_count(&size, &constraints, &layers);

    assert_eq!(p.total, 644_400_128);
    assert_eq!(QWEN35_0_8B_MEASURED_TOTAL - p.total, 107_992_896);
}

// ---------------------------------------------------------------------------
// model_parameter_count
// ---------------------------------------------------------------------------

#[test]
fn worked_example_attention_layer_params() {
    let p = attention_layer_params(&toy_size(), &qwen35_constraints());
    // d_attn = d*q_dim + 2*d*kv_dim + q_dim*d = 4*4 + 2*4*2 + 4*4 = 48
    assert_eq!(p.d_attn, 48);
    // d_ffn = 3*d*d_ff = 3*4*8 = 96
    assert_eq!(p.d_ffn, 96);
    // d_norm = 2*d = 8
    assert_eq!(p.d_norm, 8);
    assert_eq!(p.total(), 152);
}

#[test]
fn worked_example_model_parameter_count() {
    let size = toy_size();
    let constraints = qwen35_constraints();
    let layers = uniform_layers(&size, &constraints);
    let p = model_parameter_count(&size, &constraints, &layers);

    assert_eq!(p.embedding, 40); // V*d = 10*4
    assert_eq!(p.layers, 304); // L*(48+96+8) = 2*152
    assert_eq!(p.final_norm, 4); // d_final = d
    assert_eq!(p.unembedding, 40); // untied lm_head = V*d
    assert_eq!(p.total, 388); // P = 40 + 304 + 4 + 40
}

#[test]
fn tied_embeddings_drop_the_trailing_v_times_d() {
    let size = toy_size();
    let mut constraints = qwen35_constraints();
    constraints.tied_embeddings = true;
    let layers = uniform_layers(&size, &constraints);
    let p = model_parameter_count(&size, &constraints, &layers);

    assert_eq!(p.unembedding, 0);
    assert_eq!(p.total, 348); // 388 - 40
}

#[test]
fn bias_adds_exactly_the_four_projection_bias_vectors() {
    let size = toy_size();
    let mut constraints = qwen35_constraints();
    constraints.has_bias = true;
    let p = attention_layer_params(&size, &constraints);
    // q_dim + 2*kv_dim + d = 4 + 4 + 4 = 12
    assert_eq!(p.d_attn, 48 + 12);
}

/// QE2E-INV-001 wants `P(Qwen3.5-9B) ∈ [9.0B, 9.2B]`. Dense/GQA accounting
/// gives 8.21B, because three of every four Qwen3.5 layers are Gated DeltaNet
/// and their `d_attn` is sized by `inner_size`/`state_size`/`conv_kernel`/
/// `group_count`, none of which exist in `ModelConstraints`. This test pins the
/// number so the gap is a measured fact, not a claim.
#[test]
fn qwen35_9b_uniform_layers_do_not_reach_the_invariant_range() {
    let size = qwen35_9b_size();
    let constraints = qwen35_constraints();
    let layers = uniform_layers(&size, &constraints);
    let p = model_parameter_count(&size, &constraints, &layers);

    assert_eq!(p.embedding, 1_017_118_720);
    assert_eq!(p.unembedding, 1_017_118_720);
    assert_eq!(p.total, 8_208_519_168);
    assert!(
        p.total < 9_000_000_000,
        "QE2E-INV-001 is NOT discharged by dense accounting: got {}",
        p.total
    );
}

/// A hybrid model is a slice of DIFFERENT per-layer costs — the reason the
/// function takes `&[LayerParams]` instead of multiplying one layer by L.
#[test]
fn hybrid_layers_sum_their_individual_costs() {
    let size = toy_size();
    let constraints = qwen35_constraints();
    let attn = attention_layer_params(&size, &constraints);
    let gdn = LayerParams {
        d_attn: 1_000,
        ..attn
    };
    let layers = vec![attn, gdn, gdn, gdn];
    let p = model_parameter_count(&size, &constraints, &layers);

    assert_eq!(p.layers, attn.total() + 3 * gdn.total());
}

// ---------------------------------------------------------------------------
// flops_per_token
// ---------------------------------------------------------------------------

#[test]
fn worked_example_flops_per_token() {
    let f = flops_per_token(388, 3, 4, 2);
    assert_eq!(f.dense, 776); // 2*P
    assert_eq!(f.attention, 48); // 2*L*S*d = 2*2*3*4
    assert_eq!(f.total, 824);
}

#[test]
fn gated_deltanet_layers_contribute_no_seq_len_term() {
    // attn_layers = 0: the whole cost is 2*P, no O(S*d) term. The contract's
    // "GDN FLOP component is O(d^2) per token (no quadratic)" invariant.
    let f = flops_per_token(388, 100_000, 4, 0);
    assert_eq!(f.attention, 0);
    assert_eq!(f.total, f.dense);
}

// ---------------------------------------------------------------------------
// memory_breakdown
// ---------------------------------------------------------------------------

fn toy_plan() -> InferencePlan {
    InferencePlan {
        seq_len: 3,
        batch_size: 1,
        kv_layers: 2,
        weights: Precision::F16,
        kv_cache: Precision::F16,
        activations: Precision::F32,
    }
}

#[test]
fn worked_example_memory_breakdown() {
    let m = memory_breakdown(&toy_size(), 388, &toy_plan());
    assert_eq!(m.weights, 776); // 388 params * 2 bytes
    assert_eq!(m.kv, 48); // 2*kv_layers*n_kv*d_k*S = 24 elements * 2 bytes
    assert_eq!(m.activations, 48); // batch*S*d = 12 elements * 4 bytes
    assert_eq!(m.total, 872);
}

#[test]
fn worked_example_precision_bytes() {
    // Two whole K-quant super-blocks.
    assert_eq!(Precision::Q4K.bytes_for(512), 288); // 2 * 144
    assert_eq!(Precision::Q5K.bytes_for(512), 352); // 2 * 176
    assert_eq!(Precision::Q6K.bytes_for(512), 420); // 2 * 210
    assert_eq!(Precision::F16.bytes_for(512), 1024);
    assert_eq!(Precision::F32.bytes_for(512), 2048);
    // A partial super-block rounds up to one whole block.
    assert_eq!(Precision::Q4K.bytes_for(1), 144);
}

#[test]
fn quantization_ordering_is_false_below_one_super_block() {
    // QE2E-ORD-003's precondition `n % 256 == 0` exists for exactly this:
    // one 210-byte Q6K block exceeds 2n F16 bytes for n <= 105.
    assert!(Precision::Q6K.bytes_for(100) > Precision::F16.bytes_for(100));
}

// ---------------------------------------------------------------------------
// throughput_model
// ---------------------------------------------------------------------------

#[test]
fn worked_example_throughput_model() {
    // bandwidth/bytes = 1000/100 = 10; compute/flops = 10000/500 = 20.
    let t = throughput_model(1000.0, 10_000.0, 100.0, 500.0);
    assert!((t.memory_bound - 10.0).abs() < 1e-9);
    assert!((t.compute_bound - 20.0).abs() < 1e-9);
    assert!((t.tokens_per_second - 10.0).abs() < 1e-9);
    assert_eq!(t.limit, RooflineLimit::MemoryBound);
}

#[test]
fn compute_bound_when_the_compute_term_is_smaller() {
    let t = throughput_model(1_000_000.0, 1000.0, 100.0, 500.0);
    assert!((t.tokens_per_second - 2.0).abs() < 1e-9);
    assert_eq!(t.limit, RooflineLimit::ComputeBound);
}

#[test]
fn zero_denominator_yields_zero_not_nan() {
    let t = throughput_model(1000.0, 10_000.0, 0.0, 0.0);
    assert!(t.tokens_per_second.is_finite());
    assert!((t.tokens_per_second - 0.0).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// contract_composition
// ---------------------------------------------------------------------------

#[test]
fn worked_example_contract_composition() {
    let size = toy_size();
    let stages = contract_composition(&size, 3);

    // embedding + L blocks + final_norm + unembed
    assert_eq!(stages.len(), 2 + 3);
    assert_eq!(stages[0].component, "embedding");
    assert_eq!(stages[0].input, vec![3]);
    assert_eq!(stages[0].output, vec![3, 4]);
    assert_eq!(stages[1].component, "block_0");
    assert_eq!(stages[2].component, "block_1");
    assert_eq!(stages[3].component, "final_norm");
    assert_eq!(stages[4].component, "unembed");
    // QE2E-CON-007: tokens in -> logits out, shape [seq_len, V].
    assert_eq!(stages[4].output, vec![3, 10]);
    assert!(composition_is_well_formed(&stages));
}

#[test]
fn every_block_stage_preserves_shape() {
    let stages = contract_composition(&toy_size(), 7);
    for stage in stages
        .iter()
        .filter(|s| s.component.starts_with("block_") || s.component == "final_norm")
    {
        assert!(
            stage.preserves_shape(),
            "QE2E-INV-006 falsified by {}",
            stage.component
        );
    }
    // The two stages that legitimately change shape.
    assert!(!stages[0].preserves_shape());
    assert!(!stages[stages.len() - 1].preserves_shape());
}

#[test]
fn a_broken_composition_is_detected() {
    let mut stages = contract_composition(&toy_size(), 3);
    stages[2].output = vec![3, 99];
    assert!(!composition_is_well_formed(&stages));
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

#[cfg(test)]
mod properties {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// `P` is additive in layers: appending one layer adds exactly that
        /// layer's `(d_attn + d_ffn + d_norm)`.
        #[test]
        fn parameter_count_is_additive_in_layers(
            n in 0usize..40,
            d_attn in 0u64..100_000,
            d_ffn in 0u64..100_000,
            d_norm in 0u64..10_000,
        ) {
            let size = toy_size();
            let constraints = qwen35_constraints();
            let layer = LayerParams { d_attn, d_ffn, d_norm };

            let short = vec![layer; n];
            let long = vec![layer; n + 1];
            let p_short = model_parameter_count(&size, &constraints, &short);
            let p_long = model_parameter_count(&size, &constraints, &long);

            prop_assert_eq!(p_long.total - p_short.total, layer.total());
        }

        /// `F` is linear in `P`: the dense term is exactly `2*P` for any `P`.
        #[test]
        fn flops_dense_term_is_twice_the_parameter_count(p in 0u64..1_000_000_000) {
            let f = flops_per_token(p, 512, 4096, 32);
            prop_assert_eq!(f.dense, 2 * p);
            prop_assert!(f.total >= f.dense);
        }

        /// QE2E-MON-004: `bw1 < bw2 -> tok_s(bw1) <= tok_s(bw2)`.
        #[test]
        fn throughput_is_monotone_in_bandwidth(
            bw1 in 0.0f64..1e12,
            delta in 0.0f64..1e12,
            compute in 1.0f64..1e15,
            bytes in 1.0f64..1e9,
            flops in 1.0f64..1e12,
        ) {
            let slow = throughput_model(bw1, compute, bytes, flops);
            let fast = throughput_model(bw1 + delta, compute, bytes, flops);
            prop_assert!(
                fast.tokens_per_second >= slow.tokens_per_second,
                "monotonicity falsified: {} -> {}",
                slow.tokens_per_second,
                fast.tokens_per_second
            );
        }

        /// QE2E-ORD-003 at whole super-blocks:
        /// `M(Q4K) < M(Q5K) < M(Q6K) < M(F16) < M(F32)`.
        #[test]
        fn quantization_memory_ordering_holds_at_whole_super_blocks(blocks in 1u64..10_000) {
            let n = blocks * 256;
            prop_assert!(Precision::Q4K.bytes_for(n) < Precision::Q5K.bytes_for(n));
            prop_assert!(Precision::Q5K.bytes_for(n) < Precision::Q6K.bytes_for(n));
            prop_assert!(Precision::Q6K.bytes_for(n) < Precision::F16.bytes_for(n));
            prop_assert!(Precision::F16.bytes_for(n) < Precision::F32.bytes_for(n));
        }

        /// `M_kv` is linear in sequence length: doubling `S` doubles the KV term
        /// and leaves `M_weights` alone.
        #[test]
        fn kv_memory_is_linear_in_sequence_length(seq_len in 1u64..4096, params in 0u64..1_000_000) {
            let size = toy_size();
            let plan = InferencePlan { seq_len, ..toy_plan() };
            let doubled = InferencePlan { seq_len: seq_len * 2, ..toy_plan() };

            let m1 = memory_breakdown(&size, params, &plan);
            let m2 = memory_breakdown(&size, params, &doubled);

            prop_assert_eq!(m2.kv, m1.kv * 2);
            prop_assert_eq!(m2.weights, m1.weights);
        }

        /// The composition always chains, every block preserves the residual
        /// shape, and the pipeline ends at `[seq_len, V]`.
        #[test]
        fn composition_chains_and_ends_in_logits(
            seq_len in 1usize..512,
            num_layers in 0usize..64,
        ) {
            let mut size = toy_size();
            size.num_layers = num_layers;
            let stages = contract_composition(&size, seq_len);

            prop_assert_eq!(stages.len(), num_layers + 3);
            prop_assert!(composition_is_well_formed(&stages));
            for stage in stages.iter().filter(|s| s.component.starts_with("block_")) {
                prop_assert!(stage.preserves_shape());
            }
            let last = &stages[stages.len() - 1];
            prop_assert_eq!(last.output.clone(), vec![seq_len, size.vocab_size]);
        }
    }
}
