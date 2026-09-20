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
