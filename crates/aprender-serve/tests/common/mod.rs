//! Shared fixtures for aprender-serve's integration tests (#3809).
//!
//! WHY THIS DIRECTORY EXISTS. `create_test_model()` lived inside
//! `gpu_cpu_trace_compare.rs`'s `#[cfg(all(test, feature = "cuda"))]` module, and
//! Rust integration tests cannot import from one another — so when #3809 needed a
//! GPU-free CPU control over the SAME fixture, the only way to get one was to copy
//! it (`apr_trace_fixture_cpu_half.rs`). A copy that drifts stops testing the thing
//! it exists for: the two would silently diverge and the control would stop being a
//! control. This module is the single definition both use.
//!
//! It is deliberately free of `feature = "cuda"`: the whole point is that the CPU
//! control can build it on a machine with no card.
//!
//! `tests/common/mod.rs` (the directory form) rather than `tests/common.rs`, so
//! cargo treats it as a module and not as its own test binary.
#![allow(dead_code)] // each test target uses only the fixtures it needs
#![allow(clippy::needless_range_loop)]

use realizar::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};

/// The fixture both the CUDA comparison and the CPU control drive.
///
/// `lm_head_tied: false` is not a guess: this fixture materialises a separate
/// `lm_head_weight` below, which is exactly the case the field's own doc names
/// as its default ("models that always materialized a separate lm_head_weight").
pub fn create_test_model() -> AprTransformer {
    let hidden_dim = 64;
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let intermediate_dim = 128;
    let vocab_size = 256;

    let config = AprTransformerConfig {
        architecture: "test".to_string(),
        hidden_dim,
        num_layers: 1,
        num_heads,
        num_kv_heads,
        vocab_size,
        intermediate_dim,
        context_length: 32,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let token_embedding: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();

    let output_norm_weight = vec![1.0f32; hidden_dim];

    let lm_head_weight: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| ((i as f32) * 0.001).cos())
        .collect();

    let qkv_out_dim = hidden_dim + 2 * kv_dim;
    let qkv_weight: Vec<f32> = (0..qkv_out_dim * hidden_dim)
        .map(|i| ((i as f32) * 0.01).sin() * 0.1)
        .collect();

    let attn_output_weight: Vec<f32> = (0..hidden_dim * hidden_dim)
        .map(|i| ((i as f32) * 0.02).cos() * 0.1)
        .collect();

    let attn_norm_weight = vec![1.0f32; hidden_dim];

    let ffn_up_weight: Vec<f32> = (0..intermediate_dim * hidden_dim)
        .map(|i| ((i as f32) * 0.03).sin() * 0.1)
        .collect();
    let ffn_down_weight: Vec<f32> = (0..hidden_dim * intermediate_dim)
        .map(|i| ((i as f32) * 0.04).cos() * 0.1)
        .collect();
    let ffn_gate_weight: Vec<f32> = (0..intermediate_dim * hidden_dim)
        .map(|i| ((i as f32) * 0.05).sin() * 0.1)
        .collect();
    let ffn_norm_weight = vec![1.0f32; hidden_dim];

    let layer = AprTransformerLayer {
        qkv_weight,
        qkv_bias: None,
        attn_output_weight,
        attn_output_bias: None,
        attn_norm_weight,
        attn_norm_bias: None,
        ffn_up_weight,
        ffn_up_bias: None,
        ffn_down_weight,
        ffn_down_bias: None,
        ffn_gate_weight: Some(ffn_gate_weight),
        ffn_gate_bias: None,
        ffn_norm_weight: Some(ffn_norm_weight),
        ffn_norm_bias: None,
        attn_q_norm_weight: None,
        attn_k_norm_weight: None,
        linear_attn_z_weight: None,
        linear_attn_b_weight: None,
        linear_attn_a_weight: None,
        linear_attn_conv1d_weight: None,
        linear_attn_a_log: None,
        linear_attn_dt_bias: None,
        linear_attn_norm_weight: None,
        moe_gate_weight: None,
        moe_expert_gate_up: None,
        moe_expert_down: None,
        moe_shared_gate: None,
        moe_shared_up: None,
        moe_shared_down: None,
        moe_shared_expert_gate_weight: None,
    };

    AprTransformer {
        config,
        // This fixture materialises a separate `lm_head_weight`, so the tied
        // path is off — the field's own documented default.
        lm_head_tied: false,
        token_embedding,
        layers: vec![layer],
        output_norm_weight,
        output_norm_bias: None,
        lm_head_weight,
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q4k: None,
        lm_head_weight_q6k: None,
    }
}
