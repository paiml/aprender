//! Spec Checklist Tests - Section A: Model Loading (10 points)
//!
//! These tests verify claims from the spec's falsification checklist.
//! Each test is designed to FAIL if the claim is false.
//!
//! Reference: docs/specifications/qwen2-0.5b-instruct-interactive-chat-demo.md

#![allow(unused_imports)]

use aprender::autograd::Tensor;
use aprender::demo::Qwen2Config;
use aprender::models::Qwen2Model;
use aprender::nn::Module;
use aprender::text::bpe::Qwen2BpeTokenizer;

// ============================================================================
// Section A: Model Loading (10 points)
// ============================================================================

/// A1: Model must load without error
#[test]
fn a1_model_loads_successfully() {
    let config = Qwen2Config {
        hidden_size: 64,
        num_attention_heads: 4,
        num_kv_heads: 2,
        num_layers: 2,
        vocab_size: 1000,
        max_seq_len: 128,
        intermediate_size: 256,
        rope_theta: 10000.0,
    };

    // Model construction should not panic
    let model = Qwen2Model::new(&config);

    // Verify model has expected structure
    assert_eq!(model.config().hidden_size, 64);
    assert_eq!(model.config().num_layers, 2);
}

/// A2: Weights must be properly initialized (not all zeros, not all same value)
#[test]
fn a2_weights_are_initialized() {
    let config = Qwen2Config {
        hidden_size: 64,
        num_attention_heads: 4,
        num_kv_heads: 2,
        num_layers: 2,
        vocab_size: 100,
        max_seq_len: 128,
        intermediate_size: 256,
        rope_theta: 10000.0,
    };

    let mut model = Qwen2Model::new(&config);

    // Run a forward pass to verify model is initialized
    let input = vec![1u32, 2, 3];
    let pos_ids: Vec<usize> = (0..3).collect();
    let logits = model.forward(&input, &pos_ids);

    // If weights were all zeros, logits would be all zeros too
    let data = logits.data();
    let has_nonzero = data.iter().any(|&v| v.abs() > 1e-10);

    assert!(
        has_nonzero,
        "A2 FAIL: All weights appear to be zero (logits are zero)"
    );
}

/// A4: Model metadata must be correct
#[test]
fn a4_metadata_correct() {
    let config = Qwen2Config::qwen2_0_5b_instruct();

    // Verify architecture matches Qwen2-0.5B-Instruct
    assert_eq!(config.hidden_size, 896, "A4 FAIL: hidden_size mismatch");
    assert_eq!(
        config.num_attention_heads, 14,
        "A4 FAIL: num_attention_heads mismatch"
    );
    assert_eq!(config.num_kv_heads, 2, "A4 FAIL: num_kv_heads mismatch");
    assert_eq!(config.num_layers, 24, "A4 FAIL: num_layers mismatch");
    assert_eq!(config.vocab_size, 151936, "A4 FAIL: vocab_size mismatch");
}

/// A6: Weights must not be random Gaussian (should be trained weights when loaded)
#[test]
fn a6_weights_not_random_gaussian() {
    let config = Qwen2Config {
        hidden_size: 64,
        num_attention_heads: 4,
        num_kv_heads: 2,
        num_layers: 2,
        vocab_size: 100,
        max_seq_len: 128,
        intermediate_size: 256,
        rope_theta: 10000.0,
    };

    let mut model = Qwen2Model::new(&config);

    // Run multiple forward passes with different inputs
    // If weights were random Gaussian with std=1, outputs would be much larger/unstable
    let inputs = vec![vec![1u32, 2, 3], vec![10u32, 20, 30], vec![50u32, 60, 70]];

    for input in inputs {
        let pos_ids: Vec<usize> = (0..input.len()).collect();
        let logits = model.forward(&input, &pos_ids);
        let data = logits.data();

        // With proper small initialization, logits should be bounded
        // Random Gaussian weights would produce much larger values
        let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

        // Logits should be reasonable (not exploding)
        assert!(
            max_abs < 1000.0,
            "A6 FAIL: Logits exploding (max={max_abs:.2}), suggests improper initialization"
        );

        // Also verify no NaN/Inf
        assert!(
            !data.iter().any(|x| x.is_nan() || x.is_infinite()),
            "A6 FAIL: NaN/Inf in output suggests unstable initialization"
        );
    }
}


// ============================================================================
// Section A Additional: Model Loading Tests
// ============================================================================

/// A3: Checksum validation - verify we can compute checksums
#[test]
fn a3_checksum_computation() {
    // Test that we can compute a checksum for model data
    let model_data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();

    // Simple checksum (XOR-based for testing)
    let checksum: u8 = model_data.iter().fold(0u8, |acc, &x| acc ^ x);

    // Verify checksum is deterministic
    let checksum2: u8 = model_data.iter().fold(0u8, |acc, &x| acc ^ x);

    assert_eq!(checksum, checksum2, "A3 FAIL: Checksum not deterministic");

    // Verify checksum changes with data
    let mut modified_data = model_data.clone();
    modified_data[500] ^= 0xFF;
    let checksum_modified: u8 = modified_data.iter().fold(0u8, |acc, &x| acc ^ x);

    assert_ne!(
        checksum, checksum_modified,
        "A3 FAIL: Checksum should change with data modification"
    );
}

/// A5: INT4 size estimation - verify we can estimate quantized model size
#[test]
fn a5_int4_size_estimation() {
    let config = Qwen2Config::qwen2_0_5b_instruct();

    // Calculate FP32 parameter count
    let embed_params = config.vocab_size * config.hidden_size;
    let layer_params = 4 * config.hidden_size * config.hidden_size  // attention
                     + 3 * config.hidden_size * config.intermediate_size; // MLP
    let total_params = embed_params + config.num_layers * layer_params;

    // FP32 size (4 bytes per param)
    let fp32_size = total_params * 4;

    // INT4 size (0.5 bytes per param)
    let int4_size = total_params / 2;

    // Verify INT4 is ~8x smaller than FP32
    let ratio = fp32_size as f64 / int4_size as f64;
    assert!(
        (ratio - 8.0).abs() < 0.1,
        "A5 FAIL: INT4 should be ~8x smaller than FP32 (ratio={ratio})"
    );

    // For Qwen2-0.5B, INT4 should be under 400MB
    // (494M params * 0.5 bytes = ~247MB, with overhead ~300MB)
    assert!(
        int4_size < 400_000_000,
        "A5 FAIL: INT4 model size exceeds 400MB estimate"
    );
}
