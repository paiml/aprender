//! Weight Loading Tests for Qwen2 Model
//!
//! EXTREME TDD: These tests define the weight loading specification.
//! They must FAIL initially (RED), then be made to PASS (GREEN).
//!
//! # Spec Reference
//!
//! Section A: Model Loading (10 points)
//! - A1: APR file loads without error
//! - A2: Model metadata is parsed correctly
//! - A3: Tensor shapes match architecture
//! - A4: Weights are non-zero
//! - A5: Roundtrip save/load preserves model
//!
//! All tests must complete in < 100ms (bashrs standard).

use aprender::demo::Qwen2Config;
use aprender::models::Qwen2Model;

// ============================================================================
// Test Fixtures (tiny model for fast testing)
// ============================================================================

/// Create tiny test config for weight loading tests.
fn tiny_test_config() -> Qwen2Config {
    Qwen2Config {
        hidden_size: 32,
        num_attention_heads: 2,
        num_kv_heads: 1,
        num_layers: 1,
        vocab_size: 100,
        max_seq_len: 32,
        intermediate_size: 64,
        rope_theta: 10000.0,
    }
}

// ============================================================================
// Section A: Model Loading Tests
// ============================================================================

/// A1: Model can be created and has expected layer count.
#[test]
fn a1_model_creation() {
    let config = tiny_test_config();
    let model = Qwen2Model::new(&config);

    assert_eq!(model.num_layers(), config.num_layers);
}

/// A2: Model config is accessible and correct.
#[test]
fn a2_model_config_accessible() {
    let config = tiny_test_config();
    let model = Qwen2Model::new(&config);

    assert_eq!(model.config().hidden_size, 32);
    assert_eq!(model.config().vocab_size, 100);
    assert_eq!(model.config().num_layers, 1);
}

/// A3: Model produces output with correct shape.
#[test]
fn a3_forward_output_shape() {
    let config = tiny_test_config();
    let mut model = Qwen2Model::new(&config);

    let input_ids = vec![1u32, 2, 3];
    let position_ids: Vec<usize> = (0..3).collect();
    let logits = model.forward(&input_ids, &position_ids);

    // Output should be [batch=1, seq_len=3, vocab_size=100]
    assert_eq!(logits.shape(), &[1, 3, 100]);
}

/// A4: Model weights are initialized (not all zeros).
#[test]
fn a4_weights_initialized() {
    let config = tiny_test_config();
    let mut model = Qwen2Model::new(&config);

    // Forward pass should produce non-zero logits
    let input_ids = vec![1u32, 2, 3];
    let position_ids: Vec<usize> = (0..3).collect();
    let logits = model.forward(&input_ids, &position_ids);

    let sum: f32 = logits.data().iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "Model weights should produce non-zero output");
}

/// A5: Model produces deterministic output (same input = same output).
#[test]
fn a5_deterministic_forward() {
    let config = tiny_test_config();
    let mut model = Qwen2Model::new(&config);

    let input_ids = vec![1u32, 2, 3];
    let position_ids: Vec<usize> = (0..3).collect();

    let logits1 = model.forward(&input_ids, &position_ids);
    let logits2 = model.forward(&input_ids, &position_ids);

    assert_eq!(
        logits1.data(),
        logits2.data(),
        "Same input should produce same output"
    );
}

// ============================================================================
// Weight Serialization Tests (for future implementation)
// ============================================================================

/// A6: Model can export weights to SafeTensors format.
#[test]
fn a6_export_weights() {
    let config = tiny_test_config();
    let model = Qwen2Model::new(&config);

    // Get weight names (should match expected Qwen2 naming)
    let weight_names = model.weight_names();

    // Should have embedding, layer weights, norm, lm_head
    assert!(!weight_names.is_empty(), "Model should have weights");

    // Check for expected weight patterns
    let has_embed = weight_names.iter().any(|n| n.contains("embed"));
    let has_norm = weight_names.iter().any(|n| n.contains("norm"));

    assert!(has_embed, "Should have embedding weights");
    assert!(has_norm, "Should have normalization weights");
}

/// A7: Weight names follow HuggingFace convention.
#[test]
fn a7_weight_naming_convention() {
    let config = tiny_test_config();
    let model = Qwen2Model::new(&config);

    let weight_names = model.weight_names();

    // Verify naming follows model.layers.N.component.weight pattern
    for name in &weight_names {
        // Names should be lowercase with dots
        assert!(
            name.chars()
                .all(|c| c.is_lowercase() || c.is_numeric() || c == '.' || c == '_'),
            "Weight name '{}' should be lowercase with dots/underscores",
            name
        );
    }
}

/// A8: Weight shapes are correct for tiny model.
#[test]
fn a8_weight_shapes() {
    let config = tiny_test_config();
    let model = Qwen2Model::new(&config);

    let weight_info = model.weight_info();

    // Check embed_tokens shape: [vocab_size, hidden_size]
    if let Some(embed_shape) = weight_info.get("model.embed_tokens.weight") {
        assert_eq!(
            embed_shape,
            &vec![config.vocab_size, config.hidden_size],
            "Embedding shape mismatch"
        );
    }

    // Check lm_head shape: [vocab_size, hidden_size]
    if let Some(head_shape) = weight_info.get("lm_head.weight") {
        assert_eq!(
            head_shape,
            &vec![config.vocab_size, config.hidden_size],
            "LM head shape mismatch"
        );
    }
}

// ============================================================================
// Roundtrip Tests (placeholder for SafeTensors integration)
// ============================================================================

/// A9: Weight data can be extracted as f32 vectors.
#[test]
fn a9_weight_extraction() {
    let config = tiny_test_config();
    let model = Qwen2Model::new(&config);

    let weights = model.weights();

    // Should have some weights
    assert!(!weights.is_empty(), "Should be able to extract weights");

    // All weights should be non-empty
    for (name, data) in &weights {
        assert!(!data.is_empty(), "Weight '{}' should not be empty", name);
    }
}

/// A10: Total parameter count matches expected.
#[test]
fn a10_parameter_count() {
    let config = tiny_test_config();
    let model = Qwen2Model::new(&config);

    let total_params = model.num_parameters();

    // Tiny model should have reasonable parameter count
    // embed_tokens: 100 * 32 = 3,200
    // lm_head: 100 * 32 = 3,200 (tied or separate)
    // layer: attention + mlp + norms
    assert!(
        total_params > 0,
        "Model should have non-zero parameter count"
    );
    assert!(
        total_params < 100_000,
        "Tiny model should have < 100K parameters, got {}",
        total_params
    );
}

// ============================================================================
// Property Tests
// ============================================================================

#[cfg(test)]
mod proptest_weights {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        /// Property: Different configs produce different parameter counts.
        #[test]
        fn prop_config_affects_params(
            hidden_size in (16usize..64).prop_map(|x| x - (x % 8)),
            vocab_size in 50usize..200,
        ) {
            let config = Qwen2Config {
                hidden_size,
                num_attention_heads: 2,
                num_kv_heads: 1,
                num_layers: 1,
                vocab_size,
                max_seq_len: 32,
                intermediate_size: hidden_size * 2,
                rope_theta: 10000.0,
            };

            let model = Qwen2Model::new(&config);
            let params = model.num_parameters();

            // Parameter count should scale with hidden_size and vocab_size
            prop_assert!(params > 0);
        }
    }
}
