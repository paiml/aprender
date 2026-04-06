//! Tests for GGUF generation module (generation.rs)
//!
//! Covers:
//! - Generation loop logic (generate, generate_with_cache, generate_with_scratch)
//! - Sampling methods (argmax, sample_topk)
//! - Stopping conditions (stop_tokens, max_tokens)
//! - Streaming callback (generate_with_cache_streaming)

use crate::gguf::test_helpers::create_test_model_with_config;
use crate::gguf::{GGUFConfig, OwnedQuantizedModel, QuantizedGenerateConfig};

/// Create a minimal test config for generation tests
fn make_test_config() -> GGUFConfig {
    GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 1,
        vocab_size: 100,
        rope_theta: 10000.0,
        context_length: 256,
        eps: 1e-5,
        rope_type: 0,
        explicit_head_dim: None,
        bos_token_id: None,
        eos_token_id: None,
    }
}

// =============================================================================
// Argmax Tests
// =============================================================================

#[test]
fn test_argmax_basic() {
    let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
    let result = OwnedQuantizedModel::argmax(&logits);
    assert_eq!(result, 3, "argmax should return index of max value");
}

#[test]
fn test_argmax_first_element() {
    let logits = vec![1.0, 0.5, 0.3, 0.2];
    let result = OwnedQuantizedModel::argmax(&logits);
    assert_eq!(
        result, 0,
        "argmax should return 0 when first element is max"
    );
}

#[test]
fn test_argmax_last_element() {
    let logits = vec![0.1, 0.2, 0.3, 0.9];
    let result = OwnedQuantizedModel::argmax(&logits);
    assert_eq!(
        result, 3,
        "argmax should return last index when last element is max"
    );
}

#[test]
fn test_argmax_negative_values() {
    let logits = vec![-5.0, -2.0, -3.0, -1.0];
    let result = OwnedQuantizedModel::argmax(&logits);
    assert_eq!(result, 3, "argmax should work with negative values");
}

#[test]
fn test_argmax_ties_returns_last() {
    // max_by returns the last maximum element on ties
    let logits = vec![0.5, 0.5, 0.5];
    let result = OwnedQuantizedModel::argmax(&logits);
    assert_eq!(result, 2, "argmax with max_by returns last index on tie");
}

#[test]
fn test_argmax_empty_returns_zero() {
    let logits: Vec<f32> = vec![];
    let result = OwnedQuantizedModel::argmax(&logits);
    assert_eq!(result, 0, "argmax on empty slice returns 0");
}

#[test]
fn test_argmax_single_element() {
    let logits = vec![42.0];
    let result = OwnedQuantizedModel::argmax(&logits);
    assert_eq!(result, 0, "argmax on single element returns 0");
}

#[test]
fn test_argmax_with_nan_handling() {
    // NaN comparisons return Ordering::Equal via unwrap_or, so max_by returns last element
    // after the max before NaN
    let logits = vec![0.1, 0.5, f32::NAN, 0.3];
    let result = OwnedQuantizedModel::argmax(&logits);
    // NaN comparison returns Equal, so last element (0.3 at index 3) becomes max
    assert_eq!(
        result, 3,
        "NaN treated as Equal causes last element to be returned"
    );
}

#[test]
fn test_argmax_infinity() {
    let logits = vec![0.1, f32::INFINITY, 0.3];
    let result = OwnedQuantizedModel::argmax(&logits);
    assert_eq!(result, 1, "argmax should select infinity as max");
}

#[test]
fn test_argmax_neg_infinity() {
    let logits = vec![f32::NEG_INFINITY, -1.0, -2.0];
    let result = OwnedQuantizedModel::argmax(&logits);
    assert_eq!(result, 1, "argmax should not select neg_infinity");
}

// =============================================================================
// Sample Top-K Tests
// =============================================================================

#[test]
fn test_sample_topk_deterministic_with_top1() {
    // With top_k=1, should behave like argmax
    let logits = vec![0.1, 0.9, 0.3, 0.5];
    let result = OwnedQuantizedModel::sample_topk(&logits, 1.0, 1);
    assert_eq!(result, 1, "top_k=1 should select argmax");
}

#[test]
fn test_sample_topk_returns_valid_index() {
    let logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
    // With temperature=1.0 and top_k=3, should return one of top 3 indices
    for _ in 0..10 {
        let result = OwnedQuantizedModel::sample_topk(&logits, 1.0, 3);
        // Top 3 are indices 3 (0.9), 1 (0.5), 2 (0.3)
        assert!(
            result == 3 || result == 1 || result == 2,
            "sample_topk should return one of top 3 indices, got {}",
            result
        );
    }
}

#[test]
fn test_sample_topk_low_temperature_concentrates() {
    // Very low temperature should concentrate probability on max
    let logits = vec![0.0, 1.0, 0.5];
    let mut count_max = 0;
    for _ in 0..50 {
        let result = OwnedQuantizedModel::sample_topk(&logits, 0.01, 3);
        if result == 1 {
            count_max += 1;
        }
    }
    // With very low temp, should almost always pick max
    assert!(
        count_max >= 45,
        "Low temperature should heavily favor max token, got {} out of 50",
        count_max
    );
}

#[test]
fn test_sample_topk_high_temperature_distributes() {
    // High temperature should distribute more evenly
    let logits = vec![1.0, 1.0, 1.0]; // Equal logits
    let mut counts = [0, 0, 0];
    for _ in 0..300 {
        let result = OwnedQuantizedModel::sample_topk(&logits, 2.0, 3) as usize;
        counts[result] += 1;
    }
    // Each should get roughly 1/3 of samples
    for (i, &count) in counts.iter().enumerate() {
        assert!(
            count >= 50 && count <= 200,
            "Token {} got {} samples, expected ~100",
            i,
            count
        );
    }
}

#[test]
fn test_sample_topk_respects_topk_limit() {
    // With 5 logits and top_k=2, should only sample from top 2
    let logits = vec![0.1, 0.9, 0.2, 0.8, 0.3];
    // Top 2 are indices 1 (0.9) and 3 (0.8)
    for _ in 0..20 {
        let result = OwnedQuantizedModel::sample_topk(&logits, 1.0, 2);
        assert!(
            result == 1 || result == 3,
            "With top_k=2, should only sample indices 1 or 3, got {}",
            result
        );
    }
}

#[test]
fn test_sample_topk_empty_logits() {
    let logits: Vec<f32> = vec![];
    let result = OwnedQuantizedModel::sample_topk(&logits, 1.0, 5);
    assert_eq!(result, 0, "Empty logits should return 0");
}

#[test]
fn test_sample_topk_single_element() {
    let logits = vec![0.5];
    let result = OwnedQuantizedModel::sample_topk(&logits, 1.0, 5);
    assert_eq!(result, 0, "Single element should return 0");
}

// =============================================================================
// QuantizedGenerateConfig Tests
// =============================================================================

#[test]
fn test_generate_config_default() {
    let config = QuantizedGenerateConfig::default();
    assert_eq!(config.max_tokens, 64);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert!(config.stop_tokens.is_empty());
    assert!(!config.trace);
}

#[test]
fn test_generate_config_deterministic() {
    let config = QuantizedGenerateConfig::deterministic(32);
    assert_eq!(config.max_tokens, 32);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
}

#[test]
fn test_generate_config_builder_methods() {
    let config = QuantizedGenerateConfig::default()
        .with_max_tokens(128)
        .with_temperature(0.7)
        .with_top_k(40)
        .with_stop_tokens(vec![1, 2, 3])
        .with_trace(true);

    assert_eq!(config.max_tokens, 128);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_k, 40);
    assert_eq!(config.stop_tokens, vec![1, 2, 3]);
    assert!(config.trace);
}

// =============================================================================
// Generate Method Tests
// =============================================================================

#[test]
fn test_generate_empty_prompt_error() {
    let cfg = make_test_config();
    let model = create_test_model_with_config(&cfg);
    let gen_config = QuantizedGenerateConfig::deterministic(5);

    let result = model.generate(&[], &gen_config);
    assert!(result.is_err(), "Empty prompt should return error");

    let err = result.unwrap_err();
    let err_str = format!("{:?}", err);
    assert!(
        err_str.contains("empty") || err_str.contains("Empty"),
        "Error should mention empty prompt"
    );
}

#[test]
fn test_generate_returns_prompt_plus_tokens() {
    let cfg = make_test_config();
    let model = create_test_model_with_config(&cfg);
    let gen_config = QuantizedGenerateConfig::deterministic(3);
    let prompt = vec![1, 2, 3];

    let result = model.generate(&prompt, &gen_config).unwrap();

    // Should contain at least the prompt
    assert!(result.len() >= 3, "Result should contain prompt");
    assert_eq!(&result[..3], &prompt, "Result should start with prompt");
}

#[test]
fn test_generate_respects_max_tokens() {
    let cfg = make_test_config();
    let model = create_test_model_with_config(&cfg);
    let gen_config = QuantizedGenerateConfig::deterministic(2);
    let prompt = vec![1];

    let result = model.generate(&prompt, &gen_config).unwrap();

    // Max length = prompt.len() + max_tokens = 1 + 2 = 3
    assert!(
        result.len() <= 3,
        "Result should respect max_tokens, got len={}",
        result.len()
    );
}

#[test]
fn test_generate_stops_on_stop_token() {
    let cfg = make_test_config();
    let model = create_test_model_with_config(&cfg);

    // Use a stop token that matches whatever the model generates
    // Since test model is deterministic, first generated token will be consistent
    let prompt = vec![1];
    let first_gen = model
        .generate(&prompt, &QuantizedGenerateConfig::deterministic(1))
        .unwrap();

    if first_gen.len() > 1 {
        let stop_token = first_gen[1];
        let gen_config =
            QuantizedGenerateConfig::deterministic(10).with_stop_tokens(vec![stop_token]);

        let result = model.generate(&prompt, &gen_config).unwrap();

        // Should stop before generating stop_token
        assert!(
            !result[1..].contains(&stop_token),
            "Result should not contain stop token in generated portion"
        );
    }
}

#[test]
fn test_generate_greedy_is_deterministic() {
    let cfg = make_test_config();
    let model = create_test_model_with_config(&cfg);
    let gen_config = QuantizedGenerateConfig::deterministic(5);
    let prompt = vec![1, 2];

    let result1 = model.generate(&prompt, &gen_config).unwrap();
    let result2 = model.generate(&prompt, &gen_config).unwrap();

    assert_eq!(result1, result2, "Greedy decoding should be deterministic");
}

// =============================================================================
// Generate With Cache Tests
// =============================================================================

#[test]
fn test_generate_with_cache_empty_prompt_error() {
    let cfg = make_test_config();
    let model = create_test_model_with_config(&cfg);
    let gen_config = QuantizedGenerateConfig::deterministic(5);

    let result = model.generate_with_cache(&[], &gen_config);
    assert!(result.is_err(), "Empty prompt should return error");
}

#[test]
fn test_generate_with_cache_returns_prompt() {
    let cfg = make_test_config();
    let model = create_test_model_with_config(&cfg);
    let gen_config = QuantizedGenerateConfig::deterministic(3);
    let prompt = vec![5, 10, 15];

    let result = model.generate_with_cache(&prompt, &gen_config).unwrap();

    assert!(result.len() >= 3);
    assert_eq!(&result[..3], &prompt, "Result should start with prompt");
}

#[test]
fn test_generate_with_cache_respects_max_tokens() {
    let cfg = make_test_config();
    let model = create_test_model_with_config(&cfg);
    let gen_config = QuantizedGenerateConfig::deterministic(2);
    let prompt = vec![1, 2];

    let result = model.generate_with_cache(&prompt, &gen_config).unwrap();

    // Max = prompt.len() + max_tokens = 2 + 2 = 4
    assert!(result.len() <= 4, "Should respect max_tokens limit");
}

#[test]
fn test_generate_with_cache_deterministic() {
    let cfg = make_test_config();
    let model = create_test_model_with_config(&cfg);
    let gen_config = QuantizedGenerateConfig::deterministic(3);
    let prompt = vec![7];

    let result1 = model.generate_with_cache(&prompt, &gen_config).unwrap();
    let result2 = model.generate_with_cache(&prompt, &gen_config).unwrap();

    assert_eq!(
        result1, result2,
        "Greedy decoding with cache should be deterministic"
    );
}

// =============================================================================
// Generate With Scratch Tests
// =============================================================================

#[test]
fn test_generate_with_scratch_empty_prompt_error() {
    let cfg = make_test_config();
    let model = create_test_model_with_config(&cfg);
    let gen_config = QuantizedGenerateConfig::deterministic(5);

    let result = model.generate_with_scratch(&[], &gen_config);
    assert!(result.is_err(), "Empty prompt should return error");
}

#[test]
fn test_generate_with_scratch_returns_prompt() {
    let cfg = make_test_config();
    let model = create_test_model_with_config(&cfg);
    let gen_config = QuantizedGenerateConfig::deterministic(3);
    let prompt = vec![1, 2, 3];

    let result = model.generate_with_scratch(&prompt, &gen_config).unwrap();

    assert!(result.len() >= 3);
    assert_eq!(&result[..3], &prompt);
}

#[test]
fn test_generate_with_scratch_respects_max_tokens() {
    let cfg = make_test_config();
    let model = create_test_model_with_config(&cfg);
    let gen_config = QuantizedGenerateConfig::deterministic(1);
    let prompt = vec![1];

    let result = model.generate_with_scratch(&prompt, &gen_config).unwrap();

    // Max = 1 + 1 = 2
    assert!(result.len() <= 2, "Should respect max_tokens");
}

// =============================================================================
// Streaming Generation Tests
// =============================================================================

#[test]
fn test_generate_streaming_empty_prompt_error() {
    let cfg = make_test_config();
    let model = create_test_model_with_config(&cfg);
    let gen_config = QuantizedGenerateConfig::deterministic(5);

    let result = model.generate_with_cache_streaming(&[], &gen_config, |_| true);
    assert!(result.is_err(), "Empty prompt should return error");
}

include!("generation_tests_generate_streaming.rs");
