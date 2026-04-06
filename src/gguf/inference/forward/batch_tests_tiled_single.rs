
#[test]
fn test_tiled_single_head_attention_various_tile_sizes() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    let seq_len = 4;
    let head_dim = 4;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 7) as f32 * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 5) as f32 * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 3) as f32 * 0.1)
        .collect();

    let standard = model
        .standard_single_head_attention(&q, &k, &v, seq_len, head_dim, scale)
        .expect("expected value");

    for tile_size in [1, 2, 3, 4, 8] {
        let tiled = model
            .tiled_single_head_attention(&q, &k, &v, seq_len, head_dim, scale, tile_size)
            .expect("expected value");
        for (s, t) in standard.iter().zip(tiled.iter()) {
            assert!((s - t).abs() < 1e-4, "tile_size={}", tile_size);
        }
    }
}

// ============================================================================
// tiled_causal_attention tests
// ============================================================================

#[test]
fn test_tiled_causal_attention_basic() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    let seq_len = 3;
    let head_dim = 4;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q = vec![1.0; seq_len * head_dim];
    let k = vec![1.0; seq_len * head_dim];
    let v = vec![1.0; seq_len * head_dim];

    let result = model.tiled_causal_attention(&q, &k, &v, seq_len, head_dim, scale, 2);
    assert!(result.is_ok());
    let output = result.expect("output");
    assert_eq!(output.len(), seq_len * head_dim);
}

#[test]
fn test_tiled_causal_attention_first_position() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    // First position can only attend to itself
    let seq_len = 3;
    let head_dim = 2;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q = vec![1.0; seq_len * head_dim];
    let k = vec![1.0; seq_len * head_dim];
    // Different V values for each position
    let v = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];

    let result = model.tiled_causal_attention(&q, &k, &v, seq_len, head_dim, scale, 1);
    assert!(result.is_ok());
    let output = result.expect("output");
    // First position output should be first V (since it only attends to itself)
    assert!((output[0] - 1.0).abs() < 1e-5);
    assert!((output[1] - 1.0).abs() < 1e-5);
}

#[test]
fn test_tiled_causal_attention_various_tile_sizes() {
    let config = test_config();
    let model = create_test_model_with_config(&config);

    let seq_len = 4;
    let head_dim = 4;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 7) as f32 * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 5) as f32 * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 3) as f32 * 0.1)
        .collect();

    // Use tile_size=1 as reference
    let reference = model
        .tiled_causal_attention(&q, &k, &v, seq_len, head_dim, scale, 1)
        .expect("expected value");

    for tile_size in [2, 3, 4, 8] {
        let tiled = model
            .tiled_causal_attention(&q, &k, &v, seq_len, head_dim, scale, tile_size)
            .expect("expected value");
        for (r, t) in reference.iter().zip(tiled.iter()) {
            assert!((r - t).abs() < 1e-4, "tile_size={}", tile_size);
        }
    }
}

// ============================================================================
// GPU feature tests (conditional compilation)
// ============================================================================

#[cfg(feature = "gpu")]
mod gpu_tests {
    use super::*;
    use crate::gguf::DispatchMetrics;
    use std::sync::Arc;

    #[test]
    fn test_forward_batch_with_cache_empty_tokens_error() {
        let config = test_config();
        let model = create_test_model_with_config(&config);
        let mut cache = OwnedQuantizedKVCache::from_config(&config, 100);
        let metrics = Arc::new(DispatchMetrics::new());

        let result = model.forward_batch_with_cache(&[], &mut cache, &metrics);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{:?}", err).contains("empty"));
    }

    #[test]
    fn test_forward_batch_with_cache_single_token() {
        let config = test_config();
        let model = create_test_model_with_config(&config);
        let mut cache = OwnedQuantizedKVCache::from_config(&config, 100);
        let metrics = Arc::new(DispatchMetrics::new());

        let result = model.forward_batch_with_cache(&[1], &mut cache, &metrics);
        assert!(result.is_ok());
        let logits = result.expect("logits");
        assert_eq!(logits.len(), config.vocab_size);
        // Check that CPU dispatch was recorded
        assert!(metrics.cpu_dispatches() > 0);
    }

    #[test]
    fn test_generate_with_batched_prefill_empty_prompt_error() {
        let config = test_config();
        let model = create_test_model_with_config(&config);
        let gen_config = QuantizedGenerateConfig::deterministic(5);
        let metrics = Arc::new(DispatchMetrics::new());

        let result = model.generate_with_batched_prefill(&[], &gen_config, &metrics);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{:?}", err).contains("empty"));
    }

    #[test]
    fn test_generate_with_batched_prefill_basic() {
        let config = test_config();
        let model = create_test_model_with_config(&config);
        let gen_config = QuantizedGenerateConfig::deterministic(2);
        let metrics = Arc::new(DispatchMetrics::new());

        let result = model.generate_with_batched_prefill(&[1], &gen_config, &metrics);
        assert!(result.is_ok());
        let tokens = result.expect("tokens");
        assert!(!tokens.is_empty());
        assert_eq!(tokens[0], 1);
    }

    #[test]
    fn test_generate_with_batched_prefill_temperature() {
        let config = test_config();
        let model = create_test_model_with_config(&config);
        let gen_config = QuantizedGenerateConfig::default()
            .with_max_tokens(2)
            .with_temperature(0.8)
            .with_top_k(5);
        let metrics = Arc::new(DispatchMetrics::new());

        let result = model.generate_with_batched_prefill(&[1], &gen_config, &metrics);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reshape_for_parallel_heads_basic() {
        let config = test_config();
        let model = create_test_model_with_config(&config);

        let seq_len = 2;
        let num_heads = 4;
        let head_dim = 16;
        let input: Vec<f32> = (0..seq_len * num_heads * head_dim)
            .map(|i| i as f32)
            .collect();

        let result = model.reshape_for_parallel_heads(&input, seq_len, num_heads, head_dim);
        assert!(result.is_ok());
        let reshaped = result.expect("reshaped");
        assert_eq!(reshaped.len(), num_heads * seq_len * head_dim);
    }

    #[test]
    fn test_reshape_for_parallel_heads_invalid_size() {
        let config = test_config();
        let model = create_test_model_with_config(&config);

        let seq_len = 2;
        let num_heads = 4;
        let head_dim = 16;
        // Wrong input size
        let input: Vec<f32> = vec![1.0; 10];

        let result = model.reshape_for_parallel_heads(&input, seq_len, num_heads, head_dim);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_causal_mask_softmax_basic() {
        let config = test_config();
        let model = create_test_model_with_config(&config);

        let seq_len = 3;
        // 3x3 scores
        let scores: Vec<f32> = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];

        let weights = model.apply_causal_mask_softmax(&scores, seq_len);
        assert_eq!(weights.len(), seq_len * seq_len);

        // First row: only position 0 is unmasked
        assert!((weights[0] - 1.0).abs() < 1e-6);
        assert!((weights[1]).abs() < 1e-6);
        assert!((weights[2]).abs() < 1e-6);

        // Second row: positions 0,1 are unmasked
        let row1_sum = weights[3] + weights[4];
        assert!((row1_sum - 1.0).abs() < 1e-6);
        assert!((weights[5]).abs() < 1e-6);

        // Third row: all positions are unmasked
        let row2_sum = weights[6] + weights[7] + weights[8];
        assert!((row2_sum - 1.0).abs() < 1e-6);
    }
}
