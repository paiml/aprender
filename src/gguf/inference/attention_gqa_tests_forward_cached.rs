
// =============================================================================
// Tests: forward_cached with GQA (autoregressive generation path)
// =============================================================================

/// Test forward_cached doesn't panic with GQA
#[test]
fn test_forward_cached_gqa_no_panic() {
    use crate::gguf::OwnedQuantizedKVCache;

    let model = create_gqa_model(64, 8, 2);
    // OwnedQuantizedKVCache::new(num_layers, hidden_dim, max_seq_len)
    let mut cache = OwnedQuantizedKVCache::new(
        model.config.num_layers,
        model.config.hidden_dim,
        128, // max_seq_len
    );

    // Process tokens sequentially (autoregressive)
    for pos in 0..5 {
        let token_id = (pos + 10) as u32;
        let result = model.forward_cached(token_id, &mut cache, pos);
        assert!(
            result.is_ok(),
            "forward_cached should succeed at position {}",
            pos
        );

        let logits = result.expect("logits");
        assert_eq!(logits.len(), 100, "Should have vocab_size logits");
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "All logits should be finite at position {}",
            pos
        );
    }
}

/// Test forward_cached with GQA 8:1 ratio (TinyLlama-like)
#[test]
fn test_forward_cached_gqa_8_to_1() {
    use crate::gguf::OwnedQuantizedKVCache;

    let model = create_gqa_model(256, 32, 4);
    let mut cache =
        OwnedQuantizedKVCache::new(model.config.num_layers, model.config.hidden_dim, 128);

    for pos in 0..3 {
        let result = model.forward_cached((pos + 1) as u32, &mut cache, pos);
        assert!(result.is_ok(), "forward_cached GQA 8:1 should succeed");
    }
}

/// Test forward_cached produces consistent outputs for GQA
#[test]
fn test_forward_cached_gqa_consistency() {
    use crate::gguf::OwnedQuantizedKVCache;

    let model = create_gqa_model(64, 8, 2);
    let mut cache =
        OwnedQuantizedKVCache::new(model.config.num_layers, model.config.hidden_dim, 128);

    // First token
    let logits1 = model
        .forward_cached(42, &mut cache, 0)
        .expect("first token");

    // Reset cache and try again - should get same result
    let mut cache2 =
        OwnedQuantizedKVCache::new(model.config.num_layers, model.config.hidden_dim, 128);
    let logits2 = model
        .forward_cached(42, &mut cache2, 0)
        .expect("same token");

    // Same input should produce same output
    assert_eq!(logits1.len(), logits2.len());
    for (a, b) in logits1.iter().zip(logits2.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "Logits should be identical for same input"
        );
    }
}
