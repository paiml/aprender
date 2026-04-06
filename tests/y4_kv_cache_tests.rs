//! Y4: APR KV Cache Tests (EXTREME TDD - RED Phase)
//!
//! Per Section Y of the spec, APR transformer MUST have optimized KV cache.
//! These tests define Popperian falsification conditions for Y4.
//!
//! FALSIFICATION: KV cache allocations during decode

// ============================================================================
// Y4.1: AprKVCache Struct Exists
// ============================================================================

/// Y4.1a: AprKVCache struct exists with correct interface
/// FALSIFICATION: Struct missing or wrong methods
#[test]
fn y4_1a_apr_kv_cache_struct_exists() {
    use realizar::apr_transformer::{AprKVCache, AprTransformerConfig};

    let config = AprTransformerConfig {
        hidden_dim: 256,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        context_length: 128,
        ..Default::default()
    };

    // Must be constructible with config
    let cache = AprKVCache::new(&config);

    // Must have len() method
    assert_eq!(cache.len(), 0, "New cache should be empty");

    // Must have capacity() method
    assert!(
        cache.capacity() >= 128,
        "Cache should have context_length capacity"
    );
}

/// Y4.1b: AprKVCache stores K and V separately per layer
/// FALSIFICATION: Single storage for K and V
#[test]
fn y4_1b_kv_cache_separate_storage() {
    use realizar::apr_transformer::{AprKVCache, AprTransformerConfig};

    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 4,
        context_length: 32,
        ..Default::default()
    };

    let mut cache = AprKVCache::new(&config);

    // Append K and V for layer 0
    let head_dim = config.hidden_dim / config.num_heads; // 16
    let k = vec![1.0f32; config.num_kv_heads * head_dim]; // [4, 16]
    let v = vec![2.0f32; config.num_kv_heads * head_dim];

    cache.append(0, &k, &v);

    // Retrieve should return separate K and V
    let (k_cache, v_cache) = cache.get(0);

    // K should contain 1.0s, V should contain 2.0s
    assert!(
        k_cache.iter().all(|&x| (x - 1.0).abs() < 1e-6),
        "K cache should contain K values"
    );
    assert!(
        v_cache.iter().all(|&x| (x - 2.0).abs() < 1e-6),
        "V cache should contain V values"
    );
}

// ============================================================================
// Y4.2: Forward with Cache
// ============================================================================

/// Y4.2a: AprTransformer has forward_with_cache method
/// FALSIFICATION: Method missing or wrong signature
#[test]
fn y4_2a_forward_with_cache_exists() {
    use realizar::apr_transformer::{AprKVCache, AprTransformer, AprTransformerConfig};

    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 32,
        ..Default::default()
    };

    let transformer = AprTransformer::new(config.clone());
    let mut cache = AprKVCache::new(&config);

    // forward_with_cache should exist and produce valid output
    let token_id = 1u32;
    let position = 0usize;
    let result = transformer.forward_with_cache(token_id, &mut cache, position);

    assert!(result.is_ok(), "forward_with_cache should succeed");

    let logits = result.expect("test");
    assert_eq!(
        logits.len(),
        config.vocab_size,
        "Should return vocab_size logits"
    );
}

/// Y4.2b: Cache grows after forward_with_cache
/// FALSIFICATION: Cache len() doesn't increase
#[test]
fn y4_2b_cache_grows_after_forward() {
    use realizar::apr_transformer::{AprKVCache, AprTransformer, AprTransformerConfig};

    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 32,
        ..Default::default()
    };

    let transformer = AprTransformer::new(config.clone());
    let mut cache = AprKVCache::new(&config);

    assert_eq!(cache.len(), 0, "Cache should start empty");

    // First token
    let _ = transformer.forward_with_cache(1, &mut cache, 0);
    assert_eq!(
        cache.len(),
        1,
        "Cache should have 1 position after first token"
    );

    // Second token
    let _ = transformer.forward_with_cache(2, &mut cache, 1);
    assert_eq!(
        cache.len(),
        2,
        "Cache should have 2 positions after second token"
    );
}

// ============================================================================
// Y4.3: No Allocations During Decode
// ============================================================================

/// Y4.3a: Cache pre-allocates to context_length
/// FALSIFICATION: capacity() < context_length
#[test]
fn y4_3a_cache_preallocates() {
    use realizar::apr_transformer::{AprKVCache, AprTransformerConfig};

    let config = AprTransformerConfig {
        hidden_dim: 256,
        num_layers: 4,
        num_heads: 8,
        num_kv_heads: 8,
        context_length: 2048,
        ..Default::default()
    };

    let cache = AprKVCache::new(&config);

    // Cache should pre-allocate for full context
    assert!(
        cache.capacity() >= config.context_length,
        "Cache capacity {} should be >= context_length {}",
        cache.capacity(),
        config.context_length
    );
}

/// Y4.3b: Multiple appends don't reallocate
/// FALSIFICATION: Memory address changes during append
#[test]
fn y4_3b_no_reallocation_during_append() {
    use realizar::apr_transformer::{AprKVCache, AprTransformerConfig};

    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        context_length: 32,
        ..Default::default()
    };

    let mut cache = AprKVCache::new(&config);
    let head_dim = config.hidden_dim / config.num_heads;
    let kv_size = config.num_kv_heads * head_dim;

    let k = vec![1.0f32; kv_size];
    let v = vec![2.0f32; kv_size];

    // Get initial capacity
    let initial_capacity = cache.capacity();

    // Append up to capacity
    for _ in 0..initial_capacity.min(16) {
        cache.append(0, &k, &v);
    }

    // Capacity should not have changed (no reallocation)
    assert_eq!(
        cache.capacity(),
        initial_capacity,
        "Capacity should not change during append within pre-allocated space"
    );
}

// ============================================================================
// Y4.4: Attention with Cache
// ============================================================================

/// Y4.4a: attention_with_cache produces valid output
/// FALSIFICATION: Returns NaN/Inf or wrong size
#[test]
fn y4_4a_attention_with_cache_valid() {
    use realizar::apr_transformer::{AprKVCache, AprTransformer, AprTransformerConfig};

    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 32,
        ..Default::default()
    };

    let transformer = AprTransformer::new(config.clone());
    let mut cache = AprKVCache::new(&config);

    // Process a few tokens
    for i in 0..5 {
        let result = transformer.forward_with_cache(i as u32, &mut cache, i);
        assert!(result.is_ok(), "Token {} should succeed", i);

        let logits = result.expect("test");
        // Check no NaN/Inf
        for &logit in &logits {
            assert!(
                logit.is_finite(),
                "Logit should be finite at position {}",
                i
            );
        }
    }
}

/// Y4.4b: Cached forward matches non-cached for single token
/// FALSIFICATION: Different output for same token at position 0
#[test]
fn y4_4b_cached_matches_non_cached_first_token() {
    use realizar::apr_transformer::{AprKVCache, AprTransformer, AprTransformerConfig};

    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 32,
        ..Default::default()
    };

    let transformer = AprTransformer::new(config.clone());

    // Non-cached forward (processes whole sequence)
    let non_cached = transformer.forward(&[1]).expect("Non-cached should work");

    // Cached forward (single token)
    let mut cache = AprKVCache::new(&config);
    let cached = transformer
        .forward_with_cache(1, &mut cache, 0)
        .expect("Cached should work");

    // Results should match
    assert_eq!(
        non_cached.len(),
        cached.len(),
        "Output lengths should match"
    );

    for (i, (&nc, &c)) in non_cached.iter().zip(cached.iter()).enumerate() {
        let diff = (nc - c).abs();
        assert!(
            diff < 1e-5,
            "Logit {} differs: non_cached={}, cached={}, diff={}",
            i,
            nc,
            c,
            diff
        );
    }
}

// ============================================================================
// Y4.5: Generate with Cache
// ============================================================================

/// Y4.5a: generate_with_cache produces valid tokens
/// FALSIFICATION: Returns empty or invalid tokens
#[test]
fn y4_5a_generate_with_cache_works() {
    use realizar::apr_transformer::{AprTransformer, AprTransformerConfig, GenerateConfig};

    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 64,
        ..Default::default()
    };

    let transformer = AprTransformer::new(config.clone());
    let gen_config = GenerateConfig {
        max_tokens: 5,
        temperature: 0.0, // Greedy
        ..Default::default()
    };

    let prompt = vec![1u32, 2, 3];
    let result = transformer.generate_with_cache(&prompt, &gen_config);

    assert!(result.is_ok(), "generate_with_cache should succeed");

    let tokens = result.expect("test");
    // Should have prompt + generated tokens
    assert!(
        tokens.len() >= prompt.len(),
        "Should have at least prompt tokens"
    );
    assert!(
        tokens.len() <= prompt.len() + gen_config.max_tokens,
        "Should not exceed max_tokens"
    );

    // All tokens should be valid (< vocab_size)
    for &token in &tokens {
        assert!(
            (token as usize) < config.vocab_size,
            "Token {} should be < vocab_size {}",
            token,
            config.vocab_size
        );
    }
}

/// Y4.5b: generate_with_cache is deterministic with temperature=0
/// FALSIFICATION: Different outputs for same input
#[test]
fn y4_5b_generate_deterministic() {
    use realizar::apr_transformer::{AprTransformer, AprTransformerConfig, GenerateConfig};

    let config = AprTransformerConfig {
        hidden_dim: 64,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 128,
        context_length: 64,
        ..Default::default()
    };

    let transformer = AprTransformer::new(config);
    let gen_config = GenerateConfig {
        max_tokens: 5,
        temperature: 0.0, // Greedy = deterministic
        ..Default::default()
    };

    let prompt = vec![1u32, 2, 3];

    let run1 = transformer
        .generate_with_cache(&prompt, &gen_config)
        .expect("test");
    let run2 = transformer
        .generate_with_cache(&prompt, &gen_config)
        .expect("test");

    assert_eq!(run1, run2, "Greedy generation should be deterministic");
}

// ============================================================================
// Y4.6: QuantizedAprTransformer Cache Support
// ============================================================================

/// Y4.6a: QuantizedAprTransformer has forward_with_cache
/// FALSIFICATION: Method missing
#[test]
fn y4_6a_quantized_forward_with_cache() {
    use realizar::apr_transformer::{
        AprKVCache, AprQuantizationType, AprTransformerConfig, QuantizedAprTransformer,
    };

    let config = AprTransformerConfig {
        hidden_dim: 256,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 100,
        intermediate_dim: 512,
        context_length: 32,
        ..Default::default()
    };

    let transformer = QuantizedAprTransformer::new(config.clone(), AprQuantizationType::Q4_K);
    let mut cache = AprKVCache::new(&config);

    let result = transformer.forward_with_cache(1, &mut cache, 0);
    assert!(
        result.is_ok(),
        "Quantized forward_with_cache should succeed"
    );
}

// ============================================================================
// Summary: Y4 Popperian Falsification Matrix
// ============================================================================
//
// | Test | Claim | Falsification Condition |
// |------|-------|------------------------|
// | Y4.1a | AprKVCache exists | Struct missing |
// | Y4.1b | Separate K/V storage | Mixed storage |
// | Y4.2a | forward_with_cache exists | Method missing |
// | Y4.2b | Cache grows after forward | len() doesn't increase |
// | Y4.3a | Cache pre-allocates | capacity < context_length |
// | Y4.3b | No realloc during append | Capacity changes |
// | Y4.4a | attention_with_cache valid | NaN/Inf output |
// | Y4.4b | Cached matches non-cached | Different output |
// | Y4.5a | generate_with_cache works | Empty/invalid tokens |
// | Y4.5b | Deterministic generation | Different outputs |
// | Y4.6a | Quantized cache support | Method missing |
