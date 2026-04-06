
/// IMP-101c: Attention with cache produces normalized output
#[test]
fn test_imp_101c_attention_with_cache_softmax_normalized() {
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 4,
        intermediate_dim: 16,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 100,
        context_length: 2048,
        eps: 1e-5,
        rope_type: 0,
        rope_theta: 10000.0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = OwnedQuantizedModel {
        config,
        token_embedding: vec![],
        position_embedding: None,
        layers: vec![],
        encoder_layers: vec![],
        encoder_output_norm_weight: None,
        encoder_output_norm_bias: None,
        output_norm_weight: vec![],
        output_norm_bias: None,
        lm_head_weight: OwnedQuantizedTensor {
            data: vec![],
            in_dim: 4,
            out_dim: 100,
            qtype: 0,
        },
        lm_head_bias: None,
        #[cfg(feature = "cuda")]
        cuda_executor: None,
        #[cfg(feature = "cuda")]
        cuda_kernel_count: std::sync::atomic::AtomicU64::new(0),
        #[cfg(feature = "cuda")]
        cached_weight_names: std::sync::Mutex::new(std::collections::HashSet::new()),
    };

    // Test attention with cache
    // Q = [1, 0, 0, 0], cached K/V for one position, current K/V
    let q = vec![1.0, 0.0, 0.0, 0.0];
    let k_cache = vec![1.0, 0.0, 0.0, 0.0]; // cached position 0
    let v_cache = vec![1.0, 0.0, 0.0, 0.0];
    let current_k = vec![1.0, 0.0, 0.0, 0.0]; // current position 1
    let current_v = vec![0.0, 1.0, 0.0, 0.0];

    let output = model.attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v);

    // Output should be weighted combination of v_cache and current_v
    // Both K vectors are identical to Q, so scores are equal -> 50/50 weights
    // Output should be approximately [0.5, 0.5, 0, 0]
    assert_eq!(
        output.len(),
        4,
        "IMP-101c: Output should have hidden_dim elements"
    );

    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.1,
        "IMP-101c: Attention output should be normalized weighted sum"
    );
}

/// IMP-101c: Cache handles multiple positions correctly
#[test]
fn test_imp_101c_kv_cache_multiple_positions() {
    let mut cache = OwnedQuantizedKVCache::new(1, 4, 100);

    // Add 3 positions
    for i in 0..3 {
        let k = vec![i as f32; 4];
        let v = vec![(i as f32) * 0.1; 4];
        cache.append(0, &k, &v);
        cache.advance();
    }

    assert_eq!(cache.len(), 3, "IMP-101c: Cache should have 3 positions");

    let k_data = cache.get_k(0);
    assert_eq!(
        k_data.len(),
        12,
        "IMP-101c: K cache should have 3 * 4 = 12 elements"
    );

    // Verify first position K values
    assert!(
        (k_data[0] - 0.0).abs() < 1e-6,
        "IMP-101c: First K should be 0"
    );
    // Verify second position K values
    assert!(
        (k_data[4] - 1.0).abs() < 1e-6,
        "IMP-101c: Second K should be 1"
    );
    // Verify third position K values
    assert!(
        (k_data[8] - 2.0).abs() < 1e-6,
        "IMP-101c: Third K should be 2"
    );
}

#[test]
fn test_imp_105_gqa_attention_multiple_q_per_kv() {
    // IMP-105: GQA (Grouped Query Attention) support
    // 8 Q heads share 2 KV heads (4 Q heads per KV head)
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 32, // 8 heads * 4 head_dim
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 8,    // Q heads
        num_kv_heads: 2, // KV heads (4:1 ratio)
        vocab_size: 100,
        context_length: 1024,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    // Create model with dummy weights
    let hidden_dim = config.hidden_dim;
    let head_dim = hidden_dim / config.num_heads; // 4
    let kv_dim = config.num_kv_heads * head_dim; // 2 * 4 = 8

    // Q: [hidden_dim] = [32] - 8 heads
    // K/V: [kv_dim] = [8] - 2 heads
    let q = vec![1.0f32; hidden_dim];
    let current_k = vec![1.0f32; kv_dim];
    let current_v = vec![1.0f32; kv_dim];

    // Empty cache for first position
    let k_cache: Vec<f32> = vec![];
    let v_cache: Vec<f32> = vec![];

    // Test that GQA attention computes correctly
    // Q heads 0-3 should use KV head 0
    // Q heads 4-7 should use KV head 1
    let model = create_test_model_with_config(&config);
    let output = model.attention_with_cache_gqa(&q, &k_cache, &v_cache, &current_k, &current_v);

    // Output should have hidden_dim elements
    assert_eq!(
        output.len(),
        hidden_dim,
        "IMP-105: GQA output should have hidden_dim={hidden_dim} elements"
    );

    // Each head's output should be non-zero (softmax weight = 1.0 for single position)
    for head in 0..config.num_heads {
        let head_start = head * head_dim;
        let head_sum: f32 = output[head_start..head_start + head_dim].iter().sum();
        assert!(
            head_sum.abs() > 1e-6,
            "IMP-105: GQA head {head} output should be non-zero"
        );
    }
}

#[test]
fn test_imp_105_gqa_kv_head_sharing() {
    // IMP-105: Verify that multiple Q heads correctly share KV heads
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 16, // 4 heads * 4 head_dim
        intermediate_dim: 64,
        num_layers: 1,
        num_heads: 4,    // Q heads
        num_kv_heads: 2, // KV heads (2:1 ratio)
        vocab_size: 100,
        context_length: 1024,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let hidden_dim = config.hidden_dim;
    let head_dim = hidden_dim / config.num_heads; // 4
    let kv_dim = config.num_kv_heads * head_dim; // 8

    // Create Q with different values per head
    let mut q = vec![0.0f32; hidden_dim];
    for head in 0..config.num_heads {
        for d in 0..head_dim {
            q[head * head_dim + d] = (head + 1) as f32;
        }
    }

    // Create K with different values per KV head
    let mut current_k = vec![0.0f32; kv_dim];
    for kv_head in 0..config.num_kv_heads {
        for d in 0..head_dim {
            current_k[kv_head * head_dim + d] = (kv_head + 1) as f32 * 0.5;
        }
    }

    // V values
    let mut current_v = vec![0.0f32; kv_dim];
    for kv_head in 0..config.num_kv_heads {
        for d in 0..head_dim {
            current_v[kv_head * head_dim + d] = (kv_head + 1) as f32;
        }
    }

    let k_cache: Vec<f32> = vec![];
    let v_cache: Vec<f32> = vec![];

    let model = create_test_model_with_config(&config);
    let output = model.attention_with_cache_gqa(&q, &k_cache, &v_cache, &current_k, &current_v);

    // Q heads 0,1 should use KV head 0 (value=1.0)
    // Q heads 2,3 should use KV head 1 (value=2.0)
    // With softmax weight = 1.0 (single position), output = V
    let eps = 1e-5;

    // Head 0 and 1 should have similar outputs (both use KV head 0)
    let head0_sum: f32 = output[0..head_dim].iter().sum();
    let head1_sum: f32 = output[head_dim..2 * head_dim].iter().sum();

    // Head 2 and 3 should have similar outputs (both use KV head 1)
    let head2_sum: f32 = output[2 * head_dim..3 * head_dim].iter().sum();
    let head3_sum: f32 = output[3 * head_dim..4 * head_dim].iter().sum();

    // Verify KV head sharing pattern
    assert!(
        (head0_sum - head1_sum).abs() < eps,
        "IMP-105: Heads 0,1 should produce same output (share KV head 0)"
    );
    assert!(
        (head2_sum - head3_sum).abs() < eps,
        "IMP-105: Heads 2,3 should produce same output (share KV head 1)"
    );
    assert!(
        (head0_sum - head2_sum).abs() > eps,
        "IMP-105: Heads using different KV heads should have different outputs"
    );
}
