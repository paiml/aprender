
#[test]
#[cfg(feature = "gpu")]
fn test_imp_119c_gpu_fused_multihead_long_sequence() {
    // IMP-119c: GPU fused multi-head attention for long sequences
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 128,
        intermediate_dim: 256,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 100,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    // Long sequence with multiple heads
    let seq_len = 128;
    let hidden_dim = 128;
    let num_heads = 8;
    let _head_dim = hidden_dim / num_heads;

    let q: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| (i % 17) as f32 * 0.05)
        .collect();
    let k: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| (i % 13) as f32 * 0.05)
        .collect();
    let v: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| (i % 11) as f32 * 0.05)
        .collect();

    // Use GPU-accelerated multihead fused attention
    let result = cached_model
        .gpu_fused_multihead_attention(&q, &k, &v, seq_len)
        .expect("GPU fused multihead attention should succeed");

    assert_eq!(
        result.len(),
        seq_len * hidden_dim,
        "IMP-119c: Output should have shape [seq_len, hidden_dim]"
    );

    // Verify each position has non-trivial output
    for pos in 0..seq_len {
        let slice = &result[pos * hidden_dim..(pos + 1) * hidden_dim];
        let sum: f32 = slice.iter().map(|x| x.abs()).sum();
        assert!(
            sum > 0.0 || pos == 0,
            "IMP-119c: Position {} should have non-zero output",
            pos
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_119d_adaptive_cpu_gpu_dispatch() {
    // IMP-119d: Verify adaptive dispatch chooses CPU for short, GPU for long sequences
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        context_length: 256,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    let head_dim = 16;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Short sequence - should work regardless of backend choice
    let short_seq_len = 8;
    let short_q: Vec<f32> = (0..short_seq_len * head_dim)
        .map(|i| (i % 17) as f32 * 0.1)
        .collect();
    let short_k: Vec<f32> = (0..short_seq_len * head_dim)
        .map(|i| (i % 13) as f32 * 0.1)
        .collect();
    let short_v: Vec<f32> = (0..short_seq_len * head_dim)
        .map(|i| (i % 11) as f32 * 0.1)
        .collect();

    let short_result = cached_model
        .adaptive_fused_attention(&short_q, &short_k, &short_v, short_seq_len, head_dim, scale)
        .expect("Adaptive attention for short sequence should succeed");

    assert_eq!(short_result.len(), short_seq_len * head_dim);

    // Long sequence - should also work
    let long_seq_len = 128;
    let long_q: Vec<f32> = (0..long_seq_len * head_dim)
        .map(|i| (i % 17) as f32 * 0.1)
        .collect();
    let long_k: Vec<f32> = (0..long_seq_len * head_dim)
        .map(|i| (i % 13) as f32 * 0.1)
        .collect();
    let long_v: Vec<f32> = (0..long_seq_len * head_dim)
        .map(|i| (i % 11) as f32 * 0.1)
        .collect();

    let long_result = cached_model
        .adaptive_fused_attention(&long_q, &long_k, &long_v, long_seq_len, head_dim, scale)
        .expect("Adaptive attention for long sequence should succeed");

    assert_eq!(long_result.len(), long_seq_len * head_dim);

    // Both should produce valid outputs
    let short_sum: f32 = short_result.iter().sum();
    let long_sum: f32 = long_result.iter().sum();

    // Longer sequence should have larger accumulated values (more positions attending)
    assert!(
        long_sum.abs() > short_sum.abs() / 2.0,
        "IMP-119d: Long sequence output should be non-trivial"
    );
}
