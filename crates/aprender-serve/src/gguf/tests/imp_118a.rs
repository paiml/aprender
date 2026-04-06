
// ========================================================================
// IMP-118: True GPU Batched GEMM Kernel Tests
// ========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_118a_true_batched_gemm_correctness() {
    // IMP-118a: Verify true batched GEMM produces correct results
    // Strategy: Process all batches in single kernel invocation
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    let batch_size = 8;
    let m = 16;
    let k = 32;
    let n = 16;

    // Create batched input data
    let mut batched_a = vec![0.0f32; batch_size * m * k];
    let mut batched_b = vec![0.0f32; batch_size * k * n];

    for b in 0..batch_size {
        for i in 0..m * k {
            batched_a[b * m * k + i] = ((b * m * k + i) % 17) as f32 * 0.1;
        }
        for i in 0..k * n {
            batched_b[b * k * n + i] = ((b * k * n + i) % 13) as f32 * 0.1;
        }
    }

    // True batched GEMM should process all batches together
    let result = cached_model
        .true_batched_gemm(&batched_a, &batched_b, batch_size, m, k, n)
        .expect("True batched GEMM should succeed");

    assert_eq!(
        result.len(),
        batch_size * m * n,
        "IMP-118a: Output should have shape [batch, m, n]"
    );

    // Verify by computing reference per-batch
    for b in 0..batch_size {
        let a_start = b * m * k;
        let b_start = b * k * n;
        let out_start = b * m * n;

        for i in 0..m {
            for j in 0..n {
                let mut expected = 0.0f32;
                for kk in 0..k {
                    expected += batched_a[a_start + i * k + kk] * batched_b[b_start + kk * n + j];
                }
                let actual = result[out_start + i * n + j];
                let diff = (expected - actual).abs();
                assert!(
                    diff < 1e-2,
                    "IMP-118a: Batch {} pos ({},{}) mismatch: expected={}, got={}, diff={}",
                    b,
                    i,
                    j,
                    expected,
                    actual,
                    diff
                );
            }
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_118b_true_batched_gemm_matches_flattened() {
    // IMP-118b: True batched GEMM should match flattened implementation
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    let batch_size = 4;
    let m = 8;
    let k = 16;
    let n = 8;

    let mut batched_a = vec![0.0f32; batch_size * m * k];
    let mut batched_b = vec![0.0f32; batch_size * k * n];

    for i in 0..batched_a.len() {
        batched_a[i] = (i % 19) as f32 * 0.05;
    }
    for i in 0..batched_b.len() {
        batched_b[i] = (i % 23) as f32 * 0.05;
    }

    // Compare true batched vs flattened
    let true_result = cached_model
        .true_batched_gemm(&batched_a, &batched_b, batch_size, m, k, n)
        .expect("True batched GEMM should succeed");

    let flat_result = cached_model
        .flattened_batched_gemm(&batched_a, &batched_b, batch_size, m, k, n)
        .expect("Flattened GEMM should succeed");

    assert_eq!(true_result.len(), flat_result.len());
    for i in 0..true_result.len() {
        let diff = (true_result[i] - flat_result[i]).abs();
        assert!(
            diff < 1e-3,
            "IMP-118b: Results differ at {}: true={}, flat={}, diff={}",
            i,
            true_result[i],
            flat_result[i],
            diff
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_118c_true_batched_gemm_large_batch() {
    // IMP-118c: True batched GEMM should handle large batch sizes efficiently
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 128,
        intermediate_dim: 256,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 8,
        vocab_size: 100,
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

    // Large batch that benefits from true GPU batching
    let batch_size = 32;
    let m = 16;
    let k = 64;
    let n = 16;

    let mut batched_a = vec![0.0f32; batch_size * m * k];
    let mut batched_b = vec![0.0f32; batch_size * k * n];

    for i in 0..batched_a.len() {
        batched_a[i] = (i % 31) as f32 * 0.02;
    }
    for i in 0..batched_b.len() {
        batched_b[i] = (i % 29) as f32 * 0.02;
    }

    let result = cached_model
        .true_batched_gemm(&batched_a, &batched_b, batch_size, m, k, n)
        .expect("Large batch true GEMM should succeed");

    assert_eq!(
        result.len(),
        batch_size * m * n,
        "IMP-118c: Large batch output should have correct dimensions"
    );

    // Verify non-trivial output
    let sum: f32 = result.iter().map(|x| x.abs()).sum();
    assert!(
        sum > 0.01,
        "IMP-118c: Output should have non-trivial values, got sum={}",
        sum
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_118d_true_batched_attention() {
    // IMP-118d: Use true batched GEMM for multi-head attention
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        context_length: 64,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    let num_heads = 4;
    let seq_len = 8;
    let head_dim = 16;

    // Create Q, K, V tensors
    let q: Vec<f32> = (0..num_heads * seq_len * head_dim)
        .map(|i| (i % 17) as f32 * 0.1)
        .collect();
    let k: Vec<f32> = (0..num_heads * seq_len * head_dim)
        .map(|i| (i % 13) as f32 * 0.1)
        .collect();
    let v: Vec<f32> = (0..num_heads * seq_len * head_dim)
        .map(|i| (i % 11) as f32 * 0.1)
        .collect();

    // Use true batched GEMM for attention
    let result = cached_model
        .true_batched_multihead_attention(&q, &k, &v, seq_len, num_heads, head_dim)
        .expect("True batched attention should succeed");

    assert_eq!(
        result.len(),
        num_heads * seq_len * head_dim,
        "IMP-118d: Attention output should have correct shape"
    );

    // Verify normalized attention (each position should have weighted values)
    for h in 0..num_heads {
        for pos in 0..seq_len {
            let out_start = h * seq_len * head_dim + pos * head_dim;
            let slice = &result[out_start..out_start + head_dim];
            let sum: f32 = slice.iter().map(|x| x.abs()).sum();
            assert!(
                sum > 0.0 || pos == 0,
                "IMP-118d: Head {} pos {} should have non-zero output",
                h,
                pos
            );
        }
    }
}

// ========================================================================
// IMP-119: GPU-Accelerated Fused Attention for Long Sequences
// ========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_119a_gpu_fused_attention_correctness() {
    // IMP-119a: Verify GPU fused attention produces correct results
    // Uses GPU for long sequences where compute dominates transfer overhead
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

    // Long sequence that benefits from GPU
    let seq_len = 64;
    let head_dim = 16;

    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 17) as f32 * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 13) as f32 * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 11) as f32 * 0.1)
        .collect();

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Use GPU-accelerated fused attention
    let result = cached_model
        .gpu_fused_causal_attention(&q, &k, &v, seq_len, head_dim, scale)
        .expect("GPU fused attention should succeed");

    assert_eq!(
        result.len(),
        seq_len * head_dim,
        "IMP-119a: Output should have shape [seq_len, head_dim]"
    );

    // Verify causality: later positions should have different values than if
    // they could attend to all positions
    // Position 0 can only attend to itself
    let pos0_sum: f32 = result[0..head_dim].iter().sum();
    // Position seq_len-1 can attend to all previous positions
    let last_pos_sum: f32 = result[(seq_len - 1) * head_dim..].iter().sum();

    // These sums should be different due to causal masking
    assert!(
        (pos0_sum - last_pos_sum).abs() > 0.001 || seq_len == 1,
        "IMP-119a: Causal masking should affect output"
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_119b_gpu_fused_matches_cpu_fused() {
    // IMP-119b: GPU fused attention should match CPU fused attention
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size: 50,
        context_length: 128,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0,
            explicit_head_dim: None,
        bos_token_id: None,
            eos_token_id: None,
    };

    let model = create_test_model_with_config(&config);
    let cached_model = OwnedQuantizedModelCached::new(model);

    let seq_len = 32;
    let head_dim = 16;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 19) as f32 * 0.05)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 23) as f32 * 0.05)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i % 29) as f32 * 0.05)
        .collect();

    // CPU fused attention (IMP-115)
    let cpu_result = cached_model
        .fused_causal_attention(&q, &k, &v, seq_len, head_dim, scale)
        .expect("CPU fused attention should succeed");

    // GPU fused attention (IMP-119)
    let gpu_result = cached_model
        .gpu_fused_causal_attention(&q, &k, &v, seq_len, head_dim, scale)
        .expect("GPU fused attention should succeed");

    assert_eq!(cpu_result.len(), gpu_result.len());
    for i in 0..cpu_result.len() {
        let diff = (cpu_result[i] - gpu_result[i]).abs();
        assert!(
            diff < 1e-2,
            "IMP-119b: Results differ at {}: cpu={}, gpu={}, diff={}",
            i,
            cpu_result[i],
            gpu_result[i],
            diff
        );
    }
}
