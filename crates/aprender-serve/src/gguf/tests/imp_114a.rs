
// ========================================================================
// IMP-114: True GPU Batched GEMM Kernel Tests
// ========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_114a_flattened_batched_gemm_correctness() {
    // IMP-114a: Verify flattened batched GEMM computes correct results
    // Strategy: Flatten [batch, m, k] @ [batch, k, n] into single large matmul
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

    // Create batched matrices
    let batched_a: Vec<f32> = (0..batch_size * m * k)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let batched_b: Vec<f32> = (0..batch_size * k * n)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();

    // Use flattened batched GEMM (true single dispatch)
    let result = cached_model
        .flattened_batched_gemm(&batched_a, &batched_b, batch_size, m, k, n)
        .expect("Flattened batched GEMM should succeed");

    // Output should be [batch_size, m, n]
    assert_eq!(
        result.len(),
        batch_size * m * n,
        "IMP-114a: Output should have shape [batch, m, n]"
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
                    diff < 1e-3,
                    "IMP-114a: Batch {} mismatch at ({},{}): expected={}, actual={}, diff={}",
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
fn test_imp_114b_flattened_matches_loop() {
    // IMP-114b: Verify flattened approach matches loop-based approach
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 8,
        num_kv_heads: 8,
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
    let k = 8;
    let n = 16;

    let batched_a: Vec<f32> = (0..batch_size * m * k)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
        .collect();
    let batched_b: Vec<f32> = (0..batch_size * k * n)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.05)
        .collect();

    // Loop-based (IMP-113)
    let loop_result = cached_model
        .batched_gemm_single_dispatch(&batched_a, &batched_b, batch_size, m, k, n)
        .expect("Loop GEMM should succeed");

    // Flattened (IMP-114)
    let flat_result = cached_model
        .flattened_batched_gemm(&batched_a, &batched_b, batch_size, m, k, n)
        .expect("Flattened GEMM should succeed");

    assert_eq!(loop_result.len(), flat_result.len());
    for i in 0..loop_result.len() {
        let diff = (loop_result[i] - flat_result[i]).abs();
        assert!(
            diff < 1e-3,
            "IMP-114b: Results differ at {}: loop={}, flat={}, diff={}",
            i,
            loop_result[i],
            flat_result[i],
            diff
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_114c_flattened_attention_correctness() {
    // IMP-114c: Verify flattened attention matches reference
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

    let seq_len = 8;
    let hidden_dim = config.hidden_dim;

    let q: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();

    // Reference (IMP-113 single dispatch)
    let reference = cached_model
        .single_dispatch_multihead_attention(&q, &k, &v, seq_len)
        .expect("Reference attention should succeed");

    // Flattened (IMP-114)
    let result = cached_model
        .flattened_multihead_attention(&q, &k, &v, seq_len)
        .expect("Flattened attention should succeed");

    assert_eq!(result.len(), reference.len());
    for i in 0..result.len() {
        let diff = (result[i] - reference[i]).abs();
        assert!(
            diff < 1e-3,
            "IMP-114c: Attention differs at {}: ref={}, flat={}, diff={}",
            i,
            reference[i],
            result[i],
            diff
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_114d_large_batch_flattened() {
    // IMP-114d: Test with larger batch sizes where flattening benefits
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 128,
        intermediate_dim: 256,
        num_layers: 1,
        num_heads: 16, // Larger number of heads
        num_kv_heads: 16,
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

    let batch_size = 16;
    let m = 8;
    let k = 8;
    let n = 8;

    let batched_a: Vec<f32> = (0..batch_size * m * k)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.04)
        .collect();
    let batched_b: Vec<f32> = (0..batch_size * k * n)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.04)
        .collect();

    let result = cached_model
        .flattened_batched_gemm(&batched_a, &batched_b, batch_size, m, k, n)
        .expect("Large batch flattened GEMM should succeed");

    assert_eq!(
        result.len(),
        batch_size * m * n,
        "IMP-114d: Output should have correct dimensions"
    );

    // Verify non-trivial output
    let sum: f32 = result.iter().map(|x| x.abs()).sum();
    assert!(
        sum > 0.01,
        "IMP-114d: Output should have non-trivial values, got sum={}",
        sum
    );
}
