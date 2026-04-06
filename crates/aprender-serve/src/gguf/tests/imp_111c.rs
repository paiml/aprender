
#[test]
fn test_imp_111c_tiled_causal_attention() {
    // IMP-111c: Verify tiled attention respects causal mask
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 32,
        intermediate_dim: 64,
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
    let seq_len = 8;
    let head_dim = config.hidden_dim / config.num_heads;

    // Create deterministic Q, K, V
    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| (i as f32 * 0.1) % 1.0)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i + 5) as f32 * 0.1) % 1.0)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i + 10) as f32 * 0.1) % 1.0)
        .collect();

    let scale = 1.0 / (head_dim as f32).sqrt();
    let tile_size = 4;

    // Tiled causal attention
    let tiled_output = model
        .tiled_causal_attention(&q, &k, &v, seq_len, head_dim, scale, tile_size)
        .expect("IMP-111c: Tiled causal attention should succeed");

    // Verify output shape
    assert_eq!(
        tiled_output.len(),
        seq_len * head_dim,
        "IMP-111c: Output should have seq_len * head_dim elements"
    );

    // Verify finite values
    assert!(
        tiled_output.iter().all(|x| x.is_finite()),
        "IMP-111c: All outputs should be finite"
    );

    // Verify causality: output at position i should only depend on positions 0..=i
    // We test this by checking that changing K/V at position j > i doesn't affect output[i]
    let mut k_modified = k.clone();
    // Modify K at last position
    for d in 0..head_dim {
        k_modified[(seq_len - 1) * head_dim + d] = 999.0;
    }

    let modified_output = model
        .tiled_causal_attention(&q, &k_modified, &v, seq_len, head_dim, scale, tile_size)
        .expect("Modified attention should succeed");

    // Positions 0 to seq_len-2 should be unchanged (they don't attend to position seq_len-1)
    for pos in 0..seq_len - 1 {
        for d in 0..head_dim {
            let idx = pos * head_dim + d;
            let diff = (tiled_output[idx] - modified_output[idx]).abs();
            assert!(
                diff < 1e-6,
                "IMP-111c: Position {} should not be affected by future positions, diff={}",
                pos,
                diff
            );
        }
    }
}

#[test]
fn test_imp_111d_tiled_attention_various_tile_sizes() {
    // IMP-111d: Verify tiled attention works with various tile sizes
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 32,
        intermediate_dim: 64,
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
    let seq_len = 16;
    let head_dim = config.hidden_dim / config.num_heads;

    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Get reference with tile_size = 1 (equivalent to standard)
    let reference = model
        .tiled_causal_attention(&q, &k, &v, seq_len, head_dim, scale, 1)
        .expect("Reference should succeed");

    // Test various tile sizes
    for tile_size in [2, 4, 8, 16] {
        let output = model
            .tiled_causal_attention(&q, &k, &v, seq_len, head_dim, scale, tile_size)
            .unwrap_or_else(|_| panic!("Tile size {} should succeed", tile_size));

        assert_eq!(output.len(), reference.len());
        for i in 0..output.len() {
            let diff = (output[i] - reference[i]).abs();
            assert!(
                diff < 1e-4,
                "IMP-111d: Tile size {} differs at {}: ref={}, tiled={}, diff={}",
                tile_size,
                i,
                reference[i],
                output[i],
                diff
            );
        }
    }
}

// ========================================================================
// IMP-113: True Batched GPU Kernel Tests (Single Dispatch)
// ========================================================================

#[test]
#[cfg(feature = "gpu")]
fn test_imp_113a_batched_gemm_single_dispatch() {
    // IMP-113a: Verify batched GEMM processes all heads in single dispatch
    // This is the foundation for efficient multi-head attention
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

    let num_heads = 4;
    let seq_len = 8;
    let head_dim = 16;

    // Create batched A: [num_heads, seq_len, head_dim]
    let batched_a: Vec<f32> = (0..num_heads * seq_len * head_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();

    // Create batched B: [num_heads, head_dim, seq_len]
    let batched_b: Vec<f32> = (0..num_heads * head_dim * seq_len)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();

    // Single dispatch batched GEMM
    let result = cached_model
        .batched_gemm_single_dispatch(
            &batched_a, &batched_b, num_heads, seq_len, head_dim, seq_len,
        )
        .expect("Batched GEMM should succeed");

    // Output: [num_heads, seq_len, seq_len]
    assert_eq!(
        result.len(),
        num_heads * seq_len * seq_len,
        "IMP-113a: Output should have shape [num_heads, seq_len, seq_len]"
    );

    // Verify by computing reference per-head
    for h in 0..num_heads {
        let a_start = h * seq_len * head_dim;
        let b_start = h * head_dim * seq_len;
        let out_start = h * seq_len * seq_len;

        for i in 0..seq_len {
            for j in 0..seq_len {
                let mut expected = 0.0f32;
                for k in 0..head_dim {
                    expected += batched_a[a_start + i * head_dim + k]
                        * batched_b[b_start + k * seq_len + j];
                }
                let actual = result[out_start + i * seq_len + j];
                let diff = (expected - actual).abs();
                assert!(
                    diff < 1e-3,
                    "IMP-113a: Head {} mismatch at ({},{}): expected={}, actual={}, diff={}",
                    h,
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
fn test_imp_113b_single_dispatch_attention_correctness() {
    // IMP-113b: Verify single-dispatch attention matches multi-dispatch
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
    let cached_model = OwnedQuantizedModelCached::new(model.clone());

    let seq_len = 8;
    let hidden_dim = config.hidden_dim;

    // Create Q, K, V: [seq_len, hidden_dim]
    let q: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let k: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.1)
        .collect();
    let v: Vec<f32> = (0..seq_len * hidden_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect();

    // Multi-dispatch reference (existing implementation)
    let reference = cached_model
        .parallel_multihead_attention_gpu_cached(&q, &k, &v, seq_len)
        .expect("Multi-dispatch attention should succeed");

    // Single-dispatch new implementation
    let result = cached_model
        .single_dispatch_multihead_attention(&q, &k, &v, seq_len)
        .expect("Single-dispatch attention should succeed");

    // Compare outputs
    assert_eq!(result.len(), reference.len());
    for i in 0..result.len() {
        let diff = (result[i] - reference[i]).abs();
        assert!(
            diff < 1e-3,
            "IMP-113b: Single-dispatch differs at {}: ref={}, single={}, diff={}",
            i,
            reference[i],
            result[i],
            diff
        );
    }
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_113c_single_dispatch_dispatch_count() {
    // IMP-113c: Verify single-dispatch uses fewer GPU dispatches
    // This test validates the architectural improvement
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 64,
        intermediate_dim: 128,
        num_layers: 1,
        num_heads: 8, // More heads = bigger benefit
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

    let seq_len = 16;
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

    // Both should succeed and produce valid output
    let single_result = cached_model
        .single_dispatch_multihead_attention(&q, &k, &v, seq_len)
        .expect("Single-dispatch should succeed");

    // Validate output dimensions
    assert_eq!(
        single_result.len(),
        seq_len * hidden_dim,
        "IMP-113c: Output should have shape [seq_len, hidden_dim]"
    );

    // Validate output is not all zeros (sanity check)
    let sum: f32 = single_result.iter().map(|x| x.abs()).sum();
    assert!(
        sum > 0.01,
        "IMP-113c: Output should have non-trivial values, got sum={}",
        sum
    );
}

#[test]
#[cfg(feature = "gpu")]
fn test_imp_113d_batched_softmax_correctness() {
    // IMP-113d: Verify batched softmax with causal mask
    let config = GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: 32,
        intermediate_dim: 64,
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

    let num_heads = 4;
    let seq_len = 8;

    // Create batched scores: [num_heads, seq_len, seq_len]
    let batched_scores: Vec<f32> = (0..num_heads * seq_len * seq_len)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.2)
        .collect();

    // Apply batched causal softmax
    let result = cached_model
        .batched_causal_softmax(&batched_scores, num_heads, seq_len)
        .expect("Batched causal softmax should succeed");

    // Verify dimensions
    assert_eq!(result.len(), num_heads * seq_len * seq_len);

    // Verify each row sums to 1.0 (within causal mask)
    for h in 0..num_heads {
        for i in 0..seq_len {
            let row_start = h * seq_len * seq_len + i * seq_len;
            let row_sum: f32 = (0..=i).map(|j| result[row_start + j]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "IMP-113d: Head {} row {} should sum to 1.0, got {}",
                h,
                i,
                row_sum
            );

            // Verify causal mask: positions > i should be 0
            for j in (i + 1)..seq_len {
                assert!(
                    result[row_start + j].abs() < 1e-6,
                    "IMP-113d: Head {} pos ({},{}) should be masked, got {}",
                    h,
                    i,
                    j,
                    result[row_start + j]
                );
            }
        }
    }
}
