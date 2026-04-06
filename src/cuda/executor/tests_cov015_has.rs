
// ==============================================================================
// COV-015: layer.rs error paths and validation coverage
// Target: Increase layer.rs coverage from 17.61% by testing error branches
// ==============================================================================

#[test]
#[serial]
fn test_cov015_has_rmsnorm_weights_false_initially() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Should return false for any layer when no weights cached
    assert!(
        !executor.has_rmsnorm_weights(0),
        "Layer 0 should have no RMSNorm weights"
    );
    assert!(
        !executor.has_rmsnorm_weights(5),
        "Layer 5 should have no RMSNorm weights"
    );
    assert!(
        !executor.has_rmsnorm_weights(100),
        "Layer 100 should have no RMSNorm weights"
    );
}

#[test]
#[serial]
fn test_cov015_has_rmsnorm_weights_after_preload() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let gamma = vec![1.0f32; 64];
    let attn_norms: Vec<&[f32]> = vec![gamma.as_slice()];
    let ffn_norms: Vec<&[f32]> = vec![gamma.as_slice()];

    executor
        .preload_rmsnorm_weights(1, &attn_norms, &ffn_norms)
        .expect("preload");

    assert!(
        executor.has_rmsnorm_weights(0),
        "Layer 0 should have RMSNorm weights after preload"
    );
    assert!(
        !executor.has_rmsnorm_weights(1),
        "Layer 1 should not have weights"
    );
}

#[test]
#[serial]
fn test_cov015_forward_all_layers_missing_attn_norm() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 64u32;
    let input = vec![0.1f32; hidden_dim as usize];
    let mut output = vec![0.0f32; hidden_dim as usize];

    // Try forward without any cached weights - should fail
    let result = executor.forward_all_layers_gpu(
        &input,
        &mut output,
        0, // position
        1, // num_layers
        hidden_dim,
        128,  // intermediate_dim
        1e-5, // epsilon
    );

    assert!(result.is_err(), "Should fail without cached attn_norm");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("attn_norm not cached"),
        "Error should mention missing attn_norm: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov015_forward_all_layers_missing_ffn_norm() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 64u32;
    let gamma = vec![1.0f32; hidden_dim as usize];

    // Only cache attn_norm, not ffn_norm
    executor
        .cache_rmsnorm_gamma("blk.0.attn_norm.gamma", &gamma)
        .expect("cache attn");

    let input = vec![0.1f32; hidden_dim as usize];
    let mut output = vec![0.0f32; hidden_dim as usize];

    let result = executor.forward_all_layers_gpu(&input, &mut output, 0, 1, hidden_dim, 128, 1e-5);

    assert!(result.is_err(), "Should fail without cached ffn_norm");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("ffn_norm not cached"),
        "Error should mention missing ffn_norm: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov015_forward_to_logits_missing_output_norm() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 64u32;
    let vocab_size = 128u32;
    let gamma = vec![1.0f32; hidden_dim as usize];

    // Cache layer norms but not output norm
    executor
        .cache_rmsnorm_gamma("blk.0.attn_norm.gamma", &gamma)
        .expect("cache attn");
    executor
        .cache_rmsnorm_gamma("blk.0.ffn_norm.gamma", &gamma)
        .expect("cache ffn");

    let input = vec![0.1f32; hidden_dim as usize];
    let mut logits = vec![0.0f32; vocab_size as usize];

    let result = executor.forward_all_layers_gpu_to_logits(
        &input,
        &mut logits,
        0,
        1,
        hidden_dim,
        128,
        vocab_size,
        1e-5,
    );

    assert!(result.is_err(), "Should fail without cached output_norm");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("output_norm not cached"),
        "Error should mention missing output_norm: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov015_forward_batched_batch_size_zero() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 64u32;
    let inputs: Vec<f32> = vec![]; // Empty - batch size 0
    let positions: Vec<u32> = vec![]; // Empty positions

    let result =
        executor.forward_batched_to_token_ids(&inputs, &positions, 1, hidden_dim, 128, 256, 1e-5);

    assert!(result.is_err(), "Should fail with batch size 0");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("batch size must be 1-32"),
        "Error should mention batch size constraint: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov015_forward_batched_batch_size_exceeds_max() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 64u32;
    let m = 33; // Exceeds max of 32
    let inputs = vec![0.1f32; m * hidden_dim as usize];
    let positions: Vec<u32> = (0..m as u32).collect();

    let result =
        executor.forward_batched_to_token_ids(&inputs, &positions, 1, hidden_dim, 128, 256, 1e-5);

    assert!(result.is_err(), "Should fail with batch size > 32");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("batch size must be 1-32"),
        "Error should mention batch size constraint: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov015_forward_batched_wrong_input_length() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 64u32;
    let positions: Vec<u32> = vec![0, 1]; // M=2
    let inputs = vec![0.1f32; 50]; // Wrong length: should be 2 * 64 = 128

    let result =
        executor.forward_batched_to_token_ids(&inputs, &positions, 1, hidden_dim, 128, 256, 1e-5);

    assert!(result.is_err(), "Should fail with wrong input length");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("inputs.len()") || err_msg.contains("M*hidden_dim"),
        "Error should mention input length mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov015_forward_batched_workspace_not_initialized() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 64u32;
    let m = 2;
    let inputs = vec![0.1f32; m * hidden_dim as usize];
    let positions: Vec<u32> = vec![0, 1];

    // Don't initialize workspace - should fail
    let result =
        executor.forward_batched_to_token_ids(&inputs, &positions, 1, hidden_dim, 128, 256, 1e-5);

    assert!(result.is_err(), "Should fail without initialized workspace");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("workspace not initialized") || err_msg.contains("Batched workspace"),
        "Error should mention workspace: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov015_preload_lm_head_bias_empty() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Preload with empty bias (should return 0 bytes, not error)
    let empty_bias: Vec<f32> = vec![];
    let result = executor.preload_lm_head_bias(Some(&empty_bias));
    assert!(
        result.is_ok(),
        "preload_lm_head_bias with empty should succeed"
    );
    assert_eq!(result.unwrap(), 0, "Empty bias should upload 0 bytes");
    assert!(
        !executor.has_lm_head_bias(),
        "Should not have LM head bias with empty input"
    );
}

#[test]
#[serial]
fn test_cov015_cache_rmsnorm_gamma_duplicate() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let gamma = vec![1.0f32; 64];

    // First cache should upload bytes
    let result1 = executor.cache_rmsnorm_gamma("test_gamma", &gamma);
    assert!(result1.is_ok(), "First cache should succeed");
    let bytes1 = result1.unwrap();
    assert!(bytes1 > 0, "First cache should upload bytes");

    // Second cache of same name should return 0 (already cached)
    let result2 = executor.cache_rmsnorm_gamma("test_gamma", &gamma);
    assert!(result2.is_ok(), "Second cache should succeed");
    let bytes2 = result2.unwrap();
    assert_eq!(bytes2, 0, "Duplicate cache should return 0 bytes");
}

#[test]
#[serial]
fn test_cov015_preload_output_norm_duplicate() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let gamma = vec![1.0f32; 64];

    // First preload
    let result1 = executor.preload_output_norm(&gamma);
    assert!(result1.is_ok(), "First preload should succeed");
    let bytes1 = result1.unwrap();
    assert!(bytes1 > 0, "First preload should upload bytes");

    // Second preload of same norm should return 0
    let result2 = executor.preload_output_norm(&gamma);
    assert!(result2.is_ok(), "Second preload should succeed");
    let bytes2 = result2.unwrap();
    assert_eq!(bytes2, 0, "Duplicate preload should return 0 bytes");
}

#[test]
#[serial]
fn test_cov015_preload_rmsnorm_weights_duplicate() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let gamma = vec![1.0f32; 64];
    let attn_norms: Vec<&[f32]> = vec![gamma.as_slice()];
    let ffn_norms: Vec<&[f32]> = vec![gamma.as_slice()];

    // First preload
    let result1 = executor.preload_rmsnorm_weights(1, &attn_norms, &ffn_norms);
    assert!(result1.is_ok(), "First preload should succeed");
    let bytes1 = result1.unwrap();
    assert!(bytes1 > 0, "First preload should upload bytes");

    // Second preload should return 0 (already cached)
    let result2 = executor.preload_rmsnorm_weights(1, &attn_norms, &ffn_norms);
    assert!(result2.is_ok(), "Second preload should succeed");
    let bytes2 = result2.unwrap();
    assert_eq!(bytes2, 0, "Duplicate preload should return 0 bytes");
}

#[test]
#[serial]
fn test_cov015_preload_qkv_bias_with_none_values() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // All biases are None (no bias model)
    let q_biases: Vec<Option<&[f32]>> = vec![None];
    let k_biases: Vec<Option<&[f32]>> = vec![None];
    let v_biases: Vec<Option<&[f32]>> = vec![None];

    let result = executor.preload_qkv_bias(1, &q_biases, &k_biases, &v_biases);
    assert!(
        result.is_ok(),
        "preload_qkv_bias with None values should succeed"
    );
    assert_eq!(
        result.unwrap(),
        0,
        "No bytes should be uploaded for None biases"
    );
    assert!(
        !executor.has_qkv_bias(0),
        "Should not have QKV bias when all None"
    );
}

#[test]
#[serial]
fn test_cov015_preload_qkv_bias_partial() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let q_bias_data = vec![0.1f32; 64];
    // Only Q bias present, K and V are None
    let q_biases: Vec<Option<&[f32]>> = vec![Some(q_bias_data.as_slice())];
    let k_biases: Vec<Option<&[f32]>> = vec![None];
    let v_biases: Vec<Option<&[f32]>> = vec![None];

    let result = executor.preload_qkv_bias(1, &q_biases, &k_biases, &v_biases);
    assert!(result.is_ok(), "preload_qkv_bias partial should succeed");
    let bytes = result.unwrap();
    assert!(bytes > 0, "Should upload Q bias bytes");
    assert!(executor.has_qkv_bias(0), "Should have QKV bias (Q only)");
}

#[test]
#[serial]
fn test_cov015_forward_batched_graphed_batch_size_zero() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let inputs: Vec<f32> = vec![];
    let positions: Vec<u32> = vec![];

    let result =
        executor.forward_batched_to_token_ids_graphed(&inputs, &positions, 1, 64, 128, 256, 1e-5);

    assert!(result.is_err(), "Should fail with batch size 0");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("batch size must be 1-32"),
        "Error should mention batch size: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov015_forward_batched_graphed_batch_size_exceeds_max() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 64u32;
    let m = 33; // > 32
    let inputs = vec![0.1f32; m * hidden_dim as usize];
    let positions: Vec<u32> = (0..m as u32).collect();

    let result = executor
        .forward_batched_to_token_ids_graphed(&inputs, &positions, 1, hidden_dim, 128, 256, 1e-5);

    assert!(result.is_err(), "Should fail with batch size > 32");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("batch size must be 1-32"),
        "Error should mention batch size: {}",
        err_msg
    );
}

