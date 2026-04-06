
// =============================================================================
// COV-022: layer.rs utility functions (Refs PMAT-802)
// =============================================================================

#[test]
#[serial]
fn test_cov022_has_rmsnorm_weights_false_when_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Layer 0 RMSNorm weights not cached - should return false
    assert!(
        !executor.has_rmsnorm_weights(0),
        "has_rmsnorm_weights should return false when not cached"
    );
    assert!(
        !executor.has_rmsnorm_weights(5),
        "has_rmsnorm_weights should return false for any layer"
    );
}

#[test]
#[serial]
fn test_cov022_has_output_norm_false_when_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Output norm not cached - should return false
    assert!(
        !executor.has_output_norm(),
        "has_output_norm should return false when not cached"
    );
}

#[test]
#[serial]
fn test_cov022_has_qkv_bias_false_when_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // QKV bias not cached - should return false
    assert!(
        !executor.has_qkv_bias(0),
        "has_qkv_bias should return false when not cached"
    );
    assert!(
        !executor.has_qkv_bias(10),
        "has_qkv_bias should return false for any layer"
    );
}

#[test]
#[serial]
fn test_cov022_has_lm_head_bias_false_when_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // LM head bias not cached - should return false
    assert!(
        !executor.has_lm_head_bias(),
        "has_lm_head_bias should return false when not cached"
    );
}

#[test]
#[serial]
fn test_cov022_preload_output_norm_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Preload output norm gamma
    let gamma = vec![1.0f32; 256]; // hidden_dim = 256
    let result = executor.preload_output_norm(&gamma);

    assert!(
        result.is_ok(),
        "preload_output_norm should succeed: {:?}",
        result.err()
    );
    let bytes = result.unwrap();
    assert!(bytes > 0, "Should have uploaded some bytes");
    assert!(
        executor.has_output_norm(),
        "Output norm should be cached now"
    );

    // Preloading again should return 0 (already cached)
    let result2 = executor.preload_output_norm(&gamma);
    assert!(result2.is_ok(), "Second preload should succeed");
    assert_eq!(result2.unwrap(), 0, "Second preload should return 0 bytes");
}

#[test]
#[serial]
fn test_cov022_preload_lm_head_bias_none() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Preload with None bias - should return 0
    let result = executor.preload_lm_head_bias(None);

    assert!(result.is_ok(), "preload_lm_head_bias(None) should succeed");
    assert_eq!(result.unwrap(), 0, "None bias should return 0 bytes");
    assert!(
        !executor.has_lm_head_bias(),
        "LM head bias should not be cached"
    );
}

#[test]
#[serial]
fn test_cov022_preload_lm_head_bias_empty() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Preload with empty bias slice - should return 0
    let empty_bias: [f32; 0] = [];
    let result = executor.preload_lm_head_bias(Some(&empty_bias));

    assert!(result.is_ok(), "preload_lm_head_bias(empty) should succeed");
    assert_eq!(result.unwrap(), 0, "Empty bias should return 0 bytes");
    assert!(
        !executor.has_lm_head_bias(),
        "LM head bias should not be cached"
    );
}

#[test]
#[serial]
fn test_cov022_preload_lm_head_bias_valid() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Preload with valid bias
    let bias = vec![0.1f32; 32000]; // vocab_size = 32000
    let result = executor.preload_lm_head_bias(Some(&bias));

    assert!(
        result.is_ok(),
        "preload_lm_head_bias should succeed: {:?}",
        result.err()
    );
    let bytes = result.unwrap();
    assert!(bytes > 0, "Should have uploaded some bytes");
    assert!(
        executor.has_lm_head_bias(),
        "LM head bias should be cached now"
    );

    // Preloading again should return 0 (already cached)
    let result2 = executor.preload_lm_head_bias(Some(&bias));
    assert!(result2.is_ok(), "Second preload should succeed");
    assert_eq!(result2.unwrap(), 0, "Second preload should return 0 bytes");
}

#[test]
#[serial]
fn test_cov022_cache_rmsnorm_gamma_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Cache a gamma weight
    let gamma = vec![1.0f32; 256];
    let result = executor.cache_rmsnorm_gamma("test_layer.gamma", &gamma);

    assert!(
        result.is_ok(),
        "cache_rmsnorm_gamma should succeed: {:?}",
        result.err()
    );
    let bytes = result.unwrap();
    assert!(bytes > 0, "Should have uploaded some bytes");

    // Caching again should return 0 (already cached)
    let result2 = executor.cache_rmsnorm_gamma("test_layer.gamma", &gamma);
    assert!(result2.is_ok(), "Second cache should succeed");
    assert_eq!(result2.unwrap(), 0, "Second cache should return 0 bytes");
}

#[test]
#[serial]
fn test_cov022_workspace_output_none_initially() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Workspace output should be None before initialization
    assert!(
        executor.workspace_output().is_none(),
        "workspace_output should be None initially"
    );
}

#[test]
#[serial]
fn test_cov022_read_hidden_state_workspace_not_initialized() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Workspace not initialized - should fail
    let result = executor.read_hidden_state_to_cpu();

    assert!(
        result.is_err(),
        "Should fail when workspace not initialized"
    );
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("workspace not initialized") || err_msg.contains("APR-TRACE-001"),
        "Error should mention workspace not initialized: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov022_preload_qkv_bias_empty() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Empty biases - should succeed with 0 bytes
    let q_biases: Vec<Option<&[f32]>> = vec![None, None];
    let k_biases: Vec<Option<&[f32]>> = vec![None, None];
    let v_biases: Vec<Option<&[f32]>> = vec![None, None];

    let result = executor.preload_qkv_bias(2, &q_biases, &k_biases, &v_biases);

    assert!(result.is_ok(), "preload_qkv_bias with None should succeed");
    assert_eq!(result.unwrap(), 0, "Should return 0 bytes for None biases");
    assert!(!executor.has_qkv_bias(0), "Should not have QKV bias");
}

#[test]
#[serial]
fn test_cov022_preload_rmsnorm_weights_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Create norm weights for 2 layers
    let attn_norm_0 = vec![1.0f32; 256];
    let attn_norm_1 = vec![1.0f32; 256];
    let ffn_norm_0 = vec![1.0f32; 256];
    let ffn_norm_1 = vec![1.0f32; 256];

    let attn_norms: Vec<&[f32]> = vec![&attn_norm_0, &attn_norm_1];
    let ffn_norms: Vec<&[f32]> = vec![&ffn_norm_0, &ffn_norm_1];

    let result = executor.preload_rmsnorm_weights(2, &attn_norms, &ffn_norms);

    assert!(
        result.is_ok(),
        "preload_rmsnorm_weights should succeed: {:?}",
        result.err()
    );
    let bytes = result.unwrap();
    assert!(bytes > 0, "Should have uploaded some bytes");

    // Check that layers have weights cached
    assert!(
        executor.has_rmsnorm_weights(0),
        "Layer 0 should have RMSNorm weights"
    );
    assert!(
        executor.has_rmsnorm_weights(1),
        "Layer 1 should have RMSNorm weights"
    );
    assert!(
        !executor.has_rmsnorm_weights(2),
        "Layer 2 should not have RMSNorm weights"
    );
}

// =============================================================================
// COV-023: quantized.rs functions (Refs PMAT-802)
// =============================================================================

#[test]
#[serial]
fn test_cov023_q4k_gemv_cached_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    let input = vec![0.1f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    // Weight not cached - should fail
    let result = executor.q4k_gemv_cached("nonexistent_weight", &input, &mut output, n, k);

    assert!(result.is_err(), "Should fail when weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not cached"),
        "Error should mention not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov023_q5k_gemv_cached_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    let input = vec![0.1f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    // Weight not cached - should fail
    let result = executor.q5k_gemv_cached("nonexistent_weight", &input, &mut output, n, k);

    assert!(result.is_err(), "Should fail when weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not cached"),
        "Error should mention not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov023_q6k_gemv_cached_weight_not_cached() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    let input = vec![0.1f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    // Weight not cached - should fail
    let result = executor.q6k_gemv_cached("nonexistent_weight", &input, &mut output, n, k);

    assert!(result.is_err(), "Should fail when weight not cached");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("not cached"),
        "Error should mention not cached: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov023_gelu_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 256u32;

    // Create GPU buffer with test data
    let data = vec![0.5f32; n as usize];
    let buffer = GpuBuffer::from_host(executor.context(), &data).expect("buffer");

    // Apply GELU
    let result = executor.gelu_gpu(&buffer, n);

    assert!(
        result.is_ok(),
        "gelu_gpu should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov023_rmsnorm_gpu_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_size = 256u32;
    let epsilon = 1e-5f32;

    // Create GPU buffers
    let input_data = vec![0.5f32; hidden_size as usize];
    let gamma_data = vec![1.0f32; hidden_size as usize];

    let input = GpuBuffer::from_host(executor.context(), &input_data).expect("input");
    let gamma = GpuBuffer::from_host(executor.context(), &gamma_data).expect("gamma");

    // Apply RMSNorm
    let result = executor.rmsnorm_gpu(&input, &gamma, hidden_size, epsilon);

    assert!(
        result.is_ok(),
        "rmsnorm_gpu should succeed: {:?}",
        result.err()
    );
    let output = result.unwrap();
    assert_eq!(
        output.len(),
        hidden_size as usize,
        "Output should have hidden_size elements"
    );
}
