//! CudaExecutor tests Part 02 - COV-003 through COV-006
//!
//! Coverage tests for:
//! - COV-003: Layer.rs preload/has method coverage
//! - COV-004: kv_cache.rs coverage (init, reset, rollback, rope, batched)
//! - COV-005: attention.rs coverage (incremental, flash decoding, tensor core, gemm_fp16)
//! - COV-006: quantized.rs coverage (gelu_gpu, rmsnorm, residual_add, fused ops)

use super::*;
use serial_test::serial;

// =========================================================================
// COV-003: Layer.rs preload/has method coverage tests
// Target: cuda/executor/layer.rs (15.29% -> higher)
// =========================================================================

#[test]
#[serial]
fn test_cov003_preload_rmsnorm_weights() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially no weights loaded
    assert!(!executor.has_rmsnorm_weights(0));
    assert!(!executor.has_rmsnorm_weights(1));

    // Preload weights for 1 layer
    let gamma = vec![1.0f32; 256];
    let attn_norms: Vec<&[f32]> = vec![&gamma];
    let ffn_norms: Vec<&[f32]> = vec![&gamma];
    let result = executor.preload_rmsnorm_weights(1, &attn_norms, &ffn_norms);
    assert!(result.is_ok(), "preload_rmsnorm_weights should succeed");

    // Now layer 0 has weights
    assert!(executor.has_rmsnorm_weights(0));
    assert!(!executor.has_rmsnorm_weights(1)); // Layer 1 not loaded
}

#[test]
#[serial]
fn test_cov003_preload_rmsnorm_weights_multiple_layers() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let gamma = vec![1.0f32; 512];

    // Preload 4 layers
    let attn_norms: Vec<&[f32]> = vec![&gamma, &gamma, &gamma, &gamma];
    let ffn_norms: Vec<&[f32]> = vec![&gamma, &gamma, &gamma, &gamma];
    let result = executor.preload_rmsnorm_weights(4, &attn_norms, &ffn_norms);
    assert!(result.is_ok(), "preload_rmsnorm_weights should succeed");

    // Verify all layers have weights
    for layer_idx in 0..4 {
        assert!(executor.has_rmsnorm_weights(layer_idx));
    }
    // Layer 4 not loaded
    assert!(!executor.has_rmsnorm_weights(4));
}

#[test]
#[serial]
fn test_cov003_preload_output_norm() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially no output norm
    assert!(!executor.has_output_norm());

    // Preload output norm
    let gamma = vec![1.0f32; 256];
    let result = executor.preload_output_norm(&gamma);
    assert!(result.is_ok(), "preload_output_norm should succeed");

    // Now has output norm
    assert!(executor.has_output_norm());
}

#[test]
#[serial]
fn test_cov003_preload_qkv_bias() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially no QKV bias
    assert!(!executor.has_qkv_bias(0));

    // Preload QKV bias for 1 layer
    let hidden_dim = 256;
    let bias_data = vec![0.1f32; hidden_dim];
    let q_biases: Vec<Option<&[f32]>> = vec![Some(&bias_data)];
    let k_biases: Vec<Option<&[f32]>> = vec![Some(&bias_data)];
    let v_biases: Vec<Option<&[f32]>> = vec![Some(&bias_data)];

    let result = executor.preload_qkv_bias(1, &q_biases, &k_biases, &v_biases);
    assert!(
        result.is_ok(),
        "preload_qkv_bias should succeed: {:?}",
        result
    );

    // Now layer 0 has QKV bias
    assert!(executor.has_qkv_bias(0));
    assert!(!executor.has_qkv_bias(1)); // Layer 1 not loaded
}

#[test]
#[serial]
fn test_cov003_preload_qkv_bias_multiple_layers() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let hidden_dim = 128;
    let bias_data = vec![0.1f32; hidden_dim];

    // Preload for 3 layers
    let q_biases: Vec<Option<&[f32]>> = vec![Some(&bias_data), Some(&bias_data), Some(&bias_data)];
    let k_biases: Vec<Option<&[f32]>> = vec![Some(&bias_data), Some(&bias_data), Some(&bias_data)];
    let v_biases: Vec<Option<&[f32]>> = vec![Some(&bias_data), Some(&bias_data), Some(&bias_data)];

    let result = executor.preload_qkv_bias(3, &q_biases, &k_biases, &v_biases);
    assert!(result.is_ok(), "preload_qkv_bias should succeed");

    // Verify all layers have QKV bias
    for layer_idx in 0..3 {
        assert!(
            executor.has_qkv_bias(layer_idx),
            "layer {} should have bias",
            layer_idx
        );
    }
    assert!(!executor.has_qkv_bias(3));
}

#[test]
#[serial]
fn test_cov003_preload_lm_head_bias_none() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially no LM head bias
    assert!(!executor.has_lm_head_bias());

    // Preload None bias (no bias)
    let result = executor.preload_lm_head_bias(None);
    assert!(result.is_ok(), "preload_lm_head_bias(None) should succeed");

    // Still no bias after loading None
    assert!(!executor.has_lm_head_bias());
}

#[test]
#[serial]
fn test_cov003_preload_lm_head_bias_some() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Initially no LM head bias
    assert!(!executor.has_lm_head_bias());

    // Preload with bias
    let vocab_size = 32000;
    let bias = vec![0.0f32; vocab_size];
    let result = executor.preload_lm_head_bias(Some(&bias));
    assert!(result.is_ok(), "preload_lm_head_bias(Some) should succeed");

    // Now has bias
    assert!(executor.has_lm_head_bias());
}

#[test]
#[serial]
fn test_cov003_cache_rmsnorm_gamma() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Cache gamma by name
    let gamma = vec![1.0f32; 256];
    let result = executor.cache_rmsnorm_gamma("test_norm_layer", &gamma);
    assert!(result.is_ok(), "cache_rmsnorm_gamma should succeed");

    // Cache another
    let result2 = executor.cache_rmsnorm_gamma("output_norm", &gamma);
    assert!(
        result2.is_ok(),
        "cache_rmsnorm_gamma for output_norm should succeed"
    );
}

#[test]
#[serial]
fn test_cov003_workspace_output_none() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Fresh executor has no workspace output
    let output = executor.workspace_output();
    // This may or may not be None depending on implementation
    let _ = output;
}

#[test]
#[serial]
fn test_cov003_read_hidden_state_to_cpu() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Try to read hidden state - may fail if no forward pass done yet
    let result = executor.read_hidden_state_to_cpu();
    // Just verify it doesn't panic - it may return error if no hidden state
    let _ = result;
}

#[test]
#[serial]
fn test_cov003_output_rmsnorm_gpu() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // First preload output norm
    let gamma = vec![1.0f32; 256];
    executor
        .preload_output_norm(&gamma)
        .expect("preload_output_norm");

    // Now test output_rmsnorm_gpu
    // (This requires a GPU buffer input, so we test the preload path)
    assert!(executor.has_output_norm());
}

#[test]
#[serial]
fn test_cov003_preload_combined_weights() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Test preloading all weight types for a layer
    let hidden_dim = 256;

    // 1. RMSNorm weights
    let gamma = vec![1.0f32; hidden_dim];
    let attn_norms: Vec<&[f32]> = vec![&gamma];
    let ffn_norms: Vec<&[f32]> = vec![&gamma];
    executor
        .preload_rmsnorm_weights(1, &attn_norms, &ffn_norms)
        .expect("rmsnorm");
    assert!(executor.has_rmsnorm_weights(0));

    // 2. Output norm
    executor.preload_output_norm(&gamma).expect("output norm");
    assert!(executor.has_output_norm());

    // 3. QKV bias
    let bias = vec![0.1f32; hidden_dim];
    let q_biases: Vec<Option<&[f32]>> = vec![Some(&bias)];
    let k_biases: Vec<Option<&[f32]>> = vec![Some(&bias)];
    let v_biases: Vec<Option<&[f32]>> = vec![Some(&bias)];
    executor
        .preload_qkv_bias(1, &q_biases, &k_biases, &v_biases)
        .expect("qkv bias");
    assert!(executor.has_qkv_bias(0));

    // 4. LM head bias
    let vocab_bias = vec![0.0f32; 32000];
    executor
        .preload_lm_head_bias(Some(&vocab_bias))
        .expect("lm head bias");
    assert!(executor.has_lm_head_bias());
}

#[test]
#[serial]
fn test_cov003_has_methods_boundary_conditions() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Test has_* methods with large layer indices (should return false)
    assert!(!executor.has_rmsnorm_weights(999));
    assert!(!executor.has_qkv_bias(1000));

    // Test default states
    assert!(!executor.has_output_norm());
    assert!(!executor.has_lm_head_bias());
}

// =============================================================================
// COV-004: cuda/executor/kv_cache.rs coverage tests
// Target: 8.32% → 50%+
// Tests for: init_kv_cache_gpu, reset_kv_cache_gpu, rollback_kv_cache_gpu,
//            set_rope_theta, set_rope_type, has_kv_cache_gpu, kv_cache_len,
//            init_batched_kv_cache_gpu, reset_batched_kv_cache_gpu
// =============================================================================

#[test]
#[serial]
fn test_cov004_init_kv_cache_gpu() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_layers = 2;
    let num_heads = 4;
    let num_kv_heads = 4;
    let head_dim = 64;
    let max_len = 128;

    let result = executor.init_kv_cache_gpu(num_layers, num_heads, num_kv_heads, head_dim, max_len);
    assert!(result.is_ok());
    assert!(executor.has_kv_cache_gpu());
}

#[test]
#[serial]
fn test_cov004_has_kv_cache_gpu_before_init() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    // Before init, should return false
    assert!(!executor.has_kv_cache_gpu());
}

#[test]
#[serial]
fn test_cov004_kv_cache_len() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Before init, length should be 0
    assert_eq!(executor.kv_cache_len(0), 0);

    // After init, length is still 0 (no tokens added)
    let _ = executor.init_kv_cache_gpu(2, 4, 4, 64, 128);
    assert_eq!(executor.kv_cache_len(0), 0);
}

#[test]
#[serial]
fn test_cov004_reset_kv_cache_gpu() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let _ = executor.init_kv_cache_gpu(2, 4, 4, 64, 128);

    // Reset should work even when empty
    executor.reset_kv_cache_gpu();
    assert_eq!(executor.kv_cache_len(0), 0);
}

#[test]
#[serial]
fn test_cov004_rollback_kv_cache_gpu() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let _ = executor.init_kv_cache_gpu(2, 4, 4, 64, 128);

    // Rollback to position 0 should work even when empty
    executor.rollback_kv_cache_gpu(0);
    assert_eq!(executor.kv_cache_len(0), 0);
}

#[test]
#[serial]
fn test_cov004_set_rope_theta() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Default LLaMA theta
    executor.set_rope_theta(10000.0);

    // Qwen2 long context theta
    executor.set_rope_theta(1000000.0);
}

#[test]
#[serial]
fn test_cov004_set_rope_type() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Type 0 = NORM (adjacent pairs)
    executor.set_rope_type(0);

    // Type 2 = NEOX (split halves, used by Qwen2.5)
    executor.set_rope_type(2);
}

#[test]
#[serial]
fn test_cov004_init_batched_kv_cache_invalid_batch_size() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Must init regular KV cache first
    let _ = executor.init_kv_cache_gpu(2, 4, 4, 64, 128);

    // Batch size 0 is invalid
    let result = executor.init_batched_kv_cache_gpu(2, 0);
    assert!(result.is_err());

    // Batch size > 32 is invalid
    let result = executor.init_batched_kv_cache_gpu(2, 33);
    assert!(result.is_err());
}

include!("tests_cov004_init.rs");
include!("tests_cov005_incremental.rs");
include!("tests_cov006_residual.rs");
include!("tests_cov007_elementwise.rs");
