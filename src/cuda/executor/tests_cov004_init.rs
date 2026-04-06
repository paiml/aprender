
#[test]
#[serial]
fn test_cov004_init_batched_kv_cache_without_regular() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Without init_kv_cache_gpu, should fail
    let result = executor.init_batched_kv_cache_gpu(2, 4);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov004_init_batched_kv_cache_valid() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Init regular first
    let _ = executor.init_kv_cache_gpu(2, 4, 4, 64, 128);

    // Now batched should work
    let result = executor.init_batched_kv_cache_gpu(2, 4);
    assert!(result.is_ok());
}

#[test]
#[serial]
fn test_cov004_reset_batched_kv_cache() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let _ = executor.init_kv_cache_gpu(2, 4, 4, 64, 128);
    let _ = executor.init_batched_kv_cache_gpu(2, 4);

    // Reset batched should work
    executor.reset_batched_kv_cache_gpu();
}

#[test]
#[serial]
fn test_cov004_flash_attention_cached_dimension_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4;
    let head_dim = 64;
    let hidden_dim = num_heads * head_dim; // 256

    let _ = executor.init_kv_cache_gpu(2, num_heads, num_heads, head_dim, 128);

    // Wrong Q dimension should fail
    let q_wrong = vec![0.0f32; 128]; // Should be 256
    let k = vec![0.0f32; hidden_dim];
    let v = vec![0.0f32; hidden_dim];
    let mut output = vec![0.0f32; hidden_dim];

    let result = executor.flash_attention_cached(0, &q_wrong, &k, &v, &mut output);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov004_flash_attention_cached_valid() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Use small dimensions known to work with flash_attention_multi_head
    let num_heads = 4;
    let head_dim = 8; // Reduced from 64
    let hidden_dim = num_heads * head_dim; // 32

    let _ = executor.init_kv_cache_gpu(2, num_heads, num_heads, head_dim, 16);

    let q = vec![1.0f32; hidden_dim];
    let k = vec![1.0f32; hidden_dim];
    let v = vec![1.0f32; hidden_dim];
    let mut output = vec![0.0f32; hidden_dim];

    let result = executor.flash_attention_cached(0, &q, &k, &v, &mut output);
    assert!(
        result.is_ok(),
        "flash_attention_cached failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap(), 1); // New sequence length is 1
    assert_eq!(executor.kv_cache_len(0), 1);
}

#[test]
#[serial]
fn test_cov004_flash_attention_cached_multiple_tokens() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4;
    let head_dim = 8;
    let hidden_dim = num_heads * head_dim;

    let _ = executor.init_kv_cache_gpu(2, num_heads, num_heads, head_dim, 16);

    let q = vec![1.0f32; hidden_dim];
    let k = vec![1.0f32; hidden_dim];
    let v = vec![1.0f32; hidden_dim];
    let mut output = vec![0.0f32; hidden_dim];

    // Add 3 tokens
    for i in 1..=3 {
        let result = executor.flash_attention_cached(0, &q, &k, &v, &mut output);
        assert!(result.is_ok(), "Token {} failed: {:?}", i, result.err());
        assert_eq!(result.unwrap(), i);
    }
    assert_eq!(executor.kv_cache_len(0), 3);
}

#[test]
#[serial]
fn test_cov004_flash_attention_cached_overflow() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 2;
    let head_dim = 8;
    let hidden_dim = num_heads * head_dim;
    let max_len = 4; // Very small for fast overflow

    let _ = executor.init_kv_cache_gpu(1, num_heads, num_heads, head_dim, max_len);

    let q = vec![1.0f32; hidden_dim];
    let k = vec![1.0f32; hidden_dim];
    let v = vec![1.0f32; hidden_dim];
    let mut output = vec![0.0f32; hidden_dim];

    // Fill cache
    for i in 0..max_len {
        let result = executor.flash_attention_cached(0, &q, &k, &v, &mut output);
        assert!(
            result.is_ok(),
            "Token {} failed during fill: {:?}",
            i,
            result.err()
        );
    }

    // Next should overflow
    let result = executor.flash_attention_cached(0, &q, &k, &v, &mut output);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov004_incremental_attention_gpu_dimension_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4;
    let num_kv_heads = 2; // GQA: fewer KV heads
    let head_dim = 8;
    let q_dim = num_heads * head_dim; // 32
    let kv_dim = num_kv_heads * head_dim; // 16

    let _ = executor.init_kv_cache_gpu(2, num_heads, num_kv_heads, head_dim, 16);

    // Wrong Q dimension
    let q_wrong = vec![0.0f32; 16]; // Should be 32
    let k = vec![0.0f32; kv_dim];
    let v = vec![0.0f32; kv_dim];
    let mut output = vec![0.0f32; q_dim];

    let result = executor.incremental_attention_gpu(0, &q_wrong, &k, &v, &mut output);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov004_incremental_attention_gpu_kv_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 8;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let _ = executor.init_kv_cache_gpu(2, num_heads, num_kv_heads, head_dim, 16);

    let q = vec![0.0f32; q_dim];
    let k_wrong = vec![0.0f32; 8]; // Wrong size
    let v = vec![0.0f32; kv_dim];
    let mut output = vec![0.0f32; q_dim];

    let result = executor.incremental_attention_gpu(0, &q, &k_wrong, &v, &mut output);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov004_incremental_attention_gpu_valid() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 4;
    let num_kv_heads = 4; // MHA (not GQA)
    let head_dim = 8;
    let hidden_dim = num_heads * head_dim;

    let _ = executor.init_kv_cache_gpu(2, num_heads, num_kv_heads, head_dim, 16);

    let q = vec![1.0f32; hidden_dim];
    let k = vec![1.0f32; hidden_dim];
    let v = vec![1.0f32; hidden_dim];
    let mut output = vec![0.0f32; hidden_dim];

    let result = executor.incremental_attention_gpu(0, &q, &k, &v, &mut output);
    assert!(
        result.is_ok(),
        "incremental_attention_gpu failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap(), 1);
}

#[test]
#[serial]
fn test_cov004_incremental_attention_gpu_gqa() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // GQA: 4 Q heads, 2 KV heads
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 8;
    let q_dim = num_heads * head_dim; // 32
    let kv_dim = num_kv_heads * head_dim; // 16

    let _ = executor.init_kv_cache_gpu(2, num_heads, num_kv_heads, head_dim, 16);

    let q = vec![1.0f32; q_dim];
    let k = vec![1.0f32; kv_dim];
    let v = vec![1.0f32; kv_dim];
    let mut output = vec![0.0f32; q_dim];

    let result = executor.incremental_attention_gpu(0, &q, &k, &v, &mut output);
    assert!(
        result.is_ok(),
        "GQA incremental attention failed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov004_incremental_attention_overflow() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 2;
    let head_dim = 8;
    let hidden_dim = num_heads * head_dim;
    let max_len = 4;

    let _ = executor.init_kv_cache_gpu(1, num_heads, num_heads, head_dim, max_len);

    let q = vec![1.0f32; hidden_dim];
    let k = vec![1.0f32; hidden_dim];
    let v = vec![1.0f32; hidden_dim];
    let mut output = vec![0.0f32; hidden_dim];

    // Fill cache
    for i in 0..max_len {
        let result = executor.incremental_attention_gpu(0, &q, &k, &v, &mut output);
        assert!(
            result.is_ok(),
            "Fill token {} failed: {:?}",
            i,
            result.err()
        );
    }

    // Next should overflow
    let result = executor.incremental_attention_gpu(0, &q, &k, &v, &mut output);
    assert!(result.is_err());
}

#[test]
#[serial]
fn test_cov004_rollback_preserves_earlier_state() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 2;
    let head_dim = 8;
    let hidden_dim = num_heads * head_dim;

    let _ = executor.init_kv_cache_gpu(2, num_heads, num_heads, head_dim, 16);

    let q = vec![1.0f32; hidden_dim];
    let k = vec![1.0f32; hidden_dim];
    let v = vec![1.0f32; hidden_dim];
    let mut output = vec![0.0f32; hidden_dim];

    // Add 5 tokens
    for i in 0..5 {
        let result = executor.flash_attention_cached(0, &q, &k, &v, &mut output);
        assert!(result.is_ok(), "Token {} failed: {:?}", i, result.err());
    }
    assert_eq!(executor.kv_cache_len(0), 5);

    // Rollback to position 2
    executor.rollback_kv_cache_gpu(2);
    assert_eq!(executor.kv_cache_len(0), 2);

    // Can add more tokens from position 2
    let result = executor.flash_attention_cached(0, &q, &k, &v, &mut output);
    assert!(
        result.is_ok(),
        "Token after rollback failed: {:?}",
        result.err()
    );
    assert_eq!(executor.kv_cache_len(0), 3);
}

#[test]
#[serial]
fn test_cov004_reset_after_tokens() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let num_heads = 2;
    let head_dim = 8;
    let hidden_dim = num_heads * head_dim;

    let _ = executor.init_kv_cache_gpu(2, num_heads, num_heads, head_dim, 16);

    let q = vec![1.0f32; hidden_dim];
    let k = vec![1.0f32; hidden_dim];
    let v = vec![1.0f32; hidden_dim];
    let mut output = vec![0.0f32; hidden_dim];

    // Add tokens
    for i in 0..5 {
        let result = executor.flash_attention_cached(0, &q, &k, &v, &mut output);
        assert!(result.is_ok(), "Token {} failed: {:?}", i, result.err());
    }

    // Reset
    executor.reset_kv_cache_gpu();
    assert_eq!(executor.kv_cache_len(0), 0);

    // Can start fresh
    let result = executor.flash_attention_cached(0, &q, &k, &v, &mut output);
    assert!(
        result.is_ok(),
        "Fresh token after reset failed: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap(), 1);
}

// =============================================================================
// COV-005: cuda/executor/attention.rs coverage tests
// Target: 16.19% → 50%+
// Tests for: incremental_attention_async, incremental_attention_into,
//            batched_incremental_attention_into, init_flash_decoding,
//            tensor_core_attention, gemm_fp16, flash_attention_memory_bytes
// =============================================================================

#[test]
fn test_cov005_flash_attention_memory_bytes() {
    // Static function - no CUDA needed
    let (naive, flash) = CudaExecutor::flash_attention_memory_bytes(128, 64);

    // Naive: 128 * 128 * 4 = 65536 bytes
    assert_eq!(naive, 128 * 128 * 4);

    // Flash: block_size(64) * block_size(64) * 4 * 2 = 32768 bytes
    assert_eq!(flash, 64 * 64 * 4 * 2);

    // Flash should always be smaller for reasonable seq_len
    assert!(flash < naive);
}

#[test]
fn test_cov005_flash_attention_memory_bytes_large() {
    let (naive, flash) = CudaExecutor::flash_attention_memory_bytes(4096, 128);

    // Naive: 4096 * 4096 * 4 = 67MB
    assert_eq!(naive, 4096 * 4096 * 4);

    // Flash is constant regardless of seq_len
    assert_eq!(flash, 64 * 64 * 4 * 2);

    // Huge difference for long sequences
    assert!(naive > flash * 1000);
}
