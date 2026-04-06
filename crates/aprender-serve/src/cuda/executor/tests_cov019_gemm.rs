
#[test]
#[serial]
fn test_cov019_gemm_fp16_valid_run() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // All dimensions multiples of 16
    let m = 32u32;
    let n = 32u32;
    let k = 64u32;

    let a = vec![0.1f32; (m * k) as usize];
    let b = vec![0.1f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    // This should succeed
    let result = executor.gemm_fp16(&a, &b, &mut c, m, n, k);

    assert!(
        result.is_ok(),
        "gemm_fp16 should succeed: {:?}",
        result.err()
    );

    // Output should have non-zero values
    let has_nonzero = c.iter().any(|&x| x.abs() > 1e-10);
    assert!(has_nonzero, "Output should have non-zero values");
}

// =========================================================================
// COV-020: gemm.rs coverage tests
// Target: 63.33% -> ~70%+ coverage
// =========================================================================

#[test]
#[serial]
fn test_cov020_synchronize_compute() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.synchronize_compute();
    assert!(
        result.is_ok(),
        "synchronize_compute should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov020_synchronize_transfer() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.synchronize_transfer();
    assert!(
        result.is_ok(),
        "synchronize_transfer should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov020_synchronize_all() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.synchronize_all();
    assert!(
        result.is_ok(),
        "synchronize_all should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov020_allocate_buffer() {
    if !CudaExecutor::is_available() {
        return;
    }
    let executor = CudaExecutor::new(0).expect("CUDA executor");

    let result = executor.allocate_buffer(1024);
    assert!(
        result.is_ok(),
        "allocate_buffer should succeed: {:?}",
        result.err()
    );

    let buf = result.unwrap();
    assert_eq!(buf.len(), 1024, "Buffer should have correct length");
}

#[test]
#[serial]
fn test_cov020_gemm_cached_weight_not_found() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let b = vec![0.1f32; 64 * 32];
    let mut c = vec![0.0f32; 64 * 32];

    let result = executor.gemm_cached("nonexistent_weight", &b, &mut c, 64, 32, 64);

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
fn test_cov020_gemm_cached_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Cache a weight
    let weight_data = vec![0.1f32; 64 * 64];
    executor
        .load_weights("test_weight", &weight_data)
        .expect("load weight");

    // Wrong B size
    let b = vec![0.1f32; 100]; // Wrong size
    let mut c = vec![0.0f32; 64 * 32];

    let result = executor.gemm_cached("test_weight", &b, &mut c, 64, 32, 64);

    assert!(result.is_err(), "Should fail with size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("mismatch") || err_msg.contains("expected"),
        "Error should mention mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_gemm_b_cached_weight_not_found() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let a = vec![0.1f32; 64 * 64];
    let mut c = vec![0.0f32; 64 * 32];

    let result = executor.gemm_b_cached("nonexistent_b_weight", &a, &mut c, 64, 32, 64);

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
fn test_cov020_gemm_b_cached_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Cache a weight (B matrix: k x n = 64 x 32)
    let weight_data = vec![0.1f32; 64 * 32];
    executor
        .load_weights("test_b_weight", &weight_data)
        .expect("load weight");

    // Wrong A size
    let a = vec![0.1f32; 100]; // Wrong size (should be m * k = 64 * 64)
    let mut c = vec![0.0f32; 64 * 32];

    let result = executor.gemm_b_cached("test_b_weight", &a, &mut c, 64, 32, 64);

    assert!(result.is_err(), "Should fail with size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("mismatch") || err_msg.contains("expected"),
        "Error should mention mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_gemm_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 32u32;
    let n = 32u32;
    let k = 64u32;

    // A is wrong size
    let a = vec![0.1f32; 100]; // Should be m * k = 32 * 64 = 2048
    let b = vec![0.1f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm(&a, &b, &mut c, m, n, k);

    assert!(result.is_err(), "Should fail with size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("mismatch") || err_msg.contains("expected"),
        "Error should mention mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_gemm_valid_run() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 32u32;
    let n = 32u32;
    let k = 64u32;

    let a = vec![0.1f32; (m * k) as usize];
    let b = vec![0.1f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm(&a, &b, &mut c, m, n, k);

    assert!(result.is_ok(), "gemm should succeed: {:?}", result.err());

    // Should have non-zero output
    let has_nonzero = c.iter().any(|&x| x.abs() > 1e-10);
    assert!(has_nonzero, "Output should have non-zero values");
}

#[test]
#[serial]
fn test_cov020_gemm_gemv_path() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // M=1 triggers GEMV path
    let m = 1u32;
    let n = 64u32;
    let k = 128u32;

    let a = vec![0.1f32; (m * k) as usize]; // 1 x 128 row vector
    let b = vec![0.1f32; (k * n) as usize]; // 128 x 64 matrix
    let mut c = vec![0.0f32; (m * n) as usize]; // 1 x 64 output

    let result = executor.gemm(&a, &b, &mut c, m, n, k);

    assert!(
        result.is_ok(),
        "gemm (gemv path) should succeed: {:?}",
        result.err()
    );

    // Should have non-zero output
    let has_nonzero = c.iter().any(|&x| x.abs() > 1e-10);
    assert!(has_nonzero, "Output should have non-zero values");
}

#[test]
#[serial]
fn test_cov020_gemv_cached_input_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Cache a weight
    let k = 128u32;
    let n = 64u32;
    let weight_data = vec![0.1f32; (k * n) as usize];
    executor
        .load_weights("gemv_weight", &weight_data)
        .expect("load weight");

    // Wrong input size
    let x = vec![0.1f32; 50]; // Should be k = 128
    let mut y = vec![0.0f32; n as usize];

    let result = executor.gemv_cached("gemv_weight", &x, &mut y, k, n);

    assert!(result.is_err(), "Should fail with input size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("mismatch") || err_msg.contains("expected"),
        "Error should mention mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_gemv_cached_output_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    // Cache a weight
    let k = 128u32;
    let n = 64u32;
    let weight_data = vec![0.1f32; (k * n) as usize];
    executor
        .load_weights("gemv_weight2", &weight_data)
        .expect("load weight");

    let x = vec![0.1f32; k as usize]; // Correct input size
    let mut y = vec![0.0f32; 10]; // Wrong output size (should be n = 64)

    let result = executor.gemv_cached("gemv_weight2", &x, &mut y, k, n);

    assert!(result.is_err(), "Should fail with output size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("mismatch") || err_msg.contains("expected"),
        "Error should mention mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_gemv_cached_weight_not_found() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let k = 128u32;
    let n = 64u32;
    let x = vec![0.1f32; k as usize];
    let mut y = vec![0.0f32; n as usize];

    let result = executor.gemv_cached("nonexistent_gemv_weight", &x, &mut y, k, n);

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
fn test_cov020_gemm_optimized_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 64u32;
    let n = 64u32;
    let k = 128u32;
    let tile_size = 32u32;

    // A is wrong size
    let a = vec![0.1f32; 100]; // Should be m * k
    let b = vec![0.1f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm_optimized(&a, &b, &mut c, m, n, k, tile_size);

    assert!(result.is_err(), "Should fail with size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("mismatch") || err_msg.contains("expected"),
        "Error should mention mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_gemm_optimized_valid_run() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 64u32;
    let n = 64u32;
    let k = 128u32;
    let tile_size = 32u32;

    let a = vec![0.1f32; (m * k) as usize];
    let b = vec![0.1f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm_optimized(&a, &b, &mut c, m, n, k, tile_size);

    assert!(
        result.is_ok(),
        "gemm_optimized should succeed: {:?}",
        result.err()
    );

    let has_nonzero = c.iter().any(|&x| x.abs() > 1e-10);
    assert!(has_nonzero, "Output should have non-zero values");
}
