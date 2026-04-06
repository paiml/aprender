
#[test]
#[serial]
fn test_cov020_gemm_fused_size_mismatch() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 32u32;
    let n = 32u32;
    let k = 64u32;

    // A is wrong size
    let a = vec![0.1f32; 100];
    let b = vec![0.1f32; (k * n) as usize];
    let mut c = vec![0.0f32; (m * n) as usize];

    let result = executor.gemm_fused(&a, &b, None, &mut c, m, n, k, 0);

    assert!(result.is_err(), "Should fail with size mismatch");
}

#[test]
#[serial]
fn test_cov020_gemm_fused_bias_size_mismatch() {
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
    let bias = vec![0.1f32; 10]; // Wrong size, should be n = 32

    let result = executor.gemm_fused(&a, &b, Some(&bias), &mut c, m, n, k, 0);

    assert!(result.is_err(), "Should fail with bias size mismatch");
    let err_msg = format!("{:?}", result.err().unwrap());
    assert!(
        err_msg.contains("Bias") || err_msg.contains("mismatch"),
        "Error should mention bias mismatch: {}",
        err_msg
    );
}

#[test]
#[serial]
fn test_cov020_gemm_fused_valid_no_bias() {
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

    let result = executor.gemm_fused(&a, &b, None, &mut c, m, n, k, 0);

    assert!(
        result.is_ok(),
        "gemm_fused (no bias) should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov020_gemm_fused_with_bias_and_relu() {
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
    let bias = vec![0.1f32; n as usize]; // Correct size

    // activation = 1 (ReLU)
    let result = executor.gemm_fused(&a, &b, Some(&bias), &mut c, m, n, k, 1);

    assert!(
        result.is_ok(),
        "gemm_fused with bias+relu should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov020_softmax_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let mut data = vec![1.0f32, 2.0, 3.0, 4.0];

    let result = executor.softmax(&mut data);

    assert!(result.is_ok(), "softmax should succeed: {:?}", result.err());

    // Check output sums to 1
    let sum: f32 = data.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "Softmax should sum to 1, got {}",
        sum
    );

    // Check values are in (0, 1)
    for val in &data {
        assert!(
            *val > 0.0 && *val < 1.0,
            "Softmax values should be in (0, 1): {}",
            val
        );
    }
}

#[test]
#[serial]
fn test_cov020_gemm_cached_async_weight_not_found() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 32u32;
    let n = 32u32;
    let k = 64u32;

    let input_buf = GpuBuffer::<f32>::new(&executor.context, (k * n) as usize).expect("input");
    let output_buf = GpuBuffer::<f32>::new(&executor.context, (m * n) as usize).expect("output");

    let result =
        executor.gemm_cached_async("nonexistent_async_weight", &input_buf, &output_buf, m, n, k);

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
fn test_cov020_q4k_matvec_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let m = 32u32; // Output dimension
    let k = 256u32; // Input dimension (must be divisible by 256 for Q4K)

    // Q4K format: 144 bytes per 256 values
    let num_superblocks = (m as usize * k as usize + 255) / 256;
    let weight_bytes = num_superblocks * 144;
    let weights = vec![0u8; weight_bytes];

    let input = vec![0.1f32; k as usize];
    let mut output = vec![0.0f32; m as usize];

    let result = executor.q4k_matvec(&weights, &input, &mut output, m, k);

    assert!(
        result.is_ok(),
        "q4k_matvec should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov020_q4k_gemv_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32; // Output dimension
    let k = 256u32; // Input dimension (must be divisible by 256)

    // Q4K format: 144 bytes per 256 values
    let num_superblocks = (n as usize * k as usize + 255) / 256;
    let weight_bytes = num_superblocks * 144;
    let weights = vec![0u8; weight_bytes];

    let input = vec![0.1f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q4k_gemv(&weights, &input, &mut output, n, k);

    assert!(
        result.is_ok(),
        "q4k_gemv should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov020_q5k_gemv_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Q5K format: 176 bytes per 256 values
    let num_superblocks = (n as usize * k as usize + 255) / 256;
    let weight_bytes = num_superblocks * 176;
    let weights = vec![0u8; weight_bytes];

    let input = vec![0.1f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q5k_gemv(&weights, &input, &mut output, n, k);

    assert!(
        result.is_ok(),
        "q5k_gemv should succeed: {:?}",
        result.err()
    );
}

#[test]
#[serial]
fn test_cov020_q6k_gemv_basic() {
    if !CudaExecutor::is_available() {
        return;
    }
    let mut executor = CudaExecutor::new(0).expect("CUDA executor");

    let n = 32u32;
    let k = 256u32;

    // Q6K format: 210 bytes per 256 values
    let num_superblocks = (n as usize * k as usize + 255) / 256;
    let weight_bytes = num_superblocks * 210;
    let weights = vec![0u8; weight_bytes];

    let input = vec![0.1f32; k as usize];
    let mut output = vec![0.0f32; n as usize];

    let result = executor.q6k_gemv(&weights, &input, &mut output, n, k);

    assert!(
        result.is_ok(),
        "q6k_gemv should succeed: {:?}",
        result.err()
    );
}

