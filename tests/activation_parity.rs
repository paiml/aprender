//! PAR-023: Activation GPU Parity Tests
//!
//! Verifies that the GPU activation kernels produce the same results
//! as the CPU implementations.

/// Reference CPU SiLU implementation: x * sigmoid(x)
fn cpu_silu(input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
        .collect()
}

/// Reference CPU GELU implementation (approximate)
fn cpu_gelu(input: &[f32]) -> Vec<f32> {
    const SQRT_2_PI: f32 = 0.797_884_6;
    const C: f32 = 0.044_715;

    input
        .iter()
        .map(|&x| {
            let inner = SQRT_2_PI * (x + C * x.powi(3));
            0.5 * x * (1.0 + inner.tanh())
        })
        .collect()
}

/// Reference CPU element-wise multiply
fn cpu_elementwise_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Reference CPU fused SwiGLU: silu(gate) * up
fn cpu_fused_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    let silu_gate = cpu_silu(gate);
    cpu_elementwise_mul(&silu_gate, up)
}

// =========================================================================
// CPU Unit Tests
// =========================================================================

#[test]
fn test_cpu_silu_basic() {
    let input = vec![0.0, 1.0, -1.0, 2.0];
    let output = cpu_silu(&input);

    // silu(0) = 0
    assert!((output[0] - 0.0).abs() < 1e-5);
    // silu(1) ≈ 0.7311
    assert!((output[1] - 0.7311).abs() < 1e-3);
    // silu(-1) ≈ -0.2689
    assert!((output[2] - (-0.2689)).abs() < 1e-3);
}

#[test]
fn test_cpu_gelu_basic() {
    let input = vec![0.0, 1.0, -1.0, 2.0];
    let output = cpu_gelu(&input);

    // gelu(0) = 0
    assert!((output[0] - 0.0).abs() < 1e-5);
    // gelu(1) ≈ 0.8413
    assert!((output[1] - 0.8413).abs() < 1e-3);
    // gelu(-1) ≈ -0.1587
    assert!((output[2] - (-0.1587)).abs() < 1e-3);
}

#[test]
fn test_cpu_elementwise_mul_basic() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![0.5, 0.5, 0.5, 0.5];
    let output = cpu_elementwise_mul(&a, &b);

    assert_eq!(output, vec![0.5, 1.0, 1.5, 2.0]);
}

#[test]
fn test_cpu_fused_swiglu_basic() {
    let gate = vec![1.0, 1.0, 1.0, 1.0];
    let up = vec![2.0, 2.0, 2.0, 2.0];
    let output = cpu_fused_swiglu(&gate, &up);

    // silu(1) ≈ 0.7311, so output ≈ 0.7311 * 2 ≈ 1.4622
    for val in &output {
        assert!((val - 1.4622).abs() < 1e-2);
    }
}

// =========================================================================
// GPU Parity Tests
// =========================================================================

/// Test: CPU vs GPU SiLU parity
#[test]
#[ignore] // Run with --ignored when CUDA is available
#[cfg(feature = "cuda")]
fn test_cpu_gpu_silu_parity() {
    use realizar::cuda::CudaExecutor;

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("CUDA not available, skipping GPU test");
            return;
        },
    };

    let n = 2048usize;

    // Generate test data with some negative values
    let input: Vec<f32> = (0..n)
        .map(|i| ((i * 17) % 100) as f32 * 0.04 - 2.0) // Range [-2, 2]
        .collect();

    // CPU reference
    let cpu_output = cpu_silu(&input);

    // GPU execution using host convenience method
    let mut gpu_output = vec![0.0f32; n];
    executor
        .silu_host(&input, &mut gpu_output)
        .expect("silu_host failed");

    // Compare with tolerance for transcendental functions
    let tolerance = 1e-4;
    let mut max_diff = 0.0f32;
    let mut diff_count = 0;

    for (i, (cpu, gpu)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
        let diff = (cpu - gpu).abs();
        if diff > tolerance {
            diff_count += 1;
            if diff > max_diff {
                max_diff = diff;
            }
            if diff_count <= 10 {
                eprintln!(
                    "SiLU mismatch at {}: CPU={:.6} GPU={:.6} diff={:.6}",
                    i, cpu, gpu, diff
                );
            }
        }
    }

    if diff_count > 0 {
        eprintln!(
            "SiLU total mismatches: {} / {} (max diff: {:.6})",
            diff_count, n, max_diff
        );
    }

    assert!(
        max_diff < tolerance,
        "SiLU CPU/GPU outputs differ by more than {}: max_diff={}",
        tolerance,
        max_diff
    );
}

/// Test: CPU vs GPU GELU parity
#[test]
#[ignore] // Run with --ignored when CUDA is available
#[cfg(feature = "cuda")]
fn test_cpu_gpu_gelu_parity() {
    use realizar::cuda::CudaExecutor;

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("CUDA not available, skipping GPU test");
            return;
        },
    };

    let n = 2048usize;

    // Generate test data
    let input: Vec<f32> = (0..n)
        .map(|i| ((i * 17) % 100) as f32 * 0.04 - 2.0)
        .collect();

    // CPU reference
    let cpu_output = cpu_gelu(&input);

    // GPU execution using host convenience method
    let mut gpu_output = vec![0.0f32; n];
    executor
        .gelu_host(&input, &mut gpu_output)
        .expect("gelu_host failed");

    // Compare with slightly higher tolerance for GELU (tanh approximation)
    let tolerance = 1e-3;
    let mut max_diff = 0.0f32;
    let mut diff_count = 0;

    for (i, (cpu, gpu)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
        let diff = (cpu - gpu).abs();
        if diff > tolerance {
            diff_count += 1;
            if diff > max_diff {
                max_diff = diff;
            }
            if diff_count <= 10 {
                eprintln!(
                    "GELU mismatch at {}: CPU={:.6} GPU={:.6} diff={:.6}",
                    i, cpu, gpu, diff
                );
            }
        }
    }

    if diff_count > 0 {
        eprintln!(
            "GELU total mismatches: {} / {} (max diff: {:.6})",
            diff_count, n, max_diff
        );
    }

    assert!(
        max_diff < tolerance,
        "GELU CPU/GPU outputs differ by more than {}: max_diff={}",
        tolerance,
        max_diff
    );
}

/// Test: CPU vs GPU element-wise multiply parity
#[test]
#[ignore] // Run with --ignored when CUDA is available
#[cfg(feature = "cuda")]
fn test_cpu_gpu_elementwise_mul_parity() {
    use realizar::cuda::CudaExecutor;

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("CUDA not available, skipping GPU test");
            return;
        },
    };

    let n = 2048usize;

    // Generate test data
    let input1: Vec<f32> = (0..n)
        .map(|i| ((i * 17) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let input2: Vec<f32> = (0..n)
        .map(|i| ((i * 13) % 100) as f32 * 0.01 - 0.3)
        .collect();

    // CPU reference
    let cpu_output = cpu_elementwise_mul(&input1, &input2);

    // GPU execution using host convenience method
    let mut gpu_output = vec![0.0f32; n];
    executor
        .elementwise_mul_host(&input1, &input2, &mut gpu_output)
        .expect("elementwise_mul_host failed");

    // Compare
    let tolerance = 1e-5;
    let mut max_diff = 0.0f32;

    for (i, (cpu, gpu)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
        let diff = (cpu - gpu).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > tolerance && i < 10 {
            eprintln!(
                "Mul mismatch at {}: CPU={:.6} GPU={:.6} diff={:.6}",
                i, cpu, gpu, diff
            );
        }
    }

    assert!(
        max_diff < tolerance,
        "ElementwiseMul CPU/GPU outputs differ by more than {}: max_diff={}",
        tolerance,
        max_diff
    );
}

/// Test: CPU vs GPU fused SwiGLU parity
#[test]
#[ignore] // Run with --ignored when CUDA is available
#[cfg(feature = "cuda")]
fn test_cpu_gpu_fused_swiglu_parity() {
    use realizar::cuda::CudaExecutor;

    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("CUDA not available, skipping GPU test");
            return;
        },
    };

    let n = 2048usize;

    // Generate test data
    let gate: Vec<f32> = (0..n)
        .map(|i| ((i * 17) % 100) as f32 * 0.04 - 2.0)
        .collect();
    let up: Vec<f32> = (0..n)
        .map(|i| ((i * 13) % 100) as f32 * 0.02 - 1.0)
        .collect();

    // CPU reference
    let cpu_output = cpu_fused_swiglu(&gate, &up);

    // GPU execution using host convenience method
    let mut gpu_output = vec![0.0f32; n];
    executor
        .fused_swiglu_host(&gate, &up, &mut gpu_output)
        .expect("fused_swiglu_host failed");

    // Compare
    let tolerance = 1e-3;
    let mut max_diff = 0.0f32;
    let mut diff_count = 0;

    for (i, (cpu, gpu)) in cpu_output.iter().zip(gpu_output.iter()).enumerate() {
        let diff = (cpu - gpu).abs();
        if diff > tolerance {
            diff_count += 1;
            if diff > max_diff {
                max_diff = diff;
            }
            if diff_count <= 10 {
                eprintln!(
                    "SwiGLU mismatch at {}: CPU={:.6} GPU={:.6} diff={:.6}",
                    i, cpu, gpu, diff
                );
            }
        }
    }

    if diff_count > 0 {
        eprintln!(
            "SwiGLU total mismatches: {} / {} (max diff: {:.6})",
            diff_count, n, max_diff
        );
    }

    assert!(
        max_diff < tolerance,
        "FusedSwiGLU CPU/GPU outputs differ by more than {}: max_diff={}",
        tolerance,
        max_diff
    );
}
