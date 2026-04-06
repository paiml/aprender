//! PAR-023: Residual Add GPU Parity Test
//!
//! Verifies that the GPU residual add kernel produces the same results
//! as the CPU implementation.

/// Reference CPU residual add implementation
fn cpu_residual_add(input1: &[f32], input2: &[f32]) -> Vec<f32> {
    input1
        .iter()
        .zip(input2.iter())
        .map(|(a, b)| a + b)
        .collect()
}

/// Reference CPU fused residual add + rmsnorm implementation
fn cpu_fused_residual_rmsnorm(
    residual: &[f32],
    input: &[f32],
    gamma: &[f32],
    epsilon: f32,
) -> Vec<f32> {
    let n = residual.len();

    // Add residual + input
    let sum: Vec<f32> = residual
        .iter()
        .zip(input.iter())
        .map(|(r, i)| r + i)
        .collect();

    // Compute sum of squares
    let sq_sum: f32 = sum.iter().map(|x| x * x).sum();

    // RMS = sqrt(mean(x^2) + epsilon)
    let rms = (sq_sum / n as f32 + epsilon).sqrt();
    let rms_inv = 1.0 / rms;

    // Normalize and scale
    sum.iter()
        .zip(gamma.iter())
        .map(|(x, g)| x * rms_inv * g)
        .collect()
}

/// Test: CPU residual add basic functionality
#[test]
fn test_cpu_residual_add_basic() {
    let input1 = vec![1.0, 2.0, 3.0, 4.0];
    let input2 = vec![0.5, 0.5, 0.5, 0.5];

    let output = cpu_residual_add(&input1, &input2);

    assert_eq!(output, vec![1.5, 2.5, 3.5, 4.5]);
}

/// Test: CPU fused residual+rmsnorm basic functionality
#[test]
fn test_cpu_fused_residual_rmsnorm_basic() {
    let residual = vec![1.0f32, 2.0, 3.0, 4.0];
    let input = vec![0.5, 0.5, 0.5, 0.5];
    let gamma = vec![1.0, 1.0, 1.0, 1.0];
    let epsilon = 1e-5f32;

    let output = cpu_fused_residual_rmsnorm(&residual, &input, &gamma, epsilon);

    // Verify output is normalized
    assert_eq!(output.len(), 4);

    // Manually verify:
    // sum = [1.5, 2.5, 3.5, 4.5]
    // sq_sum = 1.5^2 + 2.5^2 + 3.5^2 + 4.5^2 = 2.25 + 6.25 + 12.25 + 20.25 = 41.0
    // rms = sqrt(41.0/4 + eps) = sqrt(10.25 + eps) ≈ 3.2016
    let sum = vec![1.5f32, 2.5, 3.5, 4.5];
    let sq_sum: f32 = sum.iter().map(|x| x * x).sum();
    let rms = (sq_sum / 4.0 + epsilon).sqrt();
    let expected: Vec<f32> = sum.iter().map(|x| x / rms).collect();

    for (o, e) in output.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-5, "Mismatch: {} vs {}", o, e);
    }
}

/// Test: CPU vs GPU residual add parity
#[test]
#[ignore] // Run with --ignored when CUDA is available
#[cfg(feature = "cuda")]
fn test_cpu_gpu_residual_add_parity() {
    use realizar::cuda::CudaExecutor;

    // Skip if CUDA not available
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
    let cpu_output = cpu_residual_add(&input1, &input2);

    // GPU execution using host convenience method
    let mut gpu_output = vec![0.0f32; n];
    executor
        .residual_add_host(&input1, &input2, &mut gpu_output)
        .expect("residual_add_host failed");

    // Compare
    let tolerance = 1e-5;
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
                    "Mismatch at {}: CPU={:.6} GPU={:.6} diff={:.6}",
                    i, cpu, gpu, diff
                );
            }
        }
    }

    if diff_count > 0 {
        eprintln!(
            "Total mismatches: {} / {} (max diff: {:.6})",
            diff_count, n, max_diff
        );
    }

    assert!(
        max_diff < tolerance,
        "CPU/GPU outputs differ by more than {}: max_diff={}",
        tolerance,
        max_diff
    );
}

/// Test: CPU vs GPU fused residual+rmsnorm parity
#[test]
#[ignore] // Run with --ignored when CUDA is available
#[cfg(feature = "cuda")]
fn test_cpu_gpu_fused_residual_rmsnorm_parity() {
    use realizar::cuda::CudaExecutor;

    // Skip if CUDA not available
    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("CUDA not available, skipping GPU test");
            return;
        },
    };

    let hidden_size = 2048usize;
    let epsilon = 1e-5f32;

    // Generate test data
    let residual: Vec<f32> = (0..hidden_size)
        .map(|i| ((i * 17) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let input: Vec<f32> = (0..hidden_size)
        .map(|i| ((i * 13) % 100) as f32 * 0.01 - 0.3)
        .collect();
    let gamma: Vec<f32> = (0..hidden_size)
        .map(|i| 0.5 + ((i * 7) % 100) as f32 * 0.01)
        .collect();

    // CPU reference
    let cpu_output = cpu_fused_residual_rmsnorm(&residual, &input, &gamma, epsilon);

    // GPU execution using host convenience method
    let mut gpu_output = vec![0.0f32; hidden_size];
    executor
        .fused_residual_rmsnorm_host(&residual, &input, &gamma, &mut gpu_output, epsilon)
        .expect("fused_residual_rmsnorm_host failed");

    // Compare
    let tolerance = 1e-3; // Slightly higher tolerance for fused ops
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
                    "Mismatch at {}: CPU={:.6} GPU={:.6} diff={:.6}",
                    i, cpu, gpu, diff
                );
            }
        }
    }

    if diff_count > 0 {
        eprintln!(
            "Total mismatches: {} / {} (max diff: {:.6})",
            diff_count, hidden_size, max_diff
        );
    }

    assert!(
        max_diff < tolerance,
        "CPU/GPU outputs differ by more than {}: max_diff={}",
        tolerance,
        max_diff
    );
}
