//! PAR-023: RMSNorm GPU Parity Test
//!
//! Verifies that the GPU RMSNorm kernel produces the same results
//! as the CPU implementation.

#![cfg(feature = "cuda")]

/// Reference CPU RMSNorm implementation
fn cpu_rmsnorm(input: &[f32], gamma: &[f32], epsilon: f32) -> Vec<f32> {
    let n = input.len();

    // Compute sum of squares
    let sq_sum: f32 = input.iter().map(|x| x * x).sum();

    // RMS = sqrt(mean(x^2) + epsilon)
    let rms = (sq_sum / n as f32 + epsilon).sqrt();
    let rms_inv = 1.0 / rms;

    // Normalize and scale
    input
        .iter()
        .zip(gamma.iter())
        .map(|(x, g)| x * rms_inv * g)
        .collect()
}

/// Test: CPU RMSNorm basic functionality
#[test]
fn test_cpu_rmsnorm_basic() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let gamma = vec![1.0, 1.0, 1.0, 1.0];
    let epsilon = 1e-5;

    let output = cpu_rmsnorm(&input, &gamma, epsilon);

    // Verify output is normalized
    assert_eq!(output.len(), 4);

    // With gamma=1, output should be input/rms
    // rms = sqrt((1+4+9+16)/4 + eps) = sqrt(7.5 + eps) ≈ 2.739
    let rms = ((1.0 + 4.0 + 9.0 + 16.0) / 4.0 + epsilon).sqrt();
    let expected: Vec<f32> = input.iter().map(|x| x / rms).collect();

    for (o, e) in output.iter().zip(expected.iter()) {
        assert!((o - e).abs() < 1e-5, "Mismatch: {} vs {}", o, e);
    }
}

/// Test: CPU vs GPU RMSNorm parity
#[test]
#[ignore] // Run with --ignored when CUDA is available
fn test_cpu_gpu_rmsnorm_parity() {
    use realizar::cuda::CudaExecutor;

    // Skip if CUDA not available
    let mut executor = match CudaExecutor::new(0) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("CUDA not available, skipping GPU test");
            return;
        },
    };

    // Test with TinyLlama hidden_dim = 2048
    let hidden_size = 2048usize;
    let epsilon = 1e-5f32;

    // Generate test data
    let input: Vec<f32> = (0..hidden_size)
        .map(|i| ((i * 17) % 100) as f32 * 0.01 - 0.5)
        .collect();
    let gamma: Vec<f32> = (0..hidden_size)
        .map(|i| 0.5 + ((i * 7) % 100) as f32 * 0.01)
        .collect();

    // CPU reference
    let cpu_output = cpu_rmsnorm(&input, &gamma, epsilon);

    // GPU execution using convenience method
    let mut gpu_output = vec![0.0f32; hidden_size];
    executor
        .rmsnorm_host(&input, &gamma, &mut gpu_output, epsilon)
        .expect("rmsnorm_host failed");

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

/// Test: RMSNorm with unit gamma (should normalize only)
#[test]
fn test_rmsnorm_unit_gamma() {
    let input = vec![3.0f32, 4.0];
    let gamma = vec![1.0f32, 1.0];
    let epsilon = 0.0f32;

    let output = cpu_rmsnorm(&input, &gamma, epsilon);

    // rms = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
    let rms = ((9.0f32 + 16.0) / 2.0).sqrt();
    assert!((output[0] - 3.0 / rms).abs() < 1e-6);
    assert!((output[1] - 4.0 / rms).abs() < 1e-6);
}
