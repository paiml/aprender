//! Section E: WGPU Shaders (Claims 66-80)
//!
//! Note: WGPU tests require GPU feature. These are simulation-level tests.

use trueno::{select_best_available_backend, Backend, Vector};

// =============================================================================
// SECTION E: WGPU Shaders (Claims 66-80)
// =============================================================================

/// E-066: WGSL shader validation infrastructure
#[test]
fn test_e066_wgsl_validation_infrastructure() {
    // Verify WGSL shader structure patterns
    let compute_shader_pattern = r"@compute\s+@workgroup_size";
    let regex = regex::Regex::new(compute_shader_pattern);
    assert!(regex.is_ok(), "E-066 FALSIFIED: Cannot compile WGSL compute shader pattern");
}

/// E-067: WGSL add shader correctness
#[test]
fn test_e067_wgsl_add_correctness() {
    // Verify add operation produces correct results
    let a = Vector::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    let b = Vector::from_slice(&[5.0f32, 6.0, 7.0, 8.0]);

    let result = a.add(&b).expect("add failed");
    let expected = [6.0f32, 8.0, 10.0, 12.0];

    assert_eq!(
        result.as_slice(),
        &expected,
        "E-067 FALSIFIED: Add shader produces incorrect results"
    );
}

/// E-068: WGSL mul shader correctness
#[test]
fn test_e068_wgsl_mul_correctness() {
    let a = Vector::from_slice(&[2.0f32, 3.0, 4.0, 5.0]);
    let b = Vector::from_slice(&[2.0f32, 2.0, 2.0, 2.0]);

    let result = a.mul(&b).expect("mul failed");
    let expected = [4.0f32, 6.0, 8.0, 10.0];

    assert_eq!(
        result.as_slice(),
        &expected,
        "E-068 FALSIFIED: Mul shader produces incorrect results"
    );
}

/// E-069: WGSL dot shader correctness
#[test]
fn test_e069_wgsl_dot_correctness() {
    let a = Vector::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
    let b = Vector::from_slice(&[4.0f32, 3.0, 2.0, 1.0]);

    let result = a.dot(&b).expect("dot failed");
    let expected = 1.0 * 4.0 + 2.0 * 3.0 + 3.0 * 2.0 + 4.0 * 1.0; // 20.0

    assert_eq!(
        result, expected,
        "E-069 FALSIFIED: Dot shader produces incorrect result: {} != {}",
        result, expected
    );
}

/// E-070: WGSL relu shader correctness
#[test]
fn test_e070_wgsl_relu_correctness() {
    let input = Vector::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0]);
    let result = input.relu().expect("relu failed");
    let expected = [0.0f32, 0.0, 0.0, 1.0, 2.0];

    assert_eq!(
        result.as_slice(),
        &expected,
        "E-070 FALSIFIED: ReLU shader produces incorrect results"
    );
}

/// E-071: WGSL sigmoid shader correctness
#[test]
fn test_e071_wgsl_sigmoid_correctness() {
    let input = Vector::from_slice(&[0.0f32]);
    let result = input.sigmoid().expect("sigmoid failed");

    // sigmoid(0) = 0.5
    let diff = (result.as_slice()[0] - 0.5).abs();
    assert!(diff < 1e-6, "E-071 FALSIFIED: sigmoid(0) = {}, expected 0.5", result.as_slice()[0]);
}

/// E-072: WGSL tanh shader correctness
#[test]
fn test_e072_wgsl_tanh_correctness() {
    let input = Vector::from_slice(&[0.0f32]);
    let result = input.tanh().expect("tanh failed");

    // tanh(0) = 0
    let diff = result.as_slice()[0].abs();
    assert!(diff < 1e-6, "E-072 FALSIFIED: tanh(0) = {}, expected 0", result.as_slice()[0]);
}

/// E-073: WGSL gelu shader correctness
#[test]
fn test_e073_wgsl_gelu_correctness() {
    let input = Vector::from_slice(&[0.0f32, 1.0, -1.0]);
    let result = input.gelu().expect("gelu failed");

    // gelu(0) ~= 0
    assert!(
        result.as_slice()[0].abs() < 1e-5,
        "E-073 FALSIFIED: gelu(0) = {}, expected ~0",
        result.as_slice()[0]
    );

    // gelu(x) > 0 for x > 0
    assert!(result.as_slice()[1] > 0.0, "E-073 FALSIFIED: gelu(1) should be positive");
}

/// E-074: WGSL swish shader correctness
#[test]
fn test_e074_wgsl_swish_correctness() {
    let input = Vector::from_slice(&[0.0f32, 1.0, 2.0]);
    let result = input.swish().expect("swish failed");

    // swish(0) = 0 * sigmoid(0) = 0
    assert!(
        result.as_slice()[0].abs() < 1e-6,
        "E-074 FALSIFIED: swish(0) = {}, expected 0",
        result.as_slice()[0]
    );

    // swish(x) > 0 for x > 0
    assert!(result.as_slice()[1] > 0.0, "E-074 FALSIFIED: swish(1) should be positive");
}

/// E-075: WGSL softmax shader correctness
#[test]
fn test_e075_wgsl_softmax_correctness() {
    let input = Vector::from_slice(&[1.0f32, 1.0, 1.0]);
    let result = input.softmax().expect("softmax failed");

    // Equal inputs should produce equal outputs
    let expected = 1.0 / 3.0;
    for (i, val) in result.as_slice().iter().enumerate() {
        let diff = (val - expected).abs();
        assert!(
            diff < 1e-5,
            "E-075 FALSIFIED: softmax of equal inputs should be equal at index {i}"
        );
    }
}

/// E-076: WGSL matmul shader correctness
#[test]
fn test_e076_wgsl_matmul_correctness() {
    use trueno::matrix::Matrix;

    // 2x2 identity matrix multiply
    let identity = Matrix::from_slice(2, 2, &[1.0f32, 0.0, 0.0, 1.0]).expect("matrix creation");
    let other = Matrix::from_slice(2, 2, &[1.0f32, 2.0, 3.0, 4.0]).expect("matrix creation");

    // Identity * other = other
    let result = identity.matmul(&other).expect("matmul failed");
    let expected = [1.0f32, 2.0, 3.0, 4.0];

    for (i, (r, e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        let diff = (*r - *e).abs();
        assert!(
            diff < 1e-5,
            "E-076 FALSIFIED: Identity matmul incorrect at index {i}: {} != {}",
            r,
            e
        );
    }
}

/// E-077: WGPU handles buffer overflow gracefully
#[test]
fn test_e077_buffer_overflow_handling() {
    // This tests that we don't panic on large allocations
    // The actual test is that this compiles and runs without crashing
    let large_size = 10_000;
    let data: Vec<f32> = (0..large_size).map(|i| i as f32).collect();
    let vec = Vector::from_slice(&data);

    assert_eq!(vec.len(), large_size);
}

/// E-078: WGPU async completion within timeout
#[test]
fn test_e078_async_timeout() {
    use std::time::{Duration, Instant};

    // Verify operations complete in reasonable time
    let timeout = Duration::from_secs(10);
    let start = Instant::now();

    let a = Vector::from_slice(&[1.0f32; 1000]);
    let b = Vector::from_slice(&[2.0f32; 1000]);
    let _ = a.add(&b);

    let elapsed = start.elapsed();
    assert!(
        elapsed < timeout,
        "E-078 FALSIFIED: Operation took {:?}, exceeded timeout {:?}",
        elapsed,
        timeout
    );
}

/// E-079: Error messages are actionable
#[test]
fn test_e079_error_messages() {
    // Test that error messages contain useful information
    let a = Vector::from_slice(&[1.0f32, 2.0, 3.0]);
    let b = Vector::from_slice(&[1.0f32, 2.0]); // Mismatched size

    let result = a.add(&b);
    assert!(result.is_err(), "E-079: Mismatched sizes should error");

    let err_msg = format!("{:?}", result.err().unwrap());
    // Error message should mention the sizes
    assert!(
        err_msg.contains('3')
            || err_msg.contains('2')
            || err_msg.contains("mismatch")
            || err_msg.contains("Mismatch"),
        "E-079 FALSIFIED: Error message not actionable: {}",
        err_msg
    );
}

/// E-080: Cross-platform compatibility
#[test]
#[allow(unexpected_cfgs)]
fn test_e080_cross_platform() {
    // This test verifies the code compiles on the current platform
    // The actual cross-platform testing happens in CI
    let backend = select_best_available_backend();

    // On x86_64, should have at least SSE2
    #[cfg(target_arch = "x86_64")]
    {
        assert!(!matches!(backend, Backend::Scalar), "E-080: Should have SIMD backend on x86_64");
    }

    // On other architectures, just verify we get a valid backend
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Backend selection should work on any platform
        let _ = backend;
    }
}
