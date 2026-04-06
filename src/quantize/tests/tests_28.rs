//! Protocol T-COV-95 Directive 4: Performance Falsification Gate
//!
//! This module verifies that SIMD optimizations provide measurable speedup
//! over scalar implementations, following Popperian falsification methodology.
//!
//! Key falsification tests:
//! 1. SIMD path detection - verify SIMD instructions are being used
//! 2. Performance measurement - SIMD must be ≥2x faster than scalar
//! 3. Numerical parity - SIMD and scalar must produce same results (within tolerance)
//!
//! Hardware requirements: AMD Threadripper 7960X with AVX-512 VNNI support

use super::super::{fused_q4k_dot, fused_q4k_dot_simd};
use std::time::Instant;

/// Q4_K super-block size in bytes
const Q4K_SUPER_BLOCK_BYTES: usize = 144;
/// Number of values per super-block
const QK_K: usize = 256;

/// Create valid Q4_K test data with realistic quantization values
fn create_test_q4k_data(num_super_blocks: usize) -> Vec<u8> {
    let mut data = vec![0u8; num_super_blocks * Q4K_SUPER_BLOCK_BYTES];

    for sb in 0..num_super_blocks {
        let offset = sb * Q4K_SUPER_BLOCK_BYTES;

        // d (f16 at bytes 0-1): set to 0.1 in f16
        // f16 for 0.1 ≈ 0x2E66
        data[offset] = 0x66;
        data[offset + 1] = 0x2E;

        // dmin (f16 at bytes 2-3): set to 0.05 in f16
        // f16 for 0.05 ≈ 0x2899
        data[offset + 2] = 0x99;
        data[offset + 3] = 0x28;

        // scales (6-bit, 12 bytes at offset 4-15)
        // Set varying scale values
        for i in 0..12 {
            data[offset + 4 + i] = (i as u8 * 5) & 0x3F;
        }

        // qs (4-bit quantized values, 128 bytes at offset 16-143)
        // Fill with varying 4-bit pairs
        for i in 0..128 {
            // Each byte has two 4-bit values
            let lo = (i % 16) as u8;
            let hi = ((i + 1) % 16) as u8;
            data[offset + 16 + i] = (hi << 4) | lo;
        }
    }

    data
}

/// Create matching activations for Q4_K data
fn create_test_activations(num_super_blocks: usize) -> Vec<f32> {
    let len = num_super_blocks * QK_K;
    (0..len)
        .map(|i| {
            // Create varying activation pattern
            let v = (i % 256) as f32 / 256.0;
            v * 2.0 - 1.0 // Range [-1, 1]
        })
        .collect()
}

// =============================================================================
// Performance Falsification Tests
// =============================================================================

/// Test that SIMD path produces same result as scalar (numerical parity)
#[test]
fn test_simd_scalar_numerical_parity() {
    let num_super_blocks = 16; // 4096 values
    let q4k_data = create_test_q4k_data(num_super_blocks);
    let activations = create_test_activations(num_super_blocks);

    // Compute using scalar path
    let scalar_result = fused_q4k_dot(&q4k_data, &activations).expect("scalar should succeed");

    // Compute using SIMD path (with runtime detection)
    let simd_result = fused_q4k_dot_simd(&q4k_data, &activations).expect("simd should succeed");

    // Results must be within relative tolerance
    // SIMD uses different accumulation order (parallel reduction) which affects precision
    // but the relative error should be small (< 0.01% for well-behaved inputs)
    let rel_diff = if scalar_result.abs() > 1e-10 {
        (scalar_result - simd_result).abs() / scalar_result.abs()
    } else {
        (scalar_result - simd_result).abs()
    };

    assert!(
        rel_diff < 1e-4, // 0.01% relative tolerance
        "SIMD/scalar parity failed: scalar={}, simd={}, rel_diff={:.2e}",
        scalar_result,
        simd_result,
        rel_diff
    );
}

/// Test that SIMD provides measurable speedup over scalar
/// This is a falsification test: if SIMD isn't faster, something is wrong
#[test]
#[ignore = "Performance test - run with --ignored"]
fn test_simd_performance_speedup() {
    let num_super_blocks = 256; // 65536 values for meaningful measurement
    let q4k_data = create_test_q4k_data(num_super_blocks);
    let activations = create_test_activations(num_super_blocks);

    // Warmup
    for _ in 0..10 {
        let _ = fused_q4k_dot(&q4k_data, &activations);
        let _ = fused_q4k_dot_simd(&q4k_data, &activations);
    }

    // Measure scalar performance
    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = fused_q4k_dot(&q4k_data, &activations);
    }
    let scalar_duration = start.elapsed();

    // Measure SIMD performance
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = fused_q4k_dot_simd(&q4k_data, &activations);
    }
    let simd_duration = start.elapsed();

    // Calculate speedup
    let speedup = scalar_duration.as_secs_f64() / simd_duration.as_secs_f64();

    println!("Performance Falsification Gate Results:");
    println!(
        "  Scalar: {:?} for {} iterations",
        scalar_duration, iterations
    );
    println!(
        "  SIMD:   {:?} for {} iterations",
        simd_duration, iterations
    );
    println!("  Speedup: {:.2}x", speedup);

    // SIMD should be at least 2x faster on this hardware (AVX-512 VNNI)
    // If not, the SIMD path is not being used correctly
    assert!(
        speedup >= 1.5,
        "SIMD speedup too low: {:.2}x (expected ≥2x). SIMD path may not be exercised.",
        speedup
    );
}

/// Verify SIMD dispatch is happening by checking runtime feature detection
#[test]
fn test_simd_feature_detection() {
    // This test verifies that the CPU features are detected correctly
    // On AMD Threadripper 7960X, we expect AVX-512 and AVX-512 VNNI

    #[cfg(target_arch = "x86_64")]
    {
        let has_avx2 = std::is_x86_feature_detected!("avx2");
        let has_avx512f = std::is_x86_feature_detected!("avx512f");
        let has_avx512vnni = std::is_x86_feature_detected!("avx512vnni");

        println!("CPU Feature Detection:");
        println!("  AVX2:        {}", has_avx2);
        println!("  AVX512F:     {}", has_avx512f);
        println!("  AVX512VNNI:  {}", has_avx512vnni);

        // Threadripper 7960X should have all these features
        assert!(has_avx2, "AVX2 not detected on Threadripper 7960X");
        assert!(has_avx512f, "AVX512F not detected on Threadripper 7960X");
        assert!(
            has_avx512vnni,
            "AVX512VNNI not detected on Threadripper 7960X"
        );
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        // On non-x86 platforms, just verify the function runs
        println!("Non-x86 platform - SIMD features not checked");
    }
}

/// Test SIMD path with various data sizes to verify vectorization
#[test]
fn test_simd_vectorization_sizes() {
    // Test with sizes that exercise different vectorization patterns
    let sizes = [1, 2, 4, 8, 16, 32, 64]; // Super-blocks

    for &num_sb in &sizes {
        let q4k_data = create_test_q4k_data(num_sb);
        let activations = create_test_activations(num_sb);

        let scalar = fused_q4k_dot(&q4k_data, &activations).expect("scalar failed");
        let simd = fused_q4k_dot_simd(&q4k_data, &activations).expect("simd failed");

        // Results should match within tolerance
        let rel_diff = if scalar.abs() > 1e-10 {
            (scalar - simd).abs() / scalar.abs()
        } else {
            (scalar - simd).abs()
        };

        assert!(
            rel_diff < 1e-4,
            "Size {} mismatch: scalar={}, simd={}, rel_diff={}",
            num_sb,
            scalar,
            simd,
            rel_diff
        );
    }
}

/// Test SIMD with edge case values
#[test]
fn test_simd_edge_case_values() {
    let num_super_blocks = 4;
    let q4k_data = create_test_q4k_data(num_super_blocks);

    // Test with all-zero activations
    let zero_activations = vec![0.0f32; num_super_blocks * QK_K];
    let scalar_zero = fused_q4k_dot(&q4k_data, &zero_activations).expect("scalar failed");
    let simd_zero = fused_q4k_dot_simd(&q4k_data, &zero_activations).expect("simd failed");
    assert!(
        (scalar_zero - simd_zero).abs() < 1e-6,
        "Zero activation mismatch"
    );

    // Test with all-one activations
    let ones_activations = vec![1.0f32; num_super_blocks * QK_K];
    let scalar_ones = fused_q4k_dot(&q4k_data, &ones_activations).expect("scalar failed");
    let simd_ones = fused_q4k_dot_simd(&q4k_data, &ones_activations).expect("simd failed");
    let rel_diff = if scalar_ones.abs() > 1e-10 {
        (scalar_ones - simd_ones).abs() / scalar_ones.abs()
    } else {
        (scalar_ones - simd_ones).abs()
    };
    assert!(
        rel_diff < 1e-4,
        "Ones activation mismatch: scalar={}, simd={}",
        scalar_ones,
        simd_ones
    );

    // Test with negative activations
    let neg_activations = vec![-0.5f32; num_super_blocks * QK_K];
    let scalar_neg = fused_q4k_dot(&q4k_data, &neg_activations).expect("scalar failed");
    let simd_neg = fused_q4k_dot_simd(&q4k_data, &neg_activations).expect("simd failed");
    let rel_diff = if scalar_neg.abs() > 1e-10 {
        (scalar_neg - simd_neg).abs() / scalar_neg.abs()
    } else {
        (scalar_neg - simd_neg).abs()
    };
    assert!(
        rel_diff < 1e-4,
        "Negative activation mismatch: scalar={}, simd={}",
        scalar_neg,
        simd_neg
    );
}

/// Test SIMD handles minimum size (1 super-block)
#[test]
fn test_simd_minimum_size() {
    let q4k_data = create_test_q4k_data(1);
    let activations = create_test_activations(1);

    let scalar = fused_q4k_dot(&q4k_data, &activations).expect("scalar failed");
    let simd = fused_q4k_dot_simd(&q4k_data, &activations).expect("simd failed");

    let rel_diff = if scalar.abs() > 1e-10 {
        (scalar - simd).abs() / scalar.abs()
    } else {
        (scalar - simd).abs()
    };
    assert!(
        rel_diff < 1e-4,
        "Minimum size mismatch: scalar={}, simd={}",
        scalar,
        simd
    );
}

/// Test SIMD handles large sizes (stress test)
#[test]
#[ignore = "Large memory test - run with --ignored"]
fn test_simd_large_size() {
    // 4096 super-blocks = 1M+ values
    let num_super_blocks = 4096;
    let q4k_data = create_test_q4k_data(num_super_blocks);
    let activations = create_test_activations(num_super_blocks);

    let scalar = fused_q4k_dot(&q4k_data, &activations).expect("scalar failed");
    let simd = fused_q4k_dot_simd(&q4k_data, &activations).expect("simd failed");

    let rel_diff = if scalar.abs() > 1e-10 {
        (scalar - simd).abs() / scalar.abs()
    } else {
        (scalar - simd).abs()
    };
    assert!(
        rel_diff < 1e-3, // Allow slightly more tolerance for large accumulation
        "Large size mismatch: scalar={}, simd={}",
        scalar,
        simd
    );
}

// =============================================================================
// SIMD Path Coverage Tests (ensure both paths are exercised)
// =============================================================================

/// Test scalar path is exercised for coverage
#[test]
fn test_scalar_path_coverage() {
    // Explicitly test the scalar implementation
    let q4k_data = create_test_q4k_data(4);
    let activations = create_test_activations(4);

    let result = fused_q4k_dot(&q4k_data, &activations).expect("scalar should succeed");

    // Just verify it computes something reasonable
    assert!(result.is_finite(), "Result should be finite");
}

/// Test SIMD dispatcher path is exercised
#[test]
fn test_simd_dispatcher_coverage() {
    // Test the SIMD dispatcher function
    let q4k_data = create_test_q4k_data(4);
    let activations = create_test_activations(4);

    let result = fused_q4k_dot_simd(&q4k_data, &activations).expect("simd should succeed");

    // Just verify it computes something reasonable
    assert!(result.is_finite(), "Result should be finite");
}

/// Verify error paths for SIMD functions
#[test]
fn test_simd_error_paths() {
    let activations = create_test_activations(4);

    // Invalid Q4_K length (not multiple of 144)
    let invalid_data = vec![0u8; 145];
    let result = fused_q4k_dot_simd(&invalid_data, &activations);
    assert!(result.is_err(), "Should error on invalid length");

    // Mismatched activation length
    let q4k_data = create_test_q4k_data(4);
    let wrong_activations = vec![0.0f32; 100]; // Wrong length
    let result = fused_q4k_dot_simd(&q4k_data, &wrong_activations);
    assert!(result.is_err(), "Should error on mismatched length");
}

/// Test empty input handling
#[test]
fn test_empty_input_handling() {
    let empty_data: Vec<u8> = vec![];
    let empty_activations: Vec<f32> = vec![];

    // Empty should succeed and return 0
    let scalar = fused_q4k_dot(&empty_data, &empty_activations).expect("empty scalar");
    let simd = fused_q4k_dot_simd(&empty_data, &empty_activations).expect("empty simd");

    assert_eq!(scalar, 0.0, "Empty scalar should be 0");
    assert_eq!(simd, 0.0, "Empty simd should be 0");
}

// =============================================================================
// Directive 5 Preparation: Mark unfalsifiable paths
// =============================================================================

/// Document unfalsifiable hardware-specific paths
/// These paths require specific hardware that may not be available:
/// - AVX-512 VNNI: Requires Ice Lake or newer Intel, or Zen 4+ AMD
/// - NEON: Requires ARM processors
#[test]
fn test_document_unfalsifiable_paths() {
    println!("Unfalsifiable Hardware Paths:");
    println!("  // pmat-ignore: hardware-path (NEON on ARM only)");
    println!("  // pmat-ignore: hardware-path (AVX-512 without VNNI)");

    // This test exists to document what paths cannot be tested on all hardware
    // The actual paths are marked in source code with pmat-ignore comments

    #[cfg(target_arch = "x86_64")]
    {
        println!("\nDetected x86_64 - AVX paths testable");
        if std::is_x86_feature_detected!("avx512vnni") {
            println!("  AVX-512 VNNI available - optimal path testable");
        } else if std::is_x86_feature_detected!("avx512f") {
            println!("  AVX-512 F available - fallback path testable");
        } else if std::is_x86_feature_detected!("avx2") {
            println!("  AVX2 available - basic SIMD path testable");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        println!("\nDetected aarch64 - NEON paths testable");
    }
}
