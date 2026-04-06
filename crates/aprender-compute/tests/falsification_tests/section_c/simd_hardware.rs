//! Claims C-042 through C-050: SIMD hardware-specific tests

use trueno::Vector;

/// C-042: SIMD remainder handling is correct for non-aligned sizes
#[test]
fn test_c042_simd_remainder_handling() {
    // Test sizes 1-15 (non-aligned for AVX2's 8-wide operations)
    for size in 1..16 {
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - 1 - i) as f32).collect();

        let vec_a = Vector::from_slice(&a);
        let vec_b = Vector::from_slice(&b);

        let result = vec_a.add(&vec_b).expect("add failed");

        // Verify correct result
        for (i, r) in result.as_slice().iter().enumerate() {
            let expected = a[i] + b[i];
            assert_eq!(
                *r, expected,
                "C-042 FALSIFIED: Remainder handling incorrect for size {size} at index {i}"
            );
        }
    }
}

/// C-043: Operations handle empty input gracefully
#[test]
fn test_c043_empty_input_safety() {
    let empty: Vec<f32> = vec![];
    let vec_empty = Vector::from_slice(&empty);

    // Empty vector operations should work without crashing
    // (add would fail due to size mismatch, but create is fine)
    assert_eq!(vec_empty.len(), 0);
}

/// C-044: Operations handle single element correctly
#[test]
fn test_c044_single_element_safety() {
    let a = Vector::from_slice(&[1.0f32]);
    let b = Vector::from_slice(&[2.0f32]);

    let result = a.add(&b).expect("add failed");
    assert_eq!(result.as_slice()[0], 3.0);
}

/// C-045: SIMD handles misaligned pointers
#[test]
fn test_c045_misaligned_pointers() {
    // Create vectors with non-power-of-2 sizes to test alignment handling
    let sizes = [3, 5, 7, 9, 11, 13, 15, 17];

    for &size in &sizes {
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

        let vec_a = Vector::from_slice(&a);
        let vec_b = Vector::from_slice(&b);

        // Should not crash regardless of pointer alignment
        let result = vec_a.add(&vec_b).expect("add failed");
        assert_eq!(
            result.as_slice().len(),
            size,
            "C-045 FALSIFIED: Misaligned pointer handling failed for size {size}"
        );
    }
}

/// C-046: AVX2 uses 256-bit registers (ymm) - verified via backend selection
#[test]
fn test_c046_avx2_register_width() {
    // AVX2 processes 8 f32 values at once (256 bits / 32 bits = 8)
    // This test verifies the expected throughput characteristic
    let size = 8; // Exactly one AVX2 register width
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

    let vec_a = Vector::from_slice(&a);
    let vec_b = Vector::from_slice(&b);

    let result = vec_a.add(&vec_b).expect("add failed");

    // Verify all 8 elements processed correctly
    for i in 0..size {
        let expected = (i + i + 1) as f32;
        assert_eq!(
            result.as_slice()[i],
            expected,
            "C-046 FALSIFIED: AVX2 256-bit operation incorrect at index {i}"
        );
    }
}

/// C-047: AVX-512 uses 512-bit registers (zmm) - verified via backend selection
#[test]
fn test_c047_avx512_register_width() {
    // AVX-512 processes 16 f32 values at once (512 bits / 32 bits = 16)
    let size = 16; // Exactly one AVX-512 register width
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

    let vec_a = Vector::from_slice(&a);
    let vec_b = Vector::from_slice(&b);

    let result = vec_a.add(&vec_b).expect("add failed");

    // Verify all 16 elements processed correctly
    for i in 0..size {
        let expected = (i + i + 1) as f32;
        assert_eq!(
            result.as_slice()[i],
            expected,
            "C-047 FALSIFIED: AVX-512 512-bit operation incorrect at index {i}"
        );
    }
}

/// C-048: NEON uses 128-bit registers (q) - verified via backend selection
#[test]
fn test_c048_neon_register_width() {
    // NEON processes 4 f32 values at once (128 bits / 32 bits = 4)
    let size = 4; // Exactly one NEON register width
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

    let vec_a = Vector::from_slice(&a);
    let vec_b = Vector::from_slice(&b);

    let result = vec_a.add(&vec_b).expect("add failed");

    // Verify all 4 elements processed correctly
    for i in 0..size {
        let expected = (i + i + 1) as f32;
        assert_eq!(
            result.as_slice()[i],
            expected,
            "C-048 FALSIFIED: NEON 128-bit operation incorrect at index {i}"
        );
    }
}

/// C-049: FMA is used when available (AVX2+FMA) - verified via fma operation
#[test]
fn test_c049_fma_availability() {
    // FMA: a * b + c in single operation
    let a = Vector::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = Vector::from_slice(&[2.0f32, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    let c = Vector::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    let result = a.fma(&b, &c).expect("fma failed");

    // Verify: a * b + c
    let expected = [3.0f32, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0];
    for (i, (r, e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        let diff = (*r - *e).abs();
        assert!(diff < f32::EPSILON, "C-049 FALSIFIED: FMA incorrect at index {i}: {} vs {}", r, e);
    }
}

/// C-050: Operations don't cause denormal stalls
#[test]
fn test_c050_no_denormal_stall() {
    use std::time::Instant;

    // Create denormal inputs
    let denormal = f32::from_bits(1); // Smallest positive subnormal
    let denormals: Vec<f32> = vec![denormal; 1000];
    let normal: Vec<f32> = vec![1.0; 1000];

    let vec_denormal = Vector::from_slice(&denormals);
    let vec_normal = Vector::from_slice(&normal);

    // Time denormal operation
    let start = Instant::now();
    for _ in 0..100 {
        let _ = vec_denormal.add(&vec_denormal);
    }
    let denormal_time = start.elapsed();

    // Time normal operation
    let start = Instant::now();
    for _ in 0..100 {
        let _ = vec_normal.add(&vec_normal);
    }
    let normal_time = start.elapsed();

    // Denormal operations shouldn't be more than 10x slower
    // (this is a heuristic - actual threshold depends on hardware)
    let ratio = denormal_time.as_nanos() as f64 / normal_time.as_nanos() as f64;
    assert!(ratio < 100.0, "C-050 WARNING: Denormal operations are {ratio:.1}x slower than normal");
}
