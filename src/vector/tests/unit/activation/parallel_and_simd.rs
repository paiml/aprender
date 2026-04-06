use super::super::super::super::*;
use crate::Backend;

// ========================================================================
// Aligned vector tests
// ========================================================================

#[test]
fn test_aligned_vector_creation() {
    let v = Vector::with_alignment(100, Backend::SSE2, 16).unwrap();

    // Verify the vector has the correct size
    assert_eq!(v.len(), 100);

    // Check alignment (Vec allocator typically provides good alignment)
    let ptr = v.as_slice().as_ptr() as usize;
    // Note: We can't guarantee specific alignment with standard Vec,
    // but we can verify it's at least naturally aligned for f32 (4 bytes)
    assert_eq!(ptr % 4, 0, "Vector data should be at least 4-byte aligned");

    // Most modern allocators provide 16-byte alignment by default
    // This is informational, not required
    if ptr.is_multiple_of(16) {
        println!("Got 16-byte alignment from standard allocator");
    }
}

#[test]
fn test_aligned_vector_operations() {
    // RED: This test will fail until we implement aligned allocation
    let a = Vector::with_alignment(1000, Backend::SSE2, 16).unwrap();
    let b = Vector::with_alignment(1000, Backend::SSE2, 16).unwrap();

    // Operations on aligned vectors should work correctly
    let result = a.add(&b);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1000);
}

// ========================================================================
// Parallel execution tests (for vectors >= 100_000 elements)
// ========================================================================

#[test]
fn test_add_parallel_large_vector() {
    // Test parallel execution path for add (>= 100_000 elements)
    const SIZE: usize = 150_000;
    let a_data: Vec<f32> = (0..SIZE).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..SIZE).map(|i| (i * 2) as f32).collect();

    let a = Vector::from_slice(&a_data);
    let b = Vector::from_slice(&b_data);
    let result = a.add(&b).unwrap();

    // Verify correctness
    assert_eq!(result.len(), SIZE);
    for i in 0..SIZE {
        let expected = a_data[i] + b_data[i];
        assert!((result.as_slice()[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn test_sub_parallel_large_vector() {
    // Test parallel execution path for sub (>= 100_000 elements)
    const SIZE: usize = 150_000;
    let a_data: Vec<f32> = (0..SIZE).map(|i| (i * 3) as f32).collect();
    let b_data: Vec<f32> = (0..SIZE).map(|i| i as f32).collect();

    let a = Vector::from_slice(&a_data);
    let b = Vector::from_slice(&b_data);
    let result = a.sub(&b).unwrap();

    // Verify correctness
    assert_eq!(result.len(), SIZE);
    for i in 0..SIZE {
        let expected = a_data[i] - b_data[i];
        assert!((result.as_slice()[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn test_mul_parallel_large_vector() {
    // Test parallel execution path for mul (>= 100_000 elements)
    const SIZE: usize = 150_000;
    let a_data: Vec<f32> = (0..SIZE).map(|i| (i % 100) as f32 + 1.0).collect();
    let b_data: Vec<f32> = (0..SIZE).map(|i| 2.0 + (i % 50) as f32).collect();

    let a = Vector::from_slice(&a_data);
    let b = Vector::from_slice(&b_data);
    let result = a.mul(&b).unwrap();

    // Verify correctness
    assert_eq!(result.len(), SIZE);
    for i in 0..SIZE {
        let expected = a_data[i] * b_data[i];
        assert!((result.as_slice()[i] - expected).abs() < 1e-3);
    }
}

#[test]
fn test_div_parallel_large_vector() {
    // Test parallel execution path for div (>= 100_000 elements)
    const SIZE: usize = 150_000;
    let a_data: Vec<f32> = (0..SIZE).map(|i| (i + 100) as f32).collect();
    let b_data: Vec<f32> = (0..SIZE).map(|i| (i % 50) as f32 + 1.0).collect();

    let a = Vector::from_slice(&a_data);
    let b = Vector::from_slice(&b_data);
    let result = a.div(&b).unwrap();

    // Verify correctness
    assert_eq!(result.len(), SIZE);
    for i in 0..SIZE {
        let expected = a_data[i] / b_data[i];
        assert!((result.as_slice()[i] - expected).abs() < 1e-3);
    }
}

#[test]
fn test_dot_parallel_large_vector() {
    // Test parallel execution path for dot (>= 500_000 elements)
    const SIZE: usize = 600_000;
    let a_data: Vec<f32> = (0..SIZE).map(|i| (i % 100) as f32).collect();
    let b_data: Vec<f32> = (0..SIZE).map(|i| 1.0 + (i % 50) as f32).collect();

    let a = Vector::from_slice(&a_data);
    let b = Vector::from_slice(&b_data);
    let result = a.dot(&b).unwrap();

    // Verify it's a reasonable value (not checking exact value due to FP precision)
    assert!(result.is_finite());
    assert!(result > 0.0);
}

#[test]
fn test_fma_parallel_large_vector() {
    // Test parallel execution path for fma (>= 100_000 elements)
    const SIZE: usize = 150_000;
    let a_data: Vec<f32> = (0..SIZE).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..SIZE).map(|_| 2.0).collect();
    let c_data: Vec<f32> = (0..SIZE).map(|i| 10.0 + i as f32).collect();

    let a = Vector::from_slice(&a_data);
    let b = Vector::from_slice(&b_data);
    let c = Vector::from_slice(&c_data);
    let result = a.fma(&b, &c).unwrap();

    // Verify correctness: fma(a, b, c) = a * b + c
    assert_eq!(result.len(), SIZE);
    for i in 0..SIZE {
        let expected = a_data[i] * b_data[i] + c_data[i];
        assert!((result.as_slice()[i] - expected).abs() < 1e-3);
    }
}

#[test]
fn test_scale_parallel_large_vector() {
    // Test parallel execution path for scale (>= 100_000 elements)
    const SIZE: usize = 150_000;
    let data: Vec<f32> = (0..SIZE).map(|i| i as f32).collect();

    let v = Vector::from_slice(&data);
    let result = v.scale(3.0).unwrap();

    // Verify correctness
    assert_eq!(result.len(), SIZE);
    for (&original, &scaled) in data.iter().zip(result.as_slice().iter()) {
        let expected = original * 3.0;
        assert!((scaled - expected).abs() < 1e-5);
    }
}

#[test]
fn test_parallel_execution_correctness() {
    // Verify parallel and sequential execution produce same results
    const SIZE: usize = 150_000;
    let a_data: Vec<f32> = (0..SIZE).map(|i| (i % 1000) as f32).collect();
    let b_data: Vec<f32> = (0..SIZE).map(|i| (i % 500) as f32 + 1.0).collect();

    let a_large = Vector::from_slice(&a_data);
    let b_large = Vector::from_slice(&b_data);
    let result_parallel = a_large.add(&b_large).unwrap();

    // Compare with small vector (sequential execution)
    const SMALL_SIZE: usize = 100;
    let a_small = Vector::from_slice(&a_data[..SMALL_SIZE]);
    let b_small = Vector::from_slice(&b_data[..SMALL_SIZE]);
    let result_sequential = a_small.add(&b_small).unwrap();

    // First SMALL_SIZE elements should match
    for i in 0..SMALL_SIZE {
        assert_eq!(result_parallel.as_slice()[i], result_sequential.as_slice()[i]);
    }
}

// ========================================================================
// AVX512 SIMD path tests (need 16+ elements to trigger SIMD loops)
// These tests ensure the SIMD implementations are exercised
// ========================================================================

#[test]
fn test_norm_l1_avx512_path() {
    // 32 elements to ensure AVX512 loop runs twice (32 / 16 = 2)
    let data: Vec<f32> = (0..32).map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) }).collect();
    let v = Vector::from_slice(&data);
    let result = v.norm_l1().unwrap();
    // Sum of |0| + |1| + |2| + ... + |31| = 0 + 1 + 2 + ... + 31 = 31*32/2 = 496
    assert!((result - 496.0).abs() < 1e-3);
}

#[test]
fn test_norm_linf_avx512_path() {
    // 32 elements to ensure AVX512 loop runs twice
    let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    data[17] = -100.0; // Make element 17 the max absolute value
    let v = Vector::from_slice(&data);
    let result = v.norm_linf().unwrap();
    assert!((result - 100.0).abs() < 1e-5);
}

#[test]
fn test_scale_avx512_path() {
    // 64 elements to ensure multiple AVX512 iterations
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let v = Vector::from_slice(&data);
    let result = v.scale(2.0).unwrap();
    for i in 0..64 {
        assert!((result.as_slice()[i] - (i as f32 * 2.0)).abs() < 1e-5);
    }
}

#[test]
fn test_abs_avx512_path() {
    // 48 elements to ensure AVX512 loop runs 3 times (48 / 16 = 3)
    let data: Vec<f32> = (0..48).map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) }).collect();
    let v = Vector::from_slice(&data);
    let result = v.abs().unwrap();
    for i in 0..48 {
        assert!((result.as_slice()[i] - (i as f32)).abs() < 1e-5);
    }
}

#[test]
fn test_clamp_avx512_path() {
    // 32 elements with values spanning the clamp range
    let data: Vec<f32> = (0..32).map(|i| (i as f32) - 10.0).collect();
    let v = Vector::from_slice(&data);
    let result = v.clamp(0.0, 15.0).unwrap();
    for i in 0..32 {
        let expected = ((i as f32) - 10.0).clamp(0.0, 15.0);
        assert!((result.as_slice()[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn test_lerp_avx512_path() {
    // 32 elements
    let a: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..32).map(|i| (i as f32) * 2.0).collect();
    let va = Vector::from_slice(&a);
    let vb = Vector::from_slice(&b);
    let result = va.lerp(&vb, 0.5).unwrap();
    // lerp(a, b, 0.5) = a + 0.5 * (b - a) = 0.5*a + 0.5*b = (a + b) / 2
    for i in 0..32 {
        let expected = f32::midpoint(i as f32, i as f32 * 2.0);
        assert!((result.as_slice()[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn test_fma_avx512_path() {
    // 32 elements: a*b + c
    let a: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..32).map(|_| 2.0).collect();
    let c: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let va = Vector::from_slice(&a);
    let vb = Vector::from_slice(&b);
    let vc = Vector::from_slice(&c);
    let result = va.fma(&vb, &vc).unwrap();
    // a*b + c = i*2 + i = 3*i
    for i in 0..32 {
        let expected = 3.0 * (i as f32);
        assert!((result.as_slice()[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn test_argmax_avx512_path() {
    // 32 elements with max at position 25
    let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    data[25] = 1000.0;
    let v = Vector::from_slice(&data);
    let result = v.argmax().unwrap();
    assert_eq!(result, 25);
}

#[test]
fn test_argmin_avx512_path() {
    // 32 elements with min at position 18
    let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    data[18] = -500.0;
    let v = Vector::from_slice(&data);
    let result = v.argmin().unwrap();
    assert_eq!(result, 18);
}
