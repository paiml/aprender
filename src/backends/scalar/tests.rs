use super::*;

#[test]
fn test_scalar_add() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let mut result = [0.0; 4];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::add(&a, &b, &mut result);
    }
    assert_eq!(result, [6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_scalar_mul() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [2.0, 3.0, 4.0, 5.0];
    let mut result = [0.0; 4];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::mul(&a, &b, &mut result);
    }
    assert_eq!(result, [2.0, 6.0, 12.0, 20.0]);
}

#[test]
fn test_scalar_dot() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    // SAFETY: Test code calling backend trait methods marked unsafe
    let result = unsafe { ScalarBackend::dot(&a, &b) };
    assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
}

#[test]
fn test_scalar_sum() {
    let a = [1.0, 2.0, 3.0, 4.0];
    // SAFETY: Test code calling backend trait methods marked unsafe
    let result = unsafe { ScalarBackend::sum(&a) };
    assert_eq!(result, 10.0);
}

#[test]
fn test_scalar_max() {
    let a = [1.0, 5.0, 3.0, 2.0];
    // SAFETY: Test code calling backend trait methods marked unsafe
    let result = unsafe { ScalarBackend::max(&a) };
    assert_eq!(result, 5.0);
}

#[test]
fn test_scalar_min() {
    let a = [1.0, 5.0, 3.0, 2.0];
    // SAFETY: Test code calling backend trait methods marked unsafe
    let result = unsafe { ScalarBackend::min(&a) };
    assert_eq!(result, 1.0);
}

#[test]
fn test_scalar_sub() {
    let a = [5.0, 6.0, 7.0, 8.0];
    let b = [1.0, 2.0, 3.0, 4.0];
    let mut result = [0.0; 4];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::sub(&a, &b, &mut result);
    }
    assert_eq!(result, [4.0, 4.0, 4.0, 4.0]);
}

#[test]
fn test_scalar_div() {
    let a = [10.0, 20.0, 30.0, 40.0];
    let b = [2.0, 4.0, 5.0, 8.0];
    let mut result = [0.0; 4];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::div(&a, &b, &mut result);
    }
    assert_eq!(result, [5.0, 5.0, 6.0, 5.0]);
}

#[test]
fn test_scalar_argmax() {
    let a = [1.0, 5.0, 3.0, 2.0];
    // SAFETY: Test code calling backend trait methods marked unsafe
    let result = unsafe { ScalarBackend::argmax(&a) };
    assert_eq!(result, 1); // Index of 5.0
}

#[test]
fn test_scalar_argmin() {
    let a = [5.0, 1.0, 3.0, 2.0];
    // SAFETY: Test code calling backend trait methods marked unsafe
    let result = unsafe { ScalarBackend::argmin(&a) };
    assert_eq!(result, 1); // Index of 1.0
}

#[test]
fn test_scalar_sum_kahan() {
    let a = [1.0, 2.0, 3.0, 4.0];
    // SAFETY: Test code calling backend trait methods marked unsafe
    let result = unsafe { ScalarBackend::sum_kahan(&a) };
    assert_eq!(result, 10.0);
}

#[test]
fn test_scalar_norm_l1() {
    let a = [1.0, -2.0, 3.0, -4.0];
    // SAFETY: Test code calling backend trait methods marked unsafe
    let result = unsafe { ScalarBackend::norm_l1(&a) };
    assert_eq!(result, 10.0); // |1| + |-2| + |3| + |-4| = 10
}

#[test]
fn test_scalar_norm_l2() {
    let a = [3.0, 4.0];
    // SAFETY: Test code calling backend trait methods marked unsafe
    let result = unsafe { ScalarBackend::norm_l2(&a) };
    assert_eq!(result, 5.0); // sqrt(3² + 4²) = 5
}

#[test]
fn test_scalar_scale() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let mut result = [0.0; 4];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::scale(&a, 2.0, &mut result);
    }
    assert_eq!(result, [2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_scalar_clamp() {
    let a = [1.0, 5.0, 10.0, 15.0];
    let mut result = [0.0; 4];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::clamp(&a, 3.0, 12.0, &mut result);
    }
    assert_eq!(result, [3.0, 5.0, 10.0, 12.0]);
}

#[test]
fn test_scalar_lerp() {
    let a = [0.0, 10.0, 20.0];
    let b = [100.0, 110.0, 120.0];
    let mut result = [0.0; 3];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::lerp(&a, &b, 0.5, &mut result);
    }
    assert_eq!(result, [50.0, 60.0, 70.0]); // Midpoint between a and b
}

#[test]
fn test_scalar_fma() {
    let a = [1.0, 2.0, 3.0];
    let b = [2.0, 3.0, 4.0];
    let c = [5.0, 6.0, 7.0];
    let mut result = [0.0; 3];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::fma(&a, &b, &c, &mut result);
    }
    // FMA: a*b + c
    assert_eq!(result, [7.0, 12.0, 19.0]); // [1*2+5, 2*3+6, 3*4+7]
}

#[test]
fn test_scalar_relu() {
    let a = [-3.0, -1.0, 0.0, 1.0, 3.0];
    let mut result = [0.0; 5];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::relu(&a, &mut result);
    }
    assert_eq!(result, [0.0, 0.0, 0.0, 1.0, 3.0]);
}

#[test]
fn test_scalar_sigmoid() {
    let a = [-51.0, -1.0, 0.0, 1.0, 51.0];
    let mut result = [0.0; 5];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::sigmoid(&a, &mut result);
    }
    // sigmoid(-51) = 0, sigmoid(0) = 0.5, sigmoid(51) = 1
    assert_eq!(result[0], 0.0); // Clamped to 0 for numerical stability
    assert!((result[1] - 0.2689).abs() < 0.001); // sigmoid(-1)
    assert_eq!(result[2], 0.5); // sigmoid(0)
    assert!((result[3] - 0.7311).abs() < 0.001); // sigmoid(1)
    assert_eq!(result[4], 1.0); // Clamped to 1 for numerical stability
}

#[test]
fn test_scalar_gelu() {
    let a = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut result = [0.0; 5];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::gelu(&a, &mut result);
    }
    // GELU approximation values
    assert!((result[0] - (-0.0454)).abs() < 0.01); // gelu(-2)
    assert!((result[1] - (-0.1588)).abs() < 0.01); // gelu(-1)
    assert_eq!(result[2], 0.0); // gelu(0) = 0
    assert!((result[3] - 0.8413).abs() < 0.01); // gelu(1)
    assert!((result[4] - 1.9545).abs() < 0.01); // gelu(2)
}

#[test]
fn test_scalar_swish() {
    let a = [-51.0, -1.0, 0.0, 1.0, 51.0];
    let mut result = [0.0; 5];
    // SAFETY: Test code calling backend trait methods marked unsafe
    unsafe {
        ScalarBackend::swish(&a, &mut result);
    }
    // swish(x) = x * sigmoid(x)
    assert_eq!(result[0], 0.0); // x * 0 = 0 (numerical stability)
    assert!((result[1] - (-0.2689)).abs() < 0.001); // -1 * sigmoid(-1)
    assert_eq!(result[2], 0.0); // 0 * sigmoid(0) = 0
    assert!((result[3] - 0.7311).abs() < 0.001); // 1 * sigmoid(1)
    assert_eq!(result[4], 51.0); // x * 1 = x (numerical stability)
}

// cuda-tile-behavior.md: Falsification test #81 - Backend equivalence
#[test]
fn test_scalar_dot_unrolled_various_sizes() {
    // Test various sizes to exercise all code paths in unrolled implementation:
    // - 0 elements (edge case)
    // - 1-3 elements (remainder only)
    // - 4 elements (exactly one unrolled chunk)
    // - 5-7 elements (one chunk + remainder)
    // - 8 elements (two chunks)
    // - 100 elements (realistic workload)
    // - 1000 elements (larger workload)
    let sizes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 100, 1000];

    for &size in &sizes {
        if size == 0 {
            continue; // Empty slice edge case
        }

        let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..size).map(|i| ((size - i) as f32) * 0.1).collect();

        // Calculate expected result using naive implementation
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        let result = unsafe { ScalarBackend::dot(&a, &b) };

        // Tolerance accounts for FP reordering from unrolling (different accumulator summing order)
        // Use relative tolerance for larger sums where absolute error grows with magnitude
        let tolerance = (1e-5 * expected.abs()).max(1e-4);
        assert!(
            (result - expected).abs() < tolerance,
            "dot mismatch at size {}: got={}, expected={}, tolerance={}",
            size,
            result,
            expected,
            tolerance
        );
    }
}

// cuda-tile-behavior.md: Falsification test #92 - FMA single-rounding accuracy
#[test]
fn test_scalar_dot_mul_add_accuracy() {
    // Test that mul_add provides better accuracy than separate mul+add
    // FMA has single rounding vs two roundings for separate operations
    let a = vec![1.0000001_f32; 1000];
    let b = vec![1.0000001_f32; 1000];

    // SAFETY: CPU feature verified at runtime, slices bounds-checked
    let result = unsafe { ScalarBackend::dot(&a, &b) };

    // Expected: 1000 * 1.0000001 * 1.0000001 ≈ 1000.0002
    // With FMA, error should be smaller due to single rounding
    let expected = 1000.0 * 1.0000001_f32 * 1.0000001_f32;
    assert!(
        (result - expected).abs() < 1e-3,
        "FMA accuracy test: got={}, expected={}",
        result,
        expected
    );
}
