use super::super::*;

// ========================================================================
// Phase 1: Scalar Reference Tests
// ========================================================================

#[test]
fn test_gemm_reference_2x2() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];

    gemm_reference(2, 2, 2, &a, &b, &mut c).unwrap();

    // [1 2] * [5 6] = [19 22]
    // [3 4]   [7 8]   [43 50]
    assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_gemm_reference_identity() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut c = vec![0.0; 9];

    gemm_reference(3, 3, 3, &a, &identity, &mut c).unwrap();

    assert_eq!(c, a);
}

#[test]
fn test_gemm_reference_accumulation() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 0.0, 0.0, 1.0];
    let mut c = vec![10.0, 20.0, 30.0, 40.0]; // Pre-existing values

    gemm_reference(2, 2, 2, &a, &b, &mut c).unwrap();

    // C += A * I = C + A
    assert_eq!(c, vec![11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn test_gemm_reference_rectangular() {
    // 2x3 * 3x2 = 2x2
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut c = vec![0.0; 4];

    gemm_reference(2, 2, 3, &a, &b, &mut c).unwrap();

    // [1 2 3] * [7  8 ] = [58  64]
    // [4 5 6]   [9  10]   [139 154]
    //           [11 12]
    assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_gemm_reference_size_mismatch() {
    let a = vec![1.0, 2.0, 3.0]; // Wrong size
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut c = vec![0.0; 4];

    let result = gemm_reference(2, 2, 2, &a, &b, &mut c);
    assert!(result.is_err());
}

#[test]
fn test_gemm_reference_b_size_mismatch() {
    let a = vec![1.0; 4];
    let b = vec![1.0, 2.0]; // Wrong: should be 4
    let mut c = vec![0.0; 4];
    assert!(gemm_reference(2, 2, 2, &a, &b, &mut c).is_err());
}

#[test]
fn test_gemm_reference_c_size_mismatch() {
    let a = vec![1.0; 4];
    let b = vec![1.0; 4];
    let mut c = vec![0.0; 2]; // Wrong: should be 4
    assert!(gemm_reference(2, 2, 2, &a, &b, &mut c).is_err());
}

#[test]
fn test_gemm_reference_with_jidoka_basic() {
    let guard = JidokaGuard::strict();
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];
    gemm_reference_with_jidoka(2, 2, 2, &a, &b, &mut c, &guard).unwrap();
    assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_gemm_reference_with_jidoka_nan_input() {
    let guard = JidokaGuard::strict();
    let a = vec![f32::NAN, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];
    assert!(gemm_reference_with_jidoka(2, 2, 2, &a, &b, &mut c, &guard).is_err());
}

#[test]
fn test_gemm_reference_with_jidoka_inf_input() {
    let guard = JidokaGuard::strict();
    let a = vec![f32::INFINITY, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];
    assert!(gemm_reference_with_jidoka(2, 2, 2, &a, &b, &mut c, &guard).is_err());
}

#[test]
fn test_gemm_reference_with_jidoka_inf_in_b() {
    let guard = JidokaGuard::strict();
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, f32::INFINITY, 7.0, 8.0];
    let mut c = vec![0.0; 4];
    assert!(gemm_reference_with_jidoka(2, 2, 2, &a, &b, &mut c, &guard).is_err());
}

#[test]
fn test_transpose_large_with_remainder() {
    // 10x13 matrix: 8x8 blocks + remainders on both edges
    let rows = 10;
    let cols = 13;
    let a: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    let mut b = vec![0.0; rows * cols];

    transpose(rows, cols, &a, &mut b).unwrap();

    // Verify: b[c * rows + r] == a[r * cols + c]
    for r in 0..rows {
        for c in 0..cols {
            assert_eq!(b[c * rows + r], a[r * cols + c], "transpose mismatch at ({}, {})", r, c);
        }
    }
}

#[test]
fn test_transpose_exact_block_size() {
    // 16x16: exactly 2x2 blocks of 8x8, no remainder
    let rows = 16;
    let cols = 16;
    let a: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    let mut b = vec![0.0; rows * cols];

    transpose(rows, cols, &a, &mut b).unwrap();

    for r in 0..rows {
        for c in 0..cols {
            assert_eq!(b[c * rows + r], a[r * cols + c]);
        }
    }
}

#[test]
fn test_transpose_size_mismatch() {
    let a = vec![1.0; 6];
    let mut b = vec![0.0; 4]; // Wrong size
    assert!(transpose(2, 3, &a, &mut b).is_err());
}

#[test]
fn test_transpose_small_scalar_path() {
    // < 64 elements goes through scalar path
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut b = vec![0.0; 6];
    transpose(2, 3, &a, &mut b).unwrap();
    // Expected: [[1,4],[2,5],[3,6]]
    assert_eq!(b, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}
