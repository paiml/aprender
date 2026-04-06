use super::super::*;

#[test]
fn test_matmul_blocking_small_matrices() {
    // Small matrices (<=32) should use simple path (no blocking overhead)
    let sizes = vec![8, 16, 32];
    for size in sizes {
        let a = Matrix::from_vec(size, size, (0..size * size).map(|i| i as f32).collect()).unwrap();
        let b = Matrix::from_vec(size, size, (0..size * size).map(|i| (i * 2) as f32).collect())
            .unwrap();

        let mut result_naive = Matrix::zeros(size, size);
        let mut result_simd = Matrix::zeros(size, size);

        a.matmul_naive(&b, &mut result_naive).unwrap();
        a.matmul_simd(&b, &mut result_simd).unwrap();

        // Verify correctness
        for i in 0..size {
            for j in 0..size {
                let naive_val = result_naive.get(i, j).unwrap();
                let simd_val = result_simd.get(i, j).unwrap();
                let diff = (naive_val - simd_val).abs();
                let tolerance = if naive_val.abs() > 1.0 { naive_val.abs() * 1e-4 } else { 1e-4 };
                assert!(
                    diff < tolerance,
                    "Size {}: Mismatch at ({}, {}): naive={}, simd={}, diff={}",
                    size,
                    i,
                    j,
                    naive_val,
                    simd_val,
                    diff
                );
            }
        }
    }
}

#[test]
fn test_matmul_blocking_medium_matrices() {
    // Medium matrices (>32, <512) should benefit from L2 blocking
    let sizes = vec![64, 128, 256];
    for size in sizes {
        let a = Matrix::from_vec(size, size, (0..size * size).map(|i| (i % 100) as f32).collect())
            .unwrap();
        let b = Matrix::from_vec(
            size,
            size,
            (0..size * size).map(|i| ((i * 3) % 100) as f32).collect(),
        )
        .unwrap();

        let mut result_naive = Matrix::zeros(size, size);
        let mut result_simd = Matrix::zeros(size, size);

        a.matmul_naive(&b, &mut result_naive).unwrap();
        a.matmul_simd(&b, &mut result_simd).unwrap();

        // Verify correctness with relative tolerance for large accumulated values
        for i in 0..size {
            for j in 0..size {
                let naive_val = result_naive.get(i, j).unwrap();
                let simd_val = result_simd.get(i, j).unwrap();
                let diff = (naive_val - simd_val).abs();
                let tolerance = if naive_val.abs() > 1.0 {
                    naive_val.abs() * 1e-3 // More relaxed for large values
                } else {
                    1e-3
                };
                assert!(
                    diff < tolerance,
                    "Size {}: Mismatch at ({}, {}): naive={}, simd={}, diff={}",
                    size,
                    i,
                    j,
                    naive_val,
                    simd_val,
                    diff
                );
            }
        }
    }
}

#[test]
fn test_matmul_blocking_non_aligned_sizes() {
    // Test matrices with sizes not aligned to block boundaries
    let test_cases = vec![
        (33, 33, 33),    // Just over small threshold
        (65, 65, 65),    // Just over L2 block size
        (100, 100, 100), // Middle of L2 block
        (127, 127, 127), // Just under 2x L2 block size
    ];

    for (m, k, n) in test_cases {
        let a = Matrix::from_vec(m, k, (0..m * k).map(|i| (i % 50) as f32).collect()).unwrap();
        let b =
            Matrix::from_vec(k, n, (0..k * n).map(|i| ((i * 2) % 50) as f32).collect()).unwrap();

        let mut result_naive = Matrix::zeros(m, n);
        let mut result_simd = Matrix::zeros(m, n);

        a.matmul_naive(&b, &mut result_naive).unwrap();
        a.matmul_simd(&b, &mut result_simd).unwrap();

        // Verify correctness
        for i in 0..m {
            for j in 0..n {
                let naive_val = result_naive.get(i, j).unwrap();
                let simd_val = result_simd.get(i, j).unwrap();
                let diff = (naive_val - simd_val).abs();
                let tolerance = if naive_val.abs() > 1.0 { naive_val.abs() * 1e-3 } else { 1e-3 };
                assert!(
                    diff < tolerance,
                    "Size {}x{}x{}: Mismatch at ({}, {}): naive={}, simd={}, diff={}",
                    m,
                    k,
                    n,
                    i,
                    j,
                    naive_val,
                    simd_val,
                    diff
                );
            }
        }
    }
}

#[test]
fn test_matmul_blocking_large_matrices() {
    // Large matrix to verify blocking algorithm correctness
    // Keep size manageable for test speed but large enough to trigger blocking
    let size = 256;
    let a =
        Matrix::from_vec(size, size, (0..size * size).map(|i| ((i % 100) as f32) / 10.0).collect())
            .unwrap();
    let b = Matrix::from_vec(
        size,
        size,
        (0..size * size).map(|i| (((i * 7) % 100) as f32) / 10.0).collect(),
    )
    .unwrap();

    let mut result_naive = Matrix::zeros(size, size);
    let mut result_simd = Matrix::zeros(size, size);

    a.matmul_naive(&b, &mut result_naive).unwrap();
    a.matmul_simd(&b, &mut result_simd).unwrap();

    // Verify correctness with appropriate tolerance for accumulated floating-point errors
    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for i in 0..size {
        for j in 0..size {
            let naive_val = result_naive.get(i, j).unwrap();
            let simd_val = result_simd.get(i, j).unwrap();
            let diff = (naive_val - simd_val).abs();
            let tolerance = if naive_val.abs() > 1.0 {
                naive_val.abs() * 1e-2 // Relaxed tolerance for large accumulated values
            } else {
                1e-2
            };
            if diff >= tolerance {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!(
                        "Mismatch at ({}, {}): naive={}, simd={}, diff={}, tolerance={}",
                        i, j, naive_val, simd_val, diff, tolerance
                    );
                }
            }
            max_diff = max_diff.max(diff);
        }
    }
    assert_eq!(
        mismatches, 0,
        "Found {} mismatches in {}x{} matmul, max_diff={}",
        mismatches, size, size, max_diff
    );
}

#[test]
fn test_matmul_3level_blocking() {
    // Phase 3: Test 3-level cache blocking for very large matrices (>=512x512)
    // This test ensures the L3 -> L2 -> micro-kernel hierarchy works correctly
    let size = 512; // Triggers 3-level blocking (L3_THRESHOLD = 512)
    let a =
        Matrix::from_vec(size, size, (0..size * size).map(|i| ((i % 100) as f32) / 10.0).collect())
            .unwrap();
    let b = Matrix::from_vec(
        size,
        size,
        (0..size * size).map(|i| (((i * 7) % 100) as f32) / 10.0).collect(),
    )
    .unwrap();

    let mut result_naive = Matrix::zeros(size, size);
    let mut result_simd = Matrix::zeros(size, size);

    a.matmul_naive(&b, &mut result_naive).unwrap();
    a.matmul_simd(&b, &mut result_simd).unwrap();

    // Verify correctness with appropriate tolerance
    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for i in 0..size {
        for j in 0..size {
            let naive_val = result_naive.get(i, j).unwrap();
            let simd_val = result_simd.get(i, j).unwrap();
            let diff = (naive_val - simd_val).abs();
            let tolerance = if naive_val.abs() > 1.0 { naive_val.abs() * 1e-2 } else { 1e-2 };
            if diff >= tolerance {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!(
                        "Mismatch at ({}, {}): naive={}, simd={}, diff={}, tolerance={}",
                        i, j, naive_val, simd_val, diff, tolerance
                    );
                }
            }
            max_diff = max_diff.max(diff);
        }
    }
    assert_eq!(
        mismatches, 0,
        "Found {} mismatches in {}x{} matmul (3-level blocking), max_diff={}",
        mismatches, size, size, max_diff
    );
}
