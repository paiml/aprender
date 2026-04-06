use super::super::*;

#[test]
#[cfg(target_arch = "x86_64")]
fn test_matmul_microkernel_8x1_avx512() {
    // Test the 8x1 AVX-512 micro-kernel for matrix multiplication (Phase 3)
    if !is_x86_feature_detected!("avx512f") {
        println!("Skipping AVX-512 micro-kernel test (CPU doesn't support AVX-512F)");
        return;
    }

    // Test case 1: Simple dot products with 32 elements (2 AVX-512 iterations)
    {
        let row0: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let row1: Vec<f32> = (33..=64).map(|x| x as f32).collect();
        let row2: Vec<f32> = (65..=96).map(|x| x as f32).collect();
        let row3: Vec<f32> = (97..=128).map(|x| x as f32).collect();
        let row4: Vec<f32> = (129..=160).map(|x| x as f32).collect();
        let row5: Vec<f32> = (161..=192).map(|x| x as f32).collect();
        let row6: Vec<f32> = (193..=224).map(|x| x as f32).collect();
        let row7: Vec<f32> = (225..=256).map(|x| x as f32).collect();
        let b_col = vec![1.0f32; 32];

        let a_rows = [
            row0.as_slice(),
            row1.as_slice(),
            row2.as_slice(),
            row3.as_slice(),
            row4.as_slice(),
            row5.as_slice(),
            row6.as_slice(),
            row7.as_slice(),
        ];
        let mut results = [0.0f32; 8];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_8x1_avx512(a_rows, &b_col, &mut results);
        }

        // Expected: sum of each row
        let expected = [
            (1..=32).sum::<i32>() as f32,
            (33..=64).sum::<i32>() as f32,
            (65..=96).sum::<i32>() as f32,
            (97..=128).sum::<i32>() as f32,
            (129..=160).sum::<i32>() as f32,
            (161..=192).sum::<i32>() as f32,
            (193..=224).sum::<i32>() as f32,
            (225..=256).sum::<i32>() as f32,
        ];

        for i in 0..8 {
            assert!(
                (results[i] - expected[i]).abs() < 1e-2,
                "Row {}: expected {}, got {}",
                i,
                expected[i],
                results[i]
            );
        }
    }

    // Test case 2: Scaled dot products
    {
        let row: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let rows: [&[f32]; 8] = [&row, &row, &row, &row, &row, &row, &row, &row];
        let b_col = vec![0.5f32; 32];
        let mut results = [0.0f32; 8];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_8x1_avx512(rows, &b_col, &mut results);
        }

        let expected = 0.5 * (1..=32).sum::<i32>() as f32;
        for (i, &result) in results.iter().enumerate() {
            assert!(
                (result - expected).abs() < 1e-2,
                "Row {}: expected {}, got {}",
                i,
                expected,
                result
            );
        }
    }

    // Test case 3: Zero accumulation
    {
        let zeros = vec![0.0f32; 32];
        let rows: [&[f32]; 8] = [&zeros, &zeros, &zeros, &zeros, &zeros, &zeros, &zeros, &zeros];
        let b_col: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let mut results = [0.0f32; 8];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_8x1_avx512(rows, &b_col, &mut results);
        }

        for (i, &result) in results.iter().enumerate() {
            assert!(result.abs() < 1e-6, "Row {}: expected 0.0, got {}", i, result);
        }
    }
}

// ===== AVX-512 Full Matmul Test =====

#[test]
#[cfg(target_arch = "x86_64")]
fn test_matmul_avx512_backend_large_matrix() {
    // Test full matmul with AVX-512 backend on large matrices
    // This exercises the AVX-512 code path in matmul_simd
    if !is_x86_feature_detected!("avx512f") {
        println!("Skipping AVX-512 matmul test (CPU doesn't support AVX-512F)");
        return;
    }

    // Create large matrices that will trigger the SIMD optimization path
    // Using 256x256 to exercise 3-level cache blocking
    let size = 256;
    let a_data: Vec<f32> = (0..size * size).map(|i| (i % 10) as f32).collect();
    let b_data: Vec<f32> = (0..size * size).map(|i| ((i + 5) % 10) as f32).collect();

    let a = Matrix::from_vec_with_backend(size, size, a_data, Backend::AVX512);
    let b = Matrix::from_vec_with_backend(size, size, b_data, Backend::AVX512);

    let result = a.matmul(&b).expect("matmul should succeed");

    // Verify result dimensions
    assert_eq!(result.rows, size);
    assert_eq!(result.cols, size);

    // Spot check a few values against scalar reference
    let a_ref = Matrix::from_vec(size, size, (0..size * size).map(|i| (i % 10) as f32).collect())
        .expect("valid data");
    let b_ref =
        Matrix::from_vec(size, size, (0..size * size).map(|i| ((i + 5) % 10) as f32).collect())
            .expect("valid data");
    let expected = a_ref.matmul(&b_ref).expect("reference matmul should succeed");

    // Check first few and last few elements
    for i in 0..5 {
        for j in 0..5 {
            let diff = (result[(i, j)] - expected[(i, j)]).abs();
            assert!(
                diff < 1e-3,
                "Mismatch at ({}, {}): AVX512={}, scalar={}",
                i,
                j,
                result[(i, j)],
                expected[(i, j)]
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_matmul_avx512_remainder_handling() {
    // Test AVX-512 matmul with non-aligned matrix sizes to exercise remainder code
    if !is_x86_feature_detected!("avx512f") {
        return;
    }

    // Size not divisible by 8 or 16 to exercise remainder handling
    let size = 67;
    let a_data: Vec<f32> = (0..size * size).map(|i| i as f32 * 0.01).collect();
    let b_data: Vec<f32> = (0..size * size).map(|i| i as f32 * 0.01 + 0.5).collect();

    let a = Matrix::from_vec_with_backend(size, size, a_data.clone(), Backend::AVX512);
    let b = Matrix::from_vec_with_backend(size, size, b_data.clone(), Backend::AVX512);

    let result = a.matmul(&b).expect("matmul should succeed");

    // Compare with scalar
    let a_scalar = Matrix::from_vec_with_backend(size, size, a_data, Backend::Scalar);
    let b_scalar = Matrix::from_vec_with_backend(size, size, b_data, Backend::Scalar);
    let expected = a_scalar.matmul(&b_scalar).expect("scalar matmul should succeed");

    for i in 0..size {
        for j in 0..size {
            let diff = (result[(i, j)] - expected[(i, j)]).abs();
            let max_val = expected[(i, j)].abs().max(1.0);
            assert!(
                diff / max_val < 1e-4,
                "Mismatch at ({}, {}): AVX512={}, scalar={}",
                i,
                j,
                result[(i, j)],
                expected[(i, j)]
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_matmul_avx512_l3_blocking() {
    // Test AVX-512 matmul with L3 blocking (requires size >= 512)
    // This exercises the L3 cache blocking path with AVX-512 8x1 micro-kernel
    if !is_x86_feature_detected!("avx512f") {
        println!("Skipping AVX-512 L3 blocking test (CPU doesn't support AVX-512F)");
        return;
    }

    // Size must be >= 512 to trigger L3 blocking path
    // Use 520 to also exercise remainder handling (520 = 512 + 8, 520 % 16 != 0)
    let size = 520;
    let a_data: Vec<f32> = (0..size * size).map(|i| (i % 7) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..size * size).map(|i| ((i + 3) % 11) as f32 * 0.1).collect();

    let a = Matrix::from_vec_with_backend(size, size, a_data.clone(), Backend::AVX512);
    let b = Matrix::from_vec_with_backend(size, size, b_data.clone(), Backend::AVX512);

    let result = a.matmul(&b).expect("AVX-512 L3 blocking matmul should succeed");

    // Verify dimensions
    assert_eq!(result.rows, size);
    assert_eq!(result.cols, size);

    // Compare with scalar reference
    let a_scalar = Matrix::from_vec_with_backend(size, size, a_data, Backend::Scalar);
    let b_scalar = Matrix::from_vec_with_backend(size, size, b_data, Backend::Scalar);
    let expected = a_scalar.matmul(&b_scalar).expect("scalar matmul should succeed");

    // Check corners and some middle elements
    let check_indices = [
        (0, 0),
        (0, size - 1),
        (size - 1, 0),
        (size - 1, size - 1),
        (size / 2, size / 2),
        (8, 8),   // Near 8x1 microkernel boundary
        (15, 15), // Near 16 element boundary
    ];
    for &(i, j) in &check_indices {
        let diff = (result[(i, j)] - expected[(i, j)]).abs();
        let max_val = expected[(i, j)].abs().max(1.0);
        assert!(
            diff / max_val < 1e-3,
            "Mismatch at ({}, {}): AVX512={}, scalar={}, diff={}",
            i,
            j,
            result[(i, j)],
            expected[(i, j)],
            diff
        );
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_matmul_avx512_l3_nonaligned_cols() {
    // Test L3 blocking with column count that exercises remainder in 8x1 microkernel
    // k-dimension (inner loop) not divisible by 16 triggers remainder handling
    if !is_x86_feature_detected!("avx512f") {
        return;
    }

    // 513 columns: 513 = 32*16 + 1, exercises remainder handling
    let rows = 512;
    let cols = 513; // Not divisible by 16
    let a_data: Vec<f32> = (0..rows * cols).map(|i| (i % 13) as f32 * 0.05).collect();
    let b_data: Vec<f32> = (0..cols * rows).map(|i| (i % 17) as f32 * 0.05).collect();

    let a = Matrix::from_vec_with_backend(rows, cols, a_data.clone(), Backend::AVX512);
    let b = Matrix::from_vec_with_backend(cols, rows, b_data.clone(), Backend::AVX512);

    let result = a.matmul(&b).expect("matmul should succeed");
    assert_eq!(result.shape(), (rows, rows));

    // Verify against scalar
    let a_scalar = Matrix::from_vec_with_backend(rows, cols, a_data, Backend::Scalar);
    let b_scalar = Matrix::from_vec_with_backend(cols, rows, b_data, Backend::Scalar);
    let expected = a_scalar.matmul(&b_scalar).expect("scalar matmul");

    // Spot check
    for i in [0, 7, 8, 15, 16, 63, 64, 255, 256, rows - 1] {
        for j in [0, 7, 8, 15, 16, 63, 64, 255, 256, rows - 1] {
            let diff = (result[(i, j)] - expected[(i, j)]).abs();
            assert!(
                diff < 0.1,
                "Mismatch at ({},{}): got={}, expected={}",
                i,
                j,
                result[(i, j)],
                expected[(i, j)]
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_matmul_avx512_l3_row_remainder() {
    // Test AVX-512 L3 blocking with row count that triggers 4x1 AVX2 and scalar remainder
    // 517 rows = 64*8 + 5 = L2_BLOCK*8 + 5, so remainder of 5 rows per L2 block
    // This exercises lines 1216-1252 (AVX2 4x1 for 4 rows, scalar for 1 row)
    if !is_x86_feature_detected!("avx512f") {
        return;
    }

    let rows = 517; // Not divisible by 8 to trigger remainder handling
    let cols = 512;
    let a_data: Vec<f32> = (0..rows * cols).map(|i| (i % 11) as f32 * 0.03).collect();
    let b_data: Vec<f32> = (0..cols * rows).map(|i| (i % 13) as f32 * 0.03).collect();

    let a = Matrix::from_vec_with_backend(rows, cols, a_data.clone(), Backend::AVX512);
    let b = Matrix::from_vec_with_backend(cols, rows, b_data.clone(), Backend::AVX512);

    let result = a.matmul(&b).expect("matmul should succeed");
    assert_eq!(result.shape(), (rows, rows));

    // Verify against scalar reference
    let a_scalar = Matrix::from_vec_with_backend(rows, cols, a_data, Backend::Scalar);
    let b_scalar = Matrix::from_vec_with_backend(cols, rows, b_data, Backend::Scalar);
    let expected = a_scalar.matmul(&b_scalar).expect("scalar matmul");

    // Check scattered points
    for i in [0, 5, 8, 63, 64, 256, 512, rows - 5, rows - 1] {
        for j in [0, 5, 8, 63, 64, 256, 512, rows - 5, rows - 1] {
            if i < rows && j < rows {
                let diff = (result[(i, j)] - expected[(i, j)]).abs();
                assert!(diff < 0.1, "Mismatch at ({},{})", i, j);
            }
        }
    }
}

#[test]
#[cfg(all(target_arch = "x86_64", feature = "parallel"))]
fn test_matmul_avx512_parallel_large() {
    // Test parallel AVX-512 matmul with 1024x1024 to hit parallel L3 blocking path
    // Requires: AVX-512F + parallel feature
    if !is_x86_feature_detected!("avx512f") {
        println!("Skipping: CPU doesn't support AVX-512F");
        return;
    }

    let size = 1024; // Triggers parallel path (PARALLEL_THRESHOLD = 1024)
    let a_data: Vec<f32> = (0..size * size).map(|i| ((i % 10) as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..size * size).map(|i| (((i + 3) % 10) as f32) * 0.1).collect();

    let a = Matrix::from_vec_with_backend(size, size, a_data.clone(), Backend::AVX512);
    let b = Matrix::from_vec_with_backend(size, size, b_data.clone(), Backend::AVX512);

    let result = a.matmul(&b).expect("parallel AVX-512 matmul should succeed");
    assert_eq!(result.shape(), (size, size));

    // Spot check against scalar reference
    let a_scalar = Matrix::from_vec_with_backend(size, size, a_data, Backend::Scalar);
    let b_scalar = Matrix::from_vec_with_backend(size, size, b_data, Backend::Scalar);
    let expected = a_scalar.matmul(&b_scalar).expect("scalar matmul");

    // Check corners
    for (i, j) in [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)] {
        let diff = (result[(i, j)] - expected[(i, j)]).abs();
        let max_val = expected[(i, j)].abs().max(1.0);
        assert!(
            diff / max_val < 0.01,
            "Mismatch at ({},{}): got={}, expected={}",
            i,
            j,
            result[(i, j)],
            expected[(i, j)]
        );
    }
}
