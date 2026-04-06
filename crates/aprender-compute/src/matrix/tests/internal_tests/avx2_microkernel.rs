use super::super::*;

// ===== Phase 2 Micro-kernel Tests (Issue #10) =====

#[test]
#[cfg(target_arch = "x86_64")]
fn test_horizontal_sum_avx2() {
    // Test the AVX2 horizontal sum helper function
    if !is_x86_feature_detected!("avx2") {
        println!("Skipping AVX2 horizontal sum test (CPU doesn't support AVX2)");
        return;
    }

    use std::arch::x86_64::*;

    // SAFETY: CPU feature verified at runtime, slices bounds-checked
    unsafe {
        // Test case 1: All ones
        let v = _mm256_set1_ps(1.0);
        let sum = Matrix::<f32>::horizontal_sum_avx2(v);
        assert!((sum - 8.0).abs() < 1e-6, "Expected 8.0, got {}", sum);

        // Test case 2: Sequence 1..8
        let v = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let sum = Matrix::<f32>::horizontal_sum_avx2(v);
        assert!((sum - 36.0).abs() < 1e-6, "Expected 36.0, got {}", sum);

        // Test case 3: Alternating signs
        let v = _mm256_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
        let sum = Matrix::<f32>::horizontal_sum_avx2(v);
        assert!(sum.abs() < 1e-6, "Expected ~0.0, got {}", sum);

        // Test case 4: Large values
        let v = _mm256_setr_ps(100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0);
        let sum = Matrix::<f32>::horizontal_sum_avx2(v);
        assert!((sum - 3600.0).abs() < 1e-3, "Expected 3600.0, got {}", sum);

        // Test case 5: Mixed positive/negative
        let v = _mm256_setr_ps(10.5, -5.25, 3.75, -8.0, 12.0, -6.5, 4.25, -2.75);
        let expected = 10.5 - 5.25 + 3.75 - 8.0 + 12.0 - 6.5 + 4.25 - 2.75;
        let sum = Matrix::<f32>::horizontal_sum_avx2(v);
        assert!((sum - expected).abs() < 1e-5, "Expected {}, got {}", expected, sum);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_matmul_microkernel_4x1_avx2() {
    // Test the 4x1 AVX2 micro-kernel for matrix multiplication
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        println!("Skipping AVX2 micro-kernel test (CPU doesn't support AVX2/FMA)");
        return;
    }

    // Test case 1: Simple dot products
    // A rows: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    // B col:  [1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1]
    // Expected: Row sums
    {
        let row0: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let row1: Vec<f32> = (17..=32).map(|x| x as f32).collect();
        let row2: Vec<f32> = (33..=48).map(|x| x as f32).collect();
        let row3: Vec<f32> = (49..=64).map(|x| x as f32).collect();
        let b_col = vec![1.0f32; 16];

        let a_rows = [row0.as_slice(), row1.as_slice(), row2.as_slice(), row3.as_slice()];
        let mut results = [0.0f32; 4];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_4x1_avx2(a_rows, &b_col, &mut results);
        }

        // Expected: sum(1..16), sum(17..32), sum(33..48), sum(49..64)
        let expected = [
            (1..=16).sum::<i32>() as f32,
            (17..=32).sum::<i32>() as f32,
            (33..=48).sum::<i32>() as f32,
            (49..=64).sum::<i32>() as f32,
        ];

        for i in 0..4 {
            assert!(
                (results[i] - expected[i]).abs() < 1e-3,
                "Row {}: expected {}, got {}",
                i,
                expected[i],
                results[i]
            );
        }
    }

    // Test case 2: Identity-like pattern
    // Each row is all zeros except one 1.0
    {
        let row0 =
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let row1 =
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let row2 =
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let row3 =
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b_col: Vec<f32> = (1..=16).map(|x| x as f32).collect();

        let a_rows = [row0.as_slice(), row1.as_slice(), row2.as_slice(), row3.as_slice()];
        let mut results = [0.0f32; 4];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_4x1_avx2(a_rows, &b_col, &mut results);
        }

        // Expected: Each result picks one element from b_col
        let expected = [1.0, 2.0, 3.0, 4.0];
        for i in 0..4 {
            assert!(
                (results[i] - expected[i]).abs() < 1e-6,
                "Row {}: expected {}, got {}",
                i,
                expected[i],
                results[i]
            );
        }
    }

    // Test case 3: Non-aligned size (not multiple of 8)
    // Size 10 (8 + 2 remainder)
    {
        let row0: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let row1: Vec<f32> = (11..=20).map(|x| x as f32).collect();
        let row2: Vec<f32> = (21..=30).map(|x| x as f32).collect();
        let row3: Vec<f32> = (31..=40).map(|x| x as f32).collect();
        let b_col = vec![2.0f32; 10];

        let a_rows = [row0.as_slice(), row1.as_slice(), row2.as_slice(), row3.as_slice()];
        let mut results = [0.0f32; 4];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_4x1_avx2(a_rows, &b_col, &mut results);
        }

        // Expected: 2x each row sum
        let expected = [
            2.0 * (1..=10).sum::<i32>() as f32,
            2.0 * (11..=20).sum::<i32>() as f32,
            2.0 * (21..=30).sum::<i32>() as f32,
            2.0 * (31..=40).sum::<i32>() as f32,
        ];

        for i in 0..4 {
            assert!(
                (results[i] - expected[i]).abs() < 1e-3,
                "Row {}: expected {}, got {}",
                i,
                expected[i],
                results[i]
            );
        }
    }

    // Test case 4: Mixed positive/negative values
    {
        let row0 = vec![
            1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0, 13.0, -14.0, 15.0,
            -16.0,
        ];
        let row1 = vec![
            2.0, -4.0, 6.0, -8.0, 10.0, -12.0, 14.0, -16.0, 18.0, -20.0, 22.0, -24.0, 26.0, -28.0,
            30.0, -32.0,
        ];
        let row2 = vec![
            0.5, -1.0, 1.5, -2.0, 2.5, -3.0, 3.5, -4.0, 4.5, -5.0, 5.5, -6.0, 6.5, -7.0, 7.5, -8.0,
        ];
        let row3 = vec![
            10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0,
            -10.0, 10.0, -10.0,
        ];
        let b_col =
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let a_rows = [row0.as_slice(), row1.as_slice(), row2.as_slice(), row3.as_slice()];
        let mut results = [0.0f32; 4];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_4x1_avx2(a_rows, &b_col, &mut results);
        }

        // Compute expected manually
        let expected = [
            row0.iter().sum::<f32>(),
            row1.iter().sum::<f32>(),
            row2.iter().sum::<f32>(),
            row3.iter().sum::<f32>(),
        ];

        for i in 0..4 {
            assert!(
                (results[i] - expected[i]).abs() < 1e-4,
                "Row {}: expected {}, got {}",
                i,
                expected[i],
                results[i]
            );
        }
    }

    // Test case 5: Zero accumulation
    {
        let row0 = vec![0.0f32; 16];
        let row1 = vec![0.0f32; 16];
        let row2 = vec![0.0f32; 16];
        let row3 = vec![0.0f32; 16];
        let b_col: Vec<f32> = (1..=16).map(|x| x as f32).collect();

        let a_rows = [row0.as_slice(), row1.as_slice(), row2.as_slice(), row3.as_slice()];
        let mut results = [0.0f32; 4];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_4x1_avx2(a_rows, &b_col, &mut results);
        }

        for (i, &result) in results.iter().enumerate() {
            assert!(result.abs() < 1e-6, "Row {}: expected 0.0, got {}", i, result);
        }
    }

    // Test case 6: Verify FMA correctness (a * b + c pattern)
    // Micro-kernel computes: sum(a[i] * b[i])
    {
        let row0 = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let row1 = vec![
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
            32.0,
        ];
        let row2 =
            vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0];
        let row3 = vec![
            3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0, 39.0, 42.0, 45.0,
            48.0,
        ];
        let b_col =
            vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

        let a_rows = [row0.as_slice(), row1.as_slice(), row2.as_slice(), row3.as_slice()];
        let mut results = [0.0f32; 4];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_4x1_avx2(a_rows, &b_col, &mut results);
        }

        // Expected: 0.5 x each row sum
        let expected = [
            0.5 * row0.iter().sum::<f32>(),
            0.5 * row1.iter().sum::<f32>(),
            0.5 * row2.iter().sum::<f32>(),
            0.5 * row3.iter().sum::<f32>(),
        ];

        for i in 0..4 {
            assert!(
                (results[i] - expected[i]).abs() < 1e-3,
                "Row {}: expected {}, got {}",
                i,
                expected[i],
                results[i]
            );
        }
    }
}
