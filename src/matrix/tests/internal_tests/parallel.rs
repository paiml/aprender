use super::super::*;

#[test]
#[cfg(feature = "parallel")]
fn test_matmul_parallel_1024() {
    // Phase 4: Test parallel matmul for 1024x1024 matrices
    // This triggers the parallel path (PARALLEL_THRESHOLD = 1024)
    let size = 1024;
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
    let mut result_parallel = Matrix::zeros(size, size);

    a.matmul_naive(&b, &mut result_naive).unwrap();
    a.matmul_simd(&b, &mut result_parallel).unwrap(); // Uses parallel path with 'parallel' feature

    // Verify correctness with appropriate tolerance
    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for i in 0..size {
        for j in 0..size {
            let naive_val = result_naive.get(i, j).unwrap();
            let parallel_val = result_parallel.get(i, j).unwrap();
            let diff = (naive_val - parallel_val).abs();
            let tolerance = if naive_val.abs() > 1.0 { naive_val.abs() * 1e-2 } else { 1e-2 };
            if diff >= tolerance {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!(
                        "Mismatch at ({}, {}): naive={}, parallel={}, diff={}, tolerance={}",
                        i, j, naive_val, parallel_val, diff, tolerance
                    );
                }
            }
            max_diff = max_diff.max(diff);
        }
    }
    assert_eq!(
        mismatches, 0,
        "Found {} mismatches in {}x{} parallel matmul, max_diff={}",
        mismatches, size, size, max_diff
    );
}

#[test]
#[cfg(feature = "parallel")]
fn test_matvec_parallel_4096() {
    // Test parallel matvec for very large matrices (>=4096 rows)
    // This triggers the parallel path (PARALLEL_THRESHOLD = 4096)
    let rows = 4096;
    let cols = 512;

    let matrix =
        Matrix::from_vec(rows, cols, (0..rows * cols).map(|i| ((i % 100) as f32) / 10.0).collect())
            .unwrap();

    let vector =
        Vector::from_slice(&(0..cols).map(|i| ((i % 50) as f32) / 5.0).collect::<Vec<f32>>());

    // Compute result (should use parallel path)
    let result = matrix.matvec(&vector).unwrap();

    // Verify result shape
    assert_eq!(result.len(), rows);

    // Verify correctness by comparing with manual dot product calculation
    // Check a few sample rows
    for sample_row in [0, 1024, 2048, 3072, 4095] {
        let row_start = sample_row * cols;
        let row = &matrix.data[row_start..(row_start + cols)];

        // Manual dot product
        let expected: f32 = row.iter().zip(vector.as_slice().iter()).map(|(a, b)| a * b).sum();

        let actual = result.as_slice()[sample_row];
        let diff = (expected - actual).abs();
        let tolerance = if expected.abs() > 1.0 { expected.abs() * 1e-3 } else { 1e-3 };

        assert!(
            diff < tolerance,
            "Mismatch at row {}: expected={}, actual={}, diff={}",
            sample_row,
            expected,
            actual,
            diff
        );
    }
}
