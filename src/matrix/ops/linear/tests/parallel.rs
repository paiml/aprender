#[cfg(feature = "parallel")]
use super::*;

#[test]
#[cfg(feature = "parallel")]
fn test_matvec_parallel_large_matrix() {
    let rows = 4096;
    let cols = 16;
    let mat_data: Vec<f32> = (0..rows * cols).map(|i| ((i % 100) as f32) * 0.01).collect();
    let vec_data: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1 + 1.0).collect();

    let m_scalar = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), Backend::Scalar);
    let v = Vector::from_slice(&vec_data);
    let result = m_scalar.matvec(&v).unwrap();
    assert_eq!(result.as_slice().len(), rows);

    let row0: f32 = (0..cols).map(|j| mat_data[j] * vec_data[j]).sum();
    assert!(
        (result.as_slice()[0] - row0).abs() < 1e-2,
        "parallel row 0: got {} expected {row0}",
        result.as_slice()[0]
    );
}

#[test]
#[cfg(feature = "parallel")]
fn test_matvec_parallel_with_simd_backends() {
    let rows = 4096;
    let cols = 32;
    let mat_data: Vec<f32> = (0..rows * cols).map(|i| ((i % 50) as f32) * 0.02 - 0.5).collect();
    let vec_data: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1).collect();
    let v = Vector::from_slice(&vec_data);

    let m_scalar = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), Backend::Scalar);
    let expected = m_scalar.matvec(&v).unwrap();

    #[cfg(target_arch = "x86_64")]
    {
        let m_sse = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), Backend::SSE2);
        let result_sse = m_sse.matvec(&v).unwrap();
        for (i, (&got, &exp)) in
            result_sse.as_slice().iter().zip(expected.as_slice().iter()).enumerate()
        {
            assert!((got - exp).abs() < 1e-2, "parallel Scalar vs SSE2 at [{i}]: {got} vs {exp}",);
        }

        if is_x86_feature_detected!("avx2") {
            let m_avx2 = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), Backend::AVX2);
            let result_avx2 = m_avx2.matvec(&v).unwrap();
            for (i, (&got, &exp)) in
                result_avx2.as_slice().iter().zip(expected.as_slice().iter()).enumerate()
            {
                assert!(
                    (got - exp).abs() < 1e-2,
                    "parallel Scalar vs AVX2 at [{i}]: {got} vs {exp}",
                );
            }
        }
    }
}

#[test]
#[cfg(feature = "parallel")]
fn test_matvec_parallel_boundary() {
    let cols = 4;
    let vec_data = vec![1.0; cols];
    let v = Vector::from_slice(&vec_data);

    // Just below the threshold (4095 rows) - should NOT hit parallel path
    let rows = 4095;
    let mat_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001).collect();
    let m = Matrix::from_vec_with_backend(rows, cols, mat_data, Backend::Scalar);
    let result = m.matvec(&v).unwrap();
    assert_eq!(result.as_slice().len(), rows);

    // Exactly at threshold (4096 rows) - should hit parallel path
    let rows = 4096;
    let mat_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001).collect();
    let m = Matrix::from_vec_with_backend(rows, cols, mat_data, Backend::Scalar);
    let result = m.matvec(&v).unwrap();
    assert_eq!(result.as_slice().len(), rows);
}
