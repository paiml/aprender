use super::*;

#[test]
fn test_matvec_all_fallback_backends() {
    let mat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let vec_data = [1.0, 2.0, 3.0];
    let expected = [14.0, 32.0];
    for &backend in
        &[Backend::Scalar, Backend::NEON, Backend::WasmSIMD, Backend::GPU, Backend::Auto]
    {
        assert_matvec_backend(
            2,
            3,
            mat.clone(),
            &vec_data,
            &expected,
            backend,
            1e-6,
            &format!("matvec {backend:?}"),
        );
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_matvec_simd_backends() {
    let mat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let vec_data = [1.0, 2.0, 3.0];
    let expected = [14.0, 32.0];
    for &backend in &[Backend::SSE2, Backend::AVX] {
        assert_matvec_backend(
            2,
            3,
            mat.clone(),
            &vec_data,
            &expected,
            backend,
            1e-6,
            &format!("matvec {backend:?}"),
        );
    }
    if is_x86_feature_detected!("avx2") {
        for &backend in &[Backend::AVX2, Backend::AVX512] {
            assert_matvec_backend(
                2,
                3,
                mat.clone(),
                &vec_data,
                &expected,
                backend,
                1e-6,
                &format!("matvec {backend:?}"),
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_matvec_backend_equivalence() {
    let rows = 4;
    let cols = 16;
    let mat_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1).collect();
    let vec_data: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.5 + 1.0).collect();

    let m_scalar = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), Backend::Scalar);
    let v = Vector::from_slice(&vec_data);
    let expected = m_scalar.matvec(&v).unwrap();

    for &backend in &[Backend::SSE2, Backend::AVX] {
        let m = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), backend);
        let result = m.matvec(&v).unwrap();
        for (i, (&got, &exp)) in
            result.as_slice().iter().zip(expected.as_slice().iter()).enumerate()
        {
            assert!(
                (got - exp).abs() < 1e-4,
                "Scalar vs {backend:?} mismatch at [{i}]: {got} vs {exp}",
            );
        }
    }

    if is_x86_feature_detected!("avx2") {
        for &backend in &[Backend::AVX2, Backend::AVX512] {
            let m = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), backend);
            let result = m.matvec(&v).unwrap();
            for (i, (&got, &exp)) in
                result.as_slice().iter().zip(expected.as_slice().iter()).enumerate()
            {
                assert!(
                    (got - exp).abs() < 1e-4,
                    "Scalar vs {backend:?} mismatch at [{i}]: {got} vs {exp}",
                );
            }
        }
    }
}

#[test]
fn test_matvec_non_aligned_dimensions() {
    let rows = 3;
    let cols = 7;
    let mat_data: Vec<f32> = (0..rows * cols).map(|i| (i + 1) as f32).collect();
    let vec_data: Vec<f32> = (0..cols).map(|i| (i + 1) as f32).collect();

    for &backend in
        &[Backend::Scalar, Backend::GPU, Backend::Auto, Backend::NEON, Backend::WasmSIMD]
    {
        let m = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), backend);
        let v = Vector::from_slice(&vec_data);
        let result = m.matvec(&v).unwrap();
        assert_eq!(result.as_slice().len(), rows, "non-aligned {backend:?}");
        assert!(
            (result.as_slice()[0] - 140.0).abs() < 1e-3,
            "non-aligned {backend:?} row 0: got {}",
            result.as_slice()[0]
        );
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_matvec_non_aligned_simd_backends() {
    let rows = 5;
    let cols = 13;
    let mat_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.3).collect();
    let vec_data: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.7 + 0.1).collect();

    let m_scalar = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), Backend::Scalar);
    let v = Vector::from_slice(&vec_data);
    let expected = m_scalar.matvec(&v).unwrap();

    for &backend in &[Backend::SSE2, Backend::AVX] {
        let m = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), backend);
        let result = m.matvec(&v).unwrap();
        for (i, (&got, &exp)) in
            result.as_slice().iter().zip(expected.as_slice().iter()).enumerate()
        {
            assert!(
                (got - exp).abs() < 1e-3,
                "non-aligned Scalar vs {backend:?} at [{i}]: {got} vs {exp}",
            );
        }
    }

    if is_x86_feature_detected!("avx2") {
        let m_avx2 = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), Backend::AVX2);
        let result = m_avx2.matvec(&v).unwrap();
        for (i, (&got, &exp)) in
            result.as_slice().iter().zip(expected.as_slice().iter()).enumerate()
        {
            assert!(
                (got - exp).abs() < 1e-3,
                "non-aligned Scalar vs AVX2 at [{i}]: {got} vs {exp}",
            );
        }
    }
}

#[test]
fn test_matvec_large_matrix_all_backends() {
    let rows = 10;
    let cols = 64;
    let mat_data: Vec<f32> = (0..rows * cols).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect();
    let vec_data: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32) * 0.2 - 1.2).collect();

    let m_scalar = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), Backend::Scalar);
    let v = Vector::from_slice(&vec_data);
    let expected = m_scalar.matvec(&v).unwrap();

    for backend in [Backend::GPU, Backend::Auto, Backend::NEON, Backend::WasmSIMD] {
        let m = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), backend);
        let result = m.matvec(&v).unwrap();
        for (i, (&got, &exp)) in
            result.as_slice().iter().zip(expected.as_slice().iter()).enumerate()
        {
            assert!(
                (got - exp).abs() < 1e-3,
                "large Scalar vs {backend:?} at [{i}]: {got} vs {exp}",
            );
        }
    }
}
