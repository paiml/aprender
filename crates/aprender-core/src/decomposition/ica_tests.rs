pub(crate) use super::*;

#[test]
fn test_ica_basic() {
    // Simple 2D case
    let data = Matrix::from_vec(
        10,
        2,
        vec![
            1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0, 5.0, 6.0, 1.5, 2.5, 2.5, 1.5, 3.5, 4.5, 4.5,
            3.5, 5.5, 6.5,
        ],
    )
    .expect("Valid matrix");

    let mut ica = ICA::new(2);
    let result = ica.fit(&data);
    assert!(result.is_ok(), "ICA should fit");

    let sources = ica.transform(&data).expect("Should transform");
    assert_eq!(sources.n_rows(), 10);
    assert_eq!(sources.n_cols(), 2);
}

#[test]
fn test_ica_invalid_n_components() {
    let data = Matrix::from_vec(5, 2, vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0])
        .expect("Valid matrix");

    let mut ica = ICA::new(3); // More components than features
    let result = ica.fit(&data);
    assert!(result.is_err());
}

#[test]
fn test_ica_transform_not_fitted() {
    let ica = ICA::new(2);
    let data = Matrix::from_vec(3, 2, vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0]).expect("Valid matrix");

    let result = ica.transform(&data);
    assert!(result.is_err());
}

#[test]
fn test_ica_dimension_mismatch() {
    let data = Matrix::from_vec(
        5,
        3,
        vec![
            1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0,
        ],
    )
    .expect("Valid matrix");

    let mut ica = ICA::new(2);
    ica.fit(&data).expect("Should fit");

    // Try to transform data with wrong dimensions
    let wrong_data =
        Matrix::from_vec(3, 2, vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0]).expect("Valid matrix");

    let result = ica.transform(&wrong_data);
    assert!(result.is_err());
}

#[test]
fn test_ica_with_options() {
    // Data with some variation between features
    let data = Matrix::from_vec(
        8,
        2,
        vec![
            1.0, 2.0, 2.0, 3.0, 3.0, 5.0, 4.0, 6.0, 5.0, 4.0, 6.0, 5.0, 7.0, 7.0, 8.0, 9.0,
        ],
    )
    .expect("Valid matrix");

    let mut ica = ICA::new(2).with_max_iter(100).with_tolerance(1e-5);

    let result = ica.fit(&data);
    assert!(result.is_ok());
}

#[test]
fn test_center_data() {
    let data = Matrix::from_vec(3, 2, vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0]).expect("Valid matrix");

    let (centered, mean) = ICA::center_data(&data).expect("Should center");

    assert_eq!(mean.len(), 2);
    assert!((mean[0] - 2.0).abs() < 1e-6); // mean of [1,2,3] is 2
    assert!((mean[1] - 4.0).abs() < 1e-6); // mean of [2,4,6] is 4

    // Centered data should have mean ~0
    let mut col0_sum = 0.0;
    let mut col1_sum = 0.0;
    for i in 0..3 {
        col0_sum += centered.get(i, 0);
        col1_sum += centered.get(i, 1);
    }
    assert!(col0_sum.abs() < 1e-6);
    assert!(col1_sum.abs() < 1e-6);
}

#[test]
fn test_power_iteration() {
    // Simple 2x2 matrix with known eigenvalues
    let matrix = Matrix::from_vec(2, 2, vec![3.0, 1.0, 1.0, 3.0]).expect("Valid matrix");

    let (eigenvalue, eigenvector) = ICA::power_iteration(&matrix, 100).expect("Should converge");

    // Largest eigenvalue should be 4.0
    assert!((eigenvalue - 4.0).abs() < 0.1, "Eigenvalue should be ~4.0");

    // Eigenvector should be normalized
    let norm: f32 = eigenvector
        .as_slice()
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
}

// ============================================================================
// Additional Coverage Tests
// ============================================================================

#[test]
fn test_ica_with_random_state() {
    let data = Matrix::from_vec(
        8,
        2,
        vec![
            1.0, 2.0, 2.0, 3.0, 3.0, 5.0, 4.0, 6.0, 5.0, 4.0, 6.0, 5.0, 7.0, 7.0, 8.0, 9.0,
        ],
    )
    .expect("Valid matrix");

    let mut ica = ICA::new(2).with_random_state(42);
    let result = ica.fit(&data);
    assert!(result.is_ok());
}

#[test]
fn test_ica_empty_data() {
    let data = Matrix::from_vec(0, 2, vec![]).expect("Valid empty matrix");
    let mut ica = ICA::new(2);
    let result = ica.fit(&data);
    assert!(result.is_err());
}

#[test]
fn test_ica_empty_features() {
    let data = Matrix::from_vec(5, 0, vec![]).expect("Valid empty matrix");
    let mut ica = ICA::new(1);
    let result = ica.fit(&data);
    assert!(result.is_err());
}

#[test]
fn test_ica_single_component() {
    let data = Matrix::from_vec(
        6,
        3,
        vec![
            1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0, 5.0, 10.0, 15.0, 6.0,
            12.0, 18.0,
        ],
    )
    .expect("Valid matrix");

    let mut ica = ICA::new(1);
    let result = ica.fit(&data);
    assert!(result.is_ok());

    let sources = ica.transform(&data).expect("Should transform");
    assert_eq!(sources.n_cols(), 1);
}

#[test]
fn test_ica_whitening() {
    let data = Matrix::from_vec(
        10,
        2,
        vec![
            1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0, 5.0, 6.0, 1.5, 2.5, 2.5, 1.5, 3.5, 4.5, 4.5,
            3.5, 5.5, 6.5,
        ],
    )
    .expect("Valid matrix");

    // Center and whiten
    let (centered, _mean) = ICA::center_data(&data).expect("Should center");
    let (whitened, _whitening_matrix) = ICA::whiten_data(&centered, 2).expect("Should whiten");

    assert_eq!(whitened.n_rows(), 10);
    assert_eq!(whitened.n_cols(), 2);

    // PMAT-847: whitening invariant — Cov(X_white) must be the identity matrix.
    // (unit variance on the diagonal, zero pairwise correlation off-diagonal)
    let n = whitened.n_rows();
    let mut cov = [[0.0_f32; 2]; 2];
    for r in 0..2 {
        for c in 0..2 {
            let mut sum = 0.0_f32;
            for i in 0..n {
                sum += whitened.get(i, r) * whitened.get(i, c);
            }
            cov[r][c] = sum / n as f32;
        }
    }
    assert!(
        (cov[0][0] - 1.0).abs() < 0.05,
        "whitened var[0] must be ~1, got {}",
        cov[0][0]
    );
    assert!(
        (cov[1][1] - 1.0).abs() < 0.05,
        "whitened var[1] must be ~1, got {}",
        cov[1][1]
    );
    assert!(
        cov[0][1].abs() < 0.05,
        "whitened cross-cov must be ~0, got {}",
        cov[0][1]
    );
}

/// PMAT-847 falsifier: whitening must yield Cov(X_white) = I on seeded correlated data.
///
/// Reference: scikit-learn `FastICA` (whiten='unit-variance'); numpy:
///   cov = (Xc.T @ Xc) / n; w, V = np.linalg.eigh(cov); W = V @ diag(1/sqrt(w));
///   ((Xc @ W).T @ (Xc @ W)) / n == I
///
/// The transposed-whitening-matrix bug makes the whitened covariance non-identity
/// (e.g. diag[0][0] ~= 5.97 instead of 1.0 on this fixture).
#[test]
fn test_ica_whitening_covariance_is_identity() {
    // Build n=200 correlated rows: f0 = 2a + b, f1 = a + 3b, where (a,b) are
    // pseudo-random in [0,1) from a deterministic LCG (reproducible fixture).
    let n = 200usize;
    let mut state: u64 = 42;
    let mut next = || -> f32 {
        // glibc LCG constants; masked to 31 bits.
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345) & 0x7fff_ffff;
        (state % 1000) as f32 / 1000.0
    };
    let mut data = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let a = next();
        let b = next();
        let f0 = 2.0 * a + b;
        let f1 = a + 3.0 * b;
        data.push(f0);
        data.push(f1);
    }
    let x = Matrix::from_vec(n, 2, data).expect("Valid matrix");

    let (centered, _mean) = ICA::center_data(&x).expect("Should center");
    let (whitened, _w) = ICA::whiten_data(&centered, 2).expect("Should whiten");

    assert_eq!(whitened.n_rows(), n);
    assert_eq!(whitened.n_cols(), 2);

    // cov = (1/n) Xw^T Xw
    let mut cov = [[0.0_f32; 2]; 2];
    for r in 0..2 {
        for c in 0..2 {
            let mut sum = 0.0_f32;
            for i in 0..n {
                sum += whitened.get(i, r) * whitened.get(i, c);
            }
            cov[r][c] = sum / n as f32;
        }
    }

    assert!(
        (cov[0][0] - 1.0).abs() < 0.05,
        "Cov(X_white)[0][0] must be ~1 (unit variance), got {}",
        cov[0][0]
    );
    assert!(
        (cov[1][1] - 1.0).abs() < 0.05,
        "Cov(X_white)[1][1] must be ~1 (unit variance), got {}",
        cov[1][1]
    );
    assert!(
        cov[0][1].abs() < 0.05,
        "Cov(X_white)[0][1] must be ~0 (decorrelated), got {}",
        cov[0][1]
    );
}

#[test]
fn test_ica_eigen_decomposition() {
    // Symmetric positive definite matrix
    let matrix = Matrix::from_vec(3, 3, vec![4.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 2.0])
        .expect("Valid matrix");

    let (eigenvalues, eigenvectors) =
        ICA::eigen_decomposition(&matrix, 2).expect("Should decompose");

    assert_eq!(eigenvalues.len(), 2);
    assert_eq!(eigenvectors.n_rows(), 3);
    assert_eq!(eigenvectors.n_cols(), 2);
}

#[test]
fn test_ica_eigen_decomposition_non_square() {
    let matrix = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("Valid matrix");

    let result = ICA::eigen_decomposition(&matrix, 2);
    assert!(result.is_err());
}

#[test]
fn test_ica_clone() {
    let ica = ICA::new(3)
        .with_max_iter(100)
        .with_tolerance(1e-5)
        .with_random_state(42);

    let cloned = ica.clone();
    // Just test that clone compiles and works
    assert_eq!(format!("{:?}", ica), format!("{:?}", cloned));
}

#[test]
fn test_ica_debug() {
    let ica = ICA::new(2);
    let debug_str = format!("{:?}", ica);
    assert!(debug_str.contains("ICA"));
    assert!(debug_str.contains("n_components"));
}

#[test]
fn test_ica_fit_then_transform_new_data() {
    let training_data = Matrix::from_vec(
        10,
        2,
        vec![
            1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0, 5.0, 6.0, 1.5, 2.5, 2.5, 1.5, 3.5, 4.5, 4.5,
            3.5, 5.5, 6.5,
        ],
    )
    .expect("Valid matrix");

    let mut ica = ICA::new(2);
    ica.fit(&training_data).expect("Should fit");

    // Transform different data with same shape
    let new_data = Matrix::from_vec(5, 2, vec![2.0, 3.0, 3.0, 2.0, 4.0, 5.0, 5.0, 4.0, 6.0, 7.0])
        .expect("Valid matrix");

    let transformed = ica.transform(&new_data).expect("Should transform");
    assert_eq!(transformed.n_rows(), 5);
    assert_eq!(transformed.n_cols(), 2);
}

#[test]
fn test_ica_3d_data() {
    // Test with 3 features/components - use data with more variance
    // Note: Highly correlated/linearly dependent data can cause issues
    let data = Matrix::from_vec(
        12,
        3,
        vec![
            1.0, 5.0, 2.0, // More variance between columns
            4.0, 2.0, 6.0, 3.0, 7.0, 1.0, 6.0, 3.0, 4.0, 2.0, 8.0, 5.0, 5.0, 1.0, 3.0, 1.5, 6.0,
            2.5, 4.5, 2.5, 5.5, 3.5, 6.5, 1.5, 5.5, 4.5, 4.5, 2.5, 7.5, 6.5, 6.5, 1.5, 3.5,
        ],
    )
    .expect("Valid matrix");

    let mut ica = ICA::new(2); // Use 2 components instead of 3 for more stability
    let result = ica.fit(&data);
    assert!(result.is_ok());

    let sources = ica.transform(&data).expect("Should transform");
    assert_eq!(sources.n_rows(), 12);
    assert_eq!(sources.n_cols(), 2);
}

#[test]
fn test_ica_strict_tolerance() {
    let data = Matrix::from_vec(
        8,
        2,
        vec![
            1.0, 2.0, 2.0, 3.0, 3.0, 5.0, 4.0, 6.0, 5.0, 4.0, 6.0, 5.0, 7.0, 7.0, 8.0, 9.0,
        ],
    )
    .expect("Valid matrix");

    // Very strict tolerance
    let mut ica = ICA::new(2).with_tolerance(1e-8).with_max_iter(500);
    let result = ica.fit(&data);
    assert!(result.is_ok());
}
