//! Correlation-Informed Transfer Learning (CITL) trainer.
//!
//! Trains a simple linear model that maps error feature vectors to fix
//! feature vectors using least-squares regression via the normal equation.
//!
//! Given a set of `ErrorFixPair` samples, the trainer computes a weight
//! matrix W such that `fix_features ≈ W * error_features`. Prediction
//! for new errors is a simple matrix-vector multiply.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// An error-fix training pair for CITL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorFixPair {
    /// Feature vector describing the error.
    pub error_features: Vec<f32>,
    /// Feature vector describing the fix.
    pub fix_features: Vec<f32>,
    /// Strength of the error-fix correlation in [0.0, 1.0].
    pub correlation_score: f32,
}

impl ErrorFixPair {
    /// Create a new error-fix pair.
    #[must_use]
    pub fn new(error_features: Vec<f32>, fix_features: Vec<f32>, correlation_score: f32) -> Self {
        Self { error_features, fix_features, correlation_score: correlation_score.clamp(0.0, 1.0) }
    }
}

/// CITL trainer that learns a linear mapping from error features to fix features.
///
/// Uses weighted least-squares regression:
///   `W = (X^T S X)^{-1} X^T S Y`
/// where S is a diagonal matrix of correlation scores used as sample weights.
///
/// # Example
///
/// ```
/// use entrenar::decision::{CitlTrainer, ErrorFixPair};
///
/// let pairs = vec![
///     ErrorFixPair::new(vec![1.0, 0.0], vec![0.0, 1.0], 0.9),
///     ErrorFixPair::new(vec![0.0, 1.0], vec![1.0, 0.0], 0.8),
/// ];
///
/// let trainer = CitlTrainer::train(&pairs).expect("training must succeed");
/// let prediction = trainer.predict_fix(&[1.0, 0.0]);
/// assert_eq!(prediction.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct CitlTrainer {
    /// Weight matrix of shape (fix_dim, error_dim).
    weights: Array2<f32>,
    /// Dimensionality of error features (input).
    error_dim: usize,
    /// Dimensionality of fix features (output).
    fix_dim: usize,
}

impl CitlTrainer {
    /// Train a linear correlation model from error-fix pairs.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `pairs` is empty
    /// - Feature dimensions are inconsistent across pairs
    /// - The normal equation matrix is singular (not invertible)
    pub fn train(pairs: &[ErrorFixPair]) -> Result<Self, crate::Error> {
        if pairs.is_empty() {
            return Err(crate::Error::InvalidParameter(
                "CITL training requires at least one error-fix pair".into(),
            ));
        }

        let error_dim = pairs[0].error_features.len();
        let fix_dim = pairs[0].fix_features.len();

        if error_dim == 0 || fix_dim == 0 {
            return Err(crate::Error::InvalidParameter(
                "Feature dimensions must be positive".into(),
            ));
        }

        // Validate consistent dimensions
        validate_pair_dimensions(pairs, error_dim, fix_dim)?;

        let n = pairs.len();

        // Build X (n x error_dim) and Y (n x fix_dim) matrices
        let mut x_data = Vec::with_capacity(n * error_dim);
        let mut y_data = Vec::with_capacity(n * fix_dim);
        let mut sample_weights = Vec::with_capacity(n);

        for pair in pairs {
            x_data.extend_from_slice(&pair.error_features);
            y_data.extend_from_slice(&pair.fix_features);
            sample_weights.push(pair.correlation_score.max(1e-6)); // avoid zero weight
        }

        let x = Array2::from_shape_vec((n, error_dim), x_data)
            .map_err(|e| crate::Error::InvalidParameter(format!("X matrix build error: {e}")))?;
        let y = Array2::from_shape_vec((n, fix_dim), y_data)
            .map_err(|e| crate::Error::InvalidParameter(format!("Y matrix build error: {e}")))?;

        // Build diagonal weight vector sqrt(S) for weighted least squares
        let sqrt_w: Array1<f32> =
            Array1::from_vec(sample_weights.iter().map(|w| w.sqrt()).collect());

        // Apply weights: X_w = diag(sqrt_w) * X,  Y_w = diag(sqrt_w) * Y
        let mut x_w = x.clone();
        let mut y_w = y.clone();
        for i in 0..n {
            let sw = sqrt_w[i];
            for j in 0..error_dim {
                x_w[[i, j]] *= sw;
            }
            for j in 0..fix_dim {
                y_w[[i, j]] *= sw;
            }
        }

        // Normal equation: W = (X_w^T X_w)^{-1} X_w^T Y_w
        // A = X_w^T X_w  (error_dim x error_dim)
        let a = x_w.t().dot(&x_w);

        // B = X_w^T Y_w  (error_dim x fix_dim)
        let b = x_w.t().dot(&y_w);

        // Solve A * W^T = B  via Tikhonov regularization (ridge)
        // A_reg = A + lambda * I  to ensure invertibility
        let lambda = 1e-4_f32;
        let mut a_reg = a;
        for i in 0..error_dim {
            a_reg[[i, i]] += lambda;
        }

        // Invert A_reg using Gauss-Jordan elimination
        let a_inv = invert_matrix(&a_reg).map_err(|_e| {
            crate::Error::InvalidParameter(
                "Normal equation matrix is singular; cannot solve for weights".into(),
            )
        })?;

        // W^T = A_inv * B  (error_dim x fix_dim)
        let w_t = a_inv.dot(&b);

        // W = (W^T)^T  (fix_dim x error_dim)
        let weights = w_t.t().to_owned();

        Ok(Self { weights, error_dim, fix_dim })
    }

    /// Predict fix features from error features.
    ///
    /// Returns a zero vector if the input dimension does not match `error_dim`.
    #[must_use]
    pub fn predict_fix(&self, error_features: &[f32]) -> Vec<f32> {
        if error_features.len() != self.error_dim {
            return vec![0.0; self.fix_dim];
        }

        let x = Array1::from_vec(error_features.to_vec());
        let y = self.weights.dot(&x);
        y.to_vec()
    }

    /// Return the error feature dimensionality.
    #[must_use]
    pub fn error_dim(&self) -> usize {
        self.error_dim
    }

    /// Return the fix feature dimensionality.
    #[must_use]
    pub fn fix_dim(&self) -> usize {
        self.fix_dim
    }

    /// Return a reference to the weight matrix.
    #[must_use]
    pub fn weights(&self) -> &Array2<f32> {
        &self.weights
    }
}

/// Validate that all pairs have consistent error/fix feature dimensions.
fn validate_pair_dimensions(
    pairs: &[ErrorFixPair],
    error_dim: usize,
    fix_dim: usize,
) -> Result<(), crate::Error> {
    for (i, pair) in pairs.iter().enumerate() {
        if pair.error_features.len() != error_dim {
            return Err(crate::Error::ShapeMismatch {
                expected: vec![error_dim],
                got: vec![pair.error_features.len()],
            });
        }
        if pair.fix_features.len() != fix_dim {
            return Err(crate::Error::ShapeMismatch {
                expected: vec![fix_dim],
                got: vec![pair.fix_features.len()],
            });
        }
        if i > 0 && pair.error_features.len() != error_dim {
            return Err(crate::Error::InvalidParameter(format!(
                "Inconsistent error feature dimension at pair {i}"
            )));
        }
    }
    Ok(())
}

/// Invert a square matrix using Gauss-Jordan elimination.
///
/// Returns `Err(())` if the matrix is singular.
fn invert_matrix(m: &Array2<f32>) -> std::result::Result<Array2<f32>, ()> {
    let n = m.nrows();
    assert_eq!(n, m.ncols(), "Matrix must be square");

    let mut aug = build_augmented(m, n);

    for col in 0..n {
        pivot_column(&mut aug, col, n)?;
        eliminate_column(&mut aug, col, n);
    }

    Ok(extract_inverse(&aug, n))
}

/// Build augmented matrix [M | I].
fn build_augmented(m: &Array2<f32>, n: usize) -> Array2<f32> {
    let mut aug = Array2::<f32>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = m[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }
    aug
}

/// Partial pivoting: find largest pivot, swap rows, scale pivot row.
fn pivot_column(aug: &mut Array2<f32>, col: usize, n: usize) -> std::result::Result<(), ()> {
    let mut max_val = aug[[col, col]].abs();
    let mut max_row = col;
    for row in (col + 1)..n {
        let val = aug[[row, col]].abs();
        if val > max_val {
            max_val = val;
            max_row = row;
        }
    }

    if max_val < 1e-12 {
        return Err(());
    }

    if max_row != col {
        for j in 0..(2 * n) {
            let tmp = aug[[col, j]];
            aug[[col, j]] = aug[[max_row, j]];
            aug[[max_row, j]] = tmp;
        }
    }

    let pivot = aug[[col, col]];
    for j in 0..(2 * n) {
        aug[[col, j]] /= pivot;
    }
    Ok(())
}

/// Eliminate all rows except pivot row for a given column.
fn eliminate_column(aug: &mut Array2<f32>, col: usize, n: usize) {
    for row in 0..n {
        if row == col {
            continue;
        }
        let factor = aug[[row, col]];
        for j in 0..(2 * n) {
            aug[[row, j]] -= factor * aug[[col, j]];
        }
    }
}

/// Extract the inverse matrix from the right half of the augmented matrix.
fn extract_inverse(aug: &Array2<f32>, n: usize) -> Array2<f32> {
    let mut inv = Array2::<f32>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_pairs() -> Vec<ErrorFixPair> {
        vec![
            ErrorFixPair::new(vec![1.0, 0.0], vec![0.0, 1.0], 0.9),
            ErrorFixPair::new(vec![0.0, 1.0], vec![1.0, 0.0], 0.8),
            ErrorFixPair::new(vec![1.0, 1.0], vec![1.0, 1.0], 0.7),
        ]
    }

    #[test]
    fn test_train_produces_correct_dims() {
        let trainer = CitlTrainer::train(&simple_pairs()).expect("operation should succeed");
        assert_eq!(trainer.error_dim(), 2);
        assert_eq!(trainer.fix_dim(), 2);
        assert_eq!(trainer.weights().shape(), &[2, 2]);
    }

    #[test]
    fn test_predict_suggestion_output_length() {
        let trainer = CitlTrainer::train(&simple_pairs()).expect("operation should succeed");
        let pred = trainer.predict_fix(&[1.0, 0.0]);
        assert_eq!(pred.len(), 2);
    }

    #[test]
    fn test_predict_fix_wrong_dim_returns_zeros() {
        let trainer = CitlTrainer::train(&simple_pairs()).expect("operation should succeed");
        let pred = trainer.predict_fix(&[1.0, 0.0, 0.0]);
        assert_eq!(pred, vec![0.0, 0.0]);
    }

    #[test]
    fn test_train_empty_pairs() {
        let result = CitlTrainer::train(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_train_zero_dim_features() {
        let pairs = vec![ErrorFixPair::new(vec![], vec![1.0], 1.0)];
        let result = CitlTrainer::train(&pairs);
        assert!(result.is_err());
    }

    #[test]
    fn test_train_inconsistent_error_dims() {
        let pairs = vec![
            ErrorFixPair::new(vec![1.0, 0.0], vec![1.0], 0.9),
            ErrorFixPair::new(vec![1.0], vec![1.0], 0.8), // wrong error dim
        ];
        let result = CitlTrainer::train(&pairs);
        assert!(result.is_err());
    }

    #[test]
    fn test_train_inconsistent_fix_dims() {
        let pairs = vec![
            ErrorFixPair::new(vec![1.0], vec![1.0, 0.0], 0.9),
            ErrorFixPair::new(vec![0.0], vec![1.0], 0.8), // wrong fix dim
        ];
        let result = CitlTrainer::train(&pairs);
        assert!(result.is_err());
    }

    #[test]
    fn test_identity_mapping() {
        // Train on identity-like mapping: error = fix
        let pairs: Vec<ErrorFixPair> = (0..10)
            .map(|i| {
                let mut e = vec![0.0; 3];
                e[i % 3] = 1.0;
                ErrorFixPair::new(e.clone(), e, 1.0)
            })
            .collect();

        let trainer = CitlTrainer::train(&pairs).expect("operation should succeed");
        let pred = trainer.predict_fix(&[1.0, 0.0, 0.0]);
        // Should approximately recover the identity mapping
        assert!((pred[0] - 1.0).abs() < 0.2, "pred[0]={}", pred[0]);
        assert!(pred[1].abs() < 0.2, "pred[1]={}", pred[1]);
        assert!(pred[2].abs() < 0.2, "pred[2]={}", pred[2]);
    }

    #[test]
    fn test_correlation_score_clamped() {
        let pair = ErrorFixPair::new(vec![1.0], vec![1.0], 2.0);
        assert_eq!(pair.correlation_score, 1.0);

        let pair2 = ErrorFixPair::new(vec![1.0], vec![1.0], -1.0);
        assert_eq!(pair2.correlation_score, 0.0);
    }

    #[test]
    fn test_single_pair_training() {
        let pairs = vec![ErrorFixPair::new(vec![2.0, 0.0], vec![0.0, 4.0], 1.0)];
        let trainer = CitlTrainer::train(&pairs).expect("operation should succeed");
        let pred = trainer.predict_fix(&[2.0, 0.0]);
        // With only one sample + ridge regularization, should approximate [0.0, 4.0]
        assert!(pred.len() == 2);
        // Direction should be roughly correct
        assert!(pred[1] > pred[0], "pred={pred:?}");
    }

    #[test]
    fn test_invert_identity() {
        let eye = Array2::eye(3);
        let inv = invert_matrix(&eye).expect("operation should succeed");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[[i, j]] - expected).abs() < 1e-6, "inv[{i},{j}]={}", inv[[i, j]]);
            }
        }
    }

    #[test]
    fn test_invert_2x2() {
        // [[2, 1], [1, 1]] -> inverse [[1, -1], [-1, 2]]
        let m = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 1.0])
            .expect("operation should succeed");
        let inv = invert_matrix(&m).expect("operation should succeed");
        assert!((inv[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((inv[[0, 1]] - (-1.0)).abs() < 1e-5);
        assert!((inv[[1, 0]] - (-1.0)).abs() < 1e-5);
        assert!((inv[[1, 1]] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_weighted_training() {
        // Two conflicting samples: high-weight sample should dominate
        let pairs = vec![
            ErrorFixPair::new(vec![1.0, 0.0], vec![10.0, 0.0], 1.0), // high weight
            ErrorFixPair::new(vec![1.0, 0.0], vec![0.0, 10.0], 0.01), // low weight
        ];
        let trainer = CitlTrainer::train(&pairs).expect("operation should succeed");
        let pred = trainer.predict_fix(&[1.0, 0.0]);
        // High-weight sample's fix direction should dominate
        assert!(pred[0] > pred[1], "High-weight sample should dominate: pred={pred:?}");
    }
}
