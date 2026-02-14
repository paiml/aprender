//! Covariance and correlation computations.
//!
//! This module provides functions for computing covariance and correlation
//! between variables.
//!
//! # Mathematical Background
//!
//! ## Covariance
//!
//! Covariance measures how two variables change together:
//!
//! ```text
//! Cov(X, Y) = E[(X - E[X])(Y - E[Y])]
//!           = (1/n) Σ (x_i - x̄)(y_i - ȳ)
//! ```
//!
//! ## Pearson Correlation
//!
//! Pearson correlation normalizes covariance to [-1, 1]:
//!
//! ```text
//! ρ(X, Y) = Cov(X, Y) / (σ_X σ_Y)
//! ```
//!
//! Where `σ_X` and `σ_Y` are standard deviations.
//!
//! # Examples
//!
//! ```
//! use aprender::stats::{cov, corr};
//! use aprender::primitives::Vector;
//!
//! let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
//! let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
//!
//! // Perfect positive correlation
//! let covariance = cov(&x, &y).expect("covariance should compute");
//! let correlation = corr(&x, &y).expect("correlation should compute");
//!
//! assert!(covariance > 0.0);
//! assert!((correlation - 1.0).abs() < 1e-6);
//! ```

use crate::error::{AprenderError, Result};
use crate::primitives::{Matrix, Vector};

/// Computes the covariance between two vectors.
///
/// # Arguments
///
/// * `x` - First variable (n values)
/// * `y` - Second variable (n values)
///
/// # Returns
///
/// Covariance: `Cov(X, Y) = (1/n) Σ (x_i - x̄)(y_i - ȳ)`
///
/// # Errors
///
/// Returns error if vectors have different lengths or are empty.
///
/// # Examples
///
/// ```
/// use aprender::stats::cov;
/// use aprender::primitives::Vector;
///
/// let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
/// let y = Vector::from_slice(&[2.0, 4.0, 5.0]);
///
/// let covariance = cov(&x, &y).expect("Should compute covariance");
/// assert!(covariance > 0.0); // Positive relationship
/// ```
pub fn cov(x: &Vector<f32>, y: &Vector<f32>) -> Result<f32> {
    let n = x.len();

    if n != y.len() {
        return Err(AprenderError::DimensionMismatch {
            expected: format!("{n} values in x"),
            actual: format!("{} values in y", y.len()),
        });
    }

    if n == 0 {
        return Err(AprenderError::Other(
            "Cannot compute covariance of empty vectors".into(),
        ));
    }

    // Compute means
    let x_mean = x.as_slice().iter().sum::<f32>() / n as f32;
    let y_mean = y.as_slice().iter().sum::<f32>() / n as f32;

    // Compute covariance: (1/n) Σ (x_i - x̄)(y_i - ȳ)
    let cov_sum: f32 = x
        .as_slice()
        .iter()
        .zip(y.as_slice().iter())
        .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
        .sum();

    Ok(cov_sum / n as f32)
}

/// Computes the covariance matrix for a data matrix.
///
/// # Arguments
///
/// * `data` - Data matrix (n × p), where n is samples and p is features
///
/// # Returns
///
/// Covariance matrix (p × p) where entry (i, j) is `Cov(feature_i, feature_j)`
///
/// # Errors
///
/// Returns error if data is empty or has invalid dimensions.
///
/// # Examples
///
/// ```
/// use aprender::stats::cov_matrix;
/// use aprender::primitives::Matrix;
///
/// // 3 samples, 2 features
/// let data = Matrix::from_vec(3, 2, vec![
///     1.0, 2.0,
///     2.0, 4.0,
///     3.0, 6.0,
/// ]).expect("Valid matrix");
///
/// let cov_mat = cov_matrix(&data).expect("Should compute covariance matrix");
/// assert_eq!(cov_mat.n_rows(), 2);
/// assert_eq!(cov_mat.n_cols(), 2);
/// ```
pub fn cov_matrix(data: &Matrix<f32>) -> Result<Matrix<f32>> {
    let n = data.n_rows(); // samples
    let p = data.n_cols(); // features

    if n == 0 || p == 0 {
        return Err(AprenderError::Other(
            "Cannot compute covariance matrix for empty data".into(),
        ));
    }

    // Compute means for each feature
    let mut means = vec![0.0_f32; p];
    #[allow(clippy::needless_range_loop)]
    for j in 0..p {
        let mut sum = 0.0;
        for i in 0..n {
            sum += data.get(i, j);
        }
        means[j] = sum / n as f32;
    }

    // Compute covariance matrix
    let mut cov_data = vec![0.0_f32; p * p];
    for i in 0..p {
        for j in 0..=i {
            // Only compute lower triangle (symmetric)
            let mut cov_sum = 0.0;
            for k in 0..n {
                cov_sum += (data.get(k, i) - means[i]) * (data.get(k, j) - means[j]);
            }
            let cov_val = cov_sum / n as f32;

            // Fill both (i,j) and (j,i) for symmetry
            cov_data[i * p + j] = cov_val;
            cov_data[j * p + i] = cov_val;
        }
    }

    Matrix::from_vec(p, p, cov_data)
        .map_err(|e| AprenderError::Other(format!("Failed to create covariance matrix: {e}")))
}

/// Computes the Pearson correlation coefficient between two vectors.
///
/// # Arguments
///
/// * `x` - First variable (n values)
/// * `y` - Second variable (n values)
///
/// # Returns
///
/// Pearson correlation: `ρ(X, Y) = Cov(X, Y) / (σ_X σ_Y)`
///
/// Range: [-1, 1]
/// - 1: Perfect positive correlation
/// - 0: No linear correlation
/// - -1: Perfect negative correlation
///
/// # Errors
///
/// Returns error if vectors have different lengths, are empty, or have zero variance.
///
/// # Examples
///
/// ```
/// use aprender::stats::corr;
/// use aprender::primitives::Vector;
///
/// let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
/// let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);
///
/// let correlation = corr(&x, &y).expect("Should compute correlation");
/// assert!((correlation - 1.0).abs() < 1e-6); // Perfect positive correlation
/// ```
pub fn corr(x: &Vector<f32>, y: &Vector<f32>) -> Result<f32> {
    let n = x.len();

    if n != y.len() {
        return Err(AprenderError::DimensionMismatch {
            expected: format!("{n} values in x"),
            actual: format!("{} values in y", y.len()),
        });
    }

    if n == 0 {
        return Err(AprenderError::Other(
            "Cannot compute correlation of empty vectors".into(),
        ));
    }

    // Compute means
    let x_mean = x.as_slice().iter().sum::<f32>() / n as f32;
    let y_mean = y.as_slice().iter().sum::<f32>() / n as f32;

    // Compute covariance and variances
    let mut cov_sum = 0.0;
    let mut x_var_sum = 0.0;
    let mut y_var_sum = 0.0;

    for (&xi, &yi) in x.as_slice().iter().zip(y.as_slice().iter()) {
        let x_diff = xi - x_mean;
        let y_diff = yi - y_mean;
        cov_sum += x_diff * y_diff;
        x_var_sum += x_diff * x_diff;
        y_var_sum += y_diff * y_diff;
    }

    let x_std = (x_var_sum / n as f32).sqrt();
    let y_std = (y_var_sum / n as f32).sqrt();

    if x_std < 1e-10 || y_std < 1e-10 {
        return Err(AprenderError::Other(
            "Cannot compute correlation when variance is zero".into(),
        ));
    }

    let covariance = cov_sum / n as f32;
    Ok(covariance / (x_std * y_std))
}

/// Computes the Pearson correlation matrix for a data matrix.
///
/// # Arguments
///
/// * `data` - Data matrix (n × p), where n is samples and p is features
///
/// # Returns
///
/// Correlation matrix (p × p) where entry (i, j) is the Pearson correlation
/// between feature i and feature j. Diagonal entries are 1.0.
///
/// # Errors
///
/// Returns error if data is empty, has invalid dimensions, or features have zero variance.
///
/// # Examples
///
/// ```
/// use aprender::stats::corr_matrix;
/// use aprender::primitives::Matrix;
///
/// // 3 samples, 2 features
/// let data = Matrix::from_vec(3, 2, vec![
///     1.0, 2.0,
///     2.0, 4.0,
///     3.0, 6.0,
/// ]).expect("Valid matrix");
///
/// let corr_mat = corr_matrix(&data).expect("Should compute correlation matrix");
/// assert_eq!(corr_mat.n_rows(), 2);
/// assert_eq!(corr_mat.n_cols(), 2);
/// assert!((corr_mat.get(0, 0) - 1.0).abs() < 1e-6); // Diagonal is 1.0
/// ```
pub fn corr_matrix(data: &Matrix<f32>) -> Result<Matrix<f32>> {
    let n = data.n_rows();
    let p = data.n_cols();

    if n == 0 || p == 0 {
        return Err(AprenderError::Other(
            "Cannot compute correlation matrix for empty data".into(),
        ));
    }

    let (means, stds) = compute_feature_stats(data, n, p)?;
    let corr_data = compute_correlation_values(data, &means, &stds, n, p);

    Matrix::from_vec(p, p, corr_data)
        .map_err(|e| AprenderError::Other(format!("Failed to create correlation matrix: {e}")))
}

fn compute_feature_stats(data: &Matrix<f32>, n: usize, p: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut means = vec![0.0_f32; p];
    let mut stds = vec![0.0_f32; p];

    for j in 0..p {
        let sum: f32 = (0..n).map(|i| data.get(i, j)).sum();
        means[j] = sum / n as f32;

        let var_sum: f32 = (0..n).map(|i| (data.get(i, j) - means[j]).powi(2)).sum();
        stds[j] = (var_sum / n as f32).sqrt();

        if stds[j] < 1e-10 {
            return Err(AprenderError::Other(format!(
                "Feature {j} has zero variance"
            )));
        }
    }
    Ok((means, stds))
}

fn compute_correlation_values(
    data: &Matrix<f32>,
    means: &[f32],
    stds: &[f32],
    n: usize,
    p: usize,
) -> Vec<f32> {
    let mut corr_data = vec![0.0_f32; p * p];
    for i in 0..p {
        corr_data[i * p + i] = 1.0; // Diagonal is 1.0
        for j in 0..i {
            let cov_sum: f32 = (0..n)
                .map(|k| (data.get(k, i) - means[i]) * (data.get(k, j) - means[j]))
                .sum();
            let corr_val = cov_sum / (n as f32 * stds[i] * stds[j]);
            corr_data[i * p + j] = corr_val;
            corr_data[j * p + i] = corr_val;
        }
    }
    corr_data
}

#[cfg(test)]
#[path = "covariance_tests.rs"]
mod tests;
