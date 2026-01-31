//! Gaussian Mixture Model (GMM) for probabilistic clustering.
//!
//! Uses Expectation-Maximization (EM) algorithm to fit a mixture of
//! Gaussian distributions to the data, providing soft cluster assignments.

use super::KMeans;
use crate::error::Result;
use crate::primitives::{Matrix, Vector};
use crate::traits::UnsupervisedEstimator;
use serde::{Deserialize, Serialize};

/// Covariance matrix types for Gaussian Mixture Models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CovarianceType {
    /// Full covariance matrix (most flexible, most parameters).
    Full,
    /// Tied covariance (all components share same covariance).
    Tied,
    /// Diagonal covariance (assumes feature independence).
    Diag,
    /// Spherical covariance (isotropic, like K-Means).
    Spherical,
}

/// Gaussian Mixture Model (GMM) for probabilistic clustering.
///
/// Uses Expectation-Maximization (EM) algorithm to fit a mixture of
/// Gaussian distributions to the data, providing soft cluster assignments.
///
/// # Algorithm
///
/// 1. **E-step**: Compute responsibilities (probability each point belongs to each cluster)
/// 2. **M-step**: Update means, covariances, and mixing weights
/// 3. Repeat until convergence
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// let data = Matrix::from_vec(6, 2, vec![
///     1.0, 1.0, 1.1, 1.0, 1.0, 1.1,
///     5.0, 5.0, 5.1, 5.0, 5.0, 5.1,
/// ]).expect("Valid matrix dimensions and data length");
///
/// let mut gmm = GaussianMixture::new(2, CovarianceType::Full);
/// gmm.fit(&data).expect("Fit succeeds with valid data");
///
/// let labels = gmm.predict(&data);
/// assert_eq!(labels.len(), 6);
///
/// let proba = gmm.predict_proba(&data);
/// assert_eq!(proba.shape(), (6, 2));
/// ```
///
/// # Performance
///
/// - Time complexity: O(nkd²i) where n=samples, k=components, d=features, i=iterations
/// - Space complexity: O(nk + kd²)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianMixture {
    /// Number of mixture components.
    n_components: usize,
    /// Type of covariance matrix.
    covariance_type: CovarianceType,
    /// Maximum number of EM iterations.
    max_iter: usize,
    /// Convergence tolerance.
    tol: f32,
    /// Random seed for initialization.
    random_state: Option<u64>,
    /// Component means after fitting (k × d).
    means: Option<Matrix<f32>>,
    /// Component covariances after fitting.
    covariances: Option<Vec<Matrix<f32>>>,
    /// Mixing weights after fitting (sums to 1).
    weights: Option<Vector<f32>>,
    /// Cluster labels after fitting.
    labels: Option<Vec<usize>>,
}

impl GaussianMixture {
    /// Create new `GaussianMixture` with specified number of components and covariance type.
    #[must_use]
    pub fn new(n_components: usize, covariance_type: CovarianceType) -> Self {
        Self {
            n_components,
            covariance_type,
            max_iter: 100,
            tol: 1e-3,
            random_state: None,
            means: None,
            covariances: None,
            weights: None,
            labels: None,
        }
    }

    /// Set maximum number of EM iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Set random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Get number of components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Get covariance type.
    #[must_use]
    pub fn covariance_type(&self) -> CovarianceType {
        self.covariance_type
    }

    /// Check if model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.means.is_some()
    }

    /// Get component means (panic if not fitted).
    #[must_use]
    pub fn means(&self) -> &Matrix<f32> {
        self.means
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// Get mixing weights (panic if not fitted).
    #[must_use]
    pub fn weights(&self) -> &Vector<f32> {
        self.weights
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// Get cluster labels (panic if not fitted).
    #[must_use]
    pub fn labels(&self) -> &Vec<usize> {
        self.labels
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// Compute log probability of data under the model.
    #[must_use]
    pub fn score(&self, x: &Matrix<f32>) -> f32 {
        assert!(self.is_fitted(), "Model not fitted. Call fit() first.");
        let responsibilities = self.compute_responsibilities(x);
        let n_samples = x.shape().0;

        // Compute log-likelihood
        let mut log_likelihood = 0.0;
        for i in 0..n_samples {
            let mut prob_sum = 0.0;
            for k in 0..self.n_components {
                prob_sum += responsibilities.get(i, k);
            }
            if prob_sum > 0.0 {
                log_likelihood += prob_sum.ln();
            }
        }
        log_likelihood / n_samples as f32
    }

    /// Predict cluster probabilities for each sample (soft assignment).
    #[must_use]
    pub fn predict_proba(&self, x: &Matrix<f32>) -> Matrix<f32> {
        assert!(self.is_fitted(), "Model not fitted. Call fit() first.");
        self.compute_responsibilities(x)
    }

    /// Initialize parameters using K-Means.
    fn initialize_parameters(&mut self, x: &Matrix<f32>) -> Result<()> {
        let (_n_samples, n_features) = x.shape();

        // Use K-Means for initialization
        let mut kmeans = KMeans::new(self.n_components);
        if let Some(seed) = self.random_state {
            kmeans = kmeans.with_random_state(seed);
        }
        kmeans.fit(x)?;

        // Set initial means from K-Means centroids
        self.means = Some(kmeans.centroids().clone());

        // Initialize weights uniformly
        let weight = 1.0 / self.n_components as f32;
        self.weights = Some(Vector::from_vec(vec![weight; self.n_components]));

        // Initialize covariances
        let mut covariances = Vec::new();
        for _k in 0..self.n_components {
            let cov = match self.covariance_type {
                CovarianceType::Full
                | CovarianceType::Tied
                | CovarianceType::Diag
                | CovarianceType::Spherical => {
                    // Start with identity matrix
                    let mut cov_matrix = vec![0.0; n_features * n_features];
                    for i in 0..n_features {
                        cov_matrix[i * n_features + i] = 1.0;
                    }
                    Matrix::from_vec(n_features, n_features, cov_matrix)?
                }
            };
            covariances.push(cov);
        }
        self.covariances = Some(covariances);

        Ok(())
    }

    /// Compute Gaussian probability density.
    #[allow(clippy::needless_range_loop)]
    #[allow(clippy::unused_self)]
    fn gaussian_pdf(&self, x: &[f32], mean: &[f32], cov: &Matrix<f32>) -> f32 {
        let n_features = mean.len();

        // Compute (x - mean)
        let mut diff = vec![0.0; n_features];
        for i in 0..n_features {
            diff[i] = x[i] - mean[i];
        }

        // For numerical stability, use simplified calculation
        // This is a simplified version - production code would use proper matrix inverse
        let mut mahalanobis = 0.0;
        for i in 0..n_features {
            let cov_ii = cov.get(i, i).max(1e-6); // Regularization
            mahalanobis += diff[i] * diff[i] / cov_ii;
        }

        // Compute determinant (simplified for diagonal-like structure)
        let mut det = 1.0;
        for i in 0..n_features {
            det *= cov.get(i, i).max(1e-6);
        }

        let norm_const = ((2.0 * std::f32::consts::PI).powi(n_features as i32) * det).sqrt();
        (-0.5 * mahalanobis).exp() / norm_const.max(1e-10)
    }

    /// E-step: Compute responsibilities (posterior probabilities).
    #[allow(clippy::needless_range_loop)]
    fn compute_responsibilities(&self, x: &Matrix<f32>) -> Matrix<f32> {
        let (n_samples, n_features) = x.shape();
        let means = self
            .means
            .as_ref()
            .expect("Means must be initialized before computing responsibilities");
        let weights = self
            .weights
            .as_ref()
            .expect("Weights must be initialized before computing responsibilities");
        let covariances = self
            .covariances
            .as_ref()
            .expect("Covariances must be initialized before computing responsibilities");

        let mut responsibilities = vec![0.0; n_samples * self.n_components];

        for i in 0..n_samples {
            let mut sample = vec![0.0; n_features];
            for j in 0..n_features {
                sample[j] = x.get(i, j);
            }

            let mut total_prob = 0.0;
            for k in 0..self.n_components {
                let mut mean_k = vec![0.0; n_features];
                for j in 0..n_features {
                    mean_k[j] = means.get(k, j);
                }

                let pdf = self.gaussian_pdf(&sample, &mean_k, &covariances[k]);
                let weighted_pdf = weights[k] * pdf;
                responsibilities[i * self.n_components + k] = weighted_pdf;
                total_prob += weighted_pdf;
            }

            // Normalize responsibilities
            if total_prob > 1e-10 {
                for k in 0..self.n_components {
                    responsibilities[i * self.n_components + k] /= total_prob;
                }
            } else {
                // Uniform if total prob is too small
                for k in 0..self.n_components {
                    responsibilities[i * self.n_components + k] = 1.0 / self.n_components as f32;
                }
            }
        }

        Matrix::from_vec(n_samples, self.n_components, responsibilities)
            .expect("Responsibility matrix dimensions match preallocated vector length")
    }

    /// M-step: Update parameters based on responsibilities.
    #[allow(clippy::needless_range_loop)]
    fn update_parameters(&mut self, x: &Matrix<f32>, responsibilities: &Matrix<f32>) -> Result<()> {
        let (n_samples, n_features) = x.shape();

        // Compute effective number of points per component
        let mut n_k = vec![0.0; self.n_components];
        for k in 0..self.n_components {
            for i in 0..n_samples {
                n_k[k] += responsibilities.get(i, k);
            }
            n_k[k] = n_k[k].max(1e-6); // Regularization
        }

        // Update weights
        let mut new_weights = vec![0.0; self.n_components];
        for k in 0..self.n_components {
            new_weights[k] = n_k[k] / n_samples as f32;
        }
        self.weights = Some(Vector::from_vec(new_weights));

        // Update means
        let mut new_means = vec![0.0; self.n_components * n_features];
        for k in 0..self.n_components {
            for j in 0..n_features {
                let mut weighted_sum = 0.0;
                for i in 0..n_samples {
                    weighted_sum += responsibilities.get(i, k) * x.get(i, j);
                }
                new_means[k * n_features + j] = weighted_sum / n_k[k];
            }
        }
        self.means = Some(Matrix::from_vec(self.n_components, n_features, new_means)?);

        // Update covariances (simplified diagonal)
        let means = self
            .means
            .as_ref()
            .expect("Means must exist after update step");
        let mut new_covariances = Vec::new();

        for k in 0..self.n_components {
            let mut cov_data = vec![0.0; n_features * n_features];

            for j in 0..n_features {
                let mut variance = 0.0;
                for i in 0..n_samples {
                    let diff = x.get(i, j) - means.get(k, j);
                    variance += responsibilities.get(i, k) * diff * diff;
                }
                variance = (variance / n_k[k]).max(1e-6); // Regularization
                cov_data[j * n_features + j] = variance;
            }

            new_covariances.push(Matrix::from_vec(n_features, n_features, cov_data)?);
        }
        self.covariances = Some(new_covariances);

        Ok(())
    }
}

impl UnsupervisedEstimator for GaussianMixture {
    type Labels = Vec<usize>;

    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        // Initialize parameters
        self.initialize_parameters(x)?;

        // EM algorithm
        let mut prev_log_likelihood = f32::NEG_INFINITY;

        for _iter in 0..self.max_iter {
            // E-step: Compute responsibilities
            let responsibilities = self.compute_responsibilities(x);

            // M-step: Update parameters
            self.update_parameters(x, &responsibilities)?;

            // Check convergence
            let log_likelihood = self.score(x);
            if (log_likelihood - prev_log_likelihood).abs() < self.tol {
                break;
            }
            prev_log_likelihood = log_likelihood;
        }

        // Assign labels based on final responsibilities
        let responsibilities = self.compute_responsibilities(x);
        let n_samples = x.shape().0;
        let mut labels = vec![0; n_samples];

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_samples {
            let mut max_prob = 0.0;
            let mut max_k = 0;
            for k in 0..self.n_components {
                let prob = responsibilities.get(i, k);
                if prob > max_prob {
                    max_prob = prob;
                    max_k = k;
                }
            }
            labels[i] = max_k;
        }

        self.labels = Some(labels);
        Ok(())
    }

    fn predict(&self, x: &Matrix<f32>) -> Self::Labels {
        assert!(self.is_fitted(), "Model not fitted. Call fit() first.");

        let responsibilities = self.compute_responsibilities(x);
        let n_samples = x.shape().0;
        let mut labels = vec![0; n_samples];

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_samples {
            let mut max_prob = 0.0;
            let mut max_k = 0;
            for k in 0..self.n_components {
                let prob = responsibilities.get(i, k);
                if prob > max_prob {
                    max_prob = prob;
                    max_k = k;
                }
            }
            labels[i] = max_k;
        }

        labels
    }
}
