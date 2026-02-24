//! Spectral Clustering using graph Laplacian and eigendecomposition.
//!
//! Uses graph theory to find clusters by analyzing the spectrum (eigenvalues)
//! of the graph Laplacian. Effective for non-convex cluster shapes.

use super::KMeans;
use crate::error::Result;
use crate::primitives::Matrix;
use crate::traits::UnsupervisedEstimator;
use serde::{Deserialize, Serialize};

/// Affinity types for constructing similarity graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Affinity {
    /// Radial Basis Function (Gaussian) kernel
    RBF,
    /// k-Nearest Neighbors graph
    KNN,
}

/// Spectral Clustering using graph Laplacian and eigendecomposition.
///
/// Uses graph theory to find clusters by analyzing the spectrum (eigenvalues)
/// of the graph Laplacian. Effective for non-convex cluster shapes.
///
/// # Algorithm
///
/// 1. Construct affinity matrix W (RBF or k-NN)
/// 2. Compute graph Laplacian: L = D - W (D = degree matrix)
/// 3. Find k smallest eigenvectors of L
/// 4. Cluster rows of eigenvector matrix using K-Means
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// // Non-convex clusters (concentric circles)
/// let data = Matrix::from_vec(
///     8,
///     2,
///     vec![
///         0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0,  // Inner circle
///         0.0, 3.0, 3.0, 0.0, 0.0, -3.0, -3.0, 0.0,  // Outer circle
///     ],
/// )
/// .expect("Valid matrix dimensions and data length");
///
/// let mut sc = SpectralClustering::new(2).with_gamma(0.5);
/// sc.fit(&data).expect("Fit succeeds with valid data");
///
/// let labels = sc.predict(&data);
/// ```
///
/// # Performance
///
/// - Time complexity: O(n² + n³) for eigendecomposition
/// - Space complexity: O(n²) for affinity matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralClustering {
    /// Number of clusters
    n_clusters: usize,
    /// Affinity type (RBF or KNN)
    affinity: Affinity,
    /// RBF kernel coefficient (higher = more local)
    gamma: f32,
    /// Number of neighbors for KNN affinity
    n_neighbors: usize,
    /// Cluster labels
    labels: Option<Vec<usize>>,
}

impl SpectralClustering {
    /// Create a new Spectral Clustering with default parameters.
    ///
    /// Default: RBF affinity, gamma=1.0, `n_neighbors=10`
    #[must_use]
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            affinity: Affinity::RBF,
            gamma: 1.0,
            n_neighbors: 10,
            labels: None,
        }
    }

    /// Set the affinity type.
    #[must_use]
    pub fn with_affinity(mut self, affinity: Affinity) -> Self {
        self.affinity = affinity;
        self
    }

    /// Set gamma for RBF kernel (higher = more local similarity).
    #[must_use]
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set number of neighbors for KNN affinity.
    #[must_use]
    pub fn with_n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Check if model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.labels.is_some()
    }

    /// Get cluster labels (panics if not fitted).
    #[must_use]
    pub fn labels(&self) -> &Vec<usize> {
        assert!(self.is_fitted(), "Model not fitted. Call fit() first.");
        self.labels
            .as_ref()
            .expect("Labels must exist after successful fit")
    }

    /// Fit the Spectral Clustering model.
    pub fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let (n_samples, _) = x.shape();

        // 1. Construct affinity matrix
        let affinity_matrix = match self.affinity {
            Affinity::RBF => self.compute_rbf_affinity(x),
            Affinity::KNN => self.compute_knn_affinity(x),
        };

        // 2. Compute graph Laplacian
        let laplacian = self.compute_laplacian(&affinity_matrix, n_samples);

        // 3. Eigendecomposition - find k smallest eigenvectors
        let mut embedding = self.compute_embedding(&laplacian, n_samples)?;

        // 4. Normalize rows of embedding (critical for normalized spectral clustering)
        for i in 0..n_samples {
            let mut row_norm = 0.0;
            for j in 0..self.n_clusters {
                let val = embedding.get(i, j);
                row_norm += val * val;
            }
            row_norm = row_norm.sqrt().max(1e-10); // Avoid division by zero
            for j in 0..self.n_clusters {
                let val = embedding.get(i, j);
                embedding.set(i, j, val / row_norm);
            }
        }

        // 5. Cluster in eigenspace using K-Means
        let mut kmeans = KMeans::new(self.n_clusters).with_random_state(42);
        kmeans.fit(&embedding)?;
        let labels = kmeans.predict(&embedding);

        self.labels = Some(labels);
        Ok(())
    }

    /// Compute RBF (Gaussian) affinity matrix.
    fn compute_rbf_affinity(&self, x: &Matrix<f32>) -> Vec<f32> {
        let (n_samples, n_features) = x.shape();
        let mut affinity = vec![0.0; n_samples * n_samples];

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i == j {
                    affinity[i * n_samples + j] = 0.0;
                    continue;
                }

                // Compute squared Euclidean distance
                let mut dist_sq = 0.0;
                for k in 0..n_features {
                    let diff = x.get(i, k) - x.get(j, k);
                    dist_sq += diff * diff;
                }

                // RBF kernel: exp(-gamma * ||x_i - x_j||^2)
                let similarity = (-self.gamma * dist_sq).exp();
                affinity[i * n_samples + j] = similarity;
            }
        }

        affinity
    }

    /// Compute k-NN affinity matrix.
    fn compute_knn_affinity(&self, x: &Matrix<f32>) -> Vec<f32> {
        let (n_samples, n_features) = x.shape();
        let mut affinity = vec![0.0; n_samples * n_samples];

        // For each point, find k nearest neighbors
        for i in 0..n_samples {
            // Compute distances to all points
            let mut distances: Vec<(f32, usize)> = Vec::with_capacity(n_samples);
            for j in 0..n_samples {
                if i == j {
                    continue;
                }

                let mut dist_sq = 0.0;
                for k in 0..n_features {
                    let diff = x.get(i, k) - x.get(j, k);
                    dist_sq += diff * diff;
                }
                distances.push((dist_sq.sqrt(), j));
            }

            // Sort and take k nearest
            distances.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .expect("Distances must be valid floats for comparison")
            });
            let k_neighbors = self.n_neighbors.min(n_samples - 1);

            // Set affinity to 1 for k-nearest neighbors
            for (_, neighbor_idx) in distances.iter().take(k_neighbors) {
                affinity[i * n_samples + neighbor_idx] = 1.0;
            }
        }

        // Make symmetric
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let val = f32::max(affinity[i * n_samples + j], affinity[j * n_samples + i]);
                affinity[i * n_samples + j] = val;
                affinity[j * n_samples + i] = val;
            }
        }

        affinity
    }

    /// Compute normalized graph Laplacian.
    #[allow(clippy::unused_self)]
    fn compute_laplacian(&self, affinity: &[f32], n_samples: usize) -> Vec<f32> {
        // Compute degree matrix D
        let mut degrees = vec![0.0; n_samples];
        for i in 0..n_samples {
            let mut degree = 0.0;
            for j in 0..n_samples {
                degree += affinity[i * n_samples + j];
            }
            degrees[i] = degree;
        }

        // Compute normalized Laplacian: L = I - D^(-1/2) * W * D^(-1/2)
        let mut laplacian = vec![0.0; n_samples * n_samples];

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i == j {
                    laplacian[i * n_samples + j] = 1.0;
                } else {
                    let d_i = degrees[i].max(1e-10); // Avoid division by zero
                    let d_j = degrees[j].max(1e-10);
                    let w_ij = affinity[i * n_samples + j];
                    laplacian[i * n_samples + j] = -w_ij / (d_i * d_j).sqrt();
                }
            }
        }

        laplacian
    }

    /// Compute embedding via eigendecomposition.
    fn compute_embedding(&self, laplacian: &[f32], n_samples: usize) -> Result<Matrix<f32>> {
        use trueno::SymmetricEigen;

        // Convert to trueno Matrix for eigendecomposition
        let laplacian_matrix = trueno::Matrix::from_vec(n_samples, n_samples, laplacian.to_vec())
            .map_err(|e| format!("Failed to create Laplacian matrix: {e}"))?;

        // Eigendecomposition - trueno returns eigenvalues in descending order
        let eigen = SymmetricEigen::new(&laplacian_matrix)
            .map_err(|e| format!("Eigendecomposition failed: {e}"))?;

        let eigenvalues = eigen.eigenvalues();
        let eigenvectors = eigen.eigenvectors();

        // Spectral clustering needs k smallest eigenvalues
        // trueno returns descending order, so smallest are at the end
        let k = self.n_clusters;
        let n = eigenvalues.len();

        // Indices of k smallest eigenvalues (from end of descending list)
        let smallest_indices: Vec<usize> = (n.saturating_sub(k)..n).collect();

        let mut embedding_data = Vec::with_capacity(n_samples * k);

        // Build embedding matrix in row-major order
        for row_idx in 0..n_samples {
            for &col_idx in &smallest_indices {
                let val = *eigenvectors
                    .get(row_idx, col_idx)
                    .ok_or_else(|| format!("Invalid eigenvector index ({row_idx}, {col_idx})"))?;
                embedding_data.push(val);
            }
        }

        Ok(Matrix::from_vec(n_samples, k, embedding_data)?)
    }
}

impl UnsupervisedEstimator for SpectralClustering {
    type Labels = Vec<usize>;

    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        SpectralClustering::fit(self, x)
    }

    fn predict(&self, _x: &Matrix<f32>) -> Self::Labels {
        assert!(self.is_fitted(), "Model not fitted. Call fit() first.");
        self.labels
            .as_ref()
            .expect("Labels must exist after successful fit")
            .clone()
    }
}

impl Default for SpectralClustering {
    fn default() -> Self {
        Self::new(2)
    }
}

#[cfg(test)]
#[path = "tests_spectral_contract.rs"]
mod tests_spectral_contract;
