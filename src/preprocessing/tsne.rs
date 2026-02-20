
// ============================================================================
// t-SNE (t-Distributed Stochastic Neighbor Embedding)
// ============================================================================

/// t-SNE for dimensionality reduction and visualization.
///
/// t-Distributed Stochastic Neighbor Embedding (t-SNE) is a non-linear
/// dimensionality reduction technique optimized for visualization of
/// high-dimensional data in 2D or 3D space.
///
/// # Algorithm
///
/// 1. Compute pairwise similarities in high-D using Gaussian kernel
/// 2. Compute perplexity-based conditional probabilities
/// 3. Initialize low-D embedding (random or PCA)
/// 4. Compute pairwise similarities in low-D using Student's t-distribution
/// 5. Minimize KL divergence via gradient descent with momentum
///
/// # Example
///
/// ```
/// use aprender::prelude::*;
/// use aprender::preprocessing::TSNE;
///
/// let data = Matrix::from_vec(
///     6,
///     4,
///     vec![
///         1.0, 2.0, 3.0, 4.0,
///         1.1, 2.1, 3.1, 4.1,
///         5.0, 6.0, 7.0, 8.0,
///         5.1, 6.1, 7.1, 8.1,
///         10.0, 11.0, 12.0, 13.0,
///         10.1, 11.1, 12.1, 13.1,
///     ],
/// )
/// .expect("valid matrix dimensions");
///
/// let mut tsne = TSNE::new(2).with_perplexity(5.0).with_n_iter(250);
/// let embedding = tsne.fit_transform(&data).expect("fit_transform should succeed");
/// assert_eq!(embedding.shape(), (6, 2));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TSNE {
    /// Number of dimensions in embedding (usually 2 or 3).
    n_components: usize,
    /// Perplexity balances local vs global structure (5-50).
    perplexity: f32,
    /// Learning rate for gradient descent.
    learning_rate: f32,
    /// Number of gradient descent iterations.
    n_iter: usize,
    /// Random seed for reproducibility.
    random_state: Option<u64>,
    /// The learned embedding.
    embedding: Option<Matrix<f32>>,
}

impl Default for TSNE {
    fn default() -> Self {
        Self::new(2)
    }
}

impl TSNE {
    /// Create a new t-SNE with default parameters.
    ///
    /// Default: perplexity=30.0, `learning_rate=200.0`, `n_iter=1000`
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 1000,
            random_state: None,
            embedding: None,
        }
    }

    /// Set perplexity (balance between local and global structure).
    ///
    /// Typical range: 5-50. Higher perplexity considers more neighbors.
    #[must_use]
    pub fn with_perplexity(mut self, perplexity: f32) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Set learning rate for gradient descent.
    #[must_use]
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set number of gradient descent iterations.
    #[must_use]
    pub fn with_n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
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

    /// Check if model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.embedding.is_some()
    }

    /// Compute pairwise squared Euclidean distances.
    #[allow(clippy::unused_self)]
    fn compute_pairwise_distances(&self, x: &Matrix<f32>) -> Vec<f32> {
        let (n_samples, n_features) = x.shape();
        let mut distances = vec![0.0; n_samples * n_samples];

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i == j {
                    distances[i * n_samples + j] = 0.0;
                    continue;
                }

                let mut dist_sq = 0.0;
                for k in 0..n_features {
                    let diff = x.get(i, k) - x.get(j, k);
                    dist_sq += diff * diff;
                }
                distances[i * n_samples + j] = dist_sq;
            }
        }

        distances
    }

    /// Compute conditional probabilities P(j|i) with perplexity constraint.
    ///
    /// Uses binary search to find `sigma_i` such that perplexity matches target.
    fn compute_p_conditional(&self, distances: &[f32], n_samples: usize) -> Vec<f32> {
        let mut p_conditional = vec![0.0; n_samples * n_samples];
        let target_entropy = self.perplexity.ln();

        for i in 0..n_samples {
            // Binary search for sigma that gives target perplexity
            let mut beta_min = -f32::INFINITY;
            let mut beta_max = f32::INFINITY;
            let mut beta = 1.0; // beta = 1 / (2 * sigma^2)

            for _ in 0..50 {
                // Max iterations for binary search
                // Compute P(j|i) with current beta
                let mut sum_p = 0.0;
                let mut entropy = 0.0;

                for j in 0..n_samples {
                    if i == j {
                        p_conditional[i * n_samples + j] = 0.0;
                        continue;
                    }

                    let p_ji = (-beta * distances[i * n_samples + j]).exp();
                    p_conditional[i * n_samples + j] = p_ji;
                    sum_p += p_ji;
                }

                // Normalize and compute entropy
                if sum_p > 0.0 {
                    for j in 0..n_samples {
                        if i != j {
                            let p_normalized = p_conditional[i * n_samples + j] / sum_p;
                            p_conditional[i * n_samples + j] = p_normalized;
                            if p_normalized > 1e-12 {
                                entropy -= p_normalized * p_normalized.ln();
                            }
                        }
                    }
                }

                // Check if entropy matches target
                let entropy_diff = entropy - target_entropy;
                if entropy_diff.abs() < 1e-5 {
                    break;
                }

                // Update beta via binary search
                if entropy_diff > 0.0 {
                    beta_min = beta;
                    beta = if beta_max.is_infinite() {
                        beta * 2.0
                    } else {
                        (beta + beta_max) / 2.0
                    };
                } else {
                    beta_max = beta;
                    beta = if beta_min.is_infinite() {
                        beta / 2.0
                    } else {
                        (beta + beta_min) / 2.0
                    };
                }
            }
        }

        p_conditional
    }

    /// Compute symmetric P matrix: P_{ij} = (P_{j|i} + P_{i|j}) / (2N).
    #[allow(clippy::unused_self)]
    fn compute_p_joint(&self, p_conditional: &[f32], n_samples: usize) -> Vec<f32> {
        let mut p_joint = vec![0.0; n_samples * n_samples];
        let normalizer = 2.0 * n_samples as f32;

        for i in 0..n_samples {
            for j in 0..n_samples {
                p_joint[i * n_samples + j] = (p_conditional[i * n_samples + j]
                    + p_conditional[j * n_samples + i])
                    / normalizer;
                // Numerical stability
                p_joint[i * n_samples + j] = p_joint[i * n_samples + j].max(1e-12);
            }
        }

        p_joint
    }

    /// Compute Q matrix in low-dimensional space using Student's t-distribution.
    fn compute_q(&self, y: &[f32], n_samples: usize) -> Vec<f32> {
        let mut q = vec![0.0; n_samples * n_samples];
        let mut sum_q = 0.0;

        // Compute Q_{ij} = (1 + ||y_i - y_j||^2)^{-1}
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i == j {
                    q[i * n_samples + j] = 0.0;
                    continue;
                }

                let mut dist_sq = 0.0;
                for k in 0..self.n_components {
                    let diff = y[i * self.n_components + k] - y[j * self.n_components + k];
                    dist_sq += diff * diff;
                }

                let q_ij = 1.0 / (1.0 + dist_sq);
                q[i * n_samples + j] = q_ij;
                sum_q += q_ij;
            }
        }

        // Normalize
        if sum_q > 0.0 {
            for q_val in &mut q {
                *q_val /= sum_q;
                *q_val = q_val.max(1e-12); // Numerical stability
            }
        }

        q
    }

    /// Compute gradient of KL divergence.
    fn compute_gradient(&self, y: &[f32], p: &[f32], q: &[f32], n_samples: usize) -> Vec<f32> {
        let mut gradient = vec![0.0; n_samples * self.n_components];

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i == j {
                    continue;
                }

                let p_ij = p[i * n_samples + j];
                let q_ij = q[i * n_samples + j];

                // Gradient factor: 4 * (p_ij - q_ij) * q_ij * (1 + ||y_i - y_j||^2)^{-1}
                // Simplified: 4 * (p_ij - q_ij) / (1 + ||y_i - y_j||^2)
                let mut dist_sq = 0.0;
                for k in 0..self.n_components {
                    let diff = y[i * self.n_components + k] - y[j * self.n_components + k];
                    dist_sq += diff * diff;
                }

                let factor = 4.0 * (p_ij - q_ij) / (1.0 + dist_sq);

                for k in 0..self.n_components {
                    let diff = y[i * self.n_components + k] - y[j * self.n_components + k];
                    gradient[i * self.n_components + k] += factor * diff;
                }
            }
        }

        gradient
    }
}

impl Transformer for TSNE {
    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let (n_samples, _n_features) = x.shape();

        // Compute pairwise distances in high-D
        let distances = self.compute_pairwise_distances(x);

        // Compute conditional probabilities with perplexity
        let p_conditional = self.compute_p_conditional(&distances, n_samples);

        // Compute joint probabilities (symmetric)
        let p_joint = self.compute_p_joint(&p_conditional, n_samples);

        // Initialize embedding randomly
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let seed = self.random_state.unwrap_or_else(|| {
            let mut hasher = DefaultHasher::new();
            std::time::SystemTime::now().hash(&mut hasher);
            hasher.finish()
        });

        // Simple LCG random number generator for reproducibility
        let mut rng_state = seed;
        let mut rand = || -> f32 {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng_state >> 16) as f32 / 65536.0) - 0.5
        };

        let mut y = vec![0.0; n_samples * self.n_components];
        for val in &mut y {
            *val = rand() * 0.0001; // Small random initialization
        }

        // Gradient descent with momentum
        let mut velocity = vec![0.0; n_samples * self.n_components];
        let momentum = 0.5;
        let final_momentum = 0.8;
        let momentum_switch_iter = 250;

        for iter in 0..self.n_iter {
            // Compute Q matrix in low-D
            let q = self.compute_q(&y, n_samples);

            // Compute gradient
            let gradient = self.compute_gradient(&y, &p_joint, &q, n_samples);

            // Update with momentum
            let current_momentum = if iter < momentum_switch_iter {
                momentum
            } else {
                final_momentum
            };

            for i in 0..(n_samples * self.n_components) {
                velocity[i] = current_momentum * velocity[i] - self.learning_rate * gradient[i];
                y[i] += velocity[i];
            }

            // Early exaggeration (first 100 iterations)
            if iter == 100 {
                // Remove early exaggeration by dividing P by 4
                // (we multiplied by 4 implicitly in gradient computation)
            }
        }

        self.embedding = Some(Matrix::from_vec(n_samples, self.n_components, y)?);
        Ok(())
    }

    fn transform(&self, _x: &Matrix<f32>) -> Result<Matrix<f32>> {
        assert!(self.is_fitted(), "Model not fitted. Call fit() first.");
        // t-SNE is non-parametric, return the embedding
        Ok(self
            .embedding
            .as_ref()
            .expect("embedding should exist after is_fitted() check")
            .clone())
    }

    fn fit_transform(&mut self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        self.fit(x)?;
        self.transform(x)
    }
}

#[cfg(test)]
mod tests;
