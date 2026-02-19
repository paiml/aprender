//! Independent Component Analysis (ICA)
//!
//! ICA is a computational technique for separating a multivariate signal into
//! additive, independent components. It's a form of blind source separation.
//!
//! # Algorithm: `FastICA`
//!
//! The `FastICA` algorithm (Hyvärinen & Oja, 2000) consists of:
//!
//! 1. **Centering**: Subtract mean from each feature
//! 2. **Whitening**: Decorrelate and normalize variance via eigendecomposition
//! 3. **Optimization**: Iteratively find directions of maximum non-Gaussianity
//!
//! # Mathematical Background
//!
//! Given observed data X = AS, where:
//! - X: n×p observed signals (mixed)
//! - A: p×p mixing matrix
//! - S: n×p independent sources
//!
//! ICA recovers W such that S ≈ XW, where W ≈ A^(-1).
//!
//! # Examples
//!
//! ```
//! use aprender::decomposition::ICA;
//! use aprender::primitives::Matrix;
//!
//! // Mixed signals (3 samples, 2 sources)
//! let mixed = Matrix::from_vec(3, 2, vec![
//!     1.0, 2.0,
//!     2.0, 1.0,
//!     3.0, 4.0,
//! ]).expect("Valid matrix");
//!
//! let mut ica = ICA::new(2); // 2 components
//! ica.fit(&mixed).expect("ICA should fit");
//!
//! let sources = ica.transform(&mixed).expect("Should transform");
//! ```

use crate::error::{AprenderError, Result};
use crate::primitives::{Matrix, Vector};

/// Independent Component Analysis using `FastICA` algorithm.
///
/// ICA separates multivariate signals into independent, non-Gaussian components.
#[derive(Debug, Clone)]
pub struct ICA {
    /// Number of components to extract
    n_components: usize,

    /// Maximum iterations for `FastICA`
    max_iter: usize,

    /// Convergence tolerance
    tol: f32,

    /// Random state for initialization
    random_state: Option<u64>,

    // Fitted parameters
    /// Whitening matrix (p × `n_components`)
    whitening_matrix: Option<Matrix<f32>>,

    /// Unmixing matrix (`n_components` × p)
    unmixing_matrix: Option<Matrix<f32>>,

    /// Mean of each feature
    mean: Option<Vector<f32>>,
}

impl ICA {
    /// Creates a new ICA model.
    ///
    /// # Arguments
    ///
    /// * `n_components` - Number of independent components to extract
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::decomposition::ICA;
    ///
    /// let ica = ICA::new(3); // Extract 3 components
    /// ```
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            max_iter: 200,
            tol: 1e-4,
            random_state: None,
            whitening_matrix: None,
            unmixing_matrix: None,
            mean: None,
        }
    }

    /// Sets the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance.
    #[must_use]
    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the random state for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Fits the ICA model to the data.
    ///
    /// # Arguments
    ///
    /// * `x` - Data matrix (n × p), where n is samples and p is features
    ///
    /// # Errors
    ///
    /// Returns error if data dimensions are invalid or optimization fails.
    // Contract: ica-v1, equation = "fastica"
    #[allow(clippy::similar_names)]
    pub fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let n = x.n_rows();
        let p = x.n_cols();

        if n == 0 || p == 0 {
            return Err(AprenderError::Other("Data cannot be empty".into()));
        }

        if self.n_components > p {
            return Err(AprenderError::Other(format!(
                "n_components ({}) cannot exceed number of features ({})",
                self.n_components, p
            )));
        }

        // Step 1: Center the data
        let (x_centered, mean) = Self::center_data(x)?;
        self.mean = Some(mean);

        // Step 2: Whiten the data
        let (x_whitened, whitening_matrix) = Self::whiten_data(&x_centered, self.n_components)?;
        self.whitening_matrix = Some(whitening_matrix);

        // Step 3: Run FastICA to find unmixing matrix
        let unmixing = self.fastica(&x_whitened)?;
        self.unmixing_matrix = Some(unmixing);

        Ok(())
    }

    /// Transforms data using the fitted ICA model.
    ///
    /// # Arguments
    ///
    /// * `x` - Data matrix (n × p)
    ///
    /// # Returns
    ///
    /// Independent components (n × `n_components`)
    pub fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| AprenderError::Other("Model not fitted. Call fit() first.".into()))?;

        let whitening = self
            .whitening_matrix
            .as_ref()
            .ok_or_else(|| AprenderError::Other("Model not fitted. Call fit() first.".into()))?;

        let unmixing = self
            .unmixing_matrix
            .as_ref()
            .ok_or_else(|| AprenderError::Other("Model not fitted. Call fit() first.".into()))?;

        let n = x.n_rows();
        let p = x.n_cols();

        if p != mean.len() {
            return Err(AprenderError::DimensionMismatch {
                expected: format!("{} features", mean.len()),
                actual: format!("{p} features in data"),
            });
        }

        // Center
        let mut x_centered_data = Vec::with_capacity(n * p);
        for i in 0..n {
            for j in 0..p {
                x_centered_data.push(x.get(i, j) - mean[j]);
            }
        }
        let x_centered = Matrix::from_vec(n, p, x_centered_data)
            .map_err(|e| AprenderError::Other(format!("Centering failed: {e}")))?;

        // Whiten
        let x_whitened = x_centered
            .matmul(whitening)
            .map_err(|e| AprenderError::Other(format!("Whitening failed: {e}")))?;

        // Apply unmixing
        x_whitened
            .matmul(unmixing)
            .map_err(|e| AprenderError::Other(format!("Unmixing failed: {e}")))
    }

    /// Centers data by subtracting column means.
    #[allow(clippy::needless_range_loop)]
    fn center_data(x: &Matrix<f32>) -> Result<(Matrix<f32>, Vector<f32>)> {
        let n = x.n_rows();
        let p = x.n_cols();

        // Compute column means
        let mut means = vec![0.0_f32; p];
        #[allow(clippy::needless_range_loop)]
        for j in 0..p {
            let mut sum = 0.0;
            for i in 0..n {
                sum += x.get(i, j);
            }
            means[j] = sum / n as f32;
        }

        // Center data
        let mut centered_data = Vec::with_capacity(n * p);
        for i in 0..n {
            for j in 0..p {
                centered_data.push(x.get(i, j) - means[j]);
            }
        }

        let centered = Matrix::from_vec(n, p, centered_data)
            .map_err(|e| AprenderError::Other(format!("Failed to center data: {e}")))?;

        Ok((centered, Vector::from_vec(means)))
    }

    /// Whitens data using eigendecomposition (ZCA whitening).
    ///
    /// Returns whitened data and whitening matrix.
    #[allow(clippy::similar_names)]
    #[allow(clippy::needless_range_loop)]
    fn whiten_data(
        x_centered: &Matrix<f32>,
        n_components: usize,
    ) -> Result<(Matrix<f32>, Matrix<f32>)> {
        let n = x_centered.n_rows();
        let p = x_centered.n_cols();

        // Compute covariance matrix: (1/n) X^T X
        let xt = x_centered.transpose();
        let cov = xt
            .matmul(x_centered)
            .map_err(|e| AprenderError::Other(format!("Covariance computation failed: {e}")))?;

        // Scale by 1/n
        let mut cov_data = vec![0.0_f32; p * p];
        for i in 0..p {
            for j in 0..p {
                cov_data[i * p + j] = cov.get(i, j) / n as f32;
            }
        }
        let cov_scaled = Matrix::from_vec(p, p, cov_data)
            .map_err(|e| AprenderError::Other(format!("Covariance scaling failed: {e}")))?;

        // Eigen decomposition (simplified - use power iteration for top components)
        let (eigenvalues, eigenvectors) = Self::eigen_decomposition(&cov_scaled, n_components)?;

        // Compute whitening matrix: V Λ^(-1/2)
        // where V are eigenvectors and Λ are eigenvalues
        let mut whitening_data = Vec::with_capacity(p * n_components);
        for j in 0..n_components {
            let scale = 1.0 / eigenvalues[j].sqrt();
            for i in 0..p {
                whitening_data.push(eigenvectors.get(i, j) * scale);
            }
        }
        let whitening_matrix = Matrix::from_vec(p, n_components, whitening_data)
            .map_err(|e| AprenderError::Other(format!("Whitening matrix creation failed: {e}")))?;

        // Whiten data: X_white = X_centered * whitening_matrix
        let x_whitened = x_centered
            .matmul(&whitening_matrix)
            .map_err(|e| AprenderError::Other(format!("Data whitening failed: {e}")))?;

        Ok((x_whitened, whitening_matrix))
    }

    /// Simple eigen decomposition using power iteration for top k eigenvalues/vectors.
    #[allow(clippy::needless_range_loop)]
    fn eigen_decomposition(matrix: &Matrix<f32>, k: usize) -> Result<(Vec<f32>, Matrix<f32>)> {
        let n = matrix.n_rows();

        if matrix.n_cols() != n {
            return Err(AprenderError::Other(
                "Eigendecomposition requires square matrix".into(),
            ));
        }

        let mut eigenvalues = Vec::with_capacity(k);
        let mut eigenvectors_data = Vec::with_capacity(n * k);

        let mut residual = matrix.clone();

        for _ in 0..k {
            // Power iteration to find dominant eigenvector
            let (eigenvalue, eigenvector) = Self::power_iteration(&residual, 100)?;

            eigenvalues.push(eigenvalue);
            eigenvectors_data.extend(eigenvector.as_slice());

            // Deflate: A' = A - λvv^T
            let mut new_residual_data = vec![0.0_f32; n * n];
            for i in 0..n {
                for j in 0..n {
                    let deflation = eigenvalue * eigenvector[i] * eigenvector[j];
                    new_residual_data[i * n + j] = residual.get(i, j) - deflation;
                }
            }
            residual = Matrix::from_vec(n, n, new_residual_data)
                .map_err(|e| AprenderError::Other(format!("Deflation failed: {e}")))?;
        }

        let eigenvectors = Matrix::from_vec(n, k, eigenvectors_data).map_err(|e| {
            AprenderError::Other(format!("Eigenvector matrix creation failed: {e}"))
        })?;

        Ok((eigenvalues, eigenvectors))
    }

    /// Power iteration to find dominant eigenvector.
    #[allow(clippy::needless_range_loop)]
    fn power_iteration(matrix: &Matrix<f32>, max_iter: usize) -> Result<(f32, Vector<f32>)> {
        let n = matrix.n_rows();

        // Initialize with random vector (simple: all ones, then normalize)
        let mut v = vec![1.0_f32; n];
        let norm = (v.iter().map(|x| x * x).sum::<f32>()).sqrt();
        for val in &mut v {
            *val /= norm;
        }

        let mut eigenvalue = 0.0;

        for _ in 0..max_iter {
            // v_new = A * v
            let mut v_new = vec![0.0_f32; n];
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += matrix.get(i, j) * v[j];
                }
                v_new[i] = sum;
            }

            // Normalize
            let norm = (v_new.iter().map(|x| x * x).sum::<f32>()).sqrt();
            if norm < 1e-10 {
                return Err(AprenderError::Other(
                    "Power iteration converged to zero vector".into(),
                ));
            }

            for val in &mut v_new {
                *val /= norm;
            }

            eigenvalue = norm;
            v = v_new;
        }

        Ok((eigenvalue, Vector::from_vec(v)))
    }

    /// `FastICA` algorithm to find unmixing matrix.
    ///
    /// Uses deflation approach with tanh nonlinearity.
    #[allow(clippy::similar_names)]
    #[allow(clippy::needless_range_loop)]
    fn fastica(&self, x_white: &Matrix<f32>) -> Result<Matrix<f32>> {
        let n = x_white.n_rows();
        let p = x_white.n_cols(); // Should equal n_components after whitening

        let mut w_vectors = Vec::with_capacity(p * p);

        // Deflation: extract components one by one
        for comp in 0..p {
            // Initialize w randomly (simple: use component index for determinism)
            let mut w = vec![0.0_f32; p];
            w[comp % p] = 1.0;

            // Normalize
            let norm = (w.iter().map(|x| x * x).sum::<f32>()).sqrt();
            for val in &mut w {
                *val /= norm;
            }

            // Fixed-point iteration
            for _iter in 0..self.max_iter {
                // Compute w^T X^T for all samples
                let mut wtx = vec![0.0_f32; n];
                for i in 0..n {
                    let mut sum = 0.0;
                    for j in 0..p {
                        sum += w[j] * x_white.get(i, j);
                    }
                    wtx[i] = sum;
                }

                // E[X g(w^T X)] where g = tanh
                let mut ex_g = vec![0.0_f32; p];
                for j in 0..p {
                    let mut sum = 0.0;
                    for i in 0..n {
                        let g = wtx[i].tanh(); // g(w^T x)
                        sum += x_white.get(i, j) * g;
                    }
                    ex_g[j] = sum / n as f32;
                }

                // E[g'(w^T X)] where g' = 1 - tanh^2
                let mut eg_prime = 0.0;
                for i in 0..n {
                    let tanh_val = wtx[i].tanh();
                    eg_prime += 1.0 - tanh_val * tanh_val;
                }
                eg_prime /= n as f32;

                // w_new = E[X g(w^T X)] - E[g'(w^T X)] w
                let mut w_new = vec![0.0_f32; p];
                for j in 0..p {
                    w_new[j] = ex_g[j] - eg_prime * w[j];
                }

                // Orthogonalize against previous components
                for prev_comp in 0..comp {
                    let mut dot = 0.0;
                    for j in 0..p {
                        dot += w_new[j] * w_vectors[prev_comp * p + j];
                    }
                    for j in 0..p {
                        w_new[j] -= dot * w_vectors[prev_comp * p + j];
                    }
                }

                // Normalize
                let norm = (w_new.iter().map(|x| x * x).sum::<f32>()).sqrt();
                if norm < 1e-10 {
                    return Err(AprenderError::Other(
                        "FastICA failed: w converged to zero".into(),
                    ));
                }
                for val in &mut w_new {
                    *val /= norm;
                }

                // Check convergence
                let mut dot = 0.0;
                for j in 0..p {
                    dot += w[j] * w_new[j];
                }

                if (1.0 - dot.abs()) < self.tol {
                    w = w_new;
                    break;
                }

                w = w_new;
            }

            // Store this component
            w_vectors.extend(&w);
        }

        // Unmixing matrix is W^T (each row is a component)
        Matrix::from_vec(p, p, w_vectors)
            .map_err(|e| AprenderError::Other(format!("Failed to create unmixing matrix: {e}")))
    }
}

#[cfg(test)]
#[path = "ica_tests.rs"]
mod tests;
