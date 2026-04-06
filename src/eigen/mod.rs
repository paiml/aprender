//! Eigendecomposition for symmetric matrices
//!
//! Provides SIMD-accelerated eigenvalue and eigenvector computation for symmetric
//! (Hermitian) matrices, enabling PCA, spectral clustering, and other algorithms
//! without external dependencies like nalgebra.
//!
//! # Algorithm
//!
//! Uses the Jacobi eigenvalue algorithm, which is numerically stable and well-suited
//! for SIMD parallelization. For large matrices (>1000 dimensions), GPU acceleration
//! is available via wgpu.
//!
//! # Example
//!
//! ```
//! use trueno::{Matrix, SymmetricEigen};
//!
//! // Create a symmetric positive definite matrix
//! let cov = Matrix::from_vec(2, 2, vec![
//!     2.0, 1.0,
//!     1.0, 2.0,
//! ])?;
//!
//! let eigen = SymmetricEigen::new(&cov)?;
//!
//! // Eigenvalues in descending order (PCA convention)
//! let values = eigen.eigenvalues();
//! assert!((values[0] - 3.0).abs() < 1e-6);  // λ₁ = 3
//! assert!((values[1] - 1.0).abs() < 1e-6);  // λ₂ = 1
//! # Ok::<(), trueno::TruenoError>(())
//! ```

mod methods;
#[cfg(test)]
mod tests;

pub use methods::EigenIterator;

use crate::{Backend, Matrix, TruenoError};

/// Maximum number of sweeps for Jacobi algorithm convergence
/// Each sweep processes all n(n-1)/2 off-diagonal elements once
/// Typically converges in 5-10 sweeps for well-conditioned matrices
const MAX_JACOBI_SWEEPS: usize = 50;

/// Convergence threshold for off-diagonal elements (relative to Frobenius norm)
const CONVERGENCE_THRESHOLD: f32 = 1e-7;

/// GPU threshold - use wgpu for matrices larger than this
#[allow(dead_code)]
const GPU_THRESHOLD: usize = 1000;

/// Symmetric matrix eigendecomposition
///
/// Computes eigenvalues and eigenvectors for symmetric (Hermitian) matrices.
/// Eigenvalues are returned in descending order (largest first), which is the
/// convention used in PCA and most dimensionality reduction algorithms.
///
/// # Properties
///
/// For a symmetric matrix A, the decomposition satisfies:
/// - `A = V × D × V^T` where D is diagonal with eigenvalues
/// - Eigenvectors are orthonormal: `V^T × V = I`
/// - Eigenvalues are real (guaranteed for symmetric matrices)
///
/// # Performance
///
/// - SIMD-accelerated Jacobi rotations for CPU
/// - GPU compute shaders for matrices >1000 dimensions (with `gpu` feature)
/// - O(n³) time complexity, O(n²) space complexity
#[derive(Debug, Clone)]
pub struct SymmetricEigen {
    /// Eigenvalues sorted in descending order
    pub(crate) eigenvalues: Vec<f32>,
    /// Eigenvectors as columns (column i corresponds to eigenvalue i)
    pub(crate) eigenvectors: Matrix<f32>,
    /// Sorting indices mapping original to sorted order
    pub(crate) _sort_indices: Vec<usize>,
    /// Backend used for computation
    pub(crate) backend: Backend,
}

impl SymmetricEigen {
    /// Computes eigendecomposition of a symmetric matrix
    ///
    /// # Arguments
    ///
    /// * `matrix` - A symmetric square matrix
    ///
    /// # Returns
    ///
    /// `SymmetricEigen` containing eigenvalues (descending) and eigenvectors
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if matrix is not square
    /// - `InvalidInput` if matrix is empty
    /// - `InvalidInput` if algorithm fails to converge
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::{Matrix, SymmetricEigen};
    ///
    /// let m = Matrix::from_vec(3, 3, vec![
    ///     4.0, 2.0, 0.0,
    ///     2.0, 5.0, 3.0,
    ///     0.0, 3.0, 6.0,
    /// ])?;
    ///
    /// let eigen = SymmetricEigen::new(&m)?;
    /// assert_eq!(eigen.eigenvalues().len(), 3);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn new(matrix: &Matrix<f32>) -> Result<Self, TruenoError> {
        // Validate input
        if matrix.rows() != matrix.cols() {
            return Err(TruenoError::InvalidInput(format!(
                "Matrix must be square for eigendecomposition, got {}x{}",
                matrix.rows(),
                matrix.cols()
            )));
        }

        if matrix.rows() == 0 {
            return Err(TruenoError::InvalidInput(
                "Cannot compute eigendecomposition of empty matrix".to_string(),
            ));
        }

        let backend = Backend::select_best();

        // Dispatch to appropriate implementation based on matrix size and GPU availability
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            let n = matrix.rows();
            if n >= GPU_THRESHOLD && crate::backends::gpu::GpuBackend::is_available() {
                return Self::compute_gpu(matrix);
            }
        }

        // CPU implementation (SIMD-accelerated) - works on all platforms
        Self::compute_jacobi(matrix, backend)
    }

    /// CPU implementation using Jacobi eigenvalue algorithm
    ///
    /// The Jacobi algorithm iteratively applies Givens rotations to eliminate
    /// off-diagonal elements, converging to a diagonal matrix of eigenvalues.
    fn compute_jacobi(matrix: &Matrix<f32>, backend: Backend) -> Result<Self, TruenoError> {
        let n = matrix.rows();

        // Copy matrix data for in-place modification
        let mut a = matrix.as_slice().to_vec();

        // Compute initial Frobenius norm for relative convergence
        let frobenius_sq: f32 = a.iter().map(|x| x * x).sum();
        let tolerance = CONVERGENCE_THRESHOLD * frobenius_sq.sqrt().max(1.0);

        // Initialize eigenvectors to identity matrix
        let mut v = vec![0.0f32; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }

        // Jacobi iteration with sweep strategy
        // Each sweep processes all off-diagonal elements once
        for _sweep in 0..MAX_JACOBI_SWEEPS {
            // Cyclic Jacobi: process all pairs (i, j) where i < j
            let mut converged = true;

            for i in 0..n {
                for j in (i + 1)..n {
                    let aij = a[i * n + j];

                    // Skip if already small enough
                    if aij.abs() < tolerance {
                        continue;
                    }

                    converged = false;
                    Self::jacobi_rotate(&mut a, &mut v, n, i, j, backend);
                }
            }

            if converged {
                // Extract eigenvalues from diagonal
                let eigenvalues: Vec<f32> = (0..n).map(|i| a[i * n + i]).collect();

                // Sort eigenvalues in descending order
                let mut indices: Vec<usize> = (0..n).collect();
                indices.sort_by(|&i, &j| {
                    eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap_or(std::cmp::Ordering::Equal)
                });

                // Reorder eigenvalues
                let sorted_eigenvalues: Vec<f32> =
                    indices.iter().map(|&i| eigenvalues[i]).collect();

                // Create eigenvector matrix with sorted columns
                let mut eigenvector_data = vec![0.0f32; n * n];
                for (new_col, &old_col) in indices.iter().enumerate() {
                    for row in 0..n {
                        eigenvector_data[row * n + new_col] = v[row * n + old_col];
                    }
                }

                let eigenvectors = Matrix::from_vec(n, n, eigenvector_data)?;

                return Ok(SymmetricEigen {
                    eigenvalues: sorted_eigenvalues,
                    eigenvectors,
                    _sort_indices: indices,
                    backend,
                });
            }
        }

        // Failed to converge - this shouldn't happen for well-conditioned matrices
        Err(TruenoError::InvalidInput(format!(
            "Jacobi algorithm failed to converge after {} sweeps",
            MAX_JACOBI_SWEEPS
        )))
    }

    /// Find the largest off-diagonal element (unused in cyclic Jacobi, kept for classic Jacobi)
    #[inline]
    fn _find_max_off_diagonal(a: &[f32], n: usize) -> (usize, usize, f32) {
        let mut max_val = 0.0f32;
        let mut p = 0;
        let mut q = 1;

        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        (p, q, max_val)
    }

    /// Apply Jacobi rotation to zero out a[p][q] and a[q][p]
    ///
    /// Uses the numerically stable formula from:
    /// Golub & Van Loan, "Matrix Computations", 4th Edition
    #[inline]
    fn jacobi_rotate(
        a: &mut [f32],
        v: &mut [f32],
        n: usize,
        p: usize,
        q: usize,
        _backend: Backend,
    ) {
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        // Skip if already zero
        if apq.abs() < 1e-15 {
            return;
        }

        // Compute rotation parameters using numerically stable formula
        // tau = (aqq - app) / (2 * apq)
        // t = sign(tau) / (|tau| + sqrt(1 + tau^2))  (avoiding catastrophic cancellation)
        // c = 1 / sqrt(1 + t^2)
        // s = t * c
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };

        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Update diagonal elements
        a[p * n + p] = app - t * apq;
        a[q * n + q] = aqq + t * apq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        // Update off-diagonal elements in rows/columns p and q
        for k in 0..n {
            if k != p && k != q {
                let akp = a[k * n + p];
                let akq = a[k * n + q];
                a[k * n + p] = c * akp - s * akq;
                a[p * n + k] = a[k * n + p];
                a[k * n + q] = s * akp + c * akq;
                a[q * n + k] = a[k * n + q];
            }
        }

        // Update eigenvector matrix
        for k in 0..n {
            let vkp = v[k * n + p];
            let vkq = v[k * n + q];
            v[k * n + p] = c * vkp - s * vkq;
            v[k * n + q] = s * vkp + c * vkq;
        }
    }

    /// GPU implementation for large matrices
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    fn compute_gpu(matrix: &Matrix<f32>) -> Result<Self, TruenoError> {
        use crate::backends::gpu::GpuBackend;

        let n = matrix.rows();
        let mut gpu = GpuBackend::new();

        // Execute eigendecomposition on GPU
        let (eigenvalues, eigenvector_data) =
            gpu.symmetric_eigen(matrix.as_slice(), n).map_err(|e| {
                TruenoError::InvalidInput(format!("GPU eigendecomposition failed: {}", e))
            })?;

        // Sort eigenvalues in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_eigenvalues: Vec<f32> = indices.iter().map(|&i| eigenvalues[i]).collect();

        // Reorder eigenvectors
        let mut sorted_eigenvector_data = vec![0.0f32; n * n];
        for (new_col, &old_col) in indices.iter().enumerate() {
            for row in 0..n {
                sorted_eigenvector_data[row * n + new_col] = eigenvector_data[row * n + old_col];
            }
        }

        let eigenvectors = Matrix::from_vec(n, n, sorted_eigenvector_data)?;

        Ok(SymmetricEigen {
            eigenvalues: sorted_eigenvalues,
            eigenvectors,
            _sort_indices: indices,
            backend: Backend::GPU,
        })
    }
}
