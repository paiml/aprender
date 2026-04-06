//! Public API methods and iterator for `SymmetricEigen`.

use super::SymmetricEigen;
use crate::{Backend, Matrix, TruenoError, Vector};

impl SymmetricEigen {
    /// Returns the eigenvalues in descending order
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::{Matrix, SymmetricEigen};
    ///
    /// let m = Matrix::from_vec(2, 2, vec![3.0, 1.0, 1.0, 3.0])?;
    /// let eigen = SymmetricEigen::new(&m)?;
    ///
    /// let values = eigen.eigenvalues();
    /// assert!((values[0] - 4.0).abs() < 1e-5);  // λ₁ = 4
    /// assert!((values[1] - 2.0).abs() < 1e-5);  // λ₂ = 2
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn eigenvalues(&self) -> &[f32] {
        &self.eigenvalues
    }

    /// Returns the eigenvector matrix
    ///
    /// Columns are eigenvectors, ordered to correspond with eigenvalues.
    /// Column `i` is the eigenvector for `eigenvalues()[i]`.
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::{Matrix, SymmetricEigen};
    ///
    /// let m = Matrix::identity(3);
    /// let eigen = SymmetricEigen::new(&m)?;
    ///
    /// // Identity matrix has eigenvectors that are the standard basis
    /// let vectors = eigen.eigenvectors();
    /// assert_eq!(vectors.rows(), 3);
    /// assert_eq!(vectors.cols(), 3);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn eigenvectors(&self) -> &Matrix<f32> {
        &self.eigenvectors
    }

    /// Returns an iterator over (eigenvalue, eigenvector) pairs
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::{Matrix, SymmetricEigen};
    ///
    /// let m = Matrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 1.0])?;
    /// let eigen = SymmetricEigen::new(&m)?;
    ///
    /// for (value, vector) in eigen.iter() {
    ///     println!("λ = {}, v = {:?}", value, vector.as_slice());
    /// }
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn iter(&self) -> EigenIterator<'_> {
        EigenIterator { eigen: self, index: 0 }
    }

    /// Returns the number of eigenvalue/eigenvector pairs
    pub fn len(&self) -> usize {
        self.eigenvalues.len()
    }

    /// Returns true if there are no eigenvalues
    pub fn is_empty(&self) -> bool {
        self.eigenvalues.is_empty()
    }

    /// Returns the backend used for computation
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Get a specific eigenvector by index
    ///
    /// # Arguments
    ///
    /// * `i` - Index of the eigenvector (0 = largest eigenvalue)
    ///
    /// # Returns
    ///
    /// The eigenvector as a Vector, or None if index out of bounds
    pub fn eigenvector(&self, i: usize) -> Option<Vector<f32>> {
        contract_pre_eigenvector!();
        if i >= self.eigenvalues.len() {
            return None;
        }

        let n = self.eigenvectors.rows();
        let mut data = Vec::with_capacity(n);

        for row in 0..n {
            if let Some(&val) = self.eigenvectors.get(row, i) {
                data.push(val);
            }
        }

        Some(Vector::from_slice(&data))
    }

    /// Reconstruct the original matrix from eigendecomposition
    ///
    /// Computes `V × D × V^T` where D is the diagonal matrix of eigenvalues.
    /// This is useful for verifying the decomposition accuracy.
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::{Matrix, SymmetricEigen};
    ///
    /// let m = Matrix::from_vec(2, 2, vec![4.0, 2.0, 2.0, 4.0])?;
    /// let eigen = SymmetricEigen::new(&m)?;
    /// let reconstructed = eigen.reconstruct()?;
    ///
    /// // Should be approximately equal to original
    /// assert!((reconstructed.get(0, 0).unwrap() - 4.0).abs() < 1e-5);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn reconstruct(&self) -> Result<Matrix<f32>, TruenoError> {
        let n = self.eigenvalues.len();

        // V × D (scale each column by its eigenvalue)
        let mut vd_data = vec![0.0f32; n * n];
        for i in 0..n {
            let lambda = self.eigenvalues[i];
            for j in 0..n {
                if let Some(&v) = self.eigenvectors.get(j, i) {
                    vd_data[j * n + i] = v * lambda;
                }
            }
        }

        let vd = Matrix::from_vec(n, n, vd_data)?;
        let vt = self.eigenvectors.transpose();

        vd.matmul(&vt)
    }
}

/// Iterator over eigenvalue-eigenvector pairs
pub struct EigenIterator<'a> {
    pub(super) eigen: &'a SymmetricEigen,
    pub(super) index: usize,
}

impl<'a> Iterator for EigenIterator<'a> {
    type Item = (f32, Vector<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.eigen.len() {
            return None;
        }

        let value = self.eigen.eigenvalues[self.index];
        let vector = self.eigen.eigenvector(self.index)?;
        self.index += 1;

        Some((value, vector))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.eigen.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for EigenIterator<'a> {}
