//! Matrix storage and construction operations
//!
//! This module provides storage-related operations for Matrix:
//! - Constructors: `new()`, `from_vec()`, `from_slice()`, `zeros()`, `identity()`
//! - Accessors: `rows()`, `cols()`, `shape()`, `get()`, `get_mut()`, `as_slice()`
//!
//! ## Domain Separation (PMAT-018)
//!
//! Storage is separate from Algebra (arithmetic operations).
//! A matrix's memory layout is independent of its mathematical operations.

use crate::{Backend, TruenoError};

use super::super::Matrix;

impl std::ops::Index<(usize, usize)> for Matrix<f32> {
    type Output = f32;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row * self.cols + col]
    }
}

impl Matrix<f32> {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Creates a new matrix with uninitialized values
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Returns
    ///
    /// A new matrix with dimensions `rows x cols` containing uninitialized values
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Matrix;
    ///
    /// let m = Matrix::new(3, 4);
    /// assert_eq!(m.rows(), 3);
    /// assert_eq!(m.cols(), 4);
    /// ```
    pub fn new(rows: usize, cols: usize) -> Self {
        let backend = Backend::select_best();
        Matrix { rows, cols, data: vec![0.0; rows * cols], backend }
    }

    /// Creates a matrix from a vector of data
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// * `data` - Vector containing matrix elements in row-major order
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `data.len() != rows * cols`
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Matrix;
    ///
    /// let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// assert_eq!(m.rows(), 2);
    /// assert_eq!(m.cols(), 2);
    /// ```
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self, TruenoError> {
        if data.len() != rows * cols {
            return Err(TruenoError::InvalidInput(format!(
                "Data length {} does not match matrix dimensions {}x{} (expected {})",
                data.len(),
                rows,
                cols,
                rows * cols
            )));
        }

        let backend = Backend::select_best();
        Ok(Matrix { rows, cols, data, backend })
    }

    /// Creates a matrix from a vector with a specific backend
    ///
    /// This is useful for testing specific SIMD code paths.
    pub fn from_vec_with_backend(
        rows: usize,
        cols: usize,
        data: Vec<f32>,
        backend: Backend,
    ) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length {} does not match matrix dimensions {}x{}",
            data.len(),
            rows,
            cols
        );
        Matrix { rows, cols, data, backend }
    }

    /// Creates a matrix from a slice by copying the data
    ///
    /// This is a convenience method that copies the slice into an owned vector.
    /// For zero-copy scenarios, consider using the data directly with `from_vec`
    /// if you already have an owned `Vec`.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// * `data` - Slice containing matrix elements in row-major order
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `data.len() != rows * cols`
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Matrix;
    ///
    /// let data = [1.0, 2.0, 3.0, 4.0];
    /// let m = Matrix::from_slice(2, 2, &data).unwrap();
    /// assert_eq!(m.get(0, 0), Some(&1.0));
    /// ```
    pub fn from_slice(rows: usize, cols: usize, data: &[f32]) -> Result<Self, TruenoError> {
        Self::from_vec(rows, cols, data.to_vec())
    }

    /// Creates a matrix filled with zeros
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Matrix;
    ///
    /// let m = Matrix::zeros(3, 3);
    /// assert_eq!(m.get(1, 1), Some(&0.0));
    /// ```
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix::new(rows, cols)
    }

    /// Creates a matrix filled with zeros using a specific backend
    /// (Internal use only - reuses backend from parent matrix)
    pub(crate) fn zeros_with_backend(rows: usize, cols: usize, backend: Backend) -> Self {
        Matrix { rows, cols, data: vec![0.0; rows * cols], backend }
    }

    /// Creates an identity matrix (square matrix with 1s on diagonal)
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Matrix;
    ///
    /// let m = Matrix::identity(3);
    /// assert_eq!(m.get(0, 0), Some(&1.0));
    /// assert_eq!(m.get(0, 1), Some(&0.0));
    /// assert_eq!(m.get(1, 1), Some(&1.0));
    /// ```
    pub fn identity(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        let backend = Backend::select_best();
        Matrix { rows: n, cols: n, data, backend }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Returns the number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the shape as (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Gets a reference to an element at (row, col)
    ///
    /// Returns `None` if indices are out of bounds
    pub fn get(&self, row: usize, col: usize) -> Option<&f32> {
        if row >= self.rows || col >= self.cols {
            None
        } else {
            self.data.get(row * self.cols + col)
        }
    }

    /// Gets a mutable reference to an element at (row, col)
    ///
    /// Returns `None` if indices are out of bounds
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut f32> {
        if row >= self.rows || col >= self.cols {
            None
        } else {
            let idx = row * self.cols + col;
            self.data.get_mut(idx)
        }
    }

    /// Returns a reference to the underlying data
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Returns the backend used by this matrix
    pub fn backend(&self) -> Backend {
        self.backend
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_creates_zero_matrix() {
        let m = Matrix::new(3, 4);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        assert!(m.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_from_vec_success() {
        let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(m.get(0, 0), Some(&1.0));
        assert_eq!(m.get(1, 1), Some(&4.0));
    }

    #[test]
    fn test_from_vec_dimension_mismatch() {
        let result = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_identity() {
        let m = Matrix::identity(3);
        assert_eq!(m.get(0, 0), Some(&1.0));
        assert_eq!(m.get(1, 1), Some(&1.0));
        assert_eq!(m.get(2, 2), Some(&1.0));
        assert_eq!(m.get(0, 1), Some(&0.0));
    }

    #[test]
    fn test_index_operator() {
        let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 4.0);
    }

    #[test]
    fn test_get_out_of_bounds() {
        let m = Matrix::new(2, 2);
        assert_eq!(m.get(2, 0), None);
        assert_eq!(m.get(0, 2), None);
    }

    #[test]
    fn test_get_mut() {
        let mut m = Matrix::new(2, 2);
        if let Some(val) = m.get_mut(1, 1) {
            *val = 42.0;
        }
        assert_eq!(m.get(1, 1), Some(&42.0));
    }
}
