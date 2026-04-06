//! Matrix operations for Trueno
//!
//! Provides 2D matrix operations with SIMD optimization for linear algebra,
//! machine learning, and scientific computing.
//!
//! ## Module Structure (PMAT-018)
//!
//! Operations are organized into domain-specific submodules:
//! - **ops/storage**: Constructors and element access
//! - **ops/arithmetic**: Matrix multiplication
//! - **ops/linear**: Transpose, matvec, vecmat
//! - **ops/ml_ops**: Convolution, embedding, pooling
//!
//! # Example
//!
//! ```
//! use trueno::Matrix;
//!
//! // Create a 2x3 matrix
//! let m = Matrix::zeros(2, 3);
//! assert_eq!(m.rows(), 2);
//! assert_eq!(m.cols(), 3);
//! ```

use crate::Backend;

mod ops;

#[cfg(test)]
mod tests;

/// A 2D matrix with row-major storage
///
/// Data is stored in row-major format (C-style), where consecutive elements
/// in memory belong to the same row. This is compatible with NumPy's default
/// layout and optimal for cache locality when accessing rows.
///
/// # Storage Layout
///
/// For a 2x3 matrix:
/// ```text
/// [[a, b, c],
///  [d, e, f]]
/// ```
/// Data is stored as: [a, b, c, d, e, f]
///
/// # Example
///
/// ```
/// use trueno::Matrix;
///
/// let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// assert_eq!(m.get(0, 0), Some(&1.0));
/// assert_eq!(m.get(0, 1), Some(&2.0));
/// assert_eq!(m.get(1, 0), Some(&3.0));
/// assert_eq!(m.get(1, 1), Some(&4.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: Vec<T>,
    pub(crate) backend: Backend,
}
