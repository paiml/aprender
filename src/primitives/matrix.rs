//! Matrix type for 2D numeric data.

use super::Vector;
use serde::{Deserialize, Serialize};

/// A 2D matrix of floating-point values (row-major storage).
///
/// # Examples
///
/// ```
/// use aprender::primitives::Matrix;
///
/// let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("data length matches rows * cols");
/// assert_eq!(m.shape(), (2, 3));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Copy> Matrix<T> {
    /// Creates a new matrix from a vector of data.
    ///
    /// # Errors
    ///
    /// Returns an error if data length doesn't match rows * cols.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Result<Self, &'static str> {
        if data.len() != rows * cols {
            return Err("Data length must equal rows * cols");
        }
        Ok(Self { data, rows, cols })
    }

    /// Returns the shape as (rows, cols).
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the number of rows.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn n_cols(&self) -> usize {
        self.cols
    }

    /// Gets element at (row, col).
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> T {
        self.data[row * self.cols + col]
    }

    /// Sets element at (row, col).
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds.
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        self.data[row * self.cols + col] = value;
    }

    /// Returns a row as a Vector.
    #[must_use]
    pub fn row(&self, row_idx: usize) -> Vector<T> {
        let start = row_idx * self.cols;
        let end = start + self.cols;
        Vector::from_slice(&self.data[start..end])
    }

    /// Returns a column as a Vector.
    #[must_use]
    pub fn column(&self, col_idx: usize) -> Vector<T> {
        let data: Vec<T> = (0..self.rows)
            .map(|row| self.data[row * self.cols + col_idx])
            .collect();
        Vector::from_vec(data)
    }

    /// Returns the underlying data as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
}

impl Matrix<f32> {
    /// Creates a matrix of zeros.
    #[must_use]
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Creates a matrix of ones.
    #[must_use]
    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![1.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Creates an identity matrix.
    #[must_use]
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self {
            data,
            rows: n,
            cols: n,
        }
    }

    /// Transposes the matrix.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut data = vec![0.0; self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        Self {
            data,
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Matrix-matrix multiplication.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions don't match.
    pub fn matmul(&self, other: &Self) -> Result<Self, &'static str> {
        if self.cols != other.rows {
            return Err("Matrix dimensions don't match for multiplication");
        }

        let mut result = vec![0.0; self.rows * other.cols];
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result[i * other.cols + j] = sum;
            }
        }

        Ok(Self {
            data: result,
            rows: self.rows,
            cols: other.cols,
        })
    }

    /// Matrix-vector multiplication.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions don't match.
    pub fn matvec(&self, vec: &Vector<f32>) -> Result<Vector<f32>, &'static str> {
        if self.cols != vec.len() {
            return Err("Matrix columns must match vector length");
        }

        let result: Vec<f32> = (0..self.rows)
            .map(|i| {
                let row = self.row(i);
                row.dot(vec)
            })
            .collect();

        Ok(Vector::from_vec(result))
    }

    /// Adds another matrix element-wise.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions don't match.
    pub fn add(&self, other: &Self) -> Result<Self, &'static str> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrix dimensions must match for addition");
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Self {
            data,
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Subtracts another matrix element-wise.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions don't match.
    pub fn sub(&self, other: &Self) -> Result<Self, &'static str> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrix dimensions must match for subtraction");
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Ok(Self {
            data,
            rows: self.rows,
            cols: self.cols,
        })
    }

    /// Multiplies each element by a scalar.
    #[must_use]
    pub fn mul_scalar(&self, scalar: f32) -> Self {
        Self {
            data: self.data.iter().map(|x| x * scalar).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Solves the linear system Ax = b using Cholesky decomposition.
    ///
    /// The matrix must be symmetric positive definite.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square or not positive definite.
    pub fn cholesky_solve(&self, b: &Vector<f32>) -> Result<Vector<f32>, &'static str> {
        if self.rows != self.cols {
            return Err("Matrix must be square for Cholesky decomposition");
        }
        if self.rows != b.len() {
            return Err("Matrix rows must match vector length");
        }

        let n = self.rows;

        // Cholesky decomposition: A = L * L^T
        let mut l = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;

                if i == j {
                    for k in 0..j {
                        sum += l[j * n + k] * l[j * n + k];
                    }
                    let diag = self.get(j, j) - sum;
                    if diag <= 0.0 {
                        return Err("Matrix is not positive definite");
                    }
                    l[j * n + j] = diag.sqrt();
                } else {
                    for k in 0..j {
                        sum += l[i * n + k] * l[j * n + k];
                    }
                    l[i * n + j] = (self.get(i, j) - sum) / l[j * n + j];
                }
            }
        }

        // Forward substitution: L * y = b
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[i * n + j] * y[j];
            }
            y[i] = (b[i] - sum) / l[i * n + i];
        }

        // Backward substitution: L^T * x = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += l[j * n + i] * x[j];
            }
            x[i] = (y[i] - sum) / l[i * n + i];
        }

        Ok(Vector::from_vec(x))
    }
}

#[cfg(test)]
#[path = "matrix_tests.rs"]
mod tests;
