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
mod tests {
    use super::*;

    #[test]
    fn test_from_vec() {
        let m = Matrix::from_vec(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("test data has correct dimensions: 2*3=6 elements");
        assert_eq!(m.shape(), (2, 3));
        assert!((m.get(0, 0) - 1.0).abs() < 1e-6);
        assert!((m.get(1, 2) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_vec_error() {
        let result = Matrix::from_vec(2, 3, vec![1.0_f32, 2.0, 3.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_zeros() {
        let m = Matrix::<f32>::zeros(2, 3);
        assert_eq!(m.shape(), (2, 3));
        assert!(m.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_eye() {
        let m = Matrix::<f32>::eye(3);
        assert!((m.get(0, 0) - 1.0).abs() < 1e-6);
        assert!((m.get(1, 1) - 1.0).abs() < 1e-6);
        assert!((m.get(2, 2) - 1.0).abs() < 1e-6);
        assert!((m.get(0, 1) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_transpose() {
        let m = Matrix::from_vec(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("test data has correct dimensions: 2*3=6 elements");
        let t = m.transpose();
        assert_eq!(t.shape(), (3, 2));
        assert!((t.get(0, 0) - 1.0).abs() < 1e-6);
        assert!((t.get(0, 1) - 4.0).abs() < 1e-6);
        assert!((t.get(2, 1) - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_row() {
        let m = Matrix::from_vec(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("test data has correct dimensions: 2*3=6 elements");
        let row = m.row(1);
        assert_eq!(row.len(), 3);
        assert!((row[0] - 4.0).abs() < 1e-6);
        assert!((row[1] - 5.0).abs() < 1e-6);
        assert!((row[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_column() {
        let m = Matrix::from_vec(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("test data has correct dimensions: 2*3=6 elements");
        let col = m.column(1);
        assert_eq!(col.len(), 2);
        assert!((col[0] - 2.0).abs() < 1e-6);
        assert!((col[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul() {
        // 2x3 * 3x2 = 2x2
        let a = Matrix::from_vec(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("test data has correct dimensions: 2*3=6 elements");
        let b = Matrix::from_vec(3, 2, vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0])
            .expect("test data has correct dimensions: 3*2=6 elements");
        let c = a
            .matmul(&b)
            .expect("matrix dimensions are compatible for multiplication: 2x3 * 3x2");

        assert_eq!(c.shape(), (2, 2));
        // c[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        assert!((c.get(0, 0) - 58.0).abs() < 1e-6);
        // c[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        assert!((c.get(0, 1) - 64.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_dimension_error() {
        let a = Matrix::from_vec(2, 3, vec![1.0_f32; 6])
            .expect("test data has correct dimensions: 2*3=6 elements");
        let b = Matrix::from_vec(2, 2, vec![1.0_f32; 4])
            .expect("test data has correct dimensions: 2*2=4 elements");
        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn test_matvec() {
        let m = Matrix::from_vec(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("test data has correct dimensions: 2*3=6 elements");
        let v = Vector::from_slice(&[1.0_f32, 2.0, 3.0]);
        let result = m
            .matvec(&v)
            .expect("matrix columns match vector length: both 3");

        assert_eq!(result.len(), 2);
        // result[0] = 1*1 + 2*2 + 3*3 = 14
        assert!((result[0] - 14.0).abs() < 1e-6);
        // result[1] = 4*1 + 5*2 + 6*3 = 32
        assert!((result[1] - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_add() {
        let a = Matrix::from_vec(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])
            .expect("test data has correct dimensions: 2*2=4 elements");
        let b = Matrix::from_vec(2, 2, vec![5.0_f32, 6.0, 7.0, 8.0])
            .expect("test data has correct dimensions: 2*2=4 elements");
        let c = a.add(&b).expect("both matrices have same dimensions: 2x2");

        assert!((c.get(0, 0) - 6.0).abs() < 1e-6);
        assert!((c.get(1, 1) - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_dimension_mismatch() {
        // Test that mismatched dimensions are detected (catches || → && mutation)
        let a = Matrix::from_vec(2, 2, vec![1.0_f32; 4])
            .expect("test data has correct dimensions: 2*2=4 elements");
        let b = Matrix::from_vec(3, 2, vec![1.0_f32; 6])
            .expect("test data has correct dimensions: 3*2=6 elements");
        assert!(a.add(&b).is_err());

        let c = Matrix::from_vec(2, 3, vec![1.0_f32; 6])
            .expect("test data has correct dimensions: 2*3=6 elements");
        assert!(a.add(&c).is_err());
    }

    #[test]
    fn test_sub() {
        // Test element-wise subtraction
        let a = Matrix::from_vec(2, 2, vec![10.0_f32, 8.0, 6.0, 12.0])
            .expect("test data has correct dimensions: 2*2=4 elements");
        let b = Matrix::from_vec(2, 2, vec![4.0_f32, 3.0, 2.0, 7.0])
            .expect("test data has correct dimensions: 2*2=4 elements");
        let c = a.sub(&b).expect("both matrices have same dimensions: 2x2");

        // Verify all elements: a[i] - b[i]
        assert!((c.get(0, 0) - 6.0).abs() < 1e-6); // 10 - 4 = 6
        assert!((c.get(0, 1) - 5.0).abs() < 1e-6); // 8 - 3 = 5
        assert!((c.get(1, 0) - 4.0).abs() < 1e-6); // 6 - 2 = 4
        assert!((c.get(1, 1) - 5.0).abs() < 1e-6); // 12 - 7 = 5
    }

    #[test]
    fn test_sub_dimension_mismatch_rows() {
        // Test that mismatched rows are detected
        let a = Matrix::from_vec(2, 2, vec![1.0_f32; 4])
            .expect("test data has correct dimensions: 2*2=4 elements");
        let b = Matrix::from_vec(3, 2, vec![1.0_f32; 6])
            .expect("test data has correct dimensions: 3*2=6 elements");
        assert!(a.sub(&b).is_err());
    }

    #[test]
    fn test_sub_dimension_mismatch_cols() {
        // Test that mismatched columns are detected
        let a = Matrix::from_vec(2, 2, vec![1.0_f32; 4])
            .expect("test data has correct dimensions: 2*2=4 elements");
        let b = Matrix::from_vec(2, 3, vec![1.0_f32; 6])
            .expect("test data has correct dimensions: 2*3=6 elements");
        assert!(a.sub(&b).is_err());
    }

    #[test]
    fn test_cholesky_solve() {
        // Solve A*x = b where A is symmetric positive definite
        // A = [[4, 2], [2, 3]]
        // b = [1, 2]
        // Solution: x = [-0.125, 0.75]
        let a = Matrix::from_vec(2, 2, vec![4.0_f32, 2.0, 2.0, 3.0])
            .expect("test data has correct dimensions: 2*2=4 elements");
        let b = Vector::from_slice(&[1.0_f32, 2.0]);
        let x = a
            .cholesky_solve(&b)
            .expect("matrix is square, symmetric positive definite, and vector matches size");

        assert_eq!(x.len(), 2);
        assert!((x[0] - (-0.125)).abs() < 1e-5);
        assert!((x[1] - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_cholesky_solve_3x3() {
        // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        // b = [1, 2, 3]
        let a = Matrix::from_vec(
            3,
            3,
            vec![4.0_f32, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
        )
        .expect("test data has correct dimensions: 3*3=9 elements");
        let b = Vector::from_slice(&[1.0_f32, 2.0, 3.0]);
        let x = a
            .cholesky_solve(&b)
            .expect("matrix is square, symmetric positive definite, and vector matches size");

        // Verify A*x ≈ b
        let result = a
            .matvec(&x)
            .expect("matrix columns match vector length: both 3");
        for i in 0..3 {
            assert!((result[i] - b[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_cholesky_solve_strict() {
        // Stricter test to catch arithmetic mutations in cholesky_solve
        // Uses a 4x4 SPD matrix to exercise all accumulation loops
        // A = [[4, 2, 1, 1],
        //      [2, 5, 2, 1],
        //      [1, 2, 6, 2],
        //      [1, 1, 2, 7]]
        // This is symmetric positive definite with non-trivial decomposition
        let a = Matrix::from_vec(
            4,
            4,
            vec![
                4.0_f32, 2.0, 1.0, 1.0, 2.0, 5.0, 2.0, 1.0, 1.0, 2.0, 6.0, 2.0, 1.0, 1.0, 2.0, 7.0,
            ],
        )
        .expect("test data has correct dimensions: 4*4=16 elements");
        let b = Vector::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let x = a
            .cholesky_solve(&b)
            .expect("matrix is square, symmetric positive definite, and vector matches size");

        // Verify A*x = b with very tight tolerance
        let result = a
            .matvec(&x)
            .expect("matrix columns match vector length: both 4");
        for i in 0..4 {
            assert!(
                (result[i] - b[i]).abs() < 1e-5,
                "Failed at index {}: expected {}, got {}",
                i,
                b[i],
                result[i]
            );
        }

        // Also test a 3x3 case with known solution
        // A = [[9, 3, 3], [3, 5, 1], [3, 1, 4]], b = [15, 9, 8] => x = [1, 1, 1]
        let a3 = Matrix::from_vec(3, 3, vec![9.0_f32, 3.0, 3.0, 3.0, 5.0, 1.0, 3.0, 1.0, 4.0])
            .expect("test data has correct dimensions: 3*3=9 elements");
        let b3 = Vector::from_slice(&[15.0_f32, 9.0, 8.0]);
        let x3 = a3
            .cholesky_solve(&b3)
            .expect("matrix is square, symmetric positive definite, and vector matches size");

        // Verify exact solution [1, 1, 1] with element-by-element check
        assert!((x3[0] - 1.0).abs() < 1e-6);
        assert!((x3[1] - 1.0).abs() < 1e-6);
        assert!((x3[2] - 1.0).abs() < 1e-6);

        // Additional verification: check that A*x3 = b3 with strict tolerance
        let verify3 = a3
            .matvec(&x3)
            .expect("matrix columns match vector length: both 3");
        assert!((verify3[0] - 15.0).abs() < 1e-6);
        assert!((verify3[1] - 9.0).abs() < 1e-6);
        assert!((verify3[2] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_mul_scalar() {
        let m = Matrix::from_vec(2, 2, vec![1.0_f32, 2.0, 3.0, 4.0])
            .expect("test data has correct dimensions: 2*2=4 elements");
        let result = m.mul_scalar(2.0);
        assert!((result.get(0, 0) - 2.0).abs() < 1e-6);
        assert!((result.get(1, 1) - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_set() {
        let mut m = Matrix::<f32>::zeros(2, 2);
        m.set(0, 1, 5.0);
        assert!((m.get(0, 1) - 5.0).abs() < 1e-6);
    }
}
