//! Machine learning operations for Matrix
//!
//! This module provides ML-specific operations:
//! - `convolve2d()` - 2D convolution
//! - `embedding_lookup()` - Embedding table lookup
//! - `embedding_lookup_sparse()` - Embedding lookup with gradient tracking
//! - `max_pool2d()` - Max pooling
//! - `avg_pool2d()` - Average pooling
//! - `topk()` - Top-K selection
//! - `gather()` - Gather elements along axis
//! - `pad()` - Pad matrix with constant value

mod convolution;
mod pooling;

use crate::TruenoError;

use super::super::Matrix;

impl Matrix<f32> {
    /// Lookup embeddings by indices
    ///
    /// Performs embedding lookup where self is the embedding table with shape
    /// `[vocab_size, embed_dim]` and indices specify which rows to select.
    ///
    /// # Arguments
    ///
    /// * `indices` - Slice of indices into the embedding table
    ///
    /// # Returns
    ///
    /// A matrix with shape `[indices.len(), embed_dim]` containing the selected rows
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if any index is out of bounds
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Matrix;
    ///
    /// // Create embedding table: 4 words, 3-dimensional embeddings
    /// let embeddings = Matrix::from_vec(4, 3, vec![
    ///     1.0, 2.0, 3.0,   // word 0
    ///     4.0, 5.0, 6.0,   // word 1
    ///     7.0, 8.0, 9.0,   // word 2
    ///     10.0, 11.0, 12.0 // word 3
    /// ])?;
    ///
    /// // Lookup embeddings for indices [1, 3, 0]
    /// let result = embeddings.embedding_lookup(&[1, 3, 0])?;
    ///
    /// assert_eq!(result.rows(), 3);
    /// assert_eq!(result.cols(), 3);
    /// assert_eq!(result.get(0, 0), Some(&4.0)); // word 1
    /// assert_eq!(result.get(1, 0), Some(&10.0)); // word 3
    /// assert_eq!(result.get(2, 0), Some(&1.0)); // word 0
    /// # Ok(())
    /// # }
    /// ```
    pub fn embedding_lookup(&self, indices: &[usize]) -> Result<Matrix<f32>, TruenoError> {
        // Validate indices
        contract_pre_embedding_lookup!(indices);
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.rows {
                return Err(TruenoError::InvalidInput(format!(
                    "Index {} at position {} is out of bounds for embedding table with {} rows",
                    idx, i, self.rows
                )));
            }
        }

        // Handle empty indices
        if indices.is_empty() {
            return Ok(Matrix::zeros_with_backend(0, self.cols, self.backend));
        }

        // Allocate output matrix: [seq_len, embed_dim]
        // Uninit: every element gets written by copy_from_slice below.
        let seq_len = indices.len();
        let embed_dim = self.cols;
        let total = seq_len * embed_dim;
        let mut data: Vec<f32> = Vec::with_capacity(total);
        // SAFETY: Loop below writes every element via copy_from_slice.
        unsafe {
            data.set_len(total);
        }
        let mut result = Matrix { rows: seq_len, cols: embed_dim, data, backend: self.backend };

        // Copy rows from embedding table to result
        for (out_row, &idx) in indices.iter().enumerate() {
            let src_start = idx * embed_dim;
            let dst_start = out_row * embed_dim;

            result.data[dst_start..dst_start + embed_dim]
                .copy_from_slice(&self.data[src_start..src_start + embed_dim]);
        }

        contract_post_embedding_lookup!(&result.data);
        Ok(result)
    }

    /// Lookup embeddings with gradient tracking support (for training)
    ///
    /// Returns both the embeddings and a sparse gradient accumulator.
    /// This is useful for sparse gradient updates in training.
    ///
    /// # Arguments
    ///
    /// * `indices` - Slice of indices into the embedding table
    ///
    /// # Returns
    ///
    /// Tuple of (embeddings, unique_indices) where unique_indices can be used
    /// for sparse gradient updates
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if any index is out of bounds
    pub fn embedding_lookup_sparse(
        &self,
        indices: &[usize],
    ) -> Result<(Matrix<f32>, Vec<usize>), TruenoError> {
        let embeddings = self.embedding_lookup(indices)?;

        // Get unique indices for sparse gradient updates
        let mut unique: Vec<usize> = indices.to_vec();
        unique.sort_unstable();
        unique.dedup();

        Ok((embeddings, unique))
    }

    /// Top-K selection: returns the k largest elements and their indices
    ///
    /// Useful for beam search, sampling, and ranking operations.
    /// Searches row-major order and returns (values, indices) sorted descending.
    ///
    /// # Examples
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::matrix::Matrix;
    /// let m = Matrix::from_vec(2, 3, vec![1.0, 5.0, 3.0, 2.0, 6.0, 4.0])?;
    /// let (values, indices) = m.topk(2)?;
    /// assert_eq!(values, vec![6.0, 5.0]);
    /// assert_eq!(indices, vec![4, 1]);  // flat indices
    /// # Ok(())
    /// # }
    /// ```
    pub fn topk(&self, k: usize) -> Result<(Vec<f32>, Vec<usize>), TruenoError> {
        if k == 0 {
            return Ok((vec![], vec![]));
        }

        let k = k.min(self.data.len());
        let mut indexed: Vec<(usize, f32)> = self.data.iter().copied().enumerate().collect();

        // Partial sort - only sort k elements
        indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        indexed.truncate(k);
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let values: Vec<f32> = indexed.iter().map(|(_, v)| *v).collect();
        let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();

        Ok((values, indices))
    }

    /// Gather elements along axis using indices
    ///
    /// For 2D matrix with axis=0: output[i] = self[indices[i], :]
    /// For 2D matrix with axis=1: output[:, i] = self[:, indices[i]]
    ///
    /// # Examples
    /// ```
    /// use trueno::matrix::Matrix;
    /// let m = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// let gathered = m.gather(&[2, 0], 0).unwrap();  // Select rows 2 and 0
    /// assert_eq!(gathered.shape(), (2, 2));
    /// assert_eq!(gathered.get(0, 0), Some(&5.0));  // Row 2
    /// assert_eq!(gathered.get(1, 0), Some(&1.0));  // Row 0
    /// ```
    pub fn gather(&self, indices: &[usize], axis: usize) -> Result<Matrix<f32>, TruenoError> {
        match axis {
            0 => self.gather_rows(indices),
            1 => self.gather_cols(indices),
            _ => Err(TruenoError::InvalidInput(format!(
                "Axis {} not supported for 2D matrix (use 0 or 1)",
                axis
            ))),
        }
    }

    fn gather_rows(&self, indices: &[usize]) -> Result<Matrix<f32>, TruenoError> {
        let mut result = Matrix::new(indices.len(), self.cols);
        for (out_i, &idx) in indices.iter().enumerate() {
            if idx >= self.rows {
                return Err(TruenoError::InvalidInput(format!(
                    "Index {} out of bounds for axis 0 with size {}",
                    idx, self.rows
                )));
            }
            result.data[out_i * self.cols..(out_i + 1) * self.cols]
                .copy_from_slice(&self.data[idx * self.cols..(idx + 1) * self.cols]);
        }
        Ok(result)
    }

    fn gather_cols(&self, indices: &[usize]) -> Result<Matrix<f32>, TruenoError> {
        let mut result = Matrix::new(self.rows, indices.len());
        for i in 0..self.rows {
            for (out_j, &idx) in indices.iter().enumerate() {
                if idx >= self.cols {
                    return Err(TruenoError::InvalidInput(format!(
                        "Index {} out of bounds for axis 1 with size {}",
                        idx, self.cols
                    )));
                }
                result.data[i * indices.len() + out_j] = self.data[i * self.cols + idx];
            }
        }
        Ok(result)
    }

    /// Pad matrix with a constant value
    ///
    /// # Arguments
    /// * `padding` - ((top, bottom), (left, right)) padding amounts
    /// * `value` - constant value to pad with (usually 0.0)
    ///
    /// # Examples
    /// ```
    /// use trueno::matrix::Matrix;
    /// let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let padded = m.pad(((1, 1), (1, 1)), 0.0).unwrap();
    /// assert_eq!(padded.shape(), (4, 4));
    /// assert_eq!(padded.get(0, 0), Some(&0.0));  // top-left padding
    /// assert_eq!(padded.get(1, 1), Some(&1.0));  // original (0,0)
    /// ```
    pub fn pad(
        &self,
        padding: ((usize, usize), (usize, usize)),
        value: f32,
    ) -> Result<Matrix<f32>, TruenoError> {
        let ((top, bottom), (left, right)) = padding;
        let new_rows = self.rows + top + bottom;
        let new_cols = self.cols + left + right;

        let mut result = Matrix::from_vec(new_rows, new_cols, vec![value; new_rows * new_cols])?;

        // Copy original data
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[(i + top) * new_cols + (j + left)] = self.data[i * self.cols + j];
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests;
