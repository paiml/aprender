//! Dense N-dimensional tensor.

use crate::error::TensorError;

/// Dense tensor with arbitrary dimensions.
///
/// Row-major (C-order) storage. Shape `[d0, d1, ..., dk]` means
/// element `[i0, i1, ..., ik]` is at offset `i0*stride[0] + i1*stride[1] + ...`.
#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    /// Create a new tensor with the given shape and data.
    ///
    /// # Errors
    ///
    /// Returns error if data length doesn't match shape product.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, TensorError> {
        let product: usize = shape.iter().product();
        if data.len() != product {
            return Err(TensorError::DataLengthMismatch {
                len: data.len(),
                shape: shape.clone(),
                product,
            });
        }
        let strides = compute_strides(&shape);
        Ok(Self { shape, strides, data })
    }

    /// Create a zero tensor with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let product: usize = shape.iter().product();
        let strides = compute_strides(&shape);
        Self { shape, strides, data: vec![0.0; product] }
    }

    /// Tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Raw data slice.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Mutable data slice.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get element at multi-index.
    pub fn get(&self, indices: &[usize]) -> f32 {
        let offset = self.offset(indices);
        self.data[offset]
    }

    /// Set element at multi-index.
    pub fn set(&mut self, indices: &[usize], value: f32) {
        let offset = self.offset(indices);
        self.data[offset] = value;
    }

    /// Compute linear offset from multi-index.
    fn offset(&self, indices: &[usize]) -> usize {
        indices.iter().zip(self.strides.iter()).map(|(&i, &s)| i * s).sum()
    }

    /// Reshape tensor (must have same total elements).
    ///
    /// # Errors
    ///
    /// Returns error if new shape has different total elements.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, TensorError> {
        let new_product: usize = new_shape.iter().product();
        if new_product != self.data.len() {
            return Err(TensorError::ShapeMismatch {
                expected: new_shape,
                got: self.shape.clone(),
            });
        }
        Self::new(new_shape, self.data.clone())
    }

    /// Transpose: permute dimensions according to `perm`.
    pub fn transpose(&self, perm: &[usize]) -> Self {
        let ndim = self.ndim();
        let mut new_shape = vec![0usize; ndim];
        for (i, &p) in perm.iter().enumerate() {
            new_shape[i] = self.shape[p];
        }
        let new_product: usize = new_shape.iter().product();
        let mut new_data = vec![0.0f32; new_product];
        let new_strides = compute_strides(&new_shape);

        // Iterate over all elements
        let mut old_indices = vec![0usize; ndim];
        for flat in 0..self.data.len() {
            // Convert flat index to multi-index (old)
            let mut rem = flat;
            for d in 0..ndim {
                old_indices[d] = rem / self.strides[d];
                rem %= self.strides[d];
            }

            // Permute indices
            let new_offset: usize = perm
                .iter()
                .enumerate()
                .map(|(new_d, &old_d)| old_indices[old_d] * new_strides[new_d])
                .sum();

            new_data[new_offset] = self.data[flat];
        }

        Self { shape: new_shape, strides: new_strides, data: new_data }
    }
}

/// Compute row-major strides for a shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
