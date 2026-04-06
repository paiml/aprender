//! TensorView - GPU Memory Layout Abstraction
//!
//! Provides a view into GPU buffer memory with shape, stride, and layout information.
//! Enables zero-copy slicing and transposition operations.
//!
//! # cuda-tile-behavior.md References
//!
//! - Section 3.2: Two-Level Memory Hierarchy
//! - Falsification tests #31-40: TensorView correctness
//!
//! # Academic Foundation
//!
//! Based on Halide (PLDI 2013): Schedule/algorithm separation improves portability.

use std::marker::PhantomData;
use std::ops::Range;

/// Memory layout for tensor storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryLayout {
    /// Row-major (C-style): last dimension varies fastest
    #[default]
    RowMajor,
    /// Column-major (Fortran-style): first dimension varies fastest
    ColumnMajor,
    /// Tiled layout for GPU shared memory optimization
    Tiled {
        /// Tile dimensions
        tile_size: [usize; 2],
    },
}

/// A view into a contiguous memory region with shape and stride information.
///
/// TensorView does not own the data - it provides a structured view over
/// existing memory, enabling zero-copy operations like slicing and transposition.
///
/// # Type Parameters
///
/// * `T` - Element type (typically f32 for GPU compute)
///
/// # cuda-tile-behavior.md References
///
/// - Falsification test #31: TensorView preserves data integrity
/// - Falsification test #32: Slicing produces correct views
/// - Falsification test #33: Transpose swaps dimensions correctly
#[derive(Debug)]
pub struct TensorView<T> {
    /// Shape of the tensor (up to 4 dimensions: N, C, H, W)
    shape: [usize; 4],
    /// Strides for each dimension (in elements, not bytes)
    strides: [usize; 4],
    /// Offset from the start of the buffer (in elements)
    offset: usize,
    /// Memory layout hint for optimization
    layout: MemoryLayout,
    /// Number of active dimensions (1-4)
    ndim: usize,
    /// Phantom data for type safety
    _marker: PhantomData<T>,
}

impl<T> TensorView<T> {
    /// Create a new TensorView with the given shape.
    ///
    /// Strides are computed automatically based on row-major layout.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the tensor (unused dimensions should be 1)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let view = TensorView::<f32>::new([2, 3, 4, 1]); // 2x3x4 tensor
    /// assert_eq!(view.numel(), 24);
    /// ```
    pub fn new(shape: [usize; 4]) -> Self {
        let ndim = Self::compute_ndim(&shape);
        let strides = Self::compute_row_major_strides(&shape);
        Self {
            shape,
            strides,
            offset: 0,
            layout: MemoryLayout::RowMajor,
            ndim,
            _marker: PhantomData,
        }
    }

    /// Create a TensorView with explicit strides.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the tensor
    /// * `strides` - Strides for each dimension (in elements)
    pub fn with_strides(shape: [usize; 4], strides: [usize; 4]) -> Self {
        let ndim = Self::compute_ndim(&shape);
        Self {
            shape,
            strides,
            offset: 0,
            layout: MemoryLayout::RowMajor,
            ndim,
            _marker: PhantomData,
        }
    }

    /// Create a 1D TensorView.
    pub fn new_1d(len: usize) -> Self {
        Self::new([len, 1, 1, 1])
    }

    /// Create a 2D TensorView (matrix).
    pub fn new_2d(rows: usize, cols: usize) -> Self {
        Self::new([rows, cols, 1, 1])
    }

    /// Create a 3D TensorView.
    pub fn new_3d(d0: usize, d1: usize, d2: usize) -> Self {
        Self::new([d0, d1, d2, 1])
    }

    /// Create a 4D TensorView.
    pub fn new_4d(d0: usize, d1: usize, d2: usize, d3: usize) -> Self {
        Self::new([d0, d1, d2, d3])
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[usize; 4] {
        &self.shape
    }

    /// Get the strides of the tensor.
    pub fn strides(&self) -> &[usize; 4] {
        &self.strides
    }

    /// Get the offset from the start of the buffer.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the memory layout.
    pub fn layout(&self) -> MemoryLayout {
        self.layout
    }

    /// Get the number of active dimensions.
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if the tensor is contiguous in memory.
    ///
    /// A tensor is contiguous if elements are stored without gaps
    /// in row-major order.
    pub fn is_contiguous(&self) -> bool {
        let expected_strides = Self::compute_row_major_strides(&self.shape);
        self.strides == expected_strides
    }

    /// Check if the tensor is empty (has zero elements).
    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }

    /// Get dimension size at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `dim >= 4`.
    pub fn dim(&self, dim: usize) -> usize {
        self.shape[dim]
    }

    /// Get stride at the given dimension.
    ///
    /// # Panics
    ///
    /// Panics if `dim >= 4`.
    pub fn stride(&self, dim: usize) -> usize {
        self.strides[dim]
    }

    /// Create a slice of this tensor along the first dimension.
    ///
    /// # Arguments
    ///
    /// * `range` - Range of indices to include
    ///
    /// # Returns
    ///
    /// A new TensorView representing the slice.
    ///
    /// # cuda-tile-behavior.md References
    ///
    /// - Falsification test #32: Slicing produces correct views
    pub fn slice(&self, range: Range<usize>) -> Self {
        assert!(range.end <= self.shape[0], "Slice range out of bounds");
        let new_offset = self.offset + range.start * self.strides[0];
        let mut new_shape = self.shape;
        new_shape[0] = range.end - range.start;

        Self {
            shape: new_shape,
            strides: self.strides,
            offset: new_offset,
            layout: self.layout,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }

    /// Create a slice along a specific dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to slice along
    /// * `range` - Range of indices to include
    pub fn slice_dim(&self, dim: usize, range: Range<usize>) -> Self {
        assert!(dim < 4, "Dimension out of bounds");
        assert!(range.end <= self.shape[dim], "Slice range out of bounds");

        let new_offset = self.offset + range.start * self.strides[dim];
        let mut new_shape = self.shape;
        new_shape[dim] = range.end - range.start;

        Self {
            shape: new_shape,
            strides: self.strides,
            offset: new_offset,
            layout: self.layout,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }

    /// Transpose the tensor by swapping two dimensions.
    ///
    /// # Arguments
    ///
    /// * `dim0` - First dimension to swap
    /// * `dim1` - Second dimension to swap
    ///
    /// # Returns
    ///
    /// A new TensorView with swapped dimensions.
    ///
    /// # cuda-tile-behavior.md References
    ///
    /// - Falsification test #33: Transpose swaps dimensions correctly
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        assert!(dim0 < 4 && dim1 < 4, "Dimension out of bounds");

        let mut new_shape = self.shape;
        let mut new_strides = self.strides;
        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);

        Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            layout: self.layout,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }

    /// Reshape the tensor to a new shape.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - New shape (must have same number of elements)
    ///
    /// # Returns
    ///
    /// A new TensorView with the new shape, or None if reshape is invalid.
    pub fn reshape(&self, new_shape: [usize; 4]) -> Option<Self> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return None;
        }

        // Reshape requires contiguous memory
        if !self.is_contiguous() {
            return None;
        }

        Some(Self::new(new_shape))
    }

    /// Squeeze dimensions of size 1.
    ///
    /// Returns a view with all size-1 dimensions removed.
    pub fn squeeze(&self) -> Self {
        let mut new_shape = [1usize; 4];
        let mut new_strides = [1usize; 4];
        let mut new_ndim = 0;

        for i in 0..4 {
            if self.shape[i] > 1 {
                new_shape[new_ndim] = self.shape[i];
                new_strides[new_ndim] = self.strides[i];
                new_ndim += 1;
            }
        }

        // If all dimensions were 1, keep at least one
        if new_ndim == 0 {
            new_ndim = 1;
        }

        Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            layout: self.layout,
            ndim: new_ndim,
            _marker: PhantomData,
        }
    }

    /// Unsqueeze: add a dimension of size 1 at the specified position.
    ///
    /// # Arguments
    ///
    /// * `dim` - Position to insert the new dimension
    pub fn unsqueeze(&self, dim: usize) -> Option<Self> {
        if dim > self.ndim || self.ndim >= 4 {
            return None;
        }

        let mut new_shape = [1usize; 4];
        let mut new_strides = [1usize; 4];

        // Copy dimensions before the insertion point
        // Using manual loop since we're copying from two separate arrays to two separate arrays
        #[allow(clippy::manual_memcpy)]
        for i in 0..dim {
            new_shape[i] = self.shape[i];
            new_strides[i] = self.strides[i];
        }

        // Insert the new dimension
        new_shape[dim] = 1;
        new_strides[dim] = if dim < self.ndim { self.strides[dim] * self.shape[dim] } else { 1 };

        // Copy remaining dimensions (offset by 1 for insertion)
        #[allow(clippy::manual_memcpy)]
        for i in dim..self.ndim {
            new_shape[i + 1] = self.shape[i];
            new_strides[i + 1] = self.strides[i];
        }

        Some(Self {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            layout: self.layout,
            ndim: self.ndim + 1,
            _marker: PhantomData,
        })
    }

    /// Set the memory layout hint.
    pub fn with_layout(mut self, layout: MemoryLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Compute linear index from multi-dimensional indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - Array of indices for each dimension
    ///
    /// # Returns
    ///
    /// Linear offset into the underlying buffer.
    pub fn linear_index(&self, indices: [usize; 4]) -> usize {
        self.offset
            + indices[0] * self.strides[0]
            + indices[1] * self.strides[1]
            + indices[2] * self.strides[2]
            + indices[3] * self.strides[3]
    }

    /// Compute row-major strides for a given shape.
    fn compute_row_major_strides(shape: &[usize; 4]) -> [usize; 4] {
        let mut strides = [1usize; 4];
        // Strides: s[i] = product of shape[i+1..4]
        strides[3] = 1;
        strides[2] = shape[3];
        strides[1] = shape[3] * shape[2];
        strides[0] = shape[3] * shape[2] * shape[1];
        strides
    }

    /// Compute the number of active dimensions.
    fn compute_ndim(shape: &[usize; 4]) -> usize {
        // Count from the end: find last dimension > 1
        for i in (0..4).rev() {
            if shape[i] > 1 {
                return i + 1;
            }
        }
        1 // At least 1 dimension
    }
}

impl<T> Clone for TensorView<T> {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape,
            strides: self.strides,
            offset: self.offset,
            layout: self.layout,
            ndim: self.ndim,
            _marker: PhantomData,
        }
    }
}

impl<T> Default for TensorView<T> {
    fn default() -> Self {
        Self::new([1, 1, 1, 1])
    }
}

#[cfg(test)]
mod tests;
