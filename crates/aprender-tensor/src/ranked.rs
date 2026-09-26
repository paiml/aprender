//! Rank-typed dense tensor: the rank is a const generic, so a rank mismatch
//! does not compile (#3150).
//!
//! [`Tensor`](crate::Tensor) keeps its rank in a `Vec<usize>`, so every rank
//! error is found at runtime, if at all. [`RankedTensor<D, L>`] fixes the rank
//! `D` and the storage layout `L` in the type:
//!
//! - `matmul` exists only on `RankedTensor<2, RowMajor>`.
//! - `reshape::<D2>` returns a tensor whose rank is in its type.
//! - `unsqueeze` / `squeeze` change the rank in the type (ranks 0..=6).
//! - A [`ColMajor`] tensor has no arithmetic. The only ways out are the
//!   conversions to [`RowMajor`], which is how `contracts/tensor-layout-v1.yaml`'s
//!   "transpose at the GGUF import boundary" becomes a type-level boundary.
//!
//! The compile-fail proofs live in `tests/ui/*.rs` (run by `tests/rank_typed_ui.rs`).
//! Rank is typed and dimensions stay runtime values; full shape typing is out of scope.

use std::marker::PhantomData;

use crate::error::TensorError;
use crate::tensor::Tensor;

mod sealed {
    pub trait Sealed {}
}

/// Storage layout marker. Sealed: only [`RowMajor`] and [`ColMajor`] exist.
pub trait Layout: sealed::Sealed + Copy + Default + std::fmt::Debug {
    /// Name as used in `contracts/tensor-layout-v1.yaml` (`formats.*.layout`).
    const NAME: &'static str;
    /// True when the LAST axis is contiguous (row-major), false when the FIRST is.
    const LAST_AXIS_CONTIGUOUS: bool;
}

/// Row-major (C-order): the LAST axis is contiguous. The APR / SafeTensors layout.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RowMajor;

/// Column-major: the FIRST axis is contiguous. The GGUF (GGML `ne[0]`) layout.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ColMajor;

impl sealed::Sealed for RowMajor {}
impl sealed::Sealed for ColMajor {}
impl Layout for RowMajor {
    const NAME: &'static str = "row-major";
    const LAST_AXIS_CONTIGUOUS: bool = true;
}
impl Layout for ColMajor {
    const NAME: &'static str = "column-major";
    const LAST_AXIS_CONTIGUOUS: bool = false;
}

/// Dense `f32` tensor of rank `D` stored in layout `L`.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedTensor<const D: usize, L: Layout = RowMajor> {
    shape: [usize; D],
    data: Vec<f32>,
    layout: PhantomData<L>,
}

/// A rank-2 row-major matrix.
pub type Matrix = RankedTensor<2, RowMajor>;

impl<const D: usize, L: Layout> RankedTensor<D, L> {
    /// Build a tensor from a shape and data in layout `L`.
    ///
    /// # Errors
    ///
    /// `DataLengthMismatch` if `data.len()` is not the product of `shape`.
    pub fn new(shape: [usize; D], data: Vec<f32>) -> Result<Self, TensorError> {
        let product: usize = shape.iter().product();
        if data.len() != product {
            return Err(TensorError::DataLengthMismatch {
                len: data.len(),
                shape: shape.to_vec(),
                product,
            });
        }
        Ok(Self {
            shape,
            data,
            layout: PhantomData,
        })
    }

    /// All-zero tensor.
    pub fn zeros(shape: [usize; D]) -> Self {
        Self {
            shape,
            data: vec![0.0; shape.iter().product()],
            layout: PhantomData,
        }
    }

    /// The shape; its length is the rank `D`.
    pub fn shape(&self) -> [usize; D] {
        self.shape
    }

    /// The rank, known at compile time.
    pub const fn rank() -> usize {
        D
    }

    /// Raw data in layout `L`.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Consume into raw data in layout `L`.
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Reinterpret the same data under a new rank-`D2` shape. The layout is
    /// kept, so the element order is unchanged.
    ///
    /// # Errors
    ///
    /// `ShapeMismatch` if the element counts differ.
    pub fn reshape<const D2: usize>(
        self,
        shape: [usize; D2],
    ) -> Result<RankedTensor<D2, L>, TensorError> {
        let product: usize = shape.iter().product();
        if product != self.data.len() {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.to_vec(),
                got: shape.to_vec(),
            });
        }
        Ok(RankedTensor {
            shape,
            data: self.data,
            layout: PhantomData,
        })
    }

    /// Offset of `index` in the flat data, honouring the layout.
    fn offset(&self, index: [usize; D]) -> usize {
        let mut offset = 0;
        let mut stride = 1;
        if L::LAST_AXIS_CONTIGUOUS {
            for axis in (0..D).rev() {
                offset += index[axis] * stride;
                stride *= self.shape[axis];
            }
        } else {
            for axis in 0..D {
                offset += index[axis] * stride;
                stride *= self.shape[axis];
            }
        }
        offset
    }

    /// Element at `index`.
    ///
    /// # Panics
    ///
    /// If any index component is out of bounds.
    pub fn get(&self, index: [usize; D]) -> f32 {
        for (axis, (&i, &n)) in index.iter().zip(&self.shape).enumerate() {
            assert!(i < n, "index {i} out of bounds for axis {axis} of size {n}");
        }
        self.data[self.offset(index)]
    }
}

impl<const D: usize> RankedTensor<D, RowMajor> {
    /// Rank-check a dynamic [`Tensor`] (row-major) into a rank-`D` tensor.
    ///
    /// # Errors
    ///
    /// `RankMismatch` if the dynamic tensor's rank is not `D`.
    pub fn from_dynamic(t: &Tensor) -> Result<Self, TensorError> {
        let shape: [usize; D] = t
            .shape()
            .try_into()
            .map_err(|_| TensorError::RankMismatch {
                expected: D,
                got: t.ndim(),
            })?;
        Self::new(shape, t.data().to_vec())
    }

    /// Back to a dynamic [`Tensor`].
    pub fn into_dynamic(self) -> Tensor {
        Tensor::from_trusted(self.shape.to_vec(), self.data)
    }
}

impl RankedTensor<2, ColMajor> {
    /// A GGUF 2-D tensor exactly as the file describes it: `ne = [ne0, ne1]`
    /// with `ne0` contiguous (`contracts/tensor-layout-v1.yaml`
    /// `formats.gguf.shape_convention`).
    ///
    /// # Errors
    ///
    /// `DataLengthMismatch` if `data.len() != ne0 * ne1`.
    pub fn from_gguf(ne: [usize; 2], data: Vec<f32>) -> Result<Self, TensorError> {
        Self::new(ne, data)
    }

    /// The GGUF import boundary: GGUF `[ne0, ne1]` becomes APR `[ne1, ne0]`
    /// row-major over the SAME bytes. This is the contract's `transpose: 'true'`
    /// rows (`gguf_shape` reversed is `apr_shape`). It is zero-copy: a
    /// column-major `[r, c]` buffer is a row-major `[c, r]` buffer.
    pub fn into_apr(self) -> Matrix {
        let [ne0, ne1] = self.shape;
        RankedTensor {
            shape: [ne1, ne0],
            data: self.data,
            layout: PhantomData,
        }
    }

    /// Same logical matrix, row-major storage (a physical transpose of the
    /// buffer; the shape is unchanged). Use this when the column-major matrix
    /// is meant as-is, not as a GGUF weight.
    pub fn to_row_major(&self) -> Matrix {
        let [rows, cols] = self.shape;
        let mut data = vec![0.0; rows * cols];
        for c in 0..cols {
            for r in 0..rows {
                data[r * cols + c] = self.data[c * rows + r];
            }
        }
        RankedTensor {
            shape: self.shape,
            data,
            layout: PhantomData,
        }
    }
}

impl Matrix {
    /// `self @ other`. Rank 2 is a type fact, so only the inner dimension is
    /// checked. The i-k-j loop keeps both inner accesses contiguous.
    ///
    /// # Errors
    ///
    /// `ContractionDimensionMismatch` (index `'j'`) if `self` is `m x k` and
    /// `other` is not `k x n`.
    pub fn matmul(&self, other: &Matrix) -> Result<Matrix, TensorError> {
        let [m, k] = self.shape;
        let [k2, n] = other.shape;
        if k != k2 {
            return Err(TensorError::ContractionDimensionMismatch {
                index: 'j',
                size_a: k,
                size_b: k2,
            });
        }
        let mut out = vec![0.0_f32; m * n];
        for (a_row, out_row) in self
            .data
            .chunks_exact(k.max(1))
            .zip(out.chunks_exact_mut(n.max(1)))
        {
            for (&a, b_row) in a_row.iter().zip(other.data.chunks_exact(n.max(1))) {
                for (o, &b) in out_row.iter_mut().zip(b_row) {
                    *o += a * b;
                }
            }
        }
        RankedTensor::new([m, n], out)
    }

    /// Transpose (the result is row-major too).
    pub fn t(&self) -> Matrix {
        let [rows, cols] = self.shape;
        let mut data = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                data[c * rows + r] = self.data[r * cols + c];
            }
        }
        RankedTensor {
            shape: [cols, rows],
            data,
            layout: PhantomData,
        }
    }
}

/// `unsqueeze` (rank `D` -> `D + 1`) and `squeeze` (rank `D + 1` -> `D`) for
/// each concrete rank. Stable Rust cannot write `D + 1` in a const generic, so
/// each step is spelled out once.
macro_rules! rank_steps {
    ($($lo:literal => $hi:literal),* $(,)?) => {$(
        impl<L: Layout> RankedTensor<$lo, L> {
            /// Insert a size-1 axis at `axis` (`0..=rank`). The rank grows by
            /// one in the type; the element order is unchanged.
            ///
            /// # Errors
            ///
            /// `ShapeMismatch` if `axis > rank`.
            pub fn unsqueeze(self, axis: usize) -> Result<RankedTensor<$hi, L>, TensorError> {
                if axis > $lo {
                    return Err(TensorError::ShapeMismatch {
                        expected: self.shape.to_vec(),
                        got: vec![axis],
                    });
                }
                let mut shape = [1_usize; $hi];
                shape[..axis].copy_from_slice(&self.shape[..axis]);
                shape[axis + 1..].copy_from_slice(&self.shape[axis..]);
                Ok(RankedTensor { shape, data: self.data, layout: PhantomData })
            }
        }

        impl<L: Layout> RankedTensor<$hi, L> {
            /// Remove the size-1 axis at `axis`. The rank shrinks by one in
            /// the type.
            ///
            /// # Errors
            ///
            /// `ShapeMismatch` if `axis` is out of range or its size is not 1.
            pub fn squeeze(self, axis: usize) -> Result<RankedTensor<$lo, L>, TensorError> {
                if axis >= $hi || self.shape[axis] != 1 {
                    return Err(TensorError::ShapeMismatch {
                        expected: self.shape.to_vec(),
                        got: vec![axis],
                    });
                }
                let mut shape = [0_usize; $lo];
                shape[..axis].copy_from_slice(&self.shape[..axis]);
                shape[axis..].copy_from_slice(&self.shape[axis + 1..]);
                Ok(RankedTensor { shape, data: self.data, layout: PhantomData })
            }
        }
    )*};
}

rank_steps!(0 => 1, 1 => 2, 2 => 3, 3 => 4, 4 => 5, 5 => 6);
