//! Tensor operations with SIMD acceleration (v2.0 Phase 3).
//!
//! This module provides a high-level API for distributed tensor operations
//! using the trueno library for SIMD-accelerated computation.

#[cfg(feature = "tensor")]
use crate::error::Result;
#[cfg(feature = "tensor")]
use crate::task::Backend;

/// Tensor type alias for `trueno::Vector<f32>` (v2.0 Phase 3).
///
/// Provides a convenient type for SIMD-accelerated vector operations.
/// Uses f32 (single precision) for optimal SIMD performance on most CPUs.
#[cfg(feature = "tensor")]
pub type Tensor = trueno::Vector<f32>;

/// Executor for distributed tensor operations (v2.0 Phase 3).
///
/// Wraps the trueno library to provide SIMD-accelerated tensor operations
/// that can be distributed across CPU and GPU backends.
///
/// # Example
///
/// ```no_run
/// # #[cfg(feature = "tensor")]
/// # use repartir::tensor::{TensorExecutor, Tensor};
/// # #[cfg(feature = "tensor")]
/// # use repartir::task::Backend;
/// # #[cfg(feature = "tensor")]
/// # fn main() -> repartir::error::Result<()> {
/// let executor = TensorExecutor::builder()
///     .backend(Backend::Cpu)
///     .build()?;
///
/// // SIMD tensor operations
/// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
/// let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0]);
///
/// let result = executor.add(&a, &b)?;
/// println!("Result: {:?}", result.as_slice());
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "tensor")]
#[derive(Clone)]
pub struct TensorExecutor {
    backend: Backend,
    // Future: Add distributed worker pool for tensor parallelism
}

#[cfg(feature = "tensor")]
impl TensorExecutor {
    /// Creates a new tensor executor builder.
    #[must_use]
    pub fn builder() -> TensorExecutorBuilder {
        TensorExecutorBuilder::default()
    }

    /// Element-wise addition of two tensors (SIMD-accelerated).
    ///
    /// # Errors
    ///
    /// Returns an error if tensor shapes don't match.
    ///
    /// # Example
    ///
    /// ```
    /// # #[cfg(feature = "tensor")]
    /// # use repartir::tensor::{TensorExecutor, Tensor};
    /// # #[cfg(feature = "tensor")]
    /// # use repartir::task::Backend;
    /// # #[cfg(feature = "tensor")]
    /// # fn main() -> repartir::error::Result<()> {
    /// let executor = TensorExecutor::builder()
    ///     .backend(Backend::Cpu)
    ///     .build()?;
    ///
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0]);
    /// let b = Tensor::from_slice(&[4.0, 5.0, 6.0]);
    ///
    /// let result = executor.add(&a, &b)?;
    /// assert_eq!(result.as_slice(), &[5.0, 7.0, 9.0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // trueno provides SIMD-optimized operations
        a.add(b).map_err(Into::into)
    }

    /// Element-wise subtraction of two tensors (SIMD-accelerated).
    ///
    /// # Errors
    ///
    /// Returns an error if tensor shapes don't match.
    pub fn sub(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        a.sub(b).map_err(Into::into)
    }

    /// Element-wise multiplication of two tensors (SIMD-accelerated).
    ///
    /// # Errors
    ///
    /// Returns an error if tensor shapes don't match.
    pub fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        a.mul(b).map_err(Into::into)
    }

    /// Element-wise division of two tensors (SIMD-accelerated).
    ///
    /// # Errors
    ///
    /// Returns an error if tensor shapes don't match or division by zero.
    pub fn div(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        a.div(b).map_err(Into::into)
    }

    /// Dot product of two tensors (SIMD-accelerated).
    ///
    /// Returns a scalar value.
    ///
    /// # Errors
    ///
    /// Returns an error if tensor shapes are incompatible for dot product.
    pub fn dot(&self, a: &Tensor, b: &Tensor) -> Result<f32> {
        a.dot(b).map_err(Into::into)
    }

    /// Scalar multiplication (SIMD-accelerated).
    ///
    /// Multiplies each element of the tensor by a scalar value.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails.
    pub fn scalar_mul(&self, tensor: &Tensor, scalar: f32) -> Result<Tensor> {
        // Manually implement scalar multiplication
        let data: Vec<f32> = tensor.as_slice().iter().map(|&x| x * scalar).collect();
        Ok(Tensor::from_vec(data))
    }

    /// Returns the backend used by this executor.
    #[must_use]
    pub const fn backend(&self) -> Backend {
        self.backend
    }
}

/// Builder for `TensorExecutor`.
#[cfg(feature = "tensor")]
#[derive(Debug, Default)]
pub struct TensorExecutorBuilder {
    backend: Option<Backend>,
}

#[cfg(feature = "tensor")]
impl TensorExecutorBuilder {
    /// Sets the execution backend.
    #[must_use]
    pub const fn backend(mut self, backend: Backend) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Builds the tensor executor.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend is not supported.
    pub fn build(self) -> Result<TensorExecutor> {
        let backend = self.backend.unwrap_or(Backend::Cpu);
        Ok(TensorExecutor { backend })
    }
}

#[cfg(all(test, feature = "tensor"))]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_executor_creation() {
        let executor = TensorExecutor::builder().backend(Backend::Cpu).build().unwrap();

        assert_eq!(executor.backend(), Backend::Cpu);
    }

    #[test]
    fn test_tensor_addition() {
        let executor = TensorExecutor::builder().backend(Backend::Cpu).build().unwrap();

        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0]);

        let result = executor.add(&a, &b).unwrap();
        assert_eq!(result.as_slice(), &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_tensor_subtraction() {
        let executor = TensorExecutor::builder().backend(Backend::Cpu).build().unwrap();

        let a = Tensor::from_slice(&[10.0, 9.0, 8.0, 7.0]);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let result = executor.sub(&a, &b).unwrap();
        assert_eq!(result.as_slice(), &[9.0, 7.0, 5.0, 3.0]);
    }

    #[test]
    fn test_tensor_multiplication() {
        let executor = TensorExecutor::builder().backend(Backend::Cpu).build().unwrap();

        let a = Tensor::from_slice(&[2.0, 3.0, 4.0]);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0]);

        let result = executor.mul(&a, &b).unwrap();
        assert_eq!(result.as_slice(), &[10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_tensor_division() {
        let executor = TensorExecutor::builder().backend(Backend::Cpu).build().unwrap();

        let a = Tensor::from_slice(&[10.0, 20.0, 30.0]);
        let b = Tensor::from_slice(&[2.0, 4.0, 5.0]);

        let result = executor.div(&a, &b).unwrap();
        assert_eq!(result.as_slice(), &[5.0, 5.0, 6.0]);
    }

    #[test]
    fn test_tensor_scalar_multiplication() {
        let executor = TensorExecutor::builder().backend(Backend::Cpu).build().unwrap();

        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let result = executor.scalar_mul(&a, 2.5).unwrap();
        assert_eq!(result.as_slice(), &[2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_tensor_dot_product() {
        let executor = TensorExecutor::builder().backend(Backend::Cpu).build().unwrap();

        let a = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0]);

        let result = executor.dot(&a, &b).unwrap();
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_tensor_executor_default_backend() {
        let executor = TensorExecutor::builder().build().unwrap();
        assert_eq!(executor.backend(), Backend::Cpu);
    }
}
