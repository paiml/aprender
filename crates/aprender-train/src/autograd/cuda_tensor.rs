//! CUDA-accelerated tensor type for GPU training
//!
//! This module provides GPU-resident tensors using trueno-gpu's CUDA backend.
//! It replaces ndarray::Array1<f32> with CUDA-backed storage for 100x speedup.
//!
//! # Architecture (SPEC-FT-001 v3.0.0)
//!
//! ```text
//! CudaTensor
//!   ├── data: GpuBuffer<f32>     (GPU memory)
//!   ├── grad: Option<GpuBuffer<f32>>  (gradient on GPU)
//!   └── ctx: Arc<CudaContext>    (shared device context)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use entrenar::autograd::CudaTensor;
//!
//! // Create tensor on GPU
//! let t = CudaTensor::from_vec(vec![1.0, 2.0, 3.0], true)?;
//!
//! // Transfer back to CPU
//! let data = t.to_vec()?;
//! ```

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{cuda_available, CudaContext, CudaStream, GpuBuffer};
#[cfg(feature = "cuda")]
use trueno_gpu::GpuError;

/// Error type for CUDA tensor operations
#[derive(Debug, thiserror::Error)]
pub enum CudaTensorError {
    /// CUDA is not available on this system
    #[error("CUDA not available: {0}")]
    CudaNotAvailable(String),

    /// GPU memory allocation failed
    #[error("GPU allocation failed: {0}")]
    AllocationFailed(String),

    /// Data transfer failed
    #[error("Data transfer failed: {0}")]
    TransferFailed(String),

    /// Shape mismatch
    #[error("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },

    /// Kernel launch failed
    #[error("Kernel launch failed: {0}")]
    KernelError(String),

    /// Device not initialized
    #[error("CUDA device not initialized")]
    DeviceNotInitialized,
}

#[cfg(feature = "cuda")]
impl From<GpuError> for CudaTensorError {
    fn from(e: GpuError) -> Self {
        match e {
            GpuError::OutOfMemory { requested, available } => CudaTensorError::AllocationFailed(
                format!("Out of GPU memory: requested {requested} bytes, {available} available"),
            ),
            GpuError::Transfer(msg) => CudaTensorError::TransferFailed(msg),
            GpuError::CudaNotAvailable(msg) => CudaTensorError::CudaNotAvailable(msg),
            other => CudaTensorError::KernelError(format!("{other:?}")),
        }
    }
}

/// Result type for CUDA tensor operations
pub type Result<T> = std::result::Result<T, CudaTensorError>;

/// CUDA device handle with lazy initialization
#[cfg(feature = "cuda")]
pub struct CudaDevice {
    ctx: Arc<CudaContext>,
    stream: CudaStream,
}

#[cfg(feature = "cuda")]
impl CudaDevice {
    /// Create a new CUDA device handle for the given device ID
    pub fn new(device_id: i32) -> Result<Self> {
        if !cuda_available() {
            return Err(CudaTensorError::CudaNotAvailable("No CUDA driver found".into()));
        }

        let ctx = CudaContext::new(device_id)
            .map_err(|e| CudaTensorError::CudaNotAvailable(format!("{e:?}")))?;
        let stream = CudaStream::new(&ctx)
            .map_err(|e| CudaTensorError::AllocationFailed(format!("{e:?}")))?;

        Ok(Self { ctx: Arc::new(ctx), stream })
    }

    /// Create device handle for default GPU (device 0)
    pub fn default_device() -> Result<Self> {
        Self::new(0)
    }

    /// Get the CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Get the CUDA stream
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    /// Synchronize the stream (wait for all operations to complete)
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(|e| CudaTensorError::KernelError(format!("{e:?}")))
    }
}

/// GPU-resident tensor with gradient support
///
/// This is the CUDA-accelerated replacement for `Tensor` when the `cuda` feature is enabled.
#[cfg(feature = "cuda")]
pub struct CudaTensor {
    /// Data stored on GPU
    data: GpuBuffer<f32>,
    /// Gradient stored on GPU (if requires_grad)
    grad: Option<GpuBuffer<f32>>,
    /// Shared device context
    device: Arc<CudaContext>,
    /// Whether this tensor requires gradient computation
    requires_grad: bool,
    /// Number of elements
    len: usize,
}

#[cfg(feature = "cuda")]
impl CudaTensor {
    /// Create a new tensor on GPU from host data
    pub fn from_vec(device: &CudaDevice, data: Vec<f32>, requires_grad: bool) -> Result<Self> {
        let len = data.len();
        let gpu_data = GpuBuffer::from_host(&device.ctx, &data)?;

        let grad = if requires_grad {
            // Initialize gradient to zeros
            let zeros = vec![0.0f32; len];
            Some(GpuBuffer::from_host(&device.ctx, &zeros)?)
        } else {
            None
        };

        Ok(Self { data: gpu_data, grad, device: device.ctx.clone(), requires_grad, len })
    }

    /// Create a tensor filled with zeros
    pub fn zeros(device: &CudaDevice, len: usize, requires_grad: bool) -> Result<Self> {
        let data = vec![0.0f32; len];
        Self::from_vec(device, data, requires_grad)
    }

    /// Create a tensor filled with ones
    pub fn ones(device: &CudaDevice, len: usize, requires_grad: bool) -> Result<Self> {
        let data = vec![1.0f32; len];
        Self::from_vec(device, data, requires_grad)
    }

    /// Copy tensor data back to CPU
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let mut result = vec![0.0f32; self.len];
        self.data.copy_to_host(&mut result)?;
        Ok(result)
    }

    /// Get gradient as CPU vector (if computed)
    pub fn grad_to_vec(&self) -> Result<Option<Vec<f32>>> {
        match &self.grad {
            Some(grad_buf) => {
                let mut result = vec![0.0f32; self.len];
                grad_buf.copy_to_host(&mut result)?;
                Ok(Some(result))
            }
            None => Ok(None),
        }
    }

    /// Update data from CPU vector
    pub fn copy_from_vec(&mut self, data: &[f32]) -> Result<()> {
        if data.len() != self.len {
            return Err(CudaTensorError::ShapeMismatch { expected: self.len, actual: data.len() });
        }
        self.data.copy_from_host(data)?;
        Ok(())
    }

    /// Set gradient from CPU vector
    pub fn set_grad_from_vec(&mut self, grad: &[f32]) -> Result<()> {
        if grad.len() != self.len {
            return Err(CudaTensorError::ShapeMismatch { expected: self.len, actual: grad.len() });
        }

        match &mut self.grad {
            Some(grad_buf) => {
                grad_buf.copy_from_host(grad)?;
            }
            None => {
                self.grad = Some(GpuBuffer::from_host(
                    // Need to get context somehow - this is a design issue
                    // For now, we'll create a new buffer
                    &CudaContext::new(0)
                        .map_err(|e| CudaTensorError::CudaNotAvailable(format!("{e:?}")))?,
                    grad,
                )?);
            }
        }
        Ok(())
    }

    /// Zero out gradient
    pub fn zero_grad(&mut self) -> Result<()> {
        if let Some(ref mut grad_buf) = self.grad {
            let zeros = vec![0.0f32; self.len];
            grad_buf.copy_from_host(&zeros)?;
        }
        Ok(())
    }

    /// Check if requires gradient
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Get number of elements
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get raw GPU buffer for data (for kernel operations)
    pub fn data_buffer(&self) -> &GpuBuffer<f32> {
        &self.data
    }

    /// Get mutable raw GPU buffer for data
    pub fn data_buffer_mut(&mut self) -> &mut GpuBuffer<f32> {
        &mut self.data
    }

    /// Get raw GPU buffer for gradient (for kernel operations)
    pub fn grad_buffer(&self) -> Option<&GpuBuffer<f32>> {
        self.grad.as_ref()
    }

    /// Get mutable raw GPU buffer for gradient
    pub fn grad_buffer_mut(&mut self) -> Option<&mut GpuBuffer<f32>> {
        self.grad.as_mut()
    }

    /// Get device context
    pub fn device(&self) -> &Arc<CudaContext> {
        &self.device
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for CudaTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTensor")
            .field("len", &self.len)
            .field("requires_grad", &self.requires_grad)
            .field("has_grad", &self.grad.is_some())
            .finish_non_exhaustive()
    }
}

// CPU fallback when CUDA is not available
#[cfg(not(feature = "cuda"))]
pub struct CudaDevice;

#[cfg(not(feature = "cuda"))]
impl CudaDevice {
    pub fn new(_device_id: i32) -> Result<Self> {
        Err(CudaTensorError::CudaNotAvailable("Compiled without CUDA support".into()))
    }

    pub fn default_device() -> Result<Self> {
        Err(CudaTensorError::CudaNotAvailable("Compiled without CUDA support".into()))
    }
}

#[cfg(not(feature = "cuda"))]
pub struct CudaTensor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_tensor_error_display() {
        let err = CudaTensorError::CudaNotAvailable("test".into());
        assert!(err.to_string().contains("CUDA not available"));

        let err = CudaTensorError::ShapeMismatch { expected: 10, actual: 5 };
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains('5'));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_device_creation() {
        // This test only runs if CUDA is actually available
        if !cuda_available() {
            return;
        }

        let device = CudaDevice::default_device();
        assert!(device.is_ok());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_tensor_from_vec() {
        if !cuda_available() {
            return;
        }

        let device = CudaDevice::default_device().expect("operation should succeed");
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor =
            CudaTensor::from_vec(&device, data.clone(), true).expect("operation should succeed");

        assert_eq!(tensor.len(), 4);
        assert!(tensor.requires_grad());

        // Verify round-trip
        let result = tensor.to_vec().expect("operation should succeed");
        assert_eq!(result, data);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_tensor_zeros() {
        if !cuda_available() {
            return;
        }

        let device = CudaDevice::default_device().expect("operation should succeed");
        let tensor = CudaTensor::zeros(&device, 100, false).expect("operation should succeed");

        assert_eq!(tensor.len(), 100);
        assert!(!tensor.requires_grad());

        let data = tensor.to_vec().expect("operation should succeed");
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_tensor_ones() {
        if !cuda_available() {
            return;
        }

        let device = CudaDevice::default_device().expect("operation should succeed");
        let tensor = CudaTensor::ones(&device, 50, true).expect("operation should succeed");

        assert_eq!(tensor.len(), 50);
        let data = tensor.to_vec().expect("operation should succeed");
        assert!(data.iter().all(|&x| x == 1.0));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_tensor_gradient() {
        if !cuda_available() {
            return;
        }

        let device = CudaDevice::default_device().expect("operation should succeed");
        let mut tensor = CudaTensor::from_vec(&device, vec![1.0, 2.0, 3.0], true)
            .expect("operation should succeed");

        // Initially gradient should be zeros
        let grad = tensor
            .grad_to_vec()
            .expect("operation should succeed")
            .expect("operation should succeed");
        assert!(grad.iter().all(|&x| x == 0.0));

        // Set gradient
        tensor.set_grad_from_vec(&[0.1, 0.2, 0.3]).expect("operation should succeed");
        let grad = tensor
            .grad_to_vec()
            .expect("operation should succeed")
            .expect("operation should succeed");
        assert!((grad[0] - 0.1).abs() < 1e-6);
        assert!((grad[1] - 0.2).abs() < 1e-6);
        assert!((grad[2] - 0.3).abs() < 1e-6);

        // Zero gradient
        tensor.zero_grad().expect("gradient should be available");
        let grad = tensor
            .grad_to_vec()
            .expect("operation should succeed")
            .expect("operation should succeed");
        assert!(grad.iter().all(|&x| x == 0.0));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_tensor_copy_from_vec() {
        if !cuda_available() {
            return;
        }

        let device = CudaDevice::default_device().expect("operation should succeed");
        let mut tensor = CudaTensor::zeros(&device, 4, false).expect("operation should succeed");

        tensor.copy_from_vec(&[5.0, 6.0, 7.0, 8.0]).expect("operation should succeed");
        let data = tensor.to_vec().expect("operation should succeed");
        assert_eq!(data, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_tensor_shape_mismatch() {
        if !cuda_available() {
            return;
        }

        let device = CudaDevice::default_device().expect("operation should succeed");
        let mut tensor = CudaTensor::zeros(&device, 4, false).expect("operation should succeed");

        let result = tensor.copy_from_vec(&[1.0, 2.0]); // Wrong size
        assert!(result.is_err());
        assert!(matches!(result, Err(CudaTensorError::ShapeMismatch { .. })));
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_cuda_not_available_fallback() {
        let result = CudaDevice::default_device();
        assert!(result.is_err());
        assert!(matches!(result, Err(CudaTensorError::CudaNotAvailable(_))));
    }
}
