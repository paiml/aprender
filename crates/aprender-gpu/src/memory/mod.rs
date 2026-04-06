//! GPU Memory Management
//!
//! Provides memory pooling and transfer utilities for efficient GPU memory usage.
//!
//! ## Features
//!
//! - **Pool allocator**: Reduces allocation overhead
//! - **Transfer utilities**: Efficient H2D/D2H transfers
//! - **Fragmentation tracking**: Per PagedAttention [12]
//! - **GPU-Resident Tensors**: Minimize host↔device transfers (WAPR-PERF-004)

mod pool;
pub mod resident;

pub use pool::{AllocationInfo, MemoryPool, PoolConfig};

use crate::driver::DevicePtr;
use crate::error::Result;

/// GPU buffer wrapper
#[derive(Debug)]
pub struct GpuBuffer<T> {
    ptr: DevicePtr<T>,
    len: usize,
    capacity: usize,
}

impl<T> GpuBuffer<T> {
    /// Create a new uninitialized buffer
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self {
            ptr: DevicePtr::null(),
            len,
            capacity: len,
        }
    }

    /// Get buffer length
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get device pointer
    #[must_use]
    pub const fn as_ptr(&self) -> DevicePtr<T> {
        self.ptr
    }

    /// Get size in bytes
    #[must_use]
    pub const fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

/// Copy data from host to device
///
/// Note: For CUDA operations, use `driver::GpuBuffer::copy_from_host()` directly.
/// This function is a no-op placeholder for the abstract memory API.
pub fn copy_h2d<T: Copy>(_dst: &mut GpuBuffer<T>, _src: &[T]) -> Result<()> {
    // Abstract API - actual transfer via driver::GpuBuffer when CUDA enabled
    Ok(())
}

/// Copy data from device to host
///
/// Note: For CUDA operations, use `driver::GpuBuffer::copy_to_host()` directly.
/// This function is a no-op placeholder for the abstract memory API.
pub fn copy_d2h<T: Copy>(_src: &GpuBuffer<T>, _dst: &mut [T]) -> Result<()> {
    // Abstract API - actual transfer via driver::GpuBuffer when CUDA enabled
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_buffer_creation() {
        let buffer: GpuBuffer<f32> = GpuBuffer::new(1024);
        assert_eq!(buffer.len(), 1024);
        assert_eq!(buffer.size_bytes(), 1024 * 4);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_gpu_buffer_empty() {
        let buffer: GpuBuffer<f32> = GpuBuffer::new(0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_gpu_buffer_as_ptr() {
        let buffer: GpuBuffer<f32> = GpuBuffer::new(100);
        let ptr = buffer.as_ptr();
        assert!(ptr.is_null()); // New buffer starts with null ptr
    }

    #[test]
    fn test_copy_h2d_d2h() {
        let mut buffer: GpuBuffer<f32> = GpuBuffer::new(4);
        let src = [1.0f32, 2.0, 3.0, 4.0];
        // Without CUDA feature, these are no-ops but should not error
        assert!(copy_h2d(&mut buffer, &src).is_ok());

        let mut dst = [0.0f32; 4];
        assert!(copy_d2h(&buffer, &mut dst).is_ok());
    }

    #[test]
    fn test_gpu_buffer_different_types() {
        let buffer_f64: GpuBuffer<f64> = GpuBuffer::new(512);
        assert_eq!(buffer_f64.size_bytes(), 512 * 8);

        let buffer_u8: GpuBuffer<u8> = GpuBuffer::new(1024);
        assert_eq!(buffer_u8.size_bytes(), 1024);
    }
}
