//! GPU memory transfer operations
//!
//! Host-to-device, device-to-host, and device-to-device copy methods
//! for `GpuBuffer<T>`. Both synchronous and asynchronous variants.

use std::ffi::c_void;

use crate::driver::context::{get_driver, CudaContext};
use crate::driver::stream::CudaStream;
use crate::driver::sys::CudaDriver;
use crate::GpuError;

// PMAT-420: All transfer methods call self.ensure_context() before any
// cuMemcpy* call.  Without this, cuMemcpyHtoD/DtoH silently produces
// zeros when the CUDA context is not current on the calling thread
// (see paiml/trueno#232).

use super::buffer::GpuBuffer;

// ============================================================================
// Host <-> Device Transfers
// ============================================================================

impl<T: Copy> GpuBuffer<T> {
    /// Copy data from host to device (synchronous)
    ///
    /// # Arguments
    ///
    /// * `data` - Host data to copy (must have same length as buffer)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if copy fails.
    /// Returns `Err(GpuError::InvalidValue)` if lengths don't match.
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<(), GpuError> {
        if data.len() != self.len {
            return Err(GpuError::Transfer(format!(
                "Length mismatch: host {} vs device {}",
                data.len(),
                self.len
            )));
        }

        if self.len == 0 {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = self.size_bytes();

        // SAFETY: data is valid for size bytes, ptr is valid device pointer
        let result =
            unsafe { (driver.cuMemcpyHtoD)(self.ptr, data.as_ptr() as *const c_void, size) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Copy data from device to host (synchronous)
    ///
    /// Supports partial readback: if `data.len() < self.len`, copies only the first
    /// `data.len()` elements. This is safe because cuMemcpyDtoH respects the size parameter.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if `data.len() > self.len` or copy fails.
    pub fn copy_to_host(&self, data: &mut [T]) -> Result<(), GpuError> {
        if data.len() > self.len {
            return Err(GpuError::Transfer(format!(
                "Host buffer too large: host {} > device {}",
                data.len(),
                self.len
            )));
        }

        if data.is_empty() {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = data.len() * std::mem::size_of::<T>();

        // SAFETY: data is valid for size bytes, ptr is valid device pointer
        let result =
            unsafe { (driver.cuMemcpyDtoH)(data.as_mut_ptr() as *mut c_void, self.ptr, size) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Copy data from host to device (asynchronous)
    ///
    /// # Arguments
    ///
    /// * `data` - Host data to copy (must have same length as buffer)
    /// * `stream` - Stream for async operation
    ///
    /// # Safety
    ///
    /// The host data must remain valid until the stream is synchronized.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if copy fails.
    pub unsafe fn copy_from_host_async(
        &mut self,
        data: &[T],
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        if data.len() != self.len {
            return Err(GpuError::Transfer(format!(
                "Length mismatch: host {} vs device {}",
                data.len(),
                self.len
            )));
        }

        if self.len == 0 {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = self.size_bytes();

        // SAFETY: data is valid for size bytes, caller ensures data outlives stream ops
        let result = unsafe {
            (driver.cuMemcpyHtoDAsync)(self.ptr, data.as_ptr() as *const c_void, size, stream.raw())
        };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Copy data from device to host (asynchronous)
    ///
    /// # Arguments
    ///
    /// * `data` - Host buffer to copy into
    /// * `stream` - Stream for async operation
    ///
    /// # Safety
    ///
    /// The host buffer must remain valid until the stream is synchronized.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if copy fails.
    pub unsafe fn copy_to_host_async(
        &self,
        data: &mut [T],
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        if data.len() != self.len {
            return Err(GpuError::Transfer(format!(
                "Length mismatch: host {} vs device {}",
                data.len(),
                self.len
            )));
        }

        if self.len == 0 {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = self.size_bytes();

        // SAFETY: data is valid for size bytes, caller ensures data outlives stream ops
        let result = unsafe {
            (driver.cuMemcpyDtoHAsync)(
                data.as_mut_ptr() as *mut c_void,
                self.ptr,
                size,
                stream.raw(),
            )
        };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Create buffer and initialize from host data
    ///
    /// Convenience method combining allocation and upload.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `data` - Host data to upload
    ///
    /// # Errors
    ///
    /// Returns allocation or transfer errors.
    pub fn from_host(ctx: &CudaContext, data: &[T]) -> Result<Self, GpuError> {
        let mut buf = Self::new(ctx, data.len())?;
        buf.copy_from_host(data)?;
        Ok(buf)
    }

    /// Copy partial data from host to device at specific offset (PAR-018)
    ///
    /// # Arguments
    ///
    /// * `data` - Host data to copy
    /// * `offset` - Element offset in device buffer where copy begins
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if offset + data.len() exceeds buffer size.
    pub fn copy_from_host_at(&mut self, data: &[T], offset: usize) -> Result<(), GpuError> {
        if offset + data.len() > self.len {
            return Err(GpuError::Transfer(format!(
                "Partial copy out of bounds: offset {} + len {} > buffer {}",
                offset,
                data.len(),
                self.len
            )));
        }

        if data.is_empty() {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = std::mem::size_of_val(data);
        let dst_ptr = self.ptr + (offset * std::mem::size_of::<T>()) as u64;

        // SAFETY: bounds checked above, data and ptr are valid
        let result =
            unsafe { (driver.cuMemcpyHtoD)(dst_ptr, data.as_ptr() as *const c_void, size) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Copy partial data from device to host at specific offset (PAR-018)
    ///
    /// # Arguments
    ///
    /// * `data` - Host buffer to copy into
    /// * `offset` - Element offset in device buffer where copy begins
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if offset + data.len() exceeds buffer size.
    pub fn copy_to_host_at(&self, data: &mut [T], offset: usize) -> Result<(), GpuError> {
        if offset + data.len() > self.len {
            return Err(GpuError::Transfer(format!(
                "Partial copy out of bounds: offset {} + len {} > buffer {}",
                offset,
                data.len(),
                self.len
            )));
        }

        if data.is_empty() {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = std::mem::size_of_val(data);
        let src_ptr = self.ptr + (offset * std::mem::size_of::<T>()) as u64;

        // SAFETY: bounds checked above, data and ptr are valid
        let result =
            unsafe { (driver.cuMemcpyDtoH)(data.as_mut_ptr() as *mut c_void, src_ptr, size) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    // =========================================================================
    // PAR-023: Device-to-Device Copy (Zero-Sync Pipeline)
    // =========================================================================

    /// Clone buffer to new GPU memory (device-to-device copy)
    ///
    /// Allocates new GPU memory and copies contents from self.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context (must be current)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::MemoryAllocation)` if allocation fails.
    /// Returns `Err(GpuError::Transfer)` if copy fails.
    pub fn clone(&self, ctx: &CudaContext) -> Result<Self, GpuError> {
        let mut new_buffer = GpuBuffer::new(ctx, self.len)?;
        new_buffer.copy_from_buffer(self)?;
        Ok(new_buffer)
    }

    /// Copy data from another GPU buffer (device-to-device, synchronous)
    ///
    /// Enables zero-sync GPU pipelines by keeping data on device.
    ///
    /// # Arguments
    ///
    /// * `src` - Source GPU buffer (must have same length)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if lengths don't match or copy fails.
    pub fn copy_from_buffer(&mut self, src: &GpuBuffer<T>) -> Result<(), GpuError> {
        if src.len != self.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: D2D length mismatch: src {} vs dst {}",
                src.len, self.len
            )));
        }

        if self.len == 0 {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = self.size_bytes();

        // SAFETY: both buffers are valid, size is correct
        let result = unsafe { (driver.cuMemcpyDtoD)(self.ptr, src.ptr, size) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Copy partial data from another GPU buffer at specific offset (PAR-023)
    ///
    /// Enables GPU-resident KV cache updates without host round-trip.
    ///
    /// # Arguments
    ///
    /// * `src` - Source GPU buffer
    /// * `dst_offset` - Element offset in destination (this buffer)
    /// * `src_offset` - Element offset in source buffer
    /// * `count` - Number of elements to copy
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if copy would exceed buffer bounds.
    pub fn copy_from_buffer_at(
        &mut self,
        src: &GpuBuffer<T>,
        dst_offset: usize,
        src_offset: usize,
        count: usize,
    ) -> Result<(), GpuError> {
        if dst_offset + count > self.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: D2D dst out of bounds: {} + {} > {}",
                dst_offset, count, self.len
            )));
        }
        if src_offset + count > src.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: D2D src out of bounds: {} + {} > {}",
                src_offset, count, src.len
            )));
        }

        if count == 0 {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = count * std::mem::size_of::<T>();
        let dst_ptr = self.ptr + (dst_offset * std::mem::size_of::<T>()) as u64;
        let src_ptr = src.ptr + (src_offset * std::mem::size_of::<T>()) as u64;

        // SAFETY: bounds checked above, both ptrs are valid
        let result = unsafe { (driver.cuMemcpyDtoD)(dst_ptr, src_ptr, size) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Async copy from another GPU buffer (PAR-023)
    ///
    /// # Safety
    ///
    /// Both buffers must remain valid until stream is synchronized.
    pub unsafe fn copy_from_buffer_async(
        &mut self,
        src: &GpuBuffer<T>,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        if src.len != self.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: Async D2D length mismatch: src {} vs dst {}",
                src.len, self.len
            )));
        }

        if self.len == 0 {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = self.size_bytes();

        // SAFETY: both buffers valid, caller ensures lifetime
        let result = unsafe { (driver.cuMemcpyDtoDAsync)(self.ptr, src.ptr, size, stream.raw()) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// PAR-023: Async D2D copy with offsets
    ///
    /// Copies a region from source buffer to destination buffer asynchronously.
    /// Does not synchronize - caller must ensure stream sync before accessing data.
    ///
    /// # Arguments
    ///
    /// * `src` - Source GPU buffer
    /// * `dst_offset` - Element offset in destination (this buffer)
    /// * `src_offset` - Element offset in source buffer
    /// * `count` - Number of elements to copy
    /// * `stream` - CUDA stream for async operation
    ///
    /// # Safety
    ///
    /// Both buffers must remain valid until stream is synchronized.
    pub unsafe fn copy_from_buffer_at_async(
        &mut self,
        src: &GpuBuffer<T>,
        dst_offset: usize,
        src_offset: usize,
        count: usize,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        if dst_offset + count > self.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: Async D2D dst out of bounds: {} + {} > {}",
                dst_offset, count, self.len
            )));
        }
        if src_offset + count > src.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: Async D2D src out of bounds: {} + {} > {}",
                src_offset, count, src.len
            )));
        }

        if count == 0 {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = count * std::mem::size_of::<T>();
        let dst_ptr = self.ptr + (dst_offset * std::mem::size_of::<T>()) as u64;
        let src_ptr = src.ptr + (src_offset * std::mem::size_of::<T>()) as u64;

        // SAFETY: bounds checked above, both ptrs are valid, caller ensures lifetime
        let result = unsafe { (driver.cuMemcpyDtoDAsync)(dst_ptr, src_ptr, size, stream.raw()) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// PAR-023: Async D2D copy with raw stream handle
    ///
    /// Same as `copy_from_buffer_at_async` but takes raw CUstream handle.
    /// Useful when borrow checker prevents passing &CudaStream due to other borrows.
    ///
    /// # Safety
    ///
    /// - Both buffers must remain valid until stream is synchronized.
    /// - Stream handle must be valid.
    pub unsafe fn copy_from_buffer_at_async_raw(
        &mut self,
        src: &GpuBuffer<T>,
        dst_offset: usize,
        src_offset: usize,
        count: usize,
        stream_handle: crate::driver::sys::CUstream,
    ) -> Result<(), GpuError> {
        if dst_offset + count > self.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: Async D2D dst out of bounds: {} + {} > {}",
                dst_offset, count, self.len
            )));
        }
        if src_offset + count > src.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: Async D2D src out of bounds: {} + {} > {}",
                src_offset, count, src.len
            )));
        }

        if count == 0 {
            return Ok(());
        }

        self.ensure_context()?;

        let driver = get_driver()?;
        let size = count * std::mem::size_of::<T>();
        let dst_ptr = self.ptr + (dst_offset * std::mem::size_of::<T>()) as u64;
        let src_ptr = src.ptr + (src_offset * std::mem::size_of::<T>()) as u64;

        // SAFETY: bounds checked above, both ptrs valid, caller ensures lifetime + stream valid
        let result = unsafe { (driver.cuMemcpyDtoDAsync)(dst_ptr, src_ptr, size, stream_handle) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }
}
