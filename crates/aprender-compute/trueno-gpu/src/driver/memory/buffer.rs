//! GPU buffer types and core operations
//!
//! Defines `GpuBuffer<T>` (owning) and `GpuBufferView<T>` (non-owning)
//! with allocation, deallocation, and metadata access.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem;
use std::ptr;

use crate::driver::context::{get_driver, CudaContext};
use crate::driver::sys::{CUcontext, CUdeviceptr, CudaDriver, CUDA_SUCCESS};
use crate::GpuError;

// ============================================================================
// GPU Buffer
// ============================================================================

/// GPU memory buffer with RAII cleanup
///
/// Allocates device memory and provides safe transfer operations.
/// Memory is automatically freed when dropped.
///
/// # Type Parameter
///
/// * `T` - Element type (must be `Copy` for safe transfer)
///
/// # Example
///
/// ```ignore
/// let ctx = CudaContext::new(0)?;
/// let mut buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 1024)?;
///
/// // Upload data
/// let host_data: Vec<f32> = vec![1.0; 1024];
/// buf.copy_from_host(&host_data)?;
///
/// // Download data
/// let mut result = vec![0.0f32; 1024];
/// buf.copy_to_host(&mut result)?;
/// ```
pub struct GpuBuffer<T> {
    /// Device pointer
    pub(super) ptr: CUdeviceptr,
    /// Number of elements
    pub(super) len: usize,
    /// PMAT-396: Original host pointer for registered buffers (None = device-allocated)
    host_ptr: Option<*mut c_void>,
    /// PMAT-420: Raw CUDA context handle for thread-safe transfers.
    /// Stored at allocation time so every transfer can call cuCtxSetCurrent
    /// even when the buffer has been sent to a different thread.
    pub(crate) ctx: Option<CUcontext>,
    /// Phantom for type parameter
    pub(super) _marker: PhantomData<T>,
}

// SAFETY: GPU memory is accessible from any thread
unsafe impl<T: Send> Send for GpuBuffer<T> {}
unsafe impl<T: Sync> Sync for GpuBuffer<T> {}

impl<T> GpuBuffer<T> {
    /// PAR-023: Create a non-owning buffer from raw device pointer
    ///
    /// # Safety
    ///
    /// - `ptr` must be a valid CUDA device pointer
    /// - The pointed-to memory must be at least `len * size_of::<T>()` bytes
    /// - The caller is responsible for not freeing this buffer's memory
    ///   (use `std::mem::forget` after use)
    ///
    /// # Use Case
    ///
    /// This is useful for creating temporary buffers from cached device pointers
    /// without triggering the borrow checker.
    #[must_use]
    pub unsafe fn from_raw_parts(ptr: CUdeviceptr, len: usize) -> Self {
        Self { ptr, len, host_ptr: None, ctx: None, _marker: PhantomData }
    }

    /// Allocate a new GPU buffer
    ///
    /// # Arguments
    ///
    /// * `_ctx` - CUDA context (must be current)
    /// * `len` - Number of elements to allocate
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::MemoryAllocation)` if allocation fails.
    /// Returns `Err(GpuError::OutOfMemory)` if insufficient GPU memory.
    pub fn new(ctx: &CudaContext, len: usize) -> Result<Self, GpuError> {
        let ctx_handle = Some(ctx.raw());

        if len == 0 {
            return Ok(Self {
                ptr: 0,
                len: 0,
                host_ptr: None,
                ctx: ctx_handle,
                _marker: PhantomData,
            });
        }

        // PMAT-394: Use managed memory on Grace Blackwell when MANAGED_MEMORY=1
        static USE_MANAGED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let managed =
            *USE_MANAGED.get_or_init(|| std::env::var("MANAGED_MEMORY").as_deref() == Ok("1"));
        if managed {
            return Self::new_managed(ctx, len);
        }

        let driver = get_driver()?;
        let size = len * mem::size_of::<T>();

        let mut ptr: CUdeviceptr = 0;
        // SAFETY: ptr is valid, size is computed correctly
        let result = unsafe { (driver.cuMemAlloc)(&mut ptr, size) };
        CudaDriver::check(result).map_err(|e| GpuError::MemoryAllocation(e.to_string()))?;

        Ok(Self { ptr, len, host_ptr: None, ctx: ctx_handle, _marker: PhantomData })
    }

    /// PMAT-394: Allocate managed (unified) memory for Grace Blackwell.
    /// GPU accesses via NVLink-C2C, no explicit copy needed.
    /// `cuMemFree` works for both managed and device allocations.
    pub fn new_managed(ctx: &CudaContext, len: usize) -> Result<Self, GpuError> {
        let ctx_handle = Some(ctx.raw());

        if len == 0 {
            return Ok(Self {
                ptr: 0,
                len: 0,
                host_ptr: None,
                ctx: ctx_handle,
                _marker: PhantomData,
            });
        }
        let driver = get_driver()?;
        let size = len * mem::size_of::<T>();
        let mut ptr: CUdeviceptr = 0;
        const CU_MEM_ATTACH_GLOBAL: u32 = 1;
        let result = unsafe { (driver.cuMemAllocManaged)(&mut ptr, size, CU_MEM_ATTACH_GLOBAL) };
        CudaDriver::check(result).map_err(|e| {
            GpuError::MemoryAllocation(format!("cuMemAllocManaged({} bytes): {}", size, e))
        })?;
        Ok(Self { ptr, len, host_ptr: None, ctx: ctx_handle, _marker: PhantomData })
    }

    /// PMAT-396: Register existing host memory for GPU access (zero-copy).
    /// On Grace Blackwell, GPU accesses same physical pages via NVLink-C2C.
    ///
    /// # Safety
    /// `host_ptr` must be page-aligned, valid for `len * size_of::<T>()`,
    /// and must outlive this buffer. Drop does NOT free the host memory.
    pub unsafe fn from_host_registered(host_ptr: *mut T, len: usize) -> Result<Self, GpuError> {
        if len == 0 {
            return Ok(Self { ptr: 0, len: 0, host_ptr: None, ctx: None, _marker: PhantomData });
        }
        let driver = get_driver()?;
        let size = len * mem::size_of::<T>();
        const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 0x02;
        // SAFETY: cuMemHostRegister/cuMemHostGetDevicePointer are FFI calls.
        // host_ptr is a valid allocation provided by the caller.
        let result = unsafe {
            (driver.cuMemHostRegister)(host_ptr as *mut c_void, size, CU_MEMHOSTREGISTER_DEVICEMAP)
        };
        CudaDriver::check(result).map_err(|e| {
            GpuError::MemoryAllocation(format!("cuMemHostRegister({} bytes): {}", size, e))
        })?;
        let mut dev_ptr: CUdeviceptr = 0;
        let result =
            unsafe { (driver.cuMemHostGetDevicePointer)(&mut dev_ptr, host_ptr as *mut c_void, 0) };
        CudaDriver::check(result)
            .map_err(|e| GpuError::MemoryAllocation(format!("cuMemHostGetDevicePointer: {}", e)))?;
        Ok(Self {
            ptr: dev_ptr,
            len,
            host_ptr: Some(host_ptr as *mut c_void),
            ctx: None,
            _marker: PhantomData,
        })
    }

    /// Zero buffer on GPU asynchronously (no PCIe transfer).
    pub fn zero_async(&mut self, stream: &crate::driver::CudaStream) -> Result<(), GpuError> {
        if self.len == 0 {
            return Ok(());
        }
        self.ensure_context()?;
        let driver = get_driver()?;
        let result = unsafe { (driver.cuMemsetD32Async)(self.ptr, 0, self.len, stream.raw()) };
        if result != CUDA_SUCCESS {
            return Err(GpuError::Transfer(format!("cuMemsetD32Async failed: {result}")));
        }
        Ok(())
    }

    /// Get device pointer as raw u64
    #[must_use]
    pub fn as_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Get number of elements
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// PMAT-420: Set the CUDA context for thread-safe transfers.
    ///
    /// Normally the context is captured automatically at allocation time.
    /// Use this only for buffers created via `from_raw_parts` or
    /// `from_host_registered` where no `CudaContext` was available.
    pub fn set_context(&mut self, ctx: &CudaContext) {
        self.ctx = Some(ctx.raw());
    }

    /// PMAT-420: Ensure the CUDA context stored at allocation time is current
    /// on the calling thread before any driver API call (memcpy, kernel launch).
    ///
    /// cuMemcpyHtoD / cuMemcpyDtoH silently produce zeros when the context
    /// is not current, which is the root cause of paiml/trueno#232.
    pub(crate) fn ensure_context(&self) -> Result<(), GpuError> {
        if let Some(ctx_handle) = self.ctx {
            let driver = get_driver()?;
            // SAFETY: ctx_handle was obtained from CudaContext::raw() which
            // returns a primary-context handle that remains valid for the
            // lifetime of the process (ref-counted by cuDevicePrimaryCtxRetain).
            let result = unsafe { (driver.cuCtxSetCurrent)(ctx_handle) };
            if result != CUDA_SUCCESS {
                return Err(GpuError::DeviceInit(format!(
                    "PMAT-420: cuCtxSetCurrent failed with code {} — \
                     context may have been destroyed",
                    result
                )));
            }
        }
        Ok(())
    }

    /// Get size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.len * mem::size_of::<T>()
    }

    /// PAR-023: Create a non-owning clone of the buffer metadata
    ///
    /// Creates a new GpuBuffer that points to the same device memory but
    /// does NOT own it. The returned buffer will NOT free the memory when dropped.
    ///
    /// # Safety
    ///
    /// The caller MUST ensure the original buffer outlives any clones.
    /// The returned buffer should typically be wrapped with `ManuallyDrop` or
    /// `std::mem::forget` to prevent the Drop impl from running.
    ///
    /// # Use Case
    ///
    /// This is useful for passing cached GPU buffers to functions that take
    /// `&GpuBuffer<T>` while avoiding borrow checker conflicts.
    #[must_use]
    pub fn clone_metadata(&self) -> GpuBufferView<T> {
        GpuBufferView { ptr: self.ptr, len: self.len, _marker: PhantomData }
    }
}

// ============================================================================
// GPU Buffer View (non-owning)
// ============================================================================

/// PAR-023: Non-owning view of a GPU buffer
///
/// This struct points to GPU memory but does NOT free it when dropped.
/// Use this for temporary references to cached GPU buffers.
pub struct GpuBufferView<T> {
    ptr: CUdeviceptr,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> GpuBufferView<T> {
    /// Get device pointer as raw u64
    #[must_use]
    pub fn as_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Get number of elements
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

// ============================================================================
// Drop + Kernel Arg
// ============================================================================

impl<T> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            if let Ok(driver) = get_driver() {
                unsafe {
                    if let Some(host_ptr) = self.host_ptr {
                        // PMAT-396: Unregister host memory (don't free it)
                        let _ = (driver.cuMemHostUnregister)(host_ptr);
                    } else {
                        // Standard device/managed memory
                        let _ = (driver.cuMemFree)(self.ptr);
                    }
                }
            }
        }
    }
}

impl<T> GpuBuffer<T> {
    /// Get pointer to device pointer for kernel arguments
    ///
    /// Returns a pointer that can be passed to kernel launch.
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid while this buffer is alive.
    #[must_use]
    pub fn as_kernel_arg(&self) -> *mut c_void {
        // The kernel expects a pointer to the device pointer
        ptr::addr_of!(self.ptr) as *mut c_void
    }
}
