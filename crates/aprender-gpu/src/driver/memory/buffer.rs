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

// CUDA driver device attribute IDs (cuda.h, CU_DEVICE_ATTRIBUTE_*).
// Local consts keep the buffer module self-contained without growing the
// public sys API.
const CU_DEVICE_ATTRIBUTE_INTEGRATED: i32 = 18;

/// Memory architecture class of a CUDA device.
///
/// Used by [`GpuBuffer::new`] to decide which allocator to dispatch to.
/// See `contracts/trueno-gpu/cuda-unified-memory-allocator-v1.yaml` for
/// the full contract; this enum is the runtime witness of the
/// `device_class_classification` equation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemoryClass {
    /// Integrated GPU sharing the system memory pool (Grace Blackwell GB10,
    /// Tegra Jetson, future NVL-class). Default allocator must use
    /// `cuMemAllocManaged` to access the full unified pool.
    UnifiedMemory,
    /// Classic discrete GPU with its own VRAM partition (Ada, Hopper,
    /// Ampere, Turing, Volta). Default allocator uses `cuMemAlloc` —
    /// no behavior change from pre-PMAT-701.
    ClassicDevice,
}

/// Query the CUDA driver for the device's memory architecture class.
///
/// Reads `CU_DEVICE_ATTRIBUTE_INTEGRATED` via `cuDeviceGetAttribute`:
/// returns `1` for integrated GPUs (Grace, Tegra), `0` for discrete dGPUs.
/// This is the cleanest single-attribute classification — INTEGRATED
/// implies the device has no separate VRAM partition and therefore the
/// distinction between "device memory" and "system memory" collapses.
///
/// # Errors
///
/// Returns `Err(GpuError::CudaDriver)` if the attribute query fails.
pub fn classify_device_memory(ctx: &CudaContext) -> Result<DeviceMemoryClass, GpuError> {
    let driver = get_driver()?;
    let mut integrated: i32 = 0;
    // SAFETY: device handle from CudaContext is valid; out-pointer is on the
    // stack; integrated attribute is a documented driver attribute.
    let result = unsafe {
        (driver.cuDeviceGetAttribute)(
            &mut integrated,
            CU_DEVICE_ATTRIBUTE_INTEGRATED,
            ctx.device(),
        )
    };
    CudaDriver::check(result)?;
    if integrated == 1 {
        Ok(DeviceMemoryClass::UnifiedMemory)
    } else {
        Ok(DeviceMemoryClass::ClassicDevice)
    }
}

/// Allocator-selection decision after consulting env override and device class.
///
/// `MANAGED_MEMORY` env var values:
///   - `"1"`         -> force `cuMemAllocManaged` (legacy opt-in, still honored)
///   - `"0"`         -> force `cuMemAlloc` (new diagnostics escape hatch)
///   - unset / other -> follow `classify_device_memory(ctx)` (PMAT-701 default)
fn should_use_managed_memory(ctx: &CudaContext) -> bool {
    match std::env::var("MANAGED_MEMORY").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => classify_device_memory(ctx)
            .map(|c| c == DeviceMemoryClass::UnifiedMemory)
            .unwrap_or(false),
    }
}

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
        Self {
            ptr,
            len,
            host_ptr: None,
            ctx: None,
            _marker: PhantomData,
        }
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

        // PMAT-701: Autodetect unified-memory devices (Grace Blackwell) and
        // route to cuMemAllocManaged by default. PMAT-394's env-var opt-in
        // is preserved for forcing/forbidding managed mode explicitly.
        // Contract: contracts/trueno-gpu/cuda-unified-memory-allocator-v1.yaml
        if should_use_managed_memory(ctx) {
            return Self::new_managed(ctx, len);
        }

        let driver = get_driver()?;
        let size = len * mem::size_of::<T>();

        let mut ptr: CUdeviceptr = 0;
        // SAFETY: ptr is valid, size is computed correctly
        let result = unsafe { (driver.cuMemAlloc)(&mut ptr, size) };
        CudaDriver::check(result).map_err(|e| GpuError::MemoryAllocation(e.to_string()))?;

        Ok(Self {
            ptr,
            len,
            host_ptr: None,
            ctx: ctx_handle,
            _marker: PhantomData,
        })
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
        Ok(Self {
            ptr,
            len,
            host_ptr: None,
            ctx: ctx_handle,
            _marker: PhantomData,
        })
    }

    /// PMAT-769: Register host memory for GPU access, or copy into a managed
    /// buffer on unified-memory devices that reject `cuMemHostRegister`.
    ///
    /// Grace-Blackwell (GB10), Grace-Hopper, and Jetson expose a single coherent
    /// physical memory pool. On those parts `cuMemHostRegister` is both
    /// **unnecessary** (host and device pages are already coherent) and
    /// **unsupported** — the driver returns `CUDA_ERROR_UNKNOWN (712)` /
    /// `CUDA_ERROR_INVALID_VALUE (1)`, which previously aborted weight loading and
    /// left `workspace.q8_activation_buf` uninitialized (the cascade that broke
    /// ~21 `cuda::` serve lib tests on gx10).
    ///
    /// Behavior, gated strictly on device class so discrete GPUs are unaffected:
    /// - **Unified memory** (`CU_DEVICE_ATTRIBUTE_INTEGRATED == 1`): allocate a
    ///   managed buffer via `cuMemAllocManaged` and copy the host bytes in. The
    ///   buffer is device-accessible (plain pointer is valid on the GPU) and is
    ///   freed via `cuMemFree` on drop. `cuMemHostRegister`/`cuMemHostUnregister`
    ///   are skipped entirely.
    /// - **Discrete GPU** (`INTEGRATED == 0`, e.g. RTX 4090): identical to the
    ///   legacy [`from_host_registered`](Self::from_host_registered) — host pages
    ///   are pinned + device-mapped (a real H2D-DMA perf win). No behavior change.
    ///
    /// As a belt-and-suspenders fallback, a `cuMemHostRegister` failure on a
    /// device we classified as discrete is treated as "this host can't pin" and
    /// also degrades to the managed-copy path rather than failing the load.
    ///
    /// # Safety
    /// `host_ptr` must be valid for reads of `len * size_of::<T>()` bytes for the
    /// duration of this call. On the discrete path it must additionally be
    /// page-aligned and outlive the buffer (Drop does NOT free host memory); on
    /// the unified path the bytes are copied, so the host allocation may be freed
    /// independently afterward.
    pub unsafe fn from_host_registered_or_managed(
        ctx: &CudaContext,
        host_ptr: *mut T,
        len: usize,
    ) -> Result<Self, GpuError>
    where
        T: Copy,
    {
        if len == 0 {
            return Ok(Self {
                ptr: 0,
                len: 0,
                host_ptr: None,
                ctx: Some(ctx.raw()),
                _marker: PhantomData,
            });
        }

        // Unified-memory devices reject cuMemHostRegister — skip pinning and use
        // a coherent managed allocation instead. Discrete GPUs keep pinning.
        let is_unified = classify_device_memory(ctx)
            .map(|c| c == DeviceMemoryClass::UnifiedMemory)
            .unwrap_or(false);

        if !is_unified {
            // SAFETY: caller upholds from_host_registered's contract on the
            // discrete path (page-aligned, valid, outlives buffer).
            match unsafe { Self::from_host_registered(host_ptr, len) } {
                Ok(buf) => return Ok(buf),
                // Belt-and-suspenders: a host that reports discrete but still
                // refuses to pin (ambiguous hardware) degrades to managed copy
                // rather than failing the model load.
                Err(GpuError::MemoryAllocation(_)) => {}
                Err(e) => return Err(e),
            }
        }

        // Unified (or pin-refusing) path: managed buffer + host copy.
        // SAFETY: host_ptr is valid for len elements per the caller's contract.
        let host_slice = unsafe { std::slice::from_raw_parts(host_ptr as *const T, len) };
        let mut buf = Self::new_managed(ctx, len)?;
        buf.copy_from_host(host_slice)?;
        Ok(buf)
    }

    /// PMAT-396: Register existing host memory for GPU access (zero-copy).
    /// On Grace Blackwell, GPU accesses same physical pages via NVLink-C2C.
    ///
    /// NOTE: On unified-memory parts (GB10/Jetson) `cuMemHostRegister` is
    /// unsupported; prefer [`from_host_registered_or_managed`](Self::from_host_registered_or_managed)
    /// which auto-detects and falls back. This raw entry point is retained for
    /// discrete GPUs and existing callers.
    ///
    /// # Safety
    /// `host_ptr` must be page-aligned, valid for `len * size_of::<T>()`,
    /// and must outlive this buffer. Drop does NOT free the host memory.
    pub unsafe fn from_host_registered(host_ptr: *mut T, len: usize) -> Result<Self, GpuError> {
        if len == 0 {
            return Ok(Self {
                ptr: 0,
                len: 0,
                host_ptr: None,
                ctx: None,
                _marker: PhantomData,
            });
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
            return Err(GpuError::Transfer(format!(
                "cuMemsetD32Async failed: {result}"
            )));
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
        GpuBufferView {
            ptr: self.ptr,
            len: self.len,
            _marker: PhantomData,
        }
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
