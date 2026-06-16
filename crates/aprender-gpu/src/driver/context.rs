//! CUDA Context Management
//!
//! Provides safe RAII wrapper for CUDA contexts using the Primary Context API.
//!
//! # Design Philosophy
//!
//! Uses Primary Context API (cuDevicePrimaryCtxRetain) instead of cuCtxCreate:
//! - Shared across all modules in the process
//! - Reference counted by CUDA driver
//! - More efficient for multi-module applications
//!
//! # Citation
//!
//! [5] NVIDIA CUDA C++ Programming Guide v12.3, Section 3.2 "CUDA Contexts"
//!     recommends Primary Context API for applications using multiple modules.

use std::os::raw::c_char;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

use super::sys::{CUcontext, CUdevice, CudaDriver, CUDA_SUCCESS};
use crate::GpuError;

// ============================================================================
// Global Initialization State
// ============================================================================

/// Serialize the one-time `cuInit` call. A bare `AtomicBool::swap` guard is a
/// TOCTOU race: a second thread could observe the flag already `true` and call
/// `cuDeviceGetCount` / `cuDevicePrimaryCtxRetain` *before* the first thread's
/// `cuInit` returned, getting `count == 0` or a spurious init failure. This
/// surfaced as flaky `num_devices()`/`CudaExecutor::new(0)` failures when GPU
/// tests run in parallel. `Once` makes every caller block until `cuInit`
/// completes exactly once.
static CUDA_INIT_ONCE: Once = Once::new();
/// Records whether the one-time `cuInit` succeeded (read after `call_once`).
static CUDA_INIT_OK: AtomicBool = AtomicBool::new(false);

/// Get the CUDA driver, initializing if needed
///
/// # Errors
///
/// Returns `Err(GpuError::CudaNotAvailable)` if CUDA driver is not installed.
/// Returns `Err(GpuError::DeviceInit)` if cuInit fails.
pub(crate) fn get_driver() -> Result<&'static CudaDriver, GpuError> {
    let driver = CudaDriver::load()
        .ok_or_else(|| GpuError::CudaNotAvailable("CUDA driver not found".to_string()))?;

    // Initialize CUDA exactly once; concurrent callers BLOCK here until the
    // single cuInit completes, then all observe the result via CUDA_INIT_OK.
    CUDA_INIT_ONCE.call_once(|| {
        // SAFETY: cuInit(0) is the standard CUDA driver init entry point.
        let result = unsafe { (driver.cuInit)(0) };
        CUDA_INIT_OK.store(result == CUDA_SUCCESS, Ordering::SeqCst);
    });

    if !CUDA_INIT_OK.load(Ordering::SeqCst) {
        return Err(GpuError::DeviceInit(
            "cuInit failed during CUDA driver initialization".to_string(),
        ));
    }

    Ok(driver)
}

// ============================================================================
// CUDA Context
// ============================================================================

/// CUDA context with RAII cleanup
///
/// Uses Primary Context API for efficient multi-module sharing.
/// Automatically releases context when dropped.
///
/// # Example
///
/// ```ignore
/// let ctx = CudaContext::new(0)?; // Device 0
/// let (free, total) = ctx.memory_info()?;
/// println!("GPU memory: {} / {} bytes free", free, total);
/// ```
pub struct CudaContext {
    /// Device ordinal
    device: CUdevice,
    /// Primary context handle
    context: CUcontext,
}

// SAFETY: CUcontext handles are thread-safe when using Primary Context API
unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    /// Create a new CUDA context for the specified device
    ///
    /// Uses Primary Context API (cuDevicePrimaryCtxRetain) which shares
    /// context across all users in the process.
    ///
    /// # Arguments
    ///
    /// * `device_ordinal` - Device index (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::DeviceNotFound)` if device doesn't exist.
    /// Returns `Err(GpuError::DeviceInit)` if context creation fails.
    pub fn new(device_ordinal: i32) -> Result<Self, GpuError> {
        let driver = get_driver()?;

        // Get device count
        let mut count: i32 = 0;
        // SAFETY: count is a valid pointer
        let result = unsafe { (driver.cuDeviceGetCount)(&mut count) };
        CudaDriver::check(result)?;

        if device_ordinal < 0 || device_ordinal >= count {
            return Err(GpuError::DeviceNotFound(device_ordinal, count as usize));
        }

        // Get device handle
        let mut device: CUdevice = 0;
        // SAFETY: device_ordinal is validated above
        let result = unsafe { (driver.cuDeviceGet)(&mut device, device_ordinal) };
        CudaDriver::check(result)?;

        // Retain primary context
        let mut context: CUcontext = ptr::null_mut();
        // SAFETY: device is a valid handle from cuDeviceGet
        let result = unsafe { (driver.cuDevicePrimaryCtxRetain)(&mut context, device) };
        CudaDriver::check(result)?;

        // Set as current context
        // SAFETY: context is valid from cuDevicePrimaryCtxRetain
        let result = unsafe { (driver.cuCtxSetCurrent)(context) };
        if result != CUDA_SUCCESS {
            // Release context on failure
            unsafe { (driver.cuDevicePrimaryCtxRelease)(device) };
            return Err(GpuError::DeviceInit(format!(
                "cuCtxSetCurrent failed with code {}",
                result
            )));
        }

        Ok(Self { device, context })
    }

    /// Get device ordinal
    #[must_use]
    pub fn device(&self) -> i32 {
        self.device
    }

    /// Get raw context handle
    ///
    /// # Safety
    ///
    /// The returned handle is only valid while this `CudaContext` is alive.
    #[must_use]
    pub fn raw(&self) -> CUcontext {
        self.context
    }

    /// Query free and total device memory
    ///
    /// # Returns
    ///
    /// Tuple of (free_bytes, total_bytes)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::CudaDriver)` if query fails.
    pub fn memory_info(&self) -> Result<(usize, usize), GpuError> {
        let driver = get_driver()?;

        let mut free: usize = 0;
        let mut total: usize = 0;

        // SAFETY: pointers are valid
        let result = unsafe { (driver.cuMemGetInfo)(&mut free, &mut total) };
        CudaDriver::check(result)?;

        Ok((free, total))
    }

    /// Set this context as the current context for the calling thread
    ///
    /// CUDA contexts are thread-local. When using async/multi-threaded code,
    /// you must call `make_current()` before any CUDA operation if the operation
    /// might run on a different thread than where the context was created.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::DeviceInit)` if cuCtxSetCurrent fails.
    pub fn make_current(&self) -> Result<(), GpuError> {
        let driver = get_driver()?;

        // SAFETY: context is valid (from cuDevicePrimaryCtxRetain in constructor)
        let result = unsafe { (driver.cuCtxSetCurrent)(self.context) };
        if result != CUDA_SUCCESS {
            return Err(GpuError::DeviceInit(format!(
                "cuCtxSetCurrent failed with code {}",
                result
            )));
        }
        Ok(())
    }

    /// Synchronize all work on this context
    ///
    /// Blocks until all preceding commands have completed.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::StreamSync)` if synchronization fails.
    pub fn synchronize(&self) -> Result<(), GpuError> {
        let driver = get_driver()?;

        // SAFETY: context is current (set in constructor)
        let result = unsafe { (driver.cuCtxSynchronize)() };
        CudaDriver::check(result).map_err(|e| GpuError::StreamSync(e.to_string()))
    }

    /// Get device name
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::CudaDriver)` if query fails.
    pub fn device_name(&self) -> Result<String, GpuError> {
        let driver = get_driver()?;

        let mut name = [0 as c_char; 256];
        // SAFETY: buffer is valid and large enough
        let result = unsafe { (driver.cuDeviceGetName)(name.as_mut_ptr(), 256, self.device) };
        CudaDriver::check(result)?;

        // Convert to Rust string
        let name_str = unsafe {
            std::ffi::CStr::from_ptr(name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        Ok(name_str)
    }

    /// Get device compute capability as (major, minor)
    ///
    /// Returns the SM architecture version, e.g. (8, 9) for RTX 4090.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::CudaDriver)` if query fails.
    pub fn compute_capability(&self) -> Result<(i32, i32), GpuError> {
        let driver = get_driver()?;

        // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
        // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
        let mut major: i32 = 0;
        let mut minor: i32 = 0;

        // SAFETY: pointer is valid, device is valid
        let result = unsafe { (driver.cuDeviceGetAttribute)(&mut major, 75, self.device) };
        CudaDriver::check(result)?;

        let result = unsafe { (driver.cuDeviceGetAttribute)(&mut minor, 76, self.device) };
        CudaDriver::check(result)?;

        Ok((major, minor))
    }

    /// Get device compute capability as SM target string (e.g. "sm_89", "sm_121")
    ///
    /// Returns the REAL compute capability without clamping. The PTX ISA version
    /// is selected by `ptx_version()` to match -- PTX 8.8 for sm_100+, PTX 8.0
    /// for sm_90 and below.
    ///
    /// trueno#188: Previously clamped to sm_90 on Blackwell (GH-480 workaround).
    /// Now returns native target so kernels generate sm_121 SASS directly.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::CudaDriver)` if query fails.
    pub fn sm_target(&self) -> Result<String, GpuError> {
        let (major, minor) = self.compute_capability()?;
        Ok(format!("sm_{major}{minor}"))
    }

    /// Get the PTX ISA version required for this device's compute capability.
    ///
    /// Returns `(major, minor)` of the PTX version:
    /// - sm_70..sm_90: PTX 8.0 (baseline)
    /// - sm_100+:      PTX 8.8 (Blackwell support, requires CUDA 13.0+)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::CudaDriver)` if compute capability query fails.
    pub fn ptx_version(&self) -> Result<(u32, u32), GpuError> {
        let (major, _minor) = self.compute_capability()?;
        if major >= 10 {
            // Blackwell (sm_100+) requires PTX 8.8 (CUDA 13.0 toolkit)
            Ok((8, 8))
        } else {
            // Volta through Hopper: PTX 8.0
            Ok((8, 0))
        }
    }

    /// Get number of streaming multiprocessors (SMs) on the device
    ///
    /// Used for scaling kernel grid dimensions to GPU size.
    /// E.g., Jetson Orin Nano = 8, RTX 4090 = 128.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::CudaDriver)` if query fails.
    pub fn multiprocessor_count(&self) -> Result<i32, GpuError> {
        let driver = get_driver()?;

        // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
        let mut count: i32 = 0;
        // SAFETY: pointer is valid, device is valid
        let result = unsafe { (driver.cuDeviceGetAttribute)(&mut count, 16, self.device) };
        CudaDriver::check(result)?;

        Ok(count)
    }

    /// Get total device memory in bytes
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::CudaDriver)` if query fails.
    pub fn total_memory(&self) -> Result<usize, GpuError> {
        let driver = get_driver()?;

        let mut bytes: usize = 0;
        // SAFETY: pointer is valid, device is valid
        let result = unsafe { (driver.cuDeviceTotalMem)(&mut bytes, self.device) };
        CudaDriver::check(result)?;

        Ok(bytes)
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if let Ok(driver) = get_driver() {
            // SAFETY: device is valid from constructor
            unsafe {
                let _ = (driver.cuDevicePrimaryCtxRelease)(self.device);
            }
        }
    }
}

// ============================================================================
// Device Enumeration
// ============================================================================

/// Get the number of CUDA devices
///
/// # Errors
///
/// Returns `Err(GpuError::CudaNotAvailable)` if CUDA is not available.
pub fn device_count() -> Result<usize, GpuError> {
    let driver = get_driver()?;

    let mut count: i32 = 0;
    // SAFETY: count is a valid pointer
    let result = unsafe { (driver.cuDeviceGetCount)(&mut count) };
    CudaDriver::check(result)?;

    Ok(count as usize)
}

/// Check if CUDA is available
///
/// Returns `true` if CUDA driver is installed and at least one device exists.
#[must_use]
pub fn cuda_available() -> bool {
    device_count().map(|c| c > 0).unwrap_or(false)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_get_driver_without_feature() {
        use super::get_driver;
        let result = get_driver();
        assert!(result.is_err());
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_cuda_available_without_feature() {
        use super::cuda_available;
        assert!(!cuda_available());
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_device_count_without_feature() {
        use super::device_count;
        let result = device_count();
        assert!(result.is_err());
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_context_new_without_feature() {
        use super::CudaContext;
        let result = CudaContext::new(0);
        assert!(result.is_err());
    }
}
