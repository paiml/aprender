//! PTX Module Loading and JIT Compilation with Disk Cache
//!
//! Loads PTX source into GPU-executable modules.
//! Uses OUR OWN FFI from driver/sys.rs - no external dependencies.
//!
//! # Disk Cache (PTX-CACHE)
//!
//! Compiled cubin blobs are cached to `~/.cache/trueno/ptx/{hash}.cubin`
//! to eliminate the ~35-minute JIT warmup on training restarts.
//!
//! Cache key = SHA-256(patched PTX + compute capability + driver version).
//! On cache hit, `cuModuleLoadData` loads the pre-compiled cubin instantly.
//! On cache miss, the CUDA linker API compiles PTX to cubin, saves to disk,
//! then loads the module.
//!
//! # Design Philosophy
//!
//! PTX is JIT-compiled to SASS (device assembly) at load time.
//! This incurs one-time cost but enables runtime architecture targeting.
//! The disk cache amortizes this cost across process restarts.
//!
//! # Citation
//!
//! [5] NVIDIA CUDA C++ Programming Guide v12.3, Section 3.3 "Modules"

use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;

use std::ffi::c_void;
use std::os::raw::c_uint;

use super::context::{get_driver, CudaContext};
use super::ptx_cache::{load_cached_cubin, ptx_cache_dir, ptx_cache_key, save_cached_cubin};
use super::ptx_patch::patch_backward_branches_sm121;
use super::sys::{
    CUfunction, CUmodule, CudaDriver, CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    CU_JIT_INPUT_PTX, CU_JIT_TARGET,
};
use crate::GpuError;

/// CU_JIT_INFO_LOG_BUFFER - Pointer to buffer for info log
const CU_JIT_INFO_LOG_BUFFER: c_uint = 3;
/// CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES - Size of info log buffer
const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_uint = 4;

// ============================================================================
// PTX Disk Cache Helpers (CUDA-dependent)
// ============================================================================

/// Query the CUDA driver version via `cuDriverGetVersion`.
///
/// Returns 0 if the query fails (cache key still valid, just less specific).
fn query_driver_version(driver: &CudaDriver) -> i32 {
    let mut version: i32 = 0;
    // SAFETY: version is a valid pointer to stack-allocated i32
    let result = unsafe { (driver.cuDriverGetVersion)(&mut version) };
    if CudaDriver::check(result).is_ok() {
        version
    } else {
        0
    }
}

/// Compile PTX to cubin via the CUDA linker API and return the cubin bytes.
///
/// Uses `cuLinkCreate` -> `cuLinkAddData(PTX)` -> `cuLinkComplete` -> copy cubin.
/// The cubin output pointer is owned by the link state and freed on `cuLinkDestroy`,
/// so we copy the bytes before destroying.
fn compile_ptx_to_cubin(
    driver: &CudaDriver,
    ptx: &str,
    jit_target: c_uint,
) -> Result<Vec<u8>, GpuError> {
    use super::sys::CUlinkState;

    let mut link_state: CUlinkState = ptr::null_mut();

    // PMAT-290: Pass CU_JIT_TARGET to the linker for proper optimization.
    // trueno#188: Now that PTX uses the correct ISA version (8.8 for sm_121),
    // the linker can use the real JIT target for all architectures.
    let mut opt_keys: [c_uint; 1] = [CU_JIT_TARGET];
    // CU_JIT_TARGET is a value-type option: the integer value is cast directly
    // to *mut c_void (NOT passed as a pointer). This matches cuModuleLoadDataEx
    // convention and the CUDA Driver API specification.
    // Bug fix: previously passed &mut jit_target_val (a pointer TO the value),
    // which caused CUDA_ERROR_INVALID_VALUE on sm_89 (trueno#231).
    let mut opt_vals: [*mut c_void; 1] = [jit_target as *mut c_void];

    // SAFETY: option arrays are valid for the specified count
    let result = unsafe {
        (driver.cuLinkCreate)(1, opt_keys.as_mut_ptr(), opt_vals.as_mut_ptr(), &mut link_state)
    };
    CudaDriver::check(result)
        .map_err(|e| GpuError::ModuleLoad(format!("cuLinkCreate failed: {e}")))?;

    // Add PTX data to the linker
    // cuLinkAddData with CU_JIT_INPUT_PTX requires null-terminated text.
    // Rust &str is NOT null-terminated, so we must use CString.
    // Bug fix: previously passed raw bytes without null terminator,
    // causing CUDA_ERROR_INVALID_PTX (218) on sm_89 (trueno#231).
    let ptx_cstring = CString::new(ptx.as_bytes().to_vec())
        .map_err(|_| GpuError::ModuleLoad("PTX contains null bytes".to_string()))?;
    let ptx_name = CString::new("kernel.ptx").expect("static string has no null bytes");

    // SAFETY: link_state is valid, ptx_cstring is valid null-terminated string.
    // cuLinkAddData takes a mutable pointer but does not modify the data.
    // Size includes the null terminator per CUDA driver API convention.
    let result = unsafe {
        (driver.cuLinkAddData)(
            link_state,
            CU_JIT_INPUT_PTX,
            ptx_cstring.as_ptr() as *mut c_void,
            ptx_cstring.as_bytes_with_nul().len(),
            ptx_name.as_ptr(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    if CudaDriver::check(result).is_err() {
        // Clean up on failure
        unsafe { (driver.cuLinkDestroy)(link_state) };
        return Err(GpuError::ModuleLoad(format!("cuLinkAddData failed: result={result}")));
    }

    // Complete the link and extract cubin
    let mut cubin_ptr: *mut c_void = ptr::null_mut();
    let mut cubin_size: usize = 0;

    // SAFETY: link_state is valid, output pointers are valid
    let result = unsafe { (driver.cuLinkComplete)(link_state, &mut cubin_ptr, &mut cubin_size) };
    if CudaDriver::check(result).is_err() {
        unsafe { (driver.cuLinkDestroy)(link_state) };
        return Err(GpuError::ModuleLoad(format!("cuLinkComplete failed: result={result}")));
    }

    // Copy the cubin bytes -- the pointer is owned by link_state and freed on destroy
    let cubin = if !cubin_ptr.is_null() && cubin_size > 0 {
        // SAFETY: cubin_ptr points to cubin_size bytes of valid memory owned by link_state
        let slice = unsafe { std::slice::from_raw_parts(cubin_ptr as *const u8, cubin_size) };
        slice.to_vec()
    } else {
        unsafe { (driver.cuLinkDestroy)(link_state) };
        return Err(GpuError::ModuleLoad("cuLinkComplete returned null cubin".to_string()));
    };

    // Destroy the link state (frees cubin_ptr)
    unsafe { (driver.cuLinkDestroy)(link_state) };

    Ok(cubin)
}

// ============================================================================
// CUDA Module
// ============================================================================

/// Compiled CUDA module containing kernels
///
/// Loads PTX source and JIT compiles to device-specific SASS.
/// Caches function handles for efficient lookup.
///
/// # Disk Cache
///
/// Compiled cubin blobs are cached to `~/.cache/trueno/ptx/{sha256}.cubin`.
/// On subsequent loads with identical PTX + device + driver, the cached cubin
/// is loaded directly via `cuModuleLoadData`, skipping the ~35s JIT per kernel.
///
/// # RAII
///
/// Module is automatically unloaded when dropped.
pub struct CudaModule {
    /// Module handle
    module: CUmodule,
    /// Cached function handles
    functions: HashMap<String, CUfunction>,
}

// SAFETY: CUmodule handles are thread-safe for read-only operations
unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl CudaModule {
    /// Load PTX using ONLY cuModuleLoadData (no cuModuleLoadDataEx).
    ///
    /// trueno#200: On Blackwell (sm_121), cuModuleLoadDataEx with CU_JIT_TARGET=90
    /// poisons the CUDA context. This function bypasses that entirely.
    ///
    /// Applies GH-480 backward branch patch before compilation.
    pub fn from_ptx_direct(ctx: &CudaContext, ptx: &str) -> Result<Self, GpuError> {
        let driver = get_driver()?;
        ctx.make_current()?;

        let (major, _) = ctx.compute_capability()?;
        let ptx_patched = if major >= 12 { patch_backward_branches_sm121(ptx) } else { None };
        let ptx = ptx_patched.as_deref().unwrap_or(ptx);

        let ptx_cstring = CString::new(ptx.as_bytes().to_vec())
            .map_err(|_| GpuError::ModuleLoad("PTX contains null bytes".to_string()))?;

        let mut module: CUmodule = ptr::null_mut();
        let result =
            unsafe { (driver.cuModuleLoadData)(&mut module, ptx_cstring.as_ptr() as *const _) };

        CudaDriver::check(result).map_err(|e| {
            let kernel_name = ptx.lines().find(|l| l.contains(".entry")).unwrap_or("unknown");
            // GH-561: Dump PTX on compilation failure for debugging
            let dump_path = format!("/tmp/ptx-fail-{}.ptx", std::process::id());
            let _ = std::fs::write(&dump_path, ptx);
            eprintln!("[PTX-FAIL] Invalid PTX dumped to {dump_path} ({} bytes)", ptx.len());
            GpuError::ModuleLoad(format!(
                "cuModuleLoadData failed: result={result} (kernel: {kernel_name}), error: {e}"
            ))
        })?;

        Ok(Self { module, functions: HashMap::new() })
    }
}

impl CudaModule {
    /// Load PTX source and JIT compile to device code
    ///
    /// Uses `cuModuleLoadDataEx` with explicit JIT target architecture
    /// derived from the device's compute capability. This ensures the JIT
    /// compiler knows exactly which SASS to generate.
    ///
    /// # Disk Cache
    ///
    /// Before JIT compilation, checks `~/.cache/trueno/ptx/` for a cached cubin
    /// matching the SHA-256 of (patched PTX + compute capability + driver version).
    /// On cache hit, loads the cubin directly (instant). On miss, compiles via the
    /// CUDA linker API, saves the cubin to disk, then loads the module.
    ///
    /// # Contract: F-PTX-002 (Context Currency)
    ///
    /// Ensures the CUDA context is current on the calling thread before
    /// JIT compilation. CUDA contexts are thread-local.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context (will be made current)
    /// * `ptx` - PTX assembly source code
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::ModuleLoad)` if PTX is invalid or compilation fails.
    pub fn from_ptx(ctx: &CudaContext, ptx: &str) -> Result<Self, GpuError> {
        // GH-611: Removed debug print — was spamming stderr with 50+ lines per inference
        let driver = get_driver()?;

        // F-PTX-002: Ensure context is current on this thread before JIT compilation.
        ctx.make_current()?;

        // Detect device compute capability for JIT target.
        // trueno#188: Now that PTX uses the real target (sm_121 with PTX 8.8),
        // the JIT target should also be the real compute capability.
        // The previous GH-480 clamp to sm_90 is no longer needed because
        // PTX ISA 8.8 supports sm_121 natively.
        let (major, minor) = ctx.compute_capability()?;
        let jit_target: c_uint = (major * 10 + minor) as c_uint;

        // GH-480: Patch unconditional backward branches for Blackwell (sm_121+).
        // The CUDA 13.0 JIT miscompiles while-loops (unconditional backward
        // branches) on sm_121, silently dropping loop iterations. Converting
        // them to conditional branches avoids the buggy JIT optimizer path.
        //
        // NOTE: Patching happens BEFORE cache key computation so the cached
        // cubin matches the patched PTX that was actually compiled.
        let ptx_patched = if major >= 12 { patch_backward_branches_sm121(ptx) } else { None };
        let ptx: &str = ptx_patched.as_deref().unwrap_or(ptx);

        // Query driver version for cache key
        let driver_version = query_driver_version(driver);

        // Compute cache key: SHA-256(patched PTX + jit_target + driver_version)
        let cache_key = ptx_cache_key(ptx, jit_target, driver_version);

        // ----------------------------------------------------------------
        // Cache hit path: load pre-compiled cubin from disk
        // ----------------------------------------------------------------
        if let Some(cubin) = load_cached_cubin(&cache_key) {
            let mut module: CUmodule = ptr::null_mut();
            // SAFETY: cubin is valid binary data, context is current
            let result =
                unsafe { (driver.cuModuleLoadData)(&mut module, cubin.as_ptr() as *const c_void) };
            if CudaDriver::check(result).is_ok() {
                return Ok(Self { module, functions: HashMap::new() });
            }
            // Cache corruption or stale cubin -- fall through to JIT compilation.
            // Remove the corrupt cache entry so we don't keep failing.
            if let Some(dir) = ptx_cache_dir() {
                let _ = std::fs::remove_file(dir.join(format!("{cache_key}.cubin")));
            }
            eprintln!(
                "[PTX-CACHE] Cache hit but cuModuleLoadData failed (result={result}), \
                 falling through to JIT compilation"
            );
        }

        // ----------------------------------------------------------------
        // Cache miss path: compile PTX -> cubin via linker API, save, load
        // ----------------------------------------------------------------

        // trueno#188: With PTX 8.8 + sm_121 native target, the normal linker/cache
        // path works for Blackwell. No more special-case skip needed.
        // The backward branch patch (above) still applies for the JIT bug.

        // Try to compile PTX to cubin via linker API for caching
        let cubin_result = compile_ptx_to_cubin(driver, ptx, jit_target);

        if let Err(ref e) = cubin_result {
            eprintln!("[PTX-CACHE] Linker compilation failed: {e}, falling through to legacy JIT");
        }
        if let Ok(cubin) = &cubin_result {
            // Save cubin to disk cache (best-effort, failures are silent)
            save_cached_cubin(&cache_key, cubin);

            // Load the cubin as a module
            let mut module: CUmodule = ptr::null_mut();
            // SAFETY: cubin is valid compiled binary, context is current
            let result =
                unsafe { (driver.cuModuleLoadData)(&mut module, cubin.as_ptr() as *const c_void) };
            if CudaDriver::check(result).is_ok() {
                return Ok(Self { module, functions: HashMap::new() });
            }
            // cubin compiled but wouldn't load -- fall through to legacy path
            eprintln!(
                "[PTX-CACHE] Linker produced cubin but cuModuleLoadData failed (result={result}), \
                 falling through to legacy JIT"
            );
        }

        // ----------------------------------------------------------------
        // Legacy fallback: cuModuleLoadDataEx (original behavior, no caching)
        // ----------------------------------------------------------------
        Self::from_ptx_legacy(driver, ptx, jit_target, major, minor)
    }

    /// Legacy PTX loading path without disk caching.
    ///
    /// This is the original `from_ptx` implementation, used as a fallback
    /// when the linker-based caching path fails.
    fn from_ptx_legacy(
        driver: &CudaDriver,
        ptx: &str,
        jit_target: c_uint,
        major: i32,
        minor: i32,
    ) -> Result<Self, GpuError> {
        // Ensure PTX is null-terminated
        let ptx_cstring = CString::new(ptx.as_bytes().to_vec())
            .map_err(|_| GpuError::ModuleLoad("PTX contains null bytes".to_string()))?;

        // trueno#188: With PTX 8.8 + native sm_121, cuModuleLoadDataEx with
        // CU_JIT_TARGET=121 should work correctly on Blackwell. The previous
        // GH-480 workaround (skip cuModuleLoadDataEx) is no longer needed.

        // Try 1: cuModuleLoadDataEx with explicit JIT target + log buffers
        let mut info_log = vec![0u8; 4096];
        let mut error_log = vec![0u8; 4096];
        let info_log_size: usize = info_log.len();
        let error_log_size: usize = error_log.len();

        let mut options: [c_uint; 5] = [
            CU_JIT_TARGET,
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        ];
        let mut option_values: [*mut c_void; 5] = [
            jit_target as *mut c_void,
            info_log.as_mut_ptr() as *mut c_void,
            info_log_size as *mut c_void,
            error_log.as_mut_ptr() as *mut c_void,
            error_log_size as *mut c_void,
        ];

        // SAFETY: ptx_cstring is valid null-terminated string, options arrays
        // are valid for the lifetime of this call, context is current.
        let mut module: CUmodule = ptr::null_mut();
        let result = unsafe {
            (driver.cuModuleLoadDataEx)(
                &mut module,
                ptx_cstring.as_ptr() as *const _,
                5,
                options.as_mut_ptr(),
                option_values.as_mut_ptr(),
            )
        };

        if CudaDriver::check(result).is_ok() {
            return Ok(Self { module, functions: HashMap::new() });
        }

        // Try 1 failed -- capture diagnostics
        let kernel_name =
            ptx.lines().find(|l| l.contains(".entry")).map(|l| l.trim()).unwrap_or("<unknown>");
        let jit_info = String::from_utf8_lossy(&info_log).trim_end_matches('\0').to_string();
        let jit_err = String::from_utf8_lossy(&error_log).trim_end_matches('\0').to_string();

        eprintln!(
            "[PTX-JIT] Try 1 failed: {kernel_name}, target: sm_{major}{minor}, \
             PTX: {} bytes, result: {result}",
            ptx.len()
        );
        if !jit_info.is_empty() {
            eprintln!("[PTX-JIT] Info log: {jit_info}");
        }
        if !jit_err.is_empty() {
            eprintln!("[PTX-JIT] Error log: {jit_err}");
        }

        // Dump PTX to /tmp for offline diagnosis (#127)
        let dump_path = format!(
            "/tmp/failed-ptx-sm_{major}{minor}-{}.ptx",
            kernel_name.replace(|c: char| !c.is_alphanumeric() && c != '_', "_")
        );
        if let Ok(()) = std::fs::write(&dump_path, ptx) {
            eprintln!("[PTX-JIT] PTX dumped to {dump_path}");
        }

        // Try 2: cuModuleLoadData without explicit JIT target (let driver auto-detect)
        eprintln!("[PTX-JIT] Retrying with cuModuleLoadData (no explicit target)...");
        let mut module2: CUmodule = ptr::null_mut();
        let result2 =
            unsafe { (driver.cuModuleLoadData)(&mut module2, ptx_cstring.as_ptr() as *const _) };

        if CudaDriver::check(result2).is_ok() {
            eprintln!("[PTX-JIT] Fallback succeeded for {kernel_name}");
            return Ok(Self { module: module2, functions: HashMap::new() });
        }

        // Both attempts failed
        eprintln!("[PTX-JIT] Both attempts failed for {kernel_name}");
        Err(GpuError::ModuleLoad(format!(
            "CUDA module loading failed: try1={result} try2={result2} (JIT target: sm_{major}{minor})"
        )))
    }

    /// Get kernel function handle by name
    ///
    /// Function handles are cached for efficient repeated lookup.
    ///
    /// # Arguments
    ///
    /// * `name` - Kernel function name (must match PTX .entry name)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::FunctionNotFound)` if function doesn't exist.
    pub fn get_function(&mut self, name: &str) -> Result<CUfunction, GpuError> {
        // Check cache first
        if let Some(&func) = self.functions.get(name) {
            return Ok(func);
        }

        let driver = get_driver()?;
        let name_cstring =
            CString::new(name).map_err(|_| GpuError::FunctionNotFound(name.to_string()))?;

        // SAFETY: module is valid, name_cstring is null-terminated
        let mut func: CUfunction = ptr::null_mut();
        let result =
            unsafe { (driver.cuModuleGetFunction)(&mut func, self.module, name_cstring.as_ptr()) };
        CudaDriver::check(result).map_err(|_| GpuError::FunctionNotFound(name.to_string()))?;

        // Cache for future lookups
        self.functions.insert(name.to_string(), func);
        Ok(func)
    }

    /// Get raw module handle
    ///
    /// # Safety
    ///
    /// The returned handle is only valid while this `CudaModule` is alive.
    #[must_use]
    pub fn raw(&self) -> CUmodule {
        self.module
    }

    /// Check if a function exists in the module
    pub fn has_function(&mut self, name: &str) -> bool {
        self.get_function(name).is_ok()
    }

    /// Get list of cached function names
    #[must_use]
    pub fn cached_functions(&self) -> Vec<&str> {
        self.functions.keys().map(String::as_str).collect()
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        if let Ok(driver) = get_driver() {
            // SAFETY: module is valid from constructor
            unsafe {
                let _ = (driver.cuModuleUnload)(self.module);
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_module_requires_cuda_feature() {
        // Without cuda feature, we can't create modules
        // This test just verifies the module compiles
        assert!(true);
    }
}
