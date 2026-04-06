//! Minimal CUDA Driver API FFI Bindings
//!
//! Hand-written FFI for the ~20 CUDA driver functions we actually need.
//! No external dependencies. Dynamic loading via libcuda.so/nvcuda.dll.
//!
//! # Design Philosophy
//!
//! **OWN THE STACK**: We built 5,500 lines of PTX generation from scratch.
//! We can build 400 lines of CUDA FFI. Total control, no third-party surprises.
//!
//! # Safety
//!
//! All functions in this module are unsafe. Safe wrappers are provided in
//! sibling modules (context.rs, module.rs, stream.rs, memory.rs).
//!
//! # Clippy Allows
//!
//! This module uses FFI-specific patterns that trigger clippy lints:
//! - `borrow_as_ptr`: FFI requires `&mut T` -> `*mut T` conversion
//! - `ptr_as_ptr`: FFI pointer casts are intentional
//! - `cast_sign_loss`: CUDA uses i32 for counts, we use usize
//!
//! # Citation
//!
//! [1] RustBelt (Jung et al., POPL 2017) proves Rust's type system safely
//!     encapsulates unsafe FFI operations via ownership and borrowing.

use std::ffi::c_void;
use std::os::raw::{c_char, c_int, c_uint};

use crate::GpuError;

// ============================================================================
// CUDA Type Definitions (from cuda.h)
// ============================================================================

/// CUDA error code
pub type CUresult = c_int;

/// CUDA device handle
pub type CUdevice = c_int;

/// CUDA context handle (opaque pointer)
pub type CUcontext = *mut c_void;

/// CUDA module handle (opaque pointer)
pub type CUmodule = *mut c_void;

/// CUDA function handle (opaque pointer)
pub type CUfunction = *mut c_void;

/// CUDA stream handle (opaque pointer)
pub type CUstream = *mut c_void;

/// CUDA device pointer (GPU memory address)
pub type CUdeviceptr = u64;

/// CUDA graph handle (opaque pointer)
pub type CUgraph = *mut c_void;

/// CUDA graph executable handle (opaque pointer)
pub type CUgraphExec = *mut c_void;

/// CUDA event handle (opaque pointer)
pub type CUevent = *mut c_void;

/// CUDA graph node handle (opaque pointer, for cuGraphExecUpdate error reporting)
pub type CUgraphNode = *mut c_void;

/// CUDA graph exec update result (PMAT-291)
pub type CUgraphExecUpdateResult = c_uint;
/// Update succeeded
pub const CU_GRAPH_EXEC_UPDATE_SUCCESS: CUgraphExecUpdateResult = 0;
/// Update failed (topology changed, need re-instantiate)
pub const CU_GRAPH_EXEC_UPDATE_ERROR: CUgraphExecUpdateResult = 1;

/// CUDA linker state handle (opaque pointer) — for PTX→cubin compilation
pub type CUlinkState = *mut c_void;

/// trueno#243: CUDA_KERNEL_NODE_PARAMS for manual graph construction.
/// Matches CUDA driver API struct layout.
#[repr(C)]
pub struct CudaKernelNodeParams {
    /// Kernel function handle
    pub func: CUfunction,
    /// Grid dimensions (x, y, z)
    pub grid_dim_x: c_uint,
    pub grid_dim_y: c_uint,
    pub grid_dim_z: c_uint,
    /// Block dimensions (x, y, z)
    pub block_dim_x: c_uint,
    pub block_dim_y: c_uint,
    pub block_dim_z: c_uint,
    /// Dynamic shared memory size in bytes
    pub shared_mem_bytes: c_uint,
    /// Kernel parameters (array of pointers to argument values)
    pub kernel_params: *mut *mut c_void,
    /// Extra options (must be null for kernel_params path)
    pub extra: *mut *mut c_void,
}

// ============================================================================
// CUDA Error Codes (subset we handle)
// ============================================================================

/// CUDA success
pub const CUDA_SUCCESS: CUresult = 0;
/// Invalid value passed
pub const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
/// Out of memory
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
/// CUDA not initialized
pub const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
/// CUDA deinitialized
pub const CUDA_ERROR_DEINITIALIZED: CUresult = 4;
/// No CUDA device
pub const CUDA_ERROR_NO_DEVICE: CUresult = 100;
/// Invalid device
pub const CUDA_ERROR_INVALID_DEVICE: CUresult = 101;
/// Invalid device kernel image
pub const CUDA_ERROR_INVALID_IMAGE: CUresult = 200;
/// Invalid CUDA context (not current on this thread)
pub const CUDA_ERROR_INVALID_CONTEXT: CUresult = 201;
/// No binary for GPU (PTX JIT failed for this architecture)
pub const CUDA_ERROR_NO_BINARY_FOR_GPU: CUresult = 209;
/// Invalid PTX
pub const CUDA_ERROR_INVALID_PTX: CUresult = 218;
/// Function not found
pub const CUDA_ERROR_NOT_FOUND: CUresult = 500;
/// Operation not ready
pub const CUDA_ERROR_NOT_READY: CUresult = 600;
/// Illegal memory address accessed by a kernel (sticky — context unrecoverable)
pub const CUDA_ERROR_ILLEGAL_ADDRESS: CUresult = 700;
/// Illegal instruction encountered
pub const CUDA_ERROR_ILLEGAL_INSTRUCTION: CUresult = 715;
/// Unspecified launch failure (sticky — context unrecoverable)
pub const CUDA_ERROR_LAUNCH_FAILED: CUresult = 719;

// ============================================================================
// CUDA JIT Option Codes (subset we use)
// ============================================================================

/// CU_JIT_TARGET - Specifies target architecture for JIT compilation
pub const CU_JIT_TARGET: c_uint = 9;
/// CU_JIT_ERROR_LOG_BUFFER - Pointer to buffer for error log
pub const CU_JIT_ERROR_LOG_BUFFER: c_uint = 5;
/// CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES - Size of error log buffer
pub const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_uint = 6;

// ============================================================================
// CUDA JIT Input Types (for cuLinkAddData)
// ============================================================================

/// CU_JIT_INPUT_PTX - PTX source input for the linker
pub const CU_JIT_INPUT_PTX: c_uint = 1;

// ============================================================================
// CUDA Compute Capability Target Values (for CU_JIT_TARGET)
// ============================================================================

/// SM 7.0 (Volta)
pub const CU_TARGET_COMPUTE_70: c_uint = 70;
/// SM 7.5 (Turing)
pub const CU_TARGET_COMPUTE_75: c_uint = 75;
/// SM 8.0 (Ampere)
pub const CU_TARGET_COMPUTE_80: c_uint = 80;
/// SM 8.6 (Ampere consumer)
pub const CU_TARGET_COMPUTE_86: c_uint = 86;
/// SM 8.7 (Jetson Orin)
pub const CU_TARGET_COMPUTE_87: c_uint = 87;
/// SM 8.9 (Ada Lovelace)
pub const CU_TARGET_COMPUTE_89: c_uint = 89;
/// SM 9.0 (Hopper)
pub const CU_TARGET_COMPUTE_90: c_uint = 90;

// ============================================================================
// CUDA Stream Flags
// ============================================================================

/// Default stream creation flag
pub const CU_STREAM_DEFAULT: c_uint = 0;
/// Non-blocking stream (doesn't synchronize with stream 0)
pub const CU_STREAM_NON_BLOCKING: c_uint = 1;

// ============================================================================
// CUDA Event Flags (PMAT-283)
// ============================================================================

/// Default event creation flag
pub const CU_EVENT_DEFAULT: c_uint = 0;
/// Disable timing (faster event operations when timing is not needed)
pub const CU_EVENT_DISABLE_TIMING: c_uint = 0x02;

// ============================================================================
// CUDA Driver Function Pointers
// ============================================================================

/// Dynamically loaded CUDA driver functions
///
/// All function pointers are loaded at runtime from libcuda.so (Linux)
/// or nvcuda.dll (Windows). This avoids link-time dependency on CUDA.
#[allow(non_snake_case)]
pub struct CudaDriver {
    // Initialization
    /// cuInit - Initialize the CUDA driver
    pub cuInit: unsafe extern "C" fn(flags: c_uint) -> CUresult,

    // Device Management
    /// cuDeviceGetCount - Get number of CUDA devices
    pub cuDeviceGetCount: unsafe extern "C" fn(count: *mut c_int) -> CUresult,
    /// cuDeviceGet - Get device handle by ordinal
    pub cuDeviceGet: unsafe extern "C" fn(device: *mut CUdevice, ordinal: c_int) -> CUresult,
    /// cuDeviceGetName - Get device name
    pub cuDeviceGetName:
        unsafe extern "C" fn(name: *mut c_char, len: c_int, device: CUdevice) -> CUresult,
    /// cuDeviceTotalMem - Get total device memory
    pub cuDeviceTotalMem: unsafe extern "C" fn(bytes: *mut usize, device: CUdevice) -> CUresult,
    /// cuDeviceGetAttribute - Get device attribute
    pub cuDeviceGetAttribute:
        unsafe extern "C" fn(pi: *mut c_int, attrib: c_int, device: CUdevice) -> CUresult,

    // Context Management (Primary Context API - preferred)
    /// cuDevicePrimaryCtxRetain - Retain primary context
    pub cuDevicePrimaryCtxRetain:
        unsafe extern "C" fn(ctx: *mut CUcontext, device: CUdevice) -> CUresult,
    /// cuDevicePrimaryCtxRelease - Release primary context
    pub cuDevicePrimaryCtxRelease: unsafe extern "C" fn(device: CUdevice) -> CUresult,
    /// cuCtxSetCurrent - Set current context
    pub cuCtxSetCurrent: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    /// cuCtxSynchronize - Synchronize current context
    pub cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,

    // Module Management
    /// cuModuleLoadData - Load module from PTX/cubin data
    pub cuModuleLoadData:
        unsafe extern "C" fn(module: *mut CUmodule, image: *const c_void) -> CUresult,
    /// cuModuleLoadDataEx - Load module with JIT options (target arch, error log)
    #[allow(clippy::type_complexity)]
    pub cuModuleLoadDataEx: unsafe extern "C" fn(
        module: *mut CUmodule,
        image: *const c_void,
        num_options: c_uint,
        options: *mut c_uint,
        option_values: *mut *mut c_void,
    ) -> CUresult,
    /// cuModuleUnload - Unload module
    pub cuModuleUnload: unsafe extern "C" fn(module: CUmodule) -> CUresult,
    /// cuModuleGetFunction - Get function from module
    pub cuModuleGetFunction: unsafe extern "C" fn(
        func: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult,

    // Memory Management
    /// cuMemAlloc - Allocate device memory
    pub cuMemAlloc: unsafe extern "C" fn(ptr: *mut CUdeviceptr, size: usize) -> CUresult,
    /// PMAT-394: cuMemAllocManaged - Allocate unified memory (Grace Blackwell zero-copy)
    pub cuMemAllocManaged:
        unsafe extern "C" fn(ptr: *mut CUdeviceptr, size: usize, flags: u32) -> CUresult,
    /// PMAT-396: cuMemHostRegister - Register host memory for GPU access (zero-copy on unified)
    pub cuMemHostRegister:
        unsafe extern "C" fn(ptr: *mut c_void, size: usize, flags: u32) -> CUresult,
    /// PMAT-396: cuMemHostUnregister - Unregister previously registered host memory
    pub cuMemHostUnregister: unsafe extern "C" fn(ptr: *mut c_void) -> CUresult,
    /// PMAT-396: cuMemHostGetDevicePointer - Get device pointer for registered host memory
    pub cuMemHostGetDevicePointer:
        unsafe extern "C" fn(pdptr: *mut CUdeviceptr, p: *mut c_void, flags: u32) -> CUresult,
    /// cuMemFree - Free device memory
    pub cuMemFree: unsafe extern "C" fn(ptr: CUdeviceptr) -> CUresult,
    /// cuMemcpyHtoD - Copy from host to device
    pub cuMemcpyHtoD:
        unsafe extern "C" fn(dst: CUdeviceptr, src: *const c_void, size: usize) -> CUresult,
    /// cuMemcpyDtoH - Copy from device to host
    pub cuMemcpyDtoH:
        unsafe extern "C" fn(dst: *mut c_void, src: CUdeviceptr, size: usize) -> CUresult,
    /// cuMemcpyHtoDAsync - Async copy from host to device
    pub cuMemcpyHtoDAsync: unsafe extern "C" fn(
        dst: CUdeviceptr,
        src: *const c_void,
        size: usize,
        stream: CUstream,
    ) -> CUresult,
    /// cuMemcpyDtoHAsync - Async copy from device to host
    pub cuMemcpyDtoHAsync: unsafe extern "C" fn(
        dst: *mut c_void,
        src: CUdeviceptr,
        size: usize,
        stream: CUstream,
    ) -> CUresult,
    /// cuMemcpyDtoD - Sync copy from device to device (PAR-023)
    pub cuMemcpyDtoD:
        unsafe extern "C" fn(dst: CUdeviceptr, src: CUdeviceptr, size: usize) -> CUresult,
    /// cuMemcpyDtoDAsync - Async copy from device to device (PAR-023)
    pub cuMemcpyDtoDAsync: unsafe extern "C" fn(
        dst: CUdeviceptr,
        src: CUdeviceptr,
        size: usize,
        stream: CUstream,
    ) -> CUresult,
    /// cuMemsetD32Async - GPU-side memset (entrenar#318 Tier 1)
    pub cuMemsetD32Async: unsafe extern "C" fn(
        dst: CUdeviceptr,
        value: u32,
        count: usize,
        stream: CUstream,
    ) -> CUresult,
    /// cuMemGetInfo - Get free and total memory
    pub cuMemGetInfo: unsafe extern "C" fn(free: *mut usize, total: *mut usize) -> CUresult,

    // Stream Management
    /// cuStreamCreate - Create a stream
    pub cuStreamCreate: unsafe extern "C" fn(stream: *mut CUstream, flags: c_uint) -> CUresult,
    /// cuStreamDestroy - Destroy a stream
    pub cuStreamDestroy: unsafe extern "C" fn(stream: CUstream) -> CUresult,
    /// cuStreamSynchronize - Synchronize a stream
    pub cuStreamSynchronize: unsafe extern "C" fn(stream: CUstream) -> CUresult,

    // Kernel Launch
    /// cuLaunchKernel - Launch a kernel
    #[allow(clippy::type_complexity)]
    pub cuLaunchKernel: unsafe extern "C" fn(
        func: CUfunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult,

    // Graph Management (PAR-037)
    /// cuGraphCreate - Create an empty graph
    pub cuGraphCreate: unsafe extern "C" fn(graph: *mut CUgraph, flags: c_uint) -> CUresult,
    /// cuGraphDestroy - Destroy a graph
    pub cuGraphDestroy: unsafe extern "C" fn(graph: CUgraph) -> CUresult,
    /// cuGraphInstantiateWithFlags - Create executable from graph
    pub cuGraphInstantiateWithFlags:
        unsafe extern "C" fn(exec: *mut CUgraphExec, graph: CUgraph, flags: u64) -> CUresult,
    /// cuGraphExecDestroy - Destroy graph executable
    pub cuGraphExecDestroy: unsafe extern "C" fn(exec: CUgraphExec) -> CUresult,
    /// cuGraphLaunch - Launch graph on stream
    pub cuGraphLaunch: unsafe extern "C" fn(exec: CUgraphExec, stream: CUstream) -> CUresult,
    /// trueno#243: cuGraphAddKernelNode — add kernel to graph (manual construction)
    pub cuGraphAddKernelNode: unsafe extern "C" fn(
        node: *mut CUgraphNode,
        graph: CUgraph,
        deps: *const CUgraphNode,
        num_deps: usize,
        params: *const CudaKernelNodeParams,
    ) -> CUresult,
    /// cuGraphExecUpdate - Update graph executable in-place (PMAT-291)
    /// Avoids recapture when only kernel arguments change (same topology).
    /// CUDA 11 API: returns error node + result enum.
    pub cuGraphExecUpdate: unsafe extern "C" fn(
        exec: CUgraphExec,
        graph: CUgraph,
        error_node: *mut CUgraphNode,
        update_result: *mut CUgraphExecUpdateResult,
    ) -> CUresult,
    /// cuStreamBeginCapture - Begin stream capture
    pub cuStreamBeginCapture: unsafe extern "C" fn(stream: CUstream, mode: c_uint) -> CUresult,
    /// cuStreamEndCapture - End stream capture and return graph
    pub cuStreamEndCapture: unsafe extern "C" fn(stream: CUstream, graph: *mut CUgraph) -> CUresult,

    // Event Management (PMAT-283: CPU-GPU pipelining)
    /// cuEventCreate - Create an event
    pub cuEventCreate: unsafe extern "C" fn(event: *mut CUevent, flags: c_uint) -> CUresult,
    /// cuEventDestroy - Destroy an event
    pub cuEventDestroy: unsafe extern "C" fn(event: CUevent) -> CUresult,
    /// cuEventRecord - Record an event on a stream (non-blocking)
    pub cuEventRecord: unsafe extern "C" fn(event: CUevent, stream: CUstream) -> CUresult,
    /// cuEventQuery - Query event completion (non-blocking: returns NOT_READY or SUCCESS)
    pub cuEventQuery: unsafe extern "C" fn(event: CUevent) -> CUresult,
    /// cuEventSynchronize - Wait for event completion (blocking)
    pub cuEventSynchronize: unsafe extern "C" fn(event: CUevent) -> CUresult,
    /// cuStreamWaitEvent - Make stream wait for event (non-blocking cross-stream dependency)
    /// GH-559-PERF: Replaces cuStreamSynchronize for cross-stream ordering.
    pub cuStreamWaitEvent:
        unsafe extern "C" fn(stream: CUstream, event: CUevent, flags: c_uint) -> CUresult,

    // Linker API (PTX disk cache: PTX→cubin compilation + extraction)
    /// cuLinkCreate - Create a linker invocation
    #[allow(clippy::type_complexity)]
    pub cuLinkCreate: unsafe extern "C" fn(
        num_options: c_uint,
        options: *mut c_uint,
        option_values: *mut *mut c_void,
        state_out: *mut CUlinkState,
    ) -> CUresult,
    /// cuLinkAddData - Add PTX/cubin data to a linker invocation
    #[allow(clippy::type_complexity)]
    pub cuLinkAddData: unsafe extern "C" fn(
        state: CUlinkState,
        input_type: c_uint,
        data: *mut c_void,
        size: usize,
        name: *const c_char,
        num_options: c_uint,
        options: *mut c_uint,
        option_values: *mut *mut c_void,
    ) -> CUresult,
    /// cuLinkComplete - Complete a linker invocation and get cubin output
    pub cuLinkComplete: unsafe extern "C" fn(
        state: CUlinkState,
        cubin_out: *mut *mut c_void,
        size_out: *mut usize,
    ) -> CUresult,
    /// cuLinkDestroy - Destroy a linker invocation
    pub cuLinkDestroy: unsafe extern "C" fn(state: CUlinkState) -> CUresult,

    // Driver Version Query (PTX disk cache key component)
    /// cuDriverGetVersion - Get CUDA driver version
    pub cuDriverGetVersion: unsafe extern "C" fn(version: *mut c_int) -> CUresult,
}

// ============================================================================
// Dynamic Loading
// ============================================================================

#[cfg(feature = "cuda")]
mod loading {
    use super::*;
    use libloading::{Library, Symbol};
    use std::sync::OnceLock;

    /// Global driver instance (loaded once)
    static DRIVER: OnceLock<Option<CudaDriver>> = OnceLock::new();

    /// Library handle (must outlive function pointers)
    static LIBRARY: OnceLock<Option<Library>> = OnceLock::new();

    impl CudaDriver {
        /// Load CUDA driver dynamically
        ///
        /// Returns `None` if CUDA is not available (no driver installed).
        /// This is NOT an error - it's expected on systems without NVIDIA GPUs.
        ///
        /// # Safety
        ///
        /// This function loads a shared library and extracts function pointers.
        /// The library must remain loaded for the lifetime of the returned driver.
        #[must_use]
        pub fn load() -> Option<&'static Self> {
            // Initialize library first
            let _ = LIBRARY.get_or_init(|| {
                // Try platform-specific library names
                #[cfg(target_os = "linux")]
                let lib_names = ["libcuda.so.1", "libcuda.so"];
                #[cfg(target_os = "windows")]
                let lib_names = ["nvcuda.dll"];
                #[cfg(target_os = "macos")]
                let lib_names: [&str; 0] = []; // No CUDA on macOS

                for name in lib_names {
                    // SAFETY: We're loading a well-known system library
                    if let Ok(lib) = unsafe { Library::new(name) } {
                        return Some(lib);
                    }
                }
                None
            });

            // Then initialize driver
            DRIVER
                .get_or_init(|| {
                    let lib = LIBRARY.get()?.as_ref()?;
                    Self::load_from_library(lib)
                })
                .as_ref()
        }

        /// Load function pointers from library
        fn load_from_library(lib: &Library) -> Option<Self> {
            // SAFETY: All symbols are standard CUDA driver API functions
            unsafe {
                // Helper macro to load symbols with explicit type
                macro_rules! load_sym {
                    ($name:ident, $ty:ty) => {{
                        let sym: Symbol<'_, $ty> = lib.get(stringify!($name).as_bytes()).ok()?;
                        *sym
                    }};
                }

                type FnInit = unsafe extern "C" fn(c_uint) -> CUresult;
                type FnDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CUresult;
                type FnDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
                type FnDeviceGetName =
                    unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;
                type FnDeviceTotalMem = unsafe extern "C" fn(*mut usize, CUdevice) -> CUresult;
                type FnDeviceGetAttribute =
                    unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult;
                type FnPrimaryCtxRetain =
                    unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult;
                type FnPrimaryCtxRelease = unsafe extern "C" fn(CUdevice) -> CUresult;
                type FnCtxSetCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
                type FnCtxSync = unsafe extern "C" fn() -> CUresult;
                type FnModuleLoadData =
                    unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
                type FnModuleLoadDataEx = unsafe extern "C" fn(
                    *mut CUmodule,
                    *const c_void,
                    c_uint,
                    *mut c_uint,
                    *mut *mut c_void,
                ) -> CUresult;
                type FnModuleUnload = unsafe extern "C" fn(CUmodule) -> CUresult;
                type FnModuleGetFunction =
                    unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
                type FnMemAlloc = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
                type FnMemAllocManaged =
                    unsafe extern "C" fn(*mut CUdeviceptr, usize, u32) -> CUresult;
                type FnMemHostRegister = unsafe extern "C" fn(*mut c_void, usize, u32) -> CUresult;
                type FnMemHostUnregister = unsafe extern "C" fn(*mut c_void) -> CUresult;
                type FnMemHostGetDevicePointer =
                    unsafe extern "C" fn(*mut CUdeviceptr, *mut c_void, u32) -> CUresult;
                type FnMemFree = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
                type FnMemcpyHtoD =
                    unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
                type FnMemcpyDtoH =
                    unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
                type FnMemcpyHtoDAsync =
                    unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult;
                type FnMemcpyDtoHAsync =
                    unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult;
                type FnMemcpyDtoD =
                    unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult;
                type FnMemcpyDtoDAsync =
                    unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult;
                type FnMemsetD32Async =
                    unsafe extern "C" fn(CUdeviceptr, u32, usize, CUstream) -> CUresult;
                type FnMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult;
                type FnStreamCreate = unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult;
                type FnStreamDestroy = unsafe extern "C" fn(CUstream) -> CUresult;
                type FnStreamSync = unsafe extern "C" fn(CUstream) -> CUresult;
                type FnLaunchKernel = unsafe extern "C" fn(
                    CUfunction,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    CUstream,
                    *mut *mut c_void,
                    *mut *mut c_void,
                ) -> CUresult;
                // Graph types (PAR-037)
                type FnGraphCreate = unsafe extern "C" fn(*mut CUgraph, c_uint) -> CUresult;
                type FnGraphDestroy = unsafe extern "C" fn(CUgraph) -> CUresult;
                type FnGraphInstantiate =
                    unsafe extern "C" fn(*mut CUgraphExec, CUgraph, u64) -> CUresult;
                type FnGraphExecDestroy = unsafe extern "C" fn(CUgraphExec) -> CUresult;
                type FnGraphLaunch = unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult;
                type FnGraphExecUpdate = unsafe extern "C" fn(
                    CUgraphExec,
                    CUgraph,
                    *mut CUgraphNode,
                    *mut CUgraphExecUpdateResult,
                ) -> CUresult;
                type FnGraphAddKernelNode = unsafe extern "C" fn(
                    *mut CUgraphNode,
                    CUgraph,
                    *const CUgraphNode,
                    usize,
                    *const CudaKernelNodeParams,
                ) -> CUresult;
                type FnStreamBeginCapture = unsafe extern "C" fn(CUstream, c_uint) -> CUresult;
                type FnStreamEndCapture = unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUresult;
                // Event types (PMAT-283)
                type FnEventCreate = unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult;
                type FnEventDestroy = unsafe extern "C" fn(CUevent) -> CUresult;
                type FnEventRecord = unsafe extern "C" fn(CUevent, CUstream) -> CUresult;
                type FnEventQuery = unsafe extern "C" fn(CUevent) -> CUresult;
                type FnEventSync = unsafe extern "C" fn(CUevent) -> CUresult;
                type FnStreamWaitEvent =
                    unsafe extern "C" fn(CUstream, CUevent, c_uint) -> CUresult;
                // Linker types (PTX disk cache)
                type FnLinkCreate = unsafe extern "C" fn(
                    c_uint,
                    *mut c_uint,
                    *mut *mut c_void,
                    *mut CUlinkState,
                ) -> CUresult;
                type FnLinkAddData = unsafe extern "C" fn(
                    CUlinkState,
                    c_uint,
                    *mut c_void,
                    usize,
                    *const c_char,
                    c_uint,
                    *mut c_uint,
                    *mut *mut c_void,
                ) -> CUresult;
                type FnLinkComplete =
                    unsafe extern "C" fn(CUlinkState, *mut *mut c_void, *mut usize) -> CUresult;
                type FnLinkDestroy = unsafe extern "C" fn(CUlinkState) -> CUresult;
                // Driver version type
                type FnDriverGetVersion = unsafe extern "C" fn(*mut c_int) -> CUresult;

                Some(CudaDriver {
                    cuInit: load_sym!(cuInit, FnInit),
                    cuDeviceGetCount: load_sym!(cuDeviceGetCount, FnDeviceGetCount),
                    cuDeviceGet: load_sym!(cuDeviceGet, FnDeviceGet),
                    cuDeviceGetName: load_sym!(cuDeviceGetName, FnDeviceGetName),
                    cuDeviceTotalMem: load_sym!(cuDeviceTotalMem_v2, FnDeviceTotalMem),
                    cuDeviceGetAttribute: load_sym!(cuDeviceGetAttribute, FnDeviceGetAttribute),
                    cuDevicePrimaryCtxRetain: load_sym!(
                        cuDevicePrimaryCtxRetain,
                        FnPrimaryCtxRetain
                    ),
                    cuDevicePrimaryCtxRelease: load_sym!(
                        cuDevicePrimaryCtxRelease_v2,
                        FnPrimaryCtxRelease
                    ),
                    cuCtxSetCurrent: load_sym!(cuCtxSetCurrent, FnCtxSetCurrent),
                    cuCtxSynchronize: load_sym!(cuCtxSynchronize, FnCtxSync),
                    cuModuleLoadData: load_sym!(cuModuleLoadData, FnModuleLoadData),
                    cuModuleLoadDataEx: load_sym!(cuModuleLoadDataEx, FnModuleLoadDataEx),
                    cuModuleUnload: load_sym!(cuModuleUnload, FnModuleUnload),
                    cuModuleGetFunction: load_sym!(cuModuleGetFunction, FnModuleGetFunction),
                    cuMemAlloc: load_sym!(cuMemAlloc_v2, FnMemAlloc),
                    cuMemAllocManaged: load_sym!(cuMemAllocManaged, FnMemAllocManaged),
                    cuMemHostRegister: load_sym!(cuMemHostRegister_v2, FnMemHostRegister),
                    cuMemHostUnregister: load_sym!(cuMemHostUnregister, FnMemHostUnregister),
                    cuMemHostGetDevicePointer: load_sym!(
                        cuMemHostGetDevicePointer_v2,
                        FnMemHostGetDevicePointer
                    ),
                    cuMemFree: load_sym!(cuMemFree_v2, FnMemFree),
                    cuMemcpyHtoD: load_sym!(cuMemcpyHtoD_v2, FnMemcpyHtoD),
                    cuMemcpyDtoH: load_sym!(cuMemcpyDtoH_v2, FnMemcpyDtoH),
                    cuMemcpyHtoDAsync: load_sym!(cuMemcpyHtoDAsync_v2, FnMemcpyHtoDAsync),
                    cuMemcpyDtoHAsync: load_sym!(cuMemcpyDtoHAsync_v2, FnMemcpyDtoHAsync),
                    cuMemcpyDtoD: load_sym!(cuMemcpyDtoD_v2, FnMemcpyDtoD),
                    cuMemcpyDtoDAsync: load_sym!(cuMemcpyDtoDAsync_v2, FnMemcpyDtoDAsync),
                    cuMemsetD32Async: load_sym!(cuMemsetD32Async, FnMemsetD32Async),
                    cuMemGetInfo: load_sym!(cuMemGetInfo_v2, FnMemGetInfo),
                    cuStreamCreate: load_sym!(cuStreamCreate, FnStreamCreate),
                    cuStreamDestroy: load_sym!(cuStreamDestroy_v2, FnStreamDestroy),
                    cuStreamSynchronize: load_sym!(cuStreamSynchronize, FnStreamSync),
                    cuLaunchKernel: load_sym!(cuLaunchKernel, FnLaunchKernel),
                    // Graph functions (PAR-037)
                    cuGraphCreate: load_sym!(cuGraphCreate, FnGraphCreate),
                    cuGraphDestroy: load_sym!(cuGraphDestroy, FnGraphDestroy),
                    cuGraphInstantiateWithFlags: load_sym!(
                        cuGraphInstantiateWithFlags,
                        FnGraphInstantiate
                    ),
                    cuGraphExecDestroy: load_sym!(cuGraphExecDestroy, FnGraphExecDestroy),
                    cuGraphLaunch: load_sym!(cuGraphLaunch, FnGraphLaunch),
                    cuGraphExecUpdate: load_sym!(cuGraphExecUpdate, FnGraphExecUpdate),
                    cuGraphAddKernelNode: load_sym!(cuGraphAddKernelNode, FnGraphAddKernelNode),
                    cuStreamBeginCapture: load_sym!(cuStreamBeginCapture, FnStreamBeginCapture),
                    cuStreamEndCapture: load_sym!(cuStreamEndCapture, FnStreamEndCapture),
                    // Event functions (PMAT-283)
                    cuEventCreate: load_sym!(cuEventCreate, FnEventCreate),
                    cuEventDestroy: load_sym!(cuEventDestroy_v2, FnEventDestroy),
                    cuEventRecord: load_sym!(cuEventRecord, FnEventRecord),
                    cuEventQuery: load_sym!(cuEventQuery, FnEventQuery),
                    cuEventSynchronize: load_sym!(cuEventSynchronize, FnEventSync),
                    cuStreamWaitEvent: load_sym!(cuStreamWaitEvent, FnStreamWaitEvent),
                    // Linker functions (PTX disk cache)
                    cuLinkCreate: load_sym!(cuLinkCreate, FnLinkCreate),
                    cuLinkAddData: load_sym!(cuLinkAddData_v2, FnLinkAddData),
                    cuLinkComplete: load_sym!(cuLinkComplete, FnLinkComplete),
                    cuLinkDestroy: load_sym!(cuLinkDestroy, FnLinkDestroy),
                    // Driver version
                    cuDriverGetVersion: load_sym!(cuDriverGetVersion, FnDriverGetVersion),
                })
            }
        }

        /// Check CUDA result and convert to GpuError
        ///
        /// # Errors
        ///
        /// Returns `Err(GpuError::CudaDriver)` if result is not CUDA_SUCCESS.
        pub fn check(result: CUresult) -> Result<(), GpuError> {
            if result == CUDA_SUCCESS {
                Ok(())
            } else {
                Err(GpuError::CudaDriver(cuda_error_string(result).to_string(), result))
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod loading {
    use super::*;

    impl CudaDriver {
        /// CUDA not available without feature
        #[must_use]
        pub fn load() -> Option<&'static Self> {
            None
        }

        /// Check is a no-op without CUDA
        pub fn check(_result: CUresult) -> Result<(), GpuError> {
            Err(GpuError::CudaNotAvailable("cuda feature not enabled".to_string()))
        }
    }
}

// ============================================================================
// Error String Conversion
// ============================================================================

/// Convert CUDA error code to human-readable string
#[must_use]
pub fn cuda_error_string(code: CUresult) -> &'static str {
    match code {
        CUDA_SUCCESS => "CUDA_SUCCESS",
        CUDA_ERROR_INVALID_VALUE => "CUDA_ERROR_INVALID_VALUE",
        CUDA_ERROR_OUT_OF_MEMORY => "CUDA_ERROR_OUT_OF_MEMORY",
        CUDA_ERROR_NOT_INITIALIZED => "CUDA_ERROR_NOT_INITIALIZED",
        CUDA_ERROR_DEINITIALIZED => "CUDA_ERROR_DEINITIALIZED",
        CUDA_ERROR_NO_DEVICE => "CUDA_ERROR_NO_DEVICE",
        CUDA_ERROR_INVALID_DEVICE => "CUDA_ERROR_INVALID_DEVICE",
        CUDA_ERROR_INVALID_IMAGE => "CUDA_ERROR_INVALID_IMAGE",
        CUDA_ERROR_INVALID_CONTEXT => "CUDA_ERROR_INVALID_CONTEXT",
        CUDA_ERROR_NO_BINARY_FOR_GPU => "CUDA_ERROR_NO_BINARY_FOR_GPU",
        CUDA_ERROR_INVALID_PTX => "CUDA_ERROR_INVALID_PTX",
        CUDA_ERROR_NOT_FOUND => "CUDA_ERROR_NOT_FOUND",
        CUDA_ERROR_NOT_READY => "CUDA_ERROR_NOT_READY",
        CUDA_ERROR_ILLEGAL_ADDRESS => "CUDA_ERROR_ILLEGAL_ADDRESS",
        CUDA_ERROR_ILLEGAL_INSTRUCTION => "CUDA_ERROR_ILLEGAL_INSTRUCTION",
        CUDA_ERROR_LAUNCH_FAILED => "CUDA_ERROR_LAUNCH_FAILED",
        _ => "CUDA_ERROR_UNKNOWN",
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod proptests;
