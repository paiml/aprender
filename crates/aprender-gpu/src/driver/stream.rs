//! CUDA Stream Management
//!
//! Provides async execution streams for overlapping computation with data transfer.
//!
//! # Design Philosophy
//!
//! Streams enable:
//! - Overlapping H2D copy with kernel execution
//! - Overlapping kernel execution with D2H copy
//! - Parallel kernel execution on different streams
//!
//! # Citation
//!
//! [2] Sourouri et al. (ICPADS 2014) demonstrates that overlapping computation
//!     with communication via CUDA streams is essential for hiding PCIe latency.

use std::ffi::{c_uint, c_void};
use std::ptr;

use super::context::{get_driver, CudaContext};
use super::graph::{CaptureMode, CudaGraph, CudaGraphExec};
use super::module::CudaModule;
use super::sys::{
    CUevent, CUfunction, CUstream, CudaDriver, CUDA_ERROR_NOT_READY, CU_EVENT_DISABLE_TIMING,
    CU_STREAM_DEFAULT, CU_STREAM_NON_BLOCKING,
};
use super::types::LaunchConfig;
use crate::GpuError;

// ============================================================================
// CUDA Stream
// ============================================================================

/// CUDA execution stream
///
/// Commands submitted to a stream execute in order.
/// Commands on different streams may execute concurrently.
///
/// # RAII
///
/// Stream is automatically destroyed when dropped.
pub struct CudaStream {
    /// Stream handle
    stream: CUstream,
}

// SAFETY: CUstream handles are thread-safe
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

/// PERF-053 (aprender#2767): choose the `cuStreamCreate` flags.
///
/// `CU_STREAM_NON_BLOCKING` is explicitly EXCLUDED from legacy default-stream ordering, while
/// `GpuBuffer::copy_from_host` / `copy_to_host` are `cuMemcpyHtoD` / `cuMemcpyDtoH` -- LEGACY
/// stream transfers. A non-blocking stream therefore does not order against them, so every host
/// transfer in this crate races whatever kernels are in flight, while the surrounding code is
/// written throughout as though the transfers were ordered against it.
///
/// Measured at the level a user sees (aprender#2767): ten rounds of four concurrent IDENTICAL
/// greedy requests returned **11 distinct continuations out of 40** under `CU_STREAM_NON_BLOCKING`,
/// most of them garbage, against **1 out of 40** -- the correct answer -- under
/// `CU_STREAM_DEFAULT`. So `CU_STREAM_DEFAULT` is the default here.
///
/// **This is not free.** The same measurement put wall-clock aggregate throughput at 1.006x of the
/// non-blocking arm at c=1 (no measurable cost), **0.893x at c=4** and **0.817x at c=8**, ranges
/// disjoint at both. `APR_STREAM_NONBLOCKING=1` restores the old, faster, RACY behaviour so that
/// cost stays reproducible in ONE binary rather than across two builds. It is a measurement knob,
/// not a supported production setting: it is known to return wrong answers under concurrency.
#[must_use]
pub fn stream_create_flags(nonblocking_env: Option<&str>) -> c_uint {
    if nonblocking_env == Some("1") {
        CU_STREAM_NON_BLOCKING
    } else {
        CU_STREAM_DEFAULT
    }
}

impl CudaStream {
    /// Create a new CUDA stream
    ///
    /// Creates a stream that IS ordered against the legacy default stream, because this
    /// crate's host transfers are legacy-stream transfers. See [`stream_create_flags`].
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::StreamCreate)` if stream creation fails.
    pub fn new(_ctx: &CudaContext) -> Result<Self, GpuError> {
        let driver = get_driver()?;

        let mut stream: CUstream = ptr::null_mut();
        let flags = stream_create_flags(std::env::var("APR_STREAM_NONBLOCKING").ok().as_deref());
        // SAFETY: stream pointer is valid
        let result = unsafe { (driver.cuStreamCreate)(&mut stream, flags) };
        CudaDriver::check(result).map_err(|e| GpuError::StreamCreate(e.to_string()))?;

        Ok(Self { stream })
    }

    /// Get raw stream handle
    ///
    /// # Safety
    ///
    /// The returned handle is only valid while this `CudaStream` is alive.
    #[must_use]
    pub fn raw(&self) -> CUstream {
        self.stream
    }

    /// Synchronize this stream
    ///
    /// Blocks until all commands in this stream have completed.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::StreamSync)` if synchronization fails.
    pub fn synchronize(&self) -> Result<(), GpuError> {
        let driver = get_driver()?;

        // SAFETY: stream is valid from constructor
        let result = unsafe { (driver.cuStreamSynchronize)(self.stream) };
        CudaDriver::check(result).map_err(|e| GpuError::StreamSync(e.to_string()))
    }

    /// PMAT-044: Synchronous device-to-device memory copy.
    ///
    /// Copies `size_bytes` from `src_ptr` to `dst_ptr` on the device.
    /// Both pointers must be valid device pointers with sufficient allocated memory.
    ///
    /// # Safety
    ///
    /// The caller must ensure both device pointers are valid and the copy
    /// does not exceed allocated memory bounds.
    pub fn memcpy_dtod_sync(
        &self,
        dst_ptr: u64,
        src_ptr: u64,
        size_bytes: usize,
    ) -> Result<(), GpuError> {
        if size_bytes == 0 {
            return Ok(());
        }
        let driver = get_driver()?;
        let result = unsafe { (driver.cuMemcpyDtoD)(dst_ptr, src_ptr, size_bytes) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(format!("D2D copy failed: {e}")))
    }

    /// Launch a kernel on this stream
    ///
    /// # Arguments
    ///
    /// * `module` - Module containing the kernel
    /// * `func_name` - Name of the kernel function
    /// * `config` - Launch configuration (grid, block, shared memory)
    /// * `args` - Kernel arguments as raw pointers
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `args` contains valid pointers to kernel arguments
    /// - Arguments match the kernel signature
    /// - Device pointers in args are valid
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::KernelLaunch)` if launch fails.
    pub unsafe fn launch_kernel(
        &self,
        module: &mut CudaModule,
        func_name: &str,
        config: &LaunchConfig,
        args: &mut [*mut c_void],
    ) -> Result<(), GpuError> {
        let driver = get_driver()?;
        let func = module.get_function(func_name)?;

        // SAFETY: Caller guarantees args are valid pointers matching kernel signature
        unsafe { self.launch_function(driver, func, config, args) }
    }

    /// Launch a kernel function directly
    ///
    /// # Safety
    ///
    /// Same safety requirements as `launch_kernel`.
    pub unsafe fn launch_function(
        &self,
        driver: &CudaDriver,
        func: CUfunction,
        config: &LaunchConfig,
        args: &mut [*mut c_void],
    ) -> Result<(), GpuError> {
        // SAFETY: func is valid, args contains valid pointers (caller's responsibility)
        let result = unsafe {
            (driver.cuLaunchKernel)(
                func,
                config.grid.0,
                config.grid.1,
                config.grid.2,
                config.block.0,
                config.block.1,
                config.block.2,
                config.shared_mem,
                self.stream,
                args.as_mut_ptr(),
                ptr::null_mut(), // extra (not used)
            )
        };

        CudaDriver::check(result).map_err(|e| GpuError::KernelLaunch(e.to_string()))?;

        Ok(())
    }

    // ========================================================================
    // PAR-037: CUDA Graph Capture
    // ========================================================================

    /// Begin stream capture (PAR-037)
    ///
    /// All subsequent operations on this stream will be recorded into a graph.
    /// Call `end_capture()` to get the captured graph.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::GraphCapture)` if capture cannot be started.
    pub fn begin_capture(&self, mode: CaptureMode) -> Result<(), GpuError> {
        let driver = get_driver()?;
        // SAFETY: stream is valid from constructor
        let result = unsafe { (driver.cuStreamBeginCapture)(self.stream, mode.to_cuda_mode()) };
        CudaDriver::check(result).map_err(|e| GpuError::GraphCapture(e.to_string()))
    }

    /// End stream capture and return the captured graph (PAR-037)
    ///
    /// Returns the graph containing all operations recorded since `begin_capture()`.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::GraphCapture)` if capture cannot be ended.
    pub fn end_capture(&self) -> Result<CudaGraph, GpuError> {
        let driver = get_driver()?;
        let mut graph = ptr::null_mut();
        // SAFETY: stream is valid from constructor
        let result = unsafe { (driver.cuStreamEndCapture)(self.stream, &mut graph) };
        CudaDriver::check(result).map_err(|e| GpuError::GraphCapture(e.to_string()))?;
        Ok(CudaGraph::from_raw(graph))
    }

    /// Launch a captured graph on this stream (PAR-037)
    ///
    /// Replays all operations in the graph with minimal launch overhead.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::GraphLaunch)` if launch fails.
    pub fn launch_graph(&self, exec: &CudaGraphExec) -> Result<(), GpuError> {
        exec.launch(self.stream)
    }

    /// Record an event on this stream (PMAT-283: non-blocking)
    ///
    /// The event will be marked as completed when all preceding operations
    /// on this stream have finished. This does NOT block the CPU.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::EventRecord)` if recording fails.
    pub fn record_event(&self, event: &CudaEvent) -> Result<(), GpuError> {
        let driver = get_driver()?;
        // SAFETY: stream and event are valid from constructors
        let result = unsafe { (driver.cuEventRecord)(event.event, self.stream) };
        CudaDriver::check(result).map_err(|e| GpuError::StreamSync(format!("event record: {e}")))
    }

    /// GH-559-PERF: Make this stream wait for an event recorded on another stream.
    ///
    /// Non-blocking cross-stream dependency: all future work on this stream will
    /// wait until the event completes, but the CPU is NOT blocked. This replaces
    /// `compute_stream.synchronize()` which blocks both GPU and CPU.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::StreamSync)` if the wait fails.
    pub fn wait_event(&self, event: &CudaEvent) -> Result<(), GpuError> {
        let driver = get_driver()?;
        // SAFETY: stream and event are valid from constructors. flags=0 per CUDA spec.
        let result = unsafe { (driver.cuStreamWaitEvent)(self.stream, event.event, 0) };
        CudaDriver::check(result)
            .map_err(|e| GpuError::StreamSync(format!("stream wait event: {e}")))
    }
}

// ============================================================================
// CUDA Event (PMAT-283: CPU-GPU pipelining)
// ============================================================================

/// CUDA event for non-blocking completion queries
///
/// Events enable CPU-GPU pipelining: record an event after GPU work,
/// then query/wait for completion without blocking the CPU thread.
///
/// # PMAT-283
///
/// This replaces `CudaStream::synchronize()` in the decode loop to enable
/// serving overhead (HTTP, tokenizer, scheduling) to overlap with GPU decode.
pub struct CudaEvent {
    event: CUevent,
}

// SAFETY: CUevent handles are thread-safe
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    /// Create a new CUDA event (timing disabled for minimal overhead)
    ///
    /// # Errors
    ///
    /// Returns `Err` if event creation fails.
    pub fn new() -> Result<Self, GpuError> {
        let driver = get_driver()?;
        let mut event: CUevent = ptr::null_mut();
        // SAFETY: event pointer is valid
        let result = unsafe { (driver.cuEventCreate)(&mut event, CU_EVENT_DISABLE_TIMING) };
        CudaDriver::check(result)
            .map_err(|e| GpuError::StreamCreate(format!("event create: {e}")))?;
        Ok(Self { event })
    }

    /// Query whether the event has completed (non-blocking)
    ///
    /// Returns `true` if all work preceding the event has completed,
    /// `false` if work is still in progress.
    ///
    /// # Errors
    ///
    /// Returns `Err` only on actual errors (not for NOT_READY).
    pub fn is_complete(&self) -> Result<bool, GpuError> {
        let driver = get_driver()?;
        // SAFETY: event is valid from constructor
        let result = unsafe { (driver.cuEventQuery)(self.event) };
        if result == CUDA_ERROR_NOT_READY {
            return Ok(false);
        }
        CudaDriver::check(result).map_err(|e| GpuError::StreamSync(format!("event query: {e}")))?;
        Ok(true)
    }

    /// Wait for the event to complete (blocking)
    ///
    /// Use `is_complete()` for non-blocking polling.
    ///
    /// # Errors
    ///
    /// Returns `Err` if synchronization fails.
    pub fn synchronize(&self) -> Result<(), GpuError> {
        let driver = get_driver()?;
        // SAFETY: event is valid from constructor
        let result = unsafe { (driver.cuEventSynchronize)(self.event) };
        CudaDriver::check(result).map_err(|e| GpuError::StreamSync(format!("event sync: {e}")))
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if let Ok(driver) = get_driver() {
            // SAFETY: event is valid from constructor
            unsafe {
                let _ = (driver.cuEventDestroy)(self.event);
            }
        }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if let Ok(driver) = get_driver() {
            // SAFETY: stream is valid from constructor
            unsafe {
                let _ = (driver.cuStreamDestroy)(self.stream);
            }
        }
    }
}

// ============================================================================
// Default Stream
// ============================================================================

/// Null stream handle (default stream)
///
/// Operations on the default stream synchronize with all other streams.
/// Use `CudaStream::new()` for non-blocking streams.
pub const DEFAULT_STREAM: CUstream = ptr::null_mut();

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_stream_is_null() {
        assert!(DEFAULT_STREAM.is_null());
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_stream_requires_cuda_feature() {
        // This test verifies the module compiles without cuda feature
        assert!(true);
    }

    // ------------------------------------------------------------------
    // PERF-053 (aprender#2767): the ordering default, at the flag level.
    //
    // This is the SHAPE half of the falsification. It cannot see a race, so
    // it is not the proof -- that is
    // `aprender-serve/tests/falsify_stream_ordering_2767.rs`, which requires a
    // GPU. What this pins is that the default is CU_STREAM_DEFAULT and that the
    // escape hatch is opt-IN, on a host with no CUDA at all.
    // ------------------------------------------------------------------

    #[test]
    fn perf053_default_stream_flag_is_legacy_ordered() {
        // Unset env => ordered. This is the line the whole ticket is about.
        assert_eq!(stream_create_flags(None), CU_STREAM_DEFAULT);
    }

    #[test]
    fn perf053_nonblocking_is_opt_in_only() {
        assert_eq!(stream_create_flags(Some("1")), CU_STREAM_NON_BLOCKING);
        // Discrimination: anything that is not exactly "1" must NOT silently
        // re-arm the racy path. A truthy-looking value is the classic way an
        // opt-in knob turns itself back on.
        for v in ["0", "", "true", "yes", "on", "TRUE", "2", "1 ", " 1"] {
            assert_eq!(
                stream_create_flags(Some(v)),
                CU_STREAM_DEFAULT,
                "APR_STREAM_NONBLOCKING={v:?} must not select the racy non-blocking path"
            );
        }
    }

    #[test]
    fn perf053_the_two_flags_actually_differ() {
        // Guard against the whole knob collapsing to a no-op if the two
        // constants were ever defined the same. Without this, both tests above
        // would pass on a tree where the escape hatch does nothing.
        assert_ne!(CU_STREAM_DEFAULT, CU_STREAM_NON_BLOCKING);
        assert_eq!(CU_STREAM_DEFAULT, 0);
    }
}
