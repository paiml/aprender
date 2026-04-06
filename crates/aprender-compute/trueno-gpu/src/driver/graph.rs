//! PAR-037: CUDA Graph Capture and Execution
//!
//! Provides CUDA graph capture for pre-recording kernel sequences.
//!
//! ## Benefits
//!
//! - ~3-10µs graph launch vs ~20-50µs per kernel launch
//! - Pre-validated kernel parameters
//! - Reduced CPU overhead for repetitive sequences
//!
//! ## Usage
//!
//! ```text
//! 1. stream.begin_capture()        - Start capture mode
//! 2. Execute kernel sequence       - Record operations
//! 3. stream.end_capture()          - Get captured graph
//! 4. graph.instantiate()           - Create executable
//! 5. stream.launch_graph(&exec)    - Replay sequence
//! ```

use std::ptr;

use super::context::get_driver;
use super::sys::{
    CUgraph, CUgraphExec, CUgraphExecUpdateResult, CUgraphNode, CUstream, CudaDriver,
    CU_GRAPH_EXEC_UPDATE_SUCCESS,
};
use crate::GpuError;

// ============================================================================
// CUDA Graph
// ============================================================================

/// Captured CUDA graph
///
/// Represents a dependency graph of operations captured from stream execution.
/// Must be instantiated before execution.
pub struct CudaGraph {
    /// Graph handle
    graph: CUgraph,
}

// SAFETY: CUgraph handles are thread-safe
unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

impl CudaGraph {
    /// Create a new empty CUDA graph (for manual construction)
    pub fn new() -> Result<Self, GpuError> {
        let driver = get_driver()?;

        let mut graph: CUgraph = ptr::null_mut();
        // SAFETY: graph pointer is valid
        let result = unsafe { (driver.cuGraphCreate)(&mut graph, 0) };
        CudaDriver::check(result).map_err(|e| GpuError::GraphCreate(e.to_string()))?;

        Ok(Self { graph })
    }

    /// Create from a captured stream
    ///
    /// This is called internally by `CudaStream::end_capture`.
    pub(crate) fn from_raw(graph: CUgraph) -> Self {
        Self { graph }
    }

    /// Get raw graph handle
    #[must_use]
    pub fn raw(&self) -> CUgraph {
        self.graph
    }

    /// trueno#243: Add a kernel launch node to the graph (manual construction).
    ///
    /// Bypasses stream capture entirely — works even when `cuStreamBeginCapture`
    /// triggers driver bug code 901 on 570.207.
    ///
    /// # Arguments
    /// * `func` - CUDA function handle from `CudaModule::get_function()`
    /// * `grid` - Grid dimensions (blocks_x, blocks_y, blocks_z)
    /// * `block` - Block dimensions (threads_x, threads_y, threads_z)
    /// * `shared_mem` - Dynamic shared memory bytes
    /// * `args` - Kernel argument pointers
    /// * `deps` - Graph nodes this node depends on (empty for first node)
    #[allow(clippy::too_many_arguments)]
    pub fn add_kernel_node(
        &mut self,
        func: super::sys::CUfunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        args: &mut [*mut std::ffi::c_void],
        deps: &[super::sys::CUgraphNode],
    ) -> Result<super::sys::CUgraphNode, GpuError> {
        let driver = get_driver()?;

        let params = super::sys::CudaKernelNodeParams {
            func,
            grid_dim_x: grid.0,
            grid_dim_y: grid.1,
            grid_dim_z: grid.2,
            block_dim_x: block.0,
            block_dim_y: block.1,
            block_dim_z: block.2,
            shared_mem_bytes: shared_mem,
            kernel_params: args.as_mut_ptr(),
            extra: ptr::null_mut(),
        };

        let mut node: super::sys::CUgraphNode = ptr::null_mut();
        let result = unsafe {
            (driver.cuGraphAddKernelNode)(
                &mut node,
                self.graph,
                if deps.is_empty() { ptr::null() } else { deps.as_ptr() },
                deps.len(),
                &params,
            )
        };
        CudaDriver::check(result)
            .map_err(|e| GpuError::GraphCreate(format!("add_kernel_node: {e}")))?;

        Ok(node)
    }

    /// Instantiate the graph for execution
    ///
    /// Creates an executable graph instance that can be launched on streams.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::GraphInstantiate)` if instantiation fails.
    pub fn instantiate(&self) -> Result<CudaGraphExec, GpuError> {
        let driver = get_driver()?;

        let mut graph_exec: CUgraphExec = ptr::null_mut();
        // SAFETY: graph and graph_exec pointers are valid
        let result =
            unsafe { (driver.cuGraphInstantiateWithFlags)(&mut graph_exec, self.graph, 0) };
        CudaDriver::check(result).map_err(|e| GpuError::GraphInstantiate(e.to_string()))?;

        Ok(CudaGraphExec::from_raw(graph_exec))
    }
}

impl Default for CudaGraph {
    fn default() -> Self {
        Self::new().expect("Failed to create CUDA graph")
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        if !self.graph.is_null() {
            if let Ok(driver) = get_driver() {
                // SAFETY: graph is valid from constructor
                unsafe { (driver.cuGraphDestroy)(self.graph) };
            }
        }
    }
}

// ============================================================================
// CUDA Graph Executable
// ============================================================================

/// Executable CUDA graph instance
///
/// Instantiated from a `CudaGraph`, this can be launched on streams
/// with minimal overhead (~3-10µs vs ~20-50µs per kernel).
pub struct CudaGraphExec {
    /// Graph executable handle
    exec: CUgraphExec,
}

// SAFETY: CUgraphExec handles are thread-safe
unsafe impl Send for CudaGraphExec {}
unsafe impl Sync for CudaGraphExec {}

impl CudaGraphExec {
    /// Create from raw handle (internal)
    pub(crate) fn from_raw(exec: CUgraphExec) -> Self {
        Self { exec }
    }

    /// Get raw executable handle
    #[must_use]
    pub fn raw(&self) -> CUgraphExec {
        self.exec
    }

    /// Update this executable in-place from a new graph (PMAT-291).
    ///
    /// Tries to update kernel arguments without re-instantiation.
    /// Returns `true` if update succeeded, `false` if topology changed
    /// (caller should re-instantiate).
    ///
    /// This is the llama.cpp approach: capture a new graph each step,
    /// then update the existing executable. Much cheaper than
    /// destroy + re-instantiate for argument-only changes.
    pub fn update(&self, new_graph: &CudaGraph) -> Result<bool, GpuError> {
        let driver = get_driver()?;

        let mut error_node: CUgraphNode = ptr::null_mut();
        let mut update_result: CUgraphExecUpdateResult = 0;

        // SAFETY: exec, graph, and output pointers are valid
        let result = unsafe {
            (driver.cuGraphExecUpdate)(
                self.exec,
                new_graph.graph,
                &mut error_node,
                &mut update_result,
            )
        };

        // CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE is not a fatal error --
        // it means topology changed and we need to re-instantiate
        if result != 0 && update_result != CU_GRAPH_EXEC_UPDATE_SUCCESS {
            return Ok(false); // Topology changed, need re-instantiate
        }

        CudaDriver::check(result)
            .map_err(|e| GpuError::GraphInstantiate(format!("graph update: {e}")))?;

        Ok(update_result == CU_GRAPH_EXEC_UPDATE_SUCCESS)
    }

    /// Launch this graph on a stream
    ///
    /// Replays all captured operations with minimal overhead.
    ///
    /// # Arguments
    ///
    /// * `stream` - Stream to launch on (use raw handle)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::GraphLaunch)` if launch fails.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `stream` is a valid CUDA stream handle.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn launch(&self, stream: CUstream) -> Result<(), GpuError> {
        let driver = get_driver()?;

        // SAFETY: exec and stream handles are valid
        let result = unsafe { (driver.cuGraphLaunch)(self.exec, stream) };
        CudaDriver::check(result).map_err(|e| GpuError::GraphLaunch(e.to_string()))
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            if let Ok(driver) = get_driver() {
                // SAFETY: exec is valid from constructor
                unsafe { (driver.cuGraphExecDestroy)(self.exec) };
            }
        }
    }
}

// ============================================================================
// Stream Capture Mode Extension
// ============================================================================

/// Capture mode for CUDA graphs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CaptureMode {
    /// Global mode - all operations on any stream are captured
    #[default]
    Global,
    /// Thread-local mode - only operations from capturing thread are captured
    ThreadLocal,
    /// Relaxed mode - allows dependencies from other streams
    Relaxed,
}

impl CaptureMode {
    /// Convert to CUDA capture mode constant
    #[must_use]
    pub fn to_cuda_mode(self) -> u32 {
        match self {
            CaptureMode::Global => 0,      // CU_STREAM_CAPTURE_MODE_GLOBAL
            CaptureMode::ThreadLocal => 1, // CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
            CaptureMode::Relaxed => 2,     // CU_STREAM_CAPTURE_MODE_RELAXED
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capture_mode_values() {
        assert_eq!(CaptureMode::Global.to_cuda_mode(), 0);
        assert_eq!(CaptureMode::ThreadLocal.to_cuda_mode(), 1);
        assert_eq!(CaptureMode::Relaxed.to_cuda_mode(), 2);
    }

    #[test]
    fn test_capture_mode_default() {
        assert_eq!(CaptureMode::default(), CaptureMode::Global);
    }
}
