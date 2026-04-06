// trueno#243: Manual CUDA graph construction for decode loop.
//
// Bypasses cuStreamBeginCapture (broken on driver 570.207 code 901)
// by building the graph explicitly via cuGraphAddKernelNode.
//
// Protocol:
// 1. First decode token: set graph_recording=true, run eager forward
//    (kernels execute AND get recorded)
// 2. Build CudaGraph from recorded kernels
// 3. Subsequent tokens: replay graph (single cuGraphLaunch)
// 4. Before replay: update position_buf + seq_len_buf via async memcpy

#![allow(clippy::wildcard_imports)]

use super::super::*;

impl CudaExecutor {
    /// Start recording kernel launches for manual graph construction.
    /// Kernels still execute (eager) but are also recorded.
    pub(crate) fn begin_graph_recording(&mut self) {
        self.graph_recorded_kernels.clear();
        self.graph_recording = true;
    }

    /// Stop recording and build a CudaGraph from recorded kernels.
    ///
    /// Returns the number of kernels captured.
    pub(crate) fn end_graph_recording(&mut self) -> Result<usize, GpuError> {
        self.graph_recording = false;
        let num_kernels = self.graph_recorded_kernels.len();

        if num_kernels == 0 {
            return Ok(0);
        }

        // Build graph from recorded kernels (linear dependency chain)
        let mut graph = trueno_gpu::driver::CudaGraph::new()?;
        let mut prev_node = None;

        for record in &self.graph_recorded_kernels {
            // Reconstruct arg pointers from stored u64 values
            let mut arg_storage: Vec<u64> = record.arg_data.clone();
            let mut arg_ptrs: Vec<*mut std::ffi::c_void> = arg_storage
                .iter_mut()
                .map(|v| std::ptr::from_mut(v) as *mut std::ffi::c_void)
                .collect();

            let deps: Vec<trueno_gpu::driver::sys::CUgraphNode> = match prev_node {
                Some(node) => vec![node],
                None => vec![],
            };

            let node = graph.add_kernel_node(
                record.func.0,
                (
                    record.config.grid.0,
                    record.config.grid.1,
                    record.config.grid.2,
                ),
                (
                    record.config.block.0,
                    record.config.block.1,
                    record.config.block.2,
                ),
                record.config.shared_mem,
                &mut arg_ptrs,
                &deps,
            )?;

            prev_node = Some(node);
        }

        // Instantiate
        let graph_exec = graph.instantiate()?;
        self.decode_graph = Some(graph_exec);
        self.decode_token_count = 1;

        // realizr#198 DEBUG: Log recorded arg pointers for first AND last kernel
        // First kernel = RMSNorm (reads graph_input_buf)
        // Last kernel = LM head GEMV or bias add (writes logits_buf)
        let first_args = self.graph_recorded_kernels.first().map(|k| {
            k.arg_data
                .iter()
                .map(|a| format!("{:#x}", a))
                .collect::<Vec<_>>()
        });
        let last_args = self.graph_recorded_kernels.last().map(|k| {
            k.arg_data
                .iter()
                .map(|a| format!("{:#x}", a))
                .collect::<Vec<_>>()
        });
        // Also log logits_buf current pointer for comparison
        let logits_ptr = self
            .workspace
            .logits_buf
            .as_ref()
            .map(|b| b.as_ptr())
            .unwrap_or(0);
        eprintln!(
            "[trueno#243] ✓ Manual graph: {} kernels. first_args={:?}, last_args={:?}, current_logits_buf={:#x}",
            num_kernels, first_args, last_args, logits_ptr
        );

        Ok(num_kernels)
    }

    /// Record a kernel launch for manual graph construction.
    /// Called by kernel dispatch functions when graph_recording is true.
    pub(crate) fn record_kernel_launch(
        &mut self,
        func: trueno_gpu::driver::sys::CUfunction,
        config: &LaunchConfig,
        args: &[u64],
    ) {
        if self.graph_recording {
            self.graph_recorded_kernels.push(RecordedKernel {
                func: SendCUfunction(func),
                config: config.clone(),
                arg_data: args.to_vec(),
            });
        }
    }
}
