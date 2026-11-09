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

    /// #5279: forget the cached Q8 activation, so a capture that follows records
    /// every quantize its GEMVs need instead of skipping one the cache served
    /// on the host side only.
    pub(crate) fn forget_q8_activation(&mut self) {
        self.q8_activation_valid = false;
    }

    /// #5279: the device address of every scratch buffer the executor owns
    /// and may grow (0 for one not yet allocated). A recorded launch bakes
    /// these addresses in, and another path (prefill, the single-slot step)
    /// can reallocate any of them, so a graph is only valid while they all
    /// stay where they were at capture.
    pub(crate) fn graph_scratch_ptrs(&self) -> Vec<u64> {
        fn p<T>(b: Option<&GpuBuffer<T>>) -> u64 {
            b.map_or(0, GpuBuffer::as_ptr)
        }
        let w = &self.workspace;
        vec![
            p(w.hidden_buf1.as_ref()),
            p(w.hidden_buf2.as_ref()),
            p(w.input_staging.as_ref()),
            p(w.q_buf.as_ref()),
            p(w.k_buf.as_ref()),
            p(w.v_buf.as_ref()),
            p(w.ffn_gate_buf.as_ref()),
            p(w.ffn_up_buf.as_ref()),
            p(w.ffn_act_buf.as_ref()),
            p(w.attn_out_buf.as_ref()),
            p(w.logits_buf.as_ref()),
            p(w.normed_hidden_buf.as_ref()),
            p(w.positions_buf.as_ref()),
            p(w.q8_activation_buf.as_ref()),
            p(w.batched_q8_buf.as_ref()),
            p(self.gemv_input_buffer.as_ref()),
            p(self.gemv_output_buffer.as_ref()),
            p(self.gemv_output_buffer_b.as_ref()),
            p(self.gemv_output_buffer_c.as_ref()),
            p(self.decode_attn_partials.as_ref()),
            p(self.argmax_block_vals.as_ref()),
            p(self.argmax_block_idxs.as_ref()),
            p(self.argmax_result.as_ref()),
            p(self.cublas_workspace.as_ref()),
            p(self.dequant_scratch.as_ref()),
            p(self.fp16_activation_scratch.as_ref()),
            p(self.fp16_dequant_temp.as_ref()),
            p(self.fp8_activation_scratch.as_ref()),
            p(self.fp8_act_scale_buf.as_ref()),
            p(self.fp8_act_row_absmax.as_ref()),
            p(self.wmma_scratch.as_ref()),
            p(self.dp4a_q8_scratch.as_ref()),
        ]
    }

    /// aprender#4233: hand the graph [`Self::end_graph_recording`] just built to
    /// a caller that owns its own replay (the Qwen3.5 decode step), instead of
    /// leaving it in the dense path's `decode_graph` slot.
    pub(crate) fn take_decode_graph(&mut self) -> Option<CudaGraphExec> {
        self.decode_graph.take()
    }

    /// aprender#4233: upload `host` into `buf` on THIS executor's stream, so the
    /// copy is ordered before the next launch or graph replay on it.
    pub(crate) fn upload_on_stream<T: Copy>(
        &self,
        buf: &mut GpuBuffer<T>,
        host: &[T],
    ) -> Result<(), GpuError> {
        // SAFETY: `host` is a pageable slice; cuMemcpyHtoDAsync stages pageable
        // memory before it returns. The callers also sync the stream before
        // `host` goes out of scope (the ONE sync in front of the logits read).
        unsafe { buf.copy_from_host_async(host, &self.stream) }
    }

    /// aprender#4233: replay a graph on this executor's stream. The inputs the
    /// graph reads must already be uploaded with [`Self::upload_on_stream`].
    pub(crate) fn launch_graph_exec(&self, exec: &CudaGraphExec) -> Result<(), GpuError> {
        self.stream.launch_graph(exec)
    }
}
