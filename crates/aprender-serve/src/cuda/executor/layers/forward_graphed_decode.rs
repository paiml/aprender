// Graph-captured forward pass for single-token decode with kernel launch coalescing.

impl CudaExecutor {
    /// PAR-054: Graph-captured forward pass for decode (M=1)
    ///
    /// Uses CUDA graph capture to reduce kernel launch overhead from ~280 launches
    /// to 1 graph launch (~10µs vs ~5.6ms overhead).
    ///
    /// First decode token: captures the kernel sequence into a graph
    /// Subsequent tokens: replays the captured graph with updated position
    ///
    /// # Performance
    ///
    /// - Without graphs: ~280 kernel launches × ~20µs = ~5.6ms overhead/token
    /// - With graphs: 1 graph launch × ~10µs = ~0.01ms overhead/token
    /// - Expected speedup: ~500x reduction in launch overhead
    #[allow(clippy::too_many_arguments)]
    pub fn forward_all_layers_gpu_to_logits_graphed(
        &mut self,
        input: &[f32],
        logits: &mut [f32],
        position: u32,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // C-GDP-001: Profiling requires eager path for per-brick instrumentation.
        // CUDA graph replay executes all kernels in one opaque launch, hiding bricks.
        // Contract: gpu-decode-profiling-v1 FALSIFY-GDP-001.
        // PAR-118: Skip capture if previous attempt failed (error 901 → CUDA corruption).
        if self.should_use_eager_decode() || self.graph_capture_failed {
            return self.forward_all_layers_gpu_to_logits(
                input, logits, position, num_layers, hidden_dim,
                intermediate_dim, vocab_size, epsilon,
            );
        }

        // PAR-054: Replay captured graph if available
        if self.decode_graph.is_some() && self.decode_token_count > 0 {
            if self.decode_token_count <= 3 && verbose() {
                eprintln!(
                    "[PAR-054] Graph replay #{} (pos={})",
                    self.decode_token_count, position
                );
            }
            return self.forward_graphed_replay(input, logits, position);
        }

        // First token: attempt graph capture (requires workspace + lm_head)
        self.try_first_token_graph_capture(
            input, logits, position, num_layers, hidden_dim,
            intermediate_dim, vocab_size, epsilon,
        )
    }

    /// Returns true if the eager (non-graphed) decode path should be used.
    /// GH-559-PERF: Default to eager. Graph capture poisons CUDA context on
    /// drivers 570.207 and 590.48.01 (CUDA_ERROR_UNKNOWN code 901).
    /// Opt-in with CUDA_GRAPH_ENABLE=1.
    fn should_use_eager_decode(&self) -> bool {
        static GRAPH_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let graph_enabled = *GRAPH_ENABLED.get_or_init(|| {
            std::env::var("CUDA_GRAPH_ENABLE")
                .map(|v| v == "1")
                .unwrap_or(false)
        });
        !graph_enabled || self.profiler.is_enabled()
    }

    /// First-token graph capture: initialize buffers, attempt capture, fallback on failure.
    #[allow(clippy::too_many_arguments)]
    fn try_first_token_graph_capture(
        &mut self,
        input: &[f32],
        logits: &mut [f32],
        position: u32,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let use_workspace = self.has_workspace()
            && self.has_indexed_weights()
            && self.indexed_layer_weights.len() == num_layers;

        if !use_workspace {
            eprintln!("[PAR-054] Workspace not ready, using non-graphed path (has_workspace={}, has_indexed={}, layers={})",
                self.has_workspace(), self.has_indexed_weights(), self.indexed_layer_weights.len());
            return self.forward_all_layers_gpu_to_logits(
                input, logits, position, num_layers, hidden_dim,
                intermediate_dim, vocab_size, epsilon,
            );
        }

        if self.lm_head_ptr == 0 {
            eprintln!("[PAR-054] lm_head_ptr not set, using non-graphed path");
            return self.forward_all_layers_gpu_to_logits(
                input, logits, position, num_layers, hidden_dim,
                intermediate_dim, vocab_size, epsilon,
            );
        }

        self.prepare_capture_buffers(input, position, hidden_dim, vocab_size)?;
        self.preload_modules_for_capture(num_layers, hidden_dim, intermediate_dim, vocab_size)?;

        // trueno#243: Skip stream capture (code 901 poisons context on driver 570.207).
        // Go directly to manual graph construction via cuGraphAddKernelNode.
        // Run one eager forward with recording, then build graph from records.
        eprintln!("[trueno#243] Manual graph construction (skipping stream capture)...");
        self.begin_graph_recording();
        self.is_capturing = true;
        let eager_result = self.forward_workspace_captured(
            num_layers, hidden_dim, intermediate_dim, vocab_size, epsilon,
        );
        self.is_capturing = false;

        if let Err(eager_err) = eager_result {
            self.graph_recording = false;
            self.graph_capture_failed = true;
            eprintln!("[trueno#243] Eager forward during recording failed: {:?}", eager_err);
            return self.forward_all_layers_gpu_to_logits(
                input, logits, position, num_layers, hidden_dim,
                intermediate_dim, vocab_size, epsilon,
            );
        }

        // Eager pass succeeded — now build graph from recorded kernels
        match self.end_graph_recording() {
            Ok(n) if n > 0 => {
                // Manual graph built! First token already computed by eager pass.
                // Download logits from workspace buffer.
                self.stream.synchronize()?;
                if let Some(ref logits_buf) = self.workspace.logits_buf {
                    logits_buf.copy_to_host(logits)?;
                }
                Ok(())
            },
            Ok(_) => {
                eprintln!("[trueno#243] No kernels recorded (recording not wired to all ops yet)");
                // First token was computed by eager pass, download logits
                self.stream.synchronize()?;
                if let Some(ref logits_buf) = self.workspace.logits_buf {
                    logits_buf.copy_to_host(logits)?;
                }
                // Mark graph as failed so subsequent tokens use eager path
                self.graph_capture_failed = true;
                Ok(())
            },
            Err(graph_err) => {
                self.graph_capture_failed = true;
                eprintln!("[trueno#243] Manual graph build failed: {:?}, using eager", graph_err);
                // First token was already computed, just download logits
                self.stream.synchronize()?;
                if let Some(ref logits_buf) = self.workspace.logits_buf {
                    logits_buf.copy_to_host(logits)?;
                }
                Ok(())
            },
        }
    }

    /// Initialize GPU buffers required for CUDA graph capture.
    fn prepare_capture_buffers(
        &mut self,
        input: &[f32],
        position: u32,
        hidden_dim: u32,
        vocab_size: u32,
    ) -> Result<(), GpuError> {
        // Position buffer
        match self.position_buf {
            None => { self.position_buf = Some(GpuBuffer::from_host(&self.context, &[position])?); },
            Some(ref mut buf) => { buf.copy_from_host(&[position])?; },
        }

        // PAR-061: seq_len = position + 1
        let seq_len = position + 1;
        match self.seq_len_buf {
            None => { self.seq_len_buf = Some(GpuBuffer::from_host(&self.context, &[seq_len])?); },
            Some(ref mut buf) => { buf.copy_from_host(&[seq_len])?; },
        }

        // Stable input buffer
        let hidden_size = hidden_dim as usize;
        let needs_new = self.graph_input_buf.as_ref().map_or(true, |b| b.len() != hidden_size);
        if needs_new {
            self.graph_input_buf = Some(GpuBuffer::from_host(&self.context, input)?);
        } else {
            self.graph_input_buf.as_mut().unwrap().copy_from_host(input)?;
        }

        // Pre-allocate workspace buffers
        if self.workspace.normed_hidden_buf.is_none() {
            self.workspace.normed_hidden_buf = Some(GpuBuffer::new(&self.context, hidden_size)?);
        }
        // PMAT-088: Logits buffer may have been resized by batched decode (M*vocab_size).
        // Must reallocate to exact vocab_size for M=1 graph capture, otherwise
        // copy_to_host will fail with "Length mismatch" (host=vocab vs device=M*vocab).
        let needs_logits = self.workspace.logits_buf.as_ref().map_or(true, |b| {
            b.len() != vocab_size as usize
        });
        if needs_logits {
            self.workspace.logits_buf = Some(GpuBuffer::new(&self.context, vocab_size as usize)?);
        }

        Ok(())
    }
}
