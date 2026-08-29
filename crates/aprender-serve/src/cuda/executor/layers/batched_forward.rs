// Batched forward pass dispatching M parallel sequences through all transformer layers.

impl CudaExecutor {
    /// PERF-050: `APR_LAYER_TRACE=1` enables the per-layer hidden-state trace used to bisect
    /// FALSIFY-CB-006. Read once per process.
    fn layer_trace_enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("APR_LAYER_TRACE").as_deref() == Ok("1"))
    }

    /// PERF-050: dump per-slot hidden-state statistics after one layer.
    ///
    /// Supersedes the realizr#220 diagnostic that sat inline in batched_forward_run_layers:
    /// layer 0 only, m >= 3 only, and UNCONDITIONAL, so every decode step of every m >= 3 batch
    /// paid a stream synchronize plus a device-to-host copy on the hot path with no way to turn
    /// it off. `nonfinite` is reported because CB-006's residue returns all-NaN logits on the
    /// second batched request and the bisect has to name the layer that first sees them.
    /// PERF-050 round 5: content fingerprint of one device buffer.
    ///
    /// The CB-006 bisect is down to two stages whose INPUT is already proven identical across
    /// two invocations of the batched forward, so the split is decided by which of their two
    /// OUTPUTS first differs. Pointer and length cannot answer that; contents can.
    ///
    /// `ALL-ZERO` is called out rather than printed as `sum=0 absmax=0`, because an unwritten
    /// buffer and a legitimately zero tensor produce identical statistics. This investigation
    /// has already had to withdraw one result to that exact confusion.
    fn trace_buffer(&self, label: &str, ptr: u64, len: usize) {
        if self.stream.synchronize().is_err() {
            eprintln!("[CB-006-OUT] {label} SYNC FAILED");
            return;
        }
        // SAFETY: non-owning view over a live device allocation of at least `len` f32; leaked
        // below so Drop never frees the borrowed allocation.
        let buf = unsafe { GpuBuffer::<f32>::from_raw_parts(ptr, len) };
        let mut host = vec![0.0f32; len];
        let res = buf.copy_to_host(&mut host);
        std::mem::forget(buf);
        if res.is_err() {
            eprintln!("[CB-006-OUT] {label} DOWNLOAD FAILED ptr={ptr:#x} len={len}");
            return;
        }
        let nonfinite = host.iter().filter(|v| !v.is_finite()).count();
        let sum: f64 = host.iter().filter(|v| v.is_finite()).map(|v| f64::from(*v)).sum();
        let absmax = host.iter().filter(|v| v.is_finite()).fold(0.0f32, |a, v| a.max(v.abs()));
        let zero = host.iter().all(|v| *v == 0.0);
        eprintln!(
            "[CB-006-OUT] {label} ptr={ptr:#x} len={len} sum={sum:.6} absmax={absmax:.6} \
             nonfinite={nonfinite} first2={:?}{}",
            &host[..2.min(len)],
            if zero { "  ALL-ZERO(SUSPECT)" } else { "" }
        );
    }

    fn trace_layer_hidden(
        &self,
        layer_idx: usize,
        hidden_buf2_ptr: u64,
        m: usize,
        hidden_dim: u32,
    ) -> Result<(), GpuError> {
        self.stream.synchronize()?;
        let hd = hidden_dim as usize;
        // SAFETY: non-owning view over the already-allocated hidden_buf2 device region; leaked
        // below so Drop never frees the borrowed allocation.
        let buf = unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, m * hd) };
        let mut host = vec![0.0f32; m * hd];
        let res = buf.copy_to_host(&mut host);
        std::mem::forget(buf);
        res?;
        for seq in 0..m {
            let slice = &host[seq * hd..(seq + 1) * hd];
            let nonfinite = slice.iter().filter(|v| !v.is_finite()).count();
            let sum: f32 = slice.iter().filter(|v| v.is_finite()).sum();
            let absmax = slice
                .iter()
                .filter(|v| v.is_finite())
                .fold(0.0f32, |a, v| a.max(v.abs()));
            eprintln!(
                "[CB-006-LAYER] layer={layer_idx} seq={seq} sum={sum:.4} absmax={absmax:.4} \
                 nonfinite={nonfinite}/{hd} first4={:?}",
                &slice[..4.min(hd)]
            );
        }
        Ok(())
    }

    /// PAR-111: Batched forward pass for M sequences returning M token IDs
    ///
    /// Processes M sequences in parallel through all transformer layers using
    /// batched GEMV kernels that read/dequantize weights ONCE for all M inputs.
    ///
    /// # Performance
    ///
    /// - M=1: Baseline (~360 tok/s)
    /// - M=4: 16x GEMV speedup → 857+ tok/s aggregate throughput
    ///
    /// # Arguments
    ///
    /// * `inputs` - M embeddings packed [M × hidden_dim]
    /// * `positions` - M sequence positions for RoPE
    /// * `num_layers` - Number of transformer layers
    /// * `hidden_dim` - Hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `vocab_size` - Vocabulary size
    /// * `epsilon` - RMSNorm epsilon
    ///
    /// # Returns
    ///
    /// M token IDs (greedy argmax)
    ///
    /// PMAT-764: the embed + per-layer forward is shared via batched_forward_run_layers so a
    /// sibling (forward_batched_to_logits) can reuse it for per-request sampling.
    #[allow(clippy::too_many_arguments)]
    fn batched_forward_run_layers(
        &mut self,
        inputs: &[f32],
        positions: &[u32],
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        let m = positions.len();
        // PAR-129: Extended to M=32 via 4-warp kernel
        if m == 0 || m > 32 {
            return Err(GpuError::InvalidParameter(format!(
                "PAR-111: batch size must be 1-32, got {}",
                m
            )));
        }
        let expected_input_len = m * hidden_dim as usize;
        if inputs.len() != expected_input_len {
            return Err(GpuError::InvalidParameter(format!(
                "PAR-111: inputs.len() {} != M*hidden_dim = {}",
                inputs.len(),
                expected_input_len
            )));
        }

        // Verify batched workspace initialized
        if !self.workspace.initialized || self.workspace.batch_size != m {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-111: Batched workspace not initialized for M={}",
                m
            )));
        }

        // 1. Upload M embeddings to GPU
        // PMAT-088: Buffer may be over-sized (high-water mark from larger M).
        // copy_from_host requires exact length match, so reallocate on size change.
        if self.batched_decode_input_buf.as_ref().map_or(true, |b| b.len() != expected_input_len)
        {
            self.batched_decode_input_buf =
                Some(GpuBuffer::new(&self.context, expected_input_len)?);
            self.batched_decode_input_cap = expected_input_len;
        }
        self.batched_decode_input_buf
            .as_mut()
            .expect("batched_decode_input_buf must be allocated before copy")
            .copy_from_host(inputs)
            .map_err(|e| GpuError::Transfer(format!(
                "PMAT-088c batched_decode_input_buf: host={} device={}: {e}",
                inputs.len(),
                self.batched_decode_input_buf.as_ref().map_or(0, |b| b.len()),
            )))?;
        let input_buf_ptr = self.batched_decode_input_buf.as_ref().expect("batched_decode_input_buf must be allocated before batched forward pass").as_ptr();
        let input_buf_len = expected_input_len;

        // Get workspace buffer pointers to avoid borrow conflicts
        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: hidden_buf2 missing".to_string())
            })?
            .as_ptr();
        // PMAT-088: Use logical size (M * hidden_dim), not allocated capacity.
        // Workspace buffers use high-water mark allocation — may be larger than current M.
        let hidden_buf2_len = expected_input_len;

        // 2. Process all layers with batched GEMV
        for layer_idx in 0..num_layers {
            // Get indexed layer weights (must be pre-built via build_indexed_weights)
            if layer_idx >= self.indexed_layer_weights.len() {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PAR-111: Layer {} weights not indexed (have {})",
                    layer_idx,
                    self.indexed_layer_weights.len()
                )));
            }
            let layer_weights = self.get_indexed_layer(layer_idx).clone();

            // Use workspace output from previous layer (or input_buf for first layer)
            // SAFETY: Pointers valid from allocation, length verified, used within scope
            let layer_input_buf = if layer_idx == 0 {
                // PMAT-086: Use pre-allocated input buffer pointer (same pattern as hidden_buf2)
                // SAFETY: constructs a non-owning `GpuBuffer` view over an already-allocated device region (`ptr`, element count `len`) that stays live for the kernel call; the view is `leak()`ed afterwards so its Drop never frees the borrowed device allocation (no double-free).
                unsafe { GpuBuffer::<f32>::from_raw_parts(input_buf_ptr, input_buf_len) }
            } else {
                // SAFETY: constructs a non-owning `GpuBuffer` view over an already-allocated device region (`ptr`, element count `len`) that stays live for the kernel call; the view is `leak()`ed afterwards so its Drop never frees the borrowed device allocation (no double-free).
                unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) }
            };

            let layer_input = &layer_input_buf;

            // PMAT-291: Graph dispatch path (GRAPH_DISPATCH=1)
            if self.use_graph_dispatch() {
                self.transformer_layer_batched_graph(
                    layer_input,
                    layer_idx,
                    &layer_weights,
                    m as u32,
                    positions,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                )?;
            } else {
                self.transformer_layer_batched(
                    layer_input,
                    layer_idx,
                    &layer_weights,
                    m as u32,
                    positions,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                )?;
            }

            // Prevent drop of borrowed buffer (from_raw_parts doesn't own the memory)
            std::mem::forget(layer_input_buf);

            // PERF-050: per-layer hidden-state trace, the layer bisect for CB-006.
            // Behind APR_LAYER_TRACE=1; see trace_layer_hidden.
            if Self::layer_trace_enabled() {
                self.trace_layer_hidden(layer_idx, hidden_buf2_ptr, m, hidden_dim)?;
            }
        }

        Ok(())
    }

    /// PMAT-764: batched forward → M greedy token IDs (on-GPU argmax). Behavior is identical
    /// to the pre-refactor single-method version.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batched_to_token_ids(
        &mut self,
        inputs: &[f32],
        positions: &[u32],
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<Vec<u32>, GpuError> {
        self.batched_forward_run_layers(
            inputs,
            positions,
            num_layers,
            hidden_dim,
            intermediate_dim,
            epsilon,
        )?;
        let m = positions.len();
        self.batched_output_norm_lm_head_argmax(m, hidden_dim, vocab_size, epsilon)
    }

    /// PMAT-764: batched forward → per-slot LOGITS downloaded to host (m × vocab_size,
    /// row-major per slot), so the batched decode loop can apply per-request temperature/top_k
    /// sampling instead of forced greedy argmax. Used only when a slot requests temperature>0.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batched_to_logits(
        &mut self,
        inputs: &[f32],
        positions: &[u32],
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<Vec<f32>, GpuError> {
        self.batched_forward_run_layers(
            inputs,
            positions,
            num_layers,
            hidden_dim,
            intermediate_dim,
            epsilon,
        )?;
        let m = positions.len();
        self.batched_output_norm_lm_head_logits(m, hidden_dim, vocab_size, epsilon)
    }

    /// PMAT-764: output norm → LM head projection, leaving logits in workspace.logits_buf.
    /// Returns the device pointer (u64) + element count (m × vocab_size) so the caller can
    /// either argmax on GPU (greedy) or download for CPU sampling. Extracted unchanged from
    /// the former batched_output_norm_lm_head_argmax body.
    #[allow(clippy::too_many_arguments)]
    fn batched_output_norm_lm_head_into_logits(
        &mut self,
        m: usize,
        hidden_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(u64, usize), GpuError> {
        // Output norm (PAR-115: Batched - single launch for M sequences)
        let output_norm_buf = self.rmsnorm_cache.get("output_norm.gamma").ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-111: output_norm not cached".to_string())
        })?;
        let output_norm_ptr = output_norm_buf.as_ptr();
        let output_norm_len = hidden_dim as usize;

        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: hidden_buf2 missing".to_string())
            })?
            .as_ptr();
        let hidden_buf2_len = m * hidden_dim as usize;
        let normed_hidden_ptr = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: normed_hidden_buf missing".to_string())
            })?
            .as_ptr();
        let normed_hidden_len = m * hidden_dim as usize;

        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let hidden_buf2 =
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) };
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let normed_hidden_buf =
            unsafe { GpuBuffer::<f32>::from_raw_parts(normed_hidden_ptr, normed_hidden_len) };

        self.batched_rmsnorm_ptr_into(
            &hidden_buf2,
            output_norm_ptr,
            output_norm_len,
            &normed_hidden_buf,
            hidden_dim,
            m as u32,
            epsilon,
        )?;

        std::mem::forget(hidden_buf2);
        std::mem::forget(normed_hidden_buf);

        // PERF-050 round 5: stage 1 of 2. If this differs between two identical forwards the
        // batched output RMSNorm is at fault; if it matches and the logits differ, the LM head
        // GEMM is.
        if Self::layer_trace_enabled() {
            self.trace_buffer("normed_hidden", normed_hidden_ptr, m * hidden_dim as usize);
        }

        // LM head projection
        if self.lm_head_ptr == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "PAR-111: LM head not indexed".to_string(),
            ));
        }
        let lm_head_ptr = self.lm_head_ptr;
        let lm_head_qtype = self.lm_head_qtype;

        // PMAT-086: Reuse workspace logits buffer (grow-only, avoids cuMemAlloc per step)
        let logits_size = m * vocab_size as usize;
        if self.workspace.logits_buf.is_none()
            || self
                .workspace
                .logits_buf
                .as_ref()
                .map_or(0, trueno_gpu::driver::GpuBuffer::len)
                < logits_size
        {
            self.workspace.logits_buf = Some(GpuBuffer::new(&self.context, logits_size)?);
        }
        let logits_buf_ptr = self.workspace.logits_buf.as_ref().expect("CUDA buffer must be allocated").as_ptr();
        // SAFETY: logits_buf_ptr valid from workspace allocation, size verified above
        let logits_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(logits_buf_ptr, logits_size) };

        let normed_hidden_buf_len = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-111: normed_hidden_buf missing".to_string())
            })?
            .len();
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let normed_hidden_buf_wrapper =
            unsafe { GpuBuffer::<f32>::from_raw_parts(normed_hidden_ptr, normed_hidden_buf_len) };

        // PMAT-105: Route LmHead through full dispatch chain (batched_gemv_or_gemm)
        // instead of batched_gemv_with_fallback. At M>=5, this enables FP8 cuBLASLt
        // GEMM which reads FP8 weights ONCE (233 MB at 1 B/elem) vs Q6K batched GEMV
        // which reads Q6K weights M times (175 MB × M_effective with L2 sharing).
        // At M=12: estimated ~2.3ms savings per step (3.5ms → 1.2ms, -13% ITL).
        self.batched_gemv_or_gemm(
            lm_head_qtype,
            lm_head_ptr,
            &normed_hidden_buf_wrapper,
            &logits_buf,
            normed_hidden_ptr,
            logits_buf.as_ptr(),
            m as u32,
            vocab_size,
            hidden_dim,
        )?;

        std::mem::forget(normed_hidden_buf_wrapper);

        // PERF-050 round 5: stage 2 of 2.
        if Self::layer_trace_enabled() {
            self.trace_buffer("logits", logits_buf.as_ptr(), m * vocab_size as usize);
        }

        // PMAT-086: Removed redundant stream.synchronize() before argmax.
        // LM head GEMV and argmax both execute on self.stream — CUDA stream
        // ordering guarantees GEMV completes before argmax reads logits.
        // batched_gpu_argmax does its own sync after all 2×M kernels (reduces.rs:370).
        let logits_ptr = logits_buf.as_ptr();
        std::mem::forget(logits_buf); // Non-owning wrapper — prevent double-free
        Ok((logits_ptr, m * vocab_size as usize))
    }

    /// PMAT-764: batched output norm → LM head → on-GPU argmax (greedy). Behavior identical
    /// to the pre-refactor batched_output_norm_lm_head_argmax.
    fn batched_output_norm_lm_head_argmax(
        &mut self,
        m: usize,
        hidden_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<Vec<u32>, GpuError> {
        let (logits_ptr, _len) =
            self.batched_output_norm_lm_head_into_logits(m, hidden_dim, vocab_size, epsilon)?;
        self.batched_gpu_argmax(logits_ptr, vocab_size, m)
    }

    /// PMAT-764: batched output norm → LM head → DOWNLOAD logits to host (m × vocab_size,
    /// slot s occupies host[s*vocab_size .. (s+1)*vocab_size]) for per-request CPU sampling.
    fn batched_output_norm_lm_head_logits(
        &mut self,
        m: usize,
        hidden_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<Vec<f32>, GpuError> {
        let (logits_ptr, len) =
            self.batched_output_norm_lm_head_into_logits(m, hidden_dim, vocab_size, epsilon)?;
        // SAFETY: logits_ptr is workspace.logits_buf (valid, `len` elements); non-owning wrapper.
        let logits_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(logits_ptr, len) };
        let mut host = vec![0.0f32; len];
        let dl = logits_buf.copy_to_host(&mut host);
        std::mem::forget(logits_buf); // non-owning — prevent double-free
        dl.map_err(|e| GpuError::Transfer(format!("PMAT-764 batched logits download: {e}")))?;
        Ok(host)
    }

    /// PAR-121: Graph-captured batched forward pass for M sequences
    ///
    /// Uses CUDA graph capture to reduce kernel launch overhead for batched decode.
    /// First call with batch size M: captures the kernel sequence into a graph.
    /// Subsequent calls with same M: replays captured graph with updated inputs.
    ///
    /// # Performance
    ///
    /// - Without graphs (M=2): 404.6 tok/s
    /// - With graphs (M=2): Target ~550+ tok/s (2x Ollama)
    /// - Key: Combines batched GEMV efficiency + CUDA graph launch reduction
    #[allow(clippy::too_many_arguments)]
    pub fn forward_batched_to_token_ids_graphed(
        &mut self,
        inputs: &[f32],
        positions: &[u32],
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<Vec<u32>, GpuError> {
        let m = positions.len();
        // PAR-129: Extended to M=32 via 4-warp kernel
        if m == 0 || m > 32 {
            return Err(GpuError::InvalidParameter(format!(
                "PAR-121: batch size must be 1-32, got {}",
                m
            )));
        }
        let expected_input_len = m * hidden_dim as usize;
        if inputs.len() != expected_input_len {
            return Err(GpuError::InvalidParameter(format!(
                "PAR-121: inputs.len() {} != M*hidden_dim = {}",
                inputs.len(),
                expected_input_len
            )));
        }

        // Verify batched workspace initialized
        if !self.workspace.initialized || self.workspace.batch_size != m {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-121: Batched workspace not initialized for M={}",
                m
            )));
        }

        // Check if we have a captured graph for this batch size
        if self.batched_decode_graphs.contains_key(&m) && self.batched_graph_batch_size == m {
            // Replay path: update inputs and replay graph
            return self.forward_batched_graphed_replay(inputs, positions, m, vocab_size);
        }

        // First call or batch size changed: need to capture graph
        // Initialize stable buffers for graph capture
        self.init_batched_graph_buffers(m, hidden_dim, vocab_size)?;

        // Pre-load all kernel modules before capture
        self.preload_modules_for_batched_capture(
            num_layers,
            hidden_dim,
            intermediate_dim,
            vocab_size,
        )?;

        // Copy inputs to stable buffer
        if let Some(ref mut input_buf) = self.batched_graph_input_buf {
            input_buf.copy_from_host(inputs)?;
        }

        // Copy positions to stable buffer
        if let Some(ref mut pos_buf) = self.batched_graph_positions_buf {
            pos_buf.copy_from_host(positions)?;
        }

        // Copy seq_lens (position + 1 for each) to stable buffer
        let seq_lens: Vec<u32> = positions.iter().map(|&p| p + 1).collect();
        if let Some(ref mut len_buf) = self.batched_graph_seq_lens_buf {
            len_buf.copy_from_host(&seq_lens)?;
        }

        // PMAT-037: Pre-populate FP16 weight cache + warm cuBLAS before graph capture.
        // Graph capture doesn't allow dynamic allocation, so FP16 weights and cuBLAS
        // workspace must be allocated beforehand.
        if self.gpu_profile.hgemm_decode {
            self.ensure_cublas()?;
            self.warmup_hgemm_cache(num_layers, hidden_dim, intermediate_dim, vocab_size)?;
        }

        // Try to capture graph
        // PMAT-285: Pass real positions for realistic seq_lens during capture
        let capture_result = self.try_batched_graph_capture(
            m,
            num_layers,
            hidden_dim,
            intermediate_dim,
            vocab_size,
            epsilon,
            positions,
        );

        match capture_result {
            Ok(()) => {
                // Graph captured successfully
                self.batched_graph_batch_size = m;
                eprintln!("[PAR-121] ✓ Batched CUDA graph captured for M={}", m);

                // GH-141: Graph capture RECORDS kernels but doesn't EXECUTE them.
                // Must replay the graph to get actual logits from the real inputs.
                self.forward_batched_graphed_replay(inputs, positions, m, vocab_size)
            },
            Err(e) => {
                // Graph capture failed, fall back to non-graphed path
                eprintln!(
                    "[PAR-121] Graph capture failed for M={}: {:?}, using non-graphed path",
                    m, e
                );
                self.forward_batched_to_token_ids(
                    inputs,
                    positions,
                    num_layers,
                    hidden_dim,
                    intermediate_dim,
                    vocab_size,
                    epsilon,
                )
            },
        }
    }

    /// PAR-121: Initialize stable buffers for batched graph capture
    fn init_batched_graph_buffers(
        &mut self,
        m: usize,
        hidden_dim: u32,
        vocab_size: u32,
    ) -> Result<(), GpuError> {
        let input_size = m * hidden_dim as usize;

        // Allocate or reallocate input buffer
        if self.batched_graph_input_buf.is_none()
            || self
                .batched_graph_input_buf
                .as_ref()
                .map_or(0, trueno_gpu::driver::GpuBuffer::len)
                != input_size
        {
            self.batched_graph_input_buf = Some(GpuBuffer::new(&self.context, input_size)?);
        }

        // Allocate or reallocate positions buffer
        if self.batched_graph_positions_buf.is_none()
            || self
                .batched_graph_positions_buf
                .as_ref()
                .map_or(0, trueno_gpu::driver::GpuBuffer::len)
                != m
        {
            self.batched_graph_positions_buf = Some(GpuBuffer::new(&self.context, m)?);
        }

        // Allocate or reallocate seq_lens buffer
        if self.batched_graph_seq_lens_buf.is_none()
            || self
                .batched_graph_seq_lens_buf
                .as_ref()
                .map_or(0, trueno_gpu::driver::GpuBuffer::len)
                != m
        {
            self.batched_graph_seq_lens_buf = Some(GpuBuffer::new(&self.context, m)?);
        }

        // Ensure workspace logits buffer is allocated for graph capture
        let logits_size = m * vocab_size as usize;
        if self.workspace.logits_buf.is_none()
            || self
                .workspace
                .logits_buf
                .as_ref()
                .map_or(0, trueno_gpu::driver::GpuBuffer::len)
                != logits_size
        {
            self.workspace.logits_buf = Some(GpuBuffer::new(&self.context, logits_size)?);
            // PMAT-088: Logits buffer reallocation invalidates M=1 decode graph
            // (it captured a pointer to the old logits_buf address/size).
            self.clear_decode_graph();
        }

        Ok(())
    }
}
