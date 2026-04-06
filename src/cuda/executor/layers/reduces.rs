// CUDA graph capture with ThreadLocal mode and stream poison recovery (PMAT-374).

impl CudaExecutor {

    /// PAR-054: Try to capture CUDA graph
    /// PMAT-374: If capture fails, ensure stream returns to non-capture state.
    fn try_graph_capture(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        contract_pre_cuda_graph_guard!();
        // Begin graph capture
        // PMAT-374: Use ThreadLocal instead of Global to avoid poisoning
        // the entire CUDA context if capture fails (driver 590.48.01 bug).
        self.stream.begin_capture(CaptureMode::ThreadLocal)?;

        // Run workspace forward pass (all kernels will be captured)
        let capture_result = self.forward_workspace_captured(
            num_layers,
            hidden_dim,
            intermediate_dim,
            vocab_size,
            epsilon,
        );

        // End capture regardless of result — MUST be called to leave capture mode
        let graph = match self.stream.end_capture() {
            Ok(g) => g,
            Err(e) => {
                // PMAT-374: end_capture failed — stream may be stuck in capture mode.
                // Synchronize to force clean state before returning error.
                let _ = self.stream.synchronize();
                return Err(e);
            }
        };

        // Check capture result
        capture_result?;

        // Instantiate the graph
        let graph_exec = graph.instantiate()?;
        self.decode_graph = Some(graph_exec);
        self.decode_token_count = 1;

        // PMAT-283: Create event for non-blocking decode completion tracking
        self.init_decode_event()?;

        if verbose() {
            eprintln!(
                "[PAR-054] ✓ CUDA graph captured successfully ({} layers + LM head)",
                num_layers
            );
        }

        Ok(())
    }

    /// PAR-054: Replay captured graph with updated position
    fn forward_graphed_replay(
        &mut self,
        input: &[f32],
        logits: &mut [f32],
        position: u32,
    ) -> Result<(), GpuError> {
        // CORRECTNESS-013: Stateless GPU mode - force position=0, seq_len=1
        static STATELESS_MODE_REPLAY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_stateless = *STATELESS_MODE_REPLAY.get_or_init(|| {
            std::env::var("STATELESS_GPU")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        // CORRECTNESS-013 FIX: Use async H2D on self.stream instead of copy_from_host
        // (stream 0). CU_STREAM_NON_BLOCKING has no ordering between stream 0 and
        // the non-blocking stream used for graph launch. Same root cause as prefill.
        // CORRECTNESS-013: In stateless mode, always use position=0
        if let Some(ref mut pos_buf) = self.position_buf {
            let pos_to_write = if use_stateless { 0 } else { position };
            // SAFETY: pos_to_write is stack-local, valid until stream sync at line 88
            unsafe { pos_buf.copy_from_host_async(&[pos_to_write], &self.stream)?; }
        }

        // PAR-061-FIX: Update seq_len buffer (seq_len = position + 1)
        // CORRECTNESS-013: In stateless mode, always use seq_len=1
        if let Some(ref mut seq_len_buf) = self.seq_len_buf {
            let seq_len = if use_stateless { 1 } else { position + 1 };
            // SAFETY: seq_len is stack-local, valid until stream sync at line 88
            unsafe { seq_len_buf.copy_from_host_async(&[seq_len], &self.stream)?; }
        }

        // Update input buffer
        if let Some(ref mut input_buf) = self.graph_input_buf {
            // SAFETY: input slice valid until stream sync at line 88
            unsafe { input_buf.copy_from_host_async(input, &self.stream)?; }
        }

        // Launch captured graph
        if let Some(ref graph_exec) = self.decode_graph {
            self.stream.launch_graph(graph_exec)?;
        }

        self.decode_token_count += 1;

        // PMAT-283: Record event for non-blocking completion tracking.
        // The event is queryable via decode_event_complete() for pipelining.
        if let Some(ref event) = self.decode_event {
            self.stream.record_event(event)?;
        }

        // Sync and download (still blocking for now — pipelining requires
        // iteration_scheduler restructuring to overlap token distribution)
        self.stream.synchronize()?;
        if let Some(ref logits_buf) = self.workspace.logits_buf {
            logits_buf.copy_to_host(logits)?;
        }

        Ok(())
    }

    /// PAR-062: GPU-side argmax to eliminate logits transfer bottleneck
    ///
    /// Instead of copying all 152064 logits (600KB) from GPU to CPU for argmax,
    /// this method runs argmax entirely on GPU and only copies the result token ID (4 bytes).
    /// This is a 150,000x reduction in data transfer per token.
    ///
    /// # Algorithm
    ///
    /// Two-pass reduction:
    /// 1. Block-level: Each block finds local (max_val, max_idx) using shared memory
    /// 2. Final: Single block reduces block results to find global argmax
    ///
    /// # Arguments
    ///
    /// * `logits_ptr` - Device pointer to logits (vocab_size f32s)
    /// * `vocab_size` - Number of vocabulary entries (e.g., 152064)
    ///
    /// # Returns
    ///
    /// The token ID with the maximum logit value
    pub fn gpu_argmax(&mut self, logits_ptr: u64, vocab_size: u32) -> Result<u32, GpuError> {
        if logits_ptr == 0 {
            return Err(GpuError::InvalidLaunchConfig(
                "null logits pointer in gpu_argmax".to_string(),
            ));
        }
        // PAR-068: Optimized GPU argmax with pre-allocated buffers
        // Eliminates 3 GPU allocations per token and removes intermediate sync
        let block_size = 256u32;
        let elements_per_block = block_size * 4; // 4 elements per thread
        let num_blocks = (vocab_size + elements_per_block - 1) / elements_per_block;

        // PAR-068: Lazy allocate argmax buffers on first use, reuse thereafter
        if self.argmax_block_vals.is_none() || self.argmax_num_blocks != num_blocks {
            self.argmax_block_vals = Some(GpuBuffer::new(&self.context, num_blocks as usize)?);
            self.argmax_block_idxs = Some(GpuBuffer::new(&self.context, num_blocks as usize)?);
            self.argmax_result = Some(GpuBuffer::new(&self.context, 1)?);
            self.argmax_num_blocks = num_blocks;
        }

        let block_max_vals = self
            .argmax_block_vals
            .as_ref()
            .expect("argmax_block_vals must be initialized");
        let block_max_idxs = self
            .argmax_block_idxs
            .as_ref()
            .expect("argmax_block_idxs must be initialized");
        let result_buf = self
            .argmax_result
            .as_ref()
            .expect("argmax_result must be initialized");

        // Load first-pass kernel module (cached after first use)
        let argmax_kernel_type = KernelType::ArgMax { length: vocab_size };
        let argmax_key = format!("argmax_{}", vocab_size);
        if !self.modules.contains_key(&argmax_key) {
            let ptx = self.kernels.generate_ptx(&argmax_kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(argmax_key.clone(), module);
        }

        // Load second-pass kernel module (cached after first use)
        let final_kernel_type = KernelType::ArgMaxFinal { num_blocks };
        let final_key = format!("argmax_final_{}", num_blocks);
        if !self.modules.contains_key(&final_key) {
            let ptx = self.kernels.generate_ptx(&final_kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(final_key.clone(), module);
        }

        // Prepare kernel arguments
        let kernel_name = self.kernels.kernel_name(&argmax_kernel_type);
        // PAR-068-FIX: Do NOT use .with_shared_mem() - PTX declares static shared memory via .shared directive
        let launch_config = LaunchConfig::grid_2d(num_blocks, 1, block_size, 1);

        let mut input_ptr = logits_ptr;
        let mut block_vals_ptr = block_max_vals.as_ptr();
        let mut block_idxs_ptr = block_max_idxs.as_ptr();
        let mut length_val = vocab_size;

        // Launch first-pass kernel (block-level reduction)
        // SAFETY: Buffers are valid, args match kernel signature
        unsafe {
            let module = self
                .modules
                .get_mut(&argmax_key)
                .expect("argmax module just inserted");
            self.stream.launch_kernel(
                module,
                kernel_name,
                &launch_config,
                &mut [
                    std::ptr::from_mut(&mut input_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut block_vals_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut block_idxs_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut length_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-068: NO intermediate sync - launch both kernels back-to-back
        // The kernels are on the same stream, so execution is serialized

        // Launch second-pass kernel (final reduction)
        let final_kernel_name = self.kernels.kernel_name(&final_kernel_type);
        let final_launch_config = LaunchConfig::grid_2d(1, 1, 256, 1);

        let mut result_ptr = result_buf.as_ptr();
        let mut num_blocks_val = num_blocks;

        // SAFETY: Buffers are valid, args match kernel signature
        unsafe {
            let final_module = self
                .modules
                .get_mut(&final_key)
                .expect("argmax_final module just inserted");
            self.stream.launch_kernel(
                final_module,
                final_kernel_name,
                &final_launch_config,
                &mut [
                    std::ptr::from_mut(&mut block_vals_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut block_idxs_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut result_ptr) as *mut std::ffi::c_void,
                    std::ptr::from_mut(&mut num_blocks_val) as *mut std::ffi::c_void,
                ],
            )?;
        }

        // PAR-068: Single sync after both kernels complete
        self.stream.synchronize()?;
        let mut result = [0u32];
        result_buf.copy_to_host(&mut result)?;

        // CORRECTNESS-005: Debug GPU argmax result
        static ARGMAX_DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ARGMAX_DEBUG.get_or_init(|| {
            std::env::var("GPU_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
        }) {
            eprintln!(
                "[CORRECTNESS-005] GPU argmax: token_id={}, vocab_size={}",
                result[0], vocab_size
            );
        }

        Ok(result[0])
    }

    /// PMAT-045: Batched GPU argmax — process M logit vectors with ONE sync.
    ///
    /// Five-Whys root cause: c=4 decode regression (140→50 tok/s per request).
    /// Why? Sequential `gpu_argmax` calls: M syncs × ~0.3ms = 1.2ms/token at c=4.
    /// Why? Each `gpu_argmax` calls `stream.synchronize()` + `copy_to_host`.
    /// Why? API designed for single-sequence M=1 path.
    /// Fix: Launch all 2×M kernel passes back-to-back, then ONE sync + ONE memcpy.
    /// Same-stream kernels serialize naturally — intermediate buffers safely reused.
    pub fn batched_gpu_argmax(
        &mut self,
        logits_base_ptr: u64,
        vocab_size: u32,
        m: usize,
    ) -> Result<Vec<u32>, GpuError> {
        if m == 0 {
            return Ok(Vec::new());
        }
        if m == 1 {
            return Ok(vec![self.gpu_argmax(logits_base_ptr, vocab_size)?]);
        }

        let block_size = 256u32;
        let elements_per_block = block_size * 4;
        let num_blocks = (vocab_size + elements_per_block - 1) / elements_per_block;

        // Ensure block-level buffers (shared across M seqs — safe on same stream)
        if self.argmax_block_vals.is_none() || self.argmax_num_blocks != num_blocks {
            self.argmax_block_vals = Some(GpuBuffer::new(&self.context, num_blocks as usize)?);
            self.argmax_block_idxs = Some(GpuBuffer::new(&self.context, num_blocks as usize)?);
            self.argmax_result = Some(GpuBuffer::new(&self.context, 1)?);
            self.argmax_num_blocks = num_blocks;
        }

        // Allocate/grow batched result buffer
        if self.batched_argmax_results.is_none() || self.batched_argmax_results_cap < m {
            self.batched_argmax_results = Some(GpuBuffer::new(&self.context, m)?);
            self.batched_argmax_results_cap = m;
        }

        let block_max_vals_ptr = self.argmax_block_vals.as_ref().unwrap().as_ptr();
        let block_max_idxs_ptr = self.argmax_block_idxs.as_ref().unwrap().as_ptr();
        let batched_results_base = self.batched_argmax_results.as_ref().unwrap().as_ptr();

        // Ensure kernels are compiled (cached after first use)
        let argmax_kernel_type = KernelType::ArgMax { length: vocab_size };
        let argmax_key = format!("argmax_{}", vocab_size);
        if !self.modules.contains_key(&argmax_key) {
            let ptx = self.kernels.generate_ptx(&argmax_kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(argmax_key.clone(), module);
        }
        let final_kernel_type = KernelType::ArgMaxFinal { num_blocks };
        let final_key = format!("argmax_final_{}", num_blocks);
        if !self.modules.contains_key(&final_key) {
            let ptx = self.kernels.generate_ptx(&final_kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(final_key.clone(), module);
        }

        let kernel_name = self.kernels.kernel_name(&argmax_kernel_type);
        let final_kernel_name = self.kernels.kernel_name(&final_kernel_type);
        let launch_config = LaunchConfig::grid_2d(num_blocks, 1, block_size, 1);
        let final_launch_config = LaunchConfig::grid_2d(1, 1, 256, 1);

        // Launch all 2*M kernels back-to-back — no sync between sequences.
        // Same-stream kernels serialize: pass2[i] completes before pass1[i+1],
        // so block_max_vals/idxs buffers are safely reused across sequences.
        for seq_idx in 0..m {
            let v_offset = seq_idx * vocab_size as usize;
            let mut input_ptr =
                logits_base_ptr + (v_offset * std::mem::size_of::<f32>()) as u64;
            let mut result_ptr =
                batched_results_base + (seq_idx * std::mem::size_of::<u32>()) as u64;

            let mut bv_ptr = block_max_vals_ptr;
            let mut bi_ptr = block_max_idxs_ptr;
            let mut length_val = vocab_size;

            // Pass 1: block-level reduction
            unsafe {
                let module = self
                    .modules
                    .get_mut(&argmax_key)
                    .expect("argmax module just compiled");
                self.stream.launch_kernel(
                    module,
                    kernel_name,
                    &launch_config,
                    &mut [
                        std::ptr::from_mut(&mut input_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut bv_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut bi_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut length_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }

            // Pass 2: final reduction → writes to batched_results[seq_idx]
            let mut nb_val = num_blocks;
            unsafe {
                let final_module = self
                    .modules
                    .get_mut(&final_key)
                    .expect("argmax_final module just compiled");
                self.stream.launch_kernel(
                    final_module,
                    final_kernel_name,
                    &final_launch_config,
                    &mut [
                        std::ptr::from_mut(&mut bv_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut bi_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut result_ptr) as *mut std::ffi::c_void,
                        std::ptr::from_mut(&mut nb_val) as *mut std::ffi::c_void,
                    ],
                )?;
            }
        }

        // ONE sync after all 2*M kernels
        self.stream.synchronize()?;

        // ONE memcpy for all M results
        // PMAT-088c: Buffer may be over-sized from high-water-mark allocation
        // (e.g., allocated for M=4 but current batch M=2). Create exact-M view.
        let mut results = vec![0u32; m];
        let results_ptr = self.batched_argmax_results.as_ref().unwrap().as_ptr();
        let results_view = unsafe { GpuBuffer::<u32>::from_raw_parts(results_ptr, m) };
        results_view.copy_to_host(&mut results[..m])?;
        std::mem::forget(results_view);

        Ok(results)
    }

    /// PAR-062: Forward pass with GPU-side argmax returning token ID directly
    ///
    /// Like `forward_graphed_replay` but uses GPU argmax instead of downloading all logits.
    /// Reduces data transfer from 600KB to 4 bytes per token.
    ///
    /// # Performance Target
    ///
    /// - Before: ~3ms logits transfer per token on PCIe
    /// - After: ~0.001ms token ID transfer
    /// - Expected speedup: ~1.2x overall throughput
    pub fn forward_graphed_replay_to_token_id(
        &mut self,
        input: &[f32],
        position: u32,
        vocab_size: u32,
    ) -> Result<u32, GpuError> {
        // PAR-083: Sub-phase timing for Five-Whys decode gap diagnosis
        static DECODE_TIMING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let timing = *DECODE_TIMING.get_or_init(|| {
            std::env::var("DECODE_TIMING")
                .map(|v| v == "1")
                .unwrap_or(false)
        });
        let t_start = if timing {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // PAR-072: Use ASYNC H2D copies to eliminate blocking overhead
        // Root cause: cuMemcpyHtoD has ~10-30µs overhead per call
        // Fix: Use cuMemcpyHtoDAsync on the same stream as graph launch

        // Update position buffer (async memcpy on same stream)
        if let Some(ref mut pos_buf) = self.position_buf {
            // SAFETY: position is stack-allocated and we synchronize before returning
            unsafe {
                pos_buf.copy_from_host_async(&[position], &self.stream)?;
            }
        }

        // PAR-061-FIX: Update seq_len buffer (seq_len = position + 1)
        let seq_len = position + 1;
        if let Some(ref mut seq_len_buf) = self.seq_len_buf {
            // SAFETY: seq_len is stack-allocated and we synchronize before returning
            unsafe {
                seq_len_buf.copy_from_host_async(&[seq_len], &self.stream)?;
            }
        }

        // Update input buffer (async - largest copy, ~14KB for Qwen 7B)
        if let Some(ref mut input_buf) = self.graph_input_buf {
            // SAFETY: input slice is valid for the duration of this function
            // and we synchronize in gpu_argmax before returning
            unsafe {
                input_buf.copy_from_host_async(input, &self.stream)?;
            }
        }
        let t_h2d = t_start.map(|_| std::time::Instant::now());

        // Launch captured graph (all H2D copies are ordered before this on same stream)
        if let Some(ref graph_exec) = self.decode_graph {
            self.stream.launch_graph(graph_exec)?;
        }
        let t_graph = t_start.map(|_| std::time::Instant::now());

        self.decode_token_count += 1;

        // PAR-068: GPU argmax instead of downloading 600KB logits
        // This reduces D2H transfer from 600KB to 4 bytes per token
        let logits_ptr = self
            .workspace
            .logits_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidParameter("logits_buf not allocated".into()))?
            .as_ptr();

        // CORRECTNESS-004: Debug graph-replayed logits and compare argmax
        static GPU_DEBUG_FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let debug_enabled = *GPU_DEBUG_FLAG.get_or_init(|| {
            std::env::var("GPU_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        if debug_enabled {
            self.stream.synchronize()?;
            // Download ALL logits to compute CPU argmax for comparison
            let mut all_logits = vec![0.0f32; vocab_size as usize];
            // SAFETY: Pointer valid from allocation, length verified, used within scope
            // SAFETY: Pointer valid from allocation, length verified, used within scope
            let debug_view =
                unsafe { GpuBuffer::<f32>::from_raw_parts(logits_ptr, vocab_size as usize) };
            debug_view.copy_to_host(&mut all_logits)?;
            std::mem::forget(debug_view);

            // CPU argmax
            let (cpu_argmax_idx, cpu_argmax_val) = all_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .expect("CUDA operation failed");

            eprintln!(
                "[CORRECTNESS-004] Graph logits[0..20]: {:?}",
                all_logits.get(..20).expect("logits buffer has at least 20 elements")
            );
            eprintln!(
                "[CORRECTNESS-004] GPU argmax: idx={}, val={}",
                cpu_argmax_idx, cpu_argmax_val
            );

            // Compare against CPU's expected top tokens: 19 ("4"), 17 ("2"), 785 (" The")
            eprintln!(
                "[CORRECTNESS-004] GPU logit for token 19 ('4'): {}",
                all_logits.get(19).unwrap_or(&f32::NAN)
            );
            eprintln!(
                "[CORRECTNESS-004] GPU logit for token 17 ('2'): {}",
                all_logits.get(17).unwrap_or(&f32::NAN)
            );
            eprintln!(
                "[CORRECTNESS-004] GPU logit for token 785: {}",
                all_logits.get(785).unwrap_or(&f32::NAN)
            );

            // Top 5 GPU logits
            let mut top5: Vec<(usize, f32)> = all_logits.iter().copied().enumerate().collect();
            top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            top5.truncate(10);
            eprintln!("[CORRECTNESS-004] GPU top10 logits: {:?}", top5);
        }

        let t_pre_argmax = t_start.map(|_| std::time::Instant::now());
        let gpu_result = self.gpu_argmax(logits_ptr, vocab_size)?;
        let t_done = t_start.map(|_| std::time::Instant::now());

        if debug_enabled {
            eprintln!("[CORRECTNESS-004] GPU argmax result: {}", gpu_result);
        }

        // PAR-083: Sub-phase timing output
        if let (Some(ts), Some(th), Some(tg), Some(ta), Some(td)) =
            (t_start, t_h2d, t_graph, t_pre_argmax, t_done)
        {
            let h2d_us = th.duration_since(ts).as_micros();
            let graph_us = tg.duration_since(th).as_micros();
            let wait_us = ta.duration_since(tg).as_micros();
            let argmax_us = td.duration_since(ta).as_micros();
            let total_us = td.duration_since(ts).as_micros();
            eprintln!(
                "[GRAPH-TIMING] h2d={}µs launch={}µs wait={}µs argmax+sync={}µs total={}µs",
                h2d_us, graph_us, wait_us, argmax_us, total_us
            );
        }

        Ok(gpu_result)
    }
}
