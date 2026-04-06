impl CudaExecutor {

    /// PAR-121: Pre-load kernel modules for batched graph capture
    fn preload_modules_for_batched_capture(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
    ) -> Result<(), GpuError> {
        // Reuse existing preload_modules_for_capture which loads all needed kernels
        self.preload_modules_for_capture(num_layers, hidden_dim, intermediate_dim, vocab_size)
    }

    /// PAR-121: Try to capture batched forward pass into CUDA graph
    fn try_batched_graph_capture(
        &mut self,
        m: usize,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
        positions: &[u32], // PMAT-285: pass real positions for realistic seq_lens
    ) -> Result<(), GpuError> {
        // PMAT-045: Pre-upload static data BEFORE capture.
        // cuMemcpyHtoD is not capturable — all host-to-device copies must happen
        // outside the capture region. Guards inside batched ops skip copy_from_host
        // when is_capturing == true.
        self.pre_upload_batched_state_for_capture(m, positions)?;

        // PMAT-045: Set capture flag so copy_from_host calls are skipped
        self.is_capturing = true;

        // Begin graph capture
        self.stream.begin_capture(CaptureMode::Global)?;

        // Run batched forward pass (all kernels will be captured)
        let capture_result = self.forward_batched_captured(
            m,
            num_layers,
            hidden_dim,
            intermediate_dim,
            vocab_size,
            epsilon,
        );

        // End capture regardless of result
        let graph = self.stream.end_capture()?;
        self.is_capturing = false;

        // Check capture result
        capture_result?;

        // Instantiate the graph
        let graph_exec = graph.instantiate()?;
        self.batched_decode_graphs.insert(m, graph_exec);

        Ok(())
    }

    /// PMAT-045: Pre-upload static batched state before graph capture.
    /// Uploads k_ptrs, v_ptrs, seq_lens, and positions to device buffers.
    /// These are the same buffers the graph will reference during replay.
    fn pre_upload_batched_state_for_capture(
        &mut self,
        m: usize,
        positions: &[u32], // PMAT-285: real positions for realistic seq_lens
    ) -> Result<(), GpuError> {
        // PMAT-285: Upload REAL positions (not dummies) to workspace.positions_buf.
        // H-CB11 root cause: dummy positions=0 caused suboptimal kernel behavior
        // during capture, resulting in −32% regression on replay.
        let positions_to_upload: Vec<u32> = if positions.len() >= m {
            positions[..m].to_vec()
        } else {
            // Fallback: use position 32 (realistic mid-generation) instead of 0
            vec![32; m]
        };
        if let Some(ref mut pos_buf) = self.workspace.positions_buf {
            if pos_buf.len() >= m {
                let mut wrapper =
                    unsafe { GpuBuffer::<u32>::from_raw_parts(pos_buf.as_ptr(), m) };
                wrapper.copy_from_host(&positions_to_upload)?;
                std::mem::forget(wrapper);
            }
        }

        // GH-141: Per-layer k_ptrs/v_ptrs are now pre-populated in
        // init_batched_kv_cache_gpu (batched_k_ptrs_per_layer). During capture,
        // batch.rs reads from these per-layer buffers instead of the shared
        // batched_k_ptrs. No need to pre-upload shared buffer here.

        // PMAT-285: Upload REAL seq_lens (position + 1) instead of dummy vec![1; m].
        // H-CB11 root cause: seq_lens=1 during capture caused attention kernels
        // to see only 1 KV entry, producing wrong compute pattern for graph.
        let real_seq_lens: Vec<u32> = positions_to_upload.iter().map(|&p| p + 1).collect();
        if let Some(ref mut seq_buf) = self.batched_seq_lens_gpu {
            if seq_buf.len() >= m {
                let mut wrapper =
                    unsafe { GpuBuffer::<u32>::from_raw_parts(seq_buf.as_ptr(), m) };
                wrapper.copy_from_host(&real_seq_lens)?;
                std::mem::forget(wrapper);
            }
        }

        Ok(())
    }

    /// PAR-121: Forward pass for batched graph capture (uses stable buffers)
    fn forward_batched_captured(
        &mut self,
        m: usize,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // Use stable input buffer
        let input_ptr = self
            .batched_graph_input_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(
                    "PAR-121: batched_graph_input_buf missing".to_string(),
                )
            })?
            .as_ptr();
        let input_len = m * hidden_dim as usize;
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let input_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(input_ptr, input_len) };

        // Get workspace buffer pointers
        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-121: hidden_buf2 missing".to_string())
            })?
            .as_ptr();
        let hidden_buf2_len = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-121: hidden_buf2 missing".to_string())
            })?
            .len();

        // Use stable positions buffer for RoPE and attention
        let positions_ptr = self
            .batched_graph_positions_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig(
                    "PAR-121: batched_graph_positions_buf missing".to_string(),
                )
            })?
            .as_ptr();

        // Process all layers with batched GEMV
        for layer_idx in 0..num_layers {
            if layer_idx >= self.indexed_layer_weights.len() {
                std::mem::forget(input_buf);
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PAR-121: Layer {} weights not indexed",
                    layer_idx
                )));
            }
            let layer_weights = self.get_indexed_layer(layer_idx).clone();

            let layer_input_buf = if layer_idx == 0 {
                None
            } else {
                // SAFETY: Pointer valid from allocation, length verified, used within scope
                Some(unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) })
            };

            let layer_input = match &layer_input_buf {
                Some(buf) => buf,
                None => &input_buf,
            };

            // Call batched layer with positions from stable buffer
            self.transformer_layer_batched_captured(
                layer_input,
                layer_idx,
                &layer_weights,
                m as u32,
                positions_ptr,
                hidden_dim,
                intermediate_dim,
                epsilon,
            )?;

            if let Some(buf) = layer_input_buf {
                std::mem::forget(buf);
            }
        }

        // Output norm
        let output_norm_buf = self.rmsnorm_cache.get("output_norm.gamma").ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-121: output_norm not cached".to_string())
        })?;
        let output_norm_ptr = output_norm_buf.as_ptr();
        let output_norm_len = hidden_dim as usize;

        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-121: hidden_buf2 missing".to_string())
            })?
            .as_ptr();
        let hidden_buf2_len = m * hidden_dim as usize;
        let normed_hidden_ptr = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-121: normed_hidden_buf missing".to_string())
            })?
            .as_ptr();
        let normed_hidden_len = m * hidden_dim as usize;

        // SAFETY: Raw pointer from valid allocation, length verified by caller
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let hidden_buf2 =
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_buf2_ptr, hidden_buf2_len) };
        // SAFETY: Raw pointer from valid allocation, length verified by caller
        // SAFETY: Pointer valid from allocation, length verified, used within scope
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

        // LM head projection
        let lm_head_ptr = self.lm_head_ptr;
        let lm_head_qtype = self.lm_head_qtype;

        // Get logits buffer pointer to avoid borrow conflict
        let logits_buf_ptr = self
            .workspace
            .logits_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-121: logits_buf missing".to_string())
            })?
            .as_ptr();
        let logits_buf_len = m * vocab_size as usize;

        // Create wrapper for logits buffer
        // SAFETY: Unsafe operation with validated invariants
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let logits_buf =
            unsafe { GpuBuffer::<f32>::from_raw_parts(logits_buf_ptr, logits_buf_len) };

        // SAFETY: Unsafe operation with validated invariants
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let normed_hidden_buf_wrapper =
            unsafe { GpuBuffer::<f32>::from_raw_parts(normed_hidden_ptr, normed_hidden_len) };

        self.batched_gemv_with_fallback(
            lm_head_qtype,
            lm_head_ptr,
            &normed_hidden_buf_wrapper,
            &logits_buf,
            normed_hidden_ptr,
            logits_buf_ptr,
            m as u32,
            vocab_size,
            hidden_dim,
        )?;

        std::mem::forget(normed_hidden_buf_wrapper);
        std::mem::forget(logits_buf);
        std::mem::forget(input_buf);

        Ok(())
    }

    /// PAR-121: Batched transformer layer using positions from device pointer
    #[allow(clippy::too_many_arguments)]
    fn transformer_layer_batched_captured(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &ValidatedLayerWeights,
        m: u32,
        _positions_ptr: u64,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // Uses batched version with positions read back from device
        // Direct device-side position access planned for PAR-200

        // For graph capture, we need to avoid host-device transfers
        // The positions are already on device, kernels can read from there
        // PMAT-285: Use real positions for realistic seq_lens during capture
        let dummy_positions: Vec<u32> = (0..m as usize).map(|i| i as u32).collect();

        // PMAT-291: Use graph dispatch during capture if enabled (default ON).
        // The tensor graph path produces the same kernel sequence with ~40% fewer
        // logical nodes (392 vs 654), reducing CUDA graph management overhead.
        if self.use_graph_dispatch() {
            self.transformer_layer_batched_graph(
                input,
                layer_idx,
                layer_weights,
                m,
                &dummy_positions,
                hidden_dim,
                intermediate_dim,
                epsilon,
            )
        } else {
            self.transformer_layer_batched(
                input,
                layer_idx,
                layer_weights,
                m,
                &dummy_positions,
                hidden_dim,
                intermediate_dim,
                epsilon,
            )
        }
    }

    /// PAR-121: Replay captured batched graph with updated inputs
    fn forward_batched_graphed_replay(
        &mut self,
        inputs: &[f32],
        positions: &[u32],
        m: usize,
        vocab_size: u32,
    ) -> Result<Vec<u32>, GpuError> {
        // PMAT-088b: Async H2D copies on self.stream — no sync stalls before graph launch.
        // Prior PMAT-075 used sync copy_from_host (stream 0), adding 2.8ms overhead.
        // All host data (&inputs, &positions, seq_lens Vec) lives until stream.synchronize().
        if let Some(ref mut input_buf) = self.batched_graph_input_buf {
            unsafe { input_buf.copy_from_host_async(inputs, &self.stream)?; }
        }

        if let Some(ref mut pos_buf) = self.batched_graph_positions_buf {
            unsafe { pos_buf.copy_from_host_async(positions, &self.stream)?; }
        }

        let seq_lens: Vec<u32> = positions
            .iter()
            .enumerate()
            .map(|(seq_idx, &p)| {
                // PMAT-076: Zero seq_lens for done slots in graph replay path.
                if seq_idx < self.batched_done_mask.len() && self.batched_done_mask[seq_idx] {
                    0
                } else {
                    p + 1
                }
            })
            .collect();
        if let Some(ref mut len_buf) = self.batched_graph_seq_lens_buf {
            unsafe { len_buf.copy_from_host_async(&seq_lens, &self.stream)?; }
        }

        // Also update the batched KV cache seq_lens for attention
        if let Some(ref mut seq_lens_gpu) = self.batched_seq_lens_gpu {
            // PMAT-088b: Use exact-M view for async copy (buffer may be high-water-mark sized)
            let ptr = seq_lens_gpu.as_ptr();
            let mut view = unsafe { GpuBuffer::<u32>::from_raw_parts(ptr, m) };
            unsafe { view.copy_from_host_async(&seq_lens, &self.stream)?; }
            std::mem::forget(view);
        }

        // PMAT-045: Update workspace.positions_buf (read by RoPE kernels in graph)
        if let Some(ref mut pos_buf) = self.workspace.positions_buf {
            if pos_buf.len() >= m {
                let mut wrapper =
                    unsafe { GpuBuffer::<u32>::from_raw_parts(pos_buf.as_ptr(), m) };
                unsafe { wrapper.copy_from_host_async(positions, &self.stream)?; }
                std::mem::forget(wrapper);
            }
        }

        // PMAT-045: Update CPU-side batched_kv_lengths for attention seq_lens computation.
        // The graph's attention kernel reads from batched_seq_lens_gpu (already updated above),
        // but we also need CPU-side tracking for the next step's seq_lens calculation.
        for (seq_idx, &pos) in positions.iter().enumerate() {
            if seq_idx < self.batched_kv_lengths.len() {
                self.batched_kv_lengths[seq_idx] = pos as usize + 1;
            }
        }

        // Launch the captured graph
        if let Some(graph_exec) = self.batched_decode_graphs.get(&m) {
            graph_exec.launch(self.stream.raw())?;
        } else {
            return Err(GpuError::InvalidLaunchConfig(format!(
                "PAR-121: No captured graph for M={}",
                m
            )));
        }

        // Get token IDs from logits
        self.stream.synchronize()?;
        self.batched_argmax_from_logits(m, vocab_size)
    }

    /// PAR-121: Extract token IDs from batched logits using GPU argmax
    /// PMAT-045: Batched argmax — ONE sync for all M sequences instead of M syncs.
    fn batched_argmax_from_logits(
        &mut self,
        m: usize,
        vocab_size: u32,
    ) -> Result<Vec<u32>, GpuError> {
        let logits_base_ptr = self
            .workspace
            .logits_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-121: logits_buf missing".to_string())
            })?
            .as_ptr();

        self.batched_gpu_argmax(logits_base_ptr, vocab_size, m)
    }
}
