
impl CudaExecutor {

    /// PAR-054: Forward pass for graph capture (uses pre-allocated workspace)
    ///
    /// # Safety
    ///
    /// This function must only be called while stream capture is active.
    /// All output buffers (workspace.logits_buf) must be pre-allocated before capture.
    fn forward_workspace_captured(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // Layer 0: input from graph_input_buf
        // PAR-070: Position is read from position_buf in indirect mode (graph capture)
        // The position parameter here is ignored since position_buf.is_some() triggers indirect mode
        if num_layers > 0 {
            let input_ptr = self
                .graph_input_buf
                .as_ref()
                .expect("graph_input_buf must be initialized")
                .as_ptr();
            let input_len = self
                .graph_input_buf
                .as_ref()
                .expect("graph_input_buf must be initialized")
                .len();
            // SAFETY: Memory safety ensured by bounds checking and alignment
            // SAFETY: Pointer valid from allocation, length verified, used within scope
            let input_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(input_ptr, input_len) };
            let layer_weights = self.indexed_layer_weights[0].clone();
            // PAR-054: Use capture-safe version (no debug sync/copy_to_host)
            self.transformer_layer_workspace_for_capture(
                &input_buf,
                0,
                &layer_weights,
                hidden_dim,
                intermediate_dim,
                epsilon,
                0, // PAR-070: Ignored in graph capture mode (uses position_buf)
            )?;
            std::mem::forget(input_buf);
        }

        // Layers 1+: input from hidden_buf2
        for layer_idx in 1..num_layers {
            let layer_weights = self.indexed_layer_weights[layer_idx].clone();
            let buf_ptr = self
                .workspace
                .hidden_buf2
                .as_ref()
                .expect("hidden_buf2 must be initialized")
                .as_ptr();
            let buf_len = self
                .workspace
                .hidden_buf2
                .as_ref()
                .expect("hidden_buf2 must be initialized")
                .len();
            // SAFETY: Memory safety ensured by bounds checking and alignment
            // SAFETY: Pointer valid from allocation, length verified, used within scope
            let input_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(buf_ptr, buf_len) };
            // PAR-054: Use capture-safe version (no debug sync/copy_to_host)
            self.transformer_layer_workspace_for_capture(
                &input_buf,
                layer_idx,
                &layer_weights,
                hidden_dim,
                intermediate_dim,
                epsilon,
                0, // PAR-070: Ignored in graph capture mode (uses position_buf)
            )?;
            std::mem::forget(input_buf);
        }

        // Output RMSNorm - PAR-054: Use pre-allocated normed_hidden_buf
        let output_norm_gamma = self.rmsnorm_cache.get("output_norm.gamma").ok_or_else(|| {
            GpuError::InvalidLaunchConfig("PAR-054: output_norm not cached".to_string())
        })?;
        let output_gamma_ptr = output_norm_gamma.as_ptr();
        let output_gamma_len = output_norm_gamma.len();

        let hidden_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .expect("hidden_buf2 must be initialized")
            .as_ptr();
        let hidden_len = self
            .workspace
            .hidden_buf2
            .as_ref()
            .expect("hidden_buf2 must be initialized")
            .len();
        // SAFETY: Memory safety ensured by bounds checking and alignment
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let hidden_input = unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_ptr, hidden_len) };

        // PAR-054: Write to pre-allocated normed_hidden_buf (no allocation during capture)
        let normed_ptr = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .expect("normed_hidden_buf must be initialized")
            .as_ptr();
        let normed_len = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .expect("normed_hidden_buf must be initialized")
            .len();
        // SAFETY: Memory safety ensured by bounds checking and alignment
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let normed_output = unsafe { GpuBuffer::<f32>::from_raw_parts(normed_ptr, normed_len) };

        // GQA-DEBUG: Print hidden before output norm
        static GPU_DEBUG_FLAG2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let debug_enabled2 = *GPU_DEBUG_FLAG2.get_or_init(|| {
            std::env::var("GPU_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
        });
        if debug_enabled2 {
            self.stream.synchronize()?;
            let mut hidden_check = vec![0.0f32; hidden_len.min(896)];
            hidden_input.copy_to_host(&mut hidden_check)?;
            let sum: f32 = hidden_check.iter().sum();
            let sq_sum: f32 = hidden_check.iter().map(|x| x * x).sum();
            let rms = (sq_sum / hidden_check.len() as f32).sqrt();
            eprintln!(
                "[GQA-DEBUG] Hidden before output_norm: first 5 = {:?}, sum={:.4}, rms={:.4}",
                &hidden_check[..5.min(hidden_check.len())],
                sum,
                rms
            );
        }

        self.rmsnorm_ptr_into(
            &hidden_input,
            output_gamma_ptr,
            output_gamma_len,
            &normed_output,
            hidden_dim,
            epsilon,
        )?;

        // GQA-DEBUG: Print normed hidden
        if debug_enabled2 {
            self.stream.synchronize()?;
            let mut normed_check = vec![0.0f32; normed_len.min(896)];
            normed_output.copy_to_host(&mut normed_check)?;
            let sum: f32 = normed_check.iter().sum();
            let sq_sum: f32 = normed_check.iter().map(|x| x * x).sum();
            let rms = (sq_sum / normed_check.len() as f32).sqrt();
            eprintln!(
                "[GQA-DEBUG] Normed hidden: first 5 = {:?}, sum={:.4}, rms={:.4}",
                &normed_check[..5.min(normed_check.len())],
                sum,
                rms
            );
        }

        std::mem::forget(hidden_input);
        std::mem::forget(normed_output);

        // LM head projection - PAR-054: Use pre-allocated logits_buf
        // PAR-058: Use correct kernel based on LM head quantization type
        let logits_ptr = self
            .workspace
            .logits_buf
            .as_ref()
            .expect("logits_buf must be initialized")
            .as_ptr();
        let logits_len = self
            .workspace
            .logits_buf
            .as_ref()
            .expect("logits_buf must be initialized")
            .len();
        // SAFETY: Memory safety ensured by bounds checking and alignment
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let logits_output = unsafe { GpuBuffer::<f32>::from_raw_parts(logits_ptr, logits_len) };

        let normed_ptr = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .expect("normed_hidden_buf must be initialized")
            .as_ptr();
        let normed_len = self
            .workspace
            .normed_hidden_buf
            .as_ref()
            .expect("normed_hidden_buf must be initialized")
            .len();
        // SAFETY: Memory safety ensured by bounds checking and alignment
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let normed_input = unsafe { GpuBuffer::<f32>::from_raw_parts(normed_ptr, normed_len) };

        // PMAT-027: Invalidate Q8 cache — LM head input is normed_input (different from layer GEMVs).
        self.q8_activation_valid = false;

        // PAR-058: Dispatch to correct kernel based on LM head quant type
        // Validate qtype against actual size - GGUF metadata can lie!
        let lm_head_qtype =
            WeightQuantType::from_size(self.lm_head_len, vocab_size as usize, hidden_dim as usize)
                .unwrap_or(self.lm_head_qtype);

        // Log if we overrode the type
        if lm_head_qtype != self.lm_head_qtype {
            eprintln!(
                "[PAR-058] LM head qtype override: {:?} -> {:?} (size-based detection)",
                self.lm_head_qtype, lm_head_qtype
            );
        }

        self.gemv_dispatch(
            lm_head_qtype,
            self.lm_head_ptr,
            &normed_input,
            &logits_output,
            vocab_size,
            hidden_dim,
        )?;

        // PAR-064-FIX: Add LM head bias after GEMV (if present)
        // Without this, GPU inference produces incorrect token predictions
        if self.lm_head_bias_ptr != 0 && self.lm_head_bias_len > 0 {
            // Create non-owning buffer wrapper from device pointer
            // SAFETY: bias_ptr is valid device memory owned by bias_cache
            let bias_buf = unsafe {
                GpuBuffer::<f32>::from_raw_parts(self.lm_head_bias_ptr, self.lm_head_bias_len)
            };

            // Add bias in-place: logits = logits + bias
            self.residual_add_into(&logits_output, &bias_buf, &logits_output, vocab_size)?;

            // Prevent Drop from freeing borrowed memory
            std::mem::forget(bias_buf);
        }

        // GQA-DEBUG: Print final logits and top token
        if debug_enabled2 {
            self.stream.synchronize()?;
            let mut logits_check = vec![0.0f32; logits_len.min(100)];
            logits_output.copy_to_host(&mut logits_check)?;
            eprintln!(
                "[GQA-DEBUG] Final logits: first 10 = {:?}",
                &logits_check[..10.min(logits_check.len())]
            );
            // Find argmax
            let mut full_logits = vec![0.0f32; logits_len];
            logits_output.copy_to_host(&mut full_logits)?;
            let argmax = full_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i);
            eprintln!(
                "[GQA-DEBUG] Argmax token = {}, logit = {:.4}",
                argmax, full_logits[argmax]
            );
        }

        std::mem::forget(normed_input);
        std::mem::forget(logits_output);

        Ok(())
    }
}

include!("graphed_capture.rs");
include!("preload_modules.rs");
include!("par-062.rs");
