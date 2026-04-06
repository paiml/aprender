impl CudaExecutor {
    /// PAR-023: Run ALL transformer layers GPU-resident (minimal syncs)
    ///
    /// Chains all layers on GPU, only syncing at the very end.
    /// Requires RMSNorm weights pre-cached via `preload_rmsnorm_weights()`.
    ///
    /// # Sync Count
    ///
    /// - Input upload: 1 sync
    /// - Per layer: 0 syncs (attention has internal D2D)
    /// - Output download: 1 sync
    /// - Total: ~2 syncs vs 22 syncs (per-layer) or 176 syncs (original)
    ///
    /// # Arguments
    ///
    /// * `input` - Embedding input [hidden_dim]
    /// * `output` - Output buffer [hidden_dim]
    /// * `num_layers` - Number of transformer layers
    /// * `hidden_dim` - Hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `epsilon` - RMSNorm epsilon
    #[allow(clippy::too_many_arguments)]
    pub fn forward_all_layers_gpu(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        position: u32, // PAR-070: Explicit position for RoPE and KV cache
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // 1. Validate all RMSNorm weights are cached
        for layer_idx in 0..num_layers {
            let attn_name = format!("blk.{}.attn_norm.gamma", layer_idx);
            let ffn_name = format!("blk.{}.ffn_norm.gamma", layer_idx);
            if !self.rmsnorm_cache.contains_key(&attn_name) {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: attn_norm not cached for layer {}",
                    layer_idx
                )));
            }
            if !self.rmsnorm_cache.contains_key(&ffn_name) {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: ffn_norm not cached for layer {}",
                    layer_idx
                )));
            }
        }

        // 2. Collect all cache key names (avoids repeated string allocs in loop)
        let layer_keys: Vec<(String, String)> = (0..num_layers)
            .map(|i| {
                (
                    format!("blk.{}.attn_norm.gamma", i),
                    format!("blk.{}.ffn_norm.gamma", i),
                )
            })
            .collect();

        // 3. Upload input embedding - sync point #1
        // PAR-044: Check if we can use zero-allocation workspace path
        let use_workspace = self.has_workspace()
            && self.has_indexed_weights()
            && self.indexed_layer_weights.len() == num_layers;

        let mut hidden_gpu = GpuBuffer::from_host(&self.context, input)?;

        // 4. Chain all transformer layers (no intermediate syncs)
        // PAR-044: Use workspace path for zero-allocation forward (fastest)
        // PAR-043: Use indexed path if weights are pre-indexed (10x faster per-token)
        // PAR-044 FIX: Track which buffer has output to avoid unnecessary D2D copy
        // PAR-044: Workspace path enabled - confirmed same performance as indexed path
        // See five-whys-gpu-performance-gap for analysis
        let mut workspace_used = false;
        if use_workspace {
            // PAR-044: Zero-allocation path - workspace buffers + indexed weights
            // Eliminates ~288 buffer allocations per token
            workspace_used = true;

            // Layer 0: input from external hidden_gpu
            if num_layers > 0 {
                let layer_weights = self.indexed_layer_weights[0].clone();
                self.transformer_layer_workspace(
                    &hidden_gpu,
                    0,
                    &layer_weights,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                    position,
                )?;
            }

            // Layers 1+: input from hidden_buf2 (output of previous layer)
            // Use raw pointer to avoid borrow conflict with &mut self
            for layer_idx in 1..num_layers {
                let layer_weights = self.indexed_layer_weights[layer_idx].clone();
                // SAFETY: hidden_buf2 is initialized and remains valid throughout
                // We get ptr/len before the mutable borrow, avoiding conflict
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
                // Create temporary non-owning view of hidden_buf2
                // SAFETY: Memory safety ensured by bounds checking and alignment
                // SAFETY: Pointer valid from allocation, length verified, used within scope
                let input_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(buf_ptr, buf_len) };
                self.transformer_layer_workspace(
                    &input_buf,
                    layer_idx,
                    &layer_weights,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                    position,
                )?;
                // Prevent Drop from freeing the borrowed memory
                std::mem::forget(input_buf);
            }

            // PAR-044 FIX: Output is in hidden_buf2, use it directly
            // (removed unnecessary copy_from_buffer - saves one D2D copy per token)
        } else if self.has_indexed_weights() && self.indexed_layer_weights.len() == num_layers {
            // PAR-043: Fast path - O(1) weight access, no string formatting
            for layer_idx in 0..num_layers {
                // Clone the layer weights to avoid borrow conflict
                let layer_weights = self.indexed_layer_weights[layer_idx].clone();
                hidden_gpu = self.transformer_layer_indexed(
                    &hidden_gpu,
                    layer_idx,
                    &layer_weights,
                    hidden_dim,
                    intermediate_dim,
                    epsilon,
                )?;
            }
        } else {
            // Legacy path - HashMap lookups + string formatting (~10ms overhead)
            for layer_idx in 0..num_layers {
                let prefix = format!("blk.{}", layer_idx);
                let (ref attn_name, ref ffn_name) = layer_keys[layer_idx];

                // Get cached gamma buffer pointers (no data copy, just metadata)
                let attn_gamma = self.rmsnorm_cache.get(attn_name).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PAR-023: Missing cached gamma for {}",
                        attn_name
                    ))
                })?;
                let attn_ptr = attn_gamma.as_ptr();
                let attn_len = attn_gamma.len();
                let ffn_gamma = self.rmsnorm_cache.get(ffn_name).ok_or_else(|| {
                    GpuError::InvalidLaunchConfig(format!(
                        "PAR-023: Missing cached gamma for {}",
                        ffn_name
                    ))
                })?;
                let ffn_ptr = ffn_gamma.as_ptr();
                let ffn_len = ffn_gamma.len();

                // Run layer GPU-resident using cached gamma buffers
                hidden_gpu = self.transformer_layer_gpu_cached(
                    &hidden_gpu,
                    layer_idx,
                    &prefix,
                    hidden_dim,
                    intermediate_dim,
                    attn_ptr,
                    attn_len,
                    ffn_ptr,
                    ffn_len,
                    epsilon,
                )?;
            }
        }

        // 5. Final sync and download - sync point #2
        // PAR-044 FIX: Copy from correct buffer based on which path was used
        self.stream.synchronize()?;
        if workspace_used {
            // Output is in hidden_buf2
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
            let output_buf = unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_ptr, hidden_len) };
            output_buf.copy_to_host(output)?;
            std::mem::forget(output_buf);
        } else {
            hidden_gpu.copy_to_host(output)?;
        }

        Ok(())
    }
}
