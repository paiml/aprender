
impl CudaExecutor {

    /// Validate that all RMSNorm gamma weights (per-layer + output) are cached.
    ///
    /// Returns an error if any required gamma buffer is missing from `rmsnorm_cache`.
    fn validate_rmsnorm_cache_for_logits(
        &self,
        num_layers: usize,
    ) -> Result<(), GpuError> {
        for layer_idx in 0..num_layers {
            let attn_name = format!("blk.{}.attn_norm.gamma", layer_idx);
            if !self.rmsnorm_cache.contains_key(&attn_name) {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: attn_norm not cached for layer {}",
                    layer_idx
                )));
            }
            let ffn_name = format!("blk.{}.ffn_norm.gamma", layer_idx);
            if !self.rmsnorm_cache.contains_key(&ffn_name) {
                return Err(GpuError::InvalidLaunchConfig(format!(
                    "PAR-023: ffn_norm not cached for layer {}",
                    layer_idx
                )));
            }
        }
        if !self.rmsnorm_cache.contains_key("output_norm.gamma") {
            return Err(GpuError::InvalidLaunchConfig(
                "PAR-023: output_norm not cached".to_string(),
            ));
        }
        Ok(())
    }

    /// GH-215 FIX: Pad input embedding to Q4K super-block boundary (256) and upload.
    ///
    /// Q4K GEMV kernels access `activations[sb_idx*256+val_idx]` which can exceed
    /// the logical dimension for non-256-aligned models (e.g., `hidden_dim=896`).
    fn pad_and_upload_input(&self, input: &[f32]) -> Result<GpuBuffer<f32>, GpuError> {
        let padded_len = ((input.len() + 255) / 256) * 256;
        let padded_input: std::borrow::Cow<'_, [f32]> = if padded_len > input.len() {
            let mut padded = vec![0.0f32; padded_len];
            padded[..input.len()].copy_from_slice(input);
            std::borrow::Cow::Owned(padded)
        } else {
            std::borrow::Cow::Borrowed(input)
        };
        GpuBuffer::from_host(&self.context, &padded_input)
    }

    /// PAR-044: Run all transformer layers via the zero-allocation workspace path.
    ///
    /// Layer 0 reads from the externally-provided `hidden_gpu`; layers 1+ read from
    /// `workspace.hidden_buf2` (the output of the previous layer).
    #[allow(clippy::too_many_arguments)]
    fn run_workspace_layers(
        &mut self,
        hidden_gpu: &GpuBuffer<f32>,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
        position: u32,
    ) -> Result<(), GpuError> {
        // GH-559: Per-layer debug (moved OnceLock here for early use)
        static DEBUG_LAYERS_EARLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let debug_layers_early = *DEBUG_LAYERS_EARLY.get_or_init(|| {
            std::env::var("GPU_DEBUG_ALL_LAYERS")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        // GH-559: Check embedding upload
        if debug_layers_early {
            let n = (hidden_dim as usize).min(hidden_gpu.len());
            let mut host = vec![0.0f32; n];
            if hidden_gpu.copy_to_host(&mut host).is_ok() {
                let sum: f32 = host.iter().sum();
                let rms = (host.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
                eprintln!(
                    "[GH-559] Layer 0 INPUT (hidden_gpu embed): sum={:.6}, rms={:.6}, first5={:?}",
                    sum, rms, &host[..5.min(n)]
                );
            }
        }

        // Layer 0: input from external hidden_gpu
        // PAR-070: Pass explicit position for RoPE and KV cache
        if num_layers > 0 {
            let layer_weights = self.indexed_layer_weights[0].clone();
            self.transformer_layer_workspace(
                hidden_gpu,
                0,
                &layer_weights,
                hidden_dim,
                intermediate_dim,
                epsilon,
                position,
            )?;
        }

        // GH-559: Per-layer debug checksum for GPU parity diagnosis
        static DEBUG_LAYERS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let debug_layers = *DEBUG_LAYERS.get_or_init(|| {
            std::env::var("GPU_DEBUG_ALL_LAYERS")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        // GH-559: Check layer 0 output
        if debug_layers {
            if let Some(ref buf2) = self.workspace.hidden_buf2 {
                let n = (hidden_dim as usize).min(buf2.len());
                let mut host = vec![0.0f32; n];
                if buf2.copy_to_host(&mut host).is_ok() {
                    let sum: f32 = host.iter().sum();
                    let rms = (host.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
                    eprintln!(
                        "[GH-559] Layer 0/28 OUTPUT (hidden_buf2): sum={:.6}, rms={:.6}, first5={:?}",
                        sum, rms, &host[..5.min(n)]
                    );
                    // GH-559: Dump per-super-block sums to find divergent region
                    for sb in 0..(n / 256) {
                        let idx = sb * 256;
                        let end = (idx + 5).min(n);
                        let sb_sum: f32 = host[idx..idx+256.min(n-idx)].iter().sum();
                        eprintln!(
                            "[GH-559-GPU] L0 sb{}: idx={}, sum={:.4}, vals={:?}",
                            sb, idx, sb_sum, &host[idx..end]
                        );
                    }
                }
            }
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

            // GH-559: Download hidden state checksum BEFORE this layer
            if debug_layers {
                let mut host_buf = vec![0.0f32; buf_len.min(hidden_dim as usize)];
                if input_buf.copy_to_host(&mut host_buf).is_ok() {
                    let sum: f32 = host_buf.iter().sum();
                    let rms: f32 = (host_buf.iter().map(|x| x * x).sum::<f32>()
                        / host_buf.len() as f32)
                        .sqrt();
                    eprintln!(
                        "[GH-559] Layer {}/{} input: sum={:.6}, rms={:.6}, first5={:?}",
                        layer_idx,
                        num_layers,
                        sum,
                        rms,
                        &host_buf[..5.min(host_buf.len())]
                    );
                }
            }

            // PAR-070: Pass explicit position for RoPE and KV cache
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
        Ok(())
    }

    /// PAR-043: Run all transformer layers via the indexed path (O(1) weight access).
    #[allow(clippy::too_many_arguments)]
    fn run_indexed_layers(
        &mut self,
        mut hidden_gpu: GpuBuffer<f32>,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        for layer_idx in 0..num_layers {
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
        Ok(hidden_gpu)
    }

    /// Legacy path: run all transformer layers via HashMap lookups + string formatting.
    #[allow(clippy::too_many_arguments)]
    fn run_legacy_layers(
        &mut self,
        mut hidden_gpu: GpuBuffer<f32>,
        num_layers: usize,
        layer_keys: &[(String, String)],
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        for layer_idx in 0..num_layers {
            let prefix = format!("blk.{}", layer_idx);
            let (ref attn_name, ref ffn_name) = layer_keys[layer_idx];

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
        Ok(hidden_gpu)
    }

    /// CORRECTNESS-001: Debug-dump the hidden state before output norm (GPU_DEBUG=1 only).
    fn debug_dump_hidden_state(
        &mut self,
        hidden_gpu: &GpuBuffer<f32>,
        workspace_used: bool,
        debug_enabled: bool,
    ) -> Result<(), GpuError> {
        if !debug_enabled {
            return Ok(());
        }
        self.stream.synchronize()?;
        let hidden_to_check = if workspace_used {
            let ptr = self
                .workspace
                .hidden_buf2
                .as_ref()
                .expect("hidden_buf2 must be initialized")
                .as_ptr();
            let len = self
                .workspace
                .hidden_buf2
                .as_ref()
                .expect("hidden_buf2 must be initialized")
                .len();
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe { GpuBuffer::<f32>::from_raw_parts(ptr, len) }
        } else {
            // SAFETY: Memory safety ensured by bounds checking and alignment
            unsafe { GpuBuffer::<f32>::from_raw_parts(hidden_gpu.as_ptr(), hidden_gpu.len()) }
        };
        let mut hidden_host = vec![0.0f32; hidden_to_check.len()];
        hidden_to_check.copy_to_host(&mut hidden_host)?;
        std::mem::forget(hidden_to_check);
        let sum: f32 = hidden_host.iter().sum();
        let sum_sq: f32 = hidden_host.iter().map(|x| x * x).sum();
        eprintln!(
            "[CORRECTNESS-001] Hidden before output_norm: first 5 = {:?}, sum = {:.4}, rms = {:.4}",
            &hidden_host[..5.min(hidden_host.len())],
            sum,
            (sum_sq / hidden_host.len() as f32).sqrt()
        );
        Ok(())
    }

    /// Apply output RMSNorm on GPU, selecting workspace or external hidden buffer.
    #[allow(clippy::too_many_arguments)]
    fn apply_output_rmsnorm(
        &mut self,
        hidden_gpu: &GpuBuffer<f32>,
        workspace_used: bool,
        hidden_dim: u32,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        let output_norm_gamma = self.rmsnorm_cache.get("output_norm.gamma").ok_or_else(|| {
            GpuError::InvalidLaunchConfig(
                "PAR-023: Missing cached gamma for output_norm.gamma".to_string(),
            )
        })?;
        let output_gamma_ptr = output_norm_gamma.as_ptr();
        let output_gamma_len = output_norm_gamma.len();

        if workspace_used {
            // PAR-044 FIX: Use hidden_buf2 directly (no D2D copy)
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
            let result = self.rmsnorm_gpu_ptr(
                &hidden_input,
                output_gamma_ptr,
                output_gamma_len,
                hidden_dim,
                epsilon,
            )?;
            std::mem::forget(hidden_input);
            Ok(result)
        } else {
            self.rmsnorm_gpu_ptr(
                hidden_gpu,
                output_gamma_ptr,
                output_gamma_len,
                hidden_dim,
                epsilon,
            )
        }
    }

    /// CORRECTNESS-002: Debug-dump the normed hidden state before LM head (GPU_DEBUG=1 only).
    fn debug_dump_normed_hidden(
        &mut self,
        normed_hidden: &GpuBuffer<f32>,
        debug_enabled: bool,
    ) -> Result<(), GpuError> {
        if !debug_enabled {
            return Ok(());
        }
        self.stream.synchronize()?;
        let mut normed_host = vec![0.0f32; normed_hidden.len()];
        normed_hidden.copy_to_host(&mut normed_host)?;
        let sum: f32 = normed_host.iter().sum();
        let sum_sq: f32 = normed_host.iter().map(|x| x * x).sum();
        eprintln!(
            "[CORRECTNESS-002] Normed hidden: first 5 = {:?}, sum = {:.4}, rms = {:.4}",
            &normed_host[..5.min(normed_host.len())],
            sum,
            (sum_sq / normed_host.len() as f32).sqrt()
        );
        Ok(())
    }

    /// PAR-023: Fully GPU-resident forward to logits (minimal syncs)
    ///
    /// Runs all transformer layers + output norm + LM head projection entirely on GPU,
    /// only downloading the final logits. This eliminates the CPU round-trip for output norm.
    ///
    /// # Sync Count
    ///
    /// - Input embedding upload: 1 sync
    /// - All transformer layers: 0 syncs (attention has internal D2D)
    /// - Output RMSNorm: 0 syncs (on GPU)
    /// - LM head projection: 0 syncs (on GPU)
    /// - Logits download: 1 sync
    /// - **Total: 2 syncs** vs 3+ syncs (with CPU output norm)
    ///
    /// # Requirements
    ///
    /// Must call `preload_rmsnorm_weights()` and `preload_output_norm()` before use.
    /// LM head weights must be pre-cached via `load_quantized_weights("output.weight", ...)`.
    ///
    /// # Arguments
    ///
    /// * `input` - Input embedding [hidden_dim]
    /// * `logits` - Output logits buffer [vocab_size]
    /// * `position` - Token position for RoPE and KV cache (PAR-070: CORRECTNESS-001 fix)
    /// * `num_layers` - Number of transformer layers
    /// * `hidden_dim` - Hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `vocab_size` - Output vocabulary size
    /// * `epsilon` - RMSNorm epsilon
    #[allow(clippy::too_many_arguments)]
    pub fn forward_all_layers_gpu_to_logits(
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
        // PERF-002: Debug code removed for performance (was PAR-058-DEBUG)
        // NaN checks required D2H transfer on every token - ~10ms overhead each

        // 1. Validate all RMSNorm weights are cached (including output norm)
        self.validate_rmsnorm_cache_for_logits(num_layers)?;

        // 2. Collect all cache key names
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

        let mut hidden_gpu = self.pad_and_upload_input(input)?;

        // 4. Chain all transformer layers (no intermediate syncs)
        // PAR-044: Use workspace path for zero-allocation forward (fastest)
        // PAR-043: Use indexed path if weights are pre-indexed (10x faster per-token)
        // PAR-044 FIX: Track which buffer has output to avoid unnecessary D2D copy
        // PAR-044: Workspace path enabled - confirmed same performance as indexed path
        // See five-whys-gpu-performance-gap for analysis
        let workspace_used = if use_workspace {
            self.run_workspace_layers(
                &hidden_gpu, num_layers, hidden_dim, intermediate_dim, epsilon, position,
            )?;
            true
        } else if self.has_indexed_weights() && self.indexed_layer_weights.len() == num_layers {
            hidden_gpu = self.run_indexed_layers(
                hidden_gpu, num_layers, hidden_dim, intermediate_dim, epsilon,
            )?;
            false
        } else {
            hidden_gpu = self.run_legacy_layers(
                hidden_gpu, num_layers, &layer_keys, hidden_dim, intermediate_dim, epsilon,
            )?;
            false
        };

        // PERF-002: Debug code removed (was PAR-058-DEBUG hidden state check)
        // D2H transfer + NaN check was ~15ms overhead per token

        // CORRECTNESS-001: Compare hidden state before output norm
        static HIDDEN_DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let debug_enabled = *HIDDEN_DEBUG.get_or_init(|| {
            std::env::var("GPU_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
        });
        self.debug_dump_hidden_state(&hidden_gpu, workspace_used, debug_enabled)?;

        // 5. Output RMSNorm on GPU (no sync)
        // PAR-PROFILE: Brick timer for output norm
        let profiling = self.profiler.is_enabled();
        let timer_output_norm = if profiling {
            self.start_brick_id(trueno::BrickId::RmsNorm)
        } else {
            None
        };

        let normed_hidden = self.apply_output_rmsnorm(
            &hidden_gpu, workspace_used, hidden_dim, epsilon,
        )?;

        // CORRECTNESS-002: Debug normed_hidden output (before LM head)
        self.debug_dump_normed_hidden(&normed_hidden, debug_enabled)?;

        // PAR-PROFILE: Stop output norm timer, start LM head timer
        if profiling {
            self.stop_brick_id(timer_output_norm, 1);
        }

        // 6-7. LM head projection + final sync + download
        self.dispatch_lm_head_and_download(
            &normed_hidden, logits, vocab_size, hidden_dim, debug_enabled,
        )
    }

    /// realizr#203: Apply output norm + LM head to a pre-computed hidden state.
    ///
    /// Used by batched prefill PPL: prefill computes hidden states for all positions,
    /// then this function extracts logits per-position via the standard M=1 path.
    pub fn hidden_to_logits(
        &mut self,
        hidden_state: &[f32],
        logits: &mut [f32],
        hidden_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // Upload hidden state
        let hidden_gpu = self.pad_and_upload_input(hidden_state)?;

        // Output RMSNorm
        let normed_hidden = self.apply_output_rmsnorm(
            &hidden_gpu, false, hidden_dim, epsilon,
        )?;

        // LM head + download
        self.dispatch_lm_head_and_download(
            &normed_hidden, logits, vocab_size, hidden_dim, false,
        )
    }
}

include!("logits.rs");
include!("forward_from_forward_from_forward.rs");
