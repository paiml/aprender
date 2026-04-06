impl CudaExecutor {
    // =========================================================================
    // PAR-023: GPU-Resident SwiGLU FFN (LLaMA-style)
    // Reduces 3 syncs per layer to 1 by chaining: gate→up→swiglu→down
    // =========================================================================

    /// PAR-023: GPU-resident SwiGLU FFN operating entirely on GPU buffers
    ///
    /// Implements LLaMA-style FFN: down(swiglu(gate(x), up(x)))
    /// All operations chained without sync - only syncs when output needed.
    ///
    /// PAR-063-V5: Set TRUE_DP4A=1 to use Q8 activation quantization + Q4K×Q8
    /// integer dot product for 4x instruction reduction (llama.cpp-style).
    ///
    /// # Arguments
    /// * `input` - GPU buffer containing hidden state [hidden_dim]
    /// * `ffn_gate_name` - Cache key for FFN gate weight
    /// * `ffn_up_name` - Cache key for FFN up weight
    /// * `ffn_down_name` - Cache key for FFN down weight
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    ///
    /// # Returns
    /// GPU buffer containing FFN output [hidden_dim] - not synchronized
    #[allow(clippy::too_many_arguments)]
    pub fn fused_ffn_swiglu_gpu(
        &mut self,
        input: &GpuBuffer<f32>,
        ffn_gate_name: &str,
        ffn_up_name: &str,
        ffn_down_name: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        super::layers::fused_ffn_swiglu_gpu(
            self,
            input,
            ffn_gate_name,
            ffn_up_name,
            ffn_down_name,
            hidden_dim,
            intermediate_dim,
        )
    }

    /// PAR-063-V5: GPU-resident SwiGLU FFN using TRUE DP4A kernels (async, no sync)
    ///
    /// Uses Q8 activation quantization + Q4K×Q8 integer dot product for 4x instruction reduction.
    /// This is the llama.cpp-style approach:
    /// 1. Quantize f32 activations to Q8_1 (per-block scale + 32 × int8)
    /// 2. Use dp4a.u32.s32 for 4 multiply-adds per instruction
    /// 3. Apply scales at the end
    ///
    /// # Arguments
    /// * `input` - GPU buffer containing hidden state [hidden_dim]
    /// * `ffn_gate_name` - Cache key for FFN gate weight
    /// * `ffn_up_name` - Cache key for FFN up weight
    /// * `ffn_down_name` - Cache key for FFN down weight
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    #[allow(clippy::too_many_arguments)]
    pub fn fused_ffn_swiglu_gpu_true_dp4a(
        &mut self,
        input: &GpuBuffer<f32>,
        ffn_gate_name: &str,
        ffn_up_name: &str,
        ffn_down_name: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        super::layers::fused_ffn_swiglu_gpu_true_dp4a(
            self,
            input,
            ffn_gate_name,
            ffn_up_name,
            ffn_down_name,
            hidden_dim,
            intermediate_dim,
        )
    }

    /// PAR-043: SwiGLU FFN using pre-indexed device pointers (async, no sync)
    ///
    /// This eliminates 3 HashMap lookups + string formatting per FFN call.
    /// Pointers must be from `indexed_layer_weights` populated by `build_indexed_weights()`.
    pub fn fused_ffn_swiglu_indexed_gpu(
        &mut self,
        input: &GpuBuffer<f32>,
        ffn_gate_ptr: u64,
        ffn_up_ptr: u64,
        ffn_down_ptr: u64,
        hidden_dim: u32,
        intermediate_dim: u32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // 1. Gate projection: [hidden_dim] -> [intermediate_dim] (no sync)
        let gate =
            self.q4k_gemv_indexed_async(ffn_gate_ptr, input, intermediate_dim, hidden_dim)?;

        // 2. Up projection: [hidden_dim] -> [intermediate_dim] (no sync)
        let up = self.q4k_gemv_indexed_async(ffn_up_ptr, input, intermediate_dim, hidden_dim)?;

        // 3. Fused SwiGLU: silu(gate) * up (no sync)
        let activated = self.fused_swiglu_gpu(&gate, &up, intermediate_dim)?;

        // 4. Down projection: [intermediate_dim] -> [hidden_dim] (no sync)
        let output =
            self.q4k_gemv_indexed_async(ffn_down_ptr, &activated, hidden_dim, intermediate_dim)?;

        // PAR-043: NO sync here - caller chains more operations or syncs when needed
        Ok(output)
    }

    /// PAR-023: SwiGLU FFN with host memory (convenience wrapper)
    ///
    /// Uploads input, runs GPU-resident FFN, syncs, downloads result.
    /// For testing and single-FFN use cases.
    #[allow(clippy::too_many_arguments)]
    pub fn fused_ffn_swiglu_host(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        ffn_gate_name: &str,
        ffn_up_name: &str,
        ffn_down_name: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
    ) -> Result<(), GpuError> {
        // Upload input
        let input_gpu = GpuBuffer::from_host(&self.context, input)?;

        // Run GPU-resident FFN (no intermediate syncs)
        let output_gpu = self.fused_ffn_swiglu_gpu(
            &input_gpu,
            ffn_gate_name,
            ffn_up_name,
            ffn_down_name,
            hidden_dim,
            intermediate_dim,
        )?;

        // Single sync and download
        self.stream.synchronize()?;
        output_gpu.copy_to_host(output)?;

        Ok(())
    }

    // =========================================================================
    // PAR-023: GPU-Resident Transformer Layer
    // Chains all operations with minimal syncs for maximum throughput
    // Target: Reduce 176 syncs/token to ~22 syncs/token (1 per layer)
    // =========================================================================

    /// PAR-023: GPU-resident transformer layer (LLaMA-style)
    ///
    /// Chains all layer operations on GPU with single sync at end:
    /// 1. Pre-attention RMSNorm
    /// 2. Q/K/V projections
    /// 3. Incremental attention
    /// 4. Output projection
    /// 5. Residual add
    /// 6. Pre-FFN RMSNorm
    /// 7. Gate/Up projections + SwiGLU + Down projection
    /// 8. Residual add
    ///
    /// # Arguments
    /// * `input` - GPU buffer containing hidden state [hidden_dim]
    /// * `layer_idx` - Layer index for weight lookup and KV cache
    /// * `layer_prefix` - Weight name prefix (e.g., "blk.0")
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `attn_norm_gamma` - Pre-attention RMSNorm weights
    /// * `ffn_norm_gamma` - Pre-FFN RMSNorm weights
    /// * `epsilon` - RMSNorm epsilon
    ///
    /// # Returns
    /// GPU buffer containing layer output [hidden_dim] - NOT synchronized
    #[allow(clippy::too_many_arguments)]
    pub fn transformer_layer_gpu(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_prefix: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
        attn_norm_gamma: &GpuBuffer<f32>,
        ffn_norm_gamma: &GpuBuffer<f32>,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Weight names follow GGML convention
        let q_name = format!("{}.attn_q.weight", layer_prefix);
        let k_name = format!("{}.attn_k.weight", layer_prefix);
        let v_name = format!("{}.attn_v.weight", layer_prefix);
        let o_name = format!("{}.attn_output.weight", layer_prefix);
        let gate_name = format!("{}.ffn_gate.weight", layer_prefix);
        let up_name = format!("{}.ffn_up.weight", layer_prefix);
        let down_name = format!("{}.ffn_down.weight", layer_prefix);

        // 1. Pre-attention RMSNorm (no sync)
        let normed = self.rmsnorm_gpu(input, attn_norm_gamma, hidden_dim, epsilon)?;

        // 2. Q/K/V projections (no sync)
        // Q: [hidden_dim] -> [num_heads * head_dim]
        // K: [hidden_dim] -> [num_kv_heads * head_dim]
        // V: [hidden_dim] -> [num_kv_heads * head_dim]
        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        // PAR-056: Tiled kernel selection based on K dimension
        // - TiledQ4KGemv: K <= 8192 (fits in 48KB shared memory)
        // - ChunkedTiledQ4KGemv: K > 8192 (uses 32KB chunks)
        const CHUNK_THRESHOLD: u32 = 8192;
        let hidden_aligned = hidden_dim.is_multiple_of(256);
        let q_aligned = q_dim.is_multiple_of(256);
        let kv_aligned = kv_dim.is_multiple_of(256);

        // Q/K/V projections: K = hidden_dim
        // CORRECTNESS-001: Temporarily disable DP4A to test fixed TiledQ4K kernel
        // PAR-063: Use DP4A kernel for aligned dimensions (fastest)
        let _use_dp4a = hidden_aligned && q_aligned && hidden_dim <= CHUNK_THRESHOLD;
        let q = {
            // Force TiledQ4K for now - dp4a_q4k has scale extraction issue
            self.q4k_gemv_cached_async(&q_name, &normed, q_dim, hidden_dim)?
        };
        let _use_dp4a_kv = hidden_aligned && kv_aligned && hidden_dim <= CHUNK_THRESHOLD;
        let k = { self.q4k_gemv_cached_async(&k_name, &normed, kv_dim, hidden_dim)? };
        let v = { self.q4k_gemv_cached_async(&v_name, &normed, kv_dim, hidden_dim)? };

        // 3. Incremental attention (has internal sync for KV cache update)
        let (attn_out, _seq_len) = self.incremental_attention_async(layer_idx, &q, &k, &v)?;

        // 4. Output projection (no sync) - K = q_dim
        // CORRECTNESS-001: Force TiledQ4K kernel
        let projected = { self.q4k_gemv_cached_async(&o_name, &attn_out, hidden_dim, q_dim)? };

        // 5. First residual add (no sync)
        let residual1 = self.residual_add_gpu(input, &projected, hidden_dim)?;

        // 6. Pre-FFN RMSNorm (no sync)
        let ffn_normed = self.rmsnorm_gpu(&residual1, ffn_norm_gamma, hidden_dim, epsilon)?;

        // 7. FFN SwiGLU (no sync)
        let ffn_out = self.fused_ffn_swiglu_gpu(
            &ffn_normed,
            &gate_name,
            &up_name,
            &down_name,
            hidden_dim,
            intermediate_dim,
        )?;

        // 8. Second residual add (no sync)
        let output = self.residual_add_gpu(&residual1, &ffn_out, hidden_dim)?;

        // PAR-023: NO sync here - caller can chain multiple layers
        Ok(output)
    }

    /// TILING-SPEC-001: Tile-profiled transformer layer for bottleneck identification.
    ///
    /// This method wraps `transformer_layer_gpu` with tile-level profiling instrumentation
    /// to identify whether the 0.07% efficiency bottleneck is:
    /// - Kernel launch overhead (many small kernels)
    /// - CPU dequantization in the hot path
    /// - Memory transfer overhead (H2D/D2H)
    /// - Specific operation bottlenecks (QKV, attention, FFN)
    ///
    /// # Profiling Levels
    ///
    /// | Level | Operation | FLOPs Formula |
    /// |-------|-----------|---------------|
    /// | Macro | QKV Projections | 2 × M × K × 3 |
    /// | Macro | Output Projection | 2 × M × K |
    /// | Midi  | Attention | 2 × seq × head_dim × num_heads |
    /// | Macro | FFN (SwiGLU) | 2 × M × K × 3 |
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// cuda_model.enable_tile_profiling();
    /// let output = cuda_model.transformer_layer_gpu_tiled_profiled(...)?;
    /// println!("{}", cuda_model.tile_summary());
    /// // Output shows per-operation GFLOP/s and identifies bottlenecks
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn transformer_layer_gpu_tiled_profiled(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_prefix: &str,
        hidden_dim: u32,
        intermediate_dim: u32,
        attn_norm_gamma: &GpuBuffer<f32>,
        ffn_norm_gamma: &GpuBuffer<f32>,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // Weight names follow GGML convention
        let q_name = format!("{}.attn_q.weight", layer_prefix);
        let k_name = format!("{}.attn_k.weight", layer_prefix);
        let v_name = format!("{}.attn_v.weight", layer_prefix);
        let o_name = format!("{}.attn_output.weight", layer_prefix);
        let gate_name = format!("{}.ffn_gate.weight", layer_prefix);
        let up_name = format!("{}.ffn_up.weight", layer_prefix);
        let down_name = format!("{}.ffn_down.weight", layer_prefix);

        // Q/K/V dimensions
        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        // 1. Pre-attention RMSNorm (tracked as Micro - very fast)
        let timer_norm1 = self.start_tile_timer(trueno::TileLevel::Micro, layer_idx as u32, 0);
        let normed = self.rmsnorm_gpu(input, attn_norm_gamma, hidden_dim, epsilon)?;
        // RMSNorm FLOPs: 5N (square, sum, rsqrt, multiply, multiply) per element
        let norm_flops = (hidden_dim as u64) * 5;
        self.stop_tile_timer(timer_norm1, hidden_dim as u64, norm_flops);

        // 2. Q/K/V projections (Macro tile - largest compute block)
        // FLOPs: 2 * M * K for each matrix-vector multiply (M=1 for single token)
        let timer_qkv = self.start_tile_timer(trueno::TileLevel::Macro, layer_idx as u32, 1);

        let q = self.q4k_gemv_cached_async(&q_name, &normed, q_dim, hidden_dim)?;
        let k = self.q4k_gemv_cached_async(&k_name, &normed, kv_dim, hidden_dim)?;
        let v = self.q4k_gemv_cached_async(&v_name, &normed, kv_dim, hidden_dim)?;

        // QKV FLOPs: Q(hidden→q) + K(hidden→kv) + V(hidden→kv)
        let qkv_flops = 2 * (hidden_dim as u64) * (q_dim as u64 + kv_dim as u64 + kv_dim as u64);
        let qkv_elements = (q_dim + kv_dim + kv_dim) as u64;
        self.stop_tile_timer(timer_qkv, qkv_elements, qkv_flops);

        // 3. Incremental attention (Midi tile - head-level parallelism)
        let timer_attn = self.start_tile_timer(trueno::TileLevel::Midi, layer_idx as u32, 2);
        let (attn_out, seq_len) = self.incremental_attention_async(layer_idx, &q, &k, &v)?;
        // Attention FLOPs: 2 * seq * head_dim * num_heads (Q×K^T + softmax×V)
        let attn_flops =
            2 * (seq_len as u64) * (self.kv_head_dim as u64) * (self.kv_num_heads as u64) * 2;
        self.stop_tile_timer(timer_attn, q_dim as u64, attn_flops);

        // 4. Output projection (Macro tile)
        let timer_proj = self.start_tile_timer(trueno::TileLevel::Macro, layer_idx as u32, 3);
        let projected = self.q4k_gemv_cached_async(&o_name, &attn_out, hidden_dim, q_dim)?;
        let proj_flops = 2 * (q_dim as u64) * (hidden_dim as u64);
        self.stop_tile_timer(timer_proj, hidden_dim as u64, proj_flops);

        // 5. First residual add (Micro - very fast)
        let timer_res1 = self.start_tile_timer(trueno::TileLevel::Micro, layer_idx as u32, 4);
        let residual1 = self.residual_add_gpu(input, &projected, hidden_dim)?;
        self.stop_tile_timer(timer_res1, hidden_dim as u64, hidden_dim as u64);

        // 6. Pre-FFN RMSNorm (Micro)
        let timer_norm2 = self.start_tile_timer(trueno::TileLevel::Micro, layer_idx as u32, 5);
        let ffn_normed = self.rmsnorm_gpu(&residual1, ffn_norm_gamma, hidden_dim, epsilon)?;
        self.stop_tile_timer(timer_norm2, hidden_dim as u64, norm_flops);

        // 7. FFN SwiGLU (Macro tile - second largest compute block)
        // FLOPs: gate(hidden→inter) + up(hidden→inter) + down(inter→hidden) + SiLU
        let timer_ffn = self.start_tile_timer(trueno::TileLevel::Macro, layer_idx as u32, 6);
        let ffn_out = self.fused_ffn_swiglu_gpu(
            &ffn_normed,
            &gate_name,
            &up_name,
            &down_name,
            hidden_dim,
            intermediate_dim,
        )?;
        // FFN FLOPs: 3 GEMV (gate+up+down) + SiLU (~3 ops per element)
        let ffn_flops =
            2 * (hidden_dim as u64) * (intermediate_dim as u64) * 3 + (intermediate_dim as u64) * 3;
        self.stop_tile_timer(timer_ffn, hidden_dim as u64, ffn_flops);

        // 8. Second residual add (Micro)
        let timer_res2 = self.start_tile_timer(trueno::TileLevel::Micro, layer_idx as u32, 7);
        let output = self.residual_add_gpu(&residual1, &ffn_out, hidden_dim)?;
        self.stop_tile_timer(timer_res2, hidden_dim as u64, hidden_dim as u64);

        Ok(output)
    }
}
