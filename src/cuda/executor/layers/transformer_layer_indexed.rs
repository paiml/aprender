impl CudaExecutor {
    /// PAR-043: Transformer layer using pre-indexed device pointers (async, no sync)
    ///
    /// This is the **hot path** for decode. Eliminates ALL string formatting and HashMap
    /// lookups (7 per layer = ~224 allocations/lookups per forward pass for 32 layers).
    ///
    /// Measured improvement: ~10ms per token overhead → ~0.1ms per token overhead
    ///
    /// # Arguments
    ///
    /// * `input` - GPU buffer with hidden states [hidden_dim]
    /// * `layer_idx` - Layer index (for incremental attention KV cache)
    /// * `layer_weights` - Pre-indexed pointers from `indexed_layer_weights`
    /// * `hidden_dim` - Model hidden dimension
    /// * `intermediate_dim` - FFN intermediate dimension
    /// * `epsilon` - RMSNorm epsilon
    #[allow(clippy::too_many_arguments)]
    pub fn transformer_layer_indexed(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &ValidatedLayerWeights,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<GpuBuffer<f32>, GpuError> {
        // PROHIBITION-OF-MIRACLES (T-COV-95): Validate pointers BEFORE kernel launch
        // Null pointers corrupt GPU context - fail loudly at API boundary
        if layer_weights.attn_norm_ptr == 0 {
            return Err(GpuError::InvalidParameter(
                "attn_norm_ptr is null (0)".into(),
            ));
        }
        if layer_weights.attn_q_ptr == 0 {
            return Err(GpuError::InvalidParameter("attn_q_ptr is null (0)".into()));
        }
        if layer_weights.attn_k_ptr == 0 {
            return Err(GpuError::InvalidParameter("attn_k_ptr is null (0)".into()));
        }
        if layer_weights.attn_v_ptr == 0 {
            return Err(GpuError::InvalidParameter("attn_v_ptr is null (0)".into()));
        }
        if layer_weights.attn_output_ptr == 0 {
            return Err(GpuError::InvalidParameter(
                "attn_output_ptr is null (0)".into(),
            ));
        }
        if layer_weights.ffn_norm_ptr == 0 {
            return Err(GpuError::InvalidParameter(
                "ffn_norm_ptr is null (0)".into(),
            ));
        }
        if layer_weights.ffn_gate_ptr == 0 {
            return Err(GpuError::InvalidParameter(
                "ffn_gate_ptr is null (0)".into(),
            ));
        }
        if layer_weights.ffn_up_ptr == 0 {
            return Err(GpuError::InvalidParameter("ffn_up_ptr is null (0)".into()));
        }
        if layer_weights.ffn_down_ptr == 0 {
            return Err(GpuError::InvalidParameter(
                "ffn_down_ptr is null (0)".into(),
            ));
        }

        // 1. Pre-attention RMSNorm using indexed gamma pointer
        let normed = self.rmsnorm_gpu_ptr(
            input,
            layer_weights.attn_norm_ptr,
            layer_weights.attn_norm_len,
            hidden_dim,
            epsilon,
        )?;

        // 2. Q/K/V projections using indexed pointers (no sync)
        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        let q =
            self.q4k_gemv_indexed_async(layer_weights.attn_q_ptr, &normed, q_dim, hidden_dim)?;
        let k =
            self.q4k_gemv_indexed_async(layer_weights.attn_k_ptr, &normed, kv_dim, hidden_dim)?;
        // PMAT-232 CONTRACT: Exhaustive dispatch — no catch-all. See tensor-layout-v1.yaml quant_dispatch.
        // NOTE: Indexed async path only has Q4K/Q6K kernels. Other types must use workspace path.
        let v = match layer_weights.attn_v_qtype {
            WeightQuantType::Q6K => {
                self.q6k_gemv_indexed_async(layer_weights.attn_v_ptr, &normed, kv_dim, hidden_dim)?
            },
            WeightQuantType::Q4K => {
                self.q4k_gemv_indexed_async(layer_weights.attn_v_ptr, &normed, kv_dim, hidden_dim)?
            },
            WeightQuantType::Q5K
            | WeightQuantType::Q8_0
            | WeightQuantType::Q4_0
            | WeightQuantType::Q5_0
            | WeightQuantType::Q4_1 => {
                return Err(GpuError::InvalidParameter(format!(
                    "PMAT-232: V qtype {:?} not supported in indexed async path (use workspace path)",
                    layer_weights.attn_v_qtype
                )));
            },
        };

        // 3. Incremental attention (has internal sync for KV cache update)
        let (attn_out, _seq_len) = self.incremental_attention_async(layer_idx, &q, &k, &v)?;

        // 4. Output projection using indexed pointer (no sync)
        let projected = self.q4k_gemv_indexed_async(
            layer_weights.attn_output_ptr,
            &attn_out,
            hidden_dim,
            q_dim,
        )?;

        // 5. First residual add (no sync)
        let residual1 = self.residual_add_gpu(input, &projected, hidden_dim)?;

        // 6. Pre-FFN RMSNorm using indexed gamma pointer
        let ffn_normed = self.rmsnorm_gpu_ptr(
            &residual1,
            layer_weights.ffn_norm_ptr,
            layer_weights.ffn_norm_len,
            hidden_dim,
            epsilon,
        )?;

        // 7. FFN SwiGLU using indexed pointers (no sync)
        let ffn_out = self.fused_ffn_swiglu_indexed_gpu(
            &ffn_normed,
            layer_weights.ffn_gate_ptr,
            layer_weights.ffn_up_ptr,
            layer_weights.ffn_down_ptr,
            hidden_dim,
            intermediate_dim,
        )?;

        // 8. Second residual add (no sync)
        let output = self.residual_add_gpu(&residual1, &ffn_out, hidden_dim)?;

        Ok(output)
    }

    /// PAR-044: Transformer layer with zero allocations using workspace buffers
    ///
    /// Uses pre-allocated workspace buffers for all intermediate tensors.
    /// Eliminates ~288 buffer allocations per token.
    ///
    /// # Buffer Usage
    ///
    /// Workspace buffers used:
    /// - hidden_buf1: normed, projected, ffn_normed, ffn_out (reused)
    /// - hidden_buf2: residual1, final output
    /// - q_buf: Q projection, then attention output
    /// - k_buf: K projection
    /// - v_buf: V projection
    /// - ffn_gate_buf: FFN gate projection
    /// - ffn_up_buf: FFN up projection
    /// - ffn_act_buf: SwiGLU activation result
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success. Output is written to workspace.hidden_buf2.
    /// PAR-054: Transformer layer for graph capture (no debug output)
    /// PAR-070: Takes position but uses indirect mode (reads from position_buf) during graph capture
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transformer_layer_workspace_for_capture(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &ValidatedLayerWeights,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
        position: u32,
    ) -> Result<(), GpuError> {
        self.transformer_layer_workspace_inner(
            input,
            layer_idx,
            layer_weights,
            hidden_dim,
            intermediate_dim,
            epsilon,
            position,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transformer_layer_workspace(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &ValidatedLayerWeights,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
        position: u32, // PAR-070: Explicit position for RoPE and KV cache
    ) -> Result<(), GpuError> {
        // PERF-001: skip_debug=true disables stream.synchronize() calls and debug prints
        // that were causing ~4x slowdown (70 tok/s -> target 280+ tok/s)
        // CORRECTNESS-001: Set GPU_DEBUG=1 to enable layer-by-layer debug output
        static GPU_DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let skip_debug = !*GPU_DEBUG.get_or_init(|| {
            std::env::var("GPU_DEBUG")
                .map(|v| v == "1")
                .unwrap_or(false)
        });
        self.transformer_layer_workspace_inner(
            input,
            layer_idx,
            layer_weights,
            hidden_dim,
            intermediate_dim,
            epsilon,
            position,
            skip_debug,
        )
    }
}
