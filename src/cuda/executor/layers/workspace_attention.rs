impl CudaExecutor {
    /// Phase 3-5: Attention + output projection + residual1
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn workspace_attention_residual_phase(
        &mut self,
        input: &GpuBuffer<f32>,
        hidden_buf1: &GpuBuffer<f32>,
        q_buf: &GpuBuffer<f32>,
        k_buf: &GpuBuffer<f32>,
        v_buf: &GpuBuffer<f32>,
        attn_out_buf: &GpuBuffer<f32>,
        input_staging: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &ValidatedLayerWeights,
        hidden_dim: u32,
        q_dim: u32,
        skip_debug: bool,
        profiling: bool,
    ) -> Result<(), GpuError> {
        // 3. PAR-051: Incremental attention into pre-allocated workspace buffer
        // Eliminates 28 GPU allocations per token
        // PAR-054-FIX: Use capture-safe version during graph capture to skip debug sync
        let timer_attn = if profiling {
            self.start_brick_id(trueno::BrickId::AttentionScore)
        } else {
            None
        };
        let _seq_len = if skip_debug {
            self.incremental_attention_into_for_capture(
                layer_idx,
                q_buf,
                k_buf,
                v_buf,
                attn_out_buf,
            )?
        } else {
            self.incremental_attention_into(layer_idx, q_buf, k_buf, v_buf, attn_out_buf)?
        };
        if profiling {
            self.stop_brick_id(timer_attn, 1);
        }

        // PAR-058-DEBUG: Check attention output (skip during graph capture)
        // Note: attention runs on compute_stream, so sync that first
        if !skip_debug && (layer_idx < 4 || (layer_idx >= 10 && layer_idx <= 12)) {
            self.compute_stream.synchronize()?;
            self.debug_check_buf(attn_out_buf, "Attn", layer_idx)?;
        }

        // GH-559 ROOT CAUSE FIX: Synchronize compute_stream before output projection.
        // Attention runs on compute_stream, output projection reads attn_out_buf on self.stream.
        // Without this sync, output projection reads stale/incomplete attention data, causing
        // layers to produce garbage (no-op layers where output == input).
        // Five-Whys: GPU wrong logits → layers 13-27 are no-ops → attn_out_buf stale →
        // compute_stream not synced → race condition between attention and output projection.
        self.compute_stream.synchronize()?;

        // PMAT-027: Invalidate Q8 cache — input is now attn_out_buf (different from QKV's hidden_buf1).
        self.q8_activation_valid = false;

        // 4. Output projection: attn_out_buf -> hidden_buf1 (reuse, normed no longer needed)
        let timer_oproj = if profiling {
            self.start_brick_id(trueno::BrickId::OutputProjection)
        } else {
            None
        };
        self.gemv_dispatch(
            layer_weights.attn_output_qtype,
            layer_weights.attn_output_ptr,
            attn_out_buf, hidden_buf1, hidden_dim, q_dim,
        )?;
        if profiling {
            self.stop_brick_id(timer_oproj, 1);
        }

        // PAR-058-DEBUG: Check output projection (skip during graph capture)
        if !skip_debug && (layer_idx < 4 || (layer_idx >= 10 && layer_idx <= 12)) {
            self.debug_check_buf(hidden_buf1, "Output proj", layer_idx)?;
        }

        // 5. First residual: input + projected -> input_staging (PAR-044 FIX)
        // NOTE: Using input_staging instead of hidden_buf2 to avoid read/write conflict
        // when input IS hidden_buf2 (layers 1+)
        // PAR-075: Cannot fuse with RmsNorm2 because we need input_staging for second residual
        let timer_res1 = if profiling {
            self.start_brick_timer("Residual1")
        } else {
            None
        };
        self.residual_add_into(input, hidden_buf1, input_staging, hidden_dim)?;
        if profiling {
            self.stop_brick_timer(timer_res1, 1);
        }

        // PAR-058-DEBUG: Check residual1 output (skip during graph capture)
        if !skip_debug && (layer_idx < 4 || (layer_idx >= 10 && layer_idx <= 12)) {
            self.debug_check_buf(input_staging, "Residual1", layer_idx)?;
        }

        Ok(())
    }
}
