impl CudaExecutor {
    /// Phase 6-10: FFN RMSNorm + gate/up projections + SwiGLU + down projection + residual2
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn workspace_ffn_phase(
        &mut self,
        hidden_buf1: &GpuBuffer<f32>,
        hidden_buf2: &GpuBuffer<f32>,
        input_staging: &GpuBuffer<f32>,
        ffn_gate_buf: &GpuBuffer<f32>,
        ffn_up_buf: &GpuBuffer<f32>,
        ffn_act_buf: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &ValidatedLayerWeights,
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
        skip_debug: bool,
        profiling: bool,
    ) -> Result<(), GpuError> {
        // 6. Pre-FFN RMSNorm: residual1 (input_staging) -> hidden_buf1 (ffn_normed)
        let timer_rmsnorm2 = if profiling {
            self.start_brick_id(trueno::BrickId::RmsNorm)
        } else {
            None
        };
        self.rmsnorm_ptr_into(
            input_staging,
            layer_weights.ffn_norm_ptr,
            layer_weights.ffn_norm_len,
            hidden_buf1,
            hidden_dim,
            epsilon,
        )?;
        if profiling {
            self.stop_brick_id(timer_rmsnorm2, 1);
        }

        // 7+8. FFN gate/up projections + SwiGLU
        // PMAT-027: Invalidate Q8 cache — hidden_buf1 was just written by FFN RMSNorm.
        self.q8_activation_valid = false;

        // PMAT-034: Fused gate+up+SwiGLU when both weights are Q4K and HW DP4A available
        let use_fused = self.gpu_profile.fused_gate_up
            && layer_weights.ffn_gate_qtype == WeightQuantType::Q4K
            && layer_weights.ffn_up_qtype == WeightQuantType::Q4K;

        if use_fused {
            self.fused_gate_up_swiglu_hw_dp4a_q4k_gemv_into(
                layer_weights.ffn_gate_ptr,
                layer_weights.ffn_up_ptr,
                hidden_buf1, ffn_act_buf, hidden_dim, intermediate_dim,
            )?;
        } else {
            self.workspace_ffn_gate_up_swiglu_separate(
                hidden_buf1, ffn_gate_buf, ffn_up_buf, ffn_act_buf,
                layer_weights, intermediate_dim, hidden_dim, profiling,
            )?;
        }

        if !skip_debug && (layer_idx < 4 || (layer_idx >= 10 && layer_idx <= 12)) {
            self.debug_check_buf(ffn_act_buf, "SwiGLU", layer_idx)?;
        }

        // 9. FFN down projection: ffn_act -> hidden_buf1 (reuse, ffn_normed no longer needed)
        // PAR-058: Use correct kernel based on FFN down quantization type
        // PAR-105-FIX: Only override qtype if metadata qtype doesn't match expected size
        let metadata_qtype = layer_weights.ffn_down_qtype;
        let metadata_matches = metadata_qtype.matches_size(
            layer_weights.ffn_down_len,
            hidden_dim as usize,
            intermediate_dim as usize,
        );
        let ffn_down_qtype = if metadata_matches {
            metadata_qtype
        } else {
            WeightQuantType::from_size(
                layer_weights.ffn_down_len,
                hidden_dim as usize,
                intermediate_dim as usize,
            )
            .unwrap_or(metadata_qtype)
        };

        // PMAT-027: Invalidate Q8 cache — input is now ffn_act_buf (different from gate/up).
        self.q8_activation_valid = false;

        let timer_ffn_down = if profiling {
            self.start_brick_id(trueno::BrickId::DownProjection)
        } else {
            None
        };
        self.gemv_dispatch(
            ffn_down_qtype,
            layer_weights.ffn_down_ptr,
            ffn_act_buf, hidden_buf1, hidden_dim, intermediate_dim,
        )?;
        if profiling {
            self.stop_brick_id(timer_ffn_down, 1);
        }

        // PAR-058-DEBUG: Check FFN down output (skip during graph capture)
        if !skip_debug && (layer_idx < 4 || (layer_idx >= 10 && layer_idx <= 12)) {
            self.debug_check_buf(hidden_buf1, "FFN down", layer_idx)?;
        }

        // 10. Second residual: residual1 (input_staging) + ffn_out (hidden_buf1) -> hidden_buf2
        // PAR-044 FIX: Now safe because residual1 is in input_staging, not hidden_buf2
        let timer_res2 = if profiling {
            self.start_brick_timer("Residual2")
        } else {
            None
        };
        self.residual_add_into(input_staging, hidden_buf1, hidden_buf2, hidden_dim)?;
        if profiling {
            self.stop_brick_timer(timer_res2, 1);
        }

        // GH-559: Force stream sync after final residual to ensure hidden_buf2 is
        // fully written before the next layer reads it. On Blackwell sm_121, the
        // CUDA JIT may not guarantee same-stream ordering for all kernel types.
        // Without this sync, layers 13-27 see stale hidden_buf2 data (no-op layers).
        self.stream.synchronize()?;

        // PAR-058-DEBUG: Check layer output (skip during graph capture)
        if !skip_debug && (layer_idx < 10 || (layer_idx >= 10 && layer_idx <= 12)) {
            self.debug_check_buf(hidden_buf2, "Layer output", layer_idx)?;
        }

        Ok(())
    }

    /// Separate gate + up + SwiGLU path (default when fused kernel not available)
    #[allow(clippy::too_many_arguments)]
    fn workspace_ffn_gate_up_swiglu_separate(
        &mut self,
        hidden_buf1: &GpuBuffer<f32>,
        ffn_gate_buf: &GpuBuffer<f32>,
        ffn_up_buf: &GpuBuffer<f32>,
        ffn_act_buf: &GpuBuffer<f32>,
        layer_weights: &ValidatedLayerWeights,
        intermediate_dim: u32,
        hidden_dim: u32,
        profiling: bool,
    ) -> Result<(), GpuError> {
        let timer = if profiling {
            self.start_brick_id(trueno::BrickId::GateProjection)
        } else {
            None
        };

        self.gemv_dispatch(
            layer_weights.ffn_gate_qtype,
            layer_weights.ffn_gate_ptr,
            hidden_buf1, ffn_gate_buf, intermediate_dim, hidden_dim,
        )?;

        self.gemv_dispatch(
            layer_weights.ffn_up_qtype,
            layer_weights.ffn_up_ptr,
            hidden_buf1, ffn_up_buf, intermediate_dim, hidden_dim,
        )?;

        if profiling {
            self.stop_brick_id(timer, 1);
        }

        self.fused_swiglu_into(ffn_gate_buf, ffn_up_buf, ffn_act_buf, intermediate_dim)?;
        Ok(())
    }
}
