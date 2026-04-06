impl CudaExecutor {
    /// Phase 2: Attention + output projection + residuals + FFN (batched)
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(crate) fn batched_attn_ffn_phase(
        &mut self,
        input: &GpuBuffer<f32>,
        hidden_buf1: &GpuBuffer<f32>,
        hidden_buf2: &GpuBuffer<f32>,
        input_staging: &GpuBuffer<f32>,
        q_buf: &GpuBuffer<f32>,
        k_buf: &GpuBuffer<f32>,
        v_buf: &GpuBuffer<f32>,
        attn_out_buf: &GpuBuffer<f32>,
        ffn_gate_buf: &GpuBuffer<f32>,
        ffn_up_buf: &GpuBuffer<f32>,
        ffn_act_buf: &GpuBuffer<f32>,
        q_buf_ptr: u64,
        k_buf_ptr: u64,
        v_buf_ptr: u64,
        attn_out_ptr: u64,
        hidden_buf1_ptr: u64,
        ffn_gate_ptr: u64,
        ffn_up_ptr: u64,
        ffn_act_ptr: u64,
        layer_idx: usize,
        layer_weights: &ValidatedLayerWeights,
        m: u32,
        positions: &[u32],
        hidden_dim: u32,
        intermediate_dim: u32,
        q_dim: u32,
        kv_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // ========== 4. Attention ==========
        // PAR-119: Use batched attention if batched KV caches are initialized
        if self.batched_kv_stride > 0 && self.batched_kv_k_caches.contains_key(&layer_idx) {
            let max_seq_len = self
                .batched_kv_lengths
                .iter()
                .take(m as usize)
                .copied()
                .max()
                .unwrap_or(0);

            if self.flash_decode_enabled && max_seq_len > 1024 {
                self.flash_decoding_attention_into(
                    layer_idx, q_buf, k_buf, v_buf, attn_out_buf,
                    m as usize, positions,
                )?;
            } else {
                self.batched_incremental_attention_into(
                    layer_idx, q_buf, k_buf, v_buf, attn_out_buf,
                    m as usize, positions,
                )?;
            }
        } else if self.is_prefilling && m > 1 && self.cublas_handle.is_some() {
            // PMAT-032: Parallel prefill attention via cuBLAS strided batched GEMM
            // Replaces M sequential attention calls with bulk scatter + batched GEMM
            self.prefill_attention_cublas(
                layer_idx, q_buf, k_buf, v_buf, attn_out_buf,
                q_buf_ptr, k_buf_ptr, v_buf_ptr, attn_out_ptr,
                m, q_dim, kv_dim,
            )?;
        } else {
            // Sequential attention fallback (shared KV cache)
            self.sequential_attention_loop(
                layer_idx, q_buf_ptr, k_buf_ptr, v_buf_ptr, attn_out_ptr,
                m, q_dim, kv_dim,
            )?;
        }

        // ========== 5. Output Projection (BATCHED GEMV or cuBLAS GEMM) ==========
        self.batched_gemv_or_gemm(
            layer_weights.attn_output_qtype, layer_weights.attn_output_ptr,
            attn_out_buf, hidden_buf1, attn_out_ptr, hidden_buf1_ptr,
            m, hidden_dim, q_dim,
        )?;

        // ========== 6. First Residual (PAR-114: BATCHED) ==========
        self.batched_residual_add_into(input, hidden_buf1, input_staging, hidden_dim, m)?;

        // ========== 7. Pre-FFN RMSNorm (BATCHED - PAR-112) ==========
        // PMAT-092: Fused residual+RMSNorm kernel was FALSIFIED here — 5% regression.
        // Root cause: Fused kernel restricts to (1,M) grid (one CTA per batch element for
        // RMSNorm reduction), losing 6x parallelism vs separate residual_add (6×M grid).
        // The 28 saved launches (~560μs) don't compensate for reduced BW saturation.
        self.batched_rmsnorm_ptr_into(
            input_staging,
            layer_weights.ffn_norm_ptr,
            layer_weights.ffn_norm_len,
            hidden_buf1,
            hidden_dim,
            m,
            epsilon,
        )?;

        // ========== 8. FFN Gate/Up (BATCHED GEMV or cuBLAS GEMM) ==========
        // GH-141: Fuse gate+up Q8_1 quantization when both are Q4K and DP4A is active.
        // Same input (hidden_buf1) → quantize once, launch both GEMV with shared Q8_1.
        // PMAT-056: Removed !self.is_capturing guard — DP4A kernels are pure GPU
        // kernels (no H2D copies), graph-capturable. Old guard forced FP32 fallback.
        // PMAT-061: Disable fused gate+up DP4A when HGEMM batched decode is active.
        // Individual gate/up projections go through batched_gemv_or_gemm → cuBLAS HGEMM.
        // PMAT-088b RESULT: Even with fused gate+up preserved (hybrid), HGEMM does NOT
        // beat DP4A at M=4 (260.5 vs 261.5 tok/s). FP16's 3.5x BW penalty not compensated.
        // PMAT-090: Skip fused DP4A gate+up when FP8 decode will actually fire.
        // PMAT-093: FP8 threshold raised to M>=5. At M<=4, DP4A fused gate+up is
        // faster than FP8 individual projections (saves conversion overhead + launches).
        // Only skip fused DP4A when FP8 will actually be used for these projections.
        let fp8_will_fire = self.gpu_profile.fp8_decode && m >= 5;
        let use_fused_gate_up_dp4a = layer_weights.ffn_gate_qtype == WeightQuantType::Q4K
            && layer_weights.ffn_up_qtype == WeightQuantType::Q4K
            && m >= 2
            && m <= 8
            && self.gpu_profile.q4k == crate::cuda::gpu_profile::Q4kVariant::HwDp4a
            && !self.is_prefilling
            && !self.hgemm_batched_decode_active
            && !fp8_will_fire
            && std::env::var("BATCHED_DP4A").as_deref() != Ok("0");

        if use_fused_gate_up_dp4a {
            self.batched_gate_up_dp4a_q4k_gemv_into(
                layer_weights.ffn_gate_ptr,
                layer_weights.ffn_up_ptr,
                hidden_buf1,
                ffn_gate_buf,
                ffn_up_buf,
                m,
                intermediate_dim,
                hidden_dim,
            )?;
        } else {
            self.batched_gemv_or_gemm(
                layer_weights.ffn_gate_qtype, layer_weights.ffn_gate_ptr,
                hidden_buf1, ffn_gate_buf, hidden_buf1_ptr, ffn_gate_ptr,
                m, intermediate_dim, hidden_dim,
            )?;
            self.batched_gemv_or_gemm(
                layer_weights.ffn_up_qtype, layer_weights.ffn_up_ptr,
                hidden_buf1, ffn_up_buf, hidden_buf1_ptr, ffn_up_ptr,
                m, intermediate_dim, hidden_dim,
            )?;
        }

        // ========== 9. SwiGLU (PAR-114: BATCHED) ==========
        self.batched_swiglu_into(ffn_gate_buf, ffn_up_buf, ffn_act_buf, intermediate_dim, m)?;

        // ========== 10. FFN Down (Batched DP4A / cuBLAS GEMM / BATCHED GEMV) ==========
        // GH-141: Route through batched_gemv_or_gemm for consistent DP4A dispatch
        self.batched_gemv_or_gemm(
            layer_weights.ffn_down_qtype, layer_weights.ffn_down_ptr,
            ffn_act_buf, hidden_buf1, ffn_act_ptr, hidden_buf1_ptr,
            m, hidden_dim, intermediate_dim,
        )?;

        // ========== 11. Second Residual (PAR-114: BATCHED) ==========
        self.batched_residual_add_into(input_staging, hidden_buf1, hidden_buf2, hidden_dim, m)?;

        Ok(())
    }

    /// Sequential attention: process M tokens one at a time through incremental attention.
    /// Extracted from `batched_attn_ffn_phase` for complexity reduction.
    #[allow(clippy::too_many_arguments)]
    fn sequential_attention_loop(
        &mut self,
        layer_idx: usize,
        q_buf_ptr: u64,
        k_buf_ptr: u64,
        v_buf_ptr: u64,
        attn_out_ptr: u64,
        m: u32,
        q_dim: u32,
        kv_dim: u32,
    ) -> Result<(), GpuError> {
        for seq_idx in 0..m as usize {
            let q_offset = seq_idx * q_dim as usize;
            let kv_offset = seq_idx * kv_dim as usize;
            let attn_offset = seq_idx * q_dim as usize;

            // SAFETY: q/k/v/attn_out buf ptrs are valid GPU allocs, offsets bounded by seq_idx * dim
            let q_view = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    q_buf_ptr + (q_offset * std::mem::size_of::<f32>()) as u64,
                    q_dim as usize,
                )
            };
            let k_view = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    k_buf_ptr + (kv_offset * std::mem::size_of::<f32>()) as u64,
                    kv_dim as usize,
                )
            };
            let v_view = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    v_buf_ptr + (kv_offset * std::mem::size_of::<f32>()) as u64,
                    kv_dim as usize,
                )
            };
            let attn_out_view = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    attn_out_ptr + (attn_offset * std::mem::size_of::<f32>()) as u64,
                    q_dim as usize,
                )
            };

            self.incremental_attention_into_for_capture(
                layer_idx, &q_view, &k_view, &v_view, &attn_out_view,
            )?;

            std::mem::forget(q_view);
            std::mem::forget(k_view);
            std::mem::forget(v_view);
            std::mem::forget(attn_out_view);
        }
        Ok(())
    }
}
