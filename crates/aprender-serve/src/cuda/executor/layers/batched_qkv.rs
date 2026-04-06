impl CudaExecutor {
    /// Batched QKV projection dispatch: Q4K/Q6K → batched kernel, others → sequential fallback.
    ///
    /// PMAT-046: Added Q6K batched path. Previously Q6K fell through to sequential
    /// loop (M individual kernel launches), causing 2.87x ITL overhead at c=4.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_gemv_with_fallback(
        &mut self,
        qtype: WeightQuantType,
        weight_ptr: u64,
        packed_input: &GpuBuffer<f32>,
        packed_output: &GpuBuffer<f32>,
        packed_input_ptr: u64,
        packed_output_ptr: u64,
        m: u32,
        n_per_seq: u32,
        k_per_seq: u32,
    ) -> Result<(), GpuError> {
        // realizr#220 DIAGNOSTIC: Force sequential M=1 GEMV for all qtypes
        // to isolate whether batched GEMV kernel codegen is the root cause.
        if qtype == WeightQuantType::Q4K && false {
            self.batched_q4k_gemv_into(
                weight_ptr, packed_input, packed_output, m, n_per_seq, k_per_seq,
            )
        } else if qtype == WeightQuantType::Q6K && false {
            self.batched_q6k_gemv_into(
                weight_ptr, packed_input, packed_output, m, n_per_seq, k_per_seq,
            )
        } else {
            for seq_idx in 0..m as usize {
                let in_offset = seq_idx * k_per_seq as usize;
                let out_offset = seq_idx * n_per_seq as usize;
                // SAFETY: packed_input_ptr/packed_output_ptr are valid GPU allocs, offsets bounded by m * k/n
                let input_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        packed_input_ptr + (in_offset * std::mem::size_of::<f32>()) as u64,
                        k_per_seq as usize,
                    )
                };
                // SAFETY: packed_output_ptr is valid GPU alloc, out_offset bounded by seq_idx * n
                let output_view = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        packed_output_ptr + (out_offset * std::mem::size_of::<f32>()) as u64,
                        n_per_seq as usize,
                    )
                };
                self.gemv_dispatch(qtype, weight_ptr, &input_view, &output_view, n_per_seq, k_per_seq)?;
                std::mem::forget(input_view);
                std::mem::forget(output_view);
            }
            Ok(())
        }
    }

    /// Phase 1: RMSNorm + QKV projections + bias + RoPE (batched)
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(crate) fn batched_qkv_rope_phase(
        &mut self,
        input: &GpuBuffer<f32>,
        hidden_buf1: &GpuBuffer<f32>,
        q_buf: &GpuBuffer<f32>,
        k_buf: &GpuBuffer<f32>,
        v_buf: &GpuBuffer<f32>,
        q_buf_ptr: u64,
        k_buf_ptr: u64,
        v_buf_ptr: u64,
        hidden_buf1_ptr: u64,
        _layer_idx: usize,
        layer_weights: &ValidatedLayerWeights,
        m: u32,
        positions: &[u32],
        hidden_dim: u32,
        q_dim: u32,
        kv_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // ========== 1. Pre-attention RMSNorm (BATCHED - PAR-112) ==========
        self.batched_rmsnorm_ptr_into(
            input,
            layer_weights.attn_norm_ptr,
            layer_weights.attn_norm_len,
            hidden_buf1,
            hidden_dim,
            m,
            epsilon,
        )?;

        // ========== 2. Q/K/V Projections (BATCHED GEMV or cuBLAS GEMM) ==========
        // PMAT-024: During prefill (M > threshold), use cuBLAS GEMM for Q4K weights.
        // This reads weights once instead of M/8 times, closing the 86x prefill gap.
        //
        // PMAT-054A: Fused QKV DP4A — quantize hidden_buf1 to Q8_1 ONCE, then
        // launch 3 GEMV kernels (Q, K, V) reusing the same Q8 buffer.
        // Saves 2 Q8 quantize kernel launches per layer (56 total across 28 layers).
        // PMAT-056: Removed !self.is_capturing guard — DP4A kernels are pure GPU
        // kernels (no H2D copies), so they are graph-capturable. The old guard
        // forced fallback to FP32-activation GEMV during capture, 2x slower.
        // PMAT-088b: Keep fused QKV DP4A even when HGEMM batched decode is active.
        // Fused QKV quantizes input once, launches 3 GEMV with shared Q8 — saves 2 Q8
        // quantize launches per layer (56 total across 28 layers). HGEMM benefits are
        // only significant for non-fused paths (output proj, down proj).
        // PMAT-088b RESULT: Confirmed — fused QKV DP4A stays active regardless of HGEMM.
        // Hybrid (fused QKV DP4A + HGEMM output/down) = 260.5 vs full DP4A 261.5 tok/s.
        // HGEMM crossover fully falsified at M=4 on RTX 4060L.
        // PMAT-291: Skip fused QKV DP4A when FP8 cuBLASLt will fire.
        // At M>=5, FP8 via batched_gemv_or_gemm is faster than fused DP4A
        // (same pattern as ffn gate+up in batched_ffn.rs). Graph dispatch
        // bypassed this and got +7.8% at c=4 — this is the direct fix.
        let fp8_will_fire = self.gpu_profile.fp8_decode && m >= 5;
        let use_fused_qkv_dp4a = layer_weights.attn_q_qtype == WeightQuantType::Q4K
            && layer_weights.attn_k_qtype == WeightQuantType::Q4K
            && layer_weights.attn_v_qtype == WeightQuantType::Q4K
            && m >= 2
            && m <= 8
            && self.gpu_profile.q4k == crate::cuda::gpu_profile::Q4kVariant::HwDp4a
            && !self.is_prefilling
            && !fp8_will_fire
            && std::env::var("BATCHED_DP4A").as_deref() != Ok("0");

        if use_fused_qkv_dp4a {
            self.batched_qkv_dp4a_q4k_gemv_into(
                layer_weights.attn_q_ptr,
                layer_weights.attn_k_ptr,
                layer_weights.attn_v_ptr,
                hidden_buf1,
                q_buf,
                k_buf,
                v_buf,
                m,
                q_dim,
                kv_dim,
                hidden_dim,
            )?;
        } else {
            self.batched_gemv_or_gemm(
                layer_weights.attn_q_qtype, layer_weights.attn_q_ptr,
                hidden_buf1, q_buf, hidden_buf1_ptr, q_buf_ptr,
                m, q_dim, hidden_dim,
            )?;
            self.batched_gemv_or_gemm(
                layer_weights.attn_k_qtype, layer_weights.attn_k_ptr,
                hidden_buf1, k_buf, hidden_buf1_ptr, k_buf_ptr,
                m, kv_dim, hidden_dim,
            )?;
            self.batched_gemv_or_gemm(
                layer_weights.attn_v_qtype, layer_weights.attn_v_ptr,
                hidden_buf1, v_buf, hidden_buf1_ptr, v_buf_ptr,
                m, kv_dim, hidden_dim,
            )?;
        }

        // ========== 2b. QKV Bias (PMAT-046: batched broadcast) ==========
        // Single launch per bias vector: adds bias[dim] to packed[M×dim] in-place.
        // Eliminates 3×M sequential launches per layer (was 12 for M=4, now 3).
        if layer_weights.attn_q_bias_len > 0 {
            self.batched_bias_broadcast_add(
                q_buf_ptr, layer_weights.attn_q_bias_ptr, q_dim, m,
            )?;
        }
        if layer_weights.attn_k_bias_len > 0 {
            self.batched_bias_broadcast_add(
                k_buf_ptr, layer_weights.attn_k_bias_ptr, kv_dim, m,
            )?;
        }
        if layer_weights.attn_v_bias_len > 0 {
            self.batched_bias_broadcast_add(
                v_buf_ptr, layer_weights.attn_v_bias_ptr, kv_dim, m,
            )?;
        }

        // ========== 3. RoPE on Q/K (PAR-114) ==========
        let num_heads = self.kv_num_heads as u32;
        let num_kv_heads = self.kv_num_kv_heads as u32;
        let head_dim = self.kv_head_dim as u32;
        let theta = self.rope_theta;

        let positions_buf_ptr = self
            .workspace
            .positions_buf
            .as_ref()
            .ok_or_else(|| {
                GpuError::InvalidLaunchConfig("PAR-114: positions_buf not initialized".to_string())
            })?
            .as_ptr();
        // SAFETY: positions_buf_ptr is valid GPU allocation from workspace.positions_buf
        let mut positions_buf =
            unsafe { GpuBuffer::<u32>::from_raw_parts(positions_buf_ptr, m as usize) };

        // PMAT-045: Skip copy_from_host during CUDA graph capture — positions are
        // already on device from batched_graph_positions_buf (pre-loaded before capture).
        // cuMemcpyHtoD during capture causes CUDA_ERROR_ILLEGAL_ADDRESS.
        if !self.is_capturing {
            let positions_u32: Vec<u32> = positions.to_vec();
            positions_buf.copy_from_host(&positions_u32).map_err(|e| {
                GpuError::Transfer(format!(
                    "PMAT-088c positions_buf: host={} device_view={} workspace_buf={}: {e}",
                    positions_u32.len(),
                    m,
                    self.workspace.positions_buf.as_ref().map_or(0, |b| b.len()),
                ))
            })?;
        }

        if self.rope_type == 2 {
            // PMAT-046: Batched NEOX RoPE (all M sequences in one launch per Q/K)
            // Replaces 2×M sequential launches with 2 launches.
            self.batched_rope_neox_into(q_buf, q_buf, &positions_buf, num_heads, head_dim, m, theta)?;
            self.batched_rope_neox_into(k_buf, k_buf, &positions_buf, num_kv_heads, head_dim, m, theta)?;
        } else {
            // PAR-114: Batched RoPE (all M sequences in one launch)
            self.batched_rope_into(q_buf, q_buf, &positions_buf, num_heads, head_dim, m, theta)?;
            self.batched_rope_into(k_buf, k_buf, &positions_buf, num_kv_heads, head_dim, m, theta)?;
        }

        std::mem::forget(positions_buf);
        Ok(())
    }
}
