impl CudaExecutor {
    /// Phase 1-2: Pre-attention RMSNorm + Q/K/V projections + bias + RoPE
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn workspace_qkv_rope_phase(
        &mut self,
        input: &GpuBuffer<f32>,
        hidden_buf1: &GpuBuffer<f32>,
        q_buf: &GpuBuffer<f32>,
        k_buf: &GpuBuffer<f32>,
        v_buf: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &ValidatedLayerWeights,
        hidden_dim: u32,
        q_dim: u32,
        kv_dim: u32,
        epsilon: f32,
        position: u32,
        skip_debug: bool,
        profiling: bool,
    ) -> Result<(), GpuError> {
        // 1. Pre-attention RMSNorm: input -> hidden_buf1 (normed)
        let timer_rmsnorm1 = if profiling {
            self.start_brick_id(trueno::BrickId::RmsNorm)
        } else {
            None
        };
        self.rmsnorm_ptr_into(
            input,
            layer_weights.attn_norm_ptr,
            layer_weights.attn_norm_len,
            hidden_buf1,
            hidden_dim,
            epsilon,
        )?;
        if profiling {
            self.stop_brick_id(timer_rmsnorm1, 1);
        }

        // GH-559: Dump GPU input sum-of-squares for Layer 0 to verify RMS computation
        if !skip_debug && layer_idx == 0 {
            self.stream.synchronize()?;
            let mut input_host = vec![0.0f32; hidden_dim as usize];
            input.copy_to_host(&mut input_host)?;
            let input_sum: f32 = input_host.iter().sum();
            let input_sq_sum: f32 = input_host.iter().map(|x| x * x).sum();
            let cpu_rms = (input_sq_sum / hidden_dim as f32 + epsilon).sqrt();
            eprintln!(
                "[GH-559-RMS] L0 input: sum={:.6}, sq_sum={:.10}, cpu_rms={:.10}, first5={:?}",
                input_sum, input_sq_sum, cpu_rms, &input_host[..5.min(input_host.len())]
            );
        }

        // PAR-058-DEBUG: Check after RMSNorm (skip during graph capture)
        // GH-559: Extended to layers 10-12 to debug no-op layers
        if !skip_debug && (layer_idx < 4 || (layer_idx >= 10 && layer_idx <= 12)) {
            self.stream.synchronize()?;
            let mut rmsnorm_out = vec![0.0f32; hidden_buf1.len()];
            hidden_buf1.copy_to_host(&mut rmsnorm_out)?;
            let nan_count = rmsnorm_out.iter().filter(|x| x.is_nan()).count();
            if nan_count > 0 {
                eprintln!(
                    "[PAR-058-L{}] RMSNorm output has {} NaN",
                    layer_idx, nan_count
                );
            } else {
                eprintln!(
                    "[PAR-058-L{}] RMSNorm OK, first 3: {:?}",
                    layer_idx,
                    &rmsnorm_out[..3.min(rmsnorm_out.len())]
                );
            }
        }

        // 2. Q/K/V projections using indexed pointers -> workspace buffers
        let timer_qkv = if profiling {
            self.start_brick_id(trueno::BrickId::QkvProjection)
        } else {
            None
        };

        // CORRECTNESS-011: Debug Q4K GEMV parameters for layer 0
        if !skip_debug && layer_idx == 0 && layer_weights.attn_q_qtype == WeightQuantType::Q4K {
            self.stream.synchronize()?;
            let mut input_check = vec![0.0f32; hidden_buf1.len()];
            hidden_buf1.copy_to_host(&mut input_check)?;
            eprintln!(
                "[CORRECTNESS-011-L0] Q4K GEMV params: n={}, k={}",
                q_dim, hidden_dim
            );
            eprintln!(
                "[CORRECTNESS-011-L0] Input (hidden_buf1): first 5 = {:?}",
                &input_check[..5.min(input_check.len())]
            );
            eprintln!(
                "[CORRECTNESS-011-L0] Weight ptr = {:#x}, len = {}",
                layer_weights.attn_q_ptr, layer_weights.attn_q_len
            );
        }

        // PMAT-027: Invalidate Q8 cache — hidden_buf1 was just written by RMSNorm.
        // Q, K, V all read the same hidden_buf1; first GEMV quantizes, rest reuse.
        self.q8_activation_valid = false;

        // GH-288: Fused QKV when all three weights are Q4K and HW DP4A available
        let use_fused_qkv = self.gpu_profile.fused_gate_up  // reuse same capability flag
            && layer_weights.attn_q_qtype == WeightQuantType::Q4K
            && layer_weights.attn_k_qtype == WeightQuantType::Q4K
            && layer_weights.attn_v_qtype == WeightQuantType::Q4K;

        if use_fused_qkv {
            self.fused_qkv_hw_dp4a_q4k_gemv_into(
                layer_weights.attn_q_ptr,
                layer_weights.attn_k_ptr,
                layer_weights.attn_v_ptr,
                hidden_buf1, q_buf, k_buf, v_buf,
                hidden_dim, q_dim, kv_dim,
            )?;
        } else {
            // Fallback: separate Q, K, V projections
            self.gemv_dispatch(
                layer_weights.attn_q_qtype,
                layer_weights.attn_q_ptr,
                hidden_buf1, q_buf, q_dim, hidden_dim,
            )?;

            self.gemv_dispatch(
                layer_weights.attn_k_qtype,
                layer_weights.attn_k_ptr,
                hidden_buf1, k_buf, kv_dim, hidden_dim,
            )?;

            self.gemv_dispatch(
                layer_weights.attn_v_qtype,
                layer_weights.attn_v_ptr,
                hidden_buf1, v_buf, kv_dim, hidden_dim,
            )?;
        }

        // CORRECTNESS-011: Debug Q output for Q4K layer 0
        if !skip_debug && layer_idx == 0 && layer_weights.attn_q_qtype == WeightQuantType::Q4K {
            self.stream.synchronize()?;
            let mut q_check = vec![0.0f32; q_buf.len()];
            q_buf.copy_to_host(&mut q_check)?;
            eprintln!(
                "[CORRECTNESS-011-L0] Q output: first 5 = {:?}",
                &q_check[..5.min(q_check.len())]
            );
        }

        // GQA-DEBUG: Print K qtype for debugging
        if !skip_debug && layer_idx == 0 {
            eprintln!(
                "[GQA-DEBUG-GPU-L0] K qtype = {:?}, ptr = {:#x}, len = {}",
                layer_weights.attn_k_qtype, layer_weights.attn_k_ptr, layer_weights.attn_k_len
            );
        }

        if profiling {
            self.stop_brick_id(timer_qkv, 1);
        }

        // BIAS-FIX: Add QKV bias after GEMV (Qwen2.5 models have QKV bias)
        self.apply_qkv_bias(q_buf, k_buf, v_buf, layer_idx, layer_weights, q_dim, kv_dim, skip_debug)?;

        // GH-279: Per-head QK RMSNorm (Qwen3) — after bias, before RoPE
        self.apply_qk_norm(q_buf, k_buf, layer_weights, epsilon)?;

        // PAR-058-DEBUG: Check Q/K/V after projections (skip during graph capture)
        // GH-559: Extended to layers 10-12 to debug no-op layers
        if !skip_debug
            && (layer_idx < 4
                || layer_idx == 5
                || (layer_idx >= 10 && layer_idx <= 12))
        {
            self.stream.synchronize()?;
            eprintln!(
                "[PAR-058-L{}] Weight ptrs: Q={:#x}, K={:#x}, V={:#x}",
                layer_idx,
                layer_weights.attn_q_ptr,
                layer_weights.attn_k_ptr,
                layer_weights.attn_v_ptr
            );
            eprintln!(
                "[PAR-058-L{}] Weight lens: Q={}, K={}, V={}",
                layer_idx,
                layer_weights.attn_q_len,
                layer_weights.attn_k_len,
                layer_weights.attn_v_len
            );

            let mut q_out = vec![0.0f32; q_buf.len()];
            q_buf.copy_to_host(&mut q_out)?;
            let q_nan = q_out.iter().filter(|x| x.is_nan()).count();
            if q_nan > 0 {
                eprintln!("[PAR-058-L{}] Q has {} NaN", layer_idx, q_nan);
            } else {
                eprintln!(
                    "[PAR-058-L{}] Q OK, first 3: {:?}",
                    layer_idx,
                    &q_out[..3.min(q_out.len())]
                );
            }
            let mut k_out = vec![0.0f32; k_buf.len()];
            k_buf.copy_to_host(&mut k_out)?;
            let k_nan = k_out.iter().filter(|x| x.is_nan()).count();
            let k_max = k_out.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let k_min = k_out.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            eprintln!(
                "[PAR-058-L{}] K stats: nan={}, min={:.4}, max={:.4}, first 5: {:?}",
                layer_idx, k_nan, k_min, k_max,
                &k_out[..5.min(k_out.len())]
            );
            let mut v_out = vec![0.0f32; v_buf.len()];
            v_buf.copy_to_host(&mut v_out)?;
            let v_nan = v_out.iter().filter(|x| x.is_nan()).count();
            let v_max = v_out.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let v_min = v_out.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            eprintln!(
                "[PAR-058-L{}] V stats: nan={}, min={:.4}, max={:.4}, first 5: {:?}",
                layer_idx, v_nan, v_min, v_max,
                &v_out[..5.min(v_out.len())]
            );
        }

        // PAR-060: Apply RoPE to Q and K before attention using GPU kernel
        self.apply_rope_to_qk(q_buf, k_buf, layer_idx, position, skip_debug, profiling)?;

        Ok(())
    }

    /// Apply QKV bias after GEMV (Qwen2.5 models have QKV bias).
    /// Only adds bias if bias exists (len > 0).
    #[allow(clippy::too_many_arguments)]
    fn apply_qkv_bias(
        &mut self,
        q_buf: &GpuBuffer<f32>,
        k_buf: &GpuBuffer<f32>,
        v_buf: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &ValidatedLayerWeights,
        q_dim: u32,
        kv_dim: u32,
        skip_debug: bool,
    ) -> Result<(), GpuError> {
        if layer_weights.attn_q_bias_len > 0 {
            // SAFETY: bias_ptr is valid device memory owned by bias_cache
            let q_bias_buf = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    layer_weights.attn_q_bias_ptr,
                    layer_weights.attn_q_bias_len,
                )
            };
            self.residual_add_into(q_buf, &q_bias_buf, q_buf, q_dim)?;
            std::mem::forget(q_bias_buf);

            if !skip_debug && (layer_idx == 0 || layer_idx == 4 || layer_idx == 5) {
                self.stream.synchronize()?;
                let mut q_check = vec![0.0f32; q_buf.len()];
                q_buf.copy_to_host(&mut q_check)?;
                eprintln!(
                    "[BIAS-FIX-L{}] Q after bias: first 5 = {:?}",
                    layer_idx,
                    &q_check[..5.min(q_check.len())]
                );
            }
        }
        if layer_weights.attn_k_bias_len > 0 {
            // GQA-DEBUG: Print K values BEFORE bias to isolate issue
            if !skip_debug && layer_idx == 0 {
                self.stream.synchronize()?;
                let mut k_pre = vec![0.0f32; k_buf.len()];
                k_buf.copy_to_host(&mut k_pre)?;
                eprintln!(
                    "[GQA-DEBUG-L0] K BEFORE bias: first 5 = {:?}",
                    &k_pre[..5.min(k_pre.len())]
                );
                // SAFETY: bias ptr/len from validated layer_weights, GPU memory pre-allocated
                let k_bias_buf_check = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        layer_weights.attn_k_bias_ptr,
                        layer_weights.attn_k_bias_len,
                    )
                };
                let mut k_bias_vals = vec![0.0f32; k_bias_buf_check.len()];
                k_bias_buf_check.copy_to_host(&mut k_bias_vals)?;
                eprintln!(
                    "[GQA-DEBUG-L0] K bias values: first 5 = {:?}",
                    &k_bias_vals[..5.min(k_bias_vals.len())]
                );
                std::mem::forget(k_bias_buf_check);
            }

            // SAFETY: Pointer and length from layer_weights validated at model load time
            let k_bias_buf = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    layer_weights.attn_k_bias_ptr,
                    layer_weights.attn_k_bias_len,
                )
            };
            self.residual_add_into(k_buf, &k_bias_buf, k_buf, kv_dim)?;
            std::mem::forget(k_bias_buf);

            if !skip_debug && (layer_idx == 0 || layer_idx == 4 || layer_idx == 5) {
                self.stream.synchronize()?;
                let mut k_check = vec![0.0f32; k_buf.len()];
                k_buf.copy_to_host(&mut k_check)?;
                eprintln!(
                    "[BIAS-FIX-L{}] K after bias: first 5 = {:?}",
                    layer_idx,
                    &k_check[..5.min(k_check.len())]
                );
            }
        }
        if layer_weights.attn_v_bias_len > 0 {
            // SAFETY: Pointer and length from layer_weights validated at model load time
            let v_bias_buf = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    layer_weights.attn_v_bias_ptr,
                    layer_weights.attn_v_bias_len,
                )
            };
            self.residual_add_into(v_buf, &v_bias_buf, v_buf, kv_dim)?;
            std::mem::forget(v_bias_buf);

            if !skip_debug && (layer_idx == 0 || layer_idx == 4 || layer_idx == 5) {
                self.stream.synchronize()?;
                let mut v_check = vec![0.0f32; v_buf.len()];
                v_buf.copy_to_host(&mut v_check)?;
                eprintln!(
                    "[BIAS-FIX-L{}] V after bias: first 5 = {:?}",
                    layer_idx,
                    &v_check[..5.min(v_check.len())]
                );
            }
        }
        Ok(())
    }

    /// GH-279: Apply per-head QK RMSNorm (Qwen3) after bias, before RoPE.
    /// Matches CPU path at single_part_02.rs:211-225.
    /// No-op if the model doesn't have QkNorm weights (len == 0).
    fn apply_qk_norm(
        &mut self,
        q_buf: &GpuBuffer<f32>,
        k_buf: &GpuBuffer<f32>,
        layer_weights: &ValidatedLayerWeights,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        if layer_weights.attn_q_norm_len > 0 {
            // SAFETY: Pointer valid from rmsnorm_cache, length verified at model load time
            let q_norm_buf = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    layer_weights.attn_q_norm_ptr,
                    layer_weights.attn_q_norm_len,
                )
            };
            let num_heads = self.kv_num_heads as u32;
            let head_dim = self.kv_head_dim as u32;
            self.per_head_rmsnorm_into(q_buf, &q_norm_buf, q_buf, head_dim, num_heads, epsilon)?;
            std::mem::forget(q_norm_buf);
        }
        if layer_weights.attn_k_norm_len > 0 {
            // SAFETY: Pointer valid from rmsnorm_cache, length verified at model load time
            let k_norm_buf = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    layer_weights.attn_k_norm_ptr,
                    layer_weights.attn_k_norm_len,
                )
            };
            let num_kv_heads = self.kv_num_kv_heads as u32;
            let head_dim = self.kv_head_dim as u32;
            self.per_head_rmsnorm_into(k_buf, &k_norm_buf, k_buf, head_dim, num_kv_heads, epsilon)?;
            std::mem::forget(k_norm_buf);
        }
        Ok(())
    }

    /// PAR-060: Apply RoPE (Rotary Position Embedding) to Q and K buffers.
    /// Supports both NORM (adjacent pairs) and NEOX (split halves) styles,
    /// and both direct position values and indirect position buffers for CUDA graph capture.
    fn apply_rope_to_qk(
        &mut self,
        q_buf: &GpuBuffer<f32>,
        k_buf: &GpuBuffer<f32>,
        _layer_idx: usize,
        position: u32,
        skip_debug: bool,
        profiling: bool,
    ) -> Result<(), GpuError> {
        let timer_rope = if profiling {
            self.start_brick_id(trueno::BrickId::RopeEmbedding)
        } else {
            None
        };

        let num_heads = self.kv_num_heads as u32;
        let num_kv_heads = self.kv_num_kv_heads as u32;
        let head_dim = self.kv_head_dim as u32;
        let theta = self.rope_theta;

        if skip_debug && self.position_buf.is_some() {
            self.apply_rope_indirect(q_buf, k_buf, num_heads, num_kv_heads, head_dim, theta)?;
        } else {
            self.apply_rope_direct(q_buf, k_buf, position, num_heads, num_kv_heads, head_dim, theta)?;
        }

        if profiling {
            self.stop_brick_id(timer_rope, 1);
        }
        Ok(())
    }

    /// Apply RoPE using indirect position buffer (CUDA graph capture mode).
    fn apply_rope_indirect(
        &mut self,
        q_buf: &GpuBuffer<f32>,
        k_buf: &GpuBuffer<f32>,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), GpuError> {
        let pos_buf_ptr = self
            .position_buf
            .as_ref()
            .expect("position_buf must be initialized")
            .as_ptr();
        let pos_buf_len = self
            .position_buf
            .as_ref()
            .expect("position_buf must be initialized")
            .len();
        // SAFETY: Pointer valid from allocation, length verified, used within scope
        let pos_buf = unsafe { GpuBuffer::<u32>::from_raw_parts(pos_buf_ptr, pos_buf_len) };
        if self.rope_type == 2 {
            self.rope_neox_indirect_into(q_buf, q_buf, &pos_buf, num_heads, head_dim, theta)?;
            self.rope_neox_indirect_into(k_buf, k_buf, &pos_buf, num_kv_heads, head_dim, theta)?;
        } else {
            self.rope_indirect_into(q_buf, q_buf, &pos_buf, num_heads, head_dim, theta)?;
            self.rope_indirect_into(k_buf, k_buf, &pos_buf, num_kv_heads, head_dim, theta)?;
        }
        std::mem::forget(pos_buf);
        Ok(())
    }

    /// Apply RoPE using direct position value (normal mode).
    fn apply_rope_direct(
        &mut self,
        q_buf: &GpuBuffer<f32>,
        k_buf: &GpuBuffer<f32>,
        position: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), GpuError> {
        if self.rope_type == 2 {
            self.rope_neox_into(q_buf, q_buf, position, num_heads, head_dim, theta)?;
            self.rope_neox_into(k_buf, k_buf, position, num_kv_heads, head_dim, theta)?;
        } else {
            self.rope_into(q_buf, q_buf, position, num_heads, head_dim, theta)?;
            self.rope_into(k_buf, k_buf, position, num_kv_heads, head_dim, theta)?;
        }
        Ok(())
    }
}
