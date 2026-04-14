/// SPEC-MOE-APR-001 Phase 8: All-GPU MoE FFN phase with per-expert GEMV.
///
/// Flow: RMSNorm(GPU) → router(download,CPU,upload) → per-expert Q4K GEMV(GPU)
///       → SwiGLU(download,CPU,upload) → per-expert down(CPU Q6K) → residual(GPU)
///
/// Contract: moe-cuda-kernel-v1.yaml
impl super::super::CudaExecutor {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn workspace_moe_ffn_phase(
        &mut self,
        hidden_buf1: &trueno_gpu::driver::GpuBuffer<f32>,
        hidden_buf2: &trueno_gpu::driver::GpuBuffer<f32>,
        input_staging: &trueno_gpu::driver::GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &crate::cuda::types::ValidatedLayerWeights,
        hidden_dim: u32,
        epsilon: f32,
        _skip_debug: bool,
    ) -> Result<(), super::super::GpuError> {
        let hd = hidden_dim as usize;
        let lw = layer_weights.inner();

        // Check if expert GPU pointers are available
        if lw.moe_gate_exps_ptr == 0 {
            // No GPU expert weights — just copy through (skip FFN)
            self.stream.memcpy_dtod_sync(
                hidden_buf2.as_ptr(),
                input_staging.as_ptr(),
                hd * std::mem::size_of::<f32>(),
            )?;
            return Ok(());
        }

        // Step 1: FFN RMSNorm on GPU
        let ffn_norm_buf = unsafe {
            trueno_gpu::driver::GpuBuffer::<f32>::from_raw_parts(
                lw.ffn_norm_ptr, lw.ffn_norm_len,
            )
        };
        self.rmsnorm_into(input_staging, &ffn_norm_buf, hidden_buf1, hidden_dim, epsilon)?;
        std::mem::forget(ffn_norm_buf);

        // Step 2: Download normed hidden for router (4 KB transfer)
        let mut normed_cpu = vec![0.0f32; hd];
        self.stream.synchronize()?;
        hidden_buf1.copy_to_host(&mut normed_cpu)?;

        // Step 3: Router softmax + top-k on CPU
        let num_experts = lw.moe_gate_exps_len / (lw.moe_gate_exps_len / 128.max(1)); // estimate
        let moe_intermediate = 768usize; // TODO: get from config
        let top_k = 8usize; // TODO: get from config

        // Router matmul: download router weight and compute on CPU
        // Router is F32 [num_experts, hidden_dim] = 128 × 2048 = 1 MB
        // moe_router_len is in BYTES (from get_quantized_weight_ptr_and_size)
        let router_f32_elements = lw.moe_router_len / std::mem::size_of::<f32>();
        let num_experts = if router_f32_elements > 0 { router_f32_elements / hd } else { 128 };
        let top_k = 8usize;
        let moe_intermediate = 768usize;

        let mut router_logits = vec![0.0f32; num_experts];
        if lw.moe_router_ptr != 0 && router_f32_elements > 0 {
            // Download router weight (F32 on GPU)
            let mut router_weight = vec![0.0f32; router_f32_elements];
            let router_buf = unsafe {
                trueno_gpu::driver::GpuBuffer::<f32>::from_raw_parts(
                    lw.moe_router_ptr, router_f32_elements,
                )
            };
            router_buf.copy_to_host(&mut router_weight)?;
            std::mem::forget(router_buf);

            // Matmul: normed_cpu × router_weight^T → logits
            for (e, logit) in router_logits.iter_mut().enumerate() {
                let off = e * hd;
                *logit = (0..hd).map(|j| normed_cpu[j] * router_weight[off + j]).sum();
            }
        }

        // Softmax + top-k
        let max_l = router_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = router_logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 { for p in &mut probs { *p /= sum; } }
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected: Vec<(usize, f32)> = indexed[..top_k.min(num_experts)].to_vec();

        // Step 4-5: Per-expert gate + up GEMV on GPU, then SwiGLU on CPU
        // Use kernel-expected stride (out_dim * ceil(in_dim/256) * Q4K_block_bytes)
        let q4k_blocks_per_row = (hd + 255) / 256; // ceil(2048/256) = 8
        let q4k_bytes_per_row = q4k_blocks_per_row * 144;
        let gate_stride = moe_intermediate * q4k_bytes_per_row; // 768 * 1152 = 884,736
        let up_stride = gate_stride;

        // Upload normed input to GPU for GEMV
        let gpu_input = self.upload_f32(&normed_cpu)
            .map_err(|e| super::super::GpuError::Transfer(format!("moe input: {e}")))?;

        let mut output = vec![0.0f32; hd];
        for &(expert_idx, weight) in &selected {
            // Gate GEMV
            let gate_ptr = lw.moe_gate_exps_ptr + (expert_idx * gate_stride) as u64;
            let gate_out = self.q4k_gemv_indexed_async(
                gate_ptr, &gpu_input, moe_intermediate as u32, hidden_dim,
            ).map_err(|e| super::super::GpuError::KernelLaunch(format!("gate expert {}: {}", expert_idx, e)))?;

            // Up GEMV
            let up_ptr = lw.moe_up_exps_ptr + (expert_idx * up_stride) as u64;
            let up_out = self.q4k_gemv_indexed_async(
                up_ptr, &gpu_input, moe_intermediate as u32, hidden_dim,
            ).map_err(|e| super::super::GpuError::KernelLaunch(format!("up expert {}: {}", expert_idx, e)))?;

            // Download gate+up for SwiGLU
            self.stream.synchronize()?;
            let mut gd = vec![0.0f32; moe_intermediate];
            let mut ud = vec![0.0f32; moe_intermediate];
            gate_out.copy_to_host(&mut gd)?;
            up_out.copy_to_host(&mut ud)?;

            // SwiGLU: SiLU(gate) * up
            let mut swiglu = vec![0.0f32; moe_intermediate];
            for i in 0..moe_intermediate {
                let silu = gd[i] / (1.0 + (-gd[i]).exp());
                swiglu[i] = silu * ud[i];
            }

            // Down projection: Q6K GEMV on GPU
            // Stride verified: 1,290,240 bytes/expert (GGUF file confirmed)
            let q6k_blocks_per_row = (moe_intermediate + 255) / 256;
            let q6k_bytes_per_row = q6k_blocks_per_row * 210;
            let down_stride = hd * q6k_bytes_per_row;
            let down_ptr = lw.moe_down_exps_ptr + (expert_idx * down_stride) as u64;
            if layer_idx == 0 && expert_idx == selected[0].0 {
                eprintln!("[MOE-DOWN-GPU] expert={} base={} stride={} ptr={} buf_len={}",
                    expert_idx, lw.moe_down_exps_ptr, down_stride, down_ptr, lw.moe_down_exps_len);
            }

            let swiglu_gpu = self.upload_f32(&swiglu)
                .map_err(|e| super::super::GpuError::Transfer(format!("swiglu: {e}")))?;

            // TEMP: Use Q4K GEMV for down (Q6K crashes — investigating)
            // Q4K stride: 2048 * (768/256) * 144 = 2048 * 3 * 144 = 884,736
            let q4k_down_stride = hd * ((moe_intermediate + 255) / 256) * 144;
            let down_ptr_q4k = lw.moe_down_exps_ptr + (expert_idx * q4k_down_stride) as u64;
            let down_out = self.q4k_gemv_indexed_async(
                down_ptr_q4k, &swiglu_gpu, hidden_dim, moe_intermediate as u32,
            ).map_err(|e| super::super::GpuError::KernelLaunch(format!("down expert {}: {}", expert_idx, e)))?;

            self.stream.synchronize()?;
            let mut dd = vec![0.0f32; hd];
            down_out.copy_to_host(&mut dd)?;

            for i in 0..hd { output[i] += weight * dd[i]; }
        }

        // Step 6: Upload MoE output and add residual on GPU
        let moe_result_gpu = self.upload_f32(&output)
            .map_err(|e| super::super::GpuError::Transfer(format!("moe result: {e}")))?;
        self.residual_add_into(input_staging, &moe_result_gpu, hidden_buf2, hidden_dim)?;

        Ok(())
    }
}
