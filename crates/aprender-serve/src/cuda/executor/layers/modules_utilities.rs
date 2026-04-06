impl CudaExecutor {
    /// Pre-load LM head, RoPE, SwiGLU, residual, scatter, and attention modules.
    /// Split from preload_modules_for_capture sections 5-10.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn preload_lm_head_and_utility_modules(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        max_len: u32,
        _q_dim: u32,
        _kv_dim: u32,
        nw: u32,
    ) -> Result<(), GpuError> {
        // CORRECTNESS-013: Check if precise mode is requested
        static PRECISE_MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_precise = *PRECISE_MODE.get_or_init(|| {
            std::env::var("CORRECTNESS_MODE")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        // 5. LM head (hidden_dim -> vocab_size) - pre-load both Q4K and Q6K
        let mwv_lm_head_q4k_key = format!("mwv_q4k_gemv_{}_{}_{}", hidden_dim, vocab_size, nw);
        if !self.modules.contains_key(&mwv_lm_head_q4k_key) {
            let kernel_type = KernelType::MwvQ4KGemv {
                k: hidden_dim,
                n: vocab_size,
                num_warps: nw,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(mwv_lm_head_q4k_key, module);
        }
        let lm_head_q6k_key = format!("q6k_gemv_{}_{}", hidden_dim, vocab_size);
        if !self.modules.contains_key(&lm_head_q6k_key) {
            let kernel_type = KernelType::Q6KGemv {
                k: hidden_dim,
                n: vocab_size,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(lm_head_q6k_key, module);
        }
        if hidden_dim.is_multiple_of(256) {
            let coalesced_lm_head_q6k_key =
                format!("coalesced_q6k_gemv_{}_{}", hidden_dim, vocab_size);
            if !self.modules.contains_key(&coalesced_lm_head_q6k_key) {
                let kernel_type = KernelType::CoalescedQ6KGemv {
                    k: hidden_dim,
                    n: vocab_size,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(coalesced_lm_head_q6k_key, module);
            }
        }

        // 6. RoPE kernels — extracted to reduce cyclomatic complexity
        self.preload_rope_modules(num_heads, num_kv_heads, head_dim, use_precise)?;

        // 7. SwiGLU kernel
        let swiglu_key = format!("fused_swiglu_{}", intermediate_dim);
        if !self.modules.contains_key(&swiglu_key) {
            let kernel_type = KernelType::FusedSwiglu { n: intermediate_dim };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(swiglu_key, module);
        }

        // 8. Residual add kernel
        // GH-129: PTX is n-independent, use constant cache key
        let residual_key = "residual_add".to_string();
        if !self.modules.contains_key(&residual_key) {
            let kernel_type = KernelType::ResidualAdd { n: hidden_dim };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(residual_key, module);
        }

        // 9. KV cache scatter kernel
        let scatter_key = format!("kv_scatter_{}_{}", num_kv_heads, head_dim);
        if !self.modules.contains_key(&scatter_key) {
            let kernel_type = KernelType::KvCacheScatter { num_kv_heads, head_dim, max_len };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(scatter_key, module);
        }

        // 10. Incremental attention kernel (direct + indirect)
        let attn_key = format!("incremental_attention_{}_{}_{}_{}",
            max_len, head_dim, num_heads, num_kv_heads);
        if !self.modules.contains_key(&attn_key) {
            let kernel_type = KernelType::IncrementalAttention {
                max_seq_len: max_len, head_dim,
                n_heads: num_heads, n_kv_heads: num_kv_heads, indirect: false,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(attn_key, module);
        }
        let attn_indirect_key = format!("incremental_attention_indirect_{}_{}_{}_{}",
            max_len, head_dim, num_heads, num_kv_heads);
        if !self.modules.contains_key(&attn_indirect_key) {
            let kernel_type = KernelType::IncrementalAttention {
                max_seq_len: max_len, head_dim,
                n_heads: num_heads, n_kv_heads: num_kv_heads, indirect: true,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(attn_indirect_key, module);
        }

        // Multi-warp attention kernels (for head_dim > 64)
        let num_warps_per_head = 4u32;
        let multi_warp_key = format!("multi_warp_attention_{}_{}_{}_{}_{}", max_len, head_dim, num_heads, num_kv_heads, num_warps_per_head);
        if !self.modules.contains_key(&multi_warp_key) {
            let kernel_type = KernelType::MultiWarpAttention {
                max_seq_len: max_len, head_dim,
                n_heads: num_heads, n_kv_heads: num_kv_heads,
                num_warps_per_head, indirect: false,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(multi_warp_key, module);
        }
        let multi_warp_indirect_key = format!("multi_warp_attention_indirect_{}_{}_{}_{}_{}", max_len, head_dim, num_heads, num_kv_heads, num_warps_per_head);
        if !self.modules.contains_key(&multi_warp_indirect_key) {
            let kernel_type = KernelType::MultiWarpAttention {
                max_seq_len: max_len, head_dim,
                n_heads: num_heads, n_kv_heads: num_kv_heads,
                num_warps_per_head, indirect: true,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(multi_warp_indirect_key, module);
        }

        // 11. Batched prefill kernels (GH-129)
        self.preload_batched_prefill_modules(
            hidden_dim, intermediate_dim, num_heads, num_kv_heads, head_dim,
        )?;

        // 12. Flash Decoding chunk + reduce kernels (PAR-118)
        // Pre-load to avoid PTX compilation during graph capture (error 901).
        if self.flash_decode_enabled {
            self.preload_flash_decoding_modules(max_len, head_dim, num_heads, num_kv_heads)?;
        }

        if verbose() {
            eprintln!(
                "[PAR-054-FIX] Pre-loaded {} kernel modules for {} layers",
                self.modules.len(), num_layers
            );
        }
        Ok(())
    }

    /// Pre-load RoPE base + indirect kernel variants.
    fn preload_rope_modules(
        &mut self,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        use_precise: bool,
    ) -> Result<(), GpuError> {
        let theta = self.rope_theta;

        let rope_q_key = format!("rope_{}_{}", num_heads, head_dim);
        if !self.modules.contains_key(&rope_q_key) {
            let kernel_type = KernelType::Rope { num_heads, head_dim, theta };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(rope_q_key, module);
        }
        let rope_k_key = format!("rope_{}_{}", num_kv_heads, head_dim);
        if !self.modules.contains_key(&rope_k_key) {
            let kernel_type = KernelType::Rope { num_heads: num_kv_heads, head_dim, theta };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(rope_k_key, module);
        }

        let rope_q_indirect_key = format!("rope_indirect_{}_{}", num_heads, head_dim);
        if !self.modules.contains_key(&rope_q_indirect_key) {
            let kernel_type = KernelType::RopeIndirect { num_heads, head_dim, theta };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(rope_q_indirect_key, module);
        }
        let rope_k_indirect_key = format!("rope_indirect_{}_{}", num_kv_heads, head_dim);
        if !self.modules.contains_key(&rope_k_indirect_key) {
            let kernel_type = KernelType::RopeIndirect { num_heads: num_kv_heads, head_dim, theta };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(rope_k_indirect_key, module);
        }

        if self.rope_type == 2 {
            self.preload_rope_neox_modules(num_heads, num_kv_heads, head_dim, use_precise)?;
        }

        Ok(())
    }

    /// Pre-load RoPE NEOX variants (Qwen2.5 rope_type=2).
    fn preload_rope_neox_modules(
        &mut self,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        use_precise: bool,
    ) -> Result<(), GpuError> {
        let theta = self.rope_theta;

        if use_precise {
            let rope_precise_q_indirect_key = format!("rope_precise_indirect_{}_{}", num_heads, head_dim);
            if !self.modules.contains_key(&rope_precise_q_indirect_key) {
                let kernel_type = KernelType::PreciseRopeNeoxIndirect { num_heads, head_dim, theta };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(rope_precise_q_indirect_key, module);
            }
            let rope_precise_k_indirect_key = format!("rope_precise_indirect_{}_{}", num_kv_heads, head_dim);
            if !self.modules.contains_key(&rope_precise_k_indirect_key) {
                let kernel_type = KernelType::PreciseRopeNeoxIndirect { num_heads: num_kv_heads, head_dim, theta };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(rope_precise_k_indirect_key, module);
            }
        } else {
            let rope_neox_q_indirect_key = format!("rope_neox_indirect_{}_{}", num_heads, head_dim);
            if !self.modules.contains_key(&rope_neox_q_indirect_key) {
                let kernel_type = KernelType::RopeNeoxIndirect { num_heads, head_dim, theta };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(rope_neox_q_indirect_key, module);
            }
            let rope_neox_k_indirect_key = format!("rope_neox_indirect_{}_{}", num_kv_heads, head_dim);
            if !self.modules.contains_key(&rope_neox_k_indirect_key) {
                let kernel_type = KernelType::RopeNeoxIndirect { num_heads: num_kv_heads, head_dim, theta };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(rope_neox_k_indirect_key, module);
            }
        }
        let rope_neox_q_key = format!("rope_neox_{}_{}", num_heads, head_dim);
        if !self.modules.contains_key(&rope_neox_q_key) {
            let kernel_type = KernelType::RopeNeox { num_heads, head_dim, theta };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(rope_neox_q_key, module);
        }
        let rope_neox_k_key = format!("rope_neox_{}_{}", num_kv_heads, head_dim);
        if !self.modules.contains_key(&rope_neox_k_key) {
            let kernel_type = KernelType::RopeNeox { num_heads: num_kv_heads, head_dim, theta };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(rope_neox_k_key, module);
        }

        Ok(())
    }

    /// GH-129: Pre-load batched kernels used by batched prefill path.
    ///
    /// These kernels embed dimensional constants as immediates but NOT `batch_size`
    /// (which only affects the grid dimension). Pre-loading prevents JIT
    /// compilation on memory-constrained devices (Jetson unified memory).
    #[allow(clippy::too_many_arguments)]
    fn preload_batched_prefill_modules(
        &mut self,
        hidden_dim: u32,
        intermediate_dim: u32,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> Result<(), GpuError> {
        // Batched RMSNorm (called twice per layer: attn_norm + ffn_norm)
        let batched_rmsnorm_key = format!("batched_rmsnorm_vectorized_{}", hidden_dim);
        if !self.modules.contains_key(&batched_rmsnorm_key) {
            let kernel_type = KernelType::BatchedVectorizedRmsNorm {
                hidden_size: hidden_dim, batch_size: 1, epsilon: 1e-5,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(batched_rmsnorm_key, module);
        }

        // Batched RoPE (for non-NEOX rope_type; batch_size is grid-only)
        let batched_rope_q_key = format!("batched_rope_{}_{}", num_heads, head_dim);
        if !self.modules.contains_key(&batched_rope_q_key) {
            let kernel_type = KernelType::BatchedRope {
                num_heads, head_dim, batch_size: 1, theta: self.rope_theta,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(batched_rope_q_key, module);
        }
        let batched_rope_k_key = format!("batched_rope_{}_{}", num_kv_heads, head_dim);
        if !self.modules.contains_key(&batched_rope_k_key) {
            let kernel_type = KernelType::BatchedRope {
                num_heads: num_kv_heads, head_dim, batch_size: 1, theta: self.rope_theta,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(batched_rope_k_key, module);
        }

        // Batched residual add
        let batched_residual_key = format!("batched_residual_add_{}", hidden_dim);
        if !self.modules.contains_key(&batched_residual_key) {
            let kernel_type = KernelType::BatchedResidualAdd { n: hidden_dim, batch_size: 1 };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(batched_residual_key, module);
        }

        // Batched SwiGLU
        let batched_swiglu_key = format!("batched_swiglu_{}", intermediate_dim);
        if !self.modules.contains_key(&batched_swiglu_key) {
            let kernel_type = KernelType::BatchedSwiglu { n: intermediate_dim, batch_size: 1 };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(batched_swiglu_key, module);
        }

        Ok(())
    }

    /// Pre-load RMSNorm kernel (precise or vectorized based on CORRECTNESS_MODE).
    fn preload_rmsnorm_module(&mut self, hidden_dim: u32) -> Result<(), GpuError> {
        static PRECISE_MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_precise = *PRECISE_MODE.get_or_init(|| {
            std::env::var("CORRECTNESS_MODE")
                .map(|v| v == "1")
                .unwrap_or(false)
        });

        if use_precise {
            let rmsnorm_key = format!("rmsnorm_precise_{}", hidden_dim);
            if !self.modules.contains_key(&rmsnorm_key) {
                let kernel_type = KernelType::PreciseRmsNorm {
                    hidden_size: hidden_dim,
                    epsilon: 1e-5,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(rmsnorm_key, module);
            }
        } else {
            let rmsnorm_key = format!("rmsnorm_vectorized_{}", hidden_dim);
            if !self.modules.contains_key(&rmsnorm_key) {
                let kernel_type = KernelType::VectorizedRmsNorm {
                    hidden_size: hidden_dim,
                    epsilon: 1e-5,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(rmsnorm_key, module);
            }
        }
        Ok(())
    }

    /// Pre-load all GEMV kernels: Q4K, Q5_0, Q6K, Q8_0 for Q/K/V, output, and FFN.
    #[allow(clippy::too_many_lines)]
    fn preload_gemv_modules(
        &mut self,
        hidden_dim: u32,
        intermediate_dim: u32,
        q_dim: u32,
        kv_dim: u32,
        nw: u32,
    ) -> Result<(), GpuError> {
        // MWV Q4K for Q and KV projections
        let mwv_q4k_q_key = format!("mwv_q4k_gemv_{}_{}_{}", hidden_dim, q_dim, nw);
        if !self.modules.contains_key(&mwv_q4k_q_key) {
            let kernel_type = KernelType::MwvQ4KGemv {
                k: hidden_dim, n: q_dim, num_warps: nw,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(mwv_q4k_q_key, module);
        }
        let mwv_q4k_kv_key = format!("mwv_q4k_gemv_{}_{}_{}", hidden_dim, kv_dim, nw);
        if !self.modules.contains_key(&mwv_q4k_kv_key) {
            let kernel_type = KernelType::MwvQ4KGemv {
                k: hidden_dim, n: kv_dim, num_warps: nw,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(mwv_q4k_kv_key, module);
        }

        // Q5_0 GEMV (for Qwen 0.5B which uses Q5_0 for Q/K)
        let q5_0_q_key = format!("q5_0_gemv_{}_{}", hidden_dim, q_dim);
        if !self.modules.contains_key(&q5_0_q_key) {
            let kernel_type = KernelType::Q5_0Gemv { k: hidden_dim, n: q_dim };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(q5_0_q_key, module);
        }
        let q5_0_kv_key = format!("q5_0_gemv_{}_{}", hidden_dim, kv_dim);
        if !self.modules.contains_key(&q5_0_kv_key) {
            let kernel_type = KernelType::Q5_0Gemv { k: hidden_dim, n: kv_dim };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(q5_0_kv_key, module);
        }

        // Q6K GEMV — original + coalesced variants
        self.preload_q6k_gemv_pair(hidden_dim, q_dim)?;
        self.preload_q6k_gemv_pair(hidden_dim, kv_dim)?;

        // Q8_0 GEMV
        let q8_0_q_key = format!("q8_0_gemv_{}_{}", hidden_dim, q_dim);
        if !self.modules.contains_key(&q8_0_q_key) {
            let kernel_type = KernelType::Q8_0Gemv { k: hidden_dim, n: q_dim };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(q8_0_q_key, module);
        }
        let q8_0_kv_key = format!("q8_0_gemv_{}_{}", hidden_dim, kv_dim);
        if !self.modules.contains_key(&q8_0_kv_key) {
            let kernel_type = KernelType::Q8_0Gemv { k: hidden_dim, n: kv_dim };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(q8_0_kv_key, module);
        }

        // Output projection (q_dim -> hidden_dim)
        let mwv_q4k_o_key = format!("mwv_q4k_gemv_{}_{}_{}", q_dim, hidden_dim, nw);
        if !self.modules.contains_key(&mwv_q4k_o_key) {
            let kernel_type = KernelType::MwvQ4KGemv {
                k: q_dim, n: hidden_dim, num_warps: nw,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(mwv_q4k_o_key, module);
        }

        // FFN gate/up (hidden->intermediate) and down (intermediate->hidden)
        let mwv_q4k_up_key = format!("mwv_q4k_gemv_{}_{}_{}", hidden_dim, intermediate_dim, nw);
        if !self.modules.contains_key(&mwv_q4k_up_key) {
            let kernel_type = KernelType::MwvQ4KGemv {
                k: hidden_dim, n: intermediate_dim, num_warps: nw,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(mwv_q4k_up_key, module);
        }
        let mwv_q4k_down_key = format!("mwv_q4k_gemv_{}_{}_{}", intermediate_dim, hidden_dim, nw);
        if !self.modules.contains_key(&mwv_q4k_down_key) {
            let kernel_type = KernelType::MwvQ4KGemv {
                k: intermediate_dim, n: hidden_dim, num_warps: nw,
            };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(mwv_q4k_down_key, module);
        }

        // Q6K FFN down + coalesced variant
        self.preload_q6k_gemv_pair(intermediate_dim, hidden_dim)?;

        // PMAT-038: HW DP4A Q4K + Q8 quantize preloading for CUDA graph capture.
        if self.gpu_profile.q4k == crate::cuda::gpu_profile::Q4kVariant::HwDp4a {
            self.preload_hw_dp4a_modules(hidden_dim, intermediate_dim, q_dim, kv_dim, nw)?;
        }

        Ok(())
    }

    /// PMAT-038: Pre-load HW DP4A Q4K GEMV + Q8 quantize + fused gate+up+SwiGLU.
    ///
    /// Without this, graph capture fails (error 901) because JIT compile allocates during capture.
    #[allow(clippy::too_many_arguments)]
    fn preload_hw_dp4a_modules(
        &mut self,
        hidden_dim: u32,
        intermediate_dim: u32,
        q_dim: u32,
        kv_dim: u32,
        nw: u32,
    ) -> Result<(), GpuError> {
        // Q8 quantize for all GEMV input dimensions
        for &q8_n in &[hidden_dim, q_dim, intermediate_dim] {
            let q8_key = format!("q8_quantize_{}", q8_n);
            if !self.modules.contains_key(&q8_key) {
                let kernel_type = KernelType::Q8Quantize { n: q8_n };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(q8_key, module);
            }
        }
        // HW DP4A Q4K for all dimension pairs
        let hw_dims: [(u32, u32); 5] = [
            (hidden_dim, q_dim),           // Q projection
            (hidden_dim, kv_dim),          // K/V projection
            (q_dim, hidden_dim),           // output projection
            (hidden_dim, intermediate_dim), // FFN gate/up
            (intermediate_dim, hidden_dim), // FFN down
        ];
        for &(k, n) in &hw_dims {
            let key = format!("hw_dp4a_q4k_gemv_{}_{}_{}", k, n, nw);
            if !self.modules.contains_key(&key) {
                let kernel_type = KernelType::HwDp4aQ4KGemv { k, n, num_warps: nw };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(key, module);
            }
        }
        // Fused gate+up+SwiGLU kernel (PMAT-034)
        if self.gpu_profile.fused_gate_up {
            let fused_key = format!("fused_gate_up_swiglu_hw_dp4a_q4k_{}_{}", hidden_dim, intermediate_dim);
            if !self.modules.contains_key(&fused_key) {
                let kernel_type = KernelType::FusedGateUpSwigluHwDp4aQ4KGemv {
                    k: hidden_dim, n: intermediate_dim,
                };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(fused_key, module);
            }
        }
        Ok(())
    }

    /// Pre-load Q6K GEMV (original + coalesced if K is 256-aligned) for given dimensions.
    fn preload_q6k_gemv_pair(&mut self, k: u32, n: u32) -> Result<(), GpuError> {
        let q6k_key = format!("q6k_gemv_{}_{}", k, n);
        if !self.modules.contains_key(&q6k_key) {
            let kernel_type = KernelType::Q6KGemv { k, n };
            let ptx = self.kernels.generate_ptx(&kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(q6k_key, module);
        }
        if k.is_multiple_of(256) {
            let coalesced_key = format!("coalesced_q6k_gemv_{}_{}", k, n);
            if !self.modules.contains_key(&coalesced_key) {
                let kernel_type = KernelType::CoalescedQ6KGemv { k, n };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(coalesced_key, module);
            }
        }
        Ok(())
    }

    /// GH-129: Pre-load DP4A Q6K + Q8 quantize kernels when DP4A_Q6K=1.
    ///
    /// Prevents JIT compilation during inference on memory-constrained devices.
    fn preload_dp4a_q6k_modules(
        &mut self,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        num_warps: u32,
    ) -> Result<(), GpuError> {
        use crate::cuda::gpu_profile::Q6kVariant;
        let variant = self.gpu_profile.q6k;
        if variant != Q6kVariant::Dp4a && variant != Q6kVariant::HwDp4a {
            return Ok(());
        }
        // Q8 quantize for GEMV input dimensions (shared by both DP4A variants)
        for &q8_n in &[hidden_dim, intermediate_dim] {
            let q8_key = format!("q8_quantize_{}", q8_n);
            if !self.modules.contains_key(&q8_key) {
                let kernel_type = KernelType::Q8Quantize { n: q8_n };
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(q8_key, module);
            }
        }
        // Preload variant-specific Q6K GEMV kernels for FFN down + LM head
        let dims = [(intermediate_dim, hidden_dim), (hidden_dim, vocab_size)];
        for &(k_dim, n_dim) in &dims {
            let (key, kernel_type) = match variant {
                Q6kVariant::HwDp4a => (
                    format!("hw_dp4a_q6k_gemv_{}_{}_{}", k_dim, n_dim, num_warps),
                    KernelType::HwDp4aQ6KGemv { k: k_dim, n: n_dim, num_warps },
                ),
                _ => (
                    format!("dp4a_q6k_gemv_{}_{}_{}", k_dim, n_dim, num_warps),
                    KernelType::Dp4aQ6KGemv { k: k_dim, n: n_dim, num_warps },
                ),
            };
            if !self.modules.contains_key(&key) {
                let ptx = self.kernels.generate_ptx(&kernel_type);
                let module = self.compile_ptx(&ptx)?;
                self.modules.insert(key, module);
            }
        }
        Ok(())
    }

    /// PAR-118: Pre-load Flash Decoding chunk + reduce kernels for graph capture.
    ///
    /// Module keys must match those in flash_decoding_graphed.rs exactly.
    fn preload_flash_decoding_modules(
        &mut self,
        max_len: u32,
        head_dim: u32,
        num_heads: u32,
        num_kv_heads: u32,
    ) -> Result<(), GpuError> {
        use trueno_gpu::kernels::{FlashDecodingChunkKernel, FlashDecodingReduceKernel, Kernel};

        let chunk_module_key = format!(
            "flash_decode_chunk_{}_{}_{}_{}",
            max_len, head_dim, num_heads, num_kv_heads
        );
        if !self.modules.contains_key(&chunk_module_key) {
            let chunk_kernel = FlashDecodingChunkKernel::new(
                max_len, head_dim, num_heads, num_kv_heads, 1,
            );
            let ptx = chunk_kernel.emit_ptx_for_target(&self.kernels.sm_target);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(chunk_module_key, module);
        }

        let reduce_module_key = format!("flash_decode_reduce_{}_{}", head_dim, num_heads);
        if !self.modules.contains_key(&reduce_module_key) {
            let reduce_kernel = FlashDecodingReduceKernel::new(head_dim, num_heads, 1);
            let ptx = reduce_kernel.emit_ptx_for_target(&self.kernels.sm_target);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(reduce_module_key, module);
        }

        Ok(())
    }
}
