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

        if verbose() {
            eprintln!(
                "[PAR-054-FIX] Pre-loaded {} kernel modules for {} layers",
                self.modules.len(), num_layers
            );
        }
        Ok(())
    }
}
