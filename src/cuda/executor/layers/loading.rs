impl CudaExecutor {

    /// Compile and cache a kernel module if not already loaded.
    fn ensure_module(&mut self, key: String, kernel_type: &KernelType) -> Result<(), GpuError> {
        if !self.modules.contains_key(&key) {
            let ptx = self.kernels.generate_ptx(kernel_type);
            let module = self.compile_ptx(&ptx)?;
            self.modules.insert(key, module);
        }
        Ok(())
    }

    /// PAR-054-FIX: Pre-load all kernel modules needed for graph capture
    ///
    /// Root cause of CUDA graph capture failure (code 901):
    /// `CudaModule::from_ptx` calls CUDA driver which allocates memory.
    /// Any allocation during graph capture causes error 901.
    /// Solution: Pre-load ALL modules before `begin_capture()`.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn preload_modules_for_capture(
        &mut self,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
    ) -> Result<(), GpuError> {
        let num_heads = self.kv_num_heads as u32;
        let num_kv_heads = self.kv_num_kv_heads as u32;
        let head_dim = self.kv_head_dim as u32;
        let max_len = self.kv_cache_max_len as u32;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let nw = self.gpu_profile.mwv_warps;

        // 1. RMSNorm kernel
        self.preload_rmsnorm_module(hidden_dim)?;

        // 2. Q/K/V GEMV kernels — all quant types that might be used
        self.preload_gemv_modules(hidden_dim, intermediate_dim, q_dim, kv_dim, nw)?;

        // GH-129: Pre-load DP4A Q6K + Q8 quantize kernels
        self.preload_dp4a_q6k_modules(hidden_dim, intermediate_dim, vocab_size, nw)?;

        // 5-9. LM head, RoPE, SwiGLU, residual, scatter, attention kernels
        self.preload_lm_head_and_utility_modules(
            num_layers, hidden_dim, intermediate_dim, vocab_size,
            num_heads, num_kv_heads, head_dim, max_len, q_dim, kv_dim, nw,
        )
    }
}

include!("preload_utilities.rs");
