impl CudaExecutor {

    /// PAR-054-FIX: Pre-load all kernel modules needed for graph capture
    ///
    /// Root cause of CUDA graph capture failure (code 901):
    /// - `CudaModule::from_ptx` calls CUDA driver which allocates memory
    /// - Any memory allocation during graph capture causes error 901
    /// - Solution: Pre-load ALL modules before `begin_capture()`
    ///
    /// Five-Whys Analysis:
    /// 1. Why does capture fail? Memory allocation detected during capture
    /// 2. Why allocation during capture? Lazy module loading in kernel dispatch
    /// 3. Why lazy loading? Performance optimization for unused kernels
    /// 4. Why does lazy loading allocate? PTX compilation requires driver memory
    /// 5. Why not pre-loaded? Missing pre-loading step before capture
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

        // 1. RMSNorm kernel (used for attn_norm, ffn_norm, output_norm)
        self.preload_rmsnorm_module(hidden_dim)?;

        // 2-4. Q/K/V, output projection, and FFN GEMV kernels
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

include!("modules_utilities.rs");
