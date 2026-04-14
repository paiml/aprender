/// SPEC-MOE-APR-001 Phase 8: Fused MoE WMMA GGUF kernel dispatch.
///
/// Loads pre-compiled PTX from candle's moe_wmma_gguf kernel and dispatches
/// fused expert GEMM with Q4K/Q6K dequant + WMMA tensor cores.
///
/// Contract: moe-cuda-kernel-v1.yaml (FALSIFY-CUDA-MOE-002: >= 61 tok/s)
/// Reference: candle/candle-kernels/src/moe/moe_wmma_gguf.cu

use super::GpuError;
use trueno_gpu::driver::{CudaModule, GpuBuffer, LaunchConfig};

/// MoE kernel entry point names in the pre-compiled PTX
const MOE_Q4K_KERNEL: &str = "_Z28moe_gemm_gguf_prefill_kernelI6__halfLi256E10block_q4_KLi32EEvPKT_PKhPKiS8_PKfPfiiiiii";
const MOE_Q6K_KERNEL: &str = "_Z28moe_gemm_gguf_prefill_kernelI6__halfLi256E10block_q6_KLi64EEvPKT_PKhPKiS8_PKfPfiiiiii";
const MOE_COUNT_TOKENS_KERNEL: &str = "_Z30count_tokens_per_expert_kernelPKiPii";
const MOE_PREFIX_SUM_KERNEL: &str = "_Z24expert_prefix_sum_kernelPKiPii";

impl super::CudaExecutor {
    /// Check if the fused MoE kernel is loaded
    #[must_use]
    pub fn has_moe_kernel(&self) -> bool {
        self.modules.contains_key("moe_wmma_gguf")
    }

    /// Load the fused MoE WMMA GGUF kernel from PTX file.
    /// Called once during model init.
    pub fn load_moe_kernel(&mut self, ptx_path: &str) -> Result<(), GpuError> {
        let ptx = std::fs::read_to_string(ptx_path).map_err(|e| {
            GpuError::ModuleLoad(format!("Failed to read MoE PTX from {}: {}", ptx_path, e))
        })?;
        let module = self.compile_ptx(&ptx)?;
        self.modules.insert("moe_wmma_gguf".to_string(), module);
        eprintln!("[SPEC-MOE-APR-001] Loaded fused MoE WMMA GGUF kernel from {}", ptx_path);
        Ok(())
    }

    /// Dispatch fused MoE GEMM for one projection (gate, up, or down).
    ///
    /// Processes ALL top-k experts in ONE kernel launch using candle's
    /// moe_wmma_gguf pattern: tokens sorted by expert, WMMA tensor cores,
    /// in-kernel Q4K/Q6K dequant.
    ///
    /// # Arguments
    /// * `input` - Input activations [1, k] on GPU (f16)
    /// * `weights_ptr` - Packed 3D expert weights on GPU (Q4K or Q6K)
    /// * `sorted_token_ids` - Token indices sorted by expert [num_tokens * topk]
    /// * `expert_offsets` - [num_experts + 1] prefix sum of tokens per expert
    /// * `topk_weights` - Router weights [num_tokens * topk]
    /// * `output` - Output buffer [1, n] on GPU
    /// * `num_experts` - Total number of experts
    /// * `topk` - Number of active experts per token
    /// * `n` - Output dimension (moe_intermediate or hidden_dim)
    /// * `k` - Input dimension (hidden_dim or moe_intermediate)
    /// * `is_q6k` - True for Q6K (down_exps), false for Q4K (gate/up_exps)
    #[allow(clippy::too_many_arguments)]
    pub fn fused_moe_gemv(
        &mut self,
        input_ptr: u64,
        weights_ptr: u64,
        sorted_token_ids_ptr: u64,
        expert_offsets_ptr: u64,
        topk_weights_ptr: u64,
        output_ptr: u64,
        num_experts: u32,
        topk: u32,
        size_m: u32,  // num tokens
        size_n: u32,  // output dim
        size_k: u32,  // input dim
        is_q6k: bool,
    ) -> Result<(), GpuError> {
        let module = self.modules.get_mut("moe_wmma_gguf")
            .ok_or_else(|| GpuError::ModuleLoad(
                "MoE WMMA kernel not loaded. Call load_moe_kernel() first.".into()
            ))?;

        let kernel_name = if is_q6k { MOE_Q6K_KERNEL } else { MOE_Q4K_KERNEL };
        let gguf_dtype: i32 = if is_q6k { 5 } else { 1 }; // Q6K=5, Q4K=1 in candle convention

        // Grid: (num_experts, ceil(size_n / 32))
        // Block: (128,) = 4 warps × 32 threads
        let n_tiles = (size_n + 31) / 32;
        // Grid: (num_experts, n_tiles), Block: (128,1,1), shared memory for tiles
        let qk: u32 = 256;
        let block_size_bytes: u32 = if is_q6k { 210 } else { 144 };
        let a_sh = 32 * qk * 2; // M_BLK * qk * sizeof(half)
        let b_sh = 32 * qk * 2; // N_BLK * qk * sizeof(half)
        let b_quant = 32 * block_size_bytes; // N_BLK * block_size_bytes
        let c_sh = 32 * 32 * 4; // M_BLK * N_BLK * sizeof(float)
        let shared_mem = a_sh + b_sh + b_quant + c_sh + 64; // + alignment padding
        let config = LaunchConfig {
            grid: (num_experts, n_tiles, 1),
            block: (128, 1, 1),
            shared_mem,
        };

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            &input_ptr as *const u64 as *mut std::ffi::c_void,
            &weights_ptr as *const u64 as *mut std::ffi::c_void,
            &sorted_token_ids_ptr as *const u64 as *mut std::ffi::c_void,
            &expert_offsets_ptr as *const u64 as *mut std::ffi::c_void,
            &topk_weights_ptr as *const u64 as *mut std::ffi::c_void,
            &output_ptr as *const u64 as *mut std::ffi::c_void,
            &num_experts as *const u32 as *mut std::ffi::c_void,
            &topk as *const u32 as *mut std::ffi::c_void,
            &size_m as *const u32 as *mut std::ffi::c_void,
            &size_n as *const u32 as *mut std::ffi::c_void,
            &size_k as *const u32 as *mut std::ffi::c_void,
            &gguf_dtype as *const i32 as *mut std::ffi::c_void,
        ];

        unsafe {
            self.stream.launch_kernel(
                module,
                kernel_name,
                &config,
                &mut args,
            )?;
        }
        self.stream.synchronize()?;

        Ok(())
    }
}
