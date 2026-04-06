
impl CudaKernels {

    /// Get kernel name for the specified type
    #[must_use]
    pub fn kernel_name(&self, kernel_type: &KernelType) -> &'static str {
        Self::gemm_kernel_name(kernel_type)
            .or_else(|| Self::gemv_kernel_name(kernel_type))
            .or_else(|| Self::q4k_gemv_kernel_name(kernel_type))
            .or_else(|| Self::attention_kernel_name(kernel_type))
            .or_else(|| Self::norm_rope_kernel_name(kernel_type))
            .or_else(|| Self::activation_misc_kernel_name(kernel_type))
            .unwrap_or("unknown")
    }

    /// GEMM kernel names (matrix-matrix multiplication)
    fn gemm_kernel_name(kernel_type: &KernelType) -> Option<&'static str> {
        let name = match kernel_type {
            KernelType::GemmNaive { .. } => "gemm_naive",
            KernelType::GemmTiled { .. }
            | KernelType::GemmOptimized { .. }
            | KernelType::GemmBiasActivation { .. } => "gemm_tiled",
            KernelType::GemmTensorCore { .. } => "gemm_tensor_core",
            KernelType::GemmFp16TensorCore { .. } => "gemm_wmma_fp16",
            KernelType::QuantizedGemm { .. } => "q4k_gemm_fused",
            KernelType::QuantizedGemmGgml { .. } => "q4k_gemm_ggml",
            KernelType::QuantizedGemmGgmlTiled { .. } => "q4k_gemm_ggml_tiled",
            KernelType::Q5KQuantizedGemm { .. } => "q5k_gemm_ggml",
            KernelType::Q6KQuantizedGemm { .. } => "q6k_gemm_ggml",
            KernelType::FusedQ4Q8Dot { .. } => "q4k_gemm_ggml",
            KernelType::TensorCoreQ4KGemm { .. } => "tensor_core_q4k_gemm",
            KernelType::MultiWarpTensorCoreQ4KGemm { .. } => "mw_tensor_core_q4k_gemm",
            KernelType::InterleavedWmmaQ4KGemm { .. } => "interleaved_wmma_q4k_gemm",
            KernelType::W4a16WmmaQ4KGemm { .. } => "w4a16_wmma_q4k_gemm",
            KernelType::Dp4aQ4KGemm { .. } => "dp4a_q4k_gemm",
            _ => return None,
        };
        Some(name)
    }

    /// Standard GEMV kernel names (matrix-vector, non-Q4K)
    fn gemv_kernel_name(kernel_type: &KernelType) -> Option<&'static str> {
        let name = match kernel_type {
            KernelType::Gemv { .. } => "gemv_warp_reduce",
            KernelType::CoalescedGemv { .. } => "gemv_coalesced",
            KernelType::Q5KGemv { .. } => "q5k_gemv_warp_reduce",
            KernelType::Q6KGemv { .. } => "q6k_gemv_warp_reduce",
            KernelType::CoalescedQ6KGemv { .. } => "coalesced_q6k_gemv",
            KernelType::BatchedQ6KGemv { .. } => "batched_q6k_gemv_warp_reduce",
            KernelType::MwvQ6KGemv { .. } => "mwv_q6k_gemv",
            KernelType::Dp4aQ6KGemv { .. } => "dp4a_q6k_gemv",
            KernelType::HwDp4aQ6KGemv { .. } => "hw_dp4a_q6k_gemv",
            KernelType::Fp16Q4KGemv { .. } => "fp16_q4k_gemv",
            KernelType::Q8_0Gemv { .. } => "q8_0_gemv_warp_reduce",
            KernelType::Q5_0Gemv { .. } => "q5_0_gemv_warp_reduce",
            KernelType::Q4_0Gemv { .. } => "q4_0_gemv_warp_reduce",
            KernelType::Q4_1Gemv { .. } => "q4_1_gemv_warp_reduce",
            _ => return None,
        };
        Some(name)
    }

    /// Q4K GEMV kernel names (quantized 4-bit matrix-vector variants)
    fn q4k_gemv_kernel_name(kernel_type: &KernelType) -> Option<&'static str> {
        let name = match kernel_type {
            KernelType::Q4KGemv { .. } => "q4k_gemv_warp_reduce",
            KernelType::TiledQ4KGemv { .. } => "tiled_q4k_gemv",
            KernelType::ChunkedTiledQ4KGemv { .. } => "chunked_tiled_q4k_gemv",
            KernelType::CoalescedQ4KGemv { .. } => "coalesced_q4k_gemv",
            KernelType::WideQ4KGemv { .. } => "wide_q4k_gemv",
            KernelType::VectorizedQ4KGemv { .. } => "vectorized_q4k_gemv",
            KernelType::MwvQ4KGemv { .. } => "mwv_q4k_gemv",
            KernelType::MwvDp4aQ4KGemv { .. } => "mwv_dp4a_q4k_gemv",
            KernelType::HwDp4aQ4KGemv { .. } => "hw_dp4a_q4k_gemv",
            KernelType::Dp4aQ4KGemv { .. } => "dp4a_q4k_gemv",
            KernelType::Dp4aSIMDQ4KGemv { .. } => "dp4a_simd_q4k_gemv",
            KernelType::TrueDp4aQ4KGemv { .. } => "true_dp4a_q4k_gemv",
            KernelType::BatchedQ4KGemv { .. }
            | KernelType::MultiWarpBatchedQ4KGemv { .. } => "batched_q4k_gemv_warp_reduce",
            KernelType::BatchedHwDp4aQ4KGemv { .. } => "batched_hw_dp4a_q4k_gemv",
            KernelType::FusedFp32Q4KGemv { .. } => "fused_fp32_q4k_gemv",
            KernelType::InlineQ8Dp4aQ4KGemv { .. } => "inline_q8_dp4a_q4k_gemv",
            KernelType::FusedKVHwDp4aQ4KGemv { .. } => "fused_qkv_hw_dp4a_q4k_gemv",
            KernelType::Q4KDequant { .. } => "q4k_dequant_to_f32",
            KernelType::Q4KDequantFp16 { .. } => "q4k_dequant_to_f16",
            KernelType::Q6KDequant { .. } => "q6k_dequant_to_f32",
            _ => return None,
        };
        Some(name)
    }

    /// Attention kernel names (flash attention, incremental, multi-warp)
    fn attention_kernel_name(kernel_type: &KernelType) -> Option<&'static str> {
        let name = match kernel_type {
            KernelType::Softmax { .. } => "softmax_warp_shuffle",
            KernelType::LayerNorm { .. } => "layernorm_warp_shuffle",
            KernelType::Attention { causal, .. }
            | KernelType::MultiHeadAttention { causal, .. } => {
                if *causal { "flash_attention_causal" } else { "flash_attention" }
            },
            KernelType::AttentionTensorCore { causal, .. } => {
                if *causal { "flash_attention_tensor_core_causal" } else { "flash_attention_tensor_core" }
            },
            KernelType::IncrementalAttention { indirect, .. } => {
                if *indirect { "incremental_attention_indirect" } else { "incremental_attention" }
            },
            KernelType::MultiWarpAttention { indirect, .. } => {
                if *indirect { "multi_warp_attention_indirect" } else { "multi_warp_attention" }
            },
            _ => return None,
        };
        Some(name)
    }

    /// Normalization and RoPE kernel names
    fn norm_rope_kernel_name(kernel_type: &KernelType) -> Option<&'static str> {
        let name = match kernel_type {
            KernelType::RmsNorm { .. } => "rmsnorm",
            KernelType::VectorizedRmsNorm { .. } => "rmsnorm_vectorized",
            KernelType::BatchedVectorizedRmsNorm { .. } => "batched_rmsnorm_vectorized",
            KernelType::BatchedFusedResidualRmsNorm { .. } => "batched_fused_residual_rmsnorm",
            KernelType::PreciseRmsNorm { .. } => "rmsnorm_precise",
            KernelType::PerHeadRmsNorm { .. } => "per_head_rmsnorm",
            KernelType::FusedResidualRmsNorm { .. } => "fused_residual_rmsnorm",
            KernelType::FusedRmsNormQ4KGemv { .. } => "fused_rmsnorm_q4k_gemv",
            KernelType::FusedRmsNormGateUpSwigluQ4K { .. } => "fused_rmsnorm_gate_up_swiglu_q4k",
            KernelType::Rope { .. } => "rope",
            KernelType::RopeIndirect { .. } => "rope_indirect",
            KernelType::RopeNeox { .. } => "rope_neox",
            KernelType::RopeNeoxIndirect { .. } => "rope_neox_indirect",
            KernelType::PreciseRopeNeoxIndirect { .. } => "rope_precise_indirect",
            KernelType::BatchedRope { .. } => "batched_rope",
            _ => return None,
        };
        Some(name)
    }

    /// Activation, fusion, and miscellaneous kernel names
    fn activation_misc_kernel_name(kernel_type: &KernelType) -> Option<&'static str> {
        let name = match kernel_type {
            KernelType::BiasActivation { .. } => "bias_activation",
            KernelType::Silu { .. } => "silu",
            KernelType::Gelu { .. } => "gelu",
            KernelType::ElementwiseMul { .. } => "elementwise_mul",
            KernelType::FusedSwiglu { .. } => "fused_swiglu",
            KernelType::FusedQKV { .. } => "fused_qkv_gemv",
            KernelType::FusedGateUp { .. } => "fused_gate_up_swiglu",
            KernelType::FusedGateUpQ4KGemv { .. } => "fused_gate_up_q4k_gemv",
            KernelType::FusedGateUpSwigluHwDp4aQ4KGemv { .. } => "fused_gate_up_swiglu_hw_dp4a_q4k_gemv",
            KernelType::ResidualAdd { .. } => "residual_add",
            KernelType::BatchedResidualAdd { .. } => "batched_residual_add",
            KernelType::BatchedSwiglu { .. } => "batched_swiglu",
            KernelType::KvCacheScatter { .. } => "kv_cache_scatter",
            KernelType::KvCacheScatterIndirect { .. } => "kv_cache_scatter_indirect",
            KernelType::Q8Quantize { .. } => "q8_quantize",
            KernelType::Q4KQ8Dot { .. } => "q4k_q8_dot",
            KernelType::PackedDp4aQ4KQ8 { .. } => "packed_dp4a_q4k_q8",
            KernelType::ArgMax { .. } => "argmax_block_reduce",
            KernelType::ArgMaxFinal { .. } => "argmax_final_reduce",
            KernelType::Q8Dequant { .. } => "q8_dequant",
            _ => return None,
        };
        Some(name)
    }

    /// Check if CUDA is likely available on this system
    ///
    /// Note: This is a heuristic check. Actual CUDA availability requires
    /// driver API initialization.
    #[must_use]
    pub fn cuda_likely_available() -> bool {
        // Check for NVIDIA GPU indicators
        std::path::Path::new("/dev/nvidia0").exists()
            || std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
    }
}

include!("cuda.rs");
