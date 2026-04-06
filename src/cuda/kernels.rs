//! CUDA Kernel Type Definitions and PTX Generation
//!
//! This module contains the `KernelType` enum for all supported GPU kernels
//! and the `CudaKernels` struct for generating PTX assembly.

/// PAR-082-V3: Get configured MWV warp count (default: 3)
/// Shared by q4k.rs dispatch and graphed.rs preload to ensure consistent kernel selection.
/// Set `MWV_WARPS=N` to override (2, 3, 4, 6, 8).
/// Empirical sweet spot: 3 warps = 110.7 tok/s on RTX 4090 for 7B Q4K.
pub fn mwv_warp_count() -> u32 {
    static MWV_WARPS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *MWV_WARPS.get_or_init(|| {
        std::env::var("MWV_WARPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3)
    })
}

// All kernel types are imported for exhaustive KernelType enum coverage
#[allow(unused_imports)]
use trueno_gpu::kernels::{
    Activation, ArgMaxFinalKernel, ArgMaxKernel, AttentionKernel, BatchedHwDp4aQ4KGemvKernel,
    BatchedIncrementalAttentionKernel, BatchedQ4KGemvKernel, BatchedQ6KGemvKernel,
    BatchedResidualAddKernel, BatchedRopeKernel, BatchedSwigluKernel,
    BatchedVectorizedRmsNormKernel, BiasActivationKernel, ChunkedTiledQ4KGemvKernel,
    CoalescedGemvKernel, CoalescedQ4KGemvKernel, CoalescedQ6KGemvKernel, Dp4aQ4KGemmKernel,
    Dp4aQ4KGemvKernel, Dp4aQ6KGemvKernel, ElementwiseMulKernel, Fp16Q4KGemvKernel,
    FusedFp32Q4KGemvKernel, FusedGateUpKernel, FusedGateUpQ4KGemvKernel,
    FusedGateUpSwigluHwDp4aQ4KGemvKernel, FusedQKVHwDp4aQ4KGemvKernel, FusedQKVKernel,
    FusedResidualRmsNormKernel, FusedRmsNormGateUpSwigluQ4KKernel, FusedRmsNormQ4KGemvKernel,
    FusedSwigluKernel, GeluKernel, GemmKernel, GemvKernel, HalfWarpDp4aQ4KGemvKernel,
    HalfWarpDp4aQ6KGemvKernel, IncrementalAttentionKernel, InlineQ8Dp4aQ4KGemvKernel, Kernel,
    KvCacheScatterIndirectKernel, KvCacheScatterKernel, LayerNormKernel,
    MultiWarpIncrementalAttentionKernel, MultiWarpQ6KGemvKernel, MultiWarpTensorCoreQ4KGemmKernel,
    MultiWarpVectorizedQ4KGemvKernel, MwvDp4aQ4KGemvKernel, PackedDp4aQ4KQ8Kernel,
    PerHeadRmsNormKernel, PreciseRmsNormKernel, PreciseRopeIndirectKernel, Q4KDequantFp16Kernel,
    Q4KDequantKernel, Q4KGemvKernel, Q4KQ8DotKernel, Q4_0GemvKernel, Q4_1GemvKernel, Q5KGemvKernel,
    Q5KKernel, Q5_0GemvKernel, Q6KDequantKernel, Q6KGemvKernel, Q6KKernel, Q8QuantizeKernel,
    Q8_0GemvKernel, QuantizeKernel, ResidualAddKernel, RmsNormKernel, RopeIndirectKernel,
    RopeKernel, RopeNeoxIndirectKernel, RopeNeoxKernel, SiluKernel, SoftmaxKernel,
    TensorCoreQ4KGemmKernel, TiledQ4KGemvKernel, TrueDp4aQ4KGemvKernel, VectorizedQ4KGemvKernel,
    VectorizedRmsNormKernel, WideQ4KGemvKernel,
};
// GH-143: WMMA kernel types only available on x86_64 (not on aarch64/Blackwell GB10)
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use trueno_gpu::kernels::{
    BatchedFusedResidualRmsNormKernel, InterleavedWmmaQ4KGemmKernel, W4a16WmmaQ4KGemmKernel,
};

include!("kernel_type.rs");
include!("kernel_generator.rs");
include!("kernel.rs");
include!("layout.rs");
include!("kernels_generate.rs");
