//! Hand-Optimized GPU Kernels
//!
//! Pre-built kernels for common operations with optimal memory access patterns.
//!
//! ## Available Kernels
//!
//! - **GEMM**: Matrix multiplication (naive, tiled, Tensor Core)
//! - **Softmax**: Numerically stable softmax with warp shuffle
//! - **LayerNorm**: Fused layer normalization
//! - **Attention**: FlashAttention (SRAM tiling) + Paged (VRAM block management)
//! - **Quantize**: Q4_K/Q5_K/Q6_K dequantization fused with matmul (PARITY-115/116/117)
//! - **BiasActivation**: Fused bias + activation epilogue (ReLU, GELU)
//! - **GEMV**: Matrix-vector multiply for M=1 decode throughput (CoalescedGemvKernel)
//! - **Backward**: Training backward passes (GEMM, Softmax, LayerNorm, RMSNorm, Attention, Activations)
//! - **Optimizer**: Fused weight updates (AdamW, Adam, gradient clipping)
//!
//! ## Barrier Safety (PARITY-114)
//!
//! All kernels are validated for barrier safety to prevent thread divergence bugs.
//! Use `emit_ptx_validated()` for production to ensure no early-exit-before-barrier patterns.

mod argmax;
mod attention;
pub mod backward;
mod bias_activation;
mod conv1d;
mod elementwise;
mod fused;
pub(crate) mod gemm;
mod gemv;
mod layernorm;
pub mod lz4;
#[cfg(test)]
mod lz4_hash_store_test;
mod megakernel;
pub mod optimizer;
mod parity_impls;
mod persistent;
mod quantize;
mod softmax;

pub use argmax::{ArgMaxFinalKernel, ArgMaxKernel};
pub use attention::{
    AttentionKernel, BatchedIncrementalAttentionKernel, FlashDecodingChunkKernel,
    FlashDecodingChunkKernel2Warp, FlashDecodingReduceKernel, IncrementalAttentionKernel,
    MultiWarpIncrementalAttentionKernel, PrefillAttentionKernel, FLASH_DECODE_CHUNK_SIZE,
};
pub use bias_activation::{Activation, BiasActivationKernel};
pub use conv1d::{Conv1dKernel, TiledConv1dKernel};
pub use elementwise::{
    BatchedFusedResidualRmsNormKernel,
    BatchedResidualAddKernel,
    BatchedRopeBackwardKernel,
    BatchedRopeKernel,
    BatchedRopeNeoxBackwardKernel,
    BatchedRopeNeoxKernel,
    BatchedScaleKernel,
    BatchedSoftmaxKernel,
    BatchedSwigluKernel,
    BatchedToInterleavedKernel,
    BatchedTransposeKernel,
    CastF32ToF16Kernel,
    CopySingleHeadKernel,
    ElementwiseMulKernel,
    ExtractSingleHeadKernel,
    FusedResidualRmsNormKernel,
    FusedSwigluKernel,
    GeluKernel,
    InterleavedToBatchedKernel, // WAPR-PERF-004: Multi-head attention
    KvCacheScatterIndirectKernel,
    KvCacheScatterKernel,
    PreciseRopeIndirectKernel,
    PreciseRopeKernel, // CORRECTNESS-013
    ReluKernel,        // Issue #88: Forward ReLU kernel
    ResidualAddKernel,
    RopeIndirectKernel,
    RopeKernel,
    RopeNeoxIndirectKernel,
    RopeNeoxKernel,
    ScaleKernel,
    SiluKernel,
    TransposeKernel, // WAPR-PERF-004
};
pub use fused::{FusedGateUpKernel, FusedGemmBiasGeluKernel, FusedQKVKernel};
pub use gemm::basic::tensor_core::cta64_wmma::build_cta64x128_mma_pipeline_fp16;
pub use gemm::{
    Batched4DGemmConfig, Batched4DGemmKernel, BatchedGemmConfig, BatchedGemmKernel, GemmConfig,
    GemmKernel,
};
pub use gemv::{CoalescedGemvKernel, GemvKernel};
pub use layernorm::{
    BatchedVectorizedRmsNormKernel, LayerNormKernel, PerHeadRmsNormKernel, PreciseRmsNormKernel,
    RmsNormKernel, VectorizedRmsNormKernel,
};
pub use lz4::{Lz4WarpCompressKernel, Lz4WarpDecompressKernel};
pub use megakernel::TransformerBlockMegakernel;
pub use optimizer::{
    AdamStepKernel, AdamWStepKernel, ClipScaleReduceKernel, GradientClipGpuScaleKernel,
    GradientClipKernel, SquaredSumKernel,
};
pub use persistent::PersistentDecoderKernel;
pub use quantize::fused_kv_scatter::FusedKvScatterKernel;
pub use quantize::{
    dequantize_nf4, pack_nf4_for_gpu, quantize_nf4, repack_q4k_interleaved, repack_q4k_w4a16,
    unpack_nf4_from_gpu, BatchedHwDp4aQ4KGemvKernel, BatchedQ4KGemvKernel, BatchedQ6KGemvKernel,
    ChunkedTiledQ4KGemvKernel, CoalescedQ4KGemvKernel, CoalescedQ6KGemvKernel, Dp4aQ4KGemmKernel,
    Dp4aQ4KGemvKernel, Dp4aQ6KGemvKernel, Fp16Q4KGemvKernel, FusedFp32Q4KGemvKernel,
    FusedGateUpQ4KGemvKernel, FusedGateUpSwigluHwDp4aQ4KGemvKernel, FusedNf4GateUpGemmKernel,
    FusedQKVHwDp4aQ4KGemvKernel, FusedRmsNormGateUpSwigluQ4KKernel, FusedRmsNormNf4GemvKernel,
    FusedRmsNormQ4KGemvKernel, HalfWarpDp4aQ4KGemvKernel, HalfWarpDp4aQ6KGemvKernel,
    InlineQ8Dp4aQ4KGemvKernel, InterleavedWmmaQ4KGemmKernel, MultiWarpQ6KGemvKernel,
    MultiWarpTensorCoreQ4KGemmKernel, MultiWarpVectorizedQ4KGemvKernel, MwvDp4aQ4KGemvKernel,
    Nf4GemmKernel, Nf4GemmTransposeKernel, Nf4Quantized, Nf4TensorCoreGemmKernel,
    PackedDp4aQ4KQ8Kernel, Q4KDequantFp16Kernel, Q4KDequantKernel, Q4KGemvKernel, Q4KQ8DotKernel,
    Q4_0GemvKernel, Q4_1GemvKernel, Q5KGemvKernel, Q5KKernel, Q5_0GemvKernel, Q6KDequantKernel,
    Q6KGemvKernel, Q6KKernel, Q8QuantizeKernel, Q8_0GemvKernel, QuantizeKernel,
    TensorCoreQ4KGemmKernel, TiledQ4KGemvKernel, TrueDp4aQ4KGemvKernel, VectorizedQ4KGemvKernel,
    W4a16WmmaQ4KGemmKernel, WideQ4KGemvKernel, NF4_BLOCK_BYTES, NF4_BLOCK_SIZE, NF4_LUT,
};
pub use softmax::{LongRowSoftmaxKernel, SoftmaxKernel};

use crate::ptx::optimize::barrier_safety::{self, BarrierSafetyResult};
use crate::ptx::parity::{self, ParityResult};
use crate::ptx::{PtxKernel, PtxModule};

/// Select the correct PTX ISA version for a given SM target.
///
/// trueno#188: Blackwell (sm_100+) requires PTX 8.8 (CUDA 13.0 toolkit).
/// Older architectures use PTX 8.0.
///
/// Returns `(major, minor)` of the PTX ISA version.
pub fn ptx_version_for_target(target: &str) -> (u32, u32) {
    // Parse "sm_XYZ" -> XYZ as integer
    let sm_num: u32 = target
        .strip_prefix("sm_")
        .and_then(|s| s.parse().ok())
        .unwrap_or(70);
    if sm_num >= 100 {
        // Blackwell (sm_100, sm_120, sm_121, etc.) requires PTX 8.8
        (8, 8)
    } else {
        // Volta through Hopper: PTX 8.0
        (8, 0)
    }
}

/// Kernel trait for GPU kernels
pub trait Kernel {
    /// Get kernel name
    fn name(&self) -> &str;

    /// Build PTX kernel
    fn build_ptx(&self) -> PtxKernel;

    /// Get PTX module containing this kernel with specified compute target.
    ///
    /// trueno#188: Automatically selects the correct PTX ISA version based on
    /// the target. sm_100+ (Blackwell) requires PTX 8.8; older targets use 8.0.
    fn as_module_for_target(&self, target: &str) -> PtxModule {
        let (ptx_major, ptx_minor) = ptx_version_for_target(target);
        PtxModule::new()
            .version(ptx_major, ptx_minor)
            .target(target)
            .address_size(64)
            .add_kernel(self.build_ptx())
    }

    /// Get PTX module containing this kernel
    /// Uses sm_70 (Volta) as minimum baseline for broad compatibility
    fn as_module(&self) -> PtxModule {
        self.as_module_for_target("sm_70")
    }

    /// Emit PTX source for a specific compute capability
    fn emit_ptx_for_target(&self, target: &str) -> String {
        self.as_module_for_target(target).emit()
    }

    /// Emit PTX source
    fn emit_ptx(&self) -> String {
        self.as_module().emit()
    }

    /// Analyze PTX for barrier safety (PARITY-114 prevention)
    ///
    /// Returns detailed analysis of barrier safety, including any violations found.
    fn analyze_barrier_safety(&self) -> BarrierSafetyResult {
        let ptx = self.emit_ptx();
        barrier_safety::analyze(&ptx)
    }

    /// Validate PTX is barrier-safe (PARITY-114 prevention)
    ///
    /// Returns `Ok(())` if safe, `Err` with violation details if not.
    fn validate_barrier_safety(&self) -> Result<(), String> {
        let ptx = self.emit_ptx();
        barrier_safety::validate(&ptx)
    }

    /// Emit PTX with barrier safety validation (recommended for production)
    ///
    /// # Panics
    ///
    /// Panics if the PTX contains barrier safety violations (PARITY-114).
    /// Use this in production to catch bugs at compile time rather than runtime.
    fn emit_ptx_validated(&self) -> String {
        let ptx = self.emit_ptx();
        if let Err(e) = barrier_safety::validate(&ptx) {
            panic!(
                "PARITY-114: Barrier safety violation in kernel '{}': {}",
                self.name(),
                e
            );
        }
        ptx
    }
}

/// Trait for validating parity between a batched kernel and its single-vector counterpart.
///
/// # GH-219: PTX Contract Validation
///
/// When a batched kernel variant is created (e.g., `BatchedVectorizedRmsNormKernel`),
/// it MUST produce identical results to the single-vector kernel for M=1.
/// This trait enables compile-time structural validation of that contract.
///
/// ## What it checks
///
/// - Parameter count matches
/// - Shared memory size matches
/// - Loop structure matches (sum_loop, norm_loop, etc.)
/// - Batched kernel uses `ctaid.y` for row dispatch
/// - Shared memory addressed with u32 (not u64)
pub trait KernelParity: Kernel {
    /// The single-vector kernel type that this batched kernel must match
    type SingleVector: Kernel;

    /// Create the single-vector reference kernel for parity comparison
    fn single_vector_reference(&self) -> Self::SingleVector;

    /// Validate PTX structural parity between this batched kernel and its
    /// single-vector counterpart.
    fn validate_parity(&self) -> ParityResult {
        let single = self.single_vector_reference();
        let single_ptx = single.emit_ptx();
        let batched_ptx = self.emit_ptx();
        parity::validate_parity(&single_ptx, &batched_ptx, single.name(), self.name())
    }

    /// Validate that this batched kernel has correct batch dispatch patterns.
    /// Does not require a single-vector reference.
    fn validate_batch_dispatch(&self) -> ParityResult {
        let ptx = self.emit_ptx();
        parity::validate_batched_kernel(&ptx, self.name())
    }
}

#[cfg(test)]
mod tests;
