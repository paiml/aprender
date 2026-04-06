//! BLIS-Style Matrix Multiplication
//!
//! High-performance GEMM implementation based on the BLIS framework.
//!
//! # References
//!
//! - Goto, K., & Van de Geijn, R. A. (2008). Anatomy of High-Performance Matrix Multiplication.
//!   ACM TOMS, 34(3). <https://doi.org/10.1145/1356052.1356053>
//! - Van Zee, F. G., & Van de Geijn, R. A. (2015). BLIS: A Framework for Rapidly Instantiating
//!   BLAS Functionality. ACM TOMS, 41(3). <https://doi.org/10.1145/2764454>
//! - Low, T. M., et al. (2016). Analytical Modeling Is Enough for High-Performance BLIS.
//!   ACM TOMS, 43(2). <https://doi.org/10.1145/2925987>
//!
//! # Toyota Production System Integration
//!
//! - **Jidoka**: Runtime guards that stop on numerical errors (see [`jidoka`] module)
//! - **Poka-Yoke**: Compile-time type safety for panel dimensions
//! - **Heijunka**: Load-balanced parallel execution
//! - **Kaizen**: Performance tracking for continuous improvement (see [`profiler`] module)
//!
//! # Module Structure
//!
//! - [`jidoka`]: Runtime validation guards (stop-on-defect)
//! - [`profiler`]: Performance tracking at all BLIS hierarchy levels
//! - [`microkernels`]: High-performance SIMD compute kernels
//! - [`backend_selection`]: Automatic CPU/GPU backend selection
//! - [`reference`]: Scalar reference GEMM for validation
//! - [`packing`]: Cache-optimized matrix packing routines
//! - [`compute`]: Core BLIS blocked GEMM computation
//! - [`parallel`]: Parallel GEMM with Heijunka scheduling
//! - [`transpose`]: Matrix transpose operations

pub mod attention;
pub mod backend_selection;
pub mod cache_topology;
pub mod compute;
pub mod elementwise;
pub mod gemv;
pub mod jidoka;
pub mod microkernels;
pub mod norms;
pub mod packing;
pub mod parallel;
pub mod prepacked;
pub mod profiler;
pub mod reference;
pub mod softmax;
pub mod transpose;

// Re-export jidoka types for backwards compatibility
pub use jidoka::{JidokaError, JidokaGuard};

// Re-export profiler types for backwards compatibility
pub use profiler::{BlisLevelStats, BlisProfileLevel, BlisProfiler, KaizenMetrics};

// Re-export microkernel functions
#[cfg(target_arch = "aarch64")]
pub use microkernels::microkernel_8x8_neon;
pub use microkernels::microkernel_scalar;
#[cfg(target_arch = "x86_64")]
pub use microkernels::{microkernel_8x6_avx2, microkernel_8x6_avx2_asm, microkernel_8x6_true_asm};

// Re-export backend selection types
pub use backend_selection::{
    gemm_auto, BackendCostModel, BrickLevel, ComputeBackend, PtxMicrokernelSpec, RooflineResult,
    UnifiedBrickProfiler, WgslMicrokernelSpec,
};

// Re-export reference GEMM
pub use reference::{gemm_reference, gemm_reference_with_jidoka};

// Re-export packing functions
pub use packing::{pack_a, pack_b, packed_a_size, packed_b_size};

// Re-export compute
#[cfg(target_arch = "x86_64")]
pub use compute::gemm_blis_broadcast_b;
pub use compute::{gemm_blis, gemm_blis_with_prepacked_b};

// Re-export parallel
#[cfg(feature = "parallel")]
pub use parallel::gemm_blis_parallel_shared_b;
pub use parallel::{gemm_blis_parallel, gemm_blis_parallel_with_prepacked_b, HeijunkaScheduler};

// Re-export prepacked
pub use prepacked::PrepackedB;

// Re-export transpose
pub use transpose::transpose;

use crate::error::TruenoError;

// ============================================================================
// BLIS Configuration Constants
// ============================================================================

/// Microkernel row dimension (AVX2: 8 f32 per ymm register)
pub const MR: usize = 8;

/// Microkernel column dimension (6 columns fit in remaining registers)
pub const NR: usize = 6;

/// K-dimension blocking for L1 cache (256 elements = 1KB)
pub const KC: usize = 256;

/// M-dimension blocking for L2 cache.
/// Must be a multiple of MR. 128 = 16×MR for AVX2 (vs old 72 = 9×MR).
/// Larger MC reduces packing overhead per macroblock (fewer ic-loop iterations).
/// Zen 4 L2 = 1MB per core; MC×KC×4B = 128×256×4 = 128KB << 1MB.
pub const MC: usize = 128;

/// N-dimension blocking for L3 cache
pub const NC: usize = 4096;

// ============================================================================
// AVX-512 BLIS Configuration Constants
// ============================================================================

/// AVX-512 microkernel row dimension (16 f32 per zmm register)
pub const MR_512: usize = 16;

/// AVX-512 microkernel column dimension (8 columns in remaining zmm registers)
pub const NR_512: usize = 8;

/// AVX-512 K-dimension blocking (same as AVX2, L1 limited)
pub const KC_512: usize = 256;

/// AVX-512 M-dimension blocking for L2 cache.
/// 128 = 8×MR_512. Zen 4 L2 = 1MB; 128×256×4 = 128KB.
pub const MC_512: usize = 128;

/// AVX-512 N-dimension blocking for L3 cache
pub const NC_512: usize = 4096;

// ============================================================================
// AVX-512 32×6 Microkernel Constants (Phase 4, Appendix D optimization #1)
// ============================================================================

/// 32×6 microkernel: 2 zmm rows × 6 columns = 12 accumulators.
/// 1.5× more FMAs per K step than 16×8 (12 vs 8).
pub const MR_512V2: usize = 32;

/// 6 columns: balances register pressure (12 acc + 2 A load = 14 zmm).
pub const NR_512V2: usize = 6;

/// Increased KC for 32×6: 32×256×4 = 32 KB fits L1 (32 KB on Zen 4).
pub const KC_512V2: usize = 256;

/// MC for 32×6: 192 = 6×MR_512V2. Packed A = 192×256×4 = 192 KB fits L2.
pub const MC_512V2: usize = 192;

/// NC for 32×6: same L3 blocking.
pub const NC_512V2: usize = 4096;

// ============================================================================
// Public API
// ============================================================================

/// High-performance GEMM using BLIS algorithm
///
/// Computes C += A * B where:
/// - A is M x K (row-major)
/// - B is K x N (row-major)
/// - C is M x N (row-major)
///
/// Automatically selects single-threaded or parallel execution based on matrix size.
pub fn gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    // Contract: matmul-kernel-v1.yaml precondition (pv codegen)
    contract_pre_matmul!(a);

    let result = {
        #[cfg(feature = "parallel")]
        {
            gemm_blis_parallel(m, n, k, a, b, c)
        }
        #[cfg(not(feature = "parallel"))]
        {
            gemm_blis(m, n, k, a, b, c, None)
        }
    };
    if result.is_ok() {
        contract_post_matmul!(c);
    }
    result
}

/// GEMM with profiling enabled
pub fn gemm_profiled(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    profiler: &mut BlisProfiler,
) -> Result<(), TruenoError> {
    gemm_blis(m, n, k, a, b, c, Some(profiler))
}

/// Fused GEMM + bias + ReLU: C = max(0, A×B + bias)
///
/// Performs matmul then applies bias addition and ReLU activation in a single
/// pass over C while the output tiles are still in L1/L2 cache. This avoids
/// two extra full-matrix memory passes that separate add+relu would require.
///
/// For GEMM 64: saves ~2µs (bias+relu would cost 2×0.8µs on cold data).
/// For GEMM 128: saves ~5µs.
///
/// # Arguments
///
/// * `bias` - Per-column bias vector of length `n` (broadcast across rows)
///
/// # Errors
///
/// Returns `Err` if dimensions don't match or bias length != n.
pub fn gemm_bias_relu(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    if bias.len() != n {
        return Err(TruenoError::InvalidInput(format!(
            "gemm_bias_relu: bias.len()={} != n={}",
            bias.len(),
            n
        )));
    }
    // Step 1: GEMM (C = A×B)
    gemm(m, n, k, a, b, c)?;

    // Step 2: Fused bias + ReLU in-place on hot cache data.
    // C is still in L1/L2 from the GEMM store — no DRAM reads needed.
    for row in 0..m {
        let row_start = row * n;
        for col in 0..n {
            let val = c[row_start + col] + bias[col];
            c[row_start + col] = val.max(0.0);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
