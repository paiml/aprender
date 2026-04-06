//! Batched GEMM Kernel (3D batched matrix multiplication)
//!
//! Implements C[b] = A[b] @ B[b] for batch_size independent matrix multiplications.

#![allow(clippy::similar_names)]

mod naive;
mod tiled;
mod tiled_unrolled;
mod wmma_fp16;

use crate::kernels::Kernel;
use crate::ptx::PtxKernel;

/// Batched GEMM configuration
#[derive(Debug, Clone)]
pub struct BatchedGemmConfig {
    /// Batch size (number of independent matrix multiplications)
    pub batch: u32,
    /// M dimension (rows of A and C)
    pub m: u32,
    /// N dimension (cols of B and C)
    pub n: u32,
    /// K dimension (cols of A, rows of B)
    pub k: u32,
    /// Tile size for shared memory
    pub tile_size: u32,
}

impl Default for BatchedGemmConfig {
    fn default() -> Self {
        Self { batch: 1, m: 1024, n: 1024, k: 1024, tile_size: 16 }
    }
}

/// Batched GEMM kernel for 3D tensor matmul
/// Each batch is processed by a separate thread block in the z-dimension
#[derive(Debug, Clone)]
pub struct BatchedGemmKernel {
    /// Kernel configuration
    pub config: BatchedGemmConfig,
    variant: BatchedGemmVariant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchedGemmVariant {
    Naive,
    Tiled,
    /// Tiled with 4x unrolled inner loop (WAPR-PERF-009)
    TiledUnrolled,
    /// WMMA FP16 using Tensor Core PTX intrinsics (WAPR-PERF-011)
    /// Requires sm_70+ (Volta or later). Dimensions must be multiples of 16.
    WmmaFp16,
}

impl BatchedGemmKernel {
    /// Create naive batched GEMM kernel (for correctness testing)
    /// Pattern: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
    #[must_use]
    pub fn naive(batch: u32, m: u32, n: u32, k: u32) -> Self {
        Self {
            config: BatchedGemmConfig { batch, m, n, k, ..Default::default() },
            variant: BatchedGemmVariant::Naive,
        }
    }

    /// Create tiled batched GEMM kernel (for performance)
    /// Pattern: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
    #[must_use]
    pub fn tiled(batch: u32, m: u32, n: u32, k: u32, tile_size: u32) -> Self {
        Self {
            config: BatchedGemmConfig { batch, m, n, k, tile_size },
            variant: BatchedGemmVariant::Tiled,
        }
    }

    /// Create tiled batched GEMM kernel with 4x unrolled inner loop (WAPR-PERF-009)
    /// Reduces loop overhead from 12:1 to ~3:1 instructions per FMA
    /// Pattern: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
    #[must_use]
    pub fn tiled_unrolled(batch: u32, m: u32, n: u32, k: u32, tile_size: u32) -> Self {
        Self {
            config: BatchedGemmConfig { batch, m, n, k, tile_size },
            variant: BatchedGemmVariant::TiledUnrolled,
        }
    }

    /// Create WMMA FP16 batched GEMM kernel using Tensor Core PTX intrinsics (WAPR-PERF-011)
    /// Requires sm_70+ (Volta or later). Input is FP32, converted to FP16 internally.
    /// Pattern: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
    /// Dimensions m, n must be multiples of 16 for optimal performance.
    #[must_use]
    pub fn wmma_fp16(batch: u32, m: u32, n: u32, k: u32) -> Self {
        Self {
            config: BatchedGemmConfig {
                batch,
                m,
                n,
                k,
                tile_size: 16, // WMMA uses 16x16x16 tiles
            },
            variant: BatchedGemmVariant::WmmaFp16,
        }
    }
}

impl Kernel for BatchedGemmKernel {
    fn name(&self) -> &str {
        match self.variant {
            BatchedGemmVariant::Naive => "batched_gemm_naive",
            BatchedGemmVariant::Tiled => "batched_gemm_tiled",
            BatchedGemmVariant::TiledUnrolled => "batched_gemm_tiled_unrolled",
            BatchedGemmVariant::WmmaFp16 => "batched_gemm_wmma_fp16",
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        match self.variant {
            BatchedGemmVariant::Naive => self.build_naive(),
            BatchedGemmVariant::Tiled => self.build_tiled(),
            BatchedGemmVariant::TiledUnrolled => self.build_tiled_unrolled(),
            BatchedGemmVariant::WmmaFp16 => self.build_wmma_fp16(),
        }
    }
}

#[cfg(test)]
mod tests;
