//! Tiling configuration and backend selection.

use super::error::TilingError;
use super::geometry::TcbGeometry;
use serde::{Deserialize, Serialize};

/// Complete tiling configuration for a kernel
///
/// Contains geometry for all three tiling levels, enabling hierarchical
/// cache-aware execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TilingConfig {
    /// Kernel name for identification
    pub name: String,
    /// Macro-tile geometry (L3/Global)
    pub macro_tile: TcbGeometry,
    /// Midi-tile geometry (L2/Shared)
    pub midi_tile: TcbGeometry,
    /// Micro-tile geometry (Registers)
    pub micro_tile: TcbGeometry,
    /// Target backend
    pub backend: TilingBackend,
}

/// Backend target for tiling configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TilingBackend {
    /// CPU with AVX2 (256-bit SIMD)
    CpuAvx2,
    /// CPU with AVX-512 (512-bit SIMD)
    CpuAvx512,
    /// CPU with NEON (128-bit SIMD)
    CpuNeon,
    /// GPU (CUDA/wgpu)
    Gpu,
    /// Scalar fallback
    Scalar,
}

impl TilingConfig {
    /// Create configuration for GPU Q4_K MatVec
    ///
    /// Optimized for single-token generation where M=1.
    #[must_use]
    pub fn gpu_q4k_matvec() -> Self {
        Self {
            name: "Q4K_MatVec_GPU".into(),
            macro_tile: TcbGeometry::with_alignment(1, 4096, 256, 64),
            midi_tile: TcbGeometry::with_alignment(1, 256, 256, 64),
            micro_tile: TcbGeometry::with_alignment(1, 32, 256, 64),
            backend: TilingBackend::Gpu,
        }
    }

    /// Create configuration for GPU Q4_K MatMul (batched)
    ///
    /// Optimized for prefill where M > 1.
    #[must_use]
    pub fn gpu_q4k_matmul() -> Self {
        Self {
            name: "Q4K_MatMul_GPU".into(),
            macro_tile: TcbGeometry::with_alignment(128, 128, 256, 64),
            midi_tile: TcbGeometry::with_alignment(32, 32, 256, 64),
            micro_tile: TcbGeometry::with_alignment(8, 8, 256, 64),
            backend: TilingBackend::Gpu,
        }
    }

    /// Create configuration for GPU Softmax
    #[must_use]
    pub fn gpu_softmax() -> Self {
        Self {
            name: "Softmax_GPU".into(),
            macro_tile: TcbGeometry::with_alignment(1, 32000, 1, 64),
            midi_tile: TcbGeometry::with_alignment(1, 1024, 1, 64),
            micro_tile: TcbGeometry::with_alignment(1, 32, 1, 64),
            backend: TilingBackend::Gpu,
        }
    }

    /// Create configuration for CPU AVX-512 MatMul
    ///
    /// Optimized for 512-bit wide SIMD:
    /// - 16 floats per ZMM register
    /// - 32 ZMM registers available
    /// - 4×16 micro-kernel uses 8 registers (4 accumulators + 4 scratch)
    #[must_use]
    pub fn cpu_avx512_matmul() -> Self {
        Self {
            name: "MatMul_AVX512".into(),
            macro_tile: TcbGeometry::with_alignment(512, 512, 512, 64),
            midi_tile: TcbGeometry::with_alignment(128, 128, 128, 64),
            // 16 floats wide × 4 rows = 64 elements in registers
            micro_tile: TcbGeometry::with_alignment(4, 16, 128, 64),
            backend: TilingBackend::CpuAvx512,
        }
    }

    /// Create configuration for CPU AVX-512 Q4K MatVec
    ///
    /// Optimized for Q4_K quantized inference with 512-bit SIMD.
    /// Key differences from AVX2:
    /// - 64-byte aligned for cache line optimization
    /// - 4×1 micro-kernel processes 4 rows simultaneously
    /// - K=256 aligned to Q4_K superblock
    #[must_use]
    pub fn cpu_avx512_q4k_matvec() -> Self {
        Self {
            name: "Q4K_MatVec_AVX512".into(),
            // Large macro-tile to amortize L3 access
            macro_tile: TcbGeometry::with_alignment(4096, 1, 4096, 64),
            // Midi-tile fits in L2 (256KB)
            // 64 rows × 256 K × 0.5625 bytes/element ≈ 9KB weights
            midi_tile: TcbGeometry::with_alignment(64, 1, 256, 64),
            // 4 rows × 1 output, K=256 (Q4_K superblock)
            micro_tile: TcbGeometry::with_alignment(4, 1, 256, 64),
            backend: TilingBackend::CpuAvx512,
        }
    }

    /// Create configuration for AVX-512 VNNI Q4K×Q8K integer dot product
    ///
    /// AVX-512 VNNI (Vector Neural Network Instructions) provides:
    /// - VPDPBUSD: 8-bit unsigned × 8-bit signed multiply-add to i32
    /// - VPDPWSSD: 16-bit signed × 16-bit signed multiply-add to i32
    ///
    /// This enables pure integer Q4K×Q8K without intermediate f32 conversion.
    #[must_use]
    pub fn cpu_avx512_vnni_q4k_q8k() -> Self {
        Self {
            name: "Q4K_Q8K_VNNI".into(),
            macro_tile: TcbGeometry::with_alignment(4096, 1, 4096, 64),
            midi_tile: TcbGeometry::with_alignment(64, 1, 256, 64),
            // VNNI processes 64 i8 values per ZMM register
            micro_tile: TcbGeometry::with_alignment(4, 1, 256, 64),
            backend: TilingBackend::CpuAvx512,
        }
    }

    /// Create configuration for CPU AVX2 MatMul
    #[must_use]
    pub fn cpu_avx2_matmul() -> Self {
        Self {
            name: "MatMul_AVX2".into(),
            macro_tile: TcbGeometry::with_alignment(256, 256, 256, 32),
            midi_tile: TcbGeometry::with_alignment(64, 64, 64, 32),
            // 8 floats wide × 4 rows = 32 elements in registers
            micro_tile: TcbGeometry::with_alignment(4, 8, 64, 32),
            backend: TilingBackend::CpuAvx2,
        }
    }

    /// Create configuration for CPU Q4_K MatVec (AVX2)
    #[must_use]
    pub fn cpu_avx2_q4k_matvec() -> Self {
        Self {
            name: "Q4K_MatVec_AVX2".into(),
            // Process 4 rows at a time (4×1 micro-kernel)
            macro_tile: TcbGeometry::with_alignment(4096, 1, 4096, 32),
            midi_tile: TcbGeometry::with_alignment(64, 1, 256, 32),
            // 4 rows × 1 output, K=256 (Q4_K superblock)
            micro_tile: TcbGeometry::with_alignment(4, 1, 256, 32),
            backend: TilingBackend::CpuAvx2,
        }
    }

    /// Create configuration for RMSNorm (CPU)
    #[must_use]
    pub fn cpu_rmsnorm() -> Self {
        Self {
            name: "RMSNorm_CPU".into(),
            macro_tile: TcbGeometry::with_alignment(1, 4096, 1, 32),
            midi_tile: TcbGeometry::with_alignment(1, 256, 1, 32),
            micro_tile: TcbGeometry::with_alignment(1, 16, 1, 32),
            backend: TilingBackend::CpuAvx512,
        }
    }

    /// Validate that tiling configuration is internally consistent
    pub fn validate(&self) -> Result<(), TilingError> {
        // Macro must be >= Midi >= Micro
        if self.midi_tile.m > self.macro_tile.m
            || self.midi_tile.n > self.macro_tile.n
            || self.midi_tile.k > self.macro_tile.k
        {
            return Err(TilingError::InvalidHierarchy {
                reason: "Midi-tile larger than macro-tile".into(),
            });
        }

        if self.micro_tile.m > self.midi_tile.m
            || self.micro_tile.n > self.midi_tile.n
            || self.micro_tile.k > self.midi_tile.k
        {
            return Err(TilingError::InvalidHierarchy {
                reason: "Micro-tile larger than midi-tile".into(),
            });
        }

        // Check divisibility
        if self.macro_tile.m % self.midi_tile.m != 0 {
            return Err(TilingError::DivisibilityError {
                level: "macro/midi",
                dimension: "M",
                larger: self.macro_tile.m,
                smaller: self.midi_tile.m,
            });
        }

        if self.midi_tile.m % self.micro_tile.m != 0 {
            return Err(TilingError::DivisibilityError {
                level: "midi/micro",
                dimension: "M",
                larger: self.midi_tile.m,
                smaller: self.micro_tile.m,
            });
        }

        Ok(())
    }

    /// Calculate total number of macro-tiles for given problem size
    #[must_use]
    pub fn num_macro_tiles(&self, m: u32, n: u32) -> u32 {
        let m_tiles = (m + self.macro_tile.m - 1) / self.macro_tile.m;
        let n_tiles = (n + self.macro_tile.n - 1) / self.macro_tile.n;
        m_tiles * n_tiles
    }

    /// Calculate total number of midi-tiles within a macro-tile
    #[must_use]
    pub fn midi_tiles_per_macro(&self) -> u32 {
        let m_tiles = self.macro_tile.m / self.midi_tile.m;
        let n_tiles = self.macro_tile.n / self.midi_tile.n;
        m_tiles * n_tiles
    }

    /// Calculate total number of micro-tiles within a midi-tile
    #[must_use]
    pub fn micro_tiles_per_midi(&self) -> u32 {
        let m_tiles = self.midi_tile.m / self.micro_tile.m;
        let n_tiles = self.midi_tile.n / self.micro_tile.n;
        m_tiles * n_tiles
    }
}
