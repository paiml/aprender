//! TCB geometry types and level definitions.

use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// TILE-001: TcbGeometry Struct
// ============================================================================

/// Dimensions for a Tiling Compute Block
///
/// Represents the (M, N, K) dimensions of a tile in matrix operations:
/// - M: Output rows
/// - N: Output columns
/// - K: Reduction dimension (inner product)
///
/// # Alignment Constraints
///
/// Per the TCB-03 pattern (Tile Quantization Alignment), K must align with
/// the quantization superblock size:
/// - Q4_0: K % 32 == 0
/// - Q4_K: K % 256 == 0
/// - Q8_0: K % 32 == 0
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TcbGeometry {
    /// Items processed in M dimension (rows)
    pub m: u32,
    /// Items processed in N dimension (columns)
    pub n: u32,
    /// Reduction dimension (inner product)
    pub k: u32,
    /// Alignment requirement in bytes (typically 16 for SIMD, 32 for AVX2, 64 for AVX-512)
    pub alignment: u32,
}

impl TcbGeometry {
    /// Create a new TCB geometry
    ///
    /// # Panics
    /// Panics if any dimension is zero.
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        assert!(m > 0 && n > 0 && k > 0, "TCB dimensions must be non-zero");
        Self {
            m,
            n,
            k,
            alignment: 16, // Default to SSE/NEON alignment
        }
    }

    /// Create geometry with explicit alignment
    #[must_use]
    pub fn with_alignment(m: u32, n: u32, k: u32, alignment: u32) -> Self {
        assert!(m > 0 && n > 0 && k > 0, "TCB dimensions must be non-zero");
        assert!(alignment.is_power_of_two(), "Alignment must be power of 2");
        Self { m, n, k, alignment }
    }

    /// Calculate arithmetic intensity (FLOPS per byte loaded)
    ///
    /// For GEMM: AI = (2 * M * N * K) / (M*K + K*N) * sizeof(f32)
    ///
    /// Higher AI means compute-bound; lower means memory-bound.
    #[must_use]
    pub fn arithmetic_intensity(&self) -> f32 {
        let flops = 2.0 * self.m as f64 * self.n as f64 * self.k as f64;
        let bytes = (self.m as f64 * self.k as f64 + self.k as f64 * self.n as f64) * 4.0;
        (flops / bytes) as f32
    }

    /// Calculate total elements in the tile
    #[must_use]
    pub fn total_elements(&self) -> u64 {
        self.m as u64 * self.n as u64
    }

    /// Calculate total FLOPs for this tile
    #[must_use]
    pub fn total_flops(&self) -> u64 {
        2 * self.m as u64 * self.n as u64 * self.k as u64
    }

    /// Check if K dimension aligns with Q4_K superblock (256)
    #[must_use]
    pub fn is_q4k_aligned(&self) -> bool {
        self.k % 256 == 0
    }

    /// Check if K dimension aligns with Q4_0/Q8_0 block (32)
    #[must_use]
    pub fn is_q4_0_aligned(&self) -> bool {
        self.k % 32 == 0
    }

    /// Calculate bytes needed for A tile (M × K × sizeof(f32))
    #[must_use]
    pub fn a_tile_bytes(&self) -> usize {
        self.m as usize * self.k as usize * 4
    }

    /// Calculate bytes needed for B tile (K × N × sizeof(f32))
    #[must_use]
    pub fn b_tile_bytes(&self) -> usize {
        self.k as usize * self.n as usize * 4
    }

    /// Calculate bytes needed for C tile (M × N × sizeof(f32))
    #[must_use]
    pub fn c_tile_bytes(&self) -> usize {
        self.m as usize * self.n as usize * 4
    }

    /// Check if tile fits in given cache size (bytes)
    #[must_use]
    pub fn fits_in_cache(&self, cache_bytes: usize) -> bool {
        self.a_tile_bytes() + self.b_tile_bytes() <= cache_bytes
    }
}

impl Default for TcbGeometry {
    fn default() -> Self {
        // Sensible default: 4×4 micro-tile for SIMD
        Self { m: 4, n: 4, k: 4, alignment: 16 }
    }
}

impl fmt::Display for TcbGeometry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TCB({}×{}×{}, align={}, AI={:.2})",
            self.m,
            self.n,
            self.k,
            self.alignment,
            self.arithmetic_intensity()
        )
    }
}
// ============================================================================
// TILE-001: Tiling Levels
// ============================================================================

/// Tiling hierarchy level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TcbLevel {
    /// Macro-tile: L3 cache / GPU global memory partitioning
    Macro,
    /// Midi-tile: L2 cache / GPU shared memory
    Midi,
    /// Micro-tile: Registers / SIMD lanes
    Micro,
}

impl TcbLevel {
    /// Get typical cache size for this level (x86_64)
    #[must_use]
    pub fn typical_cache_bytes(&self) -> usize {
        match self {
            TcbLevel::Macro => 32 * 1024 * 1024, // 32 MB L3
            TcbLevel::Midi => 256 * 1024,        // 256 KB L2
            TcbLevel::Micro => 32 * 1024,        // 32 KB L1
        }
    }
}
