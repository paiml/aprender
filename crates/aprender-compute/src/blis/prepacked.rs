//! Pre-packed B matrix for BLIS GEMM.
//!
//! Eliminates redundant B packing in parallel GEMM by pre-packing all
//! (jc, pc) tiles once at weight-load time. The packed data can then be
//! shared immutably across all threads without per-thread repacking.
//!
//! # Motivation (WAPR-KAIZEN Cycle 12)
//!
//! In `gemm_blis_parallel`, each thread independently calls `gemm_blis` which
//! packs B internally via `pack_b_block()`. For encoder FFN with 16 threads,
//! 2 GEMMs per block, and 4 layers, this results in 128 redundant B packings
//! per encoder pass. Pre-packing eliminates this entirely.
//!
//! # References
//!
//! - Van Zee & Van de Geijn (2015): BLIS framework, Section 3.2 (packing)

use super::packing::{pack_b_block, packed_b_size};
use super::{KC, NC};

/// Pre-packed B matrix in BLIS tile format.
///
/// Stores all (jc, pc) tiles of a B matrix (k × n, row-major) in the packed
/// micro-panel layout expected by BLIS microkernels. Once constructed, this
/// is immutable and can be shared across threads via `&PrepackedB`.
#[derive(Debug, Clone)]
pub struct PrepackedB {
    /// Flat buffer of all pre-packed B tiles
    data: Vec<f32>,
    /// Original K dimension
    pub k: usize,
    /// Original N dimension
    pub n: usize,
    /// Offset of each (jc_idx, pc_idx) tile in `data`
    tile_offsets: Vec<usize>,
    /// Size of each (jc_idx, pc_idx) tile
    tile_sizes: Vec<usize>,
    /// Number of pc tiles (K-dimension)
    num_pc_tiles: usize,
}

impl PrepackedB {
    /// Pre-pack a B matrix (k × n, row-major) into BLIS tile format.
    ///
    /// This iterates the same (jc, pc) loop as `gemm_blis` and packs each
    /// B tile into the NR-aligned micro-panel layout. The result can be
    /// reused across many GEMM calls with different A matrices.
    ///
    /// # Panics
    ///
    /// Panics if `b.len() != k * n`.
    pub fn pack(b: &[f32], k: usize, n: usize) -> Self {
        assert_eq!(b.len(), k * n, "B size mismatch: expected {}, got {}", k * n, b.len());

        if k == 0 || n == 0 {
            return Self {
                data: Vec::new(),
                k,
                n,
                tile_offsets: Vec::new(),
                tile_sizes: Vec::new(),
                num_pc_tiles: 0,
            };
        }

        let num_jc = (n + NC - 1) / NC;
        let num_pc = (k + KC - 1) / KC;
        let num_tiles = num_jc * num_pc;

        // First pass: compute tile sizes and cumulative offsets
        let mut tile_offsets = Vec::with_capacity(num_tiles);
        let mut tile_sizes = Vec::with_capacity(num_tiles);
        let mut total_size = 0;

        for jc in (0..n).step_by(NC) {
            let nc_block = NC.min(n - jc);
            for pc in (0..k).step_by(KC) {
                let kc_block = KC.min(k - pc);
                let size = packed_b_size(kc_block, nc_block);
                tile_offsets.push(total_size);
                tile_sizes.push(size);
                total_size += size;
            }
        }

        // Second pass: pack all tiles
        let mut data = vec![0.0_f32; total_size];
        let mut tile_idx = 0;

        for jc in (0..n).step_by(NC) {
            let nc_block = NC.min(n - jc);
            for pc in (0..k).step_by(KC) {
                let kc_block = KC.min(k - pc);
                let offset = tile_offsets[tile_idx];
                let size = tile_sizes[tile_idx];
                pack_b_block(b, n, pc, jc, kc_block, nc_block, &mut data[offset..offset + size]);
                tile_idx += 1;
            }
        }

        Self { data, k, n, tile_offsets, tile_sizes, num_pc_tiles: num_pc }
    }

    /// Get the pre-packed tile for the given (jc, pc) tile indices.
    ///
    /// `jc_idx` = jc / NC, `pc_idx` = pc / KC
    #[inline]
    pub fn tile(&self, jc_idx: usize, pc_idx: usize) -> &[f32] {
        let idx = jc_idx * self.num_pc_tiles + pc_idx;
        let offset = self.tile_offsets[idx];
        let size = self.tile_sizes[idx];
        &self.data[offset..offset + size]
    }

    /// Total memory used by packed data (bytes).
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// Number of packed tiles.
    #[must_use]
    pub fn num_tiles(&self) -> usize {
        self.tile_offsets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepack_empty() {
        let pb = PrepackedB::pack(&[], 0, 0);
        assert_eq!(pb.k, 0);
        assert_eq!(pb.n, 0);
        assert_eq!(pb.num_tiles(), 0);
        assert_eq!(pb.memory_bytes(), 0);
    }

    #[test]
    fn test_prepack_small() {
        // 4x4 matrix — small enough for a single tile
        let b: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let pb = PrepackedB::pack(&b, 4, 4);
        assert_eq!(pb.k, 4);
        assert_eq!(pb.n, 4);
        assert!(pb.num_tiles() > 0);
        assert!(pb.memory_bytes() > 0);
    }

    #[test]
    fn test_prepack_dimensions() {
        // Whisper-tiny fc1: B is 384×1536 (transposed weights)
        let k = 384;
        let n = 1536;
        let b = vec![0.0_f32; k * n];
        let pb = PrepackedB::pack(&b, k, n);
        assert_eq!(pb.k, k);
        assert_eq!(pb.n, n);

        let num_jc = (n + NC - 1) / NC;
        let num_pc = (k + KC - 1) / KC;
        assert_eq!(pb.num_tiles(), num_jc * num_pc);
    }

    #[test]
    fn test_prepack_tile_access() {
        let k = 384;
        let n = 384;
        let b = vec![1.0_f32; k * n];
        let pb = PrepackedB::pack(&b, k, n);

        // Access first tile
        let tile = pb.tile(0, 0);
        assert!(!tile.is_empty());
    }

    #[test]
    #[should_panic(expected = "B size mismatch")]
    fn test_prepack_size_mismatch() {
        PrepackedB::pack(&[1.0, 2.0], 4, 4);
    }

    /// Golden test: gemm_blis_with_prepacked_b must produce identical output to gemm_blis.
    #[test]
    fn test_prepacked_matches_gemm_blis() {
        use crate::blis::compute::{gemm_blis, gemm_blis_with_prepacked_b};

        let m = 128;
        let k = 64;
        let n = 96;

        // Deterministic pseudo-random data
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 13) % 97) as f32 / 97.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 11 + 3) % 89) as f32 / 89.0).collect();

        // Standard GEMM
        let mut c_standard = vec![0.0_f32; m * n];
        gemm_blis(m, n, k, &a, &b, &mut c_standard, None).unwrap();

        // Pre-packed GEMM
        let prepacked = PrepackedB::pack(&b, k, n);
        let mut c_prepacked = vec![0.0_f32; m * n];
        gemm_blis_with_prepacked_b(m, n, k, &a, &prepacked, &mut c_prepacked, None).unwrap();

        // Must be bit-identical (same packing, same microkernel)
        for i in 0..m * n {
            assert!(
                (c_standard[i] - c_prepacked[i]).abs() < 1e-5,
                "Mismatch at index {i}: standard={}, prepacked={}",
                c_standard[i],
                c_prepacked[i]
            );
        }
    }

    /// Golden test for parallel pre-packed GEMM.
    #[test]
    fn test_prepacked_parallel_matches_standard() {
        use crate::blis::parallel::{gemm_blis_parallel, gemm_blis_parallel_with_prepacked_b};

        // Use dimensions large enough to trigger parallel path (m*n*k >= 1_000_000)
        let m = 256;
        let k = 128;
        let n = 64;

        let a: Vec<f32> = (0..m * k).map(|i| ((i * 7 + 13) % 97) as f32 / 97.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i * 11 + 3) % 89) as f32 / 89.0).collect();

        // Standard parallel GEMM
        let mut c_standard = vec![0.0_f32; m * n];
        gemm_blis_parallel(m, n, k, &a, &b, &mut c_standard).unwrap();

        // Pre-packed parallel GEMM
        let prepacked = PrepackedB::pack(&b, k, n);
        let mut c_prepacked = vec![0.0_f32; m * n];
        gemm_blis_parallel_with_prepacked_b(m, n, k, &a, &prepacked, &mut c_prepacked).unwrap();

        for i in 0..m * n {
            assert!(
                (c_standard[i] - c_prepacked[i]).abs() < 1e-5,
                "Mismatch at index {i}: standard={}, prepacked={}",
                c_standard[i],
                c_prepacked[i]
            );
        }
    }
}
