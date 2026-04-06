//! PMAT-091: Column-interleaved Q4K weight layout for coalesced WMMA access.
//!
//! The standard Q4K layout stores super-blocks per output neuron:
//!   weights[n][sb_idx] = SB(144 bytes)
//!
//! When a WMMA tile loads 16 columns simultaneously, adjacent threads access
//! super-blocks ~864 bytes apart (num_sb × 144B stride). This causes
//! zero coalescing — each thread's load generates a separate 128-byte cache
//! line fetch, resulting in 8-16× bandwidth amplification.
//!
//! The interleaved layout groups 16 columns' super-blocks into tiles:
//!   tiles[n_tile][sb_idx] = InterleavedTile(2304 bytes)
//!
//! Within each tile, the qs nibble bytes are byte-interleaved across columns:
//!   tile.qs[byte_i * 16 + col] = original col's qs[byte_i]
//!
//! This ensures that 16 threads reading from 16 different columns access
//! 16 consecutive bytes — a single 128-byte cache line transaction.
//!
//! ## Tile Layout (2304 bytes = 16 × 144)
//!
//! ```text
//! Offset 0-31:    d values     (16 × f16, col-interleaved)
//! Offset 32-63:   dmin values  (16 × f16, col-interleaved)
//! Offset 64-255:  scales       (16 × 12 bytes, per-column contiguous)
//! Offset 256-2303: qs nibbles  (16 × 128 bytes, byte-interleaved)
//! ```
//!
//! Bandwidth savings: metadata loaded once per SB (amortized over 16 WMMA
//! iterations), qs bytes perfectly coalesced. Combined with W4A16 tensor
//! core compute: reads 0.5625 B/elem (Q4K) with WMMA FP16 matmul.

use super::super::{Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};

/// Size of one interleaved tile in bytes (16 columns × 144 bytes/SB)
pub const INTERLEAVED_TILE_BYTES: usize = 2304;

/// Number of columns per interleaved tile
pub const TILE_COLS: usize = 16;

/// Byte offset of d values within an interleaved tile
pub const TILE_D_OFFSET: usize = 0;
/// Byte offset of dmin values within an interleaved tile
pub const TILE_DMIN_OFFSET: usize = 32;
/// Byte offset of scales within an interleaved tile
pub const TILE_SCALES_OFFSET: usize = 64;
/// Byte offset of byte-interleaved qs within an interleaved tile
pub const TILE_QS_OFFSET: usize = 256;

/// Q4K super-block field offsets within original format
const SB_D_OFFSET: usize = 0;
const SB_DMIN_OFFSET: usize = 2;
const SB_SCALES_OFFSET: usize = 4;
const SB_QS_OFFSET: usize = 16;
const SB_SCALES_SIZE: usize = 12;
const SB_QS_SIZE: usize = 128;

/// Repack Q4K weights from per-neuron layout to column-interleaved tile layout.
///
/// Input layout: `[N × num_sb × 144B]` — each output neuron's super-blocks contiguous.
/// Output layout: `[N_tiles × num_sb × 2304B]` — 16-column tiles with interleaved qs.
///
/// # Arguments
/// * `src` — Raw Q4K weight bytes in original GGML format
/// * `n` — Number of output neurons (rows)
/// * `k` — Input dimension (must be multiple of 256)
///
/// # Returns
/// Interleaved weight bytes. Size: `ceil(N/16) * num_sb * 2304`.
///
/// # Panics
/// Panics if `k` is not a multiple of 256 or `src` length doesn't match `n * num_sb * 144`.
pub fn repack_q4k_interleaved(src: &[u8], n: usize, k: usize) -> Vec<u8> {
    assert!(k % Q4K_SUPER_BLOCK_SIZE as usize == 0, "K must be multiple of 256");
    let num_sb = k / Q4K_SUPER_BLOCK_SIZE as usize;
    let sb_bytes = Q4K_SUPER_BLOCK_BYTES as usize;
    assert_eq!(
        src.len(),
        n * num_sb * sb_bytes,
        "src length {} != N({}) × num_sb({}) × 144({})",
        src.len(),
        n,
        num_sb,
        n * num_sb * sb_bytes
    );

    let n_tiles = (n + TILE_COLS - 1) / TILE_COLS;
    let mut dst = vec![0u8; n_tiles * num_sb * INTERLEAVED_TILE_BYTES];

    for tile_idx in 0..n_tiles {
        let col_base = tile_idx * TILE_COLS;

        for sb_idx in 0..num_sb {
            let tile_offset = (tile_idx * num_sb + sb_idx) * INTERLEAVED_TILE_BYTES;

            for col_in_tile in 0..TILE_COLS {
                let global_col = col_base + col_in_tile;
                // Clamp to last valid column for padding tiles
                let clamped_col = global_col.min(n - 1);
                let sb_src_offset = (clamped_col * num_sb + sb_idx) * sb_bytes;

                // Copy d (2 bytes FP16)
                let d_dst = tile_offset + TILE_D_OFFSET + col_in_tile * 2;
                dst[d_dst..d_dst + 2].copy_from_slice(
                    &src[sb_src_offset + SB_D_OFFSET..sb_src_offset + SB_D_OFFSET + 2],
                );

                // Copy dmin (2 bytes FP16)
                let dmin_dst = tile_offset + TILE_DMIN_OFFSET + col_in_tile * 2;
                dst[dmin_dst..dmin_dst + 2].copy_from_slice(
                    &src[sb_src_offset + SB_DMIN_OFFSET..sb_src_offset + SB_DMIN_OFFSET + 2],
                );

                // Copy scales (12 bytes, per-column contiguous)
                let scales_dst = tile_offset + TILE_SCALES_OFFSET + col_in_tile * SB_SCALES_SIZE;
                dst[scales_dst..scales_dst + SB_SCALES_SIZE].copy_from_slice(
                    &src[sb_src_offset + SB_SCALES_OFFSET
                        ..sb_src_offset + SB_SCALES_OFFSET + SB_SCALES_SIZE],
                );

                // Byte-interleave qs (128 bytes → stride-16 interleaved)
                for byte_i in 0..SB_QS_SIZE {
                    let qs_dst = tile_offset + TILE_QS_OFFSET + byte_i * TILE_COLS + col_in_tile;
                    dst[qs_dst] = src[sb_src_offset + SB_QS_OFFSET + byte_i];
                }
            }
        }
    }

    dst
}

/// Compute the size of interleaved weight buffer for given dimensions.
#[must_use]
pub fn interleaved_size(n: usize, k: usize) -> usize {
    let num_sb = k / Q4K_SUPER_BLOCK_SIZE as usize;
    let n_tiles = (n + TILE_COLS - 1) / TILE_COLS;
    n_tiles * num_sb * INTERLEAVED_TILE_BYTES
}

#[cfg(test)]
#[allow(clippy::identity_op, clippy::erasing_op)]
mod tests {
    use super::*;

    /// Encode f32 to FP16 LE bytes (minimal, no half crate dependency)
    fn f32_to_f16_bytes(val: f32) -> [u8; 2] {
        let bits = val.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let frac = bits & 0x7FFFFF;
        let h = if exp > 15 {
            (sign << 15) | 0x7C00 // inf
        } else if exp < -14 {
            sign << 15 // zero/subnormal
        } else {
            let h_exp = ((exp + 15) as u32) & 0x1F;
            let h_frac = frac >> 13;
            (sign << 15) | (h_exp << 10) | h_frac
        };
        (h as u16).to_le_bytes()
    }

    /// Decode FP16 LE bytes to f32 (minimal)
    fn f16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
        let h = u16::from_le_bytes(bytes) as u32;
        let sign = (h >> 15) & 1;
        let exp = (h >> 10) & 0x1F;
        let frac = h & 0x3FF;
        if exp == 0 {
            if frac == 0 {
                f32::from_bits(sign << 31)
            } else {
                // Subnormal
                let val = (frac as f32) / 1024.0 * (2.0f32).powi(-14);
                if sign == 1 {
                    -val
                } else {
                    val
                }
            }
        } else if exp == 31 {
            if frac == 0 {
                if sign == 1 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                f32::NAN
            }
        } else {
            let f_exp = (exp as i32 - 15 + 127) as u32;
            let f_frac = frac << 13;
            f32::from_bits((sign << 31) | (f_exp << 23) | f_frac)
        }
    }

    #[test]
    fn test_interleaved_size() {
        // N=1536, K=1536 → 6 SBs, 96 tiles
        assert_eq!(interleaved_size(1536, 1536), 96 * 6 * 2304);
        // N=8960, K=1536 → 6 SBs, 560 tiles
        assert_eq!(interleaved_size(8960, 1536), 560 * 6 * 2304);
    }

    #[test]
    fn test_interleaved_size_non_aligned() {
        // N=17 → 2 tiles (16 + 1 with padding)
        assert_eq!(interleaved_size(17, 256), 2 * 1 * 2304);
    }

    #[test]
    fn test_repack_preserves_size() {
        let n = 32;
        let k = 256;
        let src = vec![0u8; n * 1 * Q4K_SUPER_BLOCK_BYTES as usize];
        let dst = repack_q4k_interleaved(&src, n, k);
        assert_eq!(dst.len(), 2 * 1 * INTERLEAVED_TILE_BYTES);
    }

    #[test]
    fn test_repack_d_values() {
        let n = 16;
        let k = 256;
        let sb_bytes = Q4K_SUPER_BLOCK_BYTES as usize;
        let mut src = vec![0u8; n * sb_bytes];

        // Set d for each column to a distinct FP16 value
        for col in 0..16 {
            let offset = col * sb_bytes;
            let bytes = f32_to_f16_bytes(col as f32 + 1.0);
            src[offset] = bytes[0];
            src[offset + 1] = bytes[1];
        }

        let dst = repack_q4k_interleaved(&src, n, k);

        // Verify d values are interleaved
        for col in 0..16 {
            let d_offset = TILE_D_OFFSET + col * 2;
            let d = f16_bytes_to_f32([dst[d_offset], dst[d_offset + 1]]);
            let expected = col as f32 + 1.0;
            assert!((d - expected).abs() < 0.1, "col {} d={} expected={}", col, d, expected);
        }
    }

    #[test]
    fn test_repack_qs_interleaving() {
        let n = 16;
        let k = 256;
        let sb_bytes = Q4K_SUPER_BLOCK_BYTES as usize;
        let mut src = vec![0u8; n * sb_bytes];

        // Set qs[0] for each column to col_idx
        for col in 0..16u8 {
            let offset = col as usize * sb_bytes + SB_QS_OFFSET;
            src[offset] = col;
        }

        let dst = repack_q4k_interleaved(&src, n, k);

        // Verify qs are byte-interleaved
        for col in 0..16u8 {
            let qs_offset = TILE_QS_OFFSET + 0 * TILE_COLS + col as usize;
            assert_eq!(dst[qs_offset], col, "qs interleave failed for col {}", col);
        }
    }

    #[test]
    fn test_repack_scales_per_column() {
        let n = 16;
        let k = 256;
        let sb_bytes = Q4K_SUPER_BLOCK_BYTES as usize;
        let mut src = vec![0u8; n * sb_bytes];

        for col in 0..16u8 {
            let offset = col as usize * sb_bytes + SB_SCALES_OFFSET;
            src[offset] = col + 100;
        }

        let dst = repack_q4k_interleaved(&src, n, k);

        for col in 0..16u8 {
            let scales_offset = TILE_SCALES_OFFSET + col as usize * SB_SCALES_SIZE;
            assert_eq!(dst[scales_offset], col + 100, "scales failed for col {}", col);
        }
    }

    #[test]
    fn test_repack_roundtrip_dequant() {
        let n = 16;
        let k = 256;
        let sb_bytes = Q4K_SUPER_BLOCK_BYTES as usize;
        let mut src = vec![0u8; n * sb_bytes];

        // Fill with semi-realistic data
        for col in 0..n {
            let offset = col * sb_bytes;
            let d_bytes = f32_to_f16_bytes(0.5);
            src[offset] = d_bytes[0];
            src[offset + 1] = d_bytes[1];
            let dmin_bytes = f32_to_f16_bytes(0.1);
            src[offset + 2] = dmin_bytes[0];
            src[offset + 3] = dmin_bytes[1];
            for i in 0..12 {
                src[offset + 4 + i] = 1;
            }
            for i in 0..128 {
                src[offset + 16 + i] = ((i % 16) | ((i % 16) << 4)) as u8;
            }
        }

        let dst = repack_q4k_interleaved(&src, n, k);

        // Dequant col 5, k=0 from original
        let col = 5;
        let sb_offset = col * sb_bytes;
        let d = f16_bytes_to_f32([src[sb_offset], src[sb_offset + 1]]);
        let dmin = f16_bytes_to_f32([src[sb_offset + 2], src[sb_offset + 3]]);
        let scale = (src[sb_offset + SB_SCALES_OFFSET] & 0x3F) as f32;
        let min = (src[sb_offset + SB_SCALES_OFFSET + 4] & 0x3F) as f32;
        let quant = (src[sb_offset + SB_QS_OFFSET] & 0x0F) as f32;
        let original_val = d * scale * quant - dmin * min;

        // Dequant from interleaved
        let d_il =
            f16_bytes_to_f32([dst[TILE_D_OFFSET + col * 2], dst[TILE_D_OFFSET + col * 2 + 1]]);
        let dmin_il = f16_bytes_to_f32([
            dst[TILE_DMIN_OFFSET + col * 2],
            dst[TILE_DMIN_OFFSET + col * 2 + 1],
        ]);
        let scale_il = (dst[TILE_SCALES_OFFSET + col * SB_SCALES_SIZE] & 0x3F) as f32;
        let min_il = (dst[TILE_SCALES_OFFSET + col * SB_SCALES_SIZE + 4] & 0x3F) as f32;
        let qs_byte_il = dst[TILE_QS_OFFSET + 0 * TILE_COLS + col];
        let quant_il = (qs_byte_il & 0x0F) as f32;
        let interleaved_val = d_il * scale_il * quant_il - dmin_il * min_il;

        assert!(
            (original_val - interleaved_val).abs() < 1e-4,
            "Roundtrip mismatch: original={} interleaved={}",
            original_val,
            interleaved_val
        );
    }

    #[test]
    fn test_repack_padding_columns() {
        let n = 17;
        let k = 256;
        let sb_bytes = Q4K_SUPER_BLOCK_BYTES as usize;
        let mut src = vec![0u8; n * sb_bytes];

        // Set col 16 d to 1.0
        let offset = 16 * sb_bytes;
        let d_bytes = f32_to_f16_bytes(1.0);
        src[offset] = d_bytes[0];
        src[offset + 1] = d_bytes[1];

        let dst = repack_q4k_interleaved(&src, n, k);

        // Tile 1, col_in_tile 0 = global col 16
        let tile1_offset = INTERLEAVED_TILE_BYTES;
        let d = f16_bytes_to_f32([
            dst[tile1_offset + TILE_D_OFFSET],
            dst[tile1_offset + TILE_D_OFFSET + 1],
        ]);
        assert!((d - 1.0).abs() < 0.01, "Padded tile col 0 d={}", d);

        // Padded columns clone last valid column (16)
        let d_pad = f16_bytes_to_f32([
            dst[tile1_offset + TILE_D_OFFSET + 2],
            dst[tile1_offset + TILE_D_OFFSET + 3],
        ]);
        assert!((d_pad - 1.0).abs() < 0.01, "Padded col should clone: d_pad={}", d_pad);
    }

    #[test]
    fn test_repack_multiple_sbs() {
        // N=16, K=512 → 2 SBs per column
        let n = 16;
        let k = 512;
        let num_sb = 2;
        let sb_bytes = Q4K_SUPER_BLOCK_BYTES as usize;
        let mut src = vec![0u8; n * num_sb * sb_bytes];

        // Set col 3, sb 1 qs[0] to 0xAB
        let sb_offset = (3 * num_sb + 1) * sb_bytes;
        src[sb_offset + SB_QS_OFFSET] = 0xAB;

        let dst = repack_q4k_interleaved(&src, n, k);

        // Tile 0, sb 1, col 3, byte 0
        let tile_offset = 1 * INTERLEAVED_TILE_BYTES; // sb_idx=1
        let qs_byte = dst[tile_offset + TILE_QS_OFFSET + 0 * TILE_COLS + 3];
        assert_eq!(qs_byte, 0xAB, "Multi-SB repack failed");
    }
}
