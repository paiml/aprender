//! PMAT-054B: W4A16 pre-computed scale Q4K layout for fast tensor core dequant.
//!
//! The key insight from vLLM's Marlin kernel: GPU-side dequantization must be
//! trivially simple (2-3 instructions) for tensor core kernels to be competitive.
//! Standard Q4K requires ~20 instructions per element (nested GGML scale/min
//! decoding with 5-8 scattered global loads). This format pre-computes the
//! effective FP16 scales on CPU, reducing GPU dequant to:
//!
//!   1. Load 4-bit nibble (1 byte load, coalesced)
//!   2. Extract nibble (shift + mask, 2 insn)
//!   3. FMA: val = nibble * eff_scale - eff_min (2 insn)
//!
//! ## Tile Layout (2560 bytes = 16 columns × 1 Q4K super-block)
//!
//! ```text
//! Offset 0-255:    eff_scale[8 sub-blocks × 16 cols × 2B FP16]  = 256 bytes
//! Offset 256-511:  eff_min[8 sub-blocks × 16 cols × 2B FP16]    = 256 bytes
//! Offset 512-2559: qs nibbles [128 bytes × 16 cols, interleaved] = 2048 bytes
//! ```
//!
//! Scale layout: `eff_scale[sub_block * 16 + col]` — 16 threads loading the same
//! sub-block access 16 consecutive FP16 values (32 bytes, perfect coalescing).
//!
//! qs layout: same byte-interleaving as PMAT-091 interleaved format —
//! `qs[byte_i * 16 + col]` for coalesced 128-byte cache line access.

use super::super::{Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};

/// Size of one W4A16 tile in bytes
pub const W4A16_TILE_BYTES: usize = 2560;

/// Number of columns per tile (same as interleaved)
pub const W4A16_TILE_COLS: usize = 16;

/// Byte offset of pre-computed effective scales within a tile
pub const W4A16_SCALE_OFFSET: usize = 0;
/// Byte offset of pre-computed effective mins within a tile
pub const W4A16_MIN_OFFSET: usize = 256;
/// Byte offset of byte-interleaved qs nibbles within a tile
pub const W4A16_QS_OFFSET: usize = 512;

/// Number of sub-blocks per Q4K super-block
const NUM_SUB_BLOCKS: usize = 8;
/// Values per sub-block
const SUB_BLOCK_SIZE: usize = 32;

/// Q4K super-block field offsets
const SB_D_OFFSET: usize = 0;
const SB_DMIN_OFFSET: usize = 2;
const SB_SCALES_OFFSET: usize = 4;
const SB_QS_OFFSET: usize = 16;
const SB_QS_SIZE: usize = 128;

/// Decode FP16 LE bytes to f32 (minimal, no half crate dependency)
fn f16_to_f32(bytes: [u8; 2]) -> f32 {
    let h = u16::from_le_bytes(bytes) as u32;
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let frac = h & 0x3FF;
    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
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

/// Encode f32 to FP16 LE bytes (minimal)
fn f32_to_f16(val: f32) -> [u8; 2] {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let frac = bits & 0x7FFFFF;
    let h = if exp > 15 {
        (sign << 15) | 0x7C00
    } else if exp < -14 {
        sign << 15
    } else {
        let h_exp = ((exp + 15) as u32) & 0x1F;
        let h_frac = frac >> 13;
        (sign << 15) | (h_exp << 10) | h_frac
    };
    (h as u16).to_le_bytes()
}

/// Extract Q4K scale and min integers for a given sub-block index (0-7).
///
/// Returns (scale_int, min_int) decoded from the 12-byte GGML scale format.
fn extract_scale_min(scales: &[u8], sub_block: usize) -> (u8, u8) {
    debug_assert!(scales.len() == 12);
    debug_assert!(sub_block < 8);

    if sub_block < 4 {
        // Low sub-blocks (0-3): 6-bit values from bytes 0-3 (scale) and 4-7 (min)
        let scale = scales[sub_block] & 0x3F;
        let min = scales[4 + sub_block] & 0x3F;
        (scale, min)
    } else {
        // High sub-blocks (4-7): combine low bits from combo bytes 8-11 with
        // high 2 bits from the upper portion of bytes 0-3 and 4-7
        let i = sub_block - 4;
        let combo = scales[8 + i];

        // Scale: low 4 bits from combo, high 2 bits from scales[i] >> 6
        let sc_low4 = combo & 0x0F;
        let sc_high2 = (scales[i] >> 6) & 0x03;
        let scale = sc_low4 | (sc_high2 << 4);

        // Min: bits 4-7 from combo, high 2 bits from scales[4+i] >> 6
        let mn_low4 = (combo >> 4) & 0x0F;
        let mn_high2 = (scales[4 + i] >> 6) & 0x03;
        let min = mn_low4 | (mn_high2 << 4);

        (scale, min)
    }
}

/// Repack Q4K weights to W4A16 format with pre-computed FP16 scales.
///
/// Pre-computes `eff_scale[sb] = d * scale_int[sb]` and `eff_min[sb] = dmin * min_int[sb]`
/// as FP16, eliminating all complex scale decoding from the GPU kernel.
///
/// # Arguments
/// * `src` — Raw Q4K weight bytes in original GGML format
/// * `n` — Number of output neurons (rows)
/// * `k` — Input dimension (must be multiple of 256)
///
/// # Returns
/// W4A16 weight bytes. Size: `ceil(N/16) * num_sb * 2560`.
pub fn repack_q4k_w4a16(src: &[u8], n: usize, k: usize) -> Vec<u8> {
    assert!(k % Q4K_SUPER_BLOCK_SIZE as usize == 0, "K must be multiple of 256");
    let num_sb = k / Q4K_SUPER_BLOCK_SIZE as usize;
    let sb_bytes = Q4K_SUPER_BLOCK_BYTES as usize;
    assert_eq!(
        src.len(),
        n * num_sb * sb_bytes,
        "src length {} != N({}) × num_sb({}) × {}",
        src.len(),
        n,
        num_sb,
        n * num_sb * sb_bytes
    );

    let n_tiles = (n + W4A16_TILE_COLS - 1) / W4A16_TILE_COLS;
    let mut dst = vec![0u8; n_tiles * num_sb * W4A16_TILE_BYTES];

    for tile_idx in 0..n_tiles {
        let col_base = tile_idx * W4A16_TILE_COLS;

        for sb_idx in 0..num_sb {
            let tile_offset = (tile_idx * num_sb + sb_idx) * W4A16_TILE_BYTES;

            for col_in_tile in 0..W4A16_TILE_COLS {
                let global_col = col_base + col_in_tile;
                let clamped_col = global_col.min(n - 1);
                let sb_src_offset = (clamped_col * num_sb + sb_idx) * sb_bytes;

                // Read d and dmin as FP16→f32
                let d = f16_to_f32([
                    src[sb_src_offset + SB_D_OFFSET],
                    src[sb_src_offset + SB_D_OFFSET + 1],
                ]);
                let dmin = f16_to_f32([
                    src[sb_src_offset + SB_DMIN_OFFSET],
                    src[sb_src_offset + SB_DMIN_OFFSET + 1],
                ]);

                // Pre-compute effective scale and min for each sub-block
                let scales =
                    &src[sb_src_offset + SB_SCALES_OFFSET..sb_src_offset + SB_SCALES_OFFSET + 12];
                for sb_sub in 0..NUM_SUB_BLOCKS {
                    let (scale_int, min_int) = extract_scale_min(scales, sb_sub);
                    let eff_scale = d * scale_int as f32;
                    let eff_min = dmin * min_int as f32;

                    // Store as FP16: eff_scale[sb_sub * 16 + col]
                    let scale_dst = tile_offset
                        + W4A16_SCALE_OFFSET
                        + (sb_sub * W4A16_TILE_COLS + col_in_tile) * 2;
                    let scale_bytes = f32_to_f16(eff_scale);
                    dst[scale_dst] = scale_bytes[0];
                    dst[scale_dst + 1] = scale_bytes[1];

                    // Store as FP16: eff_min[sb_sub * 16 + col]
                    let min_dst = tile_offset
                        + W4A16_MIN_OFFSET
                        + (sb_sub * W4A16_TILE_COLS + col_in_tile) * 2;
                    let min_bytes = f32_to_f16(eff_min);
                    dst[min_dst] = min_bytes[0];
                    dst[min_dst + 1] = min_bytes[1];
                }

                // Byte-interleave qs (same as PMAT-091 interleaved format)
                for byte_i in 0..SB_QS_SIZE {
                    let qs_dst =
                        tile_offset + W4A16_QS_OFFSET + byte_i * W4A16_TILE_COLS + col_in_tile;
                    dst[qs_dst] = src[sb_src_offset + SB_QS_OFFSET + byte_i];
                }
            }
        }
    }

    dst
}

/// Compute the size of W4A16 weight buffer for given dimensions.
#[must_use]
pub fn w4a16_size(n: usize, k: usize) -> usize {
    let num_sb = k / Q4K_SUPER_BLOCK_SIZE as usize;
    let n_tiles = (n + W4A16_TILE_COLS - 1) / W4A16_TILE_COLS;
    n_tiles * num_sb * W4A16_TILE_BYTES
}

#[cfg(test)]
#[allow(clippy::identity_op, clippy::erasing_op)]
mod tests {
    use super::*;

    #[test]
    fn test_w4a16_size() {
        // N=1536, K=1536 → 6 SBs, 96 tiles
        assert_eq!(w4a16_size(1536, 1536), 96 * 6 * W4A16_TILE_BYTES);
        // N=8960, K=1536 → 6 SBs, 560 tiles
        assert_eq!(w4a16_size(8960, 1536), 560 * 6 * W4A16_TILE_BYTES);
    }

    #[test]
    fn test_w4a16_size_non_aligned() {
        // N=17 → 2 tiles (16 + 1 with padding)
        assert_eq!(w4a16_size(17, 256), 2 * 1 * W4A16_TILE_BYTES);
    }

    #[test]
    fn test_repack_preserves_total_size() {
        let n = 32;
        let k = 256;
        let src = vec![0u8; n * 1 * Q4K_SUPER_BLOCK_BYTES as usize];
        let dst = repack_q4k_w4a16(&src, n, k);
        assert_eq!(dst.len(), 2 * 1 * W4A16_TILE_BYTES);
    }

    #[test]
    fn test_extract_scale_min_low() {
        // Sub-blocks 0-3: scale = byte & 0x3F, min = byte+4 & 0x3F
        let mut scales = [0u8; 12];
        scales[0] = 0x15; // scale[0] = 0x15 & 0x3F = 21
        scales[4] = 0x0A; // min[0] = 0x0A & 0x3F = 10
        let (s, m) = extract_scale_min(&scales, 0);
        assert_eq!(s, 21);
        assert_eq!(m, 10);
    }

    #[test]
    fn test_extract_scale_min_high() {
        // Sub-block 4: uses combo byte at index 8
        let mut scales = [0u8; 12];
        // For sub-block 4 (i=0):
        // combo = scales[8], sc_low4 = combo & 0xF, sc_high2 = scales[0] >> 6
        // mn_low4 = (combo >> 4) & 0xF, mn_high2 = scales[4] >> 6
        scales[8] = 0x37; // sc_low4 = 7, mn_low4 = 3
        scales[0] = 0x80; // sc_high2 = (0x80 >> 6) & 3 = 2
        scales[4] = 0x40; // mn_high2 = (0x40 >> 6) & 3 = 1
        let (s, m) = extract_scale_min(&scales, 4);
        assert_eq!(s, 7 | (2 << 4)); // 7 + 32 = 39
        assert_eq!(m, 3 | (1 << 4)); // 3 + 16 = 19
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
            // d = 0.5
            let d_bytes = f32_to_f16(0.5);
            src[offset] = d_bytes[0];
            src[offset + 1] = d_bytes[1];
            // dmin = 0.1
            let dmin_bytes = f32_to_f16(0.1);
            src[offset + 2] = dmin_bytes[0];
            src[offset + 3] = dmin_bytes[1];
            // scales: all 1s
            for i in 0..12 {
                src[offset + 4 + i] = 1;
            }
            // qs: repeating pattern
            for i in 0..128 {
                src[offset + 16 + i] = ((i % 16) | ((i % 16) << 4)) as u8;
            }
        }

        let dst = repack_q4k_w4a16(&src, n, k);

        // Dequant col 5, sub-block 0, value 0 from original
        let col = 5;
        let sb_offset = col * sb_bytes;
        let d = f16_to_f32([src[sb_offset], src[sb_offset + 1]]);
        let dmin = f16_to_f32([src[sb_offset + 2], src[sb_offset + 3]]);
        let (scale_int, min_int) = extract_scale_min(&src[sb_offset + 4..sb_offset + 16], 0);
        let quant = (src[sb_offset + SB_QS_OFFSET] & 0x0F) as f32;
        let original_val = d * scale_int as f32 * quant - dmin * min_int as f32;

        // Dequant from W4A16 format
        let eff_scale_off = W4A16_SCALE_OFFSET + (0 * W4A16_TILE_COLS + col) * 2;
        let eff_scale = f16_to_f32([dst[eff_scale_off], dst[eff_scale_off + 1]]);
        let eff_min_off = W4A16_MIN_OFFSET + (0 * W4A16_TILE_COLS + col) * 2;
        let eff_min = f16_to_f32([dst[eff_min_off], dst[eff_min_off + 1]]);
        let qs_byte = dst[W4A16_QS_OFFSET + 0 * W4A16_TILE_COLS + col];
        let quant_w4 = (qs_byte & 0x0F) as f32;
        let w4a16_val = eff_scale * quant_w4 - eff_min;

        assert!(
            (original_val - w4a16_val).abs() < 0.01,
            "Roundtrip mismatch: original={} w4a16={}",
            original_val,
            w4a16_val
        );
    }

    #[test]
    fn test_repack_qs_interleaving() {
        let n = 16;
        let k = 256;
        let sb_bytes = Q4K_SUPER_BLOCK_BYTES as usize;
        let mut src = vec![0u8; n * sb_bytes];

        for col in 0..16u8 {
            let offset = col as usize * sb_bytes + SB_QS_OFFSET;
            src[offset] = col + 0x10; // distinct value per column
        }

        let dst = repack_q4k_w4a16(&src, n, k);

        // Verify byte-interleaving
        for col in 0..16u8 {
            let qs_offset = W4A16_QS_OFFSET + 0 * W4A16_TILE_COLS + col as usize;
            assert_eq!(dst[qs_offset], col + 0x10, "qs interleave failed col {}", col);
        }
    }

    #[test]
    fn test_repack_effective_scales_coalesced() {
        let n = 16;
        let k = 256;
        let sb_bytes = Q4K_SUPER_BLOCK_BYTES as usize;
        let mut src = vec![0u8; n * sb_bytes];

        // Set d=2.0, scale[0]=3 for all columns
        for col in 0..16 {
            let offset = col * sb_bytes;
            let d_bytes = f32_to_f16(2.0);
            src[offset] = d_bytes[0];
            src[offset + 1] = d_bytes[1];
            // scale[0] = 3 (low sub-block, byte & 0x3F)
            src[offset + SB_SCALES_OFFSET] = 3;
        }

        let dst = repack_q4k_w4a16(&src, n, k);

        // eff_scale[sub_block=0, col=0..15] should all be 2.0 * 3 = 6.0
        for col in 0..16 {
            let off = W4A16_SCALE_OFFSET + (0 * W4A16_TILE_COLS + col) * 2;
            let val = f16_to_f32([dst[off], dst[off + 1]]);
            assert!((val - 6.0).abs() < 0.1, "col {} eff_scale = {} expected 6.0", col, val);
        }
    }

    #[test]
    fn test_repack_padding_columns() {
        let n = 17;
        let k = 256;
        let sb_bytes = Q4K_SUPER_BLOCK_BYTES as usize;
        let mut src = vec![0u8; n * sb_bytes];

        // Set col 16 d to 1.0
        let offset = 16 * sb_bytes;
        let d_bytes = f32_to_f16(1.0);
        src[offset] = d_bytes[0];
        src[offset + 1] = d_bytes[1];
        src[offset + SB_SCALES_OFFSET] = 5; // scale[0] = 5

        let dst = repack_q4k_w4a16(&src, n, k);

        // Tile 1, col_in_tile 0 = global col 16: eff_scale = 1.0 * 5 = 5.0
        let tile1_offset = W4A16_TILE_BYTES;
        let off = tile1_offset + W4A16_SCALE_OFFSET + (0 * W4A16_TILE_COLS + 0) * 2;
        let val = f16_to_f32([dst[off], dst[off + 1]]);
        assert!((val - 5.0).abs() < 0.1, "Padded tile col 0 eff_scale={}", val);

        // Padded columns (1-15) clone col 16
        let off_pad = tile1_offset + W4A16_SCALE_OFFSET + (0 * W4A16_TILE_COLS + 1) * 2;
        let val_pad = f16_to_f32([dst[off_pad], dst[off_pad + 1]]);
        assert!((val_pad - 5.0).abs() < 0.1, "Padded col should clone: eff_scale={}", val_pad);
    }

    #[test]
    fn test_repack_multiple_sbs() {
        let n = 16;
        let k = 512;
        let num_sb = 2;
        let sb_bytes = Q4K_SUPER_BLOCK_BYTES as usize;
        let mut src = vec![0u8; n * num_sb * sb_bytes];

        // Set col 3, sb 1 qs[0] to 0xAB
        let sb_offset = (3 * num_sb + 1) * sb_bytes;
        src[sb_offset + SB_QS_OFFSET] = 0xAB;

        let dst = repack_q4k_w4a16(&src, n, k);

        // Tile 0, sb 1, col 3, byte 0
        let tile_offset = 1 * W4A16_TILE_BYTES; // sb_idx=1
        let qs_byte = dst[tile_offset + W4A16_QS_OFFSET + 0 * W4A16_TILE_COLS + 3];
        assert_eq!(qs_byte, 0xAB, "Multi-SB repack failed");
    }
}
