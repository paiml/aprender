//! `IQ3_XXS` (ggml type 18) dequantization — 3.0625 bits/weight: eight 8-bit codebook indices per 32-value sub-block plus a u32 holding four 7-bit sign codes and the scale.
//!
//! Layout and constants transcribed from llama.cpp's `ggml/src/ggml-quants.c`
//! (`dequantize_row_*`) and `ggml/src/ggml-common.h` (codebooks) — MIT, the ggml
//! authors; reference tree on this box `/mnt/nvme-raid0/llama.cpp-master` @ df03399.
//! The expected values below were produced from that reference and independently
//! confirmed against llama.cpp's `gguf-py` numpy dequantizers (50 random blocks
//! per type, PMAT-3477).

use crate::error::{RealizarError, Result};
use crate::quantize::f16_to_f32_lut;
use crate::quantize::iq_grids::{IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};

/// ggml type id of `IQ3_XXS`.
pub const GGML_TYPE_IQ3_XXS: u32 = 18;

/// Elements in one `IQ3_XXS` super-block.
pub const IQ3_XXS_BLOCK_ELEMS: usize = 256;

/// Bytes in one `IQ3_XXS` super-block (`sizeof(block_iq3_xxs)`).
pub const IQ3_XXS_BLOCK_BYTES: usize = 98;

/// Dequantize one `IQ3_XXS` super-block into `out` (IQ3_XXS_BLOCK_ELEMS values).
///
/// # Panics
/// Panics if `block` is shorter than [`IQ3_XXS_BLOCK_BYTES`] or `out` shorter than
/// [`IQ3_XXS_BLOCK_ELEMS`] — callers slice exact blocks.
pub fn dequantize_iq3_xxs_block(block: &[u8], out: &mut [f32]) {
    assert!(
        block.len() >= IQ3_XXS_BLOCK_BYTES && out.len() >= IQ3_XXS_BLOCK_ELEMS,
        "IQ3_XXS block needs {} bytes and {} outputs, got {} and {}",
        IQ3_XXS_BLOCK_BYTES,
        IQ3_XXS_BLOCK_ELEMS,
        block.len(),
        out.len()
    );

    let d = f16_to_f32_lut(u16::from_le_bytes([block[0], block[1]]));
    // 64 grid indices, then 32 bytes of packed per-sub-block scale + sign codes.
    let qs = &block[2..66];
    let scales_and_signs = &block[66..98];
    for ib32 in 0..8 {
        let s = 4 * ib32;
        let aux = u32::from_le_bytes([
            scales_and_signs[s],
            scales_and_signs[s + 1],
            scales_and_signs[s + 2],
            scales_and_signs[s + 3],
        ]);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            let g1 = IQ3XXS_GRID[usize::from(qs[8 * ib32 + 2 * l])];
            let g2 = IQ3XXS_GRID[usize::from(qs[8 * ib32 + 2 * l + 1])];
            let base = 32 * ib32 + 8 * l;
            for j in 0..4 {
                let m1 = ((g1 >> (8 * j)) & 0xff) as f32;
                let m2 = ((g2 >> (8 * j)) & 0xff) as f32;
                let s1 = if signs & KMASK_IQ2XS[j] != 0 {
                    -1.0
                } else {
                    1.0
                };
                let s2 = if signs & KMASK_IQ2XS[j + 4] != 0 {
                    -1.0
                } else {
                    1.0
                };
                out[base + j] = db * m1 * s1;
                out[base + j + 4] = db * m2 * s2;
            }
        }
    }
}

/// Dequantize a whole `IQ3_XXS` byte run (a multiple of [`IQ3_XXS_BLOCK_BYTES`]).
///
/// # Errors
/// Returns [`RealizarError::InvalidShape`] when `data` is not a whole number of
/// super-blocks.
pub fn dequantize_iq3_xxs(data: &[u8]) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(IQ3_XXS_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "IQ3_XXS data length {} is not a multiple of block size {}",
                data.len(),
                IQ3_XXS_BLOCK_BYTES
            ),
        });
    }
    let nb = data.len() / IQ3_XXS_BLOCK_BYTES;
    let mut out = vec![0.0f32; nb * IQ3_XXS_BLOCK_ELEMS];
    for (i, block) in data.chunks_exact(IQ3_XXS_BLOCK_BYTES).enumerate() {
        dequantize_iq3_xxs_block(
            block,
            &mut out[i * IQ3_XXS_BLOCK_ELEMS..(i + 1) * IQ3_XXS_BLOCK_ELEMS],
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One super-block whose bytes are a fixed pseudo-random draw (scale pinned
    /// to a representable f16). Expected values come from the ggml reference.
    const IQ3_XXS_BLOCK: [u8; IQ3_XXS_BLOCK_BYTES] = [
        0x00, 0x26, 0x79, 0x42, 0xbd, 0xf2, 0x21, 0x06, 0xf0, 0x84, 0x77, 0x62, 0xf0, 0xf3, 0xcb,
        0x4d, 0x76, 0x4d, 0xc7, 0x07, 0x20, 0x51, 0x15, 0x9a, 0x0f, 0x89, 0xf2, 0xc6, 0xda, 0xca,
        0xe3, 0x44, 0xbb, 0x31, 0x12, 0x45, 0xfd, 0x6f, 0x84, 0xdf, 0x9a, 0xd7, 0xc5, 0xb3, 0xd0,
        0x76, 0xac, 0x0e, 0x8f, 0x53, 0xa7, 0x35, 0x6c, 0x88, 0x91, 0x3f, 0x20, 0xf6, 0xf7, 0x2d,
        0xb0, 0x22, 0xd2, 0x4d, 0x0a, 0x96, 0xda, 0xd4, 0x3c, 0x16, 0x17, 0xc1, 0xa9, 0x8e, 0x78,
        0x12, 0x9e, 0x03, 0x27, 0x37, 0x10, 0x65, 0xd0, 0x95, 0x86, 0x4f, 0x15, 0xad, 0xa0, 0xb8,
        0x46, 0xc1, 0xc0, 0xeb, 0xc5, 0x34, 0x8a, 0xdc,
    ];

    /// `dequantize_row_iq3_xxs` of IQ3_XXS_BLOCK, from ggml-quants.c.
    #[rustfmt::skip]
    const IQ3_XXS_EXPECTED: [f32; IQ3_XXS_BLOCK_ELEMS] = [
        3.515625000e-01, -3.515625000e-01, 3.515625000e-01, -3.515625000e-01, -6.328125000e-01, 4.921875000e-01,
        -7.031250000e-02, 2.109375000e-01, -2.109375000e-01, 7.734375000e-01, 6.328125000e-01, -6.328125000e-01,
        7.031250000e-02, -6.328125000e-01, 2.109375000e-01, -1.089843750e+00, -1.089843750e+00, -7.734375000e-01,
        3.515625000e-01, 7.031250000e-02, -7.031250000e-02, -3.515625000e-01, -7.031250000e-02, -7.031250000e-02,
        -3.515625000e-01, 7.734375000e-01, 7.031250000e-02, 1.089843750e+00, -4.921875000e-01, -7.031250000e-02,
        7.734375000e-01, -3.515625000e-01, -6.175781250e+00, -1.195312500e+00, -1.992187500e+00, 1.992187500e+00,
        -1.992187500e+00, 3.984375000e-01, 3.984375000e-01, 1.992187500e+00, 1.992187500e+00, -4.382812500e+00,
        3.984375000e-01, 6.175781250e+00, 1.992187500e+00, 1.195312500e+00, 1.992187500e+00, -6.175781250e+00,
        -5.179687500e+00, -1.992187500e+00, -1.195312500e+00, 4.382812500e+00, 1.992187500e+00, -1.195312500e+00,
        1.992187500e+00, 1.195312500e+00, -1.195312500e+00, 1.195312500e+00, -1.992187500e+00, 1.992187500e+00,
        -1.992187500e+00, -1.195312500e+00, -1.992187500e+00, -1.195312500e+00, 7.031250000e-02, 2.109375000e-01,
        2.343750000e-02, -2.578125000e-01, -1.171875000e-01, -1.171875000e-01, -2.343750000e-02, 2.343750000e-02,
        7.031250000e-02, 2.578125000e-01, -1.171875000e-01, 2.343750000e-02, 2.343750000e-02, -2.343750000e-02,
        1.640625000e-01, 7.031250000e-02, 2.109375000e-01, 2.578125000e-01, 7.031250000e-02, -2.343750000e-02,
        -7.031250000e-02, -2.343750000e-02, -1.171875000e-01, 1.640625000e-01, 1.171875000e-01, 7.031250000e-02,
        -7.031250000e-02, -2.343750000e-02, -1.640625000e-01, 2.343750000e-02, 3.632812500e-01, -1.171875000e-01,
        -3.046875000e-01, -2.742187500e+00, -9.140625000e-01, 4.722656250e+00, 1.523437500e+00, -9.140625000e-01,
        3.046875000e-01, 3.351562500e+00, 1.523437500e+00, -2.742187500e+00, -4.722656250e+00, -3.351562500e+00,
        3.960937500e+00, -3.046875000e-01, -9.140625000e-01, -3.351562500e+00, 3.046875000e-01, 2.132812500e+00,
        2.132812500e+00, 3.960937500e+00, 3.046875000e-01, 3.351562500e+00, -3.046875000e-01, -9.140625000e-01,
        3.351562500e+00, 2.742187500e+00, 2.132812500e+00, -2.742187500e+00, 1.523437500e+00, -2.132812500e+00,
        3.351562500e+00, 3.046875000e-01, 2.109375000e-01, 1.476562500e+00, 6.328125000e-01, 2.109375000e-01,
        -2.109375000e-01, 2.109375000e-01, -6.328125000e-01, 6.328125000e-01, -1.054687500e+00, -2.109375000e-01,
        2.320312500e+00, -3.269531250e+00, 6.328125000e-01, -1.054687500e+00, 6.328125000e-01, 1.054687500e+00,
        1.476562500e+00, -2.109375000e-01, 2.320312500e+00, -1.054687500e+00, -6.328125000e-01, 1.054687500e+00,
        6.328125000e-01, -2.742187500e+00, 6.328125000e-01, 2.109375000e-01, -1.054687500e+00, -1.476562500e+00,
        -6.328125000e-01, -1.476562500e+00, -2.320312500e+00, -2.320312500e+00, -5.390625000e-01, 3.773437500e+00,
        -8.355468750e+00, 4.851562500e+00, -8.355468750e+00, 5.929687500e+00, 5.390625000e-01, -4.851562500e+00,
        2.695312500e+00, -5.390625000e-01, 3.773437500e+00, -5.929687500e+00, -1.617187500e+00, 1.617187500e+00,
        -2.695312500e+00, 2.695312500e+00, 3.773437500e+00, -3.773437500e+00, 8.355468750e+00, 3.773437500e+00,
        5.390625000e-01, 1.617187500e+00, 1.617187500e+00, -5.390625000e-01, -1.617187500e+00, 2.695312500e+00,
        -5.390625000e-01, 3.773437500e+00, 5.390625000e-01, 2.695312500e+00, -3.773437500e+00, -1.617187500e+00,
        3.398437500e+00, -3.398437500e+00, -7.476562500e+00, 4.757812500e+00, 6.796875000e-01, 2.039062500e+00,
        -1.053515625e+01, -6.796875000e-01, 7.476562500e+00, -6.796875000e-01, 2.039062500e+00, 3.398437500e+00,
        2.039062500e+00, 6.796875000e-01, 1.053515625e+01, -3.398437500e+00, -6.796875000e-01, -7.476562500e+00,
        6.796875000e-01, 4.757812500e+00, 4.757812500e+00, 3.398437500e+00, 6.796875000e-01, 2.039062500e+00,
        2.039062500e+00, -7.476562500e+00, -3.398437500e+00, -6.796875000e-01, -6.796875000e-01, 6.796875000e-01,
        -4.757812500e+00, -1.053515625e+01, -6.960937500e+00, 1.898437500e+00, -4.429687500e+00, 9.808593750e+00,
        4.429687500e+00, 9.808593750e+00, -5.695312500e+00, -6.328125000e-01, -6.960937500e+00, 4.429687500e+00,
        6.328125000e-01, -5.695312500e+00, 6.960937500e+00, -9.808593750e+00, -3.164062500e+00, 6.328125000e-01,
        6.328125000e-01, 1.898437500e+00, 5.695312500e+00, -6.960937500e+00, 3.164062500e+00, -1.898437500e+00,
        3.164062500e+00, 1.898437500e+00, 4.429687500e+00, 9.808593750e+00, -6.328125000e-01, 6.328125000e-01,
        6.328125000e-01, -3.164062500e+00, -1.898437500e+00, -4.429687500e+00,
    ];

    #[test]
    fn iq3_xxs_block_matches_the_ggml_reference() {
        let mut got = [0.0f32; IQ3_XXS_BLOCK_ELEMS];
        dequantize_iq3_xxs_block(&IQ3_XXS_BLOCK, &mut got);
        for (i, (&g, &e)) in got.iter().zip(IQ3_XXS_EXPECTED.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-6 * e.abs().max(1.0),
                "iq3_xxs[{i}]: got {g}, ggml reference {e}"
            );
        }
    }

    #[test]
    fn iq3_xxs_whole_run_dequantizes_every_block() {
        let mut data = Vec::new();
        data.extend_from_slice(&IQ3_XXS_BLOCK);
        data.extend_from_slice(&IQ3_XXS_BLOCK);
        let out = dequantize_iq3_xxs(&data).expect("two whole blocks");
        assert_eq!(out.len(), 2 * IQ3_XXS_BLOCK_ELEMS);
        assert_eq!(out[..IQ3_XXS_BLOCK_ELEMS], out[IQ3_XXS_BLOCK_ELEMS..]);
        assert!((out[0] - IQ3_XXS_EXPECTED[0]).abs() <= 1e-6 * IQ3_XXS_EXPECTED[0].abs().max(1.0));
    }

    #[test]
    fn iq3_xxs_rejects_a_partial_block() {
        let err = dequantize_iq3_xxs(&[0u8; IQ3_XXS_BLOCK_BYTES - 1]).expect_err("partial block");
        assert!(format!("{err}").contains("not a multiple of block size"));
    }

    /// A zero super-block dequantizes to all zeros (`d == 0`), which pins that
    /// the scale is read from the first two bytes rather than assumed.
    #[test]
    fn iq3_xxs_zero_block_is_all_zeros() {
        let mut got = [1.0f32; IQ3_XXS_BLOCK_ELEMS];
        dequantize_iq3_xxs_block(&[0u8; IQ3_XXS_BLOCK_BYTES], &mut got);
        assert!(got.iter().all(|v| *v == 0.0), "zero scale must give zeros");
    }
}
