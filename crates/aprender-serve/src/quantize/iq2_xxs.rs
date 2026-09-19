//! `IQ2_XXS` (ggml type 16) dequantization — 2.0625 bits/weight: one f16 super-block scale and, per 32-value sub-block, four 8-bit codebook indices plus four 7-bit sign codes packed into two u32.
//!
//! Layout and constants transcribed from llama.cpp's `ggml/src/ggml-quants.c`
//! (`dequantize_row_*`) and `ggml/src/ggml-common.h` (codebooks) — MIT, the ggml
//! authors; reference tree on this box `/mnt/nvme-raid0/llama.cpp-master` @ df03399.
//! The expected values below were produced from that reference and independently
//! confirmed against llama.cpp's `gguf-py` numpy dequantizers (50 random blocks
//! per type, PMAT-3477).

use crate::error::{RealizarError, Result};
use crate::quantize::f16_to_f32_lut;
use crate::quantize::iq_grids::{IQ2XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};

/// ggml type id of `IQ2_XXS`.
pub const GGML_TYPE_IQ2_XXS: u32 = 16;

/// Elements in one `IQ2_XXS` super-block.
pub const IQ2_XXS_BLOCK_ELEMS: usize = 256;

/// Bytes in one `IQ2_XXS` super-block (`sizeof(block_iq2_xxs)`).
pub const IQ2_XXS_BLOCK_BYTES: usize = 66;

/// Dequantize one `IQ2_XXS` super-block into `out` (IQ2_XXS_BLOCK_ELEMS values).
///
/// # Panics
/// Panics if `block` is shorter than [`IQ2_XXS_BLOCK_BYTES`] or `out` shorter than
/// [`IQ2_XXS_BLOCK_ELEMS`] — callers slice exact blocks.
pub fn dequantize_iq2_xxs_block(block: &[u8], out: &mut [f32]) {
    assert!(
        block.len() >= IQ2_XXS_BLOCK_BYTES && out.len() >= IQ2_XXS_BLOCK_ELEMS,
        "IQ2_XXS block needs {} bytes and {} outputs, got {} and {}",
        IQ2_XXS_BLOCK_BYTES,
        IQ2_XXS_BLOCK_ELEMS,
        block.len(),
        out.len()
    );

    let d = f16_to_f32_lut(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..IQ2_XXS_BLOCK_BYTES];
    for ib32 in 0..8 {
        let base = 8 * ib32;
        let aux0 = u32::from_le_bytes([qs[base], qs[base + 1], qs[base + 2], qs[base + 3]]);
        let aux1 = u32::from_le_bytes([qs[base + 4], qs[base + 5], qs[base + 6], qs[base + 7]]);
        let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
        for l in 0..4 {
            let grid = IQ2XXS_GRID[((aux0 >> (8 * l)) & 0xff) as usize];
            let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                let mag = ((grid >> (8 * j)) & 0xff) as f32;
                let sign = if signs & KMASK_IQ2XS[j] != 0 {
                    -1.0
                } else {
                    1.0
                };
                out[32 * ib32 + 8 * l + j] = db * mag * sign;
            }
        }
    }
}

/// Dequantize a whole `IQ2_XXS` byte run (a multiple of [`IQ2_XXS_BLOCK_BYTES`]).
///
/// # Errors
/// Returns [`RealizarError::InvalidShape`] when `data` is not a whole number of
/// super-blocks.
pub fn dequantize_iq2_xxs(data: &[u8]) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(IQ2_XXS_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "IQ2_XXS data length {} is not a multiple of block size {}",
                data.len(),
                IQ2_XXS_BLOCK_BYTES
            ),
        });
    }
    let nb = data.len() / IQ2_XXS_BLOCK_BYTES;
    let mut out = vec![0.0f32; nb * IQ2_XXS_BLOCK_ELEMS];
    for (i, block) in data.chunks_exact(IQ2_XXS_BLOCK_BYTES).enumerate() {
        dequantize_iq2_xxs_block(
            block,
            &mut out[i * IQ2_XXS_BLOCK_ELEMS..(i + 1) * IQ2_XXS_BLOCK_ELEMS],
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One super-block whose bytes are a fixed pseudo-random draw (scale pinned
    /// to a representable f16). Expected values come from the ggml reference.
    const IQ2_XXS_BLOCK: [u8; IQ2_XXS_BLOCK_BYTES] = [
        0x00, 0x26, 0x44, 0x20, 0x82, 0x3c, 0xfd, 0xe6, 0xf1, 0xc2, 0x6b, 0x30, 0xf9, 0x0e, 0xc7,
        0xdd, 0x01, 0xe4, 0x88, 0x75, 0x34, 0xa2, 0x0f, 0x0b, 0x0d, 0x04, 0xc3, 0x6e, 0xd8, 0x0e,
        0x71, 0xe0, 0xfd, 0x77, 0xb0, 0x76, 0x70, 0xeb, 0x94, 0x0b, 0xd5, 0x33, 0x5f, 0x97, 0x3d,
        0xaa, 0xd8, 0x61, 0x9b, 0x91, 0xff, 0xc9, 0x11, 0xf5, 0x7c, 0xce, 0xd4, 0x58, 0xbb, 0xbf,
        0x2c, 0xe0, 0x37, 0x53, 0xc9, 0xbd,
    ];

    /// `dequantize_row_iq2_xxs` of IQ2_XXS_BLOCK, from ggml-quants.c.
    #[rustfmt::skip]
    const IQ2_XXS_EXPECTED: [f32; IQ2_XXS_BLOCK_ELEMS] = [
        -5.859375000e-01, 1.831054688e+00, -5.859375000e-01, -5.859375000e-01, -1.831054688e+00, -3.149414062e+00,
        -5.859375000e-01, 5.859375000e-01, -1.831054688e+00, 5.859375000e-01, -5.859375000e-01, -5.859375000e-01,
        5.859375000e-01, 1.831054688e+00, -5.859375000e-01, 5.859375000e-01, -5.859375000e-01, -1.831054688e+00,
        -1.831054688e+00, 3.149414062e+00, 5.859375000e-01, 3.149414062e+00, -3.149414062e+00, 5.859375000e-01,
        -5.859375000e-01, -3.149414062e+00, -5.859375000e-01, 5.859375000e-01, -5.859375000e-01, 3.149414062e+00,
        5.859375000e-01, 5.859375000e-01, -2.124023438e+00, -6.796875000e-01, -2.124023438e+00, 2.124023438e+00,
        3.653320312e+00, 3.653320312e+00, -2.124023438e+00, 6.796875000e-01, -6.796875000e-01, -6.796875000e-01,
        3.653320312e+00, -6.796875000e-01, -2.124023438e+00, -2.124023438e+00, 6.796875000e-01, -6.796875000e-01,
        -6.796875000e-01, -6.796875000e-01, -2.124023438e+00, 2.124023438e+00, 6.796875000e-01, 6.796875000e-01,
        3.653320312e+00, -3.653320312e+00, 6.796875000e-01, 3.653320312e+00, 2.124023438e+00, 2.124023438e+00,
        6.796875000e-01, -6.796875000e-01, 6.796875000e-01, -6.796875000e-01, -7.324218750e-02, -2.343750000e-02,
        -1.259765625e-01, -2.343750000e-02, 2.343750000e-02, 2.343750000e-02, 2.343750000e-02, 7.324218750e-02,
        2.343750000e-02, -2.343750000e-02, -2.343750000e-02, 2.343750000e-02, -1.259765625e-01, 2.343750000e-02,
        1.259765625e-01, -2.343750000e-02, 7.324218750e-02, 2.343750000e-02, -7.324218750e-02, 1.259765625e-01,
        -7.324218750e-02, -7.324218750e-02, 2.343750000e-02, -2.343750000e-02, 2.343750000e-02, 2.343750000e-02,
        2.343750000e-02, 2.343750000e-02, 1.259765625e-01, -7.324218750e-02, 2.343750000e-02, -7.324218750e-02,
        -3.515625000e-01, 1.889648438e+00, 3.515625000e-01, 3.515625000e-01, -1.889648438e+00, -1.889648438e+00,
        -1.098632812e+00, 1.098632812e+00, 1.889648438e+00, 1.889648438e+00, 3.515625000e-01, 3.515625000e-01,
        3.515625000e-01, 3.515625000e-01, -1.889648438e+00, -3.515625000e-01, -3.515625000e-01, -1.889648438e+00,
        -1.098632812e+00, 3.515625000e-01, -1.098632812e+00, -3.515625000e-01, -3.515625000e-01, 1.889648438e+00,
        -3.515625000e-01, -1.889648438e+00, -1.098632812e+00, -1.098632812e+00, -3.515625000e-01, -3.515625000e-01,
        3.515625000e-01, 3.515625000e-01, 5.126953125e-01, 1.640625000e-01, -5.126953125e-01, 1.640625000e-01,
        -1.640625000e-01, 1.640625000e-01, 5.126953125e-01, 5.126953125e-01, -8.818359375e-01, -1.640625000e-01,
        -1.640625000e-01, 1.640625000e-01, -8.818359375e-01, 1.640625000e-01, 8.818359375e-01, 1.640625000e-01,
        5.126953125e-01, 1.640625000e-01, -8.818359375e-01, 5.126953125e-01, -1.640625000e-01, 1.640625000e-01,
        -8.818359375e-01, -1.640625000e-01, 1.640625000e-01, -1.640625000e-01, -5.126953125e-01, -1.640625000e-01,
        -1.640625000e-01, 1.640625000e-01, 5.126953125e-01, 8.818359375e-01, 4.453125000e-01, 2.393554688e+00,
        1.391601562e+00, -4.453125000e-01, -1.391601562e+00, 1.391601562e+00, -1.391601562e+00, -4.453125000e-01,
        -4.453125000e-01, -4.453125000e-01, 1.391601562e+00, 4.453125000e-01, 2.393554688e+00, 4.453125000e-01,
        -4.453125000e-01, -1.391601562e+00, -4.453125000e-01, 1.391601562e+00, -1.391601562e+00, -4.453125000e-01,
        4.453125000e-01, -2.393554688e+00, -4.453125000e-01, -4.453125000e-01, 1.391601562e+00, 4.453125000e-01,
        -2.393554688e+00, -1.391601562e+00, 1.391601562e+00, 2.393554688e+00, 4.453125000e-01, 1.391601562e+00,
        2.578125000e-01, 8.056640625e-01, -2.578125000e-01, -2.578125000e-01, -8.056640625e-01, -1.385742188e+00,
        -1.385742188e+00, -1.385742188e+00, 8.056640625e-01, 8.056640625e-01, -8.056640625e-01, -8.056640625e-01,
        -8.056640625e-01, 2.578125000e-01, 1.385742188e+00, -8.056640625e-01, -2.578125000e-01, -2.578125000e-01,
        2.578125000e-01, 1.385742188e+00, -2.578125000e-01, 2.578125000e-01, -2.578125000e-01, 2.578125000e-01,
        8.056640625e-01, -2.578125000e-01, -8.056640625e-01, 8.056640625e-01, 8.056640625e-01, 8.056640625e-01,
        -8.056640625e-01, -1.385742188e+00, -2.897460938e+00, -2.897460938e+00, -5.390625000e-01, 5.390625000e-01,
        -1.684570312e+00, -1.684570312e+00, 1.684570312e+00, -1.684570312e+00, 1.684570312e+00, -5.390625000e-01,
        -1.684570312e+00, 5.390625000e-01, 5.390625000e-01, -2.897460938e+00, 1.684570312e+00, -1.684570312e+00,
        -5.390625000e-01, 1.684570312e+00, -2.897460938e+00, 2.897460938e+00, 5.390625000e-01, -1.684570312e+00,
        5.390625000e-01, -5.390625000e-01, 5.390625000e-01, -5.390625000e-01, -2.897460938e+00, -1.684570312e+00,
        5.390625000e-01, -1.684570312e+00, -5.390625000e-01, -2.897460938e+00,
    ];

    #[test]
    fn iq2_xxs_block_matches_the_ggml_reference() {
        let mut got = [0.0f32; IQ2_XXS_BLOCK_ELEMS];
        dequantize_iq2_xxs_block(&IQ2_XXS_BLOCK, &mut got);
        for (i, (&g, &e)) in got.iter().zip(IQ2_XXS_EXPECTED.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-6 * e.abs().max(1.0),
                "iq2_xxs[{i}]: got {g}, ggml reference {e}"
            );
        }
    }

    #[test]
    fn iq2_xxs_whole_run_dequantizes_every_block() {
        let mut data = Vec::new();
        data.extend_from_slice(&IQ2_XXS_BLOCK);
        data.extend_from_slice(&IQ2_XXS_BLOCK);
        let out = dequantize_iq2_xxs(&data).expect("two whole blocks");
        assert_eq!(out.len(), 2 * IQ2_XXS_BLOCK_ELEMS);
        assert_eq!(out[..IQ2_XXS_BLOCK_ELEMS], out[IQ2_XXS_BLOCK_ELEMS..]);
        assert!((out[0] - IQ2_XXS_EXPECTED[0]).abs() <= 1e-6 * IQ2_XXS_EXPECTED[0].abs().max(1.0));
    }

    #[test]
    fn iq2_xxs_rejects_a_partial_block() {
        let err = dequantize_iq2_xxs(&[0u8; IQ2_XXS_BLOCK_BYTES - 1]).expect_err("partial block");
        assert!(format!("{err}").contains("not a multiple of block size"));
    }

    /// A zero super-block dequantizes to all zeros (`d == 0`), which pins that
    /// the scale is read from the first two bytes rather than assumed.
    #[test]
    fn iq2_xxs_zero_block_is_all_zeros() {
        let mut got = [1.0f32; IQ2_XXS_BLOCK_ELEMS];
        dequantize_iq2_xxs_block(&[0u8; IQ2_XXS_BLOCK_BYTES], &mut got);
        assert!(got.iter().all(|v| *v == 0.0), "zero scale must give zeros");
    }
}
