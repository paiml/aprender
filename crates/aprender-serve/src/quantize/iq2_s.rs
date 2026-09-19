//! `IQ2_S` (ggml type 22) dequantization — 2.5 bits/weight: 10-bit codebook indices (8 bits in `qs`, 2 in `qh`), explicit sign bytes and 4-bit per-sub-block scales.
//!
//! Layout and constants transcribed from llama.cpp's `ggml/src/ggml-quants.c`
//! (`dequantize_row_*`) and `ggml/src/ggml-common.h` (codebooks) — MIT, the ggml
//! authors; reference tree on this box `/mnt/nvme-raid0/llama.cpp-master` @ df03399.
//! The expected values below were produced from that reference and independently
//! confirmed against llama.cpp's `gguf-py` numpy dequantizers (50 random blocks
//! per type, PMAT-3477).

use crate::error::{RealizarError, Result};
use crate::quantize::f16_to_f32_lut;
use crate::quantize::iq_grids::{IQ2S_GRID, KMASK_IQ2XS};

/// ggml type id of `IQ2_S`.
pub const GGML_TYPE_IQ2_S: u32 = 22;

/// Elements in one `IQ2_S` super-block.
pub const IQ2_S_BLOCK_ELEMS: usize = 256;

/// Bytes in one `IQ2_S` super-block (`sizeof(block_iq2_s)`).
pub const IQ2_S_BLOCK_BYTES: usize = 82;

/// Dequantize one `IQ2_S` super-block into `out` (IQ2_S_BLOCK_ELEMS values).
///
/// # Panics
/// Panics if `block` is shorter than [`IQ2_S_BLOCK_BYTES`] or `out` shorter than
/// [`IQ2_S_BLOCK_ELEMS`] — callers slice exact blocks.
pub fn dequantize_iq2_s_block(block: &[u8], out: &mut [f32]) {
    assert!(
        block.len() >= IQ2_S_BLOCK_BYTES && out.len() >= IQ2_S_BLOCK_ELEMS,
        "IQ2_S block needs {} bytes and {} outputs, got {} and {}",
        IQ2_S_BLOCK_BYTES,
        IQ2_S_BLOCK_ELEMS,
        block.len(),
        out.len()
    );

    let d = f16_to_f32_lut(u16::from_le_bytes([block[0], block[1]]));
    // `qs` holds 32 grid indices followed by 32 sign bytes (ggml reads the sign
    // run as `qs + QK_K/8`), then 8 `qh` high-bit bytes and 8 packed scales.
    let qs = &block[2..34];
    let signs = &block[34..66];
    let qh = &block[66..74];
    let scales = &block[74..82];
    for ib32 in 0..8 {
        let db = [
            d * (0.5 + f32::from(scales[ib32] & 0xf)) * 0.25,
            d * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25,
        ];
        for l in 0..4 {
            let dl = db[l / 2];
            let idx =
                usize::from(qs[4 * ib32 + l]) | ((usize::from(qh[ib32]) << (8 - 2 * l)) & 0x300);
            let grid = IQ2S_GRID[idx];
            let sign_byte = signs[4 * ib32 + l];
            for j in 0..8 {
                let mag = ((grid >> (8 * j)) & 0xff) as f32;
                let sign = if sign_byte & KMASK_IQ2XS[j] != 0 {
                    -1.0
                } else {
                    1.0
                };
                out[32 * ib32 + 8 * l + j] = dl * mag * sign;
            }
        }
    }
}

/// Dequantize a whole `IQ2_S` byte run (a multiple of [`IQ2_S_BLOCK_BYTES`]).
///
/// # Errors
/// Returns [`RealizarError::InvalidShape`] when `data` is not a whole number of
/// super-blocks.
pub fn dequantize_iq2_s(data: &[u8]) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(IQ2_S_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "IQ2_S data length {} is not a multiple of block size {}",
                data.len(),
                IQ2_S_BLOCK_BYTES
            ),
        });
    }
    let nb = data.len() / IQ2_S_BLOCK_BYTES;
    let mut out = vec![0.0f32; nb * IQ2_S_BLOCK_ELEMS];
    for (i, block) in data.chunks_exact(IQ2_S_BLOCK_BYTES).enumerate() {
        dequantize_iq2_s_block(
            block,
            &mut out[i * IQ2_S_BLOCK_ELEMS..(i + 1) * IQ2_S_BLOCK_ELEMS],
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One super-block whose bytes are a fixed pseudo-random draw (scale pinned
    /// to a representable f16). Expected values come from the ggml reference.
    const IQ2_S_BLOCK: [u8; IQ2_S_BLOCK_BYTES] = [
        0x00, 0x26, 0x1c, 0x2e, 0x2b, 0xb8, 0x56, 0x9d, 0x80, 0x6c, 0x12, 0x51, 0xdc, 0xc9, 0xbe,
        0xe3, 0x89, 0x12, 0x0e, 0xba, 0xee, 0xa3, 0xc2, 0xd8, 0x54, 0x5a, 0x78, 0x76, 0x0c, 0x5a,
        0xa6, 0x58, 0x45, 0xb8, 0x5d, 0xe4, 0xd4, 0xba, 0xb5, 0xb9, 0xe4, 0x52, 0xcc, 0xec, 0x7f,
        0xfa, 0x8e, 0xff, 0xb5, 0xe8, 0xec, 0xb3, 0xe9, 0xf9, 0x71, 0xa6, 0x55, 0x89, 0xf5, 0x9e,
        0x9b, 0xd0, 0x9f, 0x6a, 0xfa, 0xbb, 0x26, 0xae, 0x04, 0x61, 0x36, 0x1e, 0x19, 0x8b, 0x74,
        0x36, 0x45, 0x88, 0x7d, 0x6b, 0x1e, 0xd8,
    ];

    /// `dequantize_row_iq2_s` of IQ2_S_BLOCK, from ggml-quants.c.
    #[rustfmt::skip]
    const IQ2_S_EXPECTED: [f32; IQ2_S_BLOCK_ELEMS] = [
        -6.591796875e-01, 2.109375000e-01, -6.591796875e-01, -6.591796875e-01, -1.133789062e+00, 2.109375000e-01,
        -2.109375000e-01, 6.591796875e-01, 1.133789062e+00, 2.109375000e-01, -2.109375000e-01, 1.133789062e+00,
        2.109375000e-01, -6.591796875e-01, -6.591796875e-01, -2.109375000e-01, 1.889648438e+00, 3.515625000e-01,
        -1.889648438e+00, 3.515625000e-01, -3.515625000e-01, 1.098632812e+00, -3.515625000e-01, -1.098632812e+00,
        3.515625000e-01, -1.098632812e+00, 3.515625000e-01, -3.515625000e-01, -1.098632812e+00, -1.889648438e+00,
        3.515625000e-01, -3.515625000e-01, -9.521484375e-01, 9.521484375e-01, -3.046875000e-01, 3.046875000e-01,
        -1.637695312e+00, -9.521484375e-01, 3.046875000e-01, -9.521484375e-01, -3.046875000e-01, 1.637695312e+00,
        9.521484375e-01, -3.046875000e-01, -3.046875000e-01, -3.046875000e-01, 9.521484375e-01, -1.637695312e+00,
        1.640625000e-01, 1.640625000e-01, -5.126953125e-01, 1.640625000e-01, 8.818359375e-01, -8.818359375e-01,
        -1.640625000e-01, -5.126953125e-01, 1.640625000e-01, -8.818359375e-01, 1.640625000e-01, 5.126953125e-01,
        -1.640625000e-01, 8.818359375e-01, -1.640625000e-01, 5.126953125e-01, 1.385742188e+00, 2.578125000e-01,
        -8.056640625e-01, -8.056640625e-01, 2.578125000e-01, 2.578125000e-01, -2.578125000e-01, -2.578125000e-01,
        2.578125000e-01, 2.578125000e-01, -2.578125000e-01, -1.385742188e+00, 1.385742188e+00, -8.056640625e-01,
        -8.056640625e-01, -2.578125000e-01, -2.109375000e-01, -1.133789062e+00, -2.109375000e-01, -6.591796875e-01,
        -2.109375000e-01, -2.109375000e-01, -6.591796875e-01, 2.109375000e-01, 1.133789062e+00, -2.109375000e-01,
        1.133789062e+00, -2.109375000e-01, -1.133789062e+00, -1.133789062e+00, -2.109375000e-01, -2.109375000e-01,
        2.141601562e+00, -1.245117188e+00, -2.141601562e+00, -1.245117188e+00, 1.245117188e+00, 1.245117188e+00,
        2.141601562e+00, -3.984375000e-01, -1.245117188e+00, -1.245117188e+00, -2.141601562e+00, -1.245117188e+00,
        -3.984375000e-01, -3.984375000e-01, -1.245117188e+00, -3.984375000e-01, -1.245117188e+00, 2.141601562e+00,
        -1.245117188e+00, 3.984375000e-01, -3.984375000e-01, -3.984375000e-01, 1.245117188e+00, -1.245117188e+00,
        1.245117188e+00, 1.245117188e+00, 3.984375000e-01, -1.245117188e+00, 2.141601562e+00, -3.984375000e-01,
        -1.245117188e+00, -3.984375000e-01, 6.328125000e-01, 3.401367188e+00, -6.328125000e-01, -3.401367188e+00,
        1.977539062e+00, -6.328125000e-01, -6.328125000e-01, -1.977539062e+00, -6.328125000e-01, -6.328125000e-01,
        3.401367188e+00, 6.328125000e-01, -1.977539062e+00, -1.977539062e+00, 3.401367188e+00, -6.328125000e-01,
        -3.515625000e-01, 3.515625000e-01, 3.515625000e-01, -1.098632812e+00, 3.515625000e-01, -1.098632812e+00,
        -1.889648438e+00, -1.889648438e+00, -3.515625000e-01, 1.098632812e+00, 1.098632812e+00, -3.515625000e-01,
        -3.515625000e-01, -1.889648438e+00, -3.515625000e-01, -3.515625000e-01, -2.897460938e+00, 5.390625000e-01,
        1.684570312e+00, 5.390625000e-01, -5.390625000e-01, -1.684570312e+00, -1.684570312e+00, 1.684570312e+00,
        5.390625000e-01, -1.684570312e+00, -1.684570312e+00, 5.390625000e-01, 5.390625000e-01, -5.390625000e-01,
        2.897460938e+00, -2.897460938e+00, -3.046875000e-01, 9.521484375e-01, -3.046875000e-01, 3.046875000e-01,
        -3.046875000e-01, 1.637695312e+00, -9.521484375e-01, 3.046875000e-01, -3.046875000e-01, 9.521484375e-01,
        1.637695312e+00, -3.046875000e-01, 3.046875000e-01, 9.521484375e-01, 3.046875000e-01, -3.046875000e-01,
        -2.124023438e+00, 6.796875000e-01, -2.124023438e+00, 6.796875000e-01, -6.796875000e-01, -6.796875000e-01,
        -3.653320312e+00, -6.796875000e-01, 2.124023438e+00, -6.796875000e-01, -2.124023438e+00, -6.796875000e-01,
        -2.124023438e+00, 3.653320312e+00, 6.796875000e-01, -2.124023438e+00, -2.197265625e-01, -3.779296875e-01,
        7.031250000e-02, -7.031250000e-02, -3.779296875e-01, 7.031250000e-02, 2.197265625e-01, -7.031250000e-02,
        7.031250000e-02, 2.197265625e-01, 3.779296875e-01, 7.031250000e-02, -7.031250000e-02, 2.197265625e-01,
        -7.031250000e-02, -7.031250000e-02, -1.245117188e+00, -3.984375000e-01, -3.984375000e-01, -2.141601562e+00,
        -3.984375000e-01, 3.984375000e-01, 1.245117188e+00, -2.141601562e+00, 1.245117188e+00, -3.984375000e-01,
        1.245117188e+00, -3.984375000e-01, 2.141601562e+00, -1.245117188e+00, -3.984375000e-01, 1.245117188e+00,
        3.401367188e+00, -3.401367188e+00, 3.401367188e+00, -6.328125000e-01, -3.401367188e+00, -6.328125000e-01,
        -6.328125000e-01, -6.328125000e-01, -1.977539062e+00, -6.328125000e-01, 6.328125000e-01, -1.977539062e+00,
        -3.401367188e+00, -6.328125000e-01, 1.977539062e+00, -1.977539062e+00,
    ];

    #[test]
    fn iq2_s_block_matches_the_ggml_reference() {
        let mut got = [0.0f32; IQ2_S_BLOCK_ELEMS];
        dequantize_iq2_s_block(&IQ2_S_BLOCK, &mut got);
        for (i, (&g, &e)) in got.iter().zip(IQ2_S_EXPECTED.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-6 * e.abs().max(1.0),
                "iq2_s[{i}]: got {g}, ggml reference {e}"
            );
        }
    }

    #[test]
    fn iq2_s_whole_run_dequantizes_every_block() {
        let mut data = Vec::new();
        data.extend_from_slice(&IQ2_S_BLOCK);
        data.extend_from_slice(&IQ2_S_BLOCK);
        let out = dequantize_iq2_s(&data).expect("two whole blocks");
        assert_eq!(out.len(), 2 * IQ2_S_BLOCK_ELEMS);
        assert_eq!(out[..IQ2_S_BLOCK_ELEMS], out[IQ2_S_BLOCK_ELEMS..]);
        assert!((out[0] - IQ2_S_EXPECTED[0]).abs() <= 1e-6 * IQ2_S_EXPECTED[0].abs().max(1.0));
    }

    #[test]
    fn iq2_s_rejects_a_partial_block() {
        let err = dequantize_iq2_s(&[0u8; IQ2_S_BLOCK_BYTES - 1]).expect_err("partial block");
        assert!(format!("{err}").contains("not a multiple of block size"));
    }

    /// A zero super-block dequantizes to all zeros (`d == 0`), which pins that
    /// the scale is read from the first two bytes rather than assumed.
    #[test]
    fn iq2_s_zero_block_is_all_zeros() {
        let mut got = [1.0f32; IQ2_S_BLOCK_ELEMS];
        dequantize_iq2_s_block(&[0u8; IQ2_S_BLOCK_BYTES], &mut got);
        assert!(got.iter().all(|v| *v == 0.0), "zero scale must give zeros");
    }
}
