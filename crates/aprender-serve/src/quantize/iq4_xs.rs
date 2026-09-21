//! `IQ4_XS` (ggml type 23) dequantization — 4.25 bits/weight: 6-bit per-sub-block scales split across `scales_l` and `scales_h`, and 4-bit indices into the 16 non-linear IQ4_NL levels.
//!
//! Layout and constants transcribed from llama.cpp's `ggml/src/ggml-quants.c`
//! (`dequantize_row_*`) and `ggml/src/ggml-common.h` (codebooks) — MIT, the ggml
//! authors; reference tree on this box `/mnt/nvme-raid0/llama.cpp-master` @ df03399.
//! The expected values below were produced from that reference and independently
//! confirmed against llama.cpp's `gguf-py` numpy dequantizers (50 random blocks
//! per type, PMAT-3477).

use crate::error::{RealizarError, Result};
use crate::quantize::f16_to_f32_lut;
use crate::quantize::iq_grids::KVALUES_IQ4NL;

/// ggml type id of `IQ4_XS`.
pub const GGML_TYPE_IQ4_XS: u32 = 23;

/// Elements in one `IQ4_XS` super-block.
pub const IQ4_XS_BLOCK_ELEMS: usize = 256;

/// Bytes in one `IQ4_XS` super-block (`sizeof(block_iq4_xs)`).
pub const IQ4_XS_BLOCK_BYTES: usize = 136;

/// Dequantize one `IQ4_XS` super-block into `out` (IQ4_XS_BLOCK_ELEMS values).
///
/// # Panics
/// Panics if `block` is shorter than [`IQ4_XS_BLOCK_BYTES`] or `out` shorter than
/// [`IQ4_XS_BLOCK_ELEMS`] — callers slice exact blocks.
pub fn dequantize_iq4_xs_block(block: &[u8], out: &mut [f32]) {
    assert!(
        block.len() >= IQ4_XS_BLOCK_BYTES && out.len() >= IQ4_XS_BLOCK_ELEMS,
        "IQ4_XS block needs {} bytes and {} outputs, got {} and {}",
        IQ4_XS_BLOCK_BYTES,
        IQ4_XS_BLOCK_ELEMS,
        block.len(),
        out.len()
    );

    let d = f16_to_f32_lut(u16::from_le_bytes([block[0], block[1]]));
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[4..8];
    let qs = &block[8..IQ4_XS_BLOCK_BYTES];
    for ib in 0..8 {
        let ls =
            ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as u16 | ((scales_h >> (2 * ib)) & 3) << 4;
        let dl = d * (f32::from(ls) - 32.0);
        for j in 0..16 {
            let byte = qs[16 * ib + j];
            out[32 * ib + j] = dl * f32::from(KVALUES_IQ4NL[usize::from(byte & 0xf)]);
            out[32 * ib + j + 16] = dl * f32::from(KVALUES_IQ4NL[usize::from(byte >> 4)]);
        }
    }
}

/// Dequantize a whole `IQ4_XS` byte run (a multiple of [`IQ4_XS_BLOCK_BYTES`]).
///
/// # Errors
/// Returns [`RealizarError::InvalidShape`] when `data` is not a whole number of
/// super-blocks.
pub fn dequantize_iq4_xs(data: &[u8]) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(IQ4_XS_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "IQ4_XS data length {} is not a multiple of block size {}",
                data.len(),
                IQ4_XS_BLOCK_BYTES
            ),
        });
    }
    let nb = data.len() / IQ4_XS_BLOCK_BYTES;
    let mut out = vec![0.0f32; nb * IQ4_XS_BLOCK_ELEMS];
    for (i, block) in data.chunks_exact(IQ4_XS_BLOCK_BYTES).enumerate() {
        dequantize_iq4_xs_block(
            block,
            &mut out[i * IQ4_XS_BLOCK_ELEMS..(i + 1) * IQ4_XS_BLOCK_ELEMS],
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One super-block whose bytes are a fixed pseudo-random draw (scale pinned
    /// to a representable f16). Expected values come from the ggml reference.
    const IQ4_XS_BLOCK: [u8; IQ4_XS_BLOCK_BYTES] = [
        0x00, 0x26, 0x82, 0xb7, 0x0e, 0xee, 0x7f, 0x1a, 0x50, 0x39, 0xbe, 0xf0, 0x7e, 0xc2, 0x34,
        0x7f, 0x06, 0x6e, 0xd0, 0x8f, 0x5d, 0xc7, 0x51, 0x24, 0x47, 0xe3, 0x40, 0x43, 0x00, 0x02,
        0x6b, 0x6e, 0x54, 0x55, 0x94, 0xa0, 0x65, 0x68, 0x5d, 0x64, 0xc4, 0x98, 0x0b, 0xb8, 0xd4,
        0x54, 0x4a, 0x87, 0x21, 0xa9, 0x9a, 0x01, 0xad, 0x21, 0x9e, 0xb5, 0x9c, 0xf6, 0xa1, 0x5e,
        0xf6, 0xf1, 0x5a, 0x1d, 0x83, 0x0b, 0xb7, 0xce, 0x09, 0xd6, 0xbb, 0xc0, 0x04, 0xe7, 0x17,
        0x5c, 0x64, 0x3c, 0x7d, 0xec, 0xb0, 0xb5, 0x80, 0xec, 0x37, 0xbc, 0x97, 0x12, 0xdd, 0x2e,
        0x6a, 0xae, 0xb9, 0x4b, 0xae, 0x8d, 0x2f, 0x9f, 0xa2, 0x9c, 0x5a, 0x28, 0x4c, 0x9e, 0xf7,
        0x52, 0x18, 0x29, 0xcf, 0x10, 0x79, 0xb0, 0x80, 0xe9, 0xd7, 0x4a, 0x1c, 0x10, 0xfc, 0xab,
        0x6a, 0x42, 0x43, 0xd3, 0x36, 0x56, 0xde, 0xbe, 0x4c, 0x1e, 0xd7, 0x96, 0x48, 0xe8, 0x56,
        0xe8,
    ];

    /// `dequantize_row_iq4_xs` of IQ4_XS_BLOCK, from ggml-quants.c.
    #[rustfmt::skip]
    const IQ4_XS_EXPECTED: [f32; IQ4_XS_BLOCK_ELEMS] = [
        -4.167187500e+01, 4.265625000e+00, 2.920312500e+01, -4.167187500e+01, 2.920312500e+01, -2.723437500e+01,
        -1.607812500e+01, 3.707812500e+01, -7.218750000e+00, 2.920312500e+01, -4.167187500e+01, 3.707812500e+01,
        2.264062500e+01, -3.281250000e+00, -3.412500000e+01, -1.607812500e+01, -1.148437500e+01, -2.132812500e+01,
        1.246875000e+01, 3.707812500e+01, -3.281250000e+00, 1.739062500e+01, -2.132812500e+01, -3.281250000e+00,
        -4.167187500e+01, -7.218750000e+00, 2.264062500e+01, 3.281250000e-01, -1.148437500e+01, 1.739062500e+01,
        -1.148437500e+01, -2.723437500e+01, 7.500000000e+00, 4.875000000e+01, 9.525000000e+01, 4.875000000e+01,
        9.525000000e+01, 6.225000000e+01, -2.850000000e+01, -6.675000000e+01, 3.675000000e+01, 2.625000000e+01,
        3.675000000e+01, 9.525000000e+01, 2.625000000e+01, -7.500000000e-01, -5.175000000e+01, 3.675000000e+01,
        3.675000000e+01, -6.675000000e+01, 3.675000000e+01, 3.675000000e+01, 9.525000000e+01, 9.525000000e+01,
        1.650000000e+01, 1.650000000e+01, 2.625000000e+01, 2.625000000e+01, -9.750000000e+00, -1.875000000e+01,
        1.650000000e+01, 1.650000000e+01, 2.625000000e+01, 1.650000000e+01, 2.067187500e+01, -4.218750000e-01,
        -1.603125000e+01, -4.218750000e-01, 2.067187500e+01, 2.067187500e+01, -1.054687500e+01, 4.218750000e+00,
        4.387500000e+01, -5.484375000e+00, -1.054687500e+01, 4.387500000e+01, -2.910937500e+01, 4.387500000e+01,
        -3.754687500e+01, 1.476562500e+01, -2.235937500e+01, -5.484375000e+00, 5.357812500e+01, -1.603125000e+01,
        -2.910937500e+01, 1.476562500e+01, 2.067187500e+01, -4.218750000e-01, 3.501562500e+01, -1.054687500e+01,
        -5.484375000e+00, 5.357812500e+01, -1.054687500e+01, 3.501562500e+01, -5.484375000e+00, -1.603125000e+01,
        1.739062500e+01, -7.218750000e+00, -3.412500000e+01, 2.920312500e+01, -7.218750000e+00, -3.412500000e+01,
        8.203125000e+00, 2.264062500e+01, -2.132812500e+01, 1.246875000e+01, -3.281250000e+00, 2.920312500e+01,
        4.265625000e+00, -7.218750000e+00, 1.246875000e+01, -4.167187500e+01, 4.265625000e+00, 3.707812500e+01,
        8.203125000e+00, -1.148437500e+01, 3.707812500e+01, 3.707812500e+01, -1.148437500e+01, -3.412500000e+01,
        3.281250000e-01, -4.167187500e+01, 1.246875000e+01, 1.739062500e+01, -4.167187500e+01, 2.264062500e+01,
        1.246875000e+01, 1.739062500e+01, -3.560156250e+01, -7.265625000e+00, -7.265625000e+00, 3.850781250e+01,
        -3.560156250e+01, 3.850781250e+01, 5.013281250e+01, 3.850781250e+01, -9.227343750e+01, -2.542968750e+01,
        -9.227343750e+01, 3.850781250e+01, -7.265625000e+00, 3.850781250e+01, -7.265625000e+00, -6.030468750e+01,
        -9.227343750e+01, 6.466406250e+01, -7.556250000e+01, -2.542968750e+01, -1.598437500e+01, -4.722656250e+01,
        -7.265625000e+00, 6.466406250e+01, 2.760937500e+01, 2.760937500e+01, 7.265625000e-01, 6.466406250e+01,
        -4.722656250e+01, 2.760937500e+01, 9.445312500e+00, -7.556250000e+01, -1.455468750e+01, -1.877343750e+01,
        -5.273437500e+00, -1.877343750e+01, -2.742187500e+00, -8.015625000e+00, -1.877343750e+01, -1.455468750e+01,
        -2.383593750e+01, -2.383593750e+01, 1.750781250e+01, -1.117968750e+01, -5.273437500e+00, -2.109375000e-01,
        -1.117968750e+01, -1.877343750e+01, -1.455468750e+01, 1.750781250e+01, 4.640625000e+00, -5.273437500e+00,
        -8.015625000e+00, 1.033593750e+01, -5.273437500e+00, -2.109375000e-01, 1.750781250e+01, -2.742187500e+00,
        -5.273437500e+00, -2.742187500e+00, 7.382812500e+00, 1.750781250e+01, 1.033593750e+01, -2.742187500e+00,
        -6.093750000e+00, -5.057812500e+01, 6.093750000e-01, 7.921875000e+00, 6.885937500e+01, -7.739062500e+01,
        7.921875000e+00, -7.739062500e+01, -7.739062500e+01, 7.921875000e+00, -6.093750000e+00, 1.523437500e+01,
        3.229687500e+01, -7.739062500e+01, 3.229687500e+01, 2.315625000e+01, 6.885937500e+01, -2.132812500e+01,
        -6.337500000e+01, -5.057812500e+01, 3.229687500e+01, -6.337500000e+01, -6.093750000e+00, 2.315625000e+01,
        6.093750000e-01, 5.423437500e+01, 4.204687500e+01, -2.985937500e+01, -6.337500000e+01, -6.337500000e+01,
        6.885937500e+01, 1.523437500e+01, 5.859375000e-01, -1.945312500e+00, -1.523437500e+00, -1.523437500e+00,
        -5.156250000e-01, -5.156250000e-01, 2.085937500e+00, 2.085937500e+00, 1.242187500e+00, 2.085937500e+00,
        -2.343750000e-01, -5.156250000e-01, 2.343750000e-02, 2.343750000e-02, -5.156250000e-01, 2.343750000e-02,
        -5.156250000e-01, -1.148437500e+00, -1.148437500e+00, 1.617187500e+00, -1.523437500e+00, -8.203125000e-01,
        1.617187500e+00, 8.906250000e-01, -1.148437500e+00, -2.437500000e+00, 1.617187500e+00, 3.046875000e-01,
        -1.148437500e+00, 2.085937500e+00, -8.203125000e-01, 2.085937500e+00,
    ];

    #[test]
    fn iq4_xs_block_matches_the_ggml_reference() {
        let mut got = [0.0f32; IQ4_XS_BLOCK_ELEMS];
        dequantize_iq4_xs_block(&IQ4_XS_BLOCK, &mut got);
        for (i, (&g, &e)) in got.iter().zip(IQ4_XS_EXPECTED.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-6 * e.abs().max(1.0),
                "iq4_xs[{i}]: got {g}, ggml reference {e}"
            );
        }
    }

    #[test]
    fn iq4_xs_whole_run_dequantizes_every_block() {
        let mut data = Vec::new();
        data.extend_from_slice(&IQ4_XS_BLOCK);
        data.extend_from_slice(&IQ4_XS_BLOCK);
        let out = dequantize_iq4_xs(&data).expect("two whole blocks");
        assert_eq!(out.len(), 2 * IQ4_XS_BLOCK_ELEMS);
        assert_eq!(out[..IQ4_XS_BLOCK_ELEMS], out[IQ4_XS_BLOCK_ELEMS..]);
        assert!((out[0] - IQ4_XS_EXPECTED[0]).abs() <= 1e-6 * IQ4_XS_EXPECTED[0].abs().max(1.0));
    }

    #[test]
    fn iq4_xs_rejects_a_partial_block() {
        let err = dequantize_iq4_xs(&[0u8; IQ4_XS_BLOCK_BYTES - 1]).expect_err("partial block");
        assert!(format!("{err}").contains("not a multiple of block size"));
    }

    /// A zero super-block dequantizes to all zeros (`d == 0`), which pins that
    /// the scale is read from the first two bytes rather than assumed.
    #[test]
    fn iq4_xs_zero_block_is_all_zeros() {
        let mut got = [1.0f32; IQ4_XS_BLOCK_ELEMS];
        dequantize_iq4_xs_block(&[0u8; IQ4_XS_BLOCK_BYTES], &mut got);
        assert!(got.iter().all(|v| *v == 0.0), "zero scale must give zeros");
    }
}
