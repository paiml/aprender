//! `IQ3_S` (ggml type 21) dequantization — 3.4375 bits/weight: 9-bit codebook indices (8 in `qs`, 1 in `qh`), explicit sign bytes and 4-bit scales shared by two sub-blocks.
//!
//! Layout and constants transcribed from llama.cpp's `ggml/src/ggml-quants.c`
//! (`dequantize_row_*`) and `ggml/src/ggml-common.h` (codebooks) — MIT, the ggml
//! authors; reference tree on this box `/mnt/nvme-raid0/llama.cpp-master` @ df03399.
//! The expected values below were produced from that reference and independently
//! confirmed against llama.cpp's `gguf-py` numpy dequantizers (50 random blocks
//! per type, PMAT-3477).

use crate::error::{RealizarError, Result};
use crate::quantize::f16_to_f32_lut;
use crate::quantize::iq_grids::{IQ3S_GRID, KMASK_IQ2XS};

/// ggml type id of `IQ3_S`.
pub const GGML_TYPE_IQ3_S: u32 = 21;

/// Elements in one `IQ3_S` super-block.
pub const IQ3_S_BLOCK_ELEMS: usize = 256;

/// Bytes in one `IQ3_S` super-block (`sizeof(block_iq3_s)`).
pub const IQ3_S_BLOCK_BYTES: usize = 110;

/// Dequantize one `IQ3_S` super-block into `out` (IQ3_S_BLOCK_ELEMS values).
///
/// # Panics
/// Panics if `block` is shorter than [`IQ3_S_BLOCK_BYTES`] or `out` shorter than
/// [`IQ3_S_BLOCK_ELEMS`] — callers slice exact blocks.
pub fn dequantize_iq3_s_block(block: &[u8], out: &mut [f32]) {
    assert!(
        block.len() >= IQ3_S_BLOCK_BYTES && out.len() >= IQ3_S_BLOCK_ELEMS,
        "IQ3_S block needs {} bytes and {} outputs, got {} and {}",
        IQ3_S_BLOCK_BYTES,
        IQ3_S_BLOCK_ELEMS,
        block.len(),
        out.len()
    );

    let d = f16_to_f32_lut(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..66];
    let qh = &block[66..74];
    let signs = &block[74..106];
    let scales = &block[106..110];
    // ggml walks two 32-value sub-blocks per packed scale byte; `half` is 0 for
    // the low nibble's sub-block and 1 for the high nibble's.
    for ib32 in (0..8).step_by(2) {
        let sc = scales[ib32 / 2];
        let dbs = [
            d * (1.0 + 2.0 * f32::from(sc & 0xf)),
            d * (1.0 + 2.0 * f32::from(sc >> 4)),
        ];
        for half in 0..2 {
            let db = dbs[half];
            let qs_off = 8 * (ib32 + half);
            let sg_off = 4 * (ib32 + half);
            let qh_byte = usize::from(qh[ib32 + half]);
            for l in 0..4 {
                let i1 = usize::from(qs[qs_off + 2 * l]) | ((qh_byte << (8 - 2 * l)) & 256);
                let i2 = usize::from(qs[qs_off + 2 * l + 1]) | ((qh_byte << (7 - 2 * l)) & 256);
                let g1 = IQ3S_GRID[i1];
                let g2 = IQ3S_GRID[i2];
                let sign_byte = signs[sg_off + l];
                let base = 32 * (ib32 + half) + 8 * l;
                for j in 0..4 {
                    let m1 = ((g1 >> (8 * j)) & 0xff) as f32;
                    let m2 = ((g2 >> (8 * j)) & 0xff) as f32;
                    let s1 = if sign_byte & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let s2 = if sign_byte & KMASK_IQ2XS[j + 4] != 0 {
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
}

/// Dequantize a whole `IQ3_S` byte run (a multiple of [`IQ3_S_BLOCK_BYTES`]).
///
/// # Errors
/// Returns [`RealizarError::InvalidShape`] when `data` is not a whole number of
/// super-blocks.
pub fn dequantize_iq3_s(data: &[u8]) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(IQ3_S_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "IQ3_S data length {} is not a multiple of block size {}",
                data.len(),
                IQ3_S_BLOCK_BYTES
            ),
        });
    }
    let nb = data.len() / IQ3_S_BLOCK_BYTES;
    let mut out = vec![0.0f32; nb * IQ3_S_BLOCK_ELEMS];
    for (i, block) in data.chunks_exact(IQ3_S_BLOCK_BYTES).enumerate() {
        dequantize_iq3_s_block(
            block,
            &mut out[i * IQ3_S_BLOCK_ELEMS..(i + 1) * IQ3_S_BLOCK_ELEMS],
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One super-block whose bytes are a fixed pseudo-random draw (scale pinned
    /// to a representable f16). Expected values come from the ggml reference.
    const IQ3_S_BLOCK: [u8; IQ3_S_BLOCK_BYTES] = [
        0x00, 0x26, 0x78, 0x9b, 0x34, 0xca, 0xf5, 0x4f, 0x2e, 0x22, 0x0a, 0xcd, 0x94, 0x1e, 0x71,
        0xb8, 0x8d, 0x58, 0x36, 0x86, 0x6d, 0x0d, 0x85, 0x8b, 0x63, 0x54, 0x9e, 0x94, 0xbe, 0x2c,
        0xac, 0xc6, 0x7f, 0x5b, 0x7e, 0xf2, 0x8f, 0x2d, 0x99, 0x03, 0x95, 0x9f, 0x63, 0xd3, 0xd8,
        0x93, 0xdc, 0xe7, 0x52, 0x77, 0x9c, 0x84, 0x16, 0x29, 0x17, 0xec, 0x8f, 0xf1, 0xaf, 0x4a,
        0x64, 0x22, 0xd3, 0x67, 0xe1, 0x8d, 0x5e, 0xb6, 0xdf, 0xa4, 0x65, 0xa5, 0x33, 0x1f, 0x75,
        0x8e, 0x79, 0x3e, 0xa9, 0x5a, 0x94, 0xeb, 0x0d, 0x15, 0xb6, 0x2a, 0x92, 0xa7, 0x09, 0xa5,
        0x93, 0xa4, 0x4e, 0xd2, 0x27, 0x96, 0x62, 0xe3, 0x95, 0x45, 0x80, 0xc3, 0x51, 0xa9, 0x04,
        0xba, 0x16, 0xe8, 0x56, 0xba,
    ];

    /// `dequantize_row_iq3_s` of IQ3_S_BLOCK, from ggml-quants.c.
    #[rustfmt::skip]
    const IQ3_S_EXPECTED: [f32; IQ3_S_BLOCK_ELEMS] = [
        -9.140625000e-01, 2.132812500e+00, -3.046875000e-01, 9.140625000e-01, -2.742187500e+00, -4.570312500e+00,
        -3.046875000e-01, 3.351562500e+00, 3.046875000e-01, -1.523437500e+00, -1.523437500e+00, -2.132812500e+00,
        1.523437500e+00, 2.742187500e+00, 1.523437500e+00, -3.960937500e+00, -2.132812500e+00, 3.351562500e+00,
        2.132812500e+00, -4.570312500e+00, -3.046875000e-01, -2.742187500e+00, -2.742187500e+00, 3.046875000e-01,
        3.046875000e-01, -2.742187500e+00, -9.140625000e-01, -2.132812500e+00, -4.570312500e+00, -1.523437500e+00,
        9.140625000e-01, 3.046875000e-01, -7.031250000e-02, 3.515625000e-01, 7.031250000e-02, -7.031250000e-02,
        7.031250000e-02, -1.054687500e+00, 3.515625000e-01, -9.140625000e-01, 6.328125000e-01, -2.109375000e-01,
        7.031250000e-02, -7.734375000e-01, -2.109375000e-01, 2.109375000e-01, -2.109375000e-01, 7.031250000e-02,
        7.031250000e-02, 2.109375000e-01, -3.515625000e-01, 6.328125000e-01, -2.109375000e-01, 7.734375000e-01,
        7.734375000e-01, -7.734375000e-01, -7.031250000e-02, -7.031250000e-02, 3.515625000e-01, -2.109375000e-01,
        7.031250000e-02, -3.515625000e-01, -1.054687500e+00, -4.921875000e-01, -3.585937500e+00, 2.789062500e+00,
        -1.992187500e+00, -2.789062500e+00, 4.382812500e+00, 3.984375000e-01, 4.382812500e+00, 3.585937500e+00,
        -1.195312500e+00, 4.382812500e+00, -1.195312500e+00, 3.585937500e+00, -3.984375000e-01, 3.984375000e-01,
        5.179687500e+00, 1.992187500e+00, 1.195312500e+00, -5.976562500e+00, -3.585937500e+00, 3.585937500e+00,
        -3.984375000e-01, -5.976562500e+00, 1.195312500e+00, -1.195312500e+00, 3.984375000e-01, -4.382812500e+00,
        3.984375000e-01, -3.585937500e+00, 5.179687500e+00, -1.195312500e+00, 5.179687500e+00, 2.789062500e+00,
        2.039062500e+00, -3.398437500e+00, 4.757812500e+00, 2.039062500e+00, -6.796875000e-01, 6.117187500e+00,
        3.398437500e+00, -2.039062500e+00, -2.039062500e+00, -2.039062500e+00, -6.796875000e-01, 8.835937500e+00,
        1.019531250e+01, -6.796875000e-01, 3.398437500e+00, -6.796875000e-01, -8.835937500e+00, 6.117187500e+00,
        6.117187500e+00, -2.039062500e+00, 1.019531250e+01, 3.398437500e+00, 2.039062500e+00, 8.835937500e+00,
        -6.796875000e-01, 6.796875000e-01, -2.039062500e+00, 2.039062500e+00, 6.796875000e-01, -6.796875000e-01,
        6.796875000e-01, -6.117187500e+00, -9.140625000e-01, -2.132812500e+00, 2.132812500e+00, 2.742187500e+00,
        -3.046875000e-01, 3.046875000e-01, 2.132812500e+00, -1.523437500e+00, 3.046875000e-01, 2.132812500e+00,
        -4.570312500e+00, 2.742187500e+00, 3.046875000e-01, -9.140625000e-01, 1.523437500e+00, -3.046875000e-01,
        9.140625000e-01, -3.046875000e-01, -2.132812500e+00, -9.140625000e-01, 9.140625000e-01, 2.742187500e+00,
        -2.742187500e+00, 1.523437500e+00, 1.523437500e+00, -1.523437500e+00, 3.046875000e-01, 3.351562500e+00,
        -4.570312500e+00, 1.523437500e+00, -2.132812500e+00, -9.140625000e-01, -2.578125000e-01, -2.835937500e+00,
        -2.578125000e-01, 2.320312500e+00, 1.804687500e+00, -2.320312500e+00, 2.578125000e-01, 1.289062500e+00,
        2.578125000e-01, -3.351562500e+00, -2.835937500e+00, 3.351562500e+00, -1.289062500e+00, 1.804687500e+00,
        1.289062500e+00, -7.734375000e-01, 1.289062500e+00, -2.578125000e-01, 7.734375000e-01, 1.289062500e+00,
        7.734375000e-01, -7.734375000e-01, -7.734375000e-01, 3.867187500e+00, -2.578125000e-01, -3.867187500e+00,
        2.320312500e+00, 2.578125000e-01, 1.289062500e+00, -3.351562500e+00, -1.289062500e+00, -2.320312500e+00,
        -1.476562500e+00, 4.921875000e-01, -1.476562500e+00, 5.414062500e+00, -4.921875000e-01, 4.429687500e+00,
        4.429687500e+00, -4.429687500e+00, -1.476562500e+00, 7.382812500e+00, -4.921875000e-01, 4.921875000e-01,
        4.921875000e-01, 4.921875000e-01, -2.460937500e+00, 4.921875000e-01, 3.445312500e+00, 4.429687500e+00,
        7.382812500e+00, 2.460937500e+00, 4.429687500e+00, 4.921875000e-01, 2.460937500e+00, -7.382812500e+00,
        -5.414062500e+00, -1.476562500e+00, 2.460937500e+00, 1.476562500e+00, 4.429687500e+00, 7.382812500e+00,
        -2.460937500e+00, -2.460937500e+00, -5.929687500e+00, 7.007812500e+00, 3.773437500e+00, 5.929687500e+00,
        -5.390625000e-01, 7.007812500e+00, -4.851562500e+00, 3.773437500e+00, -5.390625000e-01, 8.085937500e+00,
        5.390625000e-01, -4.851562500e+00, 2.695312500e+00, -4.851562500e+00, 5.390625000e-01, -3.773437500e+00,
        3.773437500e+00, 4.851562500e+00, -4.851562500e+00, 7.007812500e+00, 5.390625000e-01, 2.695312500e+00,
        8.085937500e+00, 5.390625000e-01, 5.929687500e+00, -2.695312500e+00, 1.617187500e+00, -2.695312500e+00,
        -5.390625000e-01, -5.390625000e-01, 2.695312500e+00, -1.617187500e+00,
    ];

    #[test]
    fn iq3_s_block_matches_the_ggml_reference() {
        let mut got = [0.0f32; IQ3_S_BLOCK_ELEMS];
        dequantize_iq3_s_block(&IQ3_S_BLOCK, &mut got);
        for (i, (&g, &e)) in got.iter().zip(IQ3_S_EXPECTED.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-6 * e.abs().max(1.0),
                "iq3_s[{i}]: got {g}, ggml reference {e}"
            );
        }
    }

    #[test]
    fn iq3_s_whole_run_dequantizes_every_block() {
        let mut data = Vec::new();
        data.extend_from_slice(&IQ3_S_BLOCK);
        data.extend_from_slice(&IQ3_S_BLOCK);
        let out = dequantize_iq3_s(&data).expect("two whole blocks");
        assert_eq!(out.len(), 2 * IQ3_S_BLOCK_ELEMS);
        assert_eq!(out[..IQ3_S_BLOCK_ELEMS], out[IQ3_S_BLOCK_ELEMS..]);
        assert!((out[0] - IQ3_S_EXPECTED[0]).abs() <= 1e-6 * IQ3_S_EXPECTED[0].abs().max(1.0));
    }

    #[test]
    fn iq3_s_rejects_a_partial_block() {
        let err = dequantize_iq3_s(&[0u8; IQ3_S_BLOCK_BYTES - 1]).expect_err("partial block");
        assert!(format!("{err}").contains("not a multiple of block size"));
    }

    /// A zero super-block dequantizes to all zeros (`d == 0`), which pins that
    /// the scale is read from the first two bytes rather than assumed.
    #[test]
    fn iq3_s_zero_block_is_all_zeros() {
        let mut got = [1.0f32; IQ3_S_BLOCK_ELEMS];
        dequantize_iq3_s_block(&[0u8; IQ3_S_BLOCK_BYTES], &mut got);
        assert!(got.iter().all(|v| *v == 0.0), "zero scale must give zeros");
    }
}
