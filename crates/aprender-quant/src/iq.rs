//! IQ (importance-quantized) dequantization: `IQ4_NL` (ggml type 20), `IQ3_S` (21)
//! and `IQ4_XS` (23).
//!
//! #3947: `aprender-core` had no decoder for these types, so its GGUF reader
//! refused every IQ tensor (#3656), and `apr qa`'s `tensor_contract` gate
//! could not inspect three release models. The decoders and tables are
//! transcribed from `aprender-serve/src/quantize/{iq4_nl,iq3_s,iq4_xs,iq_grids}.rs`,
//! which transcribed them from llama.cpp's `ggml-quants.c` / `ggml-common.h`
//! (MIT, the ggml authors). They live in this crate because it is the one both
//! `aprender-core` and `aprender-serve` already depend on.
//!
//! Oracle: gguf-py 0.17.1 `gguf.quants.dequantize`. Before this port, serve's
//! decoders were bit-exact against it on all 271 IQ tensors (913,735,680
//! elements) of the three #3947 models. This port was re-measured against the
//! same oracle, and `tests` below pins blocks taken from those real files.

use crate::f16_to_f32;
use std::fmt;

/// ggml type id of `IQ4_NL`.
pub const GGML_TYPE_IQ4_NL: u32 = 20;
/// ggml type id of `IQ3_S`.
pub const GGML_TYPE_IQ3_S: u32 = 21;
/// ggml type id of `IQ4_XS`.
pub const GGML_TYPE_IQ4_XS: u32 = 23;

/// Elements in one `IQ4_NL` block (`QK4_NL`). **32, not 256** — the odd one out.
pub const IQ4_NL_BLOCK_ELEMS: usize = 32;
/// Bytes in one `IQ4_NL` block: an f16 scale plus 16 packed nibble bytes.
pub const IQ4_NL_BLOCK_BYTES: usize = 18;
/// Elements in one `IQ3_S` super-block.
pub const IQ3_S_BLOCK_ELEMS: usize = 256;
/// Bytes in one `IQ3_S` super-block (`sizeof(block_iq3_s)`).
pub const IQ3_S_BLOCK_BYTES: usize = 110;
/// Elements in one `IQ4_XS` super-block.
pub const IQ4_XS_BLOCK_ELEMS: usize = 256;
/// Bytes in one `IQ4_XS` super-block (`sizeof(block_iq4_xs)`).
pub const IQ4_XS_BLOCK_BYTES: usize = 136;

/// The 16 non-linear int8 levels shared by `IQ4_NL` and `IQ4_XS`.
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Bit j of a sign byte selects element j.
const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// 512 packed 4-magnitude lattice points for `IQ3_S` (`iq3s_grid`).
/// Transcribed table, kept byte-for-byte comparable with `ggml-common.h`.
#[rustfmt::skip]
#[allow(clippy::unreadable_literal)]
const IQ3S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
];

/// Why an IQ byte run could not be dequantized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IqDequantError {
    /// `ggml_type` is not one of the IQ types this module decodes.
    UnsupportedType(u32),
    /// The element count is not a whole number of blocks.
    PartialBlock {
        /// ggml type id.
        ggml_type: u32,
        /// Requested element count.
        num_elements: usize,
        /// Elements per block for this type.
        block_elems: usize,
    },
    /// `data` holds fewer bytes than `num_elements` needs.
    Truncated {
        /// ggml type id.
        ggml_type: u32,
        /// Bytes needed.
        needed: usize,
        /// Bytes available.
        available: usize,
    },
}

impl fmt::Display for IqDequantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedType(t) => write!(f, "ggml type {t} is not an IQ type decoded here"),
            Self::PartialBlock { ggml_type, num_elements, block_elems } => write!(
                f,
                "ggml type {ggml_type}: {num_elements} elements is not a multiple of block size {block_elems}"
            ),
            Self::Truncated { ggml_type, needed, available } => write!(
                f,
                "ggml type {ggml_type}: needs {needed} bytes, only {available} available"
            ),
        }
    }
}

impl std::error::Error for IqDequantError {}

/// `(elements, bytes)` per block for the IQ types decoded here, else `None`.
#[must_use]
pub fn iq_block_layout(ggml_type: u32) -> Option<(usize, usize)> {
    match ggml_type {
        GGML_TYPE_IQ4_NL => Some((IQ4_NL_BLOCK_ELEMS, IQ4_NL_BLOCK_BYTES)),
        GGML_TYPE_IQ3_S => Some((IQ3_S_BLOCK_ELEMS, IQ3_S_BLOCK_BYTES)),
        GGML_TYPE_IQ4_XS => Some((IQ4_XS_BLOCK_ELEMS, IQ4_XS_BLOCK_BYTES)),
        _ => None,
    }
}

/// Dequantize `num_elements` values of IQ type `ggml_type` from the start of `data`.
///
/// `data` may extend past the tensor (the caller can pass the rest of the file);
/// exactly `num_elements / block_elems * block_bytes` bytes are read.
///
/// # Errors
/// [`IqDequantError::UnsupportedType`] for a type not decoded here — never a
/// fallback to another format (#3850); [`IqDequantError::PartialBlock`] when
/// `num_elements` is not whole blocks; [`IqDequantError::Truncated`] when
/// `data` is too short.
pub fn dequantize_iq_to_f32(
    ggml_type: u32,
    data: &[u8],
    num_elements: usize,
) -> Result<Vec<f32>, IqDequantError> {
    let (block_elems, block_bytes) =
        iq_block_layout(ggml_type).ok_or(IqDequantError::UnsupportedType(ggml_type))?;
    if !num_elements.is_multiple_of(block_elems) {
        return Err(IqDequantError::PartialBlock {
            ggml_type,
            num_elements,
            block_elems,
        });
    }
    let nb = num_elements / block_elems;
    let needed = nb * block_bytes;
    if data.len() < needed {
        return Err(IqDequantError::Truncated {
            ggml_type,
            needed,
            available: data.len(),
        });
    }
    let decode: fn(&[u8], &mut [f32]) = match ggml_type {
        GGML_TYPE_IQ4_NL => dequantize_iq4_nl_block,
        GGML_TYPE_IQ3_S => dequantize_iq3_s_block,
        _ => dequantize_iq4_xs_block,
    };
    let mut out = vec![0.0f32; num_elements];
    for (block, dst) in data[..needed]
        .chunks_exact(block_bytes)
        .zip(out.chunks_exact_mut(block_elems))
    {
        decode(block, dst);
    }
    Ok(out)
}

/// One `IQ4_NL` block. The two nibbles of byte `j` land at `j` and `j + 16`,
/// not at `2j` / `2j + 1` (`y[j]` / `y[j + QK4_NL/2]` in the reference).
fn dequantize_iq4_nl_block(block: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for (j, &byte) in block[2..IQ4_NL_BLOCK_BYTES].iter().enumerate() {
        out[j] = d * f32::from(KVALUES_IQ4NL[usize::from(byte & 0xf)]);
        out[j + 16] = d * f32::from(KVALUES_IQ4NL[usize::from(byte >> 4)]);
    }
}

/// One `IQ4_XS` super-block: `IQ4_NL`'s codebook under a 6-bit per-sub-block
/// scale split across `scales_l` (low 4 bits) and `scales_h` (high 2 bits).
fn dequantize_iq4_xs_block(block: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[4..8];
    let qs = &block[8..IQ4_XS_BLOCK_BYTES];
    for ib in 0..8 {
        let ls = u16::from((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf)
            | ((scales_h >> (2 * ib)) & 3) << 4;
        let dl = d * (f32::from(ls) - 32.0);
        for j in 0..16 {
            let byte = qs[16 * ib + j];
            out[32 * ib + j] = dl * f32::from(KVALUES_IQ4NL[usize::from(byte & 0xf)]);
            out[32 * ib + j + 16] = dl * f32::from(KVALUES_IQ4NL[usize::from(byte >> 4)]);
        }
    }
}

/// One `IQ3_S` super-block: 9-bit grid indices (8 bits in `qs`, 1 in `qh`),
/// explicit sign bytes, and 4-bit scales shared by two 32-value sub-blocks.
fn dequantize_iq3_s_block(block: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..66];
    let qh = &block[66..74];
    let signs = &block[74..106];
    let scales = &block[106..110];
    for ib32 in (0..8).step_by(2) {
        let sc = scales[ib32 / 2];
        let dbs = [
            d * (1.0 + 2.0 * f32::from(sc & 0xf)),
            d * (1.0 + 2.0 * f32::from(sc >> 4)),
        ];
        for (half, &db) in dbs.iter().enumerate() {
            let sub = ib32 + half;
            let qh_byte = usize::from(qh[sub]);
            for l in 0..4 {
                let i1 = usize::from(qs[8 * sub + 2 * l]) | ((qh_byte << (8 - 2 * l)) & 256);
                let i2 = usize::from(qs[8 * sub + 2 * l + 1]) | ((qh_byte << (7 - 2 * l)) & 256);
                let (g1, g2) = (IQ3S_GRID[i1].to_le_bytes(), IQ3S_GRID[i2].to_le_bytes());
                let sign_byte = signs[4 * sub + l];
                let base = 32 * sub + 8 * l;
                for j in 0..4 {
                    let m1 = f32::from(g1[j]);
                    let m2 = f32::from(g2[j]);
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

#[cfg(test)]
#[allow(clippy::unreadable_literal)] // f32 bit patterns copied from gguf-py
mod tests {
    use super::*;

    /// Assert `got` at each `(index, f32 bits)` is bit-identical to gguf-py.
    fn assert_bits(ty: &str, got: &[f32], want: &[(usize, u32)]) {
        for &(i, bits) in want {
            assert_eq!(
                got[i].to_bits(),
                bits,
                "{ty}[{i}]: got {}, gguf-py {}",
                got[i],
                f32::from_bits(bits)
            );
        }
    }

    // Real blocks, not synthetic ones: block 0 of the first tensor of each type in
    // the #3947 release models, with expected values from gguf-py 0.17.1
    // `gguf.quants.dequantize` on that same tensor. A synthetic block can share the
    // decoder's misreading of the layout; bytes a real quantizer wrote cannot.

    /// `IQ4_NL`: block 0 of `blk.0.ffn_gate.weight`, Qwen2.5-0.5B-Instruct-IQ4_XS.gguf.
    const IQ4_NL_BLOCK: [u8; IQ4_NL_BLOCK_BYTES] = [
        0x29, 0x8e, 0x7d, 0x04, 0xb9, 0xc3, 0x16, 0x62, 0x54, 0x49, 0x9b, 0xb8, 0x66, 0x84, 0xe3,
        0x6a, 0xc5, 0xb2,
    ];
    /// gguf-py's value at each index, as f32 bits.
    const IQ4_NL_EXPECTED: [(usize, u32); 32] = [
        (0, 0xbcd48680),
        (1, 0x3c96ec80),
        (2, 0xbba02a00),
        (3, 0x3cc83480),
        (4, 0x3c078600),
        (5, 0x3cffa580),
        (6, 0x3c96ec80),
        (7, 0xbba02a00),
        (8, 0xbc6a1600),
        (9, 0xb9c52000),
        (10, 0x3c078600),
        (11, 0x3c96ec80),
        (12, 0x3cc83480),
        (13, 0xbc1a0100),
        (14, 0x3c579b00),
        (15, 0x3cffa580),
        (16, 0x3b766800),
        (17, 0x3d4395c0),
        (18, 0xbc6a1600),
        (19, 0xbca33e80),
        (20, 0x3d202a00),
        (21, 0x3c078600),
        (22, 0x3c579b00),
        (23, 0x3c96ec80),
        (24, 0xbba02a00),
        (25, 0xbc6a1600),
        (26, 0x3c078600),
        (27, 0xb9c52000),
        (28, 0xbd091040),
        (29, 0x3c078600),
        (30, 0xbca33e80),
        (31, 0xbc6a1600),
    ];

    /// `IQ4_XS`: block 0 of `blk.0.ffn_down.weight`, Qwen2.5-0.5B-Instruct-IQ4_XS.gguf.
    const IQ4_XS_BLOCK: [u8; IQ4_XS_BLOCK_BYTES] = [
        0x2e, 0x01, 0x3d, 0xfc, 0x40, 0x04, 0xa4, 0x4b, 0x8a, 0x36, 0x31, 0xd6, 0xaa, 0xcb, 0xde,
        0xb7, 0xa6, 0x97, 0x95, 0x40, 0x5a, 0x67, 0xdc, 0xf1, 0xab, 0x61, 0xa8, 0x8a, 0x5a, 0x82,
        0x9d, 0x58, 0x61, 0xd4, 0x77, 0xc0, 0xc5, 0xc5, 0x45, 0xce, 0x09, 0x8c, 0xaa, 0xb0, 0xa8,
        0xa6, 0x20, 0xa4, 0x2d, 0x86, 0xec, 0x15, 0x93, 0x95, 0x88, 0x2c, 0x8b, 0x77, 0x6a, 0xa9,
        0x63, 0x9b, 0x77, 0x88, 0x57, 0x40, 0xa7, 0x77, 0x24, 0xa8, 0x6a, 0x7b, 0x97, 0x88, 0xc8,
        0x90, 0xbe, 0xa6, 0x6b, 0x77, 0x7c, 0x45, 0x4c, 0x99, 0xb7, 0x1c, 0x6b, 0x83, 0xea, 0x57,
        0xc2, 0x75, 0x87, 0x45, 0x04, 0xa7, 0x9a, 0x69, 0x7a, 0x56, 0x8a, 0xc4, 0x67, 0x3a, 0x83,
        0x18, 0x86, 0x40, 0x08, 0x48, 0x55, 0xa7, 0x82, 0xf8, 0x49, 0x96, 0xa3, 0xc5, 0x86, 0xc9,
        0x77, 0x2c, 0xa8, 0x52, 0x93, 0x24, 0xac, 0x27, 0x48, 0x4b, 0xc4, 0x86, 0x64, 0xba, 0x39,
        0xd0,
    ];
    /// gguf-py's value at each index, as f32 bits.
    const IQ4_XS_EXPECTED: [(usize, u32); 38] = [
        (0, 0xbbebf000),
        (7, 0x3b3cc000),
        (14, 0xbc7a1800),
        (21, 0xbc7a1800),
        (28, 0x3c252800),
        (35, 0x3c137600),
        (42, 0xbb6bf000),
        (49, 0xbc01c400),
        (56, 0xbc01c400),
        (63, 0x3c9c4f00),
        (70, 0xbd3b4680),
        (77, 0xbc4e7200),
        (84, 0x3c137600),
        (91, 0xbd195c00),
        (98, 0xbc6bf000),
        (105, 0x3d95d200),
        (112, 0xba170000),
        (119, 0xba170000),
        (126, 0x3c4fa000),
        (133, 0x3c35ac00),
        (140, 0x3ba52800),
        (147, 0xbbd6b400),
        (154, 0x3cca5100),
        (161, 0xbb995c00),
        (168, 0x3c3fb300),
        (175, 0x3c3fb300),
        (182, 0xbd737540),
        (189, 0x3ccb3380),
        (196, 0x39fed000),
        (203, 0xbc2f2f00),
        (210, 0x39fed000),
        (217, 0x3d60f3a0),
        (224, 0xbb6bf000),
        (231, 0xbb6bf000),
        (238, 0x3b995c00),
        (245, 0xbcf4c900),
        (252, 0xbc01c400),
        (255, 0x3ccb7f00),
    ];

    /// `IQ3_S`: block 0 of `blk.11.ffn_down.weight`, Qwen2.5-0.5B-Instruct-IQ3_M.gguf.
    const IQ3_S_BLOCK: [u8; IQ3_S_BLOCK_BYTES] = [
        0xf9, 0x02, 0x68, 0xdd, 0x4c, 0xf8, 0x5f, 0x70, 0x78, 0x34, 0x2d, 0x6a, 0xb3, 0x72, 0x7d,
        0xcf, 0x15, 0x86, 0x3d, 0x6f, 0x14, 0x76, 0xa6, 0x4b, 0x4e, 0x6e, 0x90, 0x64, 0x91, 0xa8,
        0x0a, 0xe1, 0x8a, 0xa3, 0xdc, 0xeb, 0x7d, 0xe0, 0x23, 0x43, 0x1f, 0x30, 0x39, 0xe5, 0x7f,
        0x67, 0x2d, 0x89, 0xe0, 0x3f, 0x81, 0xaf, 0xf3, 0x23, 0x05, 0x0e, 0x53, 0x13, 0x99, 0xc1,
        0x6a, 0x51, 0x3b, 0x28, 0x79, 0xa8, 0x08, 0xda, 0xd8, 0x53, 0x40, 0xd8, 0x04, 0x0a, 0xf6,
        0xf9, 0x0b, 0xa9, 0x1c, 0x05, 0x2a, 0xa4, 0x17, 0xa4, 0xe3, 0xb9, 0x15, 0x86, 0x7e, 0x89,
        0x7b, 0xf1, 0xd0, 0x6d, 0x84, 0xaf, 0x1e, 0x1a, 0xde, 0x45, 0x8b, 0x3c, 0xaa, 0x02, 0xb8,
        0x56, 0xef, 0xbb, 0xdf, 0xdd,
    ];
    /// gguf-py's value at each index, as f32 bits.
    const IQ3_S_EXPECTED: [(usize, u32); 38] = [
        (0, 0x3be66180),
        (7, 0xbbe66180),
        (14, 0xbc4f57c0),
        (21, 0x3ab84e00),
        (28, 0x3be66180),
        (35, 0xbaac6a00),
        (42, 0xbc6d11c0),
        (49, 0xbbd78480),
        (56, 0x3c8c1620),
        (63, 0xbc41f740),
        (70, 0x3a88be00),
        (77, 0xbc3c0540),
        (84, 0x3c803220),
        (91, 0xbbef4c80),
        (98, 0xbc803220),
        (105, 0xbbaaed80),
        (112, 0x3a88be00),
        (119, 0x3baaed80),
        (126, 0x3bef4c80),
        (133, 0xbc214440),
        (140, 0xbbe66180),
        (147, 0x3ab84e00),
        (154, 0xbab84e00),
        (161, 0x3c967da0),
        (168, 0xbaa08600),
        (175, 0xbc3496c0),
        (182, 0x3b70c900),
        (189, 0x3c3496c0),
        (196, 0xbb70c900),
        (203, 0x3c967da0),
        (210, 0x3aa08600),
        (217, 0x3aa08600),
        (224, 0x3b70c900),
        (231, 0xbc826ce0),
        (238, 0x3c5cb840),
        (245, 0xbc967da0),
        (252, 0xbc3496c0),
        (255, 0x3b70c900),
    ];

    #[test]
    fn iq4_nl_real_block_matches_gguf_py_bit_for_bit() {
        let got = dequantize_iq_to_f32(GGML_TYPE_IQ4_NL, &IQ4_NL_BLOCK, IQ4_NL_BLOCK_ELEMS)
            .expect("one whole block");
        assert_bits("IQ4_NL", &got, &IQ4_NL_EXPECTED);
    }

    #[test]
    fn iq4_xs_real_block_matches_gguf_py_bit_for_bit() {
        let got = dequantize_iq_to_f32(GGML_TYPE_IQ4_XS, &IQ4_XS_BLOCK, IQ4_XS_BLOCK_ELEMS)
            .expect("one whole block");
        assert_bits("IQ4_XS", &got, &IQ4_XS_EXPECTED);
    }

    #[test]
    fn iq3_s_real_block_matches_gguf_py_bit_for_bit() {
        let got = dequantize_iq_to_f32(GGML_TYPE_IQ3_S, &IQ3_S_BLOCK, IQ3_S_BLOCK_ELEMS)
            .expect("one whole block");
        assert_bits("IQ3_S", &got, &IQ3_S_EXPECTED);
    }

    /// Positive control for the three tests above: one flipped bit in the first
    /// quant byte must move element 0 off the gguf-py value. If it did not, those
    /// tests could not tell a wrong decode from a right one.
    #[test]
    fn a_single_flipped_quant_bit_is_seen_by_the_pinned_values() {
        for (ty, block, qs_byte, elems, want0) in [
            (
                GGML_TYPE_IQ4_NL,
                &IQ4_NL_BLOCK[..],
                2,
                IQ4_NL_BLOCK_ELEMS,
                IQ4_NL_EXPECTED[0],
            ),
            (
                GGML_TYPE_IQ4_XS,
                &IQ4_XS_BLOCK[..],
                8,
                IQ4_XS_BLOCK_ELEMS,
                IQ4_XS_EXPECTED[0],
            ),
            (
                GGML_TYPE_IQ3_S,
                &IQ3_S_BLOCK[..],
                2,
                IQ3_S_BLOCK_ELEMS,
                IQ3_S_EXPECTED[0],
            ),
        ] {
            let mut mutant = block.to_vec();
            mutant[qs_byte] ^= 1;
            let got = dequantize_iq_to_f32(ty, &mutant, elems).expect("one whole block");
            assert_ne!(
                got[want0.0].to_bits(),
                want0.1,
                "type {ty}: flip went unseen"
            );
        }
    }

    /// #3850: a type this module does not decode is refused, never decoded as
    /// something else. `IQ2_XXS` (16) is a real IQ type with no decoder here.
    #[test]
    fn an_undecoded_iq_type_is_refused_not_approximated() {
        for ty in [16, 17, 18, 19, 22, 12, 999] {
            assert_eq!(
                dequantize_iq_to_f32(ty, &[0u8; 4096], 256),
                Err(IqDequantError::UnsupportedType(ty))
            );
            assert_eq!(iq_block_layout(ty), None);
        }
    }

    #[test]
    fn a_partial_block_is_refused_rather_than_truncated() {
        assert!(matches!(
            dequantize_iq_to_f32(GGML_TYPE_IQ4_XS, &[0u8; 4096], 255),
            Err(IqDequantError::PartialBlock {
                block_elems: 256,
                ..
            })
        ));
        // IQ4_NL's block is 32 elements, not 256: 32 is whole, 16 is not.
        assert!(dequantize_iq_to_f32(GGML_TYPE_IQ4_NL, &[0u8; 18], 32).is_ok());
        assert!(dequantize_iq_to_f32(GGML_TYPE_IQ4_NL, &[0u8; 18], 16).is_err());
    }

    #[test]
    fn short_data_is_refused_and_trailing_bytes_are_ignored() {
        assert_eq!(
            dequantize_iq_to_f32(GGML_TYPE_IQ3_S, &IQ3_S_BLOCK[..109], 256),
            Err(IqDequantError::Truncated {
                ggml_type: 21,
                needed: 110,
                available: 109
            })
        );
        let mut padded = IQ3_S_BLOCK.to_vec();
        padded.extend_from_slice(&[0xff; 64]);
        let got = dequantize_iq_to_f32(GGML_TYPE_IQ3_S, &padded, 256).expect("trailing bytes");
        assert_eq!(got.len(), 256);
        assert_bits("IQ3_S", &got, &IQ3_S_EXPECTED);
    }

    #[test]
    fn block_layouts_match_ggml_sizeof() {
        assert_eq!(iq_block_layout(20), Some((32, 18)));
        assert_eq!(iq_block_layout(21), Some((256, 110)));
        assert_eq!(iq_block_layout(23), Some((256, 136)));
    }
}
