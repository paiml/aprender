//! `IQ4_NL` (ggml type 20) dequantization — 4 bits/weight: one f16 scale per
//! 32-element block and 4-bit indices into the same 16 non-linear levels
//! `IQ4_XS` uses.
//!
//! Layout and constants transcribed from llama.cpp's `ggml/src/ggml-quants.c`
//! (`dequantize_row_iq4_nl`) and `ggml/src/ggml-common.h` (`block_iq4_nl`,
//! `QK4_NL`) — MIT, the ggml authors; reference tree on this box
//! `/mnt/nvme-raid0/llama.cpp-master`.
//!
//! **`IQ4_NL` IS THE ODD ONE OUT.** Every other IQ type here is a 256-element
//! super-block; this one is **32 elements in 18 bytes**. `ggml-common.h:447`
//! says `#define QK4_NL 32`, and the struct's own static assert says
//! `sizeof(block_iq4_nl) == sizeof(ggml_half) + QK4_NL/2` = 18. Code that
//! assumes 256 elements per block will under-read an `IQ4_NL` row by 8x and
//! produce garbage rather than an error — see `iq_dispatch::iq_block_elems`,
//! which exists because of exactly this.

use crate::error::{RealizarError, Result};
use crate::quantize::f16_to_f32_lut;
use crate::quantize::iq_grids::KVALUES_IQ4NL;

/// ggml type id of `IQ4_NL`.
pub const GGML_TYPE_IQ4_NL: u32 = 20;

/// Elements in one `IQ4_NL` block (`QK4_NL`). **32, not 256.**
pub const IQ4_NL_BLOCK_ELEMS: usize = 32;

/// Bytes in one `IQ4_NL` block (`sizeof(block_iq4_nl)`): an f16 scale plus
/// `QK4_NL/2` packed nibbles.
pub const IQ4_NL_BLOCK_BYTES: usize = 18;

/// Dequantize one `IQ4_NL` block into `out` ([`IQ4_NL_BLOCK_ELEMS`] values).
///
/// The two nibbles of byte `j` are **not** adjacent outputs: the low nibble
/// lands at `j` and the high nibble at `j + 16`, exactly as the reference's
/// `y[j+0]` / `y[j+QK4_NL/2]` do. Reading them as `2*j` / `2*j+1` transposes
/// every block's two halves and still produces plausible-looking numbers.
///
/// # Panics
/// Panics if `block` is shorter than [`IQ4_NL_BLOCK_BYTES`] or `out` shorter
/// than [`IQ4_NL_BLOCK_ELEMS`] — callers slice exact blocks.
pub fn dequantize_iq4_nl_block(block: &[u8], out: &mut [f32]) {
    assert!(
        block.len() >= IQ4_NL_BLOCK_BYTES && out.len() >= IQ4_NL_BLOCK_ELEMS,
        "IQ4_NL block needs {IQ4_NL_BLOCK_BYTES} bytes and {IQ4_NL_BLOCK_ELEMS} outputs, got {} and {}",
        block.len(),
        out.len()
    );

    let d = f16_to_f32_lut(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..IQ4_NL_BLOCK_BYTES];
    for j in 0..16 {
        let byte = qs[j];
        out[j] = d * f32::from(KVALUES_IQ4NL[usize::from(byte & 0xf)]);
        out[j + 16] = d * f32::from(KVALUES_IQ4NL[usize::from(byte >> 4)]);
    }
}

/// Dequantize a whole `IQ4_NL` byte run (a multiple of [`IQ4_NL_BLOCK_BYTES`]).
///
/// # Errors
/// Returns [`RealizarError::InvalidShape`] when `data` is not a whole number of
/// blocks.
pub fn dequantize_iq4_nl(data: &[u8]) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(IQ4_NL_BLOCK_BYTES) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "IQ4_NL data length {} is not a multiple of block size {}",
                data.len(),
                IQ4_NL_BLOCK_BYTES
            ),
        });
    }
    let nb = data.len() / IQ4_NL_BLOCK_BYTES;
    let mut out = vec![0.0f32; nb * IQ4_NL_BLOCK_ELEMS];
    for (i, block) in data.chunks_exact(IQ4_NL_BLOCK_BYTES).enumerate() {
        dequantize_iq4_nl_block(
            block,
            &mut out[i * IQ4_NL_BLOCK_ELEMS..(i + 1) * IQ4_NL_BLOCK_ELEMS],
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One block: f16 scale pinned to 0x2600 (0.0234375, exactly representable)
    /// and a deterministic payload. Expected values produced from llama.cpp's
    /// `dequantize_row_iq4_nl`.
    const IQ4_NL_BLOCK: [u8; IQ4_NL_BLOCK_BYTES] = [
        0x00, 0x26, 0xf3, 0x42, 0x9e, 0xd6, 0x9f, 0xf3, 0x2a, 0xad, 0xec, 0x14, 0x6d, 0xc1, 0xae,
        0xa9, 0xef, 0x20,
    ];

    const EXPECTED: [f32; IQ4_NL_BLOCK_ELEMS] = [
        -1.5234375, -1.9453125, 2.0859375, -0.515625, 2.6484375, -1.5234375, 0.5859375, 1.6171875,
        1.2421875, -1.1484375, 1.6171875, -2.4375, 2.0859375, 0.3046875, 2.6484375, -2.9765625,
        2.6484375, -1.1484375, 0.3046875, 1.6171875, 0.3046875, 2.6484375, -1.9453125, 0.5859375,
        2.0859375, -2.4375, -0.515625, 1.2421875, 0.5859375, 0.5859375, 2.0859375, -1.9453125,
    ];

    #[test]
    fn one_block_matches_the_ggml_reference() {
        let mut out = [0.0f32; IQ4_NL_BLOCK_ELEMS];
        dequantize_iq4_nl_block(&IQ4_NL_BLOCK, &mut out);
        for (i, (got, want)) in out.iter().zip(EXPECTED.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1e-6,
                "element {i}: got {got}, reference {want}"
            );
        }
    }

    /// The structural cross-check, and the one that does not merely restate the
    /// arithmetic above.
    ///
    /// `IQ4_XS` is `IQ4_NL`'s codebook with a 6-bit per-sub-block scale layered
    /// on: its `dl = d * (ls - 32)`. So an `IQ4_XS` super-block whose eight
    /// sub-block scales are all `ls = 33` has `dl = d`, which is exactly
    /// `IQ4_NL`'s per-block scale — and its eight 32-element sub-blocks must
    /// then equal eight `IQ4_NL` blocks carrying the same nibbles.
    ///
    /// The two modules were transcribed independently from the reference, so a
    /// disagreement here means one of them is wrong, which is a much stronger
    /// statement than either module checking its own expected values.
    #[test]
    fn eight_iq4_nl_blocks_equal_one_flat_scaled_iq4_xs_superblock() {
        use crate::quantize::iq4_xs::{dequantize_iq4_xs_block, IQ4_XS_BLOCK_BYTES};

        // 128 payload bytes = 8 sub-blocks x 16 bytes, shared by both layouts.
        let mut qs = [0u8; 128];
        let mut state: u32 = 12345;
        for b in &mut qs {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *b = (state >> 16) as u8;
        }

        // IQ4_XS: d at [0..2], scales_h at [2..4], scales_l at [4..8], qs at [8..].
        // ls = 33 for every sub-block => low nibble 1 in scales_l, bit 1 set in
        // scales_h (33 = 0b100001: low 4 bits = 1, high 2 bits = 0b10).
        let mut xs = [0u8; IQ4_XS_BLOCK_BYTES];
        xs[0] = 0x00;
        xs[1] = 0x26; // d = 0.0234375
        let scales_h: u16 = (0..8).map(|ib| 2u16 << (2 * ib)).sum();
        xs[2..4].copy_from_slice(&scales_h.to_le_bytes());
        for b in &mut xs[4..8] {
            *b = 0x11; // low nibble 1 for both sub-blocks packed in each byte
        }
        xs[8..].copy_from_slice(&qs);

        let mut xs_out = [0.0f32; 256];
        dequantize_iq4_xs_block(&xs, &mut xs_out);

        // The same bytes as eight independent IQ4_NL blocks.
        let mut nl = Vec::with_capacity(8 * IQ4_NL_BLOCK_BYTES);
        for ib in 0..8 {
            nl.push(0x00);
            nl.push(0x26);
            nl.extend_from_slice(&qs[16 * ib..16 * (ib + 1)]);
        }
        let nl_out = dequantize_iq4_nl(&nl).expect("8 whole blocks");

        assert_eq!(nl_out.len(), 256);
        for (i, (a, b)) in nl_out.iter().zip(xs_out.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6,
                "element {i}: IQ4_NL {a} vs flat-scaled IQ4_XS {b} — the two \
                 transcriptions of the shared codebook disagree"
            );
        }
    }

    #[test]
    fn the_two_nibbles_of_a_byte_are_16_apart_not_adjacent() {
        // A block whose only non-zero byte is qs[0] = 0x0f: low nibble 15, high
        // nibble 0. KVALUES_IQ4NL[15] = 113 and [0] = -127, so element 0 must be
        // the large positive and element 16 the large negative. If the halves
        // were read as 2*j / 2*j+1 this lands at 0 and 1 instead.
        let mut block = [0u8; IQ4_NL_BLOCK_BYTES];
        block[1] = 0x3c; // f16 1.0
        block[2] = 0x0f;
        let mut out = [0.0f32; IQ4_NL_BLOCK_ELEMS];
        dequantize_iq4_nl_block(&block, &mut out);
        assert!((out[0] - 113.0).abs() <= 1e-6, "out[0] = {}", out[0]);
        assert!((out[16] - -127.0).abs() <= 1e-6, "out[16] = {}", out[16]);
        assert!((out[1] - -127.0).abs() <= 1e-6, "out[1] = {}", out[1]);
    }

    #[test]
    fn a_partial_block_is_refused_rather_than_truncated() {
        let err = dequantize_iq4_nl(&[0u8; IQ4_NL_BLOCK_BYTES + 1]).expect_err("not whole blocks");
        assert!(
            format!("{err}").contains("not a multiple of block size 18"),
            "{err}"
        );
    }

    #[test]
    fn the_block_is_32_elements_in_18_bytes_not_256() {
        // Stated as a test because everything else in this module's neighbourhood
        // is 256/136 and a reader will assume the same here.
        assert_eq!(IQ4_NL_BLOCK_ELEMS, 32);
        assert_eq!(IQ4_NL_BLOCK_BYTES, 18);
        assert_eq!(IQ4_NL_BLOCK_BYTES, 2 + IQ4_NL_BLOCK_ELEMS / 2);
    }
}
