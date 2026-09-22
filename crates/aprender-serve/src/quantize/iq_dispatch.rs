//! Dequantize-then-dot CPU path for the IQ (importance-quantized) ggml types.
//!
//! PMAT-3477 / #3091. Real unsloth Qwen3.5 GGUFs carry IQ2_XXS, IQ2_S, IQ3_XXS,
//! IQ3_S and IQ4_XS tensors. None of them has a fused kernel here; this module
//! gives the *correct* path — dequantize one row into a 256-element scratch and
//! dot it with the activation — so those files run. It is deliberately the
//! simple shape: no SIMD, no fused activation quantization. A fused kernel can
//! replace it later without changing the dispatch.
//!
//! Row layout matches the loader's sizing (`ggml_type_table.rs`): each row of a
//! 2-D tensor is padded to a whole number of super-blocks, so row `r` starts at
//! `r * blocks_per_row * block_bytes`.

use crate::error::{RealizarError, Result};
use crate::quantize::iq2_s::{dequantize_iq2_s_block, GGML_TYPE_IQ2_S, IQ2_S_BLOCK_BYTES};
use crate::quantize::iq2_xxs::{dequantize_iq2_xxs_block, GGML_TYPE_IQ2_XXS, IQ2_XXS_BLOCK_BYTES};
use crate::quantize::iq3_s::{dequantize_iq3_s_block, GGML_TYPE_IQ3_S, IQ3_S_BLOCK_BYTES};
use crate::quantize::iq3_xxs::{dequantize_iq3_xxs_block, GGML_TYPE_IQ3_XXS, IQ3_XXS_BLOCK_BYTES};
use crate::quantize::iq4_nl::{
    dequantize_iq4_nl_block, GGML_TYPE_IQ4_NL, IQ4_NL_BLOCK_BYTES, IQ4_NL_BLOCK_ELEMS,
};
use crate::quantize::iq4_xs::{dequantize_iq4_xs_block, GGML_TYPE_IQ4_XS, IQ4_XS_BLOCK_BYTES};

/// Elements in an IQ **super**-block (`QK_K`), and the scratch size every type
/// here fits in.
///
/// #3869: this is NO LONGER "elements per block for every IQ type". `IQ4_NL` is
/// 32 elements in 18 bytes (`ggml-common.h`: `#define QK4_NL 32`), and it is the
/// only one. Ask [`iq_block_elems`] for a type's element count; this constant is
/// now only the upper bound used to size the per-row scratch.
pub const IQ_BLOCK_ELEMS: usize = 256;

/// Bytes per super-block for an IQ type, or `None` if `qtype` is not one.
///
/// This doubles as the "can this module run it" predicate — the dispatch in
/// `fused_matmul` asks exactly this question.
#[must_use]
pub const fn iq_block_bytes(qtype: u32) -> Option<usize> {
    match qtype {
        GGML_TYPE_IQ2_XXS => Some(IQ2_XXS_BLOCK_BYTES),
        GGML_TYPE_IQ2_S => Some(IQ2_S_BLOCK_BYTES),
        GGML_TYPE_IQ3_XXS => Some(IQ3_XXS_BLOCK_BYTES),
        GGML_TYPE_IQ3_S => Some(IQ3_S_BLOCK_BYTES),
        GGML_TYPE_IQ4_XS => Some(IQ4_XS_BLOCK_BYTES),
        GGML_TYPE_IQ4_NL => Some(IQ4_NL_BLOCK_BYTES),
        _ => None,
    }
}

/// Elements per block for an IQ type, or `None` if `qtype` is not one.
///
/// #3869. Every type here except one is a 256-element super-block, and the
/// dispatch used to hardcode that in three places: `blocks_per_row =
/// in_dim.div_ceil(256)`, `col0 = b * 256`, and `n = 256.min(in_dim - col0)`.
///
/// `IQ4_NL` is **32 elements in 18 bytes**. Wiring it in without this accessor
/// does not produce an error — it produces `blocks_per_row = in_dim/256`, so a
/// row that occupies 8 blocks is read as 1, every row after the first lands at
/// the wrong offset, and the untouched scratch tail silently contributes zeros.
/// A plausible-looking wrong answer where a refusal belongs.
#[must_use]
pub const fn iq_block_elems(qtype: u32) -> Option<usize> {
    match qtype {
        GGML_TYPE_IQ2_XXS | GGML_TYPE_IQ2_S | GGML_TYPE_IQ3_XXS | GGML_TYPE_IQ3_S
        | GGML_TYPE_IQ4_XS => Some(IQ_BLOCK_ELEMS),
        GGML_TYPE_IQ4_NL => Some(IQ4_NL_BLOCK_ELEMS),
        _ => None,
    }
}

/// Dequantize one super-block of an IQ type into `out` (256 values).
///
/// # Errors
/// Returns [`RealizarError::UnsupportedOperation`] if `qtype` is not an IQ type
/// this module implements.
pub fn dequantize_iq_block(qtype: u32, block: &[u8], out: &mut [f32]) -> Result<()> {
    match qtype {
        GGML_TYPE_IQ2_XXS => dequantize_iq2_xxs_block(block, out),
        GGML_TYPE_IQ2_S => dequantize_iq2_s_block(block, out),
        GGML_TYPE_IQ3_XXS => dequantize_iq3_xxs_block(block, out),
        GGML_TYPE_IQ3_S => dequantize_iq3_s_block(block, out),
        GGML_TYPE_IQ4_XS => dequantize_iq4_xs_block(block, out),
        GGML_TYPE_IQ4_NL => dequantize_iq4_nl_block(block, out),
        other => {
            return Err(RealizarError::UnsupportedOperation {
                operation: "dequantize_iq_block".to_string(),
                reason: format!("ggml type {other} is not an IQ type with a dequantizer here"),
            });
        },
    }
    Ok(())
}

/// Dequantize a whole IQ tensor (flat, one super-block after another).
///
/// # Errors
/// Returns an error for a non-IQ `qtype` or a byte length that is not a whole
/// number of super-blocks.
pub fn dequantize_iq_tensor(qtype: u32, data: &[u8]) -> Result<Vec<f32>> {
    let block_bytes = iq_block_bytes(qtype).ok_or_else(|| RealizarError::UnsupportedOperation {
        operation: "dequantize_iq_tensor".to_string(),
        reason: format!("ggml type {qtype} is not an IQ type with a dequantizer here"),
    })?;
    if !data.len().is_multiple_of(block_bytes) {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "IQ tensor length {} is not a multiple of block size {block_bytes}",
                data.len()
            ),
        });
    }
    let elems = iq_block_elems(qtype).unwrap_or(IQ_BLOCK_ELEMS);
    let nb = data.len() / block_bytes;
    let mut out = vec![0.0f32; nb * elems];
    for (i, block) in data.chunks_exact(block_bytes).enumerate() {
        dequantize_iq_block(qtype, block, &mut out[i * elems..(i + 1) * elems])?;
    }
    Ok(out)
}

/// Row-major matvec for an IQ weight: `out[row] = dot(W[row], x)`.
///
/// Each row is dequantized one super-block at a time into a 256-element
/// scratch, so the peak extra memory is `threads * 1 KiB` rather than a full
/// f32 copy of the tensor.
///
/// # Errors
/// Returns an error for a non-IQ `qtype`, a short `x`/`data`, or an output
/// buffer smaller than `out_dim`.
pub fn iq_parallel_matvec_into(
    qtype: u32,
    data: &[u8],
    x: &[f32],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    use rayon::prelude::*;

    let block_bytes = iq_block_bytes(qtype).ok_or_else(|| RealizarError::UnsupportedOperation {
        operation: "iq_parallel_matvec_into".to_string(),
        reason: format!("ggml type {qtype} is not an IQ type with a dequantizer here"),
    })?;
    if x.len() < in_dim || output.len() < out_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "IQ matvec needs x >= {in_dim} (got {}) and output >= {out_dim} (got {})",
                x.len(),
                output.len()
            ),
        });
    }
    let elems = iq_block_elems(qtype).unwrap_or(IQ_BLOCK_ELEMS);
    let blocks_per_row = in_dim.div_ceil(elems);
    let row_bytes = blocks_per_row * block_bytes;
    let needed = row_bytes * out_dim;
    if data.len() < needed {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "IQ weight has {} bytes, needs {needed} for {out_dim}x{in_dim} (type {qtype})",
                data.len()
            ),
        });
    }

    output[..out_dim]
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(row, dst)| -> Result<()> {
            let row_data = &data[row * row_bytes..row * row_bytes + row_bytes];
            let mut scratch = [0.0f32; IQ_BLOCK_ELEMS];
            let mut sum = 0.0f32;
            for (b, block) in row_data.chunks_exact(block_bytes).enumerate() {
                dequantize_iq_block(qtype, block, &mut scratch[..elems])?;
                let col0 = b * elems;
                // The last block of a row may be padding past `in_dim`.
                let n = elems.min(in_dim - col0);
                for (j, w) in scratch[..n].iter().enumerate() {
                    sum += w * x[col0 + j];
                }
            }
            *dst = sum;
            Ok(())
        })?;
    Ok(())
}

/// Allocating form of [`iq_parallel_matvec_into`].
///
/// # Errors
/// Same conditions as [`iq_parallel_matvec_into`].
pub fn iq_parallel_matvec(
    qtype: u32,
    data: &[u8],
    x: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; out_dim];
    iq_parallel_matvec_into(qtype, data, x, in_dim, out_dim, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IQ_TYPES: [u32; 6] = [
        GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ4_XS,
        GGML_TYPE_IQ4_NL,
    ];

    /// A deterministic, well-formed super-block: an f16 scale of 0.0234375 and
    /// pseudo-random payload bytes.
    fn block_of(qtype: u32, salt: u8) -> Vec<u8> {
        let n = iq_block_bytes(qtype).expect("iq type");
        let mut b = vec![0u8; n];
        b[0] = 0x00;
        b[1] = 0x26; // half::f16 bits for 0.0234375, little-endian
        let mut state = u32::from(salt).wrapping_mul(2_654_435_761).wrapping_add(1);
        for byte in b.iter_mut().skip(2) {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            *byte = (state >> 16) as u8;
        }
        b
    }

    #[test]
    fn iq_block_bytes_knows_the_six_types_and_refuses_others() {
        assert_eq!(iq_block_bytes(GGML_TYPE_IQ2_XXS), Some(66));
        assert_eq!(iq_block_bytes(GGML_TYPE_IQ2_S), Some(82));
        assert_eq!(iq_block_bytes(GGML_TYPE_IQ3_XXS), Some(98));
        assert_eq!(iq_block_bytes(GGML_TYPE_IQ3_S), Some(110));
        assert_eq!(iq_block_bytes(GGML_TYPE_IQ4_XS), Some(136));
        // #3869: 18, and 32 elements — the only type here that is not a 256
        // element super-block.
        assert_eq!(iq_block_bytes(GGML_TYPE_IQ4_NL), Some(18));
        // Q4_K (12) has its own fused kernel and must NOT be claimed here.
        assert_eq!(iq_block_bytes(12), None);
        assert_eq!(iq_block_bytes(0), None);
    }

    /// A one-hot activation turns the matvec into a read of one weight, so the
    /// matvec and the dequantizer must agree element by element.
    #[test]
    fn matvec_with_a_one_hot_input_reads_the_dequantized_weight() {
        for qtype in IQ_TYPES {
            let elems = expected_block_elems(qtype);
            let data = block_of(qtype, 7);
            let dequantized = dequantize_iq_tensor(qtype, &data).expect("dequant");
            // #3869: probe columns that EXIST for this type. IQ4_NL blocks are 32
            // elements, so 130 and 255 are past the end of one block.
            for col in [0usize, 1, 31, 130, 255].into_iter().filter(|c| *c < elems) {
                let mut x = vec![0.0f32; elems];
                x[col] = 1.0;
                let got = iq_parallel_matvec(qtype, &data, &x, elems, 1).expect("matvec");
                assert!(
                    (got[0] - dequantized[col]).abs() <= 1e-6 * dequantized[col].abs().max(1.0),
                    "type {qtype} col {col}: matvec {} vs dequant {}",
                    got[0],
                    dequantized[col]
                );
            }
        }
    }

    /// Two rows with different payloads must not be read from the same offset —
    /// this is the test that fails if the row stride is wrong.
    #[test]
    fn rows_are_read_at_their_own_stride() {
        for qtype in IQ_TYPES {
            let mut data = block_of(qtype, 1);
            let row1 = block_of(qtype, 2);
            data.extend_from_slice(&row1);
            let elems = expected_block_elems(qtype);
            let x = vec![1.0f32; elems];
            let got = iq_parallel_matvec(qtype, &data, &x, elems, 2).expect("matvec");
            let d0: f32 = dequantize_iq_tensor(qtype, &data[..data.len() / 2])
                .expect("dequant")
                .iter()
                .sum();
            let d1: f32 = dequantize_iq_tensor(qtype, &row1)
                .expect("dequant")
                .iter()
                .sum();
            assert!(
                (got[0] - d0).abs() <= 1e-4 * d0.abs().max(1.0),
                "type {qtype} row 0"
            );
            assert!(
                (got[1] - d1).abs() <= 1e-4 * d1.abs().max(1.0),
                "type {qtype} row 1"
            );
        }
    }

    #[test]
    fn a_non_iq_type_is_refused_rather_than_guessed() {
        let err = iq_parallel_matvec(12, &[0u8; 144], &[0.0; 256], 256, 1).expect_err("Q4_K");
        assert!(format!("{err}").contains("not an IQ type"));
        let err = dequantize_iq_tensor(0, &[0u8; 4]).expect_err("F32");
        assert!(format!("{err}").contains("not an IQ type"));
    }

    #[test]
    fn a_short_weight_is_refused_rather_than_read_out_of_bounds() {
        let data = block_of(GGML_TYPE_IQ4_XS, 3);
        let err = iq_parallel_matvec(GGML_TYPE_IQ4_XS, &data, &[0.0; 256], 256, 2)
            .expect_err("one block cannot feed two rows");
        assert!(format!("{err}").contains("needs"));
    }

    /// #3869: a row that spans MORE THAN ONE block, which is what every real
    /// tensor has and what no existing row here had.
    ///
    /// `rows_are_read_at_their_own_stride` passes for IQ4_NL even with the
    /// arithmetic wrong, because it builds one block per row and calls the
    /// matvec with `in_dim = 256`. For a 32-element type that is not one row of
    /// 256 — it is 8 blocks. With `blocks_per_row = in_dim.div_ceil(256) = 1`
    /// the length check still passes, both rows still land on their own block,
    /// the 224 unwritten scratch slots are zero, and an all-ones activation
    /// makes them contribute nothing. Green, about nothing.
    ///
    /// This row builds `in_dim = 2 * elems_per_block` so the stride is actually
    /// exercised, and uses a POSITION-DEPENDENT activation so a block read at
    /// the wrong offset cannot coincidentally sum the same.
    #[test]
    fn a_row_spanning_two_blocks_is_read_at_the_types_own_stride() {
        for qtype in IQ_TYPES {
            let elems = expected_block_elems(qtype);
            let bytes = iq_block_bytes(qtype).expect("iq type");
            let in_dim = 2 * elems;
            let out_dim = 2;

            // Four blocks: rows [b0 b1] and [b2 b3], laid out exactly as a GGUF
            // row-major tensor does.
            let mut data = Vec::new();
            for salt in 1..=4u8 {
                data.extend_from_slice(&block_of(qtype, salt));
            }
            assert_eq!(data.len(), 4 * bytes);

            let x: Vec<f32> = (0..in_dim).map(|i| ((i % 11) as f32) - 5.0).collect();
            let got = iq_parallel_matvec(qtype, &data, &x, in_dim, out_dim)
                .unwrap_or_else(|e| panic!("type {qtype}: {e}"));

            // The answer, computed independently of the matvec's own arithmetic.
            let flat = dequantize_iq_tensor(qtype, &data).expect("dequant");
            for row in 0..out_dim {
                let want: f32 = (0..in_dim).map(|j| flat[row * in_dim + j] * x[j]).sum();
                assert!(
                    (got[row] - want).abs() <= 1e-3 * want.abs().max(1.0),
                    "type {qtype} ({elems} elems/{bytes} B per block), row {row}: matvec {} vs \
                     dequantized dot {want} — the row stride is wrong for this type",
                    got[row]
                );
            }
        }
    }

    /// The refusal that used to be a wrong answer.
    ///
    /// Two rows of 256 elements in IQ4_NL is 8 blocks per row — 144 bytes a row,
    /// 288 in total. Handed 36 bytes, the dispatch now says so:
    ///
    /// ```text
    /// IQ weight has 36 bytes, needs 288 for 2x256 (type 20)
    /// ```
    ///
    /// Before elements-per-block became a property of the type, the same call
    /// computed `blocks_per_row = 256.div_ceil(256) = 1`, decided 36 bytes were
    /// exactly enough, read one 18-byte block per row and summed it against 256
    /// activations — 224 of them against scratch it never wrote. It returned a
    /// float. That is the failure mode this whole change exists to remove: not a
    /// crash, not an error, a plausible number.
    #[test]
    fn a_short_iq4_nl_weight_is_refused_with_the_right_arithmetic() {
        let mut data = block_of(GGML_TYPE_IQ4_NL, 1);
        data.extend_from_slice(&block_of(GGML_TYPE_IQ4_NL, 2));
        assert_eq!(data.len(), 36);

        let err = iq_parallel_matvec(GGML_TYPE_IQ4_NL, &data, &[0.0; 256], 256, 2)
            .expect_err("36 bytes cannot be two 256-element rows of a 32-element type");
        let msg = format!("{err}");
        assert!(
            msg.contains("needs 288"),
            "the refusal must name the size the TYPE requires (8 blocks x 18 B x 2 rows), \
             not the size a 256-element type would: {msg}"
        );
    }

    /// `iq_block_elems` and `iq_block_bytes` must agree about which types exist,
    /// or a type gets a byte size and no element count and silently falls back
    /// to 256 via `unwrap_or`.
    #[test]
    fn every_type_with_a_block_size_has_an_element_count() {
        for qtype in 0u32..=40 {
            assert_eq!(
                iq_block_bytes(qtype).is_some(),
                iq_block_elems(qtype).is_some(),
                "type {qtype}: iq_block_bytes and iq_block_elems disagree about whether \
                 this is an IQ type, so one of them would fall back to a default"
            );
        }
        assert_eq!(iq_block_elems(GGML_TYPE_IQ4_NL), Some(32));
        assert_eq!(iq_block_elems(GGML_TYPE_IQ4_XS), Some(256));
    }

    /// Elements per block, stated independently of the production code so this
    /// file's tests cannot be satisfied by agreeing with the thing they check.
    fn expected_block_elems(qtype: u32) -> usize {
        if qtype == GGML_TYPE_IQ4_NL {
            32
        } else {
            256
        }
    }
}
