//! #3931: IQ2_XXS block geometry, asserted as a PRECONDITION of the GPU kernel
//! rather than recorded in a comment.
//!
//! `iq_dispatch`'s own header states the failure this prevents, about IQ4_NL:
//! wiring a 32-element type against a 256-element assumption "does not produce
//! an error — it produces `blocks_per_row = in_dim/256`, so a row that occupies
//! 8 blocks is read as 1, every row after the first lands at the wrong offset,
//! and the untouched scratch tail silently contributes zeros. A plausible-looking
//! wrong answer where a refusal belongs."
//!
//! That is LAYOUT-001's shape in a different dimension, and with codebooks and
//! grids in the decode path a layout error and a decode error are
//! indistinguishable by eye. So the geometry is pinned before any output value
//! is compared, and the A/B asserts these same numbers about the buffer it
//! builds.

use super::iq2_xxs::{GGML_TYPE_IQ2_XXS, IQ2_XXS_BLOCK_BYTES, IQ2_XXS_BLOCK_ELEMS};
use super::iq_dispatch::{iq_block_bytes, iq_block_elems};

/// 2.0625 bits/weight: 66 bytes per 256 values.
///   2 bytes  f16 super-block scale
///  64 bytes  8 sub-blocks x (u32 codebook indices + u32 signs/scale)
#[test]
fn iq2_xxs_is_256_elements_in_66_bytes_3931() {
    assert_eq!(IQ2_XXS_BLOCK_ELEMS, 256);
    assert_eq!(IQ2_XXS_BLOCK_BYTES, 66);
    assert_eq!(
        2 + 8 * 8,
        IQ2_XXS_BLOCK_BYTES,
        "2 scale + 8 sub-blocks x 8 bytes"
    );
    // 66 bytes * 8 bits / 256 values
    let bits_per_weight = (IQ2_XXS_BLOCK_BYTES * 8) as f64 / IQ2_XXS_BLOCK_ELEMS as f64;
    assert!(
        (bits_per_weight - 2.0625).abs() < 1e-9,
        "IQ2_XXS is 2.0625 bpw by definition; got {bits_per_weight}"
    );
}

/// The dispatch must agree with the module's own constants. These are two
/// recordings of one fact and the kernel reads the dispatch, not the constants.
#[test]
fn the_dispatch_agrees_with_the_block_constants_3931() {
    assert_eq!(iq_block_elems(GGML_TYPE_IQ2_XXS), Some(IQ2_XXS_BLOCK_ELEMS));
    assert_eq!(iq_block_bytes(GGML_TYPE_IQ2_XXS), Some(IQ2_XXS_BLOCK_BYTES));
}

/// The row stride, which is the number the GPU kernel indexes with and the one
/// the IQ4_NL warning is about. A row of `k` values is `k/256` blocks of 66
/// bytes — NOT `k/256` blocks of some other size, and not `k/32`.
#[test]
fn the_row_stride_is_blocks_per_row_times_66_3931() {
    for (k, n) in [(256usize, 64usize), (512, 8), (2048, 3), (5120, 2)] {
        let blocks_per_row = k.div_ceil(IQ2_XXS_BLOCK_ELEMS);
        let row_bytes = blocks_per_row * IQ2_XXS_BLOCK_BYTES;
        assert_eq!(
            row_bytes,
            (k / 256) * 66,
            "k={k}: a row is (k/256) super-blocks of 66 bytes"
        );
        // The whole-weight size the A/B will assert about its own buffer.
        assert_eq!(n * row_bytes, n * blocks_per_row * IQ2_XXS_BLOCK_BYTES);
    }
}

/// The trap, stated as a test: computing the stride with the WRONG element count
/// gives a smaller buffer, and a kernel using it reads every row after the first
/// from the wrong offset. Pinned so the number cannot be changed silently.
#[test]
fn a_32_element_assumption_would_undersize_the_buffer_3931() {
    let (k, n) = (2048usize, 4usize);
    let right = n * k.div_ceil(256) * 66;
    let wrong_elems = n * k.div_ceil(32) * 66; // IQ4_NL's element count
    let wrong_bytes = n * k.div_ceil(256) * 18; // IQ4_NL's block size
    assert_ne!(
        right, wrong_elems,
        "a 32-element assumption must not coincide"
    );
    assert_ne!(
        right, wrong_bytes,
        "an 18-byte assumption must not coincide"
    );
    assert!(
        wrong_bytes < right,
        "the 18-byte assumption UNDER-sizes ({wrong_bytes} < {right}), which is the \
         silent case: the tail is never written and contributes zeros"
    );
}

/// The decoder is the A/B's oracle, so it must be shown to produce values at all
/// — an oracle returning zeros would agree with an all-zero kernel perfectly.
#[test]
fn the_cpu_decoder_produces_a_nondegenerate_block_3931() {
    let mut block = vec![0u8; IQ2_XXS_BLOCK_BYTES];
    block[0] = 0x00;
    block[1] = 0x3c; // f16 1.0 — exactly representable, so disagreement is indexing
    let mut state: u32 = 0x9E37_79B9;
    for b in block.iter_mut().skip(2) {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *b = (state >> 24) as u8;
    }
    let mut out = vec![0.0f32; IQ2_XXS_BLOCK_ELEMS];
    super::iq2_xxs::dequantize_iq2_xxs_block(&block, &mut out);

    let nonzero = out.iter().filter(|v| v.abs() > 1e-9).count();
    assert!(
        nonzero >= IQ2_XXS_BLOCK_ELEMS / 2,
        "only {nonzero} of {IQ2_XXS_BLOCK_ELEMS} decoded values are non-zero; an A/B \
         against this oracle would pass on a kernel that writes zeros"
    );
    assert!(
        out.iter().any(|v| *v < 0.0) && out.iter().any(|v| *v > 0.0),
        "the sign codes must produce both signs, or the sign path is untested"
    );
    let max = out.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    assert!(
        max.is_finite() && max > 0.0,
        "decoded magnitudes must be finite and non-zero"
    );
}

/// Deterministic IQ2_XXS weights for `n` rows of `k` values, with every block's
/// f16 scale pinned to an exactly representable 1.0 so that any GPU/CPU
/// disagreement is about INDEXING rather than rounding. Same construction the
/// device A/B will use, kept here so the oracle and the A/B cannot drift.
pub(crate) fn iq2_xxs_weights(n: usize, k: usize) -> Vec<u8> {
    let blocks_per_row = k.div_ceil(IQ2_XXS_BLOCK_ELEMS);
    let mut data = Vec::with_capacity(n * blocks_per_row * IQ2_XXS_BLOCK_BYTES);
    let mut state: u32 = 0x9E37_79B9;
    for _ in 0..(n * blocks_per_row) {
        data.push(0x00);
        data.push(0x3c); // f16 1.0
        for _ in 2..IQ2_XXS_BLOCK_BYTES {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            data.push((state >> 24) as u8);
        }
    }
    data
}

/// The oracle the device A/B will compare against, exercised end to end on the
/// CPU. If this cannot produce a non-degenerate result, the A/B is not a
/// measurement — it is two zeros agreeing.
#[test]
fn iq_parallel_matvec_decodes_iq2_xxs_end_to_end_3931() {
    let (k, n) = (512usize, 64usize); // 2 blocks per row, so the row stride is exercised
    let weights = iq2_xxs_weights(n, k);
    assert_eq!(
        weights.len(),
        n * (k / IQ2_XXS_BLOCK_ELEMS) * IQ2_XXS_BLOCK_BYTES,
        "the buffer must be exactly (n * blocks_per_row * 66) bytes"
    );

    let input: Vec<f32> = (0..k).map(|i| ((i % 13) as f32) - 6.0).collect();
    let got = super::iq_dispatch::iq_parallel_matvec(GGML_TYPE_IQ2_XXS, &weights, &input, k, n)
        .expect("IQ2_XXS is dispatched by iq_parallel_matvec — this is the A/B's oracle");

    assert_eq!(got.len(), n);
    let nonzero = got.iter().filter(|v| v.abs() > 1e-6).count();
    assert!(
        nonzero >= n / 2,
        "only {nonzero} of {n} oracle rows are non-zero; an A/B against this would pass \
         on a kernel that writes zeros"
    );
    assert!(
        got.iter().all(|v| v.is_finite()),
        "the oracle must not produce NaN/Inf"
    );
    // Rows must differ: identical rows would hide a kernel that ignores the row index.
    let distinct = got
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::collections::BTreeSet<_>>();
    assert!(
        distinct.len() >= n / 2,
        "only {} of {n} oracle rows are distinct; a kernel ignoring the row index would \
         still match",
        distinct.len()
    );
}

/// A row whose last block is PADDING — `k` not a multiple of 256. #3884's receipt
/// records that the IQ3_S A/B used `in_dim=512` specifically so the row stride
/// spans two super-blocks; this is the neighbouring case where the tail is short.
#[test]
fn the_oracle_handles_a_row_whose_last_block_is_padding_3931() {
    let (k, n) = (300usize, 8usize); // 300 -> 2 blocks, the second only 44/256 used
    assert_eq!(k.div_ceil(IQ2_XXS_BLOCK_ELEMS), 2);
    let weights = iq2_xxs_weights(n, k);
    let input: Vec<f32> = (0..k).map(|i| ((i % 7) as f32) - 3.0).collect();
    let got = super::iq_dispatch::iq_parallel_matvec(GGML_TYPE_IQ2_XXS, &weights, &input, k, n)
        .expect("a padded tail must decode, not error");
    assert_eq!(got.len(), n);
    assert!(got.iter().all(|v| v.is_finite()));
    assert!(
        got.iter().filter(|v| v.abs() > 1e-6).count() >= n / 2,
        "the padded-tail shape must still produce a non-degenerate result"
    );
}
