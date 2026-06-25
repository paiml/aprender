//! Forward-invariant round-trip fidelity gate for the k-quant schemes.
//!
//! PMAT-917 (Pillar-4 / CRUX-M verify-wall): pins quantize -> dequantize
//! reconstruction error within each scheme's theoretical quantization bound, so a
//! future regression in scale/offset/sub-block handling is caught the moment the
//! reconstructed block drifts past the bound that the bit-width can possibly achieve.
//!
//! For an affine (min + scale) uniform quantizer with `L` representable levels
//! covering a value range `R`, the worst-case per-element reconstruction error is
//! `R / (L - 1) / 2` (round-to-nearest). We allow a small multiplicative slack to
//! absorb the f16 rounding of the per-block `d`/`dmin` scales (GGML stores them as
//! f16) without making the gate vacuous.
//!
//! | scheme | bits | levels | base step bound          |
//! |--------|------|--------|--------------------------|
//! | Q4_K   | 4    | 16     | R / 15 / 2               |
//! | Q5_K   | 5    | 32     | R / 31 / 2               |
//! | Q6_K   | 6    | 64     | R / 63 / 2               |

use crate::{
    dequantize_q4_k_to_f32, dequantize_q5_k_to_f32, dequantize_q6_k_to_f32, quantize_q4_k,
    quantize_q5_k, quantize_q6_k,
};

/// f16 slack: the per-block `d`/`dmin` scales are stored as f16 (≈11-bit mantissa),
/// so a reconstructed level can be off by a relative ~2^-11 of the block scale on
/// top of the ideal mid-rise rounding error. 1.30x of the ideal step comfortably
/// covers that while staying far below 2x (the bound a one-bit-too-coarse scheme
/// would need), keeping the gate non-vacuous.
const F16_SLACK: f32 = 1.30;

/// A representative super-block exercising the hard cases for a fidelity gate:
/// large-magnitude weights, near-zero weights, the block-scale sign boundary, and
/// a smoothly-varying tail. NOT near-constant (a constant block would make any
/// dropped-offset/halved-scale mutation vacuously pass).
fn representative_block() -> Vec<f32> {
    (0..256)
        .map(|i| {
            let fi = i as f32;
            match i % 4 {
                // Large-magnitude span (~[-4, 4)) — drives the block scale.
                0 => 8.0 * (fi / 256.0 - 0.5),
                // Near-zero weights — sensitive to offset/min handling.
                1 => 1e-4 * fi,
                // Block-scale sign boundary — flips halfway through the block.
                2 => {
                    if i < 128 {
                        0.5
                    } else {
                        -0.5
                    }
                }
                // Smoothly varying tail.
                _ => fi.sin(),
            }
        })
        .collect()
}

/// max |reconstructed - original| over the block.
fn max_abs_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Value range of the block (max - min).
fn value_range(data: &[f32]) -> f32 {
    let hi = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lo = data.iter().copied().fold(f32::INFINITY, f32::min);
    hi - lo
}

/// Theoretical worst-case per-element bound for `levels` representable levels over
/// range `range`, inflated by the f16-scale slack.
fn fidelity_bound(range: f32, levels: u32) -> f32 {
    range / (levels as f32 - 1.0) / 2.0 * F16_SLACK
}

/// Assert a scheme's round-trip stays within its theoretical fidelity bound AND that
/// every reconstructed value is finite (a NaN/Inf scale bug must not slip through).
fn assert_within_bound(label: &str, original: &[f32], reconstructed: &[f32], levels: u32) {
    for (i, &v) in reconstructed.iter().enumerate() {
        assert!(
            v.is_finite(),
            "{label} round-trip produced non-finite value at index {i}: {v}"
        );
    }
    let range = value_range(original);
    let bound = fidelity_bound(range, levels);
    let err = max_abs_error(original, reconstructed);
    assert!(
        err <= bound,
        "{label} round-trip max error {err} exceeds theoretical fidelity bound {bound} \
         (range {range}, {levels} levels) — quantize/dequantize fidelity regression"
    );
}

#[test]
fn falsify_q4k_roundtrip_fidelity() {
    let data = representative_block();
    let bytes = quantize_q4_k(&data);
    assert_eq!(bytes.len(), 144, "Q4_K super-block must be 144 bytes");
    let recon = dequantize_q4_k_to_f32(&bytes, 256);
    assert_within_bound("Q4_K", &data, &recon, 16);
}

#[test]
fn falsify_q5k_roundtrip_fidelity() {
    let data = representative_block();
    let bytes = quantize_q5_k(&data);
    assert_eq!(bytes.len(), 176, "Q5_K super-block must be 176 bytes");
    let recon = dequantize_q5_k_to_f32(&bytes, 256);
    assert_within_bound("Q5_K", &data, &recon, 32);
}

#[test]
fn falsify_q6k_roundtrip_fidelity() {
    let data = representative_block();
    let bytes = quantize_q6_k(&data);
    assert_eq!(bytes.len(), 210, "Q6_K super-block must be 210 bytes");
    let recon = dequantize_q6_k_to_f32(&bytes, 256);
    assert_within_bound("Q6_K", &data, &recon, 64);
}

/// Cross-scheme monotonicity: more bits must NOT round-trip worse than fewer bits on
/// the same block (`Q6_K` ≤ `Q5_K` ≤ `Q4_K` error). A scale/offset bug in one scheme that
/// keeps it under its own (looser) bound is still caught here by the ordering.
#[test]
fn falsify_kquant_bitwidth_error_monotonic() {
    let data = representative_block();
    let e4 = max_abs_error(&data, &dequantize_q4_k_to_f32(&quantize_q4_k(&data), 256));
    let e5 = max_abs_error(&data, &dequantize_q5_k_to_f32(&quantize_q5_k(&data), 256));
    let e6 = max_abs_error(&data, &dequantize_q6_k_to_f32(&quantize_q6_k(&data), 256));
    // Small absolute tie tolerance for f16-scale jitter between schemes.
    let tol = 1e-3;
    assert!(
        e5 <= e4 + tol,
        "Q5_K error {e5} should not exceed Q4_K error {e4} (more bits, worse fidelity)"
    );
    assert!(
        e6 <= e5 + tol,
        "Q6_K error {e6} should not exceed Q5_K error {e5} (more bits, worse fidelity)"
    );
}
