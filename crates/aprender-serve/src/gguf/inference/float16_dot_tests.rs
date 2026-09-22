//! #3076: the F16/BF16 row dot that replaced `float16_matmul`'s per-element `fn`-pointer loop.
//!
//! Every length below is checked against an f64 reference, which reaches every branch of the
//! kernels: the 32-wide unrolled loop, the 8-wide loop, the scalar tail, and the portable
//! path's 64-element chunks and remainders. `float16_matmul` itself is checked against a
//! transcription of the loop it replaced, including a truncated weight buffer.

use super::matmul::{float16_matmul, float16_row_dot, float16_row_dot_portable, Float16Kind};

/// Deterministic values in roughly [-2, 2]: a 64-bit LCG, so a failure reproduces exactly.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

fn encode(kind: Float16Kind, w: &[f32]) -> Vec<u8> {
    w.iter()
        .flat_map(|&v| match kind {
            Float16Kind::F16 => half::f16::from_f32(v).to_bits().to_le_bytes(),
            Float16Kind::Bf16 => half::bf16::from_f32(v).to_bits().to_le_bytes(),
        })
        .collect()
}

fn decode(kind: Float16Kind, b: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(b);
    match kind {
        Float16Kind::F16 => half::f16::from_bits(bits).to_f32(),
        Float16Kind::Bf16 => half::bf16::from_bits(bits).to_f32(),
    }
}

/// (exact dot in f64 of the decoded weights, sum of |w * x| for the error bound).
fn reference(kind: Float16Kind, row: &[u8], x: &[f32]) -> (f64, f64) {
    row.chunks_exact(2)
        .zip(x)
        .fold((0.0, 0.0), |(dot, mag), (b, &xv)| {
            let p = f64::from(decode(kind, [b[0], b[1]])) * f64::from(xv);
            (dot + p, mag + p.abs())
        })
}

const LENGTHS: [usize; 16] = [
    0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 100, 129, 896, 4864, 4867,
];

fn assert_matches_reference(kind: Float16Kind, dot: fn(Float16Kind, &[u8], &[f32]) -> f32) {
    for (k, &n) in LENGTHS.iter().enumerate() {
        let row = encode(kind, &values(n, 2 * k as u64 + 1));
        let x = values(n, 2 * k as u64 + 2);
        let (exact, mag) = reference(kind, &row, &x);
        let got = f64::from(dot(kind, &row, &x));
        // f32 accumulation over n terms: error well inside 1e-5 of the magnitude sum here.
        assert!(
            (got - exact).abs() <= 1e-5 * mag + 1e-6,
            "{kind:?} n={n}: got {got}, exact {exact}, |w*x| sum {mag}"
        );
    }
}

#[test]
fn test_3076_f16_row_dot_matches_the_f64_reference_at_every_length() {
    assert_matches_reference(Float16Kind::F16, float16_row_dot);
}

#[test]
fn test_3076_bf16_row_dot_matches_the_f64_reference_at_every_length() {
    assert_matches_reference(Float16Kind::Bf16, float16_row_dot);
}

/// The fallback must be right on its own: on an AVX2 host `float16_row_dot` never reaches it.
#[test]
fn test_3076_portable_path_matches_the_f64_reference_at_every_length() {
    assert_matches_reference(Float16Kind::F16, float16_row_dot_portable);
    assert_matches_reference(Float16Kind::Bf16, float16_row_dot_portable);
}

/// Non-finite and subnormal weights reach the sum the way the scalar loop let them.
#[test]
fn test_3076_specials_propagate_through_every_lane_position() {
    for kind in [Float16Kind::F16, Float16Kind::Bf16] {
        let (nan, inf) = match kind {
            Float16Kind::F16 => (0x7E00u16, 0x7C00u16),
            Float16Kind::Bf16 => (0x7FC0u16, 0x7F80u16),
        };
        // Position 3 sits in the 32-wide loop, 37 in the 8-wide loop, 42 in the tail.
        for pos in [3usize, 37, 42] {
            let mut row = encode(kind, &vec![0.5; 43]);
            row[2 * pos..2 * pos + 2].copy_from_slice(&nan.to_le_bytes());
            assert!(
                float16_row_dot(kind, &row, &vec![1.0; 43]).is_nan(),
                "{kind:?} NaN at {pos}"
            );
            row[2 * pos..2 * pos + 2].copy_from_slice(&inf.to_le_bytes());
            let got = float16_row_dot(kind, &row, &vec![1.0; 43]);
            assert!(
                got.is_infinite() && got > 0.0,
                "{kind:?} +inf at {pos}: {got}"
            );
        }
    }
    // The smallest F16 subnormal is 2^-24, not zero: 40 of them times 2^20 is 40 / 16 = 2.5.
    let row: Vec<u8> = std::iter::repeat_n(0x0001u16.to_le_bytes(), 40)
        .flatten()
        .collect();
    let got = float16_row_dot(Float16Kind::F16, &row, &vec![(1u32 << 20) as f32; 40]);
    assert!((got - 2.5).abs() < 1e-6, "F16 subnormals: {got}");
}

/// The element count is `min(row.len() / 2, x.len())`: an odd trailing byte is not an element.
#[test]
fn test_3076_element_count_is_whole_elements_of_the_shorter_operand() {
    let kind = Float16Kind::F16;
    let row = encode(kind, &[1.0; 20]);
    let mut odd = row.clone();
    odd.push(0x3C); // half an element: must be ignored, not read as a weight
    assert_eq!(float16_row_dot(kind, &odd, &[2.0; 20]), 40.0);
    assert_eq!(float16_row_dot(kind, &row, &[2.0; 9]), 18.0);
    assert_eq!(float16_row_dot(kind, &row[..10], &[2.0; 20]), 10.0);
}

/// `float16_matmul` against the loop it replaced, transcribed: one element at a time, an
/// element counted only when both of its bytes are inside `data`.
fn pre_3076_float16_matmul(
    kind: Float16Kind,
    input: &[f32],
    data: &[u8],
    in_dim: usize,
    out_dim: usize,
    seq_len: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(seq_len * out_dim);
    for s in 0..seq_len {
        let x = &input[s * in_dim..(s + 1) * in_dim];
        for row in 0..out_dim {
            let mut sum = 0.0f64;
            for (col, &xv) in x.iter().enumerate() {
                let offset = row * in_dim * 2 + col * 2;
                if offset + 1 < data.len() {
                    sum +=
                        f64::from(decode(kind, [data[offset], data[offset + 1]])) * f64::from(xv);
                }
            }
            out.push(sum);
        }
    }
    out
}

#[test]
fn test_3076_float16_matmul_matches_the_pre_3076_loop_including_a_truncated_buffer() {
    let (in_dim, out_dim, seq_len) = (37, 5, 3);
    let input = values(in_dim * seq_len, 7);
    for kind in [Float16Kind::F16, Float16Kind::Bf16] {
        let full = encode(kind, &values(in_dim * out_dim, 8));
        // Full buffer, then one cut mid-row that also leaves an odd trailing byte.
        for data in [&full[..], &full[..full.len() - 2 * in_dim - 21]] {
            let got = float16_matmul(&input, data, in_dim, out_dim, seq_len, kind);
            let want = pre_3076_float16_matmul(kind, &input, data, in_dim, out_dim, seq_len);
            assert_eq!(got.len(), want.len());
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (f64::from(g) - w).abs() <= 1e-5 * w.abs().max(1.0),
                    "{kind:?} len={} out[{i}]: got {g}, want {w}",
                    data.len()
                );
            }
        }
    }
}
