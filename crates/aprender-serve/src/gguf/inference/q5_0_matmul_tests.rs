//! #3602: the `Q5_0` CPU matmul that decodes one row at a time.
//!
//! The arm it replaced expanded the whole tensor through `dequantize_q5_0` on
//! every call. These tests hold the new kernel to that decoder: every output is
//! checked against an f64 dot over `dequantize_q5_0`'s weights, a batched call
//! must give each position exactly what a one-position call gives, and a weight
//! buffer that is not a whole number of rows is refused rather than read short.

use super::matmul::q5_0_matmul;
use crate::error::RealizarError;
use crate::quantize::dequantize_q5_0;

/// A 64-bit LCG, so a failure reproduces exactly.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 16
    }

    /// Roughly uniform in [-2, 2].
    fn unit(&mut self) -> f32 {
        (self.next() & 0xFF_FFFF) as f32 / (1u32 << 24) as f32 * 4.0 - 2.0
    }
}

/// `out_dim` rows of `in_dim / 32` random `Q5_0` blocks: every qs and qh bit
/// random, and a signed f16 scale, so all 32 quant levels and both signs occur.
fn q5_0_weights(in_dim: usize, out_dim: usize, rng: &mut Lcg) -> Vec<u8> {
    let blocks = out_dim * in_dim / 32;
    let mut data = Vec::with_capacity(blocks * 22);
    for _ in 0..blocks {
        let d = half::f16::from_f32(rng.unit() * 0.05);
        data.extend_from_slice(&d.to_bits().to_le_bytes());
        data.extend((0..20).map(|_| rng.next() as u8));
    }
    data
}

/// The output `fused_matmul`'s Q5_0 arm produced before #3602, in f64:
/// `dequantize_q5_0` over the whole tensor, then one dot per (position, row).
fn reference(data: &[u8], input: &[f32], in_dim: usize, out_dim: usize) -> Vec<(f64, f64)> {
    let w = dequantize_q5_0(data).expect("whole blocks");
    let seq_len = input.len() / in_dim;
    let mut out = Vec::with_capacity(seq_len * out_dim);
    for x in input.chunks_exact(in_dim) {
        for row in w.chunks_exact(in_dim) {
            let dot: f64 = row
                .iter()
                .zip(x)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let mag: f64 = row
                .iter()
                .zip(x)
                .map(|(&a, &b)| f64::from((a * b).abs()))
                .sum();
            out.push((dot, mag));
        }
    }
    out
}

#[test]
fn q5_0_matmul_matches_the_whole_tensor_dequant_it_replaced() {
    let mut rng = Lcg(3602);
    // 96 and 896 are not powers of two (896 is Qwen2.5-0.5B's hidden size, whose
    // K-quant files fall back to Q5_0); 65 and 130 rows straddle the 64-row
    // minimum chunk the parallel split uses.
    for &(in_dim, out_dim) in &[(32, 1), (64, 3), (96, 65), (896, 130)] {
        let data = q5_0_weights(in_dim, out_dim, &mut rng);
        for seq_len in [1, 2, 5] {
            let input: Vec<f32> = (0..seq_len * in_dim).map(|_| rng.unit()).collect();
            let got = q5_0_matmul(&input, &data, in_dim, out_dim, seq_len).expect("valid shape");
            let want = reference(&data, &input, in_dim, out_dim);
            assert_eq!(got.len(), want.len(), "{in_dim}x{out_dim} seq {seq_len}");
            for (i, (&g, &(w, mag))) in got.iter().zip(&want).enumerate() {
                let tol = mag * 2e-5 + 1e-6;
                assert!(
                    (f64::from(g) - w).abs() <= tol,
                    "{in_dim}x{out_dim} seq {seq_len} [{i}]: got {g}, want {w} (tol {tol})"
                );
            }
        }
    }
}

#[test]
fn q5_0_matmul_gives_each_position_what_a_single_position_call_gives() {
    let mut rng = Lcg(42);
    let (in_dim, out_dim, seq_len) = (128, 70, 4);
    let data = q5_0_weights(in_dim, out_dim, &mut rng);
    let input: Vec<f32> = (0..seq_len * in_dim).map(|_| rng.unit()).collect();
    let batched = q5_0_matmul(&input, &data, in_dim, out_dim, seq_len).expect("valid shape");
    for (s, x) in input.chunks_exact(in_dim).enumerate() {
        let single = q5_0_matmul(x, &data, in_dim, out_dim, 1).expect("valid shape");
        let at_s = &batched[s * out_dim..(s + 1) * out_dim];
        // Bit-identical, not merely close: a transposition slip moves values
        // between rows or positions and would not survive this.
        assert_eq!(
            at_s.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            single.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "position {s}"
        );
    }
}

#[test]
fn q5_0_matmul_refuses_a_buffer_that_is_not_whole_rows() {
    let mut rng = Lcg(7);
    let (in_dim, out_dim) = (64, 4);
    let data = q5_0_weights(in_dim, out_dim, &mut rng);
    let x = vec![1.0f32; in_dim];
    let short = &data[..data.len() - 1];
    let mut long = data.clone();
    long.extend_from_slice(&data[..22]);
    for (label, buf, dim) in [
        ("one byte short", short, in_dim),
        ("one block over", long.as_slice(), in_dim),
        ("in_dim not a multiple of 32", data.as_slice(), 48),
    ] {
        let input = vec![1.0f32; dim];
        match q5_0_matmul(&input, buf, dim, out_dim, 1) {
            Err(RealizarError::InvalidShape { reason }) => {
                assert!(reason.contains("Q5_0"), "{label}: {reason}");
            },
            other => panic!("{label}: expected InvalidShape, got {other:?}"),
        }
    }
    assert!(q5_0_matmul(&x, &data, in_dim, out_dim, 1).is_ok());
}
