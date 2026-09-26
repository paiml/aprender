//! #2880: the aarch64 NEON Q4K×Q8K dot against the scalar reference.
//! Bench: cargo test -p aprender-serve --release --lib neon_q4k_q8k_bench -- --ignored --nocapture

use super::fused_k::{fused_q4k_q8k_dot, fused_q4k_q8k_dot_neon, fused_q4k_q8k_dot_simd};
use super::{quantize_activations_q8k_into, QK_K};
use std::time::Instant;

const SB: usize = 144;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *seed >> 33
}

fn weights(nsb: usize, seed: &mut u64) -> Vec<u8> {
    let mut w = vec![0u8; nsb * SB];
    for b in w.chunks_mut(SB) {
        for x in b.iter_mut() {
            *x = lcg(seed) as u8;
        }
        let d = 0x2000u16 | (lcg(seed) as u16 & 0x03ff);
        let m = 0x1c00u16 | (lcg(seed) as u16 & 0x03ff);
        b[0..2].copy_from_slice(&d.to_le_bytes());
        b[2..4].copy_from_slice(&m.to_le_bytes());
    }
    w
}

fn activations(n: usize, seed: &mut u64) -> (Vec<f32>, Vec<i8>) {
    let act: Vec<f32> = (0..n).map(|_| (lcg(seed) as f32 / 2e9) - 1.0).collect();
    let mut sc = vec![0f32; n / QK_K];
    let mut qq = vec![0i8; n];
    quantize_activations_q8k_into(&act, &mut sc, &mut qq).expect("q8k");
    (sc, qq)
}

#[test]
fn neon_matches_scalar_bitwise() {
    let mut seed = 0x2880_0002u64;
    for nsb in [1usize, 2, 7, 16, 24] {
        for _ in 0..8 {
            let w = weights(nsb, &mut seed);
            let (sc, mut qq) = activations(nsb * QK_K, &mut seed);
            // Extremes: -128 and 127 must survive the i16 products and sums.
            qq[0] = -128;
            qq[1] = 127;
            let want = fused_q4k_q8k_dot(&w, &sc, &qq).expect("scalar");
            let got = fused_q4k_q8k_dot_neon(&w, &sc, &qq).expect("neon");
            let simd = fused_q4k_q8k_dot_simd(&w, &sc, &qq).expect("simd");
            assert_eq!(got.to_bits(), want.to_bits(), "nsb={nsb}: {got} vs {want}");
            assert_eq!(simd.to_bits(), want.to_bits(), "dispatch must reach NEON");
        }
    }
}

#[test]
fn neon_rejects_what_scalar_rejects() {
    let mut seed = 7u64;
    let w = weights(2, &mut seed);
    let (sc, qq) = activations(2 * QK_K, &mut seed);
    assert!(fused_q4k_q8k_dot_neon(&w[..=SB], &sc, &qq).is_err());
    assert!(fused_q4k_q8k_dot_neon(&w, &sc[..1], &qq).is_err());
    assert!(fused_q4k_q8k_dot_neon(&w, &sc, &qq[..QK_K]).is_err());
}

#[test]
#[ignore = "perf measurement for #2880"]
fn neon_q4k_q8k_bench() {
    let mut seed = 0x2880u64;
    for &(in_dim, out_dim) in &[
        (2048usize, 2048usize),
        (2048, 6144),
        (6144, 2048),
        (4096, 4096),
    ] {
        let nsb = in_dim / QK_K;
        let bpr = nsb * SB;
        let w: Vec<u8> = (0..out_dim).flat_map(|_| weights(nsb, &mut seed)).collect();
        let (sc, qq) = activations(in_dim, &mut seed);
        let (mut t_s, mut t_n) = (f64::MAX, f64::MAX);
        let mut acc = 0f32;
        for _ in 0..20 {
            let t = Instant::now();
            for r in 0..out_dim {
                acc += fused_q4k_q8k_dot(&w[r * bpr..(r + 1) * bpr], &sc, &qq).expect("scalar");
            }
            t_s = t_s.min(t.elapsed().as_secs_f64());
            let t = Instant::now();
            for r in 0..out_dim {
                acc -= fused_q4k_q8k_dot_neon(&w[r * bpr..(r + 1) * bpr], &sc, &qq).expect("neon");
            }
            t_n = t_n.min(t.elapsed().as_secs_f64());
        }
        println!(
            "neon_q4k {in_dim}x{out_dim}: scalar {:.1}us  neon {:.1}us  speedup {:.2}x  (residual {acc:e})",
            t_s * 1e6,
            t_n * 1e6,
            t_s / t_n
        );
    }
}
