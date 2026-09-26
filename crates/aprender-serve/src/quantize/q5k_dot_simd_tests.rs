//! #2880 slice 5: the Q5_K × f32 SIMD kernels (AVX2+FMA, NEON) against an f64
//! reference built from `dequantize_q5_k`, and the dispatcher reaching them.
//! Bench: cargo test -p aprender-serve --release --lib q5k_simd_bench -- --ignored --nocapture

use super::dequantize_q5_k;
use super::fused_q5k_q6k::{fused_q5k_dot, fused_q5k_dot_simd};
use super::QK_K;
use std::time::Instant;

const SB5: usize = 176;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *seed >> 33
}

fn q5k_weights(nsb: usize, seed: &mut u64) -> Vec<u8> {
    let mut w = vec![0u8; nsb * SB5];
    for b in w.chunks_mut(SB5) {
        for x in b.iter_mut() {
            *x = lcg(seed) as u8;
        }
        // d, dmin: small finite f16 with a random low mantissa
        let d = 0x2000u16 | (lcg(seed) as u16 & 0x03ff);
        let m = 0x1c00u16 | (lcg(seed) as u16 & 0x03ff);
        b[0..2].copy_from_slice(&d.to_le_bytes());
        b[2..4].copy_from_slice(&m.to_le_bytes());
    }
    w
}

fn activations(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n).map(|_| (lcg(seed) as f32 / 2e9) - 1.0).collect()
}

/// f64 dot of the dequantized weights, and Σ|w·a| as the rounding scale.
fn q5k_reference(w: &[u8], act: &[f32]) -> (f64, f64) {
    let deq = dequantize_q5_k(w).expect("dequant");
    deq.iter().zip(act).fold((0.0, 0.0), |(s, m), (&x, &a)| {
        let t = f64::from(x) * f64::from(a);
        (s + t, m + t.abs())
    })
}

/// The arch kernel this host dispatches to, called directly.
fn arch_kernel(w: &[u8], act: &[f32]) -> crate::error::Result<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        assert!(
            is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
            "test host lacks avx2+fma"
        );
        // SAFETY: avx2 and fma verified above.
        unsafe { super::fused_q5k_q6k::fused_q5k_dot_avx2(w, act) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        super::fused_q5k_q6k::fused_q5k_dot_neon(w, act)
    }
}

#[test]
fn q5k_simd_matches_f64_reference() {
    let mut seed = 0x2880_0005u64;
    for nsb in [1usize, 2, 7, 16, 24] {
        for _ in 0..8 {
            let w = q5k_weights(nsb, &mut seed);
            let act = activations(nsb * QK_K, &mut seed);
            let (want, mag) = q5k_reference(&w, &act);
            let tol = 1e-5 * mag;
            let scalar = f64::from(fused_q5k_dot(&w, &act).expect("scalar"));
            let simd = f64::from(arch_kernel(&w, &act).expect("simd"));
            assert!(
                (scalar - want).abs() <= tol,
                "scalar off: {scalar} vs {want}"
            );
            assert!(
                (simd - want).abs() <= tol,
                "nsb={nsb}: simd {simd} vs {want} (tol {tol})"
            );
            assert_eq!(
                fused_q5k_dot_simd(&w, &act).expect("dispatch").to_bits(),
                arch_kernel(&w, &act).expect("simd").to_bits(),
                "dispatch must reach the arch kernel"
            );
        }
    }
}

/// One weight at a time: a wrong nibble/qh-bit/scale index moves the result by
/// a whole quantum, far outside the rounding tolerance.
#[test]
fn q5k_simd_single_value_probe() {
    let mut seed = 0x2880_0055u64;
    let w = q5k_weights(1, &mut seed);
    for i in 0..QK_K {
        let mut act = vec![0f32; QK_K];
        act[i] = 1.0;
        let (want, _) = q5k_reference(&w, &act);
        let got = f64::from(arch_kernel(&w, &act).expect("simd"));
        assert!(
            (got - want).abs() <= 1e-5 * want.abs().max(1e-3),
            "value {i}: simd {got} vs {want}"
        );
    }
}

#[test]
fn q5k_simd_rejects_what_scalar_rejects() {
    let mut seed = 9u64;
    let w = q5k_weights(2, &mut seed);
    let act = vec![0.5f32; 2 * QK_K];
    assert!(arch_kernel(&w[..=SB5], &act).is_err());
    assert!(arch_kernel(&w, &act[..QK_K]).is_err());
    assert!(fused_q5k_dot_simd(&w, &act[..QK_K]).is_err());
}

#[test]
#[ignore = "perf measurement for #2880"]
fn q5k_simd_bench() {
    let mut seed = 0x2880u64;
    for &(in_dim, out_dim) in &[(2048usize, 2048usize), (2048, 6144), (6144, 2048)] {
        let nsb = in_dim / QK_K;
        let w = q5k_weights(out_dim * nsb, &mut seed);
        let act = activations(in_dim, &mut seed);
        let bpr = nsb * SB5;
        let (mut t_s, mut t_v) = (f64::MAX, f64::MAX);
        let mut sink = 0f32;
        for _ in 0..20 {
            let t = Instant::now();
            for r in 0..out_dim {
                sink += fused_q5k_dot(&w[r * bpr..(r + 1) * bpr], &act).expect("scalar");
            }
            t_s = t_s.min(t.elapsed().as_secs_f64());
            let t = Instant::now();
            for r in 0..out_dim {
                sink += fused_q5k_dot_simd(&w[r * bpr..(r + 1) * bpr], &act).expect("simd");
            }
            t_v = t_v.min(t.elapsed().as_secs_f64());
        }
        println!(
            "q5k_simd {} {in_dim}x{out_dim}: scalar {:.1}us  simd {:.1}us  speedup {:.2}x  (sink {sink:.1})",
            std::env::consts::ARCH,
            t_s * 1e6,
            t_v * 1e6,
            t_s / t_v
        );
    }
}
