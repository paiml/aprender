//! #2880 measurement: on an AVX-512 VNNI host, is the VNNI 4-row Q4K×Q8K kernel
//! faster than the lean AVX2 row kernel that `fused_q4k_q8k_parallel_matvec_into`
//! returns through first? Single-threaded, interleaved, min-of-N.
//! Run: cargo test -p aprender-serve --release --lib vnni_reach_bench -- --ignored --nocapture

use super::fused_k::{
    fused_q4k_q8k_dot_4rows_avx512vnni, ggml_style_q4k_q8k_dot_avx2_raw, precompute_q8k_bsums_i16,
};
use super::{quantize_activations_q8k_into, QK_K};
use std::time::Instant;

const SB: usize = 144;

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *seed >> 33
}

fn weights(rows: usize, nsb: usize, seed: &mut u64) -> Vec<u8> {
    let mut w = vec![0u8; rows * nsb * SB];
    for b in w.chunks_mut(SB) {
        for x in b.iter_mut() {
            *x = lcg(seed) as u8;
        }
        // d, dmin: small finite f16 (0x2000 ~ 0.0078 with a random low mantissa)
        let d = 0x2000u16 | (lcg(seed) as u16 & 0x03ff);
        let m = 0x1c00u16 | (lcg(seed) as u16 & 0x03ff);
        b[0..2].copy_from_slice(&d.to_le_bytes());
        b[2..4].copy_from_slice(&m.to_le_bytes());
    }
    w
}

#[test]
#[ignore = "perf measurement for #2880; needs avx512vnni"]
fn vnni_reach_bench() {
    if !(is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vnni")
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma"))
    {
        eprintln!("SKIP: host lacks avx512vnni");
        return;
    }
    let mut seed = 0x2880u64;
    for &(in_dim, out_dim) in &[
        (2048usize, 2048usize),
        (2048, 6144),
        (6144, 2048),
        (4096, 4096),
    ] {
        let nsb = in_dim / QK_K;
        let bpr = nsb * SB;
        let w = weights(out_dim, nsb, &mut seed);
        let act: Vec<f32> = (0..in_dim)
            .map(|_| (lcg(&mut seed) as f32 / 2e9) - 1.0)
            .collect();
        let mut sc = vec![0f32; nsb];
        let mut qq = vec![0i8; in_dim];
        quantize_activations_q8k_into(&act, &mut sc, &mut qq).expect("q8k");
        // SAFETY: avx2 verified above; qq holds nsb*256 quants.
        let bs = unsafe { precompute_q8k_bsums_i16(&qq, nsb) };
        let mut lean = vec![0f32; out_dim];
        let mut vnni = vec![0f32; out_dim];
        let (mut t_lean, mut t_vnni) = (f64::MAX, f64::MAX);
        for _ in 0..40 {
            let t = Instant::now();
            for (r, o) in lean.iter_mut().enumerate() {
                // SAFETY: avx2+fma verified; row r < out_dim is inside w, sc/qq/bs hold nsb blocks.
                *o = unsafe {
                    ggml_style_q4k_q8k_dot_avx2_raw(
                        w.as_ptr().add(r * bpr),
                        sc.as_ptr(),
                        qq.as_ptr(),
                        bs.as_ptr(),
                        nsb,
                    )
                };
            }
            t_lean = t_lean.min(t.elapsed().as_secs_f64());
            let t = Instant::now();
            for r in (0..out_dim).step_by(4) {
                let p = |k: usize| w.as_ptr().wrapping_add((r + k) * bpr);
                // SAFETY: avx512f/bw/vnni verified; out_dim % 4 == 0 so rows r..r+4 are inside w.
                let o = unsafe {
                    fused_q4k_q8k_dot_4rows_avx512vnni([p(0), p(1), p(2), p(3)], bpr, &sc, &qq)
                };
                vnni[r..r + 4].copy_from_slice(&o);
            }
            t_vnni = t_vnni.min(t.elapsed().as_secs_f64());
        }
        let max_rel = lean
            .iter()
            .zip(&vnni)
            .map(|(a, b)| (a - b).abs() / a.abs().max(1e-3))
            .fold(0f32, f32::max);
        println!(
            "vnni_reach {in_dim}x{out_dim}: lean_avx2 {:.1}us  vnni_4row {:.1}us  vnni/lean {:.3}  max_rel_diff {:.2e}",
            t_lean * 1e6,
            t_vnni * 1e6,
            t_vnni / t_lean,
            max_rel
        );
    }
}
