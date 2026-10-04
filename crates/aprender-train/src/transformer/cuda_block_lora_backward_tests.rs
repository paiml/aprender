//! FALSIFY-LORA_GRADIENT_FLOW_V1_004 (0.72 R15a C1): extracting `lora_backward` changes no
//! number.
//!
//! The reference, `inline_316dee2cd4`, makes the same five GEMMs and the same add, in the same
//! order, that `backward_nf4_attention` ran inline for Q (and, with the KV width, for V) at
//! 316dee2cd4. Both run on one device on the same inputs, with B nonzero as it is from step 2:
//! B starts at zero, so at step 1 dA and the adapter's share of dX are zero on both sides. dA,
//! dB and dX must agree bit for bit, and both must match an f64 CPU replay of the chain rule.
//! The helper's scratch starts as NaN, so a helper that read its scratch before writing it
//! would fail. Child module of `cuda_block` so it can call the private helper.

use super::*;
use crate::autograd::cuda_training::CudaTrainer;

/// The 316dee2cd4 inline sequence: the same calls in the same order, with neutral names.
#[allow(clippy::too_many_arguments)]
fn inline_316dee2cd4(
    norm1_out: &GpuBuffer<f32>,
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    grad_out: &GpuBuffer<f32>,
    grad_a: &mut GpuBuffer<f32>,
    grad_b: &mut GpuBuffer<f32>,
    grad_norm1: &mut GpuBuffer<f32>,
    lora_inter: &mut GpuBuffer<f32>,
    lora_temp: &mut GpuBuffer<f32>,
    (s, h, r, n): (u32, u32, u32, u32),
    stream: &CudaStream,
) -> Result<()> {
    gemm_forward(norm1_out, a, lora_inter, s, h, r, stream)?;
    gemm_backward_b(lora_inter, grad_out, grad_b, s, r, n, stream)?;
    gemm_backward_a(grad_out, b, lora_inter, s, n, r, stream)?;
    gemm_backward_b(norm1_out, lora_inter, grad_a, s, h, r, stream)?;
    gemm_backward_a(lora_inter, a, lora_temp, s, r, h, stream)?;
    cuda_add_inplace(grad_norm1, lora_temp, s as usize * h as usize, stream)
}

/// Deterministic values in [-1, 1) that differ per `seed`.
fn pseudo(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

/// `c[m, n] = a[m, k] · b[k, n]` in f64; `ta`/`tb` read `a`/`b` transposed from storage.
fn matmul64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, ta: bool, tb: bool) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = (0..k)
                .map(|p| {
                    let av = if ta { a[p * m + i] } else { a[i * k + p] };
                    let bv = if tb { b[j * k + p] } else { b[p * n + j] };
                    av * bv
                })
                .sum();
        }
    }
    c
}

fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

fn assert_same_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let diff: Vec<usize> =
        (0..got.len()).filter(|&i| got[i].to_bits() != want[i].to_bits()).collect();
    assert!(
        diff.is_empty(),
        "{what}: {} of {} values differ in their bits; first at {}: {} vs {}",
        diff.len(),
        got.len(),
        diff[0],
        got[diff[0]],
        want[diff[0]]
    );
}

fn assert_close(got: &[f32], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    let worst = got.iter().zip(want).map(|(&g, &w)| (f64::from(g) - w).abs()).fold(0.0, f64::max);
    assert!(worst <= 1e-4 * scale, "{what}: max |GPU - f64| = {worst:e}, scale {scale:e}");
}

#[test]
#[ignore = "needs a CUDA device; hold the GPU lock around the test binary"]
fn falsify_lora_gradient_flow_v1_004_lora_backward_is_the_inline_path() {
    let trainer = CudaTrainer::new().expect("CUDA trainer");
    let stream = trainer.stream();
    let (s, h, r) = (7u32, 32u32, 8u32);
    // A Q-shaped projection, then a V-shaped one (GQA: fewer output columns).
    for n in [32u32, 16] {
        let (su, hu, ru, nu) = (s as usize, h as usize, r as usize, n as usize);
        let x = pseudo(su * hu, 1);
        let a = pseudo(hu * ru, 2);
        // B as the upload leaves it at alpha = 2·rank (times s = 2), nonzero as from step 2.
        let b: Vec<f32> = pseudo(ru * nu, 3).iter().map(|v| 2.0 * v).collect();
        let dy = pseudo(su * nu, 4);
        let dx0 = pseudo(su * hu, 5);
        let up = |v: &[f32]| trainer.upload(v).expect("upload");
        let (gx, ga, gb, gdy) = (up(&x), up(&a), up(&b), up(&dy));

        let mut want_da = up(&vec![0.0; hu * ru]);
        let mut want_db = up(&vec![0.0; ru * nu]);
        let mut want_dx = up(&dx0);
        let mut want_inter = up(&vec![0.0; su * ru]);
        let mut want_temp = up(&vec![0.0; su * hu]);
        inline_316dee2cd4(
            &gx,
            &ga,
            &gb,
            &gdy,
            &mut want_da,
            &mut want_db,
            &mut want_dx,
            &mut want_inter,
            &mut want_temp,
            (s, h, r, n),
            stream,
        )
        .expect("inline path");

        let mut got_da = up(&vec![0.0; hu * ru]);
        let mut got_db = up(&vec![0.0; ru * nu]);
        let mut got_dx = up(&dx0);
        let mut got_inter = up(&vec![f32::NAN; su * ru]);
        let mut got_temp = up(&vec![f32::NAN; su * hu]);
        lora_backward(
            &gx,
            &ga,
            &gb,
            &gdy,
            &mut got_da,
            &mut got_db,
            &mut got_dx,
            &mut got_inter,
            &mut got_temp,
            s,
            h,
            r,
            n,
            stream,
        )
        .expect("lora_backward");
        trainer.synchronize().expect("sync");

        let down = |g: &GpuBuffer<f32>| trainer.download(g).expect("download");
        let (got_da, got_db, got_dx) = (down(&got_da), down(&got_db), down(&got_dx));
        assert_same_bits(&got_da, &down(&want_da), &format!("dA, n = {n}"));
        assert_same_bits(&got_db, &down(&want_db), &format!("dB, n = {n}"));
        assert_same_bits(&got_dx, &down(&want_dx), &format!("dX, n = {n}"));

        // Both sides compute the chain rule, not merely the same thing.
        let (x64, a64, b64, dy64) = (widen(&x), widen(&a), widen(&b), widen(&dy));
        let inter = matmul64(&x64, &a64, su, hu, ru, false, false);
        let db = matmul64(&inter, &dy64, ru, su, nu, true, false);
        let d_inter = matmul64(&dy64, &b64, su, nu, ru, false, true);
        let da = matmul64(&x64, &d_inter, hu, su, ru, true, false);
        let share = matmul64(&d_inter, &a64, su, ru, hu, false, true);
        let dx: Vec<f64> = dx0.iter().zip(&share).map(|(&d, &l)| f64::from(d) + l).collect();
        for (what, v) in [("dA", &da), ("dB", &db), ("dX share", &share)] {
            assert!(v.iter().any(|e| e.abs() > 1e-3), "{what} is ~0, so the check has no power");
        }
        assert_close(&got_da, &da, &format!("dA vs f64, n = {n}"));
        assert_close(&got_db, &db, &format!("dB vs f64, n = {n}"));
        assert_close(&got_dx, &dx, &format!("dX vs f64, n = {n}"));
    }
}
