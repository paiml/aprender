//! FALSIFY-LORA_GRADIENT_FLOW_V1_004 (0.72 R15a C1): extracting `lora_backward` changes no
//! number.
//!
//! The reference, `inline_316dee2cd4`, makes the same five GEMMs and the same add, in the same
//! order, that `backward_nf4_attention` ran inline for Q (and, with the KV width, for V) at
//! 316dee2cd4. Both run on one device on the same inputs, with B nonzero as it is from step 2:
//! B starts at zero, so at step 1 dA and the adapter's share of dX are zero on both sides. dA,
//! dB and dX must agree bit for bit, and both must match an f64 CPU replay of the chain rule.
//! The helper runs at scale 1, where its two scale steps (K44) multiply by exactly 1.
//! The helper's scratch starts as NaN, so a helper that read its scratch before writing it
//! would fail. Child module of `cuda_block` so it can call the private helper.
//!
//! FALSIFY-LORA_GRADIENT_FLOW_V1_006 (K44): at scale alpha/rank ≠ 1 the helpers compute
//! `y += scale·(x·A)·B` and its gradients, and the NF4 block keeps B unscaled on the device
//! while its download and upload keep the alpha/rank·B form that callers and checkpoints use.

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
        // B nonzero, as it is from step 2.
        let b = pseudo(ru * nu, 3);
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
            1.0,
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

/// `k·v` elementwise, plus `base` when given.
fn scaled(k: f64, v: &[f64], base: Option<&[f32]>) -> Vec<f64> {
    let base = |i: usize| base.map_or(0.0, |b| f64::from(b[i]));
    v.iter().enumerate().map(|(i, &e)| base(i) + k * e).collect()
}

/// FALSIFY-LORA_GRADIENT_FLOW_V1_006 (K44): the helpers apply alpha/rank, and `b` is B itself.
///
/// For `y = y0 + scale·(x·A)·B` the gradients are `dB = scale·(x·A)ᵀ·dy`,
/// `dA = scale·xᵀ·(dy·Bᵀ)` and the input share `scale·(dy·Bᵀ)·Aᵀ`, as on the CPU path and in
/// PEFT. Before K44 the GPU folded the scale into its copy of B, so its dB was that of the
/// folded tensor and alpha cancelled out of training. Scale 2 is alpha = 2·rank, the default;
/// 0.75 is not a power of two.
#[test]
#[ignore = "needs a CUDA device; hold the GPU lock around the test binary"]
fn falsify_lora_gradient_flow_v1_006_helpers_apply_alpha_over_rank() {
    let trainer = CudaTrainer::new().expect("CUDA trainer");
    let stream = trainer.stream();
    let (s, h, r, n) = (7u32, 32u32, 8u32, 16u32);
    let (su, hu, ru, nu) = (s as usize, h as usize, r as usize, n as usize);
    let (x, a, b, dy) =
        (pseudo(su * hu, 1), pseudo(hu * ru, 2), pseudo(ru * nu, 3), pseudo(su * nu, 4));
    let (dx0, y0) = (pseudo(su * hu, 5), pseudo(su * nu, 6));
    let up = |v: &[f32]| trainer.upload(v).expect("upload");
    let down = |g: &GpuBuffer<f32>| trainer.download(g).expect("download");
    let (gx, ga, gb, gdy) = (up(&x), up(&a), up(&b), up(&dy));

    let (x64, a64, b64, dy64) = (widen(&x), widen(&a), widen(&b), widen(&dy));
    let xa = matmul64(&x64, &a64, su, hu, ru, false, false);
    let dy_bt = matmul64(&dy64, &b64, su, nu, ru, false, true);
    let xab = matmul64(&xa, &b64, su, ru, nu, false, false);
    let xa_t_dy = matmul64(&xa, &dy64, ru, su, nu, true, false);
    let x_t_dy_bt = matmul64(&x64, &dy_bt, hu, su, ru, true, false);
    let share = matmul64(&dy_bt, &a64, su, ru, hu, false, true);

    for scale in [2.0f32, 0.75] {
        let k = f64::from(scale);
        let mut gy = up(&y0);
        let (mut inter, mut temp) = (up(&vec![f32::NAN; su * ru]), up(&vec![f32::NAN; su * nu]));
        lora_forward(&gx, &ga, &gb, &mut gy, &mut inter, &mut temp, scale, s, h, r, n, stream)
            .expect("lora_forward");

        let (mut da, mut db, mut dx) = (up(&vec![0.0; hu * ru]), up(&vec![0.0; ru * nu]), up(&dx0));
        let (mut inter, mut temp) = (up(&vec![f32::NAN; su * ru]), up(&vec![f32::NAN; su * hu]));
        lora_backward(
            &gx, &ga, &gb, &gdy, &mut da, &mut db, &mut dx, &mut inter, &mut temp, scale, s, h, r,
            n, stream,
        )
        .expect("lora_backward");
        trainer.synchronize().expect("sync");

        assert_close(&down(&gy), &scaled(k, &xab, Some(&y0[..])), &format!("y, scale {scale}"));
        assert_close(&down(&db), &scaled(k, &xa_t_dy, None), &format!("dB, scale {scale}"));
        assert_close(&down(&da), &scaled(k, &x_t_dy_bt, None), &format!("dA, scale {scale}"));
        assert_close(&down(&dx), &scaled(k, &share, Some(&dx0[..])), &format!("dX, scale {scale}"));
    }
}

/// FALSIFY-LORA_GRADIENT_FLOW_V1_006 (K44), block level: the NF4 block stores B as given.
/// Download returns `scale·B`, the form every caller divides back out and every checkpoint
/// holds, and upload divides it out again, so a round trip leaves B on the device. The forward
/// then adds `scale·(norm1_out·A)·B` to V. V is checked because attention leaves it unrotated
/// (it rotates Q in place); Q runs the same helper with the same scale. At scale 2 the scale
/// steps are exact, so the download checks compare bits.
#[test]
#[ignore = "needs a CUDA device; hold the GPU lock around the test binary"]
fn falsify_lora_gradient_flow_v1_006_nf4_block_keeps_b_unscaled() {
    let config = TransformerConfig::tiny();
    let (hid, qd, inter) = (config.hidden_size, config.q_dim(), config.intermediate_size);
    let kvd = config.num_kv_heads * config.head_dim();
    let (seq, rank, scale) = (5usize, 8usize, 2.0f32);
    let trainer = CudaTrainer::new().expect("CUDA trainer");
    let stream = trainer.stream();
    let ctx = Arc::clone(trainer.context());

    let w =
        |len: usize, seed: u64| -> Vec<f32> { pseudo(len, seed).iter().map(|v| 0.1 * v).collect() };
    let norm = vec![1.0f32; hid];
    let (w_q, w_k, w_v, w_o) =
        (w(qd * hid, 10), w(kvd * hid, 11), w(kvd * hid, 12), w(hid * qd, 13));
    let (w_gate, w_up, w_down) = (w(inter * hid, 14), w(inter * hid, 15), w(hid * inter, 16));
    let (a_q, b_q) = (pseudo(hid * rank, 17), pseudo(rank * qd, 18));
    let (a_v, b_v) = (pseudo(hid * rank, 19), pseudo(rank * kvd, 20));
    let block = |lora: bool| {
        let q_lora = lora.then_some((&a_q[..], &b_q[..]));
        let v_lora = lora.then_some((&a_v[..], &b_v[..]));
        CudaNf4TransformerBlock::new(
            &config,
            0,
            Arc::clone(&ctx),
            &norm,
            &norm,
            &w_q,
            &w_k,
            &w_v,
            &w_o,
            &w_gate,
            &w_up,
            &w_down,
            seq,
            q_lora,
            v_lora,
            scale,
            rank,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("NF4 block")
    };
    let (mut with, without) = (block(true), block(false));

    let times = |b: &[f32]| -> Vec<f32> { b.iter().map(|v| scale * v).collect() };
    let (got_a_q, got_b_q, got_a_v, got_b_v) = with.download_lora_weights().expect("download");
    assert_same_bits(&got_b_q, &times(&b_q), "B_q download");
    assert_same_bits(&got_b_v, &times(&b_v), "B_v download");
    with.upload_lora_weights(&got_a_q, &got_b_q, &got_a_v, &got_b_v).expect("upload LoRA");
    let (_, again_b_q, _, again_b_v) = with.download_lora_weights().expect("download again");
    assert_same_bits(&again_b_q, &got_b_q, "B_q after a round trip");
    assert_same_bits(&again_b_v, &got_b_v, "B_v after a round trip");

    let x = trainer.upload(&pseudo(seq * hid, 21)).expect("upload");
    let run = |blk: &CudaNf4TransformerBlock| {
        let mut scratch = CudaBlockScratch::new(&config, seq, &ctx, rank).expect("scratch");
        scratch.zero_forward_buffers(stream);
        let mut out = trainer.zeros(seq * hid).expect("out");
        blk.forward(&x, &mut out, seq, stream, &mut scratch).expect("forward");
        trainer.synchronize().expect("sync");
        let take = |g: &GpuBuffer<f32>, len: usize| {
            let mut v = trainer.download(g).expect("download");
            v.truncate(len);
            v
        };
        (take(&scratch.norm1_out, seq * hid), take(&scratch.v, seq * kvd))
    };
    let (norm1, v_with) = run(&with);
    let (_, v_base) = run(&without);

    let xa = matmul64(&widen(&norm1), &widen(&a_v), seq, hid, rank, false, false);
    let want =
        scaled(f64::from(scale), &matmul64(&xa, &widen(&b_v), seq, rank, kvd, false, false), None);
    let got: Vec<f32> = v_with.iter().zip(&v_base).map(|(lora, base)| lora - base).collect();
    assert_close(&got, &want, "the V adapter's share of the projection");
}
