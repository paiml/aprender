//! FALSIFY-LORA_TARGET_SELECTION_V1_005 (0.72 R15a C2c): the NF4 block, the gradient
//! workspace, the optimizer state and the gradient clip hold one A and one B per LoRA target.
//!
//! An all_linear block takes q_proj and v_proj through `new` and the other five targets
//! through `add_lora_adapter`. The scale is 2, so dividing B by it on upload and multiplying
//! by it on download is exact and the round trips compare bits. Every 005 test needs a CUDA
//! device. Child module of `cuda_block` so it can read the private stores.
//!
//! FALSIFY-LORA_GRADIENT_FLOW_V1_007 (0.72 R15a C3): the block's forward and backward use
//! every adapter it holds. Its two device tests check each adapter's gradients against
//! central differences of the forward, and that adapters with B = 0 change no number of a
//! q_proj, v_proj block. Its third test, the LoRA temp's size, needs no device.

use super::*;
use crate::autograd::cuda_training::CudaTrainer;

const RANK: usize = 4;
const SCALE: f32 = 2.0;

/// Deterministic values in ±[0.1, 1.0], never zero, so one Adam step moves every weight.
fn vals(len: usize, seed: u64) -> Vec<f32> {
    (0..len as u64)
        .map(|i| {
            let x = i.wrapping_mul(2_654_435_761).wrapping_add(seed.wrapping_mul(40_503)) % 1000;
            let mag = 0.1 + 0.9 * x as f32 / 1000.0;
            if (i + seed) % 2 == 0 {
                mag
            } else {
                -mag
            }
        })
        .collect()
}

/// A distinct (A, B) for `target`, shaped by `LoraTarget::dims`.
fn adapter(config: &TransformerConfig, target: LoraTarget, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let (d_out, d_in) = target.dims(config);
    let seed = seed + 10 * target as u64;
    (vals(d_in * RANK, seed), vals(RANK * d_out, seed + 1))
}

/// An NF4 block for layer 0 with LoRA on `targets`: q_proj and v_proj through `new`, the
/// others through `add_lora_adapter`.
fn nf4_block(
    trainer: &CudaTrainer,
    config: &TransformerConfig,
    targets: &[LoraTarget],
) -> CudaNf4TransformerBlock {
    let (hid, qd, inter) = (config.hidden_size, config.q_dim(), config.intermediate_size);
    let kvd = config.num_kv_heads * config.head_dim();
    let w =
        |len: usize, seed: u64| -> Vec<f32> { vals(len, seed).iter().map(|v| 0.1 * v).collect() };
    let norm = vec![1.0f32; hid];
    let (a_q, b_q) = adapter(config, LoraTarget::Q, 100);
    let (a_v, b_v) = adapter(config, LoraTarget::V, 100);
    let has = |t: LoraTarget| targets.contains(&t);
    let mut block = CudaNf4TransformerBlock::new(
        config,
        0,
        Arc::clone(trainer.context()),
        &norm,
        &norm,
        &w(qd * hid, 10),
        &w(kvd * hid, 11),
        &w(kvd * hid, 12),
        &w(hid * qd, 13),
        &w(inter * hid, 14),
        &w(inter * hid, 15),
        &w(hid * inter, 16),
        8,
        has(LoraTarget::Q).then_some((&a_q[..], &b_q[..])),
        has(LoraTarget::V).then_some((&a_v[..], &b_v[..])),
        SCALE,
        RANK,
        None,
        None,
        None,
        None,
        None,
    )
    .expect("NF4 block");
    for &t in targets.iter().filter(|t| !matches!(t, LoraTarget::Q | LoraTarget::V)) {
        let (a, b) = adapter(config, t, 100);
        block.add_lora_adapter(t, &a, &b).expect("add adapter");
    }
    block
}

fn workspace(trainer: &CudaTrainer, config: &TransformerConfig) -> CudaLoraGradWorkspace {
    CudaLoraGradWorkspace::new_for_targets(trainer.context(), config, RANK, &LoraTarget::ALL)
        .expect("all_linear workspace")
}

/// Writes distinct values into every gradient buffer, in clip order (each target's A then
/// B in slot order, then the two norm gradients), and returns them in that order.
fn fill_gradients(ws: &mut CudaLoraGradWorkspace) -> Vec<Vec<f32>> {
    let bufs = ws
        .grad_lora
        .iter_mut()
        .flat_map(|p| [&mut p.a, &mut p.b])
        .chain([&mut ws.grad_input_norm, &mut ws.grad_post_attn_norm]);
    let mut written = Vec::new();
    for (i, buf) in bufs.enumerate() {
        let data = vals(buf.len(), 500 + i as u64);
        buf.copy_from_host(&data).expect("write gradient");
        written.push(data);
    }
    written
}

#[test]
#[ignore = "needs a CUDA device; hold the GPU lock around the test binary"]
fn falsify_lora_target_selection_v1_005_all_linear_stores_hold_every_target() {
    let config = TransformerConfig::tiny();
    let trainer = CudaTrainer::new().expect("CUDA trainer");
    let block = nf4_block(&trainer, &config, &LoraTarget::ALL);
    let ws = workspace(&trainer, &config);
    let state = block.init_lora_optimizer_state().expect("optimizer state");
    for (what, pairs) in [
        ("block adapters", &block.lora),
        ("gradients", &ws.grad_lora),
        ("first moments", &state.m_lora),
        ("second moments", &state.v_lora),
    ] {
        assert_eq!(LoraPair::targets(pairs), LoraTarget::ALL, "{what}: one pair per target");
        for p in pairs {
            let (d_out, d_in) = p.target.dims(&config);
            assert_eq!(
                (p.a.len(), p.b.len()),
                (d_in * RANK, RANK * d_out),
                "{what}: {:?} lengths",
                p.target
            );
        }
    }
}

#[test]
#[ignore = "needs a CUDA device; hold the GPU lock around the test binary"]
fn falsify_lora_target_selection_v1_005_step_moves_every_adapter() {
    let config = TransformerConfig::tiny();
    let trainer = CudaTrainer::new().expect("CUDA trainer");
    let stream = trainer.stream();
    let mut block = nf4_block(&trainer, &config, &LoraTarget::ALL);
    let mut state = block.init_lora_optimizer_state().expect("optimizer state");
    let mut ws = workspace(&trainer, &config);
    fill_gradients(&mut ws);

    let before = block.download_lora_adapters().expect("download");
    block
        .lora_optimizer_step(&mut state, 1, 1e-2, 0.9, 0.999, 1e-8, 0.0, stream, &ws)
        .expect("optimizer step");
    trainer.synchronize().expect("sync");
    let after = block.download_lora_adapters().expect("download");
    assert_eq!(after.len(), LoraTarget::ALL.len());
    for ((t, a0, b0), (t1, a1, b1)) in before.iter().zip(&after) {
        assert_eq!(t, t1, "slot order");
        for (name, x0, x1) in [("A", a0, a1), ("B", b0, b1)] {
            let moved = x0.iter().zip(x1).filter(|(u, v)| u != v).count();
            assert_eq!(moved, x0.len(), "{t:?} {name}: {moved} of {} weights moved", x0.len());
        }
    }

    // A q_proj, v_proj workspace does not match an all_linear block.
    let qv = CudaLoraGradWorkspace::new(trainer.context(), &config, RANK).expect("q/v workspace");
    let err = block
        .lora_optimizer_step(&mut state, 2, 1e-2, 0.9, 0.999, 1e-8, 0.0, stream, &qv)
        .expect_err("a workspace with other targets is refused");
    assert!(format!("{err:?}").contains("optimizer step: block targets"), "{err:?}");
}

#[test]
#[ignore = "needs a CUDA device; hold the GPU lock around the test binary"]
fn falsify_lora_target_selection_v1_005_clip_scales_every_gradient() {
    let config = TransformerConfig::tiny();
    let trainer = CudaTrainer::new().expect("CUDA trainer");
    let mut ws = workspace(&trainer, &config);
    let written = fill_gradients(&mut ws);
    assert_eq!(written.len(), 16, "an A and a B for 7 targets, then 2 norm gradients");
    let total = written.iter().flatten().map(|&v| f64::from(v).powi(2)).sum::<f64>().sqrt();
    let max_norm = (total / 4.0) as f32;

    ws.clip_gradients(max_norm, trainer.stream());
    trainer.synchronize().expect("sync");
    let want = f64::from(max_norm) / (total + 1e-6);
    let bufs = ws
        .grad_lora
        .iter()
        .flat_map(|p| [&p.a, &p.b])
        .chain([&ws.grad_input_norm, &ws.grad_post_attn_norm]);
    for (i, (buf, orig)) in bufs.zip(&written).enumerate() {
        let got = trainer.download(buf).expect("download");
        assert_eq!(got.len(), orig.len(), "buffer {i}");
        for (g, o) in got.iter().zip(orig) {
            let w = want * f64::from(*o);
            assert!((f64::from(*g) - w).abs() <= 1e-4 * w.abs() + 1e-7, "buffer {i}: {g} vs {w}");
        }
    }
}

#[test]
#[ignore = "needs a CUDA device; hold the GPU lock around the test binary"]
fn falsify_lora_target_selection_v1_005_upload_and_download_by_target() {
    let config = TransformerConfig::tiny();
    let trainer = CudaTrainer::new().expect("CUDA trainer");
    let mut block = nf4_block(&trainer, &config, &LoraTarget::ALL);
    let fresh: Vec<(LoraTarget, Vec<f32>, Vec<f32>)> = LoraTarget::ALL
        .iter()
        .map(|&t| {
            let (a, b) = adapter(&config, t, 900);
            (t, a, b)
        })
        .collect();

    // Given in reverse slot order: an adapter is placed by its target, not its position.
    let given: Vec<(LoraTarget, &[f32], &[f32])> =
        fresh.iter().rev().map(|(t, a, b)| (*t, a.as_slice(), b.as_slice())).collect();
    block.upload_lora_adapters(&given).expect("upload");
    assert_eq!(block.download_lora_adapters().expect("download"), fresh);

    // A target the block lacks, after a valid q_proj pair, refuses before writing q_proj.
    let mut qv = nf4_block(&trainer, &config, &[LoraTarget::Q, LoraTarget::V]);
    let before = qv.download_lora_adapters().expect("download");
    let (q, k) = (&fresh[0], &fresh[1]);
    assert_eq!((q.0, k.0), (LoraTarget::Q, LoraTarget::K));
    let err = qv
        .upload_lora_adapters(&[
            (q.0, q.1.as_slice(), q.2.as_slice()),
            (k.0, k.1.as_slice(), k.2.as_slice()),
        ])
        .expect_err("k_proj is not on the block");
    assert!(format!("{err:?}").contains("has no K adapter"), "{err:?}");
    let err = qv
        .upload_lora_adapters(&[(q.0, &q.1[1..], q.2.as_slice())])
        .expect_err("a short A is refused");
    assert!(format!("{err:?}").contains("size mismatch"), "{err:?}");
    assert_eq!(qv.download_lora_adapters().expect("download"), before, "nothing was written");
}

#[test]
#[ignore = "needs a CUDA device; hold the GPU lock around the test binary"]
fn falsify_lora_target_selection_v1_005_add_adapter_checks_target_and_length() {
    let config = TransformerConfig::tiny();
    let trainer = CudaTrainer::new().expect("CUDA trainer");
    let mut block = nf4_block(&trainer, &config, &[LoraTarget::Q, LoraTarget::V]);
    let (q_a, q_b) = adapter(&config, LoraTarget::Q, 300);
    let (k_a, k_b) = adapter(&config, LoraTarget::K, 300);

    let err = block.add_lora_adapter(LoraTarget::Q, &q_a, &q_b).expect_err("q_proj twice");
    assert!(format!("{err:?}").contains("already present"), "{err:?}");
    for (a, b) in [(&k_a[1..], &k_b[..]), (&k_a[..], &k_b[1..])] {
        let err = block.add_lora_adapter(LoraTarget::K, a, b).expect_err("wrong length");
        assert!(format!("{err:?}").contains("expected"), "{err:?}");
    }
    assert_eq!(LoraPair::targets(&block.lora), [LoraTarget::Q, LoraTarget::V], "unchanged");

    block.add_lora_adapter(LoraTarget::K, &k_a, &k_b).expect("k_proj");
    assert_eq!(
        LoraPair::targets(&block.lora),
        [LoraTarget::Q, LoraTarget::K, LoraTarget::V],
        "slot order"
    );

    for targets in [[LoraTarget::V, LoraTarget::Q], [LoraTarget::Q, LoraTarget::Q]] {
        let err =
            CudaLoraGradWorkspace::new_for_targets(trainer.context(), &config, RANK, &targets)
                .err()
                .expect("unsorted or repeated targets are refused");
        assert!(format!("{err:?}").contains("slot order without repeats"), "{err:?}");
    }
}

#[test]
#[ignore = "needs a CUDA device; hold the GPU lock around the test binary"]
fn falsify_lora_target_selection_v1_005_qv_four_tuple_is_the_by_target_form() {
    let config = TransformerConfig::tiny();
    let trainer = CudaTrainer::new().expect("CUDA trainer");
    let block = nf4_block(&trainer, &config, &[LoraTarget::Q, LoraTarget::V]);
    let (a_q, b_q, a_v, b_v) = block.download_lora_weights().expect("four-tuple download");
    assert_eq!(
        block.download_lora_adapters().expect("download by target"),
        vec![(LoraTarget::Q, a_q, b_q), (LoraTarget::V, a_v, b_v)]
    );
}

// ── FALSIFY-LORA_GRADIENT_FLOW_V1_007 (0.72 R15a C3) ────────────────────────────────────

const SEQ: usize = 4;
const EPS: f32 = 1e-2;

/// A scratch whose LoRA temp fits every target.
fn all_targets_scratch(trainer: &CudaTrainer, config: &TransformerConfig) -> CudaBlockScratch {
    CudaBlockScratch::new_for_targets(config, SEQ, trainer.context(), RANK, &LoraTarget::ALL)
        .expect("all_linear scratch")
}

/// L = Σ g·y for the block output y, the loss whose gradient with respect to y is g.
fn loss(
    trainer: &CudaTrainer,
    block: &CudaNf4TransformerBlock,
    scratch: &mut CudaBlockScratch,
    x: &GpuBuffer<f32>,
    g: &[f32],
) -> f64 {
    let stream = trainer.stream();
    scratch.zero_forward_buffers(stream);
    let mut out = trainer.zeros(g.len()).expect("out");
    block.forward(x, &mut out, SEQ, stream, scratch).expect("forward");
    trainer.synchronize().expect("sync");
    let y = trainer.download(&out).expect("download output");
    y.iter().zip(g).map(|(y, g)| f64::from(*y) * f64::from(*g)).sum()
}

/// One forward and one backward with output gradient `g`, into a zeroed all_linear
/// workspace. Returns the output, the input gradient and the workspace.
fn forward_backward(
    trainer: &CudaTrainer,
    config: &TransformerConfig,
    block: &CudaNf4TransformerBlock,
    x: &GpuBuffer<f32>,
    g: &[f32],
) -> (Vec<f32>, Vec<f32>, CudaLoraGradWorkspace) {
    let stream = trainer.stream();
    let mut scratch = all_targets_scratch(trainer, config);
    scratch.zero_forward_buffers(stream);
    let mut out = trainer.zeros(g.len()).expect("out");
    block.forward(x, &mut out, SEQ, stream, &mut scratch).expect("forward");
    let mut ws = workspace(trainer, config);
    for buf in ws.grad_lora.iter_mut().flat_map(|p| [&mut p.a, &mut p.b]) {
        let n = buf.len();
        buf.copy_from_host(&vec![0.0; n]).expect("zero gradient");
    }
    let grad_out = trainer.upload(g).expect("upload output gradient");
    let mut grad_in = trainer.zeros(g.len()).expect("input gradient");
    let mut out_scratch = trainer.zeros(g.len()).expect("output scratch");
    block
        .backward(x, &grad_out, &mut grad_in, &mut out_scratch, SEQ, stream, &mut scratch, &mut ws)
        .expect("backward");
    trainer.synchronize().expect("sync");
    let down = |buf: &GpuBuffer<f32>| trainer.download(buf).expect("download");
    (down(&out), down(&grad_in), ws)
}

/// `target`'s A (`b == false`) or B, as the device holds it.
fn adapter_buf(
    block: &mut CudaNf4TransformerBlock,
    target: LoraTarget,
    b: bool,
) -> &mut GpuBuffer<f32> {
    let pair = block.lora.iter_mut().find(|p| p.target == target).expect("adapter");
    if b {
        &mut pair.b
    } else {
        &mut pair.a
    }
}

/// The central difference of L in element `i` of `target`'s A or B. Restores the weight.
#[allow(clippy::too_many_arguments)]
fn finite_difference(
    trainer: &CudaTrainer,
    block: &mut CudaNf4TransformerBlock,
    scratch: &mut CudaBlockScratch,
    x: &GpuBuffer<f32>,
    g: &[f32],
    target: LoraTarget,
    b: bool,
    i: usize,
) -> f64 {
    let weights = trainer.download(adapter_buf(block, target, b)).expect("download adapter");
    let (up, dn) = (weights[i] + EPS, weights[i] - EPS);
    let mut at = |v: f32| {
        let mut w = weights.clone();
        w[i] = v;
        adapter_buf(block, target, b).copy_from_host(&w).expect("write adapter");
        loss(trainer, block, scratch, x, g)
    };
    let (l_up, l_dn) = (at(up), at(dn));
    adapter_buf(block, target, b).copy_from_host(&weights).expect("restore adapter");
    (l_up - l_dn) / f64::from(up - dn)
}

/// Every adapter's A and B gradient is the derivative of the block's loss: checked at the
/// largest element and at the middle one, against central differences of the forward.
#[test]
#[ignore = "needs a CUDA device; hold the GPU lock around the test binary"]
fn falsify_lora_gradient_flow_v1_007_gradients_match_finite_differences() {
    let config = TransformerConfig::tiny();
    let trainer = CudaTrainer::new().expect("CUDA trainer");
    let mut block = nf4_block(&trainer, &config, &LoraTarget::ALL);
    let n = SEQ * config.hidden_size;
    let x = trainer.upload(&vals(n, 7)).expect("upload input");
    let g = vals(n, 8);
    let (_, _, ws) = forward_backward(&trainer, &config, &block, &x, &g);
    let mut scratch = all_targets_scratch(&trainer, &config);
    for (slot, target) in LoraTarget::ALL.into_iter().enumerate() {
        let pair = &ws.grad_lora[slot];
        assert_eq!(pair.target, target, "gradient workspace slot order");
        for (b, grad) in [(false, &pair.a), (true, &pair.b)] {
            let analytic = trainer.download(grad).expect("download gradient");
            let max = analytic.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(max > 1e-3, "{target:?} (B: {b}): the gradient is zero");
            let top = analytic
                .iter()
                .enumerate()
                .max_by(|l, r| l.1.abs().total_cmp(&r.1.abs()))
                .map_or(0, |(i, _)| i);
            for i in [top, analytic.len() / 2] {
                let fd =
                    finite_difference(&trainer, &mut block, &mut scratch, &x, &g, target, b, i);
                let an = f64::from(analytic[i]);
                let tol = 0.05 * f64::from(max) + 2e-3;
                assert!(
                    (fd - an).abs() <= tol,
                    "{target:?} (B: {b}) [{i}]: backward {an}, finite difference {fd}"
                );
            }
        }
    }
}

/// A block whose k, o, gate, up and down adapters have B = 0 computes what the q_proj,
/// v_proj block computes, number for number, and still gives those adapters a nonzero dB.
/// A scratch built by `new` (q_proj, v_proj) is refused for the down_proj adapter.
#[test]
#[ignore = "needs a CUDA device; hold the GPU lock around the test binary"]
fn falsify_lora_gradient_flow_v1_007_zero_b_adapters_change_no_number() {
    let config = TransformerConfig::tiny();
    let trainer = CudaTrainer::new().expect("CUDA trainer");
    let qv = nf4_block(&trainer, &config, &[LoraTarget::Q, LoraTarget::V]);
    let mut all = nf4_block(&trainer, &config, &LoraTarget::ALL);
    let added = [LoraTarget::K, LoraTarget::O, LoraTarget::Gate, LoraTarget::Up, LoraTarget::Down];
    for &t in &added {
        let buf = adapter_buf(&mut all, t, true);
        let n = buf.len();
        buf.copy_from_host(&vec![0.0; n]).expect("zero B");
    }
    let n = SEQ * config.hidden_size;
    let x = trainer.upload(&vals(n, 7)).expect("upload input");
    let g = vals(n, 8);
    let (out_qv, dx_qv, ws_qv) = forward_backward(&trainer, &config, &qv, &x, &g);
    let (out_all, dx_all, ws_all) = forward_backward(&trainer, &config, &all, &x, &g);
    assert_eq!(out_all, out_qv, "zero-B adapters changed the block output");
    assert_eq!(dx_all, dx_qv, "zero-B adapters changed the input gradient");
    let grads = |ws: &CudaLoraGradWorkspace, t: LoraTarget| {
        let p = ws.grad_lora.iter().find(|p| p.target == t).expect("gradient pair");
        let down = |buf: &GpuBuffer<f32>| trainer.download(buf).expect("download");
        (down(&p.a), down(&p.b))
    };
    for t in [LoraTarget::Q, LoraTarget::V] {
        assert_eq!(grads(&ws_all, t), grads(&ws_qv, t), "{t:?}: the gradients changed");
    }
    for t in added {
        let (_, db) = grads(&ws_all, t);
        assert!(db.iter().any(|v| *v != 0.0), "{t:?}: dB is zero, so the backward skipped it");
    }
    let stream = trainer.stream();
    let mut small = CudaBlockScratch::new(&config, SEQ, trainer.context(), RANK).expect("scratch");
    small.zero_forward_buffers(stream);
    let mut out = trainer.zeros(n).expect("out");
    assert!(config.intermediate_size > config.hidden_size.max(config.q_dim()));
    assert!(
        all.forward(&x, &mut out, SEQ, stream, &mut small).is_err(),
        "a q_proj, v_proj scratch ran the down_proj adapter"
    );
}

/// The LoRA temp row is the widest d_out or d_in over the targets. head_dim 16 puts q_dim
/// (32) under hidden (64), so a temp sized by d_out alone is too small for q_proj's backward.
#[test]
fn falsify_lora_gradient_flow_v1_007_temp_fits_every_target() {
    let config = TransformerConfig { head_dim_override: Some(16), ..TransformerConfig::tiny() };
    let (h, q, i) = (config.hidden_size, config.q_dim(), config.intermediate_size);
    assert!(q < h && h < i, "the config must separate q_dim, hidden and intermediate");
    assert_eq!(lora_temp_dim(&config, &[LoraTarget::Q, LoraTarget::V]), h);
    assert_eq!(lora_temp_dim(&config, &[LoraTarget::O]), h);
    assert_eq!(lora_temp_dim(&config, &[LoraTarget::Down]), i);
    assert_eq!(lora_temp_dim(&config, &LoraTarget::ALL), i);
    assert_eq!(lora_temp_dim(&config, &[]), 0);
}
