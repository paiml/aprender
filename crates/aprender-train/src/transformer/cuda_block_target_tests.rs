//! FALSIFY-LORA_TARGET_SELECTION_V1_005 (0.72 R15a C2c): the NF4 block, the gradient
//! workspace, the optimizer state and the gradient clip hold one A and one B per LoRA target.
//!
//! An all_linear block takes q_proj and v_proj through `new` and the other five targets
//! through `add_lora_adapter`. The scale is 2, so dividing B by it on upload and multiplying
//! by it on download is exact and the round trips compare bits. Every test needs a CUDA
//! device. Child module of `cuda_block` so it can read the private stores.

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
