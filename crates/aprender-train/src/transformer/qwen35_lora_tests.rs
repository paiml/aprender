//! [`Qwen35Lora`]: the adapter gradient is the chain rule of the gradchecked full backward,
//! and `AdamW` on the adapters alone trains the model while the base stays untouched.

use super::super::{LoraTarget, Qwen35Lora};
use super::{write_tiny_qwen35, Lcg, Qwen35Model};
use crate::optim::{AdamW, Optimizer};
use crate::Tensor;

const TOKENS: [u32; 7] = [3, 17, 0, 39, 22, 5, 11];
const TARGETS: [u32; 7] = [17, 0, 39, 22, 5, 11, 8];

fn tiny(seed: u64) -> Qwen35Model {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("tiny.gguf");
    write_tiny_qwen35(&path, seed);
    Qwen35Model::from_gguf(&path).expect("train loads")
}

fn loss_at(lora: &Qwen35Lora, base: &Qwen35Model, work: &mut Qwen35Model) -> f32 {
    lora.merge_into(base, work);
    work.as_lm().loss(&TOKENS, &TARGETS)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `B = 0` at init, so the adapted model IS the base: merged logits are bit-equal.
#[test]
fn zero_b_merge_equals_base() {
    let base = tiny(7);
    let lora = Qwen35Lora::new(&base, &LoraTarget::ALL, 4, 8.0, 1);
    // 4 GDN layers × (qkv, gate, out, 3 FFN) + 1 attention layer × (q, k, v, out, 3 FFN).
    assert_eq!(lora.slots.len(), 4 * 6 + 7);
    let mut work = base.clone();
    lora.merge_into(&base, &mut work);
    assert_eq!(work.forward(&TOKENS), base.forward(&TOKENS));
}

/// FALSIFY-QTG-009: every adapter tensor's gradient matches a central difference of the
/// loss along a random direction in that tensor (`B` random, so `dA ≠ 0`).
#[test]
fn falsify_qtg_009_lora_grad_matches_finite_difference() {
    for seed in [4_u64, 13] {
        let base = tiny(seed);
        let mut lora = Qwen35Lora::new(&base, &LoraTarget::ALL, 3, 6.0, seed);
        let mut r = Lcg(seed ^ 0x9e37);
        for p in lora.params.iter_mut().skip(1).step_by(2) {
            *p = Tensor::from_vec(r.vec(p.len(), 0.3), true);
        }
        let mut work = base.clone();
        lora.merge_into(&base, &mut work);
        let (_, g) = work.loss_and_grads(&TOKENS, &TARGETS);
        lora.set_grads(&g);

        let (mut pred, mut fd) = (Vec::new(), Vec::new());
        let h = 1e-2_f32;
        for i in 0..lora.params.len() {
            let u = r.vec(lora.params[i].len(), 1.0);
            let norm = dot(&u, &u).sqrt();
            let u: Vec<f32> = u.iter().map(|x| x / norm).collect();
            let gi = lora.params[i].grad().expect("grad set").to_vec();
            pred.push(dot(&gi, &u));
            let p0 = lora.params[i].data().to_vec();
            let shifted = |sign: f32| p0.iter().zip(&u).map(|(p, d)| p + sign * h * d).collect();
            lora.params[i] = Tensor::from_vec(shifted(1.0), true);
            let up = loss_at(&lora, &base, &mut work);
            lora.params[i] = Tensor::from_vec(shifted(-1.0), true);
            let down = loss_at(&lora, &base, &mut work);
            lora.params[i] = Tensor::from_vec(p0, true);
            fd.push((up - down) / (2.0 * h));
        }
        let diff: Vec<f32> = pred.iter().zip(&fd).map(|(p, f)| p - f).collect();
        let rel = dot(&diff, &diff).sqrt() / dot(&fd, &fd).sqrt();
        assert!(rel <= 2e-2, "seed {seed}: rel err {rel}\npred {pred:?}\nfd   {fd:?}");
    }
}

fn train(base: &Qwen35Model, lr: f32, steps: usize) -> Vec<f32> {
    let mut lora = Qwen35Lora::new(base, &LoraTarget::ALL, 4, 8.0, 3);
    let mut opt = AdamW::new(lr, 0.9, 0.999, 1e-8, 0.0);
    let mut work = base.clone();
    let mut losses = Vec::new();
    for _ in 0..steps {
        lora.merge_into(base, &mut work);
        let (loss, g) = work.loss_and_grads(&TOKENS, &TARGETS);
        losses.push(loss);
        lora.set_grads(&g);
        opt.step(&mut lora.params);
    }
    losses.push(loss_at(&lora, base, &mut work));
    losses
}

/// `AdamW` on the adapters alone overfits the tiny model, and never writes the base.
/// Planted: `lr ≈ 0` leaves the loss exactly where it started (R4's lr = 0 falsifier).
#[test]
fn adamw_lora_overfits_and_leaves_base_untouched() {
    let base = tiny(11);
    let before = base.forward(&TOKENS);
    let losses = train(&base, 1e-2, 30);
    let (first, last) = (losses[0], losses[losses.len() - 1]);
    assert!(last < first / 2.0, "LoRA+AdamW must overfit: {losses:?}");
    assert_eq!(base.forward(&TOKENS), before, "training wrote the base weights");
    // AdamW refuses lr = 0 (precondition); the smallest normal f32 step moves no weight.
    let frozen = train(&base, f32::MIN_POSITIVE, 3);
    assert!(frozen.iter().all(|&l| l == frozen[0]), "a zero step moved the loss: {frozen:?}");
}

/// Spike S-R4a: LoRA r16 (alpha 32) on every attention, Gated `DeltaNet` and MLP projection of the
/// real 0.8B, `AdamW`, one sentence, CPU. Prints the adapter size and seconds per step.
#[test]
#[ignore = "needs a Qwen3.5 GGUF: QWEN35_GGUF=/path/to/Qwen3.5-0.8B-Q4_K_M.gguf"]
fn real_model_lora_adamw_descends() {
    let path = std::env::var("QWEN35_GGUF").expect("set QWEN35_GGUF");
    let lr: f32 = std::env::var("LORA_LR").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-4);
    let base = Qwen35Model::from_gguf(&path).expect("train loads");
    let mut lora = Qwen35Lora::new(&base, &LoraTarget::ALL, 16, 32.0, 42);
    eprintln!("adapters {} slots, {} params, lr {lr}", lora.slots.len(), lora.num_params());
    let mut opt = AdamW::new(lr, 0.9, 0.999, 1e-8, 0.0);
    let mut work = base.clone();
    let (tokens, targets) = ([9_707_u32, 11, 1_879, 374, 264], [11_u32, 1_879, 374, 264, 1_273]);
    let mut losses = Vec::new();
    for step in 0..8 {
        let t0 = std::time::Instant::now();
        lora.merge_into(&base, &mut work);
        let t1 = t0.elapsed().as_secs_f64();
        let (loss, g) = work.loss_and_grads(&tokens, &targets);
        let t2 = t0.elapsed().as_secs_f64();
        lora.set_grads(&g);
        opt.step(&mut lora.params);
        let t3 = t0.elapsed().as_secs_f64();
        eprintln!(
            "step {step}: loss {loss:.5}, merge {t1:.1} s, fwd+bwd {:.1} s, project+adamw {:.1} s",
            t2 - t1,
            t3 - t2
        );
        losses.push(loss);
    }
    lora.merge_into(&base, &mut work);
    losses.push(work.as_lm().loss(&tokens, &targets));
    eprintln!("losses {losses:?}");
    assert!(losses[1] < losses[0], "the first step must descend: {losses:?}");
    assert!(losses[losses.len() - 1] < losses[0] / 2.0, "8 steps must halve the loss: {losses:?}");
}

/// Round-trip every adapted matrix through the CUDA path's NF4 quantizer (64-element
/// absmax blocks over the row-major weight), which is the base R4 trains on.
fn nf4_base(base: &Qwen35Model, lora: &Qwen35Lora) -> Qwen35Model {
    let mut m = base.clone();
    for slot in &lora.slots {
        let w = super::super::lora::weight_mut(&mut m, slot.layer, slot.target).expect("slot");
        let q = trueno_gpu::kernels::quantize_nf4(w, slot.rows, slot.cols);
        *w = trueno_gpu::kernels::dequantize_nf4(&q);
    }
    m
}

/// The NF4 round trip is what QQE-005's CPU oracle loads: it moves the weights, stays
/// within the codebook bound (C-NF4-001), and a second round trip is bit-identical.
#[test]
fn nf4_round_trip_base_is_idempotent_and_bounded() {
    let base = tiny(5);
    let lora = Qwen35Lora::new(&base, &LoraTarget::ALL, 4, 8.0, 1);
    let once = nf4_base(&base, &lora);
    let twice = nf4_base(&once, &lora);
    assert_eq!(once.forward(&TOKENS), twice.forward(&TOKENS), "NF4 round trip is not idempotent");
    assert_ne!(once.forward(&TOKENS), base.forward(&TOKENS), "NF4 round trip changed nothing");
    for slot in &lora.slots {
        let w0 = super::super::lora::weight_mut(&mut base.clone(), slot.layer, slot.target)
            .expect("slot")
            .clone();
        let mut o = once.clone();
        let w1 = super::super::lora::weight_mut(&mut o, slot.layer, slot.target).expect("slot");
        for (b0, b1) in w0.chunks(64).zip(w1.chunks(64)) {
            let absmax = b0.iter().fold(0.0_f32, |a, x| a.max(x.abs()));
            for (x, y) in b0.iter().zip(b1.iter()) {
                assert!((x - y).abs() <= 0.16 * absmax + 1e-7, "{slot:?}: {x} -> {y}");
            }
        }
    }
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let d: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    dot(&d, &d).sqrt() / dot(b, b).sqrt().max(f32::MIN_POSITIVE)
}

/// Spike S-R4c: on the real 0.8B, how far are the adapter gradients on the Q4_K base from
/// those on the NF4 base the CUDA path trains? If that gap exceeds QQE-005's rel 1e-2, an
/// oracle on the wrong base goes RED for a non-bug reason (risk K12).
#[test]
#[ignore = "needs a Qwen3.5 GGUF: QWEN35_GGUF=/path/to/Qwen3.5-0.8B-Q4_K_M.gguf"]
fn real_model_nf4_base_gradient_gap() {
    let path = std::env::var("QWEN35_GGUF").expect("set QWEN35_GGUF");
    let base = Qwen35Model::from_gguf(&path).expect("train loads");
    let mut lora = Qwen35Lora::new(&base, &LoraTarget::ALL, 16, 32.0, 42);
    let mut r = Lcg(42 ^ 0x9e37);
    for p in lora.params.iter_mut().skip(1).step_by(2) {
        *p = Tensor::from_vec(r.vec(p.len(), 0.01), true);
    }
    let nf4 = nf4_base(&base, &lora);
    let (tokens, targets) = ([9_707_u32, 11, 1_879], [11_u32, 1_879, 374]);
    let grads = |m: &Qwen35Model| {
        let mut work = m.clone();
        lora.merge_into(m, &mut work);
        let (loss, g) = work.loss_and_grads(&tokens, &targets);
        lora.set_grads(&g);
        let gs: Vec<Vec<f32>> =
            lora.params.iter().map(|p| p.grad().expect("grad set").to_vec()).collect();
        (loss, gs)
    };
    let (l_q4k, g_q4k) = grads(&base);
    let (l_nf4, g_nf4) = grads(&nf4);
    let (_, g_nf4_again) = grads(&nf4);
    let control = g_nf4.iter().zip(&g_nf4_again).map(|(a, b)| rel(a, b)).fold(0.0_f32, f32::max);
    let mut gaps: Vec<f32> = g_q4k.iter().zip(&g_nf4).map(|(a, b)| rel(a, b)).collect();
    gaps.sort_by(f32::total_cmp);
    let over = gaps.iter().filter(|&&x| x > 1e-2).count();
    eprintln!(
        "S-R4c: loss q4k {l_q4k:.5} nf4 {l_nf4:.5}; per-tensor rel gap over {} tensors: \
         min {:.4} median {:.4} p90 {:.4} max {:.4}; {over} > 1e-2; control (same base twice) {control:e}",
        gaps.len(),
        gaps[0],
        gaps[gaps.len() / 2],
        gaps[gaps.len() * 9 / 10],
        gaps[gaps.len() - 1]
    );
    assert_eq!(control, 0.0, "the backward is not deterministic on one base");
}
