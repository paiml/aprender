//! GPU falsifiers: distill-batch-honesty-v1 (DBH-001/006/007/008) and the
//! DDP accumulate path. Each test states the planted variant it must catch.

use super::CudaTransformerTrainer;
use crate::train::transformer_trainer::batch::LMBatch;
use crate::train::transformer_trainer::config::TransformerTrainConfig;
use crate::transformer::TransformerConfig;

const VOCAB: usize = 256;
const SEQ: usize = 8;

fn tiny(accumulation_steps: usize) -> CudaTransformerTrainer {
    let model = TransformerConfig {
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_kv_heads: 4,
        vocab_size: VOCAB,
        max_position_embeddings: 32,
        ..TransformerConfig::tiny()
    };
    let mut config = TransformerTrainConfig::new(model).with_accumulation_steps(accumulation_steps);
    config.max_seq_len = SEQ;
    config.lr = 1e-3;
    config.weight_decay = 0.0;
    CudaTransformerTrainer::new(config).expect("GPU falsifier needs a CUDA device")
}

fn row(seed: u32) -> Vec<u32> {
    (0..SEQ as u32).map(|j| (seed * 31 + j * 7) % VOCAB as u32).collect()
}

fn kd_grad(seed: u32, sign: f32) -> Vec<f32> {
    #[allow(clippy::cast_precision_loss)]
    (0..VOCAB).map(|v| sign * (((v as u32 * 13 + seed) % 17) as f32 - 8.0) * 1e-2).collect()
}

fn logits(t: &mut CudaTransformerTrainer, ids: &[u32]) -> Vec<f32> {
    t.forward_logits(ids).expect("forward_logits")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Two freshly built trainers must start identical, or every A/B below is
/// comparing noise. Fails loudly instead of passing vacuously.
fn twin() -> (CudaTransformerTrainer, CudaTransformerTrainer) {
    let (mut a, mut b) = (tiny(1), tiny(1));
    let probe = row(99);
    let d = max_abs_diff(&logits(&mut a, &probe), &logits(&mut b, &probe));
    assert!(d == 0.0, "trainer init is not deterministic (max |d| = {d}); A/B tests invalid");
    (a, b)
}

/// FALSIFY-DBH-008: the batched step is ONE step from the mean gradient.
/// Mean of a row taken twice == that row, so both trainers end identical.
/// Planted variant: a micro-step per row (two optimizer steps) moves the
/// weights twice and the logits differ.
#[test]
#[ignore = "GPU"]
fn falsify_dbh_008_batch_of_a_repeated_row_equals_that_row() {
    let (mut a, mut b) = twin();
    let (r, g) = (row(1), kd_grad(1, 1.0));
    a.kd_step_batch(&[(&r, &g), (&r, &g)]).expect("batched step");
    b.kd_step_batch(&[(&r, &g)]).expect("one-row step");
    assert_eq!((a.step(), b.step()), (1, 1), "one optimizer step each");
    let probe = row(7);
    let d = max_abs_diff(&logits(&mut a, &probe), &logits(&mut b, &probe));
    assert!(d < 1e-5, "mean of (r, r) must equal r: max |d| = {d}");
}

/// FALSIFY-DBH-001 (GPU half): every row lands in the one accumulator.
/// Rows (r, g) and (r, -g) average to zero; with weight decay 0 AdamW
/// leaves the weights where they were. Planted variant: a dropped row (the
/// S-R6 last-row-only student) steps on +g or -g alone and moves them.
#[test]
#[ignore = "GPU"]
fn falsify_dbh_001_opposite_rows_cancel() {
    let mut t = tiny(1);
    let probe = row(7);
    let before = logits(&mut t, &probe);
    let r = row(1);
    let (g, neg) = (kd_grad(1, 1.0), kd_grad(1, -1.0));
    t.kd_step_batch(&[(&r, &g), (&r, &neg)]).expect("batched step");
    let d = max_abs_diff(&before, &logits(&mut t, &probe));
    assert!(d < 1e-5, "opposite rows must cancel: max |d| = {d}");
}

/// kd_step_batch reads the CPU accumulator even when the trainer holds the
/// GPU-resident one (accumulation_steps > 1). Planted variant: gradients go
/// to the GPU accumulator, the step averages zeros, logits do not move.
#[test]
#[ignore = "GPU"]
fn falsify_dbh_001_steps_with_a_gpu_accumulator_present() {
    let mut t = tiny(4);
    let probe = row(7);
    let before = logits(&mut t, &probe);
    let (r, g) = (row(1), kd_grad(1, 1.0));
    t.kd_step_batch(&[(&r, &g), (&r, &g)]).expect("batched step");
    let d = max_abs_diff(&before, &logits(&mut t, &probe));
    assert!(d > 1e-6, "the step must move the model: max |d| = {d}");
}

/// FALSIFY-DBH-006: set_lr reaches the schedule the optimizers read.
/// Planted variant: the pre-fix provider, which never set it (1e-3 stays).
#[test]
#[ignore = "GPU"]
fn falsify_dbh_006_set_lr_reaches_the_schedule() {
    let mut t = tiny(1);
    t.set_lr(2e-4);
    assert!((t.current_lr() - 2e-4).abs() < 1e-9, "current_lr = {}", t.current_lr());
}

/// FALSIFY-DBH-007: the one-row KD step moves the CPU embedding and counts
/// the step. Planted variant: forward_backward_with_grad without
/// optimizer_step() — the embedding stays at the checkpoint, step stays 0.
#[test]
#[ignore = "GPU"]
fn falsify_dbh_007_one_row_step_moves_the_embedding() {
    let mut t = tiny(1);
    let before = t.model.embed_tokens.weight.data().to_vec();
    let (r, g) = (row(1), kd_grad(1, 1.0));
    t.forward_backward_with_grad(&r, &g).expect("kd step");
    let after = t.model.embed_tokens.weight.data().to_vec();
    assert_eq!(t.step(), 1, "optimizer_step ran once");
    assert!(max_abs_diff(&before, &after) > 0.0, "embedding must move");
}

/// DDP accumulate path: after forward_backward_batch the CPU accumulator
/// DDP AllReduces holds a non-zero gradient, at accumulation 1 and 4.
/// Planted variants: ensure_grad_accum() alone — accumulation 1 panics on
/// the empty D2H staging buffer, accumulation 4 leaves the buffer zero.
#[test]
#[ignore = "GPU"]
fn falsify_ddp_backward_fills_the_cpu_accumulator() {
    for accumulation in [1, 4] {
        let mut t = tiny(accumulation);
        t.use_cpu_grad_accum();
        let seq: Vec<u32> = row(3).into_iter().chain([5]).collect();
        let batch = LMBatch::from_sequences(&[seq], 0, VOCAB as u32 - 1);
        let _ = t.forward_backward_batch(&batch);
        let accum = t.grad_accum_ref().expect("grad_accum");
        assert_eq!(accum.accumulated_count, 1, "accumulation {accumulation}");
        let lm = accum.lm_head_grad.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(lm > 0.0, "accumulation {accumulation}: lm_head grad is all zero");
    }
}
