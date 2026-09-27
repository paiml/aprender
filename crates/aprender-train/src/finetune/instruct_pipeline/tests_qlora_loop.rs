//! Falsifiers for contract qlora-training-loop-v1 (frozen_base, lora_forward,
//! response_only_loss) on the CPU `train_step` path.
//!
//! The contract was cited by `apr finetune` (`crates/apr-cli/src/commands/finetune.rs`)
//! and `WgpuTrainingPipeline::train_step` before the file existed (0.72 R16).
//! These tests are what the citation now resolves to.

use super::*;

fn pipeline() -> InstructPipeline {
    let instruct_config = InstructConfig {
        lora_rank: 4,
        lora_alpha: 8.0,
        learning_rate: 1e-2,
        max_seq_len: 64,
        gradient_clip_norm: None,
        ..InstructConfig::default()
    };
    InstructPipeline::new(&TransformerConfig::tiny(), instruct_config)
}

fn snapshot(tensors: Vec<&crate::Tensor>) -> Vec<Vec<f32>> {
    tensors.into_iter().map(|t| t.data().to_vec()).collect()
}

/// FALSIFY-QTL-001 frozen_base: a training step moves no base-model parameter.
#[test]
fn falsify_qtl_001_train_step_leaves_base_weights_bit_identical() {
    let mut p = pipeline();
    let before = snapshot(p.model.parameters());
    assert!(!before.is_empty(), "tiny transformer exposes base parameters");
    for _ in 0..3 {
        let r = p.train_step(&[1, 2, 3, 4], &[5, 6, 7, 8]);
        assert!(r.loss.is_finite() && r.loss > 0.0, "step ran and produced a loss");
    }
    let after = snapshot(p.model.parameters());
    for (i, (a, b)) in before.iter().zip(&after).enumerate() {
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "FALSIFY-QTL-001: base parameter #{i} changed during train_step — \
             the optimizer reached a frozen base weight"
        );
    }
}

/// FALSIFY-QTL-002 lora_forward: the adapters are in the forward graph, so a
/// training step moves them (a forward without LoRA leaves them orphaned and
/// their gradients zero).
#[test]
fn falsify_qtl_002_train_step_moves_lora_adapters() {
    let mut p = pipeline();
    assert!(!p.lora_layers.is_empty(), "pipeline built LoRA adapters");
    let before: Vec<Vec<f32>> = p.lora_layers.iter().map(|l| l.lora_b().data().to_vec()).collect();
    for _ in 0..3 {
        p.train_step(&[1, 2, 3, 4], &[5, 6, 7, 8]);
    }
    let moved = p
        .lora_layers
        .iter()
        .zip(&before)
        .any(|(l, b)| l.lora_b().data().iter().zip(b).any(|(x, y)| (x - y).abs() > 1e-7));
    assert!(moved, "FALSIFY-QTL-002: no LoRA B matrix moved — adapters are not in the forward");
}

/// FALSIFY-QTL-003 response_only_loss: the loss window starts at the last
/// prompt position, so exactly |response| targets are scored, prompt rows get
/// zero gradient, and prompt-row logits cannot change the loss.
#[test]
fn falsify_qtl_003_loss_scores_response_tokens_only() {
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5];
    let response: Vec<u32> = vec![6, 7, 8];
    let mut p = pipeline();
    let r = p.train_step(&prompt, &response);
    assert_eq!(
        r.num_response_tokens,
        response.len(),
        "FALSIFY-QTL-003: scored {} tokens for a {}-token response",
        r.num_response_tokens,
        response.len()
    );

    let ids: Vec<u32> = prompt.iter().chain(&response).copied().collect();
    let vocab = 16usize;
    let seq = ids.len();
    let logits: Vec<f32> = (0..seq * vocab).map(|i| ((i * 7) % 11) as f32 * 0.1).collect();
    let loss_start = prompt.len() - 1;
    let (loss, grad) =
        InstructPipeline::compute_causal_lm_loss(&logits, &ids, loss_start, seq - 1, vocab);

    for pos in 0..loss_start {
        assert!(
            grad[pos * vocab..(pos + 1) * vocab].iter().all(|g| *g == 0.0),
            "FALSIFY-QTL-003: prompt position {pos} received gradient"
        );
    }
    assert!(
        grad[loss_start * vocab..(loss_start + 1) * vocab].iter().any(|g| *g != 0.0),
        "the position predicting the first response token is scored"
    );

    let mut perturbed = logits.clone();
    for v in &mut perturbed[..loss_start * vocab] {
        *v += 3.0;
    }
    let (loss2, _) =
        InstructPipeline::compute_causal_lm_loss(&perturbed, &ids, loss_start, seq - 1, vocab);
    assert_eq!(
        loss.to_bits(),
        loss2.to_bits(),
        "FALSIFY-QTL-003: changing prompt-row logits changed the loss ({loss} -> {loss2})"
    );
}
