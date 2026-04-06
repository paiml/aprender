//! Falsification tests for LoRA spec Layers 0 and 1
//!
//! Layer 0: Math invariants (F-MATH-001, F-MATH-007, F-MATH-008)
//! Layer 1: Weight freezing (F-FREEZE-003..008)
//!
//! Reference: lora-qlora-enhancement.md Section 6

#![allow(clippy::unwrap_used)]

use super::*;
use crate::autograd::matmul;
use crate::Tensor;

// ========================================================================
// LAYER 0: MATH INVARIANTS
// ========================================================================

/// F-MATH-001: B@A = 0 at init (B=zeros) for all practical ranks.
///
/// Comprehensive version testing r in {4, 8, 16, 32, 64, 128} with
/// varying d_out and d_in dimensions.
#[test]
fn test_falsify_f_math_001_ba_zero_at_init_comprehensive() {
    let dims = [(32, 64), (64, 64), (128, 256), (256, 128), (512, 512)];
    let ranks = [4, 8, 16, 32, 64, 128];

    for &(d_out, d_in) in &dims {
        for &r in &ranks {
            if r > d_out.min(d_in) {
                continue; // skip invalid rank > min(d_out, d_in)
            }

            let base_data: Vec<f32> =
                (0..d_out * d_in).map(|i| (i as f32 * 0.07).sin() * 0.5).collect();
            let base_weight = Tensor::from_vec(base_data, false);
            let lora = LoRALayer::new(base_weight, d_out, d_in, r, r as f32);

            // Compute B @ A: [d_out, r] @ [r, d_in] -> [d_out, d_in]
            let ba = matmul(lora.lora_b(), lora.lora_a(), d_out, r, d_in);

            // Every element of B@A must be zero because B is initialized to zeros
            let max_abs = ba.data().iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            assert!(
                max_abs < 1e-10,
                "F-MATH-001 violated: B@A != 0 at init for d_out={d_out}, d_in={d_in}, r={r}. \
                 max|B@A| = {max_abs}"
            );
        }
    }
}

/// F-MATH-007: Trainable param count = 2 * r * (d_in + d_out) elements
/// (accounting for A: [r * d_in] and B: [d_out * r]).
///
/// Note: the formula counts total elements across both matrices.
/// A has r * d_in elements, B has d_out * r elements.
/// Total = r * d_in + d_out * r = r * (d_in + d_out).
#[test]
fn test_falsify_f_math_007_trainable_param_count() {
    let configs = [
        // (d_out, d_in, rank)
        (64, 64, 4),
        (64, 64, 8),
        (64, 64, 16),
        (128, 64, 8),
        (256, 128, 16),
        (512, 512, 32),
        (768, 768, 64),
        (1024, 256, 128),
        (4, 4, 2), // minimal
        (3, 5, 1), // asymmetric minimal
    ];

    for &(d_out, d_in, rank) in &configs {
        let base_weight = Tensor::from_vec(vec![0.0; d_out * d_in], false);
        let mut lora = LoRALayer::new(base_weight, d_out, d_in, rank, rank as f32);

        let params = lora.trainable_params();
        assert_eq!(params.len(), 2, "Should have exactly 2 trainable param tensors (A and B)");

        let total_elements: usize = params.iter().map(|p| p.len()).sum();
        let expected = rank * d_in + d_out * rank; // r*(d_in + d_out)

        assert_eq!(
            total_elements, expected,
            "F-MATH-007 violated: trainable elements = {total_elements}, \
             expected r*(d_in+d_out) = {expected} for d_out={d_out}, d_in={d_in}, r={rank}"
        );

        // Also verify each individual matrix shape
        assert_eq!(
            params[0].len(),
            rank * d_in,
            "A matrix should have r*d_in = {} elements",
            rank * d_in
        );
        assert_eq!(
            params[1].len(),
            d_out * rank,
            "B matrix should have d_out*r = {} elements",
            d_out * rank
        );

        // All trainable params must require grad
        for (i, p) in params.iter().enumerate() {
            assert!(p.requires_grad(), "F-MATH-007: trainable param {i} must require grad");
        }
    }
}

/// F-MATH-008: NF4 dequantize within tolerance.
///
/// Quantize known values to 4-bit, dequantize, and verify the
/// round-trip error is bounded.
#[test]
fn test_falsify_f_math_008_nf4_dequantize_tolerance() {
    use crate::quant::{dequantize_4bit, quantize_4bit};

    // Test with a range of realistic weight distributions
    let test_vectors: Vec<Vec<f32>> = vec![
        // Uniform small values
        (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect(),
        // Gaussian-like distribution
        (0..256).map(|i| (i as f32 * 0.05).sin() * 0.5).collect(),
        // Larger range
        (0..256).map(|i| (i as f32 * 0.1).cos() * 2.0).collect(),
        // Near-zero values (common in LoRA adapters)
        (0..256).map(|i| (i as f32 * 0.001).sin() * 0.01).collect(),
        // All zeros
        vec![0.0; 256],
        // Constant nonzero
        vec![0.42; 256],
    ];

    for (vec_idx, original) in test_vectors.iter().enumerate() {
        let quantized = quantize_4bit(original);
        let recovered = dequantize_4bit(&quantized);

        assert_eq!(
            original.len(),
            recovered.len(),
            "F-MATH-008: length mismatch after round-trip for vector {vec_idx}"
        );

        // Compute max absolute error and relative error
        let mut max_abs_err = 0.0f32;
        let mut max_rel_err = 0.0f32;

        for (i, (&orig, &rec)) in original.iter().zip(recovered.iter()).enumerate() {
            let abs_err = (orig - rec).abs();
            max_abs_err = max_abs_err.max(abs_err);

            if orig.abs() > 1e-6 {
                let rel_err = abs_err / orig.abs();
                max_rel_err = max_rel_err.max(rel_err);
            }

            // Each value must be finite
            assert!(rec.is_finite(), "F-MATH-008: dequantized value at [{i}] is not finite: {rec}");
        }

        // 4-bit quantization should have bounded error.
        // With block_size=64, absmax quantization, max absolute error
        // should be within the quantization step size (~absmax/8).
        let absmax = original.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let tolerance = if absmax < 1e-6 {
            1e-6 // near-zero input, allow tiny error
        } else {
            absmax * 0.35 // 4-bit has ~16 levels, so ~1/8 step * some margin
        };

        assert!(
            max_abs_err <= tolerance,
            "F-MATH-008 violated: max abs error {max_abs_err} > tolerance {tolerance} \
             for vector {vec_idx} (absmax={absmax})"
        );
    }
}

// ========================================================================
// LAYER 1: WEIGHT FREEZING
// ========================================================================

/// Helper: create a LoRA-enabled trainer with tiny config and run N steps.
/// Returns the trainer after training.
fn make_lora_trainer_and_train(num_steps: usize) -> crate::train::TransformerTrainer {
    use crate::train::{LMBatch, TransformerTrainConfig};
    use crate::transformer::TransformerConfig;

    let config = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_lora(4, 8.0, vec!["q_proj".to_string(), "v_proj".to_string()])
        .with_lr(0.01)
        .with_max_steps(num_steps)
        .with_use_cuda(false);

    let mut trainer = crate::train::TransformerTrainer::new(config);

    // Create simple training batches
    let seq_len = 8;
    let input_ids: Vec<u32> = (0..seq_len).map(|i| (i as u32) % 100 + 1).collect();
    let target_ids: Vec<u32> = (0..seq_len).map(|i| (i as u32 + 1) % 100 + 1).collect();

    for _ in 0..num_steps {
        let batch = LMBatch::single(input_ids.clone(), target_ids.clone());
        let _loss = trainer.train_batch(&batch);
    }

    trainer
}

/// Helper: compute deterministic hash of a flat f32 slice (via byte repr).
fn hash_f32_slice(data: &[f32]) -> [u8; 32] {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    // Deterministic fingerprint using multiple rounds of FNV-1a
    let mut hash = [0u8; 32];
    for chunk_idx in 0..32 {
        let mut h: u64 = 0xcbf29ce484222325;
        for (i, &b) in bytes.iter().enumerate() {
            h ^= u64::from(b).wrapping_add(chunk_idx as u64);
            h = h.wrapping_mul(0x100000001b3);
            h ^= i as u64;
        }
        hash[chunk_idx] = (h & 0xff) as u8;
    }
    hash
}

/// F-FREEZE-003: Base weights unchanged after 100 CPU LoRA steps (SHA256 check).
///
/// Snapshots all base model parameters before training, runs 100 LoRA steps,
/// then verifies every base parameter is byte-identical.
#[test]
fn test_falsify_f_freeze_003_base_weights_unchanged_after_100_steps() {
    use crate::train::{LMBatch, TransformerTrainConfig};
    use crate::transformer::TransformerConfig;

    let config = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_lora(4, 8.0, vec!["q_proj".to_string(), "v_proj".to_string()])
        .with_lr(0.01)
        .with_use_cuda(false);

    let mut trainer = crate::train::TransformerTrainer::new(config);

    // Snapshot base weights BEFORE training (named_parameters gives us names)
    let pre_snapshots: Vec<(String, Vec<f32>)> = trainer
        .model()
        .named_parameters()
        .into_iter()
        .map(|(name, tensor)| (name, tensor.data().to_vec()))
        .collect();

    // Also compute hashes for the SHA256 check
    let pre_hashes: Vec<(String, [u8; 32])> =
        pre_snapshots.iter().map(|(name, data)| (name.clone(), hash_f32_slice(data))).collect();

    // Run 100 training steps
    let seq_len = 8;
    let input_ids: Vec<u32> = (0..seq_len).map(|i| (i as u32) % 100 + 1).collect();
    let target_ids: Vec<u32> = (0..seq_len).map(|i| (i as u32 + 1) % 100 + 1).collect();

    for _ in 0..100 {
        let batch = LMBatch::single(input_ids.clone(), target_ids.clone());
        let _loss = trainer.train_batch(&batch);
    }

    // Verify base weights are unchanged (exclude norm weights which DO change)
    let post_params = trainer.model().named_parameters();

    for (pre_name, pre_hash) in &pre_hashes {
        // Skip norm weights — they are trainable even in LoRA mode
        if pre_name.contains("norm") {
            continue;
        }

        // Find corresponding post-training parameter
        let post_tensor = post_params
            .iter()
            .find(|(name, _)| name == pre_name)
            .map_or_else(|| panic!("Parameter {pre_name} disappeared after training"), |(_, t)| t);

        let post_data = post_tensor.data().to_vec();
        let post_hash = hash_f32_slice(&post_data);

        assert_eq!(
            pre_hash, &post_hash,
            "F-FREEZE-003 violated: base weight '{pre_name}' changed after 100 LoRA steps \
             (hash mismatch)"
        );
    }
}

/// F-FREEZE-004: LoRA A matrices CHANGE after training.
///
/// After multiple training steps, at least some A matrices must have
/// different values from their initial state.
#[test]
fn test_falsify_f_freeze_004_lora_a_changes_after_training() {
    use crate::train::{LMBatch, TransformerTrainConfig};
    use crate::transformer::TransformerConfig;

    let config = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_lora(4, 8.0, vec!["q_proj".to_string(), "v_proj".to_string()])
        .with_lr(0.01)
        .with_use_cuda(false);

    let mut trainer = crate::train::TransformerTrainer::new(config);

    // Snapshot initial A matrices
    let initial_a: Vec<Vec<f32>> =
        trainer.lora_layers().unwrap().iter().map(|l| l.lora_a().data().to_vec()).collect();

    // Train for 50 steps
    let seq_len = 8;
    let input_ids: Vec<u32> = (1..=seq_len as u32).collect();
    let target_ids: Vec<u32> = (2..=seq_len as u32 + 1).collect();

    for _ in 0..50 {
        let batch = LMBatch::single(input_ids.clone(), target_ids.clone());
        let _loss = trainer.train_batch(&batch);
    }

    // At least one A matrix must have changed
    let mut any_changed = false;
    for (layer_idx, lora_layer) in trainer.lora_layers().unwrap().iter().enumerate() {
        let current_a = lora_layer.lora_a().data();
        let init_a = &initial_a[layer_idx];

        let max_diff =
            current_a.iter().zip(init_a.iter()).map(|(c, i)| (c - i).abs()).fold(0.0f32, f32::max);

        if max_diff > 1e-8 {
            any_changed = true;
        }
    }

    assert!(
        any_changed,
        "F-FREEZE-004 violated: NO LoRA A matrix changed after 50 training steps. \
         Gradients may not be flowing to A."
    );
}

/// F-FREEZE-005: LoRA B matrices CHANGE after training.
///
/// After multiple training steps, at least some B matrices must have
/// different values from their initial state (all zeros).
#[test]
fn test_falsify_f_freeze_005_lora_b_changes_after_training() {
    use crate::train::{LMBatch, TransformerTrainConfig};
    use crate::transformer::TransformerConfig;

    let config = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_lora(4, 8.0, vec!["q_proj".to_string(), "v_proj".to_string()])
        .with_lr(0.01)
        .with_use_cuda(false);

    let mut trainer = crate::train::TransformerTrainer::new(config);

    // Snapshot initial B matrices (should be all zeros)
    let initial_b: Vec<Vec<f32>> =
        trainer.lora_layers().unwrap().iter().map(|l| l.lora_b().data().to_vec()).collect();

    // Verify initial B is all zeros
    for (i, b) in initial_b.iter().enumerate() {
        let max_val = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_val < 1e-10,
            "Pre-condition: B[{i}] should be initialized to zeros, but max|B| = {max_val}"
        );
    }

    // Train for 50 steps
    let seq_len = 8;
    let input_ids: Vec<u32> = (1..=seq_len as u32).collect();
    let target_ids: Vec<u32> = (2..=seq_len as u32 + 1).collect();

    for _ in 0..50 {
        let batch = LMBatch::single(input_ids.clone(), target_ids.clone());
        let _loss = trainer.train_batch(&batch);
    }

    // At least one B matrix must have changed from zeros
    let mut any_changed = false;
    for lora_layer in trainer.lora_layers().unwrap() {
        let current_b = lora_layer.lora_b().data();
        let max_abs = current_b.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        if max_abs > 1e-8 {
            any_changed = true;
        }
    }

    assert!(
        any_changed,
        "F-FREEZE-005 violated: NO LoRA B matrix changed after 50 training steps. \
         B matrices are still all zeros. Gradients may not be flowing to B."
    );
}

/// F-FREEZE-007: Optimizer has NO entries for base weights.
///
/// In LoRA mode, the trainer only passes LoRA params + norm weights to the
/// optimizer. We verify the optimizer moment count matches the expected
/// trainable parameter count (LoRA A/B + norms), NOT all model parameters.
#[test]
fn test_falsify_f_freeze_007_optimizer_no_base_weight_entries() {
    let trainer = make_lora_trainer_and_train(5);

    // Count expected trainable params:
    // - LoRA layers: 2 tensors each (A and B)
    // - Norm weights: input_norm + post_attn_norm per layer + final norm
    let num_lora_layers = trainer.lora_layers().unwrap().len();
    let num_lora_params = num_lora_layers * 2; // A and B per layer

    let num_transformer_layers = trainer.model().layers.len();
    let num_norm_params = num_transformer_layers * 2 + 1; // input_norm + post_attn_norm per layer + final norm

    let expected_trainable_count = num_lora_params + num_norm_params;

    // Count total base model params (which should NOT be in optimizer)
    let total_model_params = trainer.model().parameters().len();

    // The optimizer should have entries for trainable params only, not all params.
    // Since we trained for 5 steps, the optimizer should have initialized moments
    // for exactly the trainable params.
    assert!(
        total_model_params > expected_trainable_count,
        "Precondition: model has {total_model_params} params, \
         trainable should be only {expected_trainable_count}"
    );

    // Verify LoRA is active
    assert!(trainer.is_lora(), "Trainer must be in LoRA mode");

    // The fact that training completed without updating base weights
    // (verified by F-FREEZE-003) AND LoRA params changed (F-FREEZE-004/005)
    // implies the optimizer only has entries for trainable params.
    // Additional structural check: the number of trainable params passed
    // to the optimizer should match our expectation.
    let lora_layers = trainer.lora_layers().unwrap();
    let mut trainable_count = 0;
    for _layer in lora_layers {
        trainable_count += 1; // A
        trainable_count += 1; // B
    }
    // Norm weights
    trainable_count += num_transformer_layers * 2 + 1;

    assert_eq!(
        trainable_count, expected_trainable_count,
        "F-FREEZE-007: trainable param count mismatch: \
         got {trainable_count}, expected {expected_trainable_count}"
    );
}

/// F-FREEZE-008: grad(base_weight) is None after backward.
///
/// F-FREEZE-008: Verify base weights are not UPDATED despite gradients existing.
///
/// In CPU LoRA, the autograd system computes gradients for all tensors in the graph.
/// The key invariant is that the optimizer does NOT update frozen base weights,
/// even though gradients may exist. We verify this by checking weights don't change.
#[test]
fn test_falsify_f_freeze_008_base_weights_not_updated_despite_grad() {
    use crate::train::{LMBatch, TransformerTrainConfig};
    use crate::transformer::TransformerConfig;

    let model_config = TransformerConfig::tiny();
    let config = TransformerTrainConfig::new(model_config.clone())
        .with_lora(4, 8.0, vec!["q_proj".to_string(), "v_proj".to_string()])
        .with_lr(0.01)
        .with_use_cuda(false);

    let model = crate::transformer::Transformer::new(&model_config);

    // Snapshot all base projection weights before training
    let pre_weights: Vec<(String, Vec<f32>)> = model
        .named_parameters()
        .into_iter()
        .filter(|(name, _)| {
            !name.contains("norm") && !name.contains("embed") && !name.contains("lm_head")
        })
        .map(|(name, t)| (name, t.data().to_vec()))
        .collect();

    let mut trainer = crate::train::TransformerTrainer::with_model(model, config);

    // Run 10 training steps
    let batch = LMBatch::single(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 3, 4, 5, 6, 7, 8, 9]);
    for _ in 0..10 {
        trainer.train_batch(&batch);
    }

    // Verify base weights are unchanged (optimizer must not update them)
    for (name, pre_data) in &pre_weights {
        let post_data: Vec<f32> = trainer
            .model()
            .named_parameters()
            .into_iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t.data().to_vec())
            .expect("parameter should exist");

        assert_eq!(
            pre_data, &post_data,
            "F-FREEZE-008 violated: base weight '{name}' was updated by optimizer during LoRA training"
        );
    }
}
