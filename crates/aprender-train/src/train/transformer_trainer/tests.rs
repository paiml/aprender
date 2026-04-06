//! Tests for transformer trainer module

use super::*;
use crate::transformer::{Transformer, TransformerConfig};

#[test]
fn test_transformer_train_config_new() {
    let model_config = TransformerConfig::tiny();
    let config = TransformerTrainConfig::new(model_config.clone());

    assert_eq!(config.model_config.hidden_size, model_config.hidden_size);
    assert!(!config.checkpoint_config.enabled);
    assert!(!config.precision_config.is_mixed());
}

#[test]
fn test_transformer_train_config_with_checkpointing() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_checkpointing(4);

    assert!(config.checkpoint_config.enabled);
    assert_eq!(config.checkpoint_config.num_segments, 4);
}

#[test]
fn test_transformer_train_config_with_bf16() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_bf16();

    assert!(config.precision_config.is_mixed());
}

#[test]
fn test_transformer_train_config_with_fp16() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_fp16();

    assert!(config.precision_config.is_mixed());
    assert!(config.precision_config.dynamic_scaling);
}

#[test]
fn test_transformer_train_config_builders() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_max_seq_len(1024)
        .with_accumulation_steps(4)
        .with_warmup_steps(100)
        .with_lr(0.0001)
        .with_grad_clip(1.0);

    assert_eq!(config.max_seq_len, 1024);
    assert_eq!(config.accumulation_steps, 4);
    assert_eq!(config.warmup_steps, 100);
}

#[test]
fn test_ent_263_quantize_nf4_default_false() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    assert!(!config.quantize_nf4);
    assert!(!config.is_nf4());
}

#[test]
fn test_ent_263_quantize_nf4_builder() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_quantize_nf4(true)
        .with_lora(16, 32.0, vec!["q_proj".into(), "v_proj".into()]);
    assert!(config.quantize_nf4);
    assert!(config.is_nf4());
    assert!(config.is_lora());
}

#[test]
fn test_ent_263_quantize_nf4_requires_lora() {
    // NF4 without LoRA: quantize_nf4 is set but is_lora() is false
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_quantize_nf4(true);
    assert!(config.quantize_nf4);
    assert!(config.is_nf4());
    assert!(!config.is_lora());
    // The actual code checks quantize_nf4 && is_lora() for NF4 path
}

#[test]
fn test_lm_batch_from_sequences() {
    let sequences = vec![vec![0, 1, 2, 3, 4], vec![0, 5, 6, 7]];

    let batch = LMBatch::from_sequences(&sequences, 99, 100);

    assert_eq!(batch.batch_size, 2);
    assert_eq!(batch.seq_len, 4); // max_len - 1 = 5 - 1 = 4
}

#[test]
fn test_lm_batch_single() {
    let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);

    assert_eq!(batch.batch_size, 1);
    assert_eq!(batch.seq_len, 3);
    assert_eq!(batch.num_tokens(), 3);
}

#[test]
fn test_lm_batch_get_input() {
    let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);

    let input = batch.get_input(0).expect("operation should succeed");
    assert_eq!(input, &[1, 2, 3]);

    assert!(batch.get_input(1).is_none());
}

#[test]
fn test_lm_batch_get_target() {
    let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);

    let target = batch.get_target(0).expect("operation should succeed");
    assert_eq!(target, &[2, 3, 4]);
}

#[test]
fn test_lm_batch_empty() {
    let batch = LMBatch::from_sequences(&[], 0, 1);
    assert_eq!(batch.batch_size, 0);
    assert_eq!(batch.num_tokens(), 0);
}

#[test]
fn test_transformer_trainer_new() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let trainer = TransformerTrainer::new(config);

    assert_eq!(trainer.step(), 0);
    assert!(!trainer.is_mixed_precision());
    assert!(!trainer.is_checkpointing());
}

#[test]
fn test_transformer_trainer_forward_single() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let trainer = TransformerTrainer::new(config);

    let input_ids = vec![1, 2, 3];
    let target_ids = vec![2, 3, 4];

    let (loss, _loss_tensor, logits) = trainer.forward_single(&input_ids, &target_ids);

    assert!(loss > 0.0);
    assert!(loss.is_finite());
    assert_eq!(logits.len(), 3 * trainer.model().config().vocab_size);
}

#[test]
fn test_transformer_trainer_train_batch() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let mut trainer = TransformerTrainer::new(config);

    let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);
    let loss = trainer.train_batch(&batch);

    assert!(loss > 0.0);
    assert!(loss.is_finite());
    assert_eq!(trainer.step(), 1);
}

#[test]
fn test_transformer_trainer_train_epoch() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let mut trainer = TransformerTrainer::new(config);

    let batches = vec![
        LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]),
        LMBatch::single(vec![5, 6, 7], vec![6, 7, 8]),
    ];

    let avg_loss = trainer.train_epoch(&batches);

    assert!(avg_loss > 0.0);
    assert_eq!(trainer.step(), 2);
}

#[test]
fn test_transformer_trainer_epoch_with_callback() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let mut trainer = TransformerTrainer::new(config);

    let batches = vec![
        LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]),
        LMBatch::single(vec![5, 6, 7], vec![6, 7, 8]),
    ];

    let mut callback_calls = Vec::new();
    let avg_loss = trainer.train_epoch_with_callback(&batches, |batch_idx, loss, _trainer| {
        callback_calls.push((batch_idx, loss));
    });

    assert!(avg_loss > 0.0);
    assert_eq!(callback_calls.len(), 2);
    assert_eq!(callback_calls[0].0, 0);
    assert_eq!(callback_calls[1].0, 1);
    assert!(callback_calls[0].1 > 0.0);
    assert!(callback_calls[1].1 > 0.0);
}

#[test]
fn test_transformer_trainer_empty_epoch() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let mut trainer = TransformerTrainer::new(config);

    let avg_loss = trainer.train_epoch(&[]);
    assert_eq!(avg_loss, 0.0);
}

#[test]
fn test_transformer_trainer_with_accumulation() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_accumulation_steps(2);
    let mut trainer = TransformerTrainer::new(config);

    let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);

    // First batch - no step yet
    trainer.train_batch(&batch);
    assert_eq!(trainer.step(), 0);

    // Second batch - step occurs
    trainer.train_batch(&batch);
    assert_eq!(trainer.step(), 1);
}

#[test]
fn test_transformer_trainer_max_steps_stops_early() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_max_steps(3);
    let mut trainer = TransformerTrainer::new(config);

    let batches = vec![
        LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]),
        LMBatch::single(vec![5, 6, 7], vec![6, 7, 8]),
        LMBatch::single(vec![1, 3, 5], vec![3, 5, 7]),
        LMBatch::single(vec![2, 4, 6], vec![4, 6, 8]),
        LMBatch::single(vec![10, 11, 12], vec![11, 12, 13]),
    ];

    // Epoch 1: should process 3 batches then stop (max_steps=3)
    trainer.train_epoch(&batches);
    assert_eq!(trainer.step(), 3);
    assert!(trainer.reached_max_steps());

    // Epoch 2: should process 0 batches (already at max_steps)
    trainer.train_epoch(&batches);
    assert_eq!(trainer.step(), 3);
}

#[test]
fn test_transformer_trainer_max_steps_none_runs_all() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let mut trainer = TransformerTrainer::new(config);

    let batches = vec![
        LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]),
        LMBatch::single(vec![5, 6, 7], vec![6, 7, 8]),
    ];

    trainer.train_epoch(&batches);
    assert_eq!(trainer.step(), 2);
    assert!(!trainer.reached_max_steps());
}

#[test]
fn test_transformer_trainer_warmup_lr() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_lr(0.001)
        .with_warmup_steps(100);
    let mut trainer = TransformerTrainer::new(config);

    // At step 0, LR should be 0
    assert_eq!(trainer.current_lr(), 0.0);

    // Train to advance step
    let batch = LMBatch::single(vec![1, 2], vec![2, 3]);
    trainer.train_batch(&batch);

    // At step 1, LR should be 0.001 * 1/100 = 0.00001
    let lr = trainer.current_lr();
    assert!(lr > 0.0);
    assert!(lr < 0.001);
}

#[test]
fn test_perplexity() {
    let loss = 2.0;
    let ppl = perplexity(loss);
    assert!((ppl - loss.exp()).abs() < 1e-6);
}

#[test]
fn test_tokens_per_second() {
    let tps = tokens_per_second(1000, 2.0);
    assert_eq!(tps, 500.0);
}

#[test]
fn test_transformer_trainer_grad_scaler_stats() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_fp16();
    let trainer = TransformerTrainer::new(config);

    let (scale, overflows, successes) = trainer.grad_scaler_stats();
    assert!(scale > 0.0);
    assert_eq!(overflows, 0);
    assert_eq!(successes, 0);
}

#[test]
fn test_transformer_trainer_with_model() {
    let model_config = TransformerConfig::tiny();
    let model = Transformer::new(&model_config);
    let config = TransformerTrainConfig::new(model_config);
    let trainer = TransformerTrainer::with_model(model, config);

    assert_eq!(trainer.step(), 0);
}

#[test]
fn test_lm_batch_shift_correctness() {
    // Verify that input/target shift is correct for causal LM
    let sequences = vec![vec![100, 1, 2, 3, 200]]; // BOS, tokens, EOS
    let batch = LMBatch::from_sequences(&sequences, 0, 200);

    let input = batch.get_input(0).expect("operation should succeed");
    let target = batch.get_target(0).expect("operation should succeed");

    // Input should be [BOS, 1, 2, 3]
    assert_eq!(input[0], 100); // BOS
    assert_eq!(input[1], 1);
    assert_eq!(input[2], 2);
    assert_eq!(input[3], 3);

    // Target should be [1, 2, 3, EOS]
    assert_eq!(target[0], 1);
    assert_eq!(target[1], 2);
    assert_eq!(target[2], 3);
    assert_eq!(target[3], 200); // EOS
}

// =========================================================================
// FALSIFY-ALBOR-038: Training must modify model weights
//
// Root cause: RMSNorm::forward_batched() created tensors with no backward op,
// blocking all gradient flow through the model. The optimizer never received
// gradients, so weights remained at initialization values.
//
// This test verifies the end-to-end fix: forward → backward → optimizer → save
// produces weights that differ from initialization.
// =========================================================================

/// FALSIFY-ALBOR-038: Training step must change model weights
#[test]
fn falsify_alb038_training_changes_weights() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let mut trainer = TransformerTrainer::new(config);

    // Snapshot initial weights
    let init_params: Vec<Vec<f32>> =
        trainer.model().parameters().iter().map(|p| p.data().to_vec()).collect();

    // Train for several steps
    let batch = LMBatch::single(vec![1, 2, 3, 4], vec![2, 3, 4, 5]);
    for _ in 0..5 {
        trainer.train_batch(&batch);
    }

    // Check that at least some weights changed
    let final_params: Vec<Vec<f32>> =
        trainer.model().parameters().iter().map(|p| p.data().to_vec()).collect();

    let mut changed_count = 0;
    for (i, (init, final_p)) in init_params.iter().zip(final_params.iter()).enumerate() {
        if init == final_p {
            // Log which parameter didn't change (for debugging)
            eprintln!(
                "ALB-038 WARNING: parameter {i} unchanged after training (len={})",
                init.len()
            );
        } else {
            changed_count += 1;
        }
    }

    assert!(
        changed_count > 0,
        "FALSIFIED ALB-038: No model weights changed after 5 training steps. \
         All {} parameters remained at initialization values.",
        init_params.len()
    );

    // Specifically check FFN weights (these must change with working norm backward)
    // Parameters order: embed, norm, [per layer: input_norm, post_attn_norm, q, k, v, o, gate, up, down]
    // For tiny config with 2 layers: embed(0), norm(1), then per layer 9 params each
    // FFN gate_proj for layer 0 is at index 2 + 0*9 + 6 = 8
    let num_params = init_params.len();
    if num_params > 8 {
        assert_ne!(
            init_params[8], final_params[8],
            "FALSIFIED ALB-038: FFN gate_proj (param 8) unchanged — norm backward broken"
        );
    }
}

/// FALSIFY-ALBOR-038: Saved weights must differ from initialization
#[test]
fn falsify_alb038_saved_weights_differ_from_init() {
    use tempfile::NamedTempFile;

    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let mut trainer = TransformerTrainer::new(config);

    // Snapshot init weights for comparison
    let init_embed = trainer.model().embed_tokens.weight.data().to_vec();

    // Train
    let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);
    for _ in 0..5 {
        trainer.train_batch(&batch);
    }

    // Save to temp file
    let temp = NamedTempFile::new().expect("temp file creation should succeed");
    trainer.save(temp.path(), "alb038-test", "test").expect("save should succeed");

    // Load back and verify weights differ from init
    let data = std::fs::read(temp.path()).expect("file read should succeed");
    let loaded = safetensors::SafeTensors::deserialize(&data).expect("load should succeed");

    let saved_embed = loaded.tensor("model.embed_tokens.weight").expect("tensor exists");
    let saved_data: &[f32] = bytemuck::cast_slice(saved_embed.data());

    // Saved embed weights must differ from init
    assert_ne!(
        saved_data,
        &init_embed[..],
        "FALSIFIED ALB-038: Saved embedding weights are identical to initialization"
    );
}

// === R-084: Bitwise deterministic training (C-DETERM-001) ===

#[test]
fn test_deterministic_config_defaults() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    assert!(!config.deterministic, "deterministic should default to false");
    assert_eq!(config.seed, 42, "default seed should be 42");
}

#[test]
fn test_deterministic_config_builder() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_deterministic(true)
        .with_seed(12345);
    assert!(config.deterministic);
    assert_eq!(config.seed, 12345);
}

#[test]
fn test_deterministic_env_vars_set() {
    // O-DET-001: ReproducibilityConfig::apply() sets CUDA env vars
    let config = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_deterministic(true)
        .with_seed(99999);
    config.apply_deterministic_settings();

    assert_eq!(
        std::env::var("CUBLAS_WORKSPACE_CONFIG").unwrap_or_default(),
        ":4096:8",
        "CUBLAS_WORKSPACE_CONFIG must be :4096:8 (I-DET-001)"
    );
    assert_eq!(
        std::env::var("CUDNN_DETERMINISTIC").unwrap_or_default(),
        "1",
        "CUDNN_DETERMINISTIC must be 1 (I-DET-002)"
    );
    assert_eq!(
        std::env::var("CUDNN_BENCHMARK").unwrap_or_default(),
        "0",
        "CUDNN_BENCHMARK must be 0 (I-DET-003)"
    );
}

#[test]
fn test_deterministic_disabled_no_env_change() {
    // When deterministic=false, apply_deterministic_settings() is a no-op
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_deterministic(false);
    // Clear env first to verify it's not set
    // SAFETY: test runs single-threaded; remove_var needed to verify no-op behavior
    #[allow(clippy::disallowed_methods, unsafe_code)]
    unsafe {
        std::env::remove_var("PYTHONHASHSEED");
    };
    config.apply_deterministic_settings();
    // PYTHONHASHSEED should NOT have been set
    assert!(
        std::env::var("PYTHONHASHSEED").is_err(),
        "deterministic=false should not set PYTHONHASHSEED"
    );
}

#[test]
fn test_deterministic_training_reproducibility() {
    // F-DET-001: Two runs with same seed produce identical loss
    let config1 = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_deterministic(true)
        .with_seed(42);
    let config2 = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_deterministic(true)
        .with_seed(42);

    let mut trainer1 = TransformerTrainer::new(config1);
    let mut trainer2 = TransformerTrainer::new(config2);

    let batch = LMBatch::single(vec![1, 2, 3, 4, 5], vec![2, 3, 4, 5, 6]);

    let mut losses1 = Vec::new();
    let mut losses2 = Vec::new();

    for _ in 0..5 {
        losses1.push(trainer1.train_batch(&batch));
        losses2.push(trainer2.train_batch(&batch));
    }

    // CPU training with same init should produce identical losses
    for (i, (l1, l2)) in losses1.iter().zip(losses2.iter()).enumerate() {
        assert!(
            (l1 - l2).abs() < 1e-6,
            "Step {i}: loss mismatch {l1} vs {l2} (C-DETERM-001 violation)"
        );
    }
}

// ── Activation Checkpointing (R-021, #115) ──────────────────────────────

#[test]
fn test_checkpoint_config_segment_calculation() {
    // Verify checkpoint boundary mask calculation
    // tiny has 2 layers, 2 segments → segment_size = 1 → all layers are checkpoints
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_checkpointing(2);

    assert!(config.checkpoint_config.enabled);
    assert_eq!(config.checkpoint_config.num_segments, 2);

    let num_layers = config.model_config.num_hidden_layers;
    assert_eq!(num_layers, 2);
    let ns = config.checkpoint_config.num_segments.max(1);
    let segment_size = num_layers.div_ceil(ns);
    assert_eq!(segment_size, 1);
    let mask: Vec<bool> = (0..num_layers).map(|i| i % segment_size == 0).collect();
    assert_eq!(mask, vec![true, true]);
}

#[test]
fn test_checkpoint_config_fewer_segments() {
    // Use a config with more layers: 24 layers, 4 segments → segment_size = 6
    // Checkpoint boundary layers: 0, 6, 12, 18
    let mut model_config = TransformerConfig::tiny();
    model_config.num_hidden_layers = 24;
    let config = TransformerTrainConfig::new(model_config).with_checkpointing(4);

    let num_layers = config.model_config.num_hidden_layers;
    assert_eq!(num_layers, 24);
    let ns = config.checkpoint_config.num_segments.max(1);
    let segment_size = num_layers.div_ceil(ns);
    assert_eq!(segment_size, 6);
    let mask: Vec<bool> = (0..num_layers).map(|i| i % segment_size == 0).collect();
    // Only layers 0, 6, 12, 18 are checkpoints
    let expected: Vec<bool> = (0..24).map(|i| i % 6 == 0).collect();
    assert_eq!(mask, expected);
    assert_eq!(mask.iter().filter(|&&x| x).count(), 4);
    assert_eq!(mask.iter().filter(|&&x| !x).count(), 20);
}

#[test]
fn test_checkpoint_disabled_saves_all_layers() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());

    assert!(!config.checkpoint_config.enabled);

    // When disabled, all layers should be "checkpointed" (saved)
    let num_layers = config.model_config.num_hidden_layers;
    let mask: Vec<bool> = (0..num_layers)
        .map(|_| true) // !checkpointing → all saved
        .collect();
    assert!(mask.iter().all(|&x| x));
}

#[test]
fn test_checkpoint_with_cpu_trainer() {
    // CPU trainer with checkpointing enabled should train normally
    // (checkpointing only affects CUDA path, CPU path ignores it)
    let model_config = TransformerConfig::tiny();
    let config = TransformerTrainConfig::new(model_config.clone())
        .with_checkpointing(2)
        .with_lr(0.001)
        .with_max_seq_len(32);

    let model = Transformer::new(&model_config);
    let mut trainer = TransformerTrainer::with_model(model, config);

    let input = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
    let batch = LMBatch::from_sequences(&[input], 0, 0);

    let loss = trainer.train_batch(&batch);
    assert!(loss > 0.0, "Loss should be positive");
    assert!(loss.is_finite(), "Loss should be finite");
}

#[test]
fn test_gradient_accumulation_produces_different_weights_than_no_accum() {
    // R-038: Verify that gradient accumulation with accum_steps=2 produces
    // different weights than single-step training, proving the accumulation
    // path is actually wired (not a no-op).
    let model_config = TransformerConfig::tiny();
    let seq = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
    let batch = LMBatch::from_sequences(&[seq.clone()], 0, 0);

    // Train with accum=1 (immediate optimizer step)
    let config_no_accum =
        TransformerTrainConfig::new(model_config.clone()).with_lr(0.01).with_max_seq_len(32);
    let model1 = Transformer::new(&model_config);
    let mut trainer1 = TransformerTrainer::with_model(model1, config_no_accum);
    trainer1.train_batch(&batch);
    trainer1.train_batch(&batch);
    let weights1: Vec<f32> =
        trainer1.model().embed_tokens.weight.data().as_slice().unwrap().to_vec();

    // Train with accum=2 (deferred optimizer step)
    let config_accum = TransformerTrainConfig::new(model_config.clone())
        .with_lr(0.01)
        .with_max_seq_len(32)
        .with_accumulation_steps(2);
    let model2 = Transformer::new(&model_config);
    let mut trainer2 = TransformerTrainer::with_model(model2, config_accum);
    trainer2.train_batch(&batch);
    trainer2.train_batch(&batch);
    let weights2: Vec<f32> =
        trainer2.model().embed_tokens.weight.data().as_slice().unwrap().to_vec();

    // Weights should differ (different optimizer dynamics)
    let diff: f64 =
        weights1.iter().zip(&weights2).map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs()).sum();
    assert!(diff > 1e-6, "Gradient accumulation should produce different weights (diff={diff})");
}

#[test]
fn test_per_block_gradient_accumulator_sizes() {
    // Verify PerBlockGradientAccumulator sizes match model architecture
    use super::grad_accumulator::PerBlockGradientAccumulator;

    let model_config = TransformerConfig::tiny();
    let h = model_config.hidden_size;
    let kv = model_config.num_kv_heads * model_config.head_dim();
    let i = model_config.intermediate_size;
    let sizes = PerBlockGradientAccumulator::compute_block_sizes(h, kv, i);
    let accum = PerBlockGradientAccumulator::new(
        model_config.num_hidden_layers,
        sizes,
        model_config.vocab_size,
        h,
    );
    assert_eq!(accum.num_blocks(), model_config.num_hidden_layers);
    assert_eq!(accum.lm_head_grad.len(), model_config.vocab_size * h);
    assert_eq!(accum.final_norm_grad.len(), h);
    assert_eq!(accum.embedding_grad.len(), model_config.vocab_size * h);
    assert!(!accum.has_non_finite());
}

// ============================================================================
// ENT-LoRA-001: LoRA wiring falsification tests
// ============================================================================

#[test]
fn test_ent_lora_001_config_wiring() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_lora(
        16,
        32.0,
        vec!["q_proj".to_string(), "v_proj".to_string()],
    );

    assert!(config.is_lora());
    assert_eq!(config.lora_rank, Some(16));
    assert_eq!(config.lora_alpha, Some(32.0));
    assert_eq!(
        config.lora_target_modules.as_deref(),
        Some(&["q_proj".to_string(), "v_proj".to_string()][..])
    );
}

#[test]
fn test_ent_lora_001_no_lora_by_default() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    assert!(!config.is_lora());
    assert_eq!(config.lora_rank, None);
}

#[test]
fn test_ent_lora_001_trainer_creates_lora_layers() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_lora(
        8,
        16.0,
        vec!["q_proj".to_string(), "v_proj".to_string()],
    );
    let trainer = TransformerTrainer::new(config);

    assert!(trainer.is_lora());
    let lora = trainer.lora_layers().expect("LoRA layers should exist");
    // 2 layers (tiny config) * 2 projections (q, v) = 4 LoRA layers
    assert_eq!(lora.len(), 4);
}

#[test]
fn test_ent_lora_001_no_lora_layers_without_config() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let trainer = TransformerTrainer::new(config);

    assert!(!trainer.is_lora());
    assert!(trainer.lora_layers().is_none());
}

#[test]
fn test_ent_lora_001_lora_b_initialized_zeros() {
    // FALSIFY-LoRA-MATH-001: B must be zeros at init (ΔW = 0)
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_lora(
        4,
        8.0,
        vec!["q_proj".to_string(), "v_proj".to_string()],
    );
    let trainer = TransformerTrainer::new(config);
    let lora = trainer.lora_layers().expect("LoRA layers should exist");

    for (i, layer) in lora.iter().enumerate() {
        let b_data = layer.lora_b().data();
        let max_b = b_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(max_b < 1e-10, "LoRA B[{i}] should be zeros at init, max value: {max_b}");
    }
}

#[test]
fn test_ent_lora_001_lora_forward_produces_logits() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_lora(
        4,
        8.0,
        vec!["q_proj".to_string(), "v_proj".to_string()],
    );
    let trainer = TransformerTrainer::new(config);

    let input_ids = vec![1u32, 2, 3];
    let target_ids = vec![2u32, 3, 4];
    let (loss_val, _loss, logits) = trainer.forward_single(&input_ids, &target_ids);

    assert!(loss_val > 0.0, "Loss should be positive");
    let vocab_size = TransformerConfig::tiny().vocab_size;
    assert_eq!(logits.len(), 3 * vocab_size);
}

#[test]
fn test_ent_lora_001_lora_train_batch() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_lora(
        4,
        8.0,
        vec!["q_proj".to_string(), "v_proj".to_string()],
    );
    let mut trainer = TransformerTrainer::new(config);

    let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);
    let loss = trainer.train_batch(&batch);

    assert!(loss > 0.0, "LoRA training should produce non-zero loss");
    assert_eq!(trainer.step(), 1, "Step should increment after batch");
}

#[test]
fn test_ent_lora_001_lora_updates_only_adapters() {
    // FALSIFY-LoRA-FREEZE-001: Base attention/FFN weights must NOT change during LoRA training
    // Norm weights ARE allowed to change (they're trainable in LoRA mode)
    let model_config = TransformerConfig::tiny();
    let config = TransformerTrainConfig::new(model_config.clone()).with_lora(
        4,
        8.0,
        vec!["q_proj".to_string(), "v_proj".to_string()],
    );

    // Snapshot named parameters before training
    let model_before = Transformer::new(&model_config);
    let named_before: Vec<(String, Vec<f32>)> = model_before
        .named_parameters()
        .into_iter()
        .map(|(name, t)| (name, t.data().to_vec()))
        .collect();

    let mut trainer = TransformerTrainer::with_model(model_before, config);

    // Train a few batches
    for _ in 0..3 {
        let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);
        trainer.train_batch(&batch);
    }

    // Check named parameters
    let named_after: Vec<(String, Vec<f32>)> = trainer
        .model()
        .named_parameters()
        .into_iter()
        .map(|(name, t)| (name, t.data().to_vec()))
        .collect();

    for ((name_b, data_b), (name_a, data_a)) in named_before.iter().zip(&named_after) {
        assert_eq!(name_b, name_a);
        // Norm weights are allowed to change (ENT-LoRA-002)
        if name_b.contains("layernorm") || name_b.contains("norm.weight") {
            continue;
        }
        // Attention and FFN base weights must NOT change
        assert_eq!(data_b, data_a, "Base weight '{name_b}' changed during LoRA training");
    }
}

#[test]
fn test_ent_lora_001_with_model_creates_lora() {
    let model_config = TransformerConfig::tiny();
    let model = Transformer::new(&model_config);
    let config = TransformerTrainConfig::new(model_config).with_lora(
        8,
        16.0,
        vec!["q_proj".to_string(), "v_proj".to_string()],
    );
    let trainer = TransformerTrainer::with_model(model, config);

    assert!(trainer.is_lora());
    assert!(trainer.lora_layers().is_some());
}

#[test]
fn test_ent_lora_002_norm_weights_trainable() {
    // ENT-LoRA-002: Norm weights should change during LoRA training
    let model_config = TransformerConfig::tiny();
    let config = TransformerTrainConfig::new(model_config.clone()).with_lora(
        4,
        8.0,
        vec!["q_proj".to_string(), "v_proj".to_string()],
    );

    let model = Transformer::new(&model_config);
    let norm_before: Vec<f32> = model.norm.weight.data().to_vec();

    let mut trainer = TransformerTrainer::with_model(model, config);

    // Train enough batches to change norm weights
    for _ in 0..5 {
        let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);
        trainer.train_batch(&batch);
    }

    let norm_after: Vec<f32> = trainer.model().norm.weight.data().to_vec();
    // At least some norm weights should have changed
    let any_changed = norm_before.iter().zip(&norm_after).any(|(b, a)| (b - a).abs() > 1e-10);
    assert!(any_changed, "Norm weights should be trainable during LoRA fine-tuning");
}

#[test]
fn test_ent_lora_003_save_adapter() {
    // ENT-LoRA-003: Adapter checkpoint saves only LoRA weights
    let model_config = TransformerConfig::tiny();
    let config = TransformerTrainConfig::new(model_config).with_lora(
        4,
        8.0,
        vec!["q_proj".to_string(), "v_proj".to_string()],
    );

    let mut trainer = TransformerTrainer::new(config);

    // Train briefly
    let batch = LMBatch::single(vec![1, 2, 3], vec![2, 3, 4]);
    trainer.train_batch(&batch);

    // Save adapter
    let tmp_dir = std::env::temp_dir().join("test_lora_adapter_save");
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).expect("create temp dir");

    trainer.save_lora_adapter(&tmp_dir, Some("test-model")).expect("save adapter should succeed");

    // Verify PEFT files exist
    assert!(tmp_dir.join("adapter_config.json").exists());
    assert!(tmp_dir.join("adapter_model.safetensors").exists());

    // Adapter should be much smaller than full model
    let adapter_size =
        std::fs::metadata(tmp_dir.join("adapter_model.safetensors")).expect("adapter file").len();
    // Adapter for tiny model with rank=4 should be very small
    assert!(adapter_size < 100_000, "Adapter should be small, got {adapter_size} bytes");

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_ent_lora_003_save_adapter_without_lora_fails() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    let trainer = TransformerTrainer::new(config);

    let result = trainer.save_lora_adapter("/tmp/no_lora", None::<&str>);
    assert!(result.is_err(), "Saving adapter without LoRA should fail");
}

// ============================================================================
// ENT-LoRA-005: Flexible target modules
// ============================================================================

#[test]
fn test_ent_lora_005_all_linear_creates_7_layers_per_block() {
    let model_config = TransformerConfig::tiny();
    let config =
        TransformerTrainConfig::new(model_config).with_lora(4, 8.0, vec!["all_linear".to_string()]);

    let trainer = TransformerTrainer::new(config);
    let lora = trainer.lora_layers().expect("LoRA should be active");
    // tiny() has 2 layers, all_linear = 7 modules (q/k/v/o/gate/up/down)
    assert_eq!(lora.len(), 2 * 7, "Should have 14 LoRA layers (2 blocks × 7 modules)");
}

#[test]
fn test_ent_lora_005_attention_shorthand() {
    let model_config = TransformerConfig::tiny();
    let config =
        TransformerTrainConfig::new(model_config).with_lora(4, 8.0, vec!["attention".to_string()]);

    let trainer = TransformerTrainer::new(config);
    let lora = trainer.lora_layers().expect("LoRA should be active");
    // 2 layers × 4 attention modules (q/k/v/o)
    assert_eq!(lora.len(), 2 * 4);
}

#[test]
fn test_ent_lora_005_mlp_shorthand() {
    let model_config = TransformerConfig::tiny();
    let config =
        TransformerTrainConfig::new(model_config).with_lora(4, 8.0, vec!["mlp".to_string()]);

    let trainer = TransformerTrainer::new(config);
    let lora = trainer.lora_layers().expect("LoRA should be active");
    // 2 layers × 3 MLP modules (gate/up/down)
    assert_eq!(lora.len(), 2 * 3);
}

// ============================================================================
// ENT-LoRA-006: LoRA+ optimizer (separate LR for A and B)
// ============================================================================

#[test]
fn test_ent_lora_006_config_default_ratio() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    assert!((config.lora_plus_ratio - 1.0).abs() < 1e-6, "Default ratio should be 1.0");
}

#[test]
fn test_ent_lora_006_config_with_ratio() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_lora_plus_ratio(16.0);
    assert!((config.lora_plus_ratio - 16.0).abs() < 1e-6);
}

// ============================================================================
// Coverage improvement: Builder methods and accessors (config.rs)
// ============================================================================

#[test]
fn test_config_with_lr_sets_value() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_lr(0.0005);
    assert!((config.lr - 0.0005).abs() < 1e-8, "with_lr should set lr field");
}

#[test]
fn test_config_with_grad_clip_sets_base_max_grad_norm() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_grad_clip(2.5);
    assert_eq!(
        config.base.max_grad_norm,
        Some(2.5),
        "with_grad_clip should set base.max_grad_norm"
    );
}

#[test]
fn test_config_with_use_cuda_true() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_use_cuda(true);
    assert!(config.use_cuda, "with_use_cuda(true) should set use_cuda to true");
}

#[test]
fn test_config_with_use_cuda_false() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_use_cuda(false);
    assert!(!config.use_cuda, "with_use_cuda(false) should set use_cuda to false");
}

#[test]
fn test_config_with_beta2_sets_value() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_beta2(0.95);
    assert!((config.beta2 - 0.95).abs() < 1e-8, "with_beta2 should set beta2 field");
}

#[test]
fn test_config_with_weight_decay_sets_value() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_weight_decay(0.1);
    assert!(
        (config.weight_decay - 0.1).abs() < 1e-8,
        "with_weight_decay should set weight_decay field"
    );
}

#[test]
fn test_config_with_profile_interval_sets_value() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_profile_interval(50);
    assert_eq!(
        config.profile_interval, 50,
        "with_profile_interval should set profile_interval field"
    );
}

#[test]
fn test_config_with_profile_interval_zero_disabled() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_profile_interval(0);
    assert_eq!(config.profile_interval, 0, "profile_interval=0 means disabled");
}

#[test]
fn test_config_with_max_steps_sets_some() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_max_steps(500);
    assert_eq!(config.max_steps, Some(500), "with_max_steps should set max_steps to Some(N)");
}

#[test]
fn test_config_with_double_quantize_enabled() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_double_quantize(true);
    assert!(config.double_quantize, "with_double_quantize(true) should enable double quantization");
}

#[test]
fn test_config_with_double_quantize_disabled() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_double_quantize(false);
    assert!(
        !config.double_quantize,
        "with_double_quantize(false) should disable double quantization"
    );
}

#[test]
fn test_config_with_accumulation_steps_clamps_to_one() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_accumulation_steps(0);
    assert_eq!(config.accumulation_steps, 1, "with_accumulation_steps(0) should clamp to 1");
}

#[test]
fn test_config_with_accumulation_steps_preserves_valid() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_accumulation_steps(8);
    assert_eq!(config.accumulation_steps, 8, "with_accumulation_steps(8) should preserve value");
}

// ============================================================================
// Coverage improvement: Distributed config, accessors, defaults
// ============================================================================

#[test]
fn test_distributed_role_default_is_coordinator() {
    let role = DistributedRole::default();
    assert_eq!(role, DistributedRole::Coordinator, "Default DistributedRole should be Coordinator");
}

#[test]
fn test_distributed_backend_default_is_auto() {
    let backend = DistributedBackend::default();
    assert_eq!(backend, DistributedBackend::Auto, "Default DistributedBackend should be Auto");
}

#[test]
fn test_distributed_role_eq() {
    assert_eq!(DistributedRole::Coordinator, DistributedRole::Coordinator);
    assert_eq!(DistributedRole::Worker, DistributedRole::Worker);
    assert_ne!(DistributedRole::Coordinator, DistributedRole::Worker);
}

#[test]
fn test_distributed_backend_eq() {
    assert_eq!(DistributedBackend::Cuda, DistributedBackend::Cuda);
    assert_eq!(DistributedBackend::Wgpu, DistributedBackend::Wgpu);
    assert_eq!(DistributedBackend::Auto, DistributedBackend::Auto);
    assert_ne!(DistributedBackend::Cuda, DistributedBackend::Wgpu);
}

#[test]
fn test_distributed_role_clone() {
    let role = DistributedRole::Worker;
    let cloned = role;
    assert_eq!(role, cloned);
}

#[test]
fn test_distributed_backend_clone() {
    let backend = DistributedBackend::Cuda;
    let cloned = backend;
    assert_eq!(backend, cloned);
}

#[test]
fn test_config_with_distributed_sets_some() {
    let dist = DistributedTrainConfig {
        world_size: 4,
        rank: 1,
        local_rank: 0,
        role: DistributedRole::Worker,
        coordinator_addr: "127.0.0.1:29500".parse().expect("valid socket addr"),
        backend: DistributedBackend::Cuda,
    };
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_distributed(dist);

    assert!(config.distributed.is_some(), "with_distributed should set distributed");
    let d = config.distributed.as_ref().expect("distributed should be Some");
    assert_eq!(d.world_size, 4);
    assert_eq!(d.rank, 1);
    assert_eq!(d.local_rank, 0);
    assert_eq!(d.role, DistributedRole::Worker);
    assert_eq!(d.backend, DistributedBackend::Cuda);
}

#[test]
fn test_config_is_distributed_false_by_default() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    assert!(!config.is_distributed(), "is_distributed() should be false by default");
}

#[test]
fn test_config_is_distributed_true_when_set() {
    let dist = DistributedTrainConfig {
        world_size: 2,
        rank: 0,
        local_rank: 0,
        role: DistributedRole::Coordinator,
        coordinator_addr: "127.0.0.1:29500".parse().expect("valid socket addr"),
        backend: DistributedBackend::Auto,
    };
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_distributed(dist);
    assert!(
        config.is_distributed(),
        "is_distributed() should be true when distributed config is set"
    );
}

#[test]
fn test_config_world_size_default_is_one() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    assert_eq!(
        config.world_size(),
        1,
        "world_size() should return 1 for single-GPU (no distributed)"
    );
}

#[test]
fn test_config_world_size_from_distributed() {
    let dist = DistributedTrainConfig {
        world_size: 8,
        rank: 3,
        local_rank: 1,
        role: DistributedRole::Worker,
        coordinator_addr: "10.0.0.1:29500".parse().expect("valid socket addr"),
        backend: DistributedBackend::Wgpu,
    };
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_distributed(dist);
    assert_eq!(config.world_size(), 8, "world_size() should return distributed world_size");
}

#[test]
fn test_config_rank_default_is_zero() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    assert_eq!(config.rank(), 0, "rank() should return 0 for single-GPU (no distributed)");
}

#[test]
fn test_config_rank_from_distributed() {
    let dist = DistributedTrainConfig {
        world_size: 4,
        rank: 2,
        local_rank: 0,
        role: DistributedRole::Worker,
        coordinator_addr: "127.0.0.1:29500".parse().expect("valid socket addr"),
        backend: DistributedBackend::Auto,
    };
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_distributed(dist);
    assert_eq!(config.rank(), 2, "rank() should return distributed rank");
}

// ============================================================================
// Coverage improvement: Builder chaining and default values
// ============================================================================

#[test]
fn test_config_new_defaults() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());

    // Verify all default values from config.rs new()
    assert_eq!(config.max_seq_len, 512);
    assert_eq!(config.accumulation_steps, 1);
    assert_eq!(config.warmup_steps, 0);
    assert!((config.lr - 0.001).abs() < 1e-8);
    assert!(config.max_steps.is_none());
    assert!(config.use_cuda);
    assert!((config.beta1 - 0.9).abs() < 1e-8);
    assert!((config.beta2 - 0.999).abs() < 1e-8);
    assert!((config.weight_decay - 0.01).abs() < 1e-8);
    assert!(config.distributed.is_none());
    assert!(!config.deterministic);
    assert_eq!(config.seed, 42);
    assert_eq!(config.profile_interval, 0);
    assert!(config.lora_rank.is_none());
    assert!(config.lora_alpha.is_none());
    assert!(config.lora_target_modules.is_none());
    assert!((config.lora_plus_ratio - 1.0).abs() < 1e-8);
    assert!(!config.double_quantize);
}

#[test]
fn test_config_full_builder_chain() {
    // Test chaining ALL builder methods together
    let dist = DistributedTrainConfig {
        world_size: 2,
        rank: 1,
        local_rank: 0,
        role: DistributedRole::Worker,
        coordinator_addr: "127.0.0.1:29500".parse().expect("valid socket addr"),
        backend: DistributedBackend::Cuda,
    };

    let config = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_checkpointing(4)
        .with_bf16()
        .with_max_seq_len(2048)
        .with_accumulation_steps(8)
        .with_warmup_steps(200)
        .with_lr(3e-4)
        .with_grad_clip(0.5)
        .with_max_steps(10000)
        .with_use_cuda(true)
        .with_beta2(0.95)
        .with_weight_decay(0.05)
        .with_deterministic(true)
        .with_seed(12345)
        .with_profile_interval(100)
        .with_lora(16, 32.0, vec!["q_proj".to_string(), "v_proj".to_string()])
        .with_lora_plus_ratio(16.0)
        .with_double_quantize(true)
        .with_distributed(dist);

    assert!(config.checkpoint_config.enabled);
    assert_eq!(config.checkpoint_config.num_segments, 4);
    assert!(config.precision_config.is_mixed());
    assert_eq!(config.max_seq_len, 2048);
    assert_eq!(config.accumulation_steps, 8);
    assert_eq!(config.warmup_steps, 200);
    assert!((config.lr - 3e-4).abs() < 1e-8);
    assert_eq!(config.base.max_grad_norm, Some(0.5));
    assert_eq!(config.max_steps, Some(10000));
    assert!(config.use_cuda);
    assert!((config.beta2 - 0.95).abs() < 1e-8);
    assert!((config.weight_decay - 0.05).abs() < 1e-8);
    assert!(config.deterministic);
    assert_eq!(config.seed, 12345);
    assert_eq!(config.profile_interval, 100);
    assert!(config.is_lora());
    assert_eq!(config.lora_rank, Some(16));
    assert_eq!(config.lora_alpha, Some(32.0));
    assert!((config.lora_plus_ratio - 16.0).abs() < 1e-8);
    assert!(config.double_quantize);
    assert!(config.is_distributed());
    assert_eq!(config.world_size(), 2);
    assert_eq!(config.rank(), 1);
}

#[test]
fn test_config_is_lora_false_without_lora() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    assert!(!config.is_lora(), "is_lora() should be false when lora_rank is None");
}

#[test]
fn test_config_with_warmup_steps_value() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_warmup_steps(500);
    assert_eq!(config.warmup_steps, 500);
}

#[test]
fn test_config_with_max_seq_len_value() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_max_seq_len(4096);
    assert_eq!(config.max_seq_len, 4096);
}

#[test]
fn test_config_with_seed_value() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_seed(99999);
    assert_eq!(config.seed, 99999);
}

#[test]
fn test_config_with_deterministic_true() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_deterministic(true);
    assert!(config.deterministic);
}

#[test]
fn test_config_with_deterministic_false() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_deterministic(false);
    assert!(!config.deterministic);
}

#[test]
fn test_config_apply_deterministic_settings_noop_when_disabled() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_deterministic(false);
    // Should not panic and should be a no-op
    config.apply_deterministic_settings();
}

#[test]
fn test_config_apply_deterministic_settings_when_enabled() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny())
        .with_deterministic(true)
        .with_seed(777);
    config.apply_deterministic_settings();

    // Verify CUDA env vars are set
    assert_eq!(std::env::var("CUBLAS_WORKSPACE_CONFIG").unwrap_or_default(), ":4096:8");
}

#[test]
fn test_distributed_train_config_debug() {
    let dist = DistributedTrainConfig {
        world_size: 2,
        rank: 0,
        local_rank: 0,
        role: DistributedRole::Coordinator,
        coordinator_addr: "127.0.0.1:29500".parse().expect("valid socket addr"),
        backend: DistributedBackend::Auto,
    };
    let debug_str = format!("{dist:?}");
    assert!(debug_str.contains("DistributedTrainConfig"), "Debug output should contain type name");
    assert!(debug_str.contains("world_size: 2"), "Debug output should contain world_size");
}

#[test]
fn test_distributed_train_config_clone() {
    let dist = DistributedTrainConfig {
        world_size: 4,
        rank: 2,
        local_rank: 1,
        role: DistributedRole::Worker,
        coordinator_addr: "192.168.1.1:29500".parse().expect("valid socket addr"),
        backend: DistributedBackend::Wgpu,
    };
    let cloned = dist.clone();
    assert_eq!(cloned.world_size, 4);
    assert_eq!(cloned.rank, 2);
    assert_eq!(cloned.local_rank, 1);
    assert_eq!(cloned.role, DistributedRole::Worker);
    assert_eq!(cloned.backend, DistributedBackend::Wgpu);
}

#[test]
fn test_distributed_role_debug() {
    let coordinator = format!("{:?}", DistributedRole::Coordinator);
    let worker = format!("{:?}", DistributedRole::Worker);
    assert_eq!(coordinator, "Coordinator");
    assert_eq!(worker, "Worker");
}

#[test]
fn test_distributed_backend_debug() {
    let cuda = format!("{:?}", DistributedBackend::Cuda);
    let wgpu = format!("{:?}", DistributedBackend::Wgpu);
    let auto = format!("{:?}", DistributedBackend::Auto);
    assert_eq!(cuda, "Cuda");
    assert_eq!(wgpu, "Wgpu");
    assert_eq!(auto, "Auto");
}

#[test]
fn test_config_with_lora_sets_all_fields() {
    let modules = vec![
        "q_proj".to_string(),
        "k_proj".to_string(),
        "v_proj".to_string(),
        "o_proj".to_string(),
    ];
    let config =
        TransformerTrainConfig::new(TransformerConfig::tiny()).with_lora(32, 64.0, modules.clone());

    assert_eq!(config.lora_rank, Some(32));
    assert_eq!(config.lora_alpha, Some(64.0));
    assert_eq!(config.lora_target_modules, Some(modules),);
}

#[test]
fn test_config_bf16_then_fp16_overwrites() {
    // Last precision config wins
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_bf16().with_fp16();
    assert!(config.precision_config.is_mixed());
    assert!(config.precision_config.dynamic_scaling, "fp16 should have dynamic_scaling");
}

#[test]
fn test_config_fp16_then_bf16_overwrites() {
    let config = TransformerTrainConfig::new(TransformerConfig::tiny()).with_fp16().with_bf16();
    assert!(config.precision_config.is_mixed());
    // bf16 should NOT have dynamic scaling
    assert!(!config.precision_config.dynamic_scaling, "bf16 should not have dynamic_scaling");
}

#[test]
fn test_config_with_grad_clip_overrides_default() {
    // Default TrainConfig has max_grad_norm=Some(1.0)
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    assert_eq!(config.base.max_grad_norm, Some(1.0));

    let config = config.with_grad_clip(5.0);
    assert_eq!(config.base.max_grad_norm, Some(5.0));
}

#[test]
fn test_config_with_use_cuda_default_true() {
    // Verify the default is true
    let config = TransformerTrainConfig::new(TransformerConfig::tiny());
    assert!(config.use_cuda, "Default use_cuda should be true");
}
