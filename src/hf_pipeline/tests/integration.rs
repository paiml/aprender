//! Integration tests and property tests for HuggingFace pipeline
//!
//! ENT-082: Integration tests
//! ENT-083: Property tests

use crate::hf_pipeline::*;

// =========================================================================
// ENT-082: Integration Tests
// =========================================================================

#[test]
fn test_integration_full_pipeline_flow() {
    // 1. Create teacher model
    let teacher = SafeTensorsTeacher::mock(12, 768);

    // 2. Create trainer config
    let config = TrainerConfig::new("teacher/model", "student/model")
        .temperature(4.0)
        .alpha(0.7)
        .with_progressive(vec![(0, 3), (1, 7), (2, 11)])
        .epochs(3);

    // 3. Create trainer
    let mut trainer = DistillationTrainer::new(config, teacher);

    // 4. Simulate training loop
    for epoch in 0..3 {
        for step in 0..10 {
            let loss = 1.0 / (1.0 + (epoch * 10 + step) as f32);
            trainer.simulate_step(loss);
        }
        trainer.simulate_epoch();
    }

    // 5. Verify training state
    assert_eq!(trainer.state().epoch, 3);
    assert_eq!(trainer.state().global_step, 30);
    assert!(trainer.state().avg_loss(10).expect("operation should succeed") < 0.5);
}

#[test]
fn test_integration_dataset_to_batch() {
    // 1. Create dataset
    let dataset = Dataset::mock(100, 64);
    assert_eq!(dataset.len(), 100);

    // 2. Create collator
    let collator = DistillationCollator::new(0).max_length(64);

    // 3. Create batches
    let batches = collator.batch_dataset(&dataset, 16);
    assert_eq!(batches.len(), 7); // 100 / 16 = 6 full + 1 partial

    // 4. Verify batch shapes
    assert_eq!(batches[0].batch_size(), 16);
    assert!(batches[0].max_seq_len() <= 64);
    assert!(batches[0].labels.is_some());
}

#[test]
fn test_integration_config_to_trainer() {
    let yaml = r#"
teacher:
  model_id: "microsoft/codebert-base"
student:
  model_id: "distilbert-base-uncased"
  lora:
    rank: 8
    alpha: 16.0
distillation:
  temperature: 6.0
  progressive:
    layer_mapping: [[0, 2], [1, 5]]
training:
  epochs: 5
  batch_size: 32
dataset:
  path: "wikitext"
"#;

    let config = DistillationYamlConfig::from_yaml(yaml).expect("config should be valid");
    assert!(config.validate().is_ok());

    let trainer_config = config.to_trainer_config().expect("config should be valid");
    assert_eq!(trainer_config.epochs, 5);
    assert!(trainer_config.progressive.is_some());
}

#[test]
fn test_integration_loss_with_trainer() {
    use ndarray::Array2;

    let teacher = SafeTensorsTeacher::mock(12, 768);
    let config = TrainerConfig::new("t", "s")
        .temperature(4.0)
        .alpha(0.7)
        .with_progressive(vec![(0, 0)])
        .with_attention_transfer(0.1);

    let trainer = DistillationTrainer::new(config, teacher);

    // Create mock data
    let student_logits = Array2::from_shape_fn((8, 100), |(i, j)| (i + j) as f32 * 0.01);
    let teacher_logits = Array2::from_shape_fn((8, 100), |(i, j)| (i + j + 1) as f32 * 0.01);
    let targets: Vec<usize> = (0..8).collect();

    let sh = vec![Array2::<f32>::zeros((8, 768))];
    let th = vec![Array2::<f32>::ones((8, 768))];
    let sa = vec![Array2::<f32>::zeros((8, 8))];
    let ta = vec![Array2::<f32>::ones((8, 8))];

    // Compute combined loss
    let loss = trainer.compute_loss(
        &student_logits,
        &teacher_logits,
        &targets,
        Some(&sh),
        Some(&th),
        Some(&sa),
        Some(&ta),
    );

    assert!(loss > 0.0);
    assert!(loss.is_finite());
}

// =========================================================================
// ENT-083: Property Tests
// =========================================================================

mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(proptest::test_runner::Config::with_cases(100))]

        /// Memory estimation should scale linearly with param count
        #[test]
        fn prop_memory_scales_linearly(
            param_count in 1_000_000u64..1_000_000_000,
            batch_size in 1usize..64,
            seq_len in 64usize..2048,
            hidden_size in 256usize..4096,
        ) {
            let mem1 = MemoryEstimate::fp16(param_count, batch_size, seq_len, hidden_size);
            let mem2 = MemoryEstimate::fp16(param_count * 2, batch_size, seq_len, hidden_size);

            // Weights should double
            prop_assert_eq!(mem2.weights, mem1.weights * 2);
        }

        /// Distillation loss should be positive for different logits
        #[test]
        fn prop_distillation_loss_positive(
            temp in 1.0f32..20.0,
            alpha in 0.0f32..1.0,
        ) {
            use ndarray::Array2;

            let loss_fn = DistillationLoss::new(temp, alpha);
            let student = Array2::from_shape_fn((2, 10), |(i, j)| (i + j) as f32);
            let teacher = Array2::from_shape_fn((2, 10), |(i, j)| (i + j + 1) as f32);

            let loss = loss_fn.forward(&student, &teacher, &[5, 3]);
            prop_assert!(loss >= 0.0);
            prop_assert!(loss.is_finite());
        }

        /// Collator should preserve sequence content
        #[test]
        fn prop_collator_preserves_content(
            seq_len in 1usize..100,
            batch_size in 1usize..16,
        ) {
            let examples: Vec<Example> = (0..batch_size)
                .map(|i| Example::from_tokens((0..seq_len).map(|j| (i * 1000 + j) as u32).collect()))
                .collect();

            let collator = DistillationCollator::new(0).max_length(512);
            let batch = collator.collate(&examples);

            // First token of each sequence should match
            for (i, example) in examples.iter().enumerate() {
                prop_assert_eq!(batch.input_ids[[i, 0]], example.input_ids[0]);
            }
        }

        /// Dataset shuffle should be deterministic with same seed
        #[test]
        fn prop_shuffle_deterministic(seed in 0u64..10000) {
            let mut ds1 = Dataset::mock(50, 16);
            let mut ds2 = Dataset::mock(50, 16);

            ds1.shuffle(seed);
            ds2.shuffle(seed);

            // Same seed should produce same order
            for (e1, e2) in ds1.examples().iter().zip(ds2.examples().iter()) {
                prop_assert_eq!(&e1.input_ids, &e2.input_ids);
            }
        }

        /// Export format detection should round-trip
        #[test]
        fn prop_format_detection_roundtrip(format_idx in 0usize..3) {
            let formats = [ExportFormat::SafeTensors, ExportFormat::APR, ExportFormat::GGUF];
            let format = formats[format_idx];

            let path = format!("model.{}", format.extension());
            let detected = ExportFormat::from_path(std::path::Path::new(&path));

            prop_assert_eq!(detected, Some(format));
        }

        /// Trainer state should track progress correctly
        #[test]
        fn prop_trainer_state_consistent(
            num_steps in 1usize..1000,
            num_epochs in 1usize..10,
        ) {
            let mut state = TrainingState::new();

            for _ in 0..num_epochs {
                for _ in 0..num_steps {
                    state.record_loss(1.0);
                    state.step();
                }
                state.new_epoch();
            }

            prop_assert_eq!(state.global_step, num_steps * num_epochs);
            prop_assert_eq!(state.epoch, num_epochs);
            prop_assert_eq!(state.loss_history.len(), num_steps * num_epochs);
        }

        /// Model weights should count params correctly
        #[test]
        fn prop_weights_param_count(
            num_tensors in 1usize..10,
            tensor_size in 100usize..1000,
        ) {
            let mut weights = ModelWeights::new();
            for i in 0..num_tensors {
                let data = vec![0.0f32; tensor_size];
                weights.add_tensor(format!("tensor_{i}"), data, vec![tensor_size]);
            }

            prop_assert_eq!(weights.param_count(), (num_tensors * tensor_size) as u64);
            prop_assert_eq!(weights.tensor_names().len(), num_tensors);
        }

        /// FineTuneConfig memory estimate should be positive
        #[test]
        fn prop_finetune_memory_positive(
            total_params in 1_000_000u64..10_000_000_000,
        ) {
            let config = FineTuneConfig::new("model");
            let mem = config.estimate_memory(total_params);

            prop_assert!(mem.total() > 0);
            prop_assert!(mem.model > 0);
        }
    }
}
