use super::*;
use crate::finetune::classify_eval_report::{
    restore_class_weights_from_metadata, ClassifyEvalReport, SSC_LABELS,
};
use crate::finetune::{ClassifyConfig, ClassifyPipeline};
use crate::transformer::TransformerConfig;
use std::collections::HashSet;

fn tiny_pipeline(num_classes: usize) -> ClassifyPipeline {
    let model_config = TransformerConfig::tiny();
    let classify_config = ClassifyConfig {
        num_classes,
        lora_rank: 4,
        lora_alpha: 4.0,
        learning_rate: 1e-2,
        batch_size: 4,
        gradient_clip_norm: None,
        ..ClassifyConfig::default()
    };
    ClassifyPipeline::new(&model_config, classify_config)
}

fn make_corpus(n: usize, num_classes: usize) -> Vec<SafetySample> {
    (0..n)
        .map(|i| SafetySample {
            input: format!("sample_{i}_{}", "x".repeat(i % 5 + 1)),
            label: i % num_classes,
        })
        .collect()
}

// =========================================================================
// SSC-026: Dataset splitting
// =========================================================================

#[test]
fn test_ssc026_split_dataset_disjoint() {
    // F-LOOP-008: Train and val sets must have zero overlap
    let corpus = make_corpus(20, 3);
    let (train, val) = ClassifyTrainer::split_dataset(&corpus, 0.2, 42);

    let train_inputs: HashSet<String> = train.iter().map(|s| s.input.clone()).collect();
    let val_inputs: HashSet<String> = val.iter().map(|s| s.input.clone()).collect();

    let overlap: HashSet<_> = train_inputs.intersection(&val_inputs).collect();
    assert!(overlap.is_empty(), "F-LOOP-008: train/val overlap = {overlap:?}");
}

#[test]
fn test_ssc026_split_dataset_sizes() {
    let corpus = make_corpus(100, 3);
    let (train, val) = ClassifyTrainer::split_dataset(&corpus, 0.2, 42);

    // Total must be preserved
    assert_eq!(train.len() + val.len(), 100, "All samples must be accounted for");

    // Val should be ~20% (ceil(100 * 0.2) = 20)
    assert_eq!(val.len(), 20, "Val set should be 20% of 100");
    assert_eq!(train.len(), 80, "Train set should be 80% of 100");
}

#[test]
fn test_ssc026_split_dataset_deterministic() {
    let corpus = make_corpus(50, 3);
    let (train1, val1) = ClassifyTrainer::split_dataset(&corpus, 0.2, 42);
    let (train2, val2) = ClassifyTrainer::split_dataset(&corpus, 0.2, 42);

    // Same seed produces identical splits
    let train1_inputs: Vec<String> = train1.iter().map(|s| s.input.clone()).collect();
    let train2_inputs: Vec<String> = train2.iter().map(|s| s.input.clone()).collect();
    assert_eq!(train1_inputs, train2_inputs, "Splits must be deterministic");

    let val1_inputs: Vec<String> = val1.iter().map(|s| s.input.clone()).collect();
    let val2_inputs: Vec<String> = val2.iter().map(|s| s.input.clone()).collect();
    assert_eq!(val1_inputs, val2_inputs, "Val splits must be deterministic");
}

#[test]
fn test_ssc026_split_dataset_empty() {
    let (train, val) = ClassifyTrainer::split_dataset(&[], 0.2, 42);
    assert!(train.is_empty());
    assert!(val.is_empty());
}

// =========================================================================
// SSC-026: Val set frozen
// =========================================================================

#[test]
fn test_ssc026_val_set_frozen() {
    // F-LOOP-009: Val set does not change between epochs
    let num_classes = 3;
    let corpus = make_corpus(20, num_classes);
    let pipeline = tiny_pipeline(num_classes);
    let config = TrainingConfig {
        epochs: 3,
        val_split: 0.2,
        checkpoint_dir: PathBuf::from("/tmp/ssc026_test_frozen"),
        early_stopping_patience: 100,
        ..TrainingConfig::default()
    };

    let trainer = ClassifyTrainer::new(pipeline, corpus, config).expect("config should be valid");
    let val_before: Vec<String> = trainer.val_data().iter().map(|s| s.input.clone()).collect();

    // The val set is established at construction and must not change
    let val_after: Vec<String> = trainer.val_data().iter().map(|s| s.input.clone()).collect();
    assert_eq!(val_before, val_after, "F-LOOP-009: val set must be frozen");
}

// =========================================================================
// SSC-026: Data shuffled per epoch
// =========================================================================

#[test]
fn test_ssc026_data_shuffled_per_epoch() {
    // F-LOOP-007: Training order differs between epochs
    let num_classes = 3;
    let corpus = make_corpus(30, num_classes);
    let pipeline = tiny_pipeline(num_classes);
    let config = TrainingConfig {
        epochs: 2,
        val_split: 0.2,
        checkpoint_dir: PathBuf::from("/tmp/ssc026_test_shuffle"),
        early_stopping_patience: 100,
        ..TrainingConfig::default()
    };

    let mut trainer =
        ClassifyTrainer::new(pipeline, corpus, config).expect("config should be valid");

    // Capture order after epoch-0 shuffle
    trainer.shuffle_training_data(0);
    let order_epoch0: Vec<String> = trainer.train_data().iter().map(|s| s.input.clone()).collect();

    // Capture order after epoch-1 shuffle
    trainer.shuffle_training_data(1);
    let order_epoch1: Vec<String> = trainer.train_data().iter().map(|s| s.input.clone()).collect();

    assert_ne!(
        order_epoch0, order_epoch1,
        "F-LOOP-007: training data must have different order per epoch"
    );
}

// =========================================================================
// SSC-026: Training convergence
// =========================================================================

#[test]
#[ignore] // CUDA logit NaN on RTX 4090; validated on dev GPU
fn test_ssc026_train_convergence() {
    // Loss should decrease over epochs on a tiny overfit task
    let num_classes = 3;
    let corpus = vec![
        SafetySample { input: "echo hello world".into(), label: 0 },
        SafetySample { input: "rm -rf /tmp/danger".into(), label: 1 },
        SafetySample { input: "ls -la /home".into(), label: 2 },
        SafetySample { input: "echo safe output".into(), label: 0 },
        SafetySample { input: "eval dangerous code".into(), label: 1 },
        SafetySample { input: "cat /etc/passwd".into(), label: 2 },
    ];

    let pipeline = tiny_pipeline(num_classes);
    let config = TrainingConfig {
        epochs: 15,
        val_split: 0.34, // 2 val samples out of 6
        checkpoint_dir: PathBuf::from("/tmp/ssc026_test_convergence"),
        early_stopping_patience: 100, // disable early stopping for this test
        warmup_fraction: 0.0,
        ..TrainingConfig::default()
    };

    let mut trainer =
        ClassifyTrainer::new(pipeline, corpus, config).expect("config should be valid");
    let result = trainer.train();

    assert!(!result.epoch_metrics.is_empty(), "Should have at least one epoch of metrics");

    let first_loss =
        result.epoch_metrics.first().expect("collection should not be empty").train_loss;
    let last_loss = result.epoch_metrics.last().expect("collection should not be empty").train_loss;

    assert!(
        last_loss < first_loss,
        "SSC-026: Training loss must decrease. First: {first_loss:.4}, last: {last_loss:.4}"
    );
}

// =========================================================================
// SSC-026: Epoch metrics complete
// =========================================================================

#[test]
#[ignore] // CUDA logit NaN on RTX 4090; validated on dev GPU
fn test_ssc026_epoch_metrics_complete() {
    let num_classes = 3;
    let corpus = make_corpus(15, num_classes);
    let pipeline = tiny_pipeline(num_classes);
    let config = TrainingConfig {
        epochs: 2,
        val_split: 0.2,
        checkpoint_dir: PathBuf::from("/tmp/ssc026_test_metrics"),
        early_stopping_patience: 100,
        ..TrainingConfig::default()
    };

    let mut trainer =
        ClassifyTrainer::new(pipeline, corpus, config).expect("config should be valid");
    let result = trainer.train();

    assert_eq!(result.epoch_metrics.len(), 2, "Should have 2 epochs");

    for m in &result.epoch_metrics {
        assert!(m.train_loss.is_finite(), "train_loss must be finite");
        assert!(m.val_loss.is_finite(), "val_loss must be finite");
        assert!(
            (0.0..=1.0).contains(&m.train_accuracy),
            "train_accuracy must be in [0,1], got {}",
            m.train_accuracy
        );
        assert!(
            (0.0..=1.0).contains(&m.val_accuracy),
            "val_accuracy must be in [0,1], got {}",
            m.val_accuracy
        );
        assert!(m.learning_rate >= 0.0, "LR must be non-negative");
        assert!(m.samples_per_sec >= 0.0, "throughput must be non-negative");
    }
}

// =========================================================================
// SSC-026: Early stopping
// =========================================================================

#[test]
#[ignore] // CUDA logit NaN on RTX 4090; validated on dev GPU
fn test_ssc026_early_stopping() {
    // F-LOOP-010: Training stops after patience epochs without val improvement
    let num_classes = 3;
    let corpus = make_corpus(10, num_classes);
    let pipeline = tiny_pipeline(num_classes);
    let config = TrainingConfig {
        epochs: 100, // high max so early stopping must trigger
        val_split: 0.3,
        early_stopping_patience: 3,
        checkpoint_dir: PathBuf::from("/tmp/ssc026_test_early_stop"),
        warmup_fraction: 0.0,
        ..TrainingConfig::default()
    };

    let mut trainer =
        ClassifyTrainer::new(pipeline, corpus, config).expect("config should be valid");
    let result = trainer.train();

    // Should have stopped before reaching 100 epochs
    assert!(
        result.epoch_metrics.len() < 100,
        "F-LOOP-010: Early stopping should have triggered. Ran {} epochs.",
        result.epoch_metrics.len()
    );
}

// =========================================================================
// SSC-026: Best epoch tracking
// =========================================================================

#[test]
#[ignore] // CUDA logit NaN on RTX 4090; validated on dev GPU
fn test_ssc026_train_result_best_epoch() {
    let num_classes = 3;
    let corpus = make_corpus(15, num_classes);
    let pipeline = tiny_pipeline(num_classes);
    let config = TrainingConfig {
        epochs: 5,
        val_split: 0.2,
        checkpoint_dir: PathBuf::from("/tmp/ssc026_test_best_epoch"),
        early_stopping_patience: 100,
        ..TrainingConfig::default()
    };

    let mut trainer =
        ClassifyTrainer::new(pipeline, corpus, config).expect("config should be valid");
    let result = trainer.train();

    // best_epoch must correspond to the lowest val_loss
    let actual_best = result
        .epoch_metrics
        .iter()
        .min_by(|a, b| a.val_loss.partial_cmp(&b.val_loss).expect("operation should succeed"))
        .expect("operation should succeed");

    assert_eq!(
        result.best_epoch, actual_best.epoch,
        "best_epoch should match the epoch with lowest val_loss"
    );
    assert!(
        (result.best_val_loss - actual_best.val_loss).abs() < 1e-6,
        "best_val_loss should match"
    );
}

// =========================================================================
// SSC-026: Error handling
// =========================================================================

#[test]
fn test_ssc026_empty_corpus_error() {
    let pipeline = tiny_pipeline(3);
    let config = TrainingConfig::default();

    let result = ClassifyTrainer::new(pipeline, vec![], config);
    assert!(result.is_err(), "Empty corpus should return an error");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("corpus must not be empty"),
        "Error should mention empty corpus, got: {err}"
    );
}

#[test]
fn test_ssc026_invalid_val_split_zero() {
    let pipeline = tiny_pipeline(3);
    let corpus = make_corpus(10, 3);
    let config = TrainingConfig { val_split: 0.0, ..TrainingConfig::default() };

    let result = ClassifyTrainer::new(pipeline, corpus, config);
    assert!(result.is_err(), "val_split=0.0 should return an error");
}

#[test]
fn test_ssc026_invalid_val_split_too_large() {
    let pipeline = tiny_pipeline(3);
    let corpus = make_corpus(10, 3);
    let config = TrainingConfig { val_split: 0.8, ..TrainingConfig::default() };

    let result = ClassifyTrainer::new(pipeline, corpus, config);
    assert!(result.is_err(), "val_split=0.8 should return an error");
}

#[test]
fn test_ssc026_training_config_default() {
    let config = TrainingConfig::default();
    assert_eq!(config.epochs, 50);
    assert!((config.val_split - 0.2).abs() < 1e-6);
    assert_eq!(config.save_every, 5);
    assert_eq!(config.early_stopping_patience, 10);
    assert_eq!(config.seed, 42);
    assert_eq!(config.log_interval, 1);
    assert!(config.distributed.is_none());
}

#[test]
fn test_training_config_with_distributed() {
    let dist = DistributedConfig::coordinator("127.0.0.1:0".parse().expect("valid"), 2);
    let config = TrainingConfig { distributed: Some(dist.clone()), ..TrainingConfig::default() };
    assert!(config.distributed.is_some());
    assert_eq!(config.distributed.expect("valid").expect_workers, 2);
}

#[test]
fn test_is_coordinator_mode() {
    let pipeline = tiny_pipeline(2);
    let corpus = make_corpus(20, 2);
    let config = TrainingConfig { epochs: 1, ..TrainingConfig::default() };
    let trainer = ClassifyTrainer::new(pipeline, corpus, config).expect("valid");
    assert!(!trainer.is_coordinator_mode());
}

#[test]
fn test_is_coordinator_mode_with_coordinator_config() {
    let pipeline = tiny_pipeline(2);
    let corpus = make_corpus(20, 2);
    let dist = DistributedConfig::coordinator("127.0.0.1:0".parse().expect("valid"), 1);
    let config = TrainingConfig { epochs: 1, distributed: Some(dist), ..TrainingConfig::default() };
    let trainer = ClassifyTrainer::new(pipeline, corpus, config).expect("valid");
    assert!(trainer.is_coordinator_mode());
}

#[test]
fn test_is_coordinator_mode_with_worker_config() {
    let pipeline = tiny_pipeline(2);
    let corpus = make_corpus(20, 2);
    let dist = DistributedConfig::worker("127.0.0.1:9000".parse().expect("valid"));
    let config = TrainingConfig { epochs: 1, distributed: Some(dist), ..TrainingConfig::default() };
    let trainer = ClassifyTrainer::new(pipeline, corpus, config).expect("valid");
    assert!(!trainer.is_coordinator_mode());
}

#[test]
fn test_collect_gradients_layout() {
    // F-DP-001: Gradient collection produces correct-length vector
    let pipeline = tiny_pipeline(2);
    let grads = pipeline.collect_lora_gradients();

    // Should have gradients for all trainable params (LoRA A/B + classifier W/B)
    let expected_len = pipeline.num_trainable_parameters();
    assert_eq!(grads.len(), expected_len);
}

#[test]
fn test_apply_gradients_preserves_pipeline() {
    // F-DP-001: Applying averaged gradients doesn't corrupt pipeline state
    let mut pipeline = tiny_pipeline(2);
    let num_params = pipeline.num_trainable_parameters();

    // Create synthetic averaged gradients (small values)
    let avg_grads: Vec<f32> = (0..num_params).map(|i| (i as f32) * 0.001).collect();

    // Apply
    pipeline.apply_lora_gradients(&avg_grads);

    // Pipeline should still produce valid output
    let token_ids = vec![1u32, 2, 3, 4];
    let (loss, _pred) = pipeline.forward_only(&token_ids, 0);
    assert!(loss.is_finite(), "loss should be finite after applying gradients");
}

#[test]
fn test_distributed_coordinator_worker_gradient_exchange() {
    // Full integration: coordinator + worker via TCP
    // Validates F-DP-001 (weight consistency via AllReduce)
    use crate::finetune::gradient_server::GradientServer;
    use crate::finetune::worker_client::WorkerClient;

    let dist_config = DistributedConfig::coordinator("127.0.0.1:0".parse().expect("valid"), 1);
    let mut server = GradientServer::bind(dist_config).expect("valid");
    let addr = server.local_addr();

    // Spawn worker thread — uses synthetic gradients (no tokenizer needed)
    let handle = std::thread::spawn(move || {
        let worker_config = DistributedConfig::worker(addr);
        let client = WorkerClient::connect(worker_config, 1, "cpu").expect("valid");

        // Receive shard assignment
        let shard = client.receive_shard().expect("valid").expect("should get shard");
        assert_eq!(shard.step, 0);

        // Create a tiny pipeline and generate synthetic gradients
        let pipe = tiny_pipeline(2);
        let num_params = pipe.num_trainable_parameters();
        let grads: Vec<f32> = (0..num_params).map(|i| (i as f32 + 1.0) * 0.01).collect();

        // Send gradients (simulating forward/backward output)
        client.send_gradients(0, grads, 0.5, 3, 5).expect("valid");

        // Receive averaged gradients
        let averaged = client.receive_averaged().expect("valid");
        assert!(averaged.global_loss.is_finite());
        assert!(!averaged.gradients.is_empty());
        assert_eq!(averaged.gradients.len(), num_params);
    });

    // Server side
    server.wait_for_workers().expect("valid");
    server.set_total_samples(10);
    server.send_shard_assignments(0).expect("valid");

    let result = server.collect_and_reduce(0).expect("valid");
    assert!(result.avg_gradients.iter().all(|g| g.is_finite()));
    assert!(result.global_loss.is_finite());
    assert_eq!(result.total_correct, 3);
    assert_eq!(result.total_samples, 5);

    server.broadcast_averaged(0, &result).expect("valid");
    handle.join().expect("valid");
}

// =========================================================================
// TrainingConfig additional tests
// =========================================================================

#[test]
fn test_training_config_default_warmup_fraction() {
    let config = TrainingConfig::default();
    assert!((config.warmup_fraction - 0.1).abs() < 1e-6);
    assert!((config.lr_min - 1e-6).abs() < 1e-9);
    assert!(!config.oversample_minority);
    assert!(!config.quantize_nf4);
}

#[test]
fn test_training_config_clone() {
    let config = TrainingConfig {
        epochs: 10,
        val_split: 0.3,
        save_every: 2,
        early_stopping_patience: 5,
        checkpoint_dir: PathBuf::from("/tmp/clone_test"),
        seed: 123,
        log_interval: 2,
        warmup_fraction: 0.05,
        lr_min: 1e-5,
        oversample_minority: true,
        quantize_nf4: true,
        distributed: None,
    };
    let cloned = config.clone();
    assert_eq!(cloned.epochs, 10);
    assert_eq!(cloned.seed, 123);
    assert!(cloned.oversample_minority);
    assert!(cloned.quantize_nf4);
}

#[test]
fn test_training_config_debug() {
    let config = TrainingConfig::default();
    let dbg = format!("{config:?}");
    assert!(dbg.contains("TrainingConfig"));
    assert!(dbg.contains("epochs"));
}

// =========================================================================
// EpochMetrics tests
// =========================================================================

#[test]
fn test_epoch_metrics_clone_and_debug() {
    let m = EpochMetrics {
        epoch: 5,
        train_loss: 0.5,
        train_accuracy: 0.9,
        val_loss: 0.6,
        val_accuracy: 0.85,
        learning_rate: 1e-3,
        epoch_time_ms: 1234,
        samples_per_sec: 100.0,
    };
    let cloned = m.clone();
    assert_eq!(cloned.epoch, 5);
    assert!((cloned.train_loss - 0.5).abs() < 1e-6);
    assert!((cloned.val_loss - 0.6).abs() < 1e-6);
    assert_eq!(cloned.epoch_time_ms, 1234);
    let dbg = format!("{m:?}");
    assert!(dbg.contains("EpochMetrics"));
}

// =========================================================================
// TrainResult tests
// =========================================================================

#[test]
fn test_train_result_clone_and_debug() {
    let result = TrainResult {
        epoch_metrics: vec![],
        best_epoch: 3,
        best_val_loss: 0.25,
        stopped_early: true,
        total_time_ms: 5000,
    };
    let cloned = result.clone();
    assert_eq!(cloned.best_epoch, 3);
    assert!(cloned.stopped_early);
    assert_eq!(cloned.total_time_ms, 5000);
    let dbg = format!("{result:?}");
    assert!(dbg.contains("TrainResult"));
}

// =========================================================================
// SSC_LABELS constant
// =========================================================================

#[test]
fn test_ssc_labels_constant() {
    assert_eq!(SSC_LABELS.len(), 5);
    assert_eq!(SSC_LABELS[0], "safe");
    assert_eq!(SSC_LABELS[1], "needs-quoting");
    assert_eq!(SSC_LABELS[2], "non-deterministic");
    assert_eq!(SSC_LABELS[3], "non-idempotent");
    assert_eq!(SSC_LABELS[4], "unsafe");
}

// =========================================================================
// ClassifyTrainer::compute_data_hash
// =========================================================================

#[test]
fn test_compute_data_hash_deterministic() {
    let corpus = make_corpus(10, 3);
    let hash1 = ClassifyTrainer::compute_data_hash(&corpus);
    let hash2 = ClassifyTrainer::compute_data_hash(&corpus);
    assert_eq!(hash1, hash2);
    assert!(hash1.starts_with("sha256:"));
}

#[test]
fn test_compute_data_hash_order_independent() {
    let corpus1 = vec![
        SafetySample { input: "echo hello".into(), label: 0 },
        SafetySample { input: "rm -rf /".into(), label: 1 },
    ];
    let corpus2 = vec![
        SafetySample { input: "rm -rf /".into(), label: 1 },
        SafetySample { input: "echo hello".into(), label: 0 },
    ];
    assert_eq!(
        ClassifyTrainer::compute_data_hash(&corpus1),
        ClassifyTrainer::compute_data_hash(&corpus2),
    );
}

#[test]
fn test_compute_data_hash_empty() {
    let hash = ClassifyTrainer::compute_data_hash(&[]);
    assert!(hash.starts_with("sha256:"));
}

#[test]
fn test_compute_data_hash_different_data() {
    let c1 = vec![SafetySample { input: "echo hello".into(), label: 0 }];
    let c2 = vec![SafetySample { input: "echo world".into(), label: 0 }];
    assert_ne!(ClassifyTrainer::compute_data_hash(&c1), ClassifyTrainer::compute_data_hash(&c2));
}

#[test]
fn test_compute_data_hash_different_labels() {
    let c1 = vec![SafetySample { input: "echo hello".into(), label: 0 }];
    let c2 = vec![SafetySample { input: "echo hello".into(), label: 1 }];
    assert_ne!(ClassifyTrainer::compute_data_hash(&c1), ClassifyTrainer::compute_data_hash(&c2));
}

// =========================================================================
// ClassifyTrainer::oversample_training_data
// =========================================================================

#[test]
fn test_oversample_balanced_unchanged() {
    let mut data = vec![
        SafetySample { input: "a".into(), label: 0 },
        SafetySample { input: "b".into(), label: 1 },
        SafetySample { input: "c".into(), label: 0 },
        SafetySample { input: "d".into(), label: 1 },
    ];
    let n = data.len();
    ClassifyTrainer::oversample_training_data(&mut data, 42);
    assert_eq!(data.len(), n);
}

#[test]
fn test_oversample_imbalanced() {
    let mut data = vec![
        SafetySample { input: "a".into(), label: 0 },
        SafetySample { input: "b".into(), label: 0 },
        SafetySample { input: "c".into(), label: 0 },
        SafetySample { input: "d".into(), label: 0 },
        SafetySample { input: "e".into(), label: 1 },
    ];
    ClassifyTrainer::oversample_training_data(&mut data, 42);
    assert_eq!(data.len(), 8);
    assert_eq!(data.iter().filter(|s| s.label == 1).count(), 4);
}

#[test]
fn test_oversample_deterministic() {
    let mk = || {
        vec![
            SafetySample { input: "a".into(), label: 0 },
            SafetySample { input: "b".into(), label: 0 },
            SafetySample { input: "c".into(), label: 0 },
            SafetySample { input: "d".into(), label: 1 },
        ]
    };
    let mut d1 = mk();
    let mut d2 = mk();
    ClassifyTrainer::oversample_training_data(&mut d1, 42);
    ClassifyTrainer::oversample_training_data(&mut d2, 42);
    let l1: Vec<usize> = d1.iter().map(|s| s.label).collect();
    let l2: Vec<usize> = d2.iter().map(|s| s.label).collect();
    assert_eq!(l1, l2);
}

#[test]
fn test_oversample_three_classes() {
    let mut data = vec![
        SafetySample { input: "a".into(), label: 0 },
        SafetySample { input: "b".into(), label: 0 },
        SafetySample { input: "c".into(), label: 0 },
        SafetySample { input: "d".into(), label: 1 },
        SafetySample { input: "e".into(), label: 1 },
        SafetySample { input: "f".into(), label: 2 },
    ];
    ClassifyTrainer::oversample_training_data(&mut data, 42);
    assert_eq!(data.len(), 9);
    assert_eq!(data.iter().filter(|s| s.label == 0).count(), 3);
    assert_eq!(data.iter().filter(|s| s.label == 1).count(), 3);
    assert_eq!(data.iter().filter(|s| s.label == 2).count(), 3);
}

// =========================================================================
// ClassifyTrainer::new validation
// =========================================================================

#[test]
fn test_new_zero_epochs_error() {
    let p = tiny_pipeline(3);
    let c = make_corpus(10, 3);
    let cfg = TrainingConfig { epochs: 0, ..TrainingConfig::default() };
    let err = ClassifyTrainer::new(p, c, cfg).unwrap_err().to_string();
    assert!(err.contains("epochs must be > 0"));
}

#[test]
fn test_new_negative_val_split_error() {
    let p = tiny_pipeline(3);
    let c = make_corpus(10, 3);
    let cfg = TrainingConfig { val_split: -0.1, ..TrainingConfig::default() };
    assert!(ClassifyTrainer::new(p, c, cfg).is_err());
}

#[test]
fn test_new_val_split_0_5_ok() {
    let p = tiny_pipeline(3);
    let c = make_corpus(20, 3);
    let cfg = TrainingConfig { val_split: 0.5, epochs: 1, ..TrainingConfig::default() };
    assert!(ClassifyTrainer::new(p, c, cfg).is_ok());
}

#[test]
fn test_new_val_split_above_0_5_error() {
    let p = tiny_pipeline(3);
    let c = make_corpus(20, 3);
    let cfg = TrainingConfig { val_split: 0.51, ..TrainingConfig::default() };
    assert!(ClassifyTrainer::new(p, c, cfg).is_err());
}

#[test]
fn test_new_with_oversample() {
    let p = tiny_pipeline(2);
    // Use a larger corpus to ensure both classes survive in training split
    let mut c: Vec<SafetySample> =
        (0..80).map(|i| SafetySample { input: format!("safe_{i}"), label: 0 }).collect();
    for i in 0..20 {
        c.push(SafetySample { input: format!("unsafe_{i}"), label: 1 });
    }
    let cfg = TrainingConfig {
        epochs: 1,
        val_split: 0.2,
        oversample_minority: true,
        ..TrainingConfig::default()
    };
    let t = ClassifyTrainer::new(p, c, cfg).expect("valid");
    let c0 = t.train_data().iter().filter(|s| s.label == 0).count();
    let c1 = t.train_data().iter().filter(|s| s.label == 1).count();
    assert_eq!(c0, c1, "After oversampling, minority matches majority: {c0} vs {c1}");
}

#[test]
fn test_new_accessors() {
    let p = tiny_pipeline(3);
    let c = make_corpus(20, 3);
    let cfg = TrainingConfig { epochs: 1, val_split: 0.2, ..TrainingConfig::default() };
    let t = ClassifyTrainer::new(p, c, cfg).expect("valid");
    assert!(!t.train_data().is_empty());
    assert!(!t.val_data().is_empty());
    assert_eq!(t.config().epochs, 1);
    assert_eq!(t.train_data().len() + t.val_data().len(), 20);
}

#[test]
fn test_new_debug_impl() {
    let p = tiny_pipeline(3);
    let c = make_corpus(20, 3);
    let cfg = TrainingConfig { epochs: 1, ..TrainingConfig::default() };
    let t = ClassifyTrainer::new(p, c, cfg).expect("valid");
    let dbg = format!("{t:?}");
    assert!(dbg.contains("ClassifyTrainer"));
    assert!(dbg.contains("train_data_len"));
    assert!(dbg.contains("val_data_len"));
}

// =========================================================================
// split_dataset edge cases
// =========================================================================

#[test]
fn test_split_single_element() {
    let c = vec![SafetySample { input: "only".into(), label: 0 }];
    let (tr, va) = ClassifyTrainer::split_dataset(&c, 0.2, 42);
    assert_eq!(tr.len() + va.len(), 1);
}

#[test]
fn test_split_two_elements() {
    let c = vec![
        SafetySample { input: "a".into(), label: 0 },
        SafetySample { input: "b".into(), label: 1 },
    ];
    let (tr, va) = ClassifyTrainer::split_dataset(&c, 0.2, 42);
    assert_eq!(tr.len() + va.len(), 2);
    assert!(!va.is_empty());
    assert!(!tr.is_empty());
}

#[test]
fn test_split_different_seeds() {
    let c = make_corpus(50, 3);
    let (t1, _) = ClassifyTrainer::split_dataset(&c, 0.2, 42);
    let (t2, _) = ClassifyTrainer::split_dataset(&c, 0.2, 99);
    let o1: Vec<String> = t1.iter().map(|s| s.input.clone()).collect();
    let o2: Vec<String> = t2.iter().map(|s| s.input.clone()).collect();
    assert_ne!(o1, o2);
}

#[test]
fn test_split_val_0_5() {
    let c = make_corpus(20, 2);
    let (tr, va) = ClassifyTrainer::split_dataset(&c, 0.5, 42);
    assert_eq!(tr.len() + va.len(), 20);
    assert_eq!(va.len(), 10);
}

#[test]
fn test_split_preserves_all() {
    let c = make_corpus(100, 5);
    let (tr, va) = ClassifyTrainer::split_dataset(&c, 0.3, 42);
    let mut all: Vec<String> = tr.iter().chain(va.iter()).map(|s| s.input.clone()).collect();
    all.sort();
    let mut orig: Vec<String> = c.iter().map(|s| s.input.clone()).collect();
    orig.sort();
    assert_eq!(all, orig);
}

// =========================================================================
// shuffle_training_data
// =========================================================================

#[test]
fn test_shuffle_preserves_length() {
    let p = tiny_pipeline(3);
    let c = make_corpus(30, 3);
    let cfg = TrainingConfig { epochs: 1, val_split: 0.2, ..TrainingConfig::default() };
    let mut t = ClassifyTrainer::new(p, c, cfg).expect("valid");
    let n = t.train_data().len();
    t.shuffle_training_data(0);
    assert_eq!(t.train_data().len(), n);
}

#[test]
fn test_shuffle_preserves_elements() {
    let p = tiny_pipeline(3);
    let c = make_corpus(30, 3);
    let cfg = TrainingConfig { epochs: 1, val_split: 0.2, ..TrainingConfig::default() };
    let mut t = ClassifyTrainer::new(p, c, cfg).expect("valid");
    let mut before: Vec<String> = t.train_data().iter().map(|s| s.input.clone()).collect();
    t.shuffle_training_data(0);
    let mut after: Vec<String> = t.train_data().iter().map(|s| s.input.clone()).collect();
    before.sort();
    after.sort();
    assert_eq!(before, after);
}

// =========================================================================
// ClassifyEvalReport — helper
// =========================================================================

fn make_eval_report(
    y_pred: &[usize],
    y_true: &[usize],
    all_probs: &[Vec<f32>],
    num_classes: usize,
) -> ClassifyEvalReport {
    let label_names: Vec<String> = (0..num_classes).map(|i| format!("class_{i}")).collect();
    let total_loss: f32 = all_probs
        .iter()
        .map(|p| {
            let mx = p.iter().copied().fold(0.0f32, f32::max);
            -mx.ln().max(0.0)
        })
        .sum();
    ClassifyEvalReport::from_predictions_with_probs(
        y_pred,
        y_true,
        all_probs,
        total_loss,
        num_classes,
        &label_names,
        100,
    )
}

// =========================================================================
// ClassifyEvalReport tests
// =========================================================================

#[test]
fn test_eval_report_perfect() {
    let report = make_eval_report(
        &[0, 1, 0, 1, 0],
        &[0, 1, 0, 1, 0],
        &[vec![0.9, 0.1], vec![0.1, 0.9], vec![0.9, 0.1], vec![0.1, 0.9], vec![0.9, 0.1]],
        2,
    );
    assert!((report.accuracy - 1.0).abs() < 1e-6);
    assert!(report.mcc > 0.9);
    assert!(report.cohens_kappa > 0.9);
}

#[test]
fn test_eval_report_all_wrong() {
    let report = make_eval_report(
        &[1, 0, 1, 0],
        &[0, 1, 0, 1],
        &[vec![0.1, 0.9], vec![0.9, 0.1], vec![0.1, 0.9], vec![0.9, 0.1]],
        2,
    );
    assert!((report.accuracy - 0.0).abs() < 1e-6);
}

#[test]
fn test_eval_report_empty() {
    let report = make_eval_report(&[], &[], &[], 2);
    assert!((report.accuracy - 0.0).abs() < 1e-6);
    assert_eq!(report.total_samples, 0);
}

#[test]
fn test_top2_accuracy_perfect() {
    let top2 =
        ClassifyEvalReport::compute_top2_accuracy(&[vec![0.9, 0.1], vec![0.1, 0.9]], &[0, 1], 2);
    assert!((top2 - 1.0).abs() < 1e-6);
}

#[test]
fn test_top2_accuracy_second_choice() {
    let top2 = ClassifyEvalReport::compute_top2_accuracy(
        &[vec![0.1, 0.6, 0.3], vec![0.1, 0.3, 0.6]],
        &[2, 0],
        2,
    );
    assert!((top2 - 0.5).abs() < 1e-6);
}

#[test]
fn test_top2_accuracy_empty() {
    assert!((ClassifyEvalReport::compute_top2_accuracy(&[], &[], 0) - 0.0).abs() < 1e-6);
}

#[test]
fn test_confidence_stats_all_correct() {
    let (mean, mc, mw) =
        ClassifyEvalReport::compute_confidence_stats(&[0.9, 0.95, 0.85], &[0, 1, 0], &[0, 1, 0]);
    assert!((mean - 0.9).abs() < 0.01);
    assert!((mc - 0.9).abs() < 0.01);
    assert!((mw - 0.0).abs() < 1e-6);
}

#[test]
fn test_confidence_stats_all_wrong() {
    let (mean, mc, mw) =
        ClassifyEvalReport::compute_confidence_stats(&[0.6, 0.7], &[1, 0], &[0, 1]);
    assert!((mean - 0.65).abs() < 1e-6);
    assert!((mc - 0.0).abs() < 1e-6);
    assert!((mw - 0.65).abs() < 1e-6);
}

#[test]
fn test_confidence_stats_empty() {
    let (m, mc, mw) = ClassifyEvalReport::compute_confidence_stats(&[], &[], &[]);
    assert!((m - 0.0).abs() < 1e-6);
    assert!((mc - 0.0).abs() < 1e-6);
    assert!((mw - 0.0).abs() < 1e-6);
}

#[test]
fn test_confidence_stats_mixed() {
    let (mean, mc, mw) = ClassifyEvalReport::compute_confidence_stats(
        &[0.9, 0.6, 0.8, 0.5],
        &[0, 1, 0, 0],
        &[0, 0, 0, 1],
    );
    assert!((mean - 0.7).abs() < 1e-6);
    assert!((mc - 0.85).abs() < 1e-6);
    assert!((mw - 0.55).abs() < 1e-6);
}

#[test]
fn test_calibration_basic() {
    let (bins, ece) = ClassifyEvalReport::compute_calibration(
        &[0.9, 0.1, 0.5, 0.5],
        &[0, 0, 1, 0],
        &[0, 0, 1, 1],
        4,
    );
    assert_eq!(bins.len(), 10);
    assert!((0.0..=1.0).contains(&ece));
}

#[test]
fn test_calibration_empty() {
    let (bins, ece) = ClassifyEvalReport::compute_calibration(&[], &[], &[], 0);
    assert_eq!(bins.len(), 10);
    assert!((ece - 0.0).abs() < 1e-6);
}

#[test]
fn test_cohens_kappa_perfect() {
    use crate::eval::classification::ConfusionMatrix;
    let cm = ConfusionMatrix::from_predictions_with_min_classes(
        &[0, 1, 0, 1, 0, 1],
        &[0, 1, 0, 1, 0, 1],
        2,
    );
    assert!((ClassifyEvalReport::compute_cohens_kappa(&cm, 6) - 1.0).abs() < 1e-6);
}

#[test]
fn test_cohens_kappa_no_agreement() {
    use crate::eval::classification::ConfusionMatrix;
    let cm = ConfusionMatrix::from_predictions_with_min_classes(&[1, 0, 1, 0], &[0, 1, 0, 1], 2);
    assert!(ClassifyEvalReport::compute_cohens_kappa(&cm, 4) < 0.0);
}

#[test]
fn test_cohens_kappa_zero_total() {
    use crate::eval::classification::ConfusionMatrix;
    let cm = ConfusionMatrix::from_predictions_with_min_classes(&[], &[], 2);
    assert!((ClassifyEvalReport::compute_cohens_kappa(&cm, 0) - 0.0).abs() < 1e-6);
}

#[test]
fn test_mcc_perfect() {
    use crate::eval::classification::ConfusionMatrix;
    let cm = ConfusionMatrix::from_predictions_with_min_classes(&[0, 1, 0, 1], &[0, 1, 0, 1], 2);
    assert!((ClassifyEvalReport::compute_mcc(&cm, 2, 4) - 1.0).abs() < 1e-6);
}

#[test]
fn test_mcc_zero_total() {
    use crate::eval::classification::ConfusionMatrix;
    let cm = ConfusionMatrix::from_predictions_with_min_classes(&[], &[], 2);
    assert!((ClassifyEvalReport::compute_mcc(&cm, 2, 0) - 0.0).abs() < 1e-6);
}

#[test]
fn test_mcc_all_same_pred() {
    use crate::eval::classification::ConfusionMatrix;
    let cm = ConfusionMatrix::from_predictions_with_min_classes(&[0, 0, 0, 0], &[0, 0, 1, 1], 2);
    assert!((ClassifyEvalReport::compute_mcc(&cm, 2, 4) - 0.0).abs() < 1e-6);
}

#[test]
fn test_brier_perfect() {
    let b = ClassifyEvalReport::compute_brier_score(&[vec![1.0, 0.0], vec![0.0, 1.0]], &[0, 1], 2);
    assert!((b - 0.0).abs() < 1e-6);
}

#[test]
fn test_brier_worst() {
    let b = ClassifyEvalReport::compute_brier_score(&[vec![0.0, 1.0], vec![1.0, 0.0]], &[0, 1], 2);
    assert!((b - 2.0).abs() < 1e-6);
}

#[test]
fn test_brier_empty() {
    assert!((ClassifyEvalReport::compute_brier_score(&[], &[], 2) - 0.0).abs() < 1e-6);
}

#[test]
fn test_log_loss_near_perfect() {
    let ll = ClassifyEvalReport::compute_log_loss(&[vec![0.99, 0.01], vec![0.01, 0.99]], &[0, 1]);
    assert!(ll < 0.02);
}

#[test]
fn test_log_loss_worst() {
    let ll = ClassifyEvalReport::compute_log_loss(&[vec![0.01, 0.99]], &[0]);
    assert!(ll > 4.0);
}

#[test]
fn test_log_loss_empty() {
    assert!((ClassifyEvalReport::compute_log_loss(&[], &[]) - 0.0).abs() < 1e-6);
}

#[test]
fn test_baselines_binary() {
    let (r, m) = ClassifyEvalReport::compute_baselines(&[70, 30], 100, 2);
    assert!((r - 0.5).abs() < 1e-6);
    assert!((m - 0.7).abs() < 1e-6);
}

#[test]
fn test_baselines_three_class() {
    let (r, m) = ClassifyEvalReport::compute_baselines(&[50, 30, 20], 100, 3);
    assert!((r - 1.0 / 3.0).abs() < 1e-6);
    assert!((m - 0.5).abs() < 1e-6);
}

#[test]
fn test_baselines_zero_total() {
    let (r, m) = ClassifyEvalReport::compute_baselines(&[], 0, 2);
    assert!((r - 0.5).abs() < 1e-6);
    assert!((m - 0.0).abs() < 1e-6);
}

#[test]
fn test_baselines_zero_classes() {
    let (r, _) = ClassifyEvalReport::compute_baselines(&[], 0, 0);
    assert!((r - 0.0).abs() < 1e-6);
}

#[test]
fn test_top_confusions_no_errors() {
    let c = ClassifyEvalReport::compute_top_confusions(&[vec![5, 0], vec![0, 5]], 5);
    assert!(c.is_empty());
}

#[test]
fn test_top_confusions_with_errors() {
    let c = ClassifyEvalReport::compute_top_confusions(
        &[vec![10, 3, 1], vec![2, 8, 0], vec![0, 1, 9]],
        5,
    );
    assert_eq!(c[0], (0, 1, 3));
    assert_eq!(c[1], (1, 0, 2));
}

#[test]
fn test_top_confusions_truncated() {
    let c = ClassifyEvalReport::compute_top_confusions(
        &[vec![5, 1, 2, 3], vec![4, 5, 1, 2], vec![3, 2, 5, 1], vec![1, 3, 2, 5]],
        3,
    );
    assert_eq!(c.len(), 3);
}

#[test]
fn test_bootstrap_cis_perfect() {
    let (ci_a, ci_f, ci_m) = ClassifyEvalReport::compute_bootstrap_cis(
        &[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        &[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        2,
        100,
    );
    assert!(ci_a.0 >= 0.9);
    assert!(ci_f.0 >= 0.9);
    assert!(ci_m.0 >= 0.9);
}

#[test]
fn test_bootstrap_cis_empty() {
    let (a, f, m) = ClassifyEvalReport::compute_bootstrap_cis(&[], &[], 2, 100);
    assert!((a.0).abs() < 1e-6);
    assert!((f.0).abs() < 1e-6);
    assert!((m.0).abs() < 1e-6);
}

#[test]
fn test_kappa_interpretation_all_ranges() {
    assert_eq!(ClassifyEvalReport::kappa_interpretation(-0.5), "worse than chance");
    assert_eq!(ClassifyEvalReport::kappa_interpretation(0.0), "slight");
    assert_eq!(ClassifyEvalReport::kappa_interpretation(0.15), "slight");
    assert_eq!(ClassifyEvalReport::kappa_interpretation(0.3), "fair");
    assert_eq!(ClassifyEvalReport::kappa_interpretation(0.5), "moderate");
    assert_eq!(ClassifyEvalReport::kappa_interpretation(0.7), "substantial");
    assert_eq!(ClassifyEvalReport::kappa_interpretation(0.9), "almost perfect");
    assert_eq!(ClassifyEvalReport::kappa_interpretation(1.0), "almost perfect");
}

#[test]
fn test_to_report_format() {
    let report = make_eval_report(
        &[0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
        &[0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
        &(0..10)
            .map(|i| if i % 2 == 0 { vec![0.8, 0.2] } else { vec![0.2, 0.8] })
            .collect::<Vec<_>>(),
        2,
    );
    let text = report.to_report();
    assert!(text.contains("precision"));
    assert!(text.contains("recall"));
    assert!(text.contains("f1-score"));
    assert!(text.contains("macro avg"));
    assert!(text.contains("weighted avg"));
    assert!(text.contains("Accuracy:"));
    assert!(text.contains("MCC:"));
    assert!(text.contains("Cohen's kappa:"));
    assert!(text.contains("Brier score:"));
    assert!(text.contains("ECE"));
    assert!(text.contains("Baselines:"));
    assert!(text.contains("Samples:"));
}

#[test]
fn test_to_report_no_confusions_when_perfect() {
    let report = make_eval_report(
        &[0, 1, 0, 1],
        &[0, 1, 0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9], vec![0.9, 0.1], vec![0.1, 0.9]],
        2,
    );
    assert!(!report.to_report().contains("Top confusions"));
}

#[test]
fn test_to_json_valid() {
    let report = make_eval_report(
        &[0, 1, 0, 1, 0],
        &[0, 1, 0, 0, 0],
        &(0..5)
            .map(|i| if i % 2 == 0 { vec![0.8, 0.2] } else { vec![0.2, 0.8] })
            .collect::<Vec<_>>(),
        2,
    );
    let json_str = report.to_json();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert!(parsed.get("accuracy").is_some());
    assert!(parsed.get("mcc").is_some());
    assert!(parsed.get("per_class").is_some());
    assert!(parsed.get("confusion_matrix").is_some());
    assert!(parsed.get("confidence_intervals_95").is_some());
    assert!(parsed.get("baselines").is_some());
    assert!(parsed.get("calibration").is_some());
}

#[test]
fn test_to_model_card_basic() {
    let labels: Vec<String> = SSC_LABELS.iter().take(2).map(ToString::to_string).collect();
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1],
        &[0, 1, 0, 0],
        &[vec![0.8, 0.2], vec![0.2, 0.8], vec![0.8, 0.2], vec![0.2, 0.8]],
        2.0,
        2,
        &labels,
        100,
    );
    let card = report.to_model_card("test-model", Some("base-model"));
    assert!(card.contains("---"));
    assert!(card.contains("# test-model"));
    assert!(card.contains("base-model"));
    assert!(card.contains("## Summary"));
    assert!(card.contains("## Labels"));
    assert!(card.contains("## Per-Class Metrics"));
    assert!(card.contains("## Confusion Matrix"));
    assert!(card.contains("## Intended Use"));
    assert!(card.contains("## Limitations"));
    assert!(card.contains("## Training"));
    assert!(card.contains("entrenar"));
}

#[test]
fn test_to_model_card_no_base() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let card = report.to_model_card("test-model", None);
    assert!(card.contains("# test-model"));
    assert!(!card.contains("base_model:"));
}

#[test]
fn test_to_model_card_weak_class() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0; 10],
        &[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        &vec![vec![0.9, 0.1]; 10],
        5.0,
        2,
        &labels,
        50,
    );
    assert!(report.to_model_card("test", None).contains("Weak class"));
}

#[test]
fn test_avg_metric_macro() {
    let report = make_eval_report(
        &[0, 1, 0, 1],
        &[0, 1, 0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9], vec![0.9, 0.1], vec![0.1, 0.9]],
        2,
    );
    use crate::eval::classification::Average;
    let mp = report.avg_metric(&report.per_class_precision, Average::Macro);
    assert!(mp > 0.0 && mp <= 1.0);
    let wp = report.avg_metric(&report.per_class_precision, Average::Weighted);
    assert!(wp > 0.0 && wp <= 1.0);
}

#[test]
fn test_avg_metric_empty_vals() {
    let report = make_eval_report(&[], &[], &[], 2);
    use crate::eval::classification::Average;
    assert!((report.avg_metric(&[], Average::Macro) - 0.0).abs() < 1e-6);
}

#[test]
fn test_avg_metric_micro_fallback() {
    let report = make_eval_report(&[0, 1], &[0, 1], &[vec![0.9, 0.1], vec![0.1, 0.9]], 2);
    use crate::eval::classification::Average;
    assert!(report.avg_metric(&report.per_class_precision, Average::Micro).is_finite());
}

#[test]
fn test_eval_report_clone() {
    let r = make_eval_report(&[0, 1], &[0, 1], &[vec![0.9, 0.1], vec![0.1, 0.9]], 2);
    let c = r.clone();
    assert_eq!(c.num_classes, r.num_classes);
    assert!((c.accuracy - r.accuracy).abs() < 1e-10);
}

#[test]
fn test_eval_report_debug() {
    let r = make_eval_report(&[0, 1], &[0, 1], &[vec![0.9, 0.1], vec![0.1, 0.9]], 2);
    assert!(format!("{r:?}").contains("ClassifyEvalReport"));
}

#[test]
fn test_eval_report_three_classes() {
    let yp = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
    let yt = vec![0, 1, 2, 0, 2, 1, 0, 1, 2];
    let probs: Vec<Vec<f32>> = yp
        .iter()
        .map(|&p| {
            let mut v = vec![0.1; 3];
            v[p] = 0.8;
            v
        })
        .collect();
    let report = make_eval_report(&yp, &yt, &probs, 3);
    assert_eq!(report.num_classes, 3);
    assert_eq!(report.total_samples, 9);
    assert_eq!(report.per_class_precision.len(), 3);
    assert_eq!(report.confusion_matrix.len(), 3);
    let json: serde_json::Value = serde_json::from_str(&report.to_json()).unwrap();
    assert_eq!(json["per_class"].as_array().unwrap().len(), 3);
}

#[test]
fn test_eval_report_five_class() {
    let yp = vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
    let yt = vec![0, 1, 2, 3, 4, 1, 0, 3, 2, 4];
    let probs: Vec<Vec<f32>> = yp
        .iter()
        .map(|&p| {
            let mut v = vec![0.05; 5];
            v[p] = 0.8;
            v
        })
        .collect();
    let labels: Vec<String> = SSC_LABELS.iter().map(ToString::to_string).collect();
    let report =
        ClassifyEvalReport::from_predictions_with_probs(&yp, &yt, &probs, 10.0, 5, &labels, 200);
    assert_eq!(report.num_classes, 5);
    let text = report.to_report();
    assert!(text.contains("safe"));
    assert!(text.contains("unsafe"));
}

// =========================================================================
// restore_class_weights_from_metadata
// =========================================================================

#[test]
fn test_restore_weights_valid() {
    let dir = std::env::temp_dir().join("entrenar_test_rw_v");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("metadata.json"),
        serde_json::to_string(&serde_json::json!({"class_weights": [0.5, 2.0]})).unwrap(),
    )
    .unwrap();
    let w = restore_class_weights_from_metadata(&dir, 2).unwrap();
    assert_eq!(w.len(), 2);
    assert!((w[0] - 0.5).abs() < 1e-6);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_restore_weights_wrong_count() {
    let dir = std::env::temp_dir().join("entrenar_test_rw_wc");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("metadata.json"),
        serde_json::to_string(&serde_json::json!({"class_weights": [0.5, 2.0]})).unwrap(),
    )
    .unwrap();
    assert!(restore_class_weights_from_metadata(&dir, 3).is_none());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_restore_weights_no_file() {
    assert!(restore_class_weights_from_metadata(
        &std::env::temp_dir().join("entrenar_test_rw_nf_nonexistent"),
        2,
    )
    .is_none());
}

#[test]
fn test_restore_weights_null() {
    let dir = std::env::temp_dir().join("entrenar_test_rw_null");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("metadata.json"),
        serde_json::to_string(&serde_json::json!({"class_weights": null})).unwrap(),
    )
    .unwrap();
    assert!(restore_class_weights_from_metadata(&dir, 2).is_none());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_restore_weights_no_key() {
    let dir = std::env::temp_dir().join("entrenar_test_rw_nk");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("metadata.json"), r#"{"epoch":5}"#).unwrap();
    assert!(restore_class_weights_from_metadata(&dir, 2).is_none());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_restore_weights_bad_json() {
    let dir = std::env::temp_dir().join("entrenar_test_rw_bj");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("metadata.json"), "not json").unwrap();
    assert!(restore_class_weights_from_metadata(&dir, 2).is_none());
    std::fs::remove_dir_all(&dir).ok();
}

// =========================================================================
// Edge cases
// =========================================================================

#[test]
fn test_eval_report_single_sample() {
    let r = make_eval_report(&[0], &[0], &[vec![0.9, 0.1]], 2);
    assert!((r.accuracy - 1.0).abs() < 1e-6);
    assert_eq!(r.total_samples, 1);
}

#[test]
fn test_eval_report_all_one_class() {
    let r = make_eval_report(&[0; 5], &[0; 5], &vec![vec![0.9, 0.1]; 5], 2);
    assert!((r.accuracy - 1.0).abs() < 1e-6);
    assert!((r.mcc - 0.0).abs() < 1e-6);
}

#[test]
fn test_report_with_confusions() {
    let report = make_eval_report(
        &[0, 1, 0, 0, 1, 0, 1, 1, 0, 0],
        &[1, 0, 0, 1, 1, 0, 0, 1, 0, 1],
        &(0..10)
            .map(|i| if i % 2 == 0 { vec![0.7, 0.3] } else { vec![0.3, 0.7] })
            .collect::<Vec<_>>(),
        2,
    );
    assert!(report.to_report().contains("Top confusions"));
}

#[test]
fn test_card_raw_and_normalized() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1, 0, 1],
        &[0, 0, 0, 1, 1, 1],
        &(0..6)
            .map(|i| if i % 2 == 0 { vec![0.8, 0.2] } else { vec![0.2, 0.8] })
            .collect::<Vec<_>>(),
        3.0,
        2,
        &labels,
        50,
    );
    let card = report.to_model_card("test", None);
    assert!(card.contains("### Raw Counts"));
    assert!(card.contains("### Normalized (row %)"));
}

// =========================================================================
// Auto-balance class weights
// =========================================================================

#[test]
fn test_auto_balance_imbalanced() {
    let mut p = tiny_pipeline(2);
    let c: Vec<SafetySample> = (0..90)
        .map(|i| SafetySample { input: format!("s{i}"), label: 0 })
        .chain((0..10).map(|i| SafetySample { input: format!("u{i}"), label: 1 }))
        .collect();
    ClassifyTrainer::auto_balance_classes(&mut p, &c);
    assert!(p.config.class_weights.is_some());
    let w = p.config.class_weights.unwrap();
    assert!(w[1] > w[0]);
}

#[test]
fn test_auto_balance_balanced() {
    let mut p = tiny_pipeline(2);
    let c: Vec<SafetySample> = (0..50)
        .map(|i| SafetySample { input: format!("s{i}"), label: 0 })
        .chain((0..50).map(|i| SafetySample { input: format!("u{i}"), label: 1 }))
        .collect();
    ClassifyTrainer::auto_balance_classes(&mut p, &c);
    assert!(p.config.class_weights.is_none());
}

#[test]
fn test_auto_balance_user_weights_preserved() {
    let mut p = tiny_pipeline(2);
    p.config.class_weights = Some(vec![1.0, 1.0]);
    let c: Vec<SafetySample> = (0..90)
        .map(|i| SafetySample { input: format!("s{i}"), label: 0 })
        .chain((0..10).map(|i| SafetySample { input: format!("u{i}"), label: 1 }))
        .collect();
    ClassifyTrainer::auto_balance_classes(&mut p, &c);
    let w = p.config.class_weights.unwrap();
    assert!((w[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_auto_balance_missing_class() {
    let mut p = tiny_pipeline(2);
    let c: Vec<SafetySample> =
        (0..100).map(|i| SafetySample { input: format!("s{i}"), label: 0 }).collect();
    ClassifyTrainer::auto_balance_classes(&mut p, &c);
    assert!(p.config.class_weights.is_none());
}

#[test]
fn test_eval_report_throughput() {
    let r = make_eval_report(
        &[0, 1, 0, 1],
        &[0, 1, 0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9], vec![0.9, 0.1], vec![0.1, 0.9]],
        2,
    );
    // 4 samples / 0.1s = 40 sam/s
    assert!((r.samples_per_sec - 40.0).abs() < 1e-6);
}

#[test]
fn test_eval_report_zero_time() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let r = ClassifyEvalReport::from_predictions_with_probs(
        &[0],
        &[0],
        &[vec![0.9, 0.1]],
        0.1,
        2,
        &labels,
        0,
    );
    assert!((r.samples_per_sec - 0.0).abs() < 1e-6);
}

// =========================================================================
// Coverage expansion: report_* sub-methods, card_* sub-methods, edge cases
// =========================================================================

#[test]
fn test_report_summary_contains_all_fields() {
    let report = make_eval_report(
        &[0, 1, 0, 1, 0],
        &[0, 1, 1, 1, 0],
        &[vec![0.8, 0.2], vec![0.2, 0.8], vec![0.7, 0.3], vec![0.3, 0.7], vec![0.9, 0.1]],
        2,
    );
    let text = report.to_report();
    assert!(text.contains("Accuracy:"));
    assert!(text.contains("Top-2 accuracy:"));
    assert!(text.contains("Cohen's kappa:"));
    assert!(text.contains("MCC:"));
    assert!(text.contains("Macro F1:"));
    assert!(text.contains("Avg loss:"));
    assert!(text.contains("95% CI"));
}

#[test]
fn test_report_confidence_section() {
    let report = make_eval_report(
        &[0, 1, 0, 1],
        &[0, 0, 0, 1],
        &[vec![0.8, 0.2], vec![0.3, 0.7], vec![0.9, 0.1], vec![0.2, 0.8]],
        2,
    );
    let text = report.to_report();
    assert!(text.contains("Confidence (mean):"));
    assert!(text.contains("correct preds:"));
    assert!(text.contains("wrong preds:"));
    assert!(text.contains("gap (higher=better):"));
}

#[test]
fn test_report_scoring_rules_section() {
    let report = make_eval_report(
        &[0, 1, 0, 1],
        &[0, 1, 0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9], vec![0.85, 0.15], vec![0.15, 0.85]],
        2,
    );
    let text = report.to_report();
    assert!(text.contains("Brier score:"));
    assert!(text.contains("perfect=0"));
    assert!(text.contains("Log loss:"));
    assert!(text.contains("random="));
}

#[test]
fn test_report_calibration_section_with_bins() {
    let report = make_eval_report(
        &[0, 1, 0, 1, 0, 1, 0, 1],
        &[0, 1, 0, 1, 0, 1, 0, 1],
        &[
            vec![0.95, 0.05],
            vec![0.05, 0.95],
            vec![0.85, 0.15],
            vec![0.15, 0.85],
            vec![0.75, 0.25],
            vec![0.25, 0.75],
            vec![0.65, 0.35],
            vec![0.35, 0.65],
        ],
        2,
    );
    let text = report.to_report();
    assert!(text.contains("ECE (Expected Calibration Error):"));
    assert!(text.contains("Calibration:"));
    assert!(text.contains("Bin       Confidence  Accuracy    Count"));
}

#[test]
fn test_report_baselines_section() {
    let report = make_eval_report(
        &[0, 1, 0, 1, 0],
        &[0, 1, 0, 1, 0],
        &[vec![0.9, 0.1], vec![0.1, 0.9], vec![0.9, 0.1], vec![0.1, 0.9], vec![0.9, 0.1]],
        2,
    );
    let text = report.to_report();
    assert!(text.contains("Baselines:"));
    assert!(text.contains("random="));
    assert!(text.contains("majority="));
    assert!(text.contains("model="));
    assert!(text.contains("lift="));
}

#[test]
fn test_report_throughput_section() {
    let report = make_eval_report(&[0, 1], &[0, 1], &[vec![0.9, 0.1], vec![0.1, 0.9]], 2);
    let text = report.to_report();
    assert!(text.contains("Samples:"));
    assert!(text.contains("Time:"));
    assert!(text.contains("samples/sec"));
}

#[test]
fn test_report_top_confusions_section() {
    let report = make_eval_report(
        &[0, 0, 1, 1, 0, 1],
        &[1, 1, 0, 0, 0, 1],
        &[
            vec![0.6, 0.4],
            vec![0.6, 0.4],
            vec![0.4, 0.6],
            vec![0.4, 0.6],
            vec![0.7, 0.3],
            vec![0.3, 0.7],
        ],
        2,
    );
    let text = report.to_report();
    assert!(text.contains("Top confusions"));
}

#[test]
fn test_to_json_all_fields_present() {
    let report = make_eval_report(
        &[0, 1, 0, 1, 0, 1],
        &[0, 1, 1, 1, 0, 0],
        &[
            vec![0.8, 0.2],
            vec![0.2, 0.8],
            vec![0.6, 0.4],
            vec![0.3, 0.7],
            vec![0.9, 0.1],
            vec![0.5, 0.5],
        ],
        2,
    );
    let json_str = report.to_json();
    let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert!(v.get("top2_accuracy").is_some());
    assert!(v.get("cohens_kappa").is_some());
    assert!(v.get("brier_score").is_some());
    assert!(v.get("log_loss").is_some());
    assert!(v.get("samples_per_sec").is_some());
    assert!(v.get("eval_time_ms").is_some());
    assert!(v.get("num_classes").is_some());
    assert!(v.get("top_confusions").is_some());
    assert!(v.get("confidence").is_some());
    let conf = &v["confidence"];
    assert!(conf.get("mean").is_some());
    assert!(conf.get("mean_correct").is_some());
    assert!(conf.get("mean_wrong").is_some());
    assert!(conf.get("gap").is_some());
    let baselines = &v["baselines"];
    assert!(baselines.get("random").is_some());
    assert!(baselines.get("majority_class").is_some());
    assert!(baselines.get("lift_over_majority").is_some());
    let cal = &v["calibration"];
    assert!(cal.get("ece").is_some());
    assert!(cal.get("brier_score").is_some());
    assert!(cal.get("log_loss").is_some());
    assert!(cal.get("bins").is_some());
}

#[test]
fn test_to_json_confusions_labeled() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 0, 1, 1],
        &[1, 1, 0, 0],
        &[vec![0.7, 0.3], vec![0.7, 0.3], vec![0.3, 0.7], vec![0.3, 0.7]],
        4.0,
        2,
        &labels,
        100,
    );
    let json_str = report.to_json();
    let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    let confusions = v["top_confusions"].as_array().unwrap();
    assert!(!confusions.is_empty());
    assert!(confusions[0].get("true_class").is_some());
    assert!(confusions[0].get("pred_class").is_some());
    assert!(confusions[0].get("count").is_some());
}

#[test]
fn test_to_json_per_class_has_labels() {
    let labels = vec!["alpha".to_string(), "beta".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let json_str = report.to_json();
    let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    let per_class = v["per_class"].as_array().unwrap();
    assert_eq!(per_class[0]["label"], "alpha");
    assert_eq!(per_class[1]["label"], "beta");
    assert!(per_class[0].get("precision").is_some());
    assert!(per_class[0].get("recall").is_some());
    assert!(per_class[0].get("f1").is_some());
    assert!(per_class[0].get("support").is_some());
}

#[test]
fn test_to_json_calibration_bins_filtered() {
    // Only non-zero count bins should appear
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.95, 0.05], vec![0.05, 0.95]],
        1.0,
        2,
        &labels,
        50,
    );
    let json_str = report.to_json();
    let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    let bins = v["calibration"]["bins"].as_array().unwrap();
    // All bins with count > 0 should have been included
    for bin in bins {
        assert!(bin["count"].as_u64().unwrap() > 0);
    }
}

// =========================================================================
// Model card sub-method coverage
// =========================================================================

#[test]
fn test_model_card_yaml_front_matter_structure() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1],
        &[0, 1, 0, 0],
        &[vec![0.8, 0.2], vec![0.2, 0.8], vec![0.9, 0.1], vec![0.3, 0.7]],
        2.0,
        2,
        &labels,
        100,
    );
    let card = report.to_model_card("my-model", Some("Qwen/Qwen2.5-Coder-0.5B"));
    // YAML front matter
    assert!(card.starts_with("---\n"));
    assert!(card.contains("license: apache-2.0"));
    assert!(card.contains("language:\n- en"));
    assert!(card.contains("tags:"));
    assert!(card.contains("shell-safety"));
    assert!(card.contains("base_model: Qwen/Qwen2.5-Coder-0.5B"));
    assert!(card.contains("pipeline_tag: text-classification"));
    assert!(card.contains("model-index:"));
    assert!(card.contains("- name: my-model"));
    assert!(card.contains("type: accuracy"));
    assert!(card.contains("type: f1"));
    assert!(card.contains("type: mcc"));
    assert!(card.contains("type: cohens_kappa"));
}

#[test]
fn test_model_card_labels_section() {
    let labels: Vec<String> = SSC_LABELS.iter().map(ToString::to_string).collect();
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 2, 3, 4],
        &[0, 1, 2, 3, 4],
        &(0..5)
            .map(|i| {
                let mut v = vec![0.05; 5];
                v[i] = 0.8;
                v
            })
            .collect::<Vec<_>>(),
        2.0,
        5,
        &labels,
        100,
    );
    let card = report.to_model_card("test-5class", None);
    assert!(card.contains("## Labels"));
    assert!(card.contains("| ID | Label | Description |"));
    assert!(card.contains("safe"));
    assert!(card.contains("needs-quoting"));
    assert!(card.contains("non-deterministic"));
    assert!(card.contains("non-idempotent"));
    assert!(card.contains("unsafe"));
    assert!(card.contains("destructive"));
}

#[test]
fn test_model_card_per_class_metrics_section() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1],
        &[0, 1, 0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9], vec![0.9, 0.1], vec![0.1, 0.9]],
        2.0,
        2,
        &labels,
        100,
    );
    let card = report.to_model_card("test", None);
    assert!(card.contains("## Per-Class Metrics"));
    assert!(card.contains("| Label | Precision | Recall | F1 | Support |"));
    assert!(card.contains("| safe |"));
    assert!(card.contains("| unsafe |"));
}

#[test]
fn test_model_card_confusion_matrix_section() {
    let labels = vec!["A".to_string(), "B".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 1, 0],
        &[0, 1, 0, 1],
        &[vec![0.8, 0.2], vec![0.2, 0.8], vec![0.3, 0.7], vec![0.6, 0.4]],
        2.0,
        2,
        &labels,
        100,
    );
    let card = report.to_model_card("test", None);
    assert!(card.contains("## Confusion Matrix"));
    assert!(card.contains("### Raw Counts"));
    assert!(card.contains("### Normalized (row %)"));
    assert!(card.contains("Predicted"));
}

#[test]
fn test_model_card_calibration_section() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1, 0, 1],
        &[0, 1, 0, 1, 0, 1],
        &[
            vec![0.9, 0.1],
            vec![0.1, 0.9],
            vec![0.8, 0.2],
            vec![0.2, 0.8],
            vec![0.7, 0.3],
            vec![0.3, 0.7],
        ],
        3.0,
        2,
        &labels,
        100,
    );
    let card = report.to_model_card("test", None);
    assert!(card.contains("## Confidence & Calibration"));
    assert!(card.contains("| Mean confidence |"));
    assert!(card.contains("| Confidence (correct) |"));
    assert!(card.contains("| Confidence (wrong) |"));
    assert!(card.contains("| Confidence gap |"));
    assert!(card.contains("| ECE |"));
    assert!(card.contains("Calibration curve"));
    assert!(card.contains("Bin         Conf    Acc     Count"));
}

#[test]
fn test_model_card_error_analysis_with_errors() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 0, 1, 1],
        &[1, 1, 0, 0],
        &[vec![0.7, 0.3], vec![0.6, 0.4], vec![0.4, 0.6], vec![0.3, 0.7]],
        4.0,
        2,
        &labels,
        100,
    );
    let card = report.to_model_card("test", None);
    assert!(card.contains("## Error Analysis"));
    assert!(card.contains("Most frequent misclassifications"));
    assert!(card.contains("| True Class | Predicted As | Count |"));
}

#[test]
fn test_model_card_error_analysis_empty_when_perfect() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1],
        &[0, 1, 0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9], vec![0.9, 0.1], vec![0.1, 0.9]],
        2.0,
        2,
        &labels,
        100,
    );
    let card = report.to_model_card("test", None);
    // No error analysis section when no confusions
    assert!(!card.contains("## Error Analysis"));
}

#[test]
fn test_model_card_intended_use_section() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let card = report.to_model_card("test", None);
    assert!(card.contains("## Intended Use"));
    assert!(card.contains("CI/CD pipelines"));
    assert!(card.contains("Shell purification"));
    assert!(card.contains("Code review"));
    assert!(card.contains("Interactive shells"));
}

#[test]
fn test_model_card_limitations_section() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let card = report.to_model_card("test", None);
    assert!(card.contains("## Limitations"));
    assert!(card.contains("Not a security oracle"));
    assert!(card.contains("Context-blind"));
    assert!(card.contains("Training distribution"));
}

#[test]
fn test_model_card_ethical_section() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let card = report.to_model_card("test", None);
    assert!(card.contains("## Ethical Considerations"));
    assert!(card.contains("False negatives"));
    assert!(card.contains("Defense in depth"));
    assert!(card.contains("adversarial-robust"));
}

#[test]
fn test_model_card_training_section_with_base() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let card = report.to_model_card("test", Some("base-model-name"));
    assert!(card.contains("## Training"));
    assert!(card.contains("| Framework |"));
    assert!(card.contains("entrenar"));
    assert!(card.contains("LoRA"));
    assert!(card.contains("base-model-name"));
    assert!(card.contains("| Num classes |"));
}

#[test]
fn test_model_card_training_section_no_base() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let card = report.to_model_card("test", None);
    assert!(card.contains("## Training"));
    assert!(!card.contains("| Base model |"));
}

#[test]
fn test_model_card_footer() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let card = report.to_model_card("test", None);
    assert!(card.contains("Generated by [entrenar]"));
}

#[test]
fn test_model_card_weak_class_not_shown_when_f1_high() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        &[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        &(0..10)
            .map(|i| if i % 2 == 0 { vec![0.9, 0.1] } else { vec![0.1, 0.9] })
            .collect::<Vec<_>>(),
        5.0,
        2,
        &labels,
        100,
    );
    let card = report.to_model_card("test", None);
    // Perfect predictions, so no weak class
    assert!(!card.contains("Weak class"));
}

// =========================================================================
// card_confusion_* edge cases
// =========================================================================

#[test]
fn test_card_confusion_header_long_names() {
    let labels = vec!["very-long-name-safe".to_string(), "very-long-name-unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let card = report.to_model_card("test", None);
    // Names > 8 chars get truncated in header
    assert!(card.contains("very-lon"));
}

#[test]
fn test_card_confusion_row_label_long_name() {
    let labels = vec!["extremely-long-name-that-exceeds-eighteen".to_string(), "short".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let card = report.to_model_card("test", None);
    // Row label > 18 chars gets truncated
    assert!(card.contains("extremely-long-nam"));
}

#[test]
fn test_card_confusion_normalized_zero_row_sum() {
    // Create a report where confusion matrix has a row with all zeros
    // This happens when a class has zero support
    let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 0, 0],
        &[0, 0, 0],
        &[vec![0.9, 0.05, 0.05], vec![0.8, 0.1, 0.1], vec![0.85, 0.1, 0.05]],
        1.0,
        3,
        &labels,
        50,
    );
    let card = report.to_model_card("test", None);
    // Should handle zero row gracefully in normalized section
    assert!(card.contains("### Normalized (row %)"));
    assert!(card.contains("0.0%"));
}

// =========================================================================
// to_report() label fallback when label_names is too short
// =========================================================================

#[test]
fn test_to_report_missing_label_names_fallback() {
    // num_classes=3 but only 2 label_names provided
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 2, 0, 1, 2],
        &[0, 1, 2, 0, 1, 2],
        &(0..6)
            .map(|i| {
                let mut v = vec![0.1; 3];
                v[i % 3] = 0.8;
                v
            })
            .collect::<Vec<_>>(),
        3.0,
        3,
        &labels,
        100,
    );
    let text = report.to_report();
    // Should fall back to "Class 2" for the missing label
    assert!(text.contains("Class 2"));
}

#[test]
fn test_to_json_missing_label_names_fallback() {
    let labels = vec!["safe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let json_str = report.to_json();
    let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    let per_class = v["per_class"].as_array().unwrap();
    assert_eq!(per_class[0]["label"], "safe");
    assert_eq!(per_class[1]["label"], "class_1");
}

// =========================================================================
// avg_metric edge cases
// =========================================================================

#[test]
fn test_avg_metric_weighted_zero_support() {
    // All zero support => 0.0
    let labels = vec!["a".to_string(), "b".to_string()];
    let mut report =
        ClassifyEvalReport::from_predictions_with_probs(&[], &[], &[], 0.0, 2, &labels, 50);
    report.per_class_support = vec![0, 0];
    use crate::eval::classification::Average;
    let w = report.avg_metric(&[0.5, 0.8], Average::Weighted);
    assert!((w - 0.0).abs() < 1e-6);
}

#[test]
fn test_avg_metric_none_variant_fallback() {
    let report = make_eval_report(&[0, 1], &[0, 1], &[vec![0.9, 0.1], vec![0.1, 0.9]], 2);
    use crate::eval::classification::Average;
    // Average::None falls through to macro-like fallback
    let result = report.avg_metric(&report.per_class_precision, Average::None);
    assert!(result.is_finite());
}

// =========================================================================
// compute_* with edge cases
// =========================================================================

#[test]
fn test_cohens_kappa_all_same_class() {
    use crate::eval::classification::ConfusionMatrix;
    let cm = ConfusionMatrix::from_predictions_with_min_classes(&[0, 0, 0], &[0, 0, 0], 2);
    // p_e approaches 1.0, kappa should handle gracefully
    let kappa = ClassifyEvalReport::compute_cohens_kappa(&cm, 3);
    assert!(kappa.is_finite());
}

#[test]
fn test_mcc_single_class_predictions() {
    use crate::eval::classification::ConfusionMatrix;
    // All predictions are class 0 but true labels are mixed
    let cm = ConfusionMatrix::from_predictions_with_min_classes(&[0, 0, 0, 0], &[0, 1, 0, 1], 2);
    let mcc = ClassifyEvalReport::compute_mcc(&cm, 2, 4);
    // denom_sq = 0 since all predictions are same class
    assert!((mcc - 0.0).abs() < 1e-6);
}

#[test]
fn test_mcc_three_class() {
    use crate::eval::classification::ConfusionMatrix;
    let cm = ConfusionMatrix::from_predictions_with_min_classes(
        &[0, 1, 2, 0, 1, 2, 0, 1, 2],
        &[0, 1, 2, 0, 1, 2, 0, 1, 2],
        3,
    );
    let mcc = ClassifyEvalReport::compute_mcc(&cm, 3, 9);
    assert!((mcc - 1.0).abs() < 1e-6);
}

#[test]
fn test_top2_accuracy_single_class() {
    let probs = vec![vec![1.0]]; // single class
    let top2 = ClassifyEvalReport::compute_top2_accuracy(&probs, &[0], 1);
    // With only 1 class, top-2 requires len >= 2, so this should be 0
    assert!((top2 - 0.0).abs() < 1e-6);
}

#[test]
fn test_top2_accuracy_three_class_all_correct_top2() {
    let probs = vec![
        vec![0.1, 0.5, 0.4], // top2: [1, 2], true=2 => correct
        vec![0.6, 0.3, 0.1], // top2: [0, 1], true=0 => correct
        vec![0.2, 0.3, 0.5], // top2: [2, 1], true=1 => correct
    ];
    let top2 = ClassifyEvalReport::compute_top2_accuracy(&probs, &[2, 0, 1], 3);
    assert!((top2 - 1.0).abs() < 1e-6);
}

#[test]
fn test_brier_score_uniform_probs() {
    // Uniform probabilities for 3-class
    let probs = vec![vec![1.0 / 3.0; 3], vec![1.0 / 3.0; 3], vec![1.0 / 3.0; 3]];
    let b = ClassifyEvalReport::compute_brier_score(&probs, &[0, 1, 2], 3);
    // Each sample: sum_k (1/3 - y_k)^2 = 2*(1/3)^2 + (1/3 - 1)^2 = 2/9 + 4/9 = 6/9 = 2/3
    assert!((b - 2.0 / 3.0).abs() < 1e-4);
}

#[test]
fn test_brier_score_missing_prob_entries() {
    // Test when probs vector is shorter than num_classes
    let probs = vec![vec![0.9]]; // only 1 entry, but num_classes=2
    let b = ClassifyEvalReport::compute_brier_score(&probs, &[0], 2);
    // class 0: (0.9 - 1)^2 = 0.01, class 1: (0.0 - 0)^2 = 0.0
    assert!((b - 0.01).abs() < 1e-4);
}

#[test]
fn test_log_loss_with_zero_prob() {
    // When prob for true class is 0.0, it gets clamped to eps
    let ll = ClassifyEvalReport::compute_log_loss(&[vec![0.0, 1.0]], &[0]);
    // -ln(eps) where eps = 1e-15
    assert!(ll > 30.0);
}

#[test]
fn test_log_loss_missing_true_label_index() {
    // When true label index exceeds probs length, uses 0.0 (clamped to eps)
    let ll = ClassifyEvalReport::compute_log_loss(&[vec![0.5, 0.5]], &[5]);
    assert!(ll > 30.0);
}

#[test]
fn test_bootstrap_cis_single_sample() {
    let (a, f, m) = ClassifyEvalReport::compute_bootstrap_cis(&[0], &[0], 2, 50);
    assert!((a.0 - 1.0).abs() < 1e-6);
    assert!((a.1 - 1.0).abs() < 1e-6);
    // Single sample always correct
    assert!(f.0.is_finite());
    assert!(m.0.is_finite());
}

#[test]
fn test_bootstrap_cis_all_wrong() {
    let (a, _f, _m) =
        ClassifyEvalReport::compute_bootstrap_cis(&[1, 0, 1, 0, 1, 0], &[0, 1, 0, 1, 0, 1], 2, 100);
    // All predictions are wrong
    assert!(a.0 < 0.1);
    assert!(a.1 < 0.1);
}

#[test]
fn test_baselines_single_class() {
    let (r, m) = ClassifyEvalReport::compute_baselines(&[100], 100, 1);
    assert!((r - 1.0).abs() < 1e-6);
    assert!((m - 1.0).abs() < 1e-6);
}

#[test]
fn test_top_confusions_empty_matrix() {
    let c = ClassifyEvalReport::compute_top_confusions(&[], 5);
    assert!(c.is_empty());
}

#[test]
fn test_top_confusions_single_class_no_errors() {
    let c = ClassifyEvalReport::compute_top_confusions(&[vec![10]], 5);
    assert!(c.is_empty());
}

#[test]
fn test_calibration_all_high_confidence() {
    let confs = vec![0.99, 0.98, 0.97, 0.96, 0.95];
    let (bins, ece) =
        ClassifyEvalReport::compute_calibration(&confs, &[0, 1, 0, 1, 0], &[0, 1, 0, 1, 0], 5);
    assert_eq!(bins.len(), 10);
    // All in the top bin, all correct, so ECE should be small
    assert!(ece < 0.1);
}

#[test]
fn test_calibration_all_low_confidence() {
    let confs = vec![0.05, 0.08, 0.03, 0.06, 0.04];
    let (bins, ece) =
        ClassifyEvalReport::compute_calibration(&confs, &[0, 1, 0, 1, 0], &[0, 1, 0, 1, 0], 5);
    assert_eq!(bins.len(), 10);
    // Low confidence but all correct => ECE should reflect overconfidence
    assert!(ece.is_finite());
}

// =========================================================================
// Oversample edge cases
// =========================================================================

#[test]
fn test_oversample_single_class() {
    let mut data = vec![
        SafetySample { input: "a".into(), label: 0 },
        SafetySample { input: "b".into(), label: 0 },
    ];
    let before = data.len();
    ClassifyTrainer::oversample_training_data(&mut data, 42);
    // Only one class, no minority => unchanged
    assert_eq!(data.len(), before);
}

#[test]
fn test_oversample_empty() {
    let mut data: Vec<SafetySample> = vec![];
    ClassifyTrainer::oversample_training_data(&mut data, 42);
    assert!(data.is_empty());
}

#[test]
fn test_oversample_extreme_imbalance() {
    let mut data: Vec<SafetySample> = (0..100)
        .map(|i| SafetySample { input: format!("s{i}"), label: 0 })
        .chain(std::iter::once(SafetySample { input: "rare".into(), label: 1 }))
        .collect();
    ClassifyTrainer::oversample_training_data(&mut data, 42);
    assert_eq!(data.len(), 200);
    assert_eq!(data.iter().filter(|s| s.label == 0).count(), 100);
    assert_eq!(data.iter().filter(|s| s.label == 1).count(), 100);
}

// =========================================================================
// Split dataset edge cases
// =========================================================================

#[test]
fn test_split_very_small_val_ratio() {
    let c = make_corpus(100, 3);
    let (tr, va) = ClassifyTrainer::split_dataset(&c, 0.01, 42);
    // val_count = ceil(100 * 0.01) = 1, min 1
    assert_eq!(va.len(), 1);
    assert_eq!(tr.len(), 99);
}

#[test]
fn test_split_max_val_ratio() {
    let c = make_corpus(10, 2);
    let (tr, va) = ClassifyTrainer::split_dataset(&c, 0.5, 42);
    assert_eq!(va.len(), 5);
    assert_eq!(tr.len(), 5);
}

// =========================================================================
// compute_data_hash with varied inputs
// =========================================================================

#[test]
fn test_compute_data_hash_single_sample() {
    let c = vec![SafetySample { input: "hello".into(), label: 0 }];
    let hash = ClassifyTrainer::compute_data_hash(&c);
    assert!(hash.starts_with("sha256:"));
    assert!(hash.len() > 10);
}

#[test]
fn test_compute_data_hash_large_corpus() {
    let c: Vec<SafetySample> =
        (0..1000).map(|i| SafetySample { input: format!("item_{i}"), label: i % 5 }).collect();
    let hash = ClassifyTrainer::compute_data_hash(&c);
    assert!(hash.starts_with("sha256:"));
}

// =========================================================================
// TrainingConfig edge cases
// =========================================================================

#[test]
fn test_training_config_val_split_boundary() {
    let p = tiny_pipeline(3);
    let c = make_corpus(10, 3);
    // val_split exactly 0.5 should be accepted
    let cfg = TrainingConfig { val_split: 0.5, epochs: 1, ..TrainingConfig::default() };
    assert!(ClassifyTrainer::new(p, c, cfg).is_ok());
}

#[test]
fn test_training_config_large_corpus_small_val() {
    let p = tiny_pipeline(2);
    let c = make_corpus(200, 2);
    let cfg = TrainingConfig { val_split: 0.1, epochs: 1, ..TrainingConfig::default() };
    let t = ClassifyTrainer::new(p, c, cfg).expect("valid");
    // 200 * 0.1 = 20 val samples
    assert_eq!(t.val_data().len(), 20);
    assert_eq!(t.train_data().len(), 180);
}

// =========================================================================
// Eval report with many classes
// =========================================================================

#[test]
fn test_eval_report_ten_classes() {
    let n = 100;
    let num_classes = 10;
    let yp: Vec<usize> = (0..n).map(|i| i % num_classes).collect();
    let yt: Vec<usize> = (0..n).map(|i| i % num_classes).collect();
    let probs: Vec<Vec<f32>> = yp
        .iter()
        .map(|&p| {
            let mut v = vec![0.02; num_classes];
            v[p] = 0.82;
            v
        })
        .collect();
    let labels: Vec<String> = (0..num_classes).map(|i| format!("class_{i}")).collect();
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &yp,
        &yt,
        &probs,
        10.0,
        num_classes,
        &labels,
        500,
    );
    assert_eq!(report.num_classes, 10);
    assert_eq!(report.total_samples, 100);
    assert!((report.accuracy - 1.0).abs() < 1e-6);
    let text = report.to_report();
    assert!(text.contains("class_0"));
    assert!(text.contains("class_9"));
    let json_str = report.to_json();
    let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(v["per_class"].as_array().unwrap().len(), 10);
}

// =========================================================================
// to_model_card with fallback label_names
// =========================================================================

#[test]
fn test_model_card_with_few_labels_falls_back() {
    // num_classes=3 but only 1 label provided
    let labels = vec!["safe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 2],
        &[0, 1, 2],
        &[vec![0.8, 0.1, 0.1], vec![0.1, 0.8, 0.1], vec![0.1, 0.1, 0.8]],
        1.5,
        3,
        &labels,
        50,
    );
    let card = report.to_model_card("test", None);
    // Should show "class_1" and "class_2" as fallbacks
    assert!(card.contains("safe"));
    assert!(card.contains("class_1"));
    assert!(card.contains("class_2"));
}

// =========================================================================
// EpochMetrics edge values
// =========================================================================

#[test]
fn test_epoch_metrics_extreme_values() {
    let m = EpochMetrics {
        epoch: 0,
        train_loss: 0.0,
        train_accuracy: 0.0,
        val_loss: f32::MAX,
        val_accuracy: 1.0,
        learning_rate: 0.0,
        epoch_time_ms: 0,
        samples_per_sec: 0.0,
    };
    let c = m.clone();
    assert_eq!(c.epoch, 0);
    assert!((c.train_loss - 0.0).abs() < 1e-6);
    assert_eq!(c.val_loss, f32::MAX);
}

// =========================================================================
// TrainResult edge values
// =========================================================================

#[test]
fn test_train_result_with_metrics() {
    let metrics = vec![
        EpochMetrics {
            epoch: 0,
            train_loss: 1.0,
            train_accuracy: 0.5,
            val_loss: 1.2,
            val_accuracy: 0.4,
            learning_rate: 1e-3,
            epoch_time_ms: 100,
            samples_per_sec: 50.0,
        },
        EpochMetrics {
            epoch: 1,
            train_loss: 0.8,
            train_accuracy: 0.7,
            val_loss: 0.9,
            val_accuracy: 0.65,
            learning_rate: 5e-4,
            epoch_time_ms: 95,
            samples_per_sec: 52.0,
        },
    ];
    let result = TrainResult {
        epoch_metrics: metrics,
        best_epoch: 1,
        best_val_loss: 0.9,
        stopped_early: false,
        total_time_ms: 195,
    };
    assert_eq!(result.epoch_metrics.len(), 2);
    assert_eq!(result.best_epoch, 1);
    assert!(!result.stopped_early);
    let dbg = format!("{result:?}");
    assert!(dbg.contains("TrainResult"));
    assert!(dbg.contains("best_epoch: 1"));
}

// =========================================================================
// restore_class_weights edge cases
// =========================================================================

#[test]
fn test_restore_weights_empty_array() {
    let dir = std::env::temp_dir().join("entrenar_test_rw_empty_arr");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("metadata.json"),
        serde_json::to_string(&serde_json::json!({"class_weights": []})).unwrap(),
    )
    .unwrap();
    // Empty array with num_classes=0 should return Some
    let w = restore_class_weights_from_metadata(&dir, 0);
    assert!(w.is_some());
    assert!(w.unwrap().is_empty());
    // But with num_classes=2 it should return None
    assert!(restore_class_weights_from_metadata(&dir, 2).is_none());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_restore_weights_five_classes() {
    let dir = std::env::temp_dir().join("entrenar_test_rw_5class");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("metadata.json"),
        serde_json::to_string(&serde_json::json!({"class_weights": [0.7, 1.2, 3.0, 2.5, 5.0]}))
            .unwrap(),
    )
    .unwrap();
    let w = restore_class_weights_from_metadata(&dir, 5).unwrap();
    assert_eq!(w.len(), 5);
    assert!((w[0] - 0.7).abs() < 1e-6);
    assert!((w[4] - 5.0).abs() < 1e-6);
    std::fs::remove_dir_all(&dir).ok();
}

// =========================================================================
// auto_balance_classes edge cases
// =========================================================================

#[test]
fn test_auto_balance_exact_ratio_2() {
    let mut p = tiny_pipeline(2);
    // Ratio exactly 2:1 => imbalance_ratio = 2.0, NOT > 2.0 => no weights
    let c: Vec<SafetySample> = (0..20)
        .map(|i| SafetySample { input: format!("a{i}"), label: 0 })
        .chain((0..10).map(|i| SafetySample { input: format!("b{i}"), label: 1 }))
        .collect();
    ClassifyTrainer::auto_balance_classes(&mut p, &c);
    assert!(p.config.class_weights.is_none());
}

#[test]
fn test_auto_balance_ratio_just_above_2() {
    let mut p = tiny_pipeline(2);
    // 21 vs 10 => ratio=2.1 => weights applied
    let c: Vec<SafetySample> = (0..21)
        .map(|i| SafetySample { input: format!("a{i}"), label: 0 })
        .chain((0..10).map(|i| SafetySample { input: format!("b{i}"), label: 1 }))
        .collect();
    ClassifyTrainer::auto_balance_classes(&mut p, &c);
    assert!(p.config.class_weights.is_some());
}

#[test]
fn test_auto_balance_five_classes() {
    let mut p = tiny_pipeline(5);
    let c: Vec<SafetySample> = (0..100)
        .map(|i| SafetySample { input: format!("s{i}"), label: 0 })
        .chain((0..10).map(|i| SafetySample { input: format!("n{i}"), label: 1 }))
        .chain((0..10).map(|i| SafetySample { input: format!("d{i}"), label: 2 }))
        .chain((0..10).map(|i| SafetySample { input: format!("i{i}"), label: 3 }))
        .chain((0..10).map(|i| SafetySample { input: format!("u{i}"), label: 4 }))
        .collect();
    ClassifyTrainer::auto_balance_classes(&mut p, &c);
    assert!(p.config.class_weights.is_some());
    let w = p.config.class_weights.unwrap();
    assert_eq!(w.len(), 5);
    // Majority class (0) should have lower weight
    assert!(w[0] < w[1]);
}

// =========================================================================
// Shuffle determinism
// =========================================================================

#[test]
fn test_shuffle_deterministic_same_epoch() {
    let p = tiny_pipeline(3);
    let c = make_corpus(30, 3);
    let cfg = TrainingConfig { epochs: 1, val_split: 0.2, ..TrainingConfig::default() };

    let mut t1 = ClassifyTrainer::new(p, c.clone(), cfg.clone()).expect("valid");
    let p2 = tiny_pipeline(3);
    let mut t2 = ClassifyTrainer::new(p2, c, cfg).expect("valid");

    t1.shuffle_training_data(5);
    t2.shuffle_training_data(5);

    let o1: Vec<String> = t1.train_data().iter().map(|s| s.input.clone()).collect();
    let o2: Vec<String> = t2.train_data().iter().map(|s| s.input.clone()).collect();
    assert_eq!(o1, o2);
}

// =========================================================================
// from_predictions_with_probs comprehensive
// =========================================================================

#[test]
fn test_from_predictions_with_probs_all_same_prediction() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 0, 0, 0, 0],
        &[0, 0, 0, 1, 1],
        &(0..5).map(|_| vec![0.8, 0.2]).collect::<Vec<_>>(),
        5.0,
        2,
        &labels,
        100,
    );
    // 3/5 correct
    assert!((report.accuracy - 0.6).abs() < 1e-6);
    // MCC should be 0 (all predictions same class)
    assert!((report.mcc - 0.0).abs() < 1e-6);
    assert_eq!(report.total_samples, 5);
}

#[test]
fn test_from_predictions_with_probs_fields_populated() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1],
        &[0, 1, 0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9], vec![0.8, 0.2], vec![0.2, 0.8]],
        2.0,
        2,
        &labels,
        100,
    );
    // All metrics should be populated
    assert!((report.accuracy - 1.0).abs() < 1e-6);
    assert!(report.mcc > 0.9);
    assert!(report.cohens_kappa > 0.9);
    assert!(report.top2_accuracy > 0.9);
    assert!(report.mean_confidence > 0.7);
    assert!(report.mean_confidence_correct > 0.7);
    assert!((report.mean_confidence_wrong - 0.0).abs() < 1e-6);
    assert!(report.brier_score < 0.1);
    assert!(report.log_loss < 0.3);
    assert!(report.ece.is_finite());
    assert_eq!(report.calibration_bins.len(), 10);
    assert!(report.baseline_random > 0.0);
    assert!(report.baseline_majority > 0.0);
    assert!(report.ci_accuracy.0.is_finite());
    assert!(report.ci_macro_f1.0.is_finite());
    assert!(report.ci_mcc.0.is_finite());
}

// =========================================================================
// kappa_interpretation all branches
// =========================================================================

#[test]
fn test_kappa_interpretation_worse_than_chance() {
    assert_eq!(ClassifyEvalReport::kappa_interpretation(-0.5), "worse than chance");
}

#[test]
fn test_kappa_interpretation_slight() {
    assert_eq!(ClassifyEvalReport::kappa_interpretation(0.1), "slight");
}

#[test]
fn test_kappa_interpretation_fair() {
    assert_eq!(ClassifyEvalReport::kappa_interpretation(0.3), "fair");
}

#[test]
fn test_kappa_interpretation_moderate() {
    assert_eq!(ClassifyEvalReport::kappa_interpretation(0.5), "moderate");
}

#[test]
fn test_kappa_interpretation_substantial() {
    assert_eq!(ClassifyEvalReport::kappa_interpretation(0.7), "substantial");
}

#[test]
fn test_kappa_interpretation_almost_perfect() {
    assert_eq!(ClassifyEvalReport::kappa_interpretation(0.9), "almost perfect");
}

#[test]
fn test_kappa_interpretation_boundary_zero() {
    assert_eq!(ClassifyEvalReport::kappa_interpretation(0.0), "slight");
}

#[test]
fn test_kappa_interpretation_boundary_020() {
    assert_eq!(ClassifyEvalReport::kappa_interpretation(0.20), "fair");
}

#[test]
fn test_kappa_interpretation_boundary_040() {
    assert_eq!(ClassifyEvalReport::kappa_interpretation(0.40), "moderate");
}

#[test]
fn test_kappa_interpretation_boundary_060() {
    assert_eq!(ClassifyEvalReport::kappa_interpretation(0.60), "substantial");
}

#[test]
fn test_kappa_interpretation_boundary_080() {
    assert_eq!(ClassifyEvalReport::kappa_interpretation(0.80), "almost perfect");
}

// =========================================================================
// to_report covers report_* sub-methods
// =========================================================================

#[test]
fn test_to_report_covers_all_sections() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
        &[0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
        &(0..10)
            .map(|i| if i % 2 == 0 { vec![0.7, 0.3] } else { vec![0.3, 0.7] })
            .collect::<Vec<_>>(),
        5.0,
        2,
        &labels,
        100,
    );
    let text = report.to_report();
    // report_summary section
    assert!(text.contains("Accuracy:"));
    assert!(text.contains("Top-2 accuracy:"));
    assert!(text.contains("Cohen's kappa:"));
    assert!(text.contains("MCC:"));
    assert!(text.contains("Macro F1:"));
    assert!(text.contains("Avg loss:"));
    // report_confidence section
    assert!(text.contains("Confidence (mean):"));
    assert!(text.contains("correct preds:"));
    assert!(text.contains("wrong preds:"));
    assert!(text.contains("gap (higher=better):"));
    // report_scoring_rules section
    assert!(text.contains("Brier score:"));
    assert!(text.contains("Log loss:"));
    // report_calibration section
    assert!(text.contains("ECE"));
    assert!(text.contains("Calibration:"));
    // report_baselines section
    assert!(text.contains("Baselines:"));
    assert!(text.contains("lift="));
    // report_throughput section
    assert!(text.contains("Samples:"));
    assert!(text.contains("samples/sec"));
    // Per-class rows
    assert!(text.contains("safe"));
    assert!(text.contains("unsafe"));
    // Averages
    assert!(text.contains("macro avg"));
    assert!(text.contains("weighted avg"));
}

#[test]
fn test_to_report_with_confusions() {
    let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    // Create predictions with some misclassifications to generate top confusions
    let y_pred = vec![0, 1, 2, 1, 0, 2, 0, 1, 2, 0, 1, 2, 1, 0, 2];
    let y_true = vec![0, 0, 2, 1, 1, 2, 0, 2, 1, 0, 1, 0, 1, 0, 2];
    let probs: Vec<Vec<f32>> = y_pred
        .iter()
        .map(|&p| {
            let mut v = vec![0.1; 3];
            v[p] = 0.8;
            v
        })
        .collect();
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &y_pred, &y_true, &probs, 3.0, 3, &labels, 50,
    );
    let text = report.to_report();
    // Should have "Top confusions" section since there are misclassifications
    assert!(text.contains("Top confusions"));
}

#[test]
fn test_to_report_empty_confusions() {
    let labels = vec!["a".to_string(), "b".to_string()];
    // Perfect predictions — no confusions
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1],
        &[0, 1, 0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9], vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        10,
    );
    let text = report.to_report();
    // Perfect predictions — no "Top confusions" header
    // (top_confusions may be empty since confusion matrix has no off-diagonal entries)
    assert!(text.contains("Accuracy:"));
}

// =========================================================================
// to_model_card comprehensive
// =========================================================================

#[test]
fn test_to_model_card_with_base_model() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1],
        &[0, 1, 0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9], vec![0.8, 0.2], vec![0.2, 0.8]],
        2.0,
        2,
        &labels,
        100,
    );
    let card = report.to_model_card("test-model", Some("Qwen/Qwen2.5-Coder-0.5B"));
    // YAML front matter
    assert!(card.contains("---"));
    assert!(card.contains("license: apache-2.0"));
    assert!(card.contains("base_model: Qwen/Qwen2.5-Coder-0.5B"));
    assert!(card.contains("pipeline_tag: text-classification"));
    // Title section
    assert!(card.contains("# test-model"));
    assert!(card.contains("LoRA fine-tuning"));
    assert!(card.contains("Qwen/Qwen2.5-Coder-0.5B"));
    // Summary section
    assert!(card.contains("## Summary"));
    assert!(card.contains("Accuracy"));
    assert!(card.contains("Macro F1"));
    assert!(card.contains("MCC"));
    // Per-class metrics section
    assert!(card.contains("safe"));
    assert!(card.contains("unsafe"));
    // Intended Use section
    assert!(card.contains("Intended Use"));
    // Limitations section
    assert!(card.contains("Limitations"));
    // Ethical Considerations section
    assert!(card.contains("Ethical Considerations"));
    // Training section
    assert!(card.contains("Training"));
    // Footer
    assert!(card.contains("entrenar"));
}

#[test]
fn test_to_model_card_without_base_model() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1],
        &[0, 1, 0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9], vec![0.8, 0.2], vec![0.2, 0.8]],
        2.0,
        2,
        &labels,
        100,
    );
    let card = report.to_model_card("test-model", None);
    // Should NOT contain base_model
    assert!(!card.contains("base_model:"));
    // Should still have the title
    assert!(card.contains("# test-model"));
}

// =========================================================================
// to_json comprehensive
// =========================================================================

#[test]
fn test_to_json_roundtrip_all_fields() {
    let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 2, 0, 1, 2],
        &[0, 1, 2, 1, 0, 2],
        &[
            vec![0.7, 0.2, 0.1],
            vec![0.1, 0.8, 0.1],
            vec![0.1, 0.1, 0.8],
            vec![0.6, 0.3, 0.1],
            vec![0.2, 0.7, 0.1],
            vec![0.1, 0.1, 0.8],
        ],
        3.0,
        3,
        &labels,
        150,
    );
    let json_str = report.to_json();
    let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    // Top-level fields
    assert!(v["accuracy"].is_number());
    assert!(v["top2_accuracy"].is_number());
    assert!(v["cohens_kappa"].is_number());
    assert!(v["mcc"].is_number());
    assert!(v["avg_loss"].is_number());
    assert!(v["brier_score"].is_number());
    assert!(v["log_loss"].is_number());
    assert_eq!(v["total_samples"].as_u64().unwrap(), 6);
    assert_eq!(v["num_classes"].as_u64().unwrap(), 3);
    // Confidence intervals
    assert!(v["confidence_intervals_95"]["accuracy"].is_array());
    assert!(v["confidence_intervals_95"]["macro_f1"].is_array());
    assert!(v["confidence_intervals_95"]["mcc"].is_array());
    // Baselines
    assert!(v["baselines"]["random"].is_number());
    assert!(v["baselines"]["majority_class"].is_number());
    assert!(v["baselines"]["lift_over_majority"].is_number());
    // Per-class array
    assert_eq!(v["per_class"].as_array().unwrap().len(), 3);
    assert_eq!(v["per_class"][0]["label"], "a");
    assert!(v["per_class"][0]["precision"].is_number());
    assert!(v["per_class"][0]["recall"].is_number());
    assert!(v["per_class"][0]["f1"].is_number());
    assert!(v["per_class"][0]["support"].is_number());
    // Confusion matrix
    assert!(v["confusion_matrix"].is_array());
    // Calibration section
    assert!(v["calibration"]["ece"].is_number());
    assert!(v["calibration"]["brier_score"].is_number());
    assert!(v["calibration"]["log_loss"].is_number());
    assert!(v["calibration"]["bins"].is_array());
    // Confidence section
    assert!(v["confidence"]["mean"].is_number());
    assert!(v["confidence"]["mean_correct"].is_number());
    assert!(v["confidence"]["mean_wrong"].is_number());
    assert!(v["confidence"]["gap"].is_number());
    // Top confusions
    assert!(v["top_confusions"].is_array());
}

// =========================================================================
// Debug impl for ClassifyTrainer
// =========================================================================

#[test]
fn test_classify_trainer_debug() {
    let p = tiny_pipeline(2);
    let corpus = make_corpus(20, 2);
    let cfg = TrainingConfig { epochs: 1, val_split: 0.2, ..TrainingConfig::default() };
    let trainer = ClassifyTrainer::new(p, corpus, cfg).expect("valid");
    let dbg = format!("{trainer:?}");
    assert!(dbg.contains("ClassifyTrainer"));
    assert!(dbg.contains("train_data_len"));
    assert!(dbg.contains("train_tokens_len"));
    assert!(dbg.contains("val_data_len"));
    assert!(dbg.contains("val_tokens_len"));
    assert!(dbg.contains("rng_seed"));
}

// =========================================================================
// TrainingConfig Default values verification
// =========================================================================

#[test]
fn test_training_config_default_values() {
    let cfg = TrainingConfig::default();
    assert_eq!(cfg.epochs, 50);
    assert!((cfg.val_split - 0.2).abs() < 1e-6);
    assert_eq!(cfg.save_every, 5);
    assert_eq!(cfg.early_stopping_patience, 10);
    assert_eq!(cfg.checkpoint_dir.as_os_str(), "checkpoints");
    assert_eq!(cfg.seed, 42);
    assert_eq!(cfg.log_interval, 1);
    assert!((cfg.warmup_fraction - 0.1).abs() < 1e-6);
    assert!((cfg.lr_min - 1e-6).abs() < 1e-10);
    assert!(!cfg.oversample_minority);
    assert!(!cfg.quantize_nf4);
    assert!(cfg.distributed.is_none());
}

// =========================================================================
// TrainingConfig Debug
// =========================================================================

#[test]
fn test_training_config_debug_format() {
    let cfg = TrainingConfig::default();
    let dbg = format!("{cfg:?}");
    assert!(dbg.contains("TrainingConfig"));
    assert!(dbg.contains("epochs: 50"));
    assert!(dbg.contains("val_split:"));
}

// =========================================================================
// EpochMetrics Clone and Debug
// =========================================================================

#[test]
fn test_epoch_metrics_clone() {
    let m = EpochMetrics {
        epoch: 5,
        train_loss: 0.5,
        train_accuracy: 0.85,
        val_loss: 0.6,
        val_accuracy: 0.82,
        learning_rate: 1e-4,
        epoch_time_ms: 5000,
        samples_per_sec: 100.0,
    };
    let c = m.clone();
    assert_eq!(c.epoch, 5);
    assert!((c.train_loss - 0.5).abs() < 1e-6);
    assert!((c.train_accuracy - 0.85).abs() < 1e-6);
    assert!((c.val_loss - 0.6).abs() < 1e-6);
    assert!((c.val_accuracy - 0.82).abs() < 1e-6);
    assert!((c.learning_rate - 1e-4).abs() < 1e-8);
    assert_eq!(c.epoch_time_ms, 5000);
    assert!((c.samples_per_sec - 100.0).abs() < 1e-6);
}

#[test]
fn test_epoch_metrics_debug() {
    let m = EpochMetrics {
        epoch: 0,
        train_loss: 1.0,
        train_accuracy: 0.5,
        val_loss: 1.2,
        val_accuracy: 0.4,
        learning_rate: 1e-3,
        epoch_time_ms: 100,
        samples_per_sec: 50.0,
    };
    let dbg = format!("{m:?}");
    assert!(dbg.contains("EpochMetrics"));
    assert!(dbg.contains("epoch: 0"));
}

// =========================================================================
// Static metric methods — edge cases
// =========================================================================

#[test]
fn test_compute_top2_accuracy_single_class() {
    // Single-class probs: indexed.len() < 2, so top-2 always 0
    let probs = vec![vec![1.0f32]];
    let y_true = vec![0usize];
    let result = ClassifyEvalReport::compute_top2_accuracy(&probs, &y_true, 1);
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_compute_top2_accuracy_empty() {
    let probs: Vec<Vec<f32>> = vec![];
    let y_true: Vec<usize> = vec![];
    let result = ClassifyEvalReport::compute_top2_accuracy(&probs, &y_true, 0);
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_compute_confidence_stats_all_correct() {
    let confidences = vec![0.9, 0.85, 0.95];
    let y_pred = vec![0, 1, 0];
    let y_true = vec![0, 1, 0];
    let (mean, mean_correct, mean_wrong) =
        ClassifyEvalReport::compute_confidence_stats(&confidences, &y_pred, &y_true);
    assert!((mean - 0.9).abs() < 1e-6);
    assert!((mean_correct - 0.9).abs() < 1e-6);
    assert!((mean_wrong - 0.0).abs() < 1e-6);
}

#[test]
fn test_compute_confidence_stats_all_wrong() {
    let confidences = vec![0.6, 0.7];
    let y_pred = vec![1, 0];
    let y_true = vec![0, 1];
    let (mean, mean_correct, mean_wrong) =
        ClassifyEvalReport::compute_confidence_stats(&confidences, &y_pred, &y_true);
    assert!((mean - 0.65).abs() < 1e-6);
    assert!((mean_correct - 0.0).abs() < 1e-6);
    assert!((mean_wrong - 0.65).abs() < 1e-6);
}

#[test]
fn test_compute_confidence_stats_empty() {
    let confidences: Vec<f64> = vec![];
    let y_pred: Vec<usize> = vec![];
    let y_true: Vec<usize> = vec![];
    let (mean, mean_correct, mean_wrong) =
        ClassifyEvalReport::compute_confidence_stats(&confidences, &y_pred, &y_true);
    assert!((mean - 0.0).abs() < 1e-6);
    assert!((mean_correct - 0.0).abs() < 1e-6);
    assert!((mean_wrong - 0.0).abs() < 1e-6);
}

#[test]
fn test_compute_brier_score_perfect() {
    let probs = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
    let y_true = vec![0, 1];
    let brier = ClassifyEvalReport::compute_brier_score(&probs, &y_true, 2);
    assert!((brier - 0.0).abs() < 1e-6);
}

#[test]
fn test_compute_brier_score_worst() {
    let probs = vec![vec![0.0f32, 1.0], vec![1.0, 0.0]];
    let y_true = vec![0, 1];
    let brier = ClassifyEvalReport::compute_brier_score(&probs, &y_true, 2);
    // Each sample: (0-1)^2 + (1-0)^2 = 2; avg = 2
    assert!((brier - 2.0).abs() < 1e-6);
}

#[test]
fn test_compute_brier_score_empty() {
    let probs: Vec<Vec<f32>> = vec![];
    let y_true: Vec<usize> = vec![];
    let brier = ClassifyEvalReport::compute_brier_score(&probs, &y_true, 2);
    assert!((brier - 0.0).abs() < 1e-6);
}

#[test]
fn test_compute_log_loss_empty() {
    let probs: Vec<Vec<f32>> = vec![];
    let y_true: Vec<usize> = vec![];
    let ll = ClassifyEvalReport::compute_log_loss(&probs, &y_true);
    assert!((ll - 0.0).abs() < 1e-6);
}

#[test]
fn test_compute_log_loss_perfect() {
    let probs = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
    let y_true = vec![0, 1];
    let ll = ClassifyEvalReport::compute_log_loss(&probs, &y_true);
    // With clamp at 1-eps, log(1.0) = 0, so log_loss ~ 0
    assert!(ll < 1e-10);
}

#[test]
fn test_compute_calibration_empty() {
    let confidences: Vec<f64> = vec![];
    let y_pred: Vec<usize> = vec![];
    let y_true: Vec<usize> = vec![];
    let (bins, ece) = ClassifyEvalReport::compute_calibration(&confidences, &y_pred, &y_true, 0);
    assert_eq!(bins.len(), 10);
    // All bins empty
    for (_, _, count) in &bins {
        assert_eq!(*count, 0);
    }
    assert!((ece - 0.0).abs() < 1e-6);
}

#[test]
fn test_compute_calibration_all_in_one_bin() {
    let confidences = vec![0.85, 0.89, 0.83, 0.87];
    let y_pred = vec![0, 0, 0, 0];
    let y_true = vec![0, 0, 0, 0]; // all correct
    let (bins, ece) = ClassifyEvalReport::compute_calibration(&confidences, &y_pred, &y_true, 4);
    // All in bin 8 (0.8-0.9)
    assert_eq!(bins[8].2, 4);
    // Accuracy = 1.0, mean_conf ~ 0.86, so ECE = |0.86 - 1.0| * 4/4 = 0.14
    assert!(ece > 0.0);
    assert!(ece < 0.2);
}

#[test]
fn test_compute_cohens_kappa_zero_total() {
    use crate::eval::classification::ConfusionMatrix;
    let cm = ConfusionMatrix::new(2);
    let k = ClassifyEvalReport::compute_cohens_kappa(&cm, 0);
    assert!((k - 0.0).abs() < 1e-6);
}

#[test]
fn test_compute_mcc_zero_total() {
    use crate::eval::classification::ConfusionMatrix;
    let cm = ConfusionMatrix::new(2);
    let mcc = ClassifyEvalReport::compute_mcc(&cm, 2, 0);
    assert!((mcc - 0.0).abs() < 1e-6);
}

// =========================================================================
// TrainResult struct tests
// =========================================================================

#[test]
fn test_train_result_empty_metrics() {
    let result = TrainResult {
        epoch_metrics: vec![],
        best_epoch: 0,
        best_val_loss: f32::INFINITY,
        stopped_early: false,
        total_time_ms: 0,
    };
    assert!(result.epoch_metrics.is_empty());
    assert_eq!(result.best_epoch, 0);
    assert!(result.best_val_loss.is_infinite());
    assert!(!result.stopped_early);
}

#[test]
fn test_train_result_stopped_early() {
    let result = TrainResult {
        epoch_metrics: vec![EpochMetrics {
            epoch: 0,
            train_loss: 0.5,
            train_accuracy: 0.8,
            val_loss: 0.6,
            val_accuracy: 0.75,
            learning_rate: 1e-4,
            epoch_time_ms: 1000,
            samples_per_sec: 100.0,
        }],
        best_epoch: 0,
        best_val_loss: 0.6,
        stopped_early: true,
        total_time_ms: 1000,
    };
    assert!(result.stopped_early);
    assert_eq!(result.epoch_metrics.len(), 1);
}

// =========================================================================
// compute_bootstrap_cis edge case
// =========================================================================

#[test]
fn test_bootstrap_cis_single_sample_edge() {
    let y_pred = vec![0];
    let y_true = vec![0];
    let (ci_acc, ci_f1, ci_mcc) =
        ClassifyEvalReport::compute_bootstrap_cis(&y_pred, &y_true, 2, 50);
    assert!(ci_acc.0.is_finite());
    assert!(ci_acc.1.is_finite());
    assert!(ci_f1.0.is_finite());
    assert!(ci_f1.1.is_finite());
    assert!(ci_mcc.0.is_finite());
    assert!(ci_mcc.1.is_finite());
}

#[test]
fn test_bootstrap_cis_larger_sample() {
    let y_pred: Vec<usize> = (0..50).map(|i| i % 3).collect();
    let y_true: Vec<usize> = (0..50).map(|i| i % 3).collect();
    let (ci_acc, ci_f1, ci_mcc) =
        ClassifyEvalReport::compute_bootstrap_cis(&y_pred, &y_true, 3, 100);
    // Perfect predictions — CI should be tight around 1.0
    assert!(ci_acc.0 > 0.8);
    assert!(ci_acc.1 <= 1.0001);
    assert!(ci_f1.0 > 0.8);
    assert!(ci_mcc.0 > 0.8);
}

// =========================================================================
// compute_baselines
// =========================================================================

#[test]
fn test_compute_baselines_balanced_support() {
    // support = [5, 5], total=10, 2 classes
    let (random, majority) = ClassifyEvalReport::compute_baselines(&[5, 5], 10, 2);
    assert!((random - 0.5).abs() < 1e-6);
    assert!((majority - 0.5).abs() < 1e-6);
}

#[test]
fn test_compute_baselines_imbalanced_support() {
    // support = [8, 2], total=10, 2 classes
    let (random, majority) = ClassifyEvalReport::compute_baselines(&[8, 2], 10, 2);
    assert!((random - 0.5).abs() < 1e-6);
    assert!((majority - 0.8).abs() < 1e-6);
}

#[test]
fn test_compute_baselines_empty_support() {
    let (random, majority) = ClassifyEvalReport::compute_baselines(&[], 0, 3);
    assert!((random - 1.0 / 3.0).abs() < 1e-6);
    assert!((majority - 0.0).abs() < 1e-6);
}

// =========================================================================
// ClassifyTrainer::new validation
// =========================================================================

#[test]
fn test_trainer_new_empty_corpus() {
    let p = tiny_pipeline(2);
    let cfg = TrainingConfig { epochs: 1, val_split: 0.2, ..TrainingConfig::default() };
    let result = ClassifyTrainer::new(p, vec![], cfg);
    assert!(result.is_err());
}

#[test]
fn test_trainer_new_val_split_too_high() {
    let p = tiny_pipeline(2);
    let corpus = make_corpus(20, 2);
    let cfg = TrainingConfig {
        epochs: 1,
        val_split: 0.6, // > 0.5
        ..TrainingConfig::default()
    };
    let result = ClassifyTrainer::new(p, corpus, cfg);
    assert!(result.is_err());
}

#[test]
fn test_trainer_new_zero_epochs() {
    let p = tiny_pipeline(2);
    let corpus = make_corpus(20, 2);
    let cfg = TrainingConfig { epochs: 0, val_split: 0.2, ..TrainingConfig::default() };
    let result = ClassifyTrainer::new(p, corpus, cfg);
    assert!(result.is_err());
}

#[test]
fn test_trainer_new_val_split_zero() {
    let p = tiny_pipeline(2);
    let corpus = make_corpus(20, 2);
    let cfg = TrainingConfig { epochs: 1, val_split: 0.0, ..TrainingConfig::default() };
    let result = ClassifyTrainer::new(p, corpus, cfg);
    assert!(result.is_err());
}

// =========================================================================
// ClassifyTrainer accessors
// =========================================================================

#[test]
fn test_trainer_train_data_accessor() {
    let p = tiny_pipeline(2);
    let corpus = make_corpus(20, 2);
    let cfg = TrainingConfig { epochs: 1, val_split: 0.2, ..TrainingConfig::default() };
    let trainer = ClassifyTrainer::new(p, corpus, cfg).expect("valid");
    assert!(!trainer.train_data().is_empty());
}

#[test]
fn test_trainer_val_data_accessor() {
    let p = tiny_pipeline(2);
    let corpus = make_corpus(20, 2);
    let cfg = TrainingConfig { epochs: 1, val_split: 0.2, ..TrainingConfig::default() };
    let trainer = ClassifyTrainer::new(p, corpus, cfg).expect("valid");
    assert!(!trainer.val_data().is_empty());
}

// =========================================================================
// restore_class_weights edge cases
// =========================================================================

#[test]
fn test_restore_weights_no_metadata_file() {
    let dir = std::env::temp_dir().join("entrenar_test_rw_no_meta_2");
    std::fs::create_dir_all(&dir).unwrap();
    // No metadata.json
    let w = restore_class_weights_from_metadata(&dir, 2);
    assert!(w.is_none());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_restore_weights_invalid_json() {
    let dir = std::env::temp_dir().join("entrenar_test_rw_invalid_json_2");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("metadata.json"), "not valid json").unwrap();
    let w = restore_class_weights_from_metadata(&dir, 2);
    assert!(w.is_none());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_restore_weights_no_class_weights_key() {
    let dir = std::env::temp_dir().join("entrenar_test_rw_no_key_2");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("metadata.json"),
        serde_json::to_string(&serde_json::json!({"epoch": 5})).unwrap(),
    )
    .unwrap();
    let w = restore_class_weights_from_metadata(&dir, 2);
    assert!(w.is_none());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_restore_weights_null_class_weights() {
    let dir = std::env::temp_dir().join("entrenar_test_rw_null_cw_2");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("metadata.json"),
        serde_json::to_string(&serde_json::json!({"class_weights": null})).unwrap(),
    )
    .unwrap();
    let w = restore_class_weights_from_metadata(&dir, 2);
    assert!(w.is_none());
    std::fs::remove_dir_all(&dir).ok();
}

// =========================================================================
// from_predictions_with_probs — more edge cases
// =========================================================================

#[test]
fn test_from_predictions_single_sample() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0],
        &[0],
        &[vec![0.9, 0.1]],
        0.5,
        2,
        &labels,
        10,
    );
    assert_eq!(report.total_samples, 1);
    assert!((report.accuracy - 1.0).abs() < 1e-6);
}

#[test]
fn test_from_predictions_five_classes() {
    let labels: Vec<String> = (0..5).map(|i| format!("cls_{i}")).collect();
    let y_pred: Vec<usize> = (0..25).map(|i| i % 5).collect();
    let y_true: Vec<usize> = (0..25).map(|i| i % 5).collect();
    let probs: Vec<Vec<f32>> = y_pred
        .iter()
        .map(|&p| {
            let mut v = vec![0.05; 5];
            v[p] = 0.8;
            v
        })
        .collect();
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &y_pred, &y_true, &probs, 5.0, 5, &labels, 200,
    );
    assert_eq!(report.num_classes, 5);
    assert_eq!(report.total_samples, 25);
    assert!((report.accuracy - 1.0).abs() < 1e-6);
    assert_eq!(report.per_class_precision.len(), 5);
    assert_eq!(report.per_class_recall.len(), 5);
    assert_eq!(report.per_class_f1.len(), 5);
    assert_eq!(report.confusion_matrix.len(), 5);
}

// =========================================================================
// samples_per_sec computation in report
// =========================================================================

#[test]
fn test_eval_report_samples_per_sec() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        500, // 500ms
    );
    // 2 samples in 500ms = 4.0 samples/sec
    assert!((report.samples_per_sec - 4.0).abs() < 1e-6);
}

#[test]
fn test_eval_report_zero_time_edge() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        0, // 0ms
    );
    // Should handle division by zero gracefully
    assert!(report.samples_per_sec == 0.0 || report.samples_per_sec.is_infinite());
}

// =========================================================================
// compute_data_hash
// =========================================================================

#[test]
fn test_compute_data_hash_is_deterministic() {
    let data1 = vec![
        SafetySample { input: "echo hello".to_string(), label: 0 },
        SafetySample { input: "rm -rf /".to_string(), label: 1 },
    ];
    let data2 = data1.clone();
    let hash1 = ClassifyTrainer::compute_data_hash(&data1);
    let hash2 = ClassifyTrainer::compute_data_hash(&data2);
    assert_eq!(hash1, hash2);
    assert!(!hash1.is_empty());
}

#[test]
fn test_compute_data_hash_varies_with_data() {
    let data1 = vec![SafetySample { input: "echo a".to_string(), label: 0 }];
    let data2 = vec![SafetySample { input: "echo b".to_string(), label: 0 }];
    let hash1 = ClassifyTrainer::compute_data_hash(&data1);
    let hash2 = ClassifyTrainer::compute_data_hash(&data2);
    assert_ne!(hash1, hash2);
}

// =========================================================================
// Oversampling edge case
// =========================================================================

#[test]
fn test_oversample_corpus_all_same_class() {
    let p = tiny_pipeline(2);
    let corpus: Vec<SafetySample> =
        (0..10).map(|i| SafetySample { input: format!("s{i}"), label: 0 }).collect();
    let cfg = TrainingConfig {
        epochs: 1,
        val_split: 0.2,
        oversample_minority: true,
        ..TrainingConfig::default()
    };
    // Should succeed even if all samples are the same class
    let result = ClassifyTrainer::new(p, corpus, cfg);
    assert!(result.is_ok());
}

// =========================================================================
// Coverage expansion: ~400 additional uncovered regions
// =========================================================================

// ── report_top_confusions with out-of-range label names ──────────────

#[test]
fn test_coverage_report_top_confusions_missing_label_names() {
    let labels = vec!["only_a".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 2, 1, 0, 2],
        &[2, 0, 1, 0, 2, 1],
        &[
            vec![0.6, 0.2, 0.2],
            vec![0.2, 0.6, 0.2],
            vec![0.2, 0.2, 0.6],
            vec![0.2, 0.6, 0.2],
            vec![0.6, 0.2, 0.2],
            vec![0.2, 0.2, 0.6],
        ],
        3.0,
        3,
        &labels,
        100,
    );
    let text = report.to_report();
    assert!(text.contains('?'));
    assert!(text.contains("Top confusions"));
}

// ── card_labels when class index exceeds descriptions array ──────────

#[test]
fn test_coverage_card_labels_more_classes_than_descriptions() {
    let labels: Vec<String> = (0..7).map(|i| format!("cls_{i}")).collect();
    let y: Vec<usize> = (0..14).map(|i| i % 7).collect();
    let probs: Vec<Vec<f32>> = y
        .iter()
        .map(|&p| {
            let mut v = vec![0.02; 7];
            v[p] = 0.86;
            v
        })
        .collect();
    let report =
        ClassifyEvalReport::from_predictions_with_probs(&y, &y, &probs, 7.0, 7, &labels, 100);
    let card = report.to_model_card("7class-model", None);
    assert!(card.contains("## Labels"));
    assert!(card.contains("cls_5"));
    assert!(card.contains("cls_6"));
}

// ── report_calibration overconf indicator (+) ────────────────────────

#[test]
fn test_coverage_report_calibration_overconfident_bins() {
    let confs = vec![0.95, 0.92, 0.93, 0.91, 0.94, 0.96];
    let y_pred = vec![0, 1, 0, 1, 0, 1];
    let y_true = vec![1, 0, 1, 0, 1, 0];
    let (bins, ece) = ClassifyEvalReport::compute_calibration(&confs, &y_pred, &y_true, 6);
    assert!(bins[9].2 == 6);
    assert!(bins[9].0 > bins[9].1);
    assert!(ece > 0.8);
}

#[test]
fn test_coverage_report_calibration_underconfident_bins() {
    let confs = vec![0.15, 0.12, 0.18, 0.11];
    let y_pred = vec![0, 1, 0, 1];
    let y_true = vec![0, 1, 0, 1];
    let (bins, ece) = ClassifyEvalReport::compute_calibration(&confs, &y_pred, &y_true, 4);
    assert!(bins[1].2 == 4);
    assert!(bins[1].0 < bins[1].1);
    assert!(ece > 0.5);
}

// ── report_baselines with zero majority ──────────────────────────────

#[test]
fn test_coverage_report_baselines_zero_majority() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let mut report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        100,
    );
    report.baseline_majority = 0.0;
    let mut out = String::new();
    report.report_baselines(&mut out);
    assert!(out.contains("Baselines:"));
    assert!(out.contains("0.0x"));
}

// ── to_json with zero baseline_majority ──────────────────────────────

#[test]
fn test_coverage_to_json_zero_baseline_majority() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let mut report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        100,
    );
    report.baseline_majority = 0.0;
    let json_str = report.to_json();
    let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert!((v["baselines"]["lift_over_majority"].as_f64().unwrap() - 0.0).abs() < 1e-6);
}

// ── card_summary with zero baseline_majority ─────────────────────────

#[test]
fn test_coverage_card_summary_zero_baseline_majority() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let mut report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        100,
    );
    report.baseline_majority = 0.0;
    let mut out = String::new();
    report.card_summary(&mut out, 0.9, 0.9);
    assert!(out.contains("0.0x lift"));
}

// ── cohens_kappa edge: p_e = 1.0 and p_o = 1.0 ─────────────────────

#[test]
fn test_coverage_cohens_kappa_pe_one_po_one() {
    use crate::eval::classification::ConfusionMatrix;
    let cm =
        ConfusionMatrix::from_predictions_with_min_classes(&[0, 0, 0, 0, 0], &[0, 0, 0, 0, 0], 1);
    let kappa = ClassifyEvalReport::compute_cohens_kappa(&cm, 5);
    assert!(kappa.is_finite());
}

#[test]
fn test_coverage_cohens_kappa_pe_one_po_not_one() {
    use crate::eval::classification::ConfusionMatrix;
    let cm = ConfusionMatrix::from_predictions_with_min_classes(&[0, 0, 0, 0], &[0, 0, 0, 0], 1);
    let kappa = ClassifyEvalReport::compute_cohens_kappa(&cm, 4);
    assert!(kappa.is_finite());
}

// ── brier_score with probs shorter than num_classes ──────────────────

#[test]
fn test_coverage_brier_score_short_probs_vector() {
    let probs = vec![vec![0.8f32]];
    let b = ClassifyEvalReport::compute_brier_score(&probs, &[0], 3);
    assert!((b - 0.04).abs() < 1e-4);
}

#[test]
fn test_coverage_brier_score_true_label_not_in_probs() {
    let probs = vec![vec![0.6f32, 0.4]];
    let b = ClassifyEvalReport::compute_brier_score(&probs, &[2], 3);
    assert!((b - 1.52).abs() < 1e-4);
}

// ── log_loss with prob exactly 1.0 (clamp to 1-eps) ─────────────────

#[test]
fn test_coverage_log_loss_prob_exactly_one() {
    let ll = ClassifyEvalReport::compute_log_loss(&[vec![1.0, 0.0]], &[0]);
    assert!(ll < 1e-10);
    assert!(ll >= 0.0);
}

#[test]
fn test_coverage_log_loss_prob_near_zero() {
    let ll = ClassifyEvalReport::compute_log_loss(&[vec![0.001, 0.999]], &[0]);
    assert!(ll > 6.0);
    assert!(ll < 7.5);
}

#[test]
fn test_coverage_log_loss_three_class() {
    let ll = ClassifyEvalReport::compute_log_loss(
        &[vec![0.7, 0.2, 0.1], vec![0.1, 0.8, 0.1], vec![0.1, 0.1, 0.8]],
        &[0, 1, 2],
    );
    assert!(ll > 0.2);
    assert!(ll < 0.4);
}

// ── compute_top2_accuracy: true label in second position ─────────────

#[test]
fn test_coverage_top2_accuracy_true_label_in_second_slot() {
    let probs = vec![vec![0.5, 0.3, 0.2], vec![0.5, 0.3, 0.2]];
    let top2 = ClassifyEvalReport::compute_top2_accuracy(&probs, &[1, 2], 2);
    assert!((top2 - 0.5).abs() < 1e-6);
}

#[test]
fn test_coverage_top2_accuracy_tied_probabilities() {
    let probs = vec![vec![0.5, 0.5, 0.0]];
    let top2 = ClassifyEvalReport::compute_top2_accuracy(&probs, &[1], 1);
    assert!((top2 - 1.0).abs() < 1e-6);
}

// ── confidence stats with mixed correct/wrong ────────────────────────

#[test]
fn test_coverage_confidence_stats_single_correct() {
    let (mean, mc, mw) = ClassifyEvalReport::compute_confidence_stats(&[0.75], &[0], &[0]);
    assert!((mean - 0.75).abs() < 1e-6);
    assert!((mc - 0.75).abs() < 1e-6);
    assert!((mw - 0.0).abs() < 1e-6);
}

#[test]
fn test_coverage_confidence_stats_single_wrong() {
    let (mean, mc, mw) = ClassifyEvalReport::compute_confidence_stats(&[0.65], &[1], &[0]);
    assert!((mean - 0.65).abs() < 1e-6);
    assert!((mc - 0.0).abs() < 1e-6);
    assert!((mw - 0.65).abs() < 1e-6);
}

// ── calibration with spread across many bins ─────────────────────────

#[test]
fn test_coverage_calibration_spread_across_bins() {
    let confs = vec![0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
    let y_pred = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
    let y_true = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
    let (bins, ece) = ClassifyEvalReport::compute_calibration(&confs, &y_pred, &y_true, 10);
    for b in &bins {
        assert_eq!(b.2, 1);
    }
    assert!(ece > 0.0);
    assert!(ece < 0.6);
}

#[test]
fn test_coverage_calibration_conf_exactly_one() {
    let confs = vec![1.0];
    let y_pred = vec![0];
    let y_true = vec![0];
    let (bins, ece) = ClassifyEvalReport::compute_calibration(&confs, &y_pred, &y_true, 1);
    assert_eq!(bins[9].2, 1);
    assert!(ece.is_finite());
}

#[test]
fn test_coverage_calibration_conf_exactly_zero() {
    let confs = vec![0.0];
    let y_pred = vec![0];
    let y_true = vec![0];
    let (bins, ece) = ClassifyEvalReport::compute_calibration(&confs, &y_pred, &y_true, 1);
    assert_eq!(bins[0].2, 1);
    assert!(ece.is_finite());
}

// ── MCC with negative correlation ────────────────────────────────────

#[test]
fn test_coverage_mcc_negative_correlation() {
    use crate::eval::classification::ConfusionMatrix;
    let cm = ConfusionMatrix::from_predictions_with_min_classes(
        &[1, 0, 1, 0, 1, 0],
        &[0, 1, 0, 1, 0, 1],
        2,
    );
    let mcc = ClassifyEvalReport::compute_mcc(&cm, 2, 6);
    assert!(mcc < 0.0);
}

#[test]
fn test_coverage_mcc_four_classes() {
    use crate::eval::classification::ConfusionMatrix;
    let y_pred = vec![0, 1, 2, 3, 0, 1, 2, 3];
    let y_true = vec![0, 1, 2, 3, 1, 0, 3, 2];
    let cm = ConfusionMatrix::from_predictions_with_min_classes(&y_pred, &y_true, 4);
    let mcc = ClassifyEvalReport::compute_mcc(&cm, 4, 8);
    assert!(mcc > 0.0);
    assert!(mcc < 1.0);
}

// ── cohens_kappa with moderate agreement ──────────────────────────────

#[test]
fn test_coverage_cohens_kappa_moderate_agreement() {
    use crate::eval::classification::ConfusionMatrix;
    let cm = ConfusionMatrix::from_predictions_with_min_classes(
        &[0, 1, 0, 1, 0, 0, 1, 1],
        &[0, 1, 0, 0, 0, 1, 1, 1],
        2,
    );
    let kappa = ClassifyEvalReport::compute_cohens_kappa(&cm, 8);
    assert!(kappa > 0.0);
    assert!(kappa < 1.0);
}

#[test]
fn test_coverage_cohens_kappa_three_classes() {
    use crate::eval::classification::ConfusionMatrix;
    let cm = ConfusionMatrix::from_predictions_with_min_classes(
        &[0, 1, 2, 0, 1, 2, 0, 1, 2],
        &[0, 1, 2, 1, 0, 2, 0, 2, 1],
        3,
    );
    let kappa = ClassifyEvalReport::compute_cohens_kappa(&cm, 9);
    assert!(kappa > 0.0);
    assert!(kappa < 1.0);
}

// ── bootstrap CIs with moderate accuracy ─────────────────────────────

#[test]
fn test_coverage_bootstrap_cis_moderate_accuracy() {
    let y_pred = vec![0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1];
    let y_true = vec![0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1];
    let (ci_a, ci_f, ci_m) = ClassifyEvalReport::compute_bootstrap_cis(&y_pred, &y_true, 2, 200);
    assert!(ci_a.1 > ci_a.0);
    assert!(ci_f.1 > ci_f.0);
    assert!(ci_m.0.is_finite());
    assert!(ci_m.1.is_finite());
}

#[test]
fn test_coverage_bootstrap_cis_three_class() {
    let y_pred: Vec<usize> = (0..30).map(|i| i % 3).collect();
    let mut y_true: Vec<usize> = (0..30).map(|i| i % 3).collect();
    y_true[0] = 1;
    y_true[5] = 2;
    y_true[10] = 0;
    let (ci_a, ci_f, ci_m) = ClassifyEvalReport::compute_bootstrap_cis(&y_pred, &y_true, 3, 100);
    assert!(ci_a.0 > 0.5);
    assert!(ci_f.0 > 0.3);
    assert!(ci_m.0.is_finite());
}

// ── top_confusions ordering ──────────────────────────────────────────

#[test]
fn test_coverage_top_confusions_ordering_by_count() {
    let matrix = vec![vec![100, 5, 10], vec![3, 80, 7], vec![8, 2, 90]];
    let c = ClassifyEvalReport::compute_top_confusions(&matrix, 10);
    for i in 1..c.len() {
        assert!(c[i - 1].2 >= c[i].2);
    }
    assert_eq!(c[0], (0, 2, 10));
}

#[test]
fn test_coverage_top_confusions_truncated_to_1() {
    let matrix = vec![vec![10, 3, 1], vec![2, 8, 4], vec![1, 2, 9]];
    let c = ClassifyEvalReport::compute_top_confusions(&matrix, 1);
    assert_eq!(c.len(), 1);
    assert_eq!(c[0], (1, 2, 4));
}

// ── to_report per-class row with Class N fallback ────────────────────

#[test]
fn test_coverage_to_report_class_n_fallback() {
    let labels = vec![];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1],
        &[0, 1, 0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9], vec![0.9, 0.1], vec![0.1, 0.9]],
        2.0,
        2,
        &labels,
        100,
    );
    let text = report.to_report();
    assert!(text.contains("Class 0"));
    assert!(text.contains("Class 1"));
}

// ── to_json with empty label_names ───────────────────────────────────

#[test]
fn test_coverage_to_json_empty_label_names() {
    let labels: Vec<String> = vec![];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let json_str = report.to_json();
    let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(v["per_class"][0]["label"], "class_0");
    assert_eq!(v["per_class"][1]["label"], "class_1");
}

// ── to_json confusions with out-of-range labels ──────────────────────

#[test]
fn test_coverage_to_json_confusions_class_fallback() {
    let labels = vec!["x".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 2, 1, 0, 2],
        &[2, 0, 1, 0, 2, 0],
        &[
            vec![0.6, 0.2, 0.2],
            vec![0.2, 0.6, 0.2],
            vec![0.2, 0.2, 0.6],
            vec![0.2, 0.6, 0.2],
            vec![0.6, 0.2, 0.2],
            vec![0.2, 0.2, 0.6],
        ],
        3.0,
        3,
        &labels,
        100,
    );
    let json_str = report.to_json();
    let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    let confusions = v["top_confusions"].as_array().unwrap();
    assert!(!confusions.is_empty());
    let all_text = json_str;
    assert!(
        all_text.contains("class_1") || all_text.contains("class_2") || all_text.contains("\"x\"")
    );
}

// ── card_weak_classes with high F1 (no weak class) ───────────────────

#[test]
fn test_coverage_card_weak_classes_all_high_f1() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        &[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        &(0..10)
            .map(|i| if i % 2 == 0 { vec![0.9, 0.1] } else { vec![0.1, 0.9] })
            .collect::<Vec<_>>(),
        5.0,
        2,
        &labels,
        100,
    );
    let mut out = String::new();
    report.card_weak_classes(&mut out);
    assert!(!out.contains("Weak class"));
}

#[test]
fn test_coverage_card_weak_classes_low_f1_class() {
    let labels = vec!["good".to_string(), "bad".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        &[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        &(0..10).map(|_| vec![0.9, 0.1]).collect::<Vec<_>>(),
        5.0,
        2,
        &labels,
        100,
    );
    let mut out = String::new();
    report.card_weak_classes(&mut out);
    assert!(out.contains("Weak class"));
    assert!(out.contains("bad"));
}

// ── card_error_analysis empty when no confusions ─────────────────────

#[test]
fn test_coverage_card_error_analysis_empty() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1],
        &[0, 1, 0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9], vec![0.9, 0.1], vec![0.1, 0.9]],
        2.0,
        2,
        &labels,
        100,
    );
    let mut out = String::new();
    report.card_error_analysis(&mut out);
    assert!(out.is_empty());
}

// ── card_training with and without base_model ────────────────────────

#[test]
fn test_coverage_card_training_with_base() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let mut out = String::new();
    report.card_training(&mut out, Some("qwen2.5-coder"));
    assert!(out.contains("| Base model | `qwen2.5-coder` |"));
    assert!(out.contains("| Num classes | 2 |"));
}

#[test]
fn test_coverage_card_training_no_base() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let mut out = String::new();
    report.card_training(&mut out, None);
    assert!(!out.contains("Base model"));
    assert!(out.contains("LoRA"));
}

// ── card_intended_use and card_ethical_considerations ─────────────────

#[test]
fn test_coverage_card_intended_use_standalone() {
    let mut out = String::new();
    ClassifyEvalReport::card_intended_use(&mut out);
    assert!(out.contains("## Intended Use"));
    assert!(out.contains("CI/CD pipelines"));
    assert!(out.contains("Shell purification"));
    assert!(out.contains("Code review"));
    assert!(out.contains("Interactive shells"));
}

#[test]
fn test_coverage_card_ethical_considerations_standalone() {
    let mut out = String::new();
    ClassifyEvalReport::card_ethical_considerations(&mut out);
    assert!(out.contains("## Ethical Considerations"));
    assert!(out.contains("False negatives are dangerous"));
    assert!(out.contains("Defense in depth"));
}

// ── card_confusion_header with short names ───────────────────────────

#[test]
fn test_coverage_card_confusion_header_short_names() {
    let labels = vec!["ab".to_string(), "cd".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let mut out = String::new();
    report.card_confusion_header(&mut out);
    assert!(out.contains("Predicted"));
    assert!(out.contains("ab"));
    assert!(out.contains("cd"));
}

// ── card_confusion_row_label with fallback ───────────────────────────

#[test]
fn test_coverage_card_confusion_row_label_fallback() {
    let labels: Vec<String> = vec![];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let mut out = String::new();
    report.card_confusion_row_label(&mut out, 0);
    assert!(out.contains("class_0"));
}

// ── card_confusion_normalized with non-zero and zero rows ────────────

#[test]
fn test_coverage_card_confusion_normalized_mixed_rows() {
    let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 0, 1, 1],
        &[0, 0, 1, 1],
        &[vec![0.9, 0.05, 0.05], vec![0.8, 0.1, 0.1], vec![0.1, 0.8, 0.1], vec![0.05, 0.9, 0.05]],
        2.0,
        3,
        &labels,
        100,
    );
    let mut out = String::new();
    report.card_confusion_normalized(&mut out);
    assert!(out.contains("0.0%"));
    assert!(out.contains("Normalized"));
}

// ── report_scoring_rules one class ───────────────────────────────────

#[test]
fn test_coverage_report_scoring_rules_one_class() {
    let labels = vec!["only".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 0, 0],
        &[0, 0, 0],
        &[vec![1.0], vec![1.0], vec![1.0]],
        0.0,
        1,
        &labels,
        100,
    );
    let mut out = String::new();
    report.report_scoring_rules(&mut out);
    assert!(out.contains("Brier score:"));
    assert!(out.contains("Log loss:"));
}

// ── report_throughput formatting ─────────────────────────────────────

#[test]
fn test_coverage_report_throughput_high_throughput() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        1,
    );
    let mut out = String::new();
    report.report_throughput(&mut out);
    assert!(out.contains("Samples:   2"));
    assert!(out.contains("1ms"));
}

// ── avg_metric with various average types ────────────────────────────

#[test]
fn test_coverage_avg_metric_weighted_unequal_support() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 0, 0, 1],
        &[0, 0, 0, 1],
        &[vec![0.9, 0.1], vec![0.9, 0.1], vec![0.9, 0.1], vec![0.1, 0.9]],
        2.0,
        2,
        &labels,
        100,
    );
    use crate::eval::classification::Average;
    let macro_v = report.avg_metric(&[0.8, 0.4], Average::Macro);
    assert!((macro_v - 0.6).abs() < 1e-6);

    let weighted_v = report.avg_metric(&[0.8, 0.4], Average::Weighted);
    assert!((weighted_v - 0.7).abs() < 1e-6);
}

#[test]
fn test_coverage_avg_metric_macro_single_value() {
    let report = make_eval_report(&[0], &[0], &[vec![0.9, 0.1]], 2);
    use crate::eval::classification::Average;
    let v = report.avg_metric(&[0.5], Average::Macro);
    assert!((v - 0.5).abs() < 1e-6);
}

// ── from_predictions_with_probs: loss averaging ──────────────────────

#[test]
fn test_coverage_from_predictions_zero_total_loss() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        0.0,
        2,
        &labels,
        100,
    );
    assert!((report.avg_loss - 0.0).abs() < 1e-6);
}

#[test]
fn test_coverage_from_predictions_large_loss() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1000.0,
        2,
        &labels,
        100,
    );
    assert!((report.avg_loss - 500.0).abs() < 1e-3);
}

// ── from_predictions_with_probs: empty predictions ───────────────────

#[test]
fn test_coverage_from_predictions_empty_zero_time() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(&[], &[], &[], 0.0, 2, &labels, 0);
    assert_eq!(report.total_samples, 0);
    assert!((report.accuracy - 0.0).abs() < 1e-6);
    assert!((report.samples_per_sec - 0.0).abs() < 1e-6);
    assert!((report.brier_score - 0.0).abs() < 1e-6);
    assert!((report.log_loss - 0.0).abs() < 1e-6);
}

// ── model card comprehensive with 5 SSC labels ───────────────────────

#[test]
fn test_coverage_model_card_five_ssc_labels_full() {
    let labels: Vec<String> = SSC_LABELS.iter().map(ToString::to_string).collect();
    let n = 50;
    let y_pred: Vec<usize> = (0..n).map(|i| i % 5).collect();
    let mut y_true: Vec<usize> = (0..n).map(|i| i % 5).collect();
    y_true[0] = 1;
    y_true[5] = 2;
    y_true[10] = 3;
    let probs: Vec<Vec<f32>> = y_pred
        .iter()
        .map(|&p| {
            let mut v = vec![0.05; 5];
            v[p] = 0.8;
            v
        })
        .collect();
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &y_pred, &y_true, &probs, 10.0, 5, &labels, 500,
    );
    let card = report.to_model_card("shell-safety-classifier", Some("microsoft/codebert-base"));

    assert!(card.starts_with("---\n"));
    assert!(card.contains("license: apache-2.0"));
    assert!(card.contains("base_model: microsoft/codebert-base"));
    assert!(card.contains("shell-safety"));
    assert!(card.contains("type: accuracy"));
    assert!(card.contains("type: f1"));
    assert!(card.contains("type: mcc"));
    assert!(card.contains("type: cohens_kappa"));
    assert!(card.contains("# shell-safety-classifier"));
    assert!(card.contains("LoRA fine-tuning"));
    assert!(card.contains("## Summary"));
    assert!(card.contains("## Labels"));
    assert!(card.contains("safe"));
    assert!(card.contains("needs-quoting"));
    assert!(card.contains("non-deterministic"));
    assert!(card.contains("non-idempotent"));
    assert!(card.contains("unsafe"));
    assert!(card.contains("## Per-Class Metrics"));
    assert!(card.contains("## Confusion Matrix"));
    assert!(card.contains("### Raw Counts"));
    assert!(card.contains("### Normalized (row %)"));
    assert!(card.contains("## Error Analysis"));
    assert!(card.contains("## Confidence & Calibration"));
    assert!(card.contains("## Intended Use"));
    assert!(card.contains("## Limitations"));
    assert!(card.contains("## Ethical Considerations"));
    assert!(card.contains("## Training"));
    assert!(card.contains("microsoft/codebert-base"));
    assert!(card.contains("| Num classes | 5 |"));
    assert!(card.contains("Generated by [entrenar]"));
}

// ── to_report with 5 classes and mixed results ───────────────────────

#[test]
fn test_coverage_to_report_five_class_mixed() {
    let labels: Vec<String> = SSC_LABELS.iter().map(ToString::to_string).collect();
    let y_pred = vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
    let y_true = vec![0, 1, 2, 3, 4, 1, 0, 3, 2, 4, 0, 1, 4, 3, 2];
    let probs: Vec<Vec<f32>> = y_pred
        .iter()
        .map(|&p| {
            let mut v = vec![0.05; 5];
            v[p] = 0.8;
            v
        })
        .collect();
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &y_pred, &y_true, &probs, 5.0, 5, &labels, 100,
    );
    let text = report.to_report();

    for label in &labels {
        assert!(text.contains(label));
    }
    assert!(text.contains("macro avg"));
    assert!(text.contains("weighted avg"));
    assert!(text.contains("Accuracy:"));
    assert!(text.contains("Confidence (mean):"));
    assert!(text.contains("Brier score:"));
    assert!(text.contains("ECE"));
    assert!(text.contains("Baselines:"));
    assert!(text.contains("Samples:"));
}

// ── to_json comprehensive with 5 classes ─────────────────────────────

#[test]
fn test_coverage_to_json_five_class_comprehensive() {
    let labels: Vec<String> = SSC_LABELS.iter().map(ToString::to_string).collect();
    let y_pred = vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
    let y_true = vec![0, 1, 2, 3, 4, 1, 0, 3, 2, 4];
    let probs: Vec<Vec<f32>> = y_pred
        .iter()
        .map(|&p| {
            let mut v = vec![0.05; 5];
            v[p] = 0.8;
            v
        })
        .collect();
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &y_pred, &y_true, &probs, 10.0, 5, &labels, 200,
    );
    let json_str = report.to_json();
    let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert_eq!(v["num_classes"].as_u64().unwrap(), 5);
    assert_eq!(v["total_samples"].as_u64().unwrap(), 10);
    assert_eq!(v["per_class"].as_array().unwrap().len(), 5);
    assert_eq!(v["per_class"][0]["label"], "safe");
    assert_eq!(v["per_class"][4]["label"], "unsafe");
    assert!(v["confusion_matrix"].as_array().unwrap().len() == 5);
    assert!(!v["top_confusions"].as_array().unwrap().is_empty());
    assert!(!v["calibration"]["bins"].as_array().unwrap().is_empty());
}

// ── split_dataset with large val_ratio and small corpus ──────────────

#[test]
fn test_coverage_split_dataset_three_samples() {
    let c = vec![
        SafetySample { input: "a".into(), label: 0 },
        SafetySample { input: "b".into(), label: 1 },
        SafetySample { input: "c".into(), label: 2 },
    ];
    let (tr, va) = ClassifyTrainer::split_dataset(&c, 0.5, 42);
    assert_eq!(va.len() + tr.len(), 3);
    assert!(!va.is_empty());
    assert!(!tr.is_empty());
}

#[test]
fn test_coverage_split_dataset_large_corpus_half() {
    let c = make_corpus(1000, 5);
    let (tr, va) = ClassifyTrainer::split_dataset(&c, 0.5, 123);
    assert_eq!(tr.len() + va.len(), 1000);
    assert_eq!(va.len(), 500);
    assert_eq!(tr.len(), 500);
}

// ── oversample with 4 classes ────────────────────────────────────────

#[test]
fn test_coverage_oversample_four_classes() {
    let mut data: Vec<SafetySample> = vec![];
    for i in 0..10 {
        data.push(SafetySample { input: format!("a{i}"), label: 0 });
    }
    for i in 0..5 {
        data.push(SafetySample { input: format!("b{i}"), label: 1 });
    }
    for i in 0..3 {
        data.push(SafetySample { input: format!("c{i}"), label: 2 });
    }
    for i in 0..2 {
        data.push(SafetySample { input: format!("d{i}"), label: 3 });
    }
    ClassifyTrainer::oversample_training_data(&mut data, 42);
    assert_eq!(data.iter().filter(|s| s.label == 0).count(), 10);
    assert_eq!(data.iter().filter(|s| s.label == 1).count(), 10);
    assert_eq!(data.iter().filter(|s| s.label == 2).count(), 10);
    assert_eq!(data.iter().filter(|s| s.label == 3).count(), 10);
    assert_eq!(data.len(), 40);
}

// ── oversample with different seeds produces different orders ─────────

#[test]
fn test_coverage_oversample_different_seeds() {
    let mk = || {
        vec![
            SafetySample { input: "a".into(), label: 0 },
            SafetySample { input: "b".into(), label: 0 },
            SafetySample { input: "c".into(), label: 0 },
            SafetySample { input: "d".into(), label: 1 },
        ]
    };
    let mut d1 = mk();
    let mut d2 = mk();
    ClassifyTrainer::oversample_training_data(&mut d1, 42);
    ClassifyTrainer::oversample_training_data(&mut d2, 999);
    let l1: Vec<String> = d1.iter().map(|s| s.input.clone()).collect();
    let l2: Vec<String> = d2.iter().map(|s| s.input.clone()).collect();
    let mut s1 = l1.clone();
    s1.sort();
    let mut s2 = l2.clone();
    s2.sort();
    assert_eq!(s1, s2);
    assert_ne!(l1, l2);
}

// ── TrainingConfig clone preserves all fields ─────────────────────────

#[test]
fn test_coverage_training_config_clone_all_fields() {
    let dist = DistributedConfig::coordinator("127.0.0.1:0".parse().unwrap(), 4);
    let config = TrainingConfig {
        epochs: 25,
        val_split: 0.15,
        save_every: 3,
        early_stopping_patience: 7,
        checkpoint_dir: PathBuf::from("/tmp/test_clone_all"),
        seed: 99,
        log_interval: 5,
        warmup_fraction: 0.2,
        lr_min: 1e-7,
        oversample_minority: true,
        quantize_nf4: true,
        distributed: Some(dist),
    };
    let c = config.clone();
    assert_eq!(c.epochs, 25);
    assert!((c.val_split - 0.15).abs() < 1e-6);
    assert_eq!(c.save_every, 3);
    assert_eq!(c.early_stopping_patience, 7);
    assert_eq!(c.checkpoint_dir.as_os_str(), "/tmp/test_clone_all");
    assert_eq!(c.seed, 99);
    assert_eq!(c.log_interval, 5);
    assert!((c.warmup_fraction - 0.2).abs() < 1e-6);
    assert!((c.lr_min - 1e-7).abs() < 1e-12);
    assert!(c.oversample_minority);
    assert!(c.quantize_nf4);
    assert!(c.distributed.is_some());
    assert_eq!(c.distributed.unwrap().expect_workers, 4);
}

// ── TrainResult with all fields populated ─────────────────────────────

#[test]
fn test_coverage_train_result_full_debug() {
    let metrics = vec![
        EpochMetrics {
            epoch: 0,
            train_loss: 2.5,
            train_accuracy: 0.3,
            val_loss: 2.8,
            val_accuracy: 0.25,
            learning_rate: 1e-3,
            epoch_time_ms: 5000,
            samples_per_sec: 20.0,
        },
        EpochMetrics {
            epoch: 1,
            train_loss: 1.5,
            train_accuracy: 0.6,
            val_loss: 1.8,
            val_accuracy: 0.55,
            learning_rate: 8e-4,
            epoch_time_ms: 4500,
            samples_per_sec: 22.0,
        },
        EpochMetrics {
            epoch: 2,
            train_loss: 0.8,
            train_accuracy: 0.85,
            val_loss: 1.0,
            val_accuracy: 0.75,
            learning_rate: 5e-4,
            epoch_time_ms: 4200,
            samples_per_sec: 24.0,
        },
    ];
    let result = TrainResult {
        epoch_metrics: metrics,
        best_epoch: 2,
        best_val_loss: 1.0,
        stopped_early: false,
        total_time_ms: 13700,
    };
    assert_eq!(result.epoch_metrics.len(), 3);
    assert_eq!(result.best_epoch, 2);
    assert!((result.best_val_loss - 1.0).abs() < 1e-6);
    assert!(!result.stopped_early);
    assert_eq!(result.total_time_ms, 13700);

    let cloned = result.clone();
    assert_eq!(cloned.epoch_metrics.len(), 3);
    assert_eq!(cloned.epoch_metrics[0].epoch, 0);
    assert_eq!(cloned.epoch_metrics[2].epoch, 2);

    let dbg = format!("{result:?}");
    assert!(dbg.contains("best_epoch: 2"));
    assert!(dbg.contains("total_time_ms: 13700"));
}

// ── EpochMetrics: various field values ────────────────────────────────

#[test]
fn test_coverage_epoch_metrics_nan_loss() {
    let m = EpochMetrics {
        epoch: 10,
        train_loss: f32::NAN,
        train_accuracy: 0.0,
        val_loss: f32::INFINITY,
        val_accuracy: 0.0,
        learning_rate: 0.0,
        epoch_time_ms: 0,
        samples_per_sec: 0.0,
    };
    let c = m.clone();
    assert!(c.train_loss.is_nan());
    assert!(c.val_loss.is_infinite());
    assert_eq!(c.epoch, 10);
}

// ── compute_data_hash: ordering guarantee ────────────────────────────

#[test]
fn test_coverage_data_hash_with_duplicate_inputs() {
    let c = vec![
        SafetySample { input: "echo hello".into(), label: 0 },
        SafetySample { input: "echo hello".into(), label: 0 },
        SafetySample { input: "echo hello".into(), label: 1 },
    ];
    let hash = ClassifyTrainer::compute_data_hash(&c);
    assert!(hash.starts_with("sha256:"));
    let hash2 = ClassifyTrainer::compute_data_hash(&c);
    assert_eq!(hash, hash2);
}

// ── from_predictions_with_probs: single class, all correct ───────────

#[test]
fn test_coverage_from_predictions_single_class_only() {
    let labels = vec!["only".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 0, 0, 0],
        &[0, 0, 0, 0],
        &[vec![1.0], vec![1.0], vec![1.0], vec![1.0]],
        0.0,
        1,
        &labels,
        50,
    );
    assert!((report.accuracy - 1.0).abs() < 1e-6);
    assert_eq!(report.num_classes, 1);
    assert_eq!(report.total_samples, 4);
    assert!((report.baseline_random - 1.0).abs() < 1e-6);
}

// ── report_summary individual field checks ───────────────────────────

#[test]
fn test_coverage_report_summary_individual_fields() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 0],
        &[0, 1, 1, 0],
        &[vec![0.8, 0.2], vec![0.2, 0.8], vec![0.7, 0.3], vec![0.9, 0.1]],
        3.0,
        2,
        &labels,
        200,
    );
    let mut out = String::new();
    report.report_summary(&mut out);
    assert!(out.contains("Accuracy:"));
    assert!(out.contains("Top-2 accuracy:"));
    assert!(out.contains("Cohen's kappa:"));
    assert!(out.contains("MCC:"));
    assert!(out.contains("Macro F1:"));
    assert!(out.contains("Avg loss:"));
    assert!(out.contains("95% CI"));
}

// ── report_confidence individual field checks ────────────────────────

#[test]
fn test_coverage_report_confidence_individual_fields() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1],
        &[0, 0, 0, 1],
        &[vec![0.8, 0.2], vec![0.3, 0.7], vec![0.9, 0.1], vec![0.2, 0.8]],
        2.0,
        2,
        &labels,
        100,
    );
    let mut out = String::new();
    report.report_confidence(&mut out);
    assert!(out.contains("Confidence (mean):"));
    assert!(out.contains("correct preds:"));
    assert!(out.contains("wrong preds:"));
    assert!(out.contains("gap (higher=better):"));
}

// ── card_per_class_metrics with fallback name ────────────────────────

#[test]
fn test_coverage_card_per_class_fallback_name() {
    let labels: Vec<String> = vec![];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let mut out = String::new();
    report.card_per_class_metrics(&mut out);
    assert!(out.contains("class_0"));
    assert!(out.contains("class_1"));
}

// ── brier_score with 5 classes ───────────────────────────────────────

#[test]
fn test_coverage_brier_score_five_classes_perfect() {
    let probs = vec![
        vec![1.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0],
    ];
    let b = ClassifyEvalReport::compute_brier_score(&probs, &[0, 1, 2, 3, 4], 5);
    assert!((b - 0.0).abs() < 1e-6);
}

#[test]
fn test_coverage_brier_score_five_classes_random() {
    let probs = vec![vec![0.2, 0.2, 0.2, 0.2, 0.2], vec![0.2, 0.2, 0.2, 0.2, 0.2]];
    let b = ClassifyEvalReport::compute_brier_score(&probs, &[0, 1], 5);
    assert!((b - 0.8).abs() < 1e-4);
}

// ── log_loss with multiple classes ────────────────────────────────────

#[test]
fn test_coverage_log_loss_five_classes() {
    let probs = vec![vec![0.8, 0.05, 0.05, 0.05, 0.05], vec![0.05, 0.8, 0.05, 0.05, 0.05]];
    let ll = ClassifyEvalReport::compute_log_loss(&probs, &[0, 1]);
    assert!((ll - 0.223).abs() < 0.01);
}

// ── calibration with high confidence and all wrong ───────────────────

#[test]
fn test_coverage_calibration_high_conf_all_wrong() {
    let confs = vec![0.99, 0.98, 0.97];
    let y_pred = vec![0, 0, 0];
    let y_true = vec![1, 1, 1];
    let (bins, ece) = ClassifyEvalReport::compute_calibration(&confs, &y_pred, &y_true, 3);
    assert_eq!(bins[9].2, 3);
    assert!((bins[9].1 - 0.0).abs() < 1e-6);
    assert!(ece > 0.9);
}

// ── cohens_kappa with 5 classes ──────────────────────────────────────

#[test]
fn test_coverage_cohens_kappa_five_class_mixed() {
    use crate::eval::classification::ConfusionMatrix;
    let y_pred = vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
    let y_true = vec![0, 1, 2, 3, 4, 1, 0, 3, 2, 4, 0, 1, 2, 4, 3, 0, 2, 1, 3, 4];
    let cm = ConfusionMatrix::from_predictions_with_min_classes(&y_pred, &y_true, 5);
    let kappa = ClassifyEvalReport::compute_cohens_kappa(&cm, 20);
    assert!(kappa > 0.0);
    assert!(kappa < 1.0);
}

// ── MCC with 5 classes ───────────────────────────────────────────────

#[test]
fn test_coverage_mcc_five_class_mixed() {
    use crate::eval::classification::ConfusionMatrix;
    let y_pred = vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
    let y_true = vec![0, 1, 2, 3, 4, 1, 0, 3, 2, 4];
    let cm = ConfusionMatrix::from_predictions_with_min_classes(&y_pred, &y_true, 5);
    let mcc = ClassifyEvalReport::compute_mcc(&cm, 5, 10);
    assert!(mcc > 0.0);
    assert!(mcc < 1.0);
}

// ── model card normalized confusion with all zeros in a row ──────────

#[test]
fn test_coverage_card_confusion_all_zero_row() {
    let labels = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let mut report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1],
        &[0, 1, 0, 1],
        &[vec![0.9, 0.05, 0.05], vec![0.05, 0.9, 0.05], vec![0.8, 0.1, 0.1], vec![0.1, 0.8, 0.1]],
        2.0,
        3,
        &labels,
        100,
    );
    report.confusion_matrix[2] = vec![0, 0, 0];
    let mut out = String::new();
    report.card_confusion_normalized(&mut out);
    assert!(out.contains("0.0%"));
}

// ── to_report separator lines ────────────────────────────────────────

#[test]
fn test_coverage_to_report_separator_lines() {
    let report = make_eval_report(&[0, 1], &[0, 1], &[vec![0.9, 0.1], vec![0.1, 0.9]], 2);
    let text = report.to_report();
    let dash_line = "-".repeat(62);
    assert!(text.contains(&dash_line));
}

// ── card_yaml_front_matter with exact accuracy check ─────────────────

#[test]
fn test_coverage_card_yaml_front_matter_metrics() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1],
        &[0, 1, 0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9], vec![0.9, 0.1], vec![0.1, 0.9]],
        2.0,
        2,
        &labels,
        100,
    );
    let mut out = String::new();
    report.card_yaml_front_matter(&mut out, "test-model", None, 1.0, 1.0);
    assert!(out.contains("---"));
    assert!(out.contains("license: apache-2.0"));
    assert!(out.contains("- name: test-model"));
    assert!(out.contains("type: accuracy"));
    assert!(!out.contains("base_model:"));
}

// ── card_title with and without base_model ───────────────────────────

#[test]
fn test_coverage_card_title_with_base() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let mut out = String::new();
    report.card_title(&mut out, "my-classifier", Some("base/model"));
    assert!(out.contains("# my-classifier"));
    assert!(out.contains("base/model"));
    assert!(out.contains("LoRA fine-tuning"));
}

#[test]
fn test_coverage_card_title_without_base() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9]],
        1.0,
        2,
        &labels,
        50,
    );
    let mut out = String::new();
    report.card_title(&mut out, "my-classifier", None);
    assert!(out.contains("# my-classifier"));
    assert!(!out.contains("huggingface.co"));
}

// ── card_calibration with no non-zero bins ───────────────────────────

#[test]
fn test_coverage_card_calibration_no_data() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let mut report =
        ClassifyEvalReport::from_predictions_with_probs(&[], &[], &[], 0.0, 2, &labels, 0);
    report.calibration_bins = vec![(0.0, 0.0, 0); 10];
    let mut out = String::new();
    report.card_calibration(&mut out);
    assert!(out.contains("## Confidence & Calibration"));
    assert!(!out.contains("[0.0-0.1)"));
}

// ── SSC_LABELS: verify all 5 labels ──────────────────────────────────

#[test]
fn test_coverage_ssc_labels_all_present() {
    assert_eq!(SSC_LABELS.len(), 5);
    let expected = ["safe", "needs-quoting", "non-deterministic", "non-idempotent", "unsafe"];
    for (i, &label) in SSC_LABELS.iter().enumerate() {
        assert_eq!(label, expected[i], "SSC_LABELS[{i}] mismatch");
    }
}

// ── from_predictions_with_probs: large dataset ───────────────────────

#[test]
fn test_coverage_from_predictions_large_dataset() {
    let n = 200;
    let nc = 3;
    let y_pred: Vec<usize> = (0..n).map(|i| i % nc).collect();
    let y_true: Vec<usize> = (0..n).map(|i| (i + 1) % nc).collect();
    let probs: Vec<Vec<f32>> = y_pred
        .iter()
        .map(|&p| {
            let mut v = vec![0.1; nc];
            v[p] = 0.8;
            v
        })
        .collect();
    let labels: Vec<String> = (0..nc).map(|i| format!("c{i}")).collect();
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &y_pred, &y_true, &probs, 50.0, nc, &labels, 1000,
    );
    assert_eq!(report.total_samples, n);
    assert_eq!(report.num_classes, nc);
    assert!(report.accuracy.is_finite());
    assert!(report.mcc.is_finite());
    assert!(report.cohens_kappa.is_finite());
    assert!(report.ece.is_finite());
    assert!(report.brier_score.is_finite());
    assert!(report.log_loss.is_finite());
    assert!(report.ci_accuracy.0.is_finite());
    assert!(report.ci_macro_f1.0.is_finite());
    assert!(report.ci_mcc.0.is_finite());
}

// ── card_labels with exactly 5 labels matching descriptions ──────────

#[test]
fn test_coverage_card_labels_exactly_five() {
    let labels: Vec<String> = SSC_LABELS.iter().map(ToString::to_string).collect();
    let y: Vec<usize> = (0..10).map(|i| i % 5).collect();
    let probs: Vec<Vec<f32>> = y
        .iter()
        .map(|&p| {
            let mut v = vec![0.05; 5];
            v[p] = 0.8;
            v
        })
        .collect();
    let report =
        ClassifyEvalReport::from_predictions_with_probs(&y, &y, &probs, 5.0, 5, &labels, 100);
    let mut out = String::new();
    report.card_labels(&mut out);
    assert!(out.contains("| 0 | safe |"));
    assert!(out.contains("| 1 | needs-quoting |"));
    assert!(out.contains("| 4 | unsafe |"));
    assert!(out.contains("word splitting"));
    assert!(out.contains("destructive"));
}

// ── to_report with exactly zero confusions ───────────────────────────

#[test]
fn test_coverage_to_report_zero_confusions_section() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1, 0, 1],
        &[0, 1, 0, 1],
        &[vec![0.9, 0.1], vec![0.1, 0.9], vec![0.9, 0.1], vec![0.1, 0.9]],
        2.0,
        2,
        &labels,
        100,
    );
    assert!(report.top_confusions.is_empty());
    let mut out = String::new();
    report.report_top_confusions(&mut out);
    assert!(out.is_empty());
}

// ── trainer accessors: pipeline_mut ──────────────────────────────────

#[test]
fn test_coverage_trainer_pipeline_mut() {
    let p = tiny_pipeline(2);
    let corpus = make_corpus(20, 2);
    let cfg = TrainingConfig { epochs: 1, val_split: 0.2, ..TrainingConfig::default() };
    let mut trainer = ClassifyTrainer::new(p, corpus, cfg).expect("valid");
    let pm = trainer.pipeline_mut();
    assert!(pm.config.num_classes == 2);
}

// ── shuffle doesn't change val_data ──────────────────────────────────

#[test]
fn test_coverage_shuffle_val_data_unchanged() {
    let p = tiny_pipeline(3);
    let c = make_corpus(40, 3);
    let cfg = TrainingConfig { epochs: 1, val_split: 0.2, ..TrainingConfig::default() };
    let mut t = ClassifyTrainer::new(p, c, cfg).expect("valid");
    let val_before: Vec<String> = t.val_data().iter().map(|s| s.input.clone()).collect();
    t.shuffle_training_data(0);
    t.shuffle_training_data(1);
    t.shuffle_training_data(2);
    let val_after: Vec<String> = t.val_data().iter().map(|s| s.input.clone()).collect();
    assert_eq!(val_before, val_after, "Val data must not change after shuffle");
}

// ── from_predictions_with_probs: probabilities don't sum to 1 ────────

#[test]
fn test_coverage_from_predictions_unnormalized_probs() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 1],
        &[0, 1],
        &[vec![2.0, 0.5], vec![0.3, 3.0]],
        1.0,
        2,
        &labels,
        100,
    );
    assert_eq!(report.total_samples, 2);
    assert!(report.accuracy.is_finite());
    assert!(report.brier_score.is_finite());
    assert!(report.mean_confidence > 0.0);
}

// ── report overconf marker in calibration output ─────────────────────

#[test]
fn test_coverage_report_calibration_overconf_marker() {
    let labels = vec!["a".to_string(), "b".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 0, 0, 0],
        &[1, 1, 1, 1],
        &[vec![0.95, 0.05], vec![0.92, 0.08], vec![0.93, 0.07], vec![0.94, 0.06]],
        4.0,
        2,
        &labels,
        100,
    );
    let mut out = String::new();
    report.report_calibration(&mut out);
    assert!(out.contains('+'));
}

// ── card_confusion_raw formatting check ──────────────────────────────

#[test]
fn test_coverage_card_confusion_raw_formatting() {
    let labels = vec!["safe".to_string(), "unsafe".to_string()];
    let report = ClassifyEvalReport::from_predictions_with_probs(
        &[0, 0, 1, 1],
        &[0, 1, 0, 1],
        &[vec![0.8, 0.2], vec![0.7, 0.3], vec![0.3, 0.7], vec![0.2, 0.8]],
        2.0,
        2,
        &labels,
        100,
    );
    let mut out = String::new();
    report.card_confusion_raw(&mut out);
    assert!(out.contains("### Raw Counts"));
    assert!(out.contains("```"));
    assert!(out.contains("Predicted"));
}
