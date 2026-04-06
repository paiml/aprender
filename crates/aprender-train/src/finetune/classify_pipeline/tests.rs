use super::*;

fn tiny_config() -> TransformerConfig {
    TransformerConfig::tiny()
}

#[test]
fn test_classify_pipeline_creation() {
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 5,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let pipeline = ClassifyPipeline::new(&model_config, classify_config);
    assert_eq!(pipeline.classifier.num_classes(), 5);
    assert!(!pipeline.lora_layers.is_empty(), "Should have LoRA layers");
}

#[test]
fn test_classify_pipeline_train_step() {
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 5,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
    let loss = pipeline.train_step(&[1, 2, 3], 0);
    assert!(loss.is_finite(), "F-CLASS-005: loss must be finite");
    assert!(loss > 0.0, "Cross-entropy loss must be positive");
}

#[test]
fn test_classify_pipeline_convergence() {
    // SSC-017: Training must reduce loss across epochs
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        learning_rate: 1e-2,
        ..ClassifyConfig::default()
    };
    let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);

    // Train on 3 samples for 20 epochs
    let samples = [(vec![1u32, 2, 3], 0usize), (vec![4, 5, 6], 1), (vec![7, 8, 9], 2)];

    let mut first_epoch_loss = 0.0f32;
    let mut last_epoch_loss = 0.0f32;

    for epoch in 0..20 {
        let mut epoch_loss = 0.0f32;
        for (tokens, label) in &samples {
            epoch_loss += pipeline.train_step(tokens, *label);
        }
        epoch_loss /= samples.len() as f32;

        if epoch == 0 {
            first_epoch_loss = epoch_loss;
        }
        last_epoch_loss = epoch_loss;
    }

    assert!(
            last_epoch_loss < first_epoch_loss,
            "SSC-017: Loss must decrease. First epoch: {first_epoch_loss:.4}, last: {last_epoch_loss:.4}"
        );
}

#[test]
fn test_classify_pipeline_trainable_params() {
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 5,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
    let params = pipeline.trainable_parameters_mut();
    // LoRA A + B per adapter + classifier weight + bias
    assert!(params.len() >= 3, "Should have at least classifier + 1 LoRA adapter params");
}

#[test]
fn test_classify_pipeline_summary() {
    let model_config = tiny_config();
    let classify_config = ClassifyConfig::default();
    let pipeline = ClassifyPipeline::new(&model_config, classify_config);
    let summary = pipeline.summary();
    assert!(summary.contains("ClassifyPipeline"));
    assert!(summary.contains("LoRA"));
    assert!(summary.contains("Classifier"));
}

#[test]
fn test_multi_label_train_step() {
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 5,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);

    // Multi-hot: classes 1 and 2 active (needs-quoting AND non-deterministic)
    let targets = vec![0.0, 1.0, 1.0, 0.0, 0.0];
    let loss = pipeline.multi_label_train_step(&[1, 2, 3], &targets);
    assert!(loss.is_finite(), "F-CLASS-005: loss must be finite");
    assert!(loss > 0.0, "BCE loss must be positive");
}

#[test]
fn test_multi_label_convergence() {
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        learning_rate: 1e-2,
        ..ClassifyConfig::default()
    };
    let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);

    // Train on multi-label samples
    let samples: [(Vec<u32>, Vec<f32>); 3] = [
        (vec![1, 2, 3], vec![1.0, 1.0, 0.0]), // classes 0+1
        (vec![4, 5, 6], vec![0.0, 1.0, 1.0]), // classes 1+2
        (vec![7, 8, 9], vec![1.0, 0.0, 1.0]), // classes 0+2
    ];

    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;

    for epoch in 0..20 {
        let mut epoch_loss = 0.0f32;
        for (tokens, targets) in &samples {
            epoch_loss += pipeline.multi_label_train_step(tokens, targets);
        }
        epoch_loss /= samples.len() as f32;

        if epoch == 0 {
            first_loss = epoch_loss;
        }
        last_loss = epoch_loss;
    }

    assert!(
        last_loss < first_loss,
        "SSC-021: Multi-label loss must decrease. First: {first_loss:.4}, last: {last_loss:.4}"
    );
}

#[test]
#[should_panic(expected = "F-CLASS-001")]
fn test_multi_label_wrong_target_length() {
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 5,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
    // Wrong number of targets (3 instead of 5)
    pipeline.multi_label_train_step(&[1, 2, 3], &[1.0, 0.0, 1.0]);
}

#[test]
fn test_classify_pipeline_merge() {
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 5,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);

    // Should not panic
    pipeline.merge_adapters();

    // All LoRA layers should be merged
    for lora in &pipeline.lora_layers {
        assert!(lora.is_merged(), "All adapters should be merged");
    }
}

// =========================================================================
// SSC-025: Mini-batch training with gradient accumulation
// =========================================================================

fn make_samples() -> Vec<SafetySample> {
    vec![
        SafetySample { input: "echo hello".into(), label: 0 },
        SafetySample { input: "rm -rf /".into(), label: 1 },
        SafetySample { input: "ls -la".into(), label: 2 },
    ]
}

#[test]
fn test_ssc025_batch_result_accuracy() {
    let r = BatchResult { avg_loss: 1.0, correct: 3, total: 4, grad_norm: 0.0 };
    assert!((r.accuracy() - 0.75).abs() < 1e-6);
}

#[test]
fn test_ssc025_batch_result_accuracy_empty() {
    let r = BatchResult { avg_loss: 0.0, correct: 0, total: 0, grad_norm: 0.0 };
    assert!((r.accuracy() - 0.0).abs() < 1e-6);
}

#[test]
fn test_ssc025_batch_result_accuracy_perfect() {
    let r = BatchResult { avg_loss: 0.1, correct: 10, total: 10, grad_norm: 0.0 };
    assert!((r.accuracy() - 1.0).abs() < 1e-6);
}

#[test]
fn test_ssc025_config_defaults() {
    let config = ClassifyConfig::default();
    assert_eq!(config.batch_size, 32);
    assert_eq!(config.accumulation_steps, 1);
    assert_eq!(config.gradient_clip_norm, Some(1.0));
}

#[test]
fn test_ssc025_config_custom_batch() {
    let config = ClassifyConfig {
        batch_size: 8,
        accumulation_steps: 4,
        gradient_clip_norm: Some(0.5),
        ..ClassifyConfig::default()
    };
    assert_eq!(config.batch_size, 8);
    assert_eq!(config.accumulation_steps, 4);
    assert_eq!(config.gradient_clip_norm, Some(0.5));
}

#[test]
fn test_ssc025_train_batch_finite_loss() {
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        batch_size: 3,
        ..ClassifyConfig::default()
    };
    let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
    let samples = make_samples();

    let result = pipeline.train_batch(&samples);
    assert!(
        result.avg_loss.is_finite(),
        "SSC-025: batch loss must be finite, got {}",
        result.avg_loss
    );
    assert!(result.avg_loss > 0.0, "Cross-entropy loss must be positive");
    assert_eq!(result.total, 3);
}

#[test]
fn test_ssc025_train_batch_empty() {
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);

    let result = pipeline.train_batch(&[]);
    assert_eq!(result.total, 0);
    assert_eq!(result.correct, 0);
    assert!((result.avg_loss - 0.0).abs() < 1e-6);
}

#[test]
fn test_ssc025_train_batch_convergence() {
    // SSC-025: Loss must decrease over multiple batches
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        learning_rate: 1e-2,
        gradient_clip_norm: None, // disable clipping for convergence test
        ..ClassifyConfig::default()
    };
    let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
    let samples = make_samples();

    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;

    for epoch in 0..20 {
        let result = pipeline.train_batch(&samples);
        if epoch == 0 {
            first_loss = result.avg_loss;
        }
        last_loss = result.avg_loss;
    }

    assert!(
        last_loss < first_loss,
        "SSC-025: Batch training must reduce loss. First: {first_loss:.4}, last: {last_loss:.4}"
    );
}

#[test]
fn test_ssc025_gradient_clipping_bounds_norm() {
    let model_config = tiny_config();
    let max_norm = 0.5;
    let classify_config = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        learning_rate: 1e-2,
        gradient_clip_norm: Some(max_norm),
        ..ClassifyConfig::default()
    };
    let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
    let samples = make_samples();

    // Run one batch — the internal clip should have bounded the norm
    // We verify indirectly: the pipeline should not diverge with aggressive clipping
    let result = pipeline.train_batch(&samples);
    assert!(result.avg_loss.is_finite(), "SSC-025: clipped batch loss must be finite");
}

#[test]
fn test_ssc025_gradient_clipping_disabled() {
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        gradient_clip_norm: None,
        ..ClassifyConfig::default()
    };
    let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
    let samples = make_samples();

    let result = pipeline.train_batch(&samples);
    assert!(result.avg_loss.is_finite(), "SSC-025: unclipped batch loss must be finite");
}

#[test]
fn test_ssc025_accumulate_gradients() {
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        learning_rate: 1e-2,
        gradient_clip_norm: None,
        ..ClassifyConfig::default()
    };
    let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
    let samples = make_samples();

    // Split into micro-batches of 1
    pipeline.zero_all_gradients();
    let mut total_samples = 0;
    for sample in &samples {
        let result = pipeline.accumulate_gradients(std::slice::from_ref(sample));
        assert!(result.avg_loss.is_finite());
        assert_eq!(result.total, 1);
        total_samples += result.total;
    }

    // Apply accumulated gradients
    pipeline.apply_accumulated_gradients(total_samples);

    // Pipeline should still work after accumulation
    let result = pipeline.train_batch(&samples);
    assert!(result.avg_loss.is_finite());
}

#[test]
fn test_ssc025_accumulate_gradients_convergence() {
    // Gradient accumulation should converge similarly to full-batch
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        learning_rate: 1e-2,
        gradient_clip_norm: None,
        ..ClassifyConfig::default()
    };
    let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
    let samples = make_samples();

    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;

    for epoch in 0..20 {
        // Zero grads at start of each accumulation cycle
        pipeline.zero_all_gradients();
        let mut epoch_loss = 0.0f32;
        let mut total = 0;
        for sample in &samples {
            let result = pipeline.accumulate_gradients(std::slice::from_ref(sample));
            epoch_loss += result.avg_loss;
            total += result.total;
        }
        pipeline.apply_accumulated_gradients(total);

        let avg = epoch_loss / samples.len() as f32;
        if epoch == 0 {
            first_loss = avg;
        }
        last_loss = avg;
    }

    assert!(
            last_loss < first_loss,
            "SSC-025: Accumulated gradient training must reduce loss. First: {first_loss:.4}, last: {last_loss:.4}"
        );
}

#[test]
fn test_ssc025_accumulate_gradients_empty() {
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);

    let result = pipeline.accumulate_gradients(&[]);
    assert_eq!(result.total, 0);
    assert_eq!(result.correct, 0);

    // apply with 0 should be a no-op
    pipeline.apply_accumulated_gradients(0);
}

#[test]
fn test_ssc025_safety_sample_input_ids() {
    let sample = SafetySample { input: "echo".into(), label: 0 };
    let ids = sample.input_ids();
    assert_eq!(ids, vec![u32::from(b'e'), u32::from(b'c'), u32::from(b'h'), u32::from(b'o')]);
}

#[test]
fn test_ssc025_safety_sample_input_ids_empty() {
    let sample = SafetySample { input: String::new(), label: 0 };
    assert!(sample.input_ids().is_empty());
}

#[test]
fn test_ssc025_batch_result_debug() {
    let r = BatchResult { avg_loss: 1.5, correct: 2, total: 3, grad_norm: 0.0 };
    let debug = format!("{r:?}");
    assert!(debug.contains("BatchResult"));
    assert!(debug.contains("1.5"));
}

#[test]
fn test_ssc025_single_sample_batch() {
    // A batch of 1 should behave like a single train_step
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        gradient_clip_norm: None,
        ..ClassifyConfig::default()
    };
    let mut pipeline = ClassifyPipeline::new(&model_config, classify_config);
    let samples = vec![SafetySample { input: "echo hello".into(), label: 0 }];

    let result = pipeline.train_batch(&samples);
    assert_eq!(result.total, 1);
    assert!(result.avg_loss.is_finite());
    assert!(result.avg_loss > 0.0);
}

// =========================================================================
// Tokenizer integration tests
// =========================================================================

#[test]
fn test_tokenize_byte_level_fallback() {
    // new() pipeline has no tokenizer — should use byte-level
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let pipeline = ClassifyPipeline::new(&model_config, classify_config);

    let ids = pipeline.tokenize("echo");
    assert_eq!(ids, vec![u32::from(b'e'), u32::from(b'c'), u32::from(b'h'), u32::from(b'o')]);
}

#[test]
fn test_tokenize_truncation() {
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        max_seq_len: 4,
        ..ClassifyConfig::default()
    };
    let pipeline = ClassifyPipeline::new(&model_config, classify_config);

    let ids = pipeline.tokenize("hello world");
    assert_eq!(ids.len(), 4, "Should truncate to max_seq_len");
}

#[test]
fn test_tokenize_empty_guard() {
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let pipeline = ClassifyPipeline::new(&model_config, classify_config);

    let ids = pipeline.tokenize("");
    assert_eq!(ids.len(), 1, "Empty input should produce at least 1 token");
    assert_eq!(ids[0], 0, "Empty input guard token should be 0");
}

#[test]
fn test_from_pretrained_missing_dir() {
    let model_config = tiny_config();
    let classify_config = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };

    let result =
        ClassifyPipeline::from_pretrained("/nonexistent/model/dir", &model_config, classify_config);
    assert!(result.is_err(), "from_pretrained with missing dir should fail");
}

#[test]
fn test_summary_shows_tokenizer_byte_level() {
    let model_config = tiny_config();
    let classify_config = ClassifyConfig::default();
    let pipeline = ClassifyPipeline::new(&model_config, classify_config);
    let summary = pipeline.summary();
    assert!(
        summary.contains("byte-level (256)"),
        "Summary should show byte-level tokenizer, got: {summary}"
    );
}

// ── Coverage expansion tests ─────────────────────────────────────

#[test]
fn test_cov_qlora_default_small() {
    let c = ClassifyConfig::qlora_default(4_000_000_000);
    assert_eq!(c.num_classes, 2);
    assert_eq!(c.lora_rank, 16);
    assert!((c.lora_alpha - 32.0).abs() < f32::EPSILON);
    assert!((c.learning_rate - 2e-4).abs() < 1e-6);
    assert_eq!(c.epochs, 3);
    assert_eq!(c.max_seq_len, 256);
    assert_eq!(c.batch_size, 16);
    assert_eq!(c.accumulation_steps, 1);
    assert_eq!(c.gradient_clip_norm, Some(1.0));
    assert!(c.quantize_nf4);
}

#[test]
fn test_cov_qlora_default_large() {
    let c = ClassifyConfig::qlora_default(70_000_000_000);
    assert!((c.learning_rate - 1e-4).abs() < 1e-6);
}

#[test]
fn test_cov_qlora_boundary_13b() {
    let c = ClassifyConfig::qlora_default(13_000_000_000);
    assert!((c.learning_rate - 2e-4).abs() < 1e-6);
}

#[test]
fn test_cov_hp_all_good() {
    let c = ClassifyConfig::qlora_default(4_000_000_000);
    let d = c.validate_hyperparameters(4_000_000_000);
    assert!(!d.has_errors());
}

#[test]
fn test_cov_hp_lr_too_low() {
    let c = ClassifyConfig {
        learning_rate: 1e-5,
        quantize_nf4: true,
        ..ClassifyConfig::qlora_default(4_000_000_000)
    };
    assert!(c.validate_hyperparameters(4_000_000_000).has_warning("C-HP-001"));
}

#[test]
fn test_cov_hp_lr_zero() {
    let c = ClassifyConfig { learning_rate: 0.0, ..ClassifyConfig::default() };
    let d = c.validate_hyperparameters(4_000_000_000);
    assert!(d.has_errors());
    assert!(d.has_warning("C-HP-001"));
}

#[test]
fn test_cov_hp_lr_neg() {
    let c = ClassifyConfig { learning_rate: -0.001, ..ClassifyConfig::default() };
    assert!(c.validate_hyperparameters(4_000_000_000).has_errors());
}

#[test]
fn test_cov_hp_bs_zero() {
    let c = ClassifyConfig { batch_size: 0, ..ClassifyConfig::default() };
    let d = c.validate_hyperparameters(4_000_000_000);
    assert!(d.has_errors());
    assert!(d.has_warning("C-HP-002"));
}

#[test]
fn test_cov_hp_eff_batch_not_16() {
    let c = ClassifyConfig { batch_size: 4, accumulation_steps: 2, ..ClassifyConfig::default() };
    assert!(c.validate_hyperparameters(4_000_000_000).has_warning("C-HP-002"));
}

#[test]
fn test_cov_hp_eff_batch_is_16() {
    let c = ClassifyConfig { batch_size: 4, accumulation_steps: 4, ..ClassifyConfig::default() };
    assert!(!c.validate_hyperparameters(4_000_000_000).has_warning("C-HP-002"));
}

#[test]
fn test_cov_hp_alpha_mismatch() {
    let c = ClassifyConfig { lora_rank: 16, lora_alpha: 8.0, ..ClassifyConfig::default() };
    assert!(c.validate_hyperparameters(4_000_000_000).has_warning("C-HP-003"));
}

#[test]
fn test_cov_hp_alpha_ok() {
    let c = ClassifyConfig { lora_rank: 16, lora_alpha: 32.0, ..ClassifyConfig::default() };
    assert!(!c.validate_hyperparameters(4_000_000_000).has_warning("C-HP-003"));
}

#[test]
fn test_cov_hp_no_clip() {
    let c = ClassifyConfig { gradient_clip_norm: None, ..ClassifyConfig::default() };
    assert!(c.validate_hyperparameters(4_000_000_000).has_warning("C-HP-006"));
}

#[test]
fn test_cov_hp_with_clip() {
    let c = ClassifyConfig { gradient_clip_norm: Some(1.0), ..ClassifyConfig::default() };
    assert!(!c.validate_hyperparameters(4_000_000_000).has_warning("C-HP-006"));
}

#[test]
fn test_cov_hp_lr_non_nf4() {
    let c =
        ClassifyConfig { learning_rate: 1e-5, quantize_nf4: false, ..ClassifyConfig::default() };
    assert!(!c.validate_hyperparameters(4_000_000_000).has_warning("C-HP-001"));
}

#[test]
fn test_cov_hp_lr_big_model() {
    let c = ClassifyConfig { learning_rate: 1e-5, quantize_nf4: true, ..ClassifyConfig::default() };
    assert!(!c.validate_hyperparameters(70_000_000_000).has_warning("C-HP-001"));
}

#[test]
fn test_cov_diag_empty() {
    let d = HyperparamDiagnostics::default();
    assert!(!d.has_warning("X"));
    assert!(!d.has_errors());
}

#[test]
fn test_cov_diag_info_not_warn() {
    let d = HyperparamDiagnostics {
        items: vec![HyperparamDiagnostic {
            contract_id: "C-HP-001",
            severity: DiagSeverity::Info,
            message: "i".into(),
            recommendation: "r".into(),
        }],
    };
    assert!(!d.has_warning("C-HP-001"));
}

#[test]
fn test_cov_diag_warn_counted() {
    let d = HyperparamDiagnostics {
        items: vec![HyperparamDiagnostic {
            contract_id: "C-HP-003",
            severity: DiagSeverity::Warn,
            message: "w".into(),
            recommendation: "r".into(),
        }],
    };
    assert!(d.has_warning("C-HP-003"));
    assert!(!d.has_warning("C-HP-001"));
    assert!(!d.has_errors());
}

#[test]
fn test_cov_diag_error_as_warn() {
    let d = HyperparamDiagnostics {
        items: vec![HyperparamDiagnostic {
            contract_id: "C-HP-002",
            severity: DiagSeverity::Error,
            message: "e".into(),
            recommendation: "r".into(),
        }],
    };
    assert!(d.has_warning("C-HP-002"));
    assert!(d.has_errors());
}

#[test]
fn test_cov_diag_print_all() {
    let d = HyperparamDiagnostics {
        items: vec![
            HyperparamDiagnostic {
                contract_id: "A",
                severity: DiagSeverity::Info,
                message: "i".into(),
                recommendation: "r".into(),
            },
            HyperparamDiagnostic {
                contract_id: "B",
                severity: DiagSeverity::Warn,
                message: "w".into(),
                recommendation: "r".into(),
            },
            HyperparamDiagnostic {
                contract_id: "C",
                severity: DiagSeverity::Error,
                message: "e".into(),
                recommendation: "r".into(),
            },
        ],
    };
    d.print_all();
}

#[test]
fn test_cov_diag_severity_traits() {
    assert_eq!(format!("{:?}", DiagSeverity::Info), "Info");
    assert_eq!(format!("{:?}", DiagSeverity::Warn), "Warn");
    assert_eq!(format!("{:?}", DiagSeverity::Error), "Error");
    let a = DiagSeverity::Warn;
    assert_eq!(a, a);
}

#[test]
fn test_cov_diag_diagnostic_clone() {
    let d = HyperparamDiagnostic {
        contract_id: "C-HP-001",
        severity: DiagSeverity::Info,
        message: "m".into(),
        recommendation: "r".into(),
    };
    let d2 = d.clone();
    assert_eq!(d2.contract_id, "C-HP-001");
    assert!(format!("{d2:?}").contains("C-HP-001"));
}

#[test]
fn test_cov_diags_default_clone() {
    let d = HyperparamDiagnostics::default();
    assert!(d.clone().items.is_empty());
}

#[test]
fn test_cov_data_seq_high() {
    let c = ClassifyConfig { max_seq_len: 512, ..ClassifyConfig::default() };
    let s = DataStats { p99_token_length: 100, imbalance_ratio: 1.0, minority_count: 1000 };
    assert!(c.validate_with_data(&s).has_warning("C-HP-004"));
}

#[test]
fn test_cov_data_seq_ok() {
    let c = ClassifyConfig { max_seq_len: 128, ..ClassifyConfig::default() };
    let s = DataStats { p99_token_length: 100, imbalance_ratio: 1.0, minority_count: 1000 };
    assert!(!c.validate_with_data(&s).has_warning("C-HP-004"));
}

#[test]
fn test_cov_data_seq_zero_p99() {
    let c = ClassifyConfig { max_seq_len: 512, ..ClassifyConfig::default() };
    let s = DataStats { p99_token_length: 0, imbalance_ratio: 1.0, minority_count: 1000 };
    assert!(!c.validate_with_data(&s).has_warning("C-HP-004"));
}

#[test]
fn test_cov_data_imb_few_epochs() {
    let c = ClassifyConfig {
        epochs: 1,
        batch_size: 16,
        accumulation_steps: 1,
        ..ClassifyConfig::default()
    };
    let s = DataStats { p99_token_length: 100, imbalance_ratio: 10.0, minority_count: 100 };
    assert!(c.validate_with_data(&s).has_warning("C-HP-008"));
}

#[test]
fn test_cov_data_imb_ok_epochs() {
    let c = ClassifyConfig { epochs: 3, ..ClassifyConfig::default() };
    let s = DataStats { p99_token_length: 100, imbalance_ratio: 10.0, minority_count: 100 };
    assert!(!c.validate_with_data(&s).has_warning("C-HP-008"));
}

#[test]
fn test_cov_data_low_imb() {
    let c = ClassifyConfig { epochs: 1, ..ClassifyConfig::default() };
    let s = DataStats { p99_token_length: 100, imbalance_ratio: 2.0, minority_count: 100 };
    assert!(!c.validate_with_data(&s).has_warning("C-HP-008"));
}

#[test]
fn test_cov_data_both_warn() {
    let c = ClassifyConfig {
        max_seq_len: 1024,
        epochs: 1,
        batch_size: 16,
        accumulation_steps: 1,
        ..ClassifyConfig::default()
    };
    let s = DataStats { p99_token_length: 50, imbalance_ratio: 20.0, minority_count: 80 };
    let d = c.validate_with_data(&s);
    assert!(d.has_warning("C-HP-004"));
    assert!(d.has_warning("C-HP-008"));
}

#[test]
fn test_cov_pretok_basic() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let p = ClassifyPipeline::new(&mc, cc);
    let tok = p.pre_tokenize(&make_samples());
    assert_eq!(tok.len(), 3);
    for (t, s) in tok.iter().zip(make_samples().iter()) {
        assert_eq!(t.label, s.label);
        assert!(!t.token_ids.is_empty());
    }
}

#[test]
fn test_cov_pretok_truncate() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        max_seq_len: 4,
        ..ClassifyConfig::default()
    };
    let p = ClassifyPipeline::new(&mc, cc);
    let tok = p.pre_tokenize(&[SafetySample { input: "echo hello world".into(), label: 0 }]);
    assert_eq!(tok[0].token_ids.len(), 4);
}

#[test]
fn test_cov_pretok_empty() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let p = ClassifyPipeline::new(&mc, cc);
    let tok = p.pre_tokenize(&[SafetySample { input: String::new(), label: 0 }]);
    assert!(!tok[0].token_ids.is_empty());
}

#[test]
fn test_cov_btok_empty() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    let r = p.train_batch_tokenized(&[]);
    assert_eq!(r.total, 0);
}

#[test]
fn test_cov_btok_basic() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    let s = vec![
        TokenizedSample { token_ids: vec![1, 2, 3], label: 0 },
        TokenizedSample { token_ids: vec![4, 5, 6], label: 1 },
    ];
    let r = p.train_batch_tokenized(&s);
    assert_eq!(r.total, 2);
    assert!(r.avg_loss.is_finite() && r.avg_loss > 0.0);
}

#[test]
fn test_cov_btok_converge() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        learning_rate: 1e-2,
        gradient_clip_norm: None,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    let s = vec![
        TokenizedSample { token_ids: vec![1, 2, 3], label: 0 },
        TokenizedSample { token_ids: vec![4, 5, 6], label: 1 },
        TokenizedSample { token_ids: vec![7, 8, 9], label: 2 },
    ];
    let mut first = 0.0f32;
    let mut last = 0.0f32;
    for ep in 0..20 {
        let r = p.train_batch_tokenized(&s);
        if ep == 0 {
            first = r.avg_loss;
        }
        last = r.avg_loss;
    }
    assert!(last < first);
}

#[test]
fn test_cov_btok_clip() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        gradient_clip_norm: Some(0.5),
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    let r = p.train_batch_tokenized(&[TokenizedSample { token_ids: vec![1, 2, 3], label: 0 }]);
    assert!(r.avg_loss.is_finite());
}

#[test]
fn test_cov_atok_empty() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    assert_eq!(p.accumulate_gradients_tokenized(&[]).total, 0);
}

#[test]
fn test_cov_atok_basic() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        learning_rate: 1e-2,
        gradient_clip_norm: None,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    let mb = vec![
        TokenizedSample { token_ids: vec![1, 2, 3], label: 0 },
        TokenizedSample { token_ids: vec![4, 5, 6], label: 1 },
    ];
    p.zero_all_gradients();
    let r = p.accumulate_gradients_tokenized(&mb);
    assert_eq!(r.total, 2);
    assert!(r.avg_loss.is_finite());
    p.apply_accumulated_gradients(r.total);
}

#[test]
fn test_cov_fwd_only() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    let (l, pr) = p.forward_only(&[1, 2, 3], 0);
    assert!(l.is_finite() && l > 0.0);
    assert!(pr < 3);
}

#[test]
fn test_cov_fwd_all_labels() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    for lab in 0..3 {
        let (l, _) = p.forward_only(&[1, 2, 3], lab);
        assert!(l.is_finite());
    }
}

#[test]
fn test_cov_fwd_tokenized() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    let (l, pr) = p.forward_only_tokenized(&[1, 2, 3], 0);
    assert!(l.is_finite() && pr < 3);
}

#[test]
fn test_cov_fwd_probs() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    let (l, pr, probs) = p.forward_only_with_probs(&[1, 2, 3], 0);
    assert!(l.is_finite() && l > 0.0 && pr < 3);
    assert_eq!(probs.len(), 3);
    assert!(((probs.iter().sum::<f32>()) - 1.0).abs() < 1e-5);
    for &v in &probs {
        assert!((0.0..=1.0).contains(&v));
    }
}

#[test]
fn test_cov_fwd_probs_argmax() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    let (_, pred, probs) = p.forward_only_with_probs(&[1, 2, 3], 0);
    let am = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(pred, am);
}

#[test]
fn test_cov_cw_train() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        class_weights: Some(vec![1.0, 5.0, 1.0]),
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    assert!(p.train_step(&[1, 2, 3], 1).is_finite());
}

#[test]
fn test_cov_cw_batch() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        class_weights: Some(vec![0.5, 5.0, 0.5]),
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    assert!(p.train_batch(&make_samples()).avg_loss.is_finite());
}

#[test]
fn test_cov_cw_fwd() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        class_weights: Some(vec![1.0, 2.0, 3.0]),
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    assert!(p.forward_only(&[1, 2, 3], 2).0.is_finite());
}

#[test]
fn test_cov_set_lr() {
    let mc = tiny_config();
    let mut p = ClassifyPipeline::new(
        &mc,
        ClassifyConfig { learning_rate: 1e-3, ..ClassifyConfig::default() },
    );
    assert!((p.optimizer_lr() - 1e-3).abs() < 1e-6);
    p.set_optimizer_lr(5e-4);
    assert!((p.optimizer_lr() - 5e-4).abs() < 1e-6);
}

#[test]
fn test_cov_opt_ref() {
    let mc = tiny_config();
    let p = ClassifyPipeline::new(&mc, ClassifyConfig::default());
    assert!((p.optimizer().lr() - ClassifyConfig::default().learning_rate).abs() < 1e-8);
}

#[test]
fn test_cov_opt_mut() {
    let mc = tiny_config();
    let mut p = ClassifyPipeline::new(&mc, ClassifyConfig::default());
    p.optimizer_mut().set_lr(2e-4);
    assert!((p.optimizer_lr() - 2e-4).abs() < 1e-6);
}

#[test]
fn test_cov_model_dir_none() {
    let p = ClassifyPipeline::new(&tiny_config(), ClassifyConfig::default());
    assert!(p.model_dir().is_none());
}

#[test]
fn test_cov_set_model_path() {
    let mut p = ClassifyPipeline::new(&tiny_config(), ClassifyConfig::default());
    p.set_model_path("/tmp/m");
    assert_eq!(p.model_dir(), Some(Path::new("/tmp/m")));
}

#[test]
fn test_cov_set_model_path_buf() {
    let mut p = ClassifyPipeline::new(&tiny_config(), ClassifyConfig::default());
    p.set_model_path(PathBuf::from("/opt/v1"));
    assert_eq!(p.model_dir(), Some(Path::new("/opt/v1")));
}

#[test]
fn test_cov_is_cuda() {
    let p = ClassifyPipeline::new(&tiny_config(), ClassifyConfig::default());
    #[cfg(not(feature = "cuda"))]
    assert!(!p.is_cuda());
    let _ = p.is_cuda();
}

#[test]
fn test_cov_gpu_name() {
    let p = ClassifyPipeline::new(&tiny_config(), ClassifyConfig::default());
    #[cfg(not(feature = "cuda"))]
    assert!(p.gpu_name().is_none());
    let _ = p.gpu_name();
}

#[test]
fn test_cov_gpu_mem() {
    let p = ClassifyPipeline::new(&tiny_config(), ClassifyConfig::default());
    #[cfg(not(feature = "cuda"))]
    assert!(p.gpu_total_memory().is_none());
    let _ = p.gpu_total_memory();
}

#[test]
fn test_cov_is_gpu_training() {
    let p = ClassifyPipeline::new(&tiny_config(), ClassifyConfig::default());
    #[cfg(not(feature = "cuda"))]
    assert!(!p.is_gpu_training());
    let _ = p.is_gpu_training();
}

#[test]
fn test_cov_num_params() {
    let mc = tiny_config();
    let p = ClassifyPipeline::new(
        &mc,
        ClassifyConfig {
            num_classes: 5,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        },
    );
    let n = p.num_trainable_parameters();
    assert!(n > 0);
    assert!(n >= mc.hidden_size * 5 + 5);
}

#[test]
fn test_cov_params_scale() {
    let mc = tiny_config();
    let s = ClassifyPipeline::new(
        &mc,
        ClassifyConfig {
            num_classes: 2,
            lora_rank: 2,
            lora_alpha: 2.0,
            ..ClassifyConfig::default()
        },
    );
    let l = ClassifyPipeline::new(
        &mc,
        ClassifyConfig {
            num_classes: 2,
            lora_rank: 16,
            lora_alpha: 16.0,
            ..ClassifyConfig::default()
        },
    );
    assert!(l.num_trainable_parameters() > s.num_trainable_parameters());
}

#[test]
fn test_cov_grads_len() {
    let mc = tiny_config();
    let p = ClassifyPipeline::new(
        &mc,
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        },
    );
    assert_eq!(p.collect_lora_gradients().len(), p.num_trainable_parameters());
}

#[test]
fn test_cov_grads_zero() {
    let mc = tiny_config();
    let p = ClassifyPipeline::new(
        &mc,
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        },
    );
    assert!(p.collect_lora_gradients().iter().all(|&g| g == 0.0));
}

#[test]
fn test_cov_grads_nonzero() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        gradient_clip_norm: None,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    p.zero_all_gradients();
    let _ = p.accumulate_gradients(&[SafetySample { input: "echo hi".into(), label: 0 }]);
    assert!(p.collect_lora_gradients().iter().any(|&g| g != 0.0));
}

#[test]
fn test_cov_apply_grads() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    let n = p.num_trainable_parameters();
    p.apply_lora_gradients(&(0..n).map(|i| i as f32 * 0.001).collect::<Vec<_>>());
}

#[test]
fn test_cov_apply_grads_short() {
    let mc = tiny_config();
    let mut p = ClassifyPipeline::new(
        &mc,
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        },
    );
    p.apply_lora_gradients(&[0.1, 0.2]);
}

#[test]
fn test_cov_apply_grads_empty() {
    let mc = tiny_config();
    let mut p = ClassifyPipeline::new(
        &mc,
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        },
    );
    p.apply_lora_gradients(&[]);
}

#[test]
fn test_cov_merge_idem() {
    let mc = tiny_config();
    let mut p = ClassifyPipeline::new(
        &mc,
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        },
    );
    p.merge_adapters();
    p.merge_adapters();
    for lora in &p.lora_layers {
        assert!(lora.is_merged());
    }
}

#[test]
fn test_cov_dispatch_lora() {
    let mc = tiny_config();
    let mut p = ClassifyPipeline::new(
        &mc,
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        },
    );
    assert!(!p.lora_layers.is_empty());
    assert!(!p.forward_hidden_dispatch(&[1, 2, 3]).data().is_empty());
}

#[test]
fn test_cov_summary_detail() {
    let p = ClassifyPipeline::new(
        &tiny_config(),
        ClassifyConfig {
            num_classes: 5,
            lora_rank: 8,
            lora_alpha: 16.0,
            ..ClassifyConfig::default()
        },
    );
    let s = p.summary();
    assert!(s.contains("ClassifyPipeline"));
    assert!(s.contains("64 hidden"));
    assert!(s.contains("CPU") || s.contains("CUDA"));
    assert!(s.contains("rank=8"));
}

#[test]
fn test_cov_from_pretrained_err() {
    assert!(ClassifyPipeline::from_pretrained(
        "/nonexist",
        &tiny_config(),
        ClassifyConfig::default()
    )
    .is_err());
}

#[test]
fn test_cov_from_apr_err() {
    assert!(ClassifyPipeline::from_apr(
        Path::new("/nonexist.apr"),
        &tiny_config(),
        ClassifyConfig::default()
    )
    .is_err());
}

#[test]
fn test_cov_load_corpus_err() {
    let p = ClassifyPipeline::new(
        &tiny_config(),
        ClassifyConfig { num_classes: 3, ..ClassifyConfig::default() },
    );
    assert!(p.load_corpus(Path::new("/ne.jsonl")).is_err());
}

#[test]
fn test_cov_load_ml_corpus_err() {
    let p = ClassifyPipeline::new(
        &tiny_config(),
        ClassifyConfig { num_classes: 3, ..ClassifyConfig::default() },
    );
    assert!(p.load_multi_label_corpus(Path::new("/ne.jsonl")).is_err());
}

#[test]
fn test_cov_batch_accuracy_1() {
    let r = BatchResult { avg_loss: 1.0, correct: 1, total: 100, grad_norm: 0.5 };
    assert!((r.accuracy() - 0.01).abs() < 1e-6);
}

#[test]
fn test_cov_batch_clone() {
    let r = BatchResult { avg_loss: 1.5, correct: 2, total: 3, grad_norm: 0.42 };
    let r2 = r.clone();
    assert_eq!(r2.correct, 2);
    assert!((r2.grad_norm - 0.42).abs() < 1e-6);
}

#[test]
fn test_cov_config_clone() {
    let c = ClassifyConfig::default();
    let c2 = c.clone();
    assert_eq!(c2.num_classes, c.num_classes);
    assert!(format!("{c2:?}").contains("ClassifyConfig"));
}

#[test]
fn test_cov_config_nf4_false() {
    assert!(!ClassifyConfig::default().quantize_nf4);
}

#[test]
fn test_cov_config_cw_none() {
    assert!(ClassifyConfig::default().class_weights.is_none());
}

#[test]
fn test_cov_zero_grads() {
    let mc = tiny_config();
    let mut p = ClassifyPipeline::new(
        &mc,
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        },
    );
    let _ = p.train_step(&[1, 2, 3], 0);
    p.zero_all_gradients();
    assert!(p.compute_grad_norm().abs() < 1e-6);
}

#[test]
fn test_cov_grad_norm() {
    let mc = tiny_config();
    let mut p = ClassifyPipeline::new(
        &mc,
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            gradient_clip_norm: None,
            ..ClassifyConfig::default()
        },
    );
    p.zero_all_gradients();
    let _ = p.accumulate_gradients(&[SafetySample { input: "ls".into(), label: 0 }]);
    assert!(p.compute_grad_norm() >= 0.0);
}

#[test]
fn test_cov_scale_grads() {
    let mc = tiny_config();
    let mut p = ClassifyPipeline::new(
        &mc,
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            gradient_clip_norm: None,
            ..ClassifyConfig::default()
        },
    );
    p.zero_all_gradients();
    let _ = p.accumulate_gradients(&[SafetySample { input: "ls".into(), label: 0 }]);
    let b = p.compute_grad_norm();
    p.scale_all_gradients(2.0);
    let a = p.compute_grad_norm();
    if b > 1e-8 {
        assert!((a / b - 2.0).abs() < 0.01);
    }
}

#[test]
fn test_cov_binary() {
    let mut p = ClassifyPipeline::new(
        &tiny_config(),
        ClassifyConfig {
            num_classes: 2,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        },
    );
    assert!(p.train_step(&[1, 2, 3], 0).is_finite());
    assert!(p.train_step(&[4, 5, 6], 1).is_finite());
}

#[test]
fn test_cov_many_classes() {
    let mut p = ClassifyPipeline::new(
        &tiny_config(),
        ClassifyConfig {
            num_classes: 20,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        },
    );
    assert!(p.train_step(&[1, 2, 3], 15).is_finite());
}

#[test]
fn test_cov_single_token() {
    let mut p = ClassifyPipeline::new(
        &tiny_config(),
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        },
    );
    assert!(p.train_step(&[42], 1).is_finite());
}

#[test]
fn test_cov_long_input() {
    // Input (50 tokens) exceeds max_seq_len (10); train_step must clamp without OOB.
    let mut p = ClassifyPipeline::new(
        &tiny_config(),
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            max_seq_len: 10,
            ..ClassifyConfig::default()
        },
    );
    let long_input: Vec<u32> = (0..50).collect();
    assert!(p.train_step(&long_input, 0).is_finite());
}

/// Regression: forward_only_with_probs must clamp seq_len to max_seq_len (entrenar#280).
#[test]
fn test_forward_only_with_probs_long_input() {
    let mut p = ClassifyPipeline::new(
        &tiny_config(),
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            max_seq_len: 10,
            ..ClassifyConfig::default()
        },
    );
    let long_input: Vec<u32> = (0..50).collect();
    let (loss, pred, probs) = p.forward_only_with_probs(&long_input, 0);
    assert!(loss.is_finite());
    assert!(pred < 3);
    assert_eq!(probs.len(), 3);
}

/// Regression: multi_label_train_step must clamp seq_len to max_seq_len (entrenar#280).
#[test]
fn test_multi_label_train_step_long_input() {
    let mut p = ClassifyPipeline::new(
        &tiny_config(),
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            max_seq_len: 10,
            ..ClassifyConfig::default()
        },
    );
    let long_input: Vec<u32> = (0..50).collect();
    let targets = vec![1.0, 0.0, 1.0];
    let loss = p.multi_label_train_step(&long_input, &targets);
    assert!(loss.is_finite());
}

#[test]
fn test_cov_lora_count() {
    let p = ClassifyPipeline::new(
        &tiny_config(),
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        },
    );
    assert_eq!(p.lora_layers.len(), 4);
}

#[test]
fn test_cov_lora_grad() {
    let p = ClassifyPipeline::new(
        &tiny_config(),
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            ..ClassifyConfig::default()
        },
    );
    for l in &p.lora_layers {
        assert!(l.lora_a().requires_grad() && l.lora_b().requires_grad());
    }
}

#[test]
fn test_cov_train_eval() {
    let mc = tiny_config();
    let mut p = ClassifyPipeline::new(
        &mc,
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            learning_rate: 1e-2,
            ..ClassifyConfig::default()
        },
    );
    for _ in 0..5 {
        let _ = p.train_step(&[1, 2, 3], 0);
    }
    let (l, pr) = p.forward_only(&[1, 2, 3], 0);
    assert!(l.is_finite() && pr < 3);
}

#[test]
fn test_cov_batch_then_probs() {
    let mc = tiny_config();
    let mut p = ClassifyPipeline::new(
        &mc,
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            learning_rate: 1e-2,
            ..ClassifyConfig::default()
        },
    );
    for _ in 0..5 {
        let _ = p.train_batch(&make_samples());
    }
    let (l, pr, probs) = p.forward_only_with_probs(&[1, 2, 3], 0);
    assert!(l.is_finite() && pr < 3 && probs.len() == 3);
}

#[test]
fn test_cov_multi_diag() {
    let c = ClassifyConfig {
        learning_rate: 0.0,
        batch_size: 0,
        lora_rank: 16,
        lora_alpha: 8.0,
        gradient_clip_norm: None,
        quantize_nf4: false,
        ..ClassifyConfig::default()
    };
    let d = c.validate_hyperparameters(4_000_000_000);
    assert!(d.has_errors());
    assert!(d.items.len() >= 3);
}

#[test]
fn test_cov_nf4_config() {
    let p = ClassifyPipeline::new(
        &tiny_config(),
        ClassifyConfig { quantize_nf4: true, ..ClassifyConfig::default() },
    );
    // NF4 config is set regardless of device (CPU or CUDA)
    assert!(p.config.quantize_nf4);
}

#[test]
fn test_cov_nf4_btok() {
    let mut p = ClassifyPipeline::new(
        &tiny_config(),
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            quantize_nf4: true,
            ..ClassifyConfig::default()
        },
    );
    assert!(p
        .train_batch_tokenized(&[TokenizedSample { token_ids: vec![1, 2, 3], label: 0 }])
        .avg_loss
        .is_finite());
}

#[test]
fn test_cov_apply_accum_nf4() {
    let mut p = ClassifyPipeline::new(
        &tiny_config(),
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            quantize_nf4: true,
            gradient_clip_norm: Some(1.0),
            ..ClassifyConfig::default()
        },
    );
    p.zero_all_gradients();
    let _ = p.accumulate_gradients(&[SafetySample { input: "echo t".into(), label: 0 }]);
    p.apply_accumulated_gradients(1);
}

#[test]
fn test_cov_apply_accum_fp32() {
    let mut p = ClassifyPipeline::new(
        &tiny_config(),
        ClassifyConfig {
            num_classes: 3,
            lora_rank: 4,
            lora_alpha: 4.0,
            quantize_nf4: false,
            gradient_clip_norm: Some(1.0),
            ..ClassifyConfig::default()
        },
    );
    p.zero_all_gradients();
    let _ = p.accumulate_gradients(&[SafetySample { input: "echo t".into(), label: 0 }]);
    p.apply_accumulated_gradients(1);
}

// ── test_cov4 additional coverage tests ────────────────────────

#[test]
fn test_cov4_classify_config_debug() {
    let c = ClassifyConfig::default();
    let dbg = format!("{c:?}");
    assert!(dbg.contains("num_classes"));
    assert!(dbg.contains("lora_rank"));
    assert!(dbg.contains("learning_rate"));
    assert!(dbg.contains("batch_size"));
    assert!(dbg.contains("accumulation_steps"));
    assert!(dbg.contains("gradient_clip_norm"));
    assert!(dbg.contains("class_weights"));
    assert!(dbg.contains("quantize_nf4"));
}

#[test]
fn test_cov4_classify_config_all_fields() {
    let c = ClassifyConfig {
        num_classes: 10,
        lora_rank: 32,
        lora_alpha: 64.0,
        learning_rate: 3e-4,
        epochs: 5,
        max_seq_len: 1024,
        log_interval: 50,
        batch_size: 8,
        accumulation_steps: 2,
        gradient_clip_norm: Some(2.0),
        class_weights: Some(vec![1.0; 10]),
        quantize_nf4: true,
    };
    assert_eq!(c.num_classes, 10);
    assert_eq!(c.lora_rank, 32);
    assert!((c.lora_alpha - 64.0).abs() < f32::EPSILON);
    assert_eq!(c.epochs, 5);
    assert_eq!(c.max_seq_len, 1024);
    assert_eq!(c.log_interval, 50);
    assert_eq!(c.batch_size, 8);
    assert_eq!(c.accumulation_steps, 2);
    assert_eq!(c.gradient_clip_norm, Some(2.0));
    assert!(c.class_weights.is_some());
    assert!(c.quantize_nf4);
}

#[test]
fn test_cov4_batch_result_fields() {
    let r = BatchResult { avg_loss: 2.3, correct: 7, total: 10, grad_norm: 1.5 };
    assert!((r.avg_loss - 2.3).abs() < 1e-5);
    assert_eq!(r.correct, 7);
    assert_eq!(r.total, 10);
    assert!((r.grad_norm - 1.5).abs() < 1e-5);
    assert!((r.accuracy() - 0.7).abs() < 1e-5);
}

#[test]
fn test_cov4_hp_validate_combined_diags() {
    // Config with multiple issues at once
    let c = ClassifyConfig {
        learning_rate: 1e-6,
        quantize_nf4: true,
        batch_size: 2,
        accumulation_steps: 2, // eff=4, not 16
        lora_rank: 8,
        lora_alpha: 40.0, // ratio=5, expected alpha=16
        gradient_clip_norm: None,
        ..ClassifyConfig::default()
    };
    let d = c.validate_hyperparameters(4_000_000_000);
    // Should have multiple warnings
    assert!(d.has_warning("C-HP-001")); // lr too low
    assert!(d.has_warning("C-HP-002")); // eff batch not 16
    assert!(d.has_warning("C-HP-003")); // alpha mismatch
    assert!(d.has_warning("C-HP-006")); // no grad clip
}

#[test]
fn test_cov4_hp_validate_data_both_directions() {
    // Test validate_with_data with balanced data + sufficient epochs
    let c = ClassifyConfig { max_seq_len: 64, epochs: 5, ..ClassifyConfig::default() };
    let s = DataStats { p99_token_length: 50, imbalance_ratio: 1.0, minority_count: 500 };
    let d = c.validate_with_data(&s);
    assert!(!d.has_warning("C-HP-004")); // 64 <= 2*50
    assert!(!d.has_warning("C-HP-008")); // imbalance < 5
}

#[test]
fn test_cov4_hp_diagnostics_has_warning_multiple() {
    let d = HyperparamDiagnostics {
        items: vec![
            HyperparamDiagnostic {
                contract_id: "C-HP-001",
                severity: DiagSeverity::Warn,
                message: "w1".into(),
                recommendation: "r1".into(),
            },
            HyperparamDiagnostic {
                contract_id: "C-HP-002",
                severity: DiagSeverity::Error,
                message: "e1".into(),
                recommendation: "r2".into(),
            },
            HyperparamDiagnostic {
                contract_id: "C-HP-003",
                severity: DiagSeverity::Info,
                message: "i1".into(),
                recommendation: "r3".into(),
            },
        ],
    };
    assert!(d.has_warning("C-HP-001"));
    assert!(d.has_warning("C-HP-002")); // Error counts as warning
    assert!(!d.has_warning("C-HP-003")); // Info does NOT count as warning
    assert!(d.has_errors());
    assert!(!d.has_warning("C-HP-999")); // nonexistent
}

#[test]
fn test_cov4_diag_severity_copy() {
    let s = DiagSeverity::Warn;
    let s2 = s;
    assert_eq!(s, s2);
}

#[test]
fn test_cov4_train_step_deterministic() {
    // Same input should give similar loss on fresh pipeline
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut p1 = ClassifyPipeline::new(&mc, cc.clone());
    let mut p2 = ClassifyPipeline::new(&mc, cc);

    // Only compare determinism when both pipelines use the same device.
    // VRAM pressure can cause one to fall back to CPU while the other uses CUDA.
    if p1.is_cuda() != p2.is_cuda() {
        return; // Mixed device — determinism comparison is meaningless
    }

    let loss1 = p1.train_step(&[1, 2, 3], 0);
    let loss2 = p2.train_step(&[1, 2, 3], 0);

    // Both initialized with same seed → same loss.
    // GPU (cuBLAS GEMM) has nondeterministic reduction order across
    // separate context initializations, so use a wider tolerance.
    let tol = if p1.is_cuda() { 1e-2 } else { 1e-4 };
    assert!((loss1 - loss2).abs() < tol, "Deterministic: {loss1} vs {loss2}");
}

#[test]
fn test_cov4_multi_label_different_targets() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 4,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);

    // All zeros target
    let loss_zeros = p.multi_label_train_step(&[1, 2, 3], &[0.0, 0.0, 0.0, 0.0]);
    assert!(loss_zeros.is_finite());

    // All ones target
    let loss_ones = p.multi_label_train_step(&[1, 2, 3], &[1.0, 1.0, 1.0, 1.0]);
    assert!(loss_ones.is_finite());
}

#[test]
fn test_cov4_forward_only_all_labels_tokenized() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 5,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);

    for label in 0..5 {
        let (loss, pred) = p.forward_only_tokenized(&[1, 2, 3], label);
        assert!(loss.is_finite());
        assert!(pred < 5);
    }
}

#[test]
fn test_cov4_pretokenize_multiple_samples() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        max_seq_len: 10,
        ..ClassifyConfig::default()
    };
    let p = ClassifyPipeline::new(&mc, cc);

    let samples = vec![
        SafetySample { input: "a".into(), label: 0 },
        SafetySample { input: "abcdefghijklmnop".into(), label: 1 }, // will be truncated
        SafetySample { input: String::new(), label: 2 },             // empty
    ];
    let tok = p.pre_tokenize(&samples);
    assert_eq!(tok.len(), 3);
    assert_eq!(tok[0].label, 0);
    assert_eq!(tok[1].label, 1);
    assert!(tok[1].token_ids.len() <= 10); // truncated
    assert_eq!(tok[2].label, 2);
    assert!(!tok[2].token_ids.is_empty()); // empty guard
}

#[test]
fn test_cov4_train_batch_tokenized_multiple() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        learning_rate: 1e-2,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);

    let samples = vec![
        TokenizedSample { token_ids: vec![1, 2], label: 0 },
        TokenizedSample { token_ids: vec![3, 4], label: 1 },
        TokenizedSample { token_ids: vec![5, 6], label: 2 },
        TokenizedSample { token_ids: vec![7, 8], label: 0 },
        TokenizedSample { token_ids: vec![9, 10], label: 1 },
    ];
    let r = p.train_batch_tokenized(&samples);
    assert_eq!(r.total, 5);
    assert!(r.avg_loss.is_finite() && r.avg_loss > 0.0);
}

#[test]
fn test_cov4_accumulate_gradients_tokenized_then_apply() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        learning_rate: 1e-2,
        gradient_clip_norm: None,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);

    p.zero_all_gradients();
    let r1 =
        p.accumulate_gradients_tokenized(&[TokenizedSample { token_ids: vec![1, 2], label: 0 }]);
    let r2 =
        p.accumulate_gradients_tokenized(&[TokenizedSample { token_ids: vec![3, 4], label: 1 }]);
    assert!(r1.avg_loss.is_finite());
    assert!(r2.avg_loss.is_finite());

    p.apply_accumulated_gradients(r1.total + r2.total);

    // Pipeline should still work
    let r = p.train_batch_tokenized(&[TokenizedSample { token_ids: vec![5, 6], label: 2 }]);
    assert!(r.avg_loss.is_finite());
}

#[test]
fn test_cov4_forward_only_with_probs_classes() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 4,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    let (loss, pred, probs) = p.forward_only_with_probs(&[1, 2, 3], 2);
    assert!(loss.is_finite());
    assert!(pred < 4);
    assert_eq!(probs.len(), 4);
    // Verify softmax properties
    for &v in &probs {
        assert!((0.0..=1.0).contains(&v));
    }
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_cov4_class_weights_all_labels() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        class_weights: Some(vec![0.5, 2.0, 1.5]),
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);

    for label in 0..3 {
        let loss = p.train_step(&[1, 2, 3], label);
        assert!(loss.is_finite(), "Loss for label {label} must be finite");
        assert!(loss > 0.0);
    }
}

#[test]
fn test_cov4_qlora_default_boundary_values() {
    // Exactly at 13B boundary
    let c = ClassifyConfig::qlora_default(13_000_000_000);
    assert!((c.learning_rate - 2e-4).abs() < 1e-6);
    assert!(c.quantize_nf4);

    // Just above 13B
    let c2 = ClassifyConfig::qlora_default(13_000_000_001);
    assert!((c2.learning_rate - 1e-4).abs() < 1e-6);

    // 1B model
    let c3 = ClassifyConfig::qlora_default(1_000_000_000);
    assert!((c3.learning_rate - 2e-4).abs() < 1e-6);
}

#[test]
fn test_cov4_collect_and_apply_roundtrip() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        gradient_clip_norm: None,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);

    // Accumulate some gradients
    p.zero_all_gradients();
    let _ = p.accumulate_gradients(&[
        SafetySample { input: "echo hello".into(), label: 0 },
        SafetySample { input: "rm -rf /".into(), label: 1 },
    ]);

    // Collect gradients
    let grads = p.collect_lora_gradients();
    assert_eq!(grads.len(), p.num_trainable_parameters());

    // Apply them back (simulating AllReduce)
    p.apply_lora_gradients(&grads);

    // Pipeline should still function
    let r = p.train_batch(&make_samples());
    assert!(r.avg_loss.is_finite());
}

#[test]
fn test_cov4_scale_all_gradients_zero() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        gradient_clip_norm: None,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    p.zero_all_gradients();
    let _ = p.accumulate_gradients(&[SafetySample { input: "ls".into(), label: 0 }]);

    // Scale by zero should zero all gradients
    p.scale_all_gradients(0.0);
    assert!(p.compute_grad_norm().abs() < 1e-8);
}

#[test]
fn test_cov4_forward_hidden_dispatch_single_token() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);
    let h = p.forward_hidden_dispatch(&[42]);
    assert!(!h.data().is_empty());
}

#[test]
fn test_cov4_summary_contains_all_info() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 7,
        lora_rank: 8,
        lora_alpha: 16.0,
        ..ClassifyConfig::default()
    };
    let p = ClassifyPipeline::new(&mc, cc);
    let s = p.summary();

    assert!(s.contains("ClassifyPipeline"));
    assert!(s.contains("CPU") || s.contains("CUDA"));
    assert!(s.contains("byte-level (256)"));
    assert!(s.contains("rank=8"));
    assert!(s.contains("alpha=16.0"));
    assert!(s.contains("->7"));
}

#[test]
fn test_cov4_model_dir_after_pretrained_error() {
    // Even after a failed from_pretrained, new() pipelines have None model_dir
    let p = ClassifyPipeline::new(&tiny_config(), ClassifyConfig::default());
    assert!(p.model_dir().is_none());
}

#[test]
fn test_cov4_set_model_path_overwrite() {
    let mut p = ClassifyPipeline::new(&tiny_config(), ClassifyConfig::default());
    p.set_model_path("/tmp/model1");
    assert_eq!(p.model_dir(), Some(Path::new("/tmp/model1")));
    p.set_model_path("/tmp/model2");
    assert_eq!(p.model_dir(), Some(Path::new("/tmp/model2")));
}

#[test]
fn test_cov4_train_batch_large() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        batch_size: 16,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);

    // 10 samples
    let samples: Vec<SafetySample> =
        (0..10).map(|i| SafetySample { input: format!("cmd {i}"), label: i % 3 }).collect();

    let r = p.train_batch(&samples);
    assert_eq!(r.total, 10);
    assert!(r.avg_loss.is_finite());
}

#[test]
fn test_cov4_from_apr_nonexistent() {
    let mc = tiny_config();
    let cc = ClassifyConfig::default();
    let result = ClassifyPipeline::from_apr(Path::new("/tmp/nonexistent.apr"), &mc, cc);
    assert!(result.is_err());
}

#[test]
fn test_cov4_zero_grads_then_check() {
    let mc = tiny_config();
    let cc = ClassifyConfig {
        num_classes: 3,
        lora_rank: 4,
        lora_alpha: 4.0,
        gradient_clip_norm: None,
        ..ClassifyConfig::default()
    };
    let mut p = ClassifyPipeline::new(&mc, cc);

    // Train a step to accumulate gradients
    let _ = p.train_step(&[1, 2, 3], 0);

    // Gradients should be present after a train_step followed by zero
    p.zero_all_gradients();
    let norm = p.compute_grad_norm();
    assert!(norm.abs() < 1e-6, "After zero_all_gradients, norm should be ~0, got {norm}");
}
