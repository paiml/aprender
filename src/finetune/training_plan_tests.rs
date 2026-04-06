use super::*;

#[test]
fn test_plan_missing_data_file() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/nonexistent/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 5,
        output_dir: PathBuf::from("/tmp/test-plan-out"),
        strategy: "manual".to_string(),
        budget: 10,
        scout: false,
        max_epochs: 1,
        manual_lr: Some(1e-4),
        manual_lora_rank: Some(16),
        manual_batch_size: Some(32),
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let result = plan(&config);
    assert!(result.is_err());
}

#[test]
fn test_plan_manual_strategy_warns() {
    // Create a temp JSONL file
    let dir = tempfile::tempdir().expect("valid");
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    for i in 0..50 {
        lines.push(format!(r#"{{"input": "echo test {i}", "label": {}}}"#, i % 5));
    }
    std::fs::write(&data_path, lines.join("\n")).expect("valid");

    let config = PlanConfig {
        task: "classify".to_string(),
        data_path,
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 5,
        output_dir: dir.path().to_path_buf(),
        strategy: "manual".to_string(),
        budget: 10,
        scout: false,
        max_epochs: 1,
        manual_lr: Some(1e-4),
        manual_lora_rank: Some(16),
        manual_batch_size: Some(32),
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let p = plan(&config).expect("valid");
    assert_eq!(p.hyperparameters.strategy, "manual");
    assert!(p.issues.iter().any(|i| i.category == "Hyperparameters"));
    assert!(p.hyperparameters.recommendation.is_some());
}

#[test]
fn test_plan_tpe_strategy_generates_previews() {
    let dir = tempfile::tempdir().expect("valid");
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    for i in 0..100 {
        lines.push(format!(r#"{{"input": "echo test {i}", "label": {}}}"#, i % 5));
    }
    std::fs::write(&data_path, lines.join("\n")).expect("valid");

    let config = PlanConfig {
        task: "classify".to_string(),
        data_path,
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 5,
        output_dir: dir.path().to_path_buf(),
        strategy: "tpe".to_string(),
        budget: 20,
        scout: true,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let p = plan(&config).expect("valid");
    assert_eq!(p.hyperparameters.strategy, "tpe");
    assert_eq!(p.hyperparameters.budget, 20);
    assert!(p.hyperparameters.scout);
    assert!(!p.hyperparameters.sample_configs.is_empty());
    assert_eq!(p.hyperparameters.search_space_params, 9);
}

#[test]
fn test_plan_detects_imbalance() {
    let dir = tempfile::tempdir().expect("valid");
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    // 80 class 0, 10 each for classes 1-4 = 8:1 imbalance
    for i in 0..80 {
        lines.push(format!(r#"{{"input": "safe command {i}", "label": 0}}"#));
    }
    for c in 1..5 {
        for i in 0..10 {
            lines.push(format!(r#"{{"input": "class {c} cmd {i}", "label": {c}}}"#));
        }
    }
    std::fs::write(&data_path, lines.join("\n")).expect("valid");

    let config = PlanConfig {
        task: "classify".to_string(),
        data_path,
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 5,
        output_dir: dir.path().to_path_buf(),
        strategy: "tpe".to_string(),
        budget: 10,
        scout: true,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let p = plan(&config).expect("valid");
    assert!(p.data.imbalance_ratio > 5.0);
    assert!(p.data.auto_class_weights);
    assert!(p.issues.iter().any(|i| i.message.contains("imbalance")));
}

#[test]
fn test_plan_detects_duplicates() {
    let dir = tempfile::tempdir().expect("valid");
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    for i in 0..50 {
        lines.push(format!(r#"{{"input": "echo test {i}", "label": {}}}"#, i % 5));
    }
    // Add 5 exact duplicates
    for _ in 0..5 {
        lines.push(r#"{"input": "echo test 0", "label": 0}"#.to_string());
    }
    std::fs::write(&data_path, lines.join("\n")).expect("valid");

    let config = PlanConfig {
        task: "classify".to_string(),
        data_path,
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 5,
        output_dir: dir.path().to_path_buf(),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: Some(1e-4),
        manual_lora_rank: Some(16),
        manual_batch_size: Some(32),
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let p = plan(&config).expect("valid");
    assert!(p.data.duplicates > 0);
    assert!(p.issues.iter().any(|i| i.message.contains("duplicate")));
}

#[test]
fn test_plan_serialization_roundtrip() {
    let dir = tempfile::tempdir().expect("valid");
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    for i in 0..20 {
        lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 5));
    }
    std::fs::write(&data_path, lines.join("\n")).expect("valid");

    let config = PlanConfig {
        task: "classify".to_string(),
        data_path,
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 5,
        output_dir: dir.path().to_path_buf(),
        strategy: "tpe".to_string(),
        budget: 5,
        scout: true,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let p = plan(&config).expect("valid");

    // JSON roundtrip
    let json = p.to_json();
    let deserialized: TrainingPlan = serde_json::from_str(&json).expect("valid");
    assert_eq!(deserialized.version, "1.0");
    assert_eq!(deserialized.task, "classify");
    assert_eq!(deserialized.data.train_samples, p.data.train_samples);

    // YAML roundtrip
    let yaml = p.to_yaml();
    let deserialized_yaml: TrainingPlan = serde_yaml::from_str(&yaml).expect("valid");
    assert_eq!(deserialized_yaml.data.train_samples, p.data.train_samples);
}

#[test]
fn test_plan_resource_estimation() {
    let dir = tempfile::tempdir().expect("valid");
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    for i in 0..100 {
        lines.push(format!(r#"{{"input": "echo test {i}", "label": {}}}"#, i % 5));
    }
    std::fs::write(&data_path, lines.join("\n")).expect("valid");

    let config = PlanConfig {
        task: "classify".to_string(),
        data_path,
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 5,
        output_dir: dir.path().to_path_buf(),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: Some(1e-4),
        manual_lora_rank: Some(16),
        manual_batch_size: Some(32),
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let p = plan(&config).expect("valid");
    assert!(p.resources.estimated_vram_gb > 0.0);
    assert!(p.resources.steps_per_epoch > 0);
    assert!(p.resources.estimated_checkpoint_mb > 0.0);
}

#[test]
fn test_plan_verdict_ready() {
    let dir = tempfile::tempdir().expect("valid");
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    for i in 0..100 {
        lines.push(format!(r#"{{"input": "echo test {i}", "label": {}}}"#, i % 5));
    }
    std::fs::write(&data_path, lines.join("\n")).expect("valid");

    let config = PlanConfig {
        task: "classify".to_string(),
        data_path,
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 5,
        output_dir: dir.path().to_path_buf(),
        strategy: "tpe".to_string(),
        budget: 20,
        scout: true,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let p = plan(&config).expect("valid");
    // Should be WarningsPresent (model_path not specified)
    assert_ne!(p.verdict, PlanVerdict::Blocked);
}

#[test]
fn test_plan_model_info_qwen2() {
    let dir = tempfile::tempdir().expect("valid");
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    for i in 0..20 {
        lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 5));
    }
    std::fs::write(&data_path, lines.join("\n")).expect("valid");

    let config = PlanConfig {
        task: "classify".to_string(),
        data_path,
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 5,
        output_dir: dir.path().to_path_buf(),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: Some(1e-4),
        manual_lora_rank: Some(16),
        manual_batch_size: Some(32),
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let p = plan(&config).expect("valid");
    assert_eq!(p.model.hidden_size, 896);
    assert_eq!(p.model.num_layers, 24);
    assert_eq!(p.model.architecture, "qwen2");
    assert!(p.model.lora_trainable_params > 0);
    assert!(p.model.classifier_params > 0);
}

#[test]
fn test_execute_plan_rejects_blocked() {
    let blocked_plan = TrainingPlan {
        version: "1.0".to_string(),
        task: "classify".to_string(),
        data: DataAudit {
            train_path: "/tmp/nonexistent.jsonl".to_string(),
            train_samples: 0,
            avg_input_len: 0,
            class_counts: vec![0; 5],
            imbalance_ratio: 1.0,
            auto_class_weights: false,
            val_samples: None,
            test_samples: None,
            duplicates: 0,
            preamble_count: 0,
        },
        model: ModelInfo {
            size: "0.5B".to_string(),
            hidden_size: 896,
            num_layers: 24,
            architecture: "qwen2".to_string(),
            weights_available: false,
            lora_trainable_params: 0,
            classifier_params: 0,
        },
        hyperparameters: HyperparameterPlan {
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            search_space_params: 0,
            sample_configs: Vec::new(),
            manual: None,
            recommendation: None,
        },
        resources: ResourceEstimate {
            estimated_vram_gb: 0.0,
            estimated_minutes_per_epoch: 0.0,
            estimated_total_minutes: 0.0,
            estimated_checkpoint_mb: 0.0,
            steps_per_epoch: 0,
            gpu_device: None,
        },
        pre_flight: vec![PreFlightCheck {
            name: "data_file".to_string(),
            status: CheckStatus::Fail,
            detail: "Data not found".to_string(),
        }],
        output_dir: "/tmp/test".to_string(),
        auto_diagnose: false,
        verdict: PlanVerdict::Blocked,
        issues: Vec::new(),
    };

    let apply = ApplyConfig {
        model_path: PathBuf::from("/tmp/nonexistent"),
        data_path: PathBuf::from("/tmp/nonexistent.jsonl"),
        output_dir: PathBuf::from("/tmp/test-apply"),
        on_trial_complete: None,
    };

    let result = execute_plan(&blocked_plan, &apply);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("blocked"), "Error should mention blocked: {err_msg}");
}

#[test]
fn test_resolve_class_weights_uniform() {
    let weights = resolve_class_weights("uniform", &[100, 200, 300], 3);
    assert!(weights.is_none());
}

#[test]
fn test_resolve_class_weights_sqrt_inverse() {
    let weights = resolve_class_weights("sqrt_inverse", &[100, 200, 300], 3);
    assert!(weights.is_some());
    let w = weights.expect("valid");
    assert_eq!(w.len(), 3);
    // Largest class should have smallest weight
    assert!(
        w[0] > w[2],
        "class 0 (100 samples) should have higher weight than class 2 (300 samples)"
    );
}

#[test]
fn test_resolve_class_weights_inverse_freq() {
    let weights = resolve_class_weights("inverse_freq", &[100, 200, 300], 3);
    assert!(weights.is_some());
    let w = weights.expect("valid");
    assert_eq!(w.len(), 3);
    assert!(w[0] > w[2]);
}

#[test]
fn test_resolve_class_weights_unknown() {
    let weights = resolve_class_weights("bogus", &[100, 200], 2);
    assert!(weights.is_none());
}

#[test]
fn test_execute_plan_rejects_missing_model_path() {
    let plan = TrainingPlan {
        version: "1.0".to_string(),
        task: "classify".to_string(),
        data: DataAudit {
            train_path: "/tmp/data.jsonl".to_string(),
            train_samples: 100,
            avg_input_len: 50,
            class_counts: vec![50, 50],
            imbalance_ratio: 1.0,
            auto_class_weights: false,
            val_samples: None,
            test_samples: None,
            duplicates: 0,
            preamble_count: 0,
        },
        model: ModelInfo {
            size: "0.5B".to_string(),
            hidden_size: 896,
            num_layers: 24,
            architecture: "qwen2".to_string(),
            weights_available: true,
            lora_trainable_params: 1000,
            classifier_params: 100,
        },
        hyperparameters: HyperparameterPlan {
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            search_space_params: 0,
            sample_configs: Vec::new(),
            manual: Some(ManualConfig {
                learning_rate: 1e-4,
                lora_rank: 16,
                batch_size: 32,
                lora_alpha: None,
                warmup_fraction: None,
                gradient_clip_norm: None,
                lr_min_ratio: None,
                class_weights: None,
                target_modules: None,
            }),
            recommendation: None,
        },
        resources: ResourceEstimate {
            estimated_vram_gb: 2.5,
            estimated_minutes_per_epoch: 1.0,
            estimated_total_minutes: 1.0,
            estimated_checkpoint_mb: 1.0,
            steps_per_epoch: 4,
            gpu_device: None,
        },
        pre_flight: Vec::new(),
        output_dir: "/tmp/test".to_string(),
        auto_diagnose: false,
        verdict: PlanVerdict::Ready,
        issues: Vec::new(),
    };

    let apply = ApplyConfig {
        model_path: PathBuf::from("/tmp/definitely-not-a-real-model-dir"),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        output_dir: PathBuf::from("/tmp/test-apply"),
        on_trial_complete: None,
    };

    let result = execute_plan(&plan, &apply);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Model path") || err_msg.contains("does not exist"),
        "Error should mention model path: {err_msg}"
    );
}

// ── resolve_trial_status tests ──────────────────────────────────────

#[test]
fn test_resolve_trial_status_completed() {
    assert_eq!(resolve_trial_status(false, false), "completed");
}

#[test]
fn test_resolve_trial_status_pruned() {
    assert_eq!(resolve_trial_status(true, false), "pruned");
}

#[test]
fn test_resolve_trial_status_stopped_early() {
    assert_eq!(resolve_trial_status(false, true), "stopped_early");
}

#[test]
fn test_resolve_trial_status_pruned_takes_priority() {
    // When both pruned and stopped_early, pruned wins
    assert_eq!(resolve_trial_status(true, true), "pruned");
}

// ── estimate_gpu_hours tests ────────────────────────────────────────

#[test]
fn test_estimate_gpu_hours_basic() {
    let hours = estimate_gpu_hours(128, 1, 1);
    // 128 samples / 64 batch = 2 steps, 2 * 58 = 116 seconds / 3600 ~ 0.032 hours
    assert!(hours > 0.0);
    assert!(hours < 1.0);
}

#[test]
fn test_estimate_gpu_hours_scales_with_budget() {
    let h1 = estimate_gpu_hours(100, 1, 1);
    let h10 = estimate_gpu_hours(100, 1, 10);
    assert!((h10 - h1 * 10.0).abs() < 1e-6, "Budget should scale linearly");
}

#[test]
fn test_estimate_gpu_hours_scales_with_epochs() {
    let h1 = estimate_gpu_hours(100, 1, 1);
    let h5 = estimate_gpu_hours(100, 5, 1);
    assert!((h5 - h1 * 5.0).abs() < 1e-6, "Epochs should scale linearly");
}

#[test]
fn test_estimate_gpu_hours_zero_budget() {
    let hours = estimate_gpu_hours(100, 5, 0);
    assert!((hours).abs() < 1e-10);
}

// ── resolve_model tests ─────────────────────────────────────────────

#[test]
fn test_resolve_model_qwen2_05b() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 5,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    assert_eq!(model.hidden_size, 896);
    assert_eq!(model.num_layers, 24);
    assert_eq!(model.architecture, "qwen2");
    assert!(!model.weights_available);
}

#[test]
fn test_resolve_model_500m_alias() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "500M".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    assert_eq!(model.hidden_size, 896);
    assert_eq!(model.architecture, "qwen2");
}

#[test]
fn test_resolve_model_9b() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "9B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    assert_eq!(model.hidden_size, 4096);
    assert_eq!(model.num_layers, 48);
    assert_eq!(model.architecture, "qwen3.5");
}

#[test]
fn test_resolve_model_7b() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "7B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    assert_eq!(model.hidden_size, 4096);
    assert_eq!(model.num_layers, 32);
    assert_eq!(model.architecture, "llama2");
}

#[test]
fn test_resolve_model_13b() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "13B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    assert_eq!(model.hidden_size, 5120);
    assert_eq!(model.num_layers, 40);
    assert_eq!(model.architecture, "llama2");
}

#[test]
fn test_resolve_model_unknown_size() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "99B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    assert_eq!(model.architecture, "unknown");
    assert_eq!(model.hidden_size, 896); // defaults to smallest
}

#[test]
fn test_resolve_model_with_model_path_dir() {
    // model_path pointing to an existing directory
    let dir = tempfile::tempdir().unwrap();
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: Some(dir.path().to_path_buf()),
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    assert!(model.weights_available);
    // Should warn about missing files (no model.safetensors, no tokenizer.json)
    assert!(pf.iter().any(|c| c.name == "model_weights" && c.status == CheckStatus::Warn));
}

#[test]
fn test_resolve_model_with_nonexistent_path() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: Some(PathBuf::from("/nonexistent/model/dir")),
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    assert!(!model.weights_available);
    assert!(pf.iter().any(|c| c.name == "model_weights" && c.status == CheckStatus::Fail));
}

#[test]
fn test_resolve_model_with_complete_model_dir() {
    // model_path with model.safetensors and tokenizer.json
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: Some(dir.path().to_path_buf()),
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    assert!(model.weights_available);
    assert!(pf.iter().any(|c| c.name == "model_weights" && c.status == CheckStatus::Pass));
}

#[test]
fn test_resolve_model_lora_params_calculation() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 5,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: Some(8),
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    // lora_trainable = 2 * rank * hidden * 2 * layers = 2 * 8 * 896 * 2 * 24
    assert_eq!(model.lora_trainable_params, 2 * 8 * 896 * 2 * 24);
    // classifier_params = hidden * num_classes + num_classes = 896 * 5 + 5
    assert_eq!(model.classifier_params, 896 * 5 + 5);
}

// ── estimate_resources tests ────────────────────────────────────────

#[test]
fn test_estimate_resources_05b() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 5,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 3,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let model = ModelInfo {
        size: "0.5B".to_string(),
        hidden_size: 896,
        num_layers: 24,
        architecture: "qwen2".to_string(),
        weights_available: false,
        lora_trainable_params: 100_000,
        classifier_params: 4485,
    };
    let data = DataAudit {
        train_path: "/tmp/data.jsonl".to_string(),
        train_samples: 1000,
        avg_input_len: 50,
        class_counts: vec![500, 500],
        imbalance_ratio: 1.0,
        auto_class_weights: false,
        val_samples: None,
        test_samples: None,
        duplicates: 0,
        preamble_count: 0,
    };
    let res = estimate_resources(&config, &model, &data, 64);
    assert!((res.estimated_vram_gb - 2.5).abs() < 0.01);
    assert!(res.steps_per_epoch > 0);
    assert!(res.estimated_minutes_per_epoch > 0.0);
    assert!(res.estimated_total_minutes > 0.0);
    assert!(res.estimated_checkpoint_mb > 0.0);
}

#[test]
fn test_estimate_resources_7b() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "7B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let model = ModelInfo {
        size: "7B".to_string(),
        hidden_size: 4096,
        num_layers: 32,
        architecture: "llama2".to_string(),
        weights_available: false,
        lora_trainable_params: 1_000_000,
        classifier_params: 8194,
    };
    let data = DataAudit {
        train_path: "/tmp/data.jsonl".to_string(),
        train_samples: 100,
        avg_input_len: 50,
        class_counts: vec![50, 50],
        imbalance_ratio: 1.0,
        auto_class_weights: false,
        val_samples: None,
        test_samples: None,
        duplicates: 0,
        preamble_count: 0,
    };
    let res = estimate_resources(&config, &model, &data, 32);
    assert!((res.estimated_vram_gb - 18.0).abs() < 0.01);
}

#[test]
fn test_estimate_resources_13b() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "13B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let model = ModelInfo {
        size: "13B".to_string(),
        hidden_size: 5120,
        num_layers: 40,
        architecture: "llama2".to_string(),
        weights_available: false,
        lora_trainable_params: 2_000_000,
        classifier_params: 10242,
    };
    let data = DataAudit {
        train_path: "/tmp/data.jsonl".to_string(),
        train_samples: 100,
        avg_input_len: 50,
        class_counts: vec![50, 50],
        imbalance_ratio: 1.0,
        auto_class_weights: false,
        val_samples: None,
        test_samples: None,
        duplicates: 0,
        preamble_count: 0,
    };
    let res = estimate_resources(&config, &model, &data, 32);
    assert!((res.estimated_vram_gb - 26.0).abs() < 0.01);
}

#[test]
fn test_estimate_resources_unknown_hidden_size() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "custom".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let model = ModelInfo {
        size: "custom".to_string(),
        hidden_size: 2048,
        num_layers: 16,
        architecture: "custom".to_string(),
        weights_available: false,
        lora_trainable_params: 500_000,
        classifier_params: 4098,
    };
    let data = DataAudit {
        train_path: "/tmp/data.jsonl".to_string(),
        train_samples: 100,
        avg_input_len: 50,
        class_counts: vec![50, 50],
        imbalance_ratio: 1.0,
        auto_class_weights: false,
        val_samples: None,
        test_samples: None,
        duplicates: 0,
        preamble_count: 0,
    };
    let res = estimate_resources(&config, &model, &data, 32);
    // Unknown hidden size falls through to default 3.0 GB
    assert!((res.estimated_vram_gb - 3.0).abs() < 0.01);
}

#[test]
fn test_estimate_resources_scout_mode() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "tpe".to_string(),
        budget: 10,
        scout: true,
        max_epochs: 5,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let model = ModelInfo {
        size: "0.5B".to_string(),
        hidden_size: 896,
        num_layers: 24,
        architecture: "qwen2".to_string(),
        weights_available: false,
        lora_trainable_params: 100_000,
        classifier_params: 1794,
    };
    let data = DataAudit {
        train_path: "/tmp/data.jsonl".to_string(),
        train_samples: 200,
        avg_input_len: 50,
        class_counts: vec![100, 100],
        imbalance_ratio: 1.0,
        auto_class_weights: false,
        val_samples: None,
        test_samples: None,
        duplicates: 0,
        preamble_count: 0,
    };
    let res = estimate_resources(&config, &model, &data, 64);
    // Scout: total_epochs = 1, total_trials = 10
    // Non-scout with same config would be 5 * 10 = 50
    let non_scout_config = PlanConfig { scout: false, ..config.clone() };
    let res_full = estimate_resources(&non_scout_config, &model, &data, 64);
    assert!(res.estimated_total_minutes < res_full.estimated_total_minutes);
}

// ── count_file_samples tests ────────────────────────────────────────

#[test]
fn test_count_file_samples_none() {
    assert!(count_file_samples(None, 2).is_none());
}

#[test]
fn test_count_file_samples_nonexistent() {
    let p = PathBuf::from("/nonexistent/file.jsonl");
    assert!(count_file_samples(Some(&p), 2).is_none());
}

#[test]
fn test_count_file_samples_valid() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("val.jsonl");
    let mut lines = Vec::new();
    for i in 0..10 {
        lines.push(format!(r#"{{"input": "test {i}", "label": {}}}"#, i % 2));
    }
    std::fs::write(&path, lines.join("\n")).unwrap();
    let count = count_file_samples(Some(&path), 2);
    assert_eq!(count, Some(10));
}

// ── build_hpo_plan tests ────────────────────────────────────────────

#[test]
fn test_build_hpo_plan_manual() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 5,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 5,
        manual_lr: Some(2e-5),
        manual_lora_rank: Some(8),
        manual_batch_size: Some(16),
        manual_lora_alpha: Some(16.0),
        manual_warmup: Some(0.05),
        manual_gradient_clip: Some(0.5),
        manual_lr_min_ratio: Some(0.001),
        manual_class_weights: Some("sqrt_inverse".to_string()),
        manual_target_modules: Some("qkv".to_string()),
    };
    let mut issues = Vec::new();
    let hpo = build_hpo_plan(&config, 1000, &mut issues);
    assert_eq!(hpo.strategy, "manual");
    assert_eq!(hpo.budget, 0);
    assert!(!hpo.scout);
    assert_eq!(hpo.max_epochs, 5);
    assert_eq!(hpo.search_space_params, 0);
    assert!(hpo.sample_configs.is_empty());
    let manual = hpo.manual.unwrap();
    assert!((manual.learning_rate - 2e-5).abs() < 1e-10);
    assert_eq!(manual.lora_rank, 8);
    assert_eq!(manual.batch_size, 16);
    assert_eq!(manual.lora_alpha, Some(16.0));
    assert_eq!(manual.warmup_fraction, Some(0.05));
    assert_eq!(manual.gradient_clip_norm, Some(0.5));
    assert_eq!(manual.lr_min_ratio, Some(0.001));
    assert_eq!(manual.class_weights.as_deref(), Some("sqrt_inverse"));
    assert_eq!(manual.target_modules.as_deref(), Some("qkv"));
    // Should have a warning about manual mode
    assert!(issues.iter().any(|i| i.category == "Hyperparameters"));
    assert!(hpo.recommendation.is_some());
}

#[test]
fn test_build_hpo_plan_manual_defaults() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let mut issues = Vec::new();
    let hpo = build_hpo_plan(&config, 100, &mut issues);
    let manual = hpo.manual.unwrap();
    // Defaults: lr=1e-4, rank=16, batch=32
    assert!((manual.learning_rate - 1e-4).abs() < 1e-10);
    assert_eq!(manual.lora_rank, 16);
    assert_eq!(manual.batch_size, 32);
}

#[test]
fn test_build_hpo_plan_tpe() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 5,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "tpe".to_string(),
        budget: 20,
        scout: true,
        max_epochs: 5,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let mut issues = Vec::new();
    let hpo = build_hpo_plan(&config, 1000, &mut issues);
    assert_eq!(hpo.strategy, "tpe");
    assert_eq!(hpo.budget, 20);
    assert!(hpo.scout);
    assert_eq!(hpo.max_epochs, 1); // scout mode forces 1 epoch
    assert_eq!(hpo.search_space_params, 9);
    assert!(!hpo.sample_configs.is_empty());
    assert!(hpo.manual.is_none());
}

#[test]
fn test_build_hpo_plan_grid() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "grid".to_string(),
        budget: 10,
        scout: false,
        max_epochs: 3,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let mut issues = Vec::new();
    let hpo = build_hpo_plan(&config, 100, &mut issues);
    assert_eq!(hpo.strategy, "grid");
    assert_eq!(hpo.max_epochs, 3);
}

#[test]
fn test_build_hpo_plan_random() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "random".to_string(),
        budget: 5,
        scout: false,
        max_epochs: 2,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let mut issues = Vec::new();
    let hpo = build_hpo_plan(&config, 100, &mut issues);
    assert_eq!(hpo.strategy, "random");
}

#[test]
fn test_build_hpo_plan_low_budget_warning() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "tpe".to_string(),
        budget: 3,
        scout: false,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let mut issues = Vec::new();
    build_hpo_plan(&config, 100, &mut issues);
    assert!(issues.iter().any(|i| i.message.contains("TPE budget") && i.message.contains("low")));
}

#[test]
fn test_build_hpo_plan_large_dataset_scout_warning() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "tpe".to_string(),
        budget: 20,
        scout: false,
        max_epochs: 5,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let mut issues = Vec::new();
    build_hpo_plan(&config, 50_000, &mut issues);
    // Should warn about GPU hours for large dataset without scout
    assert!(issues.iter().any(|i| i.message.contains("GPU hours")));
}

// ── detect_gpu_device tests ─────────────────────────────────────────

#[test]
fn test_detect_gpu_device_returns_option() {
    // Just verify it doesn't panic; result depends on hardware
    let _gpu = detect_gpu_device();
}

// ── TrainingPlan from_str tests ─────────────────────────────────────

#[test]
fn test_training_plan_from_str_invalid() {
    let result = TrainingPlan::from_str("not valid json or yaml {{{{");
    assert!(result.is_err());
}

#[test]
fn test_training_plan_from_str_json() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    for i in 0..20 {
        lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
    }
    std::fs::write(&data_path, lines.join("\n")).unwrap();
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path,
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: dir.path().to_path_buf(),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: Some(1e-4),
        manual_lora_rank: Some(8),
        manual_batch_size: Some(32),
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let p = plan(&config).unwrap();
    let json = p.to_json();
    let parsed = TrainingPlan::from_str(&json).unwrap();
    assert_eq!(parsed.task, "classify");
}

// ── ExperimentTracker tests ─────────────────────────────────────────

#[test]
fn test_experiment_tracker_open() {
    let dir = tempfile::tempdir().unwrap();
    let plan_data = DataAudit {
        train_path: "/tmp/data.jsonl".to_string(),
        train_samples: 100,
        avg_input_len: 50,
        class_counts: vec![50, 50],
        imbalance_ratio: 1.0,
        auto_class_weights: false,
        val_samples: None,
        test_samples: None,
        duplicates: 0,
        preamble_count: 0,
    };
    let test_plan = TrainingPlan {
        version: "1.0".to_string(),
        task: "classify".to_string(),
        data: plan_data,
        model: ModelInfo {
            size: "0.5B".to_string(),
            hidden_size: 896,
            num_layers: 24,
            architecture: "qwen2".to_string(),
            weights_available: false,
            lora_trainable_params: 100_000,
            classifier_params: 1794,
        },
        hyperparameters: HyperparameterPlan {
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            search_space_params: 0,
            sample_configs: Vec::new(),
            manual: None,
            recommendation: None,
        },
        resources: ResourceEstimate {
            estimated_vram_gb: 2.5,
            estimated_minutes_per_epoch: 1.0,
            estimated_total_minutes: 1.0,
            estimated_checkpoint_mb: 1.0,
            steps_per_epoch: 4,
            gpu_device: None,
        },
        pre_flight: Vec::new(),
        output_dir: dir.path().display().to_string(),
        auto_diagnose: false,
        verdict: PlanVerdict::Ready,
        issues: Vec::new(),
    };
    let tracker = ExperimentTracker::open(dir.path(), &test_plan);
    // Should either succeed or fail gracefully — no panic
    drop(tracker);
}

#[test]
fn test_experiment_tracker_log_failed_trial_no_store() {
    let mut tracker = ExperimentTracker { store: None, exp_id: None };
    // Should be a no-op, not panic
    tracker.log_failed_trial();
}

// ── PlanConfig serialization tests ──────────────────────────────────

#[test]
fn test_plan_config_serde_roundtrip() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: Some(PathBuf::from("/tmp/val.jsonl")),
        test_path: Some(PathBuf::from("/tmp/test.jsonl")),
        model_size: "0.5B".to_string(),
        model_path: Some(PathBuf::from("/tmp/model")),
        num_classes: 5,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "tpe".to_string(),
        budget: 20,
        scout: true,
        max_epochs: 10,
        manual_lr: Some(1e-4),
        manual_lora_rank: Some(16),
        manual_batch_size: Some(32),
        manual_lora_alpha: Some(32.0),
        manual_warmup: Some(0.1),
        manual_gradient_clip: Some(1.0),
        manual_lr_min_ratio: Some(0.01),
        manual_class_weights: Some("sqrt_inverse".to_string()),
        manual_target_modules: Some("qkv".to_string()),
    };
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: PlanConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.task, "classify");
    assert_eq!(deserialized.budget, 20);
    assert_eq!(deserialized.manual_lr, Some(1e-4));
    assert!(deserialized.val_path.is_some());
    assert!(deserialized.test_path.is_some());
}

// ── ManualConfig serde defaults test ────────────────────────────────

#[test]
fn test_manual_config_serde_defaults() {
    let json = r#"{"learning_rate": 0.001, "lora_rank": 8, "batch_size": 32}"#;
    let mc: ManualConfig = serde_json::from_str(json).unwrap();
    assert!((mc.learning_rate - 0.001).abs() < 1e-6);
    assert_eq!(mc.lora_rank, 8);
    assert_eq!(mc.batch_size, 32);
    // All optional fields should be None (serde default)
    assert!(mc.lora_alpha.is_none());
    assert!(mc.warmup_fraction.is_none());
    assert!(mc.gradient_clip_norm.is_none());
    assert!(mc.lr_min_ratio.is_none());
    assert!(mc.class_weights.is_none());
    assert!(mc.target_modules.is_none());
}

#[test]
fn test_manual_config_serde_all_fields() {
    let mc = ManualConfig {
        learning_rate: 5e-5,
        lora_rank: 4,
        batch_size: 64,
        lora_alpha: Some(8.0),
        warmup_fraction: Some(0.05),
        gradient_clip_norm: Some(0.5),
        lr_min_ratio: Some(0.001),
        class_weights: Some("inverse_freq".to_string()),
        target_modules: Some("all_linear".to_string()),
    };
    let json = serde_json::to_string(&mc).unwrap();
    let deserialized: ManualConfig = serde_json::from_str(&json).unwrap();
    assert!((deserialized.learning_rate - 5e-5).abs() < 1e-10);
    assert_eq!(deserialized.lora_alpha, Some(8.0));
    assert_eq!(deserialized.class_weights.as_deref(), Some("inverse_freq"));
}

// ── PlanIssue tests ─────────────────────────────────────────────────

#[test]
fn test_plan_issue_serde() {
    let issue = PlanIssue {
        severity: CheckStatus::Warn,
        category: "Data".to_string(),
        message: "test issue".to_string(),
        fix: Some("do this".to_string()),
    };
    let json = serde_json::to_string(&issue).unwrap();
    let deserialized: PlanIssue = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.severity, CheckStatus::Warn);
    assert_eq!(deserialized.category, "Data");
    assert_eq!(deserialized.fix.as_deref(), Some("do this"));
}

#[test]
fn test_plan_issue_no_fix() {
    let issue = PlanIssue {
        severity: CheckStatus::Fail,
        category: "Model".to_string(),
        message: "error".to_string(),
        fix: None,
    };
    let json = serde_json::to_string(&issue).unwrap();
    let deserialized: PlanIssue = serde_json::from_str(&json).unwrap();
    assert!(deserialized.fix.is_none());
}

// ── TrialPreview serde tests ────────────────────────────────────────

#[test]
fn test_trial_preview_serde() {
    let tp = TrialPreview {
        trial: 1,
        learning_rate: 1e-4,
        lora_rank: 16,
        lora_alpha: 32.0,
        batch_size: 64,
        warmup: 0.1,
        gradient_clip: 1.0,
        class_weights: "sqrt_inverse".to_string(),
        target_modules: "qv".to_string(),
        lr_min_ratio: 0.01,
    };
    let json = serde_json::to_string(&tp).unwrap();
    let deserialized: TrialPreview = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.trial, 1);
    assert_eq!(deserialized.lora_rank, 16);
    assert!((deserialized.lora_alpha - 32.0).abs() < 1e-6);
}

// ── execute_plan missing data test ──────────────────────────────────

#[test]
fn test_execute_plan_rejects_missing_data() {
    let plan = TrainingPlan {
        version: "1.0".to_string(),
        task: "classify".to_string(),
        data: DataAudit {
            train_path: "/tmp/data.jsonl".to_string(),
            train_samples: 100,
            avg_input_len: 50,
            class_counts: vec![50, 50],
            imbalance_ratio: 1.0,
            auto_class_weights: false,
            val_samples: None,
            test_samples: None,
            duplicates: 0,
            preamble_count: 0,
        },
        model: ModelInfo {
            size: "0.5B".to_string(),
            hidden_size: 896,
            num_layers: 24,
            architecture: "qwen2".to_string(),
            weights_available: true,
            lora_trainable_params: 1000,
            classifier_params: 100,
        },
        hyperparameters: HyperparameterPlan {
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            search_space_params: 0,
            sample_configs: Vec::new(),
            manual: Some(ManualConfig {
                learning_rate: 1e-4,
                lora_rank: 16,
                batch_size: 32,
                lora_alpha: None,
                warmup_fraction: None,
                gradient_clip_norm: None,
                lr_min_ratio: None,
                class_weights: None,
                target_modules: None,
            }),
            recommendation: None,
        },
        resources: ResourceEstimate {
            estimated_vram_gb: 2.5,
            estimated_minutes_per_epoch: 1.0,
            estimated_total_minutes: 1.0,
            estimated_checkpoint_mb: 1.0,
            steps_per_epoch: 4,
            gpu_device: None,
        },
        pre_flight: Vec::new(),
        output_dir: "/tmp/test".to_string(),
        auto_diagnose: false,
        verdict: PlanVerdict::Ready,
        issues: Vec::new(),
    };
    // Model path is a directory (use tempdir) but data doesn't exist
    let dir = tempfile::tempdir().unwrap();
    let apply = ApplyConfig {
        model_path: dir.path().to_path_buf(),
        data_path: PathBuf::from("/tmp/nonexistent_data_file.jsonl"),
        output_dir: PathBuf::from("/tmp/test-apply-missing-data"),
        on_trial_complete: None,
    };
    let result = execute_plan(&plan, &apply);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("not found") || err_msg.contains("data"),
        "Error should mention data: {err_msg}"
    );
}

// ── Plan with preamble detection ────────────────────────────────────

#[test]
fn test_plan_detects_preambles() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    // Many entries with shell preamble (> 10% of total)
    for i in 0..20 {
        lines.push(format!("{{\"input\": \"#!/bin/bash\\necho {i}\", \"label\": {}}}", i % 2));
    }
    for i in 0..5 {
        lines.push(format!(r#"{{"input": "echo clean {i}", "label": {}}}"#, i % 2));
    }
    std::fs::write(&data_path, lines.join("\n")).unwrap();
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path,
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: dir.path().to_path_buf(),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: Some(1e-4),
        manual_lora_rank: Some(8),
        manual_batch_size: Some(32),
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let p = plan(&config).unwrap();
    assert!(p.data.preamble_count > 0);
    assert!(p.issues.iter().any(|i| i.message.contains("preamble")));
}

// ── Plan with small dataset warning ─────────────────────────────────

#[test]
fn test_plan_small_dataset_warning() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    for i in 0..30 {
        lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
    }
    std::fs::write(&data_path, lines.join("\n")).unwrap();
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path,
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: dir.path().to_path_buf(),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: Some(1e-4),
        manual_lora_rank: Some(8),
        manual_batch_size: Some(32),
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let p = plan(&config).unwrap();
    // < 100 samples should generate a warning
    assert!(p.issues.iter().any(|i| i.message.contains("insufficient")));
}

// ── Plan output_dir with existing checkpoints ───────────────────────

#[test]
fn test_plan_output_dir_existing_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    for i in 0..20 {
        lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
    }
    std::fs::write(&data_path, lines.join("\n")).unwrap();
    // Create a metadata.json in output dir to simulate existing checkpoints
    let output_dir = dir.path().join("output");
    std::fs::create_dir_all(&output_dir).unwrap();
    std::fs::write(output_dir.join("metadata.json"), "{}").unwrap();
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path,
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir,
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: Some(1e-4),
        manual_lora_rank: Some(8),
        manual_batch_size: Some(32),
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let p = plan(&config).unwrap();
    assert!(p.pre_flight.iter().any(|c| c.name == "output_dir" && c.status == CheckStatus::Warn));
    // Output dir warning is surfaced in pre_flight; issues may or may not reference it
    assert!(!p.pre_flight.is_empty());
}

// ── Plan verdict logic tests ────────────────────────────────────────

#[test]
fn test_plan_verdict_blocked_on_empty_class() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    // Only classes 0 and 1, but num_classes = 3 (class 2 is empty)
    for i in 0..50 {
        lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
    }
    std::fs::write(&data_path, lines.join("\n")).unwrap();
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path,
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 3,
        output_dir: dir.path().to_path_buf(),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: Some(1e-4),
        manual_lora_rank: Some(8),
        manual_batch_size: Some(32),
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let p = plan(&config).unwrap();
    assert_eq!(p.verdict, PlanVerdict::Blocked);
    assert!(p
        .pre_flight
        .iter()
        .any(|c| c.name == "class_coverage" && c.status == CheckStatus::Fail));
}

// ── DataAudit serde test ────────────────────────────────────────────

#[test]
fn test_data_audit_serde() {
    let da = DataAudit {
        train_path: "/tmp/data.jsonl".to_string(),
        train_samples: 500,
        avg_input_len: 42,
        class_counts: vec![300, 200],
        imbalance_ratio: 1.5,
        auto_class_weights: false,
        val_samples: Some(50),
        test_samples: Some(25),
        duplicates: 3,
        preamble_count: 10,
    };
    let json = serde_json::to_string(&da).unwrap();
    let deserialized: DataAudit = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.train_samples, 500);
    assert_eq!(deserialized.val_samples, Some(50));
    assert_eq!(deserialized.test_samples, Some(25));
    assert_eq!(deserialized.duplicates, 3);
}

// ── ResourceEstimate serde test ─────────────────────────────────────

#[test]
fn test_resource_estimate_serde() {
    let re = ResourceEstimate {
        estimated_vram_gb: 6.5,
        estimated_minutes_per_epoch: 2.0,
        estimated_total_minutes: 100.0,
        estimated_checkpoint_mb: 50.0,
        steps_per_epoch: 32,
        gpu_device: Some("RTX 4090".to_string()),
    };
    let json = serde_json::to_string(&re).unwrap();
    let deserialized: ResourceEstimate = serde_json::from_str(&json).unwrap();
    assert!((deserialized.estimated_vram_gb - 6.5).abs() < 1e-6);
    assert_eq!(deserialized.gpu_device.as_deref(), Some("RTX 4090"));
}

// ── CheckStatus equality ────────────────────────────────────────────

#[test]
fn test_check_status_equality() {
    assert_eq!(CheckStatus::Pass, CheckStatus::Pass);
    assert_ne!(CheckStatus::Pass, CheckStatus::Warn);
    assert_ne!(CheckStatus::Warn, CheckStatus::Fail);
}

// ── PlanVerdict equality ────────────────────────────────────────────

#[test]
fn test_plan_verdict_equality() {
    assert_eq!(PlanVerdict::Ready, PlanVerdict::Ready);
    assert_ne!(PlanVerdict::Ready, PlanVerdict::Blocked);
    assert_ne!(PlanVerdict::WarningsPresent, PlanVerdict::Blocked);
}

// ── Plan with val_path and test_path ────────────────────────────────

#[test]
fn test_plan_with_val_and_test_paths() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("train.jsonl");
    let val_path = dir.path().join("val.jsonl");
    let test_path = dir.path().join("test.jsonl");
    let mut lines = Vec::new();
    for i in 0..50 {
        lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
    }
    std::fs::write(&data_path, lines.join("\n")).unwrap();
    let mut val_lines = Vec::new();
    for i in 0..10 {
        val_lines.push(format!(r#"{{"input": "val {i}", "label": {}}}"#, i % 2));
    }
    std::fs::write(&val_path, val_lines.join("\n")).unwrap();
    let mut test_lines = Vec::new();
    for i in 0..5 {
        test_lines.push(format!(r#"{{"input": "test {i}", "label": {}}}"#, i % 2));
    }
    std::fs::write(&test_path, test_lines.join("\n")).unwrap();
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path,
        val_path: Some(val_path),
        test_path: Some(test_path),
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: dir.path().to_path_buf(),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: Some(1e-4),
        manual_lora_rank: Some(8),
        manual_batch_size: Some(32),
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let p = plan(&config).unwrap();
    assert_eq!(p.data.val_samples, Some(10));
    assert_eq!(p.data.test_samples, Some(5));
}

// ── Plan with nonexistent val/test paths ────────────────────────────

#[test]
fn test_plan_with_nonexistent_val_test_paths() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    for i in 0..50 {
        lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
    }
    std::fs::write(&data_path, lines.join("\n")).unwrap();
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path,
        val_path: Some(PathBuf::from("/nonexistent/val.jsonl")),
        test_path: Some(PathBuf::from("/nonexistent/test.jsonl")),
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: dir.path().to_path_buf(),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: Some(1e-4),
        manual_lora_rank: Some(8),
        manual_batch_size: Some(32),
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let p = plan(&config).unwrap();
    // Nonexistent paths should result in None
    assert!(p.data.val_samples.is_none());
    assert!(p.data.test_samples.is_none());
}

// ── execute_plan manual without manual config ───────────────────────

#[test]
fn test_execute_plan_manual_no_manual_config() {
    let dir = tempfile::tempdir().unwrap();
    // Create a data file so data check passes
    let data_path = dir.path().join("data.jsonl");
    let mut lines = Vec::new();
    for i in 0..20 {
        lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
    }
    std::fs::write(&data_path, lines.join("\n")).unwrap();

    let plan = TrainingPlan {
        version: "1.0".to_string(),
        task: "classify".to_string(),
        data: DataAudit {
            train_path: data_path.display().to_string(),
            train_samples: 20,
            avg_input_len: 10,
            class_counts: vec![10, 10],
            imbalance_ratio: 1.0,
            auto_class_weights: false,
            val_samples: None,
            test_samples: None,
            duplicates: 0,
            preamble_count: 0,
        },
        model: ModelInfo {
            size: "0.5B".to_string(),
            hidden_size: 896,
            num_layers: 24,
            architecture: "qwen2".to_string(),
            weights_available: true,
            lora_trainable_params: 100_000,
            classifier_params: 1794,
        },
        hyperparameters: HyperparameterPlan {
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            search_space_params: 0,
            sample_configs: Vec::new(),
            manual: None, // No manual config!
            recommendation: None,
        },
        resources: ResourceEstimate {
            estimated_vram_gb: 2.5,
            estimated_minutes_per_epoch: 1.0,
            estimated_total_minutes: 1.0,
            estimated_checkpoint_mb: 1.0,
            steps_per_epoch: 1,
            gpu_device: None,
        },
        pre_flight: Vec::new(),
        output_dir: dir.path().display().to_string(),
        auto_diagnose: false,
        verdict: PlanVerdict::Ready,
        issues: Vec::new(),
    };
    let model_dir = dir.path().join("model");
    std::fs::create_dir_all(&model_dir).unwrap();
    let apply = ApplyConfig {
        model_path: model_dir,
        data_path,
        output_dir: dir.path().join("out"),
        on_trial_complete: None,
    };
    let result = execute_plan(&plan, &apply);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("manual") || err_msg.contains("Manual"),
        "Error should mention manual config: {err_msg}"
    );
}

// =========================================================================
// TrainingPlan::check_counts
// =========================================================================

#[test]
fn test_check_counts_empty() {
    let plan = TrainingPlan {
        version: "1.0".to_string(),
        task: "classify".to_string(),
        data: DataAudit {
            train_path: "/tmp/data.jsonl".to_string(),
            train_samples: 100,
            avg_input_len: 50,
            class_counts: vec![50, 50],
            imbalance_ratio: 1.0,
            auto_class_weights: false,
            val_samples: None,
            test_samples: None,
            duplicates: 0,
            preamble_count: 0,
        },
        model: ModelInfo {
            size: "0.5B".to_string(),
            hidden_size: 896,
            num_layers: 24,
            architecture: "qwen2".to_string(),
            weights_available: false,
            lora_trainable_params: 100_000,
            classifier_params: 1794,
        },
        hyperparameters: HyperparameterPlan {
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            search_space_params: 0,
            sample_configs: Vec::new(),
            manual: None,
            recommendation: None,
        },
        resources: ResourceEstimate {
            estimated_vram_gb: 2.5,
            estimated_minutes_per_epoch: 1.0,
            estimated_total_minutes: 1.0,
            estimated_checkpoint_mb: 1.0,
            steps_per_epoch: 4,
            gpu_device: None,
        },
        pre_flight: Vec::new(),
        output_dir: "/tmp/test".to_string(),
        auto_diagnose: false,
        verdict: PlanVerdict::Ready,
        issues: Vec::new(),
    };
    let (pass, warn, fail) = plan.check_counts();
    assert_eq!(pass, 0);
    assert_eq!(warn, 0);
    assert_eq!(fail, 0);
}

#[test]
fn test_check_counts_mixed() {
    let plan = TrainingPlan {
        version: "1.0".to_string(),
        task: "classify".to_string(),
        data: DataAudit {
            train_path: "/tmp/data.jsonl".to_string(),
            train_samples: 100,
            avg_input_len: 50,
            class_counts: vec![50, 50],
            imbalance_ratio: 1.0,
            auto_class_weights: false,
            val_samples: None,
            test_samples: None,
            duplicates: 0,
            preamble_count: 0,
        },
        model: ModelInfo {
            size: "0.5B".to_string(),
            hidden_size: 896,
            num_layers: 24,
            architecture: "qwen2".to_string(),
            weights_available: false,
            lora_trainable_params: 100_000,
            classifier_params: 1794,
        },
        hyperparameters: HyperparameterPlan {
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            search_space_params: 0,
            sample_configs: Vec::new(),
            manual: None,
            recommendation: None,
        },
        resources: ResourceEstimate {
            estimated_vram_gb: 2.5,
            estimated_minutes_per_epoch: 1.0,
            estimated_total_minutes: 1.0,
            estimated_checkpoint_mb: 1.0,
            steps_per_epoch: 4,
            gpu_device: None,
        },
        pre_flight: vec![
            PreFlightCheck {
                name: "data".to_string(),
                status: CheckStatus::Pass,
                detail: String::new(),
            },
            PreFlightCheck {
                name: "model".to_string(),
                status: CheckStatus::Warn,
                detail: "missing weights".to_string(),
            },
            PreFlightCheck {
                name: "output".to_string(),
                status: CheckStatus::Fail,
                detail: "no space".to_string(),
            },
            PreFlightCheck {
                name: "deps".to_string(),
                status: CheckStatus::Pass,
                detail: String::new(),
            },
        ],
        output_dir: "/tmp/test".to_string(),
        auto_diagnose: false,
        verdict: PlanVerdict::Blocked,
        issues: Vec::new(),
    };
    let (pass, warn, fail) = plan.check_counts();
    assert_eq!(pass, 2);
    assert_eq!(warn, 1);
    assert_eq!(fail, 1);
}

// =========================================================================
// TrainingPlan::from_str YAML
// =========================================================================

#[test]
fn test_training_plan_from_str_yaml() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    for i in 0..20 {
        lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
    }
    std::fs::write(&data_path, lines.join("\n")).unwrap();
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path,
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: dir.path().to_path_buf(),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: Some(1e-4),
        manual_lora_rank: Some(8),
        manual_batch_size: Some(32),
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let p = plan(&config).unwrap();
    let yaml = serde_yaml::to_string(&p).unwrap();
    let parsed = TrainingPlan::from_str(&yaml).unwrap();
    assert_eq!(parsed.task, "classify");
    assert_eq!(parsed.model.size, "0.5B");
}

// =========================================================================
// resolve_trial_status
// =========================================================================

#[test]
fn test_resolve_trial_status_is_pruned() {
    assert_eq!(resolve_trial_status(true, false), "pruned");
}

#[test]
fn test_resolve_trial_status_is_stopped_early() {
    assert_eq!(resolve_trial_status(false, true), "stopped_early");
}

#[test]
fn test_resolve_trial_status_is_completed() {
    assert_eq!(resolve_trial_status(false, false), "completed");
}

#[test]
fn test_resolve_trial_status_pruned_wins_over_stopped() {
    assert_eq!(resolve_trial_status(true, true), "pruned");
}

// =========================================================================
// ExperimentTracker with store
// =========================================================================

#[test]
fn test_experiment_tracker_log_failed_trial_with_store() {
    let dir = tempfile::tempdir().unwrap();
    let plan_data = DataAudit {
        train_path: "/tmp/data.jsonl".to_string(),
        train_samples: 100,
        avg_input_len: 50,
        class_counts: vec![50, 50],
        imbalance_ratio: 1.0,
        auto_class_weights: false,
        val_samples: None,
        test_samples: None,
        duplicates: 0,
        preamble_count: 0,
    };
    let test_plan = TrainingPlan {
        version: "1.0".to_string(),
        task: "classify".to_string(),
        data: plan_data,
        model: ModelInfo {
            size: "0.5B".to_string(),
            hidden_size: 896,
            num_layers: 24,
            architecture: "qwen2".to_string(),
            weights_available: false,
            lora_trainable_params: 100_000,
            classifier_params: 1794,
        },
        hyperparameters: HyperparameterPlan {
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            search_space_params: 0,
            sample_configs: Vec::new(),
            manual: None,
            recommendation: None,
        },
        resources: ResourceEstimate {
            estimated_vram_gb: 2.5,
            estimated_minutes_per_epoch: 1.0,
            estimated_total_minutes: 1.0,
            estimated_checkpoint_mb: 1.0,
            steps_per_epoch: 4,
            gpu_device: None,
        },
        pre_flight: Vec::new(),
        output_dir: dir.path().display().to_string(),
        auto_diagnose: false,
        verdict: PlanVerdict::Ready,
        issues: Vec::new(),
    };
    let mut tracker = ExperimentTracker::open(dir.path(), &test_plan);
    // Log a failed trial — should not panic even with a real store
    tracker.log_failed_trial();
}

// =========================================================================
// PreFlightCheck serde
// =========================================================================

#[test]
fn test_pre_flight_check_serde() {
    let check = PreFlightCheck {
        name: "data_exists".to_string(),
        status: CheckStatus::Pass,
        detail: "Found 1000 samples".to_string(),
    };
    let json = serde_json::to_string(&check).unwrap();
    let deserialized: PreFlightCheck = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.name, "data_exists");
    assert_eq!(deserialized.status, CheckStatus::Pass);
    assert_eq!(deserialized.detail.as_str(), "Found 1000 samples");
}

#[test]
fn test_pre_flight_check_no_detail() {
    let check = PreFlightCheck {
        name: "test".to_string(),
        status: CheckStatus::Fail,
        detail: String::new(),
    };
    let json = serde_json::to_string(&check).unwrap();
    let deserialized: PreFlightCheck = serde_json::from_str(&json).unwrap();
    assert!(deserialized.detail.is_empty());
}

// =========================================================================
// ModelInfo serde
// =========================================================================

#[test]
fn test_model_info_serde() {
    let mi = ModelInfo {
        size: "7B".to_string(),
        hidden_size: 4096,
        num_layers: 32,
        architecture: "llama2".to_string(),
        weights_available: true,
        lora_trainable_params: 1_000_000,
        classifier_params: 8194,
    };
    let json = serde_json::to_string(&mi).unwrap();
    let deserialized: ModelInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.size, "7B");
    assert_eq!(deserialized.hidden_size, 4096);
    assert_eq!(deserialized.num_layers, 32);
    assert!(deserialized.weights_available);
    assert_eq!(deserialized.lora_trainable_params, 1_000_000);
    assert_eq!(deserialized.classifier_params, 8194);
}

// =========================================================================
// HyperparameterPlan serde
// =========================================================================

#[test]
fn test_hyperparameter_plan_serde() {
    let hp = HyperparameterPlan {
        strategy: "tpe".to_string(),
        budget: 20,
        scout: true,
        max_epochs: 5,
        search_space_params: 9,
        sample_configs: vec![TrialPreview {
            trial: 1,
            learning_rate: 1e-4,
            lora_rank: 16,
            lora_alpha: 32.0,
            batch_size: 64,
            warmup: 0.1,
            gradient_clip: 1.0,
            class_weights: "sqrt_inverse".to_string(),
            target_modules: "qv".to_string(),
            lr_min_ratio: 0.01,
        }],
        manual: None,
        recommendation: Some("Use TPE for 20 trials".to_string()),
    };
    let json = serde_json::to_string(&hp).unwrap();
    let deserialized: HyperparameterPlan = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.strategy, "tpe");
    assert_eq!(deserialized.budget, 20);
    assert!(deserialized.scout);
    assert_eq!(deserialized.sample_configs.len(), 1);
    assert!(deserialized.manual.is_none());
    assert!(deserialized.recommendation.is_some());
}

// =========================================================================
// PlanVerdict serde
// =========================================================================

#[test]
fn test_plan_verdict_serde_roundtrip() {
    for verdict in &[PlanVerdict::Ready, PlanVerdict::WarningsPresent, PlanVerdict::Blocked] {
        let json = serde_json::to_string(verdict).unwrap();
        let deserialized: PlanVerdict = serde_json::from_str(&json).unwrap();
        assert_eq!(*verdict, deserialized);
    }
}

// =========================================================================
// Plan with duplicates in data
// =========================================================================

#[test]
fn test_plan_detects_data_duplicates() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    // Add duplicated entries
    for _ in 0..10 {
        lines.push(r#"{"input": "echo hello", "label": 0}"#.to_string());
    }
    for i in 0..20 {
        lines.push(format!(r#"{{"input": "echo unique_{i}", "label": {}}}"#, i % 2));
    }
    std::fs::write(&data_path, lines.join("\n")).unwrap();
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path,
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: dir.path().to_path_buf(),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: Some(1e-4),
        manual_lora_rank: Some(8),
        manual_batch_size: Some(32),
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let p = plan(&config).unwrap();
    assert!(p.data.duplicates > 0);
    // Should have a warning about duplicates
    assert!(p.issues.iter().any(|i| i.message.contains("duplicate")));
}

// =========================================================================
// estimate_resources with different batch sizes
// =========================================================================

#[test]
fn test_estimate_resources_large_batch_size() {
    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: "0.5B".to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    };
    let model = ModelInfo {
        size: "0.5B".to_string(),
        hidden_size: 896,
        num_layers: 24,
        architecture: "qwen2".to_string(),
        weights_available: false,
        lora_trainable_params: 100_000,
        classifier_params: 1794,
    };
    let data = DataAudit {
        train_path: "/tmp/data.jsonl".to_string(),
        train_samples: 10,
        avg_input_len: 50,
        class_counts: vec![5, 5],
        imbalance_ratio: 1.0,
        auto_class_weights: false,
        val_samples: None,
        test_samples: None,
        duplicates: 0,
        preamble_count: 0,
    };
    // Batch size larger than dataset — should still compute valid steps
    let res = estimate_resources(&config, &model, &data, 128);
    assert!(res.steps_per_epoch >= 1);
    assert!(res.estimated_total_minutes >= 0.0);
}

// =========================================================================
// test_cov2_* — Additional coverage tests
// =========================================================================

/// Helper to build a minimal PlanConfig for tests
fn mk_plan_config(model_size: &str, strategy: &str) -> PlanConfig {
    PlanConfig {
        task: "classify".to_string(),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        val_path: None,
        test_path: None,
        model_size: model_size.to_string(),
        model_path: None,
        num_classes: 2,
        output_dir: PathBuf::from("/tmp/out"),
        strategy: strategy.to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        manual_lr: None,
        manual_lora_rank: None,
        manual_batch_size: None,
        manual_lora_alpha: None,
        manual_warmup: None,
        manual_gradient_clip: None,
        manual_lr_min_ratio: None,
        manual_class_weights: None,
        manual_target_modules: None,
    }
}

fn mk_model_info(hidden: usize, layers: usize, arch: &str) -> ModelInfo {
    ModelInfo {
        size: "test".to_string(),
        hidden_size: hidden,
        num_layers: layers,
        architecture: arch.to_string(),
        weights_available: false,
        lora_trainable_params: 100_000,
        classifier_params: 1000,
    }
}

fn mk_data_audit(samples: usize) -> DataAudit {
    DataAudit {
        train_path: "/tmp/data.jsonl".to_string(),
        train_samples: samples,
        avg_input_len: 50,
        class_counts: vec![samples / 2, samples - samples / 2],
        imbalance_ratio: 1.0,
        auto_class_weights: false,
        val_samples: None,
        test_samples: None,
        duplicates: 0,
        preamble_count: 0,
    }
}

// ── resolve_model alias coverage ──────────────────────────────────────

#[test]
fn test_cov2_resolve_model_qwen2_05b_alias() {
    let config = PlanConfig {
        model_size: "qwen2-0.5b".to_string(),
        ..mk_plan_config("qwen2-0.5b", "manual")
    };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    assert_eq!(model.hidden_size, 896);
    assert_eq!(model.num_layers, 24);
    assert_eq!(model.architecture, "qwen2");
}

#[test]
fn test_cov2_resolve_model_qwen35_9b_alias() {
    let config = PlanConfig {
        model_size: "qwen3.5-9b".to_string(),
        ..mk_plan_config("qwen3.5-9b", "manual")
    };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    assert_eq!(model.hidden_size, 4096);
    assert_eq!(model.num_layers, 48);
    assert_eq!(model.architecture, "qwen3.5");
}

#[test]
fn test_cov2_resolve_model_llama2_7b_alias() {
    let config =
        PlanConfig { model_size: "llama2-7b".to_string(), ..mk_plan_config("llama2-7b", "manual") };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    assert_eq!(model.hidden_size, 4096);
    assert_eq!(model.num_layers, 32);
    assert_eq!(model.architecture, "llama2");
}

#[test]
fn test_cov2_resolve_model_llama2_13b_alias() {
    let config = PlanConfig {
        model_size: "llama2-13b".to_string(),
        ..mk_plan_config("llama2-13b", "manual")
    };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    assert_eq!(model.hidden_size, 5120);
    assert_eq!(model.num_layers, 40);
    assert_eq!(model.architecture, "llama2");
}

// ── resolve_model with sharded safetensors ───────────────────────────

#[test]
fn test_cov2_resolve_model_sharded_safetensors() {
    let dir = tempfile::tempdir().unwrap();
    // Create sharded file pattern (model-00001-of-00002.safetensors)
    std::fs::write(dir.path().join("model-00001-of-00002.safetensors"), b"fake").unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
    let config = PlanConfig {
        model_path: Some(dir.path().to_path_buf()),
        ..mk_plan_config("0.5B", "manual")
    };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    assert!(model.weights_available);
    assert!(pf.iter().any(|c| c.name == "model_weights" && c.status == CheckStatus::Pass));
}

// ── resolve_model with missing tokenizer only ────────────────────────

#[test]
fn test_cov2_resolve_model_missing_tokenizer() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
    // No tokenizer.json
    let config = PlanConfig {
        model_path: Some(dir.path().to_path_buf()),
        ..mk_plan_config("0.5B", "manual")
    };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    assert!(model.weights_available);
    let warn_check = pf.iter().find(|c| c.name == "model_weights").unwrap();
    assert_eq!(warn_check.status, CheckStatus::Warn);
    assert!(warn_check.detail.contains("tokenizer.json"));
}

// ── resolve_model with missing safetensors only ──────────────────────

#[test]
fn test_cov2_resolve_model_missing_safetensors() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
    // No model.safetensors
    let config = PlanConfig {
        model_path: Some(dir.path().to_path_buf()),
        ..mk_plan_config("0.5B", "manual")
    };
    let mut pf = Vec::new();
    let _model = resolve_model(&config, &mut pf);
    let warn_check = pf.iter().find(|c| c.name == "model_weights").unwrap();
    assert_eq!(warn_check.status, CheckStatus::Warn);
    assert!(warn_check.detail.contains("model.safetensors"));
}

// ── resolve_model no model_path → Warn ───────────────────────────────

#[test]
fn test_cov2_resolve_model_no_path_warns() {
    let config = PlanConfig { model_path: None, ..mk_plan_config("0.5B", "manual") };
    let mut pf = Vec::new();
    let _model = resolve_model(&config, &mut pf);
    let check = pf.iter().find(|c| c.name == "model_weights").unwrap();
    assert_eq!(check.status, CheckStatus::Warn);
    assert!(check.detail.contains("default model resolution"));
}

// ── estimate_resources manual vs HPO ─────────────────────────────────

#[test]
fn test_cov2_estimate_resources_manual_one_trial() {
    let config = PlanConfig {
        strategy: "manual".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 3,
        ..mk_plan_config("0.5B", "manual")
    };
    let model = mk_model_info(896, 24, "qwen2");
    let data = mk_data_audit(200);
    let res = estimate_resources(&config, &model, &data, 32);
    // manual: total_trials = 1, total_epochs = 3
    let expected_steps = 200usize.div_ceil(32);
    assert_eq!(res.steps_per_epoch, expected_steps);
    // total_minutes = minutes_per_epoch * 3 * 1
    let minutes_per_epoch = (expected_steps as f64 * 58.0) / 60.0;
    assert!((res.estimated_total_minutes - minutes_per_epoch * 3.0).abs() < 0.01);
}

#[test]
fn test_cov2_estimate_resources_hpo_multi_trial() {
    let config = PlanConfig {
        strategy: "tpe".to_string(),
        budget: 10,
        scout: false,
        max_epochs: 2,
        ..mk_plan_config("0.5B", "tpe")
    };
    let model = mk_model_info(896, 24, "qwen2");
    let data = mk_data_audit(100);
    let res = estimate_resources(&config, &model, &data, 64);
    // HPO: total_trials = 10, total_epochs = 2
    let steps = 100usize.div_ceil(64);
    let min_per_epoch = (steps as f64 * 58.0) / 60.0;
    let expected_total = min_per_epoch * 2.0 * 10.0;
    assert!((res.estimated_total_minutes - expected_total).abs() < 0.01);
}

// ── estimate_resources time scaling by model size ─────────────────────

#[test]
fn test_cov2_estimate_resources_9b_seconds_per_step() {
    let config = mk_plan_config("9B", "manual");
    let model = mk_model_info(4096, 48, "qwen3.5");
    let data = mk_data_audit(100);
    let res = estimate_resources(&config, &model, &data, 64);
    // 4096 hidden → 270 seconds per step
    let steps = 100usize.div_ceil(64);
    let expected_min = (steps as f64 * 270.0) / 60.0;
    assert!((res.estimated_minutes_per_epoch - expected_min).abs() < 0.01);
}

#[test]
fn test_cov2_estimate_resources_13b_seconds_per_step() {
    let config = mk_plan_config("13B", "manual");
    let model = mk_model_info(5120, 40, "llama2");
    let data = mk_data_audit(100);
    let res = estimate_resources(&config, &model, &data, 64);
    // 5120 hidden → 450 seconds per step
    let steps = 100usize.div_ceil(64);
    let expected_min = (steps as f64 * 450.0) / 60.0;
    assert!((res.estimated_minutes_per_epoch - expected_min).abs() < 0.01);
}

#[test]
fn test_cov2_estimate_resources_unknown_hidden_seconds() {
    let config = mk_plan_config("custom", "manual");
    let model = mk_model_info(1024, 12, "custom");
    let data = mk_data_audit(100);
    let res = estimate_resources(&config, &model, &data, 64);
    // Unknown hidden → default 90 seconds per step, default 3.0 GB
    let steps = 100usize.div_ceil(64);
    let expected_min = (steps as f64 * 90.0) / 60.0;
    assert!((res.estimated_minutes_per_epoch - expected_min).abs() < 0.01);
    assert!((res.estimated_vram_gb - 3.0).abs() < 0.01);
}

// ── estimate_resources checkpoint size ────────────────────────────────

#[test]
fn test_cov2_estimate_resources_checkpoint_mb() {
    let config = mk_plan_config("0.5B", "manual");
    let model = ModelInfo {
        size: "0.5B".to_string(),
        hidden_size: 896,
        num_layers: 24,
        architecture: "qwen2".to_string(),
        weights_available: false,
        lora_trainable_params: 500_000,
        classifier_params: 5_000,
    };
    let data = mk_data_audit(100);
    let res = estimate_resources(&config, &model, &data, 64);
    let expected_mb = f64::from(500_000 + 5_000) * 4.0 / 1_048_576.0;
    assert!((res.estimated_checkpoint_mb - expected_mb).abs() < 0.001);
}

// ── estimate_gpu_hours edge cases ─────────────────────────────────────

#[test]
fn test_cov2_estimate_gpu_hours_large_dataset() {
    let hours = estimate_gpu_hours(10_000, 5, 20);
    // 10000 / 64 = 157 steps, 157 * 58 = 9106 sec/epoch
    // total = 9106 * 5 * 20 = 910600 sec = 252.9 hours
    assert!(hours > 200.0);
    assert!(hours < 300.0);
}

#[test]
fn test_cov2_estimate_gpu_hours_single_sample() {
    let hours = estimate_gpu_hours(1, 1, 1);
    // 1 / 64 → div_ceil → 1 step, 1 * 58 / 3600 ~ 0.016 hours
    assert!(hours > 0.01);
    assert!(hours < 0.02);
}

// ── build_hpo_plan unknown strategy defaults to TPE ──────────────────

#[test]
fn test_cov2_build_hpo_plan_unknown_strategy_defaults_to_tpe() {
    let config = PlanConfig {
        strategy: "bogus".to_string(),
        budget: 5,
        scout: false,
        max_epochs: 2,
        ..mk_plan_config("0.5B", "bogus")
    };
    let mut issues = Vec::new();
    let hpo = build_hpo_plan(&config, 100, &mut issues);
    // Unknown strategy should parse to Tpe (default)
    assert_eq!(hpo.strategy, "bogus");
    assert_eq!(hpo.budget, 5);
    assert!(!hpo.sample_configs.is_empty());
}

// ── build_hpo_plan scout mode forces 1 epoch ─────────────────────────

#[test]
fn test_cov2_build_hpo_plan_scout_forces_1_epoch() {
    let config = PlanConfig {
        strategy: "random".to_string(),
        budget: 10,
        scout: true,
        max_epochs: 50,
        ..mk_plan_config("0.5B", "random")
    };
    let mut issues = Vec::new();
    let hpo = build_hpo_plan(&config, 100, &mut issues);
    assert_eq!(hpo.max_epochs, 1);
    assert!(hpo.scout);
}

// ── build_hpo_plan no budget warning for grid/random ─────────────────

#[test]
fn test_cov2_build_hpo_plan_low_budget_no_warning_for_grid() {
    let config = PlanConfig {
        strategy: "grid".to_string(),
        budget: 3,
        scout: false,
        max_epochs: 1,
        ..mk_plan_config("0.5B", "grid")
    };
    let mut issues = Vec::new();
    build_hpo_plan(&config, 100, &mut issues);
    // Low budget warning is only for TPE
    assert!(!issues.iter().any(|i| i.message.contains("TPE budget")));
}

// ── build_hpo_plan large dataset no warning when scout=true ──────────

#[test]
fn test_cov2_build_hpo_plan_large_dataset_no_scout_warning_when_scout() {
    let config = PlanConfig {
        strategy: "tpe".to_string(),
        budget: 20,
        scout: true,
        max_epochs: 5,
        ..mk_plan_config("0.5B", "tpe")
    };
    let mut issues = Vec::new();
    build_hpo_plan(&config, 50_000, &mut issues);
    // Should NOT warn about GPU hours when scout is already on
    assert!(!issues.iter().any(|i| i.message.contains("GPU hours")));
}

// ── resolve_class_weights edge cases ─────────────────────────────────

#[test]
fn test_cov2_resolve_class_weights_inverse_freq_imbalanced() {
    let weights = resolve_class_weights("inverse_freq", &[900, 100], 2);
    let w = weights.unwrap();
    assert_eq!(w.len(), 2);
    // Class 0 (900 samples) should have smaller weight
    assert!(w[0] < w[1]);
}

#[test]
fn test_cov2_resolve_class_weights_sqrt_inverse_imbalanced() {
    let weights = resolve_class_weights("sqrt_inverse", &[900, 100], 2);
    let w = weights.unwrap();
    assert_eq!(w.len(), 2);
    assert!(w[0] < w[1]);
}

#[test]
fn test_cov2_resolve_class_weights_empty_string() {
    let weights = resolve_class_weights("", &[50, 50], 2);
    assert!(weights.is_none());
}

// ── TrainingPlan to_json / to_yaml on complex plan ───────────────────

#[test]
fn test_cov2_training_plan_to_json_contains_fields() {
    let plan = TrainingPlan {
        version: "1.0".to_string(),
        task: "classify".to_string(),
        data: mk_data_audit(200),
        model: mk_model_info(896, 24, "qwen2"),
        hyperparameters: HyperparameterPlan {
            strategy: "tpe".to_string(),
            budget: 10,
            scout: true,
            max_epochs: 1,
            search_space_params: 9,
            sample_configs: Vec::new(),
            manual: None,
            recommendation: Some("use tpe".to_string()),
        },
        resources: ResourceEstimate {
            estimated_vram_gb: 2.5,
            estimated_minutes_per_epoch: 5.0,
            estimated_total_minutes: 50.0,
            estimated_checkpoint_mb: 10.0,
            steps_per_epoch: 4,
            gpu_device: Some("RTX 4090".to_string()),
        },
        pre_flight: vec![PreFlightCheck {
            name: "data".to_string(),
            status: CheckStatus::Pass,
            detail: "ok".to_string(),
        }],
        output_dir: "/tmp/out".to_string(),
        auto_diagnose: true,
        verdict: PlanVerdict::Ready,
        issues: vec![PlanIssue {
            severity: CheckStatus::Warn,
            category: "Data".to_string(),
            message: "test warning".to_string(),
            fix: Some("fix it".to_string()),
        }],
    };
    let json = plan.to_json();
    assert!(json.contains("classify"));
    assert!(json.contains("RTX 4090"));
    assert!(json.contains("test warning"));

    let yaml = plan.to_yaml();
    assert!(yaml.contains("classify"));
    assert!(yaml.contains("RTX 4090"));
}

// ── TrainingPlan from_str with empty YAML document ───────────────────

#[test]
fn test_cov2_training_plan_from_str_empty() {
    let result = TrainingPlan::from_str("");
    assert!(result.is_err());
}

// ── TrainingPlan check_counts with all pass ──────────────────────────

#[test]
fn test_cov2_check_counts_all_pass() {
    let plan = TrainingPlan {
        version: "1.0".to_string(),
        task: "classify".to_string(),
        data: mk_data_audit(100),
        model: mk_model_info(896, 24, "qwen2"),
        hyperparameters: HyperparameterPlan {
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            search_space_params: 0,
            sample_configs: Vec::new(),
            manual: None,
            recommendation: None,
        },
        resources: ResourceEstimate {
            estimated_vram_gb: 2.5,
            estimated_minutes_per_epoch: 1.0,
            estimated_total_minutes: 1.0,
            estimated_checkpoint_mb: 1.0,
            steps_per_epoch: 2,
            gpu_device: None,
        },
        pre_flight: vec![
            PreFlightCheck {
                name: "a".to_string(),
                status: CheckStatus::Pass,
                detail: "ok".to_string(),
            },
            PreFlightCheck {
                name: "b".to_string(),
                status: CheckStatus::Pass,
                detail: "ok".to_string(),
            },
            PreFlightCheck {
                name: "c".to_string(),
                status: CheckStatus::Pass,
                detail: "ok".to_string(),
            },
        ],
        output_dir: "/tmp/test".to_string(),
        auto_diagnose: false,
        verdict: PlanVerdict::Ready,
        issues: Vec::new(),
    };
    let (pass, warn, fail) = plan.check_counts();
    assert_eq!(pass, 3);
    assert_eq!(warn, 0);
    assert_eq!(fail, 0);
}

#[test]
fn test_cov2_check_counts_all_fail() {
    let plan = TrainingPlan {
        version: "1.0".to_string(),
        task: "classify".to_string(),
        data: mk_data_audit(100),
        model: mk_model_info(896, 24, "qwen2"),
        hyperparameters: HyperparameterPlan {
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            search_space_params: 0,
            sample_configs: Vec::new(),
            manual: None,
            recommendation: None,
        },
        resources: ResourceEstimate {
            estimated_vram_gb: 2.5,
            estimated_minutes_per_epoch: 1.0,
            estimated_total_minutes: 1.0,
            estimated_checkpoint_mb: 1.0,
            steps_per_epoch: 2,
            gpu_device: None,
        },
        pre_flight: vec![
            PreFlightCheck {
                name: "a".to_string(),
                status: CheckStatus::Fail,
                detail: "bad".to_string(),
            },
            PreFlightCheck {
                name: "b".to_string(),
                status: CheckStatus::Fail,
                detail: "bad".to_string(),
            },
        ],
        output_dir: "/tmp/test".to_string(),
        auto_diagnose: false,
        verdict: PlanVerdict::Blocked,
        issues: Vec::new(),
    };
    let (pass, warn, fail) = plan.check_counts();
    assert_eq!(pass, 0);
    assert_eq!(warn, 0);
    assert_eq!(fail, 2);
}

#[test]
fn test_cov2_check_counts_all_warn() {
    let plan = TrainingPlan {
        version: "1.0".to_string(),
        task: "classify".to_string(),
        data: mk_data_audit(100),
        model: mk_model_info(896, 24, "qwen2"),
        hyperparameters: HyperparameterPlan {
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            search_space_params: 0,
            sample_configs: Vec::new(),
            manual: None,
            recommendation: None,
        },
        resources: ResourceEstimate {
            estimated_vram_gb: 2.5,
            estimated_minutes_per_epoch: 1.0,
            estimated_total_minutes: 1.0,
            estimated_checkpoint_mb: 1.0,
            steps_per_epoch: 2,
            gpu_device: None,
        },
        pre_flight: vec![
            PreFlightCheck {
                name: "a".to_string(),
                status: CheckStatus::Warn,
                detail: "meh".to_string(),
            },
            PreFlightCheck {
                name: "b".to_string(),
                status: CheckStatus::Warn,
                detail: "meh".to_string(),
            },
            PreFlightCheck {
                name: "c".to_string(),
                status: CheckStatus::Warn,
                detail: "meh".to_string(),
            },
        ],
        output_dir: "/tmp/test".to_string(),
        auto_diagnose: false,
        verdict: PlanVerdict::WarningsPresent,
        issues: Vec::new(),
    };
    let (pass, warn, fail) = plan.check_counts();
    assert_eq!(pass, 0);
    assert_eq!(warn, 3);
    assert_eq!(fail, 0);
}

// ── PlanConfig clone ─────────────────────────────────────────────────

#[test]
fn test_cov2_plan_config_clone() {
    let config = PlanConfig {
        manual_lr: Some(2e-5),
        manual_lora_rank: Some(4),
        manual_batch_size: Some(16),
        manual_lora_alpha: Some(8.0),
        manual_warmup: Some(0.05),
        manual_gradient_clip: Some(0.5),
        manual_lr_min_ratio: Some(0.001),
        manual_class_weights: Some("inverse_freq".to_string()),
        manual_target_modules: Some("all_linear".to_string()),
        ..mk_plan_config("0.5B", "manual")
    };
    let cloned = config.clone();
    assert_eq!(cloned.manual_lr, Some(2e-5));
    assert_eq!(cloned.manual_lora_rank, Some(4));
    assert_eq!(cloned.manual_batch_size, Some(16));
    assert_eq!(cloned.manual_lora_alpha, Some(8.0));
    assert_eq!(cloned.manual_warmup, Some(0.05));
    assert_eq!(cloned.manual_gradient_clip, Some(0.5));
    assert_eq!(cloned.manual_lr_min_ratio, Some(0.001));
    assert_eq!(cloned.manual_class_weights.as_deref(), Some("inverse_freq"));
    assert_eq!(cloned.manual_target_modules.as_deref(), Some("all_linear"));
}

// ── PlanConfig debug ─────────────────────────────────────────────────

#[test]
fn test_cov2_plan_config_debug() {
    let config = mk_plan_config("0.5B", "manual");
    let debug = format!("{config:?}");
    assert!(debug.contains("PlanConfig"));
    assert!(debug.contains("0.5B"));
}

// ── CheckStatus copy/clone/debug ─────────────────────────────────────

#[test]
fn test_cov2_check_status_copy_clone_debug() {
    let s = CheckStatus::Pass;
    let c = s; // Copy
    let cl = s; // Clone
    assert_eq!(c, cl);
    let debug = format!("{s:?}");
    assert!(debug.contains("Pass"));

    let w = CheckStatus::Warn;
    let debug_w = format!("{w:?}");
    assert!(debug_w.contains("Warn"));

    let f = CheckStatus::Fail;
    let debug_f = format!("{f:?}");
    assert!(debug_f.contains("Fail"));
}

// ── PlanVerdict copy/clone/debug ─────────────────────────────────────

#[test]
fn test_cov2_plan_verdict_copy_clone_debug() {
    let v = PlanVerdict::Ready;
    let c = v; // Copy
    let cl = v; // Clone
    assert_eq!(c, cl);
    let debug = format!("{v:?}");
    assert!(debug.contains("Ready"));

    let wp = PlanVerdict::WarningsPresent;
    let debug_wp = format!("{wp:?}");
    assert!(debug_wp.contains("WarningsPresent"));

    let b = PlanVerdict::Blocked;
    let debug_b = format!("{b:?}");
    assert!(debug_b.contains("Blocked"));
}

// ── TrainingPlan clone ───────────────────────────────────────────────

#[test]
fn test_cov2_training_plan_clone() {
    let plan = TrainingPlan {
        version: "1.0".to_string(),
        task: "classify".to_string(),
        data: mk_data_audit(100),
        model: mk_model_info(896, 24, "qwen2"),
        hyperparameters: HyperparameterPlan {
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            search_space_params: 0,
            sample_configs: Vec::new(),
            manual: None,
            recommendation: None,
        },
        resources: ResourceEstimate {
            estimated_vram_gb: 2.5,
            estimated_minutes_per_epoch: 1.0,
            estimated_total_minutes: 1.0,
            estimated_checkpoint_mb: 1.0,
            steps_per_epoch: 4,
            gpu_device: None,
        },
        pre_flight: Vec::new(),
        output_dir: "/tmp/test".to_string(),
        auto_diagnose: true,
        verdict: PlanVerdict::Ready,
        issues: Vec::new(),
    };
    let cloned = plan.clone();
    assert_eq!(cloned.version, "1.0");
    assert_eq!(cloned.task, "classify");
    assert_eq!(cloned.verdict, PlanVerdict::Ready);
    assert!(cloned.auto_diagnose);
}

// ── ManualConfig clone ───────────────────────────────────────────────

#[test]
fn test_cov2_manual_config_clone() {
    let mc = ManualConfig {
        learning_rate: 1e-4,
        lora_rank: 16,
        batch_size: 32,
        lora_alpha: Some(32.0),
        warmup_fraction: Some(0.1),
        gradient_clip_norm: Some(1.0),
        lr_min_ratio: Some(0.01),
        class_weights: Some("sqrt_inverse".to_string()),
        target_modules: Some("qv".to_string()),
    };
    let cloned = mc.clone();
    assert!((cloned.learning_rate - 1e-4).abs() < 1e-10);
    assert_eq!(cloned.lora_rank, 16);
    assert_eq!(cloned.class_weights.as_deref(), Some("sqrt_inverse"));
}

// ── TrialPreview clone ───────────────────────────────────────────────

#[test]
fn test_cov2_trial_preview_clone() {
    let tp = TrialPreview {
        trial: 5,
        learning_rate: 3e-5,
        lora_rank: 8,
        lora_alpha: 16.0,
        batch_size: 64,
        warmup: 0.2,
        gradient_clip: 0.5,
        class_weights: "uniform".to_string(),
        target_modules: "qkv".to_string(),
        lr_min_ratio: 0.005,
    };
    let cloned = tp.clone();
    assert_eq!(cloned.trial, 5);
    assert!((cloned.learning_rate - 3e-5).abs() < 1e-10);
    assert_eq!(cloned.target_modules, "qkv");
}

// ── DataAudit clone ──────────────────────────────────────────────────

#[test]
fn test_cov2_data_audit_clone_debug() {
    let da = DataAudit {
        train_path: "/data.jsonl".to_string(),
        train_samples: 100,
        avg_input_len: 42,
        class_counts: vec![60, 40],
        imbalance_ratio: 1.5,
        auto_class_weights: false,
        val_samples: Some(10),
        test_samples: None,
        duplicates: 2,
        preamble_count: 5,
    };
    let cloned = da.clone();
    assert_eq!(cloned.train_samples, 100);
    assert_eq!(cloned.val_samples, Some(10));
    assert!(cloned.test_samples.is_none());
    let debug = format!("{da:?}");
    assert!(debug.contains("DataAudit"));
}

// ── ResourceEstimate clone ───────────────────────────────────────────

#[test]
fn test_cov2_resource_estimate_clone_debug() {
    let re = ResourceEstimate {
        estimated_vram_gb: 2.5,
        estimated_minutes_per_epoch: 3.0,
        estimated_total_minutes: 30.0,
        estimated_checkpoint_mb: 5.0,
        steps_per_epoch: 10,
        gpu_device: Some("RTX 4090".to_string()),
    };
    let cloned = re.clone();
    assert!((cloned.estimated_vram_gb - 2.5).abs() < 1e-6);
    assert_eq!(cloned.gpu_device.as_deref(), Some("RTX 4090"));
    let debug = format!("{re:?}");
    assert!(debug.contains("ResourceEstimate"));
}

// ── PreFlightCheck clone ─────────────────────────────────────────────

#[test]
fn test_cov2_pre_flight_check_clone_debug() {
    let c = PreFlightCheck {
        name: "test_check".to_string(),
        status: CheckStatus::Pass,
        detail: "all good".to_string(),
    };
    let cloned = c.clone();
    assert_eq!(cloned.name, "test_check");
    assert_eq!(cloned.status, CheckStatus::Pass);
    let debug = format!("{c:?}");
    assert!(debug.contains("PreFlightCheck"));
}

// ── PlanIssue clone ──────────────────────────────────────────────────

#[test]
fn test_cov2_plan_issue_clone_debug() {
    let issue = PlanIssue {
        severity: CheckStatus::Warn,
        category: "Data".to_string(),
        message: "something".to_string(),
        fix: Some("do thing".to_string()),
    };
    let cloned = issue.clone();
    assert_eq!(cloned.severity, CheckStatus::Warn);
    assert_eq!(cloned.fix.as_deref(), Some("do thing"));
    let debug = format!("{issue:?}");
    assert!(debug.contains("PlanIssue"));
}

// ── count_file_samples with unreadable file ──────────────────────────

#[test]
fn test_cov2_count_file_samples_invalid_jsonl() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.jsonl");
    std::fs::write(&path, "this is not valid jsonl").unwrap();
    // Should return None because load_safety_corpus fails
    let result = count_file_samples(Some(&path), 2);
    assert!(result.is_none());
}

// ── HyperparameterPlan clone ─────────────────────────────────────────

#[test]
fn test_cov2_hyperparameter_plan_clone() {
    let hp = HyperparameterPlan {
        strategy: "grid".to_string(),
        budget: 15,
        scout: false,
        max_epochs: 3,
        search_space_params: 9,
        sample_configs: vec![],
        manual: Some(ManualConfig {
            learning_rate: 1e-4,
            lora_rank: 8,
            batch_size: 32,
            lora_alpha: None,
            warmup_fraction: None,
            gradient_clip_norm: None,
            lr_min_ratio: None,
            class_weights: None,
            target_modules: None,
        }),
        recommendation: None,
    };
    let cloned = hp.clone();
    assert_eq!(cloned.strategy, "grid");
    assert_eq!(cloned.budget, 15);
    assert!(cloned.manual.is_some());
}

// ── build_hpo_plan budget=1 for tpe ──────────────────────────────────

#[test]
fn test_cov2_build_hpo_plan_tpe_budget_1() {
    let config = PlanConfig {
        strategy: "tpe".to_string(),
        budget: 1,
        scout: false,
        max_epochs: 1,
        ..mk_plan_config("0.5B", "tpe")
    };
    let mut issues = Vec::new();
    let hpo = build_hpo_plan(&config, 100, &mut issues);
    assert_eq!(hpo.budget, 1);
    // Budget 1 < 5, should warn
    assert!(issues.iter().any(|i| i.message.contains("TPE budget")));
    // Only 1 preview
    assert!(hpo.sample_configs.len() <= 1);
}

// ── build_hpo_plan budget=0 for non-manual ───────────────────────────

#[test]
fn test_cov2_build_hpo_plan_tpe_budget_0() {
    let config = PlanConfig {
        strategy: "tpe".to_string(),
        budget: 0,
        scout: false,
        max_epochs: 1,
        ..mk_plan_config("0.5B", "tpe")
    };
    let mut issues = Vec::new();
    let hpo = build_hpo_plan(&config, 100, &mut issues);
    assert_eq!(hpo.budget, 0);
    assert!(hpo.sample_configs.is_empty());
}

// ── ModelInfo debug ──────────────────────────────────────────────────

#[test]
fn test_cov2_model_info_debug() {
    let mi = mk_model_info(4096, 32, "llama2");
    let debug = format!("{mi:?}");
    assert!(debug.contains("ModelInfo"));
    assert!(debug.contains("llama2"));
}

// ── resolve_model custom lora rank ───────────────────────────────────

#[test]
fn test_cov2_resolve_model_default_rank_16() {
    let config = PlanConfig { manual_lora_rank: None, ..mk_plan_config("0.5B", "manual") };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    // Default rank is 16: lora_params = 2 * 16 * 896 * 2 * 24
    assert_eq!(model.lora_trainable_params, 2 * 16 * 896 * 2 * 24);
}

#[test]
fn test_cov2_resolve_model_custom_rank_4() {
    let config = PlanConfig { manual_lora_rank: Some(4), ..mk_plan_config("0.5B", "manual") };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    assert_eq!(model.lora_trainable_params, 2 * 4 * 896 * 2 * 24);
}

// ── resolve_model classifier_params for different num_classes ────────

#[test]
fn test_cov2_resolve_model_classifier_params_10_classes() {
    let config = PlanConfig { num_classes: 10, ..mk_plan_config("0.5B", "manual") };
    let mut pf = Vec::new();
    let model = resolve_model(&config, &mut pf);
    assert_eq!(model.classifier_params, 896 * 10 + 10);
}

// ── estimate_resources with scout + non-manual ───────────────────────

#[test]
fn test_cov2_estimate_resources_scout_hpo() {
    let config = PlanConfig {
        strategy: "tpe".to_string(),
        budget: 20,
        scout: true,
        max_epochs: 10,
        ..mk_plan_config("0.5B", "tpe")
    };
    let model = mk_model_info(896, 24, "qwen2");
    let data = mk_data_audit(100);
    let res = estimate_resources(&config, &model, &data, 64);
    // scout: total_epochs = 1, budget = 20
    let steps = 100usize.div_ceil(64);
    let min_per_epoch = (steps as f64 * 58.0) / 60.0;
    let expected = min_per_epoch * 1.0 * 20.0;
    assert!((res.estimated_total_minutes - expected).abs() < 0.01);
}

// ── Plan with output_dir that doesn't exist yet (Pass) ───────────────

#[test]
fn test_cov2_plan_output_dir_will_be_created() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    for i in 0..20 {
        lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
    }
    std::fs::write(&data_path, lines.join("\n")).unwrap();
    let output_dir = dir.path().join("nonexistent_output");
    let config = PlanConfig {
        data_path,
        output_dir: output_dir.clone(),
        ..mk_plan_config("0.5B", "manual")
    };
    let p = plan(&config).unwrap();
    let out_check = p.pre_flight.iter().find(|c| c.name == "output_dir").unwrap();
    assert_eq!(out_check.status, CheckStatus::Pass);
    assert!(out_check.detail.contains("will be created"));
}

// ── Plan with existing output_dir no checkpoints (Pass) ──────────────

#[test]
fn test_cov2_plan_output_dir_exists_no_checkpoints() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    for i in 0..20 {
        lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
    }
    std::fs::write(&data_path, lines.join("\n")).unwrap();
    let output_dir = dir.path().join("output");
    std::fs::create_dir_all(&output_dir).unwrap();
    // No metadata.json or epoch_001 — clean dir
    let config = PlanConfig { data_path, output_dir, ..mk_plan_config("0.5B", "manual") };
    let p = plan(&config).unwrap();
    let out_check = p.pre_flight.iter().find(|c| c.name == "output_dir").unwrap();
    assert_eq!(out_check.status, CheckStatus::Pass);
    assert!(out_check.detail.contains("exists"));
}

// ── Plan output_dir with epoch_001 subdir (Warn) ─────────────────────

#[test]
fn test_cov2_plan_output_dir_has_epoch_subdir() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    for i in 0..20 {
        lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
    }
    std::fs::write(&data_path, lines.join("\n")).unwrap();
    let output_dir = dir.path().join("output");
    std::fs::create_dir_all(output_dir.join("epoch_001")).unwrap();
    let config = PlanConfig { data_path, output_dir, ..mk_plan_config("0.5B", "manual") };
    let p = plan(&config).unwrap();
    let out_check = p.pre_flight.iter().find(|c| c.name == "output_dir").unwrap();
    assert_eq!(out_check.status, CheckStatus::Warn);
    assert!(out_check.detail.contains("checkpoints"));
}

// ── Plan class_weights_persist always passes ─────────────────────────

#[test]
fn test_cov2_plan_class_weights_persist_check() {
    let dir = tempfile::tempdir().unwrap();
    let data_path = dir.path().join("train.jsonl");
    let mut lines = Vec::new();
    for i in 0..20 {
        lines.push(format!(r#"{{"input": "echo {i}", "label": {}}}"#, i % 2));
    }
    std::fs::write(&data_path, lines.join("\n")).unwrap();
    let config = PlanConfig {
        data_path,
        output_dir: dir.path().to_path_buf(),
        ..mk_plan_config("0.5B", "manual")
    };
    let p = plan(&config).unwrap();
    let cw_check = p.pre_flight.iter().find(|c| c.name == "class_weights_persist").unwrap();
    assert_eq!(cw_check.status, CheckStatus::Pass);
}

// ── ExperimentTracker log methods are no-op without store ────────────

#[test]
fn test_cov2_experiment_tracker_no_store_no_panic() {
    let mut tracker = ExperimentTracker { store: None, exp_id: None };
    tracker.log_failed_trial();
    // Should be no-op
}

#[test]
fn test_cov2_experiment_tracker_with_store_no_exp_id() {
    let dir = tempfile::tempdir().unwrap();
    let store = crate::storage::SqliteBackend::open_project(dir.path()).ok();
    let mut tracker = ExperimentTracker { store, exp_id: None };
    // log_failed_trial with store but no exp_id → early return
    tracker.log_failed_trial();
}

// ── ApplyConfig debug ────────────────────────────────────────────────

#[test]
fn test_cov2_apply_config_debug() {
    let ac = ApplyConfig {
        model_path: PathBuf::from("/tmp/model"),
        data_path: PathBuf::from("/tmp/data.jsonl"),
        output_dir: PathBuf::from("/tmp/out"),
        on_trial_complete: None,
    };
    let debug = format!("{ac:?}");
    assert!(debug.contains("ApplyConfig"));
}
