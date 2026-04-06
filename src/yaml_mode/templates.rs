//! Template Generation for YAML Mode Training
//!
//! Generates starter manifests for common training scenarios.

/// Default `T_max` for cosine annealing scheduler (total steps before restart).
const DEFAULT_COSINE_ANNEALING_T_MAX: usize = 10000;

use super::manifest::{
    AlertConfig, CallbackConfig, CallbackType, ChartConfig, CheckpointConfig, DataConfig,
    DataLoader, DataSplit, EarlyStoppingConfig, GradientConfig, LoraConfig, MetricsOutputConfig,
    MixedPrecisionConfig, ModelConfig, ModelOutputConfig, MonitoringConfig, OptimizerConfig,
    OutputConfig, QuantizeConfig, RegistryConfig, ReportConfig, SchedulerConfig,
    SystemMonitorConfig, TerminalMonitor, TrackingConfig, TrainingConfig, TrainingManifest,
    WarmupConfig,
};

/// Template type for initialization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Template {
    /// Minimal manifest with required fields only
    Minimal,
    /// LoRA fine-tuning template
    Lora,
    /// QLoRA fine-tuning template
    Qlora,
    /// Full template with all sections
    Full,
}

/// Generate a training manifest from a template
pub fn generate_manifest(
    template: Template,
    name: &str,
    model: Option<&str>,
    data: Option<&str>,
) -> TrainingManifest {
    generate_manifest_with_hints(template, name, model, data, None, None)
}

/// Generate a training manifest with optional smart defaults
pub fn generate_manifest_with_hints(
    template: Template,
    name: &str,
    model: Option<&str>,
    data: Option<&str>,
    lora_rank: Option<u32>,
    learning_rate: Option<f64>,
) -> TrainingManifest {
    let mut manifest = match template {
        Template::Minimal => generate_minimal(name, model, data),
        Template::Lora => generate_lora(name, model, data),
        Template::Qlora => generate_qlora(name, model, data),
        Template::Full => generate_full(name, model, data),
    };

    // Apply smart defaults if provided
    if let Some(rank) = lora_rank {
        if let Some(ref mut lora) = manifest.lora {
            lora.rank = rank.min(1024) as usize;
            // Alpha is typically 2x rank
            lora.alpha = f64::from(rank * 2);
        }
    }
    if let Some(lr) = learning_rate {
        if let Some(ref mut optim) = manifest.optimizer {
            optim.lr = lr;
        }
    }

    manifest
}

/// Generate YAML string from a template
pub fn generate_yaml(
    template: Template,
    name: &str,
    model: Option<&str>,
    data: Option<&str>,
    lora_rank: Option<u32>,
    learning_rate: Option<f64>,
) -> String {
    let manifest =
        generate_manifest_with_hints(template, name, model, data, lora_rank, learning_rate);
    serde_yaml::to_string(&manifest).unwrap_or_else(|_err| "# Error generating YAML".to_string())
}

fn generate_minimal(name: &str, model: Option<&str>, data: Option<&str>) -> TrainingManifest {
    TrainingManifest {
        entrenar: "1.0".to_string(),
        name: name.to_string(),
        version: "1.0.0".to_string(),
        description: Some("Training experiment".to_string()),
        seed: Some(42),
        data: data.map(default_data_config),
        model: model.map(default_model_config),
        optimizer: Some(default_optimizer_config()),
        scheduler: Some(default_scheduler_config()),
        training: Some(default_training_config()),
        lora: None,
        quantize: None,
        monitoring: Some(default_monitoring_config()),
        callbacks: None,
        output: Some(default_output_config()),
        publish: None,
        // Extended configurations (YAML Mode QA Epic)
        citl: None,
        rag: None,
        graph: None,
        distillation: None,
        inspect: None,
        privacy: None,
        audit: None,
        session: None,
        stress: None,
        benchmark: None,
        debug: None,
        signing: None,
        verification: None,
        lockfile: None,
        strict: None,
        strict_validation: None,
        require_peer_review: None,
    }
}

fn default_data_config(source: &str) -> DataConfig {
    DataConfig {
        source: Some(source.to_string()),
        format: None,
        split: Some(DataSplit {
            train: 0.8,
            val: Some(0.1),
            test: Some(0.1),
            stratify: None,
            seed: Some(42),
        }),
        train: None,
        val: None,
        test: None,
        preprocessing: None,
        augmentation: None,
        loader: Some(DataLoader {
            batch_size: 32,
            shuffle: true,
            num_workers: Some(4),
            pin_memory: Some(true),
            drop_last: Some(false),
            prefetch_factor: None,
        }),
        tokenizer: None,
        seq_len: None,
        input_column: None,
        output_column: None,
        max_length: None,
    }
}

fn default_model_config(source: &str) -> ModelConfig {
    ModelConfig {
        source: source.to_string(),
        format: None,
        architecture: None,
        freeze: None,
        device: Some("auto".to_string()),
        dtype: Some("float32".to_string()),
    }
}

fn default_optimizer_config() -> OptimizerConfig {
    OptimizerConfig {
        name: "adamw".to_string(),
        lr: 0.001,
        weight_decay: Some(0.01),
        betas: Some(vec![0.9, 0.999]),
        eps: Some(1e-8),
        amsgrad: None,
        momentum: None,
        nesterov: None,
        dampening: None,
        alpha: None,
        centered: None,
        param_groups: None,
    }
}

fn default_scheduler_config() -> SchedulerConfig {
    SchedulerConfig {
        name: "cosine_annealing".to_string(),
        warmup: Some(WarmupConfig { steps: Some(100), ratio: None, start_lr: Some(1e-7) }),
        t_max: Some(DEFAULT_COSINE_ANNEALING_T_MAX),
        eta_min: Some(1e-6),
        step_size: None,
        gamma: None,
        mode: None,
        factor: None,
        patience: None,
        threshold: None,
        max_lr: None,
        pct_start: None,
        anneal_strategy: None,
        div_factor: None,
        final_div_factor: None,
    }
}

fn default_training_config() -> TrainingConfig {
    TrainingConfig {
        epochs: Some(10),
        max_steps: None,
        duration: None,
        gradient: Some(GradientConfig {
            accumulation_steps: Some(1),
            clip_norm: Some(1.0),
            clip_value: None,
        }),
        mixed_precision: None,
        distributed: None,
        checkpoint: Some(CheckpointConfig {
            save_every: Some(1000),
            keep_last: Some(3),
            save_best: Some(true),
            metric: Some("val_loss".to_string()),
            mode: Some("min".to_string()),
        }),
        early_stopping: Some(EarlyStoppingConfig {
            enabled: true,
            metric: Some("val_loss".to_string()),
            patience: Some(5),
            min_delta: Some(0.001),
            mode: Some("min".to_string()),
        }),
        validation: None,
        deterministic: None,
        benchmark: None,
        curriculum: None,
    }
}

fn default_monitoring_config() -> MonitoringConfig {
    MonitoringConfig {
        terminal: Some(TerminalMonitor {
            enabled: true,
            refresh_rate: Some(100),
            metrics: Some(vec!["loss".to_string(), "accuracy".to_string()]),
            charts: None,
        }),
        tracking: None,
        system: None,
        alerts: None,
        drift_detection: None,
    }
}

fn default_output_config() -> OutputConfig {
    OutputConfig {
        dir: "./output/{{ name }}/{{ timestamp }}".to_string(),
        model: Some(ModelOutputConfig {
            format: Some("safetensors".to_string()),
            save_optimizer: Some(true),
            save_scheduler: Some(true),
        }),
        metrics: None,
        report: Some(ReportConfig {
            enabled: true,
            format: Some("markdown".to_string()),
            include_plots: Some(true),
        }),
        registry: None,
    }
}

fn generate_lora(name: &str, model: Option<&str>, data: Option<&str>) -> TrainingManifest {
    let mut manifest = generate_minimal(name, model, data);

    // Add LoRA configuration
    manifest.lora = Some(LoraConfig {
        enabled: true,
        rank: 16,
        alpha: 32.0,
        dropout: Some(0.05),
        target_modules: vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
        ],
        target_modules_pattern: None,
        bias: Some("none".to_string()),
        init_weights: Some("gaussian".to_string()),
        quantize_base: None,
        quantize_bits: None,
        double_quantize: None,
        quant_type: None,
    });

    // Adjust training params for LoRA
    if let Some(ref mut training) = manifest.training {
        training.epochs = Some(3);
        if let Some(ref mut grad) = training.gradient {
            grad.accumulation_steps = Some(4);
        }
    }

    // Lower learning rate for fine-tuning
    if let Some(ref mut optim) = manifest.optimizer {
        optim.lr = 0.0002;
    }

    // Use float16 for model
    if let Some(ref mut model_config) = manifest.model {
        model_config.dtype = Some("float16".to_string());
    }

    manifest
}

fn generate_qlora(name: &str, model: Option<&str>, data: Option<&str>) -> TrainingManifest {
    let mut manifest = generate_lora(name, model, data);

    // Enable QLoRA (quantized LoRA)
    if let Some(ref mut lora) = manifest.lora {
        lora.quantize_base = Some(true);
        lora.quantize_bits = Some(4);
        lora.double_quantize = Some(true);
        lora.quant_type = Some("nf4".to_string());
    }

    // Enable mixed precision
    if let Some(ref mut training) = manifest.training {
        training.mixed_precision = Some(MixedPrecisionConfig {
            enabled: true,
            dtype: Some("bfloat16".to_string()),
            loss_scale: Some("dynamic".to_string()),
        });
        // Increase gradient accumulation for memory efficiency
        if let Some(ref mut grad) = training.gradient {
            grad.accumulation_steps = Some(16);
        }
    }

    manifest
}

/// Build the full quantization config section.
fn full_quantize_config() -> QuantizeConfig {
    QuantizeConfig {
        enabled: false,
        bits: 8,
        scheme: Some("symmetric".to_string()),
        granularity: Some("per_channel".to_string()),
        group_size: Some(128),
        qat: None,
        calibration: None,
        exclude: Some(vec!["lm_head".to_string(), "embed_tokens".to_string()]),
    }
}

/// Build the full monitoring config with terminal, tracking, system, and alerts.
fn full_monitoring_config(name: &str) -> MonitoringConfig {
    MonitoringConfig {
        terminal: Some(TerminalMonitor {
            enabled: true,
            refresh_rate: Some(100),
            metrics: Some(vec![
                "loss".to_string(),
                "accuracy".to_string(),
                "learning_rate".to_string(),
                "throughput".to_string(),
            ]),
            charts: Some(vec![
                ChartConfig {
                    chart_type: "sparkline".to_string(),
                    metric: Some("loss".to_string()),
                    window: Some(100),
                    show_eta: None,
                },
                ChartConfig {
                    chart_type: "progress".to_string(),
                    metric: None,
                    window: None,
                    show_eta: Some(true),
                },
            ]),
        }),
        tracking: Some(TrackingConfig {
            enabled: true,
            backend: Some("trueno-db".to_string()),
            project: Some(name.to_string()),
            experiment: Some("{{ name }}-{{ timestamp }}".to_string()),
            tags: None,
        }),
        system: Some(SystemMonitorConfig {
            enabled: true,
            interval: Some(1000),
            metrics: Some(vec![
                "cpu_percent".to_string(),
                "memory_mb".to_string(),
                "gpu_utilization".to_string(),
                "gpu_memory_mb".to_string(),
            ]),
        }),
        alerts: Some(vec![
            AlertConfig {
                condition: "loss > 10".to_string(),
                action: "warn".to_string(),
                message: "Loss explosion detected".to_string(),
            },
            AlertConfig {
                condition: "gpu_memory > 0.95".to_string(),
                action: "halt".to_string(),
                message: "GPU OOM imminent".to_string(),
            },
        ]),
        drift_detection: None,
    }
}

/// Build the full callbacks list (checkpoint, LR monitor, gradient monitor).
fn full_callbacks_config() -> Vec<CallbackConfig> {
    vec![
        CallbackConfig {
            callback_type: CallbackType::Checkpoint,
            trigger: "epoch_end".to_string(),
            interval: None,
            config: None,
            script: None,
        },
        CallbackConfig {
            callback_type: CallbackType::LrMonitor,
            trigger: "step".to_string(),
            interval: None,
            config: None,
            script: None,
        },
        CallbackConfig {
            callback_type: CallbackType::GradientMonitor,
            trigger: "step".to_string(),
            interval: Some(100),
            config: None,
            script: None,
        },
    ]
}

/// Build the full output config with model, metrics, report, and registry.
fn full_output_config() -> OutputConfig {
    OutputConfig {
        dir: "./experiments/{{ name }}/{{ timestamp }}".to_string(),
        model: Some(ModelOutputConfig {
            format: Some("safetensors".to_string()),
            save_optimizer: Some(true),
            save_scheduler: Some(true),
        }),
        metrics: Some(MetricsOutputConfig {
            format: Some("parquet".to_string()),
            include: Some(vec![
                "train_loss".to_string(),
                "val_loss".to_string(),
                "accuracy".to_string(),
                "learning_rate".to_string(),
            ]),
        }),
        report: Some(ReportConfig {
            enabled: true,
            format: Some("markdown".to_string()),
            include_plots: Some(true),
        }),
        registry: Some(RegistryConfig {
            enabled: false,
            target: Some("pacha://models/{{ name }}:{{ version }}".to_string()),
            include_config: Some(true),
            include_metrics: Some(true),
        }),
    }
}

fn generate_full(name: &str, model: Option<&str>, data: Option<&str>) -> TrainingManifest {
    let mut manifest = generate_qlora(name, model, data);

    manifest.quantize = Some(full_quantize_config());
    manifest.monitoring = Some(full_monitoring_config(name));
    manifest.callbacks = Some(full_callbacks_config());
    manifest.output = Some(full_output_config());

    manifest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_minimal() {
        let manifest = generate_manifest(
            Template::Minimal,
            "test-exp",
            Some("model.safetensors"),
            Some("./data"),
        );
        assert_eq!(manifest.entrenar, "1.0");
        assert_eq!(manifest.name, "test-exp");
        assert!(manifest.lora.is_none());
        assert!(manifest.model.is_some());
    }

    #[test]
    fn test_generate_lora() {
        let manifest =
            generate_manifest(Template::Lora, "lora-exp", Some("hf://llama"), Some("hf://data"));
        assert!(manifest.lora.is_some());
        let lora = manifest.lora.expect("operation should succeed");
        assert!(lora.enabled);
        assert_eq!(lora.rank, 16);
        assert!(lora.quantize_base.is_none());
    }

    #[test]
    fn test_generate_qlora() {
        let manifest = generate_manifest(Template::Qlora, "qlora-exp", None, None);
        assert!(manifest.lora.is_some());
        let lora = manifest.lora.expect("operation should succeed");
        assert!(lora.quantize_base.expect("operation should succeed"));
        assert_eq!(lora.quantize_bits, Some(4));
        assert!(manifest
            .training
            .as_ref()
            .expect("operation should succeed")
            .mixed_precision
            .is_some());
    }

    #[test]
    fn test_generate_full() {
        let manifest = generate_manifest(Template::Full, "full-exp", None, None);
        assert!(manifest.lora.is_some());
        assert!(manifest.quantize.is_some());
        assert!(manifest.monitoring.is_some());
        assert!(manifest.callbacks.is_some());
        assert!(manifest.output.is_some());

        let monitoring = manifest.monitoring.expect("operation should succeed");
        assert!(monitoring.tracking.is_some());
        assert!(monitoring.system.is_some());
        assert!(monitoring.alerts.is_some());
    }

    #[test]
    fn test_generate_yaml_output() {
        let yaml = generate_yaml(Template::Minimal, "yaml-test", None, None, None, None);
        assert!(yaml.contains("entrenar: '1.0'") || yaml.contains("entrenar: \"1.0\""));
        assert!(yaml.contains("yaml-test"));
    }

    #[test]
    fn test_manifest_validates() {
        use super::super::validation::validate_manifest;

        // All templates should produce valid manifests
        for template in [Template::Minimal, Template::Lora, Template::Qlora, Template::Full] {
            let manifest = generate_manifest(template, "test", None, None);
            let result = validate_manifest(&manifest);
            assert!(result.is_ok(), "Template {template:?} produced invalid manifest: {result:?}");
        }
    }

    #[test]
    fn test_smart_defaults_lora_rank() {
        let manifest = generate_manifest_with_hints(
            Template::Lora,
            "smart-test",
            Some("Qwen/Qwen2.5-Coder-0.5B"),
            None,
            Some(32),   // small model rank
            Some(3e-4), // small model lr
        );
        let lora = manifest.lora.expect("operation should succeed");
        assert_eq!(lora.rank, 32);
        assert!((lora.alpha - 64.0).abs() < 0.01); // alpha = 2 * rank
        assert!((manifest.optimizer.expect("operation should succeed").lr - 3e-4).abs() < 1e-10);
    }

    #[test]
    fn test_smart_defaults_large_model() {
        let manifest = generate_manifest_with_hints(
            Template::Qlora,
            "large-test",
            Some("meta-llama/Llama-3-13B"),
            None,
            Some(128),
            Some(1e-4),
        );
        let lora = manifest.lora.expect("operation should succeed");
        assert_eq!(lora.rank, 128);
        assert!((lora.alpha - 256.0).abs() < 0.01);
    }

    #[test]
    fn test_smart_defaults_no_hints() {
        // Without hints, defaults should be unchanged
        let manifest =
            generate_manifest_with_hints(Template::Lora, "no-hints", None, None, None, None);
        let lora = manifest.lora.expect("operation should succeed");
        assert_eq!(lora.rank, 16); // original default
        assert!((lora.alpha - 32.0).abs() < 0.01);
    }

    #[test]
    fn test_minimal_has_no_publish() {
        let manifest = generate_manifest(Template::Minimal, "test", None, None);
        assert!(manifest.publish.is_none());
    }

    #[test]
    fn test_publish_config_yaml_roundtrip() {
        use super::super::manifest::PublishConfig;

        let yaml = r#"
            repo: "myuser/my-model"
            private: false
            model_card: true
            merge_adapters: true
            format: safetensors
        "#;
        let config: PublishConfig = serde_yaml::from_str(yaml).expect("config should be valid");
        assert_eq!(config.repo, "myuser/my-model");
        assert!(!config.private);
        assert!(config.model_card);
        assert!(config.merge_adapters);
        assert_eq!(config.format, "safetensors");
    }

    #[test]
    fn test_publish_config_defaults() {
        use super::super::manifest::PublishConfig;

        let yaml = r#"repo: "org/name""#;
        let config: PublishConfig = serde_yaml::from_str(yaml).expect("config should be valid");
        assert!(!config.private);
        assert!(config.model_card); // default true
        assert!(!config.merge_adapters);
        assert_eq!(config.format, "safetensors"); // default
    }

    #[test]
    fn test_manifest_with_publish_section() {
        let yaml = r#"
entrenar: "1.0"
name: test
version: "1.0.0"
publish:
  repo: myuser/my-model
  merge_adapters: true
"#;
        let manifest: TrainingManifest =
            serde_yaml::from_str(yaml).expect("operation should succeed");
        let publish = manifest.publish.expect("operation should succeed");
        assert_eq!(publish.repo, "myuser/my-model");
        assert!(publish.merge_adapters);
        assert!(publish.model_card); // default
    }
}
