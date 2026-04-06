//! Tests for CLI argument parsing and validation

use super::*;
use std::path::PathBuf;

#[test]
fn test_parse_train_command() {
    let cli = parse_args(["entrenar", "train", "config.yaml"]).expect("parsing should succeed");
    match cli.command {
        Command::Train(args) => {
            assert_eq!(args.config, PathBuf::from("config.yaml"));
            assert!(!args.dry_run);
        }
        _ => panic!("Expected Train command"),
    }
}

#[test]
fn test_parse_train_with_overrides() {
    let cli = parse_args([
        "entrenar",
        "train",
        "config.yaml",
        "--epochs",
        "10",
        "--batch-size",
        "32",
        "--lr",
        "0.001",
        "--output-dir",
        "./output",
    ])
    .expect("operation should succeed");

    match cli.command {
        Command::Train(args) => {
            assert_eq!(args.epochs, Some(10));
            assert_eq!(args.batch_size, Some(32));
            assert!((args.lr.expect("operation should succeed") - 0.001).abs() < 1e-6);
            assert_eq!(args.output_dir, Some(PathBuf::from("./output")));
        }
        _ => panic!("Expected Train command"),
    }
}

#[test]
fn test_parse_train_with_resume() {
    let cli = parse_args(["entrenar", "train", "config.yaml", "--resume", "checkpoint.json"])
        .expect("parsing should succeed");

    match cli.command {
        Command::Train(args) => {
            assert_eq!(args.resume, Some(PathBuf::from("checkpoint.json")));
        }
        _ => panic!("Expected Train command"),
    }
}

#[test]
fn test_parse_train_dry_run() {
    let cli = parse_args(["entrenar", "train", "config.yaml", "--dry-run"])
        .expect("parsing should succeed");
    match cli.command {
        Command::Train(args) => {
            assert!(args.dry_run);
        }
        _ => panic!("Expected Train command"),
    }
}

#[test]
fn test_parse_validate_command() {
    let cli = parse_args(["entrenar", "validate", "config.yaml"]).expect("parsing should succeed");
    match cli.command {
        Command::Validate(args) => {
            assert_eq!(args.config, PathBuf::from("config.yaml"));
            assert!(!args.detailed);
        }
        _ => panic!("Expected Validate command"),
    }
}

#[test]
fn test_parse_validate_detailed() {
    let cli = parse_args(["entrenar", "validate", "config.yaml", "--detailed"])
        .expect("parsing should succeed");
    match cli.command {
        Command::Validate(args) => {
            assert!(args.detailed);
        }
        _ => panic!("Expected Validate command"),
    }
}

#[test]
fn test_parse_info_command() {
    let cli = parse_args(["entrenar", "info", "config.yaml"]).expect("parsing should succeed");
    match cli.command {
        Command::Info(args) => {
            assert_eq!(args.config, PathBuf::from("config.yaml"));
            assert_eq!(args.format, OutputFormat::Text);
        }
        _ => panic!("Expected Info command"),
    }
}

#[test]
fn test_parse_info_json_format() {
    let cli = parse_args(["entrenar", "info", "config.yaml", "--format", "json"])
        .expect("parsing should succeed");
    match cli.command {
        Command::Info(args) => {
            assert_eq!(args.format, OutputFormat::Json);
        }
        _ => panic!("Expected Info command"),
    }
}

#[test]
fn test_parse_quantize_command() {
    let cli = parse_args(["entrenar", "quantize", "model.gguf", "--output", "model_q4.gguf"])
        .expect("parsing should succeed");

    match cli.command {
        Command::Quantize(args) => {
            assert_eq!(args.model, PathBuf::from("model.gguf"));
            assert_eq!(args.output, PathBuf::from("model_q4.gguf"));
            assert_eq!(args.bits, 4);
            assert_eq!(args.method, QuantMethod::Symmetric);
        }
        _ => panic!("Expected Quantize command"),
    }
}

#[test]
fn test_parse_quantize_with_options() {
    let cli = parse_args([
        "entrenar",
        "quantize",
        "model.gguf",
        "--output",
        "model_q8.gguf",
        "--bits",
        "8",
        "--method",
        "asymmetric",
        "--per-channel",
    ])
    .expect("operation should succeed");

    match cli.command {
        Command::Quantize(args) => {
            assert_eq!(args.bits, 8);
            assert_eq!(args.method, QuantMethod::Asymmetric);
            assert!(args.per_channel);
        }
        _ => panic!("Expected Quantize command"),
    }
}

#[test]
fn test_parse_merge_command() {
    let cli =
        parse_args(["entrenar", "merge", "model1.gguf", "model2.gguf", "--output", "merged.gguf"])
            .expect("operation should succeed");

    match cli.command {
        Command::Merge(args) => {
            assert_eq!(args.models.len(), 2);
            assert_eq!(args.output, PathBuf::from("merged.gguf"));
            assert_eq!(args.method, MergeMethod::Ties);
        }
        _ => panic!("Expected Merge command"),
    }
}

#[test]
fn test_parse_merge_slerp() {
    let cli = parse_args([
        "entrenar",
        "merge",
        "model1.gguf",
        "model2.gguf",
        "--output",
        "merged.gguf",
        "--method",
        "slerp",
        "--weight",
        "0.7",
    ])
    .expect("operation should succeed");

    match cli.command {
        Command::Merge(args) => {
            assert_eq!(args.method, MergeMethod::Slerp);
            assert!((args.weight.expect("operation should succeed") - 0.7).abs() < 1e-6);
        }
        _ => panic!("Expected Merge command"),
    }
}

#[test]
fn test_parse_merge_multiple_models() {
    let cli = parse_args([
        "entrenar",
        "merge",
        "model1.gguf",
        "model2.gguf",
        "model3.gguf",
        "--output",
        "merged.gguf",
        "--method",
        "average",
    ])
    .expect("operation should succeed");

    match cli.command {
        Command::Merge(args) => {
            assert_eq!(args.models.len(), 3);
            assert_eq!(args.method, MergeMethod::Average);
        }
        _ => panic!("Expected Merge command"),
    }
}

#[test]
fn test_global_verbose_flag() {
    let cli =
        parse_args(["entrenar", "-v", "train", "config.yaml"]).expect("parsing should succeed");
    assert!(cli.verbose);
    assert!(!cli.quiet);
}

#[test]
fn test_global_quiet_flag() {
    let cli =
        parse_args(["entrenar", "-q", "train", "config.yaml"]).expect("parsing should succeed");
    assert!(!cli.verbose);
    assert!(cli.quiet);
}

#[test]
fn test_missing_config_file() {
    let result = parse_args(["entrenar", "train"]);
    assert!(result.is_err());
}

#[test]
fn test_unknown_command() {
    let result = parse_args(["entrenar", "unknown"]);
    assert!(result.is_err());
}

#[test]
fn test_apply_overrides_output_dir() {
    let mut spec = create_test_spec();
    let args = TrainArgs {
        config: PathBuf::from("config.yaml"),
        output_dir: Some(PathBuf::from("./custom_output")),
        resume: None,
        epochs: None,
        batch_size: None,
        lr: None,
        save_every: None,
        log_every: None,
        dry_run: false,
        seed: None,
    };
    apply_overrides(&mut spec, &args);
    assert_eq!(spec.training.output_dir, PathBuf::from("./custom_output"));
}

#[test]
fn test_apply_overrides_epochs() {
    let mut spec = create_test_spec();
    let args = TrainArgs {
        config: PathBuf::from("config.yaml"),
        output_dir: None,
        resume: None,
        epochs: Some(50),
        batch_size: None,
        lr: None,
        save_every: None,
        log_every: None,
        dry_run: false,
        seed: None,
    };
    apply_overrides(&mut spec, &args);
    assert_eq!(spec.training.epochs, 50);
}

#[test]
fn test_apply_overrides_batch_size() {
    let mut spec = create_test_spec();
    let args = TrainArgs {
        config: PathBuf::from("config.yaml"),
        output_dir: None,
        resume: None,
        epochs: None,
        batch_size: Some(64),
        lr: None,
        save_every: None,
        log_every: None,
        dry_run: false,
        seed: None,
    };
    apply_overrides(&mut spec, &args);
    assert_eq!(spec.data.batch_size, 64);
}

#[test]
fn test_apply_overrides_lr() {
    let mut spec = create_test_spec();
    let args = TrainArgs {
        config: PathBuf::from("config.yaml"),
        output_dir: None,
        resume: None,
        epochs: None,
        batch_size: None,
        lr: Some(0.0001),
        save_every: None,
        log_every: None,
        dry_run: false,
        seed: None,
    };
    apply_overrides(&mut spec, &args);
    assert!((spec.optimizer.lr - 0.0001).abs() < 1e-8);
}

#[test]
fn test_apply_overrides_save_every() {
    let mut spec = create_test_spec();
    let args = TrainArgs {
        config: PathBuf::from("config.yaml"),
        output_dir: None,
        resume: None,
        epochs: None,
        batch_size: None,
        lr: None,
        save_every: Some(5),
        log_every: None,
        dry_run: false,
        seed: None,
    };
    apply_overrides(&mut spec, &args);
    assert_eq!(spec.training.save_interval, 5);
}

#[test]
fn test_apply_overrides_all() {
    let mut spec = create_test_spec();
    let args = TrainArgs {
        config: PathBuf::from("config.yaml"),
        output_dir: Some(PathBuf::from("./all_overrides")),
        resume: Some(PathBuf::from("checkpoint.json")),
        epochs: Some(100),
        batch_size: Some(128),
        lr: Some(0.01),
        save_every: Some(10),
        log_every: Some(50),
        dry_run: true,
        seed: Some(42),
    };
    apply_overrides(&mut spec, &args);
    assert_eq!(spec.training.output_dir, PathBuf::from("./all_overrides"));
    assert_eq!(spec.training.epochs, 100);
    assert_eq!(spec.data.batch_size, 128);
    assert!((spec.optimizer.lr - 0.01).abs() < 1e-8);
    assert_eq!(spec.training.save_interval, 10);
}

fn create_test_spec() -> crate::config::TrainSpec {
    crate::config::TrainSpec {
        model: crate::config::ModelRef { path: PathBuf::from("model.gguf"), ..Default::default() },
        data: crate::config::DataConfig {
            train: PathBuf::from("train.parquet"),
            batch_size: 8,
            ..Default::default()
        },
        optimizer: crate::config::OptimSpec {
            name: "adam".to_string(),
            lr: 0.001,
            params: Default::default(),
        },
        lora: None,
        quantize: None,
        merge: None,
        training: Default::default(),
        publish: None,
    }
}

#[test]
fn test_verbose_and_quiet_flags() {
    let cli = parse_args(["entrenar", "--verbose", "train", "config.yaml"])
        .expect("parsing should succeed");
    assert!(cli.verbose);
    assert!(!cli.quiet);

    let cli = parse_args(["entrenar", "--quiet", "train", "config.yaml"])
        .expect("parsing should succeed");
    assert!(!cli.verbose);
    assert!(cli.quiet);
}

#[test]
fn test_info_yaml_format() {
    let cli = parse_args(["entrenar", "info", "config.yaml", "--format", "yaml"])
        .expect("parsing should succeed");
    match cli.command {
        Command::Info(args) => {
            assert_eq!(args.format, OutputFormat::Yaml);
        }
        _ => panic!("Expected Info command"),
    }
}

#[test]
fn test_parse_init_command() {
    let cli = parse_args(["entrenar", "init"]).expect("parsing should succeed");
    match cli.command {
        Command::Init(args) => {
            assert_eq!(args.template, InitTemplate::Minimal);
            assert_eq!(args.name, "my-experiment");
        }
        _ => panic!("Expected Init command"),
    }
}

#[test]
fn test_parse_init_with_template() {
    let cli = parse_args(["entrenar", "init", "--template", "lora", "--name", "test-exp"])
        .expect("parsing should succeed");
    match cli.command {
        Command::Init(args) => {
            assert_eq!(args.template, InitTemplate::Lora);
            assert_eq!(args.name, "test-exp");
        }
        _ => panic!("Expected Init command"),
    }
}
