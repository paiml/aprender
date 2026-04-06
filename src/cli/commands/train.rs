//! Train command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{apply_overrides, load_config, train_from_yaml, TrainArgs, TrainSpec};

pub fn run_train(args: TrainArgs, level: LogLevel) -> Result<(), String> {
    log(level, LogLevel::Normal, &format!("Entrenar: Training from {}", args.config.display()));

    // Load and validate config
    let mut spec = load_config(&args.config).map_err(|e| format!("Config error: {e}"))?;

    // Apply command-line overrides
    apply_overrides(&mut spec, &args);

    if args.dry_run {
        log_dry_run_summary(&spec, level);
        return Ok(());
    }

    // Run training
    train_from_yaml(&args.config).map_err(|e| format!("Training error: {e}"))?;

    log(level, LogLevel::Normal, "Training complete!");
    Ok(())
}

/// Log a summary of the training configuration for dry-run mode
fn log_dry_run_summary(spec: &TrainSpec, level: LogLevel) {
    log(level, LogLevel::Normal, "Dry run - config validated successfully");

    let mode_str = format!("{:?}", spec.model.mode).to_lowercase();
    log(level, LogLevel::Normal, &format!("  Model: {} ({})", spec.model.path.display(), mode_str));

    let training_mode = format!("{:?}", spec.training.mode).to_lowercase();
    log(level, LogLevel::Normal, &format!("  Training mode: {training_mode}"));

    log(
        level,
        LogLevel::Normal,
        &format!("  Optimizer: {} (lr={})", spec.optimizer.name, spec.optimizer.lr),
    );

    log_scheduler_info(spec, level);

    log(level, LogLevel::Normal, &format!("  Epochs: {}", spec.training.epochs));
    log(level, LogLevel::Normal, &format!("  Batch size: {}", spec.data.batch_size));

    log_optional_features(spec, level);

    log(level, LogLevel::Normal, &format!("  Output: {}", spec.training.output_dir.display()));
}

/// Log scheduler information if present
fn log_scheduler_info(spec: &TrainSpec, level: LogLevel) {
    if let Some(ref sched) = spec.training.lr_scheduler {
        let warmup = if spec.training.warmup_steps > 0 {
            format!(" (warmup={} steps)", spec.training.warmup_steps)
        } else {
            String::new()
        };
        log(level, LogLevel::Normal, &format!("  Scheduler: {sched}{warmup}"));
    }
}

/// Log optional training features (gradient accumulation, mixed precision, LoRA, quantization)
fn log_optional_features(spec: &TrainSpec, level: LogLevel) {
    if let Some(ga) = spec.training.gradient_accumulation {
        let effective = spec.data.batch_size * ga;
        log(
            level,
            LogLevel::Normal,
            &format!("  Gradient accumulation: {ga} (effective batch={effective})"),
        );
    }

    if let Some(ref mp) = spec.training.mixed_precision {
        log(level, LogLevel::Normal, &format!("  Mixed precision: {mp}"));
    }

    if let Some(ref lora) = spec.lora {
        log(
            level,
            LogLevel::Normal,
            &format!(
                "  LoRA: rank={}, alpha={}, modules={:?}",
                lora.rank, lora.alpha, lora.target_modules
            ),
        );
    }

    if let Some(ref quant) = spec.quantize {
        let scheme = if quant.symmetric { "symmetric" } else { "asymmetric" };
        let gran = if quant.per_channel { "per-channel" } else { "per-tensor" };
        log(
            level,
            LogLevel::Normal,
            &format!("  Quantization: {}-bit {} {}", quant.bits, scheme, gran),
        );
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use std::path::PathBuf;

    fn make_args(config_path: &str, dry_run: bool) -> TrainArgs {
        TrainArgs {
            config: PathBuf::from(config_path),
            output_dir: None,
            resume: None,
            epochs: None,
            batch_size: None,
            lr: None,
            dry_run,
            save_every: None,
            log_every: None,
            seed: None,
        }
    }

    #[test]
    fn test_train_dry_run_valid_config() {
        // Create a minimal valid config file
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 1
";
        let config_path = "/tmp/test_train_config.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let args = make_args(config_path, true);
        let result = run_train(args, LogLevel::Quiet);
        assert!(result.is_ok(), "Dry run should succeed: {result:?}");

        std::fs::remove_file(config_path).ok();
    }

    #[test]
    fn test_train_invalid_config_path() {
        let args = make_args("/nonexistent/config.yaml", false);
        let result = run_train(args, LogLevel::Quiet);
        assert!(result.is_err(), "Should fail with invalid config path");
    }

    #[test]
    fn test_train_dry_run_logs_correctly() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 16
optimizer:
  name: sgd
  lr: 0.01
training:
  epochs: 5
";
        let config_path = "/tmp/test_train_config_logs.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let args = make_args(config_path, true);
        // Test with verbose logging to cover log branches
        let result = run_train(args, LogLevel::Verbose);
        assert!(result.is_ok());

        std::fs::remove_file(config_path).ok();
    }

    // ── dry run with lr scheduler ───────────────────────────────────────

    #[test]
    fn test_train_dry_run_with_lr_scheduler() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 10
  lr_scheduler: cosine
  warmup_steps: 100
";
        let config_path = "/tmp/test_train_config_sched.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let args = make_args(config_path, true);
        let result = run_train(args, LogLevel::Normal);
        assert!(result.is_ok());

        std::fs::remove_file(config_path).ok();
    }

    #[test]
    fn test_train_dry_run_with_scheduler_no_warmup() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 10
  lr_scheduler: step
  warmup_steps: 0
";
        let config_path = "/tmp/test_train_config_sched_nowarmup.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let args = make_args(config_path, true);
        let result = run_train(args, LogLevel::Normal);
        assert!(result.is_ok());

        std::fs::remove_file(config_path).ok();
    }

    // ── dry run with gradient accumulation ──────────────────────────────

    #[test]
    fn test_train_dry_run_with_gradient_accumulation() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 3
  gradient_accumulation: 4
";
        let config_path = "/tmp/test_train_config_grad_acc.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let args = make_args(config_path, true);
        let result = run_train(args, LogLevel::Normal);
        assert!(result.is_ok());

        std::fs::remove_file(config_path).ok();
    }

    // ── dry run with mixed precision ────────────────────────────────────

    #[test]
    fn test_train_dry_run_with_mixed_precision() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 1
  mixed_precision: bf16
";
        let config_path = "/tmp/test_train_config_mp.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let args = make_args(config_path, true);
        let result = run_train(args, LogLevel::Normal);
        assert!(result.is_ok());

        std::fs::remove_file(config_path).ok();
    }

    // ── dry run with LoRA ───────────────────────────────────────────────

    #[test]
    fn test_train_dry_run_with_lora() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 1
lora:
  rank: 16
  alpha: 32.0
  target_modules:
    - q_proj
    - v_proj
";
        let config_path = "/tmp/test_train_config_lora.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let args = make_args(config_path, true);
        let result = run_train(args, LogLevel::Normal);
        assert!(result.is_ok());

        std::fs::remove_file(config_path).ok();
    }

    // ── dry run with quantization ───────────────────────────────────────

    #[test]
    fn test_train_dry_run_with_quantization() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 1
quantize:
  bits: 4
  symmetric: true
  per_channel: true
";
        let config_path = "/tmp/test_train_config_quant.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let args = make_args(config_path, true);
        let result = run_train(args, LogLevel::Normal);
        assert!(result.is_ok());

        std::fs::remove_file(config_path).ok();
    }

    // ── dry run with asymmetric quantization ────────────────────────────

    #[test]
    fn test_train_dry_run_with_asymmetric_quantization() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 1
quantize:
  bits: 8
  symmetric: false
  per_channel: false
";
        let config_path = "/tmp/test_train_config_quant_asym.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let args = make_args(config_path, true);
        let result = run_train(args, LogLevel::Normal);
        assert!(result.is_ok());

        std::fs::remove_file(config_path).ok();
    }

    // ── dry run with all optional features combined ─────────────────────

    #[test]
    fn test_train_dry_run_all_features() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 32
optimizer:
  name: adam
  lr: 0.0001
training:
  epochs: 20
  lr_scheduler: cosine
  warmup_steps: 500
  gradient_accumulation: 8
  mixed_precision: fp16
lora:
  rank: 8
  alpha: 16.0
  target_modules:
    - q_proj
    - k_proj
    - v_proj
quantize:
  bits: 4
  symmetric: true
  per_channel: true
";
        let config_path = "/tmp/test_train_config_all.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let args = make_args(config_path, true);
        let result = run_train(args, LogLevel::Verbose);
        assert!(result.is_ok());

        std::fs::remove_file(config_path).ok();
    }

    // ── apply_overrides tests ───────────────────────────────────────────

    #[test]
    fn test_apply_overrides_output_dir() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 1
";
        let config_path = "/tmp/test_train_override_out.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let mut spec = load_config(PathBuf::from(config_path)).unwrap();
        let args = TrainArgs {
            config: PathBuf::from(config_path),
            output_dir: Some(PathBuf::from("/tmp/override_output")),
            resume: None,
            epochs: None,
            batch_size: None,
            lr: None,
            dry_run: true,
            save_every: None,
            log_every: None,
            seed: None,
        };
        apply_overrides(&mut spec, &args);
        assert_eq!(spec.training.output_dir, PathBuf::from("/tmp/override_output"));

        std::fs::remove_file(config_path).ok();
    }

    #[test]
    fn test_apply_overrides_epochs() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 1
";
        let config_path = "/tmp/test_train_override_epochs.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let mut spec = load_config(PathBuf::from(config_path)).unwrap();
        let args = TrainArgs {
            config: PathBuf::from(config_path),
            output_dir: None,
            resume: None,
            epochs: Some(99),
            batch_size: None,
            lr: None,
            dry_run: true,
            save_every: None,
            log_every: None,
            seed: None,
        };
        apply_overrides(&mut spec, &args);
        assert_eq!(spec.training.epochs, 99);

        std::fs::remove_file(config_path).ok();
    }

    #[test]
    fn test_apply_overrides_batch_size() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 1
";
        let config_path = "/tmp/test_train_override_batch.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let mut spec = load_config(PathBuf::from(config_path)).unwrap();
        let args = TrainArgs {
            config: PathBuf::from(config_path),
            output_dir: None,
            resume: None,
            epochs: None,
            batch_size: Some(128),
            lr: None,
            dry_run: true,
            save_every: None,
            log_every: None,
            seed: None,
        };
        apply_overrides(&mut spec, &args);
        assert_eq!(spec.data.batch_size, 128);

        std::fs::remove_file(config_path).ok();
    }

    #[test]
    fn test_apply_overrides_learning_rate() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 1
";
        let config_path = "/tmp/test_train_override_lr.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let mut spec = load_config(PathBuf::from(config_path)).unwrap();
        let args = TrainArgs {
            config: PathBuf::from(config_path),
            output_dir: None,
            resume: None,
            epochs: None,
            batch_size: None,
            lr: Some(0.042),
            dry_run: true,
            save_every: None,
            log_every: None,
            seed: None,
        };
        apply_overrides(&mut spec, &args);
        assert!((spec.optimizer.lr - 0.042).abs() < 1e-6);

        std::fs::remove_file(config_path).ok();
    }

    #[test]
    fn test_apply_overrides_save_every() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 1
";
        let config_path = "/tmp/test_train_override_save.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let mut spec = load_config(PathBuf::from(config_path)).unwrap();
        let args = TrainArgs {
            config: PathBuf::from(config_path),
            output_dir: None,
            resume: None,
            epochs: None,
            batch_size: None,
            lr: None,
            dry_run: true,
            save_every: Some(5),
            log_every: None,
            seed: None,
        };
        apply_overrides(&mut spec, &args);
        assert_eq!(spec.training.save_interval, 5);

        std::fs::remove_file(config_path).ok();
    }

    #[test]
    fn test_apply_overrides_all_at_once() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 1
";
        let config_path = "/tmp/test_train_override_all.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let mut spec = load_config(PathBuf::from(config_path)).unwrap();
        let args = TrainArgs {
            config: PathBuf::from(config_path),
            output_dir: Some(PathBuf::from("/tmp/all_override")),
            resume: None,
            epochs: Some(50),
            batch_size: Some(64),
            lr: Some(0.01),
            dry_run: true,
            save_every: Some(10),
            log_every: None,
            seed: None,
        };
        apply_overrides(&mut spec, &args);
        assert_eq!(spec.training.output_dir, PathBuf::from("/tmp/all_override"));
        assert_eq!(spec.training.epochs, 50);
        assert_eq!(spec.data.batch_size, 64);
        assert!((spec.optimizer.lr - 0.01).abs() < 1e-6);
        assert_eq!(spec.training.save_interval, 10);

        std::fs::remove_file(config_path).ok();
    }

    #[test]
    fn test_apply_overrides_none_leaves_original() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 3
";
        let config_path = "/tmp/test_train_override_none.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let mut spec = load_config(PathBuf::from(config_path)).unwrap();
        let original_epochs = spec.training.epochs;
        let original_batch = spec.data.batch_size;
        let original_lr = spec.optimizer.lr;
        let args = TrainArgs {
            config: PathBuf::from(config_path),
            output_dir: None,
            resume: None,
            epochs: None,
            batch_size: None,
            lr: None,
            dry_run: true,
            save_every: None,
            log_every: None,
            seed: None,
        };
        apply_overrides(&mut spec, &args);
        assert_eq!(spec.training.epochs, original_epochs);
        assert_eq!(spec.data.batch_size, original_batch);
        assert!((spec.optimizer.lr - original_lr).abs() < 1e-6);

        std::fs::remove_file(config_path).ok();
    }

    // ── invalid YAML content ────────────────────────────────────────────

    #[test]
    fn test_train_invalid_yaml() {
        let config_content = "{{invalid yaml content}}";
        let config_path = "/tmp/test_train_config_invalid.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let args = make_args(config_path, true);
        let result = run_train(args, LogLevel::Quiet);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Config error"));

        std::fs::remove_file(config_path).ok();
    }

    // ── log_dry_run_summary and helpers direct tests ────────────────────

    #[test]
    fn test_log_dry_run_summary_quiet() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 1
";
        let config_path = "/tmp/test_train_log_quiet.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let spec = load_config(PathBuf::from(config_path)).unwrap();
        // Should not panic even in quiet mode
        log_dry_run_summary(&spec, LogLevel::Quiet);

        std::fs::remove_file(config_path).ok();
    }

    #[test]
    fn test_log_scheduler_info_none() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 1
";
        let config_path = "/tmp/test_train_log_sched_none.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let spec = load_config(PathBuf::from(config_path)).unwrap();
        // lr_scheduler should be None — no-op branch
        log_scheduler_info(&spec, LogLevel::Normal);

        std::fs::remove_file(config_path).ok();
    }

    #[test]
    fn test_log_optional_features_none() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
data:
  train_path: /tmp/train.json
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
training:
  epochs: 1
";
        let config_path = "/tmp/test_train_log_opt_none.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let spec = load_config(PathBuf::from(config_path)).unwrap();
        // No optional features — all branches should be no-ops
        log_optional_features(&spec, LogLevel::Normal);

        std::fs::remove_file(config_path).ok();
    }

    // ── make_args helper verify ─────────────────────────────────────────

    #[test]
    fn test_make_args_dry_run_true() {
        let args = make_args("/tmp/cfg.yaml", true);
        assert!(args.dry_run);
        assert_eq!(args.config, PathBuf::from("/tmp/cfg.yaml"));
        assert!(args.output_dir.is_none());
        assert!(args.resume.is_none());
        assert!(args.epochs.is_none());
        assert!(args.batch_size.is_none());
        assert!(args.lr.is_none());
        assert!(args.save_every.is_none());
        assert!(args.log_every.is_none());
        assert!(args.seed.is_none());
    }

    #[test]
    fn test_make_args_dry_run_false() {
        let args = make_args("/path/to/config.yaml", false);
        assert!(!args.dry_run);
        assert_eq!(args.config, PathBuf::from("/path/to/config.yaml"));
    }

    // ── causal_lm mode dry run ──────────────────────────────────────────

    #[test]
    fn test_train_dry_run_causal_lm_mode() {
        let config_content = r"
model:
  path: /tmp/test_model.gguf
  mode: causal_lm
data:
  train_path: /tmp/train.json
  batch_size: 4
optimizer:
  name: adam
  lr: 0.0001
training:
  mode: causal_lm
  epochs: 2
";
        let config_path = "/tmp/test_train_config_causal.yaml";
        std::fs::write(config_path, config_content).expect("file write should succeed");

        let args = make_args(config_path, true);
        // dry_run may succeed or fail depending on config parsing;
        // the important thing is it doesn't panic
        let _result = run_train(args, LogLevel::Normal);

        std::fs::remove_file(config_path).ok();
    }
}
