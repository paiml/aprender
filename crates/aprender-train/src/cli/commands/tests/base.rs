//! Base CLI command tests
//!
//! Tests for CLI command implementations to ensure coverage.

use super::super::*;
use crate::cli::LogLevel;
use crate::config::*;
use std::path::PathBuf;
use tempfile::TempDir;

/// Create a minimal valid config file for testing
pub(super) fn create_test_config(dir: &TempDir) -> PathBuf {
    let config_path = dir.path().join("test_config.yaml");
    let model_path = dir.path().join("model.safetensors");
    let data_path = dir.path().join("train.parquet");
    let output_path = dir.path().join("output");

    // Create dummy files
    std::fs::write(&model_path, b"dummy").expect("file write should succeed");
    std::fs::write(&data_path, b"dummy").expect("file write should succeed");
    std::fs::create_dir_all(&output_path).expect("operation should succeed");

    let config = format!(
        r"
model:
  path: {}
  layers: [q_proj, v_proj]

data:
  train: {}
  batch_size: 4
  seq_len: 512

optimizer:
  name: adamw
  lr: 0.0001

training:
  epochs: 1
  output_dir: {}
",
        model_path.display(),
        data_path.display(),
        output_path.display()
    );

    std::fs::write(&config_path, config).expect("file write should succeed");
    config_path
}

#[test]
fn test_validate_command_basic() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let config_path = create_test_config(&dir);

    let args = ValidateArgs { config: config_path, detailed: false };

    let result = validate::run_validate(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_validate_command_detailed() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let config_path = create_test_config(&dir);

    let args = ValidateArgs { config: config_path, detailed: true };

    let result = validate::run_validate(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_validate_command_missing_file() {
    let args = ValidateArgs { config: PathBuf::from("/nonexistent/config.yaml"), detailed: false };

    let result = validate::run_validate(args, LogLevel::Quiet);
    assert!(result.is_err());
}

#[test]
fn test_info_command() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let config_path = create_test_config(&dir);

    let args = InfoArgs { config: config_path, format: OutputFormat::Text };

    let result = info::run_info(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_init_command() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let output_path = dir.path().join("new_config.yaml");

    let args = InitArgs {
        name: "test_project".to_string(),
        output: Some(output_path.clone()),
        template: InitTemplate::Minimal,
        model: None,
        base: None,
        method: None,
        data: None,
    };

    let result = init::run_init(args, LogLevel::Quiet);
    assert!(result.is_ok());
    assert!(output_path.exists());
}

#[test]
fn test_init_command_with_lora_template() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let output_path = dir.path().join("lora_config.yaml");

    let args = InitArgs {
        name: "lora_project".to_string(),
        output: Some(output_path.clone()),
        template: InitTemplate::Lora,
        model: Some("/path/to/model".to_string()),
        base: None,
        method: None,
        data: Some("/path/to/data".to_string()),
    };

    let result = init::run_init(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_completion_command_bash() {
    let args = CompletionArgs { shell: ShellType::Bash };

    let result = completion::run_completion(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_completion_command_zsh() {
    let args = CompletionArgs { shell: ShellType::Zsh };

    let result = completion::run_completion(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_completion_command_fish() {
    let args = CompletionArgs { shell: ShellType::Fish };

    let result = completion::run_completion(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_run_command_quiet() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let output_path = dir.path().join("quiet_test.yaml");

    let cli = Cli {
        verbose: false,
        quiet: true,
        command: Command::Init(InitArgs {
            name: "test".to_string(),
            output: Some(output_path),
            template: InitTemplate::Minimal,
            model: None,
            base: None,
            method: None,
            data: None,
        }),
    };

    let result = run_command(cli);
    assert!(result.is_ok());
}

#[test]
fn test_run_command_verbose() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let config_path = create_test_config(&dir);

    let cli = Cli {
        verbose: true,
        quiet: false,
        command: Command::Validate(ValidateArgs { config: config_path, detailed: false }),
    };

    let result = run_command(cli);
    assert!(result.is_ok());
}

#[test]
fn test_log_levels() {
    assert!(matches!(LogLevel::Quiet, LogLevel::Quiet));
    assert!(matches!(LogLevel::Normal, LogLevel::Normal));
    assert!(matches!(LogLevel::Verbose, LogLevel::Verbose));
}

#[test]
fn test_init_template_from_str() {
    use std::str::FromStr;

    assert_eq!(
        InitTemplate::from_str("minimal").expect("operation should succeed"),
        InitTemplate::Minimal
    );
    assert_eq!(
        InitTemplate::from_str("lora").expect("operation should succeed"),
        InitTemplate::Lora
    );
    assert_eq!(
        InitTemplate::from_str("qlora").expect("operation should succeed"),
        InitTemplate::Qlora
    );
    assert_eq!(
        InitTemplate::from_str("full").expect("operation should succeed"),
        InitTemplate::Full
    );
    assert!(InitTemplate::from_str("invalid").is_err());
}

#[test]
fn test_shell_type_from_str() {
    use std::str::FromStr;

    assert_eq!(ShellType::from_str("bash").expect("operation should succeed"), ShellType::Bash);
    assert_eq!(ShellType::from_str("zsh").expect("operation should succeed"), ShellType::Zsh);
    assert_eq!(ShellType::from_str("fish").expect("operation should succeed"), ShellType::Fish);
    assert!(ShellType::from_str("invalid").is_err());
}

// ============================================================================
// Inspect command tests
// ============================================================================

/// Create a minimal valid SafeTensors file for testing
pub(super) fn create_safetensors_file(dir: &TempDir, name: &str) -> PathBuf {
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;

    let path = dir.path().join(name);

    // Create a simple tensor with F32 data
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();
    let shape = vec![2, 2];

    let view = TensorView::new(Dtype::F32, shape, &bytes).expect("operation should succeed");
    let tensors: Vec<(&str, TensorView<'_>)> = vec![("test_tensor", view)];

    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), "test".to_string());

    let safetensor_bytes =
        safetensors::serialize(tensors, Some(metadata)).expect("operation should succeed");
    std::fs::write(&path, safetensor_bytes).expect("file write should succeed");
    path
}

#[test]
fn test_inspect_command_safetensors() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = create_safetensors_file(&dir, "model.safetensors");

    let args = InspectArgs {
        input: model_path,
        mode: InspectMode::Summary,
        columns: None,
        z_threshold: 3.0,
    };

    let result = inspect::run_inspect(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_inspect_command_safetensors_verbose() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = create_safetensors_file(&dir, "model.safetensors");

    let args = InspectArgs {
        input: model_path,
        mode: InspectMode::Summary,
        columns: None,
        z_threshold: 3.0,
    };

    let result = inspect::run_inspect(args, LogLevel::Verbose);
    assert!(result.is_ok());
}

#[test]
fn test_inspect_command_gguf() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let gguf_path = dir.path().join("model.gguf");
    std::fs::write(&gguf_path, b"dummy gguf data").expect("file write should succeed");

    let args = InspectArgs {
        input: gguf_path,
        mode: InspectMode::Summary,
        columns: None,
        z_threshold: 3.0,
    };

    let result = inspect::run_inspect(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_inspect_command_parquet() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let parquet_path = dir.path().join("data.parquet");
    std::fs::write(&parquet_path, b"dummy parquet data").expect("file write should succeed");

    let args = InspectArgs {
        input: parquet_path,
        mode: InspectMode::Summary,
        columns: None,
        z_threshold: 3.0,
    };

    let result = inspect::run_inspect(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_inspect_command_csv() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let csv_path = dir.path().join("data.csv");
    std::fs::write(&csv_path, b"col1,col2\n1,2\n3,4").expect("file write should succeed");

    let args = InspectArgs {
        input: csv_path,
        mode: InspectMode::Summary,
        columns: None,
        z_threshold: 3.0,
    };

    let result = inspect::run_inspect(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_inspect_command_outliers_mode() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let csv_path = dir.path().join("data.csv");
    std::fs::write(&csv_path, b"col1,col2\n1,2\n3,4").expect("file write should succeed");

    let args = InspectArgs {
        input: csv_path,
        mode: InspectMode::Outliers,
        columns: None,
        z_threshold: 2.5,
    };

    let result = inspect::run_inspect(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_inspect_command_distribution_mode() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let csv_path = dir.path().join("data.csv");
    std::fs::write(&csv_path, b"col1,col2\n1,2\n3,4").expect("file write should succeed");

    let args = InspectArgs {
        input: csv_path,
        mode: InspectMode::Distribution,
        columns: None,
        z_threshold: 3.0,
    };

    let result = inspect::run_inspect(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_inspect_command_schema_mode() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let csv_path = dir.path().join("data.csv");
    std::fs::write(&csv_path, b"col1,col2\n1,2\n3,4").expect("file write should succeed");

    let args =
        InspectArgs { input: csv_path, mode: InspectMode::Schema, columns: None, z_threshold: 3.0 };

    let result = inspect::run_inspect(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_inspect_command_missing_file() {
    let args = InspectArgs {
        input: PathBuf::from("/nonexistent/model.safetensors"),
        mode: InspectMode::Summary,
        columns: None,
        z_threshold: 3.0,
    };

    let result = inspect::run_inspect(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

#[test]
fn test_inspect_command_unsupported_format() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let txt_path = dir.path().join("data.txt");
    std::fs::write(&txt_path, b"some text data").expect("file write should succeed");

    let args = InspectArgs {
        input: txt_path,
        mode: InspectMode::Summary,
        columns: None,
        z_threshold: 3.0,
    };

    let result = inspect::run_inspect(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Unsupported file format"));
}

// ============================================================================
// Bench command tests
// ============================================================================

#[test]
fn test_bench_command_basic() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = dir.path().join("model.bin");
    std::fs::write(&model_path, b"dummy model").expect("file write should succeed");

    let args = BenchArgs {
        input: model_path,
        warmup: 1,
        iterations: 5,
        batch_sizes: "1,2".to_string(),
        format: OutputFormat::Text,
    };

    let result = bench::run_bench(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_bench_command_json_output() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = dir.path().join("model.bin");
    std::fs::write(&model_path, b"dummy model").expect("file write should succeed");

    let args = BenchArgs {
        input: model_path,
        warmup: 1,
        iterations: 5,
        batch_sizes: "1".to_string(),
        format: OutputFormat::Json,
    };

    let result = bench::run_bench(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_bench_command_invalid_batch_sizes() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = dir.path().join("model.bin");
    std::fs::write(&model_path, b"dummy model").expect("file write should succeed");

    let args = BenchArgs {
        input: model_path,
        warmup: 1,
        iterations: 5,
        batch_sizes: "invalid,batch".to_string(),
        format: OutputFormat::Text,
    };

    let result = bench::run_bench(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Invalid batch sizes"));
}

// ============================================================================
// Audit command tests
// ============================================================================

#[test]
fn test_audit_command_bias() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = dir.path().join("model.bin");
    std::fs::write(&model_path, b"dummy model").expect("file write should succeed");

    let args = AuditArgs {
        input: model_path,
        audit_type: AuditType::Bias,
        protected_attr: None,
        threshold: 0.8,
        format: OutputFormat::Text,
    };

    let result = audit::run_audit(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_audit_command_bias_with_protected_attr() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = dir.path().join("model.bin");
    std::fs::write(&model_path, b"dummy model").expect("file write should succeed");

    let args = AuditArgs {
        input: model_path,
        audit_type: AuditType::Bias,
        protected_attr: Some("gender".to_string()),
        threshold: 0.8,
        format: OutputFormat::Text,
    };

    let result = audit::run_audit(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_audit_command_bias_json_output() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = dir.path().join("model.bin");
    std::fs::write(&model_path, b"dummy model").expect("file write should succeed");

    let args = AuditArgs {
        input: model_path,
        audit_type: AuditType::Bias,
        protected_attr: None,
        threshold: 0.8,
        format: OutputFormat::Json,
    };

    let result = audit::run_audit(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_audit_command_bias_fail_threshold() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = dir.path().join("model.bin");
    std::fs::write(&model_path, b"dummy model").expect("file write should succeed");

    let args = AuditArgs {
        input: model_path,
        audit_type: AuditType::Bias,
        protected_attr: None,
        threshold: 0.99, // Very high threshold, should fail
        format: OutputFormat::Text,
    };

    let result = audit::run_audit(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Bias audit failed"));
}

#[test]
fn test_audit_command_fairness() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = dir.path().join("model.bin");
    std::fs::write(&model_path, b"dummy model").expect("file write should succeed");

    let args = AuditArgs {
        input: model_path,
        audit_type: AuditType::Fairness,
        protected_attr: None,
        threshold: 0.8,
        format: OutputFormat::Text,
    };

    let result = audit::run_audit(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_audit_command_privacy() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = dir.path().join("model.bin");
    std::fs::write(&model_path, b"dummy model").expect("file write should succeed");

    let args = AuditArgs {
        input: model_path,
        audit_type: AuditType::Privacy,
        protected_attr: None,
        threshold: 0.8,
        format: OutputFormat::Text,
    };

    let result = audit::run_audit(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_audit_command_security() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = dir.path().join("model.bin");
    std::fs::write(&model_path, b"dummy model").expect("file write should succeed");

    let args = AuditArgs {
        input: model_path,
        audit_type: AuditType::Security,
        protected_attr: None,
        threshold: 0.8,
        format: OutputFormat::Text,
    };

    let result = audit::run_audit(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_audit_command_missing_file() {
    let args = AuditArgs {
        input: PathBuf::from("/nonexistent/model.bin"),
        audit_type: AuditType::Bias,
        protected_attr: None,
        threshold: 0.8,
        format: OutputFormat::Text,
    };

    let result = audit::run_audit(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

// ============================================================================
// Monitor command tests
// ============================================================================

#[test]
fn test_monitor_command_basic() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = dir.path().join("model.bin");
    std::fs::write(&model_path, b"dummy model").expect("file write should succeed");

    let args = MonitorArgs {
        input: model_path,
        baseline: None,
        threshold: 0.2,
        interval: 60,
        format: OutputFormat::Text,
    };

    let result = monitor::run_monitor(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_monitor_command_with_baseline() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = dir.path().join("model.bin");
    let baseline_path = dir.path().join("baseline.json");
    std::fs::write(&model_path, b"dummy model").expect("file write should succeed");
    std::fs::write(&baseline_path, b"{}").expect("file write should succeed");

    let args = MonitorArgs {
        input: model_path,
        baseline: Some(baseline_path),
        threshold: 0.2,
        interval: 60,
        format: OutputFormat::Text,
    };

    let result = monitor::run_monitor(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_monitor_command_json_output() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = dir.path().join("model.bin");
    std::fs::write(&model_path, b"dummy model").expect("file write should succeed");

    let args = MonitorArgs {
        input: model_path,
        baseline: None,
        threshold: 0.2,
        interval: 60,
        format: OutputFormat::Json,
    };

    let result = monitor::run_monitor(args, LogLevel::Quiet);
    assert!(result.is_ok());
}

#[test]
fn test_monitor_command_drift_detected() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = dir.path().join("model.bin");
    std::fs::write(&model_path, b"dummy model").expect("file write should succeed");

    let args = MonitorArgs {
        input: model_path,
        baseline: None,
        threshold: 0.001, // Very low threshold, should trigger drift
        interval: 60,
        format: OutputFormat::Text,
    };

    let result = monitor::run_monitor(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Drift detected"));
}

#[test]
fn test_monitor_command_missing_file() {
    let args = MonitorArgs {
        input: PathBuf::from("/nonexistent/model.bin"),
        baseline: None,
        threshold: 0.2,
        interval: 60,
        format: OutputFormat::Text,
    };

    let result = monitor::run_monitor(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not found"));
}

// ============================================================================
// Merge command tests
// ============================================================================

#[test]
fn test_merge_command_not_enough_models() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model1 = create_safetensors_file(&dir, "model1.safetensors");
    let output = dir.path().join("merged.json");

    let args = MergeArgs {
        models: vec![model1],
        output,
        method: MergeMethod::Ties,
        weight: None,
        density: None,
        weights: None,
        base: None,
        adapter: None,
    };

    let result = merge::run_merge(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("at least 2 models"));
}

#[test]
fn test_merge_command_slerp_wrong_model_count() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model1 = create_safetensors_file(&dir, "model1.safetensors");
    let model2 = create_safetensors_file(&dir, "model2.safetensors");
    let model3 = create_safetensors_file(&dir, "model3.safetensors");
    let output = dir.path().join("merged.json");

    let args = MergeArgs {
        models: vec![model1, model2, model3],
        output,
        method: MergeMethod::Slerp,
        weight: Some(0.5),
        density: None,
        weights: None,
        base: None,
        adapter: None,
    };

    let result = merge::run_merge(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("SLERP requires exactly 2 models"));
}

#[test]
fn test_merge_command_average_invalid_weights() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model1 = create_safetensors_file(&dir, "model1.safetensors");
    let model2 = create_safetensors_file(&dir, "model2.safetensors");
    let output = dir.path().join("merged.json");

    let args = MergeArgs {
        models: vec![model1, model2],
        output,
        method: MergeMethod::Average,
        weight: None,
        density: None,
        weights: Some("invalid,weights".to_string()),
        base: None,
        adapter: None,
    };

    let result = merge::run_merge(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Invalid weights"));
}

#[test]
fn test_merge_command_ties() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    // TIES requires at least 3 models: 1 base + 2 task-specific
    let model1 = create_safetensors_file(&dir, "model1.safetensors");
    let model2 = create_safetensors_file(&dir, "model2.safetensors");
    let model3 = create_safetensors_file(&dir, "model3.safetensors");
    let output = dir.path().join("merged.json");

    let args = MergeArgs {
        models: vec![model1, model2, model3],
        output: output.clone(),
        method: MergeMethod::Ties,
        weight: None,
        density: Some(0.2),
        weights: None,
        base: None,
        adapter: None,
    };

    let result = merge::run_merge(args, LogLevel::Quiet);
    assert!(result.is_ok(), "TIES merge failed: {result:?}");
    assert!(output.exists());
}

#[test]
fn test_merge_command_dare() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    // DARE requires at least 3 models: 1 base + 2 task-specific
    let model1 = create_safetensors_file(&dir, "model1.safetensors");
    let model2 = create_safetensors_file(&dir, "model2.safetensors");
    let model3 = create_safetensors_file(&dir, "model3.safetensors");
    let output = dir.path().join("merged.json");

    let args = MergeArgs {
        models: vec![model1, model2, model3],
        output: output.clone(),
        method: MergeMethod::Dare,
        weight: None,
        density: Some(0.5),
        weights: None,
        base: None,
        adapter: None,
    };

    let result = merge::run_merge(args, LogLevel::Quiet);
    assert!(result.is_ok(), "DARE merge failed: {result:?}");
    assert!(output.exists());
}

#[test]
fn test_merge_command_slerp() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model1 = create_safetensors_file(&dir, "model1.safetensors");
    let model2 = create_safetensors_file(&dir, "model2.safetensors");
    let output = dir.path().join("merged.json");

    let args = MergeArgs {
        models: vec![model1, model2],
        output: output.clone(),
        method: MergeMethod::Slerp,
        weight: Some(0.5),
        density: None,
        weights: None,
        base: None,
        adapter: None,
    };

    let result = merge::run_merge(args, LogLevel::Quiet);
    assert!(result.is_ok());
    assert!(output.exists());
}

#[test]
fn test_merge_command_average_uniform() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model1 = create_safetensors_file(&dir, "model1.safetensors");
    let model2 = create_safetensors_file(&dir, "model2.safetensors");
    let output = dir.path().join("merged.json");

    let args = MergeArgs {
        models: vec![model1, model2],
        output: output.clone(),
        method: MergeMethod::Average,
        weight: None,
        density: None,
        weights: None,
        base: None,
        adapter: None,
    };

    let result = merge::run_merge(args, LogLevel::Quiet);
    assert!(result.is_ok());
    assert!(output.exists());
}

#[test]
fn test_merge_command_average_weighted() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model1 = create_safetensors_file(&dir, "model1.safetensors");
    let model2 = create_safetensors_file(&dir, "model2.safetensors");
    let output = dir.path().join("merged.json");

    let args = MergeArgs {
        models: vec![model1, model2],
        output: output.clone(),
        method: MergeMethod::Average,
        weight: None,
        density: None,
        weights: Some("0.7,0.3".to_string()),
        base: None,
        adapter: None,
    };

    let result = merge::run_merge(args, LogLevel::Quiet);
    assert!(result.is_ok());
    assert!(output.exists());
}

#[test]
fn test_merge_command_safetensors_output() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model1 = create_safetensors_file(&dir, "model1.safetensors");
    let model2 = create_safetensors_file(&dir, "model2.safetensors");
    let output = dir.path().join("merged.safetensors");

    let args = MergeArgs {
        models: vec![model1, model2],
        output: output.clone(),
        method: MergeMethod::Average,
        weight: None,
        density: None,
        weights: None,
        base: None,
        adapter: None,
    };

    let result = merge::run_merge(args, LogLevel::Verbose);
    assert!(result.is_ok());
    assert!(output.exists());
}

// ============================================================================
// LoRA adapter merge tests (ENT-LoRA-017)
// ============================================================================

#[test]
fn test_ent_lora_017_merge_adapter_missing_base() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let output = dir.path().join("merged.safetensors");

    let args = MergeArgs {
        models: vec![],
        output,
        method: MergeMethod::LoraAdapter,
        weight: None,
        density: None,
        weights: None,
        base: None,
        adapter: Some(dir.path().to_path_buf()),
    };

    let result = merge::run_merge(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("--base required"));
}

#[test]
fn test_ent_lora_017_merge_adapter_missing_adapter() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let base = create_safetensors_file(&dir, "base.safetensors");
    let output = dir.path().join("merged.safetensors");

    let args = MergeArgs {
        models: vec![],
        output,
        method: MergeMethod::LoraAdapter,
        weight: None,
        density: None,
        weights: None,
        base: Some(base),
        adapter: None,
    };

    let result = merge::run_merge(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("--adapter required"));
}

#[test]
fn test_ent_lora_017_merge_adapter_nonexistent_base() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let output = dir.path().join("merged.safetensors");

    let args = MergeArgs {
        models: vec![],
        output,
        method: MergeMethod::LoraAdapter,
        weight: None,
        density: None,
        weights: None,
        base: Some(PathBuf::from("/nonexistent/base.safetensors")),
        adapter: Some(dir.path().to_path_buf()),
    };

    let result = merge::run_merge(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Base model not found"));
}

#[test]
fn test_ent_lora_017_merge_adapter_missing_config() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let base = create_safetensors_file(&dir, "base.safetensors");
    let adapter_dir = dir.path().join("adapter");
    std::fs::create_dir_all(&adapter_dir).expect("dir creation should succeed");
    let output = dir.path().join("merged.safetensors");

    let args = MergeArgs {
        models: vec![],
        output,
        method: MergeMethod::LoraAdapter,
        weight: None,
        density: None,
        weights: None,
        base: Some(base),
        adapter: Some(adapter_dir),
    };

    let result = merge::run_merge(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("adapter_config.json not found"));
}

#[test]
fn test_ent_lora_017_merge_method_parse() {
    assert_eq!(
        "lora-adapter".parse::<MergeMethod>().expect("parsing should succeed"),
        MergeMethod::LoraAdapter
    );
    assert_eq!(
        "lora".parse::<MergeMethod>().expect("parsing should succeed"),
        MergeMethod::LoraAdapter
    );
}

// ============================================================================
// Quantize command tests
// ============================================================================

#[test]
fn test_quantize_command_4bit() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = create_safetensors_file(&dir, "model.safetensors");
    let output = dir.path().join("quantized.json");

    let args = QuantizeArgs {
        model: model_path,
        output: output.clone(),
        bits: 4,
        method: QuantMethod::Symmetric,
        per_channel: false,
        calibration_data: None,
        safetensors: false,
    };

    let result = quantize::run_quantize(args, LogLevel::Quiet);
    assert!(result.is_ok());
    assert!(output.exists());
}

#[test]
fn test_quantize_command_8bit() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = create_safetensors_file(&dir, "model.safetensors");
    let output = dir.path().join("quantized.json");

    let args = QuantizeArgs {
        model: model_path,
        output: output.clone(),
        bits: 8,
        method: QuantMethod::Symmetric,
        per_channel: false,
        calibration_data: None,
        safetensors: false,
    };

    let result = quantize::run_quantize(args, LogLevel::Quiet);
    assert!(result.is_ok());
    assert!(output.exists());
}

#[test]
fn test_quantize_command_asymmetric() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = create_safetensors_file(&dir, "model.safetensors");
    let output = dir.path().join("quantized.json");

    let args = QuantizeArgs {
        model: model_path,
        output: output.clone(),
        bits: 4,
        method: QuantMethod::Asymmetric,
        per_channel: false,
        calibration_data: None,
        safetensors: false,
    };

    let result = quantize::run_quantize(args, LogLevel::Quiet);
    assert!(result.is_ok());
    assert!(output.exists());
}

#[test]
fn test_quantize_command_per_channel() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = create_safetensors_file(&dir, "model.safetensors");
    let output = dir.path().join("quantized.json");

    let args = QuantizeArgs {
        model: model_path,
        output: output.clone(),
        bits: 4,
        method: QuantMethod::Symmetric,
        per_channel: true,
        calibration_data: None,
        safetensors: false,
    };

    let result = quantize::run_quantize(args, LogLevel::Verbose);
    assert!(result.is_ok());
    assert!(output.exists());
}

#[test]
fn test_quantize_command_invalid_bits() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = create_safetensors_file(&dir, "model.safetensors");
    let output = dir.path().join("quantized.json");

    let args = QuantizeArgs {
        model: model_path,
        output,
        bits: 16, // Invalid - only 4 or 8 supported
        method: QuantMethod::Symmetric,
        per_channel: false,
        calibration_data: None,
        safetensors: false,
    };

    let result = quantize::run_quantize(args, LogLevel::Quiet);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Unsupported bit width"));
}

// ============================================================================
// Run command dispatch tests
// ============================================================================

#[test]
fn test_run_command_inspect() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = create_safetensors_file(&dir, "model.safetensors");

    let cli = Cli {
        verbose: false,
        quiet: true,
        command: Command::Inspect(InspectArgs {
            input: model_path,
            mode: InspectMode::Summary,
            columns: None,
            z_threshold: 3.0,
        }),
    };

    let result = run_command(cli);
    assert!(result.is_ok());
}

#[test]
fn test_run_command_bench() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = dir.path().join("model.bin");
    std::fs::write(&model_path, b"dummy model").expect("file write should succeed");

    let cli = Cli {
        verbose: false,
        quiet: true,
        command: Command::Bench(BenchArgs {
            input: model_path,
            warmup: 1,
            iterations: 2,
            batch_sizes: "1".to_string(),
            format: OutputFormat::Text,
        }),
    };

    let result = run_command(cli);
    assert!(result.is_ok());
}

#[test]
fn test_run_command_audit() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = dir.path().join("model.bin");
    std::fs::write(&model_path, b"dummy model").expect("file write should succeed");

    let cli = Cli {
        verbose: false,
        quiet: true,
        command: Command::Audit(AuditArgs {
            input: model_path,
            audit_type: AuditType::Privacy,
            protected_attr: None,
            threshold: 0.8,
            format: OutputFormat::Text,
        }),
    };

    let result = run_command(cli);
    assert!(result.is_ok());
}

#[test]
fn test_run_command_monitor() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = dir.path().join("model.bin");
    std::fs::write(&model_path, b"dummy model").expect("file write should succeed");

    let cli = Cli {
        verbose: false,
        quiet: true,
        command: Command::Monitor(MonitorArgs {
            input: model_path,
            baseline: None,
            threshold: 0.2,
            interval: 60,
            format: OutputFormat::Text,
        }),
    };

    let result = run_command(cli);
    assert!(result.is_ok());
}

// ============================================================================
// Additional run_command dispatch tests
// ============================================================================

#[test]
fn test_run_command_train() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let config_path = create_test_config(&dir);

    let cli = Cli {
        verbose: false,
        quiet: true,
        command: Command::Train(TrainArgs {
            config: config_path,
            output_dir: None,
            resume: None,
            epochs: None,
            batch_size: None,
            lr: None,
            dry_run: true,
            save_every: None,
            log_every: None,
            seed: None,
        }),
    };

    let result = run_command(cli);
    assert!(result.is_ok());
}

#[test]
fn test_run_command_info() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let config_path = create_test_config(&dir);

    let cli = Cli {
        verbose: false,
        quiet: true,
        command: Command::Info(InfoArgs { config: config_path, format: OutputFormat::Text }),
    };

    let result = run_command(cli);
    assert!(result.is_ok());
}

#[test]
fn test_run_command_quantize() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model_path = create_safetensors_file(&dir, "model.safetensors");
    let output = dir.path().join("quantized.json");

    let cli = Cli {
        verbose: false,
        quiet: true,
        command: Command::Quantize(QuantizeArgs {
            model: model_path,
            output,
            bits: 4,
            method: QuantMethod::Symmetric,
            per_channel: false,
            calibration_data: None,
            safetensors: false,
        }),
    };

    let result = run_command(cli);
    assert!(result.is_ok());
}

#[test]
fn test_run_command_merge() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let model1 = create_safetensors_file(&dir, "model1.safetensors");
    let model2 = create_safetensors_file(&dir, "model2.safetensors");
    let output = dir.path().join("merged.json");

    let cli = Cli {
        verbose: false,
        quiet: true,
        command: Command::Merge(MergeArgs {
            models: vec![model1, model2],
            output,
            method: MergeMethod::Average,
            weight: None,
            density: None,
            weights: None,
            base: None,
            adapter: None,
        }),
    };

    let result = run_command(cli);
    assert!(result.is_ok());
}

#[test]
fn test_run_command_completion() {
    let cli = Cli {
        verbose: false,
        quiet: true,
        command: Command::Completion(CompletionArgs { shell: ShellType::Bash }),
    };

    let result = run_command(cli);
    assert!(result.is_ok());
}

#[test]
fn test_run_command_normal_log_level() {
    let dir = TempDir::new().expect("temp file creation should succeed");
    let output_path = dir.path().join("normal_test.yaml");

    let cli = Cli {
        verbose: false,
        quiet: false, // Normal log level
        command: Command::Init(InitArgs {
            name: "test".to_string(),
            output: Some(output_path),
            template: InitTemplate::Minimal,
            model: None,
            base: None,
            method: None,
            data: None,
        }),
    };

    let result = run_command(cli);
    assert!(result.is_ok());
}

// ============================================================================
// Research command tests
// ============================================================================
