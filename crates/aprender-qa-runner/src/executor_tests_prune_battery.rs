
use super::*;
use crate::command::{CommandOutput, CommandRunner, MockCommandRunner};
use apr_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use std::path::Path;
use std::sync::Arc;

fn prune_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Prune,
        Backend::Cpu,
        Format::Apr,
        "prune:magnitude:0.5".to_string(),
        42,
    )
}

#[test]
fn test_prune_battery_all_pass() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/test/model.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = prune_scenario();
    let results = executor.run_prune_battery("/test/model.safetensors", &scenario, "magnitude", 0.5);

    // Should produce 6 evidence items
    assert_eq!(results.len(), 6, "Expected 6 battery checks, got {}", results.len());

    // Check gate IDs
    let gate_ids: Vec<&str> = results.iter().map(|e| e.gate_id.as_str()).collect();
    assert!(gate_ids.contains(&"T3-PRUNE-001"), "Missing prune exit gate");
    assert!(gate_ids.contains(&"T3-PRUNE-SIZE-001"), "Missing size gate");
    assert!(gate_ids.contains(&"T3-PRUNE-RATIO-001"), "Missing ratio gate");
    assert!(gate_ids.contains(&"T3-PRUNE-LOAD-001"), "Missing load gate");
    assert!(gate_ids.contains(&"T3-PRUNE-INFER-001"), "Missing inference gate");
    assert!(gate_ids.contains(&"T3-PRUNE-TENSOR-001"), "Missing tensor count gate");

    // Primary check should pass
    assert!(results[0].outcome.is_pass(), "Primary prune check should pass");
}

#[test]
fn test_prune_battery_fail_stops_early() {
    let mock_runner = MockCommandRunner::new().with_prune_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = prune_scenario();
    let results = executor.run_prune_battery("/test/model.safetensors", &scenario, "magnitude", 0.5);

    assert_eq!(results.len(), 1, "Should stop after primary failure");
    assert!(results[0].outcome.is_fail(), "Primary check should fail");
    assert_eq!(results[0].gate_id, "T3-PRUNE-001");
}

#[test]
fn test_prune_battery_validation_failure() {
    let mock_runner = MockCommandRunner::new().with_validate_strict_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = prune_scenario();
    let results = executor.run_prune_battery("/test/model.safetensors", &scenario, "magnitude", 0.5);

    assert_eq!(results.len(), 6);

    let load_result = results.iter().find(|e| e.gate_id == "T3-PRUNE-LOAD-001").unwrap();
    assert!(load_result.outcome.is_fail(), "Load validation should fail");
}

#[test]
fn test_prune_battery_inference_failure() {
    let mock_runner = MockCommandRunner::new().with_inference_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = prune_scenario();
    let results = executor.run_prune_battery("/test/model.safetensors", &scenario, "magnitude", 0.5);

    assert_eq!(results.len(), 6);

    let infer_result = results.iter().find(|e| e.gate_id == "T3-PRUNE-INFER-001").unwrap();
    assert!(infer_result.outcome.is_fail(), "Inference should fail");
}

#[test]
fn test_prune_battery_mqs_category() {
    let scenario = prune_scenario();
    assert_eq!(scenario.mqs_category(), "T3");
}

#[test]
fn test_prune_modality_is_transformation() {
    assert!(Modality::Prune.is_transformation());
}

// ─── custom runners for uncovered branches ────────────────────────────────────

macro_rules! prune_stub_methods {
    () => {
        fn convert_model(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn inspect_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn validate_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn bench_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn check_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn profile_model(&self, _: &Path, _: u32, _: u32) -> CommandOutput { CommandOutput::success("") }
        fn profile_ci(&self, _: &Path, _: Option<f64>, _: Option<f64>, _: u32, _: u32, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn diff_tensors(&self, _: &Path, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn compare_inference(&self, _: &Path, _: &Path, _: &str, _: u32, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn profile_with_flamegraph(&self, _: &Path, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn profile_with_focus(&self, _: &Path, _: &str, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn fingerprint_model(&self, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn validate_stats(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn pull_model(&self, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn run_ollama_inference(&self, _: &str, _: &str, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn pull_ollama_model(&self, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn create_ollama_model(&self, _: &str, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn serve_model(&self, _: &Path, _: u16) -> CommandOutput { CommandOutput::success("") }
        fn http_get(&self, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn profile_memory(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn run_chat(&self, _: &Path, _: &str, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("") }
        fn http_post(&self, _: &str, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn spawn_serve(&self, _: &Path, _: u16, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn quantize_model(&self, _: &Path, _: &Path, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn import_model(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn distill_model(&self, _: &Path, _: &Path, _: &Path, _: &str) -> CommandOutput { CommandOutput::success("") }
    };
}

/// Prune exits 0 but non-JSON stdout → JSON parse failure
struct InvalidJsonPruner;

impl CommandRunner for InvalidJsonPruner {
    fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput {
        CommandOutput::success("The answer is 4.")
    }
    fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
    fn inspect_model_json(&self, _: &Path) -> CommandOutput { CommandOutput::success(r#"{"tensor_count":10}"#) }
    fn prune_model(&self, _: &Path, _: &Path, _: &str, _: f64) -> CommandOutput {
        CommandOutput::success("not valid json {{{{")
    }
    prune_stub_methods!();
}

/// Prune JSON with actual_sparsity far from target → RATIO falsified
struct BadRatioPruner;

impl CommandRunner for BadRatioPruner {
    fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput {
        CommandOutput::success("The answer is 4.")
    }
    fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
    fn inspect_model_json(&self, _: &Path) -> CommandOutput { CommandOutput::success(r#"{"tensor_count":10}"#) }
    fn prune_model(&self, _: &Path, _: &Path, _: &str, _target_ratio: f64) -> CommandOutput {
        // actual_sparsity=0.0, target=0.5, diff=0.5 > 0.05 → RATIO falsified
        CommandOutput::success(
            r#"{"status":"success","method":"magnitude","target_ratio":0.5,"actual_sparsity":0.0,"output_size_bytes":1000,"tensor_count":10}"#,
        )
    }
    prune_stub_methods!();
}

/// Prune returns garbage inference output → INFER oracle falsified
struct GarbagePruneInferencer;

impl CommandRunner for GarbagePruneInferencer {
    fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput {
        CommandOutput::success("αβγδεζηθικλμνξοπρστυφχψωαβγδεζηθικλμνξοπρστυφχψωαβγδ")
    }
    fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
    fn inspect_model_json(&self, _: &Path) -> CommandOutput { CommandOutput::success(r#"{"tensor_count":10}"#) }
    fn prune_model(&self, _: &Path, _: &Path, _: &str, target_ratio: f64) -> CommandOutput {
        CommandOutput::success(format!(
            r#"{{"status":"success","method":"magnitude","actual_sparsity":{target_ratio},"output_size_bytes":1000,"tensor_count":10}}"#,
        ))
    }
    prune_stub_methods!();
}

/// Prune JSON with tensor_count matching inspect → TENSOR corroborated
struct MatchingTensorPruner;

impl CommandRunner for MatchingTensorPruner {
    fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput {
        CommandOutput::success("The answer is 4.")
    }
    fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
    fn inspect_model_json(&self, _: &Path) -> CommandOutput {
        CommandOutput::success(r#"{"format":"SafeTensors","tensor_count":42,"parameters":"1B"}"#)
    }
    fn prune_model(&self, _: &Path, _: &Path, _: &str, target_ratio: f64) -> CommandOutput {
        CommandOutput::success(format!(
            r#"{{"status":"success","actual_sparsity":{target_ratio},"output_size_bytes":1000,"tensor_count":42}}"#,
        ))
    }
    prune_stub_methods!();
}

#[test]
fn test_prune_battery_invalid_json_parse_failure() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(InvalidJsonPruner));
    let scenario = prune_scenario();
    let results = executor.run_prune_battery("/test/model.safetensors", &scenario, "magnitude", 0.5);

    assert_eq!(results.len(), 2, "Should have corroborated then JSON parse failure");
    assert!(results[0].outcome.is_pass(), "T3-PRUNE-001 should pass");
    assert!(results[1].outcome.is_fail(), "JSON parse should fail");
    let reason = &results[1].reason;
    assert!(
        reason.contains("invalid JSON") || reason.contains("JSON"),
        "Expected JSON parse error, got: {reason}"
    );
}

#[test]
fn test_prune_battery_ratio_falsified() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(BadRatioPruner));
    let scenario = prune_scenario();
    let results = executor.run_prune_battery("/test/model.safetensors", &scenario, "magnitude", 0.5);

    assert_eq!(results.len(), 6);
    let ratio = results
        .iter()
        .find(|e| e.gate_id == "T3-PRUNE-RATIO-001")
        .expect("RATIO gate should be present");
    assert!(ratio.outcome.is_fail(), "RATIO should be falsified when sparsity is off by 50%");
    assert!(
        ratio.reason.contains("outside tolerance") || ratio.reason.contains("diff="),
        "Expected tolerance error, got: {}",
        ratio.reason
    );
}

#[test]
fn test_prune_battery_infer_oracle_falsified() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(GarbagePruneInferencer));
    let scenario = prune_scenario();
    let results = executor.run_prune_battery("/test/model.safetensors", &scenario, "magnitude", 0.5);

    assert_eq!(results.len(), 6);
    let infer = results
        .iter()
        .find(|e| e.gate_id == "T3-PRUNE-INFER-001")
        .expect("INFER gate should be present");
    assert!(infer.outcome.is_fail(), "INFER should be falsified on garbage output");
}

#[test]
fn test_prune_battery_tensor_corroborated() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(MatchingTensorPruner));
    let scenario = prune_scenario();
    // Use a nonexistent path so input_size=0 → SIZE falsified (expected), TENSOR=42/42 corroborated
    let results = executor.run_prune_battery("/test/model.safetensors", &scenario, "magnitude", 0.5);

    assert_eq!(results.len(), 6);
    let tensor = results
        .iter()
        .find(|e| e.gate_id == "T3-PRUNE-TENSOR-001")
        .expect("TENSOR gate should be present");
    assert!(tensor.outcome.is_pass(), "TENSOR should be corroborated when counts match (42==42)");
    assert!(
        tensor.output.contains("42") || tensor.reason.contains("42"),
        "Expected tensor count 42, got output: {}, reason: {}",
        tensor.output,
        tensor.reason
    );
}

#[test]
fn test_prune_battery_size_corroborated() {
    // Create a real temp file so get_file_size returns nonzero
    let dir = tempfile::tempdir().expect("create temp dir");
    let model_path = dir.path().join("model.safetensors");
    // 2000-byte file → output_size_bytes=1000 < 2000 → SIZE corroborated
    std::fs::write(&model_path, vec![0u8; 2000]).expect("write temp model");
    let model_path_str = model_path.to_string_lossy().to_string();

    #[allow(clippy::items_after_statements)]
    struct SmallOutputPruner;
    #[allow(clippy::items_after_statements)]
    impl CommandRunner for SmallOutputPruner {
        fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput {
            CommandOutput::success("The answer is 4.")
        }
        fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn inspect_model_json(&self, _: &Path) -> CommandOutput {
            CommandOutput::success(r#"{"tensor_count":10}"#)
        }
        fn prune_model(&self, _: &Path, _: &Path, _: &str, target_ratio: f64) -> CommandOutput {
            CommandOutput::success(format!(
                r#"{{"status":"success","actual_sparsity":{target_ratio},"output_size_bytes":1000,"tensor_count":10}}"#,
            ))
        }
        prune_stub_methods!();
    }

    let config = ExecutionConfig {
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(SmallOutputPruner));
    let scenario = prune_scenario();
    let results = executor.run_prune_battery(&model_path_str, &scenario, "magnitude", 0.5);

    assert_eq!(results.len(), 6);
    let size = results
        .iter()
        .find(|e| e.gate_id == "T3-PRUNE-SIZE-001")
        .expect("SIZE gate should be present");
    assert!(size.outcome.is_pass(), "SIZE should be corroborated when output < input");
}
