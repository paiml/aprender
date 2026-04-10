
use super::*;
use crate::command::{CommandOutput, CommandRunner, MockCommandRunner};
use aprender_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use std::path::Path;
use std::sync::Arc;

fn quantize_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Quantize,
        Backend::Cpu,
        Format::Apr,
        "quantize:q4_k_m".to_string(),
        42,
    )
}

#[test]
fn test_quantize_battery_all_pass() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/test/model.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = quantize_scenario();
    let results = executor.run_quantize_battery("/test/model.safetensors", &scenario, "q4_k_m");

    // Should produce 6 evidence items
    assert_eq!(results.len(), 6, "Expected 6 battery checks, got {}", results.len());

    // Check gate IDs
    let gate_ids: Vec<&str> = results.iter().map(|e| e.gate_id.as_str()).collect();
    assert!(gate_ids.contains(&"T1-QUANT-001"), "Missing quantize exit gate");
    assert!(gate_ids.contains(&"T1-QUANT-SIZE-001"), "Missing size gate");
    assert!(gate_ids.contains(&"T1-QUANT-TENSOR-001"), "Missing tensor count gate");
    assert!(gate_ids.contains(&"T1-QUANT-LOAD-001"), "Missing load gate");
    assert!(gate_ids.contains(&"T1-QUANT-INFER-001"), "Missing inference gate");
    assert!(gate_ids.contains(&"T1-QUANT-DTYPE-001"), "Missing dtype gate");

    // Primary check should pass
    assert!(results[0].outcome.is_pass(), "Primary quantize check should pass");
}

#[test]
fn test_quantize_battery_fail_stops_early() {
    let mock_runner = MockCommandRunner::new().with_quantize_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = quantize_scenario();
    let results = executor.run_quantize_battery("/test/model.safetensors", &scenario, "q4_k_m");

    // Should produce only 1 evidence item (primary failure stops battery)
    assert_eq!(results.len(), 1, "Should stop after primary failure");
    assert!(results[0].outcome.is_fail(), "Primary check should fail");
    assert_eq!(results[0].gate_id, "T1-QUANT-001");
}

#[test]
fn test_quantize_battery_validation_failure() {
    let mock_runner = MockCommandRunner::new().with_validate_strict_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = quantize_scenario();
    let results = executor.run_quantize_battery("/test/model.safetensors", &scenario, "q4_k_m");

    // Should still have 6 checks
    assert_eq!(results.len(), 6);

    // LOAD check should fail
    let load_result = results.iter().find(|e| e.gate_id == "T1-QUANT-LOAD-001").unwrap();
    assert!(load_result.outcome.is_fail(), "Load validation should fail");
}

#[test]
fn test_quantize_battery_inference_failure() {
    let mock_runner = MockCommandRunner::new().with_inference_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = quantize_scenario();
    let results = executor.run_quantize_battery("/test/model.safetensors", &scenario, "q4_k_m");

    assert_eq!(results.len(), 6);

    let infer_result = results.iter().find(|e| e.gate_id == "T1-QUANT-INFER-001").unwrap();
    assert!(infer_result.outcome.is_fail(), "Inference should fail");
}

#[test]
fn test_quantize_battery_mqs_category() {
    let scenario = quantize_scenario();
    assert_eq!(scenario.mqs_category(), "T1");
}

#[test]
fn test_quantize_modality_is_transformation() {
    assert!(Modality::Quantize.is_transformation());
    assert!(!Modality::Run.is_transformation());
}

// ─── custom runners for uncovered branches ────────────────────────────────────

macro_rules! quant_stub_methods {
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
        fn import_model(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn prune_model(&self, _: &Path, _: &Path, _: &str, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn distill_model(&self, _: &Path, _: &Path, _: &Path, _: &str) -> CommandOutput { CommandOutput::success("") }
    };
}

struct InvalidJsonQuantizer;
impl CommandRunner for InvalidJsonQuantizer {
    fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("The answer is 4.") }
    fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
    fn inspect_model_json(&self, _: &Path) -> CommandOutput { CommandOutput::success(r#"{"tensor_count":5}"#) }
    fn quantize_model(&self, _: &Path, _: &Path, _: &str) -> CommandOutput { CommandOutput::success("not valid json {{{{") }
    quant_stub_methods!();
}

struct GarbageQuantInferencer;
impl CommandRunner for GarbageQuantInferencer {
    fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput {
        CommandOutput::success("αβγδεζηθικλμνξοπρστυφχψωαβγδεζηθικλμνξοπρστυφχψωαβγδ")
    }
    fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
    fn inspect_model_json(&self, _: &Path) -> CommandOutput { CommandOutput::success(r#"{"tensor_count":5}"#) }
    fn quantize_model(&self, _: &Path, _: &Path, scheme: &str) -> CommandOutput {
        CommandOutput::success(format!(
            r#"{{"status":"success","output_size_bytes":1000,"tensor_count":5,"dtype":"{scheme}"}}"#,
        ))
    }
    quant_stub_methods!();
}

struct WrongDtypeQuantizer;
impl CommandRunner for WrongDtypeQuantizer {
    fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("The answer is 4.") }
    fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
    fn inspect_model_json(&self, _: &Path) -> CommandOutput { CommandOutput::success(r#"{"tensor_count":5}"#) }
    fn quantize_model(&self, _: &Path, _: &Path, _scheme: &str) -> CommandOutput {
        // dtype is "f32" but scheme is "q4_k_m" → DTYPE falsified
        CommandOutput::success(r#"{"status":"success","output_size_bytes":1000,"tensor_count":5,"dtype":"f32"}"#)
    }
    quant_stub_methods!();
}

struct MatchingTensorQuantizer;
impl CommandRunner for MatchingTensorQuantizer {
    fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("The answer is 4.") }
    fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
    fn inspect_model_json(&self, _: &Path) -> CommandOutput {
        CommandOutput::success(r#"{"tensor_count":8}"#)
    }
    fn quantize_model(&self, _: &Path, _: &Path, scheme: &str) -> CommandOutput {
        CommandOutput::success(format!(
            r#"{{"status":"success","output_size_bytes":1000,"tensor_count":8,"dtype":"{scheme}"}}"#,
        ))
    }
    quant_stub_methods!();
}

#[test]
fn test_quantize_battery_invalid_json_parse_failure() {
    let config = ExecutionConfig { failure_policy: FailurePolicy::CollectAll, ..Default::default() };
    let executor = Executor::with_runner(config, Arc::new(InvalidJsonQuantizer));
    let scenario = quantize_scenario();
    let results = executor.run_quantize_battery("/test/model.safetensors", &scenario, "q4_k_m");
    assert_eq!(results.len(), 2, "T1-QUANT-001 corr then JSON parse fail");
    assert!(results[0].outcome.is_pass());
    assert!(results[1].outcome.is_fail());
    assert!(results[1].reason.contains("JSON") || results[1].reason.contains("invalid"),
        "Expected JSON error, got: {}", results[1].reason);
}

#[test]
fn test_quantize_battery_infer_oracle_falsified() {
    let config = ExecutionConfig { failure_policy: FailurePolicy::CollectAll, ..Default::default() };
    let executor = Executor::with_runner(config, Arc::new(GarbageQuantInferencer));
    let scenario = quantize_scenario();
    let results = executor.run_quantize_battery("/test/model.safetensors", &scenario, "q4_k_m");
    assert_eq!(results.len(), 6);
    let infer = results.iter().find(|e| e.gate_id == "T1-QUANT-INFER-001").expect("INFER gate");
    assert!(infer.outcome.is_fail(), "INFER should fail on garbage output");
}

#[test]
fn test_quantize_battery_dtype_falsified() {
    let config = ExecutionConfig { failure_policy: FailurePolicy::CollectAll, ..Default::default() };
    let executor = Executor::with_runner(config, Arc::new(WrongDtypeQuantizer));
    let scenario = quantize_scenario();
    let results = executor.run_quantize_battery("/test/model.safetensors", &scenario, "q4_k_m");
    assert_eq!(results.len(), 6);
    let dtype = results.iter().find(|e| e.gate_id == "T1-QUANT-DTYPE-001").expect("DTYPE gate");
    assert!(dtype.outcome.is_fail(), "DTYPE should fail when dtype=f32 but scheme=q4_k_m");
    assert!(dtype.reason.contains("Dtype mismatch") || dtype.reason.contains("expected="),
        "Expected dtype mismatch reason, got: {}", dtype.reason);
}

#[test]
fn test_quantize_battery_tensor_corroborated() {
    let config = ExecutionConfig { failure_policy: FailurePolicy::CollectAll, ..Default::default() };
    let executor = Executor::with_runner(config, Arc::new(MatchingTensorQuantizer));
    let scenario = quantize_scenario();
    let results = executor.run_quantize_battery("/test/model.safetensors", &scenario, "q4_k_m");
    assert_eq!(results.len(), 6);
    let tensor = results.iter().find(|e| e.gate_id == "T1-QUANT-TENSOR-001").expect("TENSOR gate");
    assert!(tensor.outcome.is_pass(), "TENSOR should pass when counts match (8==8)");
}

#[test]
fn test_quantize_battery_size_corroborated() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let model_path = dir.path().join("model.safetensors");
    std::fs::write(&model_path, vec![0u8; 3000]).expect("write temp model");
    let model_path_str = model_path.to_string_lossy().to_string();

    #[allow(clippy::items_after_statements)]
    struct SmallQuantizer;
    #[allow(clippy::items_after_statements)]
    impl CommandRunner for SmallQuantizer {
        fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("The answer is 4.") }
        fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn inspect_model_json(&self, _: &Path) -> CommandOutput { CommandOutput::success(r#"{"tensor_count":5}"#) }
        fn quantize_model(&self, _: &Path, _: &Path, scheme: &str) -> CommandOutput {
            // 1000 bytes < 3000 → SIZE corroborated
            CommandOutput::success(format!(
                r#"{{"status":"success","output_size_bytes":1000,"tensor_count":5,"dtype":"{scheme}"}}"#,
            ))
        }
        quant_stub_methods!();
    }

    let config = ExecutionConfig { failure_policy: FailurePolicy::CollectAll, ..Default::default() };
    let executor = Executor::with_runner(config, Arc::new(SmallQuantizer));
    let scenario = quantize_scenario();
    let results = executor.run_quantize_battery(&model_path_str, &scenario, "q4_k_m");
    assert_eq!(results.len(), 6);
    let size = results.iter().find(|e| e.gate_id == "T1-QUANT-SIZE-001").expect("SIZE gate");
    assert!(size.outcome.is_pass(), "SIZE should pass when output < input (1000 < 3000)");
}
