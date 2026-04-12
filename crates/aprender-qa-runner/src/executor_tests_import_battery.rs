
use super::*;
use crate::command::{CommandOutput, CommandRunner, MockCommandRunner};
use aprender_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use std::path::Path;
use std::sync::Arc;

fn import_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Import,
        Backend::Cpu,
        Format::Apr,
        "import:gguf".to_string(),
        42,
    )
}

#[test]
fn test_import_battery_all_pass() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = import_scenario();
    let results = executor.run_import_battery("/test/model.gguf", &scenario, "gguf");

    // Should produce 5 evidence items
    assert_eq!(results.len(), 5, "Expected 5 battery checks, got {}", results.len());

    // Check gate IDs
    let gate_ids: Vec<&str> = results.iter().map(|e| e.gate_id.as_str()).collect();
    assert!(gate_ids.contains(&"T2-IMPORT-001"), "Missing import exit gate");
    assert!(gate_ids.contains(&"T2-IMPORT-SIZE-001"), "Missing size gate");
    assert!(gate_ids.contains(&"T2-IMPORT-TENSOR-001"), "Missing tensor count gate");
    assert!(gate_ids.contains(&"T2-IMPORT-LOAD-001"), "Missing load gate");
    assert!(gate_ids.contains(&"T2-IMPORT-INFER-001"), "Missing inference gate");

    // Primary check should pass
    assert!(results[0].outcome.is_pass(), "Primary import check should pass");
}

#[test]
fn test_import_battery_fail_stops_early() {
    let mock_runner = MockCommandRunner::new().with_import_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = import_scenario();
    let results = executor.run_import_battery("/test/model.gguf", &scenario, "gguf");

    assert_eq!(results.len(), 1, "Should stop after primary failure");
    assert!(results[0].outcome.is_fail(), "Primary check should fail");
    assert_eq!(results[0].gate_id, "T2-IMPORT-001");
}

#[test]
fn test_import_battery_validation_failure() {
    let mock_runner = MockCommandRunner::new().with_validate_strict_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = import_scenario();
    let results = executor.run_import_battery("/test/model.gguf", &scenario, "gguf");

    assert_eq!(results.len(), 5);

    let load_result = results.iter().find(|e| e.gate_id == "T2-IMPORT-LOAD-001").unwrap();
    assert!(load_result.outcome.is_fail(), "Load validation should fail");
}

#[test]
fn test_import_battery_mqs_category() {
    let scenario = import_scenario();
    assert_eq!(scenario.mqs_category(), "T2");
}

#[test]
fn test_import_modality_is_transformation() {
    assert!(Modality::Import.is_transformation());
}

// ─── custom runners for uncovered branches ────────────────────────────────────

macro_rules! import_stub_methods {
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
        fn prune_model(&self, _: &Path, _: &Path, _: &str, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn distill_model(&self, _: &Path, _: &Path, _: &Path, _: &str) -> CommandOutput { CommandOutput::success("") }
    };
}

struct InvalidJsonImporter;
impl CommandRunner for InvalidJsonImporter {
    fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("The answer is 4.") }
    fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
    fn inspect_model_json(&self, _: &Path) -> CommandOutput { CommandOutput::success(r#"{"tensor_count":5}"#) }
    fn import_model(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("not valid json {{{{") }
    import_stub_methods!();
}

struct GarbageImportInferencer;
impl CommandRunner for GarbageImportInferencer {
    fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput {
        CommandOutput::success("αβγδεζηθικλμνξοπρστυφχψωαβγδεζηθικλμνξοπρστυφχψωαβγδ")
    }
    fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
    fn inspect_model_json(&self, _: &Path) -> CommandOutput { CommandOutput::success(r#"{"tensor_count":5}"#) }
    fn import_model(&self, _: &Path, _: &Path) -> CommandOutput {
        CommandOutput::success(r#"{"status":"success","output_size_bytes":1000,"tensor_count":5}"#)
    }
    import_stub_methods!();
}

struct MatchingTensorImporter;
impl CommandRunner for MatchingTensorImporter {
    fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("The answer is 4.") }
    fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
    fn inspect_model_json(&self, _: &Path) -> CommandOutput {
        CommandOutput::success(r#"{"tensor_count":7}"#)
    }
    fn import_model(&self, _: &Path, _: &Path) -> CommandOutput {
        CommandOutput::success(r#"{"status":"success","output_size_bytes":1000,"tensor_count":7}"#)
    }
    import_stub_methods!();
}

#[test]
fn test_import_battery_inference_failure() {
    let mock = MockCommandRunner::new().with_inference_failure();
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock));
    let scenario = import_scenario();
    let results = executor.run_import_battery("/test/model.gguf", &scenario, "gguf");
    assert_eq!(results.len(), 5);
    let infer = results.iter().find(|e| e.gate_id == "T2-IMPORT-INFER-001").expect("INFER gate");
    assert!(infer.outcome.is_fail());
}

#[test]
fn test_import_battery_invalid_json_parse_failure() {
    let config = ExecutionConfig { failure_policy: FailurePolicy::CollectAll, ..Default::default() };
    let executor = Executor::with_runner(config, Arc::new(InvalidJsonImporter));
    let scenario = import_scenario();
    let results = executor.run_import_battery("/test/model.gguf", &scenario, "gguf");
    assert_eq!(results.len(), 2, "T2-IMPORT-001 corr then JSON parse fail");
    assert!(results[0].outcome.is_pass());
    assert!(results[1].outcome.is_fail());
    assert!(results[1].reason.contains("JSON") || results[1].reason.contains("invalid"),
        "Expected JSON error, got: {}", results[1].reason);
}

#[test]
fn test_import_battery_infer_oracle_falsified() {
    let config = ExecutionConfig { failure_policy: FailurePolicy::CollectAll, ..Default::default() };
    let executor = Executor::with_runner(config, Arc::new(GarbageImportInferencer));
    let scenario = import_scenario();
    let results = executor.run_import_battery("/test/model.gguf", &scenario, "gguf");
    assert_eq!(results.len(), 5);
    let infer = results.iter().find(|e| e.gate_id == "T2-IMPORT-INFER-001").expect("INFER gate");
    assert!(infer.outcome.is_fail(), "INFER should fail on garbage output");
}

#[test]
fn test_import_battery_tensor_corroborated() {
    let config = ExecutionConfig { failure_policy: FailurePolicy::CollectAll, ..Default::default() };
    let executor = Executor::with_runner(config, Arc::new(MatchingTensorImporter));
    let scenario = import_scenario();
    let results = executor.run_import_battery("/test/model.gguf", &scenario, "gguf");
    assert_eq!(results.len(), 5);
    let tensor = results.iter().find(|e| e.gate_id == "T2-IMPORT-TENSOR-001").expect("TENSOR gate");
    assert!(tensor.outcome.is_pass(), "TENSOR should pass when counts match (7==7)");
}

#[test]
fn test_import_battery_size_corroborated() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let source_path = dir.path().join("model.gguf");
    std::fs::write(&source_path, vec![0u8; 2000]).expect("write source model");
    let source_path_str = source_path.to_string_lossy().to_string();

    #[allow(clippy::items_after_statements)]
    struct SmallImporter;
    #[allow(clippy::items_after_statements)]
    impl CommandRunner for SmallImporter {
        fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("The answer is 4.") }
        fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn inspect_model_json(&self, _: &Path) -> CommandOutput { CommandOutput::success(r#"{"tensor_count":5}"#) }
        fn import_model(&self, _: &Path, _: &Path) -> CommandOutput {
            // 3000 bytes ≤ 2 * 2000 = 4000 → SIZE corroborated
            CommandOutput::success(r#"{"status":"success","output_size_bytes":3000,"tensor_count":5}"#)
        }
        import_stub_methods!();
    }

    let config = ExecutionConfig { failure_policy: FailurePolicy::CollectAll, ..Default::default() };
    let executor = Executor::with_runner(config, Arc::new(SmallImporter));
    let scenario = import_scenario();
    let results = executor.run_import_battery(&source_path_str, &scenario, "gguf");
    assert_eq!(results.len(), 5);
    let size = results.iter().find(|e| e.gate_id == "T2-IMPORT-SIZE-001").expect("SIZE gate");
    assert!(size.outcome.is_pass(), "SIZE should pass when output ≤ 2x input (3000 ≤ 4000)");
}
