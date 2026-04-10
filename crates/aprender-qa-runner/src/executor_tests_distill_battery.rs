
use super::*;
use crate::command::{CommandOutput, CommandRunner, MockCommandRunner};
use aprender_qa_gen::{Backend, Format, Modality, ModelId, QaScenario};
use std::path::Path;
use std::sync::Arc;

fn distill_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Distill,
        Backend::Cpu,
        Format::Apr,
        "distill".to_string(),
        42,
    )
}

#[test]
fn test_distill_battery_all_pass() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/test/teacher.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = distill_scenario();
    let results = executor.run_distill_battery(
        "/test/teacher.safetensors",
        &scenario,
        "/test/student.safetensors",
        "/test/data.jsonl",
    );

    // Should produce 5 evidence items
    assert_eq!(results.len(), 5, "Expected 5 battery checks, got {}", results.len());

    // Check gate IDs
    let gate_ids: Vec<&str> = results.iter().map(|e| e.gate_id.as_str()).collect();
    assert!(gate_ids.contains(&"T4-DISTILL-001"), "Missing distill exit gate");
    assert!(gate_ids.contains(&"T4-DISTILL-SIZE-001"), "Missing size gate");
    assert!(gate_ids.contains(&"T4-DISTILL-LOAD-001"), "Missing load gate");
    assert!(gate_ids.contains(&"T4-DISTILL-INFER-001"), "Missing inference gate");
    assert!(gate_ids.contains(&"T4-DISTILL-LOSS-001"), "Missing loss gate");

    // Primary check should pass
    assert!(results[0].outcome.is_pass(), "Primary distill check should pass");
}

#[test]
fn test_distill_battery_fail_stops_early() {
    let mock_runner = MockCommandRunner::new().with_distill_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/teacher.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = distill_scenario();
    let results = executor.run_distill_battery(
        "/test/teacher.safetensors",
        &scenario,
        "/test/student.safetensors",
        "/test/data.jsonl",
    );

    assert_eq!(results.len(), 1, "Should stop after primary failure");
    assert!(results[0].outcome.is_fail(), "Primary check should fail");
    assert_eq!(results[0].gate_id, "T4-DISTILL-001");
}

#[test]
fn test_distill_battery_validation_failure() {
    let mock_runner = MockCommandRunner::new().with_validate_strict_failure();
    let config = ExecutionConfig {
        model_path: Some("/test/teacher.safetensors".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = distill_scenario();
    let results = executor.run_distill_battery(
        "/test/teacher.safetensors",
        &scenario,
        "/test/student.safetensors",
        "/test/data.jsonl",
    );

    assert_eq!(results.len(), 5);

    let load_result = results.iter().find(|e| e.gate_id == "T4-DISTILL-LOAD-001").unwrap();
    assert!(load_result.outcome.is_fail(), "Load validation should fail");
}

#[test]
fn test_distill_battery_mqs_category() {
    let scenario = distill_scenario();
    assert_eq!(scenario.mqs_category(), "T4");
}

#[test]
fn test_distill_modality_is_transformation() {
    assert!(Modality::Distill.is_transformation());
    assert!(!Modality::Serve.is_transformation());
    assert!(!Modality::Chat.is_transformation());
}

// ─── custom runners for uncovered branches ────────────────────────────────────

macro_rules! stub_runner_methods {
    ($t:ty) => {
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
        fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn fingerprint_model(&self, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn validate_stats(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn pull_model(&self, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn inspect_model_json(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
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
        fn prune_model(&self, _: &Path, _: &Path, _: &str, _: f64) -> CommandOutput { CommandOutput::success("") }
    };
}

/// Distill exits 0 but stdout is not valid JSON → JSON parse failure branch
struct InvalidJsonDistiller;

impl CommandRunner for InvalidJsonDistiller {
    fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput {
        CommandOutput::success("The answer is 4.")
    }
    fn distill_model(&self, _: &Path, _: &Path, _: &Path, _: &str) -> CommandOutput {
        CommandOutput::success("not valid json {{{{")
    }
    stub_runner_methods!(InvalidJsonDistiller);
}

/// Distill returns JSON without output_size_bytes → student_size=0 → SIZE falsified
struct NoSizeJsonDistiller;

impl CommandRunner for NoSizeJsonDistiller {
    fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput {
        CommandOutput::success("The answer is 4.")
    }
    fn distill_model(&self, _: &Path, _: &Path, _: &Path, _: &str) -> CommandOutput {
        CommandOutput::success(
            r#"{"status":"success","initial_loss":2.5,"final_loss":1.2,"teacher_size_bytes":1048576000}"#,
        )
    }
    stub_runner_methods!(NoSizeJsonDistiller);
}

/// Distill returns valid JSON but run_inference returns garbage → oracle falsified
struct GarbageInferenceDistiller;

impl CommandRunner for GarbageInferenceDistiller {
    fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput {
        // 200 non-ASCII bytes → GarbageOracle flags as garbage
        CommandOutput::success("αβγδεζηθικλμνξοπρστυφχψωαβγδεζηθικλμνξοπρστυφχψωαβγδ")
    }
    fn distill_model(&self, _: &Path, _: &Path, _: &Path, _: &str) -> CommandOutput {
        CommandOutput::success(
            r#"{"status":"success","initial_loss":2.5,"final_loss":1.2,"output_size_bytes":262144000,"teacher_size_bytes":1048576000}"#,
        )
    }
    stub_runner_methods!(GarbageInferenceDistiller);
}

/// Distill returns JSON with final_loss >= initial_loss → LOSS falsified
struct LossNotDecreasingDistiller;

impl CommandRunner for LossNotDecreasingDistiller {
    fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput {
        CommandOutput::success("The answer is 4.")
    }
    fn distill_model(&self, _: &Path, _: &Path, _: &Path, _: &str) -> CommandOutput {
        CommandOutput::success(
            r#"{"status":"success","initial_loss":1.2,"final_loss":2.5,"output_size_bytes":262144000,"teacher_size_bytes":1048576000}"#,
        )
    }
    stub_runner_methods!(LossNotDecreasingDistiller);
}

#[test]
fn test_distill_battery_invalid_json_parse_failure() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(InvalidJsonDistiller));
    let scenario = distill_scenario();
    let results = executor.run_distill_battery(
        "/test/teacher.safetensors",
        &scenario,
        "/test/student.safetensors",
        "/test/data.jsonl",
    );

    // Check 1 corroborated (distill exits 0), then JSON parse fails → 2nd T4-DISTILL-001 falsified → early return
    assert_eq!(results.len(), 2, "Expected corroborated then JSON parse failure");
    assert!(results[0].outcome.is_pass(), "distill exit check should pass");
    assert!(results[1].outcome.is_fail(), "JSON parse should fail");
    let reason = &results[1].reason;
    assert!(
        reason.contains("invalid JSON") || reason.contains("JSON"),
        "Expected JSON error, got: {reason}"
    );
}

#[test]
fn test_distill_battery_size_falsified_no_output_size() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(NoSizeJsonDistiller));
    let scenario = distill_scenario();
    let results = executor.run_distill_battery(
        "/test/teacher.safetensors",
        &scenario,
        "/test/student.safetensors",
        "/test/data.jsonl",
    );

    assert_eq!(results.len(), 5);
    let size_result = results
        .iter()
        .find(|e| e.gate_id == "T4-DISTILL-SIZE-001")
        .expect("SIZE gate should be present");
    assert!(size_result.outcome.is_fail(), "SIZE should be falsified when student_size=0");
    assert!(
        size_result.reason.contains("not smaller") || size_result.reason.contains("student=0"),
        "Expected 'not smaller' in reason, got: {}",
        size_result.reason
    );
}

#[test]
fn test_distill_battery_infer_inference_failure() {
    let mock = MockCommandRunner::new().with_inference_failure();
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(mock));
    let scenario = distill_scenario();
    let results = executor.run_distill_battery(
        "/test/teacher.safetensors",
        &scenario,
        "/test/student.safetensors",
        "/test/data.jsonl",
    );

    assert_eq!(results.len(), 5);
    let infer = results
        .iter()
        .find(|e| e.gate_id == "T4-DISTILL-INFER-001")
        .expect("INFER gate should be present");
    assert!(infer.outcome.is_fail(), "INFER should be falsified on inference failure");
    assert!(
        infer.reason.contains("failed") || infer.reason.contains("inference"),
        "Expected inference failure reason, got: {}",
        infer.reason
    );
}

#[test]
fn test_distill_battery_infer_oracle_falsified() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(GarbageInferenceDistiller));
    let scenario = distill_scenario();
    let results = executor.run_distill_battery(
        "/test/teacher.safetensors",
        &scenario,
        "/test/student.safetensors",
        "/test/data.jsonl",
    );

    assert_eq!(results.len(), 5);
    let infer = results
        .iter()
        .find(|e| e.gate_id == "T4-DISTILL-INFER-001")
        .expect("INFER gate should be present");
    assert!(infer.outcome.is_fail(), "INFER should be falsified on garbage output");
}

#[test]
fn test_distill_battery_loss_not_decreasing() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };
    let executor = Executor::with_runner(config, Arc::new(LossNotDecreasingDistiller));
    let scenario = distill_scenario();
    let results = executor.run_distill_battery(
        "/test/teacher.safetensors",
        &scenario,
        "/test/student.safetensors",
        "/test/data.jsonl",
    );

    assert_eq!(results.len(), 5);
    let loss = results
        .iter()
        .find(|e| e.gate_id == "T4-DISTILL-LOSS-001")
        .expect("LOSS gate should be present");
    assert!(loss.outcome.is_fail(), "LOSS should be falsified when loss increases");
    assert!(
        loss.reason.contains("not decreasing") || loss.reason.contains("initial="),
        "Expected loss not decreasing reason, got: {}",
        loss.reason
    );
}
