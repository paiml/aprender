/// Verify ExecutionConfig supports custom profile_ci flag
#[test]
fn test_execution_config_profile_ci_custom() {
    let config = ExecutionConfig {
        run_profile_ci: true,
        ..Default::default()
    };
    assert!(config.run_profile_ci);
}

/// Verify parse_tps_from_output extracts float from "tok/s: 12.34"
#[test]
fn test_parse_tps_from_output_valid() {
    let output = "Some text tok/s: 12.34 more text";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_some());
    assert!((tps.unwrap() - 12.34).abs() < f64::EPSILON);
}

/// Verify parse_tps_from_output handles extra whitespace after colon
#[test]
fn test_parse_tps_from_output_with_whitespace() {
    let output = "tok/s:   45.67";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_some());
    assert!((tps.unwrap() - 45.67).abs() < f64::EPSILON);
}

/// Verify parse_tps_from_output handles integer values
#[test]
fn test_parse_tps_from_output_integer() {
    let output = "tok/s: 100";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_some());
    assert!((tps.unwrap() - 100.0).abs() < f64::EPSILON);
}

/// Verify parse_tps_from_output returns None when no tok/s marker found
#[test]
fn test_parse_tps_from_output_not_found() {
    let output = "no tokens per second here";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_none());
}

/// Verify parse_tps_from_output returns None for empty string
#[test]
fn test_parse_tps_from_output_empty() {
    let output = "";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_none());
}

/// Verify parse_tps_from_output returns None for non-numeric value
#[test]
fn test_parse_tps_from_output_invalid_number() {
    let output = "tok/s: abc";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_none());
}

/// Verify extract_generated_text returns full text when no separators
#[test]
fn test_extract_generated_text_simple() {
    let output = "Hello world\nThis is text";
    let result = Executor::extract_generated_text(output);
    assert_eq!(result, "Hello world\nThis is text");
}

/// Verify extract_generated_text filters out benchmark separator lines
#[test]
fn test_extract_generated_text_filters_separator() {
    let output = "Generated text\n=== BENCHMARK ===\nMore stuff";
    let result = Executor::extract_generated_text(output);
    assert!(!result.contains("==="));
    assert!(result.contains("Generated text"));
}

/// Verify extract_generated_text filters out tok/s lines
#[test]
fn test_extract_generated_text_filters_tps() {
    let output = "Hello world\ntok/s: 12.34\nAfter tps";
    let result = Executor::extract_generated_text(output);
    assert!(!result.contains("tok/s"));
    assert!(result.contains("Hello world"));
    assert!(result.contains("After tps"));
}

/// Verify extract_generated_text returns empty for empty input
#[test]
fn test_extract_generated_text_empty() {
    let output = "";
    let result = Executor::extract_generated_text(output);
    assert!(result.is_empty());
}

/// Verify extract_generated_text returns empty when all lines filtered
#[test]
fn test_extract_generated_text_only_filtered() {
    let output = "=== START ===\ntok/s: 10\n=== END ===";
    let result = Executor::extract_generated_text(output);
    assert!(result.is_empty());
}

/// Verify extract_output_text captures text after "Output:" marker
#[test]
fn test_extract_output_text_simple() {
    let output = "Some header\nOutput:\nThe answer is 4\nCompleted in 1.2s";
    let result = Executor::extract_output_text(output);
    assert_eq!(result, "The answer is 4");
}

/// Verify extract_output_text joins multiple lines with spaces
#[test]
fn test_extract_output_text_multiline() {
    let output = "Header\nOutput:\nLine 1\nLine 2\nLine 3\nCompleted in 1s";
    let result = Executor::extract_output_text(output);
    assert_eq!(result, "Line 1 Line 2 Line 3");
}

/// Verify extract_output_text returns empty when no "Output:" marker
#[test]
fn test_extract_output_text_no_output_marker() {
    let output = "Just some text without Output marker";
    let result = Executor::extract_output_text(output);
    assert!(result.is_empty());
}

/// Verify extract_output_text returns empty for empty input
#[test]
fn test_extract_output_text_empty() {
    let output = "";
    let result = Executor::extract_output_text(output);
    assert!(result.is_empty());
}

/// Verify extract_output_text returns empty when Output section is blank
#[test]
fn test_extract_output_text_empty_output() {
    let output = "Header\nOutput:\nCompleted in 1s";
    let result = Executor::extract_output_text(output);
    assert!(result.is_empty());
}

/// Verify extract_output_text stops at first empty line
#[test]
fn test_extract_output_text_stops_at_empty_line() {
    let output = "Header\nOutput:\nThe answer\n\nMore text after blank";
    let result = Executor::extract_output_text(output);
    assert_eq!(result, "The answer");
}

/// Verify golden_scenario creates Run+Cpu+Apr scenario with Golden Rule prompt
#[test]
fn test_golden_scenario_creation() {
    let model_id = ModelId::new("test", "model");
    let scenario = Executor::golden_scenario(&model_id);
    assert_eq!(scenario.model.org, "test");
    assert_eq!(scenario.model.name, "model");
    assert_eq!(scenario.modality, Modality::Run);
    assert_eq!(scenario.backend, Backend::Cpu);
    assert_eq!(scenario.format, Format::Apr);
    assert!(scenario.prompt.contains("Golden Rule"));
}

/// Verify default ExecutionConfig enables golden rule test
#[test]
fn test_execution_config_golden_rule_default() {
    let config = ExecutionConfig::default();
    assert!(config.run_golden_rule_test);
    assert!(config.golden_reference_path.is_none());
}

/// Verify ExecutionConfig supports disabling golden rule with custom path
#[test]
fn test_execution_config_golden_rule_custom() {
    let config = ExecutionConfig {
        run_golden_rule_test: false,
        golden_reference_path: Some("/path/to/reference.json".to_string()),
        ..Default::default()
    };
    assert!(!config.run_golden_rule_test);
    assert_eq!(
        config.golden_reference_path.as_deref(),
        Some("/path/to/reference.json")
    );
}

/// Verify ToolExecutor stores model path, GPU flag, and timeout
#[test]
fn test_tool_executor_fields() {
    let executor = ToolExecutor::new("/path/model.gguf".to_string(), true, 30_000);
    assert_eq!(executor.model_path, "/path/model.gguf");
    assert!(executor.no_gpu);
    assert_eq!(executor.timeout_ms, 30_000);
}

/// Verify ToolExecutor can be created with GPU enabled
#[test]
fn test_tool_executor_no_gpu_false() {
    let executor = ToolExecutor::new("model.gguf".to_string(), false, 60_000);
    assert!(!executor.no_gpu);
}

/// Verify ToolTestResult stores custom gate ID
#[test]
fn test_tool_test_result_gate_id() {
    let result = ToolTestResult {
        tool: "custom-tool".to_string(),
        passed: true,
        exit_code: 0,
        stdout: String::new(),
        stderr: String::new(),
        duration_ms: 100,
        gate_id: "F-CUSTOM-001".to_string(),
    };
    assert_eq!(result.gate_id, "F-CUSTOM-001");
}
