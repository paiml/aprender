use super::*;

use tempfile::NamedTempFile;

#[test]
fn test_verify_logits_pass() {
    let expected = vec![0.1, 0.2, 0.3, 0.4];
    let actual = vec![0.10001, 0.20001, 0.29999, 0.40001];

    let result = verify_logits("test", &actual, &expected, 1e-4);
    assert!(result.passed);
    assert!(result.max_deviation < 1e-4);
}

#[test]
fn test_verify_logits_fail() {
    let expected = vec![0.1, 0.2, 0.3, 0.4];
    let actual = vec![0.1, 0.2, 0.3, 0.5]; // 0.4 vs 0.5 = 0.1 deviation

    let result = verify_logits("test", &actual, &expected, 1e-4);
    assert!(!result.passed);
    assert!(result.max_deviation > 0.09);
}

#[test]
fn test_verify_logits_length_mismatch() {
    let expected = vec![0.1, 0.2, 0.3];
    let actual = vec![0.1, 0.2];

    let result = verify_logits("test", &actual, &expected, 1e-4);
    assert!(!result.passed);
    assert!(result.error.is_some());
}

#[test]
fn test_logit_stats() {
    let logits = vec![0.1, 0.5, 0.2, 0.8, 0.3];
    let stats = LogitStats::compute(&logits);

    assert_eq!(stats.argmax, 3); // 0.8 is max
    assert_eq!(stats.top5.len(), 5);
    assert_eq!(stats.top5[0], 3); // 0.8
    assert_eq!(stats.top5[1], 1); // 0.5
}

#[test]
fn test_golden_trace_set() {
    let mut set = GoldenTraceSet::new("qwen2", "Qwen2-0.5B-Instruct");
    set.add_trace(GoldenTrace::new("test1", vec![1, 2, 3], vec![0.1, 0.2]));
    set.add_trace(GoldenTrace::new("test2", vec![4, 5, 6], vec![0.3, 0.4]));

    assert_eq!(set.traces.len(), 2);
    assert_eq!(set.architecture, "qwen2");
}

#[test]
fn test_trace_verify_result() {
    let pass = TraceVerifyResult::pass("test", 0.00001, 0.000005, 100, 1e-4);
    assert!(pass.passed);
    assert!(pass.error.is_none());

    let fail = TraceVerifyResult::fail("test", "deviation too high");
    assert!(!fail.passed);
    assert!(fail.error.is_some());
}

#[test]
fn test_golden_verify_report() {
    let results = vec![
        TraceVerifyResult::pass("t1", 0.00001, 0.000005, 100, 1e-4),
        TraceVerifyResult::pass("t2", 0.00002, 0.000008, 100, 1e-4),
    ];

    let report = GoldenVerifyReport::from_results(results);
    assert!(report.passed);
    assert_eq!(report.passed_count, 2);
    assert_eq!(report.total_count, 2);
}

#[test]
fn test_golden_verify_report_partial_fail() {
    let results = vec![
        TraceVerifyResult::pass("t1", 0.00001, 0.000005, 100, 1e-4),
        TraceVerifyResult::fail("t2", "deviation too high"),
    ];

    let report = GoldenVerifyReport::from_results(results);
    assert!(!report.passed);
    assert_eq!(report.passed_count, 1);
    assert_eq!(report.total_count, 2);
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_golden_trace_with_tolerance() {
    let trace = GoldenTrace::new("test", vec![1, 2, 3], vec![0.1, 0.2]).with_tolerance(1e-6);
    assert!((trace.tolerance - 1e-6).abs() < 1e-10);
}

#[test]
fn test_golden_trace_default_tolerance() {
    let trace = GoldenTrace::new("test", vec![1, 2, 3], vec![0.1, 0.2]);
    assert!((trace.tolerance - 1e-4).abs() < 1e-10);
}

#[test]
fn test_golden_trace_clone() {
    let trace = GoldenTrace::new("test", vec![1, 2, 3], vec![0.1, 0.2]);
    let cloned = trace.clone();
    assert_eq!(cloned.name, "test");
    assert_eq!(cloned.input_ids, vec![1, 2, 3]);
}

#[test]
fn test_golden_trace_debug() {
    let trace = GoldenTrace::new("test", vec![1, 2, 3], vec![0.1, 0.2]);
    let debug_str = format!("{:?}", trace);
    assert!(debug_str.contains("GoldenTrace"));
    assert!(debug_str.contains("test"));
}

#[test]
fn test_golden_trace_set_default() {
    let set = GoldenTraceSet::default();
    assert!(set.architecture.is_empty());
    assert!(set.traces.is_empty());
}

#[test]
fn test_golden_trace_set_clone() {
    let mut set = GoldenTraceSet::new("llama", "Llama-2-7B");
    set.add_trace(GoldenTrace::new("t1", vec![1], vec![0.1]));
    let cloned = set.clone();
    assert_eq!(cloned.architecture, "llama");
    assert_eq!(cloned.traces.len(), 1);
}

#[test]
fn test_golden_trace_set_debug() {
    let set = GoldenTraceSet::new("qwen2", "test");
    let debug_str = format!("{:?}", set);
    assert!(debug_str.contains("GoldenTraceSet"));
    assert!(debug_str.contains("qwen2"));
}

#[test]
fn test_golden_trace_set_to_json() {
    let mut set = GoldenTraceSet::new("qwen2", "Qwen2-0.5B");
    set.add_trace(GoldenTrace::new(
        "trace1",
        vec![1, 2, 3],
        vec![0.1, 0.2, 0.3],
    ));
    set.add_trace(GoldenTrace::new("trace2", vec![4, 5], vec![0.4, 0.5]));

    let json = set.to_json().expect("JSON serialization should work");
    assert!(json.contains("\"architecture\": \"qwen2\""));
    assert!(json.contains("\"model_name\": \"Qwen2-0.5B\""));
    assert!(json.contains("\"traces\":"));
    assert!(json.contains("\"name\": \"trace1\""));
    assert!(json.contains("\"name\": \"trace2\""));
    assert!(json.contains("\"input_ids\": [1, 2, 3]"));
}

#[test]
fn test_golden_trace_set_to_json_empty_traces() {
    let set = GoldenTraceSet::new("test", "TestModel");
    let json = set.to_json().expect("JSON serialization should work");
    assert!(json.contains("\"traces\": ["));
    assert!(json.contains("]"));
}

#[test]
fn test_golden_trace_set_from_json() {
    let json = r#"{
        "architecture": "llama",
        "model_name": "Llama-2-7B",
        "created_at": "1234567890",
        "reference": "PyTorch",
        "traces": []
    }"#;

    let set = GoldenTraceSet::from_json(json).expect("Parse should work");
    assert_eq!(set.architecture, "llama");
    assert_eq!(set.model_name, "Llama-2-7B");
    assert_eq!(set.created_at, "1234567890");
    assert_eq!(set.reference, "PyTorch");
}

#[test]
fn test_golden_trace_set_from_json_partial() {
    let json = r#"{ "architecture": "bert" }"#;
    let set = GoldenTraceSet::from_json(json).expect("Parse should work");
    assert_eq!(set.architecture, "bert");
    assert!(set.model_name.is_empty()); // Not present
}

#[test]
fn test_golden_trace_set_save_load() {
    let mut set = GoldenTraceSet::new("qwen2", "TestModel");
    set.add_trace(GoldenTrace::new("t1", vec![1, 2], vec![0.1]));

    let temp = NamedTempFile::new().expect("create temp");
    let path = temp.path().to_path_buf();

    // Save
    set.save(&path).expect("Save should work");

    // Load
    let loaded = GoldenTraceSet::load(&path).expect("Load should work");
    assert_eq!(loaded.architecture, "qwen2");
    assert_eq!(loaded.model_name, "TestModel");
}

#[test]
fn test_golden_trace_set_load_nonexistent() {
    let result = GoldenTraceSet::load(Path::new("/nonexistent/path.json"));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to read"));
}

#[test]
fn test_golden_trace_set_save_invalid_path() {
    let set = GoldenTraceSet::new("test", "model");
    let result = set.save(Path::new("/nonexistent/dir/file.json"));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to write"));
}

#[test]
fn test_trace_verify_result_clone() {
    let result = TraceVerifyResult::pass("test", 0.001, 0.0005, 50, 1e-4);
    let cloned = result.clone();
    assert_eq!(cloned.name, "test");
    assert!(cloned.passed);
}

#[test]
fn test_trace_verify_result_debug() {
    let result = TraceVerifyResult::fail("test", "error message");
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("TraceVerifyResult"));
    assert!(debug_str.contains("test"));
}

#[test]
fn test_golden_verify_report_empty() {
    let report = GoldenVerifyReport::from_results(vec![]);
    assert!(!report.passed); // Empty is not passed
    assert_eq!(report.passed_count, 0);
    assert_eq!(report.total_count, 0);
}

#[test]
fn test_golden_verify_report_clone() {
    let results = vec![TraceVerifyResult::pass("t1", 0.0, 0.0, 10, 1e-4)];
    let report = GoldenVerifyReport::from_results(results);
    let cloned = report.clone();
    assert!(cloned.passed);
    assert_eq!(cloned.results.len(), 1);
}

#[test]
fn test_golden_verify_report_debug() {
    let report = GoldenVerifyReport::from_results(vec![]);
    let debug_str = format!("{:?}", report);
    assert!(debug_str.contains("GoldenVerifyReport"));
}

#[test]
fn test_logit_stats_empty() {
    let stats = LogitStats::compute(&[]);
    assert_eq!(stats.mean, 0.0);
    assert_eq!(stats.std, 0.0);
    assert_eq!(stats.argmax, 0);
    assert!(stats.top5.is_empty());
}

#[test]
fn test_logit_stats_single_element() {
    let stats = LogitStats::compute(&[0.5]);
    assert!((stats.mean - 0.5).abs() < 1e-6);
    assert_eq!(stats.std, 0.0);
    assert_eq!(stats.min, 0.5);
    assert_eq!(stats.max, 0.5);
    assert_eq!(stats.argmax, 0);
    assert_eq!(stats.top5, vec![0]);
}

#[test]
fn test_logit_stats_clone() {
    let stats = LogitStats::compute(&[0.1, 0.2, 0.3]);
    let cloned = stats.clone();
    assert!((cloned.mean - stats.mean).abs() < 1e-6);
    assert_eq!(cloned.argmax, stats.argmax);
}

#[test]
fn test_logit_stats_debug() {
    let stats = LogitStats::compute(&[0.1, 0.2]);
    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("LogitStats"));
}

#[test]
fn test_logit_stats_fewer_than_5_elements() {
    let stats = LogitStats::compute(&[0.3, 0.1, 0.2]);
    assert_eq!(stats.top5.len(), 3);
    assert_eq!(stats.top5[0], 0); // 0.3 is max
    assert_eq!(stats.top5[1], 2); // 0.2
    assert_eq!(stats.top5[2], 1); // 0.1
}

#[test]
fn test_extract_json_string_basic() {
    let json = r#"{"key": "value", "other": 123}"#;
    let result = extract_json_string(json, "key");
    assert_eq!(result, Some("value".to_string()));
}

#[test]
fn test_extract_json_string_not_found() {
    let json = r#"{"key": "value"}"#;
    let result = extract_json_string(json, "missing");
    assert!(result.is_none());
}

#[test]
fn test_extract_json_string_not_a_string() {
    let json = r#"{"key": 123}"#;
    let result = extract_json_string(json, "key");
    assert!(result.is_none()); // Not a string value
}

#[test]
fn test_extract_json_string_with_whitespace() {
    let json = r#"{"key":   "value with spaces"}"#;
    let result = extract_json_string(json, "key");
    assert_eq!(result, Some("value with spaces".to_string()));
}

#[test]
fn test_timestamp_now() {
    // Just verify it returns a non-empty string
    let ts = timestamp_now();
    assert!(!ts.is_empty());
    // Should be parseable as a number
    let _: u64 = ts.parse().expect("Should be a number");
}

#[test]
fn test_verify_logits_exact_match() {
    let values = vec![0.1, 0.2, 0.3];
    let result = verify_logits("exact", &values, &values, 1e-10);
    assert!(result.passed);
    assert_eq!(result.max_deviation, 0.0);
    assert_eq!(result.mean_deviation, 0.0);
}

#[test]
fn test_verify_logits_failed_error_message() {
    let expected = vec![0.1, 0.2];
    let actual = vec![0.1, 0.5]; // 0.3 deviation

    let result = verify_logits("test", &actual, &expected, 1e-4);
    assert!(!result.passed);
    assert!(result.error.as_ref().unwrap().contains("exceeds tolerance"));
}
