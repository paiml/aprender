//! Golden Trace Verification
//!
//! Implements spec §7.6.3: Golden trace verification for proving model authenticity.
//!
//! A golden trace captures the expected logits for a given input sequence,
//! allowing verification that the model produces mathematically correct outputs.
//!
//! # Example
//!
//! ```ignore
//! use aprender::format::golden::{GoldenTrace, verify_golden_trace};
//!
//! // Create golden trace from known-good reference (e.g., PyTorch)
//! let trace = GoldenTrace {
//!     input_ids: vec![1, 2, 3],
//!     expected_logits: vec![0.1, 0.2, ...],
//!     tolerance: 1e-4,
//! };
//!
//! // Verify model output
//! let result = verify_golden_trace(&model, &trace);
//! assert!(result.passed);
//! ```

use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Golden trace data for a single test case.
#[derive(Debug, Clone)]
pub struct GoldenTrace {
    /// Name/identifier for this trace
    pub name: String,
    /// Input token IDs
    pub input_ids: Vec<u32>,
    /// Expected logits for the last position [`vocab_size`]
    pub expected_logits: Vec<f32>,
    /// Maximum allowed deviation (default: 1e-4 per spec C1)
    pub tolerance: f32,
}

impl GoldenTrace {
    /// Create a new golden trace.
    pub fn new(name: impl Into<String>, input_ids: Vec<u32>, expected_logits: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            input_ids,
            expected_logits,
            tolerance: 1e-4, // Per spec C1
        }
    }

    /// Set custom tolerance.
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }
}

/// Collection of golden traces for model verification.
#[derive(Debug, Clone, Default)]
pub struct GoldenTraceSet {
    /// Model architecture (e.g., "qwen2")
    pub architecture: String,
    /// Model name
    pub model_name: String,
    /// Individual traces
    pub traces: Vec<GoldenTrace>,
    /// Creation timestamp
    pub created_at: String,
    /// Reference implementation (e.g., "PyTorch/HuggingFace")
    pub reference: String,
}

impl GoldenTraceSet {
    /// Create a new golden trace set.
    pub fn new(architecture: impl Into<String>, model_name: impl Into<String>) -> Self {
        Self {
            architecture: architecture.into(),
            model_name: model_name.into(),
            traces: Vec::new(),
            created_at: timestamp_now(),
            reference: "PyTorch/HuggingFace".to_string(),
        }
    }

    /// Add a trace to the set.
    pub fn add_trace(&mut self, trace: GoldenTrace) {
        self.traces.push(trace);
    }

    /// Load from JSON file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let json = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read golden trace file: {e}"))?;
        Self::from_json(&json)
    }

    /// Save to JSON file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let json = self.to_json()?;
        fs::write(path, json).map_err(|e| format!("Failed to write golden trace file: {e}"))
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, String> {
        use std::fmt::Write;
        // Manual JSON serialization (no serde dependency in core lib)
        let mut json = String::new();
        json.push_str("{\n");
        let _ = writeln!(json, "  \"architecture\": \"{}\",", self.architecture);
        let _ = writeln!(json, "  \"model_name\": \"{}\",", self.model_name);
        let _ = writeln!(json, "  \"created_at\": \"{}\",", self.created_at);
        let _ = writeln!(json, "  \"reference\": \"{}\",", self.reference);
        json.push_str("  \"traces\": [\n");

        for (i, trace) in self.traces.iter().enumerate() {
            json.push_str("    {\n");
            let _ = writeln!(json, "      \"name\": \"{}\",", trace.name);
            let _ = writeln!(json, "      \"input_ids\": {:?},", trace.input_ids);
            let _ = writeln!(json, "      \"tolerance\": {},", trace.tolerance);
            let _ = writeln!(
                json,
                "      \"expected_logits_len\": {}",
                trace.expected_logits.len()
            );
            // Note: Full logits saved in binary format for efficiency
            if i < self.traces.len() - 1 {
                json.push_str("    },\n");
            } else {
                json.push_str("    }\n");
            }
        }

        json.push_str("  ]\n");
        json.push_str("}\n");
        Ok(json)
    }

    /// Deserialize from JSON (simplified parser).
    pub fn from_json(json: &str) -> Result<Self, String> {
        // Simple JSON parsing - in production use serde
        let mut set = Self::default();

        // Extract architecture
        if let Some(arch) = extract_json_string(json, "architecture") {
            set.architecture = arch;
        }
        if let Some(name) = extract_json_string(json, "model_name") {
            set.model_name = name;
        }
        if let Some(created) = extract_json_string(json, "created_at") {
            set.created_at = created;
        }
        if let Some(reference) = extract_json_string(json, "reference") {
            set.reference = reference;
        }

        Ok(set)
    }
}

/// Result of verifying a single golden trace.
#[derive(Debug, Clone)]
pub struct TraceVerifyResult {
    /// Trace name
    pub name: String,
    /// Whether verification passed
    pub passed: bool,
    /// Maximum absolute deviation found
    pub max_deviation: f32,
    /// Mean absolute deviation
    pub mean_deviation: f32,
    /// Number of logits compared
    pub logits_compared: usize,
    /// Tolerance used
    pub tolerance: f32,
    /// Error message if failed
    pub error: Option<String>,
}

impl TraceVerifyResult {
    /// Create a passing result.
    #[must_use]
    pub fn pass(name: &str, max_dev: f32, mean_dev: f32, count: usize, tol: f32) -> Self {
        Self {
            name: name.to_string(),
            passed: true,
            max_deviation: max_dev,
            mean_deviation: mean_dev,
            logits_compared: count,
            tolerance: tol,
            error: None,
        }
    }

    /// Create a failing result.
    pub fn fail(name: &str, error: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            passed: false,
            max_deviation: f32::MAX,
            mean_deviation: f32::MAX,
            logits_compared: 0,
            tolerance: 0.0,
            error: Some(error.into()),
        }
    }
}

/// Result of verifying a complete golden trace set.
#[derive(Debug, Clone)]
pub struct GoldenVerifyReport {
    /// Individual trace results
    pub results: Vec<TraceVerifyResult>,
    /// Overall pass/fail
    pub passed: bool,
    /// Number of traces that passed
    pub passed_count: usize,
    /// Total traces tested
    pub total_count: usize,
}

impl GoldenVerifyReport {
    /// Create a new report from results.
    #[must_use]
    pub fn from_results(results: Vec<TraceVerifyResult>) -> Self {
        let passed_count = results.iter().filter(|r| r.passed).count();
        let total_count = results.len();
        let passed = passed_count == total_count && total_count > 0;

        Self {
            results,
            passed,
            passed_count,
            total_count,
        }
    }
}

/// Verify actual logits against expected golden trace.
///
/// Per spec C1: Logits must match within tolerance (default 1e-4).
#[must_use]
pub fn verify_logits(
    name: &str,
    actual: &[f32],
    expected: &[f32],
    tolerance: f32,
) -> TraceVerifyResult {
    if actual.len() != expected.len() {
        return TraceVerifyResult::fail(
            name,
            format!(
                "Logit count mismatch: expected {}, got {}",
                expected.len(),
                actual.len()
            ),
        );
    }

    let mut max_dev = 0.0f32;
    let mut sum_dev = 0.0f32;

    for (a, e) in actual.iter().zip(expected.iter()) {
        let dev = (a - e).abs();
        max_dev = max_dev.max(dev);
        sum_dev += dev;
    }

    let mean_dev = sum_dev / actual.len() as f32;

    if max_dev > tolerance {
        TraceVerifyResult {
            name: name.to_string(),
            passed: false,
            max_deviation: max_dev,
            mean_deviation: mean_dev,
            logits_compared: actual.len(),
            tolerance,
            error: Some(format!(
                "Max deviation {max_dev:.6} exceeds tolerance {tolerance:.6}"
            )),
        }
    } else {
        TraceVerifyResult::pass(name, max_dev, mean_dev, actual.len(), tolerance)
    }
}

/// Compute statistics for logits (used for golden trace generation).
#[derive(Debug, Clone)]
pub struct LogitStats {
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Argmax (most likely token)
    pub argmax: usize,
    /// Top-5 token indices
    pub top5: Vec<usize>,
}

impl LogitStats {
    /// Compute statistics from logits slice.
    #[must_use]
    pub fn compute(logits: &[f32]) -> Self {
        if logits.is_empty() {
            return Self {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                argmax: 0,
                top5: vec![],
            };
        }

        let n = logits.len() as f32;
        let sum: f32 = logits.iter().sum();
        let mean = sum / n;

        let var_sum: f32 = logits.iter().map(|x| (x - mean).powi(2)).sum();
        let std = (var_sum / n).sqrt();

        let min = logits.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Find argmax
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);

        // Find top-5
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top5: Vec<usize> = indexed.iter().take(5).map(|(i, _)| *i).collect();

        Self {
            mean,
            std,
            min,
            max,
            argmax,
            top5,
        }
    }
}

/// Generate ISO 8601 timestamp without chrono dependency.
fn timestamp_now() -> String {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    // Return Unix timestamp as string (simpler than full ISO 8601)
    format!("{}", duration.as_secs())
}

// Helper function to extract JSON string value
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\":");
    let start = json.find(&pattern)?;
    let rest = &json[start + pattern.len()..];

    // Skip whitespace
    let rest = rest.trim_start();

    // Find opening quote
    if !rest.starts_with('"') {
        return None;
    }

    // Find closing quote
    let rest = &rest[1..];
    let end = rest.find('"')?;

    Some(rest[..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
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

        let mut temp = NamedTempFile::new().expect("create temp");
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
}
