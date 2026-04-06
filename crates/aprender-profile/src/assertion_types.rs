//! Build-time trace assertion types (Sprint 44)
//!
//! This module defines the assertion types used in renacer.toml for build-time
//! validation of trace properties. Assertions enable shift-left performance
//! validation, catching regressions before they reach production.
//!
//! # Toyota Way: Andon (Visual Control)
//!
//! The assertion system implements the Andon principle from the Toyota Production
//! System: detect defects early and stop the line. Build-time assertions fail CI/CD
//! pipelines when performance regressions are detected, preventing defects from
//! reaching production.
//!
//! # Example
//!
//! ```no_run
//! use renacer::assertion_types::{Assertion, AssertionType, CriticalPathAssertion};
//!
//! // Critical path assertion: maximum latency
//! let assertion = Assertion {
//!     name: "api_max_latency".to_string(),
//!     assertion_type: AssertionType::CriticalPath(CriticalPathAssertion {
//!         max_duration_ms: 100,
//!         trace_name_pattern: Some("api_request".to_string()),
//!     }),
//!     fail_on_violation: true,
//!     enabled: true,
//! };
//! ```

use serde::{Deserialize, Serialize};

/// Top-level assertion configuration
///
/// This represents a single assertion in renacer.toml.
///
/// # Example TOML
///
/// ```toml
/// [[assertion]]
/// name = "critical_path_max_latency"
/// type = "critical_path"
/// max_duration_ms = 100
/// fail_on_violation = true
/// enabled = true
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Assertion {
    /// Unique name for this assertion
    pub name: String,

    /// The type of assertion (critical path, anti-pattern, etc.)
    #[serde(flatten)]
    pub assertion_type: AssertionType,

    /// If true, fail cargo test when this assertion is violated
    #[serde(default = "default_true")]
    pub fail_on_violation: bool,

    /// If false, skip this assertion (useful for debugging)
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool {
    true
}

/// The type of assertion to evaluate
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssertionType {
    /// Critical path latency assertion
    ///
    /// Validates that the critical path (longest execution time) does not
    /// exceed a maximum duration.
    ///
    /// # Example TOML
    ///
    /// ```toml
    /// [[assertion]]
    /// name = "api_latency"
    /// type = "critical_path"
    /// max_duration_ms = 100
    /// ```
    CriticalPath(CriticalPathAssertion),

    /// Anti-pattern detection assertion
    ///
    /// Detects performance anti-patterns like God Process, Tight Loop, etc.
    ///
    /// # Example TOML
    ///
    /// ```toml
    /// [[assertion]]
    /// name = "no_god_process"
    /// type = "anti_pattern"
    /// pattern = "GodProcess"
    /// threshold = 0.8
    /// ```
    AntiPattern(AntiPatternAssertion),

    /// Span count assertion
    ///
    /// Validates that the number of spans in a trace does not exceed a maximum.
    ///
    /// # Example TOML
    ///
    /// ```toml
    /// [[assertion]]
    /// name = "max_syscalls"
    /// type = "span_count"
    /// max_spans = 1000
    /// ```
    SpanCount(SpanCountAssertion),

    /// Memory usage assertion
    ///
    /// Validates that memory allocations do not exceed a maximum.
    ///
    /// # Example TOML
    ///
    /// ```toml
    /// [[assertion]]
    /// name = "max_memory"
    /// type = "memory_usage"
    /// max_bytes = 10000000
    /// ```
    MemoryUsage(MemoryUsageAssertion),

    /// Custom assertion (user-defined)
    ///
    /// Allows users to define custom assertions via a Rust expression.
    ///
    /// # Example TOML
    ///
    /// ```toml
    /// [[assertion]]
    /// name = "custom_check"
    /// type = "custom"
    /// expression = "trace.spans.iter().all(|s| s.duration_ms < 50)"
    /// ```
    Custom(CustomAssertion),
}

/// Critical path latency assertion
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CriticalPathAssertion {
    /// Maximum duration in milliseconds
    pub max_duration_ms: u64,

    /// Optional trace name pattern (regex)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_name_pattern: Option<String>,
}

/// Anti-pattern detection assertion
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AntiPatternAssertion {
    /// Anti-pattern type: `GodProcess`, `TightLoop`, `PcieBottleneck`
    pub pattern: AntiPatternType,

    /// Confidence threshold (0.0 - 1.0)
    pub threshold: f64,

    /// Optional process name pattern (regex)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub process_name_pattern: Option<String>,
}

/// Anti-pattern types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "PascalCase")]
pub enum AntiPatternType {
    /// God Process: Single process doing too much work
    GodProcess,

    /// Tight Loop: Excessive iterations with minimal progress
    TightLoop,

    /// `PCIe` Bottleneck: GPU transfer overhead dominates computation
    PcieBottleneck,
}

/// Span count assertion
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpanCountAssertion {
    /// Maximum number of spans allowed
    pub max_spans: usize,

    /// Optional span name pattern (regex)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span_name_pattern: Option<String>,
}

/// Memory usage assertion
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryUsageAssertion {
    /// Maximum memory usage in bytes
    pub max_bytes: u64,

    /// Track allocations (mmap, brk) or total RSS
    #[serde(default = "default_allocations")]
    pub tracking_mode: MemoryTrackingMode,
}

fn default_allocations() -> MemoryTrackingMode {
    MemoryTrackingMode::Allocations
}

/// Memory tracking mode
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTrackingMode {
    /// Track mmap/brk syscalls
    Allocations,

    /// Track total RSS (resident set size)
    Rss,
}

/// Custom assertion (user-defined Rust expression)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CustomAssertion {
    /// Rust expression to evaluate (returns bool)
    pub expression: String,
}

/// Assertion evaluation result
#[derive(Debug, Clone, PartialEq)]
pub struct AssertionResult {
    /// Assertion name
    pub name: String,

    /// Did the assertion pass?
    pub passed: bool,

    /// Human-readable explanation
    pub message: String,

    /// Actual value (for debugging)
    pub actual_value: Option<AssertionValue>,

    /// Expected value (for debugging)
    pub expected_value: Option<AssertionValue>,
}

impl AssertionResult {
    /// Create a passing result
    pub fn pass(name: String, message: String) -> Self {
        Self { name, passed: true, message, actual_value: None, expected_value: None }
    }

    /// Create a failing result
    pub fn fail(name: String, message: String) -> Self {
        Self { name, passed: false, message, actual_value: None, expected_value: None }
    }

    /// Add actual/expected values for debugging
    pub fn with_values(mut self, actual: AssertionValue, expected: AssertionValue) -> Self {
        self.actual_value = Some(actual);
        self.expected_value = Some(expected);
        self
    }
}

/// Assertion value types (for debugging output)
#[derive(Debug, Clone, PartialEq)]
pub enum AssertionValue {
    Duration(u64),
    Count(usize),
    Bytes(u64),
    Percentage(f64),
    String(String),
}

impl std::fmt::Display for AssertionValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssertionValue::Duration(ms) => write!(f, "{ms}ms"),
            AssertionValue::Count(n) => write!(f, "{n} spans"),
            AssertionValue::Bytes(b) => write!(f, "{b} bytes"),
            AssertionValue::Percentage(p) => write!(f, "{:.1}%", p * 100.0),
            AssertionValue::String(s) => write!(f, "{s}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_critical_path_assertion_deserialize() {
        let toml = r#"
            name = "api_latency"
            type = "critical_path"
            max_duration_ms = 100
            fail_on_violation = true
        "#;

        let assertion: Assertion = toml::from_str(toml).expect("test");

        assert_eq!(assertion.name, "api_latency");
        assert!(assertion.fail_on_violation);
        assert!(assertion.enabled);

        match assertion.assertion_type {
            AssertionType::CriticalPath(cp) => {
                assert_eq!(cp.max_duration_ms, 100);
                assert_eq!(cp.trace_name_pattern, None);
            }
            _ => panic!("Expected CriticalPath assertion"),
        }
    }

    #[test]
    fn test_anti_pattern_assertion_deserialize() {
        let toml = r#"
            name = "no_god_process"
            type = "anti_pattern"
            pattern = "GodProcess"
            threshold = 0.8
            fail_on_violation = true
        "#;

        let assertion: Assertion = toml::from_str(toml).expect("test");

        assert_eq!(assertion.name, "no_god_process");

        match assertion.assertion_type {
            AssertionType::AntiPattern(ap) => {
                assert_eq!(ap.pattern, AntiPatternType::GodProcess);
                assert_eq!(ap.threshold, 0.8);
            }
            _ => panic!("Expected AntiPattern assertion"),
        }
    }

    #[test]
    fn test_span_count_assertion_deserialize() {
        let toml = r#"
            name = "max_syscalls"
            type = "span_count"
            max_spans = 1000
        "#;

        let assertion: Assertion = toml::from_str(toml).expect("test");

        match assertion.assertion_type {
            AssertionType::SpanCount(sc) => {
                assert_eq!(sc.max_spans, 1000);
            }
            _ => panic!("Expected SpanCount assertion"),
        }
    }

    #[test]
    fn test_memory_usage_assertion_deserialize() {
        let toml = r#"
            name = "max_memory"
            type = "memory_usage"
            max_bytes = 10000000
            tracking_mode = "allocations"
        "#;

        let assertion: Assertion = toml::from_str(toml).expect("test");

        match assertion.assertion_type {
            AssertionType::MemoryUsage(mu) => {
                assert_eq!(mu.max_bytes, 10_000_000);
                assert_eq!(mu.tracking_mode, MemoryTrackingMode::Allocations);
            }
            _ => panic!("Expected MemoryUsage assertion"),
        }
    }

    #[test]
    fn test_custom_assertion_deserialize() {
        let toml = r#"
            name = "custom_check"
            type = "custom"
            expression = "trace.spans.len() < 100"
        "#;

        let assertion: Assertion = toml::from_str(toml).expect("test");

        match assertion.assertion_type {
            AssertionType::Custom(c) => {
                assert_eq!(c.expression, "trace.spans.len() < 100");
            }
            _ => panic!("Expected Custom assertion"),
        }
    }

    #[test]
    fn test_assertion_result_pass() {
        let result = AssertionResult::pass("test".to_string(), "Assertion passed".to_string());

        assert!(result.passed);
        assert_eq!(result.name, "test");
        assert_eq!(result.message, "Assertion passed");
    }

    #[test]
    fn test_assertion_result_fail() {
        let result = AssertionResult::fail("test".to_string(), "Assertion failed".to_string())
            .with_values(AssertionValue::Duration(150), AssertionValue::Duration(100));

        assert!(!result.passed);
        assert_eq!(result.actual_value, Some(AssertionValue::Duration(150)));
        assert_eq!(result.expected_value, Some(AssertionValue::Duration(100)));
    }

    #[test]
    fn test_assertion_value_display() {
        assert_eq!(AssertionValue::Duration(100).to_string(), "100ms");
        assert_eq!(AssertionValue::Count(42).to_string(), "42 spans");
        assert_eq!(AssertionValue::Bytes(1024).to_string(), "1024 bytes");
        assert_eq!(AssertionValue::Percentage(0.85).to_string(), "85.0%");
    }

    #[test]
    fn test_default_values() {
        let toml = r#"
            name = "test"
            type = "critical_path"
            max_duration_ms = 100
        "#;

        let assertion: Assertion = toml::from_str(toml).expect("test");

        // Default values should be true
        assert!(assertion.fail_on_violation);
        assert!(assertion.enabled);
    }

    #[test]
    fn test_disabled_assertion() {
        let toml = r#"
            name = "test"
            type = "critical_path"
            max_duration_ms = 100
            enabled = false
        "#;

        let assertion: Assertion = toml::from_str(toml).expect("test");

        assert!(!assertion.enabled);
    }
}
