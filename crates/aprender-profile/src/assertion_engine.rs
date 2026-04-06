//! Build-time trace assertion evaluation engine (Sprint 44)
//!
//! This module evaluates assertions against traces at build time,
//! enabling shift-left performance validation.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ Build-Time Assertion Flow (Toyota Way: Andon)                   │
//! └─────────────────────────────────────────────────────────────────┘
//!
//! 1. Parse renacer.toml → Vec<Assertion>
//! 2. Run cargo test → Generate traces
//! 3. Evaluate assertions → Vec<AssertionResult>
//! 4. If any fail_on_violation → panic! (fail CI)
//! ```

use crate::assertion_types::{
    AntiPatternAssertion, Assertion, AssertionResult, AssertionType, AssertionValue,
    CriticalPathAssertion, MemoryUsageAssertion, SpanCountAssertion,
};
use crate::unified_trace::UnifiedTrace;

/// Assertion evaluation engine
///
/// Evaluates assertions against traces at build time.
///
/// # Note
///
/// Sprint 44 implementation uses simplified evaluation logic.
/// Full integration with `CausalGraph` and `AntiPatternDetector`
/// will be completed in a future sprint.
pub struct AssertionEngine {}

impl AssertionEngine {
    /// Create a new assertion engine
    pub fn new() -> Self {
        Self {}
    }

    /// Evaluate a single assertion against a trace
    ///
    /// # Arguments
    ///
    /// * `assertion` - The assertion to evaluate
    /// * `trace` - The trace to evaluate against
    ///
    /// # Returns
    ///
    /// Assertion evaluation result (pass/fail with details)
    pub fn evaluate(&self, assertion: &Assertion, trace: &UnifiedTrace) -> AssertionResult {
        if !assertion.enabled {
            return AssertionResult::pass(assertion.name.clone(), "Assertion disabled".to_string());
        }

        contract_pre_error_handling!();
        match &assertion.assertion_type {
            AssertionType::CriticalPath(cp) => {
                self.evaluate_critical_path(&assertion.name, cp, trace)
            }
            AssertionType::AntiPattern(ap) => {
                self.evaluate_anti_pattern(&assertion.name, ap, trace)
            }
            AssertionType::SpanCount(sc) => self.evaluate_span_count(&assertion.name, sc, trace),
            AssertionType::MemoryUsage(mu) => {
                self.evaluate_memory_usage(&assertion.name, mu, trace)
            }
            AssertionType::Custom(c) => {
                // Custom assertions not yet implemented
                AssertionResult::pass(
                    assertion.name.clone(),
                    format!("Custom assertion '{}' not yet implemented", c.expression),
                )
            }
        }
    }

    /// Evaluate critical path assertion
    fn evaluate_critical_path(
        &self,
        name: &str,
        assertion: &CriticalPathAssertion,
        trace: &UnifiedTrace,
    ) -> AssertionResult {
        // Planned (Sprint 44): Full integration with CausalGraph + CriticalPathAnalyzer
        // For now, calculate total duration from all syscall spans

        let duration_ms = if trace.syscall_spans.is_empty() {
            0
        } else {
            // Sum all syscall durations and convert to ms
            let total_duration_nanos: u64 =
                trace.syscall_spans.iter().map(|span| span.duration_nanos).sum();
            total_duration_nanos / 1_000_000
        };

        if duration_ms <= assertion.max_duration_ms {
            AssertionResult::pass(
                name.to_string(),
                format!(
                    "Critical path duration {}ms <= {}ms",
                    duration_ms, assertion.max_duration_ms
                ),
            )
            .with_values(
                AssertionValue::Duration(duration_ms),
                AssertionValue::Duration(assertion.max_duration_ms),
            )
        } else {
            AssertionResult::fail(
                name.to_string(),
                format!(
                    "Critical path duration {}ms exceeds maximum {}ms",
                    duration_ms, assertion.max_duration_ms
                ),
            )
            .with_values(
                AssertionValue::Duration(duration_ms),
                AssertionValue::Duration(assertion.max_duration_ms),
            )
        }
    }

    /// Evaluate anti-pattern assertion
    fn evaluate_anti_pattern(
        &self,
        name: &str,
        assertion: &AntiPatternAssertion,
        _trace: &UnifiedTrace,
    ) -> AssertionResult {
        // Planned (Sprint 44): Full integration with CausalGraph + AntiPatternDetector
        // For now, use placeholder logic

        // Placeholder: No anti-patterns detected
        AssertionResult::pass(
            name.to_string(),
            format!(
                "Anti-pattern {:?} not detected (placeholder implementation)",
                assertion.pattern
            ),
        )
    }

    /// Evaluate span count assertion
    fn evaluate_span_count(
        &self,
        name: &str,
        assertion: &SpanCountAssertion,
        trace: &UnifiedTrace,
    ) -> AssertionResult {
        let span_count = trace.syscall_spans.len();

        if span_count <= assertion.max_spans {
            AssertionResult::pass(
                name.to_string(),
                format!("Span count {} <= {}", span_count, assertion.max_spans),
            )
            .with_values(
                AssertionValue::Count(span_count),
                AssertionValue::Count(assertion.max_spans),
            )
        } else {
            AssertionResult::fail(
                name.to_string(),
                format!("Span count {} exceeds maximum {}", span_count, assertion.max_spans),
            )
            .with_values(
                AssertionValue::Count(span_count),
                AssertionValue::Count(assertion.max_spans),
            )
        }
    }

    /// Evaluate memory usage assertion
    fn evaluate_memory_usage(
        &self,
        name: &str,
        assertion: &MemoryUsageAssertion,
        trace: &UnifiedTrace,
    ) -> AssertionResult {
        // Sum up memory allocations from syscalls (mmap, brk)
        let mut total_bytes = 0u64;

        for span in &trace.syscall_spans {
            // Check if this is a memory allocation syscall
            if span.name == "mmap" || span.name == "brk" {
                // Parse allocation size from args (simplified)
                // In reality, we'd need proper syscall argument parsing
                total_bytes += 4096; // Placeholder: assume 4KB per allocation
            }
        }

        if total_bytes <= assertion.max_bytes {
            AssertionResult::pass(
                name.to_string(),
                format!("Memory usage {} bytes <= {}", total_bytes, assertion.max_bytes),
            )
            .with_values(
                AssertionValue::Bytes(total_bytes),
                AssertionValue::Bytes(assertion.max_bytes),
            )
        } else {
            AssertionResult::fail(
                name.to_string(),
                format!(
                    "Memory usage {} bytes exceeds maximum {}",
                    total_bytes, assertion.max_bytes
                ),
            )
            .with_values(
                AssertionValue::Bytes(total_bytes),
                AssertionValue::Bytes(assertion.max_bytes),
            )
        }
    }

    /// Evaluate all assertions against a trace
    ///
    /// # Arguments
    ///
    /// * `assertions` - List of assertions to evaluate
    /// * `trace` - The trace to evaluate against
    ///
    /// # Returns
    ///
    /// Vector of assertion results
    pub fn evaluate_all(
        &self,
        assertions: &[Assertion],
        trace: &UnifiedTrace,
    ) -> Vec<AssertionResult> {
        assertions.iter().map(|a| self.evaluate(a, trace)).collect()
    }

    /// Check if any assertions failed
    ///
    /// Returns true if any assertion with `fail_on_violation = true` failed.
    pub fn has_failures(results: &[AssertionResult], assertions: &[Assertion]) -> bool {
        results
            .iter()
            .zip(assertions.iter())
            .any(|(result, assertion)| !result.passed && assertion.fail_on_violation)
    }
}

impl Default for AssertionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assertion_types::{CriticalPathAssertion, SpanCountAssertion};
    use crate::trace_context::LamportClock;
    use crate::unified_trace::SyscallSpan;
    use std::borrow::Cow;

    fn create_test_span(syscall_name: &str, start_ns: u64, duration_ns: u64) -> SyscallSpan {
        let clock = LamportClock::new();
        SyscallSpan::new(
            1, // parent_span_id
            Cow::Owned(syscall_name.to_string()),
            vec![],
            0, // return_value
            start_ns,
            duration_ns,
            None, // errno
            &clock,
        )
    }

    #[test]
    fn test_evaluate_critical_path_pass() {
        let engine = AssertionEngine::new();

        let mut trace = UnifiedTrace::new(1, "test".to_string());
        trace.add_syscall(create_test_span("write", 0, 50_000_000)); // duration: 50ms

        let assertion = Assertion {
            name: "test".to_string(),
            assertion_type: AssertionType::CriticalPath(CriticalPathAssertion {
                max_duration_ms: 100,
                trace_name_pattern: None,
            }),
            fail_on_violation: true,
            enabled: true,
        };

        let result = engine.evaluate(&assertion, &trace);

        assert!(result.passed);
    }

    #[test]
    fn test_evaluate_critical_path_fail() {
        let engine = AssertionEngine::new();

        let mut trace = UnifiedTrace::new(1, "test".to_string());
        trace.add_syscall(create_test_span("write", 0, 150_000_000)); // duration: 150ms

        let assertion = Assertion {
            name: "test".to_string(),
            assertion_type: AssertionType::CriticalPath(CriticalPathAssertion {
                max_duration_ms: 100,
                trace_name_pattern: None,
            }),
            fail_on_violation: true,
            enabled: true,
        };

        let result = engine.evaluate(&assertion, &trace);

        assert!(!result.passed);
        assert!(result.message.contains("exceeds maximum"));
    }

    #[test]
    fn test_evaluate_span_count_pass() {
        let engine = AssertionEngine::new();

        let mut trace = UnifiedTrace::new(1, "test".to_string());
        for i in 0..50 {
            trace.add_syscall(create_test_span("write", i * 1000, 100)); // duration: 100ns
        }

        let assertion = Assertion {
            name: "test".to_string(),
            assertion_type: AssertionType::SpanCount(SpanCountAssertion {
                max_spans: 100,
                span_name_pattern: None,
            }),
            fail_on_violation: true,
            enabled: true,
        };

        let result = engine.evaluate(&assertion, &trace);

        assert!(result.passed);
    }

    #[test]
    fn test_evaluate_span_count_fail() {
        let engine = AssertionEngine::new();

        let mut trace = UnifiedTrace::new(1, "test".to_string());
        for i in 0..150 {
            trace.add_syscall(create_test_span("write", i * 1000, 100)); // duration: 100ns
        }

        let assertion = Assertion {
            name: "test".to_string(),
            assertion_type: AssertionType::SpanCount(SpanCountAssertion {
                max_spans: 100,
                span_name_pattern: None,
            }),
            fail_on_violation: true,
            enabled: true,
        };

        let result = engine.evaluate(&assertion, &trace);

        assert!(!result.passed);
    }

    #[test]
    fn test_evaluate_disabled_assertion() {
        let engine = AssertionEngine::new();

        let mut trace = UnifiedTrace::new(1, "test".to_string());
        trace.add_syscall(create_test_span("write", 0, 150_000_000)); // duration: 150ms (exceeds limit)

        let assertion = Assertion {
            name: "test".to_string(),
            assertion_type: AssertionType::CriticalPath(CriticalPathAssertion {
                max_duration_ms: 100,
                trace_name_pattern: None,
            }),
            fail_on_violation: true,
            enabled: false, // Disabled
        };

        let result = engine.evaluate(&assertion, &trace);

        // Should pass because assertion is disabled
        assert!(result.passed);
        assert_eq!(result.message, "Assertion disabled");
    }

    #[test]
    fn test_evaluate_all() {
        let engine = AssertionEngine::new();

        let mut trace = UnifiedTrace::new(1, "test".to_string());
        trace.add_syscall(create_test_span("write", 0, 50_000_000)); // duration: 50ms

        let assertions = vec![
            Assertion {
                name: "critical_path".to_string(),
                assertion_type: AssertionType::CriticalPath(CriticalPathAssertion {
                    max_duration_ms: 100,
                    trace_name_pattern: None,
                }),
                fail_on_violation: true,
                enabled: true,
            },
            Assertion {
                name: "span_count".to_string(),
                assertion_type: AssertionType::SpanCount(SpanCountAssertion {
                    max_spans: 10,
                    span_name_pattern: None,
                }),
                fail_on_violation: true,
                enabled: true,
            },
        ];

        let results = engine.evaluate_all(&assertions, &trace);

        assert_eq!(results.len(), 2);
        assert!(results[0].passed); // Critical path should pass
        assert!(results[1].passed); // Span count should pass (1 span < 10)
    }

    #[test]
    fn test_has_failures() {
        let results = vec![
            AssertionResult::pass("test1".to_string(), "Passed".to_string()),
            AssertionResult::fail("test2".to_string(), "Failed".to_string()),
        ];

        let assertions = vec![
            Assertion {
                name: "test1".to_string(),
                assertion_type: AssertionType::SpanCount(SpanCountAssertion {
                    max_spans: 10,
                    span_name_pattern: None,
                }),
                fail_on_violation: true,
                enabled: true,
            },
            Assertion {
                name: "test2".to_string(),
                assertion_type: AssertionType::SpanCount(SpanCountAssertion {
                    max_spans: 10,
                    span_name_pattern: None,
                }),
                fail_on_violation: true,
                enabled: true,
            },
        ];

        assert!(AssertionEngine::has_failures(&results, &assertions));
    }

    #[test]
    fn test_no_failures_when_fail_on_violation_false() {
        let results = vec![
            AssertionResult::pass("test1".to_string(), "Passed".to_string()),
            AssertionResult::fail("test2".to_string(), "Failed".to_string()),
        ];

        let assertions = vec![
            Assertion {
                name: "test1".to_string(),
                assertion_type: AssertionType::SpanCount(SpanCountAssertion {
                    max_spans: 10,
                    span_name_pattern: None,
                }),
                fail_on_violation: true,
                enabled: true,
            },
            Assertion {
                name: "test2".to_string(),
                assertion_type: AssertionType::SpanCount(SpanCountAssertion {
                    max_spans: 10,
                    span_name_pattern: None,
                }),
                fail_on_violation: false, // Don't fail on violation
                enabled: true,
            },
        ];

        // Should not have failures because test2 has fail_on_violation = false
        assert!(!AssertionEngine::has_failures(&results, &assertions));
    }

    #[test]
    fn test_evaluate_anti_pattern() {
        use crate::assertion_types::{AntiPatternAssertion, AntiPatternType};

        let engine = AssertionEngine::new();
        let mut trace = UnifiedTrace::new(1, "test".to_string());
        trace.add_syscall(create_test_span("write", 0, 100_000)); // duration: 100μs

        let assertion = Assertion {
            name: "test_anti_pattern".to_string(),
            assertion_type: AssertionType::AntiPattern(AntiPatternAssertion {
                pattern: AntiPatternType::GodProcess,
                threshold: 0.8,
                process_name_pattern: None,
            }),
            fail_on_violation: true,
            enabled: true,
        };

        let result = engine.evaluate(&assertion, &trace);
        // Should pass because anti-pattern detection is placeholder
        assert!(result.passed);
    }

    #[test]
    fn test_evaluate_memory_usage_pass() {
        use crate::assertion_types::{MemoryTrackingMode, MemoryUsageAssertion};

        let engine = AssertionEngine::new();
        let mut trace = UnifiedTrace::new(1, "test".to_string());
        // Add some non-memory syscalls
        trace.add_syscall(create_test_span("write", 0, 100_000));
        trace.add_syscall(create_test_span("read", 100_000, 100_000));

        let assertion = Assertion {
            name: "test_memory".to_string(),
            assertion_type: AssertionType::MemoryUsage(MemoryUsageAssertion {
                max_bytes: 1_000_000, // 1MB
                tracking_mode: MemoryTrackingMode::Allocations,
            }),
            fail_on_violation: true,
            enabled: true,
        };

        let result = engine.evaluate(&assertion, &trace);
        assert!(result.passed);
    }

    #[test]
    fn test_evaluate_memory_usage_with_mmap() {
        use crate::assertion_types::{MemoryTrackingMode, MemoryUsageAssertion};
        use std::borrow::Cow;

        let engine = AssertionEngine::new();
        let mut trace = UnifiedTrace::new(1, "test".to_string());

        // Add mmap syscalls
        let clock = LamportClock::new();
        let mmap_span =
            SyscallSpan::new(1, Cow::Borrowed("mmap"), vec![], 0, 0, 1000, None, &clock);
        trace.add_syscall(mmap_span);

        let assertion = Assertion {
            name: "test_memory".to_string(),
            assertion_type: AssertionType::MemoryUsage(MemoryUsageAssertion {
                max_bytes: 10_000, // 10KB (more than 4KB placeholder)
                tracking_mode: MemoryTrackingMode::Allocations,
            }),
            fail_on_violation: true,
            enabled: true,
        };

        let result = engine.evaluate(&assertion, &trace);
        assert!(result.passed);
    }

    #[test]
    fn test_evaluate_memory_usage_fail() {
        use crate::assertion_types::{MemoryTrackingMode, MemoryUsageAssertion};
        use std::borrow::Cow;

        let engine = AssertionEngine::new();
        let mut trace = UnifiedTrace::new(1, "test".to_string());

        // Add many mmap syscalls to exceed limit
        let clock = LamportClock::new();
        for _i in 0..10 {
            let mmap_span =
                SyscallSpan::new(1, Cow::Borrowed("mmap"), vec![], 0, 0, 1000, None, &clock);
            trace.add_syscall(mmap_span);
        }

        let assertion = Assertion {
            name: "test_memory".to_string(),
            assertion_type: AssertionType::MemoryUsage(MemoryUsageAssertion {
                max_bytes: 1000, // Very low limit (< 10 * 4KB placeholder)
                tracking_mode: MemoryTrackingMode::Allocations,
            }),
            fail_on_violation: true,
            enabled: true,
        };

        let result = engine.evaluate(&assertion, &trace);
        assert!(!result.passed);
    }

    #[test]
    fn test_evaluate_custom_assertion() {
        use crate::assertion_types::CustomAssertion;

        let engine = AssertionEngine::new();
        let mut trace = UnifiedTrace::new(1, "test".to_string());
        trace.add_syscall(create_test_span("write", 0, 100_000));

        let assertion = Assertion {
            name: "test_custom".to_string(),
            assertion_type: AssertionType::Custom(CustomAssertion {
                expression: "duration < 100ms".to_string(),
            }),
            fail_on_violation: true,
            enabled: true,
        };

        let result = engine.evaluate(&assertion, &trace);
        // Should pass because custom assertions not yet implemented
        assert!(result.passed);
        assert!(result.message.contains("not yet implemented"));
    }

    #[test]
    fn test_evaluate_empty_trace() {
        let engine = AssertionEngine::new();
        let trace = UnifiedTrace::new(1, "test".to_string());

        let assertion = Assertion {
            name: "test_empty".to_string(),
            assertion_type: AssertionType::CriticalPath(CriticalPathAssertion {
                max_duration_ms: 100,
                trace_name_pattern: None,
            }),
            fail_on_violation: true,
            enabled: true,
        };

        let result = engine.evaluate(&assertion, &trace);
        assert!(result.passed); // Empty trace has 0 duration
    }

    #[test]
    fn test_default_trait() {
        let engine: AssertionEngine = Default::default();
        let trace = UnifiedTrace::new(1, "test".to_string());

        let assertion = Assertion {
            name: "test".to_string(),
            assertion_type: AssertionType::SpanCount(SpanCountAssertion {
                max_spans: 10,
                span_name_pattern: None,
            }),
            fail_on_violation: true,
            enabled: true,
        };

        let result = engine.evaluate(&assertion, &trace);
        assert!(result.passed);
    }

    #[test]
    fn test_has_failures_with_all_passing() {
        let results = vec![
            AssertionResult::pass("test1".to_string(), "Passed".to_string()),
            AssertionResult::pass("test2".to_string(), "Passed".to_string()),
        ];

        let assertions = vec![
            Assertion {
                name: "test1".to_string(),
                assertion_type: AssertionType::SpanCount(SpanCountAssertion {
                    max_spans: 10,
                    span_name_pattern: None,
                }),
                fail_on_violation: true,
                enabled: true,
            },
            Assertion {
                name: "test2".to_string(),
                assertion_type: AssertionType::SpanCount(SpanCountAssertion {
                    max_spans: 10,
                    span_name_pattern: None,
                }),
                fail_on_violation: true,
                enabled: true,
            },
        ];

        assert!(!AssertionEngine::has_failures(&results, &assertions));
    }

    #[test]
    fn test_evaluate_brk_syscall() {
        use crate::assertion_types::{MemoryTrackingMode, MemoryUsageAssertion};
        use std::borrow::Cow;

        let engine = AssertionEngine::new();
        let mut trace = UnifiedTrace::new(1, "test".to_string());

        // Add brk syscall (alternative to mmap)
        let clock = LamportClock::new();
        let brk_span = SyscallSpan::new(1, Cow::Borrowed("brk"), vec![], 0, 0, 1000, None, &clock);
        trace.add_syscall(brk_span);

        let assertion = Assertion {
            name: "test_memory".to_string(),
            assertion_type: AssertionType::MemoryUsage(MemoryUsageAssertion {
                max_bytes: 10_000,
                tracking_mode: MemoryTrackingMode::Allocations,
            }),
            fail_on_violation: true,
            enabled: true,
        };

        let result = engine.evaluate(&assertion, &trace);
        assert!(result.passed);
    }
}
