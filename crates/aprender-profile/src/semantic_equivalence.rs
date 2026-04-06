//! Semantic Equivalence Validation (Specification Section 6.3)
//!
//! Implements semantic equivalence verification for transpilation validation.
//! Used by Batuta Phase 4 to verify that transpiled programs preserve the
//! observable behavior of the original programs.
//!
//! # Definition 6.5 (Semantic Equivalence)
//!
//! Two programs P₁ and P₂ are semantically equivalent (P₁ ≈ P₂) if:
//! ```text
//! ∀ inputs I: Obs(P₁(I)) ≡ Obs(P₂(I))
//! ```
//!
//! Where Obs(P) = {`syscall_sequence`, `file_contents`, `network_messages`}
//!
//! # Relaxed Equivalence
//!
//! Programs are weakly equivalent if they differ only in:
//! 1. Allocator behavior (different mmap/brk syscalls)
//! 2. Timing (different execution times)
//! 3. Intermediate results (different memory layouts)
//!
//! # Reference
//!
//! Unified Tracing for Sovereign AI: Formal Specification v1.0
//! Section 6.3: Semantic Equivalence

use crate::unified_trace::{SyscallSpan, UnifiedTrace};

/// Validation result for semantic equivalence checking
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationResult {
    /// Validation passed with confidence score
    Pass {
        /// Confidence score (0.0-1.0, higher is better)
        confidence: f64,
        /// Number of matched syscalls
        matched_syscalls: usize,
        /// Performance comparison metrics
        performance: PerformanceComparison,
    },
    /// Validation failed with divergence information
    Fail {
        /// First point where traces diverged
        divergence_point: DivergencePoint,
        /// Detailed explanation of the difference
        explanation: String,
    },
}

/// Performance comparison between two traces
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceComparison {
    /// Runtime of original program (nanoseconds)
    pub original_runtime_nanos: u64,
    /// Runtime of transpiled program (nanoseconds)
    pub transpiled_runtime_nanos: u64,
    /// Speedup factor (original / transpiled)
    pub speedup: f64,
    /// Memory usage comparison (if available)
    pub memory_delta: Option<MemoryDelta>,
}

impl PerformanceComparison {
    /// Create a new performance comparison
    pub fn new(original_runtime_nanos: u64, transpiled_runtime_nanos: u64) -> Self {
        let speedup = if transpiled_runtime_nanos > 0 {
            original_runtime_nanos as f64 / transpiled_runtime_nanos as f64
        } else {
            1.0
        };

        PerformanceComparison {
            original_runtime_nanos,
            transpiled_runtime_nanos,
            speedup,
            memory_delta: None,
        }
    }

    /// Add memory usage comparison
    pub fn with_memory(mut self, original_bytes: usize, transpiled_bytes: usize) -> Self {
        self.memory_delta = Some(MemoryDelta {
            original_bytes,
            transpiled_bytes,
            reduction_percentage: if original_bytes > 0 {
                ((original_bytes as f64 - transpiled_bytes as f64) / original_bytes as f64) * 100.0
            } else {
                0.0
            },
        });
        self
    }
}

/// Memory usage comparison
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryDelta {
    /// Original program memory usage (bytes)
    pub original_bytes: usize,
    /// Transpiled program memory usage (bytes)
    pub transpiled_bytes: usize,
    /// Memory reduction percentage (positive = reduction, negative = increase)
    pub reduction_percentage: f64,
}

/// Point where two traces diverged
#[derive(Debug, Clone, PartialEq)]
pub struct DivergencePoint {
    /// Index of first divergent syscall
    pub syscall_index: usize,
    /// Original syscall at divergence point
    pub original_syscall: String,
    /// Transpiled syscall at divergence point
    pub transpiled_syscall: String,
}

/// Observable syscall (I/O operations only)
///
/// Represents a syscall that affects observable program behavior.
/// Internal operations like memory allocation are filtered out.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ObservableSyscall {
    /// Syscall name
    pub name: String,
    /// Relevant arguments (e.g., path for open, fd for read)
    pub args: Vec<String>,
    /// Return value
    pub return_value: i64,
}

impl ObservableSyscall {
    /// Create from a `SyscallSpan`
    pub fn from_syscall_span(span: &SyscallSpan) -> Self {
        ObservableSyscall {
            name: span.name.to_string(),
            args: span.args.iter().map(|(_, v)| v.clone()).collect(),
            return_value: span.return_value,
        }
    }

    /// Check if two syscalls are equivalent (allowing for minor differences)
    pub fn is_equivalent(&self, other: &ObservableSyscall) -> bool {
        // Name must match exactly
        if self.name != other.name {
            return false;
        }

        // Return values must be compatible (both success or both failure)
        let self_success = self.return_value >= 0;
        let other_success = other.return_value >= 0;
        if self_success != other_success {
            return false;
        }

        // For success cases, allow different but valid return values
        // (e.g., different file descriptors are OK)
        true
    }
}

/// Semantic equivalence validator
///
/// Compares two `UnifiedTrace` instances to determine if they represent
/// semantically equivalent programs.
pub struct SemanticValidator {
    /// Tolerance for fuzzy matching (default: 0.05 = 5%)
    tolerance: f64,
}

impl SemanticValidator {
    /// Create a new validator with default tolerance (5%)
    pub fn new() -> Self {
        SemanticValidator { tolerance: 0.05 }
    }

    /// Create validator with custom tolerance
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Tolerance for fuzzy matching (0.0-1.0)
    pub fn with_tolerance(tolerance: f64) -> Self {
        SemanticValidator { tolerance: tolerance.clamp(0.0, 1.0) }
    }

    /// Get current tolerance
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Validate semantic equivalence between two traces
    ///
    /// # Arguments
    ///
    /// * `original` - Trace from original program
    /// * `transpiled` - Trace from transpiled program
    ///
    /// # Returns
    ///
    /// `ValidationResult` indicating whether traces are semantically equivalent
    pub fn validate(&self, original: &UnifiedTrace, transpiled: &UnifiedTrace) -> ValidationResult {
        // Extract observable syscalls (I/O operations only)
        let obs_original = self.filter_observable_syscalls(original);
        let obs_transpiled = self.filter_observable_syscalls(transpiled);

        // Compare syscall sequences with tolerance
        let diff = self.diff_with_tolerance(&obs_original, &obs_transpiled);

        if diff.is_equivalent {
            // Calculate performance comparison
            let performance = self.calculate_performance(original, transpiled);

            ValidationResult::Pass {
                confidence: diff.similarity_score,
                matched_syscalls: diff.matched_count,
                performance,
            }
        } else {
            ValidationResult::Fail {
                divergence_point: diff.divergence_point.unwrap_or_else(|| DivergencePoint {
                    syscall_index: 0,
                    original_syscall: "unknown".to_string(),
                    transpiled_syscall: "unknown".to_string(),
                }),
                explanation: diff.explanation,
            }
        }
    }

    /// Filter observable syscalls from a trace
    ///
    /// Observable syscalls are I/O operations that affect program behavior:
    /// - File operations: open, read, write, close, stat, etc.
    /// - Network operations: socket, connect, send, recv, etc.
    /// - Process operations: fork, exec, wait, etc.
    ///
    /// Filtered out (non-observable):
    /// - Memory operations: mmap, munmap, brk (allocator internals)
    /// - Internal syscalls: futex, `clock_gettime` (implementation details)
    fn filter_observable_syscalls(&self, trace: &UnifiedTrace) -> Vec<ObservableSyscall> {
        let observable_syscalls = [
            // File I/O
            "open",
            "openat",
            "read",
            "write",
            "close",
            "stat",
            "fstat",
            "lstat",
            "lseek",
            "pread",
            "pwrite",
            "readv",
            "writev",
            "fsync",
            "fdatasync",
            "rename",
            "unlink",
            "mkdir",
            "rmdir",
            "chmod",
            "chown",
            "truncate",
            "ftruncate",
            // Network I/O
            "socket",
            "connect",
            "bind",
            "listen",
            "accept",
            "send",
            "recv",
            "sendto",
            "recvfrom",
            "sendmsg",
            "recvmsg",
            "shutdown",
            // Process operations
            "fork",
            "vfork",
            "clone",
            "exec",
            "execve",
            "wait",
            "waitpid",
            "exit",
            "kill",
            // Pipes and IPC
            "pipe",
            "pipe2",
            "dup",
            "dup2",
            "dup3",
        ];

        trace
            .syscall_spans
            .iter()
            .filter(|span| {
                let name: &str = &span.name;
                observable_syscalls.contains(&name)
            })
            .map(ObservableSyscall::from_syscall_span)
            .collect()
    }

    /// Compare two syscall sequences with tolerance
    fn diff_with_tolerance(
        &self,
        original: &[ObservableSyscall],
        transpiled: &[ObservableSyscall],
    ) -> TraceDiff {
        // Check if lengths are within tolerance
        let orig_len = original.len();
        let trans_len = transpiled.len();

        let max_len = orig_len.max(trans_len) as f64;
        let length_diff = (orig_len as f64 - trans_len as f64).abs();
        let length_similarity = if max_len > 0.0 { 1.0 - (length_diff / max_len) } else { 1.0 };

        // If length difference exceeds tolerance, fail immediately
        if length_similarity < (1.0 - self.tolerance) {
            return TraceDiff {
                is_equivalent: false,
                similarity_score: length_similarity,
                matched_count: 0,
                divergence_point: Some(DivergencePoint {
                    syscall_index: orig_len.min(trans_len),
                    original_syscall: format!("<end of trace, {orig_len} syscalls>"),
                    transpiled_syscall: format!("<end of trace, {trans_len} syscalls>"),
                }),
                explanation: format!(
                    "Length mismatch: original={orig_len}, transpiled={trans_len} (diff={length_diff})"
                ),
            };
        }

        // Compare syscalls pairwise
        let mut matched = 0;
        let mut divergence_point = None;

        for (i, (orig, trans)) in original.iter().zip(transpiled.iter()).enumerate() {
            if orig.is_equivalent(trans) {
                matched += 1;
            } else if divergence_point.is_none() {
                divergence_point = Some(DivergencePoint {
                    syscall_index: i,
                    original_syscall: format!("{} -> {}", orig.name, orig.return_value),
                    transpiled_syscall: format!("{} -> {}", trans.name, trans.return_value),
                });
            }
        }

        let min_len = orig_len.min(trans_len);
        let match_rate = if min_len > 0 { matched as f64 / min_len as f64 } else { 1.0 };

        let is_equivalent = match_rate >= (1.0 - self.tolerance);

        let explanation = if is_equivalent {
            format!(
                "Traces are equivalent: {}/{} syscalls matched ({:.1}%)",
                matched,
                min_len,
                match_rate * 100.0
            )
        } else {
            format!(
                "Traces diverged: only {}/{} syscalls matched ({:.1}%)",
                matched,
                min_len,
                match_rate * 100.0
            )
        };

        TraceDiff {
            is_equivalent,
            similarity_score: match_rate,
            matched_count: matched,
            divergence_point,
            explanation,
        }
    }

    /// Calculate performance comparison
    fn calculate_performance(
        &self,
        original: &UnifiedTrace,
        transpiled: &UnifiedTrace,
    ) -> PerformanceComparison {
        // Sum up syscall durations to get total runtime
        // This is more accurate than process span duration (Lamport clock ticks)
        let orig_runtime: u64 = original.syscall_spans.iter().map(|s| s.duration_nanos).sum();
        let trans_runtime: u64 = transpiled.syscall_spans.iter().map(|s| s.duration_nanos).sum();

        PerformanceComparison::new(orig_runtime, trans_runtime)
    }
}

impl Default for SemanticValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal struct for trace comparison results
struct TraceDiff {
    is_equivalent: bool,
    similarity_score: f64,
    matched_count: usize,
    divergence_point: Option<DivergencePoint>,
    explanation: String,
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(ValidationResult: Send, Sync);
static_assertions::assert_impl_all!(PerformanceComparison: Send, Sync);
static_assertions::assert_impl_all!(MemoryDelta: Send, Sync);
static_assertions::assert_impl_all!(DivergencePoint: Send, Sync);
static_assertions::assert_impl_all!(ObservableSyscall: Send, Sync);

// ============================================================================
// UNIT TESTS (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace_context::LamportClock;
    use std::borrow::Cow;

    // Helper function to create test trace
    fn create_test_trace(pid: i32, name: &str) -> UnifiedTrace {
        UnifiedTrace::new(pid, name.to_string())
    }

    // Helper function to add observable syscall
    fn add_observable_syscall(trace: &mut UnifiedTrace, name: &'static str, return_value: i64) {
        let parent_id = trace.process_span.span_id;
        let syscall = SyscallSpan::new(
            parent_id,
            Cow::Borrowed(name),
            vec![],
            return_value,
            trace.clock.now(),
            1000,
            None,
            &trace.clock,
        );
        trace.add_syscall(syscall);
    }

    // Test 1: Create validator with default tolerance
    #[test]
    fn test_validator_default() {
        let validator = SemanticValidator::new();
        assert!((validator.tolerance() - 0.05).abs() < 1e-6);
    }

    // Test 2: Create validator with custom tolerance
    #[test]
    fn test_validator_with_tolerance() {
        let validator = SemanticValidator::with_tolerance(0.1);
        assert!((validator.tolerance() - 0.1).abs() < 1e-6);
    }

    // Test 3: Tolerance clamping (too high)
    #[test]
    fn test_tolerance_clamp_high() {
        let validator = SemanticValidator::with_tolerance(1.5);
        assert!((validator.tolerance() - 1.0).abs() < 1e-6);
    }

    // Test 4: Tolerance clamping (too low)
    #[test]
    fn test_tolerance_clamp_low() {
        let validator = SemanticValidator::with_tolerance(-0.5);
        assert!((validator.tolerance() - 0.0).abs() < 1e-6);
    }

    // Test 5: ObservableSyscall equality (same syscall)
    #[test]
    fn test_observable_syscall_equality() {
        let syscall1 = ObservableSyscall {
            name: "read".to_string(),
            args: vec!["3".to_string(), "buf".to_string(), "100".to_string()],
            return_value: 100,
        };
        let syscall2 = ObservableSyscall {
            name: "read".to_string(),
            args: vec!["4".to_string(), "buf".to_string(), "100".to_string()],
            return_value: 150, // Different FD and return value OK
        };

        assert!(syscall1.is_equivalent(&syscall2));
    }

    // Test 6: ObservableSyscall inequality (different name)
    #[test]
    fn test_observable_syscall_different_name() {
        let syscall1 =
            ObservableSyscall { name: "read".to_string(), args: vec![], return_value: 100 };
        let syscall2 =
            ObservableSyscall { name: "write".to_string(), args: vec![], return_value: 100 };

        assert!(!syscall1.is_equivalent(&syscall2));
    }

    // Test 7: ObservableSyscall inequality (success vs failure)
    #[test]
    fn test_observable_syscall_success_vs_failure() {
        let syscall1 = ObservableSyscall {
            name: "open".to_string(),
            args: vec![],
            return_value: 3, // Success
        };
        let syscall2 = ObservableSyscall {
            name: "open".to_string(),
            args: vec![],
            return_value: -1, // Failure
        };

        assert!(!syscall1.is_equivalent(&syscall2));
    }

    // Test 8: Validate identical traces
    #[test]
    fn test_validate_identical() {
        let mut trace1 = create_test_trace(1000, "test1");
        let mut trace2 = create_test_trace(2000, "test2");

        add_observable_syscall(&mut trace1, "open", 3);
        add_observable_syscall(&mut trace1, "read", 100);
        add_observable_syscall(&mut trace1, "close", 0);

        add_observable_syscall(&mut trace2, "open", 4); // Different FD OK
        add_observable_syscall(&mut trace2, "read", 100);
        add_observable_syscall(&mut trace2, "close", 0);

        trace1.end_process(0);
        trace2.end_process(0);

        let validator = SemanticValidator::new();
        let result = validator.validate(&trace1, &trace2);

        match result {
            ValidationResult::Pass { confidence, matched_syscalls, .. } => {
                assert!(confidence >= 0.95);
                assert_eq!(matched_syscalls, 3);
            }
            ValidationResult::Fail { .. } => panic!("Expected Pass"),
        }
    }

    // Test 9: Validate divergent traces
    #[test]
    fn test_validate_divergent() {
        let mut trace1 = create_test_trace(1000, "test1");
        let mut trace2 = create_test_trace(2000, "test2");

        add_observable_syscall(&mut trace1, "open", 3);
        add_observable_syscall(&mut trace1, "read", 100);

        add_observable_syscall(&mut trace2, "open", 4);
        add_observable_syscall(&mut trace2, "write", 100); // Different syscall

        let validator = SemanticValidator::new();
        let result = validator.validate(&trace1, &trace2);

        match result {
            ValidationResult::Pass { .. } => panic!("Expected Fail"),
            ValidationResult::Fail { divergence_point, .. } => {
                assert_eq!(divergence_point.syscall_index, 1);
            }
        }
    }

    // Test 10: Filter observable syscalls
    #[test]
    fn test_filter_observable() {
        let mut trace = create_test_trace(1000, "test");

        // Observable syscalls
        add_observable_syscall(&mut trace, "open", 3);
        add_observable_syscall(&mut trace, "read", 100);

        // Non-observable syscalls (should be filtered)
        add_observable_syscall(&mut trace, "mmap", 0x1000);
        add_observable_syscall(&mut trace, "futex", 0);

        add_observable_syscall(&mut trace, "close", 0);

        let validator = SemanticValidator::new();
        let observable = validator.filter_observable_syscalls(&trace);

        // Should only have open, read, close (not mmap, futex)
        assert_eq!(observable.len(), 3);
        assert_eq!(observable[0].name, "open");
        assert_eq!(observable[1].name, "read");
        assert_eq!(observable[2].name, "close");
    }

    // Test 11: Performance comparison
    #[test]
    fn test_performance_comparison() {
        let perf = PerformanceComparison::new(1000000, 500000); // 2x speedup

        assert_eq!(perf.original_runtime_nanos, 1000000);
        assert_eq!(perf.transpiled_runtime_nanos, 500000);
        assert!((perf.speedup - 2.0).abs() < 1e-6);
        assert!(perf.memory_delta.is_none());
    }

    // Test 12: Performance comparison with memory
    #[test]
    fn test_performance_with_memory() {
        let perf = PerformanceComparison::new(1000000, 500000).with_memory(1000000, 600000); // 40% reduction

        assert!(perf.memory_delta.is_some());
        let mem = perf.memory_delta.expect("test");
        assert_eq!(mem.original_bytes, 1000000);
        assert_eq!(mem.transpiled_bytes, 600000);
        assert!((mem.reduction_percentage - 40.0).abs() < 0.1);
    }

    // Test 13: Default trait
    #[test]
    fn test_default_trait() {
        let validator: SemanticValidator = Default::default();
        assert!((validator.tolerance() - 0.05).abs() < 1e-6);
    }

    // Test 14: Validate with length mismatch within tolerance
    #[test]
    fn test_validate_length_within_tolerance() {
        let mut trace1 = create_test_trace(1000, "test1");
        let mut trace2 = create_test_trace(2000, "test2");

        // Original: 20 syscalls
        for _ in 0..20 {
            add_observable_syscall(&mut trace1, "read", 100);
        }

        // Transpiled: 21 syscalls (5% more, within tolerance)
        for _ in 0..21 {
            add_observable_syscall(&mut trace2, "read", 100);
        }

        trace1.end_process(0);
        trace2.end_process(0);

        let validator = SemanticValidator::new();
        let result = validator.validate(&trace1, &trace2);

        match result {
            ValidationResult::Pass { .. } => {} // Expected
            ValidationResult::Fail { explanation, .. } => {
                panic!("Expected Pass, got Fail: {}", explanation)
            }
        }
    }

    // Test 15: Validate with length mismatch beyond tolerance
    #[test]
    fn test_validate_length_beyond_tolerance() {
        let mut trace1 = create_test_trace(1000, "test1");
        let mut trace2 = create_test_trace(2000, "test2");

        // Original: 10 syscalls
        for _ in 0..10 {
            add_observable_syscall(&mut trace1, "read", 100);
        }

        // Transpiled: 20 syscalls (100% more, beyond 5% tolerance)
        for _ in 0..20 {
            add_observable_syscall(&mut trace2, "read", 100);
        }

        let validator = SemanticValidator::new();
        let result = validator.validate(&trace1, &trace2);

        match result {
            ValidationResult::Pass { .. } => panic!("Expected Fail"),
            ValidationResult::Fail { .. } => {} // Expected
        }
    }

    // Test 16: Empty traces are equivalent
    #[test]
    fn test_empty_traces_equivalent() {
        let mut trace1 = create_test_trace(1000, "test1");
        let mut trace2 = create_test_trace(2000, "test2");

        trace1.end_process(0);
        trace2.end_process(0);

        let validator = SemanticValidator::new();
        let result = validator.validate(&trace1, &trace2);

        match result {
            ValidationResult::Pass { confidence, .. } => {
                assert!((confidence - 1.0).abs() < 1e-6);
            }
            ValidationResult::Fail { .. } => panic!("Expected Pass"),
        }
    }

    // Test 17: From syscall span
    #[test]
    fn test_from_syscall_span() {
        let clock = LamportClock::new();
        let span = SyscallSpan::new(
            1,
            Cow::Borrowed("open"),
            vec![
                (Cow::Borrowed("path"), "/tmp/test.txt".to_string()),
                (Cow::Borrowed("flags"), "O_RDONLY".to_string()),
            ],
            3,
            clock.now(),
            1000,
            None,
            &clock,
        );

        let obs = ObservableSyscall::from_syscall_span(&span);

        assert_eq!(obs.name, "open");
        assert_eq!(obs.args.len(), 2);
        assert_eq!(obs.return_value, 3);
    }

    // Test 18: Divergence point details
    #[test]
    fn test_divergence_point_details() {
        let mut trace1 = create_test_trace(1000, "test1");
        let mut trace2 = create_test_trace(2000, "test2");

        add_observable_syscall(&mut trace1, "open", 3);
        add_observable_syscall(&mut trace1, "read", 100);
        add_observable_syscall(&mut trace1, "write", 50); // Divergence here

        add_observable_syscall(&mut trace2, "open", 4);
        add_observable_syscall(&mut trace2, "read", 100);
        add_observable_syscall(&mut trace2, "close", 0); // Different syscall

        let validator = SemanticValidator::new();
        let result = validator.validate(&trace1, &trace2);

        match result {
            ValidationResult::Pass { .. } => panic!("Expected Fail"),
            ValidationResult::Fail { divergence_point, .. } => {
                assert_eq!(divergence_point.syscall_index, 2);
                assert!(divergence_point.original_syscall.contains("write"));
                assert!(divergence_point.transpiled_syscall.contains("close"));
            }
        }
    }

    // Test 19: High tolerance accepts more differences
    #[test]
    fn test_high_tolerance() {
        let mut trace1 = create_test_trace(1000, "test1");
        let mut trace2 = create_test_trace(2000, "test2");

        // 10 syscalls
        for _ in 0..10 {
            add_observable_syscall(&mut trace1, "read", 100);
        }

        // 8 matching + 2 different = 80% match
        for _ in 0..8 {
            add_observable_syscall(&mut trace2, "read", 100);
        }
        add_observable_syscall(&mut trace2, "write", 100);
        add_observable_syscall(&mut trace2, "write", 100);

        // 5% tolerance: should fail (80% < 95%)
        let validator1 = SemanticValidator::with_tolerance(0.05);
        let result1 = validator1.validate(&trace1, &trace2);
        assert!(matches!(result1, ValidationResult::Fail { .. }));

        // 25% tolerance: should pass (80% > 75%)
        let validator2 = SemanticValidator::with_tolerance(0.25);
        let result2 = validator2.validate(&trace1, &trace2);
        assert!(matches!(result2, ValidationResult::Pass { .. }));
    }

    // Test 20: Memory delta calculation
    #[test]
    fn test_memory_delta() {
        let mem =
            MemoryDelta { original_bytes: 1000, transpiled_bytes: 800, reduction_percentage: 20.0 };

        assert_eq!(mem.original_bytes, 1000);
        assert_eq!(mem.transpiled_bytes, 800);
        assert!((mem.reduction_percentage - 20.0).abs() < 1e-6);
    }
}
