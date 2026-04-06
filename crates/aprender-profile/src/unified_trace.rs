//! Unified Trace Model for Sovereign AI Stack (Specification Section 3.1)
//!
//! Implements a hierarchical tracing model where all operations are represented
//! as spans with causal relationships. This unifies observability across:
//! - System calls (ptrace)
//! - GPU kernels (wgpu-profiler, CUPTI)
//! - SIMD compute blocks (Trueno integration)
//! - Transpiler decisions (Layer 5)
//!
//! # Reference
//!
//! Unified Tracing for Sovereign AI: Formal Specification v1.0
//! Section 3: System Architecture

use std::borrow::Cow;

use crate::decision_trace::DecisionTrace;
use crate::otlp_exporter::{ComputeBlock, GpuKernel, GpuMemoryTransfer};
use crate::trace_context::LamportClock;

/// Process span ID type
pub type SpanId = u64;

/// Process span representing the root lifecycle span
///
/// Each traced process gets exactly one `ProcessSpan` as the root of its trace tree.
#[derive(Debug, Clone)]
pub struct ProcessSpan {
    /// Process ID
    pub pid: i32,
    /// Process name (executable path)
    pub name: String,
    /// Span ID (unique within trace)
    pub span_id: SpanId,
    /// Start timestamp (nanoseconds, Lamport clock)
    pub start_timestamp_nanos: u64,
    /// End timestamp (nanoseconds, Lamport clock)
    pub end_timestamp_nanos: Option<u64>,
    /// Exit code (if process has exited)
    pub exit_code: Option<i32>,
}

impl ProcessSpan {
    /// Create a new process span
    pub fn new(pid: i32, name: String, clock: &LamportClock) -> Self {
        let span_id = clock.tick(); // Generate unique span ID
        let start_timestamp = clock.now(); // Timestamp is the current clock value
        ProcessSpan {
            pid,
            name,
            span_id,
            start_timestamp_nanos: start_timestamp,
            end_timestamp_nanos: None,
            exit_code: None,
        }
    }

    /// End the process span
    pub fn end(&mut self, clock: &LamportClock, exit_code: i32) {
        self.end_timestamp_nanos = Some(clock.tick());
        self.exit_code = Some(exit_code);
    }

    /// Get duration in nanoseconds
    pub fn duration_nanos(&self) -> Option<u64> {
        self.end_timestamp_nanos.map(|end| end.saturating_sub(self.start_timestamp_nanos))
    }
}

/// System call span (ptrace)
///
/// Represents a single syscall with arguments, return value, and timing.
/// Uses zero-copy Cow for syscall names (mostly static strings).
#[derive(Debug, Clone)]
pub struct SyscallSpan {
    /// Span ID (unique within trace)
    pub span_id: SpanId,
    /// Parent span ID (typically `ProcessSpan`)
    pub parent_span_id: SpanId,
    /// Syscall name (e.g., "open", "read", "write")
    pub name: Cow<'static, str>,
    /// Arguments as (key, value) pairs (zero-copy keys)
    pub args: Vec<(Cow<'static, str>, String)>,
    /// Return value (may be negative for errors)
    pub return_value: i64,
    /// Start timestamp (nanoseconds, Lamport clock)
    pub timestamp_nanos: u64,
    /// Duration in nanoseconds
    pub duration_nanos: u64,
    /// Error number (errno) if syscall failed
    pub errno: Option<i32>,
}

impl SyscallSpan {
    /// Create a new syscall span
    #[allow(clippy::too_many_arguments)] // Legitimate case: all parameters needed for span construction
    pub fn new(
        parent_span_id: SpanId,
        name: Cow<'static, str>,
        args: Vec<(Cow<'static, str>, String)>,
        return_value: i64,
        timestamp_nanos: u64,
        duration_nanos: u64,
        errno: Option<i32>,
        clock: &LamportClock,
    ) -> Self {
        SyscallSpan {
            span_id: clock.tick(),
            parent_span_id,
            name,
            args,
            return_value,
            timestamp_nanos,
            duration_nanos,
            errno,
        }
    }

    /// Check if syscall failed (return value < 0)
    pub fn is_error(&self) -> bool {
        self.return_value < 0
    }
}

/// Unified trace containing all span types
///
/// This is the central data structure that ties together all observability
/// layers in the Sovereign AI Stack.
#[derive(Debug, Clone)]
pub struct UnifiedTrace {
    /// Root span: process lifecycle
    pub process_span: ProcessSpan,
    /// System call spans (ptrace)
    pub syscall_spans: Vec<SyscallSpan>,
    /// GPU kernel spans (wgpu-profiler, CUPTI)
    pub gpu_spans: Vec<GpuKernel>,
    /// GPU memory transfer spans
    pub gpu_memory_transfers: Vec<GpuMemoryTransfer>,
    /// SIMD compute blocks (Trueno integration)
    pub simd_spans: Vec<ComputeBlock>,
    /// Transpiler decision points (Layer 5)
    pub transpiler_spans: Vec<DecisionTrace>,
    /// Lamport clock for causal ordering
    pub clock: LamportClock,
}

impl UnifiedTrace {
    /// Create a new unified trace for a process
    pub fn new(pid: i32, process_name: String) -> Self {
        let clock = LamportClock::new();
        let process_span = ProcessSpan::new(pid, process_name, &clock);

        UnifiedTrace {
            process_span,
            syscall_spans: Vec::new(),
            gpu_spans: Vec::new(),
            gpu_memory_transfers: Vec::new(),
            simd_spans: Vec::new(),
            transpiler_spans: Vec::new(),
            clock,
        }
    }

    /// Add a syscall span
    pub fn add_syscall(&mut self, span: SyscallSpan) {
        self.syscall_spans.push(span);
    }

    /// Add a GPU kernel span
    pub fn add_gpu_kernel(&mut self, kernel: GpuKernel) {
        self.gpu_spans.push(kernel);
    }

    /// Add a GPU memory transfer span
    pub fn add_gpu_memory_transfer(&mut self, transfer: GpuMemoryTransfer) {
        self.gpu_memory_transfers.push(transfer);
    }

    /// Add a SIMD compute block span
    pub fn add_compute_block(&mut self, block: ComputeBlock) {
        self.simd_spans.push(block);
    }

    /// Add a transpiler decision span
    pub fn add_transpiler_decision(&mut self, decision: DecisionTrace) {
        self.transpiler_spans.push(decision);
    }

    /// End the process span
    pub fn end_process(&mut self, exit_code: i32) {
        self.process_span.end(&self.clock, exit_code);
    }

    /// Find GPU spans that happened within a time window
    ///
    /// Used for correlating GPU kernels with their launching syscalls.
    pub fn find_gpu_spans_in_window(&self, start_nanos: u64, end_nanos: u64) -> Vec<&GpuKernel> {
        self.gpu_spans
            .iter()
            .filter(|kernel| {
                // GPU kernels are timestamped in microseconds, convert for comparison
                let kernel_start_nanos = kernel.duration_us * 1000;
                kernel_start_nanos >= start_nanos && kernel_start_nanos <= end_nanos
            })
            .collect()
    }

    /// Find the parent syscall for a GPU kernel
    ///
    /// Returns the syscall that likely launched this GPU kernel.
    /// Heuristic: Find the most recent ioctl/mmap syscall before the GPU kernel.
    pub fn find_parent_syscall(&self, gpu_timestamp_nanos: u64) -> Option<&SyscallSpan> {
        self.syscall_spans
            .iter()
            .filter(|syscall| {
                // Only consider ioctl and mmap syscalls (GPU submission)
                let name_str: &str = &syscall.name;
                (name_str == "ioctl" || name_str == "mmap")
                    && syscall.timestamp_nanos < gpu_timestamp_nanos
            })
            .max_by_key(|syscall| syscall.timestamp_nanos)
    }

    /// Check if span A happens-before span B
    ///
    /// Implements Lamport's happens-before relation as per Section 6.2:
    /// - **Transitivity**: a → b ∧ b → c ⇒ a → c
    /// - **Irreflexivity**: ¬(a → a)
    /// - **Timestamp consistency**: a → b ⇒ timestamp(a) < timestamp(b)
    ///
    /// # Arguments
    ///
    /// * `span_a_id` - Span ID of the first span
    /// * `span_b_id` - Span ID of the second span
    ///
    /// # Returns
    ///
    /// `true` if span A causally precedes span B, `false` otherwise
    ///
    /// # Algorithm
    ///
    /// 1. Check if B is a direct child of A (parent relationship)
    /// 2. Check if A's timestamp < B's timestamp (temporal ordering)
    /// 3. Walk B's parent chain to see if A is an ancestor
    pub fn happens_before(&self, span_a_id: SpanId, span_b_id: SpanId) -> bool {
        // Irreflexivity: a span cannot happen before itself
        if span_a_id == span_b_id {
            return false;
        }

        // Check direct parent-child relationship FIRST
        // This is the strongest form of causality
        if let Some(parent_id) = self.get_parent_span_id(span_b_id) {
            if parent_id == span_a_id {
                return true; // Direct parent
            }

            // Transitive closure: walk parent chain
            let mut current = parent_id;
            let mut visited = std::collections::HashSet::new();
            visited.insert(span_b_id); // Avoid infinite loops

            while let Some(next_parent) = self.get_parent_span_id(current) {
                if visited.contains(&next_parent) {
                    break; // Cycle detected, stop
                }
                visited.insert(next_parent);

                if next_parent == span_a_id {
                    return true; // Found A in B's ancestor chain
                }

                current = next_parent;
            }
        }

        // If no parent relationship found, check timestamps
        // Get timestamps
        let timestamp_a = self.get_span_timestamp(span_a_id);
        let timestamp_b = self.get_span_timestamp(span_b_id);

        // If either span not found, no causal relationship
        let (ts_a, ts_b) = match (timestamp_a, timestamp_b) {
            (Some(a), Some(b)) => (a, b),
            _ => return false,
        };

        // For non-parent relationships, timestamp consistency is required
        // but not sufficient for happens-before (could be concurrent siblings)
        // Return false as there's no causal relationship
        if ts_a >= ts_b {
            return false; // A cannot happen before B if A's timestamp >= B's
        }

        // Timestamps suggest A happened before B, but without a parent chain,
        // we cannot establish causality (they might be concurrent/siblings)
        false
    }

    /// Get the timestamp of a span by ID
    fn get_span_timestamp(&self, span_id: SpanId) -> Option<u64> {
        // Check process span
        if self.process_span.span_id == span_id {
            return Some(self.process_span.start_timestamp_nanos);
        }

        // Check syscall spans
        for span in &self.syscall_spans {
            if span.span_id == span_id {
                return Some(span.timestamp_nanos);
            }
        }

        // Future: Add GPU kernel timestamps when parent_span_id is added to GpuKernel
        // Future: Add compute block timestamps when parent_span_id is added

        None
    }

    /// Get the parent span ID of a span
    fn get_parent_span_id(&self, span_id: SpanId) -> Option<SpanId> {
        // Check syscall spans
        for span in &self.syscall_spans {
            if span.span_id == span_id {
                return Some(span.parent_span_id);
            }
        }

        // Process span has no parent
        if self.process_span.span_id == span_id {
            return None;
        }

        // Future: Add GPU kernel parent IDs when field is added
        // Future: Add compute block parent IDs when field is added

        None
    }

    /// Establish causal relationships between spans
    ///
    /// Implements happens-before ordering as per Section 6.2 of the specification.
    /// Attaches GPU spans to their launching syscalls based on timestamp correlation.
    pub fn correlate_spans(&mut self) {
        // Note: In full implementation, this would update parent_span_id fields
        // For now, we provide query methods (find_parent_syscall, etc.)
        // Future: Add parent_span_id field to GpuKernel and update it here
    }

    /// Get total number of spans across all types
    pub fn total_spans(&self) -> usize {
        1 + // process span
        self.syscall_spans.len() +
        self.gpu_spans.len() +
        self.gpu_memory_transfers.len() +
        self.simd_spans.len() +
        self.transpiler_spans.len()
    }

    /// Get total duration of the trace in nanoseconds
    pub fn total_duration_nanos(&self) -> Option<u64> {
        self.process_span.duration_nanos()
    }

    /// Analyze trace for architectural quality and anti-patterns (§27)
    ///
    /// Returns an optional `ArchitecturalQuality` assessment. Returns `None` if
    /// the trace has no data to analyze.
    ///
    /// # Example
    ///
    /// ```
    /// use renacer::unified_trace::UnifiedTrace;
    ///
    /// let trace = UnifiedTrace::new(1234, "my_app".to_string());
    /// if let Some(quality) = trace.architectural_quality() {
    ///     println!("Quality score: {}", quality.score);
    ///     for pattern in &quality.anti_patterns {
    ///         println!("Anti-pattern detected: {:?}", pattern);
    ///     }
    /// }
    /// ```
    pub fn architectural_quality(
        &self,
    ) -> Option<crate::analysis::anti_pattern::ArchitecturalQuality> {
        // Return None for empty traces (no data to analyze)
        if self.syscall_spans.is_empty() && self.gpu_spans.is_empty() && self.simd_spans.is_empty()
        {
            return None;
        }

        let detector = crate::analysis::anti_pattern::AntiPatternDetector::default();
        Some(detector.analyze(self))
    }
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(ProcessSpan: Send, Sync);
static_assertions::assert_impl_all!(SyscallSpan: Send, Sync);
static_assertions::assert_impl_all!(UnifiedTrace: Send, Sync);

// ============================================================================
// UNIT TESTS (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: Create new unified trace
    #[test]
    fn test_new_unified_trace() {
        let trace = UnifiedTrace::new(1234, "test_process".to_string());

        assert_eq!(trace.process_span.pid, 1234);
        assert_eq!(trace.process_span.name, "test_process");
        assert_eq!(trace.syscall_spans.len(), 0);
        assert_eq!(trace.gpu_spans.len(), 0);
        assert_eq!(trace.simd_spans.len(), 0);
        assert_eq!(trace.transpiler_spans.len(), 0);
    }

    // Test 2: Process span creation
    #[test]
    fn test_process_span_creation() {
        let clock = LamportClock::new();
        let span = ProcessSpan::new(5678, "my_app".to_string(), &clock);

        assert_eq!(span.pid, 5678);
        assert_eq!(span.name, "my_app");
        assert!(span.start_timestamp_nanos > 0);
        assert!(span.end_timestamp_nanos.is_none());
        assert!(span.exit_code.is_none());
    }

    // Test 3: Process span end
    #[test]
    fn test_process_span_end() {
        let clock = LamportClock::new();
        let mut span = ProcessSpan::new(1111, "app".to_string(), &clock);

        span.end(&clock, 0);

        assert!(span.end_timestamp_nanos.is_some());
        assert_eq!(span.exit_code, Some(0));
    }

    // Test 4: Process span duration
    #[test]
    fn test_process_span_duration() {
        let clock = LamportClock::new();
        let mut span = ProcessSpan::new(2222, "app".to_string(), &clock);

        assert!(span.duration_nanos().is_none());

        span.end(&clock, 0);
        let duration = span.duration_nanos();

        assert!(duration.is_some());
        assert!(duration.expect("test") > 0);
    }

    // Test 5: Syscall span creation
    #[test]
    fn test_syscall_span_creation() {
        let clock = LamportClock::new();
        let parent_id = clock.tick();

        let span = SyscallSpan::new(
            parent_id,
            Cow::Borrowed("open"),
            vec![
                (Cow::Borrowed("path"), "/tmp/test.txt".to_string()),
                (Cow::Borrowed("flags"), "O_RDONLY".to_string()),
            ],
            3, // fd = 3
            clock.now(),
            1234,
            None,
            &clock,
        );

        assert_eq!(&*span.name, "open");
        assert_eq!(span.args.len(), 2);
        assert_eq!(span.return_value, 3);
        assert!(!span.is_error());
    }

    // Test 6: Syscall span error detection
    #[test]
    fn test_syscall_span_error() {
        let clock = LamportClock::new();
        let parent_id = clock.tick();

        let span = SyscallSpan::new(
            parent_id,
            Cow::Borrowed("open"),
            vec![],
            -1, // error
            clock.now(),
            1234,
            Some(2), // ENOENT
            &clock,
        );

        assert!(span.is_error());
        assert_eq!(span.errno, Some(2));
    }

    // Test 7: Add syscall span to trace
    #[test]
    fn test_add_syscall_span() {
        let mut trace = UnifiedTrace::new(3333, "test".to_string());
        let parent_id = trace.process_span.span_id;

        let syscall = SyscallSpan::new(
            parent_id,
            Cow::Borrowed("read"),
            vec![],
            100,
            trace.clock.now(),
            5678,
            None,
            &trace.clock,
        );

        trace.add_syscall(syscall);

        assert_eq!(trace.syscall_spans.len(), 1);
        assert_eq!(&*trace.syscall_spans[0].name, "read");
    }

    // Test 8: Add GPU kernel to trace
    #[test]
    fn test_add_gpu_kernel() {
        let mut trace = UnifiedTrace::new(4444, "gpu_app".to_string());

        let kernel = GpuKernel {
            kernel: "matmul".to_string(),
            duration_us: 1234,
            backend: "wgpu",
            workgroup_size: Some("[256,1,1]".to_string()),
            elements: Some(65536),
            is_slow: true,
        };

        trace.add_gpu_kernel(kernel);

        assert_eq!(trace.gpu_spans.len(), 1);
        assert_eq!(trace.gpu_spans[0].kernel, "matmul");
    }

    // Test 9: Add compute block to trace
    #[test]
    fn test_add_compute_block() {
        let mut trace = UnifiedTrace::new(5555, "simd_app".to_string());

        let block = ComputeBlock {
            operation: "calculate_statistics",
            duration_us: 567,
            elements: 10000,
            is_slow: false,
        };

        trace.add_compute_block(block);

        assert_eq!(trace.simd_spans.len(), 1);
        assert_eq!(trace.simd_spans[0].operation, "calculate_statistics");
    }

    // Test 10: End process
    #[test]
    fn test_end_process() {
        let mut trace = UnifiedTrace::new(6666, "ending_app".to_string());

        trace.end_process(0);

        assert!(trace.process_span.end_timestamp_nanos.is_some());
        assert_eq!(trace.process_span.exit_code, Some(0));
    }

    // Test 11: Total spans count
    #[test]
    fn test_total_spans() {
        let mut trace = UnifiedTrace::new(7777, "multi_span".to_string());

        assert_eq!(trace.total_spans(), 1); // Just process span

        let parent_id = trace.process_span.span_id;
        trace.add_syscall(SyscallSpan::new(
            parent_id,
            Cow::Borrowed("open"),
            vec![],
            3,
            trace.clock.now(),
            100,
            None,
            &trace.clock,
        ));

        assert_eq!(trace.total_spans(), 2); // Process + 1 syscall

        trace.add_gpu_kernel(GpuKernel {
            kernel: "compute".to_string(),
            duration_us: 200,
            backend: "wgpu",
            workgroup_size: None,
            elements: None,
            is_slow: false,
        });

        assert_eq!(trace.total_spans(), 3); // Process + syscall + GPU
    }

    // Test 12: Total duration
    #[test]
    fn test_total_duration() {
        let mut trace = UnifiedTrace::new(8888, "duration_app".to_string());

        assert!(trace.total_duration_nanos().is_none());

        trace.end_process(0);

        let duration = trace.total_duration_nanos();
        assert!(duration.is_some());
        assert!(duration.expect("test") > 0);
    }

    // Test 13: Find parent syscall for GPU kernel
    #[test]
    fn test_find_parent_syscall() {
        let mut trace = UnifiedTrace::new(9999, "gpu_launch".to_string());
        let parent_id = trace.process_span.span_id;

        // Add an ioctl syscall
        let ioctl_timestamp = trace.clock.now();
        trace.add_syscall(SyscallSpan::new(
            parent_id,
            Cow::Borrowed("ioctl"),
            vec![(Cow::Borrowed("cmd"), "DRM_IOCTL_SUBMIT".to_string())],
            0,
            ioctl_timestamp,
            100,
            None,
            &trace.clock,
        ));

        // Find parent for GPU kernel that happened after ioctl
        let parent = trace.find_parent_syscall(ioctl_timestamp + 1000);

        assert!(parent.is_some());
        assert_eq!(&*parent.expect("test").name, "ioctl");
    }

    // Test 14: Find parent returns None when no suitable syscall
    #[test]
    fn test_find_parent_syscall_none() {
        let trace = UnifiedTrace::new(10000, "no_gpu".to_string());

        // No ioctl/mmap syscalls, should return None
        let parent = trace.find_parent_syscall(12345);

        assert!(parent.is_none());
    }

    // Test 15: Correlate spans (basic)
    #[test]
    fn test_correlate_spans() {
        let mut trace = UnifiedTrace::new(11000, "correlate_app".to_string());

        // This is currently a no-op, but should not panic
        trace.correlate_spans();
        assert_eq!(trace.syscall_spans.len(), 0);
    }

    // Test 16: Zero-copy syscall names
    #[test]
    fn test_zero_copy_syscall_names() {
        let clock = LamportClock::new();
        let parent_id = clock.tick();

        let span = SyscallSpan::new(
            parent_id,
            Cow::Borrowed("read"), // Static string, no allocation
            vec![],
            100,
            clock.now(),
            1000,
            None,
            &clock,
        );

        // Verify it's borrowed (no allocation)
        assert!(matches!(span.name, Cow::Borrowed(_)));
    }

    // Test 17: Multiple syscalls preserve order
    #[test]
    fn test_multiple_syscalls_ordering() {
        let mut trace = UnifiedTrace::new(12000, "ordered".to_string());
        let parent_id = trace.process_span.span_id;

        for i in 0..10 {
            let syscall = SyscallSpan::new(
                parent_id,
                Cow::Borrowed("read"),
                vec![],
                i,
                trace.clock.now(),
                1000,
                None,
                &trace.clock,
            );
            trace.add_syscall(syscall);
        }

        assert_eq!(trace.syscall_spans.len(), 10);

        // Verify timestamps are increasing
        for i in 1..trace.syscall_spans.len() {
            assert!(
                trace.syscall_spans[i].timestamp_nanos > trace.syscall_spans[i - 1].timestamp_nanos
            );
        }
    }

    // Test 18: GPU memory transfer
    #[test]
    fn test_add_gpu_memory_transfer() {
        let mut trace = UnifiedTrace::new(13000, "gpu_memory".to_string());

        let transfer = GpuMemoryTransfer {
            label: "mesh_upload".to_string(),
            direction: crate::otlp_exporter::TransferDirection::CpuToGpu,
            bytes: 1048576, // 1 MB
            duration_us: 500,
            bandwidth_mbps: 2048.0,
            buffer_usage: Some("VERTEX".to_string()),
            is_slow: false,
        };

        trace.add_gpu_memory_transfer(transfer);

        assert_eq!(trace.gpu_memory_transfers.len(), 1);
        assert_eq!(trace.gpu_memory_transfers[0].label, "mesh_upload");
    }

    // Test 19: Clone trait
    #[test]
    fn test_unified_trace_clone() {
        let mut trace1 = UnifiedTrace::new(14000, "clone_test".to_string());
        let parent_id = trace1.process_span.span_id;

        trace1.add_syscall(SyscallSpan::new(
            parent_id,
            Cow::Borrowed("open"),
            vec![],
            3,
            trace1.clock.now(),
            100,
            None,
            &trace1.clock,
        ));

        let trace2 = trace1.clone();

        assert_eq!(trace2.syscall_spans.len(), 1);
        assert_eq!(trace2.process_span.pid, 14000);
    }

    // Test 20: Debug trait
    #[test]
    fn test_unified_trace_debug() {
        let trace = UnifiedTrace::new(15000, "debug_test".to_string());
        let debug_str = format!("{:?}", trace);

        assert!(debug_str.contains("UnifiedTrace"));
        assert!(debug_str.contains("ProcessSpan"));
    }

    // ========================================================================
    // HAPPENS-BEFORE ORDERING TESTS (Section 6.2)
    // ========================================================================

    // Test 21: Happens-before irreflexivity (a span doesn't happen before itself)
    #[test]
    fn test_happens_before_irreflexivity() {
        let mut trace = UnifiedTrace::new(16000, "causal_test".to_string());
        let process_span_id = trace.process_span.span_id;

        // A span cannot happen before itself
        assert!(!trace.happens_before(process_span_id, process_span_id));

        // Add a syscall and test
        let parent_id = trace.process_span.span_id;
        let syscall = SyscallSpan::new(
            parent_id,
            Cow::Borrowed("read"),
            vec![],
            100,
            trace.clock.now(),
            1000,
            None,
            &trace.clock,
        );
        let syscall_id = syscall.span_id;
        trace.add_syscall(syscall);

        assert!(!trace.happens_before(syscall_id, syscall_id));
    }

    // Test 22: Happens-before with direct parent-child relationship
    #[test]
    fn test_happens_before_direct_parent() {
        let mut trace = UnifiedTrace::new(17000, "parent_child".to_string());
        let parent_id = trace.process_span.span_id;

        let syscall = SyscallSpan::new(
            parent_id,
            Cow::Borrowed("open"),
            vec![],
            3,
            trace.clock.now(),
            1000,
            None,
            &trace.clock,
        );
        let child_id = syscall.span_id;
        trace.add_syscall(syscall);

        // Parent happens-before child
        assert!(trace.happens_before(parent_id, child_id));

        // Child does NOT happen-before parent
        assert!(!trace.happens_before(child_id, parent_id));
    }

    // Test 23: Happens-before transitivity (grandparent → parent → child)
    #[test]
    fn test_happens_before_transitivity() {
        let mut trace = UnifiedTrace::new(18000, "transitive".to_string());
        let grandparent_id = trace.process_span.span_id;

        // Create parent syscall
        let parent = SyscallSpan::new(
            grandparent_id,
            Cow::Borrowed("open"),
            vec![],
            3,
            trace.clock.now(),
            1000,
            None,
            &trace.clock,
        );
        let parent_id = parent.span_id;
        trace.add_syscall(parent);

        // Create child syscall (child of previous syscall)
        let child = SyscallSpan::new(
            parent_id,
            Cow::Borrowed("read"),
            vec![],
            100,
            trace.clock.now(),
            2000,
            None,
            &trace.clock,
        );
        let child_id = child.span_id;
        trace.add_syscall(child);

        // Transitive relationship: grandparent → parent → child
        assert!(trace.happens_before(grandparent_id, parent_id));
        assert!(trace.happens_before(parent_id, child_id));
        assert!(trace.happens_before(grandparent_id, child_id)); // Transitivity!
    }

    // Test 24: Happens-before with sibling spans (no causal relationship)
    #[test]
    fn test_happens_before_siblings() {
        let mut trace = UnifiedTrace::new(19000, "siblings".to_string());
        let parent_id = trace.process_span.span_id;

        // Create two sibling syscalls (both children of process span)
        let sibling1 = SyscallSpan::new(
            parent_id,
            Cow::Borrowed("open"),
            vec![],
            3,
            trace.clock.now(),
            1000,
            None,
            &trace.clock,
        );
        let sibling1_id = sibling1.span_id;
        trace.add_syscall(sibling1);

        let sibling2 = SyscallSpan::new(
            parent_id,
            Cow::Borrowed("read"),
            vec![],
            100,
            trace.clock.now(),
            2000,
            None,
            &trace.clock,
        );
        let sibling2_id = sibling2.span_id;
        trace.add_syscall(sibling2);

        // Siblings have no causal relationship (even though timestamps differ)
        // They both have the same parent, so neither happens-before the other
        assert!(!trace.happens_before(sibling1_id, sibling2_id));
        assert!(!trace.happens_before(sibling2_id, sibling1_id));

        // But parent happens-before both siblings
        assert!(trace.happens_before(parent_id, sibling1_id));
        assert!(trace.happens_before(parent_id, sibling2_id));
    }

    // Test 25: Happens-before with non-existent spans
    #[test]
    fn test_happens_before_nonexistent() {
        let trace = UnifiedTrace::new(20000, "nonexistent".to_string());

        // Non-existent span IDs
        assert!(!trace.happens_before(99999, 88888));
        assert!(!trace.happens_before(trace.process_span.span_id, 99999));
        assert!(!trace.happens_before(99999, trace.process_span.span_id));
    }

    // Test 26: Happens-before timestamp consistency
    #[test]
    fn test_happens_before_timestamp_consistency() {
        let mut trace = UnifiedTrace::new(21000, "timestamp_check".to_string());
        let parent_id = trace.process_span.span_id;

        let syscall1 = SyscallSpan::new(
            parent_id,
            Cow::Borrowed("open"),
            vec![],
            3,
            trace.clock.now(),
            1000,
            None,
            &trace.clock,
        );
        let span1_id = syscall1.span_id;
        let span1_ts = syscall1.timestamp_nanos;
        trace.add_syscall(syscall1);

        let syscall2 = SyscallSpan::new(
            parent_id,
            Cow::Borrowed("read"),
            vec![],
            100,
            trace.clock.now(),
            2000,
            None,
            &trace.clock,
        );
        let span2_id = syscall2.span_id;
        let span2_ts = syscall2.timestamp_nanos;
        trace.add_syscall(syscall2);

        // Verify timestamp ordering
        assert!(span1_ts < span2_ts);

        // But no happens-before relationship (siblings)
        assert!(!trace.happens_before(span1_id, span2_id));
    }

    // Test 27: Happens-before with long parent chain
    #[test]
    fn test_happens_before_long_chain() {
        let mut trace = UnifiedTrace::new(22000, "long_chain".to_string());

        // Build a chain: root → span1 → span2 → span3 → span4 → span5
        let root_id = trace.process_span.span_id;

        let span1 = SyscallSpan::new(
            root_id,
            Cow::Borrowed("open"),
            vec![],
            3,
            trace.clock.now(),
            100,
            None,
            &trace.clock,
        );
        let span1_id = span1.span_id;
        trace.add_syscall(span1);

        let span2 = SyscallSpan::new(
            span1_id,
            Cow::Borrowed("read"),
            vec![],
            100,
            trace.clock.now(),
            200,
            None,
            &trace.clock,
        );
        let span2_id = span2.span_id;
        trace.add_syscall(span2);

        let span3 = SyscallSpan::new(
            span2_id,
            Cow::Borrowed("write"),
            vec![],
            100,
            trace.clock.now(),
            300,
            None,
            &trace.clock,
        );
        let span3_id = span3.span_id;
        trace.add_syscall(span3);

        let span4 = SyscallSpan::new(
            span3_id,
            Cow::Borrowed("close"),
            vec![],
            0,
            trace.clock.now(),
            400,
            None,
            &trace.clock,
        );
        let span4_id = span4.span_id;
        trace.add_syscall(span4);

        let span5 = SyscallSpan::new(
            span4_id,
            Cow::Borrowed("fsync"),
            vec![],
            0,
            trace.clock.now(),
            500,
            None,
            &trace.clock,
        );
        let span5_id = span5.span_id;
        trace.add_syscall(span5);

        // Test transitive relationships across entire chain
        assert!(trace.happens_before(root_id, span1_id));
        assert!(trace.happens_before(root_id, span2_id));
        assert!(trace.happens_before(root_id, span3_id));
        assert!(trace.happens_before(root_id, span4_id));
        assert!(trace.happens_before(root_id, span5_id));

        assert!(trace.happens_before(span1_id, span5_id));
        assert!(trace.happens_before(span2_id, span5_id));
        assert!(trace.happens_before(span3_id, span5_id));
        assert!(trace.happens_before(span4_id, span5_id));

        // Reverse relationships should not hold
        assert!(!trace.happens_before(span5_id, span1_id));
        assert!(!trace.happens_before(span4_id, span2_id));
    }

    // Test 28: Get span timestamp (process span)
    #[test]
    fn test_get_span_timestamp_process() {
        let trace = UnifiedTrace::new(23000, "timestamp_test".to_string());
        let process_id = trace.process_span.span_id;

        let timestamp = trace.get_span_timestamp(process_id);
        assert!(timestamp.is_some());
        assert_eq!(timestamp.expect("test"), trace.process_span.start_timestamp_nanos);
    }

    // Test 29: Get span timestamp (syscall span)
    #[test]
    fn test_get_span_timestamp_syscall() {
        let mut trace = UnifiedTrace::new(24000, "syscall_ts".to_string());
        let parent_id = trace.process_span.span_id;

        let syscall = SyscallSpan::new(
            parent_id,
            Cow::Borrowed("open"),
            vec![],
            3,
            trace.clock.now(),
            1000,
            None,
            &trace.clock,
        );
        let syscall_id = syscall.span_id;
        let syscall_ts = syscall.timestamp_nanos;
        trace.add_syscall(syscall);

        let timestamp = trace.get_span_timestamp(syscall_id);
        assert!(timestamp.is_some());
        assert_eq!(timestamp.expect("test"), syscall_ts);
    }

    // Test 30: Get parent span ID (process has no parent)
    #[test]
    fn test_get_parent_span_id_process() {
        let trace = UnifiedTrace::new(25000, "parent_test".to_string());
        let process_id = trace.process_span.span_id;

        let parent = trace.get_parent_span_id(process_id);
        assert!(parent.is_none()); // Process span has no parent
    }

    // Test 31: Get parent span ID (syscall has parent)
    #[test]
    fn test_get_parent_span_id_syscall() {
        let mut trace = UnifiedTrace::new(26000, "syscall_parent".to_string());
        let parent_id = trace.process_span.span_id;

        let syscall = SyscallSpan::new(
            parent_id,
            Cow::Borrowed("open"),
            vec![],
            3,
            trace.clock.now(),
            1000,
            None,
            &trace.clock,
        );
        let syscall_id = syscall.span_id;
        trace.add_syscall(syscall);

        let retrieved_parent = trace.get_parent_span_id(syscall_id);
        assert!(retrieved_parent.is_some());
        assert_eq!(retrieved_parent.expect("test"), parent_id);
    }

    // ========================================================================
    // ARCHITECTURAL QUALITY TESTS (§27)
    // ========================================================================

    // Test 32: Empty trace returns None for architectural quality
    #[test]
    fn test_architectural_quality_empty_trace() {
        let trace = UnifiedTrace::new(27000, "empty_quality".to_string());

        let quality = trace.architectural_quality();
        assert!(quality.is_none(), "Empty trace should return None for architectural quality");
    }

    // Test 33: Trace with syscalls returns Some for architectural quality
    #[test]
    fn test_architectural_quality_with_syscalls() {
        let mut trace = UnifiedTrace::new(28000, "quality_test".to_string());
        let parent_id = trace.process_span.span_id;

        // Add some syscalls
        for _ in 0..10 {
            let syscall = SyscallSpan::new(
                parent_id,
                Cow::Borrowed("read"),
                vec![],
                100,
                trace.clock.now(),
                1000,
                None,
                &trace.clock,
            );
            trace.add_syscall(syscall);
        }

        let quality = trace.architectural_quality();
        assert!(
            quality.is_some(),
            "Trace with syscalls should return Some for architectural quality"
        );

        let quality = quality.expect("test");
        // Should have a quality score between 0 and 1
        assert!(
            quality.score >= 0.0 && quality.score <= 1.0,
            "Quality score should be between 0 and 1, got {}",
            quality.score
        );
    }

    // Test 34: Architectural quality detects anti-patterns
    #[test]
    fn test_architectural_quality_detects_antipatterns() {
        let mut trace = UnifiedTrace::new(29000, "antipattern_test".to_string());
        let parent_id = trace.process_span.span_id;

        // Create 100 syscalls (should trigger God Process detection)
        for _ in 0..100 {
            let syscall = SyscallSpan::new(
                parent_id,
                Cow::Borrowed("write"),
                vec![],
                50,
                trace.clock.now(),
                500,
                None,
                &trace.clock,
            );
            trace.add_syscall(syscall);
        }

        let quality = trace.architectural_quality().expect("Should have quality");

        // With 100% syscalls from one process, should detect God Process
        assert!(!quality.anti_patterns.is_empty(), "Should detect at least one anti-pattern");
    }
}
