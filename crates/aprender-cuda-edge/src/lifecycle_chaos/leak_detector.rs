//! GPU context memory leak detection.
//!
//! [`ContextLeakDetector`] compares memory snapshots before and after test
//! execution to detect leaked contexts. A tolerance of 1 MB accommodates
//! driver-internal allocations.

use serde::{Deserialize, Serialize};

/// Default leak tolerance in bytes (1 MB).
pub const LEAK_TOLERANCE_BYTES: u64 = 1_048_576;

/// Types of GPU resource leaks.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Leak {
    /// GPU memory was not freed.
    Memory {
        /// Bytes leaked.
        bytes: u64,
    },
    /// A CUDA context was not destroyed.
    Context {
        /// Context identifier (opaque handle).
        context_id: u64,
    },
    /// A CUDA stream was not destroyed.
    Stream {
        /// Stream identifier.
        stream_id: u64,
    },
}

impl std::fmt::Display for Leak {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Memory { bytes } => write!(f, "memory leak: {bytes} bytes"),
            Self::Context { context_id } => write!(f, "context leak: id={context_id}"),
            Self::Stream { stream_id } => write!(f, "stream leak: id={stream_id}"),
        }
    }
}

/// Report summarizing detected leaks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakReport {
    /// All detected leaks.
    pub leaks: Vec<Leak>,
    /// Memory before the test (bytes).
    pub memory_before: u64,
    /// Memory after the test (bytes).
    pub memory_after: u64,
    /// Tolerance applied (bytes).
    pub tolerance: u64,
}

impl LeakReport {
    /// Returns true if any leaks were detected.
    #[must_use]
    pub fn has_leaks(&self) -> bool {
        !self.leaks.is_empty()
    }

    /// Total bytes leaked across all memory leaks.
    #[must_use]
    pub fn total_leaked_bytes(&self) -> u64 {
        self.leaks
            .iter()
            .filter_map(|l| match l {
                Leak::Memory { bytes } => Some(*bytes),
                _ => None,
            })
            .sum()
    }
}

/// Detects GPU resource leaks by comparing pre/post memory snapshots.
#[derive(Debug, Clone)]
pub struct ContextLeakDetector {
    tolerance: u64,
}

impl ContextLeakDetector {
    /// Create a new leak detector with the default tolerance (1 MB).
    #[must_use]
    pub fn new() -> Self {
        Self {
            tolerance: LEAK_TOLERANCE_BYTES,
        }
    }

    /// Create a leak detector with a custom tolerance.
    #[must_use]
    pub fn with_tolerance(tolerance: u64) -> Self {
        Self { tolerance }
    }

    /// Returns the configured tolerance.
    #[must_use]
    pub fn tolerance(&self) -> u64 {
        self.tolerance
    }

    /// Analyze memory snapshots and report any leaks.
    ///
    /// If `memory_after > memory_before + tolerance`, a memory leak is
    /// reported.
    #[must_use]
    pub fn analyze(&self, memory_before: u64, memory_after: u64) -> LeakReport {
        let mut leaks = Vec::new();

        if memory_after > memory_before + self.tolerance {
            let leaked = memory_after - memory_before;
            leaks.push(Leak::Memory { bytes: leaked });
        }

        LeakReport {
            leaks,
            memory_before,
            memory_after,
            tolerance: self.tolerance,
        }
    }

    /// Analyze memory snapshots with additional context tracking.
    ///
    /// Each context ID in `contexts_before` that is not in `contexts_after`
    /// was destroyed. Each ID in `contexts_after` not in `contexts_before`
    /// is a leaked context.
    #[must_use]
    pub fn analyze_with_contexts(
        &self,
        memory_before: u64,
        memory_after: u64,
        contexts_before: &[u64],
        contexts_after: &[u64],
    ) -> LeakReport {
        let mut report = self.analyze(memory_before, memory_after);

        // Any context present after but not before is a leak
        for &ctx_id in contexts_after {
            if !contexts_before.contains(&ctx_id) {
                report.leaks.push(Leak::Context { context_id: ctx_id });
            }
        }

        report
    }
}

impl Default for ContextLeakDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_leak_within_tolerance() {
        let detector = ContextLeakDetector::new();
        let report = detector.analyze(100_000_000, 100_500_000);
        assert!(!report.has_leaks());
    }

    #[test]
    fn leak_above_tolerance() {
        let detector = ContextLeakDetector::new();
        let before = 100_000_000;
        let after = before + LEAK_TOLERANCE_BYTES + 1;
        let report = detector.analyze(before, after);
        assert!(report.has_leaks());
        assert_eq!(report.total_leaked_bytes(), LEAK_TOLERANCE_BYTES + 1);
    }

    #[test]
    fn memory_decrease_is_not_a_leak() {
        let detector = ContextLeakDetector::new();
        let report = detector.analyze(200_000_000, 100_000_000);
        assert!(!report.has_leaks());
    }

    #[test]
    fn context_leak_detected() {
        let detector = ContextLeakDetector::new();
        let report = detector.analyze_with_contexts(
            100_000_000,
            100_000_000,
            &[1, 2],
            &[1, 2, 3], // context 3 is new → leaked
        );
        assert!(report.has_leaks());
        assert!(report
            .leaks
            .iter()
            .any(|l| matches!(l, Leak::Context { context_id: 3 })));
    }

    #[test]
    fn custom_tolerance() {
        let detector = ContextLeakDetector::with_tolerance(100);
        let report = detector.analyze(1000, 1200);
        assert!(report.has_leaks());

        let report2 = detector.analyze(1000, 1050);
        assert!(!report2.has_leaks());
    }

    #[test]
    fn leak_display() {
        let leak = Leak::Memory { bytes: 4096 };
        assert!(leak.to_string().contains("4096"));

        let ctx_leak = Leak::Context { context_id: 42 };
        assert!(ctx_leak.to_string().contains("42"));
    }

    #[test]
    fn stream_leak_display() {
        let stream_leak = Leak::Stream { stream_id: 123 };
        assert!(stream_leak.to_string().contains("123"));
        assert!(stream_leak.to_string().contains("stream"));
    }

    #[test]
    fn detector_default() {
        let detector = ContextLeakDetector::default();
        assert_eq!(detector.tolerance(), LEAK_TOLERANCE_BYTES);
    }

    #[test]
    fn total_leaked_bytes_with_mixed_leaks() {
        let report = LeakReport {
            leaks: vec![
                Leak::Memory { bytes: 1000 },
                Leak::Context { context_id: 1 },
                Leak::Memory { bytes: 2000 },
                Leak::Stream { stream_id: 1 },
            ],
            memory_before: 0,
            memory_after: 3000,
            tolerance: LEAK_TOLERANCE_BYTES,
        };
        // Only Memory leaks are summed: 1000 + 2000 = 3000
        assert_eq!(report.total_leaked_bytes(), 3000);
    }

    #[test]
    fn total_leaked_bytes_no_memory_leaks() {
        let report = LeakReport {
            leaks: vec![
                Leak::Context { context_id: 1 },
                Leak::Stream { stream_id: 2 },
            ],
            memory_before: 0,
            memory_after: 0,
            tolerance: LEAK_TOLERANCE_BYTES,
        };
        assert_eq!(report.total_leaked_bytes(), 0);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn no_false_positive_within_tolerance(
            before in 0u64..1_000_000_000,
            delta in 0u64..LEAK_TOLERANCE_BYTES,
        ) {
            let detector = ContextLeakDetector::new();
            let after = before.saturating_add(delta);
            let report = detector.analyze(before, after);
            prop_assert!(!report.has_leaks(), "false positive: before={before}, after={after}");
        }
    }
}
