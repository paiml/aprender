//! E2E Visual Testing Framework for GPU Kernels
//!
//! This module provides pixel-level visual regression testing for GPU computations
//! using the **sovereign stack only** - NO external crates.
//!
//! # Architecture (Sovereign Stack)
//!
//! ```text
//! GPU Output → GpuPixelRenderer → trueno-viz → PNG → compare_png_bytes → Pass/Fail
//!                                                           ↑
//!                                                    Golden Baseline
//! ```
//!
//! # Dependencies (Sovereign Stack)
//!
//! - `trueno-viz` v0.1.4: PNG encoding, Framebuffer (optional, feature = "viz")
//! - `simular` v0.2.0: Deterministic RNG for reproducible tests
//! - `renacer` v0.7.0: Profiling and anomaly detection (optional)
//!
//! # Features
//!
//! - `viz`: Enable GPU pixel renderer with trueno-viz
//! - `stress-test`: Enable randomized frame-by-frame stress testing
//! - `tui-monitor`: Enable TUI monitoring mode via presentar

pub mod stress;
pub mod tui;

pub use stress::{
    verify_performance, Anomaly, AnomalyKind, FrameProfile, PerformanceResult,
    PerformanceThresholds, StressConfig, StressReport, StressRng, StressTestRunner,
};

pub use tui::{progress_bar, render_to_string, TuiConfig, TuiState};

/// GPU-specific bug classification based on diff patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BugClass {
    /// Race condition: Non-deterministic output
    RaceCondition,
    /// Floating-point precision drift
    FloatingPointDrift,
    /// Accumulator not initialized to zero
    AccumulatorInit,
    /// Loop counter SSA bug
    LoopCounter,
    /// Memory addressing error
    MemoryAddressing,
    /// Thread synchronization issue
    ThreadSync,
    /// Unknown pattern
    Unknown,
}

impl BugClass {
    /// Description of the bug class
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::RaceCondition => "Race condition: non-deterministic output",
            Self::FloatingPointDrift => "FP precision drift in accumulation",
            Self::AccumulatorInit => "Accumulator not initialized to zero",
            Self::LoopCounter => "Loop counter SSA bug (wrong iteration count)",
            Self::MemoryAddressing => "Memory addressing error (offset/alignment)",
            Self::ThreadSync => "Thread synchronization issue (barrier)",
            Self::Unknown => "Unknown bug pattern",
        }
    }

    /// Suggested fix for the bug class
    #[must_use]
    pub const fn suggested_fix(&self) -> &'static str {
        match self {
            Self::RaceCondition => "Add __syncthreads() / bar.sync; use atomics",
            Self::FloatingPointDrift => "Use Kahan summation or pairwise reduction",
            Self::AccumulatorInit => "Initialize accumulator to 0.0 before loop",
            Self::LoopCounter => "Fix loop bound; use in-place += instead of reassignment",
            Self::MemoryAddressing => "Check index calculations and stride",
            Self::ThreadSync => "Add barrier synchronization at workgroup boundaries",
            Self::Unknown => "Manual inspection required",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bug_class_descriptions() {
        // Test all variants have non-empty descriptions
        let variants = [
            BugClass::RaceCondition,
            BugClass::FloatingPointDrift,
            BugClass::AccumulatorInit,
            BugClass::LoopCounter,
            BugClass::MemoryAddressing,
            BugClass::ThreadSync,
            BugClass::Unknown,
        ];

        for variant in variants {
            assert!(
                !variant.description().is_empty(),
                "{variant:?} has empty description"
            );
            assert!(
                !variant.suggested_fix().is_empty(),
                "{variant:?} has empty fix"
            );
        }
    }

    #[test]
    fn test_bug_class_equality() {
        assert_eq!(BugClass::RaceCondition, BugClass::RaceCondition);
        assert_ne!(BugClass::RaceCondition, BugClass::Unknown);
    }

    #[test]
    fn test_bug_class_clone() {
        let original = BugClass::FloatingPointDrift;
        let cloned = original;
        assert_eq!(original, cloned);
    }
}

// Integration tests require viz feature for GpuPixelRenderer
#[cfg(all(test, feature = "viz"))]
mod integration_tests;
