//! Analysis modules for trace quality and anti-pattern detection
//!
//! This module provides architectural quality analysis tools (§27 of the specification).

pub mod anti_pattern;

pub use anti_pattern::{
    AntiPattern, AntiPatternDetector, AntiPatternThresholds, ArchitecturalQuality,
};
