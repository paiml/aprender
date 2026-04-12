//! Common Bug Pattern Detection (GH-187)
//!
//! Patterns identified from mutation testing and bug fix analysis across:
//! - aprender (6 bug fixes analyzed)
//! - realizar (7 bug fixes analyzed)
//! - organizational-intelligence-plugin (42 mutations)
//! - paiml-mcp-agent-toolkit (mutation testing config)
//!
//! # Bug Categories
//!
//! ## Code Path Bugs (aprender pattern)
//! - Alternate code path missing feature (GH-185: merges in one path, not another)
//! - Algorithm/layout mismatch between implementations (GH-177: Q4K dequant)
//!
//! ## Resource State Bugs (realizar pattern)
//! - Silent fallback to wrong resource (tokenizer from wrong model)
//! - State advancement at wrong layer (KV cache len on layer 0)
//! - GPU context corruption from prior operations
//!
//! ## Validation Gaps (both projects)
//! - Missing validation after transformation (NaN/Inf after dequant)
//! - Missing format/type detection before processing
//! - Missing companion metadata (config.json, tokenizer.json)
//!
//! ## Error Handling (aprender PMAT-189)
//! - Unchecked fallible operations (mutex lock, file I/O)
//! - Missing error propagation on alternate paths

#![allow(clippy::trivially_copy_pass_by_ref)]

use serde::{Deserialize, Serialize};

/// Bug pattern categories derived from cross-project analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BugPattern {
    // === Code Path Bugs ===
    /// Feature implemented in primary path but missing in alternate path
    /// Example: GH-185 - merges embedded in one code path, not raw GGUF path
    AlternatePathMissing,

    /// Two implementations of same algorithm with incompatible layouts
    /// Example: GH-177 - Q4K dequant: one scale vs two scales per block
    AlgorithmMismatch,

    // === Resource State Bugs ===
    /// Fallback mechanism silently uses wrong/incompatible resource
    /// Example: realizar - tokenizer fallback found different model's tokenizer
    SilentFallbackWrongResource,

    /// State advancement happens at wrong point in multi-stage pipeline
    /// Example: realizar - KV cache len auto-advanced on layer 0 instead of last
    StateAdvancementTiming,

    /// Prior operation corrupts shared state for subsequent operations
    /// Example: realizar - GPU context corrupted from earlier tests
    SharedStateCorruption,

    // === Validation Gaps ===
    /// No validation after data transformation allows corrupt values downstream
    /// Example: GH-177 - no NaN/Inf check after dequantization
    MissingPostTransformValidation,

    /// No format/type detection before processing incompatible data
    /// Example: realizar - legacy Q4_0 routed to Q4_K GPU kernel
    MissingTypeDetection,

    /// Primary data saved but required companion/metadata missing
    /// Example: GH-182 - SafeTensors missing config.json, tokenizer.json
    MissingCompanionData,

    // === Error Handling ===
    /// Unchecked fallible operation causes panic instead of error
    /// Example: PMAT-189 - mutex lock poisoning crashes server
    UnwrapOnFallible,

    /// Error not propagated on alternate code path
    /// Example: Error handling differs between primary and fallback paths
    ErrorPropagationGap,

    // === Security ===
    /// Path traversal vulnerability (untrusted path not validated)
    /// Example: realizar - could read /etc/passwd as model
    PathTraversal,

    /// Special tokens not escaped, treated as control codes
    /// Example: realizar - `<|` prompt injection
    PromptInjection,
}

impl BugPattern {
    /// Get the falsification gate ID
    #[must_use]
    pub fn gate_id(&self) -> &'static str {
        match self {
            // Code Path Bugs (F-PATH-*)
            Self::AlternatePathMissing => "F-PATH-ALT-001",
            Self::AlgorithmMismatch => "F-PATH-ALGO-001",

            // Resource State Bugs (F-STATE-*)
            Self::SilentFallbackWrongResource => "F-STATE-FALLBACK-001",
            Self::StateAdvancementTiming => "F-STATE-TIMING-001",
            Self::SharedStateCorruption => "F-STATE-CORRUPT-001",

            // Validation Gaps (F-VALID-*)
            Self::MissingPostTransformValidation => "F-VALID-POST-001",
            Self::MissingTypeDetection => "F-VALID-TYPE-001",
            Self::MissingCompanionData => "F-VALID-COMPANION-001",

            // Error Handling (F-ERR-*)
            Self::UnwrapOnFallible => "F-ERR-UNWRAP-001",
            Self::ErrorPropagationGap => "F-ERR-PROP-001",

            // Security (F-SEC-*)
            Self::PathTraversal => "F-SEC-PATH-001",
            Self::PromptInjection => "F-SEC-INJECT-001",
        }
    }

    /// Get human-readable description
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::AlternatePathMissing => {
                "Feature implemented in primary path but missing in alternate code path"
            }
            Self::AlgorithmMismatch => {
                "Two implementations of same algorithm with incompatible layouts/logic"
            }
            Self::SilentFallbackWrongResource => {
                "Fallback mechanism silently uses wrong or incompatible resource"
            }
            Self::StateAdvancementTiming => {
                "State advancement happens at wrong point in multi-stage pipeline"
            }
            Self::SharedStateCorruption => {
                "Prior operation corrupts shared state for subsequent operations"
            }
            Self::MissingPostTransformValidation => {
                "No validation after transformation allows corrupt values downstream"
            }
            Self::MissingTypeDetection => {
                "No format/type detection before processing incompatible data"
            }
            Self::MissingCompanionData => {
                "Primary data saved but required companion/metadata files missing"
            }
            Self::UnwrapOnFallible => {
                "Unchecked fallible operation causes panic instead of graceful error"
            }
            Self::ErrorPropagationGap => "Error not propagated correctly on alternate code path",
            Self::PathTraversal => "Untrusted path not validated, allows reading arbitrary files",
            Self::PromptInjection => "Special tokens not escaped, treated as control codes",
        }
    }

    /// Get the severity level (P0 = critical, P1 = high, P2 = medium)
    #[must_use]
    #[allow(clippy::match_same_arms)] // Grouping by severity is intentional
    pub fn severity(&self) -> &'static str {
        match self {
            // P0: Causes incorrect output or security vulnerability
            Self::AlternatePathMissing => "P0",
            Self::AlgorithmMismatch => "P0",
            Self::SilentFallbackWrongResource => "P0",
            Self::MissingPostTransformValidation => "P0",
            Self::PathTraversal => "P0",
            Self::PromptInjection => "P0",

            // P1: Causes crashes or data loss
            Self::StateAdvancementTiming => "P1",
            Self::SharedStateCorruption => "P1",
            Self::UnwrapOnFallible => "P1",
            Self::MissingTypeDetection => "P1",

            // P2: Causes compatibility issues
            Self::MissingCompanionData => "P2",
            Self::ErrorPropagationGap => "P2",
        }
    }

    /// Get the source project where this pattern was identified
    #[must_use]
    #[allow(clippy::match_same_arms)] // Same source is intentional - one issue revealed multiple patterns
    pub fn source(&self) -> &'static str {
        match self {
            Self::AlternatePathMissing => "aprender (GH-185)",
            Self::AlgorithmMismatch => "aprender (GH-177)",
            Self::SilentFallbackWrongResource => "realizar (33e18c2)",
            Self::StateAdvancementTiming => "realizar (62147f9)",
            Self::SharedStateCorruption => "realizar (9f9f985)",
            Self::MissingPostTransformValidation => "aprender (GH-177)", // Same issue as AlgorithmMismatch
            Self::MissingTypeDetection => "realizar (f13f39b)",
            Self::MissingCompanionData => "aprender (GH-182)",
            Self::UnwrapOnFallible => "aprender (PMAT-189)",
            Self::ErrorPropagationGap => "aprender/realizar (multiple)",
            Self::PathTraversal => "realizar (04d2774)",
            Self::PromptInjection => "realizar (1b51030)",
        }
    }

    /// All bug patterns
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::AlternatePathMissing,
            Self::AlgorithmMismatch,
            Self::SilentFallbackWrongResource,
            Self::StateAdvancementTiming,
            Self::SharedStateCorruption,
            Self::MissingPostTransformValidation,
            Self::MissingTypeDetection,
            Self::MissingCompanionData,
            Self::UnwrapOnFallible,
            Self::ErrorPropagationGap,
            Self::PathTraversal,
            Self::PromptInjection,
        ]
    }

    /// Get patterns by severity
    #[must_use]
    pub fn by_severity(severity: &str) -> Vec<Self> {
        Self::all()
            .iter()
            .filter(|p| p.severity() == severity)
            .copied()
            .collect()
    }
}

/// Result of numerical stability check (F-NUM-001..004)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalStabilityResult {
    /// Gate ID (F-NUM-001, etc.)
    pub gate_id: String,
    /// Whether the check passed
    pub is_valid: bool,
    /// Measured value
    pub value: f64,
    /// Expected range (min, max)
    pub expected_range: (f64, f64),
    /// Human-readable description
    pub description: String,
}

/// Configuration for DoS protection checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DosProtectionConfig {
    /// Maximum input size in bytes
    pub max_input_bytes: usize,
    /// Maximum estimated token count
    pub max_tokens: usize,
    /// Maximum repetition ratio (0.0-1.0)
    pub max_repetition_ratio: f64,
    /// Maximum expansion ratio
    pub max_expansion_ratio: f64,
}

impl Default for DosProtectionConfig {
    fn default() -> Self {
        Self {
            max_input_bytes: 1_000_000, // 1MB
            max_tokens: 100_000,        // 100K tokens
            max_repetition_ratio: 0.8,  // 80% repetition
            max_expansion_ratio: 100.0, // 100x expansion
        }
    }
}

/// A DoS check violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DosViolation {
    /// Check name
    pub check: String,
    /// Description of violation
    pub description: String,
    /// Severity (P0, P1, P2)
    pub severity: String,
}

/// Result of DoS protection check (F-SEC-003)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DosCheckResult {
    /// Gate ID
    pub gate_id: String,
    /// Whether input is safe
    pub is_safe: bool,
    /// Violations found
    pub violations: Vec<DosViolation>,
    /// Input size in bytes
    pub input_bytes: usize,
    /// Estimated token count
    pub estimated_tokens: usize,
    /// Repetition ratio
    pub repetition_ratio: f64,
    /// Expansion ratio
    pub expansion_ratio: f64,
}

/// Detection heuristics for each pattern
pub struct PatternDetector {
    /// Patterns to check (used for filtering which checks to run)
    #[allow(dead_code)]
    patterns: Vec<BugPattern>,
}

impl Default for PatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

include!("patterns_detectors.rs");
include!("patterns_spec_gates.rs");
include!("patterns_performance_parity.rs");
