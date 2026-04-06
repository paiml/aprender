#![allow(missing_docs)]
//! Core types for the ComputeBrick system.
//!
//! Defines execution backends, error types, assertions, and verification results.

use crate::error::TruenoError;
use std::fmt;

/// Execution backend for compute operations.
/// This is the brick-specific backend enum with additional GPU backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ComputeBackend {
    /// Pure Rust scalar fallback (always available, baseline for correctness)
    Scalar,
    /// SSE2 SIMD (x86_64 baseline)
    Sse2,
    /// AVX2 256-bit SIMD with FMA
    #[default]
    Avx2,
    /// AVX-512 512-bit SIMD
    Avx512,
    /// ARM NEON SIMD
    Neon,
    /// WebAssembly SIMD128
    Wasm,
    /// NVIDIA CUDA via PTX
    Cuda,
    /// Cross-platform GPU via wgpu
    Wgpu,
    /// Auto-select best available backend
    Auto,
}

impl fmt::Display for ComputeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeBackend::Scalar => write!(f, "Scalar"),
            ComputeBackend::Sse2 => write!(f, "SSE2"),
            ComputeBackend::Avx2 => write!(f, "AVX2"),
            ComputeBackend::Avx512 => write!(f, "AVX-512"),
            ComputeBackend::Neon => write!(f, "NEON"),
            ComputeBackend::Wasm => write!(f, "WASM"),
            ComputeBackend::Cuda => write!(f, "CUDA"),
            ComputeBackend::Wgpu => write!(f, "wgpu"),
            ComputeBackend::Auto => write!(f, "Auto"),
        }
    }
}

/// Type alias for backward compatibility
pub type Backend = ComputeBackend;

/// Errors from ComputeBrick execution.
/// Tells you exactly what failed (Jidoka: stop and signal).
#[derive(Debug, thiserror::Error)]
pub enum BrickError {
    /// Assertion failed during verification
    #[error("Assertion failed: {name} - expected {expected}, got {actual}")]
    AssertionFailed { name: String, expected: String, actual: String },

    /// Performance budget exceeded
    #[error("Budget exceeded: {limit_us:.1}µs/tok limit, {actual_us:.1}µs/tok actual ({utilization:.0}% of budget)")]
    BudgetExceeded { limit_us: f64, actual_us: f64, utilization: f64 },

    /// Underlying compute error
    #[error("Compute error: {0}")]
    ComputeError(#[from] TruenoError),

    /// No assertions defined (violates Popperian falsifiability)
    #[error("Brick has no assertions - violates Popperian falsifiability requirement")]
    NoAssertions,

    /// Backend not available
    #[error("Backend {0} not available on this system")]
    BackendUnavailable(Backend),
}

/// Type of assertion for compute verification.
#[derive(Debug, Clone)]
pub enum ComputeAssertion {
    /// Output must match baseline backend within tolerance
    Equivalence { baseline: Backend, tolerance: f64 },
    /// Output values must be within bounds
    Bounds { min: f64, max: f64 },
    /// Output must not contain NaN or infinity
    Finite,
    /// Custom assertion with name and check function index
    Custom { name: String },
}

impl ComputeAssertion {
    /// Create equivalence assertion with default tolerance (1e-5).
    pub fn equiv(baseline: Backend) -> Self {
        Self::Equivalence { baseline, tolerance: 1e-5 }
    }

    /// Create equivalence assertion with custom tolerance.
    pub fn equiv_with_tolerance(baseline: Backend, tolerance: f64) -> Self {
        Self::Equivalence { baseline, tolerance }
    }

    /// Create bounds assertion.
    pub fn bounds(min: f64, max: f64) -> Self {
        contract_pre_bounds!();
        Self::Bounds { min, max }
    }

    /// Create finite assertion (no NaN/Inf).
    pub fn finite() -> Self {
        Self::Finite
    }
}

/// Verification result from ComputeBrick.
#[derive(Debug, Clone)]
pub struct BrickVerification {
    /// Overall pass/fail
    pub passed: bool,
    /// Individual assertion results
    pub assertion_results: Vec<AssertionResult>,
    /// Verification time in microseconds
    pub verification_us: f64,
}

impl BrickVerification {
    /// Check if all assertions passed.
    pub fn is_valid(&self) -> bool {
        self.passed
    }

    /// Get failed assertions.
    pub fn failures(&self) -> impl Iterator<Item = &AssertionResult> {
        self.assertion_results.iter().filter(|r| !r.passed)
    }
}

/// Result of a single assertion check.
#[derive(Debug, Clone)]
pub struct AssertionResult {
    /// Assertion that was checked
    pub assertion: ComputeAssertion,
    /// Did it pass?
    pub passed: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Trait for compute operations that can be wrapped in a ComputeBrick.
pub trait ComputeOp: Send + Sync {
    /// Input type for this operation
    type Input;
    /// Output type for this operation
    type Output;

    /// Operation name for identification
    fn name(&self) -> &'static str;

    /// Execute the operation on the given backend
    fn execute(&self, input: Self::Input, backend: Backend) -> Result<Self::Output, TruenoError>;

    /// Number of tokens this operation processes (for budget calculation)
    fn tokens(&self, input: &Self::Input) -> usize;

    /// Clone the input for verification (if needed)
    fn clone_input(&self, input: &Self::Input) -> Option<Self::Input>
    where
        Self::Input: Clone,
    {
        Some(input.clone())
    }
}
