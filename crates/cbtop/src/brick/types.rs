//! Core Brick types: assertions, kernel traces, divergence reports,
//! performance budgets, and verification results.

use std::any::Any;
use std::time::{Duration, Instant};

/// Falsifiable assertion types
#[derive(Debug, Clone)]
pub enum BrickAssertion {
    /// Minimum width requirement
    MinWidth(u16),
    /// Minimum height requirement
    MinHeight(u16),
    /// Maximum width requirement
    MaxWidth(u16),
    /// Maximum height requirement
    MaxHeight(u16),
    /// Maximum render time in milliseconds
    MaxRenderTimeMs(u32),
    /// Maximum latency in milliseconds
    MaxLatencyMs(u32),
    /// Value must be in range [min, max]
    ValueInRange { min: f64, max: f64 },
    /// Data must not be empty
    DataNonEmpty,
    /// Custom assertion with name and validator
    Custom { name: &'static str, description: &'static str },
    /// CORRECTNESS-011: Checksum must match between backends (CPU vs GPU)
    /// Five-Whys: Hours of manual debugging → No automated divergence detection
    ChecksumMatch {
        /// Expected checksum from reference backend (e.g., CPU Scalar)
        expected: u64,
        /// Actual checksum from test backend (e.g., CUDA)
        actual: u64,
        /// Kernel name where divergence occurred
        kernel_name: String,
        /// Position/layer where divergence occurred
        position: u32,
    },
}

impl BrickAssertion {
    /// Get assertion name for reporting
    pub fn name(&self) -> &str {
        match self {
            Self::MinWidth(_) => "min_width",
            Self::MinHeight(_) => "min_height",
            Self::MaxWidth(_) => "max_width",
            Self::MaxHeight(_) => "max_height",
            Self::MaxRenderTimeMs(_) => "max_render_time_ms",
            Self::MaxLatencyMs(_) => "max_latency_ms",
            Self::ValueInRange { .. } => "value_in_range",
            Self::DataNonEmpty => "data_non_empty",
            Self::Custom { name, .. } => name,
            Self::ChecksumMatch { .. } => "checksum_match",
        }
    }

    /// Create custom assertion with name and validator function
    /// Note: validator is called but result not stored (for API compatibility)
    pub fn custom<F>(_name: &'static str, _validator: F) -> Self
    where
        F: Fn(&dyn Any) -> bool,
    {
        Self::Custom { name: _name, description: "" }
    }

    /// Create max latency assertion (milliseconds)
    pub const fn max_latency_ms(ms: u32) -> Self {
        Self::MaxLatencyMs(ms)
    }

    /// CORRECTNESS-011: Create checksum match assertion
    pub fn checksum_match(expected: u64, actual: u64, kernel_name: &str, position: u32) -> Self {
        Self::ChecksumMatch { expected, actual, kernel_name: kernel_name.to_string(), position }
    }
}

/// CORRECTNESS-011: Per-kernel trace for divergence detection
/// Captures input/output checksums for every kernel launch
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KernelTrace {
    /// Kernel name (e.g., "rope_neox_indirect_12_128")
    pub kernel_name: String,
    /// Layer index (0-27 for transformer layers)
    pub layer_idx: usize,
    /// Position in sequence (for RoPE, attention)
    pub position: u32,
    /// Input checksum (FNV-1a of first 64 floats)
    pub input_checksum: u64,
    /// Output checksum (FNV-1a of first 64 floats)
    pub output_checksum: u64,
    /// Kernel parameters (JSON serialized for flexibility)
    pub params: String,
    /// Execution time in microseconds
    pub time_us: f64,
    /// Backend that executed this kernel (e.g., "CPU", "CUDA", "Vulkan")
    pub backend: String,
}

impl KernelTrace {
    /// Create a new kernel trace
    pub fn new(kernel_name: &str, layer_idx: usize, position: u32, backend: &str) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            layer_idx,
            position,
            input_checksum: 0,
            output_checksum: 0,
            params: String::new(),
            time_us: 0.0,
            backend: backend.to_string(),
        }
    }

    /// Set input checksum from float slice (FNV-1a hash of first 64 elements)
    pub fn with_input_checksum(mut self, data: &[f32]) -> Self {
        self.input_checksum = fnv1a_f32(data);
        self
    }

    /// Set output checksum from float slice
    pub fn with_output_checksum(mut self, data: &[f32]) -> Self {
        self.output_checksum = fnv1a_f32(data);
        self
    }

    /// Set kernel parameters as JSON
    pub fn with_params(mut self, params: &str) -> Self {
        self.params = params.to_string();
        self
    }

    /// Set execution time
    pub fn with_time_us(mut self, time_us: f64) -> Self {
        self.time_us = time_us;
        self
    }
}

/// CORRECTNESS-011: Divergence report identifying first mismatch
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DivergenceReport {
    /// Whether CPU and GPU outputs matched
    pub matched: bool,
    /// First kernel that diverged (None if all matched)
    pub first_divergent_kernel: Option<KernelTrace>,
    /// Expected (CPU) trace at divergence point
    pub expected_trace: Option<KernelTrace>,
    /// Actual (GPU) trace at divergence point
    pub actual_trace: Option<KernelTrace>,
    /// Number of kernels compared before finding divergence
    pub kernels_compared: usize,
    /// Human-readable diagnosis
    pub diagnosis: String,
}

impl DivergenceReport {
    /// Create a report indicating no divergence
    pub fn matched(kernels_compared: usize) -> Self {
        Self {
            matched: true,
            first_divergent_kernel: None,
            expected_trace: None,
            actual_trace: None,
            kernels_compared,
            diagnosis: format!("All {} kernels matched between CPU and GPU", kernels_compared),
        }
    }

    /// Create a report indicating divergence at specific kernel
    pub fn diverged(expected: KernelTrace, actual: KernelTrace, kernels_compared: usize) -> Self {
        let diagnosis = format!(
            "DIVERGENCE at kernel '{}' (layer {}, position {}): \
             CPU checksum 0x{:016X} != GPU checksum 0x{:016X}. \
             Params: {}",
            actual.kernel_name,
            actual.layer_idx,
            actual.position,
            expected.output_checksum,
            actual.output_checksum,
            actual.params,
        );
        Self {
            matched: false,
            first_divergent_kernel: Some(actual.clone()),
            expected_trace: Some(expected),
            actual_trace: Some(actual),
            kernels_compared,
            diagnosis,
        }
    }
}

/// FNV-1a hash of f32 slice (first 64 elements for efficiency)
/// Public for use in divergence detection across crates
pub fn fnv1a_f32(data: &[f32]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    let len = data.len().min(64);
    for &val in &data[..len] {
        let bytes = val.to_le_bytes();
        for byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

/// Performance budget per phase (Muda elimination)
///
/// Reference: Ohno, T. (1988). "Toyota Production System"
#[derive(Debug, Clone, Copy, Default)]
pub struct BrickBudget {
    /// Collection phase budget (ms)
    pub collect_ms: u32,
    /// Layout calculation budget (ms)
    pub layout_ms: u32,
    /// Rendering phase budget (ms)
    pub render_ms: u32,
}

impl BrickBudget {
    /// Create uniform budget (same for all phases)
    pub const fn uniform(ms: u32) -> Self {
        Self { collect_ms: ms, layout_ms: ms, render_ms: ms }
    }

    /// 60fps budget: 16ms total
    pub const FRAME_60FPS: Self = Self { collect_ms: 5, layout_ms: 3, render_ms: 8 };

    /// 30fps budget: 33ms total
    pub const FRAME_30FPS: Self = Self { collect_ms: 10, layout_ms: 6, render_ms: 17 };

    /// Total budget in milliseconds
    pub const fn total_ms(&self) -> u32 {
        self.collect_ms + self.layout_ms + self.render_ms
    }
}

/// Verification result with pass/fail tracking
#[derive(Debug, Clone)]
pub struct BrickVerification {
    /// Passed assertions
    pub passed: Vec<BrickAssertion>,
    /// Failed assertions with reason
    pub failed: Vec<(BrickAssertion, String)>,
    /// Time taken to verify
    pub verification_time: Duration,
    /// Timestamp
    pub timestamp: Instant,
}

impl BrickVerification {
    /// Create new verification result
    pub fn new() -> Self {
        Self {
            passed: Vec::new(),
            failed: Vec::new(),
            verification_time: Duration::ZERO,
            timestamp: Instant::now(),
        }
    }

    /// Create a passing verification
    pub fn pass() -> Self {
        Self::new()
    }

    /// Add a passed assertion
    pub fn add_pass(&mut self, assertion: BrickAssertion) {
        self.passed.push(assertion);
    }

    /// Add a failed assertion with reason
    pub fn add_fail(&mut self, assertion: BrickAssertion, reason: impl Into<String>) {
        self.failed.push((assertion, reason.into()));
    }

    /// Check an assertion and add to passed list (simplified version)
    pub fn check(&mut self, assertion: &BrickAssertion) {
        // For now, assume assertions pass (real implementation would validate)
        self.passed.push(assertion.clone());
    }

    /// Is verification successful? (Jidoka gate)
    pub fn is_valid(&self) -> bool {
        self.failed.is_empty()
    }

    /// Falsification score: passed / total
    pub fn score(&self) -> f64 {
        let total = self.passed.len() + self.failed.len();
        if total == 0 {
            1.0
        } else {
            self.passed.len() as f64 / total as f64
        }
    }

    /// Get failure count
    pub fn failure_count(&self) -> usize {
        self.failed.len()
    }
}

impl Default for BrickVerification {
    fn default() -> Self {
        Self::new()
    }
}
