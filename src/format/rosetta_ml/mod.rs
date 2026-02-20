// Allow non-camel case for quantization format names (Q4_0, Q4_K, etc.)
// These match industry-standard naming conventions from GGUF/llama.cpp
#![allow(non_camel_case_types)]

//! ML-Powered Rosetta Diagnostics (ROSETTA-ML-001)
//!
//! "Grepping is the stone age. ML enables *automatic* root cause analysis."
//!
//! This module implements ML-based diagnostics for format conversion using
//! aprender's own algorithms (dogfooding). No external dependencies.
//!
//! # Theoretical Foundation
//!
//! - **Tarantula SBFL**: Jones et al. (2002) - fault localization via suspiciousness
//! - **Mahalanobis Distance**: Mahalanobis (1936) - multivariate anomaly detection
//! - **Wilson Score**: Wilson (1927) - confidence intervals for binomial proportions
//! - **BM25 + RRF**: Robertson (1994), Cormack (2009) - hybrid retrieval
//!
//! # Dogfooding Matrix
//!
//! | Task | Algorithm | Module |
//! |------|-----------|--------|
//! | Error prediction | `LinearRegression` | `aprender::linear_model` |
//! | Failure clustering | `KMeans` | `aprender::cluster` |
//! | Feature reduction | `PCA` | `aprender::preprocessing` |
//! | Error classification | `GaussianNB` | `aprender::classification` |
//!
//! # References
//!
//! 1. Jones, J. A., et al. (2002). Visualization of test information. ICSE '02.
//! 2. Mahalanobis, P. C. (1936). On the generalised distance in statistics.
//! 3. Wilson, E. B. (1927). Probable inference. JASA 22(158).
//! 4. Robertson, S. E., et al. (1994). Okapi at TREC-3.

use std::collections::HashMap;
use std::fmt;

// ============================================================================
// Conversion Decision Types (Granular per Dr. Popper's advice)
// ============================================================================

/// Granular conversion decisions tracked for Tarantula analysis.
///
/// Per Dr. Popper's audit: "QuantQ4_K might be too broad; consider
/// QuantQ4_K_BlockSize32". Each decision includes specific parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConversionDecision {
    // Quantization decisions (with block size granularity)
    /// Q4_0 quantization with 32-element blocks
    QuantQ4_0_Block32,
    /// Q4_K quantization with 32-element super-blocks
    QuantQ4_K_Block32,
    /// Q4_K quantization with 256-element super-blocks
    QuantQ4_K_Block256,
    /// Q6_K quantization with 256-element super-blocks
    QuantQ6_K_Block256,
    /// Q8_0 quantization with 32-element blocks
    QuantQ8_0_Block32,
    /// F16 half-precision (no quantization)
    DtypeF16,
    /// F32 single-precision (no quantization)
    DtypeF32,
    /// BF16 bfloat16 precision
    DtypeBF16,

    // Layout decisions
    /// Row-major storage (C-style)
    LayoutRowMajor,
    /// Column-major storage (Fortran-style)
    LayoutColMajor,
    /// Dimension transpose applied
    TransposeDims,
    /// No transpose (identity)
    TransposeNone,

    // Header/metadata decisions
    /// Vocabulary merge operation
    VocabMerge,
    /// Header rewrite operation
    HeaderRewrite,
    /// Metadata preserve (no changes)
    MetadataPreserve,

    // Format-specific decisions
    /// GGUF magic header write
    GgufMagicWrite,
    /// SafeTensors JSON header write
    SafeTensorsJsonWrite,
    /// APR v2 header write
    AprV2HeaderWrite,

    // Tensor operations
    /// Zero-padding applied
    TensorZeroPad,
    /// Truncation applied
    TensorTruncate,
    /// Reshape operation
    TensorReshape,
}

impl fmt::Display for ConversionDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

// ============================================================================
// Tarantula Fault Localization (Jones et al., 2002)
// ============================================================================

/// Decision statistics for Tarantula suspiciousness calculation
#[derive(Debug, Clone, Default)]
pub struct DecisionStats {
    /// Number of times decision was used in successful conversions
    pub passed: usize,
    /// Number of times decision was used in failed conversions
    pub failed: usize,
}

impl DecisionStats {
    /// Calculate pass rate for this decision
    #[must_use]
    pub fn pass_rate(&self) -> f32 {
        let total = self.passed + self.failed;
        if total == 0 {
            0.0
        } else {
            self.passed as f32 / total as f32
        }
    }
}

/// Tarantula fault localization tracker
///
/// Implements the Tarantula algorithm from Jones et al. (2002):
/// ```text
/// suspiciousness(d) = (failed(d) / total_failed) /
///                     ((failed(d) / total_failed) + (passed(d) / total_passed))
/// ```
///
/// # Falsification Criterion (F-SBFL-001)
///
/// "If Tarantula ranks an innocent decision as most suspicious in >10%
/// of fault localization sessions, the algorithm is falsified."
#[derive(Debug, Clone, Default)]
pub struct TarantulaTracker {
    /// Per-decision statistics
    stats: HashMap<ConversionDecision, DecisionStats>,
    /// Total successful conversions
    total_passed: usize,
    /// Total failed conversions
    total_failed: usize,
}

impl TarantulaTracker {
    /// Create a new Tarantula tracker
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a successful conversion with the decisions used
    pub fn record_pass(&mut self, decisions: &[ConversionDecision]) {
        self.total_passed += 1;
        for &decision in decisions {
            self.stats.entry(decision).or_default().passed += 1;
        }
    }

    /// Record a failed conversion with the decisions used
    pub fn record_fail(&mut self, decisions: &[ConversionDecision]) {
        self.total_failed += 1;
        for &decision in decisions {
            self.stats.entry(decision).or_default().failed += 1;
        }
    }

    /// Calculate Tarantula suspiciousness score for a decision
    ///
    /// Returns value in [0, 1] where 1 = most suspicious
    #[must_use]
    pub fn suspiciousness(&self, decision: ConversionDecision) -> f32 {
        if self.total_failed == 0 || self.total_passed == 0 {
            return 0.0;
        }

        let stats = self
            .stats
            .get(&decision)
            .map_or(DecisionStats::default(), std::clone::Clone::clone);

        let failed_ratio = stats.failed as f32 / self.total_failed as f32;
        let passed_ratio = stats.passed as f32 / self.total_passed as f32;

        let denominator = failed_ratio + passed_ratio;
        if denominator == 0.0 {
            0.0
        } else {
            failed_ratio / denominator
        }
    }

    /// Get all decisions ranked by suspiciousness (most suspicious first)
    #[must_use]
    pub fn ranked_suspiciousness(&self) -> Vec<(ConversionDecision, f32)> {
        let mut ranked: Vec<_> = self
            .stats
            .keys()
            .map(|&d| (d, self.suspiciousness(d)))
            .collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }

    /// Get priority level based on failure rate
    ///
    /// - Critical: failure_rate > 20%
    /// - High: failure_rate > 10%
    /// - Medium: failure_rate > 5%
    /// - Low: otherwise
    #[must_use]
    pub fn priority(&self, decision: ConversionDecision) -> Priority {
        let stats = self.stats.get(&decision);
        if stats.is_none() {
            return Priority::Low;
        }

        let stats = stats.expect("checked above");
        let total = stats.passed + stats.failed;
        if total == 0 {
            return Priority::Low;
        }

        let failure_rate = stats.failed as f32 / total as f32;
        if failure_rate > 0.20 {
            Priority::Critical
        } else if failure_rate > 0.10 {
            Priority::High
        } else if failure_rate > 0.05 {
            Priority::Medium
        } else {
            Priority::Low
        }
    }
}

/// Priority level for conversion decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Low priority (<5% failure rate)
    Low,
    /// Medium priority (5-10% failure rate)
    Medium,
    /// High priority (10-20% failure rate)
    High,
    /// Critical priority (>20% failure rate)
    Critical,
}

// ============================================================================
// Tensor Feature Extraction (12-dimensional)
// ============================================================================

/// 12-dimensional feature vector for tensor anomaly detection.
///
/// Per Dr. Popper's audit on Curse of Dimensionality: "Ensure you have
/// sufficient data (n >> p) before enabling Mahalanobis distance checks,
/// or use a regularized covariance estimator."
#[derive(Debug, Clone, Default)]
pub struct TensorFeatures {
    /// Mean: E[x]
    pub mean: f32,
    /// Standard deviation: sqrt(Var[x])
    pub std: f32,
    /// Minimum value: min(x)
    pub min: f32,
    /// Maximum value: max(x)
    pub max: f32,
    /// Kurtosis: E[(x-μ)⁴]/σ⁴ - 3 (excess kurtosis)
    pub kurtosis: f32,
    /// Skewness: E[(x-μ)³]/σ³
    pub skewness: f32,
    /// Sparsity: |{x: x=0}| / n
    pub sparsity: f32,
    /// L1 norm: Σ|x|
    pub l1_norm: f32,
    /// L2 norm: sqrt(Σx²)
    pub l2_norm: f32,
    /// Infinity norm: max(|x|)
    pub inf_norm: f32,
    /// NaN count: |{x: isnan(x)}|
    pub nan_count: f32,
    /// Inf count: |{x: isinf(x)}|
    pub inf_count: f32,
}

impl TensorFeatures {
    /// Extract features from tensor data
    #[must_use]
    pub fn from_data(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self::default();
        }

        let n = data.len() as f32;

        // Basic statistics
        let mut sum = 0.0f64;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut zeros = 0usize;
        let mut nans = 0usize;
        let mut infs = 0usize;
        let mut l1 = 0.0f64;
        let mut l2 = 0.0f64;
        let mut inf_norm = 0.0f32;

        for &x in data {
            if x.is_nan() {
                nans += 1;
                continue;
            }
            if x.is_infinite() {
                infs += 1;
                continue;
            }

            sum += f64::from(x);
            if x < min {
                min = x;
            }
            if x > max {
                max = x;
            }
            if x == 0.0 {
                zeros += 1;
            }

            let abs_x = x.abs();
            l1 += f64::from(abs_x);
            l2 += f64::from(abs_x * abs_x);
            if abs_x > inf_norm {
                inf_norm = abs_x;
            }
        }

        let valid_n = n - nans as f32 - infs as f32;
        let mean = if valid_n > 0.0 {
            (sum / f64::from(valid_n)) as f32
        } else {
            0.0
        };

        // Second pass: variance, skewness, kurtosis
        let mut m2 = 0.0f64; // Σ(x - μ)²
        let mut m3 = 0.0f64; // Σ(x - μ)³
        let mut m4 = 0.0f64; // Σ(x - μ)⁴

        for &x in data {
            if x.is_nan() || x.is_infinite() {
                continue;
            }
            let diff = f64::from(x - mean);
            m2 += diff * diff;
            m3 += diff * diff * diff;
            m4 += diff * diff * diff * diff;
        }

        let variance = if valid_n > 1.0 {
            (m2 / f64::from(valid_n - 1.0)) as f32
        } else {
            0.0
        };
        let std = variance.sqrt();

        let skewness = if std > 0.0 && valid_n > 0.0 {
            let n64 = f64::from(valid_n);
            ((m3 / n64) / f64::from(std * std * std)) as f32
        } else {
            0.0
        };

        let kurtosis = if std > 0.0 && valid_n > 0.0 {
            let n64 = f64::from(valid_n);
            let var64 = f64::from(variance);
            ((m4 / n64) / (var64 * var64)) as f32 - 3.0 // Excess kurtosis
        } else {
            0.0
        };

        Self {
            mean,
            std,
            min: if min.is_finite() { min } else { 0.0 },
            max: if max.is_finite() { max } else { 0.0 },
            kurtosis,
            skewness,
            sparsity: zeros as f32 / n,
            l1_norm: l1 as f32,
            l2_norm: (l2 as f32).sqrt(),
            inf_norm,
            nan_count: nans as f32,
            inf_count: infs as f32,
        }
    }

    /// Convert to feature vector for ML algorithms
    #[must_use]
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.mean,
            self.std,
            self.min,
            self.max,
            self.kurtosis,
            self.skewness,
            self.sparsity,
            self.l1_norm,
            self.l2_norm,
            self.inf_norm,
            self.nan_count,
            self.inf_count,
        ]
    }

    /// Check for immediate Jidoka stop conditions
    #[must_use]
    pub fn has_jidoka_violation(&self) -> Option<JidokaViolation> {
        if self.nan_count > 0.0 {
            return Some(JidokaViolation::NaN {
                count: self.nan_count as usize,
            });
        }
        if self.inf_count > 0.0 {
            return Some(JidokaViolation::Inf {
                count: self.inf_count as usize,
            });
        }
        if self.std == 0.0 && self.mean != 0.0 {
            return Some(JidokaViolation::ZeroVariance { mean: self.mean });
        }
        None
    }
}

include!("mod_part_02.rs");
include!("mod_part_03.rs");
include!("canary.rs");
