//! HuggingFace Parity Oracle
//!
//! Cross-implementation validation oracle that compares Sovereign Stack outputs
//! against HuggingFace transformers golden outputs. Implements Popperian severe
//! testing methodology with Toyota Jidoka (stop-on-defect) principles.
//!
//! # Design Philosophy
//!
//! > "The wrong view of science betrays itself in the craving to be right."
//! > — Karl Popper, *The Logic of Scientific Discovery* (1959)
//!
//! This oracle attempts to **falsify** the hypothesis that our implementation
//! produces equivalent outputs to HuggingFace. A falsification indicates a bug
//! that must be investigated before certification can proceed.
//!
//! # References
//!
//! - Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge.
//! - Ohno, T. (1988). *Toyota Production System*. Productivity Press.
//! - Goldberg, D. (1991). "What Every Computer Scientist Should Know About
//!   Floating-Point Arithmetic." ACM Computing Surveys, 23(1), 5-48.

use crate::oracle::{Oracle, OracleResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Tolerance configuration for numerical comparison.
///
/// Following IEEE 754 analysis (Goldberg, 1991) and ML reproducibility
/// guidelines (Pineau et al., 2021), tolerances are precision-specific.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Tolerance {
    /// Absolute tolerance for FP32 comparison (default: 1e-5)
    pub atol_fp32: f32,
    /// Relative tolerance for FP32 comparison (default: 1e-4)
    pub rtol_fp32: f32,
    /// Absolute tolerance for quantized comparison (default: 1e-2)
    pub atol_quant: f32,
    /// Maximum allowed mismatch ratio before falsification (default: 0.01 = 1%)
    pub max_mismatch_ratio: f32,
}

impl Default for Tolerance {
    fn default() -> Self {
        Self {
            atol_fp32: 1e-5,
            rtol_fp32: 1e-4,
            atol_quant: 1e-2,
            max_mismatch_ratio: 0.01,
        }
    }
}

impl Tolerance {
    /// Create tolerance for FP32 precision
    #[must_use]
    pub const fn fp32() -> Self {
        Self {
            atol_fp32: 1e-5,
            rtol_fp32: 1e-4,
            atol_quant: 1e-2,
            max_mismatch_ratio: 0.01,
        }
    }

    /// Create tolerance for FP16 precision
    #[must_use]
    pub const fn fp16() -> Self {
        Self {
            atol_fp32: 1e-3,
            rtol_fp32: 1e-2,
            atol_quant: 1e-1,
            max_mismatch_ratio: 0.01,
        }
    }

    /// Create tolerance for INT8 quantized models
    #[must_use]
    pub const fn int8() -> Self {
        Self {
            atol_fp32: 1e-1,
            rtol_fp32: 1e-1,
            atol_quant: 1e-1,
            max_mismatch_ratio: 0.05,
        }
    }

    /// Create tolerance for INT4 quantized models
    #[must_use]
    pub const fn int4() -> Self {
        Self {
            atol_fp32: 5e-1,
            rtol_fp32: 2e-1,
            atol_quant: 5e-1,
            max_mismatch_ratio: 0.10,
        }
    }

    /// Check if two values are within tolerance using allclose criterion.
    ///
    /// Implements: |a - b| <= atol + rtol * |b|
    ///
    /// This is the NumPy allclose criterion, which accounts for both
    /// absolute and relative error bounds.
    #[must_use]
    pub fn is_close(&self, actual: f32, expected: f32) -> bool {
        let diff = (actual - expected).abs();
        let bound = self.rtol_fp32.mul_add(expected.abs(), self.atol_fp32);
        diff <= bound
    }
}

/// Tensor comparison result when values diverge.
///
/// Implements Toyota's Andon principle: detailed diagnostic information
/// to enable rapid root cause analysis.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TensorDiff {
    /// Tensor shapes do not match
    ShapeMismatch {
        /// Expected number of elements
        expected: usize,
        /// Actual number of elements
        actual: usize,
    },
    /// Tensor values exceed tolerance
    ValueMismatch {
        /// Number of elements exceeding tolerance
        num_mismatches: usize,
        /// Total number of elements compared
        total: usize,
        /// Ratio of mismatches (num_mismatches / total)
        mismatch_ratio: f32,
        /// Maximum absolute difference observed
        max_diff: f32,
        /// Index of maximum difference
        max_diff_idx: usize,
        /// Expected value at max diff location
        expected_val: f32,
        /// Actual value at max diff location
        actual_val: f32,
        /// Mean absolute difference
        mean_diff: f32,
    },
    /// File could not be read or parsed
    ParseError {
        /// Error message
        message: String,
    },
}

impl std::fmt::Display for TensorDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "Shape mismatch: expected {expected}, got {actual}")
            }
            Self::ValueMismatch {
                num_mismatches,
                total,
                mismatch_ratio,
                max_diff,
                max_diff_idx,
                expected_val,
                actual_val,
                mean_diff,
            } => {
                write!(
                    f,
                    "Value mismatch: {num_mismatches}/{total} elements ({:.2}%) exceed tolerance. \
                     Max diff: {max_diff:.6} at idx {max_diff_idx} (expected: {expected_val:.6}, \
                     actual: {actual_val:.6}). Mean diff: {mean_diff:.6}",
                    mismatch_ratio * 100.0
                )
            }
            Self::ParseError { message } => write!(f, "Parse error: {message}"),
        }
    }
}

/// Pre-computed golden output from HuggingFace transformers.
///
/// Stored as SafeTensors with metadata for reproducibility tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenOutput {
    /// Hash of input prompt (for lookup)
    pub input_hash: String,
    /// Original prompt text
    pub prompt: String,
    /// Expected logits as raw F32 values
    pub logits: Vec<f32>,
    /// Shape of logits tensor [batch, seq, vocab]
    pub shape: Vec<usize>,
    /// Expected generated text (optional, for text comparison)
    pub text: Option<String>,
    /// HuggingFace model ID used to generate golden
    pub model_id: String,
    /// transformers library version
    pub transformers_version: String,
}

/// HuggingFace Parity Oracle
///
/// Compares model outputs against pre-computed golden outputs from
/// HuggingFace transformers. Implements Popperian falsification:
/// any divergence beyond tolerance falsifies the parity hypothesis.
#[derive(Debug, Clone)]
pub struct HfParityOracle {
    /// Path to ground truth corpus directory
    corpus_path: PathBuf,
    /// Model family (e.g., "llama", "qwen", "whisper")
    model_family: String,
    /// Numerical tolerance configuration
    tolerance: Tolerance,
    /// Cache of loaded golden outputs (keyed by input hash)
    golden_cache: HashMap<String, GoldenOutput>,
}

include!("hf_parity_oracle_impl.rs");
