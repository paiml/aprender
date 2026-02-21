//! APR Format Validation Module - 100-Point QA Checklist
//!
//! Implements the Master Falsification QA Checklist from APR-SPEC.md Section 11.
//! Each check is testable and falsifiable.
//!
//! # Categories
//! - A. Format & Structural Integrity (25 Points)
//! - B. Tensor Physics & Statistics (25 Points)
//! - C. Tooling & Operations (25 Points)
//! - D. Conversion & Interoperability (25 Points)

use crate::error::{AprenderError, Result};
use std::collections::HashMap;

/// Validation check result
#[derive(Debug, Clone, PartialEq)]
pub enum CheckStatus {
    /// Check passed
    Pass,
    /// Check failed with reason
    Fail(String),
    /// Check produced a warning
    Warn(String),
    /// Check was skipped (not applicable)
    Skip(String),
}

impl CheckStatus {
    /// Returns true if the check passed
    #[must_use]
    pub fn is_pass(&self) -> bool {
        matches!(self, Self::Pass)
    }

    /// Returns true if the check failed
    #[must_use]
    pub fn is_fail(&self) -> bool {
        matches!(self, Self::Fail(_))
    }
}

/// Individual validation check
#[derive(Debug, Clone)]
pub struct ValidationCheck {
    /// Check ID (1-100)
    pub id: u8,
    /// Check name
    pub name: &'static str,
    /// Category (A, B, C, D)
    pub category: Category,
    /// Check result
    pub status: CheckStatus,
    /// Points awarded (0 or 1)
    pub points: u8,
}

/// Validation category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    /// A. Format & Structural Integrity
    Structure,
    /// B. Tensor Physics & Statistics
    Physics,
    /// C. Tooling & Operations
    Tooling,
    /// D. Conversion & Interoperability
    Conversion,
}

impl Category {
    /// Get category letter
    #[must_use]
    pub fn letter(&self) -> char {
        match self {
            Self::Structure => 'A',
            Self::Physics => 'B',
            Self::Tooling => 'C',
            Self::Conversion => 'D',
        }
    }

    /// Get category name
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Structure => "Format & Structural Integrity",
            Self::Physics => "Tensor Physics & Statistics",
            Self::Tooling => "Tooling & Operations",
            Self::Conversion => "Conversion & Interoperability",
        }
    }
}

/// Complete validation report
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// All checks performed
    pub checks: Vec<ValidationCheck>,
    /// Total score (0-100)
    pub total_score: u8,
    /// Score by category
    pub category_scores: HashMap<Category, u8>,
}

impl ValidationReport {
    /// Create empty report
    #[must_use]
    pub fn new() -> Self {
        Self {
            checks: Vec::with_capacity(100),
            total_score: 0,
            category_scores: HashMap::new(),
        }
    }

    /// Add a check result
    pub fn add_check(&mut self, check: ValidationCheck) {
        let category = check.category;
        let points = check.points;
        self.checks.push(check);

        *self.category_scores.entry(category).or_insert(0) += points;
        self.total_score += points;
    }

    /// Get grade based on score
    #[must_use]
    pub fn grade(&self) -> &'static str {
        match self.total_score {
            95..=100 => "A+",
            90..=94 => "A",
            85..=89 => "B+",
            80..=84 => "B",
            75..=79 => "C+",
            70..=74 => "C",
            60..=69 => "D",
            _ => "F",
        }
    }

    /// Check if validation passed (score >= threshold)
    #[must_use]
    pub fn passed(&self, min_score: u8) -> bool {
        self.total_score >= min_score
    }

    /// Get failed checks
    #[must_use]
    pub fn failed_checks(&self) -> Vec<&ValidationCheck> {
        self.checks.iter().filter(|c| c.status.is_fail()).collect()
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Tensor statistics for physics validation
#[derive(Debug, Clone)]
pub struct TensorStats {
    /// Tensor name
    pub name: String,
    /// Number of elements
    pub count: usize,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// Number of NaN values
    pub nan_count: usize,
    /// Number of Inf values
    pub inf_count: usize,
    /// Number of zero values
    pub zero_count: usize,
}

impl TensorStats {
    /// Compute statistics from tensor data
    #[must_use]
    pub fn compute(name: &str, data: &[f32]) -> Self {
        let count = data.len();
        if count == 0 {
            return Self {
                name: name.to_string(),
                count: 0,
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                std: 0.0,
                nan_count: 0,
                inf_count: 0,
                zero_count: 0,
            };
        }

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut zero_count = 0;

        for &v in data {
            if v.is_nan() {
                nan_count += 1;
                continue;
            }
            if v.is_infinite() {
                inf_count += 1;
                continue;
            }
            if v == 0.0 {
                zero_count += 1;
            }
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += f64::from(v);
        }

        let valid_count = count - nan_count - inf_count;
        let mean = if valid_count > 0 {
            (sum / valid_count as f64) as f32
        } else {
            0.0
        };

        // Compute std dev
        let mut var_sum = 0.0f64;
        for &v in data {
            if !v.is_nan() && !v.is_infinite() {
                let diff = f64::from(v) - f64::from(mean);
                var_sum += diff * diff;
            }
        }
        let std = if valid_count > 1 {
            (var_sum / (valid_count - 1) as f64).sqrt() as f32
        } else {
            0.0
        };

        Self {
            name: name.to_string(),
            count,
            min: if min.is_infinite() { 0.0 } else { min },
            max: if max.is_infinite() { 0.0 } else { max },
            mean,
            std,
            nan_count,
            inf_count,
            zero_count,
        }
    }

    /// Check if tensor has no NaN values
    #[must_use]
    pub fn has_no_nan(&self) -> bool {
        self.nan_count == 0
    }

    /// Check if tensor has no Inf values
    #[must_use]
    pub fn has_no_inf(&self) -> bool {
        self.inf_count == 0
    }

    /// Check if tensor is not all zeros
    #[must_use]
    pub fn is_not_all_zeros(&self) -> bool {
        self.zero_count < self.count
    }

    /// Check if `LayerNorm` weight mean is in valid range [0.5, 3.0]
    #[must_use]
    pub fn is_valid_layernorm_weight(&self) -> bool {
        self.mean >= 0.5 && self.mean <= 3.0
    }

    /// Check if `LayerNorm` bias mean is in valid range [-0.5, 0.5]
    #[must_use]
    pub fn is_valid_layernorm_bias(&self) -> bool {
        self.mean >= -0.5 && self.mean <= 0.5
    }

    /// Check if attention/linear weight mean is approximately 0
    #[must_use]
    pub fn is_valid_linear_weight(&self) -> bool {
        self.mean.abs() < 0.1
    }
}

/// APR file header for validation
#[derive(Debug, Clone)]
pub struct AprHeader {
    /// Magic bytes (should be "APR\0" - ONE format)
    pub magic: [u8; 4],
    /// Version major
    pub version_major: u8,
    /// Version minor
    pub version_minor: u8,
    /// Feature flags
    pub flags: u32,
    /// Metadata offset
    pub metadata_offset: u32,
    /// Metadata size
    pub metadata_size: u32,
    /// Index offset
    pub index_offset: u32,
    /// Index size
    pub index_size: u32,
    /// Data offset
    pub data_offset: u32,
}

impl AprHeader {
    /// Parse header from bytes
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 32 {
            return Err(AprenderError::FormatError {
                message: "Header too small (< 32 bytes)".to_string(),
            });
        }

        let mut magic = [0u8; 4];
        magic.copy_from_slice(bytes.get(0..4).unwrap_or(&[0u8; 4]));

        Ok(Self {
            magic,
            version_major: bytes[4],
            version_minor: bytes[5],
            flags: u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
            metadata_offset: u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]),
            metadata_size: u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]),
            index_offset: u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]),
            index_size: u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            data_offset: u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]),
        })
    }

    /// Check if magic is valid (APR\0 - ONE format)
    #[must_use]
    pub fn is_valid_magic(&self) -> bool {
        self.magic == *b"APR\0"
    }

    /// Check if version is supported
    #[must_use]
    pub fn is_supported_version(&self) -> bool {
        (self.version_major == 1 && self.version_minor <= 2)
            || (self.version_major == 2 && self.version_minor == 0)
    }

    /// Check if compressed flag is set
    #[must_use]
    pub fn is_compressed(&self) -> bool {
        self.flags & 0x01 != 0
    }

    /// Check if signed flag is set
    #[must_use]
    pub fn is_signed(&self) -> bool {
        self.flags & 0x20 != 0
    }

    /// Check if encrypted flag is set
    #[must_use]
    pub fn is_encrypted(&self) -> bool {
        self.flags & 0x10 != 0
    }
}

/// Validator for APR files implementing the 100-point checklist
#[derive(Debug)]
pub struct AprValidator {
    /// Validation report
    report: ValidationReport,
    /// Tensor statistics collected during validation
    tensor_stats: Vec<TensorStats>,
}

include!("validation_impl.rs");
include!("filterbank.rs");
