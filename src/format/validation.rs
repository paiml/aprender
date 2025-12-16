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

    /// Check if LayerNorm weight mean is in valid range [0.5, 3.0]
    #[must_use]
    pub fn is_valid_layernorm_weight(&self) -> bool {
        self.mean >= 0.5 && self.mean <= 3.0
    }

    /// Check if LayerNorm bias mean is in valid range [-0.5, 0.5]
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
    /// Magic bytes (should be "APRN" or "APR2")
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
        magic.copy_from_slice(&bytes[0..4]);

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

    /// Check if magic is valid (APRN or APR2)
    #[must_use]
    pub fn is_valid_magic(&self) -> bool {
        self.magic == *b"APRN" || self.magic == *b"APR2"
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
}

impl AprValidator {
    /// Create new validator
    #[must_use]
    pub fn new() -> Self {
        Self {
            report: ValidationReport::new(),
        }
    }

    /// Run all validation checks
    pub fn validate(&mut self, data: &[u8]) -> &ValidationReport {
        self.validate_structure(data);
        &self.report
    }

    /// Run Section A: Format & Structural Integrity checks (1-25)
    fn validate_structure(&mut self, data: &[u8]) {
        // Check 1: Magic bytes valid
        self.check_magic(data);

        // Check 2: Header size fixed (32 bytes)
        self.check_header_size(data);

        // Check 3: Version supported
        if data.len() >= 32 {
            if let Ok(header) = AprHeader::parse(data) {
                self.check_version(&header);
                self.check_flags(&header);
            }
        }

        // Check 4: Checksum valid (placeholder - need footer)
        self.add_check(
            4,
            "Checksum valid",
            Category::Structure,
            CheckStatus::Skip("Footer not implemented".to_string()),
        );

        // Checks 5-25 are placeholders for now
        for id in 5..=25 {
            self.add_check(
                id,
                "Pending",
                Category::Structure,
                CheckStatus::Skip("Not implemented".to_string()),
            );
        }
    }

    /// Check 1: Magic bytes valid
    fn check_magic(&mut self, data: &[u8]) {
        let status = if data.len() >= 4 {
            let magic = &data[0..4];
            if magic == b"APRN" || magic == b"APR2" {
                CheckStatus::Pass
            } else {
                CheckStatus::Fail(format!("Invalid magic: {magic:?}"))
            }
        } else {
            CheckStatus::Fail("File too small for magic bytes".to_string())
        };

        self.add_check(1, "Magic bytes valid", Category::Structure, status);
    }

    /// Check 2: Header size fixed
    fn check_header_size(&mut self, data: &[u8]) {
        let status = if data.len() >= 32 {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail(format!("Header incomplete: {} bytes", data.len()))
        };

        self.add_check(2, "Header size fixed", Category::Structure, status);
    }

    /// Check 3: Version supported
    fn check_version(&mut self, header: &AprHeader) {
        let status = if header.is_supported_version() {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail(format!(
                "Unsupported version: {}.{}",
                header.version_major, header.version_minor
            ))
        };

        self.add_check(3, "Version supported", Category::Structure, status);
    }

    /// Check flags (11)
    fn check_flags(&mut self, header: &AprHeader) {
        // Check for undefined flag bits
        let known_flags = 0xFF; // Bits 0-7 are defined
        let unknown = header.flags & !known_flags;

        let status = if unknown == 0 {
            CheckStatus::Pass
        } else {
            CheckStatus::Warn(format!("Unknown flag bits: 0x{unknown:08X}"))
        };

        self.add_check(11, "Flags parsed", Category::Structure, status);
    }

    /// Add a check to the report
    fn add_check(&mut self, id: u8, name: &'static str, category: Category, status: CheckStatus) {
        let points = u8::from(status.is_pass());
        self.report.add_check(ValidationCheck {
            id,
            name,
            category,
            status,
            points,
        });
    }

    /// Get the validation report
    #[must_use]
    pub fn report(&self) -> &ValidationReport {
        &self.report
    }
}

impl Default for AprValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SECTION A: Format & Structural Integrity (25 Points) - TESTS FIRST
// ============================================================================

#[cfg(test)]
mod tests_section_a {
    use super::*;

    // Test 1: Magic bytes valid
    #[test]
    fn test_check_1_magic_valid_aprn() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APRN");
        let mut validator = AprValidator::new();
        validator.validate(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 1)
            .unwrap();
        assert!(check.status.is_pass(), "APRN magic should pass");
    }

    #[test]
    fn test_check_1_magic_valid_apr2() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APR2");
        let mut validator = AprValidator::new();
        validator.validate(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 1)
            .unwrap();
        assert!(check.status.is_pass(), "APR2 magic should pass");
    }

    #[test]
    fn test_check_1_magic_invalid() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"BAD!");
        let mut validator = AprValidator::new();
        validator.validate(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 1)
            .unwrap();
        assert!(check.status.is_fail(), "Invalid magic should fail");
    }

    // Test 2: Header size fixed
    #[test]
    fn test_check_2_header_complete() {
        let data = vec![0u8; 32];
        let mut validator = AprValidator::new();
        validator.validate(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 2)
            .unwrap();
        assert!(check.status.is_pass(), "32-byte header should pass");
    }

    #[test]
    fn test_check_2_header_too_small() {
        let data = vec![0u8; 16]; // Only 16 bytes
        let mut validator = AprValidator::new();
        validator.validate(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 2)
            .unwrap();
        assert!(check.status.is_fail(), "16-byte header should fail");
    }

    // Test 3: Version supported
    #[test]
    fn test_check_3_version_1_0_supported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APRN");
        data[4] = 1; // major
        data[5] = 0; // minor
        let mut validator = AprValidator::new();
        validator.validate(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(check.status.is_pass(), "Version 1.0 should be supported");
    }

    #[test]
    fn test_check_3_version_2_0_supported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APR2");
        data[4] = 2; // major
        data[5] = 0; // minor
        let mut validator = AprValidator::new();
        validator.validate(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(check.status.is_pass(), "Version 2.0 should be supported");
    }

    #[test]
    fn test_check_3_version_unsupported() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APRN");
        data[4] = 3; // major (unsupported)
        data[5] = 0;
        let mut validator = AprValidator::new();
        validator.validate(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 3)
            .unwrap();
        assert!(check.status.is_fail(), "Version 3.0 should fail");
    }

    // Test 11: Flags parsed
    #[test]
    fn test_check_11_known_flags_pass() {
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(b"APRN");
        data[4] = 1;
        data[8] = 0x01; // COMPRESSED flag
        let mut validator = AprValidator::new();
        validator.validate(&data);
        let check = validator
            .report()
            .checks
            .iter()
            .find(|c| c.id == 11)
            .unwrap();
        assert!(check.status.is_pass(), "Known flags should pass");
    }
}

// ============================================================================
// SECTION B: Tensor Physics & Statistics (25 Points) - TESTS FIRST
// ============================================================================

#[cfg(test)]
mod tests_section_b {
    use super::*;

    // Test 26: No NaNs
    #[test]
    fn test_check_26_no_nan_pass() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let stats = TensorStats::compute("test", &data);
        assert!(stats.has_no_nan(), "Clean data should have no NaN");
    }

    #[test]
    fn test_check_26_nan_detected() {
        let data = vec![1.0f32, f32::NAN, 3.0];
        let stats = TensorStats::compute("test", &data);
        assert!(!stats.has_no_nan(), "Should detect NaN");
        assert_eq!(stats.nan_count, 1);
    }

    // Test 27: No Infs
    #[test]
    fn test_check_27_no_inf_pass() {
        let data = vec![1.0f32, 2.0, 3.0];
        let stats = TensorStats::compute("test", &data);
        assert!(stats.has_no_inf(), "Clean data should have no Inf");
    }

    #[test]
    fn test_check_27_inf_detected() {
        let data = vec![1.0f32, f32::INFINITY, f32::NEG_INFINITY];
        let stats = TensorStats::compute("test", &data);
        assert!(!stats.has_no_inf(), "Should detect Inf");
        assert_eq!(stats.inf_count, 2);
    }

    // Test 28: LayerNorm Mean in [0.5, 3.0]
    #[test]
    fn test_check_28_layernorm_mean_valid() {
        // Mean should be ~1.0 for LayerNorm weights
        let data = vec![1.0f32; 384];
        let stats = TensorStats::compute("encoder.layer_norm.weight", &data);
        assert!(
            stats.is_valid_layernorm_weight(),
            "Mean of 1.0 should be valid"
        );
    }

    #[test]
    fn test_check_28_layernorm_mean_too_high() {
        // Bug case: mean=11.0 (10x too high)
        let data = vec![11.0f32; 384];
        let stats = TensorStats::compute("decoder.layer_norm.weight", &data);
        assert!(
            !stats.is_valid_layernorm_weight(),
            "Mean of 11.0 should FAIL - this is the bug we're catching"
        );
    }

    #[test]
    fn test_check_28_layernorm_mean_too_low() {
        let data = vec![0.1f32; 384];
        let stats = TensorStats::compute("encoder.layer_norm.weight", &data);
        assert!(
            !stats.is_valid_layernorm_weight(),
            "Mean of 0.1 should fail"
        );
    }

    // Test 29: LayerNorm Bias in [-0.5, 0.5]
    #[test]
    fn test_check_29_layernorm_bias_valid() {
        let data = vec![0.0f32; 384];
        let stats = TensorStats::compute("encoder.layer_norm.bias", &data);
        assert!(
            stats.is_valid_layernorm_bias(),
            "Mean of 0.0 should be valid"
        );
    }

    #[test]
    fn test_check_29_layernorm_bias_invalid() {
        let data = vec![5.0f32; 384];
        let stats = TensorStats::compute("decoder.layer_norm.bias", &data);
        assert!(!stats.is_valid_layernorm_bias(), "Mean of 5.0 should fail");
    }

    // Test 31: Zero Tensors
    #[test]
    fn test_check_31_not_all_zeros_pass() {
        let data = vec![0.0f32, 0.0, 1.0, 0.0];
        let stats = TensorStats::compute("test", &data);
        assert!(stats.is_not_all_zeros(), "Should pass with some non-zero");
    }

    #[test]
    fn test_check_31_all_zeros_fail() {
        let data = vec![0.0f32; 100];
        let stats = TensorStats::compute("test", &data);
        assert!(!stats.is_not_all_zeros(), "All zeros should fail");
    }

    // Test 35: Attention/Linear Mean ~0
    #[test]
    fn test_check_35_linear_weight_valid() {
        let data = vec![0.01f32, -0.02, 0.03, -0.01];
        let stats = TensorStats::compute("encoder.layers.0.self_attn.q_proj.weight", &data);
        assert!(stats.is_valid_linear_weight(), "Mean ~0 should be valid");
    }

    #[test]
    fn test_check_35_linear_weight_invalid() {
        let data = vec![1.0f32; 100];
        let stats = TensorStats::compute("encoder.layers.0.fc1.weight", &data);
        assert!(!stats.is_valid_linear_weight(), "Mean of 1.0 should fail");
    }

    // Test statistics computation
    #[test]
    fn test_stats_compute_mean() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let stats = TensorStats::compute("test", &data);
        assert!((stats.mean - 3.0).abs() < 0.001, "Mean should be 3.0");
    }

    #[test]
    fn test_stats_compute_std() {
        let data = vec![2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let stats = TensorStats::compute("test", &data);
        // Mean = 5.0, Variance = 32/7 ≈ 4.57, Std ≈ 2.14
        assert!(
            (stats.std - 2.14).abs() < 0.1,
            "Std should be ~2.14, got {}",
            stats.std
        );
    }

    #[test]
    fn test_stats_empty_data() {
        let data: Vec<f32> = vec![];
        let stats = TensorStats::compute("empty", &data);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
    }
}

// ============================================================================
// SECTION C: Tooling & Operations (25 Points) - TESTS FIRST
// ============================================================================

#[cfg(test)]
mod tests_section_c {
    // Test 56: Diff Identity
    #[test]
    fn test_check_56_diff_identity() {
        let data1 = vec![1.0f32, 2.0, 3.0];
        let data2 = vec![1.0f32, 2.0, 3.0];
        let diff = compute_l2_distance(&data1, &data2);
        assert!(diff < 1e-6, "Same data should have zero L2 distance");
    }

    // Test 57: Diff Detection
    #[test]
    fn test_check_57_diff_detection() {
        let data1 = vec![1.0f32, 2.0, 3.0];
        let data2 = vec![1.0f32, 2.0, 4.0]; // Changed last element
        let diff = compute_l2_distance(&data1, &data2);
        assert!(
            diff > 0.5,
            "Different data should have non-zero L2 distance"
        );
    }

    // Test 58: Merge Average
    #[test]
    fn test_check_58_merge_average() {
        let data1 = vec![1.0f32, 2.0, 3.0];
        let data2 = vec![3.0f32, 4.0, 5.0];
        let merged = merge_average(&data1, &data2);
        assert_eq!(merged, vec![2.0f32, 3.0, 4.0], "Average merge failed");
    }

    /// Compute L2 distance between two tensors
    fn compute_l2_distance(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Tensors must have same length");
        let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        sum.sqrt()
    }

    /// Merge two tensors by averaging
    fn merge_average(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len(), "Tensors must have same length");
        a.iter().zip(b.iter()).map(|(x, y)| (x + y) / 2.0).collect()
    }
}

// ============================================================================
// SECTION D: Conversion & Interoperability (25 Points) - TESTS FIRST
// ============================================================================

#[cfg(test)]
mod tests_section_d {
    // Test 79: Roundtrip
    #[test]
    fn test_check_79_roundtrip_tolerance() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        // Simulate roundtrip with small float error
        let roundtrip: Vec<f32> = original.iter().map(|&x| x + 1e-7).collect();
        let max_diff = compute_max_diff(&original, &roundtrip);
        assert!(max_diff < 1e-5, "Roundtrip should have drift < 1e-5");
    }

    // Test 87: Tensor Name Normalization
    #[test]
    fn test_check_87_name_normalization() {
        let hf_name = "model.encoder.conv1.weight";
        let apr_name = normalize_tensor_name(hf_name);
        assert_eq!(
            apr_name, "encoder.conv1.weight",
            "Should strip 'model.' prefix"
        );
    }

    #[test]
    fn test_check_87_name_normalization_no_prefix() {
        let name = "encoder.conv1.weight";
        let apr_name = normalize_tensor_name(name);
        assert_eq!(
            apr_name, "encoder.conv1.weight",
            "Should preserve name without prefix"
        );
    }

    /// Compute max absolute difference between tensors
    fn compute_max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, |acc, x| if x > acc { x } else { acc })
    }

    /// Normalize tensor name to APR canonical form
    fn normalize_tensor_name(name: &str) -> &str {
        name.strip_prefix("model.").unwrap_or(name)
    }
}

// ============================================================================
// ValidationReport Tests
// ============================================================================

#[cfg(test)]
mod tests_report {
    use super::*;

    #[test]
    fn test_report_grade_a_plus() {
        let mut report = ValidationReport::new();
        for i in 1..=95 {
            report.add_check(ValidationCheck {
                id: i,
                name: "test",
                category: Category::Structure,
                status: CheckStatus::Pass,
                points: 1,
            });
        }
        assert_eq!(report.grade(), "A+");
        assert_eq!(report.total_score, 95);
    }

    #[test]
    fn test_report_grade_f() {
        let mut report = ValidationReport::new();
        for i in 1..=50 {
            report.add_check(ValidationCheck {
                id: i,
                name: "test",
                category: Category::Structure,
                status: CheckStatus::Pass,
                points: 1,
            });
        }
        assert_eq!(report.grade(), "F");
        assert_eq!(report.total_score, 50);
    }

    #[test]
    fn test_report_passed_threshold() {
        let mut report = ValidationReport::new();
        for i in 1..=90 {
            report.add_check(ValidationCheck {
                id: i,
                name: "test",
                category: Category::Structure,
                status: CheckStatus::Pass,
                points: 1,
            });
        }
        assert!(report.passed(90));
        assert!(!report.passed(95));
    }

    #[test]
    fn test_report_failed_checks() {
        let mut report = ValidationReport::new();
        report.add_check(ValidationCheck {
            id: 1,
            name: "pass",
            category: Category::Structure,
            status: CheckStatus::Pass,
            points: 1,
        });
        report.add_check(ValidationCheck {
            id: 2,
            name: "fail",
            category: Category::Structure,
            status: CheckStatus::Fail("reason".to_string()),
            points: 0,
        });

        let failed = report.failed_checks();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].id, 2);
    }

    #[test]
    fn test_category_scores() {
        let mut report = ValidationReport::new();
        report.add_check(ValidationCheck {
            id: 1,
            name: "struct1",
            category: Category::Structure,
            status: CheckStatus::Pass,
            points: 1,
        });
        report.add_check(ValidationCheck {
            id: 26,
            name: "physics1",
            category: Category::Physics,
            status: CheckStatus::Pass,
            points: 1,
        });

        assert_eq!(report.category_scores.get(&Category::Structure), Some(&1));
        assert_eq!(report.category_scores.get(&Category::Physics), Some(&1));
    }
}
