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

impl AprValidator {
    /// Create new validator
    #[must_use]
    pub fn new() -> Self {
        Self {
            report: ValidationReport::new(),
            tensor_stats: Vec::new(),
        }
    }

    /// Add tensor stats for validation
    pub fn add_tensor_stats(&mut self, stats: TensorStats) {
        self.tensor_stats.push(stats);
    }

    /// Run validation on file bytes
    pub fn validate_bytes(&mut self, data: &[u8]) -> &ValidationReport {
        self.validate_structure(data);
        &self.report
    }

    /// Run all validation checks (tensor-based)
    pub fn validate(&mut self) -> ValidationReport {
        self.validate_tensors();
        std::mem::take(&mut self.report)
    }

    /// Validate tensor statistics (Section B)
    fn validate_tensors(&mut self) {
        // Check 26: No NaNs
        let nan_count: usize = self.tensor_stats.iter().map(|s| s.nan_count).sum();
        let status = if nan_count == 0 {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail(format!("{nan_count} NaN values found across tensors"))
        };
        self.add_check(26, "No NaN values", Category::Physics, status);

        // Check 27: No Infs
        let inf_count: usize = self.tensor_stats.iter().map(|s| s.inf_count).sum();
        let status = if inf_count == 0 {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail(format!("{inf_count} Inf values found across tensors"))
        };
        self.add_check(27, "No Inf values", Category::Physics, status);

        // Check 28: LayerNorm weights valid
        let invalid_ln: Vec<_> = self
            .tensor_stats
            .iter()
            .filter(|s| {
                (s.name.contains("layer_norm") || s.name.contains("ln_"))
                    && (s.name.ends_with(".weight") || s.name.ends_with(".gamma"))
                    && !s.is_valid_layernorm_weight()
            })
            .collect();

        let status = if invalid_ln.is_empty() {
            CheckStatus::Pass
        } else {
            let names: Vec<_> = invalid_ln
                .iter()
                .map(|s| format!("{} (mean={:.4})", s.name, s.mean))
                .collect();
            CheckStatus::Fail(format!("Invalid LayerNorm weights: {}", names.join(", ")))
        };
        self.add_check(28, "LayerNorm weights valid", Category::Physics, status);

        // Check 31: No all-zero tensors
        let zero_tensors: Vec<_> = self
            .tensor_stats
            .iter()
            .filter(|s| !s.is_not_all_zeros())
            .collect();

        let status = if zero_tensors.is_empty() {
            CheckStatus::Pass
        } else {
            let names: Vec<_> = zero_tensors.iter().map(|s| s.name.clone()).collect();
            CheckStatus::Fail(format!("All-zero tensors: {}", names.join(", ")))
        };
        self.add_check(31, "No all-zero tensors", Category::Physics, status);

        // Checks 29-30, 32-50 placeholders
        for id in [29, 30] {
            self.add_check(
                id,
                "Physics check",
                Category::Physics,
                CheckStatus::Skip("Not implemented".to_string()),
            );
        }
        for id in 32..=50 {
            self.add_check(
                id,
                "Physics/Tooling check",
                if id <= 35 {
                    Category::Physics
                } else {
                    Category::Tooling
                },
                CheckStatus::Skip("Not implemented".to_string()),
            );
        }

        // Checks 51-100 placeholders
        for id in 51..=100 {
            self.add_check(
                id,
                "Advanced check",
                if id <= 75 {
                    Category::Tooling
                } else {
                    Category::Conversion
                },
                CheckStatus::Skip("Not implemented".to_string()),
            );
        }
    }

    /// Run Section A: Format & Structural Integrity checks (1-25)
    ///
    /// GH-178: Detect format (APR vs GGUF) and validate appropriately
    fn validate_structure(&mut self, data: &[u8]) {
        // Check 1: Magic bytes valid
        self.check_magic(data);

        // Check 2: Header size fixed (32 bytes for APR, 8+ for GGUF)
        self.check_header_size(data);

        // GH-178: Detect format and validate version accordingly
        if data.len() >= 4 {
            let magic = data.get(0..4).unwrap_or(&[]);
            if magic == b"GGUF" {
                // GGUF format - check version at bytes 4-7 (u32 LE)
                self.check_gguf_version(data);
                // Skip APR-specific flags check for GGUF
                self.add_check(
                    11,
                    "Flags parsed",
                    Category::Structure,
                    CheckStatus::Skip("GGUF format - no APR flags".to_string()),
                );
            } else if data.len() >= 32 {
                // APR format
                if let Ok(header) = AprHeader::parse(data) {
                    self.check_version(&header);
                    self.check_flags(&header);
                }
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

    /// Check GGUF version (GH-178)
    ///
    /// GGUF versions 1, 2, and 3 are widely supported by llama.cpp
    fn check_gguf_version(&mut self, data: &[u8]) {
        let status = if data.len() >= 8 {
            let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
            // GH-178: GGUF v1, v2, v3 are all valid
            if (1..=3).contains(&version) {
                CheckStatus::Pass
            } else {
                CheckStatus::Fail(format!(
                    "Unsupported GGUF version: {version} (expected 1-3)"
                ))
            }
        } else {
            CheckStatus::Fail("File too small for GGUF version".to_string())
        };

        self.add_check(3, "Version supported", Category::Structure, status);
    }

    /// Check 1: Magic bytes valid
    ///
    /// GH-178/GH-183: Support both APR and GGUF formats:
    /// - APR: `APR\0` (0x41 0x50 0x52 0x00)
    /// - GGUF: `GGUF` (0x47 0x47 0x55 0x46 = [71, 71, 85, 70])
    fn check_magic(&mut self, data: &[u8]) {
        let status = if let Some(magic) = data.get(0..4) {
            if magic == b"APR\0" {
                CheckStatus::Pass
            } else if magic == b"GGUF" {
                // GH-178: GGUF magic is valid ([71, 71, 85, 70] = "GGUF")
                CheckStatus::Pass
            } else {
                // GH-183: Enhanced error message showing hex and ASCII
                let magic_ascii: String = magic
                    .iter()
                    .map(|&b| if b.is_ascii_graphic() { b as char } else { '.' })
                    .collect();
                CheckStatus::Fail(format!(
                    "Invalid magic: {magic:02X?} (ascii: \"{magic_ascii}\"). Expected APR\\0 or GGUF"
                ))
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
// POKA-YOKE: Extensible Model Validation (APR-POKA-001)
// Toyota Way - Mistake-proofing with self-describing quality scores
// ============================================================================

/// Poka-yoke gate result
#[derive(Debug, Clone)]
pub struct Gate {
    /// Gate name (e.g., "`filterbank_present`")
    pub name: &'static str,
    /// Whether gate passed
    pub passed: bool,
    /// Points awarded (0 if failed)
    pub points: u8,
    /// Max points possible
    pub max_points: u8,
    /// Error message if failed
    pub error: Option<String>,
}

impl Gate {
    /// Create a passing gate
    #[must_use]
    pub fn pass(name: &'static str, points: u8) -> Self {
        Self {
            name,
            passed: true,
            points,
            max_points: points,
            error: None,
        }
    }

    /// Create a failing gate with actionable error
    #[must_use]
    pub fn fail(name: &'static str, max_points: u8, error: impl Into<String>) -> Self {
        Self {
            name,
            passed: false,
            points: 0,
            max_points,
            error: Some(error.into()),
        }
    }
}

/// Poka-yoke validation result
#[derive(Debug, Clone, Default)]
pub struct PokaYokeResult {
    /// All gates evaluated
    pub gates: Vec<Gate>,
    /// Total score (0-100)
    pub score: u8,
    /// Maximum possible score
    pub max_score: u8,
}

impl PokaYokeResult {
    /// Create empty result
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create result from a vector of gates (bulk construction)
    ///
    /// # Example
    ///
    /// ```rust
    /// use aprender::format::validation::{Gate, PokaYokeResult};
    ///
    /// let gates = vec![
    ///     Gate::pass("check_a", 30),
    ///     Gate::pass("check_b", 40),
    ///     Gate::fail("check_c", 30, "Fix: implement check_c"),
    /// ];
    /// let result = PokaYokeResult::from_gates(gates);
    /// assert_eq!(result.score, 70); // 70/100
    /// assert_eq!(result.grade(), "C");
    /// ```
    #[must_use]
    pub fn from_gates(gates: Vec<Gate>) -> Self {
        let max_score: u8 = gates
            .iter()
            .map(|g| g.max_points)
            .fold(0u8, u8::saturating_add);
        let total_points: u16 = gates.iter().map(|g| u16::from(g.points)).sum();
        let max_points: u16 = gates.iter().map(|g| u16::from(g.max_points)).sum();
        let score = if max_points > 0 {
            ((total_points * 100) / max_points).min(100) as u8
        } else {
            0
        };
        Self {
            gates,
            score,
            max_score,
        }
    }

    /// Add a gate result
    pub fn add_gate(&mut self, gate: Gate) {
        self.max_score = self.max_score.saturating_add(gate.max_points);
        self.gates.push(gate);
        self.recalculate_score();
    }

    /// Recalculate score from gates
    fn recalculate_score(&mut self) {
        let total_points: u16 = self.gates.iter().map(|g| u16::from(g.points)).sum();
        let max_points: u16 = self.gates.iter().map(|g| u16::from(g.max_points)).sum();
        self.score = if max_points > 0 {
            ((total_points * 100) / max_points).min(100) as u8
        } else {
            0
        };
    }

    /// Get letter grade
    #[must_use]
    pub fn grade(&self) -> &'static str {
        match self.score {
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

    /// Check if validation passed (score >= 60)
    #[must_use]
    pub fn passed(&self) -> bool {
        self.score >= 60
    }

    /// Get all failed gates
    #[must_use]
    pub fn failed_gates(&self) -> Vec<&Gate> {
        self.gates.iter().filter(|g| !g.passed).collect()
    }

    /// Get actionable error summary
    #[must_use]
    pub fn error_summary(&self) -> String {
        let errors: Vec<String> = self
            .failed_gates()
            .iter()
            .filter_map(|g| g.error.as_ref().map(|e| format!("- {}: {}", g.name, e)))
            .collect();
        if errors.is_empty() {
            String::new()
        } else {
            format!("Poka-yoke validation failed:\n{}", errors.join("\n"))
        }
    }
}

/// Extensible Poka-yoke validation trait (implement per model type)
///
/// # Example
///
/// ```rust,ignore
/// impl PokaYoke for WhisperModel {
///     fn validate(&self) -> PokaYokeResult {
///         let mut result = PokaYokeResult::new();
///
///         // Gate 1: Filterbank must be embedded
///         if self.has_filterbank() {
///             result.add_gate(Gate::pass("filterbank_present", 20));
///         } else {
///             result.add_gate(Gate::fail("filterbank_present", 20,
///                 "Fix: Embed Slaney-normalized filterbank via MelFilterbankData::mel_80()"));
///         }
///
///         // Gate 2: Filterbank must be Slaney-normalized
///         if let Some(fb) = self.filterbank() {
///             let max = fb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
///             if max < 0.1 {
///                 result.add_gate(Gate::pass("filterbank_normalized", 30));
///             } else {
///                 result.add_gate(Gate::fail("filterbank_normalized", 30,
///                     format!("Fix: Apply 2.0/bandwidth normalization (max={max:.4}, expected <0.1)")));
///             }
///         }
///
///         result
///     }
/// }
/// ```
pub trait PokaYoke {
    /// Validate model and return quality score (0-100)
    fn poka_yoke_validate(&self) -> PokaYokeResult;

    /// Get quality score (convenience method)
    fn quality_score(&self) -> u8 {
        self.poka_yoke_validate().score
    }
}

/// Create a failing result for models without `PokaYoke` implementation.
///
/// Use this when saving models that don't implement the trait.
/// Returns a result with score=0 and a single failing gate.
///
/// # Example
///
/// ```rust
/// use aprender::format::validation::fail_no_validation_rules;
///
/// let result = fail_no_validation_rules();
/// assert_eq!(result.score, 0);
/// assert_eq!(result.grade(), "F");
/// assert!(!result.passed());
/// ```
#[must_use]
pub fn fail_no_validation_rules() -> PokaYokeResult {
    let mut result = PokaYokeResult::new();
    result.add_gate(Gate::fail(
        "no_validation_rules",
        100,
        "Fix: Implement PokaYoke trait for this model type",
    ));
    result
}

/// Alias for backwards compatibility
#[deprecated(since = "0.19.0", note = "Use fail_no_validation_rules() instead")]
#[must_use]
pub fn no_validation_result() -> PokaYokeResult {
    fail_no_validation_rules()
}

// ============================================================================
// Whisper Model Poka-yoke Validation (APR-POKA-001, D11, D12)
// Toyota Way: Jidoka - Stop and fix quality issues at the source
// ============================================================================

/// Whisper model validation context
///
/// Provides poka-yoke validation for Whisper ASR models:
/// - D11: Filterbank must be embedded for mel models
/// - D12: Filterbank must be Slaney-normalized (max < 0.1)
///
/// # Example
///
/// ```rust
/// use aprender::format::validation::WhisperValidation;
///
/// // Valid Slaney-normalized filterbank (80 bins x 201 FFT bins)
/// let filterbank: Vec<f32> = vec![0.05; 80 * 201];
/// let result = WhisperValidation::validate_filterbank(Some(&filterbank));
/// assert!(result.passed());
/// assert_eq!(result.grade(), "A+");
/// ```
#[derive(Debug, Clone, Default)]
pub struct WhisperValidation;

impl WhisperValidation {
    /// Validate Whisper filterbank (D11: present, D12: Slaney-normalized)
    ///
    /// # Arguments
    /// * `filterbank` - Optional filterbank data (80 mel bins × `n_fft` bins)
    ///
    /// # Returns
    /// `PokaYokeResult` with gates:
    /// - `filterbank_present` (50 pts): Filterbank must be embedded
    /// - `filterbank_normalized` (50 pts): Max value < 0.1 (Slaney normalization)
    #[must_use]
    pub fn validate_filterbank(filterbank: Option<&[f32]>) -> PokaYokeResult {
        let mut gates = Vec::with_capacity(2);

        // D11: Filterbank must be embedded
        match filterbank {
            Some(fb) if !fb.is_empty() => {
                gates.push(Gate::pass("filterbank_present", 50));

                // D12: Filterbank must be Slaney-normalized (max < 0.1)
                let max_val = fb.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if max_val < 0.1 {
                    gates.push(Gate::pass("filterbank_normalized", 50));
                } else {
                    gates.push(Gate::fail(
                        "filterbank_normalized",
                        50,
                        format!(
                            "Fix: Apply 2.0/bandwidth Slaney normalization (max={max_val:.4}, expected <0.1)"
                        ),
                    ));
                }
            }
            _ => {
                gates.push(Gate::fail(
                    "filterbank_present",
                    50,
                    "Fix: Embed mel filterbank via MelFilterbankData::mel_80()",
                ));
                gates.push(Gate::fail(
                    "filterbank_normalized",
                    50,
                    "Fix: Cannot verify normalization - filterbank missing",
                ));
            }
        }

        PokaYokeResult::from_gates(gates)
    }

    /// Validate encoder/decoder tensor statistics
    ///
    /// Checks for common conversion bugs:
    /// - `LayerNorm` weights should have mean ≈ 1.0
    /// - Linear weights should have mean ≈ 0.0
    /// - No NaN/Inf values
    #[must_use]
    pub fn validate_tensor_stats(stats: &[TensorStats]) -> PokaYokeResult {
        let mut gates = Vec::new();

        // Check for NaN values (catastrophic)
        let nan_count: usize = stats.iter().map(|s| s.nan_count).sum();
        if nan_count == 0 {
            gates.push(Gate::pass("no_nan_values", 30));
        } else {
            gates.push(Gate::fail(
                "no_nan_values",
                30,
                format!("Fix: {nan_count} NaN values found - check conversion pipeline"),
            ));
        }

        // Check for Inf values (catastrophic)
        let inf_count: usize = stats.iter().map(|s| s.inf_count).sum();
        if inf_count == 0 {
            gates.push(Gate::pass("no_inf_values", 20));
        } else {
            gates.push(Gate::fail(
                "no_inf_values",
                20,
                format!("Fix: {inf_count} Inf values found - check overflow in conversion"),
            ));
        }

        // Check LayerNorm weights
        let invalid_ln: Vec<_> = stats
            .iter()
            .filter(|s| {
                (s.name.contains("layer_norm") || s.name.contains("ln_"))
                    && (s.name.ends_with(".weight") || s.name.ends_with(".gamma"))
                    && !s.is_valid_layernorm_weight()
            })
            .collect();

        if invalid_ln.is_empty() {
            gates.push(Gate::pass("layernorm_weights_valid", 25));
        } else {
            let names: Vec<_> = invalid_ln
                .iter()
                .take(3)
                .map(|s| format!("{} (mean={:.4})", s.name, s.mean))
                .collect();
            gates.push(Gate::fail(
                "layernorm_weights_valid",
                25,
                format!(
                    "Fix: LayerNorm weights should have mean in [0.5, 3.0]: {}",
                    names.join(", ")
                ),
            ));
        }

        // Check for all-zero tensors (dead weights)
        let zero_tensors: Vec<_> = stats.iter().filter(|s| !s.is_not_all_zeros()).collect();

        if zero_tensors.is_empty() {
            gates.push(Gate::pass("no_zero_tensors", 25));
        } else {
            let names: Vec<_> = zero_tensors
                .iter()
                .take(3)
                .map(|s| s.name.clone())
                .collect();
            gates.push(Gate::fail(
                "no_zero_tensors",
                25,
                format!(
                    "Fix: All-zero tensors found (dead weights): {}",
                    names.join(", ")
                ),
            ));
        }

        PokaYokeResult::from_gates(gates)
    }

    /// Full Whisper model validation
    ///
    /// Combines filterbank and tensor validation into single result.
    #[must_use]
    pub fn validate_full(
        filterbank: Option<&[f32]>,
        tensor_stats: &[TensorStats],
    ) -> PokaYokeResult {
        let fb_result = Self::validate_filterbank(filterbank);
        let tensor_result = Self::validate_tensor_stats(tensor_stats);

        // Combine gates with weighted scoring
        let mut all_gates = fb_result.gates;
        all_gates.extend(tensor_result.gates);

        PokaYokeResult::from_gates(all_gates)
    }
}

// ============================================================================
// Poka-yoke Tests (APR-POKA-001)
// ============================================================================

// Tests extracted to validation_tests.rs (PMAT-197)
#[cfg(test)]
#[path = "validation_tests.rs"]
mod tests;
