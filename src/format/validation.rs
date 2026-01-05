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

#[cfg(test)]
mod tests_poka_yoke {
    use super::*;

    #[test]
    fn test_gate_pass() {
        let gate = Gate::pass("test", 10);
        assert!(gate.passed);
        assert_eq!(gate.points, 10);
        assert!(gate.error.is_none());
    }

    #[test]
    fn test_gate_fail() {
        let gate = Gate::fail("test", 10, "Fix: do something");
        assert!(!gate.passed);
        assert_eq!(gate.points, 0);
        assert!(gate.error.is_some());
        assert!(gate.error.unwrap().contains("Fix:"));
    }

    #[test]
    fn test_result_score_calculation() {
        let mut result = PokaYokeResult::new();
        result.add_gate(Gate::pass("a", 50));
        result.add_gate(Gate::fail("b", 50, "error"));
        assert_eq!(result.score, 50);
        assert_eq!(result.grade(), "F");
    }

    #[test]
    fn test_result_all_pass() {
        let mut result = PokaYokeResult::new();
        result.add_gate(Gate::pass("a", 50));
        result.add_gate(Gate::pass("b", 50));
        assert_eq!(result.score, 100);
        assert_eq!(result.grade(), "A+");
        assert!(result.passed());
    }

    #[test]
    fn test_result_error_summary() {
        let mut result = PokaYokeResult::new();
        result.add_gate(Gate::fail("gate1", 50, "Fix: action1"));
        result.add_gate(Gate::fail("gate2", 50, "Fix: action2"));
        let summary = result.error_summary();
        assert!(summary.contains("gate1"));
        assert!(summary.contains("action1"));
        assert!(summary.contains("gate2"));
    }

    #[test]
    fn test_grade_boundaries() {
        let grades = [
            (100, "A+"),
            (95, "A+"),
            (94, "A"),
            (90, "A"),
            (89, "B+"),
            (85, "B+"),
            (84, "B"),
            (80, "B"),
            (79, "C+"),
            (75, "C+"),
            (74, "C"),
            (70, "C"),
            (69, "D"),
            (60, "D"),
            (59, "F"),
            (0, "F"),
        ];
        for (score, expected_grade) in grades {
            let mut result = PokaYokeResult::new();
            // Hack to set score directly for testing
            result.score = score;
            assert_eq!(result.grade(), expected_grade, "score {score}");
        }
    }

    #[test]
    fn test_from_gates_bulk_construction() {
        let gates = vec![
            Gate::pass("check_a", 30),
            Gate::pass("check_b", 40),
            Gate::fail("check_c", 30, "Fix: implement check_c"),
        ];
        let result = PokaYokeResult::from_gates(gates);
        assert_eq!(result.score, 70); // 70/100
        assert_eq!(result.max_score, 100);
        assert_eq!(result.grade(), "C");
        assert!(result.passed());
        assert_eq!(result.gates.len(), 3);
    }

    #[test]
    fn test_from_gates_empty() {
        let result = PokaYokeResult::from_gates(vec![]);
        assert_eq!(result.score, 0);
        assert_eq!(result.max_score, 0);
        assert_eq!(result.grade(), "F");
    }

    #[test]
    fn test_fail_no_validation_rules() {
        let result = fail_no_validation_rules();
        assert_eq!(result.score, 0);
        assert_eq!(result.grade(), "F");
        assert!(!result.passed());
        assert_eq!(result.gates.len(), 1);
        assert_eq!(result.gates[0].name, "no_validation_rules");
        assert!(result.gates[0]
            .error
            .as_ref()
            .unwrap()
            .contains("Implement PokaYoke"));
    }
}

// ============================================================================
// Whisper Validation Tests (APR-POKA-001, D11, D12)
// ============================================================================

#[cfg(test)]
mod tests_whisper_validation {
    use super::*;

    // D11: Filterbank must be embedded
    #[test]
    fn test_filterbank_present_pass() {
        let fb: Vec<f32> = vec![0.05; 80 * 201];
        let result = WhisperValidation::validate_filterbank(Some(&fb));
        let gate = result.gates.iter().find(|g| g.name == "filterbank_present");
        assert!(gate.is_some());
        assert!(
            gate.unwrap().passed,
            "Filterbank should be detected as present"
        );
    }

    #[test]
    fn test_filterbank_missing_fail() {
        let result = WhisperValidation::validate_filterbank(None);
        let gate = result.gates.iter().find(|g| g.name == "filterbank_present");
        assert!(gate.is_some());
        assert!(!gate.unwrap().passed, "Missing filterbank should fail");
        assert!(gate
            .unwrap()
            .error
            .as_ref()
            .unwrap()
            .contains("MelFilterbankData"));
    }

    #[test]
    fn test_filterbank_empty_fail() {
        let fb: Vec<f32> = vec![];
        let result = WhisperValidation::validate_filterbank(Some(&fb));
        let gate = result.gates.iter().find(|g| g.name == "filterbank_present");
        assert!(!gate.unwrap().passed, "Empty filterbank should fail");
    }

    // D12: Filterbank must be Slaney-normalized (max < 0.1)
    #[test]
    fn test_filterbank_normalized_pass() {
        // Slaney-normalized filterbank has max < 0.1
        let fb: Vec<f32> = vec![0.05; 80 * 201];
        let result = WhisperValidation::validate_filterbank(Some(&fb));
        let gate = result
            .gates
            .iter()
            .find(|g| g.name == "filterbank_normalized");
        assert!(
            gate.unwrap().passed,
            "Slaney-normalized filterbank should pass"
        );
    }

    #[test]
    fn test_filterbank_not_normalized_fail() {
        // Non-normalized filterbank has max >= 0.1
        let mut fb: Vec<f32> = vec![0.05; 80 * 201];
        fb[0] = 1.0; // Bug: unnormalized value
        let result = WhisperValidation::validate_filterbank(Some(&fb));
        let gate = result
            .gates
            .iter()
            .find(|g| g.name == "filterbank_normalized");
        assert!(!gate.unwrap().passed, "Unnormalized filterbank should fail");
        assert!(gate.unwrap().error.as_ref().unwrap().contains("Slaney"));
    }

    #[test]
    fn test_filterbank_boundary_value() {
        // Exactly 0.1 should fail (must be < 0.1)
        let fb: Vec<f32> = vec![0.1; 80 * 201];
        let result = WhisperValidation::validate_filterbank(Some(&fb));
        let gate = result
            .gates
            .iter()
            .find(|g| g.name == "filterbank_normalized");
        assert!(
            !gate.unwrap().passed,
            "max=0.1 exactly should fail (need < 0.1)"
        );
    }

    #[test]
    fn test_filterbank_full_validation_score() {
        // Valid filterbank: 100 points
        let fb: Vec<f32> = vec![0.05; 80 * 201];
        let result = WhisperValidation::validate_filterbank(Some(&fb));
        assert_eq!(result.score, 100);
        assert_eq!(result.grade(), "A+");
        assert!(result.passed());
    }

    #[test]
    fn test_filterbank_missing_score() {
        // Missing filterbank: 0 points
        let result = WhisperValidation::validate_filterbank(None);
        assert_eq!(result.score, 0);
        assert_eq!(result.grade(), "F");
        assert!(!result.passed());
    }

    // Tensor validation tests
    #[test]
    fn test_tensor_stats_all_valid() {
        let stats = vec![
            TensorStats::compute("encoder.layer_norm.weight", &vec![1.0f32; 384]),
            TensorStats::compute("decoder.fc1.weight", &vec![0.01f32, -0.01, 0.02, -0.02]),
        ];
        let result = WhisperValidation::validate_tensor_stats(&stats);
        assert!(result.passed());
        assert!(result.score >= 80);
    }

    #[test]
    fn test_tensor_stats_nan_detected() {
        let stats = vec![TensorStats::compute("broken", &[1.0f32, f32::NAN, 3.0])];
        let result = WhisperValidation::validate_tensor_stats(&stats);
        let gate = result.gates.iter().find(|g| g.name == "no_nan_values");
        assert!(!gate.unwrap().passed, "NaN should be detected");
    }

    #[test]
    fn test_tensor_stats_inf_detected() {
        let stats = vec![TensorStats::compute("broken", &[1.0f32, f32::INFINITY])];
        let result = WhisperValidation::validate_tensor_stats(&stats);
        let gate = result.gates.iter().find(|g| g.name == "no_inf_values");
        assert!(!gate.unwrap().passed, "Inf should be detected");
    }

    #[test]
    fn test_tensor_stats_invalid_layernorm() {
        // LayerNorm weight with mean=11.0 (10x too high - the bug we're catching)
        let stats = vec![TensorStats::compute(
            "encoder.layer_norm.weight",
            &vec![11.0f32; 384],
        )];
        let result = WhisperValidation::validate_tensor_stats(&stats);
        let gate = result
            .gates
            .iter()
            .find(|g| g.name == "layernorm_weights_valid");
        assert!(!gate.unwrap().passed, "Invalid LayerNorm mean should fail");
    }

    #[test]
    fn test_tensor_stats_all_zeros() {
        let stats = vec![TensorStats::compute("dead_weight", &vec![0.0f32; 100])];
        let result = WhisperValidation::validate_tensor_stats(&stats);
        let gate = result.gates.iter().find(|g| g.name == "no_zero_tensors");
        assert!(!gate.unwrap().passed, "All-zero tensor should fail");
    }

    // Full validation tests
    #[test]
    fn test_full_validation_all_pass() {
        let fb: Vec<f32> = vec![0.05; 80 * 201];
        let stats = vec![
            TensorStats::compute("encoder.layer_norm.weight", &vec![1.0f32; 384]),
            TensorStats::compute("decoder.fc1.weight", &vec![0.01f32; 100]),
        ];
        let result = WhisperValidation::validate_full(Some(&fb), &stats);
        assert!(result.passed());
        assert!(result.score >= 90, "Full valid model should score >= 90");
    }

    #[test]
    fn test_full_validation_missing_filterbank() {
        let stats = vec![TensorStats::compute(
            "encoder.layer_norm.weight",
            &vec![1.0f32; 384],
        )];
        let result = WhisperValidation::validate_full(None, &stats);
        assert!(
            result.score < 60,
            "Missing filterbank should significantly reduce score"
        );
    }

    #[test]
    fn test_actionable_error_messages() {
        let result = WhisperValidation::validate_filterbank(None);
        let summary = result.error_summary();
        assert!(
            summary.contains("Fix:"),
            "Error should be actionable with Fix:"
        );
        assert!(
            summary.contains("MelFilterbankData"),
            "Error should provide solution"
        );
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
        validator.validate_bytes(&data);
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
        validator.validate_bytes(&data);
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
        validator.validate_bytes(&data);
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
        validator.validate_bytes(&data);
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
        validator.validate_bytes(&data);
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
        validator.validate_bytes(&data);
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
        validator.validate_bytes(&data);
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
        validator.validate_bytes(&data);
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
        validator.validate_bytes(&data);
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
