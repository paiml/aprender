//! APR Format Lint Module - Best Practices & Conventions
//!
//! Implements APR-SPEC §4.11: Lint Command
//!
//! Static analysis for best practices, conventions, and "soft" requirements.
//! Unlike `validate` (which checks for corruption/invalidity), `lint` checks
//! for *quality* and *standardization*.
//!
//! # Checks Performed
//!
//! - **Metadata**: Missing `license`, `model_card`, or `provenance` (WARN)
//! - **Tensor Naming**: Names not matching canonical schema (INFO/WARN)
//! - **Efficiency**: Tensors unaligned to 64 bytes (INFO)
//! - **Compression**: Uncompressed tensors >1MB (INFO)

use crate::error::Result;
use std::collections::HashMap;
use std::path::Path;

/// Severity level for lint issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LintLevel {
    /// Informational - suggestion for improvement
    Info,
    /// Warning - best practice violation
    Warn,
    /// Error - significant convention violation
    Error,
}

impl LintLevel {
    /// Get display string
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        }
    }
}

impl std::fmt::Display for LintLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Category of lint check
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LintCategory {
    /// Metadata checks (license, `model_card`, provenance)
    Metadata,
    /// Tensor naming convention checks
    Naming,
    /// Efficiency checks (alignment, compression)
    Efficiency,
}

impl LintCategory {
    /// Get display name
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Metadata => "Metadata",
            Self::Naming => "Tensor Naming",
            Self::Efficiency => "Efficiency",
        }
    }
}

impl std::fmt::Display for LintCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A single lint issue found during analysis
#[derive(Debug, Clone)]
pub struct LintIssue {
    /// Severity level
    pub level: LintLevel,
    /// Category of the issue
    pub category: LintCategory,
    /// Human-readable message
    pub message: String,
    /// Optional suggestion for fixing
    pub suggestion: Option<String>,
}

impl LintIssue {
    /// Create a new lint issue
    #[must_use]
    pub fn new(level: LintLevel, category: LintCategory, message: impl Into<String>) -> Self {
        Self {
            level,
            category,
            message: message.into(),
            suggestion: None,
        }
    }

    /// Add a suggestion for fixing
    #[must_use]
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Create metadata warning
    #[must_use]
    pub fn metadata_warn(message: impl Into<String>) -> Self {
        Self::new(LintLevel::Warn, LintCategory::Metadata, message)
    }

    /// Create naming info
    #[must_use]
    pub fn naming_info(message: impl Into<String>) -> Self {
        Self::new(LintLevel::Info, LintCategory::Naming, message)
    }

    /// Create naming warning
    #[must_use]
    pub fn naming_warn(message: impl Into<String>) -> Self {
        Self::new(LintLevel::Warn, LintCategory::Naming, message)
    }

    /// Create efficiency info
    #[must_use]
    pub fn efficiency_info(message: impl Into<String>) -> Self {
        Self::new(LintLevel::Info, LintCategory::Efficiency, message)
    }
}

impl std::fmt::Display for LintIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.level, self.category, self.message)?;
        if let Some(ref suggestion) = self.suggestion {
            write!(f, " (suggestion: {})", suggestion)?;
        }
        Ok(())
    }
}

/// Complete lint report for a model
#[derive(Debug, Clone)]
pub struct LintReport {
    /// All issues found
    pub issues: Vec<LintIssue>,
    /// Issues by category
    pub by_category: HashMap<LintCategory, Vec<usize>>,
    /// Count by level
    pub info_count: usize,
    pub warn_count: usize,
    pub error_count: usize,
}

impl LintReport {
    /// Create empty report
    #[must_use]
    pub fn new() -> Self {
        Self {
            issues: Vec::new(),
            by_category: HashMap::new(),
            info_count: 0,
            warn_count: 0,
            error_count: 0,
        }
    }

    /// Add an issue to the report
    pub fn add_issue(&mut self, issue: LintIssue) {
        let idx = self.issues.len();
        let category = issue.category;
        match issue.level {
            LintLevel::Info => self.info_count += 1,
            LintLevel::Warn => self.warn_count += 1,
            LintLevel::Error => self.error_count += 1,
        }
        self.by_category.entry(category).or_default().push(idx);
        self.issues.push(issue);
    }

    /// Check if lint passed (no warnings or errors)
    #[must_use]
    pub fn passed(&self) -> bool {
        self.warn_count == 0 && self.error_count == 0
    }

    /// Check if lint passed with only infos allowed
    #[must_use]
    pub fn passed_strict(&self) -> bool {
        self.issues.is_empty()
    }

    /// Get total issue count
    #[must_use]
    pub fn total_issues(&self) -> usize {
        self.issues.len()
    }

    /// Get issues of a specific level
    #[must_use]
    pub fn issues_at_level(&self, level: LintLevel) -> Vec<&LintIssue> {
        self.issues.iter().filter(|i| i.level == level).collect()
    }

    /// Get issues in a specific category
    #[must_use]
    pub fn issues_in_category(&self, category: LintCategory) -> Vec<&LintIssue> {
        match self.by_category.get(&category) {
            Some(indices) => indices.iter().map(|&i| &self.issues[i]).collect(),
            None => Vec::new(),
        }
    }
}

impl Default for LintReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Information extracted from a model for linting
#[derive(Debug, Clone, Default)]
pub struct ModelLintInfo {
    /// Metadata fields present
    pub has_license: bool,
    pub has_model_card: bool,
    pub has_provenance: bool,
    /// Tensor information
    pub tensors: Vec<TensorLintInfo>,
    /// Whether compression is enabled
    pub is_compressed: bool,
}

/// Information about a single tensor for linting
#[derive(Debug, Clone)]
pub struct TensorLintInfo {
    /// Tensor name
    pub name: String,
    /// Size in bytes
    pub size_bytes: usize,
    /// Alignment (in bytes)
    pub alignment: usize,
    /// Whether tensor is compressed
    pub is_compressed: bool,
}

/// Canonical tensor name patterns for Whisper models
const CANONICAL_PATTERNS: &[(&str, &str)] = &[
    ("encoder.conv1.weight", "Initial convolution weight"),
    ("encoder.conv1.bias", "Initial convolution bias"),
    ("encoder.conv2.weight", "Second convolution weight"),
    ("encoder.conv2.bias", "Second convolution bias"),
    ("encoder.positional_embedding", "Encoder position embedding"),
    (
        "encoder.layer_norm.weight",
        "Encoder final layer norm weight",
    ),
    ("encoder.layer_norm.bias", "Encoder final layer norm bias"),
    ("decoder.token_embedding", "Token embedding"),
    ("decoder.positional_embedding", "Decoder position embedding"),
    (
        "decoder.layer_norm.weight",
        "Decoder final layer norm weight",
    ),
    ("decoder.layer_norm.bias", "Decoder final layer norm bias"),
    ("proj_out.weight", "Output projection weight"),
];

/// Abbreviated names that should be expanded
const ABBREVIATION_SUGGESTIONS: &[(&str, &str)] = &[
    (".w", ".weight"),
    (".b", ".bias"),
    ("_w", ".weight"),
    ("_b", ".bias"),
    (".wt", ".weight"),
    (".bs", ".bias"),
    ("attn_", "self_attn."),
    ("ffn_", "fc"),
    ("ln_", "layer_norm."),
    ("emb_", "embedding"),
    ("embed_", "embedding"),
];

/// Run lint checks on model info
#[must_use]
pub fn lint_model(info: &ModelLintInfo) -> LintReport {
    let mut report = LintReport::new();

    // Metadata checks (WARN level)
    check_metadata(&mut report, info);

    // Tensor naming checks (INFO/WARN level)
    check_tensor_naming(&mut report, info);

    // Efficiency checks (INFO level)
    check_efficiency(&mut report, info);

    report
}

/// Check metadata requirements
fn check_metadata(report: &mut LintReport, info: &ModelLintInfo) {
    if !info.has_license {
        report.add_issue(LintIssue::metadata_warn("Missing 'license' field"));
    }

    if !info.has_model_card {
        report.add_issue(LintIssue::metadata_warn("Missing 'model_card'"));
    }

    if !info.has_provenance {
        report.add_issue(LintIssue::metadata_warn("Missing 'provenance' information"));
    }
}

/// Check tensor naming conventions
fn check_tensor_naming(report: &mut LintReport, info: &ModelLintInfo) {
    for tensor in &info.tensors {
        // Check for abbreviated names
        for (abbrev, full) in ABBREVIATION_SUGGESTIONS {
            if is_abbreviated(&tensor.name, abbrev, full) {
                let suggested = tensor.name.replace(abbrev, full);
                report.add_issue(
                    LintIssue::naming_info(format!(
                        "'{}' should be '{}' for auto-mapping",
                        tensor.name, suggested
                    ))
                    .with_suggestion(format!("Rename to '{}'", suggested)),
                );
            }
        }

        // Check for non-standard layer patterns
        if is_nonstandard_pattern(&tensor.name) {
            report.add_issue(
                LintIssue::naming_warn(format!(
                    "'{}' does not follow canonical naming schema",
                    tensor.name
                ))
                .with_suggestion("See APR-SPEC §10.8 for canonical tensor naming"),
            );
        }
    }
}

/// Check if position in string is at a word boundary (end of string or followed by separator).
fn is_at_word_boundary(name: &str, position: usize) -> bool {
    if position >= name.len() {
        return true;
    }
    matches!(name.chars().nth(position), Some('.' | '_' | '-'))
}

/// Check if a name contains an abbreviation that should be expanded
/// Returns false if the full form is already present
fn is_abbreviated(name: &str, abbrev: &str, full: &str) -> bool {
    // If the name already contains the full form, it's not abbreviated
    if name.contains(full) {
        return false;
    }

    // Check if the abbreviation appears in the name at a word boundary
    name.find(abbrev)
        .is_some_and(|pos| is_at_word_boundary(name, pos + abbrev.len()))
}

/// Known patterns where numeric suffixes are acceptable.
const STANDARD_NUMERIC_PATTERNS: &[&str] = &["layers.", "conv1", "conv2", "fc1", "fc2"];

/// Check if name contains numbers in a standard pattern.
fn has_standard_numbering(name: &str) -> bool {
    STANDARD_NUMERIC_PATTERNS.iter().any(|p| name.contains(p))
}

/// Check if name has unusual separator sequences.
fn has_unusual_separators(name: &str) -> bool {
    name.contains("__") || name.contains("--") || name.contains("..")
}

/// Check if tensor name has non-standard pattern
fn is_nonstandard_pattern(name: &str) -> bool {
    // Numbers not in standard patterns
    let has_odd_numbers = name.chars().any(|c| c.is_ascii_digit()) && !has_standard_numbering(name);

    // Names that are too short (likely abbreviated)
    let too_short = !name.is_empty() && name.len() < 5;

    has_odd_numbers || has_unusual_separators(name) || too_short
}

/// Check efficiency requirements
fn check_efficiency(report: &mut LintReport, info: &ModelLintInfo) {
    const ALIGNMENT_TARGET: usize = 64;
    const LARGE_TENSOR_THRESHOLD: usize = 1024 * 1024; // 1MB

    let mut unaligned_count = 0;
    let mut uncompressed_large_count = 0;

    for tensor in &info.tensors {
        // Check alignment
        if tensor.alignment < ALIGNMENT_TARGET && tensor.alignment > 0 {
            unaligned_count += 1;
        }

        // Check for uncompressed large tensors
        if !tensor.is_compressed && tensor.size_bytes > LARGE_TENSOR_THRESHOLD {
            uncompressed_large_count += 1;
        }
    }

    if unaligned_count > 0 {
        report.add_issue(LintIssue::efficiency_info(format!(
            "{} tensors could be aligned to 64 bytes (currently unaligned)",
            unaligned_count
        )));
    }

    if uncompressed_large_count > 0 {
        report.add_issue(LintIssue::efficiency_info(format!(
            "{} uncompressed tensors exceed 1MB - consider compression",
            uncompressed_large_count
        )));
    }
}

/// Check if license info exists in header/metadata.
fn has_license_info(header: &crate::format::Header, metadata: &crate::format::Metadata) -> bool {
    metadata.license.is_some()
        || metadata.custom.contains_key("license")
        || header.flags.is_licensed()
}

/// Check if model card info exists in header/metadata.
fn has_model_card_info(header: &crate::format::Header, metadata: &crate::format::Metadata) -> bool {
    metadata.model_card.is_some()
        || metadata.custom.contains_key("model_card")
        || header.flags.has_model_card()
}

/// Check if provenance info exists in metadata.
fn has_provenance_info(metadata: &crate::format::Metadata) -> bool {
    metadata.distillation.is_some()
        || metadata.distillation_info.is_some()
        || metadata.training.is_some()
        || metadata.custom.contains_key("provenance")
        || metadata.custom.contains_key("author")
}

/// Lint an APR file from disk
pub fn lint_apr_file(path: impl AsRef<Path>) -> Result<LintReport> {
    use crate::format::{Header, Metadata, HEADER_SIZE};
    use std::fs::File;
    use std::io::{BufReader, Read};

    let path = path.as_ref();
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read header
    let mut header_bytes = [0u8; HEADER_SIZE];
    reader.read_exact(&mut header_bytes)?;
    let header = Header::from_bytes(&header_bytes)?;

    // Read metadata
    let mut metadata_bytes = vec![0u8; header.metadata_size as usize];
    reader.read_exact(&mut metadata_bytes)?;
    let metadata: Metadata = rmp_serde::from_slice(&metadata_bytes).unwrap_or_default();

    // Build lint info from header/metadata
    let mut info = ModelLintInfo::default();
    info.has_license = has_license_info(&header, &metadata);
    info.has_model_card = has_model_card_info(&header, &metadata);
    info.has_provenance = has_provenance_info(&metadata);
    info.is_compressed = header.compression != crate::format::Compression::None;

    // For tensor info, we need to read tensor index
    let payload_size = header.payload_size as usize;
    if payload_size > 0 {
        info.tensors.push(TensorLintInfo {
            name: "payload".to_string(),
            size_bytes: payload_size,
            alignment: 64, // Assume aligned for now
            is_compressed: info.is_compressed,
        });
    }

    Ok(lint_model(&info))
}

// ============================================================================
// TESTS - Written first following EXTREME TDD
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Test LintLevel
    // ========================================================================

    #[test]
    fn test_lint_level_ordering() {
        assert!(LintLevel::Info < LintLevel::Warn);
        assert!(LintLevel::Warn < LintLevel::Error);
    }

    #[test]
    fn test_lint_level_display() {
        assert_eq!(LintLevel::Info.as_str(), "INFO");
        assert_eq!(LintLevel::Warn.as_str(), "WARN");
        assert_eq!(LintLevel::Error.as_str(), "ERROR");
    }

    // ========================================================================
    // Test LintCategory
    // ========================================================================

    #[test]
    fn test_lint_category_names() {
        assert_eq!(LintCategory::Metadata.name(), "Metadata");
        assert_eq!(LintCategory::Naming.name(), "Tensor Naming");
        assert_eq!(LintCategory::Efficiency.name(), "Efficiency");
    }

    // ========================================================================
    // Test LintIssue
    // ========================================================================

    #[test]
    fn test_lint_issue_creation() {
        let issue = LintIssue::new(LintLevel::Warn, LintCategory::Metadata, "Missing license");
        assert_eq!(issue.level, LintLevel::Warn);
        assert_eq!(issue.category, LintCategory::Metadata);
        assert_eq!(issue.message, "Missing license");
        assert!(issue.suggestion.is_none());
    }

    #[test]
    fn test_lint_issue_with_suggestion() {
        let issue = LintIssue::naming_info("Use full name")
            .with_suggestion("Rename 'enc.w' to 'encoder.weight'");

        assert_eq!(issue.level, LintLevel::Info);
        assert_eq!(issue.category, LintCategory::Naming);
        assert!(issue.suggestion.is_some());
    }

    #[test]
    fn test_lint_issue_display() {
        let issue = LintIssue::metadata_warn("Missing 'license' field");
        let display = format!("{}", issue);
        assert!(display.contains("[WARN]"));
        assert!(display.contains("Metadata"));
        assert!(display.contains("Missing 'license' field"));
    }

    // ========================================================================
    // Test LintReport
    // ========================================================================

    #[test]
    fn test_lint_report_empty() {
        let report = LintReport::new();
        assert!(report.passed());
        assert!(report.passed_strict());
        assert_eq!(report.total_issues(), 0);
    }

    #[test]
    fn test_lint_report_add_issues() {
        let mut report = LintReport::new();

        report.add_issue(LintIssue::metadata_warn("Missing license"));
        report.add_issue(LintIssue::efficiency_info("Unaligned tensors"));

        assert_eq!(report.total_issues(), 2);
        assert_eq!(report.warn_count, 1);
        assert_eq!(report.info_count, 1);
        assert!(!report.passed()); // Has warning
        assert!(!report.passed_strict()); // Has issues
    }

    #[test]
    fn test_lint_report_info_only_passes() {
        let mut report = LintReport::new();
        report.add_issue(LintIssue::efficiency_info("Suggestion"));

        assert!(report.passed()); // Info doesn't fail
        assert!(!report.passed_strict()); // But not strictly clean
    }

    #[test]
    fn test_lint_report_issues_by_category() {
        let mut report = LintReport::new();
        report.add_issue(LintIssue::metadata_warn("Missing license"));
        report.add_issue(LintIssue::metadata_warn("Missing model_card"));
        report.add_issue(LintIssue::efficiency_info("Unaligned"));

        let metadata_issues = report.issues_in_category(LintCategory::Metadata);
        assert_eq!(metadata_issues.len(), 2);

        let efficiency_issues = report.issues_in_category(LintCategory::Efficiency);
        assert_eq!(efficiency_issues.len(), 1);
    }

    #[test]
    fn test_lint_report_issues_by_level() {
        let mut report = LintReport::new();
        report.add_issue(LintIssue::metadata_warn("Warning 1"));
        report.add_issue(LintIssue::metadata_warn("Warning 2"));
        report.add_issue(LintIssue::efficiency_info("Info 1"));

        let warnings = report.issues_at_level(LintLevel::Warn);
        assert_eq!(warnings.len(), 2);

        let infos = report.issues_at_level(LintLevel::Info);
        assert_eq!(infos.len(), 1);
    }

    // ========================================================================
    // Test Metadata Checks
    // ========================================================================

    #[test]
    fn test_lint_missing_license() {
        let info = ModelLintInfo {
            has_license: false,
            has_model_card: true,
            has_provenance: true,
            ..Default::default()
        };

        let report = lint_model(&info);
        assert_eq!(report.warn_count, 1);

        let metadata_issues = report.issues_in_category(LintCategory::Metadata);
        assert_eq!(metadata_issues.len(), 1);
        assert!(metadata_issues[0].message.contains("license"));
    }

    #[test]
    fn test_lint_missing_model_card() {
        let info = ModelLintInfo {
            has_license: true,
            has_model_card: false,
            has_provenance: true,
            ..Default::default()
        };

        let report = lint_model(&info);
        assert_eq!(report.warn_count, 1);

        let metadata_issues = report.issues_in_category(LintCategory::Metadata);
        assert!(metadata_issues[0].message.contains("model_card"));
    }

    #[test]
    fn test_lint_missing_provenance() {
        let info = ModelLintInfo {
            has_license: true,
            has_model_card: true,
            has_provenance: false,
            ..Default::default()
        };

        let report = lint_model(&info);
        assert_eq!(report.warn_count, 1);

        let metadata_issues = report.issues_in_category(LintCategory::Metadata);
        assert!(metadata_issues[0].message.contains("provenance"));
    }

    #[test]
    fn test_lint_all_metadata_present() {
        let info = ModelLintInfo {
            has_license: true,
            has_model_card: true,
            has_provenance: true,
            ..Default::default()
        };

        let report = lint_model(&info);
        let metadata_issues = report.issues_in_category(LintCategory::Metadata);
        assert!(metadata_issues.is_empty());
    }

    #[test]
    fn test_lint_all_metadata_missing() {
        let info = ModelLintInfo {
            has_license: false,
            has_model_card: false,
            has_provenance: false,
            ..Default::default()
        };

        let report = lint_model(&info);
        assert_eq!(report.warn_count, 3);

        let metadata_issues = report.issues_in_category(LintCategory::Metadata);
        assert_eq!(metadata_issues.len(), 3);
    }

    // ========================================================================
    // Test Tensor Naming Checks
    // ========================================================================

    #[test]
    fn test_lint_abbreviated_weight_name() {
        let info = ModelLintInfo {
            has_license: true,
            has_model_card: true,
            has_provenance: true,
            tensors: vec![TensorLintInfo {
                name: "encoder.conv1.w".to_string(),
                size_bytes: 1000,
                alignment: 64,
                is_compressed: false,
            }],
            ..Default::default()
        };

        let report = lint_model(&info);
        let naming_issues = report.issues_in_category(LintCategory::Naming);
        assert!(!naming_issues.is_empty());
        assert!(naming_issues[0].message.contains(".weight"));
    }

    #[test]
    fn test_lint_abbreviated_bias_name() {
        let info = ModelLintInfo {
            has_license: true,
            has_model_card: true,
            has_provenance: true,
            tensors: vec![TensorLintInfo {
                name: "encoder.conv1.b".to_string(),
                size_bytes: 1000,
                alignment: 64,
                is_compressed: false,
            }],
            ..Default::default()
        };

        let report = lint_model(&info);
        let naming_issues = report.issues_in_category(LintCategory::Naming);
        assert!(!naming_issues.is_empty());
        assert!(naming_issues[0].message.contains(".bias"));
    }

    #[test]
    fn test_lint_canonical_name_passes() {
        let info = ModelLintInfo {
            has_license: true,
            has_model_card: true,
            has_provenance: true,
            tensors: vec![TensorLintInfo {
                name: "encoder.conv1.weight".to_string(),
                size_bytes: 1000,
                alignment: 64,
                is_compressed: false,
            }],
            ..Default::default()
        };

        let report = lint_model(&info);
        let naming_issues = report.issues_in_category(LintCategory::Naming);
        // Canonical name should not trigger abbreviated name warnings
        // (might trigger non-standard pattern check)
        let abbrev_issues: Vec<_> = naming_issues
            .iter()
            .filter(|i| i.message.contains("should be"))
            .collect();
        assert!(abbrev_issues.is_empty());
    }

    #[test]
    fn test_lint_nonstandard_pattern_double_underscore() {
        let info = ModelLintInfo {
            has_license: true,
            has_model_card: true,
            has_provenance: true,
            tensors: vec![TensorLintInfo {
                name: "encoder__weight".to_string(),
                size_bytes: 1000,
                alignment: 64,
                is_compressed: false,
            }],
            ..Default::default()
        };

        let report = lint_model(&info);
        let naming_issues = report.issues_in_category(LintCategory::Naming);
        let nonstandard: Vec<_> = naming_issues
            .iter()
            .filter(|i| i.message.contains("canonical naming"))
            .collect();
        assert!(!nonstandard.is_empty());
    }

    #[test]
    fn test_lint_too_short_name() {
        let info = ModelLintInfo {
            has_license: true,
            has_model_card: true,
            has_provenance: true,
            tensors: vec![TensorLintInfo {
                name: "w".to_string(),
                size_bytes: 1000,
                alignment: 64,
                is_compressed: false,
            }],
            ..Default::default()
        };

        let report = lint_model(&info);
        let naming_issues = report.issues_in_category(LintCategory::Naming);
        assert!(!naming_issues.is_empty());
    }

    // ========================================================================
    // Test Efficiency Checks
    // ========================================================================

    #[test]
    fn test_lint_unaligned_tensors() {
        let info = ModelLintInfo {
            has_license: true,
            has_model_card: true,
            has_provenance: true,
            tensors: vec![
                TensorLintInfo {
                    name: "tensor1".to_string(),
                    size_bytes: 1000,
                    alignment: 32, // Unaligned
                    is_compressed: false,
                },
                TensorLintInfo {
                    name: "tensor2".to_string(),
                    size_bytes: 1000,
                    alignment: 16, // Unaligned
                    is_compressed: false,
                },
            ],
            ..Default::default()
        };

        let report = lint_model(&info);
        let efficiency_issues = report.issues_in_category(LintCategory::Efficiency);
        let alignment_issues: Vec<_> = efficiency_issues
            .iter()
            .filter(|i| i.message.contains("aligned"))
            .collect();
        assert!(!alignment_issues.is_empty());
        assert!(alignment_issues[0].message.contains("2 tensors"));
    }

    #[test]
    fn test_lint_aligned_tensors_pass() {
        let info = ModelLintInfo {
            has_license: true,
            has_model_card: true,
            has_provenance: true,
            tensors: vec![TensorLintInfo {
                name: "tensor1".to_string(),
                size_bytes: 1000,
                alignment: 64, // Aligned
                is_compressed: false,
            }],
            ..Default::default()
        };

        let report = lint_model(&info);
        let efficiency_issues = report.issues_in_category(LintCategory::Efficiency);
        let alignment_issues: Vec<_> = efficiency_issues
            .iter()
            .filter(|i| i.message.contains("aligned"))
            .collect();
        assert!(alignment_issues.is_empty());
    }

    #[test]
    fn test_lint_large_uncompressed_tensor() {
        let info = ModelLintInfo {
            has_license: true,
            has_model_card: true,
            has_provenance: true,
            tensors: vec![TensorLintInfo {
                name: "large_tensor".to_string(),
                size_bytes: 2 * 1024 * 1024, // 2MB
                alignment: 64,
                is_compressed: false, // Not compressed
            }],
            ..Default::default()
        };

        let report = lint_model(&info);
        let efficiency_issues = report.issues_in_category(LintCategory::Efficiency);
        let compression_issues: Vec<_> = efficiency_issues
            .iter()
            .filter(|i| i.message.contains("compression"))
            .collect();
        assert!(!compression_issues.is_empty());
    }

    #[test]
    fn test_lint_large_compressed_tensor_passes() {
        let info = ModelLintInfo {
            has_license: true,
            has_model_card: true,
            has_provenance: true,
            tensors: vec![TensorLintInfo {
                name: "large_tensor".to_string(),
                size_bytes: 2 * 1024 * 1024, // 2MB
                alignment: 64,
                is_compressed: true, // Compressed
            }],
            ..Default::default()
        };

        let report = lint_model(&info);
        let efficiency_issues = report.issues_in_category(LintCategory::Efficiency);
        let compression_issues: Vec<_> = efficiency_issues
            .iter()
            .filter(|i| i.message.contains("compression"))
            .collect();
        assert!(compression_issues.is_empty());
    }

    #[test]
    fn test_lint_small_uncompressed_tensor_passes() {
        let info = ModelLintInfo {
            has_license: true,
            has_model_card: true,
            has_provenance: true,
            tensors: vec![TensorLintInfo {
                name: "small_tensor".to_string(),
                size_bytes: 500 * 1024, // 500KB
                alignment: 64,
                is_compressed: false,
            }],
            ..Default::default()
        };

        let report = lint_model(&info);
        let efficiency_issues = report.issues_in_category(LintCategory::Efficiency);
        let compression_issues: Vec<_> = efficiency_issues
            .iter()
            .filter(|i| i.message.contains("compression"))
            .collect();
        assert!(compression_issues.is_empty());
    }

    // ========================================================================
    // Test Complete Model Lint
    // ========================================================================

    #[test]
    fn test_lint_clean_model() {
        let info = ModelLintInfo {
            has_license: true,
            has_model_card: true,
            has_provenance: true,
            tensors: vec![
                TensorLintInfo {
                    name: "encoder.conv1.weight".to_string(),
                    size_bytes: 500 * 1024,
                    alignment: 64,
                    is_compressed: true,
                },
                TensorLintInfo {
                    name: "encoder.conv1.bias".to_string(),
                    size_bytes: 1024,
                    alignment: 64,
                    is_compressed: true,
                },
            ],
            is_compressed: true,
        };

        let report = lint_model(&info);
        assert!(report.passed(), "Clean model should pass lint");
    }

    #[test]
    fn test_lint_model_with_all_issues() {
        let info = ModelLintInfo {
            has_license: false,    // WARN
            has_model_card: false, // WARN
            has_provenance: false, // WARN
            tensors: vec![TensorLintInfo {
                name: "enc.w".to_string(),   // INFO (abbreviated)
                size_bytes: 2 * 1024 * 1024, // INFO (large uncompressed)
                alignment: 32,               // INFO (unaligned)
                is_compressed: false,
            }],
            is_compressed: false,
        };

        let report = lint_model(&info);
        assert!(!report.passed(), "Model with issues should not pass");
        assert!(report.warn_count >= 3, "Should have at least 3 warnings");
        assert!(report.info_count >= 1, "Should have at least 1 info");
    }

    // ========================================================================
    // Test is_nonstandard_pattern
    // ========================================================================

    #[test]
    fn test_is_nonstandard_double_underscore() {
        assert!(is_nonstandard_pattern("foo__bar"));
    }

    #[test]
    fn test_is_nonstandard_double_dot() {
        assert!(is_nonstandard_pattern("foo..bar"));
    }

    #[test]
    fn test_is_nonstandard_too_short() {
        assert!(is_nonstandard_pattern("w"));
        assert!(is_nonstandard_pattern("ab"));
    }

    #[test]
    fn test_is_nonstandard_odd_numbers() {
        // Numbers not in standard patterns
        assert!(is_nonstandard_pattern("weight_123"));
    }

    #[test]
    fn test_standard_patterns() {
        assert!(!is_nonstandard_pattern("encoder.conv1.weight"));
        assert!(!is_nonstandard_pattern(
            "encoder.layers.0.self_attn.q_proj.weight"
        ));
        assert!(!is_nonstandard_pattern("decoder.fc1.weight"));
    }

    // ========================================================================
    // Additional Coverage Tests for lint.rs
    // ========================================================================

    #[test]
    fn test_lint_level_display_impl() {
        assert_eq!(format!("{}", LintLevel::Info), "INFO");
        assert_eq!(format!("{}", LintLevel::Warn), "WARN");
        assert_eq!(format!("{}", LintLevel::Error), "ERROR");
    }

    #[test]
    fn test_lint_category_display_impl() {
        assert_eq!(format!("{}", LintCategory::Metadata), "Metadata");
        assert_eq!(format!("{}", LintCategory::Naming), "Tensor Naming");
        assert_eq!(format!("{}", LintCategory::Efficiency), "Efficiency");
    }

    #[test]
    fn test_lint_issue_display_without_suggestion() {
        let issue = LintIssue::new(LintLevel::Warn, LintCategory::Metadata, "Missing field");
        let display = format!("{}", issue);
        assert!(display.contains("[WARN]"));
        assert!(display.contains("Metadata"));
        assert!(display.contains("Missing field"));
        assert!(!display.contains("suggestion"));
    }

    #[test]
    fn test_lint_issue_display_with_suggestion() {
        let issue = LintIssue::naming_info("Use full name")
            .with_suggestion("Rename to 'encoder.weight'");
        let display = format!("{}", issue);
        assert!(display.contains("[INFO]"));
        assert!(display.contains("Tensor Naming"));
        assert!(display.contains("suggestion"));
        assert!(display.contains("encoder.weight"));
    }

    #[test]
    fn test_lint_issue_naming_warn() {
        let issue = LintIssue::naming_warn("Invalid naming pattern");
        assert_eq!(issue.level, LintLevel::Warn);
        assert_eq!(issue.category, LintCategory::Naming);
        assert_eq!(issue.message, "Invalid naming pattern");
    }

    #[test]
    fn test_lint_report_default() {
        let report = LintReport::default();
        assert!(report.issues.is_empty());
        assert!(report.by_category.is_empty());
        assert_eq!(report.info_count, 0);
        assert_eq!(report.warn_count, 0);
        assert_eq!(report.error_count, 0);
    }

    #[test]
    fn test_lint_report_add_error() {
        let mut report = LintReport::new();
        report.add_issue(LintIssue::new(
            LintLevel::Error,
            LintCategory::Naming,
            "Critical naming issue",
        ));
        assert_eq!(report.error_count, 1);
        assert!(!report.passed());
        assert!(!report.passed_strict());
    }

    #[test]
    fn test_lint_report_issues_in_nonexistent_category() {
        let report = LintReport::new();
        let issues = report.issues_in_category(LintCategory::Naming);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_model_lint_info_default() {
        let info = ModelLintInfo::default();
        assert!(!info.has_license);
        assert!(!info.has_model_card);
        assert!(!info.has_provenance);
        assert!(info.tensors.is_empty());
        assert!(!info.is_compressed);
    }

    #[test]
    fn test_tensor_lint_info_clone() {
        let tensor = TensorLintInfo {
            name: "test_tensor".to_string(),
            size_bytes: 1024,
            alignment: 64,
            is_compressed: true,
        };
        let cloned = tensor.clone();
        assert_eq!(cloned.name, "test_tensor");
        assert_eq!(cloned.size_bytes, 1024);
        assert_eq!(cloned.alignment, 64);
        assert!(cloned.is_compressed);
    }

    #[test]
    fn test_tensor_lint_info_debug() {
        let tensor = TensorLintInfo {
            name: "debug_test".to_string(),
            size_bytes: 512,
            alignment: 32,
            is_compressed: false,
        };
        let debug_str = format!("{:?}", tensor);
        assert!(debug_str.contains("debug_test"));
        assert!(debug_str.contains("512"));
    }

    #[test]
    fn test_model_lint_info_clone() {
        let info = ModelLintInfo {
            has_license: true,
            has_model_card: true,
            has_provenance: false,
            tensors: vec![],
            is_compressed: true,
        };
        let cloned = info.clone();
        assert!(cloned.has_license);
        assert!(cloned.has_model_card);
        assert!(!cloned.has_provenance);
        assert!(cloned.is_compressed);
    }

    #[test]
    fn test_model_lint_info_debug() {
        let info = ModelLintInfo::default();
        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("ModelLintInfo"));
    }

    #[test]
    fn test_lint_issue_clone() {
        let issue = LintIssue::metadata_warn("Test message").with_suggestion("Fix it");
        let cloned = issue.clone();
        assert_eq!(cloned.level, LintLevel::Warn);
        assert_eq!(cloned.category, LintCategory::Metadata);
        assert_eq!(cloned.message, "Test message");
        assert_eq!(cloned.suggestion, Some("Fix it".to_string()));
    }

    #[test]
    fn test_lint_issue_debug() {
        let issue = LintIssue::efficiency_info("Optimization hint");
        let debug_str = format!("{:?}", issue);
        assert!(debug_str.contains("Optimization hint"));
        assert!(debug_str.contains("Efficiency"));
    }

    #[test]
    fn test_lint_report_clone() {
        let mut report = LintReport::new();
        report.add_issue(LintIssue::metadata_warn("Warning 1"));
        report.add_issue(LintIssue::efficiency_info("Info 1"));

        let cloned = report.clone();
        assert_eq!(cloned.total_issues(), 2);
        assert_eq!(cloned.warn_count, 1);
        assert_eq!(cloned.info_count, 1);
    }

    #[test]
    fn test_lint_report_debug() {
        let report = LintReport::new();
        let debug_str = format!("{:?}", report);
        assert!(debug_str.contains("LintReport"));
    }

    #[test]
    fn test_is_abbreviated_already_full_form() {
        // If the full form is present, it's not abbreviated
        assert!(!is_abbreviated("encoder.weight", ".w", ".weight"));
        assert!(!is_abbreviated("layer.bias", ".b", ".bias"));
    }

    #[test]
    fn test_is_abbreviated_not_at_word_boundary() {
        // ".w" at word boundary
        assert!(is_abbreviated("encoder.w", ".w", ".weight"));
        // ".w" followed by separator
        assert!(is_abbreviated("encoder.w.test", ".w", ".weight"));
    }

    #[test]
    fn test_is_at_word_boundary_at_end() {
        // Position at end of string = word boundary
        assert!(is_at_word_boundary("test", 4));
    }

    #[test]
    fn test_is_at_word_boundary_with_separator() {
        // Position before separator = word boundary
        assert!(is_at_word_boundary("test.next", 4));
        assert!(is_at_word_boundary("test_next", 4));
        assert!(is_at_word_boundary("test-next", 4));
    }

    #[test]
    fn test_is_at_word_boundary_not_separator() {
        // Position before letter = not word boundary
        assert!(!is_at_word_boundary("testing", 4));
    }

    #[test]
    fn test_has_standard_numbering_patterns() {
        assert!(has_standard_numbering("encoder.layers.0.weight"));
        assert!(has_standard_numbering("conv1.weight"));
        assert!(has_standard_numbering("conv2.bias"));
        assert!(has_standard_numbering("fc1.weight"));
        assert!(has_standard_numbering("fc2.bias"));
    }

    #[test]
    fn test_has_standard_numbering_no_pattern() {
        assert!(!has_standard_numbering("encoder.weight"));
        assert!(!has_standard_numbering("layer.bias"));
    }

    #[test]
    fn test_has_unusual_separators() {
        assert!(has_unusual_separators("test__double"));
        assert!(has_unusual_separators("test--dash"));
        assert!(has_unusual_separators("test..dot"));
        assert!(!has_unusual_separators("test.normal"));
        assert!(!has_unusual_separators("test_normal"));
    }

    #[test]
    fn test_is_nonstandard_pattern_double_dash() {
        assert!(is_nonstandard_pattern("test--name"));
    }

    #[test]
    fn test_is_nonstandard_pattern_empty() {
        // Empty string is NOT flagged as nonstandard (special case - too short check has !name.is_empty())
        assert!(!is_nonstandard_pattern(""));
    }

    #[test]
    fn test_is_nonstandard_pattern_exact_boundary() {
        // Names of exactly 5 chars should not be "too short"
        assert!(!is_nonstandard_pattern("abcde"));
        // Names of 4 chars should be "too short"
        assert!(is_nonstandard_pattern("abcd"));
    }

    #[test]
    fn test_lint_with_all_abbreviation_patterns() {
        let info = ModelLintInfo {
            has_license: true,
            has_model_card: true,
            has_provenance: true,
            tensors: vec![
                TensorLintInfo {
                    name: "encoder.layer_w.data".to_string(), // _w -> .weight
                    size_bytes: 1000,
                    alignment: 64,
                    is_compressed: false,
                },
                TensorLintInfo {
                    name: "encoder.layer_b.data".to_string(), // _b -> .bias
                    size_bytes: 1000,
                    alignment: 64,
                    is_compressed: false,
                },
                TensorLintInfo {
                    name: "encoder.attn_qkv.data".to_string(), // attn_ -> self_attn.
                    size_bytes: 1000,
                    alignment: 64,
                    is_compressed: false,
                },
            ],
            ..Default::default()
        };

        let report = lint_model(&info);
        let naming_issues = report.issues_in_category(LintCategory::Naming);
        // Should flag abbreviation patterns
        assert!(!naming_issues.is_empty());
    }

    #[test]
    fn test_lint_category_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(LintCategory::Metadata);
        set.insert(LintCategory::Naming);
        set.insert(LintCategory::Efficiency);
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_lint_level_partial_ord() {
        // Info < Warn < Error
        assert!(LintLevel::Info < LintLevel::Warn);
        assert!(LintLevel::Warn < LintLevel::Error);
        assert!(LintLevel::Info < LintLevel::Error);

        // Equality
        assert!(LintLevel::Info == LintLevel::Info);
        assert!(LintLevel::Warn == LintLevel::Warn);
        assert!(LintLevel::Error == LintLevel::Error);
    }

    #[test]
    fn test_lint_category_equality() {
        assert_eq!(LintCategory::Metadata, LintCategory::Metadata);
        assert_ne!(LintCategory::Metadata, LintCategory::Naming);
        assert_ne!(LintCategory::Naming, LintCategory::Efficiency);
    }

    #[test]
    fn test_lint_level_copy() {
        let level = LintLevel::Warn;
        let copied = level;
        assert_eq!(copied, LintLevel::Warn);
    }

    #[test]
    fn test_lint_category_copy() {
        let cat = LintCategory::Naming;
        let copied = cat;
        assert_eq!(copied, LintCategory::Naming);
    }

    #[test]
    fn test_lint_zero_alignment() {
        // Zero alignment is a special case (skip check)
        let info = ModelLintInfo {
            has_license: true,
            has_model_card: true,
            has_provenance: true,
            tensors: vec![TensorLintInfo {
                name: "encoder.conv1.weight".to_string(),
                size_bytes: 1000,
                alignment: 0, // Zero alignment - skip check
                is_compressed: false,
            }],
            ..Default::default()
        };

        let report = lint_model(&info);
        let efficiency_issues = report.issues_in_category(LintCategory::Efficiency);
        let alignment_issues: Vec<_> = efficiency_issues
            .iter()
            .filter(|i| i.message.contains("aligned"))
            .collect();
        // Zero alignment should not be flagged as "unaligned"
        assert!(alignment_issues.is_empty());
    }
}
