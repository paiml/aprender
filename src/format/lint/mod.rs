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
    /// Layout contract checks (LAYOUT-CONTRACT-001)
    /// See: contracts/tensor-layout-v1.yaml
    Layout,
}

impl LintCategory {
    /// Get display name
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Metadata => "Metadata",
            Self::Naming => "Tensor Naming",
            Self::Efficiency => "Efficiency",
            Self::Layout => "Layout Contract",
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

    /// Create layout warning (LAYOUT-CONTRACT-001)
    #[must_use]
    pub fn layout_warn(message: impl Into<String>) -> Self {
        Self::new(LintLevel::Warn, LintCategory::Layout, message)
    }

    /// Create layout error (critical contract violation)
    #[must_use]
    pub fn layout_error(message: impl Into<String>) -> Self {
        Self::new(LintLevel::Error, LintCategory::Layout, message)
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
    /// Model vocabulary size (for layout contract validation)
    pub vocab_size: Option<usize>,
    /// Model hidden dimension (for layout contract validation)
    pub hidden_dim: Option<usize>,
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
    /// Tensor shape (dimensions)
    pub shape: Vec<usize>,
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

    // Layout contract checks (ERROR level for critical violations)
    // See: contracts/tensor-layout-v1.yaml
    check_layout_contract(&mut report, info);

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
/// BUG-LINT-001 FIX: Include GGUF naming patterns (blk.N.) alongside HF patterns (layers.)
const STANDARD_NUMERIC_PATTERNS: &[&str] = &[
    "layers.", // HuggingFace style: model.layers.0.self_attn
    "blk.",    // GGUF style: blk.0.attn_k, blk.0.ffn_gate
    "conv1", "conv2", "fc1", "fc2", // CNN/MLP patterns
];

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

/// Check layout contract compliance (LAYOUT-CONTRACT-001)
///
/// Validates tensor shapes against the authoritative tensor layout contract.
/// See: `contracts/tensor-layout-v1.yaml` for the full specification.
fn check_layout_contract(report: &mut LintReport, info: &ModelLintInfo) {
    use crate::format::layout_contract::contract;

    // Only validate if we have model dimensions
    let (vocab_size, hidden_dim) = match (info.vocab_size, info.hidden_dim) {
        (Some(v), Some(h)) => (v, h),
        _ => return, // Skip contract checks without model config
    };

    let layout = contract();
    for tensor in &info.tensors {
        let Some(tc) = layout.get_apr_contract(&tensor.name) else {
            continue;
        };

        validate_critical_tensor_shape(report, &layout, tc, tensor, vocab_size, hidden_dim);
        validate_transpose_dimensions(report, tc, tensor, vocab_size);
    }
}

include!("mod_part_02.rs");
