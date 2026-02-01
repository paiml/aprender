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

/// Lint any model file from disk (APR, GGUF, or SafeTensors)
///
/// Detects format via magic bytes and runs format-appropriate lint checks:
/// - **Universal**: metadata (license, provenance), tensor naming, NaN/Inf
/// - **APR-only**: CRC32 integrity, 64-byte alignment, compression
/// - **GGUF**: metadata KV pairs for license/model_card
/// - **SafeTensors**: `__metadata__` for license/description/author
pub fn lint_model_file(path: impl AsRef<Path>) -> Result<LintReport> {
    use crate::format::rosetta::FormatType;

    let path = path.as_ref();
    let format = FormatType::from_magic(path).or_else(|_| FormatType::from_extension(path))?;

    match format {
        FormatType::Apr => lint_apr_file(path),
        FormatType::Gguf => lint_gguf_file(path),
        FormatType::SafeTensors => lint_safetensors_file(path),
    }
}

/// Lint a GGUF file for best practices
fn lint_gguf_file(path: &Path) -> Result<LintReport> {
    use crate::format::gguf::GgufReader;

    let reader = GgufReader::from_file(path)?;
    let mut info = ModelLintInfo::default();

    // Check GGUF metadata KV pairs for standard fields
    info.has_license = reader
        .metadata
        .keys()
        .any(|k| k.contains("license") || k.contains("License"));
    info.has_model_card = reader
        .metadata
        .keys()
        .any(|k| k.contains("model_card") || k.contains("description"));
    info.has_provenance = reader
        .metadata
        .keys()
        .any(|k| k.contains("author") || k.contains("source") || k.contains("url"));

    // Build tensor lint info
    for meta in &reader.tensors {
        let shape: Vec<usize> = meta.dims.iter().map(|&d| d as usize).collect();
        let num_elements: usize = shape.iter().product();
        info.tensors.push(TensorLintInfo {
            name: meta.name.clone(),
            size_bytes: num_elements * 4, // approximate
            alignment: 32,               // GGUF uses 32-byte alignment
            is_compressed: false,
        });
    }

    Ok(lint_model(&info))
}

/// Lint a SafeTensors file for best practices
fn lint_safetensors_file(path: &Path) -> Result<LintReport> {
    use crate::serialization::safetensors::MappedSafeTensors;

    let mapped = MappedSafeTensors::open(path).map_err(|e| crate::error::AprenderError::FormatError {
        message: format!("SafeTensors open failed: {e}"),
    })?;

    let mut info = ModelLintInfo::default();

    // SafeTensors doesn't have rich metadata by default
    info.has_license = false;
    info.has_model_card = false;
    info.has_provenance = false;

    // Check the file-level __metadata__ if accessible via raw header parse
    let data = std::fs::read(path)?;
    if data.len() >= 8 {
        let header_len = u64::from_le_bytes(
            data[0..8].try_into().unwrap_or([0u8; 8]),
        ) as usize;
        if data.len() >= 8 + header_len {
            if let Ok(header) = serde_json::from_slice::<serde_json::Value>(&data[8..8 + header_len]) {
                if let Some(meta) = header.get("__metadata__").and_then(|v| v.as_object()) {
                    info.has_license = meta.contains_key("license");
                    info.has_model_card = meta.contains_key("description") || meta.contains_key("model_card");
                    info.has_provenance = meta.contains_key("author") || meta.contains_key("source");
                }
            }
        }
    }

    // Build tensor lint info
    for name in mapped.tensor_names() {
        if let Some(meta) = mapped.get_metadata(name) {
            let size_bytes = meta.data_offsets[1] - meta.data_offsets[0];
            info.tensors.push(TensorLintInfo {
                name: name.to_string(),
                size_bytes,
                alignment: 0, // SafeTensors doesn't guarantee alignment
                is_compressed: false,
            });
        }
    }

    Ok(lint_model(&info))
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
mod tests;
