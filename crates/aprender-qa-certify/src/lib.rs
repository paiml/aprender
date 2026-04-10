//! Model certification tools and README synchronization.
//!
//! This crate provides utilities for:
//! - Parsing model certification CSV data
//! - Generating markdown tables for README
//! - Synchronizing certification status with documentation

#![forbid(unsafe_code)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::doc_markdown
    )
)]

use chrono::{DateTime, Utc};
use std::fmt;
use thiserror::Error;

/// Errors that can occur during certification operations.
#[derive(Error, Debug)]
pub enum CertifyError {
    /// CSV parsing error.
    #[error("CSV parse error at line {line}: {message}")]
    CsvParse {
        /// Line number where error occurred.
        line: usize,
        /// Error message.
        message: String,
    },

    /// README marker not found.
    #[error("README marker not found: {0}")]
    MarkerNotFound(String),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for certification operations.
pub type Result<T> = std::result::Result<T, CertifyError>;

/// Certification status for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CertificationStatus {
    /// Model passed all tests with MQS >= 850.
    Certified,
    /// Model passed with MQS >= 700 but < 850.
    Provisional,
    /// Model failed tests or MQS < 700.
    Blocked,
    /// Model not yet tested.
    #[default]
    Pending,
}

impl CertificationStatus {
    /// Parse status from string representation.
    /// Returns `None` for unrecognized values (Jidoka: fail loud on bad data).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "CERTIFIED" => Some(Self::Certified),
            "PROVISIONAL" => Some(Self::Provisional),
            "BLOCKED" => Some(Self::Blocked),
            "PENDING" => Some(Self::Pending),
            _ => None,
        }
    }

    /// Get badge markdown for this status.
    #[must_use]
    pub const fn badge(&self) -> &'static str {
        match self {
            Self::Certified => "![certified](https://img.shields.io/badge/CERTIFIED-brightgreen)",
            Self::Provisional => "![provisional](https://img.shields.io/badge/PROVISIONAL-yellow)",
            Self::Blocked => "![blocked](https://img.shields.io/badge/BLOCKED-red)",
            Self::Pending => "![pending](https://img.shields.io/badge/PENDING-lightgray)",
        }
    }
}

impl fmt::Display for CertificationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Certified => write!(f, "CERTIFIED"),
            Self::Provisional => write!(f, "PROVISIONAL"),
            Self::Blocked => write!(f, "BLOCKED"),
            Self::Pending => write!(f, "PENDING"),
        }
    }
}

/// Size category for models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SizeCategory {
    /// < 1B parameters.
    Tiny,
    /// 1B - 5B parameters.
    #[default]
    Small,
    /// 5B - 10B parameters.
    Medium,
    /// 10B - 30B parameters.
    Large,
    /// > 30B parameters.
    XLarge,
}

impl SizeCategory {
    /// Parse size category from string representation.
    /// Returns `None` for unrecognized values (Jidoka: fail loud on bad data).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "tiny" => Some(Self::Tiny),
            "small" => Some(Self::Small),
            "medium" => Some(Self::Medium),
            "large" => Some(Self::Large),
            "xlarge" => Some(Self::XLarge),
            _ => None,
        }
    }
}

/// Model certification record.
///
/// Contains certification data for a single model including gateway status
/// and throughput measurements per format.
/// The four gateway bools (g1-g4) are required for the certification protocol.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
pub struct ModelCertification {
    /// `HuggingFace` model ID.
    pub model_id: String,
    /// Model family (e.g., qwen-coder, llama).
    pub family: String,
    /// Parameter count (e.g., "1.5B").
    pub parameters: String,
    /// Size category.
    pub size_category: SizeCategory,
    /// Certification status.
    pub status: CertificationStatus,
    /// Model Qualification Score (0-1000).
    pub mqs_score: u32,
    /// Letter grade.
    pub grade: String,
    /// Highest passing tier.
    pub certified_tier: String,
    /// Last certification timestamp.
    pub last_certified: Option<DateTime<Utc>>,
    /// Gateway 1 (load) status.
    pub g1: bool,
    /// Gateway 2 (inference) status.
    pub g2: bool,
    /// Gateway 3 (stability) status.
    pub g3: bool,
    /// Gateway 4 (quality) status.
    pub g4: bool,
    /// Throughput in tokens/sec for GGUF format (CPU).
    pub tps_gguf_cpu: Option<f64>,
    /// Throughput in tokens/sec for GGUF format (GPU).
    pub tps_gguf_gpu: Option<f64>,
    /// Throughput in tokens/sec for APR format (CPU).
    pub tps_apr_cpu: Option<f64>,
    /// Throughput in tokens/sec for APR format (GPU).
    pub tps_apr_gpu: Option<f64>,
    /// Throughput in tokens/sec for `SafeTensors` format (CPU).
    pub tps_st_cpu: Option<f64>,
    /// Throughput in tokens/sec for `SafeTensors` format (GPU).
    pub tps_st_gpu: Option<f64>,
    /// Provenance verified (PMAT-PROV-001).
    pub provenance_verified: bool,
    /// Kernel proof reference model (for dim-smoke tier).
    pub kernel_proof_ref: Option<String>,
}

impl ModelCertification {
    /// Get the short model name (without org prefix).
    #[must_use]
    pub fn short_name(&self) -> &str {
        self.model_id
            .split('/')
            .next_back()
            .unwrap_or(&self.model_id)
    }

    /// Get `HuggingFace` URL for this model.
    #[must_use]
    pub fn hf_url(&self) -> String {
        format!("https://huggingface.co/{}", self.model_id)
    }

    /// Get markdown link for this model.
    #[must_use]
    pub fn markdown_link(&self) -> String {
        format!("[{}]({})", self.short_name(), self.hf_url())
    }

    /// Get gateway symbol for display.
    #[must_use]
    pub const fn gateway_symbol(passed: bool, status: CertificationStatus) -> &'static str {
        if matches!(status, CertificationStatus::Pending) {
            "-"
        } else if passed {
            "\u{2713}" // checkmark
        } else {
            "\u{2717}" // x mark
        }
    }

    /// Parse numeric parameter count for sorting.
    #[must_use]
    pub fn param_count(&self) -> f64 {
        self.parameters
            .trim_end_matches('B')
            .parse::<f64>()
            .unwrap_or(0.0)
    }
}

/// Parse CSV content into model certifications.
///
/// # Errors
///
/// Required CSV columns — must all be present in the header.
const REQUIRED_CSV_COLUMNS: &[&str] = &[
    "model_id",
    "family",
    "parameters",
    "size_category",
    "status",
    "mqs_score",
    "grade",
    "certified_tier",
    "last_certified",
    "g1",
    "g2",
    "g3",
    "g4",
];

/// Parse a CSV certification table into a list of `ModelCertification` records.
///
/// # Errors
///
/// Returns `CertifyError::CsvParse` if the CSV header is missing required columns
/// or if any data row contains malformed field values.
#[allow(clippy::similar_names)]
pub fn parse_csv(content: &str) -> Result<Vec<ModelCertification>> {
    let mut models = Vec::new();
    let mut lines = content.lines().enumerate();

    // Parse header to build name→index map (prevents fragility from column reordering)
    let Some((_, header)) = lines.next() else {
        return Ok(models);
    };

    let header_fields = csv_split(header);
    let col: std::collections::HashMap<&str, usize> = header_fields
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    // Fail fast if any required columns are missing
    for &name in REQUIRED_CSV_COLUMNS {
        if !col.contains_key(name) {
            return Err(CertifyError::CsvParse {
                line: 1,
                message: format!("missing required column '{name}'"),
            });
        }
    }

    for (line_num, line) in lines {
        if line.trim().is_empty() {
            continue;
        }

        let fields = csv_split(line);
        let get = |name: &str| -> &str {
            col.get(name)
                .and_then(|&i| fields.get(i).map(String::as_str))
                .unwrap_or("")
        };
        let get_opt = |name: &str| -> Option<String> {
            col.get(name)
                .and_then(|&i| fields.get(i))
                .filter(|s| !s.is_empty())
                .cloned()
        };

        let last_certified = DateTime::parse_from_rfc3339(get("last_certified"))
            .ok()
            .map(|dt| dt.with_timezone(&Utc));

        models.push(ModelCertification {
            model_id: get("model_id").to_owned(),
            family: get("family").to_owned(),
            parameters: get("parameters").to_owned(),
            size_category: SizeCategory::parse(get("size_category")).ok_or_else(|| {
                CertifyError::CsvParse {
                    line: line_num + 1,
                    message: format!("invalid size_category: '{}'", get("size_category")),
                }
            })?,
            status: CertificationStatus::parse(get("status")).ok_or_else(|| {
                CertifyError::CsvParse {
                    line: line_num + 1,
                    message: format!("invalid status: '{}'", get("status")),
                }
            })?,
            mqs_score: get("mqs_score")
                .parse()
                .map_err(|_| CertifyError::CsvParse {
                    line: line_num + 1,
                    message: format!("invalid mqs_score: '{}'", get("mqs_score")),
                })?,
            grade: get("grade").to_owned(),
            certified_tier: get("certified_tier").to_owned(),
            last_certified,
            g1: get("g1").to_lowercase() == "true",
            g2: get("g2").to_lowercase() == "true",
            g3: get("g3").to_lowercase() == "true",
            g4: get("g4").to_lowercase() == "true",
            tps_gguf_cpu: get_opt("tps_gguf_cpu").and_then(|s| s.parse().ok()),
            tps_gguf_gpu: get_opt("tps_gguf_gpu").and_then(|s| s.parse().ok()),
            tps_apr_cpu: get_opt("tps_apr_cpu").and_then(|s| s.parse().ok()),
            tps_apr_gpu: get_opt("tps_apr_gpu").and_then(|s| s.parse().ok()),
            tps_st_cpu: get_opt("tps_st_cpu").and_then(|s| s.parse().ok()),
            tps_st_gpu: get_opt("tps_st_gpu").and_then(|s| s.parse().ok()),
            provenance_verified: get_opt("provenance_verified")
                .is_some_and(|s| s.to_lowercase() == "true"),
            kernel_proof_ref: get_opt("kernel_proof_ref"),
        });
    }

    Ok(models)
}

/// Generate certification summary statistics.
#[must_use]
pub fn generate_summary(models: &[ModelCertification], timestamp: &str) -> String {
    let total = models.len();
    let certified = models
        .iter()
        .filter(|m| matches!(m.status, CertificationStatus::Certified))
        .count();
    let provisional = models
        .iter()
        .filter(|m| matches!(m.status, CertificationStatus::Provisional))
        .count();
    let blocked = models
        .iter()
        .filter(|m| matches!(m.status, CertificationStatus::Blocked))
        .count();
    let pending = models
        .iter()
        .filter(|m| matches!(m.status, CertificationStatus::Pending))
        .count();

    format!(
        r"**Certification Summary** (updated: {timestamp})

| Status | Count |
|--------|-------|
| Certified | {certified}/{total} |
| Provisional | {provisional}/{total} |
| Blocked | {blocked}/{total} |
| Pending | {pending}/{total} |

**Priority Family:** Qwen Coder (see [Certified Testing Spec](docs/specifications/certified-testing.md))"
    )
}

/// Generate markdown table from model certifications.
#[must_use]
pub fn generate_table(models: &[ModelCertification]) -> String {
    let mut lines = Vec::new();

    // Header with tok/s columns (format × backend = 6 columns) + provenance
    lines.push(
        "| Model | Family | Size | Status | MQS | Grade | G1-4 | Prov | GGUF CPU | GGUF GPU | APR CPU | APR GPU | ST CPU | ST GPU |"
            .to_string(),
    );
    lines.push(
        "|-------|--------|------|--------|-----|-------|------|------|----------|----------|---------|---------|--------|--------|"
            .to_string(),
    );

    // Sort by family, then by parameter count
    let mut sorted: Vec<_> = models.iter().collect();
    sorted.sort_by(|a, b| {
        a.family.cmp(&b.family).then_with(|| {
            a.param_count()
                .partial_cmp(&b.param_count())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    for m in sorted {
        // Combine gateways into single column (all must pass)
        let gateways = if matches!(m.status, CertificationStatus::Pending) {
            "-".to_string()
        } else if m.g1 && m.g2 && m.g3 && m.g4 {
            "\u{2713}".to_string() // checkmark
        } else {
            "\u{2717}".to_string() // x mark
        };

        // Provenance status
        let prov = if matches!(m.status, CertificationStatus::Pending) {
            "-"
        } else if m.provenance_verified {
            "\u{2713}" // checkmark
        } else {
            "\u{2717}" // x mark
        };

        // Format tok/s values (6 columns)
        let fmt = |v: Option<f64>| v.map_or_else(|| "-".to_string(), |x| format!("{x:.1}"));

        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            m.markdown_link(),
            m.family,
            m.parameters,
            m.status.badge(),
            m.mqs_score,
            m.grade,
            gateways,
            prov,
            fmt(m.tps_gguf_cpu),
            fmt(m.tps_gguf_gpu),
            fmt(m.tps_apr_cpu),
            fmt(m.tps_apr_gpu),
            fmt(m.tps_st_cpu),
            fmt(m.tps_st_gpu),
        ));
    }

    lines.join("\n")
}

/// Markers for README table replacement.
pub const START_MARKER: &str = "<!-- CERTIFICATION_TABLE_START -->";
/// End marker for README table.
pub const END_MARKER: &str = "<!-- CERTIFICATION_TABLE_END -->";

/// Write certification records to CSV format.
///
/// Generates a CSV string with headers that can be written to models.csv.
#[must_use]
pub fn write_csv(models: &[ModelCertification]) -> String {
    let mut lines = Vec::new();

    // Header with 6 tps columns (format × backend) + provenance + kernel_proof_ref
    lines.push(
        "model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4,tps_gguf_cpu,tps_gguf_gpu,tps_apr_cpu,tps_apr_gpu,tps_st_cpu,tps_st_gpu,provenance_verified,kernel_proof_ref"
            .to_string(),
    );

    for m in models {
        let size_cat = match m.size_category {
            SizeCategory::Tiny => "tiny",
            SizeCategory::Small => "small",
            SizeCategory::Medium => "medium",
            SizeCategory::Large => "large",
            SizeCategory::XLarge => "xlarge",
        };
        let last_cert = m
            .last_certified
            .map_or_else(String::new, |dt| dt.to_rfc3339());

        // Format tps values (empty string for None)
        let fmt = |v: Option<f64>| v.map_or(String::new(), |x| format!("{x:.1}"));

        lines.push(format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            csv_quote(&m.model_id),
            csv_quote(&m.family),
            csv_quote(&m.parameters),
            size_cat,
            m.status,
            m.mqs_score,
            csv_quote(&m.grade),
            csv_quote(&m.certified_tier),
            last_cert,
            m.g1,
            m.g2,
            m.g3,
            m.g4,
            fmt(m.tps_gguf_cpu),
            fmt(m.tps_gguf_gpu),
            fmt(m.tps_apr_cpu),
            fmt(m.tps_apr_gpu),
            fmt(m.tps_st_cpu),
            fmt(m.tps_st_gpu),
            m.provenance_verified,
            csv_quote(m.kernel_proof_ref.as_deref().unwrap_or("")),
        ));
    }

    lines.join("\n") + "\n"
}

/// RFC 4180 CSV quoting: wrap in double-quotes if the field contains
/// a comma, double-quote, or newline. Internal double-quotes are escaped
/// by doubling them.
fn csv_quote(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

/// RFC 4180 CSV field splitting: handles double-quoted fields containing
/// commas and escaped double-quotes (doubled `""`).
fn csv_split(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    // Escaped double-quote
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            }
            '"' if !in_quotes && current.is_empty() => {
                in_quotes = true;
            }
            ',' if !in_quotes => {
                fields.push(std::mem::take(&mut current));
            }
            _ => {
                current.push(ch);
            }
        }
    }
    fields.push(current);
    fields
}

// Certification tier scoring logic
include!("lib_scoring.rs");

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
