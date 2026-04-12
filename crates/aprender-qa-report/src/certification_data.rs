//! Certification Data for Oracle Integration (PMAT-260)
//!
//! This module provides the data structures and CSV parsing for the certification
//! lookup table consumed by aprender's `apr oracle` CLI command.
//!
//! # Theoretical Foundation
//!
//! This implementation follows:
//! - **Toyota Production System (Ohno, 1988)**: Jidoka - automatic stop on malformed data
//! - **Poka-Yoke (Shingo, 1986)**: Schema validation prevents invalid certification states
//! - **Popperian Falsification (Popper, 1959)**: Round-trip integrity tests verify correctness
//!
//! # CSV Schema
//!
//! The `models.csv` file uses this schema:
//! ```csv
//! model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4,tps_gguf_cpu,tps_gguf_gpu,tps_apr_cpu,tps_apr_gpu,tps_st_cpu,tps_st_gpu,provenance_verified
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::{Error, Result};

/// Certification status for a model.
///
/// Status definitions follow the specification:
/// - **CERTIFIED**: MQS >= 800, all gateway gates passed, tier requirements met
/// - **BLOCKED**: MQS < 800 or gateway gate failure, cannot be used in production
/// - **PENDING**: No certification run completed, awaiting testing
/// - **UNTESTED**: Legacy status for models never tested
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ModelStatus {
    /// MQS >= 800, all gateways passed
    Certified,
    /// MQS < 800 or gateway failure
    Blocked,
    /// Awaiting certification run
    #[default]
    Pending,
    /// Never tested (legacy)
    Untested,
}

/// Display model status as an uppercase string
impl std::fmt::Display for ModelStatus {
    /// Format the status variant as its canonical uppercase name
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Certified => write!(f, "CERTIFIED"),
            Self::Blocked => write!(f, "BLOCKED"),
            Self::Pending => write!(f, "PENDING"),
            Self::Untested => write!(f, "UNTESTED"),
        }
    }
}

/// Parse model status from a case-insensitive string
impl std::str::FromStr for ModelStatus {
    type Err = Error;

    /// Parse a status string like "CERTIFIED", "BLOCKED", etc
    fn from_str(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "CERTIFIED" => Ok(Self::Certified),
            "BLOCKED" => Ok(Self::Blocked),
            "PENDING" => Ok(Self::Pending),
            "UNTESTED" => Ok(Self::Untested),
            other => Err(Error::Validation(format!("Invalid status: {other}"))),
        }
    }
}

/// Size category for resource-aware scheduling.
///
/// Matches the `SizeCategory` enum in `apr-qa-runner::playbook`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SizeCategory {
    /// < 1B params, 4 workers
    #[default]
    Tiny,
    /// 1-2B params, 4 workers
    Small,
    /// 2-7B params, 2 workers
    Medium,
    /// 7-14B params, 1 worker
    Large,
    /// 14-32B params, 1 worker
    Xlarge,
    /// > 32B params, 1 worker
    Huge,
}

/// Display size category as a lowercase string
impl std::fmt::Display for SizeCategory {
    /// Format the size category variant as its lowercase name
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tiny => write!(f, "tiny"),
            Self::Small => write!(f, "small"),
            Self::Medium => write!(f, "medium"),
            Self::Large => write!(f, "large"),
            Self::Xlarge => write!(f, "xlarge"),
            Self::Huge => write!(f, "huge"),
        }
    }
}

/// Parse size category from a case-insensitive string
impl std::str::FromStr for SizeCategory {
    type Err = Error;

    /// Parse a size string like "tiny", "small", "medium", etc
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "tiny" => Ok(Self::Tiny),
            "small" => Ok(Self::Small),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            "xlarge" => Ok(Self::Xlarge),
            "huge" => Ok(Self::Huge),
            other => Err(Error::Validation(format!("Invalid size category: {other}"))),
        }
    }
}

/// A single row from the certification lookup table (models.csv).
///
/// This struct represents the complete certification state for a model variant,
/// including MQS score, gateway results, and performance metrics.
///
/// The boolean fields (g1-g4, provenance_verified) match the CSV schema
/// and represent gateway pass/fail state directly from test results.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct CertificationRow {
    /// HuggingFace model ID (e.g., "Qwen/Qwen2.5-Coder-0.5B-Instruct")
    pub model_id: String,

    /// Model family (e.g., "qwen-coder", "llama", "mistral")
    pub family: String,

    /// Parameter count string (e.g., "0.5B", "1.5B", "7B")
    pub parameters: String,

    /// Size category for resource scheduling
    pub size_category: SizeCategory,

    /// Certification status
    pub status: ModelStatus,

    /// Model Qualification Score (0-1000)
    pub mqs_score: u32,

    /// Letter grade (A, B, C, D, F, or "-" for ungraded)
    pub grade: String,

    /// Highest certified tier (quick, smoke, mvp, full, or "none")
    pub certified_tier: String,

    /// Last certification timestamp (ISO8601)
    pub last_certified: DateTime<Utc>,

    // Gateway results (G1-G4)
    /// G1: Model loads successfully
    pub g1: bool,
    /// G2: Basic inference works
    pub g2: bool,
    /// G3: No crashes or panics
    pub g3: bool,
    /// G4: Output is not garbage
    pub g4: bool,

    // Performance metrics (tokens per second)
    /// GGUF format, CPU backend
    pub tps_gguf_cpu: Option<f64>,
    /// GGUF format, GPU backend
    pub tps_gguf_gpu: Option<f64>,
    /// APR format, CPU backend
    pub tps_apr_cpu: Option<f64>,
    /// APR format, GPU backend
    pub tps_apr_gpu: Option<f64>,
    /// SafeTensors format, CPU backend
    pub tps_st_cpu: Option<f64>,
    /// SafeTensors format, GPU backend
    pub tps_st_gpu: Option<f64>,

    /// Whether model provenance has been verified
    pub provenance_verified: bool,

    /// Kernel proof reference model for dim-smoke tier (optional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kernel_proof_ref: Option<String>,
}

/// Provide default values for a new uncertified model row
impl Default for CertificationRow {
    /// Create a default row with pending status and zero scores
    fn default() -> Self {
        Self {
            model_id: String::new(),
            family: String::new(),
            parameters: String::new(),
            size_category: SizeCategory::default(),
            status: ModelStatus::default(),
            mqs_score: 0,
            grade: "-".to_string(),
            certified_tier: "none".to_string(),
            last_certified: Utc::now(),
            g1: false,
            g2: false,
            g3: false,
            g4: false,
            tps_gguf_cpu: None,
            tps_gguf_gpu: None,
            tps_apr_cpu: None,
            tps_apr_gpu: None,
            tps_st_cpu: None,
            tps_st_gpu: None,
            provenance_verified: false,
            kernel_proof_ref: None,
        }
    }
}

impl CertificationRow {
    /// Create a new certification row for a model.
    #[must_use]
    pub fn new(model_id: impl Into<String>, family: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            family: family.into(),
            ..Default::default()
        }
    }

    /// Check if all gateway checks passed.
    #[must_use]
    pub const fn all_gateways_passed(&self) -> bool {
        self.g1 && self.g2 && self.g3 && self.g4
    }

    /// Derive status from MQS score and gateway results.
    ///
    /// Follows the specification:
    /// - CERTIFIED: MQS >= 800 AND all gateways passed
    /// - BLOCKED: otherwise
    #[must_use]
    pub fn derive_status(&self) -> ModelStatus {
        if self.mqs_score >= 800 && self.all_gateways_passed() {
            ModelStatus::Certified
        } else if self.mqs_score == 0 && !self.g1 {
            ModelStatus::Pending
        } else {
            ModelStatus::Blocked
        }
    }

    /// Derive grade from MQS score.
    ///
    /// Grade thresholds (canonical — aligned with apr-qa-certify::grade_from_score):
    /// - A+: 950-1000
    /// - A: 900-949
    /// - B+: 850-899
    /// - B: 800-849  (CERTIFIED threshold)
    /// - C: 700-799
    /// - F: 0-699
    #[must_use]
    pub fn derive_grade(&self) -> String {
        match self.mqs_score {
            950.. => "A+".to_string(),
            900..=949 => "A".to_string(),
            850..=899 => "B+".to_string(),
            800..=849 => "B".to_string(),
            700..=799 => "C".to_string(),
            _ => "F".to_string(),
        }
    }
}

/// Read certification rows from a CSV file.
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be read
/// - The CSV is malformed
/// - A row contains invalid data
pub fn read_models_csv<P: AsRef<Path>>(path: P) -> Result<Vec<CertificationRow>> {
    let file = std::fs::File::open(path.as_ref()).map_err(|e| {
        Error::Io(format!(
            "Failed to open models.csv at {}: {e}",
            path.as_ref().display()
        ))
    })?;

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(file);

    let mut rows = Vec::new();

    for (idx, result) in reader.records().enumerate() {
        let record =
            result.map_err(|e| Error::Validation(format!("CSV parse error at row {idx}: {e}")))?;

        let row = parse_csv_record(&record, idx)?;
        rows.push(row);
    }

    Ok(rows)
}

/// Parse a single CSV record into a CertificationRow.
fn parse_csv_record(record: &csv::StringRecord, idx: usize) -> Result<CertificationRow> {
    // Helper for getting field with context
    let get_field = |i: usize, name: &str| -> Result<&str> {
        record
            .get(i)
            .ok_or_else(|| Error::Validation(format!("Missing field '{name}' at row {idx}")))
    };

    let model_id = get_field(0, "model_id")?.to_string();
    let family = get_field(1, "family")?.to_string();
    let parameters = get_field(2, "parameters")?.to_string();
    let size_category: SizeCategory = get_field(3, "size_category")?.parse()?;
    let status: ModelStatus = get_field(4, "status")?.parse()?;
    let mqs_score: u32 = get_field(5, "mqs_score")?
        .parse()
        .map_err(|e| Error::Validation(format!("Invalid mqs_score at row {idx}: {e}")))?;
    let grade = get_field(6, "grade")?.to_string();
    let certified_tier = get_field(7, "certified_tier")?.to_string();

    let last_certified = get_field(8, "last_certified")?;
    let last_certified: DateTime<Utc> = DateTime::parse_from_rfc3339(last_certified)
        .map_err(|e| Error::Validation(format!("Invalid timestamp at row {idx}: {e}")))?
        .with_timezone(&Utc);

    let parse_bool = |i: usize, name: &str| -> Result<bool> {
        match get_field(i, name)?.to_lowercase().as_str() {
            "true" | "1" | "yes" => Ok(true),
            "false" | "0" | "no" | "" => Ok(false),
            other => Err(Error::Validation(format!(
                "Invalid boolean '{other}' for {name} at row {idx}"
            ))),
        }
    };

    let parse_optional_f64 = |i: usize| -> Option<f64> {
        record.get(i).and_then(|s| {
            let s = s.trim();
            if s.is_empty() {
                None
            } else {
                s.parse().ok()
            }
        })
    };

    Ok(CertificationRow {
        model_id,
        family,
        parameters,
        size_category,
        status,
        mqs_score,
        grade,
        certified_tier,
        last_certified,
        g1: parse_bool(9, "g1")?,
        g2: parse_bool(10, "g2")?,
        g3: parse_bool(11, "g3")?,
        g4: parse_bool(12, "g4")?,
        tps_gguf_cpu: parse_optional_f64(13),
        tps_gguf_gpu: parse_optional_f64(14),
        tps_apr_cpu: parse_optional_f64(15),
        tps_apr_gpu: parse_optional_f64(16),
        tps_st_cpu: parse_optional_f64(17),
        tps_st_gpu: parse_optional_f64(18),
        provenance_verified: parse_bool(19, "provenance_verified")?,
        kernel_proof_ref: record.get(20).and_then(|s| {
            let s = s.trim();
            if s.is_empty() {
                None
            } else {
                Some(s.to_string())
            }
        }),
    })
}

/// Write certification rows to a CSV file.
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn write_models_csv<P: AsRef<Path>>(rows: &[CertificationRow], path: P) -> Result<()> {
    let file = std::fs::File::create(path.as_ref()).map_err(|e| {
        Error::Io(format!(
            "Failed to create models.csv at {}: {e}",
            path.as_ref().display()
        ))
    })?;

    let mut writer = csv::Writer::from_writer(file);

    // Write header
    writer
        .write_record([
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
            "tps_gguf_cpu",
            "tps_gguf_gpu",
            "tps_apr_cpu",
            "tps_apr_gpu",
            "tps_st_cpu",
            "tps_st_gpu",
            "provenance_verified",
            "kernel_proof_ref",
        ])
        .map_err(|e| Error::Io(format!("Failed to write CSV header: {e}")))?;

    // Write rows
    for row in rows {
        let format_optional_f64 =
            |opt: Option<f64>| -> String { opt.map_or_else(String::new, |v| format!("{v:.1}")) };

        let kernel_proof = row.kernel_proof_ref.as_deref().unwrap_or("").to_string();

        writer
            .write_record([
                &row.model_id,
                &row.family,
                &row.parameters,
                &row.size_category.to_string(),
                &row.status.to_string(),
                &row.mqs_score.to_string(),
                &row.grade,
                &row.certified_tier,
                &row.last_certified.to_rfc3339(),
                &row.g1.to_string(),
                &row.g2.to_string(),
                &row.g3.to_string(),
                &row.g4.to_string(),
                &format_optional_f64(row.tps_gguf_cpu),
                &format_optional_f64(row.tps_gguf_gpu),
                &format_optional_f64(row.tps_apr_cpu),
                &format_optional_f64(row.tps_apr_gpu),
                &format_optional_f64(row.tps_st_cpu),
                &format_optional_f64(row.tps_st_gpu),
                &row.provenance_verified.to_string(),
                &kernel_proof,
            ])
            .map_err(|e| Error::Io(format!("Failed to write CSV row: {e}")))?;
    }

    writer
        .flush()
        .map_err(|e| Error::Io(format!("Failed to flush CSV writer: {e}")))?;

    Ok(())
}

/// Lookup a certification row by model ID.
///
/// Returns `None` if the model is not found.
#[must_use]
pub fn lookup_model<'a>(
    rows: &'a [CertificationRow],
    model_id: &str,
) -> Option<&'a CertificationRow> {
    rows.iter().find(|r| r.model_id == model_id)
}

/// Lookup certification rows by family.
#[must_use]
pub fn lookup_family<'a>(rows: &'a [CertificationRow], family: &str) -> Vec<&'a CertificationRow> {
    rows.iter().filter(|r| r.family == family).collect()
}

#[cfg(test)]
#[path = "certification_data_tests.rs"]
mod certification_data_tests;
