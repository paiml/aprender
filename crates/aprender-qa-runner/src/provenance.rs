//! Provenance Validation (PMAT-PROV-001)
//!
//! Ensures all derived formats come from the same SafeTensors source.
//! Prevents the critical error of comparing models from different sources.
//!
//! See: docs/specifications/certified-testing.md Section 7.5

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read as IoRead};
use std::path::Path;

/// Source model provenance information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceProvenance {
    /// Format of source file (must be "safetensors" per spec 7.4)
    pub format: String,
    /// Relative path to source file
    pub path: String,
    /// SHA256 hash of source file
    pub sha256: String,
    /// HuggingFace repository ID (e.g., "Qwen/Qwen2.5-Coder-0.5B-Instruct")
    pub hf_repo: String,
    /// ISO 8601 timestamp of download
    pub downloaded_at: String,
}

/// Derived format provenance information
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DerivedProvenance {
    /// Format of derived file (e.g., "gguf", "apr")
    pub format: String,
    /// Relative path to derived file
    pub path: String,
    /// SHA256 hash of derived file
    pub sha256: String,
    /// Converter used (must be "apr-cli" per spec 7.5.2)
    pub converter: String,
    /// Version of converter
    pub converter_version: String,
    /// Quantization applied (null for unquantized)
    pub quantization: Option<String>,
    /// ISO 8601 timestamp of conversion
    pub created_at: String,
}

/// Complete provenance record for a model directory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// Source model information
    pub source: SourceProvenance,
    /// Derived formats
    pub derived: Vec<DerivedProvenance>,
}

/// Provenance validation errors (PMAT-PROV-001)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProvenanceError {
    /// PROV-001: Source hashes don't match across formats
    SourceMismatch {
        /// Expected source hash
        expected: String,
        /// Actual source hash found
        actual: String,
        /// Format with mismatched source
        format: String,
    },
    /// PROV-002: Derived file not created by apr-cli
    InvalidConverter {
        /// Format with invalid converter
        format: String,
        /// Converter that was used
        converter: String,
    },
    /// PROV-003: Source is not SafeTensors
    InvalidSourceFormat {
        /// Invalid source format found
        format: String,
    },
    /// PROV-004: Missing provenance file
    MissingProvenance {
        /// Path where provenance was expected
        path: String,
    },
    /// PROV-005: Quantization levels don't match
    QuantizationMismatch {
        /// First format in comparison
        format_a: String,
        /// Quantization of first format
        quant_a: Option<String>,
        /// Second format in comparison
        format_b: String,
        /// Quantization of second format
        quant_b: Option<String>,
    },
    /// PROV-006: File hash doesn't match recorded hash (integrity violation)
    HashMismatch {
        /// Path to file with mismatched hash
        path: String,
        /// Expected hash from provenance
        expected: String,
        /// Actual hash computed from file
        actual: String,
    },
    /// PROV-007: Referenced file does not exist (ghost file)
    FileMissing {
        /// Path to missing file
        path: String,
    },
    /// PROV-008: Duplicate derived format entry
    DuplicateDerived {
        /// Format that already exists
        format: String,
        /// Quantization level (if any)
        quantization: Option<String>,
    },
    /// PROV-009: Format not found in derived list
    FormatNotFound {
        /// Format that was requested but not found
        format: String,
    },
}

/// Display provenance errors with their error code prefix
impl std::fmt::Display for ProvenanceError {
    /// Format the error with its PROV-NNN code and descriptive message
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SourceMismatch {
                expected,
                actual,
                format,
            } => {
                write!(
                    f,
                    "PROV-001: Source hash mismatch for {format}: expected {expected}, got {actual}"
                )
            }
            Self::InvalidConverter { format, converter } => {
                write!(
                    f,
                    "PROV-002: Invalid converter for {format}: {converter} (must be apr-cli)"
                )
            }
            Self::InvalidSourceFormat { format } => {
                write!(
                    f,
                    "PROV-003: Invalid source format: {format} (must be safetensors)"
                )
            }
            Self::MissingProvenance { path } => {
                write!(f, "PROV-004: Missing provenance file: {path}")
            }
            Self::QuantizationMismatch {
                format_a,
                quant_a,
                format_b,
                quant_b,
            } => {
                write!(
                    f,
                    "PROV-005: Quantization mismatch: {format_a}={quant_a:?} vs {format_b}={quant_b:?}"
                )
            }
            Self::HashMismatch {
                path,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "PROV-006: Hash mismatch for {path}: expected {expected}, got {actual}"
                )
            }
            Self::FileMissing { path } => {
                write!(f, "PROV-007: Referenced file missing: {path}")
            }
            Self::DuplicateDerived {
                format,
                quantization,
            } => {
                write!(
                    f,
                    "PROV-008: Duplicate derived format: {format} (quantization: {quantization:?})"
                )
            }
            Self::FormatNotFound { format } => {
                write!(f, "PROV-009: Format not found in derived list: {format}")
            }
        }
    }
}

/// Enable ProvenanceError to be used as a standard error type
impl std::error::Error for ProvenanceError {}

/// Load provenance from a model directory
///
/// # Errors
///
/// Returns error if provenance file is missing or malformed.
pub fn load_provenance(model_dir: &Path) -> Result<Provenance> {
    let provenance_path = model_dir.join(".provenance.json");
    if !provenance_path.exists() {
        return Err(Error::Provenance(ProvenanceError::MissingProvenance {
            path: provenance_path.display().to_string(),
        }));
    }

    let content = std::fs::read_to_string(&provenance_path)?;
    let provenance: Provenance = serde_json::from_str(&content)?;
    Ok(provenance)
}

/// Validate provenance for certification (PMAT-PROV-001)
///
/// Checks all rules from spec section 7.5.2:
/// - PROV-001: All formats share same source hash
/// - PROV-002: All derived files use apr-cli
/// - PROV-003: Source is safetensors
/// - PROV-004: Provenance file exists (checked by load_provenance)
/// - PROV-005: Quantization matches for comparisons
///
/// # Errors
///
/// Returns the first validation error encountered.
pub fn validate_provenance(provenance: &Provenance) -> std::result::Result<(), ProvenanceError> {
    // PROV-003: Source must be safetensors
    if provenance.source.format != "safetensors" {
        return Err(ProvenanceError::InvalidSourceFormat {
            format: provenance.source.format.clone(),
        });
    }

    // PROV-002: All derived files must use apr-cli
    for derived in &provenance.derived {
        if derived.converter != "apr-cli" {
            return Err(ProvenanceError::InvalidConverter {
                format: derived.format.clone(),
                converter: derived.converter.clone(),
            });
        }
    }

    Ok(())
}

/// Validate that two formats can be compared (same source, same quantization)
///
/// # Errors
///
/// Returns error if formats don't exist or have different quantization.
pub fn validate_comparison(
    provenance: &Provenance,
    format_a: &str,
    format_b: &str,
) -> std::result::Result<(), ProvenanceError> {
    // PROV-009: Both formats must exist in derived list
    let derived_a = provenance
        .derived
        .iter()
        .find(|d| d.format == format_a)
        .ok_or_else(|| ProvenanceError::FormatNotFound {
            format: format_a.to_string(),
        })?;

    let derived_b = provenance
        .derived
        .iter()
        .find(|d| d.format == format_b)
        .ok_or_else(|| ProvenanceError::FormatNotFound {
            format: format_b.to_string(),
        })?;

    // PROV-005: Quantization must match
    if derived_a.quantization != derived_b.quantization {
        return Err(ProvenanceError::QuantizationMismatch {
            format_a: format_a.to_string(),
            quant_a: derived_a.quantization.clone(),
            format_b: format_b.to_string(),
            quant_b: derived_b.quantization.clone(),
        });
    }

    Ok(())
}

/// Verify provenance integrity by re-hashing all files (PROV-006, PROV-007)
///
/// This function performs deep verification:
/// - Checks that all referenced files exist
/// - Re-computes SHA256 hashes and compares to recorded values
///
/// # Arguments
///
/// * `provenance` - The provenance record to verify
/// * `model_dir` - Base directory containing model files
///
/// # Errors
///
/// Returns error if any file is missing or hash doesn't match.
pub fn verify_provenance_integrity(
    provenance: &Provenance,
    model_dir: &Path,
) -> std::result::Result<(), ProvenanceError> {
    // Verify source file
    let source_path = model_dir.join(&provenance.source.path);
    if !source_path.exists() {
        return Err(ProvenanceError::FileMissing {
            path: provenance.source.path.clone(),
        });
    }

    let source_hash = compute_sha256(&source_path).map_err(|_| ProvenanceError::FileMissing {
        path: provenance.source.path.clone(),
    })?;

    if source_hash != provenance.source.sha256 {
        return Err(ProvenanceError::HashMismatch {
            path: provenance.source.path.clone(),
            expected: provenance.source.sha256.clone(),
            actual: source_hash,
        });
    }

    // Verify all derived files
    for derived in &provenance.derived {
        let derived_path = model_dir.join(&derived.path);
        if !derived_path.exists() {
            return Err(ProvenanceError::FileMissing {
                path: derived.path.clone(),
            });
        }

        let derived_hash =
            compute_sha256(&derived_path).map_err(|_| ProvenanceError::FileMissing {
                path: derived.path.clone(),
            })?;

        if derived_hash != derived.sha256 {
            return Err(ProvenanceError::HashMismatch {
                path: derived.path.clone(),
                expected: derived.sha256.clone(),
                actual: derived_hash,
            });
        }
    }

    Ok(())
}

/// Quick check that all referenced files exist (PROV-007)
///
/// Lighter than `verify_provenance_integrity` - only checks existence, not hashes.
///
/// # Errors
///
/// Returns error if any referenced file is missing.
pub fn verify_files_exist(
    provenance: &Provenance,
    model_dir: &Path,
) -> std::result::Result<(), ProvenanceError> {
    // Check source file
    let source_path = model_dir.join(&provenance.source.path);
    if !source_path.exists() {
        return Err(ProvenanceError::FileMissing {
            path: provenance.source.path.clone(),
        });
    }

    // Check all derived files
    for derived in &provenance.derived {
        let derived_path = model_dir.join(&derived.path);
        if !derived_path.exists() {
            return Err(ProvenanceError::FileMissing {
                path: derived.path.clone(),
            });
        }
    }

    Ok(())
}

// ============================================================================
// Provenance Generation (PMAT-PROV-001)
// ============================================================================

/// Compute SHA256 hash of a file
///
/// # Errors
///
/// Returns error if file cannot be read.
pub fn compute_sha256(path: &Path) -> Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();

    let mut buffer = [0u8; 8192];
    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let result = hasher.finalize();
    Ok(format!("{result:x}"))
}

/// Create initial provenance for a SafeTensors source file
///
/// # Errors
///
/// Returns error if file cannot be read or hashed.
pub fn create_source_provenance(safetensors_path: &Path, hf_repo: &str) -> Result<Provenance> {
    let sha256 = compute_sha256(safetensors_path)?;
    let now = chrono::Utc::now().to_rfc3339();

    Ok(Provenance {
        source: SourceProvenance {
            format: "safetensors".to_string(),
            path: safetensors_path.file_name().map_or_else(
                || safetensors_path.display().to_string(),
                |n| n.to_string_lossy().to_string(),
            ),
            sha256,
            hf_repo: hf_repo.to_string(),
            downloaded_at: now,
        },
        derived: Vec::new(),
    })
}

/// Add a derived format to provenance
///
/// Checks for duplicate entries (same format + quantization) before adding.
///
/// # Errors
///
/// Returns error if:
/// - Derived file cannot be read or hashed
/// - Duplicate format+quantization already exists (PROV-008)
pub fn add_derived(
    provenance: &mut Provenance,
    format: &str,
    derived_path: &Path,
    quantization: Option<&str>,
    converter_version: &str,
) -> Result<()> {
    // PROV-008: Check for duplicate format+quantization
    let exists = provenance
        .derived
        .iter()
        .any(|d| d.format == format && d.quantization.as_deref() == quantization);

    if exists {
        return Err(Error::Provenance(ProvenanceError::DuplicateDerived {
            format: format.to_string(),
            quantization: quantization.map(String::from),
        }));
    }

    let sha256 = compute_sha256(derived_path)?;
    let now = chrono::Utc::now().to_rfc3339();

    provenance.derived.push(DerivedProvenance {
        format: format.to_string(),
        path: derived_path.file_name().map_or_else(
            || derived_path.display().to_string(),
            |n| n.to_string_lossy().to_string(),
        ),
        sha256,
        converter: "apr-cli".to_string(),
        converter_version: converter_version.to_string(),
        quantization: quantization.map(String::from),
        created_at: now,
    });

    Ok(())
}

include!("provenance_utilities.rs");
