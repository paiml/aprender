//! Unified Model Loader
//!
//! Per spec §3.2 and §5: Unified loading for APR, GGUF, and SafeTensors formats.
//!
//! ## Jidoka (Built-in Quality)
//!
//! - Format auto-detection from magic bytes
//! - CRC32 verification for APR format
//! - Header validation for SafeTensors
//! - Graceful error handling with detailed messages
//!
//! ## APR Format Support (First-class)
//!
//! Per spec §3.1: APR is the primary format for classical ML models from aprender.
//! Supports all 18 model types:
//!
//! | Type | Description |
//! |------|-------------|
//! | `LinearRegression` | OLS/Ridge/Lasso |
//! | `LogisticRegression` | Binary/Multinomial classification |
//! | `DecisionTree` | CART/ID3 |
//! | `RandomForest` | Bagging ensemble |
//! | `GradientBoosting` | Boosting ensemble |
//! | `KMeans` | Lloyd's clustering |
//! | `PCA` | Dimensionality reduction |
//! | `NaiveBayes` | Gaussian NB |
//! | `KNN` | k-Nearest Neighbors |
//! | `SVM` | Linear SVM |
//! | `NgramLM` | N-gram language model |
//! | `TFIDF` | TF-IDF vectorizer |
//! | `CountVectorizer` | Count vectorizer |
//! | `NeuralSequential` | Feed-forward NN |
//! | `NeuralCustom` | Custom architecture |
//! | `ContentRecommender` | Content-based rec |
//! | `MixtureOfExperts` | Sparse/dense MoE |
//! | `Custom` | User-defined |
//!
//! ## GGUF Support (Backwards Compatible)
//!
//! Per spec §3.3: GGUF for LLM inference with llama.cpp compatibility.
//!
//! ## SafeTensors Support (Backwards Compatible)
//!
//! Per spec §3.4: SafeTensors for HuggingFace model weights.

use std::path::Path;

use crate::format::{detect_and_verify_format, detect_format, FormatError, ModelFormat};

/// Model loading errors
#[derive(Debug, Clone)]
pub enum LoadError {
    /// Format detection failed
    FormatError(FormatError),
    /// File I/O error
    IoError(String),
    /// Model parsing error
    ParseError(String),
    /// Unsupported model type for serving
    UnsupportedType(String),
    /// CRC32 checksum mismatch (APR)
    IntegrityError(String),
    /// Model type mismatch (requested vs detected)
    TypeMismatch {
        /// Expected model type
        expected: String,
        /// Actual model type in file
        actual: String,
    },
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FormatError(e) => write!(f, "Format detection error: {e}"),
            Self::IoError(msg) => write!(f, "I/O error: {msg}"),
            Self::ParseError(msg) => write!(f, "Parse error: {msg}"),
            Self::UnsupportedType(t) => write!(f, "Unsupported model type: {t}"),
            Self::IntegrityError(msg) => write!(f, "Integrity check failed: {msg}"),
            Self::TypeMismatch { expected, actual } => {
                write!(f, "Model type mismatch: expected {expected}, got {actual}")
            },
        }
    }
}

impl std::error::Error for LoadError {}

impl From<FormatError> for LoadError {
    fn from(e: FormatError) -> Self {
        Self::FormatError(e)
    }
}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e.to_string())
    }
}

/// Model metadata extracted during loading
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Detected format
    pub format: ModelFormat,
    /// Model type (if detected)
    pub model_type: Option<String>,
    /// Model version
    pub version: Option<String>,
    /// Input dimensions (for validation)
    pub input_dim: Option<usize>,
    /// Output dimensions
    pub output_dim: Option<usize>,
    /// File size in bytes
    pub file_size: u64,
}

impl ModelMetadata {
    /// Create new metadata with format only
    #[must_use]
    pub fn new(format: ModelFormat) -> Self {
        Self {
            format,
            model_type: None,
            version: None,
            input_dim: None,
            output_dim: None,
            file_size: 0,
        }
    }

    /// Set model type
    #[must_use]
    pub fn with_model_type(mut self, model_type: impl Into<String>) -> Self {
        self.model_type = Some(model_type.into());
        self
    }

    /// Set version
    #[must_use]
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Set input dimensions
    #[must_use]
    pub fn with_input_dim(mut self, dim: usize) -> Self {
        self.input_dim = Some(dim);
        self
    }

    /// Set output dimensions
    #[must_use]
    pub fn with_output_dim(mut self, dim: usize) -> Self {
        self.output_dim = Some(dim);
        self
    }

    /// Set file size
    #[must_use]
    pub fn with_file_size(mut self, size: u64) -> Self {
        self.file_size = size;
        self
    }
}

/// Detect model format from file path and contents
///
/// Per spec §3.2: Jidoka - verify both path and magic bytes match.
///
/// # Arguments
///
/// * `path` - Path to model file
///
/// # Returns
///
/// Model metadata with detected format
///
/// # Errors
///
/// Returns error if:
/// - File cannot be read
/// - Format cannot be detected
/// - Extension doesn't match magic bytes
///
/// # Example
///
/// ```rust,ignore
/// use realizar::model_loader::detect_model;
/// use std::path::Path;
///
/// let metadata = detect_model(Path::new("model.apr"))?;
/// assert_eq!(metadata.format, ModelFormat::Apr);
/// ```
pub fn detect_model(path: &Path) -> Result<ModelMetadata, LoadError> {
    // ALB-099: Read only 8 bytes for magic detection (was reading entire file)
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);
    let mut magic = [0u8; 8];
    let bytes_read = file.read(&mut magic)?;
    if bytes_read < 8 {
        return Err(LoadError::ParseError(format!(
            "File too small: {} bytes",
            bytes_read
        )));
    }

    let format = detect_and_verify_format(path, &magic)?;

    Ok(ModelMetadata::new(format).with_file_size(file_size))
}

/// Detect model format from bytes only (no path verification)
///
/// Useful for embedded models via `include_bytes!()`.
///
/// # Arguments
///
/// * `data` - Model file bytes
///
/// # Returns
///
/// Model metadata with detected format
///
/// # Errors
///
/// Returns error if:
/// - Data is too small for format detection (<8 bytes)
/// - Format cannot be detected from magic bytes
///
/// # Example
///
/// ```rust,ignore
/// use realizar::model_loader::detect_model_from_bytes;
///
/// const MODEL: &[u8] = include_bytes!("../models/model.apr");
/// let metadata = detect_model_from_bytes(MODEL)?;
/// ```
pub fn detect_model_from_bytes(data: &[u8]) -> Result<ModelMetadata, LoadError> {
    if data.len() < 8 {
        return Err(LoadError::ParseError(format!(
            "Data too small: {} bytes",
            data.len()
        )));
    }

    let format = detect_format(data.get(..8).expect("len >= 8 checked above"))?;

    Ok(ModelMetadata::new(format).with_file_size(data.len() as u64))
}

/// Extract model type from APR v2 JSON metadata
///
/// Reads the metadata offset/size from the header, parses JSON, and
/// returns model_type or model.architecture field.
fn read_apr_v2_model_type(data: &[u8]) -> Option<String> {
    if data.len() < 64 {
        return Some("Transformer".to_string()); // Default for incomplete header
    }

    let metadata_offset = u64::from_le_bytes([
        data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19],
    ]) as usize;
    let metadata_size = u32::from_le_bytes([data[20], data[21], data[22], data[23]]) as usize;

    if metadata_offset + metadata_size > data.len() || metadata_size == 0 {
        return Some("Transformer".to_string());
    }

    let metadata_bytes = &data[metadata_offset..metadata_offset + metadata_size];
    let metadata_str = std::str::from_utf8(metadata_bytes).ok()?;
    let json: serde_json::Value = serde_json::from_str(metadata_str).ok()?;

    // Check for model_type field in JSON metadata
    if let Some(model_type) = json.get("model_type").and_then(|v| v.as_str()) {
        if !model_type.is_empty() {
            return Some(model_type.to_string());
        }
    }
    // Check for model.architecture field (from GGUF import)
    if let Some(arch) = json.get("model.architecture").and_then(|v| v.as_str()) {
        return Some(format!("Transformer({})", arch));
    }

    Some("Transformer".to_string())
}

/// Map APR v1 type ID to model type name
fn read_apr_v1_model_type(type_id: u16) -> Option<&'static str> {
    match type_id {
        0x0001 => Some("LinearRegression"),
        0x0002 => Some("LogisticRegression"),
        0x0003 => Some("DecisionTree"),
        0x0004 => Some("RandomForest"),
        0x0005 => Some("GradientBoosting"),
        0x0006 => Some("KMeans"),
        0x0007 => Some("PCA"),
        0x0008 => Some("NaiveBayes"),
        0x0009 => Some("KNN"),
        0x000A => Some("SVM"),
        0x0010 => Some("NgramLM"),
        0x0011 => Some("TFIDF"),
        0x0012 => Some("CountVectorizer"),
        0x0020 => Some("NeuralSequential"),
        0x0021 => Some("NeuralCustom"),
        0x0030 => Some("ContentRecommender"),
        0x0040 => Some("MixtureOfExperts"),
        0x00FF => Some("Custom"),
        _ => None,
    }
}

/// Load APR model type from metadata bytes
///
/// Supports both APR v1 (type in header) and APR v2 (type in JSON metadata).
///
/// # Arguments
///
/// * `data` - APR file bytes (at least 8 bytes)
///
/// # Returns
///
/// APR model type string (e.g., "LogisticRegression", "Transformer")
pub fn read_apr_model_type(data: &[u8]) -> Option<String> {
    if data.len() < 8 {
        return None;
    }

    // APR v2 magic: "APR\0" (0x41, 0x50, 0x52, 0x00)
    if data[0..4] == [0x41, 0x50, 0x52, 0x00] {
        return read_apr_v2_model_type(data);
    }

    // APR v1 header layout: APRN (4 bytes) + type_id (2 bytes) + version (2 bytes)
    let type_id = u16::from_le_bytes([data[4], data[5]]);
    read_apr_v1_model_type(type_id).map(String::from)
}

/// Validate that loaded model matches expected type
///
/// Per Jidoka: fail fast if type mismatch.
///
/// # Arguments
///
/// * `expected` - Expected model type
/// * `actual` - Actual model type from file
///
/// # Returns
///
/// Ok if types match, Err otherwise
///
/// # Errors
///
/// Returns `LoadError::TypeMismatch` if expected and actual types differ.
pub fn validate_model_type(expected: &str, actual: &str) -> Result<(), LoadError> {
    if expected != actual {
        return Err(LoadError::TypeMismatch {
            expected: expected.to_string(),
            actual: actual.to_string(),
        });
    }
    Ok(())
}

include!("model_loader_load_error.rs");
