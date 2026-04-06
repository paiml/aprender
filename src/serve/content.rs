//! Core content types for WASM serving
//!
//! Defines the `ServeableContent` trait and supporting types for content
//! that can be served via browser-based WASM applications.

use std::{collections::HashMap, sync::Arc};

use arrow::record_batch::RecordBatch;
use serde::{Deserialize, Serialize};

use crate::{error::Result, serve::schema::ContentSchema};

/// Unique identifier for content types
///
/// Content types follow a namespaced format: `namespace.type`
/// Built-in types include:
/// - `alimentar.dataset` - Arrow/Parquet datasets
/// - `alimentar.raw` - Raw pasted/clipboard data
/// - `assetgen.course` - Educational courses
/// - `aprender.model` - ML models
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ContentTypeId(String);

impl ContentTypeId {
    /// Dataset content type identifier
    pub const DATASET: &'static str = "alimentar.dataset";
    /// Course content type identifier (assetgen)
    pub const COURSE: &'static str = "assetgen.course";
    /// Model content type identifier (aprender)
    pub const MODEL: &'static str = "aprender.model";
    /// Registry content type identifier
    pub const REGISTRY: &'static str = "alimentar.registry";
    /// Raw/pasted data content type identifier
    pub const RAW: &'static str = "alimentar.raw";

    /// Create a new content type ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Create dataset content type
    pub fn dataset() -> Self {
        Self(Self::DATASET.to_string())
    }

    /// Create course content type
    pub fn course() -> Self {
        Self(Self::COURSE.to_string())
    }

    /// Create model content type
    pub fn model() -> Self {
        Self(Self::MODEL.to_string())
    }

    /// Create registry content type
    pub fn registry() -> Self {
        Self(Self::REGISTRY.to_string())
    }

    /// Create raw data content type
    pub fn raw() -> Self {
        Self(Self::RAW.to_string())
    }

    /// Get the string representation
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Check if this is a built-in type
    pub fn is_builtin(&self) -> bool {
        matches!(
            self.0.as_str(),
            Self::DATASET | Self::COURSE | Self::MODEL | Self::REGISTRY | Self::RAW
        )
    }
}

impl std::fmt::Display for ContentTypeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Metadata associated with serveable content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentMetadata {
    /// Content type identifier
    pub content_type: ContentTypeId,
    /// Human-readable title
    pub title: String,
    /// Optional description
    pub description: Option<String>,
    /// Content size in bytes
    pub size: usize,
    /// Number of rows (for tabular data)
    pub row_count: Option<usize>,
    /// Content schema (if applicable)
    pub schema: Option<ContentSchema>,
    /// Source information (URL, file path, or "clipboard")
    pub source: Option<String>,
    /// Additional custom metadata
    #[serde(default)]
    pub custom: HashMap<String, serde_json::Value>,
}

impl ContentMetadata {
    /// Create new metadata with required fields
    pub fn new(content_type: ContentTypeId, title: impl Into<String>, size: usize) -> Self {
        Self {
            content_type,
            title: title.into(),
            description: None,
            size,
            row_count: None,
            schema: None,
            source: None,
            custom: HashMap::new(),
        }
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set row count
    pub fn with_row_count(mut self, count: usize) -> Self {
        self.row_count = Some(count);
        self
    }

    /// Set schema
    pub fn with_schema(mut self, schema: ContentSchema) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Set source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Add custom metadata
    pub fn with_custom(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.custom.insert(key.into(), value);
        self
    }
}

/// Validation report for content integrity checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Whether validation passed
    pub valid: bool,
    /// List of errors found
    pub errors: Vec<ValidationError>,
    /// List of warnings
    pub warnings: Vec<ValidationWarning>,
}

impl ValidationReport {
    /// Create a successful validation report
    pub fn success() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Create a failed validation report with errors
    pub fn failure(errors: Vec<ValidationError>) -> Self {
        Self {
            valid: false,
            errors,
            warnings: Vec::new(),
        }
    }

    /// Add a warning to the report
    pub fn with_warning(mut self, warning: ValidationWarning) -> Self {
        self.warnings.push(warning);
        self
    }

    /// Add an error and mark as invalid
    pub fn with_error(mut self, error: ValidationError) -> Self {
        self.valid = false;
        self.errors.push(error);
        self
    }
}

/// A validation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    /// Field or path that failed validation
    pub path: String,
    /// Error message
    pub message: String,
    /// Error code for programmatic handling
    pub code: Option<String>,
}

impl ValidationError {
    /// Create a new validation error
    pub fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
            code: None,
        }
    }

    /// Add an error code
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }
}

/// A validation warning (non-fatal issue)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Field or path with warning
    pub path: String,
    /// Warning message
    pub message: String,
}

impl ValidationWarning {
    /// Create a new validation warning
    pub fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

/// Trait for any content that can be served via WASM
///
/// This trait provides the abstraction layer between content types
/// (datasets, courses, models, raw data) and the serving infrastructure.
pub trait ServeableContent: Send + Sync {
    /// Returns the content schema for validation and UI generation
    fn schema(&self) -> ContentSchema;

    /// Validates content integrity
    ///
    /// # Errors
    ///
    /// Returns an error if validation cannot be performed.
    fn validate(&self) -> Result<ValidationReport>;

    /// Converts content to Arrow RecordBatch for efficient transfer
    ///
    /// # Errors
    ///
    /// Returns an error if the content cannot be converted to Arrow format.
    fn to_arrow(&self) -> Result<RecordBatch>;

    /// Returns content metadata for indexing and discovery
    fn metadata(&self) -> ContentMetadata;

    /// Returns content type identifier
    fn content_type(&self) -> ContentTypeId;

    /// Chunk iterator for streaming large content
    fn chunks(&self, chunk_size: usize) -> Box<dyn Iterator<Item = Result<RecordBatch>> + Send>;

    /// Get the raw bytes representation (for serialization)
    ///
    /// # Errors
    ///
    /// Returns an error if the content cannot be serialized to bytes.
    fn to_bytes(&self) -> Result<Vec<u8>>;
}

/// A boxed serveable content for dynamic dispatch
pub type BoxedContent = Box<dyn ServeableContent>;

/// Arc-wrapped serveable content for shared ownership
#[allow(dead_code)]
pub type SharedContent = Arc<dyn ServeableContent>;

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_content_type_id_new() {
        let id = ContentTypeId::new("custom.type");
        assert_eq!(id.as_str(), "custom.type");
    }

    #[test]
    fn test_content_type_is_builtin() {
        assert!(ContentTypeId::dataset().is_builtin());
        assert!(ContentTypeId::course().is_builtin());
        assert!(ContentTypeId::raw().is_builtin());
        assert!(!ContentTypeId::new("custom.type").is_builtin());
    }

    #[test]
    fn test_content_metadata_builder() {
        let meta = ContentMetadata::new(ContentTypeId::dataset(), "Test Dataset", 1024)
            .with_description("A test dataset")
            .with_row_count(100)
            .with_source("clipboard")
            .with_custom("version", serde_json::json!("1.0"));

        assert_eq!(meta.title, "Test Dataset");
        assert_eq!(meta.description, Some("A test dataset".to_string()));
        assert_eq!(meta.row_count, Some(100));
        assert_eq!(meta.source, Some("clipboard".to_string()));
        assert!(meta.custom.contains_key("version"));
    }

    #[test]
    fn test_validation_report() {
        let report = ValidationReport::success()
            .with_warning(ValidationWarning::new("field1", "Optional field missing"));

        assert!(report.valid);
        assert!(report.errors.is_empty());
        assert_eq!(report.warnings.len(), 1);

        let report = ValidationReport::failure(vec![ValidationError::new(
            "field2",
            "Required field missing",
        )
        .with_code("REQUIRED_FIELD")]);

        assert!(!report.valid);
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.errors[0].code, Some("REQUIRED_FIELD".to_string()));
    }

    #[test]
    fn test_validation_report_with_error() {
        let report = ValidationReport::success().with_error(ValidationError::new("field", "Error"));

        assert!(!report.valid);
        assert_eq!(report.errors.len(), 1);
    }

    #[test]
    fn test_content_type_id_model() {
        let model = ContentTypeId::model();
        assert_eq!(model.as_str(), "aprender.model");
        assert!(model.is_builtin());
    }

    #[test]
    fn test_content_type_id_registry() {
        let registry = ContentTypeId::registry();
        assert_eq!(registry.as_str(), "alimentar.registry");
        assert!(registry.is_builtin());
    }

    #[test]
    fn test_validation_error_without_code() {
        let err = ValidationError::new("path", "message");
        assert!(err.code.is_none());
        assert_eq!(err.path, "path");
        assert_eq!(err.message, "message");
    }
}
