//! Raw Source - Handle pasted/clipboard data with automatic format detection
//!
//! This module provides functionality for ingesting "raw" data from various
//! sources like clipboard, stdin, or direct string input. It automatically
//! detects the format (CSV, JSON, TSV, etc.) and converts to Arrow.
//!
//! # Example
//!
//! ```ignore
//! use alimentar::serve::{RawSource, SourceType};
//!
//! // From clipboard/pasted text
//! let raw = RawSource::from_string(pasted_text, SourceType::Clipboard);
//! let batch = raw.to_arrow()?;
//! ```

use std::{io::Cursor, sync::Arc};

use arrow::{
    array::{ArrayRef, RecordBatch, StringArray},
    datatypes::{DataType, Field, Schema},
};
use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    serve::{
        content::{
            ContentMetadata, ContentTypeId, ServeableContent, ValidationError, ValidationReport,
            ValidationWarning,
        },
        schema::{ContentSchema, FieldDefinition, FieldType},
    },
};

/// Source type for raw data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceType {
    /// Data pasted from clipboard
    Clipboard,
    /// Data from stdin
    Stdin,
    /// Data from a URL fetch
    Url,
    /// Data from a file
    File,
    /// Directly provided string
    Direct,
    /// Unknown source
    Unknown,
}

impl SourceType {
    /// Get human-readable name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Clipboard => "clipboard",
            Self::Stdin => "stdin",
            Self::Url => "url",
            Self::File => "file",
            Self::Direct => "direct",
            Self::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for SourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Detected format of raw data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectedFormat {
    /// Comma-separated values
    Csv,
    /// Tab-separated values
    Tsv,
    /// JSON array or object
    Json,
    /// JSON Lines (newline-delimited JSON)
    JsonLines,
    /// Plain text (line-per-row)
    PlainText,
    /// Could not detect format
    Unknown,
}

impl DetectedFormat {
    /// Get human-readable name
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Csv => "csv",
            Self::Tsv => "tsv",
            Self::Json => "json",
            Self::JsonLines => "jsonl",
            Self::PlainText => "text",
            Self::Unknown => "unknown",
        }
    }
}

/// Configuration for raw source parsing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawSourceConfig {
    /// Force a specific format (skip auto-detection)
    pub force_format: Option<DetectedFormat>,
    /// Whether to treat first line as header (for CSV/TSV)
    pub has_header: Option<bool>,
    /// Custom delimiter (for CSV-like formats)
    pub delimiter: Option<char>,
    /// Maximum rows to parse (for preview/sampling)
    pub max_rows: Option<usize>,
    /// Infer types from data (vs all strings)
    pub infer_types: bool,
    /// Source description (e.g., "Copied from Excel", "Pasted from website")
    pub source_description: Option<String>,
}

impl Default for RawSourceConfig {
    fn default() -> Self {
        Self {
            force_format: None,
            has_header: None,
            delimiter: None,
            max_rows: None,
            infer_types: true,
            source_description: None,
        }
    }
}

impl RawSourceConfig {
    /// Create a new config with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Force a specific format
    pub fn with_format(mut self, format: DetectedFormat) -> Self {
        self.force_format = Some(format);
        self
    }

    /// Set whether data has a header
    pub fn with_header(mut self, has_header: bool) -> Self {
        self.has_header = Some(has_header);
        self
    }

    /// Set custom delimiter
    pub fn with_delimiter(mut self, delimiter: char) -> Self {
        self.delimiter = Some(delimiter);
        self
    }

    /// Set maximum rows to parse
    pub fn with_max_rows(mut self, max_rows: usize) -> Self {
        self.max_rows = Some(max_rows);
        self
    }

    /// Set source description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.source_description = Some(description.into());
        self
    }
}

/// Raw source data container
///
/// Handles pasted/clipboard data with automatic format detection
/// and conversion to Arrow RecordBatch.
#[derive(Debug, Clone)]
pub struct RawSource {
    /// The raw string data
    data: String,
    /// Source type
    source_type: SourceType,
    /// Configuration
    config: RawSourceConfig,
    /// Detected format (computed lazily)
    detected_format: Option<DetectedFormat>,
    /// Cached Arrow batch
    cached_batch: Option<RecordBatch>,
}

impl RawSource {
    /// Create a new RawSource from a string
    pub fn from_string(data: impl Into<String>, source_type: SourceType) -> Self {
        Self {
            data: data.into(),
            source_type,
            config: RawSourceConfig::default(),
            detected_format: None,
            cached_batch: None,
        }
    }

    /// Create from clipboard content
    pub fn from_clipboard(data: impl Into<String>) -> Self {
        Self::from_string(data, SourceType::Clipboard)
    }

    /// Create from stdin content
    pub fn from_stdin(data: impl Into<String>) -> Self {
        Self::from_string(data, SourceType::Stdin)
    }

    /// Create with custom configuration
    pub fn with_config(mut self, config: RawSourceConfig) -> Self {
        self.config = config;
        self.detected_format = None; // Reset detection
        self.cached_batch = None;
        self
    }

    /// Get the raw data
    pub fn raw_data(&self) -> &str {
        &self.data
    }

    /// Get the source type
    pub fn source_type(&self) -> SourceType {
        self.source_type
    }

    /// Detect the format of the data
    pub fn detect_format(&self) -> DetectedFormat {
        if let Some(forced) = self.config.force_format {
            return forced;
        }

        let trimmed = self.data.trim();

        // Empty data
        if trimmed.is_empty() {
            return DetectedFormat::Unknown;
        }

        // Check for JSON
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            return DetectedFormat::Json;
        }

        // Check for JSON Lines (multiple JSON objects per line)
        let first_line = trimmed.lines().next().unwrap_or("");
        if first_line.starts_with('{') && first_line.ends_with('}') {
            let second_line = trimmed.lines().nth(1);
            if let Some(line) = second_line {
                if line.starts_with('{') {
                    return DetectedFormat::JsonLines;
                }
            }
        }

        // Count delimiters in first few lines
        let sample_lines: Vec<&str> = trimmed.lines().take(5).collect();
        if sample_lines.is_empty() {
            return DetectedFormat::PlainText;
        }

        let comma_count: usize = sample_lines.iter().map(|l| l.matches(',').count()).sum();
        let tab_count: usize = sample_lines.iter().map(|l| l.matches('\t').count()).sum();

        // Determine format based on delimiter frequency
        let lines_count = sample_lines.len();
        let avg_commas = comma_count / lines_count;
        let avg_tabs = tab_count / lines_count;

        if avg_tabs > 0 && avg_tabs >= avg_commas {
            DetectedFormat::Tsv
        } else if avg_commas > 0 {
            DetectedFormat::Csv
        } else {
            DetectedFormat::PlainText
        }
    }

    /// Parse the raw data into an Arrow RecordBatch
    ///
    /// # Errors
    ///
    /// Returns an error if the data cannot be parsed (invalid CSV, JSON, etc.).
    pub fn parse(&mut self) -> Result<RecordBatch> {
        if let Some(ref batch) = self.cached_batch {
            return Ok(batch.clone());
        }

        let format = self.detect_format();
        self.detected_format = Some(format);

        let batch = match format {
            DetectedFormat::Csv => self.parse_csv(',')?,
            DetectedFormat::Tsv => self.parse_csv('\t')?,
            DetectedFormat::Json => self.parse_json()?,
            DetectedFormat::JsonLines => self.parse_jsonl()?,
            DetectedFormat::PlainText | DetectedFormat::Unknown => self.parse_plain_text()?,
        };

        self.cached_batch = Some(batch.clone());
        Ok(batch)
    }

    /// Parse CSV/TSV data
    fn parse_csv(&self, default_delimiter: char) -> Result<RecordBatch> {
        use arrow_csv::reader::Format;

        let delimiter = self.config.delimiter.unwrap_or(default_delimiter);
        let has_header = self.config.has_header.unwrap_or(true);

        // Infer schema using Format
        let mut cursor_for_infer = Cursor::new(self.data.as_bytes());
        let format = Format::default()
            .with_delimiter(delimiter as u8)
            .with_header(has_header);
        let (inferred, _) = format
            .infer_schema(&mut cursor_for_infer, Some(1000))
            .map_err(|e| Error::transform(format!("Failed to infer CSV schema: {e}")))?;

        let schema = Arc::new(inferred);

        // Build reader with inferred schema
        let cursor = Cursor::new(self.data.as_bytes());
        let batch_size = self.config.max_rows.unwrap_or(8192);
        let builder = arrow_csv::ReaderBuilder::new(schema)
            .with_delimiter(delimiter as u8)
            .with_header(has_header)
            .with_batch_size(batch_size);

        let mut reader = builder
            .build(cursor)
            .map_err(|e| Error::transform(format!("Failed to parse CSV: {e}")))?;

        reader
            .next()
            .ok_or_else(|| Error::transform("No data in CSV"))?
            .map_err(|e| Error::transform(format!("Failed to read CSV batch: {e}")))
    }

    /// Parse JSON data
    fn parse_json(&self) -> Result<RecordBatch> {
        let cursor = Cursor::new(self.data.as_bytes());

        // Infer schema from JSON
        let (schema, _) = arrow_json::reader::infer_json_schema(cursor, Some(100))
            .map_err(|e| Error::transform(format!("Failed to infer JSON schema: {e}")))?;

        let cursor = Cursor::new(self.data.as_bytes());
        let mut reader = arrow_json::ReaderBuilder::new(Arc::new(schema))
            .build(cursor)
            .map_err(|e| Error::transform(format!("Failed to create JSON reader: {e}")))?;

        reader
            .next()
            .ok_or_else(|| Error::transform("No data in JSON"))?
            .map_err(|e| Error::transform(format!("Failed to read JSON batch: {e}")))
    }

    /// Parse JSON Lines data
    fn parse_jsonl(&self) -> Result<RecordBatch> {
        // JSON Lines is the same as JSON for Arrow reader
        self.parse_json()
    }

    /// Parse plain text (one column, one row per line)
    fn parse_plain_text(&self) -> Result<RecordBatch> {
        let lines: Vec<&str> = self.data.lines().collect();

        let max_rows = self.config.max_rows.unwrap_or(lines.len());
        let limited_lines: Vec<&str> = lines.into_iter().take(max_rows).collect();

        let schema = Arc::new(Schema::new(vec![Field::new("line", DataType::Utf8, false)]));

        let array: ArrayRef = Arc::new(StringArray::from(limited_lines));

        RecordBatch::try_new(schema, vec![array])
            .map_err(|e| Error::transform(format!("Failed to create text batch: {e}")))
    }

    /// Get the byte size of the raw data
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get the number of lines in the raw data
    pub fn line_count(&self) -> usize {
        self.data.lines().count()
    }
}

impl ServeableContent for RawSource {
    fn schema(&self) -> ContentSchema {
        ContentSchema::new(ContentTypeId::raw(), "1.0")
            .with_field(
                FieldDefinition::new("data", FieldType::String)
                    .with_description("Raw data content"),
            )
            .with_field(
                FieldDefinition::new("source_type", FieldType::String)
                    .with_description("Source type"),
            )
            .with_field(
                FieldDefinition::new("format", FieldType::String)
                    .with_description("Detected format"),
            )
    }

    fn validate(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport::success();

        if self.data.is_empty() {
            return Ok(ValidationReport::failure(vec![ValidationError::new(
                "data",
                "Raw data is empty",
            )]));
        }

        // Check for potential issues
        if self.data.len() > 100_000_000 {
            report = report.with_warning(ValidationWarning::new(
                "data",
                "Data size exceeds 100MB, consider chunking",
            ));
        }

        // Try to detect format
        let format = self.detect_format();
        if format == DetectedFormat::Unknown {
            report = report.with_warning(ValidationWarning::new(
                "format",
                "Could not detect data format, treating as plain text",
            ));
        }

        Ok(report)
    }

    fn to_arrow(&self) -> Result<RecordBatch> {
        let mut source = self.clone();
        source.parse()
    }

    fn metadata(&self) -> ContentMetadata {
        let format = self.detect_format();
        let mut meta = ContentMetadata::new(ContentTypeId::raw(), "Raw Data", self.size())
            .with_source(self.source_type.as_str())
            .with_row_count(self.line_count())
            .with_custom("format", serde_json::json!(format.as_str()));

        if let Some(ref desc) = self.config.source_description {
            meta = meta.with_description(desc.clone());
        }

        meta
    }

    fn content_type(&self) -> ContentTypeId {
        ContentTypeId::raw()
    }

    fn chunks(&self, _chunk_size: usize) -> Box<dyn Iterator<Item = Result<RecordBatch>> + Send> {
        // For raw source, we just return a single batch
        let batch_result = self.clone().parse();
        Box::new(std::iter::once(batch_result))
    }

    fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(self.data.as_bytes().to_vec())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_csv() {
        let csv_data = "name,age,city\nAlice,30,NYC\nBob,25,LA";
        let source = RawSource::from_clipboard(csv_data);
        assert_eq!(source.detect_format(), DetectedFormat::Csv);
    }

    #[test]
    fn test_detect_tsv() {
        let tsv_data = "name\tage\tcity\nAlice\t30\tNYC\nBob\t25\tLA";
        let source = RawSource::from_clipboard(tsv_data);
        assert_eq!(source.detect_format(), DetectedFormat::Tsv);
    }

    #[test]
    fn test_detect_json() {
        let json_data = r#"[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]"#;
        let source = RawSource::from_clipboard(json_data);
        assert_eq!(source.detect_format(), DetectedFormat::Json);
    }

    #[test]
    fn test_detect_json_object() {
        let json_data = r#"{"users": [{"name": "Alice"}]}"#;
        let source = RawSource::from_clipboard(json_data);
        assert_eq!(source.detect_format(), DetectedFormat::Json);
    }

    #[test]
    fn test_detect_plain_text() {
        let text_data = "Hello world\nThis is plain text\nNo delimiters here";
        let source = RawSource::from_clipboard(text_data);
        assert_eq!(source.detect_format(), DetectedFormat::PlainText);
    }

    #[test]
    fn test_force_format() {
        let data = "Hello world";
        let config = RawSourceConfig::new().with_format(DetectedFormat::Csv);
        let source = RawSource::from_clipboard(data).with_config(config);
        assert_eq!(source.detect_format(), DetectedFormat::Csv);
    }

    #[test]
    fn test_parse_plain_text() {
        let text_data = "Line 1\nLine 2\nLine 3";
        let mut source = RawSource::from_clipboard(text_data);
        let batch = source.parse().unwrap();

        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.schema().field(0).name(), "line");
    }

    #[test]
    fn test_parse_csv() {
        let csv_data = "name,age\nAlice,30\nBob,25";
        let mut source = RawSource::from_clipboard(csv_data);
        let batch = source.parse().unwrap();

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);
    }

    #[test]
    fn test_source_type() {
        let source = RawSource::from_clipboard("data");
        assert_eq!(source.source_type(), SourceType::Clipboard);

        let source = RawSource::from_stdin("data");
        assert_eq!(source.source_type(), SourceType::Stdin);
    }

    #[test]
    fn test_validation_empty() {
        let source = RawSource::from_clipboard("");
        let report = source.validate().unwrap();
        assert!(!report.valid);
    }

    #[test]
    fn test_validation_success() {
        let source = RawSource::from_clipboard("some data");
        let report = source.validate().unwrap();
        assert!(report.valid);
    }

    #[test]
    fn test_metadata() {
        let source = RawSource::from_clipboard("test data")
            .with_config(RawSourceConfig::new().with_description("Copied from spreadsheet"));

        let meta = source.metadata();
        assert_eq!(meta.content_type, ContentTypeId::raw());
        assert_eq!(meta.source, Some("clipboard".to_string()));
        assert_eq!(
            meta.description,
            Some("Copied from spreadsheet".to_string())
        );
    }

    #[test]
    fn test_max_rows() {
        let text_data = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5";
        let config = RawSourceConfig::new().with_max_rows(3);
        let mut source = RawSource::from_clipboard(text_data).with_config(config);
        let batch = source.parse().unwrap();

        assert_eq!(batch.num_rows(), 3);
    }

    #[test]
    fn test_parse_tsv() {
        let tsv_data = "name\tage\nAlice\t30\nBob\t25";
        let mut source = RawSource::from_clipboard(tsv_data);
        let batch = source.parse().unwrap();

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 2);
    }

    #[test]
    fn test_parse_json() {
        // Arrow JSON reader expects newline-delimited JSON objects
        let json_data = "{\"name\":\"Alice\",\"age\":30}\n{\"name\":\"Bob\",\"age\":25}";
        let mut source = RawSource::from_clipboard(json_data);
        let batch = source.parse().unwrap();

        assert!(batch.num_rows() >= 1);
    }

    #[test]
    fn test_to_bytes() {
        let source = RawSource::from_clipboard("test data");
        let bytes = source.to_bytes().unwrap();
        assert_eq!(bytes, b"test data");
    }

    #[test]
    fn test_chunks() {
        let source = RawSource::from_clipboard("name,age\nAlice,30");
        let chunks: Vec<_> = source.chunks(100).collect();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_ok());
    }

    #[test]
    fn test_schema() {
        let source = RawSource::from_clipboard("test");
        let schema = source.schema();
        assert_eq!(schema.content_type, ContentTypeId::raw());
        assert!(schema.get_field("data").is_some());
        assert!(schema.get_field("source_type").is_some());
        assert!(schema.get_field("format").is_some());
    }

    #[test]
    fn test_content_type() {
        let source = RawSource::from_clipboard("test");
        assert_eq!(source.content_type(), ContentTypeId::raw());
    }

    #[test]
    fn test_from_string() {
        let source = RawSource::from_string("test", SourceType::Direct);
        assert_eq!(source.source_type(), SourceType::Direct);
    }

    #[test]
    fn test_line_count() {
        let source = RawSource::from_clipboard("line1\nline2\nline3");
        assert_eq!(source.line_count(), 3);
    }

    #[test]
    fn test_detected_format_as_str() {
        assert_eq!(DetectedFormat::Csv.as_str(), "csv");
        assert_eq!(DetectedFormat::Tsv.as_str(), "tsv");
        assert_eq!(DetectedFormat::Json.as_str(), "json");
        assert_eq!(DetectedFormat::JsonLines.as_str(), "jsonl");
        assert_eq!(DetectedFormat::PlainText.as_str(), "text");
        assert_eq!(DetectedFormat::Unknown.as_str(), "unknown");
    }

    #[test]
    fn test_source_type_as_str() {
        assert_eq!(SourceType::Clipboard.as_str(), "clipboard");
        assert_eq!(SourceType::Stdin.as_str(), "stdin");
        assert_eq!(SourceType::Url.as_str(), "url");
        assert_eq!(SourceType::File.as_str(), "file");
        assert_eq!(SourceType::Direct.as_str(), "direct");
        assert_eq!(SourceType::Unknown.as_str(), "unknown");
    }

    #[test]
    fn test_config_with_delimiter() {
        let config = RawSourceConfig::new().with_delimiter(';');
        assert_eq!(config.delimiter, Some(';'));
    }

    #[test]
    fn test_config_with_header() {
        let config = RawSourceConfig::new().with_header(false);
        assert_eq!(config.has_header, Some(false));
    }

    #[test]
    fn test_config_default() {
        let config = RawSourceConfig::default();
        assert!(config.delimiter.is_none());
        assert!(config.has_header.is_none());
        assert!(config.max_rows.is_none());
    }

    #[test]
    fn test_cached_batch() {
        let mut source = RawSource::from_clipboard("name,age\nAlice,30");

        // First parse
        let batch1 = source.parse().unwrap();

        // Second parse should return cached
        let batch2 = source.parse().unwrap();

        assert_eq!(batch1.num_rows(), batch2.num_rows());
    }

    #[test]
    fn test_large_data_validation() {
        // Large data should still validate as OK
        let large_data = "x\n".repeat(2000);
        let source = RawSource::from_clipboard(&large_data);
        let report = source.validate().unwrap();
        assert!(report.valid);
    }
}
