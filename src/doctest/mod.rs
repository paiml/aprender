//! Python doctest extraction and corpus management.
//!
//! This module provides tools for extracting Python doctests from source files
//! and converting them to Arrow/Parquet format for ML training data.

mod parser;

use std::sync::Arc;

use arrow::{
    array::{ArrayRef, RecordBatch, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef},
};
use chrono::{DateTime, Utc};
pub use parser::{is_prose_continuation, DocTestParser};
use serde::{Deserialize, Serialize};

use crate::{ArrowDataset, Result};

/// A single extracted Python doctest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocTest {
    /// Module path (e.g., "collections.abc")
    pub module: String,
    /// Function/class name (e.g., "Hashable.__hash__")
    pub function: String,
    /// Input code (e.g., ">>> x = 5\n>>> x + 3")
    pub input: String,
    /// Expected output (e.g., "8")
    pub expected: String,
    /// Optional function signature (deferred to v2)
    pub signature: Option<String>,
}

impl DocTest {
    /// Create a new `DocTest`.
    #[must_use]
    pub fn new(
        module: impl Into<String>,
        function: impl Into<String>,
        input: impl Into<String>,
        expected: impl Into<String>,
    ) -> Self {
        Self {
            module: module.into(),
            function: function.into(),
            input: input.into(),
            expected: expected.into(),
            signature: None,
        }
    }

    /// Set the function signature.
    #[must_use]
    pub fn with_signature(mut self, signature: impl Into<String>) -> Self {
        self.signature = Some(signature.into());
        self
    }
}

/// A corpus of extracted doctests from a Python source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocTestCorpus {
    /// Source identifier (e.g., "cpython", "numpy", "pandas")
    pub source: String,
    /// Version or git SHA of the source
    pub version: String,
    /// When the extraction was performed
    pub extracted_at: DateTime<Utc>,
    /// The extracted doctests
    pub doctests: Vec<DocTest>,
}

impl DocTestCorpus {
    /// Create a new empty corpus.
    #[must_use]
    pub fn new(source: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            version: version.into(),
            extracted_at: Utc::now(),
            doctests: Vec::new(),
        }
    }

    /// Add a doctest to the corpus.
    pub fn push(&mut self, doctest: DocTest) {
        self.doctests.push(doctest);
    }

    /// Number of doctests in the corpus.
    #[must_use]
    pub fn len(&self) -> usize {
        self.doctests.len()
    }

    /// Check if corpus is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.doctests.is_empty()
    }

    /// Get the Arrow schema for doctest records.
    #[must_use]
    pub fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("version", DataType::Utf8, false),
            Field::new("module", DataType::Utf8, false),
            Field::new("function", DataType::Utf8, false),
            Field::new("input", DataType::Utf8, false),
            Field::new("expected", DataType::Utf8, false),
            Field::new("signature", DataType::Utf8, true),
        ]))
    }

    /// Convert the corpus to an Arrow `RecordBatch`.
    pub fn to_record_batch(&self) -> Result<RecordBatch> {
        let len = self.doctests.len();

        let source_array: ArrayRef = Arc::new(StringArray::from(vec![self.source.as_str(); len]));
        let version_array: ArrayRef = Arc::new(StringArray::from(vec![self.version.as_str(); len]));
        let module_array: ArrayRef = Arc::new(StringArray::from(
            self.doctests
                .iter()
                .map(|d| d.module.as_str())
                .collect::<Vec<_>>(),
        ));
        let function_array: ArrayRef = Arc::new(StringArray::from(
            self.doctests
                .iter()
                .map(|d| d.function.as_str())
                .collect::<Vec<_>>(),
        ));
        let input_array: ArrayRef = Arc::new(StringArray::from(
            self.doctests
                .iter()
                .map(|d| d.input.as_str())
                .collect::<Vec<_>>(),
        ));
        let expected_array: ArrayRef = Arc::new(StringArray::from(
            self.doctests
                .iter()
                .map(|d| d.expected.as_str())
                .collect::<Vec<_>>(),
        ));
        let signature_array: ArrayRef = Arc::new(StringArray::from(
            self.doctests
                .iter()
                .map(|d| d.signature.as_deref())
                .collect::<Vec<_>>(),
        ));

        let batch = RecordBatch::try_new(
            Self::schema(),
            vec![
                source_array,
                version_array,
                module_array,
                function_array,
                input_array,
                expected_array,
                signature_array,
            ],
        )?;

        Ok(batch)
    }

    /// Convert the corpus to an `ArrowDataset`.
    pub fn to_dataset(&self) -> Result<ArrowDataset> {
        let batch = self.to_record_batch()?;
        ArrowDataset::from_batch(batch)
    }

    /// Merge another corpus into this one.
    pub fn merge(&mut self, other: Self) {
        self.doctests.extend(other.doctests);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doctest_new() {
        let dt = DocTest::new("os.path", "join", ">>> join('a', 'b')", "'a/b'");
        assert_eq!(dt.module, "os.path");
        assert_eq!(dt.function, "join");
        assert!(dt.signature.is_none());
    }

    #[test]
    fn test_doctest_with_signature() {
        let dt = DocTest::new("os.path", "join", ">>> join('a', 'b')", "'a/b'")
            .with_signature("def join(*paths) -> str");
        assert_eq!(dt.signature, Some("def join(*paths) -> str".to_string()));
    }

    #[test]
    fn test_corpus_basic() {
        let mut corpus = DocTestCorpus::new("cpython", "v3.12.0");
        assert!(corpus.is_empty());

        corpus.push(DocTest::new("os", "getcwd", ">>> getcwd()", "'/home'"));
        assert_eq!(corpus.len(), 1);
    }

    #[test]
    fn test_corpus_to_record_batch() {
        let mut corpus = DocTestCorpus::new("numpy", "1.26.0");
        corpus.push(DocTest::new(
            "numpy",
            "array",
            ">>> array([1,2])",
            "array([1, 2])",
        ));
        corpus.push(DocTest::new(
            "numpy",
            "zeros",
            ">>> zeros(3)",
            "array([0., 0., 0.])",
        ));

        let batch = corpus.to_record_batch().expect("should create batch");
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 7);
    }

    #[test]
    fn test_corpus_schema() {
        let schema = DocTestCorpus::schema();
        assert_eq!(schema.fields().len(), 7);
        assert_eq!(schema.field(0).name(), "source");
        assert_eq!(schema.field(6).name(), "signature");
        assert!(schema.field(6).is_nullable());
    }
}
