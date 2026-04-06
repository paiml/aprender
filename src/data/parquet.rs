//! Parquet I/O for test case storage
//!
//! This module provides efficient columnar storage for verified test cases
//! using Apache Parquet format via Arrow.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float32Array, RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use super::{CodeFeatures, GenerationMetadata, TestCase, TestResult};
use crate::{Error, Language, Result};

/// Schema for test case storage
fn test_case_schema() -> Schema {
    Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("source_language", DataType::Utf8, false),
        Field::new("source_code", DataType::Utf8, false),
        Field::new("target_language", DataType::Utf8, false),
        Field::new("target_code", DataType::Utf8, true),
        // Flattened TestResult
        Field::new("result_type", DataType::Utf8, false),
        Field::new("result_expected", DataType::Utf8, true),
        Field::new("result_actual", DataType::Utf8, true),
        Field::new("result_error", DataType::Utf8, true),
        Field::new("result_phase", DataType::Utf8, true),
        Field::new("result_timeout_ms", DataType::UInt64, true),
        // Flattened CodeFeatures
        Field::new("feat_ast_depth", DataType::UInt32, false),
        Field::new("feat_num_operators", DataType::UInt32, false),
        Field::new("feat_num_control_flow", DataType::UInt32, false),
        Field::new("feat_cyclomatic_complexity", DataType::Float32, false),
        Field::new("feat_num_type_coercions", DataType::UInt32, false),
        Field::new("feat_uses_edge_values", DataType::Boolean, false),
        // Flattened GenerationMetadata
        Field::new("meta_strategy", DataType::Utf8, false),
        Field::new("meta_mutation_operators", DataType::Utf8, false), // JSON array
        Field::new("meta_timestamp", DataType::Utf8, false),
        Field::new("meta_transpiler_version", DataType::Utf8, false),
    ])
}

/// Writer for test cases to Parquet format
pub struct ParquetWriter {
    writer: ArrowWriter<File>,
    schema: Arc<Schema>,
    batch_size: usize,
    buffer: Vec<TestCase>,
}

impl std::fmt::Debug for ParquetWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetWriter")
            .field("batch_size", &self.batch_size)
            .field("buffer_len", &self.buffer.len())
            .finish_non_exhaustive()
    }
}

impl ParquetWriter {
    /// Create a new Parquet writer
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created
    pub fn new(path: impl AsRef<Path>, batch_size: usize) -> Result<Self> {
        let schema = Arc::new(test_case_schema());
        let file = File::create(path.as_ref())
            .map_err(|e| Error::Data(format!("Failed to create parquet file: {e}")))?;

        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();

        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
            .map_err(|e| Error::Data(format!("Failed to create arrow writer: {e}")))?;

        Ok(Self {
            writer,
            schema,
            batch_size,
            buffer: Vec::with_capacity(batch_size),
        })
    }

    /// Write a single test case (buffered)
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails
    pub fn write(&mut self, test_case: TestCase) -> Result<()> {
        self.buffer.push(test_case);

        if self.buffer.len() >= self.batch_size {
            self.flush()?;
        }

        Ok(())
    }

    /// Write multiple test cases
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails
    pub fn write_batch(&mut self, test_cases: Vec<TestCase>) -> Result<()> {
        for tc in test_cases {
            self.write(tc)?;
        }
        Ok(())
    }

    /// Flush buffered data to disk
    ///
    /// # Errors
    ///
    /// Returns an error if flushing fails
    pub fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let batch = self.create_record_batch()?;
        self.writer
            .write(&batch)
            .map_err(|e| Error::Data(format!("Failed to write batch: {e}")))?;
        self.buffer.clear();

        Ok(())
    }

    /// Close the writer and finalize the file
    ///
    /// # Errors
    ///
    /// Returns an error if closing fails
    pub fn close(mut self) -> Result<()> {
        self.flush()?;
        self.writer
            .close()
            .map_err(|e| Error::Data(format!("Failed to close writer: {e}")))?;
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn create_record_batch(&self) -> Result<RecordBatch> {
        let len = self.buffer.len();

        // Extract columns
        let ids: Vec<String> = self.buffer.iter().map(|tc| tc.id.to_string()).collect();
        let source_languages: Vec<String> = self
            .buffer
            .iter()
            .map(|tc| format!("{:?}", tc.source_language))
            .collect();
        let source_codes: Vec<String> = self
            .buffer
            .iter()
            .map(|tc| tc.source_code.clone())
            .collect();
        let target_languages: Vec<String> = self
            .buffer
            .iter()
            .map(|tc| format!("{:?}", tc.target_language))
            .collect();
        let target_codes: Vec<Option<String>> = self
            .buffer
            .iter()
            .map(|tc| tc.target_code.clone())
            .collect();

        // Flatten TestResult
        let mut result_types = Vec::with_capacity(len);
        let mut result_expected: Vec<Option<String>> = Vec::with_capacity(len);
        let mut result_actual: Vec<Option<String>> = Vec::with_capacity(len);
        let mut result_error: Vec<Option<String>> = Vec::with_capacity(len);
        let mut result_phase: Vec<Option<String>> = Vec::with_capacity(len);
        let mut result_timeout_ms: Vec<Option<u64>> = Vec::with_capacity(len);

        for tc in &self.buffer {
            match &tc.result {
                TestResult::Pass => {
                    result_types.push("Pass".to_string());
                    result_expected.push(None);
                    result_actual.push(None);
                    result_error.push(None);
                    result_phase.push(None);
                    result_timeout_ms.push(None);
                }
                TestResult::TranspileError(err) => {
                    result_types.push("TranspileError".to_string());
                    result_expected.push(None);
                    result_actual.push(None);
                    result_error.push(Some(err.clone()));
                    result_phase.push(None);
                    result_timeout_ms.push(None);
                }
                TestResult::OutputMismatch { expected, actual } => {
                    result_types.push("OutputMismatch".to_string());
                    result_expected.push(Some(expected.clone()));
                    result_actual.push(Some(actual.clone()));
                    result_error.push(None);
                    result_phase.push(None);
                    result_timeout_ms.push(None);
                }
                TestResult::Timeout { limit_ms } => {
                    result_types.push("Timeout".to_string());
                    result_expected.push(None);
                    result_actual.push(None);
                    result_error.push(None);
                    result_phase.push(None);
                    result_timeout_ms.push(Some(*limit_ms));
                }
                TestResult::RuntimeError { phase, error } => {
                    result_types.push("RuntimeError".to_string());
                    result_expected.push(None);
                    result_actual.push(None);
                    result_error.push(Some(error.clone()));
                    result_phase.push(Some(phase.clone()));
                    result_timeout_ms.push(None);
                }
            }
        }

        // Flatten CodeFeatures
        let feat_ast_depth: Vec<u32> = self.buffer.iter().map(|tc| tc.features.ast_depth).collect();
        let feat_num_operators: Vec<u32> = self
            .buffer
            .iter()
            .map(|tc| tc.features.num_operators)
            .collect();
        let feat_num_control_flow: Vec<u32> = self
            .buffer
            .iter()
            .map(|tc| tc.features.num_control_flow)
            .collect();
        let feat_cyclomatic_complexity: Vec<f32> = self
            .buffer
            .iter()
            .map(|tc| tc.features.cyclomatic_complexity)
            .collect();
        let feat_num_type_coercions: Vec<u32> = self
            .buffer
            .iter()
            .map(|tc| tc.features.num_type_coercions)
            .collect();
        let feat_uses_edge_values: Vec<bool> = self
            .buffer
            .iter()
            .map(|tc| tc.features.uses_edge_values)
            .collect();

        // Flatten GenerationMetadata
        let meta_strategy: Vec<String> = self
            .buffer
            .iter()
            .map(|tc| tc.metadata.strategy.clone())
            .collect();
        let meta_mutation_operators: Vec<String> = self
            .buffer
            .iter()
            .map(|tc| serde_json::to_string(&tc.metadata.mutation_operators).unwrap_or_default())
            .collect();
        let meta_timestamp: Vec<String> = self
            .buffer
            .iter()
            .map(|tc| tc.metadata.timestamp.clone())
            .collect();
        let meta_transpiler_version: Vec<String> = self
            .buffer
            .iter()
            .map(|tc| tc.metadata.transpiler_version.clone())
            .collect();

        // Create Arrow arrays
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(source_languages)),
            Arc::new(StringArray::from(source_codes)),
            Arc::new(StringArray::from(target_languages)),
            Arc::new(StringArray::from(target_codes)),
            Arc::new(StringArray::from(result_types)),
            Arc::new(StringArray::from(result_expected)),
            Arc::new(StringArray::from(result_actual)),
            Arc::new(StringArray::from(result_error)),
            Arc::new(StringArray::from(result_phase)),
            Arc::new(UInt64Array::from(result_timeout_ms)),
            Arc::new(UInt32Array::from(feat_ast_depth)),
            Arc::new(UInt32Array::from(feat_num_operators)),
            Arc::new(UInt32Array::from(feat_num_control_flow)),
            Arc::new(Float32Array::from(feat_cyclomatic_complexity)),
            Arc::new(UInt32Array::from(feat_num_type_coercions)),
            Arc::new(BooleanArray::from(feat_uses_edge_values)),
            Arc::new(StringArray::from(meta_strategy)),
            Arc::new(StringArray::from(meta_mutation_operators)),
            Arc::new(StringArray::from(meta_timestamp)),
            Arc::new(StringArray::from(meta_transpiler_version)),
        ];

        RecordBatch::try_new(self.schema.clone(), columns)
            .map_err(|e| Error::Data(format!("Failed to create record batch: {e}")))
    }
}

/// Reader for test cases from Parquet format
#[derive(Debug)]
pub struct ParquetReader;

impl ParquetReader {
    /// Read all test cases from a Parquet file
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails
    pub fn read(path: impl AsRef<Path>) -> Result<Vec<TestCase>> {
        let file = File::open(path.as_ref())
            .map_err(|e| Error::Data(format!("Failed to open parquet file: {e}")))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| Error::Data(format!("Failed to create reader builder: {e}")))?;

        let reader = builder
            .build()
            .map_err(|e| Error::Data(format!("Failed to build reader: {e}")))?;

        let mut test_cases = Vec::new();

        for batch_result in reader {
            let batch =
                batch_result.map_err(|e| Error::Data(format!("Failed to read batch: {e}")))?;
            let batch_cases = Self::batch_to_test_cases(&batch)?;
            test_cases.extend(batch_cases);
        }

        Ok(test_cases)
    }

    #[allow(clippy::too_many_lines)]
    fn batch_to_test_cases(batch: &RecordBatch) -> Result<Vec<TestCase>> {
        let num_rows = batch.num_rows();
        let mut test_cases = Vec::with_capacity(num_rows);

        // Get column references
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Data("Invalid id column".to_string()))?;
        let source_languages = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Data("Invalid source_language column".to_string()))?;
        let source_codes = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Data("Invalid source_code column".to_string()))?;
        let target_languages = batch
            .column(3)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Data("Invalid target_language column".to_string()))?;
        let target_codes = batch
            .column(4)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Data("Invalid target_code column".to_string()))?;
        let result_types = batch
            .column(5)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Data("Invalid result_type column".to_string()))?;
        let result_expected = batch
            .column(6)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Data("Invalid result_expected column".to_string()))?;
        let result_actual = batch
            .column(7)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Data("Invalid result_actual column".to_string()))?;
        let result_error = batch
            .column(8)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Data("Invalid result_error column".to_string()))?;
        let result_phase = batch
            .column(9)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Data("Invalid result_phase column".to_string()))?;
        let result_timeout_ms = batch
            .column(10)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| Error::Data("Invalid result_timeout_ms column".to_string()))?;
        let feat_ast_depth = batch
            .column(11)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| Error::Data("Invalid feat_ast_depth column".to_string()))?;
        let feat_num_operators = batch
            .column(12)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| Error::Data("Invalid feat_num_operators column".to_string()))?;
        let feat_num_control_flow = batch
            .column(13)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| Error::Data("Invalid feat_num_control_flow column".to_string()))?;
        let feat_cyclomatic_complexity = batch
            .column(14)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| {
            Error::Data("Invalid feat_cyclomatic_complexity column".to_string())
        })?;
        let feat_num_type_coercions = batch
            .column(15)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| Error::Data("Invalid feat_num_type_coercions column".to_string()))?;
        let feat_uses_edge_values = batch
            .column(16)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| Error::Data("Invalid feat_uses_edge_values column".to_string()))?;
        let meta_strategy = batch
            .column(17)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Data("Invalid meta_strategy column".to_string()))?;
        let meta_mutation_operators = batch
            .column(18)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Data("Invalid meta_mutation_operators column".to_string()))?;
        let meta_timestamp = batch
            .column(19)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Data("Invalid meta_timestamp column".to_string()))?;
        let meta_transpiler_version = batch
            .column(20)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| Error::Data("Invalid meta_transpiler_version column".to_string()))?;

        for i in 0..num_rows {
            let result = Self::parse_test_result(
                result_types.value(i),
                if result_expected.is_null(i) {
                    None
                } else {
                    Some(result_expected.value(i))
                },
                if result_actual.is_null(i) {
                    None
                } else {
                    Some(result_actual.value(i))
                },
                if result_error.is_null(i) {
                    None
                } else {
                    Some(result_error.value(i))
                },
                if result_phase.is_null(i) {
                    None
                } else {
                    Some(result_phase.value(i))
                },
                if result_timeout_ms.is_null(i) {
                    None
                } else {
                    Some(result_timeout_ms.value(i))
                },
            )?;

            let mutation_operators: Vec<String> =
                serde_json::from_str(meta_mutation_operators.value(i)).unwrap_or_default();

            let test_case = TestCase {
                id: uuid::Uuid::parse_str(ids.value(i))
                    .map_err(|e| Error::Data(format!("Invalid UUID: {e}")))?,
                source_language: Self::parse_language(source_languages.value(i))?,
                source_code: source_codes.value(i).to_string(),
                target_language: Self::parse_language(target_languages.value(i))?,
                target_code: if target_codes.is_null(i) {
                    None
                } else {
                    Some(target_codes.value(i).to_string())
                },
                result,
                features: CodeFeatures {
                    ast_depth: feat_ast_depth.value(i),
                    num_operators: feat_num_operators.value(i),
                    num_control_flow: feat_num_control_flow.value(i),
                    cyclomatic_complexity: feat_cyclomatic_complexity.value(i),
                    num_type_coercions: feat_num_type_coercions.value(i),
                    uses_edge_values: feat_uses_edge_values.value(i),
                },
                metadata: GenerationMetadata {
                    strategy: meta_strategy.value(i).to_string(),
                    mutation_operators,
                    timestamp: meta_timestamp.value(i).to_string(),
                    transpiler_version: meta_transpiler_version.value(i).to_string(),
                },
            };

            test_cases.push(test_case);
        }

        Ok(test_cases)
    }

    fn parse_test_result(
        result_type: &str,
        expected: Option<&str>,
        actual: Option<&str>,
        error: Option<&str>,
        phase: Option<&str>,
        timeout_ms: Option<u64>,
    ) -> Result<TestResult> {
        match result_type {
            "Pass" => Ok(TestResult::Pass),
            "TranspileError" => Ok(TestResult::TranspileError(error.unwrap_or("").to_string())),
            "OutputMismatch" => Ok(TestResult::OutputMismatch {
                expected: expected.unwrap_or("").to_string(),
                actual: actual.unwrap_or("").to_string(),
            }),
            "Timeout" => Ok(TestResult::Timeout {
                limit_ms: timeout_ms.unwrap_or(0),
            }),
            "RuntimeError" => Ok(TestResult::RuntimeError {
                phase: phase.unwrap_or("unknown").to_string(),
                error: error.unwrap_or("").to_string(),
            }),
            _ => Err(Error::Data(format!("Unknown result type: {result_type}"))),
        }
    }

    fn parse_language(s: &str) -> Result<Language> {
        match s {
            "Python" => Ok(Language::Python),
            "Rust" => Ok(Language::Rust),
            "Bash" => Ok(Language::Bash),
            "C" => Ok(Language::C),
            "Ruchy" => Ok(Language::Ruchy),
            _ => Err(Error::Data(format!("Unknown language: {s}"))),
        }
    }
}

/// Statistics about a Parquet dataset
#[derive(Debug, Clone)]
pub struct DatasetStats {
    /// Total number of test cases
    pub total_cases: usize,
    /// Number of passing test cases
    pub pass_count: usize,
    /// Number of failing test cases
    pub fail_count: usize,
    /// Unique source languages
    pub source_languages: Vec<Language>,
    /// Unique target languages
    pub target_languages: Vec<Language>,
}

impl DatasetStats {
    /// Compute statistics from a list of test cases
    #[must_use]
    pub fn from_test_cases(test_cases: &[TestCase]) -> Self {
        use std::collections::HashSet;

        let mut pass_count = 0;
        let mut fail_count = 0;
        let mut source_langs = HashSet::new();
        let mut target_langs = HashSet::new();

        for tc in test_cases {
            if tc.result == TestResult::Pass {
                pass_count += 1;
            } else {
                fail_count += 1;
            }
            source_langs.insert(tc.source_language);
            target_langs.insert(tc.target_language);
        }

        Self {
            total_cases: test_cases.len(),
            pass_count,
            fail_count,
            source_languages: source_langs.into_iter().collect(),
            target_languages: target_langs.into_iter().collect(),
        }
    }

    /// Get the pass rate as a percentage
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        if self.total_cases == 0 {
            0.0
        } else {
            (self.pass_count as f64 / self.total_cases as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use uuid::Uuid;

    fn create_test_case(result: TestResult) -> TestCase {
        TestCase {
            id: Uuid::new_v4(),
            source_language: Language::Python,
            source_code: "x = 1".to_string(),
            target_language: Language::Rust,
            target_code: Some("let x = 1;".to_string()),
            result,
            features: CodeFeatures {
                ast_depth: 2,
                num_operators: 1,
                num_control_flow: 0,
                cyclomatic_complexity: 1.0,
                num_type_coercions: 0,
                uses_edge_values: false,
            },
            metadata: GenerationMetadata {
                strategy: "exhaustive".to_string(),
                mutation_operators: vec!["AOR".to_string()],
                timestamp: "1234567890".to_string(),
                transpiler_version: "0.1.0".to_string(),
            },
        }
    }

    #[test]
    fn test_parquet_roundtrip_pass() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test.parquet");

        let tc = create_test_case(TestResult::Pass);
        let original_id = tc.id;

        // Write
        let mut writer = ParquetWriter::new(&path, 10).expect("Failed to create writer");
        writer.write(tc).expect("Failed to write");
        writer.close().expect("Failed to close");

        // Read
        let test_cases = ParquetReader::read(&path).expect("Failed to read");
        assert_eq!(test_cases.len(), 1);
        assert_eq!(test_cases[0].id, original_id);
        assert_eq!(test_cases[0].result, TestResult::Pass);
    }

    #[test]
    fn test_parquet_roundtrip_output_mismatch() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test.parquet");

        let tc = create_test_case(TestResult::OutputMismatch {
            expected: "hello".to_string(),
            actual: "world".to_string(),
        });

        let mut writer = ParquetWriter::new(&path, 10).expect("Failed to create writer");
        writer.write(tc).expect("Failed to write");
        writer.close().expect("Failed to close");

        let test_cases = ParquetReader::read(&path).expect("Failed to read");
        assert!(matches!(
            test_cases[0].result,
            TestResult::OutputMismatch { .. }
        ));
    }

    #[test]
    fn test_parquet_roundtrip_timeout() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test.parquet");

        let tc = create_test_case(TestResult::Timeout { limit_ms: 5000 });

        let mut writer = ParquetWriter::new(&path, 10).expect("Failed to create writer");
        writer.write(tc).expect("Failed to write");
        writer.close().expect("Failed to close");

        let test_cases = ParquetReader::read(&path).expect("Failed to read");
        if let TestResult::Timeout { limit_ms } = test_cases[0].result {
            assert_eq!(limit_ms, 5000);
        } else {
            panic!("Expected Timeout result");
        }
    }

    #[test]
    fn test_parquet_roundtrip_runtime_error() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test.parquet");

        let tc = create_test_case(TestResult::RuntimeError {
            phase: "source".to_string(),
            error: "division by zero".to_string(),
        });

        let mut writer = ParquetWriter::new(&path, 10).expect("Failed to create writer");
        writer.write(tc).expect("Failed to write");
        writer.close().expect("Failed to close");

        let test_cases = ParquetReader::read(&path).expect("Failed to read");
        if let TestResult::RuntimeError { phase, error } = &test_cases[0].result {
            assert_eq!(phase, "source");
            assert_eq!(error, "division by zero");
        } else {
            panic!("Expected RuntimeError result");
        }
    }

    #[test]
    fn test_parquet_roundtrip_transpile_error() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test.parquet");

        let tc = create_test_case(TestResult::TranspileError("syntax error".to_string()));

        let mut writer = ParquetWriter::new(&path, 10).expect("Failed to create writer");
        writer.write(tc).expect("Failed to write");
        writer.close().expect("Failed to close");

        let test_cases = ParquetReader::read(&path).expect("Failed to read");
        if let TestResult::TranspileError(err) = &test_cases[0].result {
            assert_eq!(err, "syntax error");
        } else {
            panic!("Expected TranspileError result");
        }
    }

    #[test]
    fn test_parquet_batch_write() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test.parquet");

        let test_cases: Vec<TestCase> = (0..100)
            .map(|_| create_test_case(TestResult::Pass))
            .collect();

        let mut writer = ParquetWriter::new(&path, 25).expect("Failed to create writer");
        writer
            .write_batch(test_cases)
            .expect("Failed to write batch");
        writer.close().expect("Failed to close");

        let read_cases = ParquetReader::read(&path).expect("Failed to read");
        assert_eq!(read_cases.len(), 100);
    }

    #[test]
    fn test_parquet_preserves_features() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test.parquet");

        let mut tc = create_test_case(TestResult::Pass);
        tc.features = CodeFeatures {
            ast_depth: 10,
            num_operators: 25,
            num_control_flow: 5,
            cyclomatic_complexity: 12.5,
            num_type_coercions: 3,
            uses_edge_values: true,
        };

        let mut writer = ParquetWriter::new(&path, 10).expect("Failed to create writer");
        writer.write(tc).expect("Failed to write");
        writer.close().expect("Failed to close");

        let test_cases = ParquetReader::read(&path).expect("Failed to read");
        assert_eq!(test_cases[0].features.ast_depth, 10);
        assert_eq!(test_cases[0].features.num_operators, 25);
        assert_eq!(test_cases[0].features.num_control_flow, 5);
        assert!((test_cases[0].features.cyclomatic_complexity - 12.5).abs() < 0.001);
        assert_eq!(test_cases[0].features.num_type_coercions, 3);
        assert!(test_cases[0].features.uses_edge_values);
    }

    #[test]
    fn test_parquet_preserves_metadata() {
        let dir = tempdir().expect("Failed to create temp dir");
        let path = dir.path().join("test.parquet");

        let mut tc = create_test_case(TestResult::Pass);
        tc.metadata = GenerationMetadata {
            strategy: "coverage-guided".to_string(),
            mutation_operators: vec!["AOR".to_string(), "ROR".to_string(), "LOR".to_string()],
            timestamp: "9876543210".to_string(),
            transpiler_version: "1.2.3".to_string(),
        };

        let mut writer = ParquetWriter::new(&path, 10).expect("Failed to create writer");
        writer.write(tc).expect("Failed to write");
        writer.close().expect("Failed to close");

        let test_cases = ParquetReader::read(&path).expect("Failed to read");
        assert_eq!(test_cases[0].metadata.strategy, "coverage-guided");
        assert_eq!(test_cases[0].metadata.mutation_operators.len(), 3);
        assert_eq!(test_cases[0].metadata.timestamp, "9876543210");
        assert_eq!(test_cases[0].metadata.transpiler_version, "1.2.3");
    }

    #[test]
    fn test_dataset_stats() {
        let test_cases = vec![
            create_test_case(TestResult::Pass),
            create_test_case(TestResult::Pass),
            create_test_case(TestResult::OutputMismatch {
                expected: "a".to_string(),
                actual: "b".to_string(),
            }),
        ];

        let stats = DatasetStats::from_test_cases(&test_cases);
        assert_eq!(stats.total_cases, 3);
        assert_eq!(stats.pass_count, 2);
        assert_eq!(stats.fail_count, 1);
        assert!((stats.pass_rate() - 66.666).abs() < 1.0);
    }

    #[test]
    fn test_dataset_stats_empty() {
        let stats = DatasetStats::from_test_cases(&[]);
        assert_eq!(stats.total_cases, 0);
        assert!((stats.pass_rate() - 0.0).abs() < 0.001);
    }
}
