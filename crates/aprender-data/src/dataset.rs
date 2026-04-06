//! Dataset types for alimentar.
//!
//! Provides the [`Dataset`] trait and [`ArrowDataset`] implementation
//! for working with Arrow-based tabular data.

use std::{path::Path, sync::Arc};

use arrow::{array::RecordBatch, datatypes::SchemaRef};
use parquet::{
    arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter},
    file::properties::WriterProperties,
};

use crate::{
    error::{Error, Result},
    transform::Transform,
};

/// A dataset that can be iterated over.
///
/// Datasets provide access to tabular data stored as Arrow RecordBatches.
/// All implementations must be thread-safe (Send + Sync).
pub trait Dataset: Send + Sync {
    /// Returns the total number of rows in the dataset.
    fn len(&self) -> usize;

    /// Returns true if the dataset contains no rows.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a single row as a RecordBatch with one row.
    ///
    /// Returns `None` if the index is out of bounds.
    fn get(&self, index: usize) -> Option<RecordBatch>;

    /// Returns the schema of the dataset.
    fn schema(&self) -> SchemaRef;

    /// Returns an iterator over all RecordBatches in the dataset.
    fn iter(&self) -> Box<dyn Iterator<Item = RecordBatch> + Send + '_>;

    /// Returns the number of batches in the dataset.
    fn num_batches(&self) -> usize;

    /// Returns a specific batch by index.
    fn get_batch(&self, index: usize) -> Option<&RecordBatch>;
}

/// An in-memory dataset backed by Arrow RecordBatches.
///
/// This is the primary dataset type for alimentar. It stores data as a
/// collection of RecordBatches and provides efficient access patterns
/// for ML training loops.
///
/// # Example
///
/// ```no_run
/// use alimentar::{ArrowDataset, Dataset};
///
/// // Load from parquet
/// let dataset = ArrowDataset::from_parquet("data.parquet").unwrap();
/// println!("Dataset has {} rows", dataset.len());
/// ```
#[derive(Debug, Clone)]
pub struct ArrowDataset {
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
    row_count: usize,
}

impl ArrowDataset {
    /// Creates a new ArrowDataset from a vector of RecordBatches.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The batches vector is empty
    /// - The batches have inconsistent schemas
    pub fn new(batches: Vec<RecordBatch>) -> Result<Self> {
        if batches.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let schema = batches[0].schema();

        // Verify all batches have the same schema
        for (i, batch) in batches.iter().enumerate().skip(1) {
            if batch.schema() != schema {
                return Err(Error::schema_mismatch(format!(
                    "Batch {} has different schema than batch 0",
                    i
                )));
            }
        }

        let row_count = batches.iter().map(|b| b.num_rows()).sum();

        Ok(Self {
            batches,
            schema,
            row_count,
        })
    }

    /// Creates an ArrowDataset from a single RecordBatch.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is empty.
    pub fn from_batch(batch: RecordBatch) -> Result<Self> {
        Self::new(vec![batch])
    }

    /// Loads a dataset from a Parquet file.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened
    /// - The file is not valid Parquet
    /// - The file is empty
    pub fn from_parquet(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|e| Error::io(e, path))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(Error::Parquet)?;

        let reader = builder.build().map_err(Error::Parquet)?;

        let batches: Vec<RecordBatch> = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::Arrow)?;

        if batches.is_empty() {
            return Err(Error::EmptyDataset);
        }

        Self::new(batches)
    }

    /// Saves the dataset to a Parquet file.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be created
    /// - Writing fails
    pub fn to_parquet(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let file = std::fs::File::create(path).map_err(|e| Error::io(e, path))?;

        let props = WriterProperties::builder().build();
        let mut writer =
            ArrowWriter::try_new(file, self.schema.clone(), Some(props)).map_err(Error::Parquet)?;

        for batch in &self.batches {
            writer.write(batch).map_err(Error::Parquet)?;
        }

        writer.close().map_err(Error::Parquet)?;
        Ok(())
    }

    /// Loads a dataset from an Arrow IPC file (Issue #2: Enhanced Data Loading)
    ///
    /// Arrow IPC (Inter-Process Communication) format enables zero-copy data
    /// sharing. This is the native Arrow file format with optimal read
    /// performance.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the Arrow IPC file (typically .arrow or .ipc
    ///   extension)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened
    /// - The file is not valid Arrow IPC format
    /// - The file is empty
    ///
    /// # Example
    ///
    /// ```ignore
    /// let dataset = ArrowDataset::from_ipc("data.arrow").unwrap();
    /// ```
    pub fn from_ipc(path: impl AsRef<Path>) -> Result<Self> {
        use arrow::ipc::reader::FileReader;

        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|e| Error::io(e, path))?;

        let reader = FileReader::try_new(file, None).map_err(Error::Arrow)?;

        let batches: Vec<RecordBatch> = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::Arrow)?;

        if batches.is_empty() {
            return Err(Error::EmptyDataset);
        }

        Self::new(batches)
    }

    /// Saves the dataset to an Arrow IPC file (Issue #2: Enhanced Data Loading)
    ///
    /// Creates a file in Arrow IPC format, the native Arrow format.
    /// This format provides optimal read performance for Arrow-based tools.
    ///
    /// # Arguments
    ///
    /// * `path` - Path for the output file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or writing fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// dataset.to_ipc("output.arrow").unwrap();
    /// ```
    pub fn to_ipc(&self, path: impl AsRef<Path>) -> Result<()> {
        use arrow::ipc::writer::FileWriter;

        let path = path.as_ref();
        let file = std::fs::File::create(path).map_err(|e| Error::io(e, path))?;

        let mut writer = FileWriter::try_new(file, &self.schema).map_err(Error::Arrow)?;

        for batch in &self.batches {
            writer.write(batch).map_err(Error::Arrow)?;
        }

        writer.finish().map_err(Error::Arrow)?;
        Ok(())
    }

    /// Loads a dataset from an Arrow IPC stream file (Issue #2: Enhanced Data
    /// Loading)
    ///
    /// Arrow IPC streaming format is designed for streaming scenarios where the
    /// schema is sent first, followed by record batches. The file extension is
    /// typically .arrows.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the Arrow IPC stream file
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails or the file is empty.
    pub fn from_ipc_stream(path: impl AsRef<Path>) -> Result<Self> {
        use arrow::ipc::reader::StreamReader;

        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|e| Error::io(e, path))?;

        let reader = StreamReader::try_new(file, None).map_err(Error::Arrow)?;

        let batches: Vec<RecordBatch> = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::Arrow)?;

        if batches.is_empty() {
            return Err(Error::EmptyDataset);
        }

        Self::new(batches)
    }

    /// Saves the dataset to an Arrow IPC stream file (Issue #2: Enhanced Data
    /// Loading)
    ///
    /// Creates a file in Arrow IPC streaming format. This format is suitable
    /// for streaming scenarios and produces slightly smaller files than the
    /// standard IPC file format.
    ///
    /// # Arguments
    ///
    /// * `path` - Path for the output file (typically .arrows extension)
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or writing fails.
    pub fn to_ipc_stream(&self, path: impl AsRef<Path>) -> Result<()> {
        use arrow::ipc::writer::StreamWriter;

        let path = path.as_ref();
        let file = std::fs::File::create(path).map_err(|e| Error::io(e, path))?;

        let mut writer = StreamWriter::try_new(file, &self.schema).map_err(Error::Arrow)?;

        for batch in &self.batches {
            writer.write(batch).map_err(Error::Arrow)?;
        }

        writer.finish().map_err(Error::Arrow)?;
        Ok(())
    }

    /// Loads a dataset from a CSV file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the CSV file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened
    /// - The file is not valid CSV
    /// - The file is empty
    pub fn from_csv(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_csv_with_options(path, CsvOptions::default())
    }

    /// Loads a dataset from a CSV file with options.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the CSV file
    /// * `options` - CSV parsing options
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails or the file is empty.
    pub fn from_csv_with_options(path: impl AsRef<Path>, options: CsvOptions) -> Result<Self> {
        use std::io::{BufReader, Seek, SeekFrom};

        use arrow_csv::{reader::Format, ReaderBuilder};

        let path = path.as_ref();
        let file = std::fs::File::open(path).map_err(|e| Error::io(e, path))?;
        let mut buf_reader = BufReader::new(file);

        // Get schema (infer or use provided)
        let schema = if let Some(schema) = options.schema {
            Arc::new(schema)
        } else {
            // Infer schema from file
            let mut format = Format::default().with_header(options.has_header);
            if let Some(delim) = options.delimiter {
                format = format.with_delimiter(delim);
            }
            let (inferred, _) = format
                .infer_schema(&mut buf_reader, Some(1000))
                .map_err(Error::Arrow)?;

            // Reset file position
            buf_reader
                .seek(SeekFrom::Start(0))
                .map_err(|e| Error::io(e, path))?;

            Arc::new(inferred)
        };

        let mut builder = ReaderBuilder::new(schema)
            .with_batch_size(options.batch_size)
            .with_header(options.has_header);

        if let Some(delim) = options.delimiter {
            builder = builder.with_delimiter(delim);
        }

        let reader = builder.build(buf_reader).map_err(Error::Arrow)?;

        let batches: Vec<RecordBatch> = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::Arrow)?;

        if batches.is_empty() {
            return Err(Error::EmptyDataset);
        }

        Self::new(batches)
    }

    /// Saves the dataset to a CSV file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or writing fails.
    pub fn to_csv(&self, path: impl AsRef<Path>) -> Result<()> {
        use arrow_csv::WriterBuilder;

        let path = path.as_ref();
        let file = std::fs::File::create(path).map_err(|e| Error::io(e, path))?;

        let mut writer = WriterBuilder::new().with_header(true).build(file);

        for batch in &self.batches {
            writer.write(batch).map_err(Error::Arrow)?;
        }

        Ok(())
    }

    /// Loads a dataset from a JSON Lines (JSONL) file.
    ///
    /// Each line in the file should be a valid JSON object representing a row.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or parsed.
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_json_with_options(path, JsonOptions::default())
    }

    /// Loads a dataset from a JSON Lines file with options.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails or the file is empty.
    pub fn from_json_with_options(path: impl AsRef<Path>, options: JsonOptions) -> Result<Self> {
        use std::io::BufReader;

        use arrow_json::ReaderBuilder;

        let path = path.as_ref();

        // Get schema (infer or use provided)
        let schema = if let Some(schema) = options.schema {
            Arc::new(schema)
        } else {
            // Infer schema from file
            let infer_file = std::fs::File::open(path).map_err(|e| Error::io(e, path))?;
            let infer_reader = BufReader::new(infer_file);
            let (inferred, _) = arrow_json::reader::infer_json_schema(infer_reader, Some(1000))
                .map_err(Error::Arrow)?;
            Arc::new(inferred)
        };

        // Open file for reading
        let file = std::fs::File::open(path).map_err(|e| Error::io(e, path))?;
        let buf_reader = BufReader::new(file);

        let builder = ReaderBuilder::new(schema).with_batch_size(options.batch_size);
        let reader = builder.build(buf_reader).map_err(Error::Arrow)?;

        let batches: Vec<RecordBatch> = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::Arrow)?;

        if batches.is_empty() {
            return Err(Error::EmptyDataset);
        }

        Self::new(batches)
    }

    /// Saves the dataset to a JSON Lines (JSONL) file.
    ///
    /// Each row is written as a single JSON object on its own line.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or writing fails.
    pub fn to_json(&self, path: impl AsRef<Path>) -> Result<()> {
        use std::io::BufWriter;

        use arrow_json::LineDelimitedWriter;

        let path = path.as_ref();
        let file = std::fs::File::create(path).map_err(|e| Error::io(e, path))?;
        let buf_writer = BufWriter::new(file);

        let mut writer = LineDelimitedWriter::new(buf_writer);

        for batch in &self.batches {
            writer.write(batch).map_err(Error::Arrow)?;
        }

        writer.finish().map_err(Error::Arrow)?;

        Ok(())
    }

    /// Loads a dataset from Parquet bytes in memory.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is not valid Parquet.
    pub fn from_parquet_bytes(data: &[u8]) -> Result<Self> {
        use bytes::Bytes;

        let bytes = Bytes::copy_from_slice(data);

        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(Error::Parquet)?;

        let reader = builder.build().map_err(Error::Parquet)?;

        let batches: Vec<RecordBatch> = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::Arrow)?;

        if batches.is_empty() {
            return Err(Error::EmptyDataset);
        }

        Self::new(batches)
    }

    /// Converts the dataset to Parquet bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_parquet_bytes(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        let cursor = std::io::Cursor::new(&mut buffer);

        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(cursor, self.schema.clone(), Some(props))
            .map_err(Error::Parquet)?;

        for batch in &self.batches {
            writer.write(batch).map_err(Error::Parquet)?;
        }

        writer.close().map_err(Error::Parquet)?;
        Ok(buffer)
    }

    /// Loads a dataset from a CSV string.
    ///
    /// # Errors
    ///
    /// Returns an error if the string is not valid CSV.
    pub fn from_csv_str(data: &str) -> Result<Self> {
        use std::io::Cursor;

        use arrow_csv::{reader::Format, ReaderBuilder};

        // Infer schema
        let mut cursor_for_infer = Cursor::new(data.as_bytes());
        let format = Format::default().with_header(true);
        let (inferred, _) = format
            .infer_schema(&mut cursor_for_infer, Some(1000))
            .map_err(Error::Arrow)?;

        let schema = Arc::new(inferred);
        let cursor = Cursor::new(data.as_bytes());

        let builder = ReaderBuilder::new(schema)
            .with_batch_size(8192)
            .with_header(true);

        let reader = builder.build(cursor).map_err(Error::Arrow)?;

        let batches: Vec<RecordBatch> = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::Arrow)?;

        if batches.is_empty() {
            return Err(Error::EmptyDataset);
        }

        Self::new(batches)
    }

    /// Loads a dataset from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if the string is not valid JSON.
    pub fn from_json_str(data: &str) -> Result<Self> {
        use std::io::Cursor;

        use arrow_json::ReaderBuilder;

        // Infer schema
        let cursor_for_infer = Cursor::new(data.as_bytes());
        let (inferred, _) = arrow_json::reader::infer_json_schema(cursor_for_infer, Some(1000))
            .map_err(Error::Arrow)?;

        let schema = Arc::new(inferred);
        let cursor = Cursor::new(data.as_bytes());

        let builder = ReaderBuilder::new(schema).with_batch_size(8192);
        let reader = builder.build(cursor).map_err(Error::Arrow)?;

        let batches: Vec<RecordBatch> = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::Arrow)?;

        if batches.is_empty() {
            return Err(Error::EmptyDataset);
        }

        Self::new(batches)
    }

    /// Returns the underlying batches.
    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }

    /// Consumes the dataset and returns the underlying batches.
    pub fn into_batches(self) -> Vec<RecordBatch> {
        self.batches
    }

    /// Applies a transform to create a new dataset.
    ///
    /// # Errors
    ///
    /// Returns an error if the transform fails on any batch.
    pub fn with_transform<T: Transform>(&self, transform: &T) -> Result<Self> {
        let new_batches: Vec<RecordBatch> = self
            .batches
            .iter()
            .map(|batch| transform.apply(batch.clone()))
            .collect::<Result<Vec<_>>>()?;

        Self::new(new_batches)
    }

    /// Returns an iterator over rows as single-row RecordBatches.
    pub fn rows(&self) -> RowIterator<'_> {
        RowIterator {
            dataset: self,
            current_batch: 0,
            current_row: 0,
        }
    }

    /// Finds the batch and local row index for a global row index.
    fn find_row(&self, global_index: usize) -> Option<(usize, usize)> {
        if global_index >= self.row_count {
            return None;
        }

        let mut remaining = global_index;
        for (batch_idx, batch) in self.batches.iter().enumerate() {
            let batch_rows = batch.num_rows();
            if remaining < batch_rows {
                return Some((batch_idx, remaining));
            }
            remaining -= batch_rows;
        }

        None
    }
}

impl Dataset for ArrowDataset {
    fn len(&self) -> usize {
        self.row_count
    }

    fn get(&self, index: usize) -> Option<RecordBatch> {
        let (batch_idx, local_idx) = self.find_row(index)?;
        let batch = &self.batches[batch_idx];
        Some(batch.slice(local_idx, 1))
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn iter(&self) -> Box<dyn Iterator<Item = RecordBatch> + Send + '_> {
        Box::new(self.batches.iter().cloned())
    }

    fn num_batches(&self) -> usize {
        self.batches.len()
    }

    fn get_batch(&self, index: usize) -> Option<&RecordBatch> {
        self.batches.get(index)
    }
}

/// Iterator over individual rows of a dataset.
pub struct RowIterator<'a> {
    dataset: &'a ArrowDataset,
    current_batch: usize,
    current_row: usize,
}

impl Iterator for RowIterator<'_> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_batch >= self.dataset.batches.len() {
                return None;
            }

            let batch = &self.dataset.batches[self.current_batch];
            if self.current_row < batch.num_rows() {
                let row = batch.slice(self.current_row, 1);
                self.current_row += 1;
                return Some(row);
            }

            self.current_batch += 1;
            self.current_row = 0;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let mut remaining = 0;
        for batch in self.dataset.batches.iter().skip(self.current_batch) {
            remaining += batch.num_rows();
        }
        if self.current_batch < self.dataset.batches.len() {
            remaining -= self.current_row;
        }
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for RowIterator<'_> {}

/// Options for CSV parsing.
#[derive(Debug, Clone)]
pub struct CsvOptions {
    /// Whether the CSV file has a header row.
    pub has_header: bool,
    /// Delimiter character (default is comma).
    pub delimiter: Option<u8>,
    /// Batch size for reading.
    pub batch_size: usize,
    /// Optional schema (inferred if not provided).
    pub schema: Option<arrow::datatypes::Schema>,
}

impl Default for CsvOptions {
    fn default() -> Self {
        Self {
            has_header: true,
            delimiter: None, // Use default comma
            batch_size: 8192,
            schema: None,
        }
    }
}

impl CsvOptions {
    /// Creates new CSV options with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether the file has a header row.
    #[must_use]
    pub fn with_header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    /// Sets the delimiter character.
    #[must_use]
    pub fn with_delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = Some(delimiter);
        self
    }

    /// Sets the batch size for reading.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets the schema for parsing.
    #[must_use]
    pub fn with_schema(mut self, schema: arrow::datatypes::Schema) -> Self {
        self.schema = Some(schema);
        self
    }
}

/// Options for JSON/JSONL parsing.
#[derive(Debug, Clone)]
pub struct JsonOptions {
    /// Batch size for reading.
    pub batch_size: usize,
    /// Optional schema (inferred if not provided).
    pub schema: Option<arrow::datatypes::Schema>,
}

impl Default for JsonOptions {
    fn default() -> Self {
        Self {
            batch_size: 8192,
            schema: None,
        }
    }
}

impl JsonOptions {
    /// Creates new JSON options with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the batch size for reading.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets the schema for parsing.
    #[must_use]
    pub fn with_schema(mut self, schema: arrow::datatypes::Schema) -> Self {
        self.schema = Some(schema);
        self
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::uninlined_format_args
)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
    };

    use super::*;

    fn create_test_batch(start: i32, count: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let ids: Vec<i32> = (start..start + count as i32).collect();
        let names: Vec<String> = ids.iter().map(|i| format!("item_{}", i)).collect();

        let id_array = Int32Array::from(ids);
        let name_array = StringArray::from(names);

        RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(name_array)])
            .ok()
            .unwrap_or_else(|| panic!("Failed to create test batch"))
    }

    #[test]
    fn test_new_dataset() {
        let batch = create_test_batch(0, 10);
        let dataset = ArrowDataset::new(vec![batch]).ok();
        assert!(dataset.is_some());
        let dataset = dataset.unwrap_or_else(|| panic!("Dataset should be Some"));
        assert_eq!(dataset.len(), 10);
    }

    #[test]
    fn test_empty_dataset_error() {
        let result = ArrowDataset::new(vec![]);
        assert!(result.is_err());
        if matches!(result, Err(Error::EmptyDataset)) {
            // Expected
        } else {
            panic!("Expected EmptyDataset error");
        }
    }

    #[test]
    fn test_from_batch() {
        let batch = create_test_batch(0, 5);
        let dataset = ArrowDataset::from_batch(batch).ok();
        assert!(dataset.is_some());
        let dataset = dataset.unwrap_or_else(|| panic!("Dataset should be Some"));
        assert_eq!(dataset.len(), 5);
        assert_eq!(dataset.num_batches(), 1);
    }

    #[test]
    fn test_get_row() {
        let batch = create_test_batch(0, 10);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let row = dataset.get(5);
        assert!(row.is_some());
        let row = row.unwrap_or_else(|| panic!("Row should exist"));
        assert_eq!(row.num_rows(), 1);

        // Out of bounds
        assert!(dataset.get(100).is_none());
    }

    #[test]
    fn test_get_row_across_batches() {
        let batch1 = create_test_batch(0, 5);
        let batch2 = create_test_batch(5, 5);
        let dataset = ArrowDataset::new(vec![batch1, batch2])
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        assert_eq!(dataset.len(), 10);
        assert_eq!(dataset.num_batches(), 2);

        // Row in first batch
        let row = dataset.get(3);
        assert!(row.is_some());

        // Row in second batch
        let row = dataset.get(7);
        assert!(row.is_some());
    }

    #[test]
    fn test_iter() {
        let batch = create_test_batch(0, 10);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let batches: Vec<RecordBatch> = dataset.iter().collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 10);
    }

    #[test]
    fn test_row_iterator() {
        let batch = create_test_batch(0, 5);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let rows: Vec<RecordBatch> = dataset.rows().collect();
        assert_eq!(rows.len(), 5);
        for row in rows {
            assert_eq!(row.num_rows(), 1);
        }
    }

    #[test]
    fn test_row_iterator_exact_size() {
        let batch = create_test_batch(0, 10);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let iter = dataset.rows();
        assert_eq!(iter.len(), 10);
    }

    #[test]
    fn test_schema() {
        let batch = create_test_batch(0, 5);
        let expected_schema = batch.schema();
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        assert_eq!(dataset.schema(), expected_schema);
    }

    #[test]
    fn test_is_empty() {
        let batch = create_test_batch(0, 5);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_get_batch() {
        let batch1 = create_test_batch(0, 5);
        let batch2 = create_test_batch(5, 5);
        let dataset = ArrowDataset::new(vec![batch1, batch2])
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        assert!(dataset.get_batch(0).is_some());
        assert!(dataset.get_batch(1).is_some());
        assert!(dataset.get_batch(2).is_none());
    }

    #[test]
    fn test_into_batches() {
        let batch = create_test_batch(0, 5);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let batches = dataset.into_batches();
        assert_eq!(batches.len(), 1);
    }

    #[test]
    fn test_parquet_roundtrip() {
        let batch = create_test_batch(0, 10);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.parquet");

        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        let loaded = ArrowDataset::from_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should load parquet"));

        assert_eq!(loaded.len(), dataset.len());
        assert_eq!(loaded.schema(), dataset.schema());
    }

    #[test]
    fn test_csv_roundtrip() {
        let batch = create_test_batch(0, 10);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.csv");

        dataset
            .to_csv(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write csv"));

        let loaded = ArrowDataset::from_csv(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should load csv"));

        assert_eq!(loaded.len(), dataset.len());
    }

    #[test]
    fn test_ipc_roundtrip() {
        let batch = create_test_batch(0, 10);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.arrow");

        dataset
            .to_ipc(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write IPC"));

        let loaded = ArrowDataset::from_ipc(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should load IPC"));

        assert_eq!(loaded.len(), dataset.len());
        assert_eq!(loaded.schema(), dataset.schema());
    }

    #[test]
    fn test_ipc_stream_roundtrip() {
        let batch = create_test_batch(0, 10);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.arrows");

        dataset
            .to_ipc_stream(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write IPC stream"));

        let loaded = ArrowDataset::from_ipc_stream(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should load IPC stream"));

        assert_eq!(loaded.len(), dataset.len());
        assert_eq!(loaded.schema(), dataset.schema());
    }

    #[test]
    fn test_ipc_error_nonexistent() {
        let result = ArrowDataset::from_ipc("/nonexistent/path/to/file.arrow");
        assert!(result.is_err());
    }

    #[test]
    fn test_ipc_stream_error_nonexistent() {
        let result = ArrowDataset::from_ipc_stream("/nonexistent/path/to/file.arrows");
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_options() {
        let options = CsvOptions::new()
            .with_header(true)
            .with_delimiter(b',')
            .with_batch_size(1024);

        assert!(options.has_header);
        assert_eq!(options.delimiter, Some(b','));
        assert_eq!(options.batch_size, 1024);
    }

    #[test]
    fn test_csv_options_default() {
        let options = CsvOptions::default();
        assert!(options.has_header);
        assert!(options.delimiter.is_none());
        assert_eq!(options.batch_size, 8192);
        assert!(options.schema.is_none());
    }

    #[test]
    fn test_json_roundtrip() {
        let batch = create_test_batch(0, 10);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.jsonl");

        dataset
            .to_json(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write json"));

        let loaded = ArrowDataset::from_json(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should load json"));

        assert_eq!(loaded.len(), dataset.len());
    }

    #[test]
    fn test_json_options() {
        let options = JsonOptions::new().with_batch_size(1024);

        assert_eq!(options.batch_size, 1024);
        assert!(options.schema.is_none());
    }

    #[test]
    fn test_json_options_default() {
        let options = JsonOptions::default();
        assert_eq!(options.batch_size, 8192);
        assert!(options.schema.is_none());
    }

    #[test]
    fn test_clone() {
        let batch = create_test_batch(0, 5);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let cloned = dataset.clone();
        assert_eq!(cloned.len(), dataset.len());
        assert_eq!(cloned.schema(), dataset.schema());
    }

    #[test]
    fn test_debug() {
        let batch = create_test_batch(0, 5);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let debug_str = format!("{:?}", dataset);
        assert!(debug_str.contains("ArrowDataset"));
    }

    #[test]
    fn test_csv_with_schema() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]);

        let options = CsvOptions::new().with_schema(schema);
        assert!(options.schema.is_some());
    }

    #[test]
    fn test_json_with_schema() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]);

        let options = JsonOptions::new().with_schema(schema);
        assert!(options.schema.is_some());
    }

    #[test]
    fn test_schema_mismatch_error() {
        let schema1 = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let schema2 = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));

        let batch1 = RecordBatch::try_new(schema1, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let batch2 = RecordBatch::try_new(
            schema2,
            vec![Arc::new(StringArray::from(vec!["a", "b", "c"]))],
        )
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"));

        let result = ArrowDataset::new(vec![batch1, batch2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_parquet_error() {
        let result = ArrowDataset::from_parquet("/nonexistent/path/to/file.parquet");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_csv_error() {
        let result = ArrowDataset::from_csv("/nonexistent/path/to/file.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_json_error() {
        let result = ArrowDataset::from_json("/nonexistent/path/to/file.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_with_transform() {
        use crate::transform::Select;

        let batch = create_test_batch(0, 10);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let transform = Select::new(vec!["id"]);
        let transformed = dataset
            .with_transform(&transform)
            .ok()
            .unwrap_or_else(|| panic!("Should apply transform"));

        assert_eq!(transformed.schema().fields().len(), 1);
        assert_eq!(transformed.len(), 10);
    }

    #[test]
    fn test_batches_accessor() {
        let batch = create_test_batch(0, 10);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let batches = dataset.batches();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 10);
    }

    #[test]
    fn test_rows_size_hint() {
        let batch = create_test_batch(0, 10);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let rows = dataset.rows();
        assert_eq!(rows.size_hint(), (10, Some(10)));
    }

    #[test]
    fn test_multiple_batches_iteration() {
        let batch1 = create_test_batch(0, 5);
        let batch2 = create_test_batch(5, 3);
        let batch3 = create_test_batch(8, 2);

        let dataset = ArrowDataset::new(vec![batch1, batch2, batch3])
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let total_rows: usize = dataset.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 10);

        // Test rows iterator across batches
        let row_count = dataset.rows().count();
        assert_eq!(row_count, 10);
    }

    #[test]
    fn test_csv_options_debug() {
        let options = CsvOptions::default();
        let debug_str = format!("{:?}", options);
        assert!(debug_str.contains("CsvOptions"));
    }

    #[test]
    fn test_json_options_debug() {
        let options = JsonOptions::default();
        let debug_str = format!("{:?}", options);
        assert!(debug_str.contains("JsonOptions"));
    }

    // === Additional coverage tests ===

    #[test]
    fn test_from_parquet_bytes_and_to_parquet_bytes() {
        let batch = create_test_batch(0, 10);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let bytes = dataset
            .to_parquet_bytes()
            .ok()
            .unwrap_or_else(|| panic!("Should convert to bytes"));

        let loaded = ArrowDataset::from_parquet_bytes(&bytes)
            .ok()
            .unwrap_or_else(|| panic!("Should load from bytes"));

        assert_eq!(loaded.len(), 10);
        assert_eq!(loaded.schema(), dataset.schema());
    }

    #[test]
    fn test_from_csv_str() {
        let csv = "id,name\n1,Alice\n2,Bob\n3,Charlie";
        let dataset = ArrowDataset::from_csv_str(csv)
            .ok()
            .unwrap_or_else(|| panic!("Should parse CSV string"));

        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.schema().fields().len(), 2);
    }

    #[test]
    fn test_from_json_str() {
        let json = r#"{"id": 1, "name": "Alice"}
{"id": 2, "name": "Bob"}
{"id": 3, "name": "Charlie"}"#;
        let dataset = ArrowDataset::from_json_str(json)
            .ok()
            .unwrap_or_else(|| panic!("Should parse JSON string"));

        assert_eq!(dataset.len(), 3);
    }

    #[test]
    fn test_csv_with_options_custom_delimiter() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.tsv");

        // Write TSV file
        std::fs::write(&path, "id\tname\n1\tAlice\n2\tBob\n3\tCharlie")
            .ok()
            .unwrap_or_else(|| panic!("Should write TSV"));

        let options = CsvOptions::new().with_delimiter(b'\t');
        let dataset = ArrowDataset::from_csv_with_options(&path, options)
            .ok()
            .unwrap_or_else(|| panic!("Should load TSV"));

        assert_eq!(dataset.len(), 3);
    }

    #[test]
    fn test_csv_without_header() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("no_header.csv");

        // Write CSV without header
        std::fs::write(&path, "1,Alice\n2,Bob\n3,Charlie")
            .ok()
            .unwrap_or_else(|| panic!("Should write CSV"));

        let options = CsvOptions::new().with_header(false);
        let dataset = ArrowDataset::from_csv_with_options(&path, options)
            .ok()
            .unwrap_or_else(|| panic!("Should load CSV"));

        assert_eq!(dataset.len(), 3);
    }

    #[test]
    fn test_json_with_options_batch_size() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.jsonl");

        let json = r#"{"id": 1, "name": "Alice"}
{"id": 2, "name": "Bob"}
{"id": 3, "name": "Charlie"}"#;
        std::fs::write(&path, json)
            .ok()
            .unwrap_or_else(|| panic!("Should write JSON"));

        let options = JsonOptions::new().with_batch_size(1024);
        let dataset = ArrowDataset::from_json_with_options(&path, options)
            .ok()
            .unwrap_or_else(|| panic!("Should load JSON"));

        assert_eq!(dataset.len(), 3);
    }

    #[test]
    fn test_row_iterator_multiple_batches_size_hint() {
        let batch1 = create_test_batch(0, 5);
        let batch2 = create_test_batch(5, 3);
        let batch3 = create_test_batch(8, 2);

        let dataset = ArrowDataset::new(vec![batch1, batch2, batch3])
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let mut iter = dataset.rows();
        assert_eq!(iter.size_hint(), (10, Some(10)));

        // Consume some elements
        iter.next();
        iter.next();
        assert_eq!(iter.size_hint(), (8, Some(8)));

        // Consume more to cross batch boundary
        for _ in 0..3 {
            iter.next();
        }
        assert_eq!(iter.size_hint(), (5, Some(5)));
    }

    #[test]
    fn test_find_row_boundary_conditions() {
        let batch1 = create_test_batch(0, 3);
        let batch2 = create_test_batch(3, 2);

        let dataset = ArrowDataset::new(vec![batch1, batch2])
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        // First row of first batch
        let row0 = dataset.get(0);
        assert!(row0.is_some());

        // Last row of first batch
        let row2 = dataset.get(2);
        assert!(row2.is_some());

        // First row of second batch
        let row3 = dataset.get(3);
        assert!(row3.is_some());

        // Last row of second batch
        let row4 = dataset.get(4);
        assert!(row4.is_some());

        // Out of bounds
        let row5 = dataset.get(5);
        assert!(row5.is_none());
    }

    #[test]
    fn test_empty_csv_str_error() {
        // Empty CSV should error
        let result = ArrowDataset::from_csv_str("");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_json_str_error() {
        // Empty JSON should error
        let result = ArrowDataset::from_json_str("");
        assert!(result.is_err());
    }

    #[test]
    fn test_dataset_trait_is_empty_for_nonempty() {
        let batch = create_test_batch(0, 1);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        assert!(!<ArrowDataset as Dataset>::is_empty(&dataset));
    }

    #[test]
    fn test_rows_exact_size_iterator_len_exhaustion() {
        let batch = create_test_batch(0, 3);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let mut iter = dataset.rows();
        assert_eq!(iter.len(), 3);

        iter.next();
        assert_eq!(iter.len(), 2);

        iter.next();
        assert_eq!(iter.len(), 1);

        iter.next();
        assert_eq!(iter.len(), 0);

        // Exhausted
        assert!(iter.next().is_none());
        assert_eq!(iter.len(), 0);
    }

    #[test]
    fn test_csv_options_clone() {
        let options = CsvOptions::new()
            .with_header(false)
            .with_delimiter(b';')
            .with_batch_size(512);
        let cloned = options.clone();

        assert_eq!(cloned.has_header, options.has_header);
        assert_eq!(cloned.delimiter, options.delimiter);
        assert_eq!(cloned.batch_size, options.batch_size);
    }

    #[test]
    fn test_json_options_clone() {
        let options = JsonOptions::new().with_batch_size(256);
        let cloned = options.clone();

        assert_eq!(cloned.batch_size, options.batch_size);
    }

    #[test]
    fn test_iter_returns_cloned_batches() {
        let batch = create_test_batch(0, 5);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let mut iter = dataset.iter();
        let first = iter.next();
        assert!(first.is_some());

        // Original dataset should still be usable
        assert_eq!(dataset.len(), 5);
    }

    #[test]
    fn test_row_iterator_empty_batch_handling() {
        // Create batches where one might be empty-ish
        let batch1 = create_test_batch(0, 2);
        let batch2 = create_test_batch(2, 1);

        let dataset = ArrowDataset::new(vec![batch1, batch2])
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let rows: Vec<_> = dataset.rows().collect();
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn test_large_batch_count() {
        let batches: Vec<RecordBatch> = (0..20).map(|i| create_test_batch(i * 2, 2)).collect();

        let dataset = ArrowDataset::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        assert_eq!(dataset.len(), 40);
        assert_eq!(dataset.num_batches(), 20);

        // Access row from last batch
        let row = dataset.get(39);
        assert!(row.is_some());
    }

    #[test]
    fn test_csv_write_and_verify_content() {
        let batch = create_test_batch(0, 3);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("verify.csv");

        dataset
            .to_csv(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write CSV"));

        // Read back and verify
        let content = std::fs::read_to_string(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        assert!(content.contains("id"));
        assert!(content.contains("name"));
    }

    #[test]
    fn test_json_write_and_verify_content() {
        let batch = create_test_batch(0, 2);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("verify.jsonl");

        dataset
            .to_json(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write JSON"));

        // Read back and verify
        let content = std::fs::read_to_string(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should read file"));
        assert!(content.contains("\"id\""));
        assert!(content.contains("\"name\""));
    }

    #[test]
    fn test_to_ipc_error_invalid_path() {
        let batch = create_test_batch(0, 5);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let result = dataset.to_ipc("/nonexistent/path/output.arrow");
        assert!(result.is_err());
    }

    #[test]
    fn test_to_ipc_stream_error_invalid_path() {
        let batch = create_test_batch(0, 5);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let result = dataset.to_ipc_stream("/nonexistent/path/output.arrows");
        assert!(result.is_err());
    }

    #[test]
    fn test_to_parquet_error_invalid_path() {
        let batch = create_test_batch(0, 5);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let result = dataset.to_parquet("/nonexistent/path/output.parquet");
        assert!(result.is_err());
    }

    #[test]
    fn test_to_csv_error_invalid_path() {
        let batch = create_test_batch(0, 5);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let result = dataset.to_csv("/nonexistent/path/output.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_to_json_error_invalid_path() {
        let batch = create_test_batch(0, 5);
        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let result = dataset.to_json("/nonexistent/path/output.jsonl");
        assert!(result.is_err());
    }
}
