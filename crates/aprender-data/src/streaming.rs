//! Streaming dataset for lazy/chunked data loading.
//!
//! Provides [`StreamingDataset`] for working with datasets that are too large
//! to fit in memory, or when you want to start processing before the full
//! dataset is loaded.

use std::{collections::VecDeque, path::Path, sync::Arc};

use arrow::{array::RecordBatch, datatypes::SchemaRef};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::{Error, Result};

/// A data source that can produce RecordBatches on demand.
pub trait DataSource: Send {
    /// Returns the schema of the data.
    fn schema(&self) -> SchemaRef;

    /// Returns the next batch of data, or None if exhausted.
    ///
    /// # Errors
    ///
    /// Returns an error if reading the next batch fails.
    fn next_batch(&mut self) -> Result<Option<RecordBatch>>;

    /// Returns an estimate of total rows, if known.
    fn size_hint(&self) -> Option<usize> {
        None
    }

    /// Resets the source to the beginning, if supported.
    ///
    /// # Errors
    ///
    /// Returns an error if this data source does not support reset.
    fn reset(&mut self) -> Result<()> {
        Err(Error::storage("This data source does not support reset"))
    }
}

/// A streaming dataset that loads data lazily in chunks.
///
/// Unlike [`ArrowDataset`](crate::ArrowDataset) which loads all data into
/// memory, `StreamingDataset` fetches data on-demand, making it suitable for:
/// - Large datasets that don't fit in memory
/// - Network-based data sources where you want to start processing early
/// - Infinite or very large data streams
///
/// # Example
///
/// ```no_run
/// use alimentar::streaming::StreamingDataset;
///
/// let dataset = StreamingDataset::from_parquet("large_data.parquet", 1024).unwrap();
///
/// for batch in dataset {
///     println!("Processing batch with {} rows", batch.num_rows());
/// }
/// ```
pub struct StreamingDataset {
    source: Box<dyn DataSource>,
    buffer: VecDeque<RecordBatch>,
    buffer_size: usize,
    prefetch: usize,
    schema: SchemaRef,
    exhausted: bool,
}

impl StreamingDataset {
    /// Creates a new streaming dataset from a data source.
    ///
    /// # Arguments
    ///
    /// * `source` - The data source to stream from
    /// * `buffer_size` - Maximum number of batches to buffer
    ///
    /// #[requires(buffer_size > 0)]
    /// #[ensures(result.buffer_size >= 1)]
    /// #[ensures(result.exhausted == false)]
    /// #[invariant(self.buffer.len() <= self.buffer_size)]
    pub fn new(source: Box<dyn DataSource>, buffer_size: usize) -> Self {
        let schema = source.schema();
        Self {
            source,
            buffer: VecDeque::with_capacity(buffer_size),
            buffer_size: buffer_size.max(1),
            prefetch: 1,
            schema,
            exhausted: false,
        }
    }

    /// Creates a streaming dataset from a Parquet file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the Parquet file
    /// * `batch_size` - Number of rows per batch
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or is invalid.
    pub fn from_parquet(path: impl AsRef<Path>, batch_size: usize) -> Result<Self> {
        let source = ParquetSource::new(path, batch_size)?;
        Ok(Self::new(Box::new(source), 4))
    }

    /// Sets the number of batches to prefetch.
    ///
    /// Higher values reduce latency but increase memory usage.
    #[must_use]
    pub fn prefetch(mut self, count: usize) -> Self {
        self.prefetch = count.max(1);
        self
    }

    /// Returns the schema of the dataset.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// Returns an estimate of total rows, if known.
    pub fn size_hint(&self) -> Option<usize> {
        self.source.size_hint()
    }

    /// Fills the buffer up to the prefetch limit.
    fn fill_buffer(&mut self) -> Result<()> {
        while !self.exhausted && self.buffer.len() < self.prefetch {
            if let Some(batch) = self.source.next_batch()? {
                self.buffer.push_back(batch);
            } else {
                self.exhausted = true;
                break;
            }
        }
        Ok(())
    }

    /// Resets the dataset to the beginning, if the source supports it.
    ///
    /// # Errors
    ///
    /// Returns an error if the source does not support reset.
    pub fn reset(&mut self) -> Result<()> {
        self.source.reset()?;
        self.buffer.clear();
        self.exhausted = false;
        Ok(())
    }
}

impl Iterator for StreamingDataset {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        // Try to fill buffer first
        if let Err(_e) = self.fill_buffer() {
            return None;
        }

        self.buffer.pop_front()
    }
}

impl std::fmt::Debug for StreamingDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingDataset")
            .field("buffer_size", &self.buffer_size)
            .field("prefetch", &self.prefetch)
            .field("buffered", &self.buffer.len())
            .field("exhausted", &self.exhausted)
            .finish_non_exhaustive()
    }
}

/// A data source that reads from a Parquet file.
pub struct ParquetSource {
    reader: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    schema: SchemaRef,
    path: std::path::PathBuf,
    batch_size: usize,
}

impl ParquetSource {
    /// Creates a new Parquet source.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened.
    pub fn new(path: impl AsRef<Path>, batch_size: usize) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path).map_err(|e| Error::io(e, &path))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(Error::Parquet)?
            .with_batch_size(batch_size);

        let schema = builder.schema().clone();
        let reader = builder.build().map_err(Error::Parquet)?;

        Ok(Self {
            reader,
            schema,
            path,
            batch_size,
        })
    }
}

impl DataSource for ParquetSource {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        match self.reader.next() {
            Some(Ok(batch)) => Ok(Some(batch)),
            Some(Err(e)) => Err(Error::Arrow(e)),
            None => Ok(None),
        }
    }

    fn reset(&mut self) -> Result<()> {
        // Re-open the file
        let file = std::fs::File::open(&self.path).map_err(|e| Error::io(e, &self.path))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(Error::Parquet)?
            .with_batch_size(self.batch_size);

        self.reader = builder.build().map_err(Error::Parquet)?;
        Ok(())
    }
}

impl std::fmt::Debug for ParquetSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetSource")
            .field("path", &self.path)
            .field("batch_size", &self.batch_size)
            .finish_non_exhaustive()
    }
}

/// A data source backed by in-memory RecordBatches.
///
/// Useful for testing or when you have data already in memory
/// but want to use the streaming interface.
#[derive(Debug)]
pub struct MemorySource {
    batches: Vec<RecordBatch>,
    schema: SchemaRef,
    position: usize,
}

impl MemorySource {
    /// Creates a new memory source from a vector of batches.
    ///
    /// # Errors
    ///
    /// Returns an error if the batches vector is empty.
    pub fn new(batches: Vec<RecordBatch>) -> Result<Self> {
        if batches.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let schema = batches[0].schema();
        Ok(Self {
            batches,
            schema,
            position: 0,
        })
    }
}

impl DataSource for MemorySource {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        if self.position >= self.batches.len() {
            return Ok(None);
        }

        let batch = self.batches[self.position].clone();
        self.position += 1;
        Ok(Some(batch))
    }

    fn size_hint(&self) -> Option<usize> {
        Some(self.batches.iter().map(|b| b.num_rows()).sum())
    }

    fn reset(&mut self) -> Result<()> {
        self.position = 0;
        Ok(())
    }
}

/// A data source that chains multiple sources together.
pub struct ChainedSource {
    sources: Vec<Box<dyn DataSource>>,
    current: usize,
    schema: SchemaRef,
}

impl ChainedSource {
    /// Creates a new chained source.
    ///
    /// # Errors
    ///
    /// Returns an error if the sources vector is empty.
    pub fn new(sources: Vec<Box<dyn DataSource>>) -> Result<Self> {
        if sources.is_empty() {
            return Err(Error::invalid_config(
                "ChainedSource requires at least one source",
            ));
        }

        let schema = sources[0].schema();
        Ok(Self {
            sources,
            current: 0,
            schema,
        })
    }
}

impl DataSource for ChainedSource {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        while self.current < self.sources.len() {
            match self.sources[self.current].next_batch()? {
                Some(batch) => return Ok(Some(batch)),
                None => self.current += 1,
            }
        }
        Ok(None)
    }

    fn size_hint(&self) -> Option<usize> {
        let mut total = 0;
        for source in &self.sources {
            match source.size_hint() {
                Some(hint) => total += hint,
                None => return None,
            }
        }
        Some(total)
    }

    fn reset(&mut self) -> Result<()> {
        for source in &mut self.sources {
            source.reset()?;
        }
        self.current = 0;
        Ok(())
    }
}

impl std::fmt::Debug for ChainedSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChainedSource")
            .field("num_sources", &self.sources.len())
            .field("current", &self.current)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
mod tests {
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

        let ids: Vec<i32> = (start..start + count as i32).collect();
        let names: Vec<String> = ids.iter().map(|i| format!("item_{}", i)).collect();

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"))
    }

    #[test]
    fn test_memory_source() {
        let batches = vec![
            create_test_batch(0, 5),
            create_test_batch(5, 5),
            create_test_batch(10, 5),
        ];

        let mut source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        assert_eq!(source.size_hint(), Some(15));

        let mut count = 0;
        while let Ok(Some(batch)) = source.next_batch() {
            count += batch.num_rows();
        }
        assert_eq!(count, 15);
    }

    #[test]
    fn test_memory_source_reset() {
        let batches = vec![create_test_batch(0, 5)];

        let mut source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        // Consume
        let _ = source.next_batch();
        assert!(source.next_batch().ok().flatten().is_none());

        // Reset and consume again
        source
            .reset()
            .ok()
            .unwrap_or_else(|| panic!("Should reset"));
        assert!(source.next_batch().ok().flatten().is_some());
    }

    #[test]
    fn test_streaming_dataset_from_memory() {
        let batches = vec![create_test_batch(0, 10), create_test_batch(10, 10)];

        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let dataset = StreamingDataset::new(Box::new(source), 4);

        let mut total = 0;
        for batch in dataset {
            total += batch.num_rows();
        }
        assert_eq!(total, 20);
    }

    #[test]
    fn test_streaming_dataset_prefetch() {
        let batches = vec![
            create_test_batch(0, 5),
            create_test_batch(5, 5),
            create_test_batch(10, 5),
        ];

        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let dataset = StreamingDataset::new(Box::new(source), 4).prefetch(2);

        let collected: Vec<RecordBatch> = dataset.collect();
        assert_eq!(collected.len(), 3);
    }

    #[test]
    fn test_streaming_dataset_schema() {
        let batches = vec![create_test_batch(0, 5)];

        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let dataset = StreamingDataset::new(Box::new(source), 4);

        assert_eq!(dataset.schema().fields().len(), 2);
        assert_eq!(dataset.schema().field(0).name(), "id");
    }

    #[test]
    fn test_streaming_dataset_reset() {
        let batches = vec![create_test_batch(0, 5)];

        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let mut dataset = StreamingDataset::new(Box::new(source), 4);

        // Consume
        let first: Vec<RecordBatch> = dataset.by_ref().collect();
        assert_eq!(first.len(), 1);

        // Reset
        dataset
            .reset()
            .ok()
            .unwrap_or_else(|| panic!("Should reset"));

        // Consume again
        let second: Vec<RecordBatch> = dataset.collect();
        assert_eq!(second.len(), 1);
    }

    #[test]
    fn test_chained_source() {
        let source1 = MemorySource::new(vec![create_test_batch(0, 5)])
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));
        let source2 = MemorySource::new(vec![create_test_batch(5, 5)])
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let mut chained = ChainedSource::new(vec![Box::new(source1), Box::new(source2)])
            .ok()
            .unwrap_or_else(|| panic!("Should create chained"));

        assert_eq!(chained.size_hint(), Some(10));

        let mut total = 0;
        while let Ok(Some(batch)) = chained.next_batch() {
            total += batch.num_rows();
        }
        assert_eq!(total, 10);
    }

    #[test]
    fn test_empty_memory_source_error() {
        let result = MemorySource::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_chained_source_error() {
        let result = ChainedSource::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parquet_source_roundtrip() {
        // Create test data
        let batch = create_test_batch(0, 100);
        let dataset = crate::ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        // Write to temp file
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.parquet");
        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        // Read back via streaming
        let streaming = StreamingDataset::from_parquet(&path, 25)
            .ok()
            .unwrap_or_else(|| panic!("Should create streaming"));

        let total: usize = streaming.map(|b| b.num_rows()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_streaming_dataset_debug() {
        let batches = vec![create_test_batch(0, 5)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));
        let dataset = StreamingDataset::new(Box::new(source), 4);

        let debug_str = format!("{:?}", dataset);
        assert!(debug_str.contains("StreamingDataset"));
    }

    #[test]
    fn test_parquet_source_debug() {
        // Create test data
        let batch = create_test_batch(0, 10);
        let dataset = crate::ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("debug_test.parquet");
        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        let source = ParquetSource::new(&path, 10)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let debug_str = format!("{:?}", source);
        assert!(debug_str.contains("ParquetSource"));
        assert!(debug_str.contains("debug_test.parquet"));
    }

    #[test]
    fn test_chained_source_debug() {
        let source1 = MemorySource::new(vec![create_test_batch(0, 5)])
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let chained = ChainedSource::new(vec![Box::new(source1)])
            .ok()
            .unwrap_or_else(|| panic!("Should create chained"));

        let debug_str = format!("{:?}", chained);
        assert!(debug_str.contains("ChainedSource"));
        assert!(debug_str.contains("num_sources"));
    }

    #[test]
    fn test_streaming_dataset_size_hint() {
        let batches = vec![create_test_batch(0, 10), create_test_batch(10, 15)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let dataset = StreamingDataset::new(Box::new(source), 4);
        assert_eq!(dataset.size_hint(), Some(25));
    }

    #[test]
    fn test_parquet_source_reset() {
        // Create test data
        let batch = create_test_batch(0, 50);
        let dataset = crate::ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("reset_test.parquet");
        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        let mut source = ParquetSource::new(&path, 10)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        // Read all batches
        let mut count = 0;
        while let Ok(Some(_)) = source.next_batch() {
            count += 1;
        }
        assert!(count > 0);

        // Reset and read again
        source
            .reset()
            .ok()
            .unwrap_or_else(|| panic!("Should reset"));

        let mut count2 = 0;
        while let Ok(Some(_)) = source.next_batch() {
            count2 += 1;
        }
        assert_eq!(count, count2);
    }

    #[test]
    fn test_chained_source_reset() {
        let source1 = MemorySource::new(vec![create_test_batch(0, 5)])
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));
        let source2 = MemorySource::new(vec![create_test_batch(5, 5)])
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let mut chained = ChainedSource::new(vec![Box::new(source1), Box::new(source2)])
            .ok()
            .unwrap_or_else(|| panic!("Should create chained"));

        // Exhaust
        let mut count = 0;
        while let Ok(Some(_)) = chained.next_batch() {
            count += 1;
        }
        assert_eq!(count, 2);

        // Reset
        chained
            .reset()
            .ok()
            .unwrap_or_else(|| panic!("Should reset"));

        // Read again
        let mut count2 = 0;
        while let Ok(Some(_)) = chained.next_batch() {
            count2 += 1;
        }
        assert_eq!(count2, 2);
    }

    #[test]
    fn test_chained_source_size_hint_unknown() {
        // Create a source that doesn't know its size
        struct UnknownSizeSource {
            schema: SchemaRef,
            count: usize,
        }

        impl DataSource for UnknownSizeSource {
            fn schema(&self) -> SchemaRef {
                Arc::clone(&self.schema)
            }

            fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
                if self.count > 0 {
                    self.count -= 1;
                    Ok(Some(create_test_batch(0, 1)))
                } else {
                    Ok(None)
                }
            }

            // Returns None, so chained source can't calculate total
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let memory_source = MemorySource::new(vec![create_test_batch(0, 5)])
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let unknown_source = UnknownSizeSource { schema, count: 1 };

        let chained = ChainedSource::new(vec![Box::new(memory_source), Box::new(unknown_source)])
            .ok()
            .unwrap_or_else(|| panic!("Should create chained"));

        // One source has unknown size, so total is None
        assert_eq!(chained.size_hint(), None);
    }

    #[test]
    fn test_streaming_dataset_buffer_size_minimum() {
        let batches = vec![create_test_batch(0, 5)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        // Buffer size 0 should be treated as 1
        let dataset = StreamingDataset::new(Box::new(source), 0);
        let collected: Vec<RecordBatch> = dataset.collect();
        assert_eq!(collected.len(), 1);
    }

    #[test]
    fn test_streaming_dataset_prefetch_minimum() {
        let batches = vec![create_test_batch(0, 5), create_test_batch(5, 5)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        // Prefetch 0 should be treated as 1
        let dataset = StreamingDataset::new(Box::new(source), 4).prefetch(0);
        let collected: Vec<RecordBatch> = dataset.collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn test_memory_source_debug() {
        let batches = vec![create_test_batch(0, 5)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let debug_str = format!("{:?}", source);
        assert!(debug_str.contains("MemorySource"));
    }

    #[test]
    fn test_parquet_source_schema() {
        let batch = create_test_batch(0, 10);
        let dataset = crate::ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("schema_test.parquet");
        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        let source = ParquetSource::new(&path, 10)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let schema = source.schema();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "id");
    }

    #[test]
    fn test_chained_source_schema() {
        let source = MemorySource::new(vec![create_test_batch(0, 5)])
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let chained = ChainedSource::new(vec![Box::new(source)])
            .ok()
            .unwrap_or_else(|| panic!("Should create chained"));

        let schema = chained.schema();
        assert_eq!(schema.fields().len(), 2);
    }

    #[test]
    fn test_parquet_source_file_not_found() {
        let result = ParquetSource::new("/nonexistent/path/to/file.parquet", 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_dataset_from_parquet_not_found() {
        let result = StreamingDataset::from_parquet("/nonexistent/file.parquet", 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_data_source_default_reset_error() {
        // Test that the default DataSource::reset returns error
        struct NoResetSource {
            schema: SchemaRef,
        }

        impl DataSource for NoResetSource {
            fn schema(&self) -> SchemaRef {
                Arc::clone(&self.schema)
            }

            fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
                Ok(None)
            }
            // Don't override reset() - uses default impl that returns error
        }

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

        let mut source = NoResetSource { schema };
        let result = source.reset();
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_dataset_reset_unsupported() {
        // Test that reset fails when source doesn't support it
        struct NoResetSource {
            schema: SchemaRef,
            done: bool,
        }

        impl DataSource for NoResetSource {
            fn schema(&self) -> SchemaRef {
                Arc::clone(&self.schema)
            }

            fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
                if self.done {
                    Ok(None)
                } else {
                    self.done = true;
                    Ok(Some(create_test_batch(0, 5)))
                }
            }
            // Uses default reset which fails
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let source = NoResetSource {
            schema,
            done: false,
        };
        let mut dataset = StreamingDataset::new(Box::new(source), 4);

        // Consume
        let _: Vec<_> = dataset.by_ref().collect();

        // Reset should fail
        let result = dataset.reset();
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_dataset_fill_buffer_error() {
        // Test that iterator returns None on fill_buffer error
        struct ErrorSource {
            schema: SchemaRef,
            error_on_call: usize,
            call_count: usize,
        }

        impl DataSource for ErrorSource {
            fn schema(&self) -> SchemaRef {
                Arc::clone(&self.schema)
            }

            fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
                self.call_count += 1;
                if self.call_count >= self.error_on_call {
                    Err(crate::Error::storage("Simulated error"))
                } else {
                    Ok(Some(create_test_batch(0, 5)))
                }
            }
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let source = ErrorSource {
            schema,
            error_on_call: 3, // Error on 3rd call (after 2 successful batches)
            call_count: 0,
        };

        // Use prefetch 1 so we get 1 batch at a time
        let mut dataset = StreamingDataset::new(Box::new(source), 4).prefetch(1);

        // First two calls should succeed
        let first = dataset.next();
        assert!(first.is_some());
        let second = dataset.next();
        assert!(second.is_some());

        // Third call triggers error in fill_buffer, returns None
        let third = dataset.next();
        assert!(third.is_none());
    }

    #[test]
    fn test_streaming_dataset_large_prefetch() {
        let batches = vec![
            create_test_batch(0, 10),
            create_test_batch(10, 10),
            create_test_batch(20, 10),
            create_test_batch(30, 10),
            create_test_batch(40, 10),
        ];

        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        // Prefetch more than available batches
        let dataset = StreamingDataset::new(Box::new(source), 10).prefetch(100);

        let collected: Vec<RecordBatch> = dataset.collect();
        assert_eq!(collected.len(), 5);
    }

    #[test]
    fn test_memory_source_multiple_iterations() {
        let batches = vec![create_test_batch(0, 5), create_test_batch(5, 5)];

        let mut source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        // First iteration
        let mut count1 = 0;
        while let Ok(Some(_)) = source.next_batch() {
            count1 += 1;
        }
        assert_eq!(count1, 2);

        // Reset and iterate again
        source
            .reset()
            .ok()
            .unwrap_or_else(|| panic!("Should reset"));

        let mut count2 = 0;
        while let Ok(Some(_)) = source.next_batch() {
            count2 += 1;
        }
        assert_eq!(count2, 2);
    }

    #[test]
    fn test_chained_source_exhaustion() {
        let source1 = MemorySource::new(vec![create_test_batch(0, 3)])
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));
        let source2 = MemorySource::new(vec![create_test_batch(3, 2)])
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));
        let source3 = MemorySource::new(vec![create_test_batch(5, 1)])
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let mut chained = ChainedSource::new(vec![
            Box::new(source1),
            Box::new(source2),
            Box::new(source3),
        ])
        .ok()
        .unwrap_or_else(|| panic!("Should create chained"));

        let mut batches = Vec::new();
        while let Ok(Some(batch)) = chained.next_batch() {
            batches.push(batch);
        }

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].num_rows(), 3);
        assert_eq!(batches[1].num_rows(), 2);
        assert_eq!(batches[2].num_rows(), 1);
    }

    #[test]
    fn test_streaming_dataset_empty_iteration() {
        struct EmptySource {
            schema: SchemaRef,
        }

        impl DataSource for EmptySource {
            fn schema(&self) -> SchemaRef {
                Arc::clone(&self.schema)
            }

            fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
                Ok(None)
            }
        }

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

        let source = EmptySource { schema };
        let dataset = StreamingDataset::new(Box::new(source), 4);

        let collected: Vec<RecordBatch> = dataset.collect();
        assert!(collected.is_empty());
    }

    #[test]
    fn test_streaming_dataset_single_batch() {
        let batches = vec![create_test_batch(0, 100)];

        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let dataset = StreamingDataset::new(Box::new(source), 1);

        let collected: Vec<RecordBatch> = dataset.collect();
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0].num_rows(), 100);
    }

    #[test]
    fn test_parquet_source_batch_size_variation() {
        // Create test data
        let batch = create_test_batch(0, 100);
        let dataset = crate::ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("batch_size_test.parquet");
        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        // Test with different batch sizes
        for batch_size in [1, 10, 50, 100, 200] {
            let source = ParquetSource::new(&path, batch_size)
                .ok()
                .unwrap_or_else(|| panic!("Should create source"));

            let streaming = StreamingDataset::new(Box::new(source), 4);
            let total: usize = streaming.map(|b| b.num_rows()).sum();
            assert_eq!(
                total, 100,
                "Batch size {} should read all 100 rows",
                batch_size
            );
        }
    }

    #[test]
    fn test_chained_source_single_source() {
        let batches = vec![create_test_batch(0, 50)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        let chained = ChainedSource::new(vec![Box::new(source)])
            .ok()
            .unwrap_or_else(|| panic!("chained"));
        let dataset = StreamingDataset::new(Box::new(chained), 4);

        let collected: Vec<RecordBatch> = dataset.collect();
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0].num_rows(), 50);
    }

    #[test]
    fn test_chained_source_multiple_sources() {
        let source1 = MemorySource::new(vec![create_test_batch(0, 30)])
            .ok()
            .unwrap_or_else(|| panic!("source1"));
        let source2 = MemorySource::new(vec![create_test_batch(30, 20)])
            .ok()
            .unwrap_or_else(|| panic!("source2"));

        let chained = ChainedSource::new(vec![Box::new(source1), Box::new(source2)])
            .ok()
            .unwrap_or_else(|| panic!("chained"));
        let dataset = StreamingDataset::new(Box::new(chained), 4);

        let collected: Vec<RecordBatch> = dataset.collect();
        assert_eq!(collected.len(), 2);

        let total: usize = collected.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 50);
    }

    #[test]
    fn test_chained_source_empty_sources_vec() {
        // Empty sources vec should return an error
        let result: std::result::Result<ChainedSource, Error> = ChainedSource::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_chained_source_with_empty_yielding_sources() {
        struct EmptyYieldSource {
            schema: SchemaRef,
        }
        impl DataSource for EmptyYieldSource {
            fn schema(&self) -> SchemaRef {
                Arc::clone(&self.schema)
            }
            fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
                Ok(None)
            }
        }

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

        let source1 = EmptyYieldSource {
            schema: Arc::clone(&schema),
        };
        let source2 = EmptyYieldSource { schema };

        let chained = ChainedSource::new(vec![Box::new(source1), Box::new(source2)])
            .ok()
            .unwrap_or_else(|| panic!("chained"));
        let dataset = StreamingDataset::new(Box::new(chained), 4);

        let collected: Vec<RecordBatch> = dataset.collect();
        assert!(collected.is_empty());
    }

    #[test]
    fn test_streaming_dataset_prefetch_config() {
        let batches = vec![create_test_batch(0, 100)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        let dataset = StreamingDataset::new(Box::new(source), 4).prefetch(2);

        let collected: Vec<RecordBatch> = dataset.collect();
        assert_eq!(collected.len(), 1);
    }

    #[test]
    fn test_chained_source_size_hint() {
        let source1 = MemorySource::new(vec![create_test_batch(0, 50)])
            .ok()
            .unwrap_or_else(|| panic!("source1"));
        let source2 = MemorySource::new(vec![create_test_batch(50, 50)])
            .ok()
            .unwrap_or_else(|| panic!("source2"));

        let chained = ChainedSource::new(vec![Box::new(source1), Box::new(source2)])
            .ok()
            .unwrap_or_else(|| panic!("chained"));

        // Should sum size hints from all sources
        assert_eq!(chained.size_hint(), Some(100));
    }

    #[test]
    fn test_streaming_dataset_buffer_size_one() {
        let batches = vec![
            create_test_batch(0, 25),
            create_test_batch(25, 25),
            create_test_batch(50, 25),
            create_test_batch(75, 25),
        ];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        // Minimal buffer
        let dataset = StreamingDataset::new(Box::new(source), 1);

        let collected: Vec<RecordBatch> = dataset.collect();
        assert_eq!(collected.len(), 4);

        let total: usize = collected.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_parquet_source_invalid_path() {
        let result = ParquetSource::new("/nonexistent/path/to/file.parquet", 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_memory_source_schema_consistency() {
        let batches = vec![create_test_batch(0, 50), create_test_batch(50, 50)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        let schema = source.schema();
        assert_eq!(schema.fields().len(), 2); // id and name columns
    }

    // === Additional coverage tests for streaming module ===

    #[test]
    fn test_parquet_source_next_batch_error_handling() {
        // Create a valid parquet file first
        let batch = create_test_batch(0, 10);
        let dataset = crate::ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("temp dir"));
        let path = temp_dir.path().join("next_batch_test.parquet");
        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("parquet"));

        let mut source = ParquetSource::new(&path, 5)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        // Read all batches
        let mut batches = Vec::new();
        while let Ok(Some(batch)) = source.next_batch() {
            batches.push(batch);
        }
        assert!(!batches.is_empty());

        // After exhaustion, should return None
        let next = source.next_batch();
        assert!(next.is_ok());
        assert!(next.ok().flatten().is_none());
    }

    #[test]
    fn test_memory_source_position_tracking() {
        let batches = vec![
            create_test_batch(0, 3),
            create_test_batch(3, 3),
            create_test_batch(6, 4),
        ];
        let mut source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        // Read first batch
        let b1 = source.next_batch().ok().flatten();
        assert!(b1.is_some());
        assert_eq!(b1.as_ref().map(|b| b.num_rows()), Some(3));

        // Read second batch
        let b2 = source.next_batch().ok().flatten();
        assert!(b2.is_some());

        // Read third batch
        let b3 = source.next_batch().ok().flatten();
        assert!(b3.is_some());

        // No more batches
        let b4 = source.next_batch().ok().flatten();
        assert!(b4.is_none());
    }

    #[test]
    fn test_chained_source_transitions_between_sources() {
        let source1 = MemorySource::new(vec![create_test_batch(0, 2), create_test_batch(2, 2)])
            .ok()
            .unwrap_or_else(|| panic!("source1"));

        let source2 = MemorySource::new(vec![create_test_batch(4, 3)])
            .ok()
            .unwrap_or_else(|| panic!("source2"));

        let mut chained = ChainedSource::new(vec![Box::new(source1), Box::new(source2)])
            .ok()
            .unwrap_or_else(|| panic!("chained"));

        let mut batches = Vec::new();
        while let Ok(Some(batch)) = chained.next_batch() {
            batches.push(batch.num_rows());
        }

        // Should get 3 batches total (2 from source1, 1 from source2)
        assert_eq!(batches.len(), 3);
        assert_eq!(batches, vec![2, 2, 3]);
    }

    #[test]
    fn test_streaming_dataset_exhaustion() {
        let batches = vec![create_test_batch(0, 5)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        let mut dataset = StreamingDataset::new(Box::new(source), 2);

        // Consume all
        let _: Vec<_> = dataset.by_ref().collect();

        // After exhaustion, next should return None
        assert!(dataset.next().is_none());
    }

    #[test]
    fn test_streaming_dataset_schema_preserved() {
        let batches = vec![create_test_batch(0, 10)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        let dataset = StreamingDataset::new(Box::new(source), 4);
        let schema = dataset.schema();

        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "name");
    }

    #[test]
    fn test_chained_source_reset_restores_all() {
        let source1 = MemorySource::new(vec![create_test_batch(0, 10)])
            .ok()
            .unwrap_or_else(|| panic!("source1"));
        let source2 = MemorySource::new(vec![create_test_batch(10, 10)])
            .ok()
            .unwrap_or_else(|| panic!("source2"));

        let mut chained = ChainedSource::new(vec![Box::new(source1), Box::new(source2)])
            .ok()
            .unwrap_or_else(|| panic!("chained"));

        // First iteration
        let count1: usize = std::iter::from_fn(|| chained.next_batch().ok().flatten())
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(count1, 20);

        // Reset
        chained.reset().ok().unwrap_or_else(|| panic!("reset"));

        // Second iteration
        let count2: usize = std::iter::from_fn(|| chained.next_batch().ok().flatten())
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(count2, 20);
    }

    #[test]
    fn test_data_source_default_size_hint() {
        // Test the default size_hint returns None
        struct NoHintSource {
            schema: SchemaRef,
        }

        impl DataSource for NoHintSource {
            fn schema(&self) -> SchemaRef {
                Arc::clone(&self.schema)
            }

            fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
                Ok(None)
            }
            // Uses default size_hint which returns None
        }

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let source = NoHintSource { schema };
        assert!(source.size_hint().is_none());
    }

    #[test]
    fn test_streaming_dataset_large_buffer() {
        let batches = vec![create_test_batch(0, 10), create_test_batch(10, 10)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        // Buffer larger than number of batches
        let dataset = StreamingDataset::new(Box::new(source), 100);
        let collected: Vec<RecordBatch> = dataset.collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn test_parquet_source_small_batch_size() {
        let batch = create_test_batch(0, 100);
        let dataset = crate::ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("temp dir"));
        let path = temp_dir.path().join("small_batch.parquet");
        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("parquet"));

        // Very small batch size
        let source = ParquetSource::new(&path, 1)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        let dataset = StreamingDataset::new(Box::new(source), 10);
        let total: usize = dataset.map(|b| b.num_rows()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_chained_source_first_source_empty() {
        struct EmptySource {
            schema: SchemaRef,
        }

        impl DataSource for EmptySource {
            fn schema(&self) -> SchemaRef {
                Arc::clone(&self.schema)
            }
            fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
                Ok(None)
            }
            fn reset(&mut self) -> Result<()> {
                Ok(())
            }
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let empty = EmptySource { schema };
        let memory = MemorySource::new(vec![create_test_batch(0, 5)])
            .ok()
            .unwrap_or_else(|| panic!("memory"));

        let mut chained = ChainedSource::new(vec![Box::new(empty), Box::new(memory)])
            .ok()
            .unwrap_or_else(|| panic!("chained"));

        // Should skip empty source and get batch from memory source
        let batch = chained.next_batch().ok().flatten();
        assert!(batch.is_some());
        assert_eq!(batch.map(|b| b.num_rows()), Some(5));
    }

    #[test]
    fn test_streaming_dataset_new_initializes_correctly() {
        let batches = vec![create_test_batch(0, 50)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        let dataset = StreamingDataset::new(Box::new(source), 4);

        // Verify initial state
        assert_eq!(dataset.size_hint(), Some(50));
        assert_eq!(dataset.schema().fields().len(), 2);
    }

    #[test]
    fn test_memory_source_single_batch() {
        let batches = vec![create_test_batch(0, 100)];
        let mut source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        assert_eq!(source.size_hint(), Some(100));

        let batch = source.next_batch().ok().flatten();
        assert!(batch.is_some());
        assert_eq!(batch.map(|b| b.num_rows()), Some(100));

        // No more batches
        assert!(source.next_batch().ok().flatten().is_none());
    }

    #[test]
    fn test_chained_source_all_sources_empty() {
        struct EmptySource {
            schema: SchemaRef,
        }

        impl DataSource for EmptySource {
            fn schema(&self) -> SchemaRef {
                Arc::clone(&self.schema)
            }
            fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
                Ok(None)
            }
        }

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

        let empty1 = EmptySource {
            schema: Arc::clone(&schema),
        };
        let empty2 = EmptySource { schema };

        let mut chained = ChainedSource::new(vec![Box::new(empty1), Box::new(empty2)])
            .ok()
            .unwrap_or_else(|| panic!("chained"));

        // Should return None when all sources are empty
        assert!(chained.next_batch().ok().flatten().is_none());
    }

    #[test]
    fn test_streaming_dataset_prefetch_larger_than_available() {
        let batches = vec![create_test_batch(0, 10)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        // Prefetch more batches than exist
        let dataset = StreamingDataset::new(Box::new(source), 4).prefetch(10);
        let collected: Vec<RecordBatch> = dataset.collect();

        // Should still work correctly
        assert_eq!(collected.len(), 1);
    }

    #[test]
    fn test_parquet_source_reset_and_reread() {
        let batch = create_test_batch(0, 50);
        let dataset = crate::ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("temp dir"));
        let path = temp_dir.path().join("reset_reread.parquet");
        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("parquet"));

        let mut source = ParquetSource::new(&path, 20)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        // Read once
        let first_pass: Vec<RecordBatch> =
            std::iter::from_fn(|| source.next_batch().ok().flatten()).collect();
        let first_count: usize = first_pass.iter().map(|b| b.num_rows()).sum();

        // Reset
        source.reset().ok().unwrap_or_else(|| panic!("reset"));

        // Read again
        let second_pass: Vec<RecordBatch> =
            std::iter::from_fn(|| source.next_batch().ok().flatten()).collect();
        let second_count: usize = second_pass.iter().map(|b| b.num_rows()).sum();

        assert_eq!(first_count, second_count);
        assert_eq!(first_count, 50);
    }

    #[test]
    fn test_chained_source_multiple_batches_per_source() {
        let source1 = MemorySource::new(vec![
            create_test_batch(0, 10),
            create_test_batch(10, 10),
            create_test_batch(20, 10),
        ])
        .ok()
        .unwrap_or_else(|| panic!("source1"));

        let source2 = MemorySource::new(vec![create_test_batch(30, 15), create_test_batch(45, 15)])
            .ok()
            .unwrap_or_else(|| panic!("source2"));

        let mut chained = ChainedSource::new(vec![Box::new(source1), Box::new(source2)])
            .ok()
            .unwrap_or_else(|| panic!("chained"));

        let batches: Vec<usize> = std::iter::from_fn(|| chained.next_batch().ok().flatten())
            .map(|b| b.num_rows())
            .collect();

        assert_eq!(batches, vec![10, 10, 10, 15, 15]);
    }

    // === Additional tests for streaming edge cases ===

    #[test]
    fn test_streaming_dataset_default_buffer() {
        let batches = vec![create_test_batch(0, 10)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        // Test default streaming behavior
        let dataset = StreamingDataset::new(Box::new(source), 4);
        assert!(!dataset.exhausted);
        let collected: Vec<RecordBatch> = dataset.collect();
        assert_eq!(collected.len(), 1);
    }

    #[test]
    fn test_memory_source_empty_error_message() {
        let result = MemorySource::new(vec![]);
        assert!(result.is_err());
        if let Err(e) = result {
            let msg = format!("{:?}", e);
            assert!(msg.contains("EmptyDataset") || msg.len() > 0);
        }
    }

    #[test]
    fn test_chained_source_empty_error_message() {
        let result: std::result::Result<ChainedSource, Error> = ChainedSource::new(vec![]);
        assert!(result.is_err());
        if let Err(e) = result {
            let msg = format!("{:?}", e);
            assert!(msg.len() > 0);
        }
    }

    #[test]
    fn test_streaming_dataset_collect_all() {
        let batches = vec![
            create_test_batch(0, 5),
            create_test_batch(5, 5),
            create_test_batch(10, 5),
            create_test_batch(15, 5),
        ];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        let dataset = StreamingDataset::new(Box::new(source), 2).prefetch(2);
        let collected: Vec<RecordBatch> = dataset.collect();

        let total_rows: usize = collected.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 20);
    }

    #[test]
    fn test_parquet_source_schema_matches() {
        let batch = create_test_batch(0, 10);
        let dataset = crate::ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("temp dir"));
        let path = temp_dir.path().join("schema_match.parquet");
        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("parquet"));

        let source = ParquetSource::new(&path, 5)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        // Schema should have 2 fields
        assert_eq!(source.schema().fields().len(), 2);
    }

    #[test]
    fn test_streaming_dataset_from_parquet_with_prefetch() {
        let batch = create_test_batch(0, 50);
        let dataset = crate::ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("temp dir"));
        let path = temp_dir.path().join("prefetch_test.parquet");
        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("parquet"));

        let streaming = StreamingDataset::from_parquet(&path, 10)
            .ok()
            .unwrap_or_else(|| panic!("streaming"))
            .prefetch(3);

        let total: usize = streaming.map(|b| b.num_rows()).sum();
        assert_eq!(total, 50);
    }

    #[test]
    fn test_chained_source_with_different_batch_counts() {
        // First source has 1 batch, second has 3
        let source1 = MemorySource::new(vec![create_test_batch(0, 100)])
            .ok()
            .unwrap_or_else(|| panic!("source1"));
        let source2 = MemorySource::new(vec![
            create_test_batch(100, 10),
            create_test_batch(110, 10),
            create_test_batch(120, 10),
        ])
        .ok()
        .unwrap_or_else(|| panic!("source2"));

        let mut chained = ChainedSource::new(vec![Box::new(source1), Box::new(source2)])
            .ok()
            .unwrap_or_else(|| panic!("chained"));

        let mut total = 0;
        while let Ok(Some(batch)) = chained.next_batch() {
            total += batch.num_rows();
        }
        assert_eq!(total, 130);
    }

    #[test]
    fn test_streaming_dataset_next_after_exhaustion() {
        let batches = vec![create_test_batch(0, 5)];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        let mut dataset = StreamingDataset::new(Box::new(source), 1);

        // Exhaust the dataset
        while dataset.next().is_some() {}

        // Additional next calls should return None
        assert!(dataset.next().is_none());
        assert!(dataset.next().is_none());
    }

    #[test]
    fn test_memory_source_size_hint_calculation() {
        let batches = vec![
            create_test_batch(0, 7),
            create_test_batch(7, 13),
            create_test_batch(20, 3),
        ];
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        // 7 + 13 + 3 = 23
        assert_eq!(source.size_hint(), Some(23));
    }

    #[test]
    fn test_chained_source_partial_size_hint() {
        // One source with known size, one without
        struct UnknownSource {
            schema: SchemaRef,
            batches: Vec<RecordBatch>,
            pos: usize,
        }

        impl DataSource for UnknownSource {
            fn schema(&self) -> SchemaRef {
                Arc::clone(&self.schema)
            }

            fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
                if self.pos < self.batches.len() {
                    let b = self.batches[self.pos].clone();
                    self.pos += 1;
                    Ok(Some(b))
                } else {
                    Ok(None)
                }
            }

            // No size_hint override - returns None
        }

        let memory = MemorySource::new(vec![create_test_batch(0, 10)])
            .ok()
            .unwrap_or_else(|| panic!("memory"));

        let unknown = UnknownSource {
            schema: create_test_batch(0, 1).schema(),
            batches: vec![create_test_batch(10, 5)],
            pos: 0,
        };

        let chained = ChainedSource::new(vec![Box::new(memory), Box::new(unknown)])
            .ok()
            .unwrap_or_else(|| panic!("chained"));

        // Size hint should be None since one source doesn't know
        assert!(chained.size_hint().is_none());
    }

    #[test]
    fn test_streaming_dataset_buffer_boundary() {
        // Create exactly as many batches as buffer size
        let batches: Vec<RecordBatch> = (0..4).map(|i| create_test_batch(i * 10, 10)).collect();

        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        // Buffer size equals batch count
        let dataset = StreamingDataset::new(Box::new(source), 4);
        let collected: Vec<RecordBatch> = dataset.collect();

        assert_eq!(collected.len(), 4);
    }

    #[test]
    fn test_parquet_source_read_all_batches() {
        let batch = create_test_batch(0, 1000);
        let dataset = crate::ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("temp dir"));
        let path = temp_dir.path().join("all_batches.parquet");
        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("parquet"));

        // Small batch size means many batches
        let mut source = ParquetSource::new(&path, 100)
            .ok()
            .unwrap_or_else(|| panic!("source"));

        let mut batch_count = 0;
        let mut total_rows = 0;
        while let Ok(Some(batch)) = source.next_batch() {
            batch_count += 1;
            total_rows += batch.num_rows();
        }

        assert!(batch_count >= 1);
        assert_eq!(total_rows, 1000);
    }
}
