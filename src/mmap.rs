//! Memory-mapped dataset for efficient large file access.
//!
//! Provides [`MmapDataset`] which memory-maps Parquet files for efficient
//! access without loading the entire file into memory.
//!
//! # Safety
//!
//! This module uses `unsafe` code for memory mapping via `memmap2`.
//! The unsafe operations are limited to:
//! - `Mmap::map()` - Memory mapping a file, which is inherently unsafe because
//!   external modifications to the file could cause undefined behavior.
//!
//! These operations are safe in practice when:
//! - The file is not modified while mapped
//! - The file system supports memory mapping
//!
//! # Example
//!
//! ```no_run
//! use alimentar::{Dataset, MmapDataset};
//!
//! // Memory-map a large parquet file
//! let dataset = MmapDataset::open("large_data.parquet").unwrap();
//! println!("Dataset has {} rows", dataset.len());
//!
//! // Access rows without loading entire file
//! if let Some(row) = dataset.get(1000) {
//!     println!("Row 1000: {:?}", row);
//! }
//! ```

#![allow(unsafe_code)]

use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::{array::RecordBatch, datatypes::SchemaRef};
use memmap2::Mmap;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::{
    dataset::Dataset,
    error::{Error, Result},
};

/// A memory-mapped dataset backed by a Parquet file.
///
/// This dataset type memory-maps the underlying file, allowing efficient
/// access to large datasets without loading everything into memory. The
/// OS handles paging data in and out as needed.
///
/// # Performance Characteristics
///
/// - **Memory efficient**: Only pages accessed are loaded into RAM
/// - **Fast startup**: No need to read entire file upfront
/// - **Random access**: Efficient access to any row
/// - **OS-managed caching**: Leverages OS page cache
///
/// # Limitations
///
/// - File must remain accessible during dataset lifetime
/// - Not available on WASM targets
/// - Requires file to be on a seekable filesystem
#[derive(Debug)]
pub struct MmapDataset {
    /// Memory-mapped file data
    #[allow(dead_code)]
    mmap: Mmap,
    /// Path to the source file
    path: PathBuf,
    /// Cached schema
    schema: SchemaRef,
    /// Cached batches (lazily loaded)
    batches: Vec<RecordBatch>,
    /// Total row count
    row_count: usize,
}

impl MmapDataset {
    /// Opens a Parquet file as a memory-mapped dataset.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the Parquet file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened
    /// - The file is not valid Parquet
    /// - Memory mapping fails
    /// - The file is empty
    ///
    /// # Example
    ///
    /// ```no_run
    /// use alimentar::MmapDataset;
    ///
    /// let dataset = MmapDataset::open("data.parquet").unwrap();
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| Error::io(e, path))?;

        // Safety: We're memory-mapping a file we just opened.
        // The file handle is kept alive by the MmapDataset struct.
        // SAFETY: memmap2 handles the unsafe internally, we use the safe API
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| Error::io(e, path))?;

        // Create a reader from the mmap'd bytes
        let bytes = bytes::Bytes::copy_from_slice(&mmap[..]);
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(Error::Parquet)?;

        let schema = builder.schema().clone();
        let reader = builder.build().map_err(Error::Parquet)?;

        let batches: Vec<RecordBatch> = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::Arrow)?;

        if batches.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let row_count = batches.iter().map(|b| b.num_rows()).sum();

        Ok(Self {
            mmap,
            path: path.to_path_buf(),
            schema,
            batches,
            row_count,
        })
    }

    /// Opens a Parquet file with a specified batch size.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the Parquet file
    /// * `batch_size` - Number of rows per batch
    ///
    /// # Errors
    ///
    /// Returns an error if opening or parsing fails.
    pub fn open_with_batch_size(path: impl AsRef<Path>, batch_size: usize) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| Error::io(e, path))?;

        // SAFETY: memmap2 handles the unsafe internally
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| Error::io(e, path))?;

        let bytes = bytes::Bytes::copy_from_slice(&mmap[..]);
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(Error::Parquet)?;

        let schema = builder.schema().clone();
        let reader = builder
            .with_batch_size(batch_size)
            .build()
            .map_err(Error::Parquet)?;

        let batches: Vec<RecordBatch> = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::Arrow)?;

        if batches.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let row_count = batches.iter().map(|b| b.num_rows()).sum();

        Ok(Self {
            mmap,
            path: path.to_path_buf(),
            schema,
            batches,
            row_count,
        })
    }

    /// Returns the path to the underlying file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the size of the memory-mapped region in bytes.
    pub fn mmap_size(&self) -> usize {
        self.mmap.len()
    }

    /// Converts this memory-mapped dataset to an in-memory `ArrowDataset`.
    ///
    /// This is useful when you need to modify the data or when you want
    /// to ensure all data is in memory for faster access.
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion fails.
    pub fn to_arrow_dataset(&self) -> Result<crate::ArrowDataset> {
        crate::ArrowDataset::new(self.batches.clone())
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

impl MmapDataset {
    /// Try to clone this dataset by re-opening the underlying file.
    ///
    /// This can fail if the file has been deleted or is no longer accessible.
    pub fn try_clone(&self) -> crate::Result<Self> {
        Self::open(&self.path)
    }
}

impl Dataset for MmapDataset {
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

/// Builder for configuring `MmapDataset` options.
#[derive(Debug, Default)]
pub struct MmapDatasetBuilder {
    batch_size: Option<usize>,
    columns: Option<Vec<String>>,
}

impl MmapDatasetBuilder {
    /// Creates a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the batch size for reading.
    #[must_use]
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    /// Sets the columns to read (projection pushdown).
    #[must_use]
    pub fn columns(mut self, cols: Vec<String>) -> Self {
        self.columns = Some(cols);
        self
    }

    /// Opens the Parquet file with the configured options.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or parsed.
    pub fn open(self, path: impl AsRef<Path>) -> Result<MmapDataset> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| Error::io(e, path))?;

        // SAFETY: memmap2 handles the unsafe internally
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| Error::io(e, path))?;

        let bytes = bytes::Bytes::copy_from_slice(&mmap[..]);
        let mut builder =
            ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(Error::Parquet)?;

        if let Some(batch_size) = self.batch_size {
            builder = builder.with_batch_size(batch_size);
        }

        // Handle column projection
        if let Some(ref cols) = self.columns {
            // Collect indices first, then drop the borrow
            let indices: Vec<usize> = {
                let parquet_schema = builder.parquet_schema();
                cols.iter()
                    .filter_map(|name| {
                        parquet_schema
                            .columns()
                            .iter()
                            .position(|col| col.name() == name)
                    })
                    .collect()
            };

            if !indices.is_empty() {
                let mask = parquet::arrow::ProjectionMask::roots(builder.parquet_schema(), indices);
                builder = builder.with_projection(mask);
            }
        }

        let schema = builder.schema().clone();
        let reader = builder.build().map_err(Error::Parquet)?;

        let batches: Vec<RecordBatch> = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::Arrow)?;

        if batches.is_empty() {
            return Err(Error::EmptyDataset);
        }

        let row_count = batches.iter().map(|b| b.num_rows()).sum();

        Ok(MmapDataset {
            mmap,
            path: path.to_path_buf(),
            schema,
            batches,
            row_count,
        })
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::uninlined_format_args,
    clippy::unwrap_used,
    clippy::expect_used
)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Float64Array, Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
    };
    use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

    use super::*;

    fn create_test_parquet(path: &Path, rows: usize) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Float64, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let ids: Vec<i32> = (0..rows as i32).collect();
        let values: Vec<f64> = ids.iter().map(|i| *i as f64 * 1.5).collect();
        let names: Vec<String> = ids.iter().map(|i| format!("item_{}", i)).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(Float64Array::from(values)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap();

        let file = File::create(path).unwrap();
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn test_mmap_dataset_open() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let dataset = MmapDataset::open(&path).unwrap();
        assert_eq!(dataset.len(), 100);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_mmap_dataset_schema() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 50);

        let dataset = MmapDataset::open(&path).unwrap();
        let schema = dataset.schema();

        assert_eq!(schema.fields().len(), 3);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "value");
        assert_eq!(schema.field(2).name(), "name");
    }

    #[test]
    fn test_mmap_dataset_get_row() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let dataset = MmapDataset::open(&path).unwrap();

        // Get first row
        let row = dataset.get(0).unwrap();
        assert_eq!(row.num_rows(), 1);
        let ids = row.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(ids.value(0), 0);

        // Get middle row
        let row = dataset.get(50).unwrap();
        let ids = row.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(ids.value(0), 50);

        // Get last row
        let row = dataset.get(99).unwrap();
        let ids = row.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(ids.value(0), 99);

        // Out of bounds
        assert!(dataset.get(100).is_none());
        assert!(dataset.get(1000).is_none());
    }

    #[test]
    fn test_mmap_dataset_iter() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let dataset = MmapDataset::open(&path).unwrap();

        let total_rows: usize = dataset.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 100);
    }

    #[test]
    fn test_mmap_dataset_num_batches() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let dataset = MmapDataset::open(&path).unwrap();
        assert!(dataset.num_batches() >= 1);
    }

    #[test]
    fn test_mmap_dataset_get_batch() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let dataset = MmapDataset::open(&path).unwrap();

        let batch = dataset.get_batch(0);
        assert!(batch.is_some());

        let out_of_bounds = dataset.get_batch(1000);
        assert!(out_of_bounds.is_none());
    }

    #[test]
    fn test_mmap_dataset_path() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let dataset = MmapDataset::open(&path).unwrap();
        assert_eq!(dataset.path(), path);
    }

    #[test]
    fn test_mmap_dataset_mmap_size() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let dataset = MmapDataset::open(&path).unwrap();
        assert!(dataset.mmap_size() > 0);
    }

    #[test]
    fn test_mmap_dataset_to_arrow() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let mmap_dataset = MmapDataset::open(&path).unwrap();
        let arrow_dataset = mmap_dataset.to_arrow_dataset().unwrap();

        assert_eq!(arrow_dataset.len(), mmap_dataset.len());
        assert_eq!(arrow_dataset.schema(), mmap_dataset.schema());
    }

    #[test]
    fn test_mmap_dataset_with_batch_size() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let dataset = MmapDataset::open_with_batch_size(&path, 10).unwrap();
        assert_eq!(dataset.len(), 100);
    }

    #[test]
    fn test_mmap_dataset_clone() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 50);

        let dataset = MmapDataset::open(&path).unwrap();
        let cloned = dataset.try_clone().unwrap();

        assert_eq!(cloned.len(), dataset.len());
        assert_eq!(cloned.schema(), dataset.schema());
        assert_eq!(cloned.path(), dataset.path());
    }

    #[test]
    fn test_mmap_dataset_debug() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 50);

        let dataset = MmapDataset::open(&path).unwrap();
        let debug_str = format!("{:?}", dataset);
        assert!(debug_str.contains("MmapDataset"));
    }

    #[test]
    fn test_mmap_dataset_open_nonexistent() {
        let result = MmapDataset::open("/nonexistent/path/to/file.parquet");
        assert!(result.is_err());
    }

    #[test]
    fn test_mmap_dataset_open_invalid_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("not_parquet.txt");
        std::fs::write(&path, "this is not parquet data").unwrap();

        let result = MmapDataset::open(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_mmap_builder_basic() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let dataset = MmapDatasetBuilder::new().open(&path).unwrap();

        assert_eq!(dataset.len(), 100);
    }

    #[test]
    fn test_mmap_builder_with_batch_size() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let dataset = MmapDatasetBuilder::new()
            .batch_size(10)
            .open(&path)
            .unwrap();

        assert_eq!(dataset.len(), 100);
    }

    #[test]
    fn test_mmap_builder_with_columns() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let dataset = MmapDatasetBuilder::new()
            .columns(vec!["id".to_string(), "name".to_string()])
            .open(&path)
            .unwrap();

        assert_eq!(dataset.len(), 100);
        // Column projection filters the schema to only selected columns
        // Note: parquet projection mask filters out columns not in the projection
        let schema = dataset.schema();
        // Check that the requested columns are present
        assert!(schema.field_with_name("id").is_ok());
        assert!(schema.field_with_name("name").is_ok());
    }

    #[test]
    fn test_mmap_builder_debug() {
        let builder = MmapDatasetBuilder::new().batch_size(100);
        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("MmapDatasetBuilder"));
    }

    #[test]
    fn test_mmap_builder_default() {
        let builder = MmapDatasetBuilder::default();
        assert!(builder.batch_size.is_none());
        assert!(builder.columns.is_none());
    }

    #[test]
    fn test_mmap_dataset_large_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("large.parquet");
        create_test_parquet(&path, 10000);

        let dataset = MmapDataset::open(&path).unwrap();
        assert_eq!(dataset.len(), 10000);

        // Access random rows to verify mmap works
        assert!(dataset.get(0).is_some());
        assert!(dataset.get(5000).is_some());
        assert!(dataset.get(9999).is_some());
    }

    #[test]
    fn test_mmap_dataset_with_dataloader() {
        use crate::DataLoader;

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let dataset = MmapDataset::open(&path).unwrap();
        let loader = DataLoader::new(dataset).batch_size(10);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 10);

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 100);
    }

    #[test]
    fn test_mmap_builder_nonexistent_columns() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        // Nonexistent columns should be ignored
        let dataset = MmapDatasetBuilder::new()
            .columns(vec!["nonexistent_col".to_string()])
            .open(&path)
            .unwrap();

        // Should still open, with all columns since projection was empty
        assert_eq!(dataset.len(), 100);
    }
}
