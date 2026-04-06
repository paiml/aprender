//! Async prefetch for parallel I/O in streaming datasets.
//!
//! Provides [`AsyncPrefetchDataset`] which spawns a background task to read
//! batches ahead of time, reducing I/O latency in the training loop.

use std::sync::Arc;

use arrow::{array::RecordBatch, datatypes::SchemaRef};
#[cfg(feature = "tokio-runtime")]
use tokio::sync::mpsc;

use crate::{
    error::{Error, Result},
    streaming::DataSource,
};

/// A streaming dataset with async prefetch for parallel I/O.
///
/// Unlike [`StreamingDataset`](crate::streaming::StreamingDataset) which reads
/// synchronously, `AsyncPrefetchDataset` spawns a background task that reads
/// batches into a channel, allowing the main thread to process while I/O
/// happens.
///
/// # Example
///
/// ```ignore
/// use alimentar::async_prefetch::AsyncPrefetchDataset;
///
/// #[tokio::main]
/// async fn main() {
///     let dataset = AsyncPrefetchDataset::from_parquet("data.parquet", 1024, 4)
///         .await
///         .unwrap();
///
///     while let Some(batch) = dataset.next().await {
///         println!("Processing batch with {} rows", batch.num_rows());
///     }
/// }
/// ```
#[cfg(feature = "tokio-runtime")]
pub struct AsyncPrefetchDataset {
    receiver: mpsc::Receiver<Result<RecordBatch>>,
    schema: SchemaRef,
    #[allow(dead_code)] // Kept alive to prevent task cancellation
    handle: tokio::task::JoinHandle<()>,
}

#[cfg(feature = "tokio-runtime")]
impl AsyncPrefetchDataset {
    /// Creates a new async prefetch dataset from a data source.
    ///
    /// # Arguments
    ///
    /// * `source` - The data source to read from
    /// * `prefetch_size` - Number of batches to buffer ahead
    pub fn new(mut source: Box<dyn DataSource>, prefetch_size: usize) -> Self {
        let schema = source.schema();
        let (tx, rx) = mpsc::channel(prefetch_size.max(1));

        let handle = tokio::spawn(async move {
            loop {
                match source.next_batch() {
                    Ok(Some(batch)) => {
                        if tx.send(Ok(batch)).await.is_err() {
                            // Receiver dropped, stop reading
                            break;
                        }
                    }
                    Ok(None) => break, // End of source
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                }
            }
        });

        Self {
            receiver: rx,
            schema,
            handle,
        }
    }

    /// Creates an async prefetch dataset from a Parquet file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the Parquet file
    /// * `batch_size` - Number of rows per batch
    /// * `prefetch_size` - Number of batches to buffer ahead
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened.
    pub fn from_parquet(
        path: impl AsRef<std::path::Path>,
        batch_size: usize,
        prefetch_size: usize,
    ) -> Result<Self> {
        let source = crate::streaming::ParquetSource::new(path, batch_size)?;
        Ok(Self::new(Box::new(source), prefetch_size))
    }

    /// Returns the schema of the dataset.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// Receives the next batch asynchronously.
    ///
    /// Returns `None` when the source is exhausted.
    pub async fn next(&mut self) -> Option<Result<RecordBatch>> {
        self.receiver.recv().await
    }

    /// Tries to receive a batch without waiting.
    ///
    /// Returns `None` if no batch is available or the source is exhausted.
    pub fn try_next(&mut self) -> Option<Result<RecordBatch>> {
        self.receiver.try_recv().ok()
    }

    /// Returns the number of batches currently buffered.
    pub fn buffered_count(&self) -> usize {
        self.receiver.len()
    }
}

#[cfg(feature = "tokio-runtime")]
impl std::fmt::Debug for AsyncPrefetchDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncPrefetchDataset")
            .field("buffered", &self.receiver.len())
            .finish_non_exhaustive()
    }
}

/// Builder for creating async prefetch datasets.
#[cfg(feature = "tokio-runtime")]
#[derive(Debug, Default)]
pub struct AsyncPrefetchBuilder {
    batch_size: Option<usize>,
    prefetch_size: Option<usize>,
}

#[cfg(feature = "tokio-runtime")]
impl AsyncPrefetchBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the batch size (rows per batch).
    #[must_use]
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    /// Sets the prefetch buffer size (number of batches).
    #[must_use]
    pub fn prefetch_size(mut self, size: usize) -> Self {
        self.prefetch_size = Some(size);
        self
    }

    /// Builds an async prefetch dataset from a Parquet file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened.
    pub fn from_parquet(self, path: impl AsRef<std::path::Path>) -> Result<AsyncPrefetchDataset> {
        let batch_size = self.batch_size.unwrap_or(1024);
        let prefetch_size = self.prefetch_size.unwrap_or(4);

        if batch_size == 0 {
            return Err(Error::invalid_config("batch_size must be greater than 0"));
        }

        AsyncPrefetchDataset::from_parquet(path, batch_size, prefetch_size)
    }

    /// Builds an async prefetch dataset from a data source.
    pub fn from_source(self, source: Box<dyn DataSource>) -> AsyncPrefetchDataset {
        let prefetch_size = self.prefetch_size.unwrap_or(4);
        AsyncPrefetchDataset::new(source, prefetch_size)
    }
}

/// Synchronous wrapper for async prefetch that works with DataLoader.
///
/// This allows using async prefetch with the existing synchronous DataLoader
/// API by blocking on the async operations internally.
#[cfg(feature = "tokio-runtime")]
pub struct SyncPrefetchDataset {
    inner: AsyncPrefetchDataset,
    runtime: tokio::runtime::Handle,
}

#[cfg(feature = "tokio-runtime")]
impl SyncPrefetchDataset {
    /// Creates a new sync wrapper around an async prefetch dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The async dataset to wrap
    /// * `runtime` - Handle to the tokio runtime
    pub fn new(dataset: AsyncPrefetchDataset, runtime: tokio::runtime::Handle) -> Self {
        Self {
            inner: dataset,
            runtime,
        }
    }

    /// Returns the schema.
    pub fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    /// Gets the next batch, blocking if necessary.
    pub fn next_blocking(&mut self) -> Option<Result<RecordBatch>> {
        self.runtime.block_on(self.inner.next())
    }
}

#[cfg(feature = "tokio-runtime")]
impl std::fmt::Debug for SyncPrefetchDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncPrefetchDataset")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[cfg(feature = "tokio-runtime")]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
    };

    use super::*;
    use crate::streaming::MemorySource;

    fn create_test_batches(count: usize, rows_per_batch: usize) -> Vec<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        (0..count)
            .map(|batch_idx| {
                let start = (batch_idx * rows_per_batch) as i32;
                let ids: Vec<i32> = (start..start + rows_per_batch as i32).collect();
                let names: Vec<String> = ids.iter().map(|i| format!("item_{}", i)).collect();

                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(Int32Array::from(ids)),
                        Arc::new(StringArray::from(names)),
                    ],
                )
                .ok()
                .unwrap_or_else(|| panic!("Should create batch"))
            })
            .collect()
    }

    #[tokio::test]
    async fn test_async_prefetch_creation() {
        let batches = create_test_batches(5, 10);
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let dataset = AsyncPrefetchDataset::new(Box::new(source), 4);
        assert_eq!(dataset.schema().fields().len(), 2);
    }

    #[tokio::test]
    async fn test_async_prefetch_iteration() {
        let batches = create_test_batches(5, 10);
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let mut dataset = AsyncPrefetchDataset::new(Box::new(source), 4);

        let mut count = 0;
        let mut total_rows = 0;
        while let Some(result) = dataset.next().await {
            let batch = result.ok().unwrap_or_else(|| panic!("Should get batch"));
            count += 1;
            total_rows += batch.num_rows();
        }

        assert_eq!(count, 5);
        assert_eq!(total_rows, 50);
    }

    #[tokio::test]
    async fn test_async_prefetch_try_next() {
        let batches = create_test_batches(3, 10);
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let mut dataset = AsyncPrefetchDataset::new(Box::new(source), 10);

        // Yield to let background task run
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;

        // Should have some batches ready
        let mut count = 0;
        while dataset.try_next().is_some() {
            count += 1;
        }

        assert!(count > 0, "Should have prefetched some batches");
    }

    #[tokio::test]
    async fn test_async_prefetch_buffered_count() {
        let batches = create_test_batches(10, 5);
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let dataset = AsyncPrefetchDataset::new(Box::new(source), 4);

        // Yield to let background task fill buffer
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }

        // Buffer should have some items (up to prefetch_size)
        let buffered = dataset.buffered_count();
        assert!(buffered <= 4, "Should not exceed prefetch size");
    }

    #[tokio::test]
    async fn test_async_prefetch_builder() {
        let batches = create_test_batches(3, 10);
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let mut dataset = AsyncPrefetchBuilder::new()
            .batch_size(10)
            .prefetch_size(2)
            .from_source(Box::new(source));

        let mut count = 0;
        while let Some(result) = dataset.next().await {
            assert!(result.is_ok());
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn test_async_prefetch_debug() {
        let batches = create_test_batches(2, 5);
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let dataset = AsyncPrefetchDataset::new(Box::new(source), 4);
        let debug_str = format!("{:?}", dataset);
        assert!(debug_str.contains("AsyncPrefetchDataset"));
    }

    #[tokio::test]
    async fn test_async_prefetch_parquet_roundtrip() {
        // Create test data
        let batch = create_test_batches(1, 100)[0].clone();
        let dataset = crate::ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        // Write to temp file
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("async_test.parquet");
        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        // Read back via async prefetch
        let mut async_dataset = AsyncPrefetchDataset::from_parquet(&path, 25, 4)
            .ok()
            .unwrap_or_else(|| panic!("Should create async dataset"));

        let mut total = 0;
        while let Some(result) = async_dataset.next().await {
            let batch = result.ok().unwrap_or_else(|| panic!("Should get batch"));
            total += batch.num_rows();
        }
        assert_eq!(total, 100);
    }

    #[tokio::test]
    async fn test_sync_prefetch_wrapper() {
        let batches = create_test_batches(3, 10);
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let async_dataset = AsyncPrefetchDataset::new(Box::new(source), 4);
        let handle = tokio::runtime::Handle::current();
        let sync_dataset = SyncPrefetchDataset::new(async_dataset, handle);

        assert_eq!(sync_dataset.schema().fields().len(), 2);

        let debug_str = format!("{:?}", sync_dataset);
        assert!(debug_str.contains("SyncPrefetchDataset"));
    }

    #[tokio::test]
    async fn test_builder_zero_batch_size_error() {
        let result = AsyncPrefetchBuilder::new()
            .batch_size(0)
            .from_parquet("/nonexistent.parquet");

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_builder_defaults() {
        let batches = create_test_batches(2, 5);
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        // Use default values
        let dataset = AsyncPrefetchBuilder::new().from_source(Box::new(source));

        assert_eq!(dataset.schema().fields().len(), 2);
    }

    #[tokio::test]
    async fn test_async_prefetch_quick_exhaustion() {
        // Test with source that quickly exhausts
        struct QuickExhaustSource {
            schema: SchemaRef,
            exhausted: bool,
        }

        impl crate::streaming::DataSource for QuickExhaustSource {
            fn schema(&self) -> SchemaRef {
                Arc::clone(&self.schema)
            }

            fn next_batch(&mut self) -> crate::Result<Option<RecordBatch>> {
                if self.exhausted {
                    Ok(None)
                } else {
                    self.exhausted = true;
                    Ok(Some(create_test_batches(1, 1)[0].clone()))
                }
            }
        }

        let source = QuickExhaustSource {
            schema: create_test_batches(1, 1)[0].schema(),
            exhausted: false,
        };

        let mut dataset = AsyncPrefetchDataset::new(Box::new(source), 4);

        // First should succeed
        let first = dataset.next().await;
        assert!(first.is_some());
        assert!(first.unwrap().is_ok());

        // Second should be None (exhausted)
        let second = dataset.next().await;
        assert!(second.is_none());
    }

    #[tokio::test]
    async fn test_async_prefetch_single_batch() {
        let batches = create_test_batches(1, 100);
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let mut dataset = AsyncPrefetchDataset::new(Box::new(source), 4);

        let batch = dataset
            .next()
            .await
            .unwrap_or_else(|| panic!("Should have batch"))
            .ok()
            .unwrap_or_else(|| panic!("Batch should be ok"));
        assert_eq!(batch.num_rows(), 100);

        // No more batches
        assert!(dataset.next().await.is_none());
    }

    #[tokio::test]
    async fn test_async_prefetch_large_prefetch_size() {
        // Prefetch size larger than available batches
        let batches = create_test_batches(3, 10);
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let mut dataset = AsyncPrefetchDataset::new(Box::new(source), 100);

        let mut count = 0;
        while let Some(result) = dataset.next().await {
            assert!(result.is_ok());
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn test_async_prefetch_prefetch_size_one() {
        // Minimal prefetch
        let batches = create_test_batches(5, 10);
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let mut dataset = AsyncPrefetchDataset::new(Box::new(source), 1);

        let mut count = 0;
        while let Some(result) = dataset.next().await {
            assert!(result.is_ok());
            count += 1;
        }
        assert_eq!(count, 5);
    }

    #[tokio::test]
    async fn test_async_prefetch_error_source() {
        // Test with source that errors
        struct ErrorSource {
            schema: SchemaRef,
            calls: usize,
        }

        impl crate::streaming::DataSource for ErrorSource {
            fn schema(&self) -> SchemaRef {
                Arc::clone(&self.schema)
            }

            fn next_batch(&mut self) -> crate::Result<Option<RecordBatch>> {
                self.calls += 1;
                if self.calls > 2 {
                    Err(crate::Error::storage("Simulated error"))
                } else {
                    Ok(Some(create_test_batches(1, 5)[0].clone()))
                }
            }
        }

        let source = ErrorSource {
            schema: create_test_batches(1, 1)[0].schema(),
            calls: 0,
        };

        let mut dataset = AsyncPrefetchDataset::new(Box::new(source), 4);

        // First two should succeed
        let b1 = dataset.next().await;
        assert!(b1.is_some());
        assert!(b1.unwrap().is_ok());

        let b2 = dataset.next().await;
        assert!(b2.is_some());
        assert!(b2.unwrap().is_ok());

        // Third should be an error
        let b3 = dataset.next().await;
        assert!(b3.is_some());
        assert!(b3.unwrap().is_err());
    }

    #[tokio::test]
    async fn test_async_prefetch_try_next_after_exhaustion() {
        // Create source with one batch
        let batches = create_test_batches(1, 5);
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let mut dataset = AsyncPrefetchDataset::new(Box::new(source), 4);

        // Consume the single batch
        let _ = dataset.next().await;

        // Allow background task to complete
        tokio::task::yield_now().await;

        // try_next should return None (exhausted)
        let result = dataset.try_next();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_builder_with_prefetch_size() {
        let batches = create_test_batches(5, 10);
        let source = MemorySource::new(batches)
            .ok()
            .unwrap_or_else(|| panic!("Should create source"));

        let mut dataset = AsyncPrefetchBuilder::new()
            .prefetch_size(2)
            .from_source(Box::new(source));

        let mut count = 0;
        while let Some(result) = dataset.next().await {
            assert!(result.is_ok());
            count += 1;
        }
        assert_eq!(count, 5);
    }

    #[tokio::test]
    async fn test_builder_from_parquet_roundtrip() {
        // Create test data
        let batch = create_test_batches(1, 50)[0].clone();
        let dataset = crate::ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("builder_test.parquet");
        dataset
            .to_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));

        // Read with builder
        let mut async_dataset = AsyncPrefetchBuilder::new()
            .batch_size(10)
            .prefetch_size(3)
            .from_parquet(&path)
            .ok()
            .unwrap_or_else(|| panic!("Should create async dataset"));

        let mut total = 0;
        while let Some(result) = async_dataset.next().await {
            total += result.ok().unwrap().num_rows();
        }
        assert_eq!(total, 50);
    }

    #[test]
    fn test_builder_debug() {
        let builder = AsyncPrefetchBuilder::new().batch_size(32).prefetch_size(8);

        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("AsyncPrefetchBuilder"));
    }
}
