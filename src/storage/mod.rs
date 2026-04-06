//! Storage backend (Arrow/Parquet)
//!
//! **OLAP-Only Design** (Append-Only Write Pattern):
//! - Trueno-DB is OLAP-optimized (columnar storage, bulk analytics)
//! - Write pattern: Append-only batches (no random updates)
//! - Use case: Full codebase re-analysis, bulk data loads
//! - NOT suitable for: OLTP workloads, incremental row updates
//!
//! See: ../paiml-mcp-agent-toolkit/docs/specifications/trueno-db-integration-review-response.md Issue #4
//!
//! Toyota Way Principles:
//! - Poka-Yoke: Morsel-based paging prevents VRAM OOM (Funke et al. 2018)
//! - Muda elimination: Late materialization (Abadi et al. 2008)

use crate::{Error, Result};
use arrow::record_batch::RecordBatch;
#[cfg(feature = "parquet-io")]
use std::path::Path;

/// Morsel size for out-of-core execution (128MB chunks)
/// Based on: Leis et al. (2014) morsel-driven parallelism
pub const MORSEL_SIZE_BYTES: usize = 128 * 1024 * 1024; // 128MB

/// Maximum number of in-flight GPU transfers
/// Bounded to prevent memory explosion while keeping `PCIe` bus busy
#[cfg(feature = "tokio")]
const MAX_IN_FLIGHT_TRANSFERS: usize = 2;

/// Storage engine for Arrow/Parquet data
pub struct StorageEngine {
    batches: Vec<RecordBatch>,
}

impl StorageEngine {
    /// Create a new storage engine from existing batches
    ///
    /// Useful for testing and benchmarking
    #[must_use]
    pub const fn new(batches: Vec<RecordBatch>) -> Self {
        Self { batches }
    }

    /// Load table from Parquet file
    ///
    /// # Errors
    /// Returns error if file cannot be read or parsed
    #[cfg(feature = "parquet-io")]
    pub fn load_parquet<P: AsRef<Path>>(path: P) -> Result<Self> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs::File;

        let file = File::open(path.as_ref())
            .map_err(|e| Error::StorageError(format!("Failed to open Parquet file: {e}")))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| Error::StorageError(format!("Failed to parse Parquet file: {e}")))?;

        let reader = builder
            .build()
            .map_err(|e| Error::StorageError(format!("Failed to create Parquet reader: {e}")))?;

        // Read all batches into memory
        let mut batches = Vec::new();
        for batch in reader {
            let batch = batch
                .map_err(|e| Error::StorageError(format!("Failed to read record batch: {e}")))?;
            batches.push(batch);
        }

        Ok(Self { batches })
    }

    /// Get all record batches
    #[must_use]
    pub fn batches(&self) -> &[RecordBatch] {
        &self.batches
    }

    /// Create iterator over morsels (128MB chunks)
    #[must_use]
    pub fn morsels(&self) -> MorselIterator<'_> {
        MorselIterator::new(&self.batches)
    }

    /// Append batches to storage (OLAP-optimized)
    ///
    /// **WARNING**: This is the ONLY supported write operation.
    /// Trueno-DB does NOT support incremental row updates (OLTP).
    ///
    /// # Design Rationale
    ///
    /// Columnar storage optimizes for bulk reads, not random writes:
    /// - Single-row update cost: O(N) (rewrite entire column)
    /// - Batch append cost: O(1) (append to new partition)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use trueno_db::storage::StorageEngine;
    /// # use arrow::array::{Int32Array, RecordBatch};
    /// # use arrow::datatypes::{DataType, Field, Schema};
    /// # use std::sync::Arc;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Good: Bulk append (OLAP-compatible)
    /// let schema = Arc::new(Schema::new(vec![
    ///     Field::new("id", DataType::Int32, false),
    /// ]));
    /// let batch = RecordBatch::try_new(
    ///     schema,
    ///     vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    /// )?;
    ///
    /// let mut storage = StorageEngine::new(vec![]);
    /// storage.append_batch(batch)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if batch schema doesn't match existing batches
    pub fn append_batch(&mut self, batch: RecordBatch) -> Result<()> {
        // Validate schema compatibility
        if !self.batches.is_empty() {
            let existing_schema = self.batches[0].schema();
            if batch.schema() != existing_schema {
                return Err(Error::StorageError(format!(
                    "Schema mismatch: expected {:?}, got {:?}",
                    existing_schema,
                    batch.schema()
                )));
            }
        }

        self.batches.push(batch);
        Ok(())
    }

    /// **DEPRECATED**: Single-row update not supported
    ///
    /// Trueno-DB is OLAP-only (columnar storage). Use [`append_batch`](Self::append_batch) instead.
    ///
    /// # Why This Fails
    ///
    /// Column stores are optimized for bulk reads, not random writes:
    /// - `SQLite` (row-store): `O(1)` update with `B-tree` index
    /// - `Trueno-DB` (column-store): `O(N)` update (rewrite entire column)
    ///
    /// # Migration Guide
    ///
    /// ```rust,no_run
    /// # use trueno_db::storage::StorageEngine;
    /// # use arrow::array::{Int32Array, RecordBatch};
    /// # use arrow::datatypes::{DataType, Field, Schema};
    /// # use std::sync::Arc;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Bad: Incremental update (OLTP pattern)
    /// // storage.update_row(row_id, new_values)?;  // NOT SUPPORTED
    ///
    /// // Good: Batch re-analysis (OLAP pattern)
    /// let schema = Arc::new(Schema::new(vec![
    ///     Field::new("id", DataType::Int32, false),
    /// ]));
    /// let new_batch = RecordBatch::try_new(
    ///     schema,
    ///     vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    /// )?;
    /// let mut storage = StorageEngine::new(vec![]);
    /// storage.append_batch(new_batch)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Always returns error (not implemented)
    #[deprecated(
        since = "0.1.0",
        note = "Trueno-DB is OLAP-only. Use append_batch() for bulk data loads."
    )]
    #[allow(clippy::unused_self)]
    pub fn update_row(&mut self, _row_id: usize, _values: RecordBatch) -> Result<()> {
        Err(Error::StorageError(
            "Single-row updates not supported in columnar storage. \
             Use append_batch() for bulk re-analysis instead."
                .to_string(),
        ))
    }
}

/// Iterator over 128MB morsels of data
pub struct MorselIterator<'a> {
    batches: &'a [RecordBatch],
    current_batch_idx: usize,
    current_offset: usize,
    morsel_rows: usize,
}

impl<'a> MorselIterator<'a> {
    /// Create new morsel iterator
    fn new(batches: &'a [RecordBatch]) -> Self {
        // Calculate morsel size based on first batch
        let morsel_rows = batches.first().map_or(0, Self::calculate_morsel_rows);

        Self { batches, current_batch_idx: 0, current_offset: 0, morsel_rows }
    }

    /// Calculate how many rows fit in a 128MB morsel
    fn calculate_morsel_rows(batch: &RecordBatch) -> usize {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return 0;
        }

        let total_bytes = batch.get_array_memory_size();
        let bytes_per_row = total_bytes / num_rows;

        if bytes_per_row == 0 {
            return num_rows; // Avoid division by zero
        }

        MORSEL_SIZE_BYTES / bytes_per_row
    }
}

impl Iterator for MorselIterator<'_> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we've exhausted all batches
        if self.current_batch_idx >= self.batches.len() {
            return None;
        }

        let current_batch = &self.batches[self.current_batch_idx];

        // Check if we've exhausted current batch
        if self.current_offset >= current_batch.num_rows() {
            self.current_batch_idx += 1;
            self.current_offset = 0;
            return self.next(); // Recurse to next batch
        }

        // Calculate slice length
        let remaining_rows = current_batch.num_rows() - self.current_offset;
        let slice_length = remaining_rows.min(self.morsel_rows);

        // Create morsel slice
        let morsel = current_batch.slice(self.current_offset, slice_length);
        self.current_offset += slice_length;

        Some(morsel)
    }
}

/// GPU Transfer Queue for async bounded transfers
///
/// Toyota Way: Heijunka (Load Balancing)
/// - Bounded queue prevents memory explosion (Poka-Yoke)
/// - Max 2 in-flight keeps `PCIe` bus busy without overwhelming GPU
/// - Async design prevents blocking Tokio reactor
///
/// References:
/// - Leis et al. (2014): Morsel-driven parallelism
#[cfg(feature = "tokio")]
pub struct GpuTransferQueue {
    sender: tokio::sync::mpsc::Sender<RecordBatch>,
    receiver: tokio::sync::mpsc::Receiver<RecordBatch>,
}

#[cfg(feature = "tokio")]
impl GpuTransferQueue {
    /// Create new GPU transfer queue with bounded capacity
    ///
    /// # Returns
    /// Queue with max 2 in-flight transfers
    #[must_use]
    pub fn new() -> Self {
        let (sender, receiver) = tokio::sync::mpsc::channel(MAX_IN_FLIGHT_TRANSFERS);
        Self { sender, receiver }
    }

    /// Enqueue a record batch for GPU transfer
    ///
    /// This will block if queue is full (2 batches in-flight)
    ///
    /// # Errors
    /// Returns error if queue is closed
    pub async fn enqueue(&self, batch: RecordBatch) -> Result<()> {
        self.sender.send(batch).await.map_err(|_| Error::QueueClosed)
    }

    /// Dequeue a record batch from GPU transfer queue
    ///
    /// # Returns
    /// Next batch if available, None if queue is empty and closed
    pub async fn dequeue(&mut self) -> Option<RecordBatch> {
        self.receiver.recv().await
    }

    /// Get sender for concurrent enqueueing
    #[must_use]
    pub fn sender(&self) -> tokio::sync::mpsc::Sender<RecordBatch> {
        self.sender.clone()
    }
}

#[cfg(feature = "tokio")]
impl Default for GpuTransferQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float32Array, Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_possible_wrap)]
    #[allow(clippy::cast_precision_loss)]
    fn create_test_batch(num_rows: usize) -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Float32, false),
            Field::new("name", DataType::Utf8, false),
        ]);

        let id_array = Int32Array::from_iter_values(0..num_rows as i32);
        let value_array = Float32Array::from_iter_values((0..num_rows).map(|i| i as f32));
        let name_array = StringArray::from_iter_values((0..num_rows).map(|i| format!("name_{i}")));

        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(id_array), Arc::new(value_array), Arc::new(name_array)],
        )
        .unwrap()
    }

    #[test]
    fn test_morsel_iterator_splits_correctly() {
        let batch = create_test_batch(1000);
        let batches = vec![batch];

        let iter = MorselIterator::new(&batches);
        let morsels: Vec<_> = iter.collect();

        // Verify all rows accounted for
        let total_rows: usize = morsels.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total_rows, 1000);

        // Verify each morsel is within size limit
        for morsel in &morsels {
            assert!(morsel.get_array_memory_size() <= MORSEL_SIZE_BYTES);
        }
    }

    #[test]
    fn test_morsel_iterator_empty_batch() {
        let batch = create_test_batch(0);
        let batches = vec![batch];

        let iter = MorselIterator::new(&batches);
        assert_eq!(iter.count(), 0);
    }

    #[test]
    fn test_append_batch_olap_pattern() {
        // OLAP-compatible: Bulk append
        let mut storage = StorageEngine::new(vec![]);
        let batch1 = create_test_batch(100);
        let batch2 = create_test_batch(200);

        storage.append_batch(batch1).unwrap();
        storage.append_batch(batch2).unwrap();

        assert_eq!(storage.batches().len(), 2);
        assert_eq!(storage.batches()[0].num_rows(), 100);
        assert_eq!(storage.batches()[1].num_rows(), 200);
    }

    #[test]
    fn test_append_batch_schema_validation() {
        // Schema mismatch should fail
        let mut storage = StorageEngine::new(vec![]);
        let batch1 = create_test_batch(100);
        storage.append_batch(batch1).unwrap();

        // Create incompatible schema
        let incompatible_schema =
            Schema::new(vec![Field::new("different_field", DataType::Int32, false)]);
        let incompatible_batch = RecordBatch::try_new(
            Arc::new(incompatible_schema),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();

        let result = storage.append_batch(incompatible_batch);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Schema mismatch"));
    }

    #[test]
    #[allow(deprecated)]
    fn test_update_row_fails_oltp_pattern() {
        // OLTP pattern (single-row update) must fail
        let mut storage = StorageEngine::new(vec![]);
        let batch = create_test_batch(100);

        let result = storage.update_row(0, batch);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Single-row updates not supported"));
    }

    #[test]
    fn test_morsel_iterator_multiple_batches() {
        let batch1 = create_test_batch(500);
        let batch2 = create_test_batch(500);
        let batches = vec![batch1, batch2];

        let iter = MorselIterator::new(&batches);
        let morsels: Vec<_> = iter.collect();

        // Verify all rows accounted for across both batches
        let total_rows: usize = morsels.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total_rows, 1000);
    }

    #[tokio::test]
    async fn test_gpu_transfer_queue_basic() {
        let mut queue = GpuTransferQueue::new();
        let batch = create_test_batch(100);

        // Enqueue
        queue.enqueue(batch.clone()).await.unwrap();

        // Dequeue
        let received = queue.dequeue().await.unwrap();
        assert_eq!(received.num_rows(), 100);
    }

    #[tokio::test]
    async fn test_gpu_transfer_queue_bounded() {
        use tokio::time::{timeout, Duration};

        let queue = GpuTransferQueue::new();
        let batch = create_test_batch(100);

        // Fill queue (capacity = 2)
        queue.enqueue(batch.clone()).await.unwrap();
        queue.enqueue(batch.clone()).await.unwrap();

        // Third enqueue should timeout (queue full)
        let result = timeout(Duration::from_millis(100), queue.enqueue(batch)).await;

        assert!(result.is_err(), "Queue should be full and block");
    }

    #[tokio::test]
    async fn test_gpu_transfer_queue_concurrent_enqueue_dequeue() {
        use tokio::task;

        let mut queue = GpuTransferQueue::new();
        let sender = queue.sender();

        // Spawn task to enqueue multiple batches
        let enqueue_handle = task::spawn(async move {
            for i in 0..5 {
                let batch = create_test_batch(100 * (i + 1));
                sender.send(batch).await.unwrap();
            }
        });

        // Dequeue and verify order
        for i in 0..5 {
            let batch = queue.dequeue().await.unwrap();
            assert_eq!(batch.num_rows(), 100 * (i + 1));
        }

        // Wait for enqueue task to complete
        enqueue_handle.await.unwrap();
    }

    // Property-based tests (EXTREME TDD - Toyota Way: Jidoka)
    mod property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// Property: MorselIterator preserves all rows (no data loss)
            #[test]
            fn prop_morsel_iterator_preserves_all_rows(
                num_rows in 1usize..100_000
            ) {
                let batch = create_test_batch(num_rows);
                let original_rows = batch.num_rows();
                let batches = vec![batch];

                let iter = MorselIterator::new(&batches);
                let total_morsel_rows: usize = iter.map(|m| m.num_rows()).sum();

                prop_assert_eq!(original_rows, total_morsel_rows);
            }

            /// Property: Each morsel respects 128MB size limit (Poka-Yoke)
            #[test]
            fn prop_morsel_size_within_limit(
                num_rows in 1usize..100_000
            ) {
                let batch = create_test_batch(num_rows);
                let batches = vec![batch];

                let iter = MorselIterator::new(&batches);

                for morsel in iter {
                    let size = morsel.get_array_memory_size();
                    prop_assert!(
                        size <= MORSEL_SIZE_BYTES,
                        "Morsel size {} exceeds limit {}",
                        size,
                        MORSEL_SIZE_BYTES
                    );
                }
            }

            /// Property: Multiple batches preserve total row count
            #[test]
            fn prop_multiple_batches_preserve_rows(
                batch_sizes in prop::collection::vec(1usize..10_000, 1..10)
            ) {
                let total_expected: usize = batch_sizes.iter().sum();
                let batches: Vec<_> = batch_sizes.iter()
                    .map(|&size| create_test_batch(size))
                    .collect();

                let iter = MorselIterator::new(&batches);
                let total_actual: usize = iter.map(|m| m.num_rows()).sum();

                prop_assert_eq!(total_expected, total_actual);
            }

            /// Property: Empty batches are handled correctly
            #[test]
            fn prop_empty_batches_handled(
                num_empty in 0usize..10
            ) {
                let batches: Vec<_> = (0..num_empty)
                    .map(|_| create_test_batch(0))
                    .collect();

                let iter = MorselIterator::new(&batches);
                let total_rows: usize = iter.map(|m| m.num_rows()).sum();

                prop_assert_eq!(0, total_rows);
            }
        }
    }
}
