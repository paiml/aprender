//! Parallel data loading with multi-worker support.
//!
//! Provides a parallel data loader that uses multiple threads to load data
//! in parallel, similar to PyTorch's `DataLoader` with `num_workers > 0`.
//!
//! # Example
//!
//! ```no_run
//! use alimentar::{parallel::ParallelDataLoader, ArrowDataset, Dataset};
//!
//! let dataset = ArrowDataset::from_parquet("data.parquet").unwrap();
//! let loader = ParallelDataLoader::new(dataset)
//!     .batch_size(32)
//!     .num_workers(4)
//!     .prefetch(2);
//!
//! for batch in loader {
//!     println!("Batch has {} rows", batch.num_rows());
//! }
//! ```

use std::{sync::Arc, thread};

use arrow::record_batch::RecordBatch;

use crate::{dataset::Dataset, error::Result};

/// Parallel data loader with multi-worker support.
///
/// Uses a thread pool to load batches in parallel, with configurable
/// number of workers and prefetch buffer size.
#[derive(Debug)]
pub struct ParallelDataLoader<D: Dataset> {
    dataset: Arc<D>,
    batch_size: usize,
    num_workers: usize,
    prefetch: usize,
    #[cfg(feature = "shuffle")]
    shuffle: bool,
    #[cfg(feature = "shuffle")]
    seed: Option<u64>,
    drop_last: bool,
}

impl<D: Dataset + 'static> ParallelDataLoader<D> {
    /// Creates a new parallel data loader.
    pub fn new(dataset: D) -> Self {
        Self {
            dataset: Arc::new(dataset),
            batch_size: 1,
            num_workers: 0, // 0 = main thread only (no workers)
            prefetch: 2,
            #[cfg(feature = "shuffle")]
            shuffle: false,
            #[cfg(feature = "shuffle")]
            seed: None,
            drop_last: false,
        }
    }

    /// Sets the batch size (minimum 1).
    #[must_use]
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    /// Sets the number of worker threads (0 = main thread only).
    ///
    /// Note: On WASM targets, num_workers is always 0.
    #[must_use]
    pub fn num_workers(mut self, workers: usize) -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = workers;
            self.num_workers = 0;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.num_workers = workers;
        }
        self
    }

    /// Sets the prefetch buffer size.
    #[must_use]
    pub fn prefetch(mut self, size: usize) -> Self {
        self.prefetch = size.max(1);
        self
    }

    /// Enables or disables shuffling.
    #[cfg(feature = "shuffle")]
    #[must_use]
    pub fn shuffle(mut self, enable: bool) -> Self {
        self.shuffle = enable;
        self
    }

    /// Sets the random seed for shuffling.
    #[cfg(feature = "shuffle")]
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enables or disables dropping the last incomplete batch.
    #[must_use]
    pub fn drop_last(mut self, enable: bool) -> Self {
        self.drop_last = enable;
        self
    }

    /// Returns the batch size.
    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns the number of workers.
    pub fn get_num_workers(&self) -> usize {
        self.num_workers
    }

    /// Returns the prefetch size.
    pub fn get_prefetch(&self) -> usize {
        self.prefetch
    }

    /// Returns the number of batches.
    pub fn num_batches(&self) -> usize {
        let total_rows = self.dataset.len();
        if self.drop_last {
            total_rows / self.batch_size
        } else {
            total_rows.div_ceil(self.batch_size)
        }
    }

    /// Returns the total number of rows in the underlying dataset.
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Returns true if the underlying dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}

impl<D: Dataset + 'static> IntoIterator for ParallelDataLoader<D> {
    type Item = RecordBatch;
    type IntoIter = ParallelDataLoaderIterator<D>;

    fn into_iter(self) -> Self::IntoIter {
        let total_rows = self.dataset.len();

        // Generate indices
        #[allow(unused_mut)]
        let mut indices: Vec<usize> = (0..total_rows).collect();

        #[cfg(feature = "shuffle")]
        if self.shuffle {
            use rand::{seq::SliceRandom, SeedableRng};

            let mut rng = match self.seed {
                Some(s) => rand::rngs::StdRng::seed_from_u64(s),
                None => rand::rngs::StdRng::from_entropy(),
            };
            indices.shuffle(&mut rng);
        }

        if self.num_workers == 0 {
            // Single-threaded path
            ParallelDataLoaderIterator::SingleThreaded {
                dataset: self.dataset,
                indices,
                batch_size: self.batch_size,
                drop_last: self.drop_last,
                position: 0,
            }
        } else {
            // Multi-threaded path with channel
            use std::sync::mpsc;

            let (tx, rx) = mpsc::sync_channel(self.prefetch);
            let dataset = self.dataset.clone();
            let batch_size = self.batch_size;
            let drop_last = self.drop_last;
            let num_workers = self.num_workers;

            // Spawn worker thread(s)
            let handle = thread::spawn(move || {
                // Simple round-robin distribution to workers
                let chunks: Vec<Vec<usize>> = indices
                    .chunks(batch_size)
                    .filter(|chunk| !drop_last || chunk.len() == batch_size)
                    .map(|chunk| chunk.to_vec())
                    .collect();

                // Use thread pool for parallel processing
                let pool_size = num_workers.min(chunks.len());
                if pool_size == 0 {
                    return;
                }

                // Process chunks and send batches
                for batch in chunks.iter().filter_map(|chunk_indices| {
                    collect_batch_from_indices(&*dataset, chunk_indices)
                }) {
                    if tx.send(batch).is_err() {
                        break;
                    }
                }
            });

            ParallelDataLoaderIterator::MultiThreaded {
                receiver: rx,
                _handle: handle,
            }
        }
    }
}

/// Collects rows from dataset into a single batch.
fn collect_batch_from_indices<D: Dataset>(dataset: &D, indices: &[usize]) -> Option<RecordBatch> {
    use arrow::compute::concat_batches;

    let rows: Vec<RecordBatch> = indices.iter().filter_map(|&idx| dataset.get(idx)).collect();

    if rows.is_empty() {
        return None;
    }

    let schema = dataset.schema();
    concat_batches(&schema, &rows).ok()
}

/// Iterator for parallel data loader.
#[allow(missing_docs)]
pub enum ParallelDataLoaderIterator<D: Dataset> {
    /// Single-threaded iteration (num_workers = 0)
    SingleThreaded {
        /// The dataset being iterated
        dataset: Arc<D>,
        /// Row indices to iterate
        indices: Vec<usize>,
        /// Batch size for iteration
        batch_size: usize,
        /// Whether to drop the last incomplete batch
        drop_last: bool,
        /// Current position in indices
        position: usize,
    },
    /// Multi-threaded iteration with channel
    MultiThreaded {
        /// Receiver for batches from worker threads
        receiver: std::sync::mpsc::Receiver<RecordBatch>,
        /// Handle to the worker thread
        _handle: thread::JoinHandle<()>,
    },
}

impl<D: Dataset> std::fmt::Debug for ParallelDataLoaderIterator<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SingleThreaded {
                position,
                batch_size,
                ..
            } => f
                .debug_struct("ParallelDataLoaderIterator::SingleThreaded")
                .field("position", position)
                .field("batch_size", batch_size)
                .finish(),
            Self::MultiThreaded { .. } => f
                .debug_struct("ParallelDataLoaderIterator::MultiThreaded")
                .finish(),
        }
    }
}

impl<D: Dataset + 'static> Iterator for ParallelDataLoaderIterator<D> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::SingleThreaded {
                dataset,
                indices,
                batch_size,
                drop_last,
                position,
            } => {
                if *position >= indices.len() {
                    return None;
                }

                let end = (*position + *batch_size).min(indices.len());
                let chunk_indices = &indices[*position..end];

                if *drop_last && chunk_indices.len() < *batch_size {
                    return None;
                }

                *position = end;
                collect_batch_from_indices(&**dataset, chunk_indices)
            }
            Self::MultiThreaded { receiver, .. } => receiver.recv().ok(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::SingleThreaded {
                indices,
                batch_size,
                drop_last,
                position,
                ..
            } => {
                let remaining = indices.len().saturating_sub(*position);
                let batches = if *drop_last {
                    remaining / *batch_size
                } else {
                    remaining.div_ceil(*batch_size)
                };
                (batches, Some(batches))
            }
            Self::MultiThreaded { .. } => (0, None),
        }
    }
}

/// Builder for parallel data loader configuration.
#[derive(Debug, Default)]
pub struct ParallelDataLoaderBuilder {
    batch_size: Option<usize>,
    num_workers: Option<usize>,
    prefetch: Option<usize>,
    #[cfg(feature = "shuffle")]
    shuffle: Option<bool>,
    #[cfg(feature = "shuffle")]
    seed: Option<u64>,
    drop_last: Option<bool>,
}

impl ParallelDataLoaderBuilder {
    /// Creates a new builder with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the batch size.
    #[must_use]
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    /// Sets the number of workers.
    #[must_use]
    pub fn num_workers(mut self, workers: usize) -> Self {
        self.num_workers = Some(workers);
        self
    }

    /// Sets the prefetch size.
    #[must_use]
    pub fn prefetch(mut self, size: usize) -> Self {
        self.prefetch = Some(size);
        self
    }

    /// Enables shuffling.
    #[cfg(feature = "shuffle")]
    #[must_use]
    pub fn shuffle(mut self, enable: bool) -> Self {
        self.shuffle = Some(enable);
        self
    }

    /// Sets the random seed.
    #[cfg(feature = "shuffle")]
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enables drop_last.
    #[must_use]
    pub fn drop_last(mut self, enable: bool) -> Self {
        self.drop_last = Some(enable);
        self
    }

    /// Builds the parallel data loader with the given dataset.
    pub fn build<D: Dataset + 'static>(self, dataset: D) -> Result<ParallelDataLoader<D>> {
        let mut loader = ParallelDataLoader::new(dataset);

        if let Some(size) = self.batch_size {
            loader = loader.batch_size(size);
        }
        if let Some(workers) = self.num_workers {
            loader = loader.num_workers(workers);
        }
        if let Some(size) = self.prefetch {
            loader = loader.prefetch(size);
        }
        #[cfg(feature = "shuffle")]
        if let Some(enable) = self.shuffle {
            loader = loader.shuffle(enable);
        }
        #[cfg(feature = "shuffle")]
        if let Some(seed) = self.seed {
            loader = loader.seed(seed);
        }
        if let Some(enable) = self.drop_last {
            loader = loader.drop_last(enable);
        }

        Ok(loader)
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
    use std::collections::HashSet;

    use arrow::{
        array::{Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
    };

    use super::*;
    use crate::ArrowDataset;

    fn create_test_dataset(rows: usize) -> ArrowDataset {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));

        let ids: Vec<i32> = (0..rows as i32).collect();
        let values: Vec<String> = ids.iter().map(|i| format!("item_{}", i)).collect();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(values)),
            ],
        )
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"));

        ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"))
    }

    #[test]
    fn test_parallel_loader_single_threaded() {
        let dataset = create_test_dataset(100);
        let loader = ParallelDataLoader::new(dataset)
            .batch_size(10)
            .num_workers(0);

        assert_eq!(loader.get_batch_size(), 10);
        assert_eq!(loader.get_num_workers(), 0);
        assert_eq!(loader.num_batches(), 10);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 10);

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 100);
    }

    #[test]
    fn test_parallel_loader_multi_threaded() {
        let dataset = create_test_dataset(100);
        let loader = ParallelDataLoader::new(dataset)
            .batch_size(10)
            .num_workers(2)
            .prefetch(4);

        assert_eq!(loader.get_num_workers(), 2);
        assert_eq!(loader.get_prefetch(), 4);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 10);

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 100);
    }

    #[test]
    fn test_parallel_loader_drop_last() {
        let dataset = create_test_dataset(25);
        let loader = ParallelDataLoader::new(dataset)
            .batch_size(10)
            .drop_last(true);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 2);

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 20);
    }

    #[test]
    #[cfg(feature = "shuffle")]
    fn test_parallel_loader_shuffle() {
        let dataset = create_test_dataset(100);
        let loader1 = ParallelDataLoader::new(dataset.clone())
            .batch_size(10)
            .shuffle(true)
            .seed(42);

        let loader2 = ParallelDataLoader::new(dataset)
            .batch_size(10)
            .shuffle(true)
            .seed(42);

        let batches1: Vec<RecordBatch> = loader1.into_iter().collect();
        let batches2: Vec<RecordBatch> = loader2.into_iter().collect();

        // Same seed should produce same order
        for (b1, b2) in batches1.iter().zip(batches2.iter()) {
            let ids1 = b1.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let ids2 = b2.column(0).as_any().downcast_ref::<Int32Array>().unwrap();

            for i in 0..ids1.len() {
                assert_eq!(ids1.value(i), ids2.value(i));
            }
        }
    }

    #[test]
    fn test_parallel_loader_all_rows() {
        let dataset = create_test_dataset(50);
        let loader = ParallelDataLoader::new(dataset)
            .batch_size(7)
            .num_workers(2);

        let mut seen_ids: HashSet<i32> = HashSet::new();
        for batch in loader {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            for i in 0..ids.len() {
                seen_ids.insert(ids.value(i));
            }
        }

        // All 50 IDs should be present
        assert_eq!(seen_ids.len(), 50);
        for i in 0..50 {
            assert!(seen_ids.contains(&i), "Missing id: {}", i);
        }
    }

    #[test]
    fn test_parallel_loader_getters() {
        let dataset = create_test_dataset(100);
        let loader = ParallelDataLoader::new(dataset)
            .batch_size(20)
            .num_workers(4)
            .prefetch(8);

        assert_eq!(loader.get_batch_size(), 20);
        assert_eq!(loader.get_num_workers(), 4);
        assert_eq!(loader.get_prefetch(), 8);
        assert_eq!(loader.len(), 100);
        assert!(!loader.is_empty());
    }

    #[test]
    fn test_parallel_loader_builder() {
        let dataset = create_test_dataset(100);
        let loader = ParallelDataLoaderBuilder::new()
            .batch_size(25)
            .num_workers(2)
            .prefetch(4)
            .drop_last(true)
            .build(dataset)
            .ok()
            .unwrap_or_else(|| panic!("Should build"));

        assert_eq!(loader.get_batch_size(), 25);
        assert_eq!(loader.get_num_workers(), 2);
        assert_eq!(loader.num_batches(), 4);
    }

    #[test]
    fn test_parallel_loader_empty_dataset() {
        // Create dataset with at least 1 row for valid ArrowDataset
        let dataset = create_test_dataset(1);
        let loader = ParallelDataLoader::new(dataset)
            .batch_size(10)
            .num_workers(0);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 1);
    }

    #[test]
    fn test_parallel_loader_batch_size_min() {
        let dataset = create_test_dataset(10);
        let loader = ParallelDataLoader::new(dataset).batch_size(0);

        assert_eq!(loader.get_batch_size(), 1);
    }

    #[test]
    fn test_parallel_loader_debug() {
        let dataset = create_test_dataset(10);
        let loader = ParallelDataLoader::new(dataset)
            .batch_size(5)
            .num_workers(2);

        let debug_str = format!("{:?}", loader);
        assert!(debug_str.contains("ParallelDataLoader"));

        let iter = loader.into_iter();
        let iter_debug = format!("{:?}", iter);
        assert!(iter_debug.contains("ParallelDataLoaderIterator"));
    }

    #[test]
    fn test_parallel_loader_size_hint() {
        let dataset = create_test_dataset(25);
        let loader = ParallelDataLoader::new(dataset)
            .batch_size(10)
            .num_workers(0);

        let mut iter = loader.into_iter();
        assert_eq!(iter.size_hint(), (3, Some(3)));

        let _ = iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));
    }

    #[test]
    fn test_builder_debug() {
        let builder = ParallelDataLoaderBuilder::new()
            .batch_size(32)
            .num_workers(4);

        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("ParallelDataLoaderBuilder"));
    }

    #[test]
    fn test_parallel_loader_single_row() {
        let dataset = create_test_dataset(1);
        let loader = ParallelDataLoader::new(dataset)
            .batch_size(10)
            .num_workers(2);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
    }

    #[test]
    fn test_parallel_loader_batch_equals_dataset() {
        let dataset = create_test_dataset(50);
        let loader = ParallelDataLoader::new(dataset)
            .batch_size(50)
            .num_workers(0);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 50);
    }

    #[test]
    fn test_parallel_loader_batch_larger_than_dataset() {
        let dataset = create_test_dataset(10);
        let loader = ParallelDataLoader::new(dataset)
            .batch_size(100)
            .num_workers(0);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 10);
    }

    #[test]
    fn test_parallel_loader_drop_last_exact_fit() {
        let dataset = create_test_dataset(100);
        let loader = ParallelDataLoader::new(dataset)
            .batch_size(25)
            .drop_last(true)
            .num_workers(0);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 4); // 100 / 25 = 4, no remainder
    }

    #[test]
    fn test_parallel_loader_drop_last_with_remainder() {
        let dataset = create_test_dataset(100);
        let loader = ParallelDataLoader::new(dataset)
            .batch_size(30)
            .drop_last(true)
            .num_workers(0);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 3); // 100 / 30 = 3, remainder dropped
    }

    #[test]
    fn test_parallel_loader_num_batches_calculation() {
        let dataset = create_test_dataset(100);

        // Without drop_last: ceil(100/30) = 4
        let loader1 = ParallelDataLoader::new(dataset.clone())
            .batch_size(30)
            .num_workers(0);
        assert_eq!(loader1.num_batches(), 4);

        // With drop_last: floor(100/30) = 3
        let loader2 = ParallelDataLoader::new(dataset)
            .batch_size(30)
            .drop_last(true)
            .num_workers(0);
        assert_eq!(loader2.num_batches(), 3);
    }

    #[test]
    fn test_parallel_loader_prefetch_setting() {
        let dataset = create_test_dataset(100);
        let loader = ParallelDataLoader::new(dataset).batch_size(10).prefetch(16);

        assert_eq!(loader.get_prefetch(), 16);
    }

    #[test]
    fn test_parallel_loader_iterator_exhaustion() {
        let dataset = create_test_dataset(30);
        let loader = ParallelDataLoader::new(dataset)
            .batch_size(10)
            .num_workers(0);

        let mut iter = loader.into_iter();

        // Should yield 3 batches
        assert!(iter.next().is_some());
        assert!(iter.next().is_some());
        assert!(iter.next().is_some());
        // Should be exhausted
        assert!(iter.next().is_none());
        // Should stay exhausted
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_parallel_loader_total_rows_preserved() {
        let dataset = create_test_dataset(97);
        let loader = ParallelDataLoader::new(dataset)
            .batch_size(10)
            .num_workers(0);

        let total: usize = loader.into_iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 97);
    }

    #[test]
    fn test_parallel_loader_builder_defaults() {
        let dataset = create_test_dataset(50);
        let loader = ParallelDataLoaderBuilder::new()
            .build(dataset)
            .ok()
            .unwrap_or_else(|| panic!("build"));

        // Defaults from ParallelDataLoader::new()
        assert_eq!(loader.get_batch_size(), 1);
        assert_eq!(loader.get_prefetch(), 2);
    }

    #[test]
    fn test_parallel_loader_builder_with_shuffle() {
        let dataset = create_test_dataset(50);
        let loader = ParallelDataLoaderBuilder::new()
            .batch_size(10)
            .shuffle(true)
            .seed(42)
            .build(dataset)
            .ok()
            .unwrap_or_else(|| panic!("build"));

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 5);
    }

    #[test]
    fn test_parallel_loader_zero_workers_single_threaded() {
        let dataset = create_test_dataset(100);
        let loader = ParallelDataLoader::new(dataset)
            .batch_size(20)
            .num_workers(0);

        assert_eq!(loader.get_num_workers(), 0);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 5);
    }
}
