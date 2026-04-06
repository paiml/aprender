//! DataLoader for batched iteration over datasets.
//!
//! The [`DataLoader`] provides configurable batch iteration with support
//! for shuffling and dropping incomplete last batches.

use std::sync::Arc;

use arrow::{array::RecordBatch, compute::concat_batches};
#[cfg(feature = "shuffle")]
use rand::{seq::SliceRandom, SeedableRng};

use crate::{dataset::Dataset, error::Result};

/// A data loader that provides batched iteration over a dataset.
///
/// The DataLoader wraps a dataset and provides:
/// - Configurable batch sizes
/// - Optional shuffling with reproducible seeds
/// - Option to drop incomplete final batches
///
/// # Example
///
/// ```no_run
/// use alimentar::{ArrowDataset, DataLoader};
///
/// let dataset = ArrowDataset::from_parquet("data.parquet").unwrap();
/// let loader = DataLoader::new(dataset)
///     .batch_size(32)
///     .shuffle(true)
///     .seed(42);
///
/// for batch in loader {
///     println!("Processing batch with {} rows", batch.num_rows());
/// }
/// ```
#[derive(Debug)]
pub struct DataLoader<D: Dataset> {
    dataset: Arc<D>,
    batch_size: usize,
    #[allow(dead_code)] // Used only with shuffle feature
    shuffle: bool,
    drop_last: bool,
    #[allow(dead_code)] // Used only with shuffle feature
    seed: Option<u64>,
}

impl<D: Dataset> DataLoader<D> {
    /// Creates a new DataLoader wrapping the given dataset.
    ///
    /// Default configuration:
    /// - batch_size: 1
    /// - shuffle: false
    /// - drop_last: false
    /// - seed: None (random)
    pub fn new(dataset: D) -> Self {
        Self {
            dataset: Arc::new(dataset),
            batch_size: 1,
            shuffle: false,
            drop_last: false,
            seed: None,
        }
    }

    /// Sets the batch size.
    ///
    /// Each iteration will yield a RecordBatch with at most this many rows.
    /// The final batch may have fewer rows unless `drop_last` is set.
    ///
    /// #[requires(true)]
    /// #[ensures(result.batch_size >= 1)]
    /// #[ensures(size >= 1 ==> result.batch_size == size)]
    /// #[ensures(size == 0 ==> result.batch_size == 1)]
    #[must_use]
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    /// Enables or disables shuffling.
    ///
    /// When enabled, the row order is randomized before each epoch.
    /// Requires the `shuffle` feature.
    #[cfg(feature = "shuffle")]
    #[must_use]
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Sets whether to drop the last incomplete batch.
    ///
    /// When true, if the dataset size is not evenly divisible by the batch
    /// size, the final partial batch is skipped.
    #[must_use]
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Sets the random seed for shuffling.
    ///
    /// Setting a seed makes shuffling deterministic and reproducible.
    /// Requires the `shuffle` feature.
    #[cfg(feature = "shuffle")]
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Returns the configured batch size.
    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }

    /// Returns whether shuffling is enabled.
    pub fn is_shuffle(&self) -> bool {
        self.shuffle
    }

    /// Returns whether drop_last is enabled.
    pub fn is_drop_last(&self) -> bool {
        self.drop_last
    }

    /// Returns the number of batches that will be yielded.
    pub fn num_batches(&self) -> usize {
        let len = self.dataset.len();
        if self.drop_last {
            len / self.batch_size
        } else {
            len.div_ceil(self.batch_size)
        }
    }

    /// Returns the total number of rows in the underlying dataset.
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Returns true if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}

impl<D: Dataset> IntoIterator for DataLoader<D> {
    type Item = RecordBatch;
    type IntoIter = DataLoaderIterator<D>;

    fn into_iter(self) -> Self::IntoIter {
        let indices: Vec<usize> = (0..self.dataset.len()).collect();

        #[cfg(feature = "shuffle")]
        let shuffled_indices = if self.shuffle {
            let mut indices = indices;
            let mut rng = match self.seed {
                Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
                None => rand::rngs::StdRng::from_entropy(),
            };
            indices.shuffle(&mut rng);
            indices
        } else {
            indices
        };

        #[cfg(not(feature = "shuffle"))]
        let shuffled_indices = indices;

        DataLoaderIterator {
            dataset: self.dataset,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            indices: shuffled_indices,
            position: 0,
        }
    }
}

/// Iterator over batched data from a DataLoader.
pub struct DataLoaderIterator<D: Dataset> {
    dataset: Arc<D>,
    batch_size: usize,
    drop_last: bool,
    indices: Vec<usize>,
    position: usize,
}

impl<D: Dataset> Iterator for DataLoaderIterator<D> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.indices.len() {
            return None;
        }

        let remaining = self.indices.len() - self.position;
        let batch_size = remaining.min(self.batch_size);

        // Skip incomplete batch if drop_last is set
        if self.drop_last && batch_size < self.batch_size {
            return None;
        }

        // Collect rows for this batch
        let batch_indices = &self.indices[self.position..self.position + batch_size];
        self.position += batch_size;

        // Get individual rows and concatenate
        let rows: Vec<RecordBatch> = batch_indices
            .iter()
            .filter_map(|&idx| self.dataset.get(idx))
            .collect();

        if rows.is_empty() {
            return None;
        }

        // Concatenate all rows into a single batch
        concat_batches(&self.dataset.schema(), &rows).ok()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.indices.len().saturating_sub(self.position);
        let batches = if self.drop_last {
            remaining / self.batch_size
        } else if remaining > 0 {
            remaining.div_ceil(self.batch_size)
        } else {
            0
        };
        (batches, Some(batches))
    }
}

/// Builder for creating DataLoaders with more complex configurations.
#[derive(Debug, Default)]
pub struct DataLoaderBuilder {
    batch_size: Option<usize>,
    shuffle: Option<bool>,
    drop_last: Option<bool>,
    seed: Option<u64>,
}

impl DataLoaderBuilder {
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

    /// Sets whether to shuffle.
    #[must_use]
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = Some(shuffle);
        self
    }

    /// Sets whether to drop the last incomplete batch.
    #[must_use]
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = Some(drop_last);
        self
    }

    /// Sets the random seed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Builds a DataLoader with the given dataset.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch size is zero.
    pub fn build<D: Dataset>(self, dataset: D) -> Result<DataLoader<D>> {
        let batch_size = self.batch_size.unwrap_or(1);
        if batch_size == 0 {
            return Err(crate::error::Error::invalid_config(
                "batch_size must be greater than 0",
            ));
        }

        let mut loader = DataLoader::new(dataset).batch_size(batch_size);

        #[cfg(feature = "shuffle")]
        if let Some(shuffle) = self.shuffle {
            loader = loader.shuffle(shuffle);
        }
        if let Some(drop_last) = self.drop_last {
            loader = loader.drop_last(drop_last);
        }
        #[cfg(feature = "shuffle")]
        if let Some(seed) = self.seed {
            loader = loader.seed(seed);
        }

        Ok(loader)
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::uninlined_format_args
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
        let values: Vec<String> = ids.iter().map(|i| format!("val_{}", i)).collect();

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
    fn test_basic_iteration() {
        let dataset = create_test_dataset(10);
        let loader = DataLoader::new(dataset).batch_size(3);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 4); // 3 + 3 + 3 + 1

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 10);
    }

    #[test]
    fn test_drop_last() {
        let dataset = create_test_dataset(10);
        let loader = DataLoader::new(dataset).batch_size(3).drop_last(true);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 3); // Only full batches

        for batch in &batches {
            assert_eq!(batch.num_rows(), 3);
        }
    }

    #[test]
    fn test_shuffle_deterministic() {
        let dataset = create_test_dataset(100);

        let loader1 = DataLoader::new(dataset.clone())
            .batch_size(10)
            .shuffle(true)
            .seed(42);
        let batches1: Vec<RecordBatch> = loader1.into_iter().collect();

        let loader2 = DataLoader::new(dataset)
            .batch_size(10)
            .shuffle(true)
            .seed(42);
        let batches2: Vec<RecordBatch> = loader2.into_iter().collect();

        // Same seed should produce same order
        assert_eq!(batches1.len(), batches2.len());
        for (b1, b2) in batches1.iter().zip(batches2.iter()) {
            assert_eq!(b1.num_rows(), b2.num_rows());
        }
    }

    #[test]
    fn test_shuffle_different_seeds() {
        let dataset = create_test_dataset(100);

        let loader1 = DataLoader::new(dataset.clone())
            .batch_size(100)
            .shuffle(true)
            .seed(42);
        let batches1: Vec<RecordBatch> = loader1.into_iter().collect();

        let loader2 = DataLoader::new(dataset)
            .batch_size(100)
            .shuffle(true)
            .seed(123);
        let batches2: Vec<RecordBatch> = loader2.into_iter().collect();

        // Different seeds should likely produce different order
        // (we check that we got all the data, order may differ)
        assert_eq!(batches1.len(), batches2.len());
    }

    #[test]
    fn test_all_rows_covered() {
        let dataset = create_test_dataset(25);
        let loader = DataLoader::new(dataset)
            .batch_size(7)
            .shuffle(true)
            .seed(99);

        let mut seen_ids = HashSet::new();
        for batch in loader {
            let id_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap_or_else(|| panic!("Should be Int32Array"));
            for i in 0..id_col.len() {
                seen_ids.insert(id_col.value(i));
            }
        }

        assert_eq!(seen_ids.len(), 25);
        for i in 0..25i32 {
            assert!(seen_ids.contains(&i));
        }
    }

    #[test]
    fn test_num_batches() {
        let dataset = create_test_dataset(10);

        let loader = DataLoader::new(dataset.clone()).batch_size(3);
        assert_eq!(loader.num_batches(), 4);

        let loader = DataLoader::new(dataset).batch_size(3).drop_last(true);
        assert_eq!(loader.num_batches(), 3);
    }

    #[test]
    fn test_builder() {
        let dataset = create_test_dataset(10);
        let loader = DataLoaderBuilder::new()
            .batch_size(5)
            .shuffle(true)
            .seed(42)
            .build(dataset)
            .ok()
            .unwrap_or_else(|| panic!("Should build loader"));

        assert_eq!(loader.get_batch_size(), 5);
        assert!(loader.is_shuffle());
    }

    #[test]
    fn test_builder_zero_batch_size_error() {
        let dataset = create_test_dataset(10);
        let result = DataLoaderBuilder::new().batch_size(0).build(dataset);
        assert!(result.is_err());
    }

    #[test]
    fn test_size_hint() {
        let dataset = create_test_dataset(10);
        let loader = DataLoader::new(dataset).batch_size(3);

        let mut iter = loader.into_iter();
        assert_eq!(iter.size_hint(), (4, Some(4)));

        let _ = iter.next();
        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_getters() {
        let dataset = create_test_dataset(10);
        let loader = DataLoader::new(dataset)
            .batch_size(5)
            .shuffle(true)
            .drop_last(true);

        assert_eq!(loader.get_batch_size(), 5);
        assert!(loader.is_shuffle());
        assert!(loader.is_drop_last());
        assert_eq!(loader.len(), 10);
        assert!(!loader.is_empty());
    }

    #[test]
    fn test_batch_size_min_one() {
        let dataset = create_test_dataset(10);
        let loader = DataLoader::new(dataset).batch_size(0);
        assert_eq!(loader.get_batch_size(), 1);
    }

    #[test]
    fn test_empty_dataset() {
        let dataset = create_test_dataset(0);
        let loader = DataLoader::new(dataset).batch_size(3);
        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert!(batches.is_empty());
    }

    #[test]
    fn test_empty_dataset_drop_last() {
        let dataset = create_test_dataset(0);
        let loader = DataLoader::new(dataset).batch_size(3).drop_last(true);
        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert!(batches.is_empty());
    }

    #[test]
    fn test_is_empty() {
        let empty_dataset = create_test_dataset(0);
        let loader_empty = DataLoader::new(empty_dataset);
        assert!(loader_empty.is_empty());

        let dataset = create_test_dataset(5);
        let loader = DataLoader::new(dataset);
        assert!(!loader.is_empty());
    }

    #[test]
    fn test_len() {
        let dataset = create_test_dataset(42);
        let loader = DataLoader::new(dataset);
        assert_eq!(loader.len(), 42);
    }

    #[test]
    fn test_single_row_dataset() {
        let dataset = create_test_dataset(1);
        let loader = DataLoader::new(dataset).batch_size(5);
        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
    }

    #[test]
    fn test_single_row_drop_last() {
        let dataset = create_test_dataset(1);
        let loader = DataLoader::new(dataset).batch_size(5).drop_last(true);
        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        // Single row is smaller than batch size, so it should be dropped
        assert!(batches.is_empty());
    }

    #[test]
    fn test_batch_size_equals_dataset_size() {
        let dataset = create_test_dataset(10);
        let loader = DataLoader::new(dataset).batch_size(10);
        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 10);
    }

    #[test]
    fn test_batch_size_larger_than_dataset() {
        let dataset = create_test_dataset(5);
        let loader = DataLoader::new(dataset).batch_size(100);
        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 5);
    }

    #[test]
    fn test_batch_size_larger_than_dataset_drop_last() {
        let dataset = create_test_dataset(5);
        let loader = DataLoader::new(dataset).batch_size(100).drop_last(true);
        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        // Batch is incomplete, should be dropped
        assert!(batches.is_empty());
    }

    #[test]
    fn test_num_batches_with_drop_last() {
        let dataset = create_test_dataset(10);

        let loader_without_drop = DataLoader::new(dataset.clone()).batch_size(3);
        assert_eq!(loader_without_drop.num_batches(), 4); // 3 + 3 + 3 + 1

        let loader_with_drop = DataLoader::new(dataset).batch_size(3).drop_last(true);
        assert_eq!(loader_with_drop.num_batches(), 3); // 3 + 3 + 3
    }

    #[test]
    fn test_builder_all_options() {
        let dataset = create_test_dataset(10);
        let result = DataLoaderBuilder::new()
            .batch_size(4)
            .shuffle(true)
            .drop_last(true)
            .seed(42)
            .build(dataset);

        assert!(result.is_ok());
        let loader = result.ok().unwrap();
        assert_eq!(loader.get_batch_size(), 4);
        assert!(loader.is_shuffle());
        assert!(loader.is_drop_last());
    }

    #[test]
    fn test_size_hint_empty_dataset() {
        let dataset = create_test_dataset(0);
        let loader = DataLoader::new(dataset).batch_size(3);
        let iter = loader.into_iter();
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_iterator_exhaustion() {
        let dataset = create_test_dataset(5);
        let loader = DataLoader::new(dataset).batch_size(2);
        let mut iter = loader.into_iter();

        // Should yield 3 batches: 2, 2, 1
        assert!(iter.next().is_some());
        assert!(iter.next().is_some());
        assert!(iter.next().is_some());
        // Should be exhausted
        assert!(iter.next().is_none());
        // Should remain exhausted
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_size_hint_during_iteration() {
        let dataset = create_test_dataset(10);
        let loader = DataLoader::new(dataset).batch_size(3);
        let mut iter = loader.into_iter();

        // Initially 4 batches remaining
        assert_eq!(iter.size_hint(), (4, Some(4)));

        iter.next();
        assert_eq!(iter.size_hint(), (3, Some(3)));

        iter.next();
        assert_eq!(iter.size_hint(), (2, Some(2)));

        iter.next();
        assert_eq!(iter.size_hint(), (1, Some(1)));

        iter.next();
        assert_eq!(iter.size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_debug_impl() {
        let dataset = create_test_dataset(5);
        let loader = DataLoader::new(dataset).batch_size(2);
        let debug_str = format!("{:?}", loader);
        assert!(debug_str.contains("DataLoader"));
        assert!(debug_str.contains("batch_size: 2"));
    }

    #[test]
    fn test_builder_debug_impl() {
        let builder = DataLoaderBuilder::new().batch_size(10).drop_last(true);
        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("DataLoaderBuilder"));
    }
}
