//! Weighted DataLoader for importance sampling.
//!
//! Provides [`WeightedDataLoader`] for sampling with per-sample weights,
//! enabling importance sampling for imbalanced datasets or CITL reweighting.

use std::sync::Arc;

use arrow::{array::RecordBatch, compute::concat_batches};
#[cfg(feature = "shuffle")]
use rand::{distributions::WeightedIndex, prelude::Distribution, SeedableRng};

use crate::{dataset::Dataset, error::Result, Error};

/// A data loader that samples with per-sample weights.
///
/// Unlike [`DataLoader`](crate::DataLoader) which samples uniformly,
/// `WeightedDataLoader` samples proportional to the provided weights.
/// This is useful for:
/// - Importance sampling in imbalanced datasets
/// - CITL reweighting (`--reweight 1.5` for compiler-verified labels)
/// - Curriculum learning with difficulty-based sampling
///
/// # Example
///
/// ```no_run
/// use alimentar::{ArrowDataset, Dataset, WeightedDataLoader};
///
/// let dataset = ArrowDataset::from_parquet("data.parquet").unwrap();
/// let weights = vec![1.0; dataset.len()]; // Uniform weights
///
/// let loader = WeightedDataLoader::new(dataset, weights)
///     .unwrap()
///     .batch_size(32)
///     .seed(42);
///
/// for batch in loader {
///     println!("Batch with {} rows", batch.num_rows());
/// }
/// ```
#[derive(Debug)]
pub struct WeightedDataLoader<D: Dataset> {
    dataset: Arc<D>,
    weights: Vec<f32>,
    batch_size: usize,
    num_samples: usize,
    drop_last: bool,
    #[allow(dead_code)] // Used only with shuffle feature
    seed: Option<u64>,
}

impl<D: Dataset> WeightedDataLoader<D> {
    /// Creates a new weighted data loader.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The dataset to sample from
    /// * `weights` - Per-sample weights (must match dataset length)
    ///
    /// # Errors
    ///
    /// Returns an error if weights length doesn't match dataset length,
    /// or if any weight is negative.
    pub fn new(dataset: D, weights: Vec<f32>) -> Result<Self> {
        let len = dataset.len();
        if weights.len() != len {
            return Err(Error::invalid_config(format!(
                "weights length {} doesn't match dataset length {}",
                weights.len(),
                len
            )));
        }

        if weights.iter().any(|&w| w < 0.0) {
            return Err(Error::invalid_config("weights must be non-negative"));
        }

        Ok(Self {
            dataset: Arc::new(dataset),
            weights,
            batch_size: 1,
            num_samples: len,
            drop_last: false,
            seed: None,
        })
    }

    /// Creates a weighted loader with a uniform reweight factor.
    ///
    /// Multiplies all weights by the given factor. Useful for CITL's
    /// `--reweight 1.5` which boosts compiler-verified samples.
    ///
    /// # Arguments
    ///
    /// * `dataset` - The dataset to sample from
    /// * `reweight` - Factor to multiply all weights by
    pub fn with_reweight(dataset: D, reweight: f32) -> Result<Self> {
        let len = dataset.len();
        let weights = vec![reweight; len];
        Self::new(dataset, weights)
    }

    /// Sets the batch size.
    #[must_use]
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    /// Sets the total number of samples per epoch.
    ///
    /// By default, samples `len()` items per epoch. Set this to oversample
    /// or undersample the dataset.
    #[must_use]
    pub fn num_samples(mut self, n: usize) -> Self {
        self.num_samples = n;
        self
    }

    /// Sets whether to drop the last incomplete batch.
    #[must_use]
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Sets the random seed for reproducibility.
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

    /// Returns the number of samples per epoch.
    pub fn get_num_samples(&self) -> usize {
        self.num_samples
    }

    /// Returns the weights.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Returns the number of batches that will be yielded.
    pub fn num_batches(&self) -> usize {
        if self.drop_last {
            self.num_samples / self.batch_size
        } else {
            self.num_samples.div_ceil(self.batch_size)
        }
    }

    /// Returns the dataset length.
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Returns true if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}

#[cfg(feature = "shuffle")]
impl<D: Dataset> IntoIterator for WeightedDataLoader<D> {
    type Item = RecordBatch;
    type IntoIter = WeightedDataLoaderIterator<D>;

    fn into_iter(self) -> Self::IntoIter {
        // Create weighted index for sampling
        let dist = WeightedIndex::new(&self.weights).ok();

        WeightedDataLoaderIterator {
            dataset: self.dataset,
            dist,
            batch_size: self.batch_size,
            num_samples: self.num_samples,
            drop_last: self.drop_last,
            rng: match self.seed {
                Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
                None => rand::rngs::StdRng::from_entropy(),
            },
            samples_yielded: 0,
        }
    }
}

/// Iterator over weighted sampled batches.
#[cfg(feature = "shuffle")]
pub struct WeightedDataLoaderIterator<D: Dataset> {
    dataset: Arc<D>,
    dist: Option<WeightedIndex<f32>>,
    batch_size: usize,
    num_samples: usize,
    drop_last: bool,
    rng: rand::rngs::StdRng,
    samples_yielded: usize,
}

#[cfg(feature = "shuffle")]
impl<D: Dataset> Iterator for WeightedDataLoaderIterator<D> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.samples_yielded >= self.num_samples {
            return None;
        }

        let remaining = self.num_samples - self.samples_yielded;
        let batch_size = remaining.min(self.batch_size);

        // Skip incomplete batch if drop_last is set
        if self.drop_last && batch_size < self.batch_size {
            return None;
        }

        // Sample indices according to weights
        let indices: Vec<usize> = if let Some(dist) = &self.dist {
            (0..batch_size)
                .map(|_| dist.sample(&mut self.rng))
                .collect()
        } else {
            // Fallback: uniform sampling if weights are all zero
            let len = self.dataset.len();
            if len == 0 {
                return None;
            }
            (0..batch_size)
                .map(|i| (self.samples_yielded + i) % len)
                .collect()
        };

        self.samples_yielded += batch_size;

        // Get rows and concatenate
        let rows: Vec<RecordBatch> = indices
            .iter()
            .filter_map(|&idx| self.dataset.get(idx))
            .collect();

        if rows.is_empty() {
            return None;
        }

        concat_batches(&self.dataset.schema(), &rows).ok()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.num_samples.saturating_sub(self.samples_yielded);
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

#[cfg(feature = "shuffle")]
impl<D: Dataset> std::fmt::Debug for WeightedDataLoaderIterator<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightedDataLoaderIterator")
            .field("batch_size", &self.batch_size)
            .field("num_samples", &self.num_samples)
            .field("samples_yielded", &self.samples_yielded)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[cfg(feature = "shuffle")]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::float_cmp
)]
mod tests {
    use std::collections::HashMap;

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
    fn test_weighted_loader_creation() {
        let dataset = create_test_dataset(10);
        let weights = vec![1.0; 10];

        let loader = WeightedDataLoader::new(dataset, weights);
        assert!(loader.is_ok());

        let loader = loader
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"));
        assert_eq!(loader.len(), 10);
        assert_eq!(loader.get_num_samples(), 10);
    }

    #[test]
    fn test_weighted_loader_wrong_length() {
        let dataset = create_test_dataset(10);
        let weights = vec![1.0; 5]; // Wrong length

        let result = WeightedDataLoader::new(dataset, weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_loader_negative_weight() {
        let dataset = create_test_dataset(10);
        let mut weights = vec![1.0; 10];
        weights[5] = -1.0; // Negative weight

        let result = WeightedDataLoader::new(dataset, weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_loader_with_reweight() {
        let dataset = create_test_dataset(10);

        let loader = WeightedDataLoader::with_reweight(dataset, 1.5)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"));

        assert!(loader.weights().iter().all(|&w| w == 1.5));
    }

    #[test]
    fn test_weighted_loader_basic_iteration() {
        let dataset = create_test_dataset(10);
        let weights = vec![1.0; 10];

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(3)
            .seed(42);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 4); // ceil(10/3) = 4

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 10);
    }

    #[test]
    fn test_weighted_loader_drop_last() {
        let dataset = create_test_dataset(10);
        let weights = vec![1.0; 10];

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(3)
            .drop_last(true)
            .seed(42);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 3); // 10/3 = 3 full batches

        for batch in &batches {
            assert_eq!(batch.num_rows(), 3);
        }
    }

    #[test]
    fn test_weighted_loader_deterministic() {
        let dataset = create_test_dataset(100);
        let weights = vec![1.0; 100];

        let loader1 = WeightedDataLoader::new(dataset.clone(), weights.clone())
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(10)
            .seed(42);
        let batches1: Vec<RecordBatch> = loader1.into_iter().collect();

        let loader2 = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(10)
            .seed(42);
        let batches2: Vec<RecordBatch> = loader2.into_iter().collect();

        assert_eq!(batches1.len(), batches2.len());
        for (b1, b2) in batches1.iter().zip(batches2.iter()) {
            let ids1 = b1
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap_or_else(|| panic!("Should be Int32Array"));
            let ids2 = b2
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap_or_else(|| panic!("Should be Int32Array"));

            for i in 0..ids1.len() {
                assert_eq!(ids1.value(i), ids2.value(i));
            }
        }
    }

    #[test]
    fn test_weighted_loader_biased_sampling() {
        // Create dataset with 10 items, heavily weight item 0
        let dataset = create_test_dataset(10);
        let mut weights = vec![0.1; 10];
        weights[0] = 10.0; // Item 0 should appear much more often

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(1)
            .num_samples(1000) // Large sample to see distribution
            .seed(42);

        let mut counts: HashMap<i32, usize> = HashMap::new();
        for batch in loader {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap_or_else(|| panic!("Should be Int32Array"));
            for i in 0..ids.len() {
                *counts.entry(ids.value(i)).or_insert(0) += 1;
            }
        }

        // Item 0 should appear significantly more than others
        let count_0 = *counts.get(&0).unwrap_or(&0);
        let count_1 = *counts.get(&1).unwrap_or(&0);

        // With weights 10.0 vs 0.1, item 0 should appear ~100x more often
        assert!(
            count_0 > count_1 * 10,
            "Item 0 ({}) should appear much more than item 1 ({})",
            count_0,
            count_1
        );
    }

    #[test]
    fn test_weighted_loader_num_samples() {
        let dataset = create_test_dataset(10);
        let weights = vec![1.0; 10];

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(5)
            .num_samples(25) // More than dataset size
            .seed(42);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 5); // ceil(25/5) = 5

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 25);
    }

    #[test]
    fn test_weighted_loader_num_batches() {
        let dataset = create_test_dataset(10);
        let weights = vec![1.0; 10];

        let loader = WeightedDataLoader::new(dataset.clone(), weights.clone())
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(3);
        assert_eq!(loader.num_batches(), 4);

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(3)
            .drop_last(true);
        assert_eq!(loader.num_batches(), 3);
    }

    #[test]
    fn test_weighted_loader_size_hint() {
        let dataset = create_test_dataset(10);
        let weights = vec![1.0; 10];

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(3)
            .seed(42);

        let mut iter = loader.into_iter();
        assert_eq!(iter.size_hint(), (4, Some(4)));

        let _ = iter.next();
        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_weighted_loader_getters() {
        let dataset = create_test_dataset(10);
        let weights = vec![1.5; 10];

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(5)
            .num_samples(20);

        assert_eq!(loader.get_batch_size(), 5);
        assert_eq!(loader.get_num_samples(), 20);
        assert_eq!(loader.len(), 10);
        assert!(!loader.is_empty());
        assert!(loader.weights().iter().all(|&w| w == 1.5));
    }

    #[test]
    fn test_weighted_loader_batch_size_min_one() {
        let dataset = create_test_dataset(10);
        let weights = vec![1.0; 10];

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(0);

        assert_eq!(loader.get_batch_size(), 1);
    }

    #[test]
    fn test_weighted_loader_debug() {
        let dataset = create_test_dataset(10);
        let weights = vec![1.0; 10];

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(5)
            .seed(42);

        let debug_str = format!("{:?}", loader);
        assert!(debug_str.contains("WeightedDataLoader"));

        let iter = loader.into_iter();
        let iter_debug = format!("{:?}", iter);
        assert!(iter_debug.contains("WeightedDataLoaderIterator"));
    }

    #[test]
    fn test_weighted_loader_all_zero_weights() {
        // All zero weights should fall back to uniform sampling
        let dataset = create_test_dataset(10);
        let weights = vec![0.0; 10];

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(5)
            .num_samples(20)
            .seed(42);

        // Should still be able to iterate (falls back to uniform)
        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 4); // 20 samples / 5 batch_size = 4 batches
    }

    #[test]
    fn test_weighted_loader_single_nonzero_weight() {
        // Only one item has weight, should sample only that item
        let dataset = create_test_dataset(10);
        let mut weights = vec![0.0; 10];
        weights[5] = 1.0; // Only item 5 has weight

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(1)
            .num_samples(10)
            .seed(42);

        let mut all_are_item_5 = true;
        for batch in loader {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap_or_else(|| panic!("Should be Int32Array"));
            for i in 0..ids.len() {
                if ids.value(i) != 5 {
                    all_are_item_5 = false;
                }
            }
        }
        assert!(all_are_item_5, "All samples should be item 5");
    }

    #[test]
    fn test_weighted_loader_large_dataset() {
        // Test with larger dataset to verify performance
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));

        let ids: Vec<i32> = (0..10000).collect();
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

        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        let weights: Vec<f32> = (0..10000).map(|i| (i % 10 + 1) as f32).collect();

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(100)
            .num_samples(5000)
            .seed(42);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 5000);
    }

    #[test]
    fn test_weighted_loader_very_small_weights() {
        // Test with very small but nonzero weights
        let dataset = create_test_dataset(10);
        let weights: Vec<f32> = (0..10).map(|i| (i + 1) as f32 * 1e-10).collect();

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(5)
            .num_samples(20)
            .seed(42);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 4);
    }

    #[test]
    fn test_weighted_loader_mixed_zero_nonzero() {
        // Half zero, half nonzero weights
        let dataset = create_test_dataset(10);
        let weights: Vec<f32> = (0..10).map(|i| if i < 5 { 0.0 } else { 1.0 }).collect();

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(1)
            .num_samples(100)
            .seed(42);

        let mut counts: HashMap<i32, usize> = HashMap::new();
        for batch in loader {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap_or_else(|| panic!("Should be Int32Array"));
            for i in 0..ids.len() {
                *counts.entry(ids.value(i)).or_insert(0) += 1;
            }
        }

        // Items 0-4 should have 0 counts, items 5-9 should have counts
        for i in 0..5 {
            assert_eq!(
                *counts.get(&i).unwrap_or(&0),
                0,
                "Item {} should not be sampled",
                i
            );
        }
        for i in 5..10 {
            assert!(
                *counts.get(&i).unwrap_or(&0) > 0,
                "Item {} should be sampled",
                i
            );
        }
    }

    #[test]
    fn test_weighted_loader_undersample() {
        // num_samples less than dataset size
        let dataset = create_test_dataset(100);
        let weights = vec![1.0; 100];

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(5)
            .num_samples(20)
            .seed(42);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 20);
    }

    #[test]
    fn test_weighted_loader_exact_batch_multiple() {
        // num_samples exactly divisible by batch_size
        let dataset = create_test_dataset(100);
        let weights = vec![1.0; 100];

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(10)
            .num_samples(50);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 5);
        for batch in &batches {
            assert_eq!(batch.num_rows(), 10);
        }
    }

    #[test]
    fn test_weighted_loader_negative_weight_error() {
        let dataset = create_test_dataset(10);
        let weights = vec![1.0, 2.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let result = WeightedDataLoader::new(dataset, weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_loader_single_item() {
        let dataset = create_test_dataset(1);
        let weights = vec![1.0];

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(1)
            .num_samples(10);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        assert_eq!(batches.len(), 10);

        // All batches should have the same single row
        for batch in batches {
            assert_eq!(batch.num_rows(), 1);
        }
    }

    #[test]
    fn test_weighted_loader_oversample() {
        // num_samples much larger than dataset size
        let dataset = create_test_dataset(5);
        let weights = vec![1.0; 5];

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(10)
            .num_samples(100);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 100);
    }

    #[test]
    fn test_weighted_loader_is_empty() {
        // is_empty() returns dataset.is_empty(), not based on num_samples
        let dataset = create_test_dataset(10);
        let weights = vec![1.0; 10];

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"));

        // Dataset has 10 items, so not empty
        assert!(!loader.is_empty());
        assert_eq!(loader.len(), 10);
    }

    #[test]
    fn test_weighted_loader_len() {
        // len() returns dataset.len(), not num_samples
        let dataset = create_test_dataset(100);
        let weights = vec![1.0; 100];

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .num_samples(42);

        // len() returns dataset length
        assert_eq!(loader.len(), 100);
        // get_num_samples() returns configured num_samples
        assert_eq!(loader.get_num_samples(), 42);
    }

    #[test]
    fn test_weighted_loader_weight_length_mismatch() {
        let dataset = create_test_dataset(10);
        let weights = vec![1.0; 5]; // Wrong length

        let result = WeightedDataLoader::new(dataset, weights);
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_loader_very_large_weight() {
        let dataset = create_test_dataset(3);
        let weights = vec![1e10, 1.0, 1.0]; // First item has huge weight

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(1)
            .num_samples(100)
            .seed(42);

        let mut counts: HashMap<i32, usize> = HashMap::new();
        for batch in loader {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap_or_else(|| panic!("Should be Int32Array"));
            for i in 0..ids.len() {
                *counts.entry(ids.value(i)).or_insert(0) += 1;
            }
        }

        // First item should be sampled almost exclusively
        let first_count = *counts.get(&0).unwrap_or(&0);
        assert!(
            first_count > 95,
            "First item should dominate: {}",
            first_count
        );
    }

    #[test]
    fn test_weighted_loader_extreme_weight_ratio() {
        let dataset = create_test_dataset(2);
        // 1000:1 weight ratio
        let weights = vec![1000.0, 1.0];

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap_or_else(|| panic!("Should create loader"))
            .batch_size(1)
            .num_samples(1000)
            .seed(42);

        let mut counts: HashMap<i32, usize> = HashMap::new();
        for batch in loader {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap_or_else(|| panic!("Should be Int32Array"));
            for i in 0..ids.len() {
                *counts.entry(ids.value(i)).or_insert(0) += 1;
            }
        }

        let first = *counts.get(&0).unwrap_or(&0);
        let second = *counts.get(&1).unwrap_or(&0);

        // First should be ~1000x more frequent than second
        assert!(
            first > 900,
            "First should dominate: {} vs {}",
            first,
            second
        );
    }

    #[test]
    fn test_weighted_loader_reweight_zero() {
        let dataset = create_test_dataset(5);
        // Zero reweight factor creates all-zero weights
        let loader = WeightedDataLoader::with_reweight(dataset, 0.0);
        assert!(loader.is_ok());
        let loader = loader.ok().unwrap();
        // All weights should be 0.0
        assert!(loader.weights().iter().all(|&w| w == 0.0));
    }

    #[test]
    fn test_weighted_loader_size_hint_drop_last_edge() {
        let dataset = create_test_dataset(10);
        let weights = vec![1.0; 10];

        // 10 samples, batch_size 3, drop_last=true -> 3 full batches
        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap()
            .batch_size(3)
            .num_samples(10)
            .drop_last(true);

        assert_eq!(loader.num_batches(), 3);
    }

    #[test]
    fn test_weighted_loader_size_hint_no_drop_last() {
        let dataset = create_test_dataset(10);
        let weights = vec![1.0; 10];

        // 10 samples, batch_size 3, drop_last=false -> 4 batches
        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap()
            .batch_size(3)
            .num_samples(10)
            .drop_last(false);

        assert_eq!(loader.num_batches(), 4);
    }

    #[test]
    fn test_weighted_loader_iteration_with_drop_last() {
        let dataset = create_test_dataset(10);
        let weights = vec![1.0; 10];

        let loader = WeightedDataLoader::new(dataset, weights)
            .ok()
            .unwrap()
            .batch_size(4)
            .num_samples(10)
            .drop_last(true)
            .seed(42);

        let batches: Vec<RecordBatch> = loader.into_iter().collect();
        // 10 samples / 4 batch_size with drop_last = 2 full batches
        assert_eq!(batches.len(), 2);
        for batch in batches {
            assert_eq!(batch.num_rows(), 4);
        }
    }
}
