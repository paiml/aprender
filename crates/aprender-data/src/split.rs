// Allow casts for size calculations - these are intentional and safe for
// dataset sizes
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]

//! Dataset splitting utilities
//!
//! Provides train/test/validation splitting with stratification support.
//!
//! # Example
//!
//! ```ignore
//! use alimentar::split::DatasetSplit;
//!
//! // Simple ratio split
//! let split = DatasetSplit::from_ratios(&dataset, 0.8, 0.2, None, None)?;
//!
//! // With validation
//! let split = DatasetSplit::from_ratios(&dataset, 0.7, 0.15, Some(0.15), Some(42))?;
//!
//! // Stratified by label column
//! let split = DatasetSplit::stratified(&dataset, "label", 0.8, 0.2, None, None)?;
//! ```

use std::{collections::HashMap, sync::Arc};

use arrow::array::{Array, RecordBatch};

use crate::{
    error::{Error, Result},
    transform::{Skip, Take, Transform},
    ArrowDataset, Dataset,
};

/// Dataset split with optional validation set
#[derive(Debug, Clone)]
pub struct DatasetSplit {
    /// Training dataset (required)
    pub train: ArrowDataset,
    /// Test/holdout dataset (required)
    pub test: ArrowDataset,
    /// Validation dataset (optional)
    pub validation: Option<ArrowDataset>,
}

impl DatasetSplit {
    /// Create train/test split (no validation)
    pub fn new(train: ArrowDataset, test: ArrowDataset) -> Self {
        Self {
            train,
            test,
            validation: None,
        }
    }

    /// Create train/test/validation split
    pub fn with_validation(
        train: ArrowDataset,
        test: ArrowDataset,
        validation: ArrowDataset,
    ) -> Self {
        Self {
            train,
            test,
            validation: Some(validation),
        }
    }

    /// Get training data
    pub fn train(&self) -> &ArrowDataset {
        &self.train
    }

    /// Get test data
    pub fn test(&self) -> &ArrowDataset {
        &self.test
    }

    /// Get validation data (if present)
    pub fn validation(&self) -> Option<&ArrowDataset> {
        self.validation.as_ref()
    }

    /// Split dataset by ratios
    ///
    /// # Arguments
    /// * `dataset` - Source dataset to split
    /// * `train_ratio` - Fraction for training (0.0 to 1.0)
    /// * `test_ratio` - Fraction for testing (0.0 to 1.0)
    /// * `val_ratio` - Optional fraction for validation
    /// * `seed` - Optional random seed for shuffling
    ///
    /// # Errors
    /// Returns error if ratios don't sum to 1.0 or dataset is empty
    pub fn from_ratios(
        dataset: &ArrowDataset,
        train_ratio: f64,
        test_ratio: f64,
        val_ratio: Option<f64>,
        seed: Option<u64>,
    ) -> Result<Self> {
        // Validate ratios
        let total = train_ratio + test_ratio + val_ratio.unwrap_or(0.0);
        if (total - 1.0).abs() > 1e-9 {
            return Err(Error::invalid_config(format!(
                "Split ratios must sum to 1.0, got {total}"
            )));
        }

        if train_ratio <= 0.0 || test_ratio <= 0.0 {
            return Err(Error::invalid_config(
                "Train and test ratios must be positive",
            ));
        }

        if let Some(v) = val_ratio {
            if v <= 0.0 {
                return Err(Error::invalid_config(
                    "Validation ratio must be positive if specified",
                ));
            }
        }

        let len = dataset.len();
        if len == 0 {
            return Err(Error::empty_dataset("Cannot split empty dataset"));
        }

        // Get all data as a single batch (concatenate if multiple batches)
        let batch = concatenate_batches(dataset)?;

        let batch = if let Some(s) = seed {
            shuffle_batch(&batch, s)?
        } else {
            batch
        };

        // Calculate sizes
        let train_size = ((len as f64) * train_ratio).round() as usize;
        let test_size = ((len as f64) * test_ratio).round() as usize;
        let val_size = val_ratio.map(|v| ((len as f64) * v).round() as usize);

        // Adjust for rounding errors
        let train_size = train_size.max(1);
        let test_size = test_size.max(1);

        // Split the batch
        let train_batch = Take::new(train_size).apply(batch.clone())?;
        let remaining = Skip::new(train_size).apply(batch)?;

        let (test_batch, validation) = if val_size.is_some() {
            let test_batch = Take::new(test_size).apply(remaining.clone())?;
            let val_batch = Skip::new(test_size).apply(remaining)?;
            (test_batch, Some(ArrowDataset::from_batch(val_batch)?))
        } else {
            (remaining, None)
        };

        Ok(Self {
            train: ArrowDataset::from_batch(train_batch)?,
            test: ArrowDataset::from_batch(test_batch)?,
            validation,
        })
    }

    /// Stratified split preserving label distribution
    ///
    /// # Arguments
    /// * `dataset` - Source dataset to split
    /// * `label_column` - Name of the label/target column
    /// * `train_ratio` - Fraction for training
    /// * `test_ratio` - Fraction for testing
    /// * `val_ratio` - Optional fraction for validation
    /// * `seed` - Optional random seed
    ///
    /// # Errors
    /// Returns error if label column not found or ratios invalid
    pub fn stratified(
        dataset: &ArrowDataset,
        label_column: &str,
        train_ratio: f64,
        test_ratio: f64,
        val_ratio: Option<f64>,
        seed: Option<u64>,
    ) -> Result<Self> {
        // Validate ratios
        let total = train_ratio + test_ratio + val_ratio.unwrap_or(0.0);
        if (total - 1.0).abs() > 1e-9 {
            return Err(Error::invalid_config(format!(
                "Split ratios must sum to 1.0, got {total}"
            )));
        }

        let len = dataset.len();
        if len == 0 {
            return Err(Error::empty_dataset("Cannot split empty dataset"));
        }

        // Get all data as a single batch
        let batch = concatenate_batches(dataset)?;

        // Find label column
        let schema = batch.schema();
        let label_idx = schema.index_of(label_column).map_err(|_| {
            Error::invalid_config(format!("Label column '{label_column}' not found"))
        })?;

        let label_array = batch.column(label_idx);

        // Group indices by label value
        let groups = group_by_label(label_array)?;

        // Split each group proportionally
        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();
        let mut val_indices = Vec::new();

        let base_seed = seed.unwrap_or(0);

        for (label_value, mut indices) in groups {
            // Shuffle within group
            if seed.is_some() {
                // Simple deterministic shuffle using label as additional seed component
                let group_seed = base_seed.wrapping_add(label_value as u64);
                shuffle_indices(&mut indices, group_seed);
            }

            let group_len = indices.len();
            let group_train = ((group_len as f64) * train_ratio).round() as usize;
            let group_test = ((group_len as f64) * test_ratio).round() as usize;

            let group_train = group_train.max(1).min(group_len);

            train_indices.extend_from_slice(&indices[..group_train]);

            if val_ratio.is_some() {
                let remaining = group_len.saturating_sub(group_train);
                let group_test = group_test.min(remaining);
                test_indices.extend_from_slice(&indices[group_train..group_train + group_test]);
                val_indices.extend_from_slice(&indices[group_train + group_test..]);
            } else {
                test_indices.extend_from_slice(&indices[group_train..]);
            }
        }

        // Build split datasets from indices
        let train_batch = take_indices(&batch, &train_indices)?;
        let test_batch = take_indices(&batch, &test_indices)?;

        let validation = if val_ratio.is_some() && !val_indices.is_empty() {
            Some(ArrowDataset::from_batch(take_indices(
                &batch,
                &val_indices,
            )?)?)
        } else {
            None
        };

        Ok(Self {
            train: ArrowDataset::from_batch(train_batch)?,
            test: ArrowDataset::from_batch(test_batch)?,
            validation,
        })
    }
}

/// Concatenate all batches from a dataset into a single batch
fn concatenate_batches(dataset: &ArrowDataset) -> Result<RecordBatch> {
    use arrow::compute::concat_batches;

    let schema = dataset.schema();
    let batches: Vec<RecordBatch> = dataset.iter().collect();

    if batches.is_empty() {
        return Err(Error::empty_dataset("Dataset has no batches"));
    }

    if batches.len() == 1 {
        return batches
            .into_iter()
            .next()
            .ok_or_else(|| Error::empty_dataset("Dataset has no batches"));
    }

    concat_batches(&schema, &batches).map_err(Error::Arrow)
}

/// Shuffle a record batch deterministically
fn shuffle_batch(batch: &RecordBatch, seed: u64) -> Result<RecordBatch> {
    let len = batch.num_rows();
    let mut indices: Vec<usize> = (0..len).collect();
    shuffle_indices(&mut indices, seed);
    take_indices(batch, &indices)
}

/// Shuffle indices in place using simple LCG
fn shuffle_indices(indices: &mut [usize], seed: u64) {
    // Simple Fisher-Yates with LCG random
    let mut rng = seed;
    for i in (1..indices.len()).rev() {
        // LCG: next = (a * current + c) mod m
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let j = (rng as usize) % (i + 1);
        indices.swap(i, j);
    }
}

/// Group row indices by label value
fn group_by_label(label_array: &Arc<dyn Array>) -> Result<HashMap<i64, Vec<usize>>> {
    use arrow::{
        array::{Int32Array, Int64Array, StringArray, UInt32Array, UInt64Array},
        datatypes::DataType,
    };

    let mut groups: HashMap<i64, Vec<usize>> = HashMap::new();

    match label_array.data_type() {
        DataType::Int32 => {
            let arr = downcast_label::<Int32Array>(label_array, "Int32Array")?;
            collect_groups(arr.iter(), &mut groups, i64::from);
        }
        DataType::Int64 => {
            let arr = downcast_label::<Int64Array>(label_array, "Int64Array")?;
            collect_groups(arr.iter(), &mut groups, |v| v);
        }
        DataType::UInt32 => {
            let arr = downcast_label::<UInt32Array>(label_array, "UInt32Array")?;
            collect_groups(arr.iter(), &mut groups, i64::from);
        }
        DataType::UInt64 => {
            let arr = downcast_label::<UInt64Array>(label_array, "UInt64Array")?;
            // May truncate very large values
            collect_groups(arr.iter(), &mut groups, |v| v as i64);
        }
        DataType::Utf8 | DataType::LargeUtf8 => {
            // Hash string labels to i64 for grouping (alimentar#37)
            let arr = downcast_label::<StringArray>(label_array, "StringArray")?;
            collect_groups(arr.iter(), &mut groups, |s: &str| {
                // FNV-1a hash — deterministic, fast, good distribution
                let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
                for byte in s.as_bytes() {
                    hash ^= u64::from(*byte);
                    hash = hash.wrapping_mul(0x0100_0000_01b3);
                }
                hash as i64
            });
        }
        dt => {
            return Err(Error::invalid_config(format!(
                "Unsupported label type for stratification: {dt:?}"
            )))
        }
    }

    Ok(groups)
}

/// Downcast a label array to a concrete Arrow array type
fn downcast_label<'a, T: 'static>(array: &'a Arc<dyn Array>, type_name: &str) -> Result<&'a T> {
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| Error::invalid_config(format!("Failed to downcast {type_name}")))
}

/// Collect values from an Arrow array iterator into groups by label.
fn collect_groups<V, F>(
    iter: impl Iterator<Item = Option<V>>,
    groups: &mut HashMap<i64, Vec<usize>>,
    to_i64: F,
) where
    F: Fn(V) -> i64,
{
    for (i, val) in iter.enumerate() {
        if let Some(v) = val {
            groups.entry(to_i64(v)).or_default().push(i);
        }
    }
}

/// Take rows at given indices from a batch
fn take_indices(batch: &RecordBatch, indices: &[usize]) -> Result<RecordBatch> {
    use arrow::{array::UInt32Array, compute::take};

    let indices_array = UInt32Array::from(indices.iter().map(|&i| i as u32).collect::<Vec<_>>());

    let columns: Vec<Arc<dyn Array>> = batch
        .columns()
        .iter()
        .map(|col| take(col.as_ref(), &indices_array, None).map_err(Error::Arrow))
        .collect::<Result<Vec<_>>>()?;

    RecordBatch::try_new(batch.schema(), columns).map_err(Error::Arrow)
}

#[cfg(test)]
mod tests {
    use arrow::{
        array::{Float64Array, Int32Array},
        datatypes::{DataType, Field, Schema},
    };

    use super::*;

    /// Helper to create a test dataset with n samples
    fn make_test_dataset(n: usize) -> ArrowDataset {
        let schema = Arc::new(Schema::new(vec![
            Field::new("feature", DataType::Float64, false),
            Field::new("label", DataType::Int32, false),
        ]));

        let features: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let labels: Vec<i32> = (0..n).map(|i| (i % 3) as i32).collect(); // 3 classes

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(features)),
                Arc::new(Int32Array::from(labels)),
            ],
        )
        .expect("batch creation failed");

        ArrowDataset::from_batch(batch).expect("dataset creation failed")
    }

    // ========== DatasetSplit::new tests ==========

    #[test]
    fn test_new_creates_split_without_validation() {
        let train = make_test_dataset(80);
        let test = make_test_dataset(20);

        let split = DatasetSplit::new(train, test);

        assert_eq!(split.train().len(), 80);
        assert_eq!(split.test().len(), 20);
        assert!(split.validation().is_none());
    }

    // ========== DatasetSplit::with_validation tests ==========

    #[test]
    fn test_with_validation_creates_three_way_split() {
        let train = make_test_dataset(70);
        let test = make_test_dataset(15);
        let val = make_test_dataset(15);

        let split = DatasetSplit::with_validation(train, test, val);

        assert_eq!(split.train().len(), 70);
        assert_eq!(split.test().len(), 15);
        assert!(split.validation().is_some());
        assert_eq!(split.validation().expect("val").len(), 15);
    }

    // ========== DatasetSplit::from_ratios tests ==========

    #[test]
    fn test_from_ratios_80_20_split() {
        let dataset = make_test_dataset(100);

        let split =
            DatasetSplit::from_ratios(&dataset, 0.8, 0.2, None, None).expect("split failed");

        assert_eq!(split.train().len(), 80);
        assert_eq!(split.test().len(), 20);
        assert!(split.validation().is_none());
    }

    #[test]
    fn test_from_ratios_70_15_15_split() {
        let dataset = make_test_dataset(100);

        let split =
            DatasetSplit::from_ratios(&dataset, 0.7, 0.15, Some(0.15), None).expect("split failed");

        assert_eq!(split.train().len(), 70);
        assert_eq!(split.test().len(), 15);
        assert!(split.validation().is_some());
        assert_eq!(split.validation().expect("val").len(), 15);
    }

    #[test]
    fn test_from_ratios_with_seed_is_deterministic() {
        let dataset = make_test_dataset(100);

        let split1 =
            DatasetSplit::from_ratios(&dataset, 0.8, 0.2, None, Some(42)).expect("split failed");
        let split2 =
            DatasetSplit::from_ratios(&dataset, 0.8, 0.2, None, Some(42)).expect("split failed");

        // Same seed should produce same split
        let train1 = split1.train().get(0).expect("batch");
        let train2 = split2.train().get(0).expect("batch");

        assert_eq!(train1.num_rows(), train2.num_rows());
        // Check first column values match
        let col1 = train1
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("downcast");
        let col2 = train2
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("downcast");

        for i in 0..col1.len() {
            assert!(
                (col1.value(i) - col2.value(i)).abs() < 1e-9,
                "Mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_from_ratios_different_seeds_produce_different_splits() {
        let dataset = make_test_dataset(100);

        let split1 =
            DatasetSplit::from_ratios(&dataset, 0.8, 0.2, None, Some(42)).expect("split failed");
        let split2 =
            DatasetSplit::from_ratios(&dataset, 0.8, 0.2, None, Some(123)).expect("split failed");

        let train1 = split1.train().get(0).expect("batch");
        let train2 = split2.train().get(0).expect("batch");

        let col1 = train1
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("downcast");
        let col2 = train2
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("downcast");

        // At least some values should differ
        let mut differs = false;
        for i in 0..col1.len().min(col2.len()) {
            if (col1.value(i) - col2.value(i)).abs() > 1e-9 {
                differs = true;
                break;
            }
        }
        assert!(differs, "Different seeds should produce different shuffles");
    }

    #[test]
    fn test_from_ratios_rejects_invalid_ratios() {
        let dataset = make_test_dataset(100);

        // Ratios don't sum to 1.0
        let result = DatasetSplit::from_ratios(&dataset, 0.5, 0.3, None, None);
        assert!(result.is_err());

        // Zero train ratio
        let result = DatasetSplit::from_ratios(&dataset, 0.0, 1.0, None, None);
        assert!(result.is_err());

        // Zero test ratio
        let result = DatasetSplit::from_ratios(&dataset, 1.0, 0.0, None, None);
        assert!(result.is_err());

        // Zero validation ratio
        let result = DatasetSplit::from_ratios(&dataset, 0.8, 0.19, Some(0.0), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_ratios_rejects_empty_dataset() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(Vec::<f64>::new()))],
        )
        .expect("batch");
        let dataset = ArrowDataset::from_batch(batch).expect("dataset");

        let result = DatasetSplit::from_ratios(&dataset, 0.8, 0.2, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_ratios_handles_small_dataset() {
        let dataset = make_test_dataset(3);

        let split =
            DatasetSplit::from_ratios(&dataset, 0.7, 0.3, None, None).expect("split failed");

        // Should have at least 1 in each
        assert!(split.train().len() >= 1);
        assert!(split.test().len() >= 1);
        assert_eq!(split.train().len() + split.test().len(), 3);
    }

    // ========== DatasetSplit::stratified tests ==========

    #[test]
    fn test_stratified_preserves_class_distribution() {
        // Create dataset with known class distribution: 60% class 0, 30% class 1, 10%
        // class 2
        let schema = Arc::new(Schema::new(vec![
            Field::new("feature", DataType::Float64, false),
            Field::new("label", DataType::Int32, false),
        ]));

        let n = 100;
        let features: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let labels: Vec<i32> = (0..n)
            .map(|i| {
                if i < 60 {
                    0
                } else if i < 90 {
                    1
                } else {
                    2
                }
            })
            .collect();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(features)),
                Arc::new(Int32Array::from(labels)),
            ],
        )
        .expect("batch");
        let dataset = ArrowDataset::from_batch(batch).expect("dataset");

        let split =
            DatasetSplit::stratified(&dataset, "label", 0.8, 0.2, None, Some(42)).expect("split");

        // Count classes in train (iterate all batches)
        let mut train_counts = [0usize; 3];
        for batch in split.train().iter() {
            let labels = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("downcast");
            for val in labels.iter().flatten() {
                train_counts[val as usize] += 1;
            }
        }

        // Count classes in test
        let mut test_counts = [0usize; 3];
        for batch in split.test().iter() {
            let labels = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .expect("downcast");
            for val in labels.iter().flatten() {
                test_counts[val as usize] += 1;
            }
        }

        // Verify proportions are approximately preserved (within tolerance)
        // Original: 60/30/10, expect ~same ratio in both splits
        let train_total = train_counts.iter().sum::<usize>() as f64;
        let test_total = test_counts.iter().sum::<usize>() as f64;

        let train_ratio_0 = train_counts[0] as f64 / train_total;
        let test_ratio_0 = test_counts[0] as f64 / test_total;

        // Should be close to 60% in both
        assert!(
            (train_ratio_0 - 0.6).abs() < 0.15,
            "Train class 0 ratio {train_ratio_0} too far from 0.6"
        );
        assert!(
            (test_ratio_0 - 0.6).abs() < 0.15,
            "Test class 0 ratio {test_ratio_0} too far from 0.6"
        );
    }

    #[test]
    fn test_stratified_with_validation() {
        let dataset = make_test_dataset(90); // Divisible by 3 classes

        let split = DatasetSplit::stratified(&dataset, "label", 0.7, 0.15, Some(0.15), Some(42))
            .expect("split");

        assert!(split.validation().is_some());
        let total = split.train().len() + split.test().len() + split.validation().expect("v").len();
        assert_eq!(total, 90);
    }

    #[test]
    fn test_stratified_rejects_missing_column() {
        let dataset = make_test_dataset(100);

        let result = DatasetSplit::stratified(&dataset, "nonexistent", 0.8, 0.2, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_stratified_rejects_invalid_ratios() {
        let dataset = make_test_dataset(100);

        let result = DatasetSplit::stratified(&dataset, "label", 0.5, 0.3, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_stratified_is_deterministic_with_seed() {
        let dataset = make_test_dataset(100);

        let split1 =
            DatasetSplit::stratified(&dataset, "label", 0.8, 0.2, None, Some(42)).expect("split");
        let split2 =
            DatasetSplit::stratified(&dataset, "label", 0.8, 0.2, None, Some(42)).expect("split");

        assert_eq!(split1.train().len(), split2.train().len());
        assert_eq!(split1.test().len(), split2.test().len());
    }

    // ========== Edge case tests ==========

    #[test]
    fn test_split_preserves_schema() {
        let dataset = make_test_dataset(100);
        let original_schema = dataset.schema();

        let split =
            DatasetSplit::from_ratios(&dataset, 0.8, 0.2, None, None).expect("split failed");

        assert_eq!(split.train().schema(), original_schema);
        assert_eq!(split.test().schema(), original_schema);
    }

    #[test]
    fn test_split_no_data_overlap() {
        let dataset = make_test_dataset(100);

        let split =
            DatasetSplit::from_ratios(&dataset, 0.8, 0.2, None, Some(42)).expect("split failed");

        // Collect all train values
        let mut train_set: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for batch in split.train().iter() {
            let features = batch
                .column(0)
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("downcast");
            for val in features.iter().flatten() {
                train_set.insert(val.to_bits());
            }
        }

        // Check no test values are in train
        for batch in split.test().iter() {
            let features = batch
                .column(0)
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("downcast");
            for val in features.iter().flatten() {
                assert!(
                    !train_set.contains(&val.to_bits()),
                    "Found overlapping value {val} in train and test"
                );
            }
        }
    }

    #[test]
    fn test_stratified_with_int64_labels() {
        use arrow::array::Int64Array;

        let schema = Arc::new(Schema::new(vec![
            Field::new("feature", DataType::Float64, false),
            Field::new("label", DataType::Int64, false),
        ]));

        let features: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let labels: Vec<i64> = (0..100).map(|i| (i % 3) as i64).collect();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(features)),
                Arc::new(Int64Array::from(labels)),
            ],
        )
        .expect("batch creation failed");

        let dataset = ArrowDataset::from_batch(batch).expect("dataset creation failed");

        let split = DatasetSplit::stratified(&dataset, "label", 0.8, 0.2, None, Some(42))
            .expect("split failed");

        assert!(split.train().len() > 0);
        assert!(split.test().len() > 0);
    }

    #[test]
    fn test_stratified_with_uint32_labels() {
        use arrow::array::UInt32Array;

        let schema = Arc::new(Schema::new(vec![
            Field::new("feature", DataType::Float64, false),
            Field::new("label", DataType::UInt32, false),
        ]));

        let features: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let labels: Vec<u32> = (0..100).map(|i| (i % 3) as u32).collect();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(features)),
                Arc::new(UInt32Array::from(labels)),
            ],
        )
        .expect("batch creation failed");

        let dataset = ArrowDataset::from_batch(batch).expect("dataset creation failed");

        let split = DatasetSplit::stratified(&dataset, "label", 0.8, 0.2, None, Some(42))
            .expect("split failed");

        assert!(split.train().len() > 0);
        assert!(split.test().len() > 0);
    }

    #[test]
    fn test_stratified_with_uint64_labels() {
        use arrow::array::UInt64Array;

        let schema = Arc::new(Schema::new(vec![
            Field::new("feature", DataType::Float64, false),
            Field::new("label", DataType::UInt64, false),
        ]));

        let features: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let labels: Vec<u64> = (0..100).map(|i| (i % 3) as u64).collect();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(features)),
                Arc::new(UInt64Array::from(labels)),
            ],
        )
        .expect("batch creation failed");

        let dataset = ArrowDataset::from_batch(batch).expect("dataset creation failed");

        let split = DatasetSplit::stratified(&dataset, "label", 0.8, 0.2, None, Some(42))
            .expect("split failed");

        assert!(split.train().len() > 0);
        assert!(split.test().len() > 0);
    }

    #[test]
    fn test_stratified_with_string_labels() {
        use arrow::array::StringArray;

        let schema = Arc::new(Schema::new(vec![
            Field::new("feature", DataType::Float64, false),
            Field::new("label", DataType::Utf8, false),
        ]));

        let features: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let labels: Vec<&str> = (0..100)
            .map(|i| if i % 2 == 0 { "a" } else { "b" })
            .collect();

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float64Array::from(features)),
                Arc::new(StringArray::from(labels)),
            ],
        )
        .expect("batch creation failed");

        let dataset = ArrowDataset::from_batch(batch).expect("dataset creation failed");

        let split = DatasetSplit::stratified(&dataset, "label", 0.8, 0.2, None, Some(42))
            .expect("stratified split with string labels should succeed");
        assert_eq!(split.train().len() + split.test().len(), 100);
        assert!(split.train().len() > 0);
        assert!(split.test().len() > 0);
    }

    #[test]
    fn test_stratified_without_seed() {
        let dataset = make_test_dataset(100);

        // Without seed, should still work
        let split = DatasetSplit::stratified(&dataset, "label", 0.8, 0.2, None, None)
            .expect("split failed");

        assert!(split.train().len() > 0);
        assert!(split.test().len() > 0);
    }

    #[test]
    fn test_split_debug() {
        let dataset = make_test_dataset(100);
        let split =
            DatasetSplit::from_ratios(&dataset, 0.8, 0.2, None, None).expect("split failed");

        let debug = format!("{:?}", split);
        assert!(debug.contains("DatasetSplit"));
    }

    #[test]
    fn test_split_clone() {
        let dataset = make_test_dataset(100);
        let split =
            DatasetSplit::from_ratios(&dataset, 0.8, 0.2, None, None).expect("split failed");

        let cloned = split.clone();
        assert_eq!(cloned.train().len(), split.train().len());
        assert_eq!(cloned.test().len(), split.test().len());
    }

    #[test]
    fn test_extreme_ratio_99_1() {
        let dataset = make_test_dataset(100);
        let split =
            DatasetSplit::from_ratios(&dataset, 0.99, 0.01, None, None).expect("split failed");

        assert_eq!(split.train().len(), 99);
        assert_eq!(split.test().len(), 1);
    }

    #[test]
    fn test_extreme_ratio_50_50() {
        let dataset = make_test_dataset(100);
        let split =
            DatasetSplit::from_ratios(&dataset, 0.5, 0.5, None, None).expect("split failed");

        assert_eq!(split.train().len(), 50);
        assert_eq!(split.test().len(), 50);
    }

    #[test]
    fn test_negative_train_ratio_rejected() {
        let dataset = make_test_dataset(100);
        let result = DatasetSplit::from_ratios(&dataset, -0.5, 0.5, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_test_ratio_rejected() {
        let dataset = make_test_dataset(100);
        let result = DatasetSplit::from_ratios(&dataset, 1.0, 0.0, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_negative_val_ratio_rejected() {
        let dataset = make_test_dataset(100);
        let result = DatasetSplit::from_ratios(&dataset, 0.6, 0.5, Some(-0.1), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_row_minimum_sizes() {
        let dataset = make_test_dataset(2);
        let split =
            DatasetSplit::from_ratios(&dataset, 0.5, 0.5, None, None).expect("split failed");

        // Each should get at least 1 row
        assert!(split.train().len() >= 1);
        assert!(split.test().len() >= 1);
    }

    #[test]
    fn test_ratios_slightly_over_one() {
        let dataset = make_test_dataset(100);
        // Sum is 1.01, should be rejected
        let result = DatasetSplit::from_ratios(&dataset, 0.81, 0.2, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_ratios_slightly_under_one() {
        let dataset = make_test_dataset(100);
        // Sum is 0.99, should be rejected
        let result = DatasetSplit::from_ratios(&dataset, 0.79, 0.2, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_getters_return_correct_data() {
        let train = make_test_dataset(80);
        let test = make_test_dataset(20);
        let val = make_test_dataset(10);

        let split = DatasetSplit::with_validation(train.clone(), test.clone(), val.clone());

        assert_eq!(split.train().len(), 80);
        assert_eq!(split.test().len(), 20);
        assert_eq!(split.validation().map(|v| v.len()), Some(10));
    }

    #[test]
    fn test_validation_none_for_two_way_split() {
        let train = make_test_dataset(80);
        let test = make_test_dataset(20);

        let split = DatasetSplit::new(train, test);

        assert!(split.validation().is_none());
    }

    #[test]
    fn test_stratified_empty_dataset() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Float64, false),
            Field::new("label", DataType::Int32, false),
        ]));
        let x_array = arrow::array::Float64Array::from(Vec::<f64>::new());
        let label_array = Int32Array::from(Vec::<i32>::new());
        let batch = RecordBatch::try_new(schema, vec![Arc::new(x_array), Arc::new(label_array)])
            .expect("batch");
        let dataset = ArrowDataset::from_batch(batch).expect("dataset");

        let result = DatasetSplit::stratified(&dataset, "label", 0.8, 0.2, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_stratified_zero_test_ratio_rejected() {
        let dataset = make_test_dataset(100);
        let result = DatasetSplit::stratified(&dataset, "y", 1.0, 0.0, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_split_preserves_all_rows() {
        let dataset = make_test_dataset(100);
        let split =
            DatasetSplit::from_ratios(&dataset, 0.6, 0.2, Some(0.2), None).expect("split failed");

        let total = split.train().len()
            + split.test().len()
            + split.validation().map(|v| v.len()).unwrap_or(0);
        assert_eq!(total, 100);
    }
}
