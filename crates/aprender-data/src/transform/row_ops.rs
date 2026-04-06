//! Row-level operations: sorting, sampling, deduplication, and slicing.

use std::sync::Arc;

use arrow::array::{Array, RecordBatch};
#[cfg(feature = "shuffle")]
use rand::{seq::SliceRandom, SeedableRng};

use super::Transform;
use crate::error::{Error, Result};

/// A transform that shuffles rows in a RecordBatch.
///
/// Requires the `shuffle` feature.
///
/// # Example
///
/// ```ignore
/// use alimentar::Shuffle;
///
/// // Random shuffle
/// let shuffle = Shuffle::new();
///
/// // Deterministic shuffle with seed
/// let shuffle = Shuffle::with_seed(42);
/// ```
#[cfg(feature = "shuffle")]
#[derive(Debug, Clone)]
pub struct Shuffle {
    seed: Option<u64>,
}

#[cfg(feature = "shuffle")]
impl Shuffle {
    /// Creates a new Shuffle transform with random ordering.
    pub fn new() -> Self {
        Self { seed: None }
    }

    /// Creates a new Shuffle transform with a fixed seed for reproducibility.
    pub fn with_seed(seed: u64) -> Self {
        Self { seed: Some(seed) }
    }
}

#[cfg(feature = "shuffle")]
impl Default for Shuffle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "shuffle")]
impl Transform for Shuffle {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let num_rows = batch.num_rows();
        if num_rows <= 1 {
            return Ok(batch);
        }

        // Create shuffled indices
        let mut indices: Vec<usize> = (0..num_rows).collect();
        let mut rng = match self.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };
        indices.shuffle(&mut rng);

        // Reorder each column according to shuffled indices
        let schema = batch.schema();
        let new_columns: Vec<Arc<dyn Array>> = (0..batch.num_columns())
            .map(|col_idx| {
                let col = batch.column(col_idx);
                let indices_array =
                    arrow::array::UInt64Array::from_iter_values(indices.iter().map(|&i| i as u64));
                arrow::compute::take(col.as_ref(), &indices_array, None)
                    .map_err(Error::Arrow)
                    .map(Arc::from)
            })
            .collect::<Result<Vec<_>>>()?;

        RecordBatch::try_new(schema, new_columns).map_err(Error::Arrow)
    }
}

/// A transform that randomly samples rows from a RecordBatch.
///
/// Useful for creating train/test splits or reducing dataset size.
/// Requires the `shuffle` feature.
///
/// # Example
///
/// ```ignore
/// use alimentar::Sample;
///
/// // Sample 100 rows with a fixed seed
/// let sample = Sample::new(100).with_seed(42);
///
/// // Sample 10% of rows
/// let sample = Sample::fraction(0.1);
/// ```
#[cfg(feature = "shuffle")]
#[derive(Debug, Clone)]
pub struct Sample {
    count: Option<usize>,
    fraction: Option<f64>,
    seed: Option<u64>,
}

#[cfg(feature = "shuffle")]
impl Sample {
    /// Creates a Sample transform that selects exactly `count` rows.
    ///
    /// If the batch has fewer rows than `count`, all rows are returned.
    pub fn new(count: usize) -> Self {
        Self {
            count: Some(count),
            fraction: None,
            seed: None,
        }
    }

    /// Creates a Sample transform that selects a fraction of rows.
    ///
    /// The fraction should be between 0.0 and 1.0.
    pub fn fraction(frac: f64) -> Self {
        Self {
            count: None,
            fraction: Some(frac.clamp(0.0, 1.0)),
            seed: None,
        }
    }

    /// Sets a seed for reproducible sampling.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Returns the sample count if set.
    pub fn count(&self) -> Option<usize> {
        self.count
    }

    /// Returns the sample fraction if set.
    pub fn sample_fraction(&self) -> Option<f64> {
        self.fraction
    }
}

#[cfg(feature = "shuffle")]
impl Transform for Sample {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return Ok(batch);
        }

        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let sample_size = match (self.count, self.fraction) {
            (Some(c), _) => c.min(num_rows),
            (None, Some(f)) => ((num_rows as f64) * f).round() as usize,
            (None, None) => return Ok(batch),
        };

        if sample_size >= num_rows {
            return Ok(batch);
        }

        // Create shuffled indices and take first sample_size
        let mut indices: Vec<usize> = (0..num_rows).collect();
        let mut rng = match self.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };
        indices.shuffle(&mut rng);
        indices.truncate(sample_size);
        indices.sort_unstable(); // Keep original order

        // Reorder each column according to sampled indices
        let schema = batch.schema();
        let new_columns: Vec<Arc<dyn Array>> = (0..batch.num_columns())
            .map(|col_idx| {
                let col = batch.column(col_idx);
                let indices_array =
                    arrow::array::UInt64Array::from_iter_values(indices.iter().map(|&i| i as u64));
                arrow::compute::take(col.as_ref(), &indices_array, None)
                    .map_err(Error::Arrow)
                    .map(Arc::from)
            })
            .collect::<Result<Vec<_>>>()?;

        RecordBatch::try_new(schema, new_columns).map_err(Error::Arrow)
    }
}

/// A transform that takes the first N rows from a RecordBatch.
///
/// # Example
///
/// ```ignore
/// use alimentar::Take;
///
/// let take = Take::new(100); // Take first 100 rows
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Take {
    count: usize,
}

impl Take {
    /// Creates a Take transform that keeps the first `count` rows.
    pub fn new(count: usize) -> Self {
        Self { count }
    }

    /// Returns the number of rows to take.
    pub fn count(&self) -> usize {
        self.count
    }
}

impl Transform for Take {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let num_rows = batch.num_rows();
        if self.count >= num_rows {
            return Ok(batch);
        }

        Ok(batch.slice(0, self.count))
    }
}

/// A transform that skips the first N rows from a RecordBatch.
///
/// # Example
///
/// ```ignore
/// use alimentar::Skip;
///
/// let skip = Skip::new(10); // Skip first 10 rows
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Skip {
    count: usize,
}

impl Skip {
    /// Creates a Skip transform that skips the first `count` rows.
    pub fn new(count: usize) -> Self {
        Self { count }
    }

    /// Returns the number of rows to skip.
    pub fn count(&self) -> usize {
        self.count
    }
}

impl Transform for Skip {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let num_rows = batch.num_rows();
        if self.count >= num_rows {
            // Skip all rows - return empty batch with same schema
            return Ok(batch.slice(0, 0));
        }

        let remaining = num_rows - self.count;
        Ok(batch.slice(self.count, remaining))
    }
}

/// Sort order for the Sort transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortOrder {
    /// Ascending order (smallest to largest)
    #[default]
    Ascending,
    /// Descending order (largest to smallest)
    Descending,
}

/// A transform that sorts rows by one or more columns.
///
/// # Example
///
/// ```ignore
/// use alimentar::{Sort, SortOrder};
///
/// // Sort by single column ascending
/// let sort = Sort::by("age");
///
/// // Sort by column descending
/// let sort = Sort::by("score").order(SortOrder::Descending);
///
/// // Sort by multiple columns
/// let sort = Sort::by_columns(vec![("name", SortOrder::Ascending), ("age", SortOrder::Descending)]);
/// ```
#[derive(Debug, Clone)]
pub struct Sort {
    columns: Vec<(String, SortOrder)>,
    nulls_first: bool,
}

impl Sort {
    /// Creates a Sort transform for a single column (ascending by default).
    pub fn by<S: Into<String>>(column: S) -> Self {
        Self {
            columns: vec![(column.into(), SortOrder::Ascending)],
            nulls_first: false,
        }
    }

    /// Creates a Sort transform for multiple columns with specified orders.
    pub fn by_columns<S: Into<String>>(columns: impl IntoIterator<Item = (S, SortOrder)>) -> Self {
        Self {
            columns: columns
                .into_iter()
                .map(|(name, order)| (name.into(), order))
                .collect(),
            nulls_first: false,
        }
    }

    /// Sets the sort order for a single-column sort.
    #[must_use]
    pub fn order(mut self, order: SortOrder) -> Self {
        if let Some((_, o)) = self.columns.first_mut() {
            *o = order;
        }
        self
    }

    /// Sets whether nulls should appear first (default: false, nulls last).
    #[must_use]
    pub fn nulls_first(mut self, nulls_first: bool) -> Self {
        self.nulls_first = nulls_first;
        self
    }

    /// Returns the columns and their sort orders.
    pub fn columns(&self) -> &[(String, SortOrder)] {
        &self.columns
    }
}

impl Transform for Sort {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        use arrow::compute::{lexsort_to_indices, take, SortColumn, SortOptions};

        if batch.num_rows() <= 1 || self.columns.is_empty() {
            return Ok(batch);
        }

        let schema = batch.schema();

        // Build sort columns
        let sort_columns: Vec<SortColumn> = self
            .columns
            .iter()
            .map(|(col_name, order)| {
                let (idx, _) = schema
                    .column_with_name(col_name)
                    .ok_or_else(|| Error::column_not_found(col_name))?;

                Ok(SortColumn {
                    values: Arc::clone(batch.column(idx)),
                    options: Some(SortOptions {
                        descending: *order == SortOrder::Descending,
                        nulls_first: self.nulls_first,
                    }),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Get sorted indices
        let indices = lexsort_to_indices(&sort_columns, None).map_err(Error::Arrow)?;

        // Reorder all columns
        let new_columns: Vec<Arc<dyn Array>> = (0..batch.num_columns())
            .map(|col_idx| {
                let col = batch.column(col_idx);
                take(col.as_ref(), &indices, None)
                    .map_err(Error::Arrow)
                    .map(Arc::from)
            })
            .collect::<Result<Vec<_>>>()?;

        RecordBatch::try_new(schema, new_columns).map_err(Error::Arrow)
    }
}

/// A transform that removes duplicate rows based on specified columns.
///
/// # Example
///
/// ```ignore
/// use alimentar::Unique;
///
/// // Keep only unique rows based on all columns
/// let unique = Unique::all();
///
/// // Keep only unique rows based on specific columns
/// let unique = Unique::by(vec!["user_id", "date"]);
///
/// // Keep first occurrence (default) or last
/// let unique = Unique::by(vec!["id"]).keep_first();
/// let unique = Unique::by(vec!["id"]).keep_last();
/// ```
#[derive(Debug, Clone)]
pub struct Unique {
    columns: Option<Vec<String>>,
    keep_last: bool,
}

impl Unique {
    /// Creates a Unique transform that considers all columns.
    pub fn all() -> Self {
        Self {
            columns: None,
            keep_last: false,
        }
    }

    /// Creates a Unique transform that considers specific columns.
    pub fn by<S: Into<String>>(columns: impl IntoIterator<Item = S>) -> Self {
        Self {
            columns: Some(columns.into_iter().map(Into::into).collect()),
            keep_last: false,
        }
    }

    /// Keep the first occurrence of duplicates (default).
    #[must_use]
    pub fn keep_first(mut self) -> Self {
        self.keep_last = false;
        self
    }

    /// Keep the last occurrence of duplicates.
    #[must_use]
    pub fn keep_last(mut self) -> Self {
        self.keep_last = true;
        self
    }

    /// Returns the columns used for uniqueness check.
    pub fn columns(&self) -> Option<&[String]> {
        self.columns.as_deref()
    }

    fn row_key(batch: &RecordBatch, row_idx: usize, key_indices: &[usize]) -> u64 {
        use std::hash::{DefaultHasher, Hash, Hasher};

        use arrow::array::{
            BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
        };

        let mut hasher = DefaultHasher::new();

        for &col_idx in key_indices {
            let col = batch.column(col_idx);
            if col.is_null(row_idx) {
                // Sentinel: hash a tag byte that cannot collide with real values
                0u8.hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
                1u8.hash(&mut hasher);
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                2u8.hash(&mut hasher);
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
                // Use bits for exact comparison (same semantics as before)
                3u8.hash(&mut hasher);
                arr.value(row_idx).to_bits().hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                4u8.hash(&mut hasher);
                arr.value(row_idx).to_bits().hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                5u8.hash(&mut hasher);
                arr.value(row_idx).hash(&mut hasher);
            } else if let Some(arr) = col.as_any().downcast_ref::<BooleanArray>() {
                6u8.hash(&mut hasher);
                arr.value(row_idx).hash(&mut hasher);
            } else {
                // Fallback: hash the data type debug repr
                7u8.hash(&mut hasher);
                format!("{:?}", col.data_type()).hash(&mut hasher);
            }
        }

        hasher.finish()
    }
}

impl Transform for Unique {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        use std::collections::HashMap;

        let num_rows = batch.num_rows();
        if num_rows <= 1 {
            return Ok(batch);
        }

        let schema = batch.schema();

        // Determine which columns to use for uniqueness
        let key_indices: Vec<usize> = match &self.columns {
            Some(cols) => cols
                .iter()
                .map(|name| {
                    schema
                        .column_with_name(name)
                        .map(|(idx, _)| idx)
                        .ok_or_else(|| Error::column_not_found(name))
                })
                .collect::<Result<Vec<_>>>()?,
            None => (0..schema.fields().len()).collect(),
        };

        // Build a hash of each row's key columns (u64 hash keys to avoid String
        // allocation storm)
        let mut seen: HashMap<u64, usize> = HashMap::new();
        let mut keep_indices: Vec<usize> = Vec::new();

        let row_iter: Box<dyn Iterator<Item = usize>> = if self.keep_last {
            Box::new((0..num_rows).rev())
        } else {
            Box::new(0..num_rows)
        };

        for row_idx in row_iter {
            let key = Self::row_key(&batch, row_idx, &key_indices);

            if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(key) {
                e.insert(row_idx);
                keep_indices.push(row_idx);
            }
        }

        if self.keep_last {
            keep_indices.reverse();
        }

        if keep_indices.len() == num_rows {
            return Ok(batch);
        }

        // Build new batch with only unique rows
        let indices_array =
            arrow::array::UInt64Array::from_iter_values(keep_indices.iter().map(|&i| i as u64));

        let new_columns: Vec<Arc<dyn Array>> = (0..batch.num_columns())
            .map(|col_idx| {
                let col = batch.column(col_idx);
                arrow::compute::take(col.as_ref(), &indices_array, None)
                    .map_err(Error::Arrow)
                    .map(Arc::from)
            })
            .collect::<Result<Vec<_>>>()?;

        RecordBatch::try_new(schema, new_columns).map_err(Error::Arrow)
    }
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::redundant_closure
)]
mod tests {
    use arrow::{
        array::{Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
    };

    use super::*;

    fn create_test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
            Field::new("value", DataType::Int32, false),
        ]));

        let id_array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let name_array = StringArray::from(vec!["a", "b", "c", "d", "e"]);
        let value_array = Int32Array::from(vec![10, 20, 30, 40, 50]);

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array),
                Arc::new(name_array),
                Arc::new(value_array),
            ],
        )
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"))
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_shuffle_transform_deterministic() {
        let batch = create_test_batch();
        let transform = Shuffle::with_seed(42);

        let result1 = transform.apply(batch.clone());
        let result2 = transform.apply(batch);

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        let result1 = result1.ok().unwrap_or_else(|| panic!("Should succeed"));
        let result2 = result2.ok().unwrap_or_else(|| panic!("Should succeed"));

        // Same seed should produce same shuffle
        let col1 = result1
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let col2 = result2
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        for i in 0..col1.len() {
            assert_eq!(col1.value(i), col2.value(i));
        }
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_shuffle_preserves_row_integrity() {
        let batch = create_test_batch();
        let transform = Shuffle::with_seed(42);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        // Check that id-name-value relationships are preserved
        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let values = result
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        // Original: id=1 -> value=10, id=2 -> value=20, etc.
        for i in 0..ids.len() {
            let id = ids.value(i);
            let value = values.value(i);
            assert_eq!(value, id * 10);
        }
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_shuffle_single_row() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Shuffle::new();
        let result = transform.apply(batch);
        assert!(result.is_ok());
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_shuffle_default() {
        let shuffle = Shuffle::default();
        let batch = create_test_batch();
        let result = shuffle.apply(batch);
        assert!(result.is_ok());
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_shuffle_debug() {
        let shuffle = Shuffle::new();
        let debug_str = format!("{:?}", shuffle);
        assert!(debug_str.contains("Shuffle"));
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_shuffle_with_seed() {
        let batch = create_test_batch();
        let shuffle = Shuffle::with_seed(12345);

        let result1 = shuffle.apply(batch.clone());
        let result2 = shuffle.apply(batch);

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        let result1 = result1.ok().unwrap_or_else(|| panic!("Should succeed"));
        let result2 = result2.ok().unwrap_or_else(|| panic!("Should succeed"));

        let ids1 = result1
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let ids2 = result2
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        for i in 0..ids1.len() {
            assert_eq!(ids1.value(i), ids2.value(i));
        }
    }

    // Sample transform tests

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_sample_by_count() {
        let batch = create_test_batch();
        let transform = Sample::new(3).with_seed(42);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3);
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_sample_by_fraction() {
        let batch = create_test_batch();
        let transform = Sample::fraction(0.4).with_seed(42);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 2); // 5 * 0.4 = 2
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_sample_deterministic() {
        let batch = create_test_batch();
        let transform = Sample::new(3).with_seed(42);

        let result1 = transform.apply(batch.clone());
        let result2 = transform.apply(batch);

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        let result1 = result1.ok().unwrap_or_else(|| panic!("Should succeed"));
        let result2 = result2.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col1 = result1
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let col2 = result2
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        for i in 0..col1.len() {
            assert_eq!(col1.value(i), col2.value(i));
        }
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_sample_preserves_row_integrity() {
        let batch = create_test_batch();
        let transform = Sample::new(3).with_seed(42);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let values = result
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        // Check that id-value relationships are preserved
        for i in 0..ids.len() {
            let id = ids.value(i);
            let value = values.value(i);
            assert_eq!(value, id * 10);
        }
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_sample_count_larger_than_batch() {
        let batch = create_test_batch();
        let transform = Sample::new(100);

        let result = transform.apply(batch.clone());
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), batch.num_rows());
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_sample_getters() {
        let sample = Sample::new(10).with_seed(42);
        assert_eq!(sample.count(), Some(10));
        assert!(sample.sample_fraction().is_none());

        let sample2 = Sample::fraction(0.5);
        assert!(sample2.count().is_none());
        assert_eq!(sample2.sample_fraction(), Some(0.5));
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_sample_debug() {
        let sample = Sample::new(10);
        let debug_str = format!("{:?}", sample);
        assert!(debug_str.contains("Sample"));
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_sample_with_seed() {
        let batch = create_test_batch();
        let sample = Sample::new(3).with_seed(42);

        let result1 = sample.apply(batch.clone());
        let result2 = sample.apply(batch);

        assert!(result1.is_ok());
        assert!(result2.is_ok());

        let result1 = result1.ok().unwrap_or_else(|| panic!("Should succeed"));
        let result2 = result2.ok().unwrap_or_else(|| panic!("Should succeed"));

        assert_eq!(result1.num_rows(), result2.num_rows());
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_sample_zero_count() {
        let batch = create_test_batch();
        let sample = Sample::new(0);
        let result = sample.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        assert_eq!(result.num_rows(), 0);
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_sample_fraction_zero() {
        let batch = create_test_batch();
        let sample = Sample::fraction(0.0);
        let result = sample.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        assert_eq!(result.num_rows(), 0);
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_sample_fraction_full() {
        let batch = create_test_batch();
        let sample = Sample::fraction(1.0);
        let result = sample.apply(batch.clone());
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        assert_eq!(result.num_rows(), batch.num_rows());
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_sample_fraction_negative_clamped() {
        let sample = Sample::fraction(-0.5);
        // Negative fraction should be clamped to 0.0
        assert_eq!(sample.sample_fraction(), Some(0.0));

        let batch = create_test_batch();
        let result = sample.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        assert_eq!(result.num_rows(), 0);
    }

    #[cfg(feature = "shuffle")]
    #[test]
    fn test_sample_fraction_over_one_clamped() {
        let sample = Sample::fraction(1.5);
        // Fraction > 1.0 should be clamped to 1.0
        assert_eq!(sample.sample_fraction(), Some(1.0));

        let batch = create_test_batch();
        let result = sample.apply(batch.clone());
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        assert_eq!(result.num_rows(), batch.num_rows());
    }

    // Take transform tests

    #[test]
    fn test_take_transform() {
        let batch = create_test_batch();
        let transform = Take::new(3);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3);

        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        assert_eq!(ids.value(0), 1);
        assert_eq!(ids.value(1), 2);
        assert_eq!(ids.value(2), 3);
    }

    #[test]
    fn test_take_more_than_available() {
        let batch = create_test_batch();
        let transform = Take::new(100);

        let result = transform.apply(batch.clone());
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), batch.num_rows());
    }

    #[test]
    fn test_take_count_getter() {
        let take = Take::new(42);
        assert_eq!(take.count(), 42);
    }

    #[test]
    fn test_take_debug() {
        let take = Take::new(10);
        let debug_str = format!("{:?}", take);
        assert!(debug_str.contains("Take"));
    }

    #[test]
    fn test_take_zero_rows() {
        let batch = create_test_batch();
        let take = Take::new(0);
        let result = take.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn test_take_beyond_bounds() {
        let batch = create_test_batch(); // 5 rows
        let take = Take::new(100); // Request more than available
        let result = take.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        assert_eq!(result.num_rows(), 5); // Should return all rows
    }

    // Skip transform tests

    #[test]
    fn test_skip_transform() {
        let batch = create_test_batch();
        let transform = Skip::new(2);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3);

        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        assert_eq!(ids.value(0), 3);
        assert_eq!(ids.value(1), 4);
        assert_eq!(ids.value(2), 5);
    }

    #[test]
    fn test_skip_all_rows() {
        let batch = create_test_batch();
        let transform = Skip::new(10);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn test_skip_count_getter() {
        let skip = Skip::new(5);
        assert_eq!(skip.count(), 5);
    }

    #[test]
    fn test_skip_debug() {
        let skip = Skip::new(5);
        let debug_str = format!("{:?}", skip);
        assert!(debug_str.contains("Skip"));
    }

    #[test]
    fn test_skip_more_than_batch_size() {
        let batch = create_test_batch();
        let skip = Skip::new(100);
        let result = skip.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn test_skip_beyond_bounds() {
        let batch = create_test_batch(); // 5 rows
        let skip = Skip::new(100); // Skip more than available
        let result = skip.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        assert_eq!(result.num_rows(), 0); // Should return empty
    }

    #[test]
    fn test_skip_zero_rows() {
        let batch = create_test_batch();
        let original_rows = batch.num_rows();
        let skip = Skip::new(0);
        let result = skip.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        // Skipping 0 should return all rows
        assert_eq!(result.num_rows(), original_rows);
    }

    // Sort transform tests

    #[test]
    fn test_sort_ascending() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let values = Int32Array::from(vec![3, 1, 4, 1, 5]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Sort::by("value");
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        assert_eq!(col.value(0), 1);
        assert_eq!(col.value(1), 1);
        assert_eq!(col.value(2), 3);
        assert_eq!(col.value(3), 4);
        assert_eq!(col.value(4), 5);
    }

    #[test]
    fn test_sort_descending() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let values = Int32Array::from(vec![3, 1, 4, 1, 5]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Sort::by("value").order(SortOrder::Descending);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        assert_eq!(col.value(0), 5);
        assert_eq!(col.value(1), 4);
        assert_eq!(col.value(2), 3);
        assert_eq!(col.value(3), 1);
        assert_eq!(col.value(4), 1);
    }

    #[test]
    fn test_sort_preserves_row_integrity() {
        let batch = create_test_batch();
        let transform = Sort::by("id").order(SortOrder::Descending);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let values = result
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        // Verify rows are still correlated
        for i in 0..ids.len() {
            let id = ids.value(i);
            let value = values.value(i);
            assert_eq!(value, id * 10);
        }

        // Verify descending order
        assert_eq!(ids.value(0), 5);
        assert_eq!(ids.value(4), 1);
    }

    #[test]
    fn test_sort_multiple_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("group", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));
        let groups = Int32Array::from(vec![1, 2, 1, 2, 1]);
        let values = Int32Array::from(vec![30, 10, 10, 20, 20]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(groups), Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Sort::by_columns(vec![
            ("group", SortOrder::Ascending),
            ("value", SortOrder::Ascending),
        ]);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let groups = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let values = result
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        // Group 1 first, sorted by value
        assert_eq!(groups.value(0), 1);
        assert_eq!(values.value(0), 10);
        assert_eq!(groups.value(1), 1);
        assert_eq!(values.value(1), 20);
        assert_eq!(groups.value(2), 1);
        assert_eq!(values.value(2), 30);
        // Then group 2
        assert_eq!(groups.value(3), 2);
        assert_eq!(values.value(3), 10);
        assert_eq!(groups.value(4), 2);
        assert_eq!(values.value(4), 20);
    }

    #[test]
    fn test_sort_column_not_found() {
        let batch = create_test_batch();
        let transform = Sort::by("nonexistent");

        let result = transform.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_sort_columns_getter() {
        let sort = Sort::by("value").order(SortOrder::Descending);
        let cols = sort.columns();
        assert_eq!(cols.len(), 1);
        assert_eq!(cols[0].0, "value");
        assert_eq!(cols[0].1, SortOrder::Descending);
    }

    #[test]
    fn test_sort_order_default() {
        let order = SortOrder::default();
        assert_eq!(order, SortOrder::Ascending);
    }

    #[test]
    fn test_sort_single_row() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Sort::by("id");
        let result = transform.apply(batch);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sort_debug() {
        let sort = Sort::by("col");
        let debug_str = format!("{:?}", sort);
        assert!(debug_str.contains("Sort"));
    }

    #[test]
    fn test_sort_empty_batch() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let empty_batch = RecordBatch::new_empty(schema);

        let sort = Sort::by("id");
        let result = sort.apply(empty_batch);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sort_empty_columns_vector() {
        let batch = create_test_batch();
        let sort = Sort::by_columns::<String>(vec![]);
        let result = sort.apply(batch.clone());
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        // Empty sort columns should return unchanged batch
        assert_eq!(result.num_rows(), batch.num_rows());
    }

    #[test]
    fn test_sort_multi_column_one_missing() {
        let batch = create_test_batch();
        let sort = Sort::by_columns(vec![
            ("value".to_string(), SortOrder::Ascending),
            ("nonexistent".to_string(), SortOrder::Ascending),
        ]);
        let result = sort.apply(batch);
        // Should error because second column doesn't exist
        assert!(result.is_err());
    }

    #[test]
    fn test_sort_single_row_unchanged() {
        // Create single-row batch
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![42]))])
            .ok()
            .unwrap();

        let sort = Sort::by_columns(vec![("id".to_string(), SortOrder::Ascending)]);
        let result = sort.apply(batch.clone());
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        // Single row should be returned unchanged
        assert_eq!(result.num_rows(), 1);
    }

    // Unique tests

    #[test]
    fn test_unique_all_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));
        let ids = Int32Array::from(vec![1, 2, 1, 2, 1]); // Duplicates
        let values = Int32Array::from(vec![10, 20, 10, 20, 30]); // Row 0 == Row 2, Row 1 == Row 3
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Unique::all();
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        assert_eq!(result.num_rows(), 3); // Only 3 unique rows
    }

    #[test]
    fn test_unique_by_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));
        let ids = Int32Array::from(vec![1, 2, 1, 2, 3]);
        let values = Int32Array::from(vec![10, 20, 30, 40, 50]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Unique::by(vec!["id"]);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        assert_eq!(result.num_rows(), 3); // ids 1, 2, 3

        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let values = result
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        // Keep first occurrences by default
        assert_eq!(ids.value(0), 1);
        assert_eq!(values.value(0), 10); // First occurrence of id=1
        assert_eq!(ids.value(1), 2);
        assert_eq!(values.value(1), 20); // First occurrence of id=2
    }

    #[test]
    fn test_unique_keep_last() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
        ]));
        let ids = Int32Array::from(vec![1, 2, 1, 2, 3]);
        let values = Int32Array::from(vec![10, 20, 30, 40, 50]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Unique::by(vec!["id"]).keep_last();
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        assert_eq!(result.num_rows(), 3);

        let ids = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));
        let values = result
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        // Keep last occurrences
        assert_eq!(ids.value(0), 1);
        assert_eq!(values.value(0), 30); // Last occurrence of id=1
        assert_eq!(ids.value(1), 2);
        assert_eq!(values.value(1), 40); // Last occurrence of id=2
    }

    #[test]
    fn test_unique_no_duplicates() {
        let batch = create_test_batch();
        let transform = Unique::all();

        let result = transform.apply(batch.clone());
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        assert_eq!(result.num_rows(), batch.num_rows()); // All unique
    }

    #[test]
    fn test_unique_column_not_found() {
        let batch = create_test_batch();
        let transform = Unique::by(vec!["nonexistent"]);

        let result = transform.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_unique_columns_getter() {
        let unique = Unique::by(vec!["a", "b"]);
        assert!(unique.columns().is_some());
        assert_eq!(
            unique
                .columns()
                .unwrap_or_else(|| panic!("Should have columns")),
            &["a", "b"]
        );

        let unique2 = Unique::all();
        assert!(unique2.columns().is_none());
    }

    #[test]
    fn test_unique_single_row() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1]))])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Unique::all();
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 1);
    }

    #[test]
    fn test_unique_debug() {
        let unique = Unique::all();
        let debug_str = format!("{:?}", unique);
        assert!(debug_str.contains("Unique"));
    }

    #[test]
    fn test_unique_with_int64_column() {
        use arrow::array::Int64Array;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let id_arr = Int64Array::from(vec![1i64, 1i64, 2i64, 2i64, 3i64]);
        let name_arr = StringArray::from(vec!["a", "b", "c", "d", "e"]);

        let batch = RecordBatch::try_new(schema, vec![Arc::new(id_arr), Arc::new(name_arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let unique = Unique::by(vec!["id"]);
        let result = unique.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_unique_with_float64_column() {
        use arrow::array::Float64Array;

        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Float64, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let val_arr = Float64Array::from(vec![1.0f64, 1.0f64, 2.0f64, 2.0f64, 3.0f64]);
        let name_arr = StringArray::from(vec!["a", "b", "c", "d", "e"]);

        let batch = RecordBatch::try_new(schema, vec![Arc::new(val_arr), Arc::new(name_arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let unique = Unique::by(vec!["val"]);
        let result = unique.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_unique_with_float32_column() {
        use arrow::array::Float32Array;

        let schema = Arc::new(Schema::new(vec![
            Field::new("val", DataType::Float32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let val_arr = Float32Array::from(vec![1.0f32, 1.0f32, 2.0f32, 2.0f32, 3.0f32]);
        let name_arr = StringArray::from(vec!["a", "b", "c", "d", "e"]);

        let batch = RecordBatch::try_new(schema, vec![Arc::new(val_arr), Arc::new(name_arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let unique = Unique::by(vec!["val"]);
        let result = unique.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_unique_with_bool_column() {
        use arrow::array::BooleanArray;

        let schema = Arc::new(Schema::new(vec![
            Field::new("flag", DataType::Boolean, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let flag_arr = BooleanArray::from(vec![true, true, false, false, true]);
        let name_arr = StringArray::from(vec!["a", "b", "c", "d", "e"]);

        let batch = RecordBatch::try_new(schema, vec![Arc::new(flag_arr), Arc::new(name_arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let unique = Unique::by(vec!["flag"]);
        let result = unique.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 2);
    }

    #[test]
    fn test_unique_with_null_values() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("name", DataType::Utf8, false),
        ]));

        let id_arr = Int32Array::from(vec![Some(1), None, Some(1), None, Some(2)]);
        let name_arr = StringArray::from(vec!["a", "b", "c", "d", "e"]);

        let batch = RecordBatch::try_new(schema, vec![Arc::new(id_arr), Arc::new(name_arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let unique = Unique::by(vec!["id"]);
        let result = unique.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3); // 1, NULL, 2
    }

    #[test]
    fn test_unique_empty_batch() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let arr = Int32Array::from(Vec::<i32>::new());
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let unique = Unique::by(["id"]);
        let result = unique.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        assert_eq!(result.num_rows(), 0);
    }
}
