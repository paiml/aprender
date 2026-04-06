//! Data transforms for alimentar.
//!
//! Transforms apply operations to RecordBatches, enabling data preprocessing
//! pipelines. All transforms are composable and can be chained together.

use std::sync::Arc;

use arrow::{
    array::{BooleanArray, RecordBatch},
    compute::filter_record_batch,
};

use crate::error::{Error, Result};

#[cfg(feature = "shuffle")]
mod fim;
mod numeric;
mod row_ops;
mod selection;

#[cfg(feature = "shuffle")]
pub use fim::{Fim, FimFormat, FimTokens};
pub use numeric::{Cast, FillNull, FillStrategy, NormMethod, Normalize};
#[cfg(feature = "shuffle")]
pub use row_ops::{Sample, Shuffle};
pub use row_ops::{Skip, Sort, SortOrder, Take, Unique};
pub use selection::{Drop, Rename, Select};

/// A transform that can be applied to RecordBatches.
///
/// Transforms are the building blocks for data preprocessing pipelines.
/// They take a RecordBatch and produce a new RecordBatch with the
/// transformation applied.
///
/// # Thread Safety
///
/// All transforms must be thread-safe (Send + Sync) to support parallel
/// data loading in future versions.
pub trait Transform: Send + Sync {
    /// Applies the transform to a RecordBatch.
    ///
    /// # Errors
    ///
    /// Returns an error if the transform cannot be applied to the batch.
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch>;
}

/// A transform that applies a function to each RecordBatch.
///
/// # Example
///
/// ```ignore
/// use alimentar::Map;
///
/// let transform = Map::new(|batch| {
///     // Process batch
///     Ok(batch)
/// });
/// ```
pub struct Map<F>
where
    F: Fn(RecordBatch) -> Result<RecordBatch> + Send + Sync,
{
    func: F,
}

impl<F> Map<F>
where
    F: Fn(RecordBatch) -> Result<RecordBatch> + Send + Sync,
{
    /// Creates a new Map transform with the given function.
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F> Transform for Map<F>
where
    F: Fn(RecordBatch) -> Result<RecordBatch> + Send + Sync,
{
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        (self.func)(batch)
    }
}

/// A transform that filters rows based on a predicate.
///
/// The predicate function receives a RecordBatch and must return a BooleanArray
/// with the same number of rows, where `true` indicates the row should be kept.
///
/// # Example
///
/// ```ignore
/// use alimentar::Filter;
/// use arrow::array::{Int32Array, BooleanArray};
///
/// let filter = Filter::new(|batch| {
///     let col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
///     let mask: Vec<bool> = (0..col.len()).map(|i| col.value(i) > 5).collect();
///     Ok(BooleanArray::from(mask))
/// });
/// ```
pub struct Filter<F>
where
    F: Fn(&RecordBatch) -> Result<BooleanArray> + Send + Sync,
{
    predicate: F,
}

impl<F> Filter<F>
where
    F: Fn(&RecordBatch) -> Result<BooleanArray> + Send + Sync,
{
    /// Creates a new Filter transform with the given predicate.
    pub fn new(predicate: F) -> Self {
        Self { predicate }
    }
}

impl<F> Transform for Filter<F>
where
    F: Fn(&RecordBatch) -> Result<BooleanArray> + Send + Sync,
{
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let mask = (self.predicate)(&batch)?;
        filter_record_batch(&batch, &mask).map_err(Error::Arrow)
    }
}

/// A chain of transforms applied in sequence.
///
/// # Example
///
/// ```ignore
/// use alimentar::{Chain, Select, Shuffle};
///
/// let chain = Chain::new()
///     .then(Select::new(vec!["id", "value"]))
///     .then(Shuffle::with_seed(42));
/// ```
pub struct Chain {
    transforms: Vec<Box<dyn Transform>>,
}

impl Chain {
    /// Creates a new empty transform chain.
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Adds a transform to the chain.
    #[must_use]
    pub fn then<T: Transform + 'static>(mut self, transform: T) -> Self {
        self.transforms.push(Box::new(transform));
        self
    }

    /// Returns the number of transforms in the chain.
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Returns true if the chain has no transforms.
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl Default for Chain {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for Chain {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let mut result = batch;
        for transform in &self.transforms {
            result = transform.apply(result)?;
        }
        Ok(result)
    }
}

// Implement Transform for boxed transforms
impl Transform for Box<dyn Transform> {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        (**self).apply(batch)
    }
}

// Implement Transform for Arc<dyn Transform>
impl Transform for Arc<dyn Transform> {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        (**self).apply(batch)
    }
}

#[cfg(test)]
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

    #[test]
    fn test_map_transform() {
        let batch = create_test_batch();
        let transform = Map::new(Ok); // Identity transform

        let result = transform.apply(batch.clone());
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), batch.num_rows());
    }

    #[test]
    fn test_filter_transform() {
        let batch = create_test_batch();
        let transform = Filter::new(|b| {
            let col = b
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| Error::transform("Expected Int32Array"))?;
            let mask: Vec<bool> = (0..col.len()).map(|i| col.value(i) > 2).collect();
            Ok(BooleanArray::from(mask))
        });

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3); // Only id > 2: 3, 4, 5
    }

    #[test]
    fn test_chain_transform() {
        let batch = create_test_batch();
        let chain = Chain::new()
            .then(Select::new(vec!["id", "value"]))
            .then(Take::new(3));

        assert_eq!(chain.len(), 2);
        assert!(!chain.is_empty());

        let result = chain.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_empty_chain() {
        let batch = create_test_batch();
        let chain = Chain::new();

        assert!(chain.is_empty());

        let result = chain.apply(batch.clone());
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), batch.num_rows());
    }

    #[test]
    fn test_filter_empty_result() {
        let batch = create_test_batch();
        let filter = Filter::new(|batch| Ok(BooleanArray::from(vec![false; batch.num_rows()])));

        let result = filter.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn test_map_with_error() {
        let batch = create_test_batch();
        let map = Map::new(|_batch| Err(crate::Error::transform("intentional error")));
        let result = map.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_filter_closure() {
        let batch = create_test_batch();
        // Test with a closure that filters to only rows where id > 2
        let filter = Filter::new(|batch: &RecordBatch| {
            let id_col = batch.column(0).as_any().downcast_ref::<Int32Array>();
            if let Some(arr) = id_col {
                let mask: Vec<bool> = (0..arr.len()).map(|i| arr.value(i) > 2).collect();
                Ok(arrow::array::BooleanArray::from(mask))
            } else {
                Ok(arrow::array::BooleanArray::from(vec![
                    false;
                    batch.num_rows()
                ]))
            }
        });
        let result = filter.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), 3); // rows with id 3, 4, 5
    }

    #[test]
    fn test_filter_all_rows_filtered() {
        let batch = create_test_batch();
        // Filter that removes all rows (5 rows in test batch)
        let filter = Filter::new(|_batch: &RecordBatch| {
            Ok(arrow::array::BooleanArray::from(vec![false; 5]))
        });
        let result = filter.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn test_map_error_propagation() {
        let batch = create_test_batch();
        // Map that returns error
        let map = Map::new(|_batch: RecordBatch| Err(crate::Error::transform("Test error")));
        let result = map.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_chain_empty_transforms() {
        let batch = create_test_batch();
        let chain: Chain = Chain::new();
        let result = chain.apply(batch.clone());
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        assert_eq!(result.num_rows(), batch.num_rows());
    }

    #[test]
    fn test_boxed_transform_delegation() {
        let batch = create_test_batch();
        let take = Take::new(2);
        let boxed: Box<dyn Transform> = Box::new(take);
        let result = boxed.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        assert_eq!(result.num_rows(), 2);
    }

    #[test]
    fn test_arc_transform_delegation() {
        use std::sync::Arc as StdArc;
        let batch = create_test_batch();
        let take = Take::new(3);
        let arced: StdArc<dyn Transform> = StdArc::new(take);
        let result = arced.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_chain_single_transform() {
        let batch = create_test_batch();
        let chain = Chain::new().then(Take::new(2));
        let result = chain.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap();
        assert_eq!(result.num_rows(), 2);
    }

    #[test]
    fn test_chain_with_multiple_transforms() {
        let batch = create_test_batch();

        let chain = Chain::new()
            .then(Select::new(vec!["id", "name"]))
            .then(Rename::from_pairs([("id", "identifier")]));

        let result = chain.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert!(result.schema().field_with_name("identifier").is_ok());
    }
}
