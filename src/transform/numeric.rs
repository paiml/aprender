//! Numeric transforms: normalization, casting, and null filling.

use std::sync::Arc;

use arrow::{
    array::{Array, RecordBatch},
    datatypes::{Field, Schema},
};

use super::Transform;
use crate::error::{Error, Result};

/// Normalization method for numeric columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormMethod {
    /// Min-max normalization: (x - min) / (max - min), scales to [0, 1]
    MinMax,
    /// Z-score normalization: (x - mean) / std, centers around 0
    ZScore,
    /// Scale to unit length (L2 norm)
    L2,
}

/// A transform that normalizes numeric columns.
///
/// Supports min-max scaling, z-score standardization, and L2 normalization.
///
/// # Example
///
/// ```ignore
/// use alimentar::{Normalize, NormMethod};
///
/// // Min-max normalize specific columns
/// let normalize = Normalize::new(vec!["feature1", "feature2"], NormMethod::MinMax);
///
/// // Z-score normalize all numeric columns
/// let normalize = Normalize::all_numeric(NormMethod::ZScore);
/// ```
#[derive(Debug, Clone)]
pub struct Normalize {
    columns: Option<Vec<String>>,
    method: NormMethod,
}

impl Normalize {
    /// Creates a Normalize transform for specific columns.
    pub fn new<S: Into<String>>(columns: impl IntoIterator<Item = S>, method: NormMethod) -> Self {
        Self {
            columns: Some(columns.into_iter().map(Into::into).collect()),
            method,
        }
    }

    /// Creates a Normalize transform that applies to all numeric columns.
    pub fn all_numeric(method: NormMethod) -> Self {
        Self {
            columns: None,
            method,
        }
    }

    /// Returns the columns to normalize (None means all numeric).
    pub fn columns(&self) -> Option<&[String]> {
        self.columns.as_deref()
    }

    /// Returns the normalization method.
    pub fn method(&self) -> NormMethod {
        self.method
    }

    fn is_numeric_type(dtype: &arrow::datatypes::DataType) -> bool {
        use arrow::datatypes::DataType;
        matches!(
            dtype,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
        )
    }

    fn normalize_array(
        &self,
        array: &dyn Array,
        _dtype: &arrow::datatypes::DataType,
    ) -> Result<Arc<dyn Array>> {
        use arrow::{array::Float64Array, compute::cast, datatypes::DataType};

        // Cast to Float64 for normalization
        let float_array = cast(array, &DataType::Float64)
            .map_err(|e| Error::transform(format!("Failed to cast to Float64: {}", e)))?;

        let float_values = float_array
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| Error::transform("Expected Float64Array after cast"))?;

        let normalized = match self.method {
            NormMethod::MinMax => Self::min_max_normalize(float_values),
            NormMethod::ZScore => Self::zscore_normalize(float_values),
            NormMethod::L2 => Self::l2_normalize(float_values),
        };

        // Normalized values are always Float64 (they're now in [0,1] or centered)
        Ok(Arc::new(normalized))
    }

    fn min_max_normalize(array: &arrow::array::Float64Array) -> arrow::array::Float64Array {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        for i in 0..array.len() {
            if !array.is_null(i) {
                let v = array.value(i);
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
        }

        let range = max - min;
        let values: Vec<Option<f64>> = (0..array.len())
            .map(|i| {
                if array.is_null(i) {
                    None
                } else if range == 0.0 {
                    Some(0.0)
                } else {
                    Some((array.value(i) - min) / range)
                }
            })
            .collect();

        arrow::array::Float64Array::from(values)
    }

    #[allow(clippy::cast_precision_loss)]
    fn zscore_normalize(array: &arrow::array::Float64Array) -> arrow::array::Float64Array {
        let mut sum = 0.0;
        let mut count = 0usize;

        for i in 0..array.len() {
            if !array.is_null(i) {
                sum += array.value(i);
                count += 1;
            }
        }

        if count == 0 {
            return array.clone();
        }

        let mean = sum / count as f64;

        let mut variance_sum = 0.0;
        for i in 0..array.len() {
            if !array.is_null(i) {
                let diff = array.value(i) - mean;
                variance_sum += diff * diff;
            }
        }

        let std = (variance_sum / count as f64).sqrt();

        let values: Vec<Option<f64>> = (0..array.len())
            .map(|i| {
                if array.is_null(i) {
                    None
                } else if std == 0.0 {
                    Some(0.0)
                } else {
                    Some((array.value(i) - mean) / std)
                }
            })
            .collect();

        arrow::array::Float64Array::from(values)
    }

    fn l2_normalize(array: &arrow::array::Float64Array) -> arrow::array::Float64Array {
        let mut sum_sq = 0.0;

        for i in 0..array.len() {
            if !array.is_null(i) {
                let v = array.value(i);
                sum_sq += v * v;
            }
        }

        let norm = sum_sq.sqrt();

        let values: Vec<Option<f64>> = (0..array.len())
            .map(|i| {
                if array.is_null(i) {
                    None
                } else if norm == 0.0 {
                    Some(0.0)
                } else {
                    Some(array.value(i) / norm)
                }
            })
            .collect();

        arrow::array::Float64Array::from(values)
    }
}

impl Transform for Normalize {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let schema = batch.schema();

        let columns_to_normalize: std::collections::HashSet<&str> = match &self.columns {
            Some(cols) => cols.iter().map(String::as_str).collect(),
            None => schema
                .fields()
                .iter()
                .filter(|f| Self::is_numeric_type(f.data_type()))
                .map(|f| f.name().as_str())
                .collect(),
        };

        let mut fields = Vec::with_capacity(schema.fields().len());
        let mut arrays = Vec::with_capacity(schema.fields().len());

        for (idx, field) in schema.fields().iter().enumerate() {
            let col = batch.column(idx);

            if columns_to_normalize.contains(field.name().as_str()) {
                if !Self::is_numeric_type(field.data_type()) {
                    return Err(Error::transform(format!(
                        "Column '{}' is not numeric (type: {:?})",
                        field.name(),
                        field.data_type()
                    )));
                }

                let normalized = self.normalize_array(col.as_ref(), field.data_type())?;
                // Normalized columns become Float64
                fields.push(Field::new(
                    field.name(),
                    arrow::datatypes::DataType::Float64,
                    field.is_nullable(),
                ));
                arrays.push(normalized);
            } else {
                fields.push(field.as_ref().clone());
                arrays.push(Arc::clone(col));
            }
        }

        let new_schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(new_schema, arrays).map_err(Error::Arrow)
    }
}

/// A transform that casts columns to different data types.
///
/// # Example
///
/// ```ignore
/// use alimentar::Cast;
/// use arrow::datatypes::DataType;
///
/// let cast = Cast::new(vec![
///     ("id", DataType::Int64),
///     ("value", DataType::Float64),
/// ]);
/// ```
#[derive(Debug, Clone)]
pub struct Cast {
    mappings: Vec<(String, arrow::datatypes::DataType)>,
}

impl Cast {
    /// Creates a new Cast transform with column-to-type mappings.
    pub fn new<S: Into<String>>(
        mappings: impl IntoIterator<Item = (S, arrow::datatypes::DataType)>,
    ) -> Self {
        Self {
            mappings: mappings
                .into_iter()
                .map(|(name, dtype)| (name.into(), dtype))
                .collect(),
        }
    }

    /// Returns the cast mappings.
    pub fn mappings(&self) -> &[(String, arrow::datatypes::DataType)] {
        &self.mappings
    }
}

impl Transform for Cast {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        use arrow::compute::cast;

        let schema = batch.schema();
        let cast_map: std::collections::HashMap<&str, &arrow::datatypes::DataType> =
            self.mappings.iter().map(|(n, t)| (n.as_str(), t)).collect();

        let mut fields = Vec::with_capacity(schema.fields().len());
        let mut arrays = Vec::with_capacity(schema.fields().len());

        for (idx, field) in schema.fields().iter().enumerate() {
            let col = batch.column(idx);

            if let Some(&target_type) = cast_map.get(field.name().as_str()) {
                let casted = cast(col.as_ref(), target_type).map_err(|e| {
                    Error::transform(format!(
                        "Failed to cast column '{}' to {:?}: {}",
                        field.name(),
                        target_type,
                        e
                    ))
                })?;
                fields.push(Field::new(
                    field.name(),
                    target_type.clone(),
                    field.is_nullable(),
                ));
                arrays.push(casted);
            } else {
                fields.push(field.as_ref().clone());
                arrays.push(Arc::clone(col));
            }
        }

        let new_schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(new_schema, arrays).map_err(Error::Arrow)
    }
}

/// Strategy for filling null values.
#[derive(Debug, Clone)]
pub enum FillStrategy {
    /// Fill with a constant integer value
    Int(i64),
    /// Fill with a constant float value
    Float(f64),
    /// Fill with a constant string value
    String(String),
    /// Fill with a constant boolean value
    Bool(bool),
    /// Fill with zero (works for numeric types)
    Zero,
    /// Forward fill (use previous non-null value)
    Forward,
    /// Backward fill (use next non-null value)
    Backward,
}

/// A transform that fills null values in specified columns.
///
/// # Example
///
/// ```ignore
/// use alimentar::{FillNull, FillStrategy};
///
/// // Fill nulls in "age" column with 0
/// let fill = FillNull::new("age", FillStrategy::Zero);
///
/// // Fill nulls with a specific value
/// let fill = FillNull::new("score", FillStrategy::Float(0.0));
///
/// // Forward fill (use previous value)
/// let fill = FillNull::new("value", FillStrategy::Forward);
/// ```
#[derive(Debug, Clone)]
pub struct FillNull {
    column: String,
    strategy: FillStrategy,
}

impl FillNull {
    /// Creates a FillNull transform for the specified column.
    pub fn new<S: Into<String>>(column: S, strategy: FillStrategy) -> Self {
        Self {
            column: column.into(),
            strategy,
        }
    }

    /// Creates a FillNull transform that fills with zero.
    pub fn with_zero<S: Into<String>>(column: S) -> Self {
        Self::new(column, FillStrategy::Zero)
    }

    /// Returns the column name.
    pub fn column(&self) -> &str {
        &self.column
    }

    /// Returns the fill strategy.
    pub fn strategy(&self) -> &FillStrategy {
        &self.strategy
    }

    fn fill_i32_array(col: &Arc<dyn Array>, fill_value: i32) -> arrow::array::Int32Array {
        use arrow::array::Int32Array;
        let arr = col.as_any().downcast_ref::<Int32Array>();
        if let Some(arr) = arr {
            let values: Vec<i32> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        fill_value
                    } else {
                        arr.value(i)
                    }
                })
                .collect();
            Int32Array::from(values)
        } else {
            Int32Array::from(Vec::<i32>::new())
        }
    }

    fn fill_i64_array(col: &Arc<dyn Array>, fill_value: i64) -> arrow::array::Int64Array {
        use arrow::array::Int64Array;
        let arr = col.as_any().downcast_ref::<Int64Array>();
        if let Some(arr) = arr {
            let values: Vec<i64> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        fill_value
                    } else {
                        arr.value(i)
                    }
                })
                .collect();
            Int64Array::from(values)
        } else {
            Int64Array::from(Vec::<i64>::new())
        }
    }

    fn fill_f32_array(col: &Arc<dyn Array>, fill_value: f32) -> arrow::array::Float32Array {
        use arrow::array::Float32Array;
        let arr = col.as_any().downcast_ref::<Float32Array>();
        if let Some(arr) = arr {
            let values: Vec<f32> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        fill_value
                    } else {
                        arr.value(i)
                    }
                })
                .collect();
            Float32Array::from(values)
        } else {
            Float32Array::from(Vec::<f32>::new())
        }
    }

    fn fill_f64_array(col: &Arc<dyn Array>, fill_value: f64) -> arrow::array::Float64Array {
        use arrow::array::Float64Array;
        let arr = col.as_any().downcast_ref::<Float64Array>();
        if let Some(arr) = arr {
            let values: Vec<f64> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        fill_value
                    } else {
                        arr.value(i)
                    }
                })
                .collect();
            Float64Array::from(values)
        } else {
            Float64Array::from(Vec::<f64>::new())
        }
    }

    fn fill_string_array(col: &Arc<dyn Array>, fill_value: &str) -> arrow::array::StringArray {
        use arrow::array::StringArray;
        let arr = col.as_any().downcast_ref::<StringArray>();
        if let Some(arr) = arr {
            let values: Vec<&str> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        fill_value
                    } else {
                        arr.value(i)
                    }
                })
                .collect();
            StringArray::from(values)
        } else {
            StringArray::from(Vec::<&str>::new())
        }
    }

    fn fill_bool_array(col: &Arc<dyn Array>, fill_value: bool) -> arrow::array::BooleanArray {
        use arrow::array::BooleanArray;
        let arr = col.as_any().downcast_ref::<BooleanArray>();
        if let Some(arr) = arr {
            let values: Vec<bool> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        fill_value
                    } else {
                        arr.value(i)
                    }
                })
                .collect();
            BooleanArray::from(values)
        } else {
            BooleanArray::from(Vec::<bool>::new())
        }
    }

    fn forward_fill_i32(col: &Arc<dyn Array>) -> arrow::array::Int32Array {
        use arrow::array::Int32Array;
        let arr = col.as_any().downcast_ref::<Int32Array>();
        if let Some(arr) = arr {
            let mut last: Option<i32> = None;
            let values: Vec<Option<i32>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        last
                    } else {
                        let v = arr.value(i);
                        last = Some(v);
                        Some(v)
                    }
                })
                .collect();
            Int32Array::from(values)
        } else {
            Int32Array::from(Vec::<i32>::new())
        }
    }

    fn forward_fill_i64(col: &Arc<dyn Array>) -> arrow::array::Int64Array {
        use arrow::array::Int64Array;
        let arr = col.as_any().downcast_ref::<Int64Array>();
        if let Some(arr) = arr {
            let mut last: Option<i64> = None;
            let values: Vec<Option<i64>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        last
                    } else {
                        let v = arr.value(i);
                        last = Some(v);
                        Some(v)
                    }
                })
                .collect();
            Int64Array::from(values)
        } else {
            Int64Array::from(Vec::<i64>::new())
        }
    }

    fn forward_fill_f64(col: &Arc<dyn Array>) -> arrow::array::Float64Array {
        use arrow::array::Float64Array;
        let arr = col.as_any().downcast_ref::<Float64Array>();
        if let Some(arr) = arr {
            let mut last: Option<f64> = None;
            let values: Vec<Option<f64>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        last
                    } else {
                        let v = arr.value(i);
                        last = Some(v);
                        Some(v)
                    }
                })
                .collect();
            Float64Array::from(values)
        } else {
            Float64Array::from(Vec::<f64>::new())
        }
    }

    fn backward_fill_i32(col: &Arc<dyn Array>) -> arrow::array::Int32Array {
        use arrow::array::Int32Array;
        let arr = col.as_any().downcast_ref::<Int32Array>();
        if let Some(arr) = arr {
            let mut next: Option<i32> = None;
            let mut values: Vec<Option<i32>> = (0..arr.len())
                .rev()
                .map(|i| {
                    if arr.is_null(i) {
                        next
                    } else {
                        let v = arr.value(i);
                        next = Some(v);
                        Some(v)
                    }
                })
                .collect();
            values.reverse();
            Int32Array::from(values)
        } else {
            Int32Array::from(Vec::<i32>::new())
        }
    }

    fn backward_fill_i64(col: &Arc<dyn Array>) -> arrow::array::Int64Array {
        use arrow::array::Int64Array;
        let arr = col.as_any().downcast_ref::<Int64Array>();
        if let Some(arr) = arr {
            let mut next: Option<i64> = None;
            let mut values: Vec<Option<i64>> = (0..arr.len())
                .rev()
                .map(|i| {
                    if arr.is_null(i) {
                        next
                    } else {
                        let v = arr.value(i);
                        next = Some(v);
                        Some(v)
                    }
                })
                .collect();
            values.reverse();
            Int64Array::from(values)
        } else {
            Int64Array::from(Vec::<i64>::new())
        }
    }

    fn backward_fill_f64(col: &Arc<dyn Array>) -> arrow::array::Float64Array {
        use arrow::array::Float64Array;
        let arr = col.as_any().downcast_ref::<Float64Array>();
        if let Some(arr) = arr {
            let mut next: Option<f64> = None;
            let mut values: Vec<Option<f64>> = (0..arr.len())
                .rev()
                .map(|i| {
                    if arr.is_null(i) {
                        next
                    } else {
                        let v = arr.value(i);
                        next = Some(v);
                        Some(v)
                    }
                })
                .collect();
            values.reverse();
            Float64Array::from(values)
        } else {
            Float64Array::from(Vec::<f64>::new())
        }
    }
}

impl Transform for FillNull {
    #[allow(clippy::cast_possible_truncation)]
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        use arrow::datatypes::DataType;

        let schema = batch.schema();
        let (col_idx, field) = schema
            .column_with_name(&self.column)
            .ok_or_else(|| Error::column_not_found(&self.column))?;

        let mut arrays: Vec<Arc<dyn Array>> = batch.columns().to_vec();
        let col = batch.column(col_idx);

        let filled: Arc<dyn Array> = match (field.data_type(), &self.strategy) {
            (DataType::Int32, FillStrategy::Int(v)) => {
                Arc::new(Self::fill_i32_array(col, *v as i32))
            }
            (DataType::Int64, FillStrategy::Int(v)) => Arc::new(Self::fill_i64_array(col, *v)),
            (DataType::Float32, FillStrategy::Float(v)) => {
                Arc::new(Self::fill_f32_array(col, *v as f32))
            }
            (DataType::Float64, FillStrategy::Float(v)) => Arc::new(Self::fill_f64_array(col, *v)),
            (DataType::Int32, FillStrategy::Zero) => Arc::new(Self::fill_i32_array(col, 0)),
            (DataType::Int64, FillStrategy::Zero) => Arc::new(Self::fill_i64_array(col, 0)),
            (DataType::Float32, FillStrategy::Zero) => Arc::new(Self::fill_f32_array(col, 0.0)),
            (DataType::Float64, FillStrategy::Zero) => Arc::new(Self::fill_f64_array(col, 0.0)),
            (DataType::Utf8, FillStrategy::String(s)) => Arc::new(Self::fill_string_array(col, s)),
            (DataType::Boolean, FillStrategy::Bool(b)) => Arc::new(Self::fill_bool_array(col, *b)),
            (DataType::Int32, FillStrategy::Forward) => Arc::new(Self::forward_fill_i32(col)),
            (DataType::Int64, FillStrategy::Forward) => Arc::new(Self::forward_fill_i64(col)),
            (DataType::Float64, FillStrategy::Forward) => Arc::new(Self::forward_fill_f64(col)),
            (DataType::Int32, FillStrategy::Backward) => Arc::new(Self::backward_fill_i32(col)),
            (DataType::Int64, FillStrategy::Backward) => Arc::new(Self::backward_fill_i64(col)),
            (DataType::Float64, FillStrategy::Backward) => Arc::new(Self::backward_fill_f64(col)),
            _ => {
                return Err(Error::transform(format!(
                    "Unsupported type {:?} for fill strategy {:?}",
                    field.data_type(),
                    self.strategy
                )));
            }
        };

        arrays[col_idx] = filled;
        RecordBatch::try_new(schema, arrays).map_err(Error::Arrow)
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
        datatypes::DataType,
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
    fn test_cast_transform() {
        let batch = create_test_batch();
        let transform = Cast::new(vec![("id", DataType::Int64), ("value", DataType::Float64)]);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        assert_eq!(result.schema().field(0).data_type(), &DataType::Int64);
        assert_eq!(result.schema().field(2).data_type(), &DataType::Float64);
    }

    #[test]
    fn test_cast_preserves_values() {
        let batch = create_test_batch();
        let transform = Cast::new(vec![("id", DataType::Float64)]);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap_or_else(|| panic!("Should be Float64Array"));

        assert_eq!(col.value(0), 1.0);
        assert_eq!(col.value(1), 2.0);
        assert_eq!(col.value(2), 3.0);
    }

    #[test]
    fn test_cast_mappings_getter() {
        let transform = Cast::new(vec![("a", DataType::Int64)]);
        assert_eq!(transform.mappings().len(), 1);
        assert_eq!(transform.mappings()[0].0, "a");
        assert_eq!(transform.mappings()[0].1, DataType::Int64);
    }

    #[test]
    fn test_cast_debug() {
        let cast = Cast::new(vec![("col", DataType::Int64)]);
        let debug_str = format!("{:?}", cast);
        assert!(debug_str.contains("Cast"));
    }

    #[test]
    fn test_cast_int_to_string() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let cast = Cast::new(vec![("id", DataType::Utf8)]);
        let result = cast.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.schema().field(0).data_type(), &DataType::Utf8);
    }

    #[test]
    fn test_cast_nonexistent_column() {
        let batch = create_test_batch();
        let cast = Cast::new(vec![("nonexistent", DataType::Int64)]);
        let result = cast.apply(batch.clone());
        // Cast on nonexistent column is a no-op (doesn't error)
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        assert_eq!(result.num_rows(), batch.num_rows());
    }

    #[test]
    fn test_cast_incompatible_type() {
        // Try to cast string to int - should fail
        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let arr = StringArray::from(vec!["hello", "world"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let cast = Cast::new(vec![("text", DataType::Int32)]);
        let result = cast.apply(batch);
        // Arrow cast will fail for incompatible types
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_minmax() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));
        let values = arrow::array::Float64Array::from(vec![0.0, 50.0, 100.0]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Normalize::new(vec!["value"], NormMethod::MinMax);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap_or_else(|| panic!("Should be Float64Array"));

        assert!((col.value(0) - 0.0).abs() < 1e-10);
        assert!((col.value(1) - 0.5).abs() < 1e-10);
        assert!((col.value(2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zscore() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));
        // Values with mean=0, std=1
        let values = arrow::array::Float64Array::from(vec![-1.0, 0.0, 1.0]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Normalize::new(vec!["value"], NormMethod::ZScore);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap_or_else(|| panic!("Should be Float64Array"));

        // Mean should be 0 (within floating point tolerance)
        let mean: f64 = (0..col.len()).map(|i| col.value(i)).sum::<f64>() / col.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_normalize_l2() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));
        let values = arrow::array::Float64Array::from(vec![3.0, 4.0]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Normalize::new(vec!["value"], NormMethod::L2);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap_or_else(|| panic!("Should be Float64Array"));

        // L2 norm of [3, 4] is 5, so normalized should be [0.6, 0.8]
        assert!((col.value(0) - 0.6).abs() < 1e-10);
        assert!((col.value(1) - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_all_numeric() {
        let batch = create_test_batch();
        let transform = Normalize::all_numeric(NormMethod::MinMax);

        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        // id and value columns should be Float64 now (normalized)
        assert_eq!(result.schema().field(0).data_type(), &DataType::Float64);
        assert_eq!(result.schema().field(1).data_type(), &DataType::Utf8); // name unchanged
        assert_eq!(result.schema().field(2).data_type(), &DataType::Float64);
    }

    #[test]
    fn test_normalize_non_numeric_error() {
        let batch = create_test_batch();
        let transform = Normalize::new(vec!["name"], NormMethod::MinMax);

        let result = transform.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_getters() {
        let transform = Normalize::new(vec!["a", "b"], NormMethod::ZScore);
        assert!(transform.columns().is_some());
        assert_eq!(
            transform
                .columns()
                .unwrap_or_else(|| panic!("Should have columns")),
            &["a", "b"]
        );
        assert_eq!(transform.method(), NormMethod::ZScore);

        let transform2 = Normalize::all_numeric(NormMethod::L2);
        assert!(transform2.columns().is_none());
        assert_eq!(transform2.method(), NormMethod::L2);
    }

    #[test]
    fn test_normalize_constant_values() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));
        // All same values - should result in 0 (no range)
        let values = arrow::array::Float64Array::from(vec![5.0, 5.0, 5.0]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Normalize::new(vec!["value"], NormMethod::MinMax);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap_or_else(|| panic!("Should be Float64Array"));

        // With constant values, minmax returns 0
        for i in 0..col.len() {
            assert_eq!(col.value(i), 0.0);
        }
    }

    #[test]
    fn test_normalize_with_nulls() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )]));
        let values = arrow::array::Float64Array::from(vec![Some(0.0), None, Some(100.0)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = Normalize::new(vec!["value"], NormMethod::MinMax);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap_or_else(|| panic!("Should be Float64Array"));

        assert!((col.value(0) - 0.0).abs() < 1e-10);
        assert!(col.is_null(1));
        assert!((col.value(2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_debug() {
        let normalize = Normalize::new(vec!["col"], NormMethod::MinMax);
        let debug_str = format!("{:?}", normalize);
        assert!(debug_str.contains("Normalize"));
    }

    #[test]
    fn test_normalize_nonexistent_column() {
        let batch = create_test_batch();
        let normalize = Normalize::new(["nonexistent"], NormMethod::MinMax);
        let result = normalize.apply(batch);
        // Normalizing nonexistent column returns Ok (skips the column)
        assert!(result.is_ok());
    }

    // FillNull tests

    #[test]
    fn test_fillnull_with_zero_i32() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            true,
        )]));
        let values = Int32Array::from(vec![Some(1), None, Some(3), None, Some(5)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = FillNull::with_zero("value");
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        assert_eq!(col.value(0), 1);
        assert_eq!(col.value(1), 0); // Was null, now 0
        assert_eq!(col.value(2), 3);
        assert_eq!(col.value(3), 0); // Was null, now 0
        assert_eq!(col.value(4), 5);
    }

    #[test]
    fn test_fillnull_with_int_value() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            true,
        )]));
        let values = Int32Array::from(vec![Some(1), None, Some(3)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = FillNull::new("value", FillStrategy::Int(-1));
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        assert_eq!(col.value(1), -1);
    }

    #[test]
    fn test_fillnull_with_float() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )]));
        let values = arrow::array::Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = FillNull::new("value", FillStrategy::Float(99.9));
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap_or_else(|| panic!("Should be Float64Array"));

        assert!((col.value(1) - 99.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_fillnull_with_string() {
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, true)]));
        let values = StringArray::from(vec![Some("a"), None, Some("c")]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = FillNull::new("name", FillStrategy::String("unknown".to_string()));
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap_or_else(|| panic!("Should be StringArray"));

        assert_eq!(col.value(1), "unknown");
    }

    #[test]
    fn test_fillnull_forward_fill() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            true,
        )]));
        let values = Int32Array::from(vec![Some(1), None, None, Some(4), None]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = FillNull::new("value", FillStrategy::Forward);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        assert_eq!(col.value(0), 1);
        assert_eq!(col.value(1), 1); // Forward filled from index 0
        assert_eq!(col.value(2), 1); // Forward filled from index 0
        assert_eq!(col.value(3), 4);
        assert_eq!(col.value(4), 4); // Forward filled from index 3
    }

    #[test]
    fn test_fillnull_backward_fill() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            true,
        )]));
        let values = Int32Array::from(vec![None, None, Some(3), None, Some(5)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(values)])
            .ok()
            .unwrap_or_else(|| panic!("Should create batch"));

        let transform = FillNull::new("value", FillStrategy::Backward);
        let result = transform.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("Should be Int32Array"));

        assert_eq!(col.value(0), 3); // Backward filled from index 2
        assert_eq!(col.value(1), 3); // Backward filled from index 2
        assert_eq!(col.value(2), 3);
        assert_eq!(col.value(3), 5); // Backward filled from index 4
        assert_eq!(col.value(4), 5);
    }

    #[test]
    fn test_fillnull_column_not_found() {
        let batch = create_test_batch();
        let transform = FillNull::with_zero("nonexistent");

        let result = transform.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_fillnull_getters() {
        let transform = FillNull::new("col", FillStrategy::Int(42));
        assert_eq!(transform.column(), "col");
        assert!(matches!(transform.strategy(), FillStrategy::Int(42)));
    }

    #[test]
    fn test_fillnull_debug() {
        let fillnull = FillNull::new("col", FillStrategy::Zero);
        let debug_str = format!("{:?}", fillnull);
        assert!(debug_str.contains("FillNull"));
    }

    #[test]
    fn test_fillnull_with_int64() {
        use arrow::array::Int64Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));
        let arr = Int64Array::from(vec![Some(1i64), None, Some(3i64), None, Some(5i64)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Int(42));
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert_eq!(col.value(1), 42);
        assert_eq!(col.value(3), 42);
    }

    #[test]
    fn test_fillnull_with_float32() {
        use arrow::array::Float32Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float32,
            true,
        )]));
        let arr = Float32Array::from(vec![Some(1.0f32), None, Some(3.0f32)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Float(2.5));
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert!((col.value(1) - 2.5f32).abs() < 0.001);
    }

    #[test]
    fn test_fillnull_with_bool() {
        use arrow::array::BooleanArray;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "flag",
            DataType::Boolean,
            true,
        )]));
        let arr = BooleanArray::from(vec![Some(true), None, Some(false)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("flag", FillStrategy::Bool(true));
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert!(col.value(1));
    }

    #[test]
    fn test_fillnull_zero_int64() {
        use arrow::array::Int64Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));
        let arr = Int64Array::from(vec![Some(1i64), None, Some(3i64)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Zero);
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert_eq!(col.value(1), 0);
    }

    #[test]
    fn test_fillnull_zero_float32() {
        use arrow::array::Float32Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float32,
            true,
        )]));
        let arr = Float32Array::from(vec![Some(1.0f32), None, Some(3.0f32)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Zero);
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert_eq!(col.value(1), 0.0f32);
    }

    #[test]
    fn test_fillnull_zero_float64() {
        use arrow::array::Float64Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )]));
        let arr = Float64Array::from(vec![Some(1.0f64), None, Some(3.0f64)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Zero);
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert_eq!(col.value(1), 0.0f64);
    }

    #[test]
    fn test_fillnull_forward_fill_int64() {
        use arrow::array::Int64Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));
        let arr = Int64Array::from(vec![Some(1i64), None, None, Some(4i64)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Forward);
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert_eq!(col.value(1), 1);
        assert_eq!(col.value(2), 1);
    }

    #[test]
    fn test_fillnull_backward_fill_int64() {
        use arrow::array::Int64Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));
        let arr = Int64Array::from(vec![Some(1i64), None, None, Some(4i64)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Backward);
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert_eq!(col.value(1), 4);
        assert_eq!(col.value(2), 4);
    }

    #[test]
    fn test_fillnull_unsupported_type_strategy() {
        use arrow::array::Date32Array;

        // Date32 with Int fill strategy is not supported
        let schema = Arc::new(Schema::new(vec![Field::new(
            "date",
            DataType::Date32,
            true,
        )]));
        let arr = Date32Array::from(vec![Some(1000), None, Some(3000)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("date", FillStrategy::Int(42));
        let result = fillnull.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_fillnull_forward_fill_float64() {
        use arrow::array::Float64Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )]));
        let arr = Float64Array::from(vec![Some(1.0f64), None, None, Some(4.0f64), None]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Forward);
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert_eq!(col.value(1), 1.0);
        assert_eq!(col.value(2), 1.0);
        assert_eq!(col.value(4), 4.0);
    }

    #[test]
    fn test_fillnull_backward_fill_float64() {
        use arrow::array::Float64Array;

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            true,
        )]));
        let arr = Float64Array::from(vec![None, Some(2.0f64), None, None, Some(5.0f64)]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(arr)])
            .ok()
            .unwrap_or_else(|| panic!("batch"));

        let fillnull = FillNull::new("value", FillStrategy::Backward);
        let result = fillnull.apply(batch);
        assert!(result.is_ok());
        let result = result.ok().unwrap_or_else(|| panic!("Should succeed"));
        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| panic!("cast"))
            .ok()
            .unwrap_or_else(|| panic!("cast"));
        assert_eq!(col.value(0), 2.0);
        assert_eq!(col.value(2), 5.0);
        assert_eq!(col.value(3), 5.0);
    }

    #[test]
    fn test_fillnull_nonexistent_column() {
        let batch = create_test_batch();
        let fillnull = FillNull::new("nonexistent", FillStrategy::Int(42));
        let result = fillnull.apply(batch);
        // FillNull on nonexistent column returns error
        assert!(result.is_err());
    }
}
