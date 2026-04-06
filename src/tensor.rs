//! Tensor conversion utilities for ML framework integration.
//!
//! Provides utilities for converting Arrow data to tensor-friendly formats
//! suitable for ML training. This module enables efficient zero-copy or
//! minimal-copy data transfer to ML frameworks.
//!
//! # Example
//!
//! ```
//! use alimentar::{ArrowDataset, Dataset};
//! use alimentar::tensor::{TensorData, TensorExtractor};
//!
//! # fn main() -> alimentar::Result<()> {
//! # use std::sync::Arc;
//! # use arrow::{array::{Int32Array, Float64Array}, datatypes::{DataType, Field, Schema}, record_batch::RecordBatch};
//! # let schema = Arc::new(Schema::new(vec![
//! #     Field::new("x", DataType::Float64, false),
//! #     Field::new("y", DataType::Float64, false),
//! # ]));
//! # let batch = RecordBatch::try_new(
//! #     schema,
//! #     vec![
//! #         Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])),
//! #         Arc::new(Float64Array::from(vec![4.0, 5.0, 6.0])),
//! #     ],
//! # )?;
//! # let dataset = ArrowDataset::from_batch(batch)?;
//! // Extract features as f32 tensor data
//! let extractor = TensorExtractor::new(&["x", "y"]);
//! let tensor_data = extractor.extract_f32(dataset.get_batch(0).unwrap())?;
//!
//! println!("Shape: {:?}", tensor_data.shape());
//! println!("Data: {:?}", tensor_data.as_slice());
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use arrow::{
    array::{
        Array, AsArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
        UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    },
    datatypes::DataType,
    record_batch::RecordBatch,
};

use crate::error::{Error, Result};

/// Tensor data in a contiguous memory layout.
///
/// This struct holds tensor data in row-major (C-style) order,
/// suitable for direct transfer to ML frameworks.
#[derive(Debug, Clone)]
pub struct TensorData<T> {
    /// The underlying data buffer
    data: Vec<T>,
    /// Shape of the tensor [rows, cols]
    shape: [usize; 2],
}

impl<T: Clone + Default> TensorData<T> {
    /// Creates a new tensor with the given shape, filled with default values.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![T::default(); rows * cols],
            shape: [rows, cols],
        }
    }

    /// Creates a tensor from existing data and shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the data length doesn't match rows * cols.
    ///
    /// #[requires(data.len() == rows * cols)]
    /// #[ensures(result.is_ok() ==> result.shape() == [rows, cols])]
    /// #[ensures(result.is_ok() ==> result.as_slice().len() == rows * cols)]
    /// #[invariant(self.data.len() == self.shape[0] * self.shape[1])]
    pub fn from_vec(data: Vec<T>, rows: usize, cols: usize) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(Error::data(format!(
                "Data length {} doesn't match shape [{}, {}]",
                data.len(),
                rows,
                cols
            )));
        }
        Ok(Self {
            data,
            shape: [rows, cols],
        })
    }

    /// Returns the shape of the tensor as [rows, cols].
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }

    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    /// Returns the number of columns.
    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    /// Returns the underlying data as a slice.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns the underlying data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consumes the tensor and returns the underlying data.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Returns a raw pointer to the underlying data.
    ///
    /// Useful for FFI integration with ML frameworks.
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Gets an element at the given row and column.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.shape[0] && col < self.shape[1] {
            Some(&self.data[row * self.shape[1] + col])
        } else {
            None
        }
    }

    /// Sets an element at the given row and column.
    ///
    /// # Panics
    ///
    /// Panics if the indices are out of bounds.
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.shape[0] && col < self.shape[1]);
        self.data[row * self.shape[1] + col] = value;
    }
}

/// Extracts tensor data from Arrow RecordBatches.
///
/// This struct configures which columns to extract and how to convert them.
#[derive(Debug, Clone)]
pub struct TensorExtractor {
    /// Column names to extract
    columns: Vec<String>,
}

impl TensorExtractor {
    /// Creates a new extractor for the specified columns.
    pub fn new(columns: &[&str]) -> Self {
        Self {
            columns: columns.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    /// Creates an extractor from owned column names.
    pub fn from_columns(columns: Vec<String>) -> Self {
        Self { columns }
    }

    /// Returns the column names being extracted.
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// Extracts data as f32 tensor.
    ///
    /// Numeric columns are converted to f32. Non-numeric columns cause an
    /// error.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A requested column doesn't exist
    /// - A column contains non-numeric data
    pub fn extract_f32(&self, batch: &RecordBatch) -> Result<TensorData<f32>> {
        let rows = batch.num_rows();
        let cols = self.columns.len();

        let mut data = vec![0.0f32; rows * cols];

        for (col_idx, col_name) in self.columns.iter().enumerate() {
            let col_index = batch
                .schema()
                .index_of(col_name)
                .map_err(|_| Error::column_not_found(col_name))?;

            let array = batch.column(col_index);
            Self::extract_column_f32(array, &mut data, col_idx, cols, rows)?;
        }

        TensorData::from_vec(data, rows, cols)
    }

    /// Extracts data as f64 tensor.
    ///
    /// Numeric columns are converted to f64. Non-numeric columns cause an
    /// error.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A requested column doesn't exist
    /// - A column contains non-numeric data
    pub fn extract_f64(&self, batch: &RecordBatch) -> Result<TensorData<f64>> {
        let rows = batch.num_rows();
        let cols = self.columns.len();

        let mut data = vec![0.0f64; rows * cols];

        for (col_idx, col_name) in self.columns.iter().enumerate() {
            let col_index = batch
                .schema()
                .index_of(col_name)
                .map_err(|_| Error::column_not_found(col_name))?;

            let array = batch.column(col_index);
            Self::extract_column_f64(array, &mut data, col_idx, cols, rows)?;
        }

        TensorData::from_vec(data, rows, cols)
    }

    /// Extracts data as i64 tensor.
    ///
    /// Integer columns are converted to i64. Non-integer columns cause an
    /// error.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A requested column doesn't exist
    /// - A column contains non-integer data
    pub fn extract_i64(&self, batch: &RecordBatch) -> Result<TensorData<i64>> {
        let rows = batch.num_rows();
        let cols = self.columns.len();

        let mut data = vec![0i64; rows * cols];

        for (col_idx, col_name) in self.columns.iter().enumerate() {
            let col_index = batch
                .schema()
                .index_of(col_name)
                .map_err(|_| Error::column_not_found(col_name))?;

            let array = batch.column(col_index);
            Self::extract_column_i64(array, &mut data, col_idx, cols, rows)?;
        }

        TensorData::from_vec(data, rows, cols)
    }

    fn extract_column_f32(
        array: &Arc<dyn Array>,
        data: &mut [f32],
        col_idx: usize,
        num_cols: usize,
        num_rows: usize,
    ) -> Result<()> {
        match array.data_type() {
            DataType::Float32 => {
                let arr = array.as_primitive::<arrow::datatypes::Float32Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = arr.value(row);
                }
            }
            DataType::Float64 => {
                let arr = array.as_primitive::<arrow::datatypes::Float64Type>();
                for row in 0..num_rows {
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        data[row * num_cols + col_idx] = arr.value(row) as f32;
                    }
                }
            }
            DataType::Int8 => {
                let arr = array.as_primitive::<arrow::datatypes::Int8Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = f32::from(arr.value(row));
                }
            }
            DataType::Int16 => {
                let arr = array.as_primitive::<arrow::datatypes::Int16Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = f32::from(arr.value(row));
                }
            }
            DataType::Int32 => {
                let arr = array.as_primitive::<arrow::datatypes::Int32Type>();
                for row in 0..num_rows {
                    #[allow(clippy::cast_precision_loss)]
                    {
                        data[row * num_cols + col_idx] = arr.value(row) as f32;
                    }
                }
            }
            DataType::Int64 => {
                let arr = array.as_primitive::<arrow::datatypes::Int64Type>();
                for row in 0..num_rows {
                    #[allow(clippy::cast_precision_loss)]
                    {
                        data[row * num_cols + col_idx] = arr.value(row) as f32;
                    }
                }
            }
            DataType::UInt8 => {
                let arr = array.as_primitive::<arrow::datatypes::UInt8Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = f32::from(arr.value(row));
                }
            }
            DataType::UInt16 => {
                let arr = array.as_primitive::<arrow::datatypes::UInt16Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = f32::from(arr.value(row));
                }
            }
            DataType::UInt32 => {
                let arr = array.as_primitive::<arrow::datatypes::UInt32Type>();
                for row in 0..num_rows {
                    #[allow(clippy::cast_precision_loss)]
                    {
                        data[row * num_cols + col_idx] = arr.value(row) as f32;
                    }
                }
            }
            DataType::UInt64 => {
                let arr = array.as_primitive::<arrow::datatypes::UInt64Type>();
                for row in 0..num_rows {
                    #[allow(clippy::cast_precision_loss)]
                    {
                        data[row * num_cols + col_idx] = arr.value(row) as f32;
                    }
                }
            }
            dt => {
                return Err(Error::data(format!(
                    "Cannot convert {:?} to f32 tensor",
                    dt
                )));
            }
        }
        Ok(())
    }

    fn extract_column_f64(
        array: &Arc<dyn Array>,
        data: &mut [f64],
        col_idx: usize,
        num_cols: usize,
        num_rows: usize,
    ) -> Result<()> {
        match array.data_type() {
            DataType::Float32 => {
                let arr = array.as_primitive::<arrow::datatypes::Float32Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = f64::from(arr.value(row));
                }
            }
            DataType::Float64 => {
                let arr = array.as_primitive::<arrow::datatypes::Float64Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = arr.value(row);
                }
            }
            DataType::Int8 => {
                let arr = array.as_primitive::<arrow::datatypes::Int8Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = f64::from(arr.value(row));
                }
            }
            DataType::Int16 => {
                let arr = array.as_primitive::<arrow::datatypes::Int16Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = f64::from(arr.value(row));
                }
            }
            DataType::Int32 => {
                let arr = array.as_primitive::<arrow::datatypes::Int32Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = f64::from(arr.value(row));
                }
            }
            DataType::Int64 => {
                let arr = array.as_primitive::<arrow::datatypes::Int64Type>();
                for row in 0..num_rows {
                    #[allow(clippy::cast_precision_loss)]
                    {
                        data[row * num_cols + col_idx] = arr.value(row) as f64;
                    }
                }
            }
            DataType::UInt8 => {
                let arr = array.as_primitive::<arrow::datatypes::UInt8Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = f64::from(arr.value(row));
                }
            }
            DataType::UInt16 => {
                let arr = array.as_primitive::<arrow::datatypes::UInt16Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = f64::from(arr.value(row));
                }
            }
            DataType::UInt32 => {
                let arr = array.as_primitive::<arrow::datatypes::UInt32Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = f64::from(arr.value(row));
                }
            }
            DataType::UInt64 => {
                let arr = array.as_primitive::<arrow::datatypes::UInt64Type>();
                for row in 0..num_rows {
                    #[allow(clippy::cast_precision_loss)]
                    {
                        data[row * num_cols + col_idx] = arr.value(row) as f64;
                    }
                }
            }
            dt => {
                return Err(Error::data(format!(
                    "Cannot convert {:?} to f64 tensor",
                    dt
                )));
            }
        }
        Ok(())
    }

    fn extract_column_i64(
        array: &Arc<dyn Array>,
        data: &mut [i64],
        col_idx: usize,
        num_cols: usize,
        num_rows: usize,
    ) -> Result<()> {
        match array.data_type() {
            DataType::Int8 => {
                let arr = array.as_primitive::<arrow::datatypes::Int8Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = i64::from(arr.value(row));
                }
            }
            DataType::Int16 => {
                let arr = array.as_primitive::<arrow::datatypes::Int16Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = i64::from(arr.value(row));
                }
            }
            DataType::Int32 => {
                let arr = array.as_primitive::<arrow::datatypes::Int32Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = i64::from(arr.value(row));
                }
            }
            DataType::Int64 => {
                let arr = array.as_primitive::<arrow::datatypes::Int64Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = arr.value(row);
                }
            }
            DataType::UInt8 => {
                let arr = array.as_primitive::<arrow::datatypes::UInt8Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = i64::from(arr.value(row));
                }
            }
            DataType::UInt16 => {
                let arr = array.as_primitive::<arrow::datatypes::UInt16Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = i64::from(arr.value(row));
                }
            }
            DataType::UInt32 => {
                let arr = array.as_primitive::<arrow::datatypes::UInt32Type>();
                for row in 0..num_rows {
                    data[row * num_cols + col_idx] = i64::from(arr.value(row));
                }
            }
            DataType::UInt64 => {
                let arr = array.as_primitive::<arrow::datatypes::UInt64Type>();
                for row in 0..num_rows {
                    #[allow(clippy::cast_possible_wrap)]
                    {
                        data[row * num_cols + col_idx] = arr.value(row) as i64;
                    }
                }
            }
            dt => {
                return Err(Error::data(format!(
                    "Cannot convert {:?} to i64 tensor",
                    dt
                )));
            }
        }
        Ok(())
    }
}

/// Extracts a single numeric column as a 1D vector.
///
/// # Errors
///
/// Returns an error if the column doesn't exist or is non-numeric.
pub fn extract_column_f32(batch: &RecordBatch, column: &str) -> Result<Vec<f32>> {
    let extractor = TensorExtractor::new(&[column]);
    let tensor = extractor.extract_f32(batch)?;
    Ok(tensor.into_vec())
}

/// Extracts a single numeric column as a 1D vector.
///
/// # Errors
///
/// Returns an error if the column doesn't exist or is non-numeric.
pub fn extract_column_f64(batch: &RecordBatch, column: &str) -> Result<Vec<f64>> {
    let extractor = TensorExtractor::new(&[column]);
    let tensor = extractor.extract_f64(batch)?;
    Ok(tensor.into_vec())
}

/// Extracts label column as integer indices.
///
/// String labels are converted to indices based on unique values.
///
/// # Errors
///
/// Returns an error if the column doesn't exist.
pub fn extract_labels_i64(batch: &RecordBatch, column: &str) -> Result<Vec<i64>> {
    let col_index = batch
        .schema()
        .index_of(column)
        .map_err(|_| Error::column_not_found(column))?;

    let array = batch.column(col_index);

    match array.data_type() {
        DataType::Int8 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int8Array>()
                .ok_or_else(|| Error::data("Failed to downcast to Int8Array"))?;
            Ok(arr.iter().map(|v| i64::from(v.unwrap_or(0))).collect())
        }
        DataType::Int16 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int16Array>()
                .ok_or_else(|| Error::data("Failed to downcast to Int16Array"))?;
            Ok(arr.iter().map(|v| i64::from(v.unwrap_or(0))).collect())
        }
        DataType::Int32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| Error::data("Failed to downcast to Int32Array"))?;
            Ok(arr.iter().map(|v| i64::from(v.unwrap_or(0))).collect())
        }
        DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| Error::data("Failed to downcast to Int64Array"))?;
            Ok(arr.iter().map(|v| v.unwrap_or(0)).collect())
        }
        DataType::UInt8 => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| Error::data("Failed to downcast to UInt8Array"))?;
            Ok(arr.iter().map(|v| i64::from(v.unwrap_or(0))).collect())
        }
        DataType::UInt16 => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt16Array>()
                .ok_or_else(|| Error::data("Failed to downcast to UInt16Array"))?;
            Ok(arr.iter().map(|v| i64::from(v.unwrap_or(0))).collect())
        }
        DataType::UInt32 => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| Error::data("Failed to downcast to UInt32Array"))?;
            Ok(arr.iter().map(|v| i64::from(v.unwrap_or(0))).collect())
        }
        DataType::UInt64 => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| Error::data("Failed to downcast to UInt64Array"))?;
            #[allow(clippy::cast_possible_wrap)]
            Ok(arr.iter().map(|v| v.unwrap_or(0) as i64).collect())
        }
        DataType::Float32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| Error::data("Failed to downcast to Float32Array"))?;
            #[allow(clippy::cast_possible_truncation)]
            Ok(arr.iter().map(|v| v.unwrap_or(0.0) as i64).collect())
        }
        DataType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| Error::data("Failed to downcast to Float64Array"))?;
            #[allow(clippy::cast_possible_truncation)]
            Ok(arr.iter().map(|v| v.unwrap_or(0.0) as i64).collect())
        }
        dt => Err(Error::data(format!("Cannot extract labels from {:?}", dt))),
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::uninlined_format_args,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp
)]
mod tests {
    use arrow::datatypes::{Field, Schema};

    use super::*;

    fn create_numeric_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("f32_col", DataType::Float32, false),
            Field::new("f64_col", DataType::Float64, false),
            Field::new("i32_col", DataType::Int32, false),
            Field::new("i64_col", DataType::Int64, false),
        ]));

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Float32Array::from(vec![1.0f32, 2.0, 3.0])),
                Arc::new(Float64Array::from(vec![4.0f64, 5.0, 6.0])),
                Arc::new(Int32Array::from(vec![7, 8, 9])),
                Arc::new(Int64Array::from(vec![10i64, 11, 12])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_tensor_data_new() {
        let tensor: TensorData<f32> = TensorData::new(3, 4);
        assert_eq!(tensor.shape(), [3, 4]);
        assert_eq!(tensor.rows(), 3);
        assert_eq!(tensor.cols(), 4);
        assert_eq!(tensor.as_slice().len(), 12);
    }

    #[test]
    fn test_tensor_data_from_vec() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = TensorData::from_vec(data, 2, 3).unwrap();
        assert_eq!(tensor.shape(), [2, 3]);
        assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_tensor_data_from_vec_invalid_shape() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let result = TensorData::from_vec(data, 2, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_data_get_set() {
        let mut tensor = TensorData::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2).unwrap();

        assert_eq!(tensor.get(0, 0), Some(&1.0f32));
        assert_eq!(tensor.get(0, 1), Some(&2.0f32));
        assert_eq!(tensor.get(1, 0), Some(&3.0f32));
        assert_eq!(tensor.get(1, 1), Some(&4.0f32));
        assert_eq!(tensor.get(2, 0), None);

        tensor.set(0, 1, 99.0);
        assert_eq!(tensor.get(0, 1), Some(&99.0f32));
    }

    #[test]
    fn test_tensor_data_into_vec() {
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = TensorData::from_vec(data.clone(), 1, 3).unwrap();
        assert_eq!(tensor.into_vec(), data);
    }

    #[test]
    fn test_tensor_data_as_ptr() {
        let tensor = TensorData::from_vec(vec![1.0f32, 2.0, 3.0], 1, 3).unwrap();
        let ptr = tensor.as_ptr();
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_tensor_data_as_mut_slice() {
        let mut tensor = TensorData::from_vec(vec![1.0f32, 2.0, 3.0], 1, 3).unwrap();
        let slice = tensor.as_mut_slice();
        slice[0] = 10.0;
        assert_eq!(tensor.as_slice()[0], 10.0);
    }

    #[test]
    fn test_tensor_data_clone() {
        let tensor = TensorData::from_vec(vec![1.0f32, 2.0, 3.0], 1, 3).unwrap();
        let cloned = tensor.clone();
        assert_eq!(cloned.shape(), tensor.shape());
        assert_eq!(cloned.as_slice(), tensor.as_slice());
    }

    #[test]
    fn test_tensor_data_debug() {
        let tensor = TensorData::from_vec(vec![1.0f32], 1, 1).unwrap();
        let debug = format!("{:?}", tensor);
        assert!(debug.contains("TensorData"));
    }

    #[test]
    fn test_extractor_new() {
        let extractor = TensorExtractor::new(&["a", "b", "c"]);
        assert_eq!(extractor.columns().len(), 3);
        assert_eq!(extractor.columns()[0], "a");
    }

    #[test]
    fn test_extractor_from_columns() {
        let extractor = TensorExtractor::from_columns(vec!["x".to_string(), "y".to_string()]);
        assert_eq!(extractor.columns().len(), 2);
    }

    #[test]
    fn test_extractor_clone() {
        let extractor = TensorExtractor::new(&["a", "b"]);
        let cloned = extractor.clone();
        assert_eq!(cloned.columns(), extractor.columns());
    }

    #[test]
    fn test_extractor_debug() {
        let extractor = TensorExtractor::new(&["col"]);
        let debug = format!("{:?}", extractor);
        assert!(debug.contains("TensorExtractor"));
    }

    #[test]
    fn test_extract_f32() {
        let batch = create_numeric_batch();
        let extractor = TensorExtractor::new(&["f32_col", "i32_col"]);
        let tensor = extractor.extract_f32(&batch).unwrap();

        assert_eq!(tensor.shape(), [3, 2]);
        assert_eq!(tensor.get(0, 0), Some(&1.0f32));
        assert_eq!(tensor.get(0, 1), Some(&7.0f32));
        assert_eq!(tensor.get(2, 0), Some(&3.0f32));
        assert_eq!(tensor.get(2, 1), Some(&9.0f32));
    }

    #[test]
    fn test_extract_f64() {
        let batch = create_numeric_batch();
        let extractor = TensorExtractor::new(&["f64_col", "i64_col"]);
        let tensor = extractor.extract_f64(&batch).unwrap();

        assert_eq!(tensor.shape(), [3, 2]);
        assert_eq!(tensor.get(0, 0), Some(&4.0f64));
        assert_eq!(tensor.get(0, 1), Some(&10.0f64));
    }

    #[test]
    fn test_extract_i64() {
        let batch = create_numeric_batch();
        let extractor = TensorExtractor::new(&["i32_col", "i64_col"]);
        let tensor = extractor.extract_i64(&batch).unwrap();

        assert_eq!(tensor.shape(), [3, 2]);
        assert_eq!(tensor.get(0, 0), Some(&7i64));
        assert_eq!(tensor.get(0, 1), Some(&10i64));
    }

    #[test]
    fn test_extract_column_not_found() {
        let batch = create_numeric_batch();
        let extractor = TensorExtractor::new(&["nonexistent"]);
        let result = extractor.extract_f32(&batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_column_f32_helper() {
        let batch = create_numeric_batch();
        let data = extract_column_f32(&batch, "f32_col").unwrap();
        assert_eq!(data, vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn test_extract_column_f64_helper() {
        let batch = create_numeric_batch();
        let data = extract_column_f64(&batch, "f64_col").unwrap();
        assert_eq!(data, vec![4.0f64, 5.0, 6.0]);
    }

    #[test]
    fn test_extract_labels_i64() {
        let batch = create_numeric_batch();
        let labels = extract_labels_i64(&batch, "i32_col").unwrap();
        assert_eq!(labels, vec![7i64, 8, 9]);
    }

    #[test]
    fn test_extract_labels_i64_from_float() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "label",
            DataType::Float64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0]))],
        )
        .unwrap();

        let labels = extract_labels_i64(&batch, "label").unwrap();
        assert_eq!(labels, vec![0i64, 1, 2]);
    }

    #[test]
    fn test_extract_labels_column_not_found() {
        let batch = create_numeric_batch();
        let result = extract_labels_i64(&batch, "nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_all_int_types() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i8", DataType::Int8, false),
            Field::new("i16", DataType::Int16, false),
            Field::new("u8", DataType::UInt8, false),
            Field::new("u16", DataType::UInt16, false),
            Field::new("u32", DataType::UInt32, false),
            Field::new("u64", DataType::UInt64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int8Array::from(vec![1i8])),
                Arc::new(Int16Array::from(vec![2i16])),
                Arc::new(UInt8Array::from(vec![3u8])),
                Arc::new(UInt16Array::from(vec![4u16])),
                Arc::new(UInt32Array::from(vec![5u32])),
                Arc::new(UInt64Array::from(vec![6u64])),
            ],
        )
        .unwrap();

        // Test f32 extraction
        let extractor = TensorExtractor::new(&["i8", "i16", "u8", "u16", "u32", "u64"]);
        let tensor = extractor.extract_f32(&batch).unwrap();
        assert_eq!(tensor.as_slice(), &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Test f64 extraction
        let tensor = extractor.extract_f64(&batch).unwrap();
        assert_eq!(tensor.as_slice(), &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Test i64 extraction
        let tensor = extractor.extract_i64(&batch).unwrap();
        assert_eq!(tensor.as_slice(), &[1i64, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_extract_f32_from_f64() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float64Array::from(vec![1.5f64, 2.5, 3.5]))],
        )
        .unwrap();

        let extractor = TensorExtractor::new(&["value"]);
        let tensor = extractor.extract_f32(&batch).unwrap();
        assert_eq!(tensor.as_slice(), &[1.5f32, 2.5, 3.5]);
    }

    #[test]
    fn test_extract_f64_from_f32() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Float32Array::from(vec![1.5f32, 2.5, 3.5]))],
        )
        .unwrap();

        let extractor = TensorExtractor::new(&["value"]);
        let tensor = extractor.extract_f64(&batch).unwrap();
        // f32 -> f64 conversion is exact for these values
        assert_eq!(tensor.as_slice(), &[1.5f64, 2.5, 3.5]);
    }

    #[test]
    fn test_extract_unsupported_type_f32() {
        use arrow::array::StringArray;

        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["hello", "world"]))],
        )
        .unwrap();

        let extractor = TensorExtractor::new(&["text"]);
        let result = extractor.extract_f32(&batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_unsupported_type_f64() {
        use arrow::array::StringArray;

        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["hello", "world"]))],
        )
        .unwrap();

        let extractor = TensorExtractor::new(&["text"]);
        let result = extractor.extract_f64(&batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_unsupported_type_i64() {
        use arrow::array::StringArray;

        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["hello", "world"]))],
        )
        .unwrap();

        let extractor = TensorExtractor::new(&["text"]);
        let result = extractor.extract_i64(&batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_labels_unsupported_type() {
        use arrow::array::StringArray;

        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["hello", "world"]))],
        )
        .unwrap();

        let result = extract_labels_i64(&batch, "text");
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_labels_all_uint_types() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("u8", DataType::UInt8, false),
            Field::new("u16", DataType::UInt16, false),
            Field::new("u32", DataType::UInt32, false),
            Field::new("u64", DataType::UInt64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt8Array::from(vec![1u8])),
                Arc::new(UInt16Array::from(vec![2u16])),
                Arc::new(UInt32Array::from(vec![3u32])),
                Arc::new(UInt64Array::from(vec![4u64])),
            ],
        )
        .unwrap();

        assert_eq!(extract_labels_i64(&batch, "u8").unwrap(), vec![1i64]);
        assert_eq!(extract_labels_i64(&batch, "u16").unwrap(), vec![2i64]);
        assert_eq!(extract_labels_i64(&batch, "u32").unwrap(), vec![3i64]);
        assert_eq!(extract_labels_i64(&batch, "u64").unwrap(), vec![4i64]);
    }

    #[test]
    fn test_extract_labels_all_int_types() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i8", DataType::Int8, false),
            Field::new("i16", DataType::Int16, false),
            Field::new("f32", DataType::Float32, false),
        ]));

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int8Array::from(vec![1i8])),
                Arc::new(Int16Array::from(vec![2i16])),
                Arc::new(Float32Array::from(vec![3.0f32])),
            ],
        )
        .unwrap();

        assert_eq!(extract_labels_i64(&batch, "i8").unwrap(), vec![1i64]);
        assert_eq!(extract_labels_i64(&batch, "i16").unwrap(), vec![2i64]);
        assert_eq!(extract_labels_i64(&batch, "f32").unwrap(), vec![3i64]);
    }
}
