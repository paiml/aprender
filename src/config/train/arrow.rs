//! Arrow array conversion utilities

use crate::error::{Error, Result};

/// Convert Arrow array to f32 vector
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
pub fn arrow_array_to_f32(array: &arrow::array::ArrayRef) -> Result<Vec<f32>> {
    use arrow::array::{Float32Array, Float64Array, Int32Array, Int64Array};
    use arrow::datatypes::DataType;

    match array.data_type() {
        DataType::Float32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| Error::ConfigError("Failed to downcast Float32Array".to_string()))?;
            Ok(arr.values().to_vec())
        }
        DataType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| Error::ConfigError("Failed to downcast Float64Array".to_string()))?;
            Ok(arr.values().iter().map(|&x| x as f32).collect())
        }
        DataType::Int32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| Error::ConfigError("Failed to downcast Int32Array".to_string()))?;
            Ok(arr.values().iter().map(|&x| x as f32).collect())
        }
        DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| Error::ConfigError("Failed to downcast Int64Array".to_string()))?;
            Ok(arr.values().iter().map(|&x| x as f32).collect())
        }
        other => Err(Error::ConfigError(format!(
            "Unsupported Arrow data type: {other:?}. Use Float32, Float64, Int32, or Int64."
        ))),
    }
}
