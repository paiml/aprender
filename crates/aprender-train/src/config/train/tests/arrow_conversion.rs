//! Tests for Arrow array to f32 conversion (non-WASM only)

use crate::config::train::arrow_array_to_f32;

#[test]
fn test_arrow_array_to_f32_float32() {
    use ::arrow::array::Float32Array;
    let array: ::arrow::array::ArrayRef =
        std::sync::Arc::new(Float32Array::from(vec![1.0f32, 2.0, 3.0]));
    let result = arrow_array_to_f32(&array).expect("operation should succeed");
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_arrow_array_to_f32_float64() {
    use ::arrow::array::Float64Array;
    let array: ::arrow::array::ArrayRef =
        std::sync::Arc::new(Float64Array::from(vec![1.0f64, 2.0, 3.0]));
    let result = arrow_array_to_f32(&array).expect("operation should succeed");
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_arrow_array_to_f32_int32() {
    use ::arrow::array::Int32Array;
    let array: ::arrow::array::ArrayRef = std::sync::Arc::new(Int32Array::from(vec![1i32, 2, 3]));
    let result = arrow_array_to_f32(&array).expect("operation should succeed");
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_arrow_array_to_f32_int64() {
    use ::arrow::array::Int64Array;
    let array: ::arrow::array::ArrayRef = std::sync::Arc::new(Int64Array::from(vec![1i64, 2, 3]));
    let result = arrow_array_to_f32(&array).expect("operation should succeed");
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_arrow_array_to_f32_unsupported_type() {
    use ::arrow::array::StringArray;
    let array: ::arrow::array::ArrayRef = std::sync::Arc::new(StringArray::from(vec!["a", "b"]));
    let result = arrow_array_to_f32(&array);
    assert!(result.is_err());
}
