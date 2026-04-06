//! Property-based tests for SafeTensors
//!
//! Verifies mathematical properties and invariants that should hold for all inputs:
//! - Roundtrip property: serialize → deserialize → extract preserves data
//! - Format validation: all valid SafeTensors files parse correctly
//! - Helper API correctness: get_tensor_f32() returns expected values

use std::collections::BTreeMap;

use proptest::prelude::*;
use realizar::safetensors::SafetensorsModel;

const EPSILON: f32 = 1e-6;

/// Type alias for tensor data: (name, data, shape)
type TensorData<'a> = (&'a str, Vec<f32>, Vec<usize>);

/// Strategy for generating valid F32 tensors
fn f32_tensor_strategy() -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1000.0f32..1000.0, 1..=100)
}

/// Strategy for generating valid tensor shapes
fn shape_strategy() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(1usize..=10, 1..=3)
}

/// Create a valid SafeTensors file from tensor data
fn create_safetensors_file(tensors: Vec<TensorData>) -> Vec<u8> {
    let mut metadata = BTreeMap::new();
    let mut offset = 0usize;

    // Build metadata and calculate offsets
    for (name, data, shape) in &tensors {
        let size = data.len() * 4; // F32 = 4 bytes
        metadata.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [offset, offset + size]
            }),
        );
        offset += size;
    }

    let metadata_json = serde_json::to_string(&metadata).expect("Failed to serialize metadata");
    let metadata_bytes = metadata_json.as_bytes();

    let mut file_data = Vec::new();

    // Header: 8-byte metadata length
    file_data.extend_from_slice(&(metadata_bytes.len() as u64).to_le_bytes());

    // Metadata: JSON
    file_data.extend_from_slice(metadata_bytes);

    // Tensor data: F32 values in little-endian
    for (_, data, _) in tensors {
        for &value in &data {
            file_data.extend_from_slice(&value.to_le_bytes());
        }
    }

    file_data
}

proptest! {
    /// Property: Any valid SafeTensors file should parse successfully
    #[test]
    fn test_parse_valid_safetensors(data in f32_tensor_strategy(), shape in shape_strategy()) {
        prop_assume!(!data.is_empty());

        let safetensors = create_safetensors_file(vec![("tensor", data, shape)]);
        let model = SafetensorsModel::from_bytes(&safetensors);

        prop_assert!(model.is_ok());
    }

    /// Property: Roundtrip - extract data matches original
    #[test]
    fn test_roundtrip_data_preservation(data in f32_tensor_strategy()) {
        prop_assume!(!data.is_empty());

        let shape = vec![data.len()];
        let safetensors = create_safetensors_file(vec![("test_tensor", data.clone(), shape)]);

        let model = SafetensorsModel::from_bytes(&safetensors).expect("test");
        let extracted = model.get_tensor_f32("test_tensor").expect("test");

        prop_assert_eq!(extracted.len(), data.len());

        for (original, extracted_val) in data.iter().zip(extracted.iter()) {
            prop_assert!((original - extracted_val).abs() < EPSILON);
        }
    }

    /// Property: get_tensor_f32 returns correct length
    #[test]
    fn test_get_tensor_f32_length(data in f32_tensor_strategy()) {
        prop_assume!(!data.is_empty());

        let shape = vec![data.len()];
        let safetensors = create_safetensors_file(vec![("data", data.clone(), shape)]);

        let model = SafetensorsModel::from_bytes(&safetensors).expect("test");
        let extracted = model.get_tensor_f32("data").expect("test");

        prop_assert_eq!(extracted.len(), data.len());
    }

    /// Property: Multiple tensors can coexist
    #[test]
    fn test_multiple_tensors(data1 in f32_tensor_strategy(), data2 in f32_tensor_strategy()) {
        prop_assume!(!data1.is_empty() && !data2.is_empty());

        let shape1 = vec![data1.len()];
        let shape2 = vec![data2.len()];

        let safetensors = create_safetensors_file(vec![
            ("tensor1", data1.clone(), shape1),
            ("tensor2", data2.clone(), shape2),
        ]);

        let model = SafetensorsModel::from_bytes(&safetensors).expect("test");

        prop_assert_eq!(model.tensors.len(), 2);

        let extracted1 = model.get_tensor_f32("tensor1").expect("test");
        let extracted2 = model.get_tensor_f32("tensor2").expect("test");

        prop_assert_eq!(extracted1.len(), data1.len());
        prop_assert_eq!(extracted2.len(), data2.len());

        for (orig, ext) in data1.iter().zip(extracted1.iter()) {
            prop_assert!((orig - ext).abs() < EPSILON);
        }

        for (orig, ext) in data2.iter().zip(extracted2.iter()) {
            prop_assert!((orig - ext).abs() < EPSILON);
        }
    }

    /// Property: Tensor names are preserved
    #[test]
    fn test_tensor_names_preserved(data in f32_tensor_strategy(), name in "[a-z]{1,20}") {
        prop_assume!(!data.is_empty());
        prop_assume!(!name.is_empty());

        let shape = vec![data.len()];
        let safetensors = create_safetensors_file(vec![(&name, data, shape)]);

        let model = SafetensorsModel::from_bytes(&safetensors).expect("test");

        prop_assert!(model.tensors.contains_key(&name));
        prop_assert_eq!(model.tensors.len(), 1);
    }

    /// Property: get_tensor_f32 on nonexistent tensor returns error
    #[test]
    fn test_nonexistent_tensor_error(data in f32_tensor_strategy()) {
        prop_assume!(!data.is_empty());

        let shape = vec![data.len()];
        let safetensors = create_safetensors_file(vec![("exists", data, shape)]);

        let model = SafetensorsModel::from_bytes(&safetensors).expect("test");
        let result = model.get_tensor_f32("does_not_exist");

        prop_assert!(result.is_err());
    }

    /// Property: Data order is preserved (order matters)
    #[test]
    fn test_data_order_preserved(data in f32_tensor_strategy()) {
        prop_assume!(data.len() >= 2);

        let shape = vec![data.len()];
        let safetensors = create_safetensors_file(vec![("ordered", data.clone(), shape)]);

        let model = SafetensorsModel::from_bytes(&safetensors).expect("test");
        let extracted = model.get_tensor_f32("ordered").expect("test");

        // Check that order is exactly preserved
        for i in 0..data.len() {
            prop_assert!((data[i] - extracted[i]).abs() < EPSILON);
        }
    }
}
