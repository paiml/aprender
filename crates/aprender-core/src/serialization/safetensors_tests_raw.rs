use crate::serialization::safetensors::{save_safetensors, MappedSafeTensors, SafeTensorsDType};
use std::collections::BTreeMap;
use std::fs;

#[test]
fn test_get_tensor_raw_tensor_not_found() {
    let path = "/tmp/test_get_tensor_raw_not_found.safetensors";

    let mut tensors = BTreeMap::new();
    tensors.insert("weight".to_string(), (vec![1.0f32], vec![1]));
    save_safetensors(path, &tensors).expect("save");

    let mapped = MappedSafeTensors::open(path).expect("open");
    let result = mapped.get_tensor_raw("nonexistent");

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.contains("not found"),
        "Error should mention 'not found': {err}"
    );
    assert!(
        err.contains("nonexistent"),
        "Error should mention tensor name: {err}"
    );

    fs::remove_file(path).ok();
}

#[test]
fn test_get_tensor_raw_f16_dtype() {
    // Build a SafeTensors file with F16 data manually
    let path = "/tmp/test_get_tensor_raw_f16.safetensors";

    // F16 representation of 1.0 = 0x3C00
    let f16_data: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x40]; // two f16 values: 1.0, 2.0

    let metadata_json = serde_json::json!({
        "tensor_f16": {
            "dtype": "F16",
            "shape": [2],
            "data_offsets": [0, 4]
        }
    });
    let metadata_str = serde_json::to_string(&metadata_json).expect("serialize");
    let metadata_bytes = metadata_str.as_bytes();

    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(&(metadata_bytes.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(metadata_bytes);
    file_bytes.extend_from_slice(&f16_data);

    fs::write(path, &file_bytes).expect("write");

    let mapped = MappedSafeTensors::open(path).expect("open");
    let raw = mapped.get_tensor_raw("tensor_f16").expect("get raw");

    assert!(matches!(raw.dtype, SafeTensorsDType::F16));
    assert_eq!(raw.shape, vec![2]);
    assert_eq!(raw.bytes.len(), 4);

    fs::remove_file(path).ok();
}

#[test]
fn test_get_tensor_raw_bf16_dtype() {
    let path = "/tmp/test_get_tensor_raw_bf16.safetensors";

    // BF16 representation of 1.0 = 0x3F80
    let bf16_data: Vec<u8> = vec![0x80, 0x3F]; // one BF16 value

    let metadata_json = serde_json::json!({
        "tensor_bf16": {
            "dtype": "BF16",
            "shape": [1],
            "data_offsets": [0, 2]
        }
    });
    let metadata_str = serde_json::to_string(&metadata_json).expect("serialize");
    let metadata_bytes = metadata_str.as_bytes();

    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(&(metadata_bytes.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(metadata_bytes);
    file_bytes.extend_from_slice(&bf16_data);

    fs::write(path, &file_bytes).expect("write");

    let mapped = MappedSafeTensors::open(path).expect("open");
    let raw = mapped.get_tensor_raw("tensor_bf16").expect("get raw");

    assert!(matches!(raw.dtype, SafeTensorsDType::BF16));
    assert_eq!(raw.shape, vec![1]);
    assert_eq!(raw.bytes.len(), 2);

    fs::remove_file(path).ok();
}

#[test]
fn test_get_tensor_raw_unsupported_dtype() {
    let path = "/tmp/test_get_tensor_raw_unsupported.safetensors";

    // Create a SafeTensors file with an unsupported dtype
    let data = vec![0u8; 8];

    let metadata_json = serde_json::json!({
        "tensor_i32": {
            "dtype": "I32",
            "shape": [2],
            "data_offsets": [0, 8]
        }
    });
    let metadata_str = serde_json::to_string(&metadata_json).expect("serialize");
    let metadata_bytes = metadata_str.as_bytes();

    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(&(metadata_bytes.len() as u64).to_le_bytes());
    file_bytes.extend_from_slice(metadata_bytes);
    file_bytes.extend_from_slice(&data);

    fs::write(path, &file_bytes).expect("write");

    let mapped = MappedSafeTensors::open(path).expect("open");
    let result = mapped.get_tensor_raw("tensor_i32");

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.contains("Unsupported dtype"),
        "Error should mention unsupported: {err}"
    );

    fs::remove_file(path).ok();
}

#[test]
fn test_get_tensor_raw_multiple_tensors() {
    let path = "/tmp/test_get_tensor_raw_multi.safetensors";

    let mut tensors = BTreeMap::new();
    tensors.insert("alpha".to_string(), (vec![1.0f32, 2.0], vec![2]));
    tensors.insert("beta".to_string(), (vec![3.0f32, 4.0, 5.0], vec![3]));
    save_safetensors(path, &tensors).expect("save");

    let mapped = MappedSafeTensors::open(path).expect("open");

    let alpha_raw = mapped.get_tensor_raw("alpha").expect("get alpha");
    assert!(matches!(alpha_raw.dtype, SafeTensorsDType::F32));
    assert_eq!(alpha_raw.shape, vec![2]);
    let alpha_f32 = alpha_raw.to_f32().expect("convert");
    assert_eq!(alpha_f32, vec![1.0, 2.0]);

    let beta_raw = mapped.get_tensor_raw("beta").expect("get beta");
    assert!(matches!(beta_raw.dtype, SafeTensorsDType::F32));
    assert_eq!(beta_raw.shape, vec![3]);
    let beta_f32 = beta_raw.to_f32().expect("convert");
    assert_eq!(beta_f32, vec![3.0, 4.0, 5.0]);

    fs::remove_file(path).ok();
}
