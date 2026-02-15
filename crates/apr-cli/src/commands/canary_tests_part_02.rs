
// ========================================================================
// Threshold Tests
// ========================================================================

#[test]
fn test_mean_threshold_value() {
    assert_eq!(MEAN_THRESHOLD, 0.1);
}

#[test]
fn test_std_threshold_value() {
    assert_eq!(STD_THRESHOLD, 0.2);
}

// ========================================================================
// Multi-Format load_tensor_data Tests
// ========================================================================

#[test]
fn test_load_tensor_data_gguf() {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

    // Create valid GGUF file with F32 tensor data
    let floats: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let tensor_data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: tensor_data,
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    file.write_all(&gguf_bytes).expect("write GGUF");

    // load_tensor_data should dispatch to GGUF path
    let result = load_tensor_data(file.path());
    assert!(result.is_ok(), "load_tensor_data should work for GGUF");

    let tensor_map = result.unwrap();
    assert_eq!(tensor_map.len(), 1);
    assert!(tensor_map.contains_key("model.weight"));
}

#[test]
fn test_load_tensor_data_safetensors() {
    // Create valid SafeTensors file manually
    let floats: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let tensor_data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

    let header_json = serde_json::json!({
        "test.weight": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 16]
        }
    });
    let header_bytes = serde_json::to_vec(&header_json).expect("serialize header");
    let header_len = header_bytes.len() as u64;

    let mut st_bytes = Vec::new();
    st_bytes.extend_from_slice(&header_len.to_le_bytes());
    st_bytes.extend_from_slice(&header_bytes);
    st_bytes.extend_from_slice(&tensor_data);

    let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    file.write_all(&st_bytes).expect("write SafeTensors");

    // load_tensor_data should dispatch to SafeTensors path
    let result = load_tensor_data(file.path());
    assert!(
        result.is_ok(),
        "load_tensor_data should work for SafeTensors"
    );

    let tensor_map = result.unwrap();
    assert_eq!(tensor_map.len(), 1);
    assert!(tensor_map.contains_key("test.weight"));
}

#[test]
fn test_load_tensor_data_gguf_multiple_tensors() {
    use aprender::format::gguf::reader::GgufReader;
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

    // Create GGUF with multiple tensors (same shape to avoid alignment issues)
    let floats1: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let floats2: Vec<u8> = [5.0f32, 6.0, 7.0, 8.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let tensors = vec![
        GgufTensor {
            name: "model.weight".to_string(),
            shape: vec![2, 2],
            dtype: GgmlType::F32,
            data: floats1,
        },
        GgufTensor {
            name: "model.bias".to_string(),
            shape: vec![2, 2],
            dtype: GgmlType::F32,
            data: floats2,
        },
    ];
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &tensors, &metadata).expect("export GGUF");

    // First verify the GGUF was written correctly
    let reader = GgufReader::from_bytes(gguf_bytes.clone()).expect("parse GGUF");
    assert_eq!(
        reader.tensors.len(),
        2,
        "GGUF should have 2 tensor metadata entries"
    );

    let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    file.write_all(&gguf_bytes).expect("write GGUF");

    let result = load_tensor_data(file.path());
    assert!(result.is_ok(), "load_tensor_data should succeed");

    let tensor_map = result.unwrap();
    // Check that we got at least the first tensor
    assert!(!tensor_map.is_empty(), "tensor_map should not be empty");
    assert!(
        tensor_map.contains_key("model.weight"),
        "Should have model.weight"
    );
}

#[test]
fn test_load_tensor_data_format_detection_by_magic() {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

    // Create GGUF with .bin extension (magic detection, not extension)
    let floats: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let tensor = GgufTensor {
        name: "test.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: floats,
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let mut file = NamedTempFile::with_suffix(".bin").expect("create temp file");
    file.write_all(&gguf_bytes).expect("write GGUF");

    // Should detect GGUF by magic, not extension
    let result = load_tensor_data(file.path());
    assert!(result.is_ok(), "Should detect GGUF by magic bytes");
}
