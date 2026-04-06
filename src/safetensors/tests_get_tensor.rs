
// ============================================================================
// SafetensorsModel::get_tensor_f32 boundary cases
// ============================================================================

#[test]
fn test_get_tensor_f32_exact_boundary() {
    // Tensor data exactly fills the data region
    let json = r#"{"w":{"dtype":"F32","shape":[3],"data_offsets":[0,12]}}"#;
    let json_bytes = json.as_bytes();
    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&2.0f32.to_le_bytes());
    data.extend_from_slice(&3.0f32.to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("parse");
    let values = model.get_tensor_f32("w").expect("get");
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_get_tensor_f32_special_values() {
    let json = r#"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
    let json_bytes = json.as_bytes();
    let mut data = Vec::new();
    data.extend_from_slice(&(json_bytes.len() as u64).to_le_bytes());
    data.extend_from_slice(json_bytes);
    data.extend_from_slice(&f32::INFINITY.to_le_bytes());
    data.extend_from_slice(&f32::NEG_INFINITY.to_le_bytes());
    data.extend_from_slice(&f32::NAN.to_le_bytes());
    data.extend_from_slice(&0.0f32.to_le_bytes());

    let model = SafetensorsModel::from_bytes(&data).expect("parse");
    let values = model.get_tensor_f32("w").expect("get");
    assert!(values[0].is_infinite() && values[0] > 0.0);
    assert!(values[1].is_infinite() && values[1] < 0.0);
    assert!(values[2].is_nan());
    assert_eq!(values[3], 0.0);
}

// ============================================================================
// SafetensorsConfig: deserialization from raw JSON
// ============================================================================

#[test]
fn test_config_deserialize_minimal() {
    let config: SafetensorsConfig = serde_json::from_str("{}").expect("parse");
    assert!(config.hidden_size.is_none());
    assert!(config.num_hidden_layers.is_none());
    assert!(config.vocab_size.is_none());
    assert_eq!(config.num_kv_heads(), 1); // all None -> default 1
    assert_eq!(config.architecture(), "unknown");
}

#[test]
fn test_config_deserialize_extra_fields_ignored() {
    let json = r#"{
        "hidden_size": 768,
        "unknown_field": "some_value",
        "another_unknown": 42
    }"#;
    let config: SafetensorsConfig = serde_json::from_str(json).expect("parse");
    assert_eq!(config.hidden_size, Some(768));
}

// ============================================================================
// MappedSafeTensorsModel: GH-213 truncated file detection
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
mod mapped_tests_part_02 {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_mapped_truncated_tensor_data_detected_at_load() {
        // GH-213: File has valid header+metadata but tensor data is truncated
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let json = r#"{"weight":{"dtype":"F32","shape":[1000],"data_offsets":[0,4000]}}"#;
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("header");
        file.write_all(json.as_bytes()).expect("metadata");
        // Only write 100 bytes of tensor data instead of 4000
        file.write_all(&[0u8; 100]).expect("partial data");
        file.flush().expect("flush");

        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(
            err.contains("truncated"),
            "Expected truncation error, got: {}",
            err
        );
    }

    #[test]
    fn test_mapped_empty_file() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(&[]).expect("write nothing");
        file.flush().expect("flush");

        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("too small") || err.contains("File too small"));
    }

    #[test]
    fn test_mapped_exactly_8_bytes_zero_metadata() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(&0u64.to_le_bytes()).expect("header");
        file.flush().expect("flush");

        let result = MappedSafeTensorsModel::load(file.path());
        // metadata_len=0 means json_bytes is empty -> parse error
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_valid_no_tensors() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let json = b"{}";
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("header");
        file.write_all(json).expect("metadata");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        assert_eq!(model.tensor_count(), 0);
        assert!(model.tensor_names().is_empty());
        assert_eq!(model.file_size(), 8 + 2); // 8-byte header + 2-byte JSON
    }

    #[test]
    fn test_mapped_debug_format() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let json = r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("header");
        file.write_all(json.as_bytes()).expect("metadata");
        file.write_all(&1.0f32.to_le_bytes()).expect("data");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let debug = format!("{:?}", model);
        assert!(debug.contains("MappedSafeTensorsModel"));
    }

    #[test]
    fn test_mapped_has_tensor_and_get_info_consistency() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let json = r#"{
            "a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},
            "b":{"dtype":"F16","shape":[4],"data_offsets":[8,16]},
            "c":{"dtype":"BF16","shape":[3],"data_offsets":[16,22]}
        }"#;
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("header");
        file.write_all(json.as_bytes()).expect("metadata");
        file.write_all(&[0u8; 22]).expect("data");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");

        // has_tensor and get_tensor_info should be consistent
        for name in ["a", "b", "c"] {
            assert!(model.has_tensor(name));
            let info = model.get_tensor_info(name).expect("info");
            assert_eq!(info.name, name);
        }
        assert!(!model.has_tensor("nonexistent"));
        assert!(model.get_tensor_info("nonexistent").is_none());
    }
}
