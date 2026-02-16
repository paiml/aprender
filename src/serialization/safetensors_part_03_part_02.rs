use crate::serialization::safetensors::{
    extract_tensor, load_safetensors, save_safetensors, save_safetensors_with_metadata,
    MappedSafeTensors, SafeTensorsDType, TensorMetadata, UserMetadata,
};
use crate::serialization::safetensors::safetensors_part_02::{extract_bf16_to_f32, extract_f16_to_f32};
use std::collections::BTreeMap;
use std::fs;

    #[test]
    fn test_save_and_load_safetensors() {
        let path = "/tmp/test_safetensors_module.safetensors";

        // Create test tensors
        let mut tensors = BTreeMap::new();
        tensors.insert("weights".to_string(), (vec![1.0, 2.0, 3.0], vec![3]));
        tensors.insert("bias".to_string(), (vec![0.5], vec![1]));

        // Save
        save_safetensors(path, &tensors)
            .expect("Failed to save test tensors to SafeTensors format");

        // Load
        let (metadata, raw_data) =
            load_safetensors(path).expect("Failed to load test SafeTensors file");

        // Verify metadata
        assert!(metadata.contains_key("weights"));
        assert!(metadata.contains_key("bias"));

        // Extract and verify tensors
        let weights = extract_tensor(&raw_data, &metadata["weights"])
            .expect("Failed to extract weights tensor from raw data");
        assert_eq!(weights, vec![1.0, 2.0, 3.0]);

        let bias = extract_tensor(&raw_data, &metadata["bias"])
            .expect("Failed to extract bias tensor from raw data");
        assert_eq!(bias, vec![0.5]);

        // Cleanup
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_save_safetensors_header_format() {
        let path = "/tmp/test_header_format.safetensors";

        let mut tensors = BTreeMap::new();
        tensors.insert("test".to_string(), (vec![1.0], vec![1]));

        save_safetensors(path, &tensors)
            .expect("Failed to save test tensor for header format verification");

        // Read and verify header
        let bytes =
            fs::read(path).expect("Failed to read test SafeTensors file for header verification");
        assert!(bytes.len() >= 8, "File must have at least 8-byte header");

        let header_bytes: [u8; 8] = bytes[0..8]
            .try_into()
            .expect("Failed to convert first 8 bytes to header array (file has at least 8 bytes)");
        let metadata_len = u64::from_le_bytes(header_bytes);
        assert!(metadata_len > 0, "Metadata length must be > 0");

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_safetensors_corrupted_header() {
        let path = "/tmp/test_corrupted_header.safetensors";

        // Write invalid file (< 8 bytes)
        fs::write(path, [1, 2, 3]).expect("Failed to write test file with corrupted header");

        let result = load_safetensors(path);
        assert!(result.is_err());
        assert!(result
            .expect_err("Should fail with corrupted header size check")
            .contains("8 bytes"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_safetensors_nonexistent_file() {
        let result = load_safetensors("/tmp/nonexistent_file_xyz.safetensors");
        assert!(result.is_err());
        let err = result.expect_err("Should fail when file not found");
        assert!(
            err.contains("No such file") || err.contains("not found"),
            "Error should mention file not found: {err}"
        );
    }

    #[test]
    fn test_extract_tensor_invalid_offsets() {
        let raw_data = vec![0u8; 16];
        let meta = TensorMetadata {
            dtype: "F32".to_string(),
            shape: vec![1],
            data_offsets: [0, 100], // Exceeds data size
        };

        let result = extract_tensor(&raw_data, &meta);
        assert!(result.is_err());
        assert!(result
            .expect_err("Should fail when tensor offset exceeds data size")
            .contains("exceeds data size"));
    }

    #[test]
    fn test_deterministic_serialization() {
        // Verify that serialization is deterministic (sorted keys)
        let path1 = "/tmp/test_det1.safetensors";
        let path2 = "/tmp/test_det2.safetensors";

        let mut tensors = BTreeMap::new();
        tensors.insert("z_last".to_string(), (vec![3.0], vec![1]));
        tensors.insert("a_first".to_string(), (vec![1.0], vec![1]));
        tensors.insert("m_middle".to_string(), (vec![2.0], vec![1]));

        // Save twice
        save_safetensors(path1, &tensors).expect("Failed to save first deterministic test file");
        save_safetensors(path2, &tensors).expect("Failed to save second deterministic test file");

        // Files should be identical (deterministic)
        let bytes1 = fs::read(path1).expect("Failed to read first deterministic test file");
        let bytes2 = fs::read(path2).expect("Failed to read second deterministic test file");
        assert_eq!(bytes1, bytes2, "Serialization must be deterministic");

        fs::remove_file(path1).ok();
        fs::remove_file(path2).ok();
    }

    // =========================================================================
    // Coverage boost: MappedSafeTensors API tests
    // =========================================================================

    #[test]
    fn test_mapped_safetensors_full_api() {
        let path = "/tmp/test_mapped_api.safetensors";

        // Create multi-tensor file
        let mut tensors = BTreeMap::new();
        tensors.insert("weight".to_string(), (vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]));
        tensors.insert("bias".to_string(), (vec![0.5, 0.5], vec![2]));
        tensors.insert("scale".to_string(), (vec![1.0], vec![1]));

        save_safetensors(path, &tensors).expect("save");

        // Test MappedSafeTensors API
        let mapped = MappedSafeTensors::open(path).expect("open");

        // len/is_empty
        assert_eq!(mapped.len(), 3);
        assert!(!mapped.is_empty());

        // tensor_names
        let names = mapped.tensor_names();
        assert!(names.contains(&"weight"));
        assert!(names.contains(&"bias"));
        assert!(names.contains(&"scale"));

        // get_metadata
        let meta = mapped.get_metadata("weight").expect("metadata");
        assert_eq!(meta.dtype, "F32");
        assert_eq!(meta.shape, vec![2, 2]);

        assert!(mapped.get_metadata("nonexistent").is_none());

        // get_tensor
        let weight = mapped.get_tensor("weight").expect("tensor");
        assert_eq!(weight, vec![1.0, 2.0, 3.0, 4.0]);

        // get_tensor_bytes
        let bytes = mapped.get_tensor_bytes("bias").expect("bytes");
        assert_eq!(bytes.len(), 8); // 2 f32 = 8 bytes

        // Error: tensor not found
        let err = mapped.get_tensor("missing").unwrap_err();
        assert!(err.contains("not found"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_mapped_safetensors_empty_file() {
        let path = "/tmp/test_empty_tensors.safetensors";

        let tensors = BTreeMap::new();
        save_safetensors(path, &tensors).expect("save empty");

        let mapped = MappedSafeTensors::open(path).expect("open empty");
        assert!(mapped.is_empty());
        assert_eq!(mapped.len(), 0);
        assert!(mapped.tensor_names().is_empty());

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_validate_header_metadata_zero() {
        let path = "/tmp/test_zero_meta.safetensors";

        // Create file with 0 metadata length
        let bytes: Vec<u8> = vec![0, 0, 0, 0, 0, 0, 0, 0];
        fs::write(path, &bytes).expect("write");

        let result = MappedSafeTensors::open(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("metadata length is 0"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_validate_header_metadata_exceeds_file() {
        let path = "/tmp/test_exceed_meta.safetensors";

        // Create file with metadata length > file size
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1000u64.to_le_bytes()); // claim 1000 bytes
        bytes.extend_from_slice(b"{}"); // only 2 bytes of metadata
        fs::write(path, &bytes).expect("write");

        let result = MappedSafeTensors::open(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeds file size"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_parse_metadata_with_dunder_keys() {
        let path = "/tmp/test_dunder.safetensors";

        // PMAT-223: __metadata__ is now extracted as user metadata, not discarded
        let metadata = r#"{"__metadata__":{"format":"pt","training_run_id":"12345"},"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let meta_bytes = metadata.as_bytes();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(meta_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(meta_bytes);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        fs::write(path, &bytes).expect("write");

        let mapped = MappedSafeTensors::open(path).expect("open");
        assert_eq!(mapped.len(), 1); // only "tensor", not "__metadata__"
        assert!(mapped.get_metadata("__metadata__").is_none()); // Not a tensor
        assert!(mapped.get_metadata("tensor").is_some());

        // PMAT-223: User metadata IS extracted
        let user_meta = mapped.user_metadata();
        assert_eq!(user_meta.len(), 2);
        assert_eq!(user_meta.get("format"), Some(&"pt".to_string()));
        assert_eq!(user_meta.get("training_run_id"), Some(&"12345".to_string()));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_save_safetensors_with_metadata_round_trip() {
        let path = "/tmp/test_metadata_roundtrip.safetensors";
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "weight".to_string(),
            (vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]),
        );

        let mut user_metadata = UserMetadata::new();
        user_metadata.insert("my_run_id".to_string(), "test_123".to_string());
        user_metadata.insert("framework".to_string(), "pytorch".to_string());

        // Write with metadata
        save_safetensors_with_metadata(path, &tensors, &user_metadata).expect("save");

        // Read back and verify metadata round-trips
        let mapped = MappedSafeTensors::open(path).expect("open");
        assert_eq!(mapped.len(), 1);
        assert!(mapped.get_metadata("weight").is_some());

        let restored = mapped.user_metadata();
        assert_eq!(restored.get("my_run_id"), Some(&"test_123".to_string()));
        assert_eq!(restored.get("framework"), Some(&"pytorch".to_string()));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_empty_user_metadata_no_dunder_section() {
        let path = "/tmp/test_no_dunder.safetensors";

        // File without __metadata__ should have empty user_metadata
        let metadata = r#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let meta_bytes = metadata.as_bytes();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(meta_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(meta_bytes);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        fs::write(path, &bytes).expect("write");

        let mapped = MappedSafeTensors::open(path).expect("open");
        assert!(mapped.user_metadata().is_empty());

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_extract_bf16() {
        // BF16: 0x3F80 = 1.0 in BF16
        let bf16_bytes = vec![0x80, 0x3F, 0x00, 0x40]; // 1.0, 2.0
        let result = extract_bf16_to_f32(&bf16_bytes).expect("bf16");
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_extract_f16() {
        // F16: 0x3C00 = 1.0 in F16
        let f16_bytes = vec![0x00, 0x3C, 0x00, 0x40]; // 1.0, 2.0
        let result = extract_f16_to_f32(&f16_bytes).expect("f16");
        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_unsupported_dtype() {
        let path = "/tmp/test_unsupported.safetensors";

        // Create file with unsupported dtype
        let metadata = r#"{"tensor":{"dtype":"INT8","shape":[1],"data_offsets":[0,1]}}"#;
        let meta_bytes = metadata.as_bytes();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(meta_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(meta_bytes);
        bytes.push(42); // 1 byte of data
        fs::write(path, &bytes).expect("write");

        let mapped = MappedSafeTensors::open(path).expect("open");
        let result = mapped.get_tensor("tensor");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unsupported dtype"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_tensor_out_of_bounds() {
        let path = "/tmp/test_oob.safetensors";

        // Create file with tensor pointing past end
        let metadata = r#"{"tensor":{"dtype":"F32","shape":[100],"data_offsets":[0,400]}}"#;
        let meta_bytes = metadata.as_bytes();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(meta_bytes.len() as u64).to_le_bytes());
        bytes.extend_from_slice(meta_bytes);
        bytes.extend_from_slice(&[0u8; 16]); // only 16 bytes, not 400
        fs::write(path, &bytes).expect("write");

        let mapped = MappedSafeTensors::open(path).expect("open");
        let result = mapped.get_tensor("tensor");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("out of bounds"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_invalid_utf8_metadata() {
        let path = "/tmp/test_invalid_utf8.safetensors";

        // Create file with invalid UTF-8 in metadata
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&4u64.to_le_bytes());
        bytes.extend_from_slice(&[0xFF, 0xFE, 0x00, 0x01]); // Invalid UTF-8
        fs::write(path, &bytes).expect("write");

        let result = MappedSafeTensors::open(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("UTF-8"));

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_invalid_json_metadata() {
        let path = "/tmp/test_invalid_json.safetensors";

        let invalid_json = b"not valid json{";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(invalid_json.len() as u64).to_le_bytes());
        bytes.extend_from_slice(invalid_json);
        fs::write(path, &bytes).expect("write");

        let result = MappedSafeTensors::open(path);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("JSON"));

        fs::remove_file(path).ok();
    }

    // ====================================================================
    // get_tensor_raw: Coverage tests (impact 3.6)
    // ====================================================================

    #[test]
    fn test_get_tensor_raw_f32() {
        let path = "/tmp/test_get_tensor_raw_f32.safetensors";

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "weight".to_string(),
            (vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]),
        );
        save_safetensors(path, &tensors).expect("save");

        let mapped = MappedSafeTensors::open(path).expect("open");
        let raw = mapped.get_tensor_raw("weight").expect("get raw");

        assert!(matches!(raw.dtype, SafeTensorsDType::F32));
        assert_eq!(raw.shape, vec![2, 2]);
        assert_eq!(raw.bytes.len(), 16); // 4 floats * 4 bytes

        // Verify data round-trips correctly
        let f32_values = raw.to_f32().expect("convert to f32");
        assert_eq!(f32_values, vec![1.0, 2.0, 3.0, 4.0]);

        fs::remove_file(path).ok();
    }
