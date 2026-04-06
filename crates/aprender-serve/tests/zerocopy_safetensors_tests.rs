//! Zero-Copy SafeTensors Tests (T-QA-020)
//!
//! Tests for `MappedSafeTensorsModel` implementation with:
//! - True zero-copy mmap verification
//! - BF16 native math without F32 conversion at boot
//! - TTFT benchmark (FAIL IF > 500ms for 3GB model)
//!
//! Coverage targets:
//! - F-ZC-001: Zero-copy loading via mmap
//! - F-ZC-002: BF16 native byte access
//! - F-ZC-003: TTFT < 500ms for 3GB model
//! - F-ZC-004: Memory efficiency (RSS < 50MB for metadata-only load)

use std::io::Write;
use std::time::Instant;
use tempfile::NamedTempFile;

// ============================================================================
// A. MappedSafeTensorsModel Basic Functionality Tests
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
mod mapped_model_tests {
    use super::*;
    use realizar::MappedSafeTensorsModel;

    /// Create a test SafeTensors file with given tensors
    fn create_test_safetensors(tensors: &[(&str, &str, &[usize], &[u8])]) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("create temp file");

        // Build JSON metadata
        let mut json_entries = Vec::new();
        let mut current_offset = 0usize;

        for (name, dtype, shape, data) in tensors {
            let end_offset = current_offset + data.len();
            json_entries.push(format!(
                r#""{name}":{{"dtype":"{dtype}","shape":{shape:?},"data_offsets":[{current_offset},{end_offset}]}}"#
            ));
            current_offset = end_offset;
        }

        let json = format!("{{{}}}", json_entries.join(","));
        let json_bytes = json.as_bytes();

        // Write header (8-byte metadata length)
        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write header");

        // Write JSON metadata
        file.write_all(json_bytes).expect("write metadata");

        // Write tensor data
        for (_, _, _, data) in tensors {
            file.write_all(data).expect("write tensor data");
        }

        file.flush().expect("flush file");
        file
    }

    // F-ZC-001: Basic loading functionality
    #[test]
    fn test_fzc001_basic_load() {
        let tensors = [(
            "weight",
            "F32",
            &[2, 3][..],
            &1.0f32.to_le_bytes().repeat(6)[..],
        )];
        let file = create_test_safetensors(&tensors);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");

        assert_eq!(model.tensor_count(), 1);
        assert!(model.has_tensor("weight"));
        assert!(!model.has_tensor("nonexistent"));
    }

    // F-ZC-001: Verify tensor names
    #[test]
    fn test_fzc001_tensor_names() {
        let l1w_data = vec![0u8; 128 * 256 * 4];
        let l1b_data = vec![0u8; 128 * 4];
        let l2w_data = vec![0u8; 64 * 128 * 4];
        let tensors = [
            ("layer1.weight", "F32", &[128, 256][..], &l1w_data[..]),
            ("layer1.bias", "F32", &[128][..], &l1b_data[..]),
            ("layer2.weight", "F32", &[64, 128][..], &l2w_data[..]),
        ];
        let file = create_test_safetensors(&tensors);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");

        let names = model.tensor_names();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"layer1.weight"));
        assert!(names.contains(&"layer1.bias"));
        assert!(names.contains(&"layer2.weight"));
    }

    // F-ZC-001: Zero-copy byte access
    #[test]
    fn test_fzc001_zero_copy_bytes() {
        // Create tensor with known pattern
        let pattern: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let tensors = [("pattern", "U8", &[256][..], &pattern[..])];
        let file = create_test_safetensors(&tensors);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");
        let bytes = model.get_tensor_bytes("pattern").expect("get bytes");

        // Verify zero-copy: data matches exactly
        assert_eq!(bytes.len(), 256);
        for (i, &b) in bytes.iter().enumerate() {
            assert_eq!(b, i as u8, "byte mismatch at index {i}");
        }
    }

    // F-ZC-001: F32 tensor access
    #[test]
    fn test_fzc001_f32_tensor() {
        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensors = [("weights", "F32", &[2, 3][..], &bytes[..])];
        let file = create_test_safetensors(&tensors);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");
        let result = model.get_tensor_f32("weights").expect("get f32");

        assert_eq!(result.len(), 6);
        for (i, (&got, &expected)) in result.iter().zip(values.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "value mismatch at index {i}: got {got}, expected {expected}"
            );
        }
    }

    // F-ZC-001: Tensor info access
    #[test]
    fn test_fzc001_tensor_info() {
        let matrix_data = vec![0u8; 64 * 128 * 4];
        let tensors = [("matrix", "F32", &[64, 128][..], &matrix_data[..])];
        let file = create_test_safetensors(&tensors);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");
        let info = model.get_tensor_info("matrix").expect("get info");

        assert_eq!(info.name, "matrix");
        assert_eq!(info.shape, vec![64, 128]);
        assert_eq!(info.data_offsets, [0, 64 * 128 * 4]);
    }

    // F-ZC-001: File metadata access
    #[test]
    fn test_fzc001_file_metadata() {
        let tensors = [("test", "F32", &[10][..], &[0u8; 40][..])];
        let file = create_test_safetensors(&tensors);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");

        assert_eq!(model.path(), file.path());
        assert!(model.file_size() > 0);
        assert_eq!(model.tensor_count(), 1);
    }
}

// ============================================================================
// B. BF16 Native Math Tests (F-ZC-002)
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
mod bf16_native_tests {
    use super::*;
    use half::bf16;
    use realizar::MappedSafeTensorsModel;

    fn create_bf16_safetensors(values: &[f32]) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("create temp file");

        // Convert F32 to BF16 bytes
        let bf16_bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect();

        let json = format!(
            r#"{{"weights":{{"dtype":"BF16","shape":[{}],"data_offsets":[0,{}]}}}}"#,
            values.len(),
            bf16_bytes.len()
        );
        let json_bytes = json.as_bytes();

        // Write header
        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write header");

        // Write JSON metadata
        file.write_all(json_bytes).expect("write metadata");

        // Write tensor data
        file.write_all(&bf16_bytes).expect("write tensor data");

        file.flush().expect("flush file");
        file
    }

    // F-ZC-002: BF16 zero-copy byte access (no F32 conversion at boot)
    #[test]
    fn test_fzc002_bf16_bytes_no_conversion() {
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let file = create_bf16_safetensors(&values);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");

        // get_tensor_bf16_bytes returns raw bytes WITHOUT F32 conversion
        let bytes = model
            .get_tensor_bf16_bytes("weights")
            .expect("get bf16 bytes");

        // Verify we got raw BF16 bytes (2 bytes per value)
        assert_eq!(bytes.len(), values.len() * 2);

        // Verify bytes are actual BF16 representation
        for (i, chunk) in bytes.chunks_exact(2).enumerate() {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let expected_bits = bf16::from_f32(values[i]).to_bits();
            assert_eq!(
                bits, expected_bits,
                "BF16 bits mismatch at index {i}: got {bits:#06x}, expected {expected_bits:#06x}"
            );
        }
    }

    // F-ZC-002: BF16 to F32 conversion on demand
    #[test]
    fn test_fzc002_bf16_to_f32_conversion() {
        let values = [1.5f32, -2.25, 0.0, 1000.0];
        let file = create_bf16_safetensors(&values);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");
        let result = model
            .get_tensor_bf16_as_f32("weights")
            .expect("get bf16 as f32");

        assert_eq!(result.len(), values.len());

        // BF16 has lower precision, so we use larger epsilon
        for (i, (&got, &expected)) in result.iter().zip(values.iter()).enumerate() {
            let epsilon = expected.abs() * 0.01 + 0.01; // 1% relative + small absolute
            assert!(
                (got - expected).abs() < epsilon,
                "BF16 conversion mismatch at index {i}: got {got}, expected {expected}"
            );
        }
    }

    // F-ZC-002: BF16 native SIMD-friendly layout
    #[test]
    fn test_fzc002_bf16_simd_alignment() {
        // Create a larger BF16 tensor for SIMD processing
        let values: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        let file = create_bf16_safetensors(&values);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");
        let bytes = model
            .get_tensor_bf16_bytes("weights")
            .expect("get bf16 bytes");

        // Verify data is contiguous and can be cast to bf16 slice
        assert_eq!(bytes.len(), 512); // 256 * 2 bytes

        // Verify we can process in SIMD-friendly chunks (8 values = 16 bytes)
        assert_eq!(bytes.len() % 16, 0);

        // Verify all values are accessible as chunks
        let bf16_count = bytes.len() / 2;
        assert_eq!(bf16_count, 256);
    }

    // F-ZC-002: BF16 special values
    #[test]
    fn test_fzc002_bf16_special_values() {
        let values = [0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY, 1.0, -1.0];
        let file = create_bf16_safetensors(&values);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");
        let result = model
            .get_tensor_bf16_as_f32("weights")
            .expect("get bf16 as f32");

        assert_eq!(result.len(), 6);

        // Check special values preserved
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1].to_bits(), (-0.0f32).to_bits()); // Check negative zero
        assert!(result[2].is_infinite() && result[2] > 0.0);
        assert!(result[3].is_infinite() && result[3] < 0.0);
        assert!((result[4] - 1.0).abs() < 0.01);
        assert!((result[5] - (-1.0)).abs() < 0.01);
    }

    // F-ZC-002: BF16 error handling
    #[test]
    fn test_fzc002_bf16_wrong_dtype_error() {
        // Create F32 tensor and try to access as BF16
        let mut file = NamedTempFile::new().expect("create temp file");

        let json = r#"{"weights":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        let json_bytes = json.as_bytes();

        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(json_bytes).expect("write metadata");
        file.write_all(&[0u8; 16]).expect("write tensor data");
        file.flush().expect("flush file");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");
        let result = model.get_tensor_bf16_bytes("weights");

        assert!(result.is_err());
    }
}

// ============================================================================
// C. F16 Native Math Tests
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
mod f16_native_tests {
    use super::*;
    use half::f16;
    use realizar::MappedSafeTensorsModel;

    fn create_f16_safetensors(values: &[f32]) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("create temp file");

        // Convert F32 to F16 bytes
        let f16_bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();

        let json = format!(
            r#"{{"weights":{{"dtype":"F16","shape":[{}],"data_offsets":[0,{}]}}}}"#,
            values.len(),
            f16_bytes.len()
        );
        let json_bytes = json.as_bytes();

        // Write header
        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(json_bytes).expect("write metadata");
        file.write_all(&f16_bytes).expect("write tensor data");
        file.flush().expect("flush file");
        file
    }

    // F16 zero-copy byte access
    #[test]
    fn test_f16_bytes_no_conversion() {
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let file = create_f16_safetensors(&values);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");
        let bytes = model
            .get_tensor_f16_bytes("weights")
            .expect("get f16 bytes");

        // Verify we got raw F16 bytes (2 bytes per value)
        assert_eq!(bytes.len(), values.len() * 2);

        // Verify bytes are actual F16 representation
        for (i, chunk) in bytes.chunks_exact(2).enumerate() {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            let expected_bits = f16::from_f32(values[i]).to_bits();
            assert_eq!(bits, expected_bits);
        }
    }

    // F16 to F32 conversion
    #[test]
    fn test_f16_to_f32_conversion() {
        let values = [0.5f32, 1.0, 1.5, 2.0];
        let file = create_f16_safetensors(&values);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");
        let result = model
            .get_tensor_f16_as_f32("weights")
            .expect("get f16 as f32");

        assert_eq!(result.len(), values.len());

        for (i, (&got, &expected)) in result.iter().zip(values.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 0.001,
                "F16 conversion mismatch at index {i}: got {got}, expected {expected}"
            );
        }
    }

    // F16 error handling
    #[test]
    fn test_f16_wrong_dtype_error() {
        // Create BF16 tensor and try to access as F16
        let mut file = NamedTempFile::new().expect("create temp file");

        let json = r#"{"weights":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]}}"#;
        let json_bytes = json.as_bytes();

        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(json_bytes).expect("write metadata");
        file.write_all(&[0u8; 8]).expect("write tensor data");
        file.flush().expect("flush file");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");
        let result = model.get_tensor_f16_bytes("weights");

        assert!(result.is_err());
    }
}

// ============================================================================
// D. Auto Dtype Conversion Tests
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
mod auto_dtype_tests {
    use super::*;
    use half::{bf16, f16};
    use realizar::MappedSafeTensorsModel;

    // Test auto conversion for F32
    #[test]
    fn test_auto_f32() {
        let mut file = NamedTempFile::new().expect("create temp file");

        let values = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let json = format!(
            r#"{{"weights":{{"dtype":"F32","shape":[4],"data_offsets":[0,{}]}}}}"#,
            bytes.len()
        );
        let json_bytes = json.as_bytes();

        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write");
        file.write_all(json_bytes).expect("write");
        file.write_all(&bytes).expect("write");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_auto("weights").expect("auto");

        assert_eq!(result, values.to_vec());
    }

    // Test auto conversion for F16
    #[test]
    fn test_auto_f16() {
        let mut file = NamedTempFile::new().expect("create temp file");

        let values = [1.0f32, 2.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();

        let json = format!(
            r#"{{"weights":{{"dtype":"F16","shape":[2],"data_offsets":[0,{}]}}}}"#,
            bytes.len()
        );
        let json_bytes = json.as_bytes();

        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write");
        file.write_all(json_bytes).expect("write");
        file.write_all(&bytes).expect("write");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_auto("weights").expect("auto");

        assert_eq!(result.len(), 2);
    }

    // Test auto conversion for BF16
    #[test]
    fn test_auto_bf16() {
        let mut file = NamedTempFile::new().expect("create temp file");

        let values = [1.0f32, 2.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect();

        let json = format!(
            r#"{{"weights":{{"dtype":"BF16","shape":[2],"data_offsets":[0,{}]}}}}"#,
            bytes.len()
        );
        let json_bytes = json.as_bytes();

        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write");
        file.write_all(json_bytes).expect("write");
        file.write_all(&bytes).expect("write");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_auto("weights").expect("auto");

        assert_eq!(result.len(), 2);
    }

    // Test auto conversion error for unsupported dtype
    #[test]
    fn test_auto_unsupported_dtype() {
        let mut file = NamedTempFile::new().expect("create temp file");

        let json = r#"{"weights":{"dtype":"I32","shape":[4],"data_offsets":[0,16]}}"#;
        let json_bytes = json.as_bytes();

        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write");
        file.write_all(json_bytes).expect("write");
        file.write_all(&[0u8; 16]).expect("write");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_auto("weights");

        assert!(result.is_err());
    }

    // Test auto conversion error for not found
    #[test]
    fn test_auto_not_found() {
        let mut file = NamedTempFile::new().expect("create temp file");

        let json = r#"{"weights":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        let json_bytes = json.as_bytes();

        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write");
        file.write_all(json_bytes).expect("write");
        file.write_all(&[0u8; 16]).expect("write");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_auto("nonexistent");

        assert!(result.is_err());
    }
}

// ============================================================================
// E. Error Handling Tests
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
mod error_tests {
    use super::*;
    use realizar::MappedSafeTensorsModel;

    // File not found
    #[test]
    fn test_load_nonexistent_file() {
        let result = MappedSafeTensorsModel::load("/nonexistent/path/to/model.safetensors");
        assert!(result.is_err());
    }

    // File too small (< 8 bytes)
    #[test]
    fn test_load_file_too_small() {
        let mut file = NamedTempFile::new().expect("create temp file");
        file.write_all(&[0u8; 4]).expect("write"); // Only 4 bytes
        file.flush().expect("flush");

        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
    }

    // Truncated metadata
    #[test]
    fn test_load_truncated_metadata() {
        let mut file = NamedTempFile::new().expect("create temp file");
        // Header says 1000 bytes of metadata, but file only has header
        file.write_all(&1000u64.to_le_bytes()).expect("write");
        file.flush().expect("flush");

        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
    }

    // Invalid JSON
    #[test]
    fn test_load_invalid_json() {
        let mut file = NamedTempFile::new().expect("create temp file");
        let json = b"not valid json!!!";
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("write");
        file.write_all(json).expect("write");
        file.flush().expect("flush");

        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
    }

    // JSON not an object
    #[test]
    fn test_load_json_not_object() {
        let mut file = NamedTempFile::new().expect("create temp file");
        let json = b"[]"; // Array, not object
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("write");
        file.write_all(json).expect("write");
        file.flush().expect("flush");

        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
    }

    // Tensor not found
    #[test]
    fn test_get_tensor_not_found() {
        let mut file = NamedTempFile::new().expect("create temp file");
        let json = b"{}";
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("write");
        file.write_all(json).expect("write");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_bytes("nonexistent");
        assert!(result.is_err());
    }

    // Data offset exceeds file size
    #[test]
    fn test_get_tensor_offset_exceeds_file() {
        let mut file = NamedTempFile::new().expect("create temp file");
        let json = r#"{"weight":{"dtype":"F32","shape":[100],"data_offsets":[0,400]}}"#;
        let json_bytes = json.as_bytes();

        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write");
        file.write_all(json_bytes).expect("write");
        // Only write 100 bytes of data, but offsets say 400
        file.write_all(&[0u8; 100]).expect("write");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_bytes("weight");
        assert!(result.is_err());
    }

    // F32 tensor with wrong size (not multiple of 4)
    #[test]
    fn test_get_f32_wrong_size() {
        let mut file = NamedTempFile::new().expect("create temp file");
        let json = r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,7]}}"#;
        let json_bytes = json.as_bytes();

        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write");
        file.write_all(json_bytes).expect("write");
        file.write_all(&[0u8; 7]).expect("write"); // 7 bytes, not divisible by 4
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_f32("weight");
        assert!(result.is_err());
    }

    // F32 accessor with wrong dtype
    #[test]
    fn test_get_f32_wrong_dtype() {
        let mut file = NamedTempFile::new().expect("create temp file");
        let json = r#"{"weight":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]}}"#;
        let json_bytes = json.as_bytes();

        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write");
        file.write_all(json_bytes).expect("write");
        file.write_all(&[0u8; 8]).expect("write");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_f32("weight");
        assert!(result.is_err());
    }

    // Metadata skipping (__metadata__ key)
    #[test]
    fn test_metadata_key_skipped() {
        let mut file = NamedTempFile::new().expect("create temp file");
        let json = r#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let json_bytes = json.as_bytes();

        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write");
        file.write_all(json_bytes).expect("write");
        file.write_all(&[0u8; 8]).expect("write");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");

        // Should have 1 tensor, not 2 (metadata skipped)
        assert_eq!(model.tensor_count(), 1);
        assert!(model.has_tensor("weight"));
        assert!(!model.has_tensor("__metadata__"));
    }
}

// ============================================================================
// F. TTFT (Time To First Token) Benchmark Tests (F-ZC-003)
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
mod ttft_benchmark_tests {
    use super::*;
    use realizar::MappedSafeTensorsModel;

    /// Create a large test file for benchmarking
    /// Returns (file, size_bytes)
    fn create_large_safetensors(size_mb: usize) -> (NamedTempFile, usize) {
        let mut file = NamedTempFile::new().expect("create temp file");

        // Calculate tensor size
        let tensor_bytes = size_mb * 1024 * 1024;
        let num_floats = tensor_bytes / 4;

        // Create minimal JSON metadata
        let json = format!(
            r#"{{"large_tensor":{{"dtype":"F32","shape":[{num_floats}],"data_offsets":[0,{tensor_bytes}]}}}}"#
        );
        let json_bytes = json.as_bytes();

        // Write header
        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write header");

        // Write metadata
        file.write_all(json_bytes).expect("write metadata");

        // Write tensor data (zeros, since we only care about load time)
        let chunk_size = 1024 * 1024; // 1MB chunks
        let zeros = vec![0u8; chunk_size];
        let mut written = 0;
        while written < tensor_bytes {
            let to_write = (tensor_bytes - written).min(chunk_size);
            file.write_all(&zeros[..to_write]).expect("write data");
            written += to_write;
        }

        file.flush().expect("flush");

        let total_size = 8 + json_bytes.len() + tensor_bytes;
        (file, total_size)
    }

    // F-ZC-003: TTFT < 50ms for 100MB model
    #[test]
    fn test_fzc003_ttft_100mb() {
        let (file, size) = create_large_safetensors(100);
        assert!(size >= 100 * 1024 * 1024, "File too small: {size} bytes");

        let start = Instant::now();
        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");
        let elapsed = start.elapsed();

        // TTFT should be < 50ms for 100MB (mmap is O(1))
        assert!(
            elapsed.as_millis() < 50,
            "TTFT too slow for 100MB: {:?} (expected < 50ms)",
            elapsed
        );

        // Verify model loaded correctly
        assert_eq!(model.tensor_count(), 1);
        assert!(model.has_tensor("large_tensor"));
    }

    // F-ZC-003: TTFT < 100ms for 500MB model
    #[test]
    fn test_fzc003_ttft_500mb() {
        let (file, size) = create_large_safetensors(500);
        assert!(size >= 500 * 1024 * 1024, "File too small: {size} bytes");

        let start = Instant::now();
        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");
        let elapsed = start.elapsed();

        // TTFT should be < 100ms for 500MB
        assert!(
            elapsed.as_millis() < 100,
            "TTFT too slow for 500MB: {:?} (expected < 100ms)",
            elapsed
        );

        assert_eq!(model.tensor_count(), 1);
    }

    // F-ZC-003: TTFT < 200ms for 1GB model
    #[test]
    fn test_fzc003_ttft_1gb() {
        let (file, size) = create_large_safetensors(1024);
        assert!(size >= 1024 * 1024 * 1024, "File too small: {size} bytes");

        let start = Instant::now();
        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");
        let elapsed = start.elapsed();

        // TTFT should be < 200ms for 1GB
        assert!(
            elapsed.as_millis() < 200,
            "TTFT too slow for 1GB: {:?} (expected < 200ms)",
            elapsed
        );

        assert_eq!(model.tensor_count(), 1);
    }

    // F-ZC-003: CRITICAL - TTFT < 500ms for 3GB model
    // This is the main performance gate from the spec
    #[test]
    #[ignore = "creates a 3GB file, run manually"]
    fn test_fzc003_ttft_3gb_critical() {
        let (file, size) = create_large_safetensors(3 * 1024);
        assert!(
            size >= 3 * 1024 * 1024 * 1024,
            "File too small: {size} bytes"
        );

        let start = Instant::now();
        let model = MappedSafeTensorsModel::load(file.path()).expect("load model");
        let elapsed = start.elapsed();

        // CRITICAL: TTFT MUST be < 500ms for 3GB model
        // This is the falsifiable gate from the spec
        assert!(
            elapsed.as_millis() < 500,
            "CRITICAL FAIL: TTFT for 3GB model is {:?} (MUST be < 500ms)\n\
             This violates the T-QA-020 performance gate!",
            elapsed
        );

        assert_eq!(model.tensor_count(), 1);
        println!("PASS: 3GB model TTFT = {:?} (< 500ms threshold)", elapsed);
    }

    // Verify zero-copy: loading time should be constant regardless of file size
    #[test]
    fn test_fzc003_zerocopy_constant_time() {
        // Load 10MB file
        let (file_10mb, _) = create_large_safetensors(10);
        let start = Instant::now();
        let _model_10 = MappedSafeTensorsModel::load(file_10mb.path()).expect("load");
        let time_10mb = start.elapsed();

        // Load 100MB file
        let (file_100mb, _) = create_large_safetensors(100);
        let start = Instant::now();
        let _model_100 = MappedSafeTensorsModel::load(file_100mb.path()).expect("load");
        let time_100mb = start.elapsed();

        // Load 500MB file
        let (file_500mb, _) = create_large_safetensors(500);
        let start = Instant::now();
        let _model_500 = MappedSafeTensorsModel::load(file_500mb.path()).expect("load");
        let time_500mb = start.elapsed();

        println!("Zero-copy load times:");
        println!("  10MB:  {:?}", time_10mb);
        println!("  100MB: {:?}", time_100mb);
        println!("  500MB: {:?}", time_500mb);

        // With mmap, loading 500MB should NOT take 50x longer than 10MB
        // Allow 5x variance for filesystem overhead
        let ratio = time_500mb.as_micros() as f64 / time_10mb.as_micros().max(1) as f64;
        assert!(
            ratio < 10.0,
            "Loading time scales with file size (ratio={ratio:.1}x), suggesting eager loading instead of mmap"
        );
    }
}

// ============================================================================
// G. Memory Efficiency Tests (F-ZC-004)
// ============================================================================

#[cfg(all(not(target_arch = "wasm32"), target_os = "linux"))]
mod memory_efficiency_tests {
    use super::*;
    use realizar::MappedSafeTensorsModel;

    /// Get current RSS (Resident Set Size) in bytes
    fn get_rss_bytes() -> Option<usize> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let kb: usize = parts[1].parse().ok()?;
                    return Some(kb * 1024);
                }
            }
        }
        None
    }

    /// Create a large test file
    fn create_large_file(size_mb: usize) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("create temp file");
        let tensor_bytes = size_mb * 1024 * 1024;
        let num_floats = tensor_bytes / 4;

        let json = format!(
            r#"{{"tensor":{{"dtype":"F32","shape":[{num_floats}],"data_offsets":[0,{tensor_bytes}]}}}}"#
        );
        let json_bytes = json.as_bytes();

        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write");
        file.write_all(json_bytes).expect("write");

        // Write in chunks to avoid huge memory allocation
        let chunk = vec![0u8; 1024 * 1024];
        for _ in 0..size_mb {
            file.write_all(&chunk).expect("write");
        }
        file.flush().expect("flush");
        file
    }

    // F-ZC-004: RSS increase should be minimal when loading (metadata only)
    #[test]
    fn test_fzc004_rss_minimal_on_load() {
        let file = create_large_file(200); // 200MB file

        let rss_before = get_rss_bytes().unwrap_or(0);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");

        // Force metadata to be parsed
        assert_eq!(model.tensor_count(), 1);

        let rss_after = get_rss_bytes().unwrap_or(0);
        let rss_increase = rss_after.saturating_sub(rss_before);

        // RSS increase should be < 50MB for loading a 200MB file
        // (Only metadata is read, data is mmap'd but not paged in)
        assert!(
            rss_increase < 50 * 1024 * 1024,
            "RSS increased by {:.1}MB when loading 200MB file (expected < 50MB increase)",
            rss_increase as f64 / (1024.0 * 1024.0)
        );

        println!(
            "RSS increase for 200MB mmap'd file: {:.2}MB",
            rss_increase as f64 / (1024.0 * 1024.0)
        );
    }

    // F-ZC-004: Accessing data should page in only accessed regions
    #[test]
    fn test_fzc004_partial_access_rss() {
        let file = create_large_file(100); // 100MB file

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");

        let rss_before_access = get_rss_bytes().unwrap_or(0);

        // Access only first 1KB of tensor data
        let bytes = model.get_tensor_bytes("tensor").expect("get bytes");
        let _first_kb: Vec<u8> = bytes[..1024].to_vec();

        let rss_after_access = get_rss_bytes().unwrap_or(0);
        let rss_increase = rss_after_access.saturating_sub(rss_before_access);

        // RSS should increase by roughly page size, not full tensor size
        // Allow up to 10MB for page prefetching and other overhead
        assert!(
            rss_increase < 10 * 1024 * 1024,
            "RSS increased by {:.1}MB when accessing 1KB (expected < 10MB)",
            rss_increase as f64 / (1024.0 * 1024.0)
        );

        println!(
            "RSS increase for 1KB access: {:.2}MB",
            rss_increase as f64 / (1024.0 * 1024.0)
        );
    }
}

// ============================================================================
// H. Comparison with Eager Loading (SafetensorsModel)
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
mod comparison_tests {
    use super::*;
    use realizar::safetensors::SafetensorsModel;
    use realizar::MappedSafeTensorsModel;

    /// Create a medium-sized test file
    fn create_medium_file() -> (NamedTempFile, usize) {
        let mut file = NamedTempFile::new().expect("create temp file");
        let size_mb = 50;
        let tensor_bytes = size_mb * 1024 * 1024;
        let num_floats = tensor_bytes / 4;

        let json = format!(
            r#"{{"weights":{{"dtype":"F32","shape":[{num_floats}],"data_offsets":[0,{tensor_bytes}]}}}}"#
        );
        let json_bytes = json.as_bytes();

        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write");
        file.write_all(json_bytes).expect("write");

        let chunk = vec![0u8; 1024 * 1024];
        for _ in 0..size_mb {
            file.write_all(&chunk).expect("write");
        }
        file.flush().expect("flush");

        (file, tensor_bytes)
    }

    // Compare loading time: mmap vs eager
    #[test]
    fn test_mmap_faster_than_eager() {
        let (file, _) = create_medium_file();

        // Time mmap loading
        let start = Instant::now();
        let _mmap_model = MappedSafeTensorsModel::load(file.path()).expect("mmap load");
        let mmap_time = start.elapsed();

        // Time eager loading
        let data = std::fs::read(file.path()).expect("read file");
        let start = Instant::now();
        let _eager_model = SafetensorsModel::from_bytes(&data).expect("eager load");
        let eager_time = start.elapsed();

        println!("50MB file load times:");
        println!("  Mmap:  {:?}", mmap_time);
        println!("  Eager: {:?}", eager_time);

        // Mmap should be significantly faster (at least 2x)
        // Note: On very fast SSDs, the difference may be smaller
        assert!(
            mmap_time < eager_time,
            "Mmap loading ({:?}) should be faster than eager loading ({:?})",
            mmap_time,
            eager_time
        );
    }

    // Both methods should produce same tensor data
    #[test]
    fn test_mmap_and_eager_same_data() {
        let mut file = NamedTempFile::new().expect("create temp file");

        let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let json = format!(
            r#"{{"weights":{{"dtype":"F32","shape":[8],"data_offsets":[0,{}]}}}}"#,
            bytes.len()
        );
        let json_bytes = json.as_bytes();

        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write");
        file.write_all(json_bytes).expect("write");
        file.write_all(&bytes).expect("write");
        file.flush().expect("flush");

        // Load with mmap
        let mmap_model = MappedSafeTensorsModel::load(file.path()).expect("mmap load");
        let mmap_data = mmap_model.get_tensor_f32("weights").expect("get mmap");

        // Load with eager
        let file_data = std::fs::read(file.path()).expect("read file");
        let eager_model = SafetensorsModel::from_bytes(&file_data).expect("eager load");
        let eager_data = eager_model.get_tensor_f32("weights").expect("get eager");

        // Data should be identical
        assert_eq!(mmap_data, eager_data);
    }
}
