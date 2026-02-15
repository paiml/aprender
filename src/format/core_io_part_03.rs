
#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    // ============================================================================
    // CRC32 Tests
    // ============================================================================

    #[test]
    fn test_crc32_empty() {
        // CRC32 of empty data (IEEE polynomial)
        assert_eq!(crc32(&[]), 0x0000_0000);
    }

    #[test]
    fn test_crc32_known_values() {
        // "123456789" should give CRC32 = 0xCBF43926
        let data = b"123456789";
        assert_eq!(crc32(data), 0xCBF4_3926);
    }

    #[test]
    fn test_crc32_single_byte() {
        // Single byte values
        assert_eq!(crc32(&[0x00]), 0xD202_EF8D);
        assert_eq!(crc32(&[0xFF]), 0xFF00_0000);
    }

    #[test]
    fn test_crc32_multiple_bytes() {
        let data = b"Hello, World!";
        let crc = crc32(data);
        // Verify it's deterministic
        assert_eq!(crc, crc32(data));
        // Verify different data gives different CRC
        assert_ne!(crc, crc32(b"Hello, World"));
    }

    // ============================================================================
    // Compression Tests
    // ============================================================================

    #[test]
    fn test_compress_payload_none() {
        let data = b"test data for compression";
        let (compressed, compression) =
            compress_payload(data, Compression::None).expect("compress");
        assert_eq!(compression, Compression::None);
        assert_eq!(compressed, data);
    }

    #[test]
    fn test_decompress_payload_none() {
        let data = b"test data for decompression";
        let decompressed = decompress_payload(data, Compression::None).expect("decompress");
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "format-compression")]
    #[test]
    fn test_compress_decompress_zstd_default() {
        let data = b"test data that should compress well with zstd compression";
        let (compressed, compression) =
            compress_payload(data, Compression::ZstdDefault).expect("compress");
        assert_eq!(compression, Compression::ZstdDefault);
        let decompressed = decompress_payload(&compressed, compression).expect("decompress");
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "format-compression")]
    #[test]
    fn test_compress_decompress_lz4() {
        let data = b"test data for lz4 compression";
        let (compressed, compression) = compress_payload(data, Compression::Lz4).expect("compress");
        assert_eq!(compression, Compression::Lz4);
        let decompressed = decompress_payload(&compressed, compression).expect("decompress");
        assert_eq!(decompressed, data);
    }

    // ============================================================================
    // Save/Load Round-Trip Tests
    // ============================================================================

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TestModel {
        name: String,
        values: Vec<f32>,
    }

    #[test]
    fn test_save_load_roundtrip() {
        let model = TestModel {
            name: "test_model".to_string(),
            values: vec![1.0, 2.0, 3.0, 4.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("test.apr");

        let options = SaveOptions::default();
        save(&model, ModelType::LinearRegression, &path, options).expect("save");

        let loaded: TestModel = load(&path, ModelType::LinearRegression).expect("load");
        assert_eq!(model, loaded);
    }

    #[test]
    fn test_save_with_metadata() {
        let model = TestModel {
            name: "metadata_test".to_string(),
            values: vec![1.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("test_metadata.apr");

        let mut metadata = Metadata::default();
        metadata.description = Some("A test model".to_string());

        let options = SaveOptions {
            metadata,
            compression: Compression::None,
            quality_score: Some(85),
        };
        save(&model, ModelType::LinearRegression, &path, options).expect("save");

        let info = inspect(&path).expect("inspect");
        assert_eq!(info.metadata.description, Some("A test model".to_string()));
    }

    #[test]
    fn test_save_rejects_quality_score_zero() {
        let model = TestModel {
            name: "bad_model".to_string(),
            values: vec![],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("should_not_exist.apr");

        let options = SaveOptions {
            quality_score: Some(0),
            ..Default::default()
        };

        let result = save(&model, ModelType::LinearRegression, &path, options);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_wrong_model_type() {
        let model = TestModel {
            name: "type_test".to_string(),
            values: vec![1.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("type_test.apr");

        save(
            &model,
            ModelType::LinearRegression,
            &path,
            SaveOptions::default(),
        )
        .expect("save");

        let result: Result<TestModel> = load(&path, ModelType::KMeans);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result: Result<TestModel> =
            load("/nonexistent/path/model.apr", ModelType::LinearRegression);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_file_too_small() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("tiny.apr");

        std::fs::write(&path, &[0u8; 10]).expect("write tiny file");

        let result: Result<TestModel> = load(&path, ModelType::LinearRegression);
        assert!(result.is_err());
    }

    // ============================================================================
    // Inspect Tests
    // ============================================================================

    #[test]
    fn test_inspect_model() {
        let model = TestModel {
            name: "inspect_test".to_string(),
            values: vec![1.0, 2.0, 3.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("inspect_test.apr");

        let mut metadata = Metadata::default();
        metadata.model_name = Some("Test Model".to_string());

        let options = SaveOptions {
            metadata,
            compression: Compression::None,
            quality_score: Some(90),
        };
        save(&model, ModelType::LinearRegression, &path, options).expect("save");

        let info = inspect(&path).expect("inspect");
        assert_eq!(info.model_type, ModelType::LinearRegression);
        assert_eq!(info.metadata.model_name, Some("Test Model".to_string()));
    }

    #[test]
    fn test_inspect_with_license_flag() {
        use super::super::{LicenseInfo, LicenseTier};

        let model = TestModel {
            name: "licensed".to_string(),
            values: vec![1.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("licensed.apr");

        let mut metadata = Metadata::default();
        metadata.license = Some(LicenseInfo {
            uuid: "test-uuid".to_string(),
            hash: "test-hash".to_string(),
            expiry: None,
            seats: None,
            licensee: Some("Test User".to_string()),
            tier: LicenseTier::Enterprise,
        });

        let options = SaveOptions {
            metadata,
            compression: Compression::None,
            quality_score: None,
        };
        save(&model, ModelType::LinearRegression, &path, options).expect("save");

        let info = inspect(&path).expect("inspect");
        assert!(info.licensed);
    }

    #[test]
    fn test_inspect_bytes_valid() {
        let model = TestModel {
            name: "bytes_test".to_string(),
            values: vec![1.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("bytes_test.apr");

        save(
            &model,
            ModelType::LinearRegression,
            &path,
            SaveOptions::default(),
        )
        .expect("save");

        let data = std::fs::read(&path).expect("read file");
        let info = inspect_bytes(&data).expect("inspect bytes");
        assert_eq!(info.model_type, ModelType::LinearRegression);
    }

    #[test]
    fn test_inspect_bytes_too_small() {
        let data = vec![0u8; 10];
        let result = inspect_bytes(&data);
        assert!(result.is_err());
    }
}
