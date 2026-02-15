
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

    // ============================================================================
    // load_from_bytes Tests (GH-219 coverage)
    // ============================================================================

    #[test]
    fn test_load_from_bytes_roundtrip() {
        let model = TestModel {
            name: "bytes_roundtrip".to_string(),
            values: vec![1.0, 2.0, 3.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("bytes_rt.apr");

        save(
            &model,
            ModelType::LinearRegression,
            &path,
            SaveOptions::default(),
        )
        .expect("save");

        let data = std::fs::read(&path).expect("read file");
        let loaded: TestModel =
            load_from_bytes(&data, ModelType::LinearRegression).expect("load from bytes");
        assert_eq!(model, loaded);
    }

    #[test]
    fn test_load_from_bytes_too_small() {
        let data = vec![0u8; 10];
        let result: Result<TestModel> = load_from_bytes(&data, ModelType::LinearRegression);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_from_bytes_wrong_model_type() {
        let model = TestModel {
            name: "wrong_type".to_string(),
            values: vec![1.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("wrong_type_bytes.apr");

        save(
            &model,
            ModelType::LinearRegression,
            &path,
            SaveOptions::default(),
        )
        .expect("save");

        let data = std::fs::read(&path).expect("read file");
        let result: Result<TestModel> = load_from_bytes(&data, ModelType::KMeans);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_from_bytes_corrupted_checksum() {
        let model = TestModel {
            name: "corrupt".to_string(),
            values: vec![1.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("corrupt_bytes.apr");

        save(
            &model,
            ModelType::LinearRegression,
            &path,
            SaveOptions::default(),
        )
        .expect("save");

        let mut data = std::fs::read(&path).expect("read file");
        // Corrupt a byte in the payload area
        if data.len() > HEADER_SIZE + 5 {
            data[HEADER_SIZE + 2] ^= 0xFF;
        }
        let result: Result<TestModel> = load_from_bytes(&data, ModelType::LinearRegression);
        assert!(result.is_err());
    }

    // ============================================================================
    // load_mmap and load_auto Tests (GH-219 coverage)
    // ============================================================================

    #[test]
    fn test_load_mmap_roundtrip() {
        let model = TestModel {
            name: "mmap_test".to_string(),
            values: vec![10.0, 20.0, 30.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("mmap_test.apr");

        save(
            &model,
            ModelType::LinearRegression,
            &path,
            SaveOptions::default(),
        )
        .expect("save");

        let loaded: TestModel =
            load_mmap(&path, ModelType::LinearRegression).expect("load mmap");
        assert_eq!(model, loaded);
    }

    #[test]
    fn test_load_auto_small_file() {
        let model = TestModel {
            name: "auto_small".to_string(),
            values: vec![1.0],
        };

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("auto_small.apr");

        save(
            &model,
            ModelType::LinearRegression,
            &path,
            SaveOptions::default(),
        )
        .expect("save");

        // Small file should use standard load path
        let loaded: TestModel =
            load_auto(&path, ModelType::LinearRegression).expect("load auto");
        assert_eq!(model, loaded);
    }

    // ============================================================================
    // ZstdMax Compression (GH-219 coverage)
    // ============================================================================

    #[cfg(feature = "format-compression")]
    #[test]
    fn test_compress_decompress_zstd_max() {
        let data = b"test data that should compress well with zstd max compression level";
        let (compressed, compression) =
            compress_payload(data, Compression::ZstdMax).expect("compress");
        assert_eq!(compression, Compression::ZstdMax);
        let decompressed = decompress_payload(&compressed, compression).expect("decompress");
        assert_eq!(decompressed, data);
    }

    // ============================================================================
    // Header Error Paths (GH-219 coverage)
    // ============================================================================

    #[test]
    fn test_header_roundtrip() {
        use super::super::{Header, ModelType};
        let mut header = Header::new(ModelType::RandomForest);
        header.metadata_size = 100;
        header.payload_size = 5000;
        header.uncompressed_size = 10000;
        header.compression = Compression::None;
        header.quality_score = 85;

        let bytes = header.to_bytes();
        let parsed = Header::from_bytes(&bytes).expect("parse header");

        assert_eq!(parsed.model_type, ModelType::RandomForest);
        assert_eq!(parsed.metadata_size, 100);
        assert_eq!(parsed.payload_size, 5000);
        assert_eq!(parsed.uncompressed_size, 10000);
        assert_eq!(parsed.compression, Compression::None);
        assert_eq!(parsed.quality_score, 85);
    }

    #[test]
    fn test_header_from_bytes_bad_magic() {
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(b"BAAD");
        let result = Header::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_from_bytes_too_short() {
        let bytes = [0u8; 10];
        let result = Header::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_from_bytes_unsupported_version() {
        use super::super::MAGIC;
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = 255; // Future major version
        bytes[5] = 0;
        // Set a valid model type
        bytes[6..8].copy_from_slice(&(ModelType::Custom as u16).to_le_bytes());
        let result = Header::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_from_bytes_unknown_model_type() {
        use super::super::{FORMAT_VERSION, MAGIC};
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = FORMAT_VERSION.0;
        bytes[5] = FORMAT_VERSION.1;
        bytes[6..8].copy_from_slice(&0xFFFEu16.to_le_bytes()); // Unknown model type
        let result = Header::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_from_bytes_unknown_compression() {
        use super::super::{FORMAT_VERSION, MAGIC};
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = FORMAT_VERSION.0;
        bytes[5] = FORMAT_VERSION.1;
        bytes[6..8].copy_from_slice(&(ModelType::Custom as u16).to_le_bytes());
        bytes[20] = 0xEE; // Unknown compression
        let result = Header::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_header_compression_bomb_protection() {
        use super::super::{FORMAT_VERSION, MAGIC, MAX_UNCOMPRESSED_SIZE};
        let mut bytes = [0u8; HEADER_SIZE];
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = FORMAT_VERSION.0;
        bytes[5] = FORMAT_VERSION.1;
        bytes[6..8].copy_from_slice(&(ModelType::Custom as u16).to_le_bytes());
        // Set uncompressed_size to exceed max
        let bomb_size = MAX_UNCOMPRESSED_SIZE + 1;
        bytes[16..20].copy_from_slice(&bomb_size.to_le_bytes());
        let result = Header::from_bytes(&bytes);
        assert!(result.is_err());
    }

    // ============================================================================
    // Flags Tests (GH-219 coverage)
    // ============================================================================

    #[test]
    fn test_flags_all_individual() {
        use super::super::Flags;

        let f = Flags::new()
            .with_encrypted()
            .with_signed()
            .with_streaming()
            .with_licensed()
            .with_trueno_native()
            .with_quantized()
            .with_model_card();

        assert!(f.is_encrypted());
        assert!(f.is_signed());
        assert!(f.is_streaming());
        assert!(f.is_licensed());
        assert!(f.is_trueno_native());
        assert!(f.is_quantized());
        assert!(f.has_model_card());
    }

    #[test]
    fn test_flags_default_empty() {
        use super::super::Flags;

        let f = Flags::new();
        assert!(!f.is_encrypted());
        assert!(!f.is_signed());
        assert!(!f.is_streaming());
        assert!(!f.is_licensed());
        assert!(!f.is_trueno_native());
        assert!(!f.is_quantized());
        assert!(!f.has_model_card());
        assert_eq!(f.bits(), 0);
    }

    #[test]
    fn test_flags_from_bits_masks_reserved() {
        use super::super::Flags;
        // Bit 7 is reserved and should be masked
        let f = Flags::from_bits(0xFF);
        assert_eq!(f.bits(), 0x7F);
    }

    // ============================================================================
    // ModelType / Compression from_u16/from_u8 (GH-219 coverage)
    // ============================================================================

    #[test]
    fn test_model_type_from_u16_all_variants() {
        assert_eq!(ModelType::from_u16(0x0001), Some(ModelType::LinearRegression));
        assert_eq!(ModelType::from_u16(0x0002), Some(ModelType::LogisticRegression));
        assert_eq!(ModelType::from_u16(0x0003), Some(ModelType::DecisionTree));
        assert_eq!(ModelType::from_u16(0x0004), Some(ModelType::RandomForest));
        assert_eq!(ModelType::from_u16(0x0005), Some(ModelType::GradientBoosting));
        assert_eq!(ModelType::from_u16(0x0006), Some(ModelType::KMeans));
        assert_eq!(ModelType::from_u16(0x0007), Some(ModelType::Pca));
        assert_eq!(ModelType::from_u16(0x0008), Some(ModelType::NaiveBayes));
        assert_eq!(ModelType::from_u16(0x0009), Some(ModelType::Knn));
        assert_eq!(ModelType::from_u16(0x000A), Some(ModelType::Svm));
        assert_eq!(ModelType::from_u16(0x0010), Some(ModelType::NgramLm));
        assert_eq!(ModelType::from_u16(0x0011), Some(ModelType::Tfidf));
        assert_eq!(ModelType::from_u16(0x0012), Some(ModelType::CountVectorizer));
        assert_eq!(ModelType::from_u16(0x0020), Some(ModelType::NeuralSequential));
        assert_eq!(ModelType::from_u16(0x0021), Some(ModelType::NeuralCustom));
        assert_eq!(ModelType::from_u16(0x0030), Some(ModelType::ContentRecommender));
        assert_eq!(ModelType::from_u16(0x0040), Some(ModelType::MixtureOfExperts));
        assert_eq!(ModelType::from_u16(0x00FF), Some(ModelType::Custom));
        assert_eq!(ModelType::from_u16(0x9999), None);
    }

    #[test]
    fn test_compression_from_u8_all_variants() {
        assert_eq!(Compression::from_u8(0x00), Some(Compression::None));
        assert_eq!(Compression::from_u8(0x01), Some(Compression::ZstdDefault));
        assert_eq!(Compression::from_u8(0x02), Some(Compression::ZstdMax));
        assert_eq!(Compression::from_u8(0x03), Some(Compression::Lz4));
        assert_eq!(Compression::from_u8(0xFF), None);
    }

    // ============================================================================
    // SaveOptions Builder (GH-219 coverage)
    // ============================================================================

    #[test]
    fn test_save_options_builder() {
        let opts = SaveOptions::new()
            .with_name("test-model")
            .with_description("A description")
            .with_compression(Compression::None)
            .with_quality_score(95);

        assert_eq!(opts.metadata.model_name, Some("test-model".to_string()));
        assert_eq!(
            opts.metadata.description,
            Some("A description".to_string())
        );
        assert_eq!(opts.compression, Compression::None);
        assert_eq!(opts.quality_score, Some(95));
    }

    #[test]
    fn test_save_options_with_license() {
        use super::super::{LicenseInfo, LicenseTier};

        let license = LicenseInfo {
            uuid: "uuid-123".to_string(),
            hash: "hash-abc".to_string(),
            expiry: Some("2027-01-01".to_string()),
            seats: Some(10),
            licensee: Some("PAIML".to_string()),
            tier: LicenseTier::Team,
        };
        let opts = SaveOptions::new().with_license(license);
        assert!(opts.metadata.license.is_some());
        let lic = opts.metadata.license.unwrap();
        assert_eq!(lic.tier, LicenseTier::Team);
        assert_eq!(lic.seats, Some(10));
    }

    // ============================================================================
    // inspect_bytes metadata boundary (GH-219 coverage)
    // ============================================================================

    #[test]
    fn test_inspect_bytes_metadata_beyond_boundary() {
        use super::super::{FORMAT_VERSION, MAGIC};
        // Create a header claiming huge metadata_size
        let mut bytes = vec![0u8; HEADER_SIZE + 4]; // minimal: header + checksum
        bytes[0..4].copy_from_slice(&MAGIC);
        bytes[4] = FORMAT_VERSION.0;
        bytes[5] = FORMAT_VERSION.1;
        bytes[6..8].copy_from_slice(&(ModelType::Custom as u16).to_le_bytes());
        // metadata_size = 10000 (way beyond the data)
        bytes[8..12].copy_from_slice(&10000u32.to_le_bytes());
        let result = inspect_bytes(&bytes);
        assert!(result.is_err());
    }

    // ============================================================================
    // load payload_end boundary check (GH-219 coverage)
    // ============================================================================

    #[test]
    fn test_load_payload_beyond_boundary() {
        use super::super::{FORMAT_VERSION, MAGIC};
        // Create a file with header claiming payload extends beyond file
        let mut content = vec![0u8; HEADER_SIZE + 10];
        content[0..4].copy_from_slice(&MAGIC);
        content[4] = FORMAT_VERSION.0;
        content[5] = FORMAT_VERSION.1;
        content[6..8].copy_from_slice(&(ModelType::Custom as u16).to_le_bytes());
        // metadata_size = 0, payload_size = 99999
        content[12..16].copy_from_slice(&99999u32.to_le_bytes());
        // Append valid checksum
        let checksum = crc32(&content);
        content.extend_from_slice(&checksum.to_le_bytes());

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("bad_boundary.apr");
        std::fs::write(&path, &content).expect("write");

        let result: Result<TestModel> = load(&path, ModelType::Custom);
        assert!(result.is_err());
    }

    // ============================================================================
    // Metadata default (GH-219 coverage)
    // ============================================================================

    #[test]
    fn test_metadata_default() {
        let m = Metadata::default();
        assert!(m.model_name.is_none());
        assert!(m.description.is_none());
        assert!(m.training.is_none());
        assert!(m.hyperparameters.is_empty());
        assert!(m.metrics.is_empty());
        assert!(m.custom.is_empty());
        assert!(m.distillation.is_none());
        assert!(m.distillation_info.is_none());
        assert!(m.license.is_none());
        assert!(m.model_card.is_none());
        assert!(!m.created_at.is_empty());
    }
}
