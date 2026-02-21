//\! Unit tests for the APR format module.

use super::super::*;

#[test]
fn test_magic_number() {
    assert_eq!(MAGIC, [0x41, 0x50, 0x52, 0x4E]);
    assert_eq!(&MAGIC, b"APRN");
}

#[test]
fn test_header_roundtrip() {
    let mut header = Header::new(ModelType::LinearRegression);
    header.metadata_size = 256;
    header.payload_size = 1024;
    header.uncompressed_size = 2048;
    header.compression = Compression::ZstdDefault;
    header.flags = Flags::new().with_signed();

    let bytes = header.to_bytes();
    assert_eq!(bytes.len(), HEADER_SIZE);

    let parsed = Header::from_bytes(&bytes).expect("valid header");
    assert_eq!(parsed.magic, MAGIC);
    assert_eq!(parsed.version, FORMAT_VERSION);
    assert_eq!(parsed.model_type, ModelType::LinearRegression);
    assert_eq!(parsed.metadata_size, 256);
    assert_eq!(parsed.payload_size, 1024);
    assert_eq!(parsed.uncompressed_size, 2048);
    assert_eq!(parsed.compression, Compression::ZstdDefault);
    assert!(parsed.flags.is_signed());
    assert!(!parsed.flags.is_encrypted());
}

#[test]
fn test_invalid_magic() {
    let mut bytes = [0u8; HEADER_SIZE];
    bytes[0..4].copy_from_slice(b"BAAD");

    let result = Header::from_bytes(&bytes);
    let err = result.expect_err("Should fail with invalid magic");
    assert!(err.to_string().contains("Invalid magic"));
}

#[test]
fn test_unsupported_version() {
    let mut header = Header::new(ModelType::LinearRegression);
    header.version = (99, 0); // Future version

    let mut bytes = header.to_bytes();
    bytes[4] = 99; // Major version

    let result = Header::from_bytes(&bytes);
    let err = result.expect_err("Should fail with unsupported version");
    assert!(err.to_string().contains("Unsupported"));
}

#[test]
fn test_compression_bomb_protection() {
    let mut header = Header::new(ModelType::LinearRegression);
    header.uncompressed_size = MAX_UNCOMPRESSED_SIZE + 1;

    let bytes = header.to_bytes();
    let result = Header::from_bytes(&bytes);
    let err = result.expect_err("Should fail with compression bomb protection");
    assert!(err.to_string().contains("compression bomb"));
}

#[test]
fn test_crc32() {
    // Known CRC32 value for "123456789"
    let data = b"123456789";
    let crc = crc32(data);
    assert_eq!(crc, 0xCBF43926);
}

#[test]
fn test_flags() {
    let flags = Flags::new()
        .with_encrypted()
        .with_signed()
        .with_streaming()
        .with_licensed()
        .with_trueno_native()
        .with_quantized();

    assert!(flags.is_encrypted());
    assert!(flags.is_signed());
    assert!(flags.is_streaming());
    assert!(flags.is_licensed());
    assert!(flags.is_trueno_native());
    assert!(flags.is_quantized());
    assert_eq!(flags.bits(), 0b0011_1111);
}

#[test]
fn test_model_type_roundtrip() {
    let types = [
        ModelType::LinearRegression,
        ModelType::LogisticRegression,
        ModelType::DecisionTree,
        ModelType::RandomForest,
        ModelType::KMeans,
        ModelType::NeuralSequential,
        ModelType::Custom,
    ];

    for model_type in types {
        let value = model_type as u16;
        let parsed = ModelType::from_u16(value).expect("valid type");
        assert_eq!(parsed, model_type);
    }
}

#[test]
fn test_save_load_simple() {
    use tempfile::tempdir;

    // Simple serializable struct for testing
    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestModel {
        weights: Vec<f32>,
        bias: f32,
    }

    let model = TestModel {
        weights: vec![1.0, 2.0, 3.0],
        bias: 0.5,
    };

    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("test.apr");

    // Save
    save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save should succeed");

    // Load
    let loaded: TestModel = load(&path, ModelType::Custom).expect("load should succeed");
    assert_eq!(loaded, model);

    // Inspect
    let info = inspect(&path).expect("inspect should succeed");
    assert_eq!(info.model_type, ModelType::Custom);
    assert_eq!(info.format_version, FORMAT_VERSION);
    assert!(!info.encrypted);
    assert!(!info.signed);
}

#[test]
fn test_load_from_bytes_roundtrip() {
    use tempfile::tempdir;

    // Simple serializable struct for testing
    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestModel {
        weights: Vec<f32>,
        bias: f32,
    }

    let model = TestModel {
        weights: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        bias: 0.5,
    };

    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("embedded.apr");

    // Save to file first
    save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save should succeed");

    // Read file into bytes (simulating include_bytes!)
    let data = std::fs::read(&path).expect("read file");

    // Load from bytes
    let loaded: TestModel =
        load_from_bytes(&data, ModelType::Custom).expect("load_from_bytes should succeed");
    assert_eq!(loaded, model);

    // Inspect from bytes
    let info = inspect_bytes(&data).expect("inspect_bytes should succeed");
    assert_eq!(info.model_type, ModelType::Custom);
    assert_eq!(info.format_version, FORMAT_VERSION);
}

#[test]
fn test_load_from_bytes_type_mismatch() {
    use tempfile::tempdir;

    #[derive(Debug, Serialize, Deserialize)]
    struct TestModel {
        value: i32,
    }

    let model = TestModel { value: 42 };
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("typed.apr");

    save(
        &model,
        ModelType::LinearRegression,
        &path,
        SaveOptions::default(),
    )
    .expect("save should succeed");

    let data = std::fs::read(&path).expect("read file");

    // Load with wrong type should fail
    let result: Result<TestModel> = load_from_bytes(&data, ModelType::KMeans);
    assert!(result.is_err());
    assert!(result
        .expect_err("should fail")
        .to_string()
        .contains("type mismatch"));
}

#[test]
fn test_load_from_bytes_checksum_failure() {
    use tempfile::tempdir;

    #[derive(Debug, Serialize, Deserialize)]
    struct TestModel {
        value: i32,
    }

    let model = TestModel { value: 42 };
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("corrupt.apr");

    save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save should succeed");

    // Read and corrupt the data
    let mut data = std::fs::read(&path).expect("read file");
    if data.len() > HEADER_SIZE + 10 {
        data[HEADER_SIZE + 5] ^= 0xFF; // Flip some bits
    }

    // Load should fail with checksum error
    let result: Result<TestModel> = load_from_bytes(&data, ModelType::Custom);
    assert!(result.is_err());
    assert!(result
        .expect_err("should fail")
        .to_string()
        .contains("Checksum"));
}

#[test]
fn test_load_from_bytes_too_small() {
    let data = vec![0u8; 10]; // Too small

    let result: Result<i32> = load_from_bytes(&data, ModelType::Custom);
    assert!(result.is_err());
    assert!(result
        .expect_err("should fail")
        .to_string()
        .contains("too small"));
}

#[test]
fn test_inspect_bytes_too_small() {
    let data = vec![0u8; 10]; // Too small

    let result = inspect_bytes(&data);
    assert!(result.is_err());
    assert!(result
        .expect_err("should fail")
        .to_string()
        .contains("too small"));
}

#[test]
fn test_checksum_corruption() {
    use tempfile::tempdir;

    #[derive(Debug, Serialize, Deserialize)]
    struct TestModel {
        value: i32,
    }

    let model = TestModel { value: 42 };
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("corrupt.apr");

    save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save should succeed");

    // Corrupt the file
    let mut content = std::fs::read(&path).expect("read file");
    if content.len() > HEADER_SIZE + 10 {
        content[HEADER_SIZE + 5] ^= 0xFF; // Flip some bits in metadata
    }
    std::fs::write(&path, &content).expect("write corrupted file");

    // Load should fail with checksum error
    let result: Result<TestModel> = load(&path, ModelType::Custom);
    let err = result.expect_err("Should fail with checksum error");
    assert!(err.to_string().contains("Checksum"));
}

#[test]
fn test_type_mismatch() {
    use tempfile::tempdir;

    #[derive(Debug, Serialize, Deserialize)]
    struct TestModel {
        value: i32,
    }

    let model = TestModel { value: 42 };
    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("typed.apr");

    save(
        &model,
        ModelType::LinearRegression,
        &path,
        SaveOptions::default(),
    )
    .expect("save should succeed");

    // Load with wrong type should fail
    let result: Result<TestModel> = load(&path, ModelType::KMeans);
    let err = result.expect_err("Should fail with type mismatch");
    assert!(err.to_string().contains("type mismatch"));
}

#[test]
#[cfg(feature = "format-compression")]
fn test_zstd_compression_roundtrip() {
    use tempfile::tempdir;

    // Model with repetitive data (compresses well)
    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct LargeModel {
        weights: Vec<f32>,
    }

    // 10,000 floats with repetitive pattern (compresses well)
    let model = LargeModel {
        weights: (0..10_000).map(|i| (i % 100) as f32).collect(),
    };

    let dir = tempdir().expect("create temp dir");
    let path = dir.path().join("compressed.apr");

    // Save with default zstd compression
    save(
        &model,
        ModelType::Custom,
        &path,
        SaveOptions::default().with_compression(Compression::ZstdDefault),
    )
    .expect("save should succeed");

    // Load and verify
    let loaded: LargeModel = load(&path, ModelType::Custom).expect("load should succeed");
    assert_eq!(loaded, model);

    // Verify compression reduced size
    let info = inspect(&path).expect("inspect should succeed");
    assert!(
        info.payload_size < info.uncompressed_size,
        "Compressed size {} should be less than uncompressed {}",
        info.payload_size,
        info.uncompressed_size
    );
}

#[test]
#[cfg(feature = "format-compression")]
fn test_zstd_max_compression() {
    use tempfile::tempdir;

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestModel {
        data: Vec<u8>,
    }

    // Highly compressible data (all zeros)
    let model = TestModel {
        data: vec![0u8; 50_000],
    };

    let dir = tempdir().expect("create temp dir");
    let path_default = dir.path().join("default.apr");
    let path_max = dir.path().join("max.apr");

    // Save with default compression
    save(
        &model,
        ModelType::Custom,
        &path_default,
        SaveOptions::default().with_compression(Compression::ZstdDefault),
    )
    .expect("save default should succeed");

    // Save with maximum compression
    save(
        &model,
        ModelType::Custom,
        &path_max,
        SaveOptions::default().with_compression(Compression::ZstdMax),
    )
    .expect("save max should succeed");

    // Both should load correctly
    let loaded_default: TestModel =
        load(&path_default, ModelType::Custom).expect("load default should succeed");
    let loaded_max: TestModel =
        load(&path_max, ModelType::Custom).expect("load max should succeed");

    assert_eq!(loaded_default, model);
    assert_eq!(loaded_max, model);

    // Max compression should be at least as small (often smaller)
    let info_default = inspect(&path_default).expect("inspect default");
    let info_max = inspect(&path_max).expect("inspect max");
    assert!(
        info_max.payload_size <= info_default.payload_size,
        "Max compression {} should be <= default {}",
        info_max.payload_size,
        info_default.payload_size
    );
}

#[test]
fn test_compression_fallback_without_feature() {
    // When feature is disabled, zstd requests should fall back to None
    let data = vec![1u8, 2, 3, 4, 5];
    let (compressed, actual_compression) =
        compress_payload(&data, Compression::ZstdDefault).expect("should fallback");

    #[cfg(not(feature = "format-compression"))]
    {
        assert_eq!(actual_compression, Compression::None);
        assert_eq!(compressed, data);
    }

    #[cfg(feature = "format-compression")]
    {
        assert_eq!(actual_compression, Compression::ZstdDefault);
        // Compressed data will be different (has zstd header)
        assert_ne!(compressed, data);
    }
}

include!("unit_compression_signing_encryption.rs");
include!("unit_encryption_distillation.rs");
include!("unit_distillation_gguf_mmap.rs");
include!("unit_core_io_pygmy.rs");
include!("unit_gguf_config.rs");
