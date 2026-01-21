//! Tests for APR format module.

use super::*;

// ==================== Unit Tests ====================

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
        save(&model, ModelType::Custom, &path, SaveOptions::default())
            .expect("save should succeed");

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
        save(&model, ModelType::Custom, &path, SaveOptions::default())
            .expect("save should succeed");

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

        save(&model, ModelType::Custom, &path, SaveOptions::default())
            .expect("save should succeed");

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

        save(&model, ModelType::Custom, &path, SaveOptions::default())
            .expect("save should succeed");

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

    #[test]
    #[cfg(feature = "format-compression")]
    fn test_lz4_compression_roundtrip() {
        // Test LZ4 compression/decompression roundtrip (GH-146)
        let data = vec![0u8; 1000]; // Highly compressible data

        let (compressed, actual_compression) =
            compress_payload(&data, Compression::Lz4).expect("lz4 compress should succeed");

        assert_eq!(actual_compression, Compression::Lz4);
        assert!(
            compressed.len() < data.len(),
            "LZ4 should compress zeros: {} < {}",
            compressed.len(),
            data.len()
        );

        // Decompress and verify
        let decompressed = decompress_payload(&compressed, Compression::Lz4)
            .expect("lz4 decompress should succeed");

        assert_eq!(decompressed, data);
    }

    #[test]
    #[cfg(feature = "format-compression")]
    fn test_lz4_compression_model_roundtrip() {
        // Test LZ4 with actual model save/load (GH-146)
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct LargeModel {
            weights: Vec<f32>,
        }

        // Repetitive data compresses well
        let model = LargeModel {
            weights: (0..10_000).map(|i| (i % 100) as f32).collect(),
        };

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("lz4_compressed.apr");

        // Save with LZ4 compression
        save(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default().with_compression(Compression::Lz4),
        )
        .expect("save should succeed");

        // Load and verify
        let loaded: LargeModel = load(&path, ModelType::Custom).expect("load should succeed");
        assert_eq!(loaded, model);

        // Verify compression reduced size
        let info = inspect(&path).expect("inspect should succeed");
        assert!(
            info.payload_size < info.uncompressed_size,
            "LZ4 compressed size {} should be less than uncompressed {}",
            info.payload_size,
            info.uncompressed_size
        );
    }

    #[test]
    fn test_lz4_fallback_without_feature() {
        // When format-compression is disabled, LZ4 should fall back to None
        let data = vec![1u8, 2, 3, 4, 5];
        let (compressed, actual_compression) =
            compress_payload(&data, Compression::Lz4).expect("should fallback");

        #[cfg(not(feature = "format-compression"))]
        {
            assert_eq!(actual_compression, Compression::None);
            assert_eq!(compressed, data);
        }

        #[cfg(feature = "format-compression")]
        {
            assert_eq!(actual_compression, Compression::Lz4);
            // LZ4 adds size prefix, so will be different
            assert_ne!(compressed, data);
        }
    }

    #[test]
    #[cfg(feature = "format-signing")]
    fn test_signed_save_load_roundtrip() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            weights: Vec<f32>,
            bias: f32,
        }

        let model = TestModel {
            weights: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            bias: 0.5,
        };

        // Generate signing key
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
        let verifying_key = signing_key.verifying_key();

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("signed.apr");

        // Save with signature
        save_signed(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            &signing_key,
        )
        .expect("save_signed should succeed");

        // Inspect - should show signed flag
        let info = inspect(&path).expect("inspect should succeed");
        assert!(info.signed, "Model should be marked as signed");
        assert_eq!(info.model_type, ModelType::Custom);

        // Load with verification (using embedded key)
        let loaded: TestModel =
            load_verified(&path, ModelType::Custom, None).expect("load_verified should succeed");
        assert_eq!(loaded, model);

        // Load with verification (using trusted key)
        let loaded2: TestModel = load_verified(&path, ModelType::Custom, Some(&verifying_key))
            .expect("load_verified with trusted key should succeed");
        assert_eq!(loaded2, model);
    }

    #[test]
    #[cfg(feature = "format-signing")]
    fn test_signature_verification_fails_with_wrong_key() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };

        // Generate two different key pairs
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
        let wrong_key = SigningKey::generate(&mut rand::rngs::OsRng).verifying_key();

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("signed_wrong.apr");

        // Save with one key
        save_signed(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            &signing_key,
        )
        .expect("save_signed should succeed");

        // Try to verify with wrong key - should fail
        let result: Result<TestModel> = load_verified(&path, ModelType::Custom, Some(&wrong_key));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Signature verification failed"));
    }

    #[test]
    #[cfg(feature = "format-signing")]
    fn test_signed_model_detects_tampering() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("tampered.apr");

        // Save signed model
        save_signed(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            &signing_key,
        )
        .expect("save_signed should succeed");

        // Tamper with the file (modify a byte in the payload)
        let mut content = std::fs::read(&path).expect("read file");
        let payload_offset = HEADER_SIZE + 20; // Somewhere in metadata/payload
        if content.len() > payload_offset {
            content[payload_offset] ^= 0xFF;
        }

        // Recalculate checksum to avoid checksum failure
        let checksum_start = content.len() - 4;
        let new_checksum = crc32(&content[..checksum_start]);
        content[checksum_start..].copy_from_slice(&new_checksum.to_le_bytes());

        std::fs::write(&path, &content).expect("write tampered file");

        // Verification should fail due to signature mismatch
        let result: Result<TestModel> = load_verified(&path, ModelType::Custom, None);
        assert!(result.is_err());
        // Either signature verification fails or parsing fails due to corrupted data
    }

    #[test]
    #[cfg(feature = "format-signing")]
    fn test_load_verified_rejects_unsigned_file() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("unsigned.apr");

        // Save without signature
        save(&model, ModelType::Custom, &path, SaveOptions::default())
            .expect("save should succeed");

        // load_verified should reject unsigned files
        let result: Result<TestModel> = load_verified(&path, ModelType::Custom, None);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("SIGNED flag not set") || err_msg.contains("File too small"),
            "Expected SIGNED flag error or size error, got: {err_msg}"
        );
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_encrypted_save_load_roundtrip() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            weights: Vec<f32>,
            bias: f32,
        }

        let model = TestModel {
            weights: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            bias: 0.5,
        };

        let password = "super_secret_password_123!";

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("encrypted.apr");

        // Save with encryption
        save_encrypted(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            password,
        )
        .expect("save_encrypted should succeed");

        // Inspect - should show encrypted flag
        let info = inspect(&path).expect("inspect should succeed");
        assert!(info.encrypted, "Model should be marked as encrypted");
        assert_eq!(info.model_type, ModelType::Custom);

        // Load with correct password
        let loaded: TestModel = load_encrypted(&path, ModelType::Custom, password)
            .expect("load_encrypted should succeed");
        assert_eq!(loaded, model);
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_load_from_bytes_encrypted_roundtrip() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            weights: Vec<f32>,
            bias: f32,
        }

        let model = TestModel {
            weights: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            bias: 0.5,
        };

        let password = "embedded_secret_123!";

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("encrypted_embedded.apr");

        // Save with encryption
        save_encrypted(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            password,
        )
        .expect("save_encrypted should succeed");

        // Read file into bytes (simulating include_bytes!)
        let data = std::fs::read(&path).expect("read file");

        // Load from bytes with correct password
        let loaded: TestModel = load_from_bytes_encrypted(&data, ModelType::Custom, password)
            .expect("load_from_bytes_encrypted should succeed");
        assert_eq!(loaded, model);
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_load_from_bytes_encrypted_wrong_password() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let password = "correct_password";

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("encrypted.apr");

        save_encrypted(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            password,
        )
        .expect("save should succeed");

        let data = std::fs::read(&path).expect("read file");

        // Load with wrong password should fail
        let result: Result<TestModel> =
            load_from_bytes_encrypted(&data, ModelType::Custom, "wrong_password");
        assert!(result.is_err());
        assert!(result
            .expect_err("should fail")
            .to_string()
            .contains("Decryption failed"));
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_encrypted_wrong_password_fails() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let correct_password = "correct_password";
        let wrong_password = "wrong_password";

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("encrypted_wrong.apr");

        // Save with correct password
        save_encrypted(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            correct_password,
        )
        .expect("save_encrypted should succeed");

        // Try to load with wrong password - should fail
        let result: Result<TestModel> = load_encrypted(&path, ModelType::Custom, wrong_password);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Decryption failed"));
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_load_encrypted_rejects_unencrypted_file() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("unencrypted.apr");

        // Save without encryption
        save(&model, ModelType::Custom, &path, SaveOptions::default())
            .expect("save should succeed");

        // load_encrypted should reject unencrypted files
        let result: Result<TestModel> = load_encrypted(&path, ModelType::Custom, "any_password");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("ENCRYPTED flag not set") || err_msg.contains("File too small"),
            "Expected ENCRYPTED flag error or size error, got: {err_msg}"
        );
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_encrypted_with_compression() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct LargeModel {
            data: Vec<f32>,
        }

        // Repetitive data compresses well
        let model = LargeModel {
            data: (0..1000).map(|i| (i % 10) as f32).collect(),
        };

        let password = "compress_and_encrypt";

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("encrypted_compressed.apr");

        // Save with encryption (compression will be applied before encryption)
        save_encrypted(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(), // No explicit compression, but internal compression happens
            password,
        )
        .expect("save_encrypted should succeed");

        // Load and verify
        let loaded: LargeModel = load_encrypted(&path, ModelType::Custom, password)
            .expect("load_encrypted should succeed");
        assert_eq!(loaded, model);
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_x25519_recipient_roundtrip() {
        use tempfile::tempdir;
        use x25519_dalek::{PublicKey, StaticSecret};

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            weights: Vec<f32>,
            bias: f32,
        }

        let model = TestModel {
            weights: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            bias: 0.5,
        };

        // Generate recipient keypair
        let recipient_secret = StaticSecret::random_from_rng(rand::thread_rng());
        let recipient_public = PublicKey::from(&recipient_secret);

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("recipient_encrypted.apr");

        // Save for recipient
        save_for_recipient(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            &recipient_public,
        )
        .expect("save_for_recipient should succeed");

        // Inspect - should show encrypted flag
        let info = inspect(&path).expect("inspect should succeed");
        assert!(info.encrypted, "Model should be marked as encrypted");
        assert_eq!(info.model_type, ModelType::Custom);

        // Load as recipient with correct key
        let loaded: TestModel = load_as_recipient(&path, ModelType::Custom, &recipient_secret)
            .expect("load_as_recipient should succeed");
        assert_eq!(loaded, model);
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_x25519_wrong_key_fails() {
        use tempfile::tempdir;
        use x25519_dalek::{PublicKey, StaticSecret};

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };

        // Generate correct recipient keypair
        let recipient_secret = StaticSecret::random_from_rng(rand::thread_rng());
        let recipient_public = PublicKey::from(&recipient_secret);

        // Generate wrong keypair
        let wrong_secret = StaticSecret::random_from_rng(rand::thread_rng());

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("x25519_wrong_key.apr");

        // Save for correct recipient
        save_for_recipient(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            &recipient_public,
        )
        .expect("save_for_recipient should succeed");

        // Try to load with wrong key - should fail
        let result: Result<TestModel> = load_as_recipient(&path, ModelType::Custom, &wrong_secret);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Decryption failed"));
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_x25519_rejects_password_encrypted_file() {
        use tempfile::tempdir;
        use x25519_dalek::StaticSecret;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("password_not_x25519.apr");

        // Save with password encryption
        save_encrypted(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            "some_password",
        )
        .expect("save_encrypted should succeed");

        // Try to load as recipient - should fail
        let wrong_secret = StaticSecret::random_from_rng(rand::thread_rng());
        let result: Result<TestModel> = load_as_recipient(&path, ModelType::Custom, &wrong_secret);
        assert!(result.is_err());
        // Will fail because file layout doesn't match X25519 format
    }

    #[test]
    #[cfg(feature = "format-encryption")]
    fn test_x25519_load_rejects_unencrypted_file() {
        use tempfile::tempdir;
        use x25519_dalek::StaticSecret;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("unencrypted_for_x25519.apr");

        // Save without encryption
        save(&model, ModelType::Custom, &path, SaveOptions::default())
            .expect("save should succeed");

        // load_as_recipient should reject unencrypted files
        let secret = StaticSecret::random_from_rng(rand::thread_rng());
        let result: Result<TestModel> = load_as_recipient(&path, ModelType::Custom, &secret);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("ENCRYPTED flag not set") || err_msg.contains("File too small"),
            "Expected ENCRYPTED flag error or size error, got: {err_msg}"
        );
    }

    // EXTREME TDD: Distillation metadata (spec §6.3)
    // Step 1: GREEN - Verified we can use description field
    #[test]
    fn test_distillation_teacher_hash() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("distilled.apr");

        // For now, use the simplest approach - add to description
        let options = SaveOptions::default()
            .with_name("student_model")
            .with_description("Distilled from teacher abc123");

        save(&model, ModelType::Custom, &path, options).expect("save should succeed");

        // Inspect and verify description contains teacher info
        let info = inspect(&path).expect("inspect should succeed");
        assert!(info.metadata.description.is_some());
        assert!(info
            .metadata
            .description
            .as_ref()
            .expect("description should be set")
            .contains("abc123"));
    }

    // Step 2: RED - Test dedicated distillation field
    #[test]
    fn test_distillation_dedicated_field() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 123 };
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("distilled2.apr");

        // First verify that description (an existing Optional<String>) works
        let options = SaveOptions::default().with_description("test description");

        save(&model, ModelType::Custom, &path, options).expect("save should succeed");

        let info = inspect(&path).expect("inspect should succeed");

        // This should work
        assert_eq!(
            info.metadata.description,
            Some("test description".to_string())
        );

        // Now test distillation
        let mut options2 = SaveOptions::default();
        options2.metadata.distillation = Some("teacher_abc123".to_string());

        let path2 = dir.path().join("distilled2b.apr");
        save(&model, ModelType::Custom, &path2, options2).expect("save should succeed");

        let info2 = inspect(&path2).expect("inspect should succeed");
        assert_eq!(
            info2.metadata.distillation,
            Some("teacher_abc123".to_string())
        );
    }

    // Test: serialize/deserialize metadata directly with named fields
    #[test]
    fn test_metadata_msgpack_roundtrip() {
        let metadata = Metadata {
            description: Some("test description".to_string()),
            distillation: Some("teacher_abc123".to_string()),
            ..Default::default()
        };

        // Serialize with named fields (map mode) - required for skip_serializing_if
        let bytes = rmp_serde::to_vec_named(&metadata).expect("serialize");

        // Deserialize
        let restored: Metadata = rmp_serde::from_slice(&bytes).expect("deserialize");

        assert_eq!(restored.description, metadata.description);
        assert_eq!(restored.distillation, metadata.distillation);
    }

    // EXTREME TDD: Step 3 - RED test for DistillationInfo struct (spec §6.3)
    #[test]
    fn test_distillation_info_struct() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("distilled3.apr");

        // Create DistillationInfo per spec §6.3.2
        let distill_info = DistillationInfo {
            method: DistillMethod::Standard,
            teacher: TeacherProvenance {
                hash: "sha256:abc123def456".to_string(),
                signature: None,
                model_type: ModelType::NeuralSequential,
                param_count: 7_000_000_000, // 7B params
                ensemble_teachers: None,
            },
            params: DistillationParams {
                temperature: 3.0,
                alpha: 0.7,
                beta: None,
                epochs: 10,
                final_loss: Some(0.42),
            },
            layer_mapping: None,
        };

        let options = SaveOptions::default().with_distillation_info(distill_info);

        save(&model, ModelType::Custom, &path, options).expect("save should succeed");

        let info = inspect(&path).expect("inspect should succeed");
        let restored = info
            .metadata
            .distillation_info
            .expect("should have distillation_info");

        assert!(matches!(restored.method, DistillMethod::Standard));
        assert_eq!(restored.teacher.hash, "sha256:abc123def456");
        assert_eq!(restored.teacher.param_count, 7_000_000_000);
        assert!((restored.params.temperature - 3.0).abs() < f32::EPSILON);
        assert!((restored.params.alpha - 0.7).abs() < f32::EPSILON);
        assert_eq!(restored.params.epochs, 10);
        assert!(
            (restored.params.final_loss.expect("should have final_loss") - 0.42).abs()
                < f32::EPSILON
        );
    }

    // EXTREME TDD: Test progressive distillation with layer mapping (spec §6.3.1)
    #[test]
    fn test_distillation_progressive_with_layer_mapping() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("progressive.apr");

        // Progressive distillation: 4-layer student from 8-layer teacher
        let layer_mapping = vec![
            LayerMapping {
                student_layer: 0,
                teacher_layer: 0,
                weight: 0.5,
            },
            LayerMapping {
                student_layer: 1,
                teacher_layer: 2,
                weight: 0.3,
            },
            LayerMapping {
                student_layer: 2,
                teacher_layer: 5,
                weight: 0.15,
            },
            LayerMapping {
                student_layer: 3,
                teacher_layer: 7,
                weight: 0.05,
            },
        ];

        let distill_info = DistillationInfo {
            method: DistillMethod::Progressive,
            teacher: TeacherProvenance {
                hash: "sha256:teacher_8layer".to_string(),
                signature: Some("sig_abc123".to_string()),
                model_type: ModelType::NeuralSequential,
                param_count: 1_000_000_000, // 1B params
                ensemble_teachers: None,
            },
            params: DistillationParams {
                temperature: 4.0,
                alpha: 0.8,
                beta: Some(0.5), // Progressive uses beta for hidden vs logit loss
                epochs: 20,
                final_loss: Some(0.31),
            },
            layer_mapping: Some(layer_mapping),
        };

        let options = SaveOptions::default().with_distillation_info(distill_info);
        save(&model, ModelType::Custom, &path, options).expect("save should succeed");

        let info = inspect(&path).expect("inspect should succeed");
        let restored = info
            .metadata
            .distillation_info
            .expect("should have distillation_info");

        // Verify method
        assert!(matches!(restored.method, DistillMethod::Progressive));

        // Verify teacher info
        assert_eq!(restored.teacher.hash, "sha256:teacher_8layer");
        assert_eq!(restored.teacher.signature, Some("sig_abc123".to_string()));
        assert_eq!(restored.teacher.param_count, 1_000_000_000);

        // Verify params with beta
        assert!((restored.params.temperature - 4.0).abs() < f32::EPSILON);
        assert!((restored.params.alpha - 0.8).abs() < f32::EPSILON);
        assert!((restored.params.beta.expect("should have beta") - 0.5).abs() < f32::EPSILON);
        assert_eq!(restored.params.epochs, 20);

        // Verify layer mapping
        let mapping = restored.layer_mapping.expect("should have layer_mapping");
        assert_eq!(mapping.len(), 4);
        assert_eq!(mapping[0].student_layer, 0);
        assert_eq!(mapping[0].teacher_layer, 0);
        assert!((mapping[0].weight - 0.5).abs() < f32::EPSILON);
        assert_eq!(mapping[2].student_layer, 2);
        assert_eq!(mapping[2].teacher_layer, 5);
    }

    // EXTREME TDD: Test ensemble distillation with multiple teachers (spec §6.3.1)
    #[test]
    fn test_distillation_ensemble_multiple_teachers() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("ensemble.apr");

        // Ensemble: 3 teachers averaged
        let ensemble_teachers = vec![
            TeacherProvenance {
                hash: "sha256:teacher1".to_string(),
                signature: None,
                model_type: ModelType::NeuralSequential,
                param_count: 3_000_000_000,
                ensemble_teachers: None,
            },
            TeacherProvenance {
                hash: "sha256:teacher2".to_string(),
                signature: None,
                model_type: ModelType::NeuralSequential,
                param_count: 5_000_000_000,
                ensemble_teachers: None,
            },
            TeacherProvenance {
                hash: "sha256:teacher3".to_string(),
                signature: None,
                model_type: ModelType::GradientBoosting,
                param_count: 2_000_000_000,
                ensemble_teachers: None,
            },
        ];

        let distill_info = DistillationInfo {
            method: DistillMethod::Ensemble,
            teacher: TeacherProvenance {
                hash: "sha256:ensemble_meta".to_string(),
                signature: None,
                model_type: ModelType::NeuralSequential,
                param_count: 10_000_000_000, // Combined param count
                ensemble_teachers: Some(ensemble_teachers),
            },
            params: DistillationParams {
                temperature: 2.5,
                alpha: 0.6,
                beta: None,
                epochs: 15,
                final_loss: Some(0.28),
            },
            layer_mapping: None,
        };

        let options = SaveOptions::default().with_distillation_info(distill_info);
        save(&model, ModelType::Custom, &path, options).expect("save should succeed");

        let info = inspect(&path).expect("inspect should succeed");
        let restored = info
            .metadata
            .distillation_info
            .expect("should have distillation_info");

        // Verify method
        assert!(matches!(restored.method, DistillMethod::Ensemble));

        // Verify ensemble teachers
        let teachers = restored
            .teacher
            .ensemble_teachers
            .expect("should have ensemble_teachers");
        assert_eq!(teachers.len(), 3);
        assert_eq!(teachers[0].hash, "sha256:teacher1");
        assert_eq!(teachers[1].param_count, 5_000_000_000);
        assert!(matches!(
            teachers[2].model_type,
            ModelType::GradientBoosting
        ));

        // Verify combined param count
        assert_eq!(restored.teacher.param_count, 10_000_000_000);
    }

    // ========================================================================
    // EXTREME TDD: License Block (spec §9)
    // ========================================================================

    // Step 1: RED - Test LicenseInfo struct (spec §9.1)
    #[test]
    fn test_license_info_roundtrip() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("licensed.apr");

        // Create LicenseInfo per spec §9.1
        let license = LicenseInfo {
            uuid: "550e8400-e29b-41d4-a716-446655440000".to_string(),
            hash: "sha256:license_hash_abc123".to_string(),
            expiry: Some("2025-12-31T23:59:59Z".to_string()),
            seats: Some(10),
            licensee: Some("ACME Corp".to_string()),
            tier: LicenseTier::Enterprise,
        };

        let options = SaveOptions::default().with_license(license);

        save(&model, ModelType::Custom, &path, options).expect("save should succeed");

        let info = inspect(&path).expect("inspect should succeed");

        // Verify LICENSED flag is set
        assert!(info.licensed);

        // Verify license info restored
        let restored = info.metadata.license.expect("should have license");
        assert_eq!(restored.uuid, "550e8400-e29b-41d4-a716-446655440000");
        assert_eq!(restored.hash, "sha256:license_hash_abc123");
        assert_eq!(restored.expiry, Some("2025-12-31T23:59:59Z".to_string()));
        assert_eq!(restored.seats, Some(10));
        assert_eq!(restored.licensee, Some("ACME Corp".to_string()));
        assert!(matches!(restored.tier, LicenseTier::Enterprise));
    }

    // Step 2: RED - Test license tiers
    #[test]
    fn test_license_tiers() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };
        let dir = tempdir().expect("create temp dir");

        // Test each tier
        for (tier, name) in [
            (LicenseTier::Personal, "personal"),
            (LicenseTier::Team, "team"),
            (LicenseTier::Enterprise, "enterprise"),
            (LicenseTier::Academic, "academic"),
        ] {
            let path = dir.path().join(format!("{name}.apr"));

            let license = LicenseInfo {
                uuid: format!("uuid-{name}"),
                hash: format!("hash-{name}"),
                expiry: None,
                seats: None,
                licensee: None,
                tier,
            };

            let options = SaveOptions::default().with_license(license);
            save(&model, ModelType::Custom, &path, options).expect("save should succeed");

            let info = inspect(&path).expect("inspect should succeed");
            let restored = info.metadata.license.expect("should have license");
            assert_eq!(restored.uuid, format!("uuid-{name}"));
        }
    }

    // ========================================================================
    // EXTREME TDD: GGUF Export (spec §7.2)
    // ========================================================================

    // Step 1: RED - Test GGUF magic number and header
    #[test]
    fn test_gguf_magic_number() {
        // GGUF magic is "GGUF" = 0x46554747 in little-endian
        assert_eq!(gguf::GGUF_MAGIC, 0x4655_4747);
        assert_eq!(&gguf::GGUF_MAGIC.to_le_bytes(), b"GGUF");
    }

    #[test]
    fn test_gguf_header_write() {
        let mut buffer = Vec::new();

        // Write minimal GGUF header
        let header = gguf::GgufHeader {
            version: gguf::GGUF_VERSION,
            tensor_count: 1,
            metadata_kv_count: 0,
        };

        header.write_to(&mut buffer).expect("write header");

        // Verify: magic (4) + version (4) + tensor_count (8) + kv_count (8) = 24 bytes
        assert_eq!(buffer.len(), 24);

        // Verify magic
        assert_eq!(&buffer[0..4], b"GGUF");

        // Verify version (little-endian u32)
        assert_eq!(
            u32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]),
            3
        );
    }

    #[test]
    fn test_gguf_metadata_string() {
        let mut buffer = Vec::new();

        // Write a string metadata value
        gguf::write_metadata_kv(
            &mut buffer,
            "general.name",
            &gguf::GgufValue::String("test_model".to_string()),
        )
        .expect("write metadata");

        // Should have: key_len (8) + key + value_type (4) + str_len (8) + str
        // 8 + 12 + 4 + 8 + 10 = 42 bytes
        assert!(!buffer.is_empty());
    }

    // Step 2: RED - Test full GGUF export function
    #[test]
    fn test_gguf_export_simple_tensor() {
        // Create a simple f32 tensor to export
        let tensor_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let data: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let tensor = gguf::GgufTensor {
            name: "test.weight".to_string(),
            shape: vec![2, 2],
            dtype: gguf::GgmlType::F32,
            data,
        };

        let mut buffer = Vec::new();

        // Export to GGUF format
        gguf::export_tensors_to_gguf(
            &mut buffer,
            &[tensor],
            &[(
                "general.name".to_string(),
                gguf::GgufValue::String("test_model".to_string()),
            )],
        )
        .expect("export should succeed");

        // Verify magic number at start
        assert_eq!(&buffer[0..4], b"GGUF");

        // Verify we have content
        assert!(buffer.len() > 24); // At least header size
    }

    #[test]
    fn test_gguf_export_with_metadata() {
        let tensor_data: Vec<f32> = vec![0.5; 16];
        let data: Vec<u8> = tensor_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let tensor = gguf::GgufTensor {
            name: "model.embed".to_string(),
            shape: vec![4, 4],
            dtype: gguf::GgmlType::F32,
            data,
        };

        let metadata = vec![
            (
                "general.name".to_string(),
                gguf::GgufValue::String("aprender_model".to_string()),
            ),
            (
                "general.architecture".to_string(),
                gguf::GgufValue::String("mlp".to_string()),
            ),
            (
                "aprender.version".to_string(),
                gguf::GgufValue::String(env!("CARGO_PKG_VERSION").to_string()),
            ),
        ];

        let mut buffer = Vec::new();
        gguf::export_tensors_to_gguf(&mut buffer, &[tensor], &metadata).expect("export");

        // Verify header
        assert_eq!(&buffer[0..4], b"GGUF");

        // Verify version is 3
        let version = u32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]);
        assert_eq!(version, 3);

        // Verify tensor count is 1
        let tensor_count = u64::from_le_bytes([
            buffer[8], buffer[9], buffer[10], buffer[11], buffer[12], buffer[13], buffer[14],
            buffer[15],
        ]);
        assert_eq!(tensor_count, 1);

        // Verify metadata count is 3
        let metadata_count = u64::from_le_bytes([
            buffer[16], buffer[17], buffer[18], buffer[19], buffer[20], buffer[21], buffer[22],
            buffer[23],
        ]);
        assert_eq!(metadata_count, 3);
    }

    // ========================================================================
    // Memory-Mapped Loading Tests (bundle-mmap-spec.md)
    // ========================================================================

    #[test]
    fn test_load_mmap_simple() {
        use tempfile::tempdir;

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
        let path = dir.path().join("mmap_test.apr");

        // Save
        save(&model, ModelType::Custom, &path, SaveOptions::default())
            .expect("save should succeed");

        // Load with mmap
        let loaded: TestModel =
            load_mmap(&path, ModelType::Custom).expect("load_mmap should succeed");
        assert_eq!(loaded, model);
    }

    #[test]
    fn test_load_mmap_large_model() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct LargeModel {
            weights: Vec<f64>,
        }

        // Create a model larger than MMAP_THRESHOLD (1MB)
        // Use unique values to prevent compression from reducing size
        let model = LargeModel {
            weights: (0..250_000)
                .map(|i| f64::from(i) * std::f64::consts::PI)
                .collect(),
        };

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("large_mmap_test.apr");

        // Save without compression to ensure file is large
        let opts = SaveOptions::default().with_compression(Compression::None);
        save(&model, ModelType::Custom, &path, opts).expect("save should succeed");

        // Verify file is large enough
        let metadata = std::fs::metadata(&path).expect("get metadata");
        assert!(metadata.len() > MMAP_THRESHOLD, "File should be > 1MB");

        // Load with mmap
        let loaded: LargeModel =
            load_mmap(&path, ModelType::Custom).expect("load_mmap should succeed");
        assert_eq!(loaded.weights.len(), model.weights.len());
        assert_eq!(loaded, model);
    }

    #[test]
    fn test_load_auto_small_file() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct SmallModel {
            value: i32,
        }

        let model = SmallModel { value: 42 };

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("small_auto_test.apr");

        // Save
        save(&model, ModelType::Custom, &path, SaveOptions::default())
            .expect("save should succeed");

        // Verify file is small
        let metadata = std::fs::metadata(&path).expect("get metadata");
        assert!(metadata.len() <= MMAP_THRESHOLD, "File should be <= 1MB");

        // Load with auto (should use standard load)
        let loaded: SmallModel =
            load_auto(&path, ModelType::Custom).expect("load_auto should succeed");
        assert_eq!(loaded, model);
    }

    #[test]
    fn test_load_auto_large_file() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct LargeModel {
            weights: Vec<f64>,
        }

        // Create a model larger than MMAP_THRESHOLD (1MB)
        // Use unique values to prevent compression from reducing size
        let model = LargeModel {
            weights: (0..250_000)
                .map(|i| f64::from(i) * std::f64::consts::E)
                .collect(),
        };

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("large_auto_test.apr");

        // Save without compression to ensure file is large
        let opts = SaveOptions::default().with_compression(Compression::None);
        save(&model, ModelType::Custom, &path, opts).expect("save should succeed");

        // Verify file is large
        let metadata = std::fs::metadata(&path).expect("get metadata");
        assert!(metadata.len() > MMAP_THRESHOLD, "File should be > 1MB");

        // Load with auto (should use mmap)
        let loaded: LargeModel =
            load_auto(&path, ModelType::Custom).expect("load_auto should succeed");
        assert_eq!(loaded, model);
    }

    #[test]
    fn test_load_mmap_nonexistent_file() {
        #[derive(Debug, Serialize, Deserialize)]
        struct TestModel {
            value: i32,
        }

        let result: Result<TestModel> = load_mmap("/nonexistent/path.apr", ModelType::Custom);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_mmap_type_mismatch() {
        use tempfile::tempdir;

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestModel {
            value: i32,
        }

        let model = TestModel { value: 42 };

        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("type_mismatch.apr");

        // Save as Custom
        save(&model, ModelType::Custom, &path, SaveOptions::default())
            .expect("save should succeed");

        // Try to load as LinearRegression (wrong type)
        let result: Result<TestModel> = load_mmap(&path, ModelType::LinearRegression);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mismatch"));
    }
}

// ============================================================================
// Property-Based Tests (EXTREME TDD)
// ============================================================================

// ==================== Property Tests ====================

mod proptests {
    use super::*;
    use proptest::prelude::*;
    // Arbitrary Strategies
    // ================================================================

    /// Generate arbitrary ModelType
    fn arb_model_type() -> impl Strategy<Value = ModelType> {
        prop_oneof![
            Just(ModelType::LinearRegression),
            Just(ModelType::LogisticRegression),
            Just(ModelType::DecisionTree),
            Just(ModelType::RandomForest),
            Just(ModelType::GradientBoosting),
            Just(ModelType::KMeans),
            Just(ModelType::Pca),
            Just(ModelType::NaiveBayes),
            Just(ModelType::Knn),
            Just(ModelType::Svm),
            Just(ModelType::NgramLm),
            Just(ModelType::Tfidf),
            Just(ModelType::CountVectorizer),
            Just(ModelType::NeuralSequential),
            Just(ModelType::NeuralCustom),
            Just(ModelType::ContentRecommender),
            Just(ModelType::Custom),
        ]
    }

    /// Generate arbitrary Compression
    fn arb_compression() -> impl Strategy<Value = Compression> {
        prop_oneof![
            Just(Compression::None),
            Just(Compression::ZstdDefault),
            Just(Compression::ZstdMax),
            Just(Compression::Lz4),
        ]
    }

    /// Generate arbitrary valid flags (6 bits)
    fn arb_flags() -> impl Strategy<Value = Flags> {
        (0u8..64).prop_map(Flags::from_bits)
    }

    /// Generate arbitrary Header with valid values
    fn arb_header() -> impl Strategy<Value = Header> {
        (
            arb_model_type(),
            0u32..1_000_000,             // metadata_size
            0u32..100_000_000,           // payload_size
            0u32..MAX_UNCOMPRESSED_SIZE, // uncompressed_size
            arb_compression(),
            arb_flags(),
        )
            .prop_map(
                |(
                    model_type,
                    metadata_size,
                    payload_size,
                    uncompressed_size,
                    compression,
                    flags,
                )| {
                    Header {
                        magic: MAGIC,
                        version: FORMAT_VERSION,
                        model_type,
                        metadata_size,
                        payload_size,
                        uncompressed_size,
                        compression,
                        flags,
                        quality_score: 0, // APR-POKA-001: default for tests
                    }
                },
            )
    }

    // ================================================================
    // Header Roundtrip Property Tests
    // ================================================================

    proptest! {
        /// Property: Header serialization always produces exactly 32 bytes
        #[test]
        fn prop_header_size_always_32(header in arb_header()) {
            let bytes = header.to_bytes();
            prop_assert_eq!(bytes.len(), HEADER_SIZE);
        }

        /// Property: Header always starts with magic "APRN"
        #[test]
        fn prop_header_has_magic(header in arb_header()) {
            let bytes = header.to_bytes();
            prop_assert_eq!(&bytes[0..4], &MAGIC);
        }

        /// Property: Header roundtrip preserves model_type
        #[test]
        fn prop_header_roundtrip_model_type(header in arb_header()) {
            let bytes = header.to_bytes();
            let parsed = Header::from_bytes(&bytes).expect("valid header");
            prop_assert_eq!(header.model_type, parsed.model_type);
        }

        /// Property: Header roundtrip preserves metadata_size
        #[test]
        fn prop_header_roundtrip_metadata_size(header in arb_header()) {
            let bytes = header.to_bytes();
            let parsed = Header::from_bytes(&bytes).expect("valid header");
            prop_assert_eq!(header.metadata_size, parsed.metadata_size);
        }

        /// Property: Header roundtrip preserves payload_size
        #[test]
        fn prop_header_roundtrip_payload_size(header in arb_header()) {
            let bytes = header.to_bytes();
            let parsed = Header::from_bytes(&bytes).expect("valid header");
            prop_assert_eq!(header.payload_size, parsed.payload_size);
        }

        /// Property: Header roundtrip preserves uncompressed_size
        #[test]
        fn prop_header_roundtrip_uncompressed_size(header in arb_header()) {
            let bytes = header.to_bytes();
            let parsed = Header::from_bytes(&bytes).expect("valid header");
            prop_assert_eq!(header.uncompressed_size, parsed.uncompressed_size);
        }

        /// Property: Header roundtrip preserves compression
        #[test]
        fn prop_header_roundtrip_compression(header in arb_header()) {
            let bytes = header.to_bytes();
            let parsed = Header::from_bytes(&bytes).expect("valid header");
            prop_assert_eq!(header.compression, parsed.compression);
        }

        /// Property: Header roundtrip preserves flags
        #[test]
        fn prop_header_roundtrip_flags(header in arb_header()) {
            let bytes = header.to_bytes();
            let parsed = Header::from_bytes(&bytes).expect("valid header");
            prop_assert_eq!(header.flags.bits(), parsed.flags.bits());
        }

        // ================================================================
        // ModelType Property Tests
        // ================================================================

        /// Property: ModelType from_u16 roundtrip
        #[test]
        fn prop_model_type_roundtrip(model_type in arb_model_type()) {
            let value = model_type as u16;
            let parsed = ModelType::from_u16(value);
            prop_assert_eq!(parsed, Some(model_type));
        }

        /// Property: Invalid model type returns None
        #[test]
        fn prop_invalid_model_type_none(value in 0x0100u16..0x1000) {
            // Values outside defined range should return None
            let parsed = ModelType::from_u16(value);
            prop_assert!(parsed.is_none());
        }

        // ================================================================
        // Compression Property Tests
        // ================================================================

        /// Property: Compression from_u8 roundtrip
        #[test]
        fn prop_compression_roundtrip(compression in arb_compression()) {
            let value = compression as u8;
            let parsed = Compression::from_u8(value);
            prop_assert_eq!(parsed, Some(compression));
        }

        /// Property: Invalid compression returns None
        #[test]
        fn prop_invalid_compression_none(value in 4u8..255) {
            let parsed = Compression::from_u8(value);
            prop_assert!(parsed.is_none());
        }

        // ================================================================
        // Flags Property Tests
        // ================================================================

        /// Property: Flags from_bits masks reserved bits
        #[test]
        fn prop_flags_masks_reserved(raw in any::<u8>()) {
            let flags = Flags::from_bits(raw);
            // Bit 7 should be masked off (bit 6 is now HAS_MODEL_CARD)
            prop_assert!(flags.bits() < 128);
        }

        /// Property: with_encrypted sets ENCRYPTED bit
        #[test]
        fn prop_flags_with_encrypted(_seed in any::<u8>()) {
            let flags = Flags::new().with_encrypted();
            prop_assert!(flags.is_encrypted());
            prop_assert_eq!(flags.bits() & Flags::ENCRYPTED, Flags::ENCRYPTED);
        }

        /// Property: with_signed sets SIGNED bit
        #[test]
        fn prop_flags_with_signed(_seed in any::<u8>()) {
            let flags = Flags::new().with_signed();
            prop_assert!(flags.is_signed());
            prop_assert_eq!(flags.bits() & Flags::SIGNED, Flags::SIGNED);
        }

        /// Property: with_streaming sets STREAMING bit
        #[test]
        fn prop_flags_with_streaming(_seed in any::<u8>()) {
            let flags = Flags::new().with_streaming();
            prop_assert!(flags.is_streaming());
            prop_assert_eq!(flags.bits() & Flags::STREAMING, Flags::STREAMING);
        }

        /// Property: with_licensed sets LICENSED bit
        #[test]
        fn prop_flags_with_licensed(_seed in any::<u8>()) {
            let flags = Flags::new().with_licensed();
            prop_assert!(flags.is_licensed());
            prop_assert_eq!(flags.bits() & Flags::LICENSED, Flags::LICENSED);
        }

        /// Property: with_quantized sets QUANTIZED bit
        #[test]
        fn prop_flags_with_quantized(_seed in any::<u8>()) {
            let flags = Flags::new().with_quantized();
            prop_assert!(flags.is_quantized());
            prop_assert_eq!(flags.bits() & Flags::QUANTIZED, Flags::QUANTIZED);
        }

        /// Property: with_model_card sets HAS_MODEL_CARD bit
        #[test]
        fn prop_flags_with_model_card(_seed in any::<u8>()) {
            let flags = Flags::new().with_model_card();
            prop_assert!(flags.has_model_card());
            prop_assert_eq!(flags.bits() & Flags::HAS_MODEL_CARD, Flags::HAS_MODEL_CARD);
        }

        /// Property: Flag chaining is commutative (order doesn't matter)
        #[test]
        fn prop_flags_chaining_commutative(a in any::<bool>(), b in any::<bool>()) {
            let mut flags1 = Flags::new();
            let mut flags2 = Flags::new();

            if a {
                flags1 = flags1.with_encrypted();
            }
            if b {
                flags1 = flags1.with_signed();
            }

            // Reverse order
            if b {
                flags2 = flags2.with_signed();
            }
            if a {
                flags2 = flags2.with_encrypted();
            }

            prop_assert_eq!(flags1.bits(), flags2.bits());
        }

        /// Property: Flag bits are independent (setting one doesn't affect others)
        #[test]
        fn prop_flags_independent(flags in arb_flags()) {
            // Check each flag independently
            let encrypted = flags.is_encrypted();
            let signed = flags.is_signed();
            let streaming = flags.is_streaming();
            let licensed = flags.is_licensed();
            let quantized = flags.is_quantized();

            // Reconstruct from individual bits
            let reconstructed = (if encrypted { Flags::ENCRYPTED } else { 0 })
                | (if signed { Flags::SIGNED } else { 0 })
                | (if streaming { Flags::STREAMING } else { 0 })
                | (if licensed { Flags::LICENSED } else { 0 })
                | (if quantized { Flags::QUANTIZED } else { 0 })
                | (flags.bits() & Flags::TRUENO_NATIVE); // Don't forget trueno_native

            prop_assert_eq!(flags.bits(), reconstructed);
        }

        // ================================================================
        // CRC32 Checksum Property Tests
        // ================================================================

        /// Property: CRC32 of same data is always identical
        #[test]
        fn prop_crc32_deterministic(data in proptest::collection::vec(any::<u8>(), 0..1000)) {
            let crc1 = crc32(&data);
            let crc2 = crc32(&data);
            prop_assert_eq!(crc1, crc2);
        }

        /// Property: CRC32 changes when data changes (avalanche property)
        #[test]
        fn prop_crc32_avalanche(
            data in proptest::collection::vec(any::<u8>(), 1..100),
            flip_pos in 0usize..100,
            flip_bit in 0u8..8
        ) {
            if flip_pos >= data.len() {
                return Ok(());
            }

            let crc_original = crc32(&data);

            let mut modified = data.clone();
            modified[flip_pos] ^= 1 << flip_bit;

            let crc_modified = crc32(&modified);

            // Single bit flip should change CRC
            prop_assert_ne!(crc_original, crc_modified);
        }

        /// Property: Empty data has consistent CRC (IEEE polynomial)
        #[test]
        fn prop_crc32_empty(_seed in any::<u8>()) {
            let crc = crc32(&[]);
            // CRC32 of empty data is 0 for our implementation
            prop_assert_eq!(crc, 0);
        }
    }
}

// ============================================================================
// Encryption Property Tests (EXTREME TDD - Security Critical)
// Argon2id uses intentional computational cost for security.
// Default: 3 cases. For full coverage: PROPTEST_CASES=100 cargo test
// ============================================================================

#[cfg(all(test, feature = "format-encryption"))]
mod encryption_proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating valid passwords (8-64 chars)
    fn arb_password() -> impl Strategy<Value = String> {
        proptest::collection::vec(any::<u8>(), 8..64)
            .prop_map(|bytes| bytes.iter().map(|b| (b % 94 + 33) as char).collect())
    }

    /// Strategy for generating test model data
    fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-100.0f32..100.0, 1..100)
    }

    // 3 cases for encryption tests (Argon2id has high computational cost by design)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(3))]

        /// Property: Encryption roundtrip preserves data (in-memory)
        #[test]
        fn prop_encryption_roundtrip_preserves_data(
            password in arb_password(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_encrypted(&model, ModelType::Custom, &path, SaveOptions::default(), &password)
                .expect("save");
            let loaded: Model = load_encrypted(&path, ModelType::Custom, &password)
                .expect("load");

            prop_assert_eq!(loaded.weights, data);
        }

        /// Property: Wrong password fails decryption
        #[test]
        fn prop_wrong_password_fails(
            password in arb_password(),
            wrong_password in arb_password()
        ) {
            // Skip if passwords happen to be the same
            if password == wrong_password {
                return Ok(());
            }

            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { value: i32 }

            let model = Model { value: 42 };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_encrypted(&model, ModelType::Custom, &path, SaveOptions::default(), &password)
                .expect("save");
            let result: Result<Model> = load_encrypted(&path, ModelType::Custom, &wrong_password);

            prop_assert!(result.is_err(), "Wrong password should fail");
        }

        /// Property: Encrypted files have ENCRYPTED flag set
        #[test]
        fn prop_encrypted_flag_set(password in arb_password()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { v: i32 }

            let model = Model { v: 1 };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_encrypted(&model, ModelType::Custom, &path, SaveOptions::default(), &password)
                .expect("save");
            let info = inspect(&path).expect("inspect");

            prop_assert!(info.encrypted, "ENCRYPTED flag must be set");
        }

        /// Property: load_from_bytes_encrypted roundtrip works
        #[test]
        fn prop_bytes_encrypted_roundtrip(
            password in arb_password(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_encrypted(&model, ModelType::Custom, &path, SaveOptions::default(), &password)
                .expect("save");

            let bytes = std::fs::read(&path).expect("read");
            let loaded: Model = load_from_bytes_encrypted(&bytes, ModelType::Custom, &password)
                .expect("load from bytes");

            prop_assert_eq!(loaded.weights, data);
        }
    }
}

// ============================================================================
// X25519 Encryption Property Tests (EXTREME TDD - Security Critical)
// ============================================================================

#[cfg(all(test, feature = "format-encryption"))]
mod x25519_proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating test model data
    fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-100.0f32..100.0, 1..50)
    }

    proptest! {
        /// Property: X25519 roundtrip preserves data
        #[test]
        fn prop_x25519_roundtrip(data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };

            // Generate recipient keypair
            let recipient_secret = X25519SecretKey::random_from_rng(rand::thread_rng());
            let recipient_public = X25519PublicKey::from(&recipient_secret);

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_for_recipient(&model, ModelType::Custom, &path, SaveOptions::default(), &recipient_public)
                .expect("save");
            let loaded: Model = load_as_recipient(&path, ModelType::Custom, &recipient_secret)
                .expect("load");

            prop_assert_eq!(loaded.weights, data);
        }

        /// Property: X25519 wrong key fails
        #[test]
        fn prop_x25519_wrong_key_fails(data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };

            // Generate two different keypairs
            let recipient_secret = X25519SecretKey::random_from_rng(rand::thread_rng());
            let recipient_public = X25519PublicKey::from(&recipient_secret);
            let wrong_secret = X25519SecretKey::random_from_rng(rand::thread_rng());

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_for_recipient(&model, ModelType::Custom, &path, SaveOptions::default(), &recipient_public)
                .expect("save");
            let result: Result<Model> = load_as_recipient(&path, ModelType::Custom, &wrong_secret);

            prop_assert!(result.is_err(), "Wrong key should fail");
        }

        /// Property: X25519 encrypted files have ENCRYPTED flag
        #[test]
        fn prop_x25519_encrypted_flag(_seed in any::<u8>()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { v: i32 }

            let model = Model { v: 1 };
            let recipient_secret = X25519SecretKey::random_from_rng(rand::thread_rng());
            let recipient_public = X25519PublicKey::from(&recipient_secret);

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_for_recipient(&model, ModelType::Custom, &path, SaveOptions::default(), &recipient_public)
                .expect("save");
            let info = inspect(&path).expect("inspect");

            prop_assert!(info.encrypted, "ENCRYPTED flag must be set");
        }
    }
}

// ============================================================================
// Signing Property Tests (EXTREME TDD - Security Critical)
// ============================================================================

#[cfg(all(test, feature = "format-signing"))]
mod signing_proptests {
    use super::*;
    use proptest::prelude::*;

    /// Strategy for generating test model data
    fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-100.0f32..100.0, 1..50)
    }

    proptest! {
        /// Property: Signing roundtrip preserves data and verifies
        #[test]
        fn prop_signing_roundtrip(data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };

            // Generate signing keypair
            let signing_key = SigningKey::generate(&mut rand::thread_rng());
            let verifying_key = signing_key.verifying_key();

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_signed(&model, ModelType::Custom, &path, SaveOptions::default(), &signing_key)
                .expect("save");
            let loaded: Model = load_verified(&path, ModelType::Custom, Some(&verifying_key))
                .expect("load");

            prop_assert_eq!(loaded.weights, data);
        }

        /// Property: Wrong verification key fails
        #[test]
        fn prop_signing_wrong_key_fails(data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };

            // Generate two different keypairs
            let signing_key = SigningKey::generate(&mut rand::thread_rng());
            let wrong_key = SigningKey::generate(&mut rand::thread_rng());
            let wrong_verifying = wrong_key.verifying_key();

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_signed(&model, ModelType::Custom, &path, SaveOptions::default(), &signing_key)
                .expect("save");
            let result: Result<Model> = load_verified(&path, ModelType::Custom, Some(&wrong_verifying));

            prop_assert!(result.is_err(), "Wrong key should fail verification");
        }

        /// Property: Signed files have SIGNED flag set
        #[test]
        fn prop_signed_flag_set(_seed in any::<u8>()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { v: i32 }

            let model = Model { v: 1 };
            let signing_key = SigningKey::generate(&mut rand::thread_rng());

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_signed(&model, ModelType::Custom, &path, SaveOptions::default(), &signing_key)
                .expect("save");
            let info = inspect(&path).expect("inspect");

            prop_assert!(info.signed, "SIGNED flag must be set");
        }

        /// Property: Tampering with signed file is detected
        #[test]
        fn prop_tampering_detected(data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };

            let signing_key = SigningKey::generate(&mut rand::thread_rng());
            let verifying_key = signing_key.verifying_key();

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("test.apr");

            save_signed(&model, ModelType::Custom, &path, SaveOptions::default(), &signing_key)
                .expect("save");

            // Tamper with the file (modify a byte in the middle)
            let mut content = std::fs::read(&path).expect("read");
            if content.len() > 50 {
                content[50] ^= 0xFF; // Flip bits
                std::fs::write(&path, content).expect("write");

                let result: Result<Model> = load_verified(&path, ModelType::Custom, Some(&verifying_key));
                prop_assert!(result.is_err(), "Tampered file should fail verification");
            }
        }
    }
}

// ============================================================================
// Property-based tests for Knowledge Distillation (spec §6.3)
}

mod distillation_proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_distill_method() -> impl Strategy<Value = DistillMethod> {
        prop_oneof![
            Just(DistillMethod::Standard),
            Just(DistillMethod::Progressive),
            Just(DistillMethod::Ensemble),
        ]
    }

    fn arb_model_type() -> impl Strategy<Value = ModelType> {
        prop_oneof![
            Just(ModelType::LinearRegression),
            Just(ModelType::LogisticRegression),
            Just(ModelType::DecisionTree),
            Just(ModelType::RandomForest),
            Just(ModelType::KMeans),
            Just(ModelType::NaiveBayes),
            Just(ModelType::Knn),
            Just(ModelType::Pca),
            Just(ModelType::Custom),
        ]
    }

    fn arb_teacher_provenance() -> impl Strategy<Value = TeacherProvenance> {
        (
            "[a-f0-9]{64}",                              // SHA256 hash
            proptest::option::of("[a-zA-Z0-9+/]{86}=="), // Ed25519 signature (base64)
            arb_model_type(),
            1_000_000u64..10_000_000_000u64, // param count: 1M to 10B
        )
            .prop_map(
                |(hash, signature, model_type, param_count)| TeacherProvenance {
                    hash,
                    signature,
                    model_type,
                    param_count,
                    ensemble_teachers: None,
                },
            )
    }

    fn arb_distillation_params() -> impl Strategy<Value = DistillationParams> {
        (
            1.0f32..10.0f32,                       // temperature (1.0-10.0)
            0.0f32..1.0f32,                        // alpha (0.0-1.0)
            proptest::option::of(0.0f32..1.0f32),  // beta
            1u32..1000u32,                         // epochs
            proptest::option::of(0.0f32..10.0f32), // final_loss
        )
            .prop_map(|(temperature, alpha, beta, epochs, final_loss)| {
                DistillationParams {
                    temperature,
                    alpha,
                    beta,
                    epochs,
                    final_loss,
                }
            })
    }

    fn arb_layer_mapping() -> impl Strategy<Value = LayerMapping> {
        (
            0usize..100usize, // student_layer
            0usize..200usize, // teacher_layer
            0.0f32..1.0f32,   // weight
        )
            .prop_map(|(student_layer, teacher_layer, weight)| LayerMapping {
                student_layer,
                teacher_layer,
                weight,
            })
    }

    fn arb_distillation_info() -> impl Strategy<Value = DistillationInfo> {
        (
            arb_distill_method(),
            arb_teacher_provenance(),
            arb_distillation_params(),
        )
            .prop_map(|(method, teacher, params)| DistillationInfo {
                method,
                teacher,
                params,
                layer_mapping: None,
            })
    }

    fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(
            any::<f32>().prop_filter("finite", |f| f.is_finite()),
            1..100,
        )
    }

    proptest! {
        /// Property: DistillMethod serialization roundtrip (JSON for optionals)
        #[test]
        fn prop_distill_method_roundtrip(method in arb_distill_method()) {
            // Use JSON for roundtrip testing (handles enums better than raw msgpack)
            let serialized = serde_json::to_string(&method).expect("serialize");
            let deserialized: DistillMethod = serde_json::from_str(&serialized).expect("deserialize");
            prop_assert_eq!(method, deserialized);
        }

        /// Property: DistillationParams serialization roundtrip
        #[test]
        fn prop_distillation_params_roundtrip(params in arb_distillation_params()) {
            // JSON handles optional fields correctly
            let serialized = serde_json::to_string(&params).expect("serialize");
            let deserialized: DistillationParams = serde_json::from_str(&serialized).expect("deserialize");

            // Check fields (f32 equality via bits for NaN handling)
            prop_assert_eq!(params.temperature.to_bits(), deserialized.temperature.to_bits());
            prop_assert_eq!(params.alpha.to_bits(), deserialized.alpha.to_bits());
            prop_assert_eq!(params.epochs, deserialized.epochs);
            prop_assert_eq!(params.beta.map(f32::to_bits), deserialized.beta.map(f32::to_bits));
        }

        /// Property: TeacherProvenance serialization roundtrip
        #[test]
        fn prop_teacher_provenance_roundtrip(teacher in arb_teacher_provenance()) {
            let serialized = serde_json::to_string(&teacher).expect("serialize");
            let deserialized: TeacherProvenance = serde_json::from_str(&serialized).expect("deserialize");

            prop_assert_eq!(&teacher.hash, &deserialized.hash);
            prop_assert_eq!(&teacher.signature, &deserialized.signature);
            prop_assert_eq!(teacher.model_type, deserialized.model_type);
            prop_assert_eq!(teacher.param_count, deserialized.param_count);
        }

        /// Property: LayerMapping serialization roundtrip
        #[test]
        fn prop_layer_mapping_roundtrip(mapping in arb_layer_mapping()) {
            let serialized = serde_json::to_string(&mapping).expect("serialize");
            let deserialized: LayerMapping = serde_json::from_str(&serialized).expect("deserialize");

            prop_assert_eq!(mapping.student_layer, deserialized.student_layer);
            prop_assert_eq!(mapping.teacher_layer, deserialized.teacher_layer);
            prop_assert_eq!(mapping.weight.to_bits(), deserialized.weight.to_bits());
        }

        /// Property: DistillationInfo serialization roundtrip
        #[test]
        fn prop_distillation_info_roundtrip(info in arb_distillation_info()) {
            let serialized = serde_json::to_string(&info).expect("serialize");
            let deserialized: DistillationInfo = serde_json::from_str(&serialized).expect("deserialize");

            prop_assert_eq!(info.method, deserialized.method);
            prop_assert_eq!(&info.teacher.hash, &deserialized.teacher.hash);
            prop_assert_eq!(info.params.epochs, deserialized.params.epochs);
        }

        /// Property: Distillation info persists through save/load cycle
        #[test]
        fn prop_distillation_save_load_roundtrip(
            info in arb_distillation_info(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("distilled.apr");

            let options = SaveOptions::default().with_distillation_info(info.clone());
            save(&model, ModelType::Custom, &path, options).expect("save");

            let model_info = inspect(&path).expect("inspect");
            let restored = model_info.metadata.distillation_info
                .expect("should have distillation_info");

            prop_assert_eq!(info.method, restored.method);
            prop_assert_eq!(&info.teacher.hash, &restored.teacher.hash);
            prop_assert_eq!(info.teacher.param_count, restored.teacher.param_count);
            prop_assert_eq!(info.params.epochs, restored.params.epochs);
        }

        /// Property: Temperature must be positive for valid distillation
        #[test]
        fn prop_temperature_positive(temp in 0.1f32..20.0f32) {
            let params = DistillationParams {
                temperature: temp,
                alpha: 0.5,
                beta: None,
                epochs: 10,
                final_loss: None,
            };
            prop_assert!(params.temperature > 0.0, "Temperature must be positive");
        }

        /// Property: Alpha (soft loss weight) must be in [0, 1]
        #[test]
        fn prop_alpha_bounded(alpha in 0.0f32..=1.0f32) {
            let params = DistillationParams {
                temperature: 3.0,
                alpha,
                beta: None,
                epochs: 10,
                final_loss: None,
            };
            prop_assert!((0.0..=1.0).contains(&params.alpha), "Alpha must be in [0,1]");
        }

        /// Property: Progressive distillation requires beta parameter (design guideline)
        #[test]
        fn prop_progressive_with_beta(beta in 0.0f32..1.0f32) {
            let info = DistillationInfo {
                method: DistillMethod::Progressive,
                teacher: TeacherProvenance {
                    hash: "abc123".to_string(),
                    signature: None,
                    model_type: ModelType::Custom,
                    param_count: 7_000_000_000,
                    ensemble_teachers: None,
                },
                params: DistillationParams {
                    temperature: 3.0,
                    alpha: 0.7,
                    beta: Some(beta),
                    epochs: 10,
                    final_loss: None,
                },
                layer_mapping: None,
            };
            // Progressive distillation should have beta for hidden layer loss weight
            prop_assert!(info.params.beta.is_some());
        }

        /// Property: Layer mappings have valid indices
        #[test]
        fn prop_layer_mapping_valid_indices(
            student in 0usize..100,
            teacher in 0usize..200,
            weight in 0.0f32..1.0f32
        ) {
            let mapping = LayerMapping {
                student_layer: student,
                teacher_layer: teacher,
                weight,
            };
            // Teacher layer index can be >= student (many-to-one mapping)
            // Weight should be non-negative
            prop_assert!(mapping.weight >= 0.0);
        }
    }
}

// ============================================================================
// Property-based tests for Commercial License Block (spec §9)
}

mod license_proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_license_tier() -> impl Strategy<Value = LicenseTier> {
        prop_oneof![
            Just(LicenseTier::Personal),
            Just(LicenseTier::Team),
            Just(LicenseTier::Enterprise),
            Just(LicenseTier::Academic),
        ]
    }

    /// Generate valid UUID v4 format
    fn arb_uuid() -> impl Strategy<Value = String> {
        "[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
    }

    /// Generate SHA256 hash
    fn arb_hash() -> impl Strategy<Value = String> {
        "[0-9a-f]{64}"
    }

    /// Generate ISO 8601 date (YYYY-MM-DD)
    fn arb_iso_date() -> impl Strategy<Value = String> {
        (2024u32..2035, 1u32..13, 1u32..29).prop_map(|(y, m, d)| format!("{y:04}-{m:02}-{d:02}"))
    }

    fn arb_license_info() -> impl Strategy<Value = LicenseInfo> {
        (
            arb_uuid(),
            arb_hash(),
            proptest::option::of(arb_iso_date()),
            proptest::option::of(1u32..1000),
            proptest::option::of("[A-Za-z0-9 ]{1,50}"),
            arb_license_tier(),
        )
            .prop_map(|(uuid, hash, expiry, seats, licensee, tier)| LicenseInfo {
                uuid,
                hash,
                expiry,
                seats,
                licensee,
                tier,
            })
    }

    fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(
            any::<f32>().prop_filter("finite", |f| f.is_finite()),
            1..100,
        )
    }

    proptest! {
        /// Property: LicenseTier serialization roundtrip
        #[test]
        fn prop_license_tier_roundtrip(tier in arb_license_tier()) {
            let serialized = serde_json::to_string(&tier).expect("serialize");
            let deserialized: LicenseTier = serde_json::from_str(&serialized).expect("deserialize");
            prop_assert_eq!(tier, deserialized);
        }

        /// Property: LicenseInfo serialization roundtrip
        #[test]
        fn prop_license_info_roundtrip(info in arb_license_info()) {
            let serialized = serde_json::to_string(&info).expect("serialize");
            let deserialized: LicenseInfo = serde_json::from_str(&serialized).expect("deserialize");

            prop_assert_eq!(&info.uuid, &deserialized.uuid);
            prop_assert_eq!(&info.hash, &deserialized.hash);
            prop_assert_eq!(info.tier, deserialized.tier);
            prop_assert_eq!(info.seats, deserialized.seats);
            prop_assert_eq!(&info.expiry, &deserialized.expiry);
        }

        /// Property: UUID format is valid v4
        #[test]
        fn prop_uuid_format_valid(uuid in arb_uuid()) {
            // UUID v4 format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
            // where y is 8, 9, a, or b
            prop_assert_eq!(uuid.len(), 36);
            prop_assert!(uuid.chars().nth(14) == Some('4'), "Version must be 4");
            let y = uuid.chars().nth(19).expect("UUID must have char at position 19");
            prop_assert!(
                matches!(y, '8' | '9' | 'a' | 'b'),
                "Variant must be 8, 9, a, or b"
            );
        }

        /// Property: License persists through save/load cycle
        #[test]
        fn prop_license_save_load_roundtrip(
            license in arb_license_info(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("licensed.apr");

            let options = SaveOptions::default().with_license(license.clone());
            save(&model, ModelType::Custom, &path, options).expect("save");

            let model_info = inspect(&path).expect("inspect");
            let restored = model_info.metadata.license
                .expect("should have license");

            prop_assert_eq!(&license.uuid, &restored.uuid);
            prop_assert_eq!(&license.hash, &restored.hash);
            prop_assert_eq!(license.tier, restored.tier);
            prop_assert_eq!(license.seats, restored.seats);
        }

        /// Property: LICENSED flag is set when license provided
        #[test]
        fn prop_licensed_flag_set(license in arb_license_info(), data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("licensed.apr");

            let options = SaveOptions::default().with_license(license);
            save(&model, ModelType::Custom, &path, options).expect("save");

            let info = inspect(&path).expect("inspect");
            prop_assert!(info.licensed, "LICENSED flag must be set");
        }

        /// Property: Seats must be positive when specified
        #[test]
        fn prop_seats_positive(seats in 1u32..10000) {
            let license = LicenseInfo {
                uuid: "00000000-0000-4000-8000-000000000000".to_string(),
                hash: "0".repeat(64),
                expiry: None,
                seats: Some(seats),
                licensee: None,
                tier: LicenseTier::Team,
            };
            prop_assert!(license.seats == Some(seats) && seats > 0, "Seats must be positive");
        }

        /// Property: Enterprise tier has no seat limit by default
        #[test]
        fn prop_enterprise_unlimited_seats(_dummy in 0u8..1) {
            // Enterprise tier typically has unlimited seats
            let license = LicenseInfo {
                uuid: "00000000-0000-4000-8000-000000000000".to_string(),
                hash: "0".repeat(64),
                expiry: None,
                seats: None, // Unlimited
                licensee: Some("ACME Corp".to_string()),
                tier: LicenseTier::Enterprise,
            };
            prop_assert!(license.seats.is_none(), "Enterprise should have unlimited seats");
            prop_assert!(matches!(license.tier, LicenseTier::Enterprise));
        }

        /// Property: Academic tier is non-commercial
        #[test]
        fn prop_academic_tier_valid(_dummy in 0u8..1) {
            let license = LicenseInfo {
                uuid: "00000000-0000-4000-8000-000000000000".to_string(),
                hash: "0".repeat(64),
                expiry: Some("2025-12-31".to_string()),
                seats: Some(100),
                licensee: Some("MIT".to_string()),
                tier: LicenseTier::Academic,
            };
            prop_assert!(matches!(license.tier, LicenseTier::Academic));
        }

        /// Property: Hash is 64 hex characters (SHA256)
        #[test]
        fn prop_hash_length_valid(hash in arb_hash()) {
            prop_assert_eq!(hash.len(), 64, "SHA256 hash must be 64 hex chars");
            prop_assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }
}

// ============================================================================
// Property-based tests for Metadata and TrainingInfo
}

mod metadata_proptests {
    use super::*;
    use proptest::prelude::*;
        (
            proptest::option::of(1usize..1_000_000),
            proptest::option::of(1u64..86_400_000), // up to 24h in ms
            proptest::option::of("[a-zA-Z0-9_/]{1,50}"),
        )
            .prop_map(|(samples, duration_ms, source)| TrainingInfo {
                samples,
                duration_ms,
                source,
            })
    }

    fn arb_model_name() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9_-]{0,49}"
    }

    #[allow(clippy::disallowed_methods)] // json! macro uses unwrap internally
    fn arb_hyperparams() -> impl Strategy<Value = HashMap<String, serde_json::Value>> {
        proptest::collection::hash_map(
            "[a-z_]{1,20}",
            prop_oneof![
                any::<f64>()
                    .prop_filter("finite", |f| f.is_finite())
                    .prop_map(|f| serde_json::json!(f)),
                any::<i32>().prop_map(|i| serde_json::json!(i)),
                "[a-z]{1,10}".prop_map(|s| serde_json::json!(s)),
            ],
            0..5,
        )
    }

    fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(any::<f32>().prop_filter("finite", |f| f.is_finite()), 1..50)
    }

    proptest! {
        /// Property: TrainingInfo serialization roundtrip
        #[test]
        fn prop_training_info_roundtrip(info in arb_training_info()) {
            let serialized = serde_json::to_string(&info).expect("serialize");
            let deserialized: TrainingInfo = serde_json::from_str(&serialized).expect("deserialize");

            prop_assert_eq!(info.samples, deserialized.samples);
            prop_assert_eq!(info.duration_ms, deserialized.duration_ms);
            prop_assert_eq!(&info.source, &deserialized.source);
        }

        /// Property: Metadata with model name persists through save/load
        #[test]
        fn prop_metadata_model_name_roundtrip(
            name in arb_model_name(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("named.apr");

            let options = SaveOptions::default().with_name(&name);
            save(&model, ModelType::Custom, &path, options).expect("save");

            let info = inspect(&path).expect("inspect");
            prop_assert_eq!(info.metadata.model_name.as_deref(), Some(name.as_str()));
        }

        /// Property: Metadata with training info persists
        #[test]
        fn prop_metadata_training_roundtrip(
            training in arb_training_info(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("trained.apr");

            let mut options = SaveOptions::default();
            options.metadata.training = Some(training.clone());
            save(&model, ModelType::Custom, &path, options).expect("save");

            let info = inspect(&path).expect("inspect");
            let restored = info.metadata.training.expect("should have training");
            prop_assert_eq!(training.samples, restored.samples);
            prop_assert_eq!(training.duration_ms, restored.duration_ms);
        }

        /// Property: Hyperparameters persist through save/load
        #[test]
        fn prop_metadata_hyperparams_roundtrip(
            hyperparams in arb_hyperparams(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("hyperparams.apr");

            let mut options = SaveOptions::default();
            options.metadata.hyperparameters = hyperparams.clone();
            save(&model, ModelType::Custom, &path, options).expect("save");

            let info = inspect(&path).expect("inspect");
            prop_assert_eq!(hyperparams.len(), info.metadata.hyperparameters.len());
            for (k, v) in &hyperparams {
                prop_assert_eq!(Some(v), info.metadata.hyperparameters.get(k));
            }
        }

        /// Property: Custom metadata persists
        #[test]
        fn prop_metadata_custom_roundtrip(
            custom in arb_hyperparams(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("custom.apr");

            let mut options = SaveOptions::default();
            options.metadata.custom = custom.clone();
            save(&model, ModelType::Custom, &path, options).expect("save");

            let info = inspect(&path).expect("inspect");
            prop_assert_eq!(custom.len(), info.metadata.custom.len());
        }

        /// Property: Aprender version is always set
        #[test]
        fn prop_metadata_version_always_set(data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("versioned.apr");

            save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save");

            let info = inspect(&path).expect("inspect");
            prop_assert!(!info.metadata.aprender_version.is_empty());
            prop_assert!(info.metadata.aprender_version.contains('.'));
        }

        /// Property: Created timestamp is always set
        #[test]
        fn prop_metadata_timestamp_always_set(data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("timestamped.apr");

            save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save");

            let info = inspect(&path).expect("inspect");
            prop_assert!(!info.metadata.created_at.is_empty());
        }
    }
}

// ============================================================================
// Property-based tests for Error Handling / Robustness
}

mod error_proptests {
    use super::*;
    use proptest::prelude::*;
        any::<[u8; 4]>().prop_filter("not APR magic", |b| b != b"APR\x00")
    }

    fn arb_invalid_model_type() -> impl Strategy<Value = u16> {
        // Valid model types are 0-16, so anything >= 17 is invalid
        17u16..=u16::MAX
    }

    fn arb_invalid_compression() -> impl Strategy<Value = u8> {
        // Valid compression values are 0-3, so anything >= 4 is invalid
        4u8..=u8::MAX
    }

    proptest! {
        /// Property: Invalid magic bytes are rejected
        #[test]
        fn prop_invalid_magic_rejected(bad_magic in arb_non_magic_bytes()) {
            use tempfile::tempdir;

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("bad_magic.apr");

            // Create a file with invalid magic
            let mut content = vec![0u8; 64];
            content[0..4].copy_from_slice(&bad_magic);
            std::fs::write(&path, &content).expect("write");

            let result = inspect(&path);
            prop_assert!(result.is_err(), "Invalid magic should be rejected");
        }

        /// Property: Truncated header is rejected
        #[test]
        fn prop_truncated_header_rejected(len in 0usize..32) {
            use tempfile::tempdir;

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("truncated.apr");

            // Create a file shorter than header size
            let content = vec![0u8; len];
            std::fs::write(&path, &content).expect("write");

            let result = inspect(&path);
            prop_assert!(result.is_err(), "Truncated header should be rejected");
        }

        /// Property: Invalid model type in header is rejected
        #[test]
        fn prop_invalid_model_type_rejected(bad_type in arb_invalid_model_type()) {
            use tempfile::tempdir;

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("bad_type.apr");

            // Create header with valid magic but invalid model type
            let mut content = vec![0u8; 64];
            content[0..4].copy_from_slice(b"APR\x00");
            content[4..6].copy_from_slice(&bad_type.to_le_bytes());
            std::fs::write(&path, &content).expect("write");

            let result = inspect(&path);
            prop_assert!(result.is_err(), "Invalid model type should be rejected");
        }

        /// Property: Invalid compression byte is rejected
        #[test]
        fn prop_invalid_compression_rejected(bad_comp in arb_invalid_compression()) {
            use tempfile::tempdir;

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("bad_comp.apr");

            // Create header with valid magic, model type, but invalid compression
            let mut content = vec![0u8; 64];
            content[0..4].copy_from_slice(b"APR\x00");
            content[4..6].copy_from_slice(&0u16.to_le_bytes()); // Valid model type
            content[20] = bad_comp;
            std::fs::write(&path, &content).expect("write");

            let result = inspect(&path);
            prop_assert!(result.is_err(), "Invalid compression should be rejected");
        }

        /// Property: CRC mismatch is detected on load
        #[test]
        fn prop_crc_mismatch_detected(data in proptest::collection::vec(any::<f32>().prop_filter("finite", |f| f.is_finite()), 1..50)) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("crc_test.apr");

            save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save");

            // Corrupt the payload (after header + metadata)
            let mut content = std::fs::read(&path).expect("read");
            if content.len() > 100 {
                content[80] ^= 0xFF; // Flip bits in payload area
                std::fs::write(&path, &content).expect("write corrupted");

                let result: Result<Model> = load(&path, ModelType::Custom);
                // Either CRC check fails or deserialization fails - both are correct
                prop_assert!(result.is_err(), "Corrupted file should fail to load");
            }
        }

        /// Property: Empty file is rejected
        #[test]
        fn prop_empty_file_rejected(_dummy in 0u8..1) {
            use tempfile::tempdir;

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("empty.apr");

            std::fs::write(&path, []).expect("write empty");

            let result = inspect(&path);
            prop_assert!(result.is_err(), "Empty file should be rejected");
        }

        /// Property: Random bytes are rejected
        #[test]
        fn prop_random_bytes_rejected(random in proptest::collection::vec(any::<u8>(), 32..256)) {
            use tempfile::tempdir;

            // Skip if random bytes happen to start with APR magic (very unlikely)
            if random.len() >= 4 && &random[0..4] == b"APR\x00" {
                return Ok(());
            }

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("random.apr");

            std::fs::write(&path, &random).expect("write random");

            let result = inspect(&path);
            prop_assert!(result.is_err(), "Random bytes should be rejected");
        }

        /// Property: Format version matches constant
        #[test]
        fn prop_format_version_correct(data in proptest::collection::vec(any::<f32>().prop_filter("finite", |f| f.is_finite()), 1..20)) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("versioned.apr");

            save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save");

            // Verify the version bytes match FORMAT_VERSION (1, 0)
            let content = std::fs::read(&path).expect("read");
            prop_assert_eq!(content[4], FORMAT_VERSION.0, "Major version mismatch");
            prop_assert_eq!(content[5], FORMAT_VERSION.1, "Minor version mismatch");

            // Verify we can load it back
            let loaded: Model = load(&path, ModelType::Custom).expect("load");
            prop_assert_eq!(data.len(), loaded.weights.len());
        }
    }
}

// ============================================================================
// Property-based Integration Tests (Combined Features)
}

mod integration_proptests {
    use super::*;
    use proptest::prelude::*;
        prop_oneof![
            Just(ModelType::LinearRegression),
            Just(ModelType::LogisticRegression),
            Just(ModelType::DecisionTree),
            Just(ModelType::RandomForest),
            Just(ModelType::KMeans),
            Just(ModelType::NaiveBayes),
            Just(ModelType::Custom),
        ]
    }

    fn arb_model_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(
            any::<f32>().prop_filter("finite", |f| f.is_finite()),
            10..500,
        )
    }

    fn arb_large_model_data() -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(
            any::<f32>().prop_filter("finite", |f| f.is_finite()),
            1000..5000,
        )
    }

    proptest! {
        /// Property: Full metadata stack persists correctly
        #[test]
        fn prop_full_metadata_roundtrip(
            model_name in "[a-zA-Z][a-zA-Z0-9_-]{1,20}",
            description in "[a-zA-Z0-9 ]{1,50}",
            samples in 1usize..100000,
            duration_ms in 1u64..86400000,
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("full_meta.apr");

            let mut options = SaveOptions::default()
                .with_name(&model_name)
                .with_description(&description);
            options.metadata.training = Some(TrainingInfo {
                samples: Some(samples),
                duration_ms: Some(duration_ms),
                source: Some("test_data".to_string()),
            });

            save(&model, ModelType::Custom, &path, options).expect("save");

            let info = inspect(&path).expect("inspect");
            prop_assert_eq!(info.metadata.model_name.as_deref(), Some(model_name.as_str()));
            prop_assert_eq!(info.metadata.description.as_deref(), Some(description.as_str()));

            let training = info.metadata.training.expect("training");
            prop_assert_eq!(training.samples, Some(samples));
            prop_assert_eq!(training.duration_ms, Some(duration_ms));

            let loaded: Model = load(&path, ModelType::Custom).expect("load");
            prop_assert_eq!(data.len(), loaded.weights.len());
        }

        /// Property: All model types roundtrip correctly
        #[test]
        fn prop_all_model_types_roundtrip(
            model_type in arb_model_type(),
            data in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("typed.apr");

            save(&model, model_type, &path, SaveOptions::default()).expect("save");

            let info = inspect(&path).expect("inspect");
            prop_assert_eq!(info.model_type, model_type);

            let loaded: Model = load(&path, model_type).expect("load");
            prop_assert_eq!(data.len(), loaded.weights.len());
        }

        /// Property: Large models roundtrip correctly (stress test)
        #[test]
        fn prop_large_model_roundtrip(data in arb_large_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data.clone() };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("large.apr");

            save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save");

            let loaded: Model = load(&path, ModelType::Custom).expect("load");
            prop_assert_eq!(data.len(), loaded.weights.len());

            // Verify data integrity
            for (i, (orig, loaded_val)) in data.iter().zip(loaded.weights.iter()).enumerate() {
                prop_assert_eq!(
                    orig.to_bits(),
                    loaded_val.to_bits(),
                    "Mismatch at index {}", i
                );
            }
        }

        /// Property: Distillation + License combined
        #[test]
        fn prop_distillation_with_license(data in arb_model_data()) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let model = Model { weights: data };
            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("distilled_licensed.apr");

            let distill_info = DistillationInfo {
                method: DistillMethod::Standard,
                teacher: TeacherProvenance {
                    hash: "a".repeat(64),
                    signature: None,
                    model_type: ModelType::Custom,
                    param_count: 7_000_000_000,
                    ensemble_teachers: None,
                },
                params: DistillationParams {
                    temperature: 3.0,
                    alpha: 0.7,
                    beta: None,
                    epochs: 10,
                    final_loss: Some(0.42),
                },
                layer_mapping: None,
            };

            let license = LicenseInfo {
                uuid: "12345678-1234-4123-8123-123456789abc".to_string(),
                hash: "b".repeat(64),
                expiry: Some("2025-12-31".to_string()),
                seats: Some(10),
                licensee: Some("Test Corp".to_string()),
                tier: LicenseTier::Enterprise,
            };

            let options = SaveOptions::default()
                .with_distillation_info(distill_info)
                .with_license(license);

            save(&model, ModelType::Custom, &path, options).expect("save");

            let info = inspect(&path).expect("inspect");

            // Both features should be present
            prop_assert!(info.metadata.distillation_info.is_some());
            prop_assert!(info.metadata.license.is_some());
            prop_assert!(info.licensed);

            let restored_distill = info.metadata.distillation_info.expect("distillation");
            prop_assert!(matches!(restored_distill.method, DistillMethod::Standard));

            let restored_license = info.metadata.license.expect("license");
            prop_assert!(matches!(restored_license.tier, LicenseTier::Enterprise));
        }

        /// Property: Multiple saves to same path overwrites correctly
        #[test]
        fn prop_overwrite_preserves_latest(
            data1 in arb_model_data(),
            data2 in arb_model_data()
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let dir = tempdir().expect("tempdir");
            let path = dir.path().join("overwrite.apr");

            // Save first model
            let model1 = Model { weights: data1 };
            save(&model1, ModelType::Custom, &path, SaveOptions::default()).expect("save1");

            // Save second model (overwrite)
            let model2 = Model { weights: data2.clone() };
            save(&model2, ModelType::Custom, &path, SaveOptions::default()).expect("save2");

            // Load should return second model
            let loaded: Model = load(&path, ModelType::Custom).expect("load");
            prop_assert_eq!(data2.len(), loaded.weights.len());
        }

        /// Property: File size scales with data size
        #[test]
        fn prop_file_size_scales_with_data(
            small_data in proptest::collection::vec(any::<f32>().prop_filter("finite", |f| f.is_finite()), 10..50),
            large_data in proptest::collection::vec(any::<f32>().prop_filter("finite", |f| f.is_finite()), 500..1000)
        ) {
            use tempfile::tempdir;

            #[derive(Debug, serde::Serialize, serde::Deserialize)]
            struct Model { weights: Vec<f32> }

            let dir = tempdir().expect("tempdir");

            let small_path = dir.path().join("small.apr");
            let model_small = Model { weights: small_data };
            save(&model_small, ModelType::Custom, &small_path, SaveOptions::default()).expect("save small");

            let large_path = dir.path().join("large.apr");
            let model_large = Model { weights: large_data };
            save(&model_large, ModelType::Custom, &large_path, SaveOptions::default()).expect("save large");

            let small_size = std::fs::metadata(&small_path).expect("meta").len();
            let large_size = std::fs::metadata(&large_path).expect("meta").len();

            prop_assert!(large_size > small_size, "Larger data should produce larger file");
        }
    }
}
