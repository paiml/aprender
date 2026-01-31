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

    // ========================================================================
    // Pygmy-Based Core I/O Tests (T-COV-95)
    // ========================================================================

    #[test]
    fn test_compress_decompress_roundtrip_none() {
        use super::super::core_io::{compress_payload, decompress_payload};

        let data = b"Hello, World! This is test data for compression.";

        // No compression roundtrip
        let (compressed, compression) = compress_payload(data, Compression::None).expect("compress");
        assert_eq!(compression, Compression::None);
        assert_eq!(compressed, data);

        let decompressed = decompress_payload(&compressed, compression).expect("decompress");
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "format-compression")]
    #[test]
    fn test_compress_decompress_roundtrip_zstd_default() {
        use super::super::core_io::{compress_payload, decompress_payload};

        let data = vec![42u8; 1000]; // Repetitive data compresses well

        let (compressed, compression) =
            compress_payload(&data, Compression::ZstdDefault).expect("compress");
        assert_eq!(compression, Compression::ZstdDefault);
        // Compressed should be smaller for repetitive data
        assert!(compressed.len() < data.len());

        let decompressed = decompress_payload(&compressed, compression).expect("decompress");
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "format-compression")]
    #[test]
    fn test_compress_decompress_roundtrip_zstd_max() {
        use super::super::core_io::{compress_payload, decompress_payload};

        let data = vec![0u8; 500];

        let (compressed, compression) =
            compress_payload(&data, Compression::ZstdMax).expect("compress");
        assert_eq!(compression, Compression::ZstdMax);

        let decompressed = decompress_payload(&compressed, compression).expect("decompress");
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "format-compression")]
    #[test]
    fn test_compress_decompress_roundtrip_lz4() {
        use super::super::core_io::{compress_payload, decompress_payload};

        let data = b"LZ4 compression test data with some repetition repetition repetition";

        let (compressed, compression) = compress_payload(data, Compression::Lz4).expect("compress");
        assert_eq!(compression, Compression::Lz4);

        let decompressed = decompress_payload(&compressed, compression).expect("decompress");
        assert_eq!(decompressed.as_slice(), data);
    }

    #[test]
    fn test_crc32_empty() {
        // CRC32 of empty data is 0 (identity element)
        let crc = crc32(&[]);
        assert_eq!(crc, 0);
    }

    #[test]
    fn test_crc32_known_values() {
        // Test multiple known CRC32 values (IEEE polynomial)
        assert_eq!(crc32(b""), 0x0000_0000);
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
        assert_eq!(crc32(b"abc"), 0x352441C2);
    }

    #[test]
    fn test_load_from_bytes_with_pygmy_apr() {
        use super::super::test_factory::{build_pygmy_apr, build_pygmy_apr_with_config, PygmyConfig};
        use super::super::v2::AprV2Reader;

        // Build a pygmy APR file in memory
        let apr_data = build_pygmy_apr();

        // Verify we can parse it
        let reader = AprV2Reader::from_bytes(&apr_data).expect("parse pygmy APR");

        // Check basic properties
        let meta = reader.metadata();
        assert_eq!(meta.model_type, "pygmy");

        let tensor_names = reader.tensor_names();
        assert!(!tensor_names.is_empty());

        // Test with different configs
        let configs = [
            ("default", PygmyConfig::default()),
            ("minimal", PygmyConfig::minimal()),
            ("embedding_only", PygmyConfig::embedding_only()),
        ];

        for (name, config) in configs {
            let apr_data = build_pygmy_apr_with_config(config);
            let reader = AprV2Reader::from_bytes(&apr_data)
                .unwrap_or_else(|e| panic!("parse pygmy APR with config {name}: {e}"));
            assert_eq!(reader.metadata().model_type, "pygmy");
        }
    }

    #[test]
    fn test_inspect_bytes_with_pygmy_safetensors() {
        use super::super::test_factory::build_pygmy_safetensors;

        // Build a pygmy SafeTensors file in memory
        let st_data = build_pygmy_safetensors();

        // Verify it has SafeTensors magic (header length as u64 LE, then JSON)
        assert!(st_data.len() > 8);
        let header_len = u64::from_le_bytes(st_data[0..8].try_into().unwrap());
        assert!(header_len < 10_000); // Reasonable header size

        // Check JSON start
        assert_eq!(&st_data[8..10], b"{\"");

        // Verify format detection from bytes
        // SafeTensors is detected by the JSON header pattern
        assert!(header_len > 0);
    }

    #[test]
    fn test_pygmy_apr_tensor_alignment() {
        use super::super::test_factory::build_pygmy_apr;
        use super::super::v2::AprV2Reader;

        let apr_data = build_pygmy_apr();
        let reader = AprV2Reader::from_bytes(&apr_data).expect("parse");

        // All tensors should be 64-byte aligned
        assert!(reader.verify_alignment());
    }

    #[test]
    fn test_pygmy_apr_quantized_formats() {
        use super::super::test_factory::{build_pygmy_apr_f16, build_pygmy_apr_q4, build_pygmy_apr_q8};
        use super::super::v2::{AprV2Reader, TensorDType};

        // Test Q8 format
        let q8_data = build_pygmy_apr_q8();
        let reader = AprV2Reader::from_bytes(&q8_data).expect("parse Q8");
        let q8_tensors: Vec<_> = reader
            .tensor_names()
            .iter()
            .filter_map(|n| reader.get_tensor(n))
            .filter(|t| t.dtype == TensorDType::Q8)
            .collect();
        assert!(!q8_tensors.is_empty(), "Should have Q8 tensors");

        // Test Q4 format
        let q4_data = build_pygmy_apr_q4();
        let reader = AprV2Reader::from_bytes(&q4_data).expect("parse Q4");
        let q4_tensors: Vec<_> = reader
            .tensor_names()
            .iter()
            .filter_map(|n| reader.get_tensor(n))
            .filter(|t| t.dtype == TensorDType::Q4)
            .collect();
        assert!(!q4_tensors.is_empty(), "Should have Q4 tensors");

        // Test F16 format
        let f16_data = build_pygmy_apr_f16();
        let reader = AprV2Reader::from_bytes(&f16_data).expect("parse F16");
        let f16_tensors: Vec<_> = reader
            .tensor_names()
            .iter()
            .filter_map(|n| reader.get_tensor(n))
            .filter(|t| t.dtype == TensorDType::F16)
            .collect();
        assert!(!f16_tensors.is_empty(), "Should have F16 tensors");
    }

    #[test]
    fn test_pygmy_safetensors_metadata_parsing() {
        use super::super::test_factory::build_pygmy_safetensors;

        let st_data = build_pygmy_safetensors();

        // Parse header length
        let header_len = u64::from_le_bytes(st_data[0..8].try_into().unwrap()) as usize;

        // Parse JSON header
        let header_bytes = &st_data[8..8 + header_len];
        let header_str = std::str::from_utf8(header_bytes).expect("valid UTF-8");

        // Should be valid JSON with tensor metadata
        assert!(header_str.starts_with('{'));
        assert!(header_str.ends_with('}'));
        assert!(header_str.contains("dtype"));
        assert!(header_str.contains("shape"));
    }

    #[test]
    fn test_inspect_bytes_error_too_small() {
        // Data too small for header
        let tiny_data = [0u8; 10];
        let result = inspect_bytes(&tiny_data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("too small"));
    }

    #[test]
    fn test_inspect_bytes_error_invalid_magic() {
        // Invalid magic number
        let mut bad_data = [0u8; HEADER_SIZE + 4];
        bad_data[0..4].copy_from_slice(b"BAAD");

        let result = inspect_bytes(&bad_data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid magic"));
    }

    #[test]
    fn test_pygmy_apr_llama_style_config() {
        use super::super::test_factory::{build_pygmy_apr_with_config, PygmyConfig};
        use super::super::v2::AprV2Reader;

        let config = PygmyConfig::llama_style();
        let apr_data = build_pygmy_apr_with_config(config);

        let reader = AprV2Reader::from_bytes(&apr_data).expect("parse llama-style");

        // Check llama-style naming conventions
        let tensor_names = reader.tensor_names();
        let has_attn_q = tensor_names.iter().any(|n| n.contains("self_attn.q_proj"));
        let has_attn_k = tensor_names.iter().any(|n| n.contains("self_attn.k_proj"));
        let has_attn_v = tensor_names.iter().any(|n| n.contains("self_attn.v_proj"));
        let has_mlp = tensor_names.iter().any(|n| n.contains("mlp"));

        assert!(has_attn_q, "Should have q_proj tensors");
        assert!(has_attn_k, "Should have k_proj tensors");
        assert!(has_attn_v, "Should have v_proj tensors");
        assert!(has_mlp, "Should have MLP tensors");
    }

    #[test]
    fn test_pygmy_apr_multiple_layers() {
        use super::super::test_factory::{build_pygmy_apr_with_config, PygmyConfig};
        use super::super::v2::AprV2Reader;

        let mut config = PygmyConfig::default();
        config.num_layers = 3;
        let apr_data = build_pygmy_apr_with_config(config);

        let reader = AprV2Reader::from_bytes(&apr_data).expect("parse multi-layer");

        let tensor_names = reader.tensor_names();

        // Check we have tensors for all layers
        for layer_idx in 0..3 {
            let layer_prefix = format!("layers.{layer_idx}");
            let has_layer = tensor_names.iter().any(|n| n.contains(&layer_prefix));
            assert!(has_layer, "Should have layer {layer_idx} tensors");
        }
    }
