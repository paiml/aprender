
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
    let decompressed =
        decompress_payload(&compressed, Compression::Lz4).expect("lz4 decompress should succeed");

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
    save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save should succeed");

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
    let loaded: TestModel =
        load_encrypted(&path, ModelType::Custom, password).expect("load_encrypted should succeed");
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
    save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save should succeed");

    // load_encrypted should reject unencrypted files
    let result: Result<TestModel> = load_encrypted(&path, ModelType::Custom, "any_password");
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("ENCRYPTED flag not set") || err_msg.contains("File too small"),
        "Expected ENCRYPTED flag error or size error, got: {err_msg}"
    );
}
