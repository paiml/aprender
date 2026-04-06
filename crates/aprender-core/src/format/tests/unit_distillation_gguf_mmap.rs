
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
    save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save should succeed");

    // Load with mmap
    let loaded: TestModel = load_mmap(&path, ModelType::Custom).expect("load_mmap should succeed");
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
    let loaded: LargeModel = load_mmap(&path, ModelType::Custom).expect("load_mmap should succeed");
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
    save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save should succeed");

    // Verify file is small
    let metadata = std::fs::metadata(&path).expect("get metadata");
    assert!(metadata.len() <= MMAP_THRESHOLD, "File should be <= 1MB");

    // Load with auto (should use standard load)
    let loaded: SmallModel = load_auto(&path, ModelType::Custom).expect("load_auto should succeed");
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
    let loaded: LargeModel = load_auto(&path, ModelType::Custom).expect("load_auto should succeed");
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
