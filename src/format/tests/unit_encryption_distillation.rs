
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
    let loaded: LargeModel =
        load_encrypted(&path, ModelType::Custom, password).expect("load_encrypted should succeed");
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
    let recipient_secret = StaticSecret::random_from_rng(rand::rng());
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
    let recipient_secret = StaticSecret::random_from_rng(rand::rng());
    let recipient_public = PublicKey::from(&recipient_secret);

    // Generate wrong keypair
    let wrong_secret = StaticSecret::random_from_rng(rand::rng());

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
    let wrong_secret = StaticSecret::random_from_rng(rand::rng());
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
    save(&model, ModelType::Custom, &path, SaveOptions::default()).expect("save should succeed");

    // load_as_recipient should reject unencrypted files
    let secret = StaticSecret::random_from_rng(rand::rng());
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
        (restored.params.final_loss.expect("should have final_loss") - 0.42).abs() < f32::EPSILON
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
