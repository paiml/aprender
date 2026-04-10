/// Verify CSV roundtrip integrity preserves all fields
#[test]
fn test_falsify_cert_001_roundtrip_integrity() {
    let temp_file = NamedTempFile::new().expect("temp file");

    // Write test CSV to temp file
    std::fs::write(temp_file.path(), TEST_CSV).expect("write");

    // Read original
    let original = read_models_csv(temp_file.path()).expect("read original");
    assert_eq!(original.len(), 3, "Expected 3 rows");

    // Write to new temp file
    let temp_file2 = NamedTempFile::new().expect("temp file 2");
    write_models_csv(&original, temp_file2.path()).expect("write");

    // Read back
    let roundtrip = read_models_csv(temp_file2.path()).expect("read roundtrip");
    assert_eq!(roundtrip.len(), original.len(), "Row count mismatch");

    // Verify each row
    for (orig, rt) in original.iter().zip(roundtrip.iter()) {
        assert_eq!(orig.model_id, rt.model_id, "model_id mismatch");
        assert_eq!(orig.family, rt.family, "family mismatch");
        assert_eq!(orig.parameters, rt.parameters, "parameters mismatch");
        assert_eq!(
            orig.size_category, rt.size_category,
            "size_category mismatch"
        );
        assert_eq!(orig.status, rt.status, "status mismatch");
        assert_eq!(orig.mqs_score, rt.mqs_score, "mqs_score mismatch");
        assert_eq!(orig.grade, rt.grade, "grade mismatch");
        assert_eq!(
            orig.certified_tier, rt.certified_tier,
            "certified_tier mismatch"
        );
        assert_eq!(orig.g1, rt.g1, "g1 mismatch");
        assert_eq!(orig.g2, rt.g2, "g2 mismatch");
        assert_eq!(orig.g3, rt.g3, "g3 mismatch");
        assert_eq!(orig.g4, rt.g4, "g4 mismatch");
        assert_eq!(
            orig.provenance_verified, rt.provenance_verified,
            "provenance_verified mismatch"
        );
    }
}

/// Verify ModelStatus parses from string including case-insensitive variants
#[test]
fn test_model_status_from_str() {
    assert_eq!(
        "CERTIFIED".parse::<ModelStatus>().unwrap(),
        ModelStatus::Certified
    );
    assert_eq!(
        "BLOCKED".parse::<ModelStatus>().unwrap(),
        ModelStatus::Blocked
    );
    assert_eq!(
        "PENDING".parse::<ModelStatus>().unwrap(),
        ModelStatus::Pending
    );
    assert_eq!(
        "UNTESTED".parse::<ModelStatus>().unwrap(),
        ModelStatus::Untested
    );
    assert_eq!(
        "certified".parse::<ModelStatus>().unwrap(),
        ModelStatus::Certified
    );
    assert!("INVALID".parse::<ModelStatus>().is_err());
}

/// Verify ModelStatus Display formats as uppercase string
#[test]
fn test_model_status_display() {
    assert_eq!(format!("{}", ModelStatus::Certified), "CERTIFIED");
    assert_eq!(format!("{}", ModelStatus::Blocked), "BLOCKED");
    assert_eq!(format!("{}", ModelStatus::Pending), "PENDING");
    assert_eq!(format!("{}", ModelStatus::Untested), "UNTESTED");
}

/// Verify SizeCategory parses from string including case-insensitive variants
#[test]
fn test_size_category_from_str() {
    assert_eq!("tiny".parse::<SizeCategory>().unwrap(), SizeCategory::Tiny);
    assert_eq!(
        "SMALL".parse::<SizeCategory>().unwrap(),
        SizeCategory::Small
    );
    assert_eq!(
        "Medium".parse::<SizeCategory>().unwrap(),
        SizeCategory::Medium
    );
    assert_eq!(
        "large".parse::<SizeCategory>().unwrap(),
        SizeCategory::Large
    );
    assert_eq!(
        "xlarge".parse::<SizeCategory>().unwrap(),
        SizeCategory::Xlarge
    );
    assert_eq!("huge".parse::<SizeCategory>().unwrap(), SizeCategory::Huge);
    assert!("invalid".parse::<SizeCategory>().is_err());
}

/// Verify SizeCategory Display formats as lowercase string
#[test]
fn test_size_category_display() {
    assert_eq!(format!("{}", SizeCategory::Tiny), "tiny");
    assert_eq!(format!("{}", SizeCategory::Small), "small");
    assert_eq!(format!("{}", SizeCategory::Medium), "medium");
    assert_eq!(format!("{}", SizeCategory::Large), "large");
    assert_eq!(format!("{}", SizeCategory::Xlarge), "xlarge");
    assert_eq!(format!("{}", SizeCategory::Huge), "huge");
}

/// Verify CertificationRow default has empty ID and pending status
#[test]
fn test_certification_row_default() {
    let row = CertificationRow::default();
    assert!(row.model_id.is_empty());
    assert_eq!(row.status, ModelStatus::Pending);
    assert_eq!(row.mqs_score, 0);
    assert!(!row.g1);
}

/// Verify CertificationRow::new sets model_id and family
#[test]
fn test_certification_row_new() {
    let row = CertificationRow::new("test/model", "test-family");
    assert_eq!(row.model_id, "test/model");
    assert_eq!(row.family, "test-family");
}

/// Verify all_gateways_passed requires all four gateways true
#[test]
fn test_all_gateways_passed() {
    // Default has no gateways passed
    let row = CertificationRow::default();
    assert!(!row.all_gateways_passed());

    // All gateways passed
    let row = CertificationRow {
        g1: true,
        g2: true,
        g3: true,
        g4: true,
        ..Default::default()
    };
    assert!(row.all_gateways_passed());

    // One gateway failed
    let row = CertificationRow {
        g1: true,
        g2: true,
        g3: false,
        g4: true,
        ..Default::default()
    };
    assert!(!row.all_gateways_passed());
}

/// Verify derive_status returns correct status for various MQS/gateway combos
#[test]
fn test_derive_status() {
    // Test CERTIFIED: MQS >= 800 and all gateways passed
    let row = CertificationRow {
        g1: true,
        g2: true,
        g3: true,
        g4: true,
        mqs_score: 850,
        ..Default::default()
    };
    assert_eq!(row.derive_status(), ModelStatus::Certified);

    // Test BLOCKED: MQS < 800
    let row = CertificationRow {
        g1: true,
        g2: true,
        g3: true,
        g4: true,
        mqs_score: 799,
        ..Default::default()
    };
    assert_eq!(row.derive_status(), ModelStatus::Blocked);

    // Test BLOCKED: gateway failure
    let row = CertificationRow {
        g1: true,
        g2: true,
        g3: false,
        g4: true,
        mqs_score: 900,
        ..Default::default()
    };
    assert_eq!(row.derive_status(), ModelStatus::Blocked);

    // Test PENDING: never tested
    let row = CertificationRow {
        g1: false,
        mqs_score: 0,
        ..Default::default()
    };
    assert_eq!(row.derive_status(), ModelStatus::Pending);
}

/// Verify derive_grade maps MQS scores to correct letter grades
#[test]
fn test_derive_grade() {
    let row_a = CertificationRow {
        mqs_score: 950,
        ..Default::default()
    };
    assert_eq!(row_a.derive_grade(), "A+");

    let row_a2 = CertificationRow {
        mqs_score: 920,
        ..Default::default()
    };
    assert_eq!(row_a2.derive_grade(), "A");

    let row_bp = CertificationRow {
        mqs_score: 850,
        ..Default::default()
    };
    assert_eq!(row_bp.derive_grade(), "B+");

    let row_b = CertificationRow {
        mqs_score: 820,
        ..Default::default()
    };
    assert_eq!(row_b.derive_grade(), "B");

    let row_c = CertificationRow {
        mqs_score: 700,
        ..Default::default()
    };
    assert_eq!(row_c.derive_grade(), "C");

    let row_f = CertificationRow {
        mqs_score: 200,
        ..Default::default()
    };
    assert_eq!(row_f.derive_grade(), "F");
}

/// Verify lookup_model finds model by ID or returns None
#[test]
fn test_lookup_model() {
    let rows = vec![
        CertificationRow::new("test/model-1", "family-a"),
        CertificationRow::new("test/model-2", "family-b"),
        CertificationRow::new("test/model-3", "family-a"),
    ];

    let found = lookup_model(&rows, "test/model-2");
    assert!(found.is_some());
    assert_eq!(found.unwrap().family, "family-b");

    let not_found = lookup_model(&rows, "nonexistent");
    assert!(not_found.is_none());
}

/// Verify lookup_family returns all rows matching family name
#[test]
fn test_lookup_family() {
    let rows = vec![
        CertificationRow::new("test/model-1", "family-a"),
        CertificationRow::new("test/model-2", "family-b"),
        CertificationRow::new("test/model-3", "family-a"),
    ];

    let family_a = lookup_family(&rows, "family-a");
    assert_eq!(family_a.len(), 2);

    let family_b = lookup_family(&rows, "family-b");
    assert_eq!(family_b.len(), 1);

    let family_c = lookup_family(&rows, "family-c");
    assert!(family_c.is_empty());
}

/// Verify read_models_csv returns error for missing file
#[test]
fn test_read_missing_file() {
    let result = read_models_csv("/nonexistent/path/models.csv");
    assert!(result.is_err());
}

/// Verify read_models_csv returns error for malformed CSV
#[test]
fn test_read_malformed_csv() {
    let temp_file = NamedTempFile::new().expect("temp file");
    std::fs::write(
        temp_file.path(),
        "model_id,family\ntest,test,extra,fields,here",
    )
    .expect("write");

    // Should handle flexible field count gracefully
    let result = read_models_csv(temp_file.path());
    // This may error due to missing required fields
    assert!(result.is_err());
}

/// Verify optional TPS fields are correctly parsed from CSV
#[test]
fn test_optional_tps_fields() {
    let temp_file = NamedTempFile::new().expect("temp file");
    std::fs::write(temp_file.path(), TEST_CSV).expect("write");

    let cert_rows = read_models_csv(temp_file.path()).expect("read");

    // First row has no TPS values
    let first_row = &cert_rows[0];
    assert!(first_row.tps_gguf_cpu.is_none());
    assert!(first_row.tps_gguf_gpu.is_none());

    // Second row has TPS values
    let second_row = &cert_rows[1];
    assert!(second_row.tps_gguf_cpu.is_some());
    assert!((second_row.tps_gguf_cpu.unwrap() - 17.9).abs() < 0.1);
}

/// Verify write_models_csv creates readable file with correct data
#[test]
fn test_write_models_csv_creates_file() {
    let temp_file = NamedTempFile::new().expect("temp file");

    let rows = vec![CertificationRow {
        model_id: "test/model".to_string(),
        family: "test-family".to_string(),
        parameters: "1B".to_string(),
        size_category: SizeCategory::Small,
        status: ModelStatus::Blocked,
        mqs_score: 500,
        grade: "F".to_string(),
        certified_tier: "mvp".to_string(),
        g1: true,
        g2: true,
        g3: false,
        g4: true,
        tps_gguf_cpu: Some(10.5),
        tps_gguf_gpu: Some(100.0),
        tps_apr_cpu: None,
        tps_apr_gpu: None,
        tps_st_cpu: None,
        tps_st_gpu: None,
        provenance_verified: true,
        ..Default::default()
    }];

    write_models_csv(&rows, temp_file.path()).expect("write");

    // Verify file exists and can be read back
    let read_back = read_models_csv(temp_file.path()).expect("read");
    assert_eq!(read_back.len(), 1);
    assert_eq!(read_back[0].model_id, "test/model");
    assert_eq!(read_back[0].tps_gguf_cpu.unwrap(), 10.5);
    assert!(read_back[0].provenance_verified);
}

/// Verify write_models_csv returns error for nonexistent directory
#[test]
fn test_write_to_nonexistent_dir() {
    let result = write_models_csv(&[], "/nonexistent/dir/models.csv");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Failed to create"));
}

/// Verify ModelStatus serializes/deserializes to/from JSON
#[test]
fn test_model_status_serde() {
    let status = ModelStatus::Certified;
    let json = serde_json::to_string(&status).expect("serialize");
    assert_eq!(json, "\"CERTIFIED\"");

    let deserialized: ModelStatus = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized, ModelStatus::Certified);
}

/// Verify SizeCategory serializes/deserializes to/from JSON
#[test]
fn test_size_category_serde() {
    let size = SizeCategory::Medium;
    let json = serde_json::to_string(&size).expect("serialize");
    assert_eq!(json, "\"medium\"");

    let deserialized: SizeCategory = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized, SizeCategory::Medium);
}

/// Verify CertificationRow serializes/deserializes to/from JSON
#[test]
fn test_certification_row_serde() {
    let row = CertificationRow {
        model_id: "test/model".to_string(),
        family: "test".to_string(),
        status: ModelStatus::Certified,
        mqs_score: 850,
        ..Default::default()
    };

    let json = serde_json::to_string(&row).expect("serialize");
    assert!(json.contains("\"model_id\":\"test/model\""));
    assert!(json.contains("\"status\":\"CERTIFIED\""));

    let deserialized: CertificationRow = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(deserialized.model_id, "test/model");
    assert_eq!(deserialized.status, ModelStatus::Certified);
}

/// Verify invalid status string returns parse error
#[test]
fn test_invalid_status_parse() {
    let result = "GARBAGE".parse::<ModelStatus>();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Invalid status"));
}

/// Verify invalid size category string returns parse error
#[test]
fn test_invalid_size_category_parse() {
    let result = "massive".parse::<SizeCategory>();
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Invalid size category")
    );
}

/// Verify default ModelStatus is Pending
#[test]
fn test_model_status_default() {
    let status = ModelStatus::default();
    assert_eq!(status, ModelStatus::Pending);
}

/// Verify default SizeCategory is Tiny
#[test]
fn test_size_category_default() {
    let size = SizeCategory::default();
    assert_eq!(size, SizeCategory::Tiny);
}



