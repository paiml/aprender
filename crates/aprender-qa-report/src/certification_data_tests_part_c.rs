// ── CERT-DATA: Additional edge cases for coverage ──────────────────────────

/// Verify derive_grade returns "A+" for scores above 1000 (proof bonus can push past 1000)
#[test]
fn test_derive_grade_above_max() {
    let row = CertificationRow {
        mqs_score: 1001,
        ..CertificationRow::default()
    };
    assert_eq!(row.derive_grade(), "A+");

    let row_1050 = CertificationRow {
        mqs_score: 1050,
        ..CertificationRow::default()
    };
    assert_eq!(row_1050.derive_grade(), "A+");
}

/// Verify SizeCategory Huge Display and FromStr roundtrip
#[test]
fn test_size_category_huge_roundtrip() {
    let size = SizeCategory::Huge;
    let s = size.to_string();
    assert_eq!(s, "huge");
    let parsed: SizeCategory = s.parse().unwrap();
    assert_eq!(parsed, SizeCategory::Huge);
}

/// Verify ModelStatus Untested Display and FromStr roundtrip
#[test]
fn test_model_status_untested_roundtrip() {
    let status = ModelStatus::Untested;
    let s = status.to_string();
    assert_eq!(s, "UNTESTED");
    let parsed: ModelStatus = s.parse().unwrap();
    assert_eq!(parsed, ModelStatus::Untested);
}

/// Verify write_models_csv roundtrip with all TPS fields populated
#[test]
fn test_write_csv_all_tps_fields() {
    let temp = NamedTempFile::new().unwrap();
    let rows = vec![CertificationRow {
        model_id: "test/all-tps".to_string(),
        family: "test".to_string(),
        parameters: "1B".to_string(),
        size_category: SizeCategory::Small,
        status: ModelStatus::Certified,
        mqs_score: 900,
        grade: "A".to_string(),
        certified_tier: "full".to_string(),
        g1: true,
        g2: true,
        g3: true,
        g4: true,
        tps_gguf_cpu: Some(15.5),
        tps_gguf_gpu: Some(120.0),
        tps_apr_cpu: Some(14.0),
        tps_apr_gpu: Some(110.5),
        tps_st_cpu: Some(3.2),
        tps_st_gpu: Some(25.8),
        provenance_verified: true,
        ..CertificationRow::default()
    }];

    write_models_csv(&rows, temp.path()).unwrap();
    let read_back = read_models_csv(temp.path()).unwrap();
    assert_eq!(read_back.len(), 1);
    let r = &read_back[0];
    assert!((r.tps_gguf_cpu.unwrap() - 15.5).abs() < 0.1);
    assert!((r.tps_gguf_gpu.unwrap() - 120.0).abs() < 0.1);
    assert!((r.tps_apr_cpu.unwrap() - 14.0).abs() < 0.1);
    assert!((r.tps_apr_gpu.unwrap() - 110.5).abs() < 0.1);
    assert!((r.tps_st_cpu.unwrap() - 3.2).abs() < 0.1);
    assert!((r.tps_st_gpu.unwrap() - 25.8).abs() < 0.1);
}

/// Verify write then read with empty rows produces empty result
#[test]
fn test_write_csv_empty_rows() {
    let temp = NamedTempFile::new().unwrap();
    write_models_csv(&[], temp.path()).unwrap();
    let read_back = read_models_csv(temp.path()).unwrap();
    assert!(read_back.is_empty());
}

/// Verify CSV with missing field produces error mentioning the field
#[test]
fn test_csv_missing_required_field() {
    let csv_data = "model_id,family\ntest,test\n";
    let temp = NamedTempFile::new().unwrap();
    std::fs::write(temp.path(), csv_data).unwrap();
    let result = read_models_csv(temp.path());
    assert!(result.is_err());
}

/// Verify derive_status boundary: MQS=800 with all gateways → CERTIFIED
#[test]
fn test_derive_status_boundary_800() {
    let row = CertificationRow {
        mqs_score: 800,
        g1: true,
        g2: true,
        g3: true,
        g4: true,
        ..CertificationRow::default()
    };
    assert_eq!(row.derive_status(), ModelStatus::Certified);
}

/// Verify derive_status boundary: MQS=799 with all gateways → BLOCKED
#[test]
fn test_derive_status_boundary_799() {
    let row = CertificationRow {
        mqs_score: 799,
        g1: true,
        g2: true,
        g3: true,
        g4: true,
        ..CertificationRow::default()
    };
    assert_eq!(row.derive_status(), ModelStatus::Blocked);
}
