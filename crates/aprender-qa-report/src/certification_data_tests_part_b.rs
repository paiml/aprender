// ── FALSIFY-CERT-002: Status derivation from MQS score ────────────────────
//
// Prediction: status is deterministically derived from mqs_score and g1-g4 gateways.
// Per Popper (1959), this test attempts to falsify the status derivation algorithm.

/// Falsify status derivation algorithm across all status outcomes
#[test]
fn test_falsify_cert_002_status_derivation() {
    // All gateways passed, high score -> CERTIFIED
    let certified = CertificationRow {
        mqs_score: 850,
        g1: true,
        g2: true,
        g3: true,
        g4: true,
        ..CertificationRow::default()
    };
    assert_eq!(
        certified.derive_status(),
        ModelStatus::Certified,
        "All gateways passed + score >= 800 should be CERTIFIED"
    );

    // All gateways passed, low score -> BLOCKED
    let blocked_low = CertificationRow {
        mqs_score: 500,
        g1: true,
        g2: true,
        g3: true,
        g4: true,
        ..CertificationRow::default()
    };
    assert_eq!(
        blocked_low.derive_status(),
        ModelStatus::Blocked,
        "All gateways passed + score < 800 should be BLOCKED"
    );

    // Gateway G3 failed, high score -> BLOCKED
    let blocked_gw = CertificationRow {
        mqs_score: 950,
        g1: true,
        g2: true,
        g3: false, // Gateway failure
        g4: true,
        ..CertificationRow::default()
    };
    assert_eq!(
        blocked_gw.derive_status(),
        ModelStatus::Blocked,
        "Gateway failed should always be BLOCKED"
    );

    // Score 0 with g1=false -> PENDING (never tested)
    let pending = CertificationRow {
        mqs_score: 0,
        g1: false,
        g2: false,
        g3: false,
        g4: false,
        ..CertificationRow::default()
    };
    assert_eq!(
        pending.derive_status(),
        ModelStatus::Pending,
        "Score 0 with g1=false should be PENDING (not yet tested)"
    );
}

// ── CERT-DATA-PARSE: CSV parsing edge cases ───────────────────────────────

/// Verify CSV with invalid boolean value produces error
#[test]
fn test_csv_invalid_boolean() {
    let csv_data = r#"model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4,tps_gguf_cpu,tps_gguf_gpu,tps_apr_cpu,tps_apr_gpu,tps_st_cpu,tps_st_gpu,provenance_verified
test/model,family,0.5B,tiny,BLOCKED,246,F,quick,2026-02-04T13:28:18.663+00:00,INVALID_BOOL,true,true,true,,,,,,,false
"#;
    let temp = NamedTempFile::new().unwrap();
    std::fs::write(temp.path(), csv_data).unwrap();
    let result = read_models_csv(temp.path());
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Invalid boolean"));
}

/// Verify CSV with invalid mqs_score produces error
#[test]
fn test_csv_invalid_mqs_score() {
    let csv_data = r#"model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4,tps_gguf_cpu,tps_gguf_gpu,tps_apr_cpu,tps_apr_gpu,tps_st_cpu,tps_st_gpu,provenance_verified
test/model,family,0.5B,tiny,BLOCKED,not_a_number,F,quick,2026-02-04T13:28:18.663+00:00,true,true,true,true,,,,,,,false
"#;
    let temp = NamedTempFile::new().unwrap();
    std::fs::write(temp.path(), csv_data).unwrap();
    let result = read_models_csv(temp.path());
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Invalid mqs_score"));
}

/// Verify CSV with invalid timestamp produces error
#[test]
fn test_csv_invalid_timestamp() {
    let csv_data = r#"model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4,tps_gguf_cpu,tps_gguf_gpu,tps_apr_cpu,tps_apr_gpu,tps_st_cpu,tps_st_gpu,provenance_verified
test/model,family,0.5B,tiny,BLOCKED,246,F,quick,not-a-date,true,true,true,true,,,,,,,false
"#;
    let temp = NamedTempFile::new().unwrap();
    std::fs::write(temp.path(), csv_data).unwrap();
    let result = read_models_csv(temp.path());
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Invalid timestamp"));
}

/// Verify CSV with kernel_proof_ref column parsed correctly
#[test]
fn test_csv_with_kernel_proof_ref() {
    let csv_data = r#"model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4,tps_gguf_cpu,tps_gguf_gpu,tps_apr_cpu,tps_apr_gpu,tps_st_cpu,tps_st_gpu,provenance_verified,kernel_proof_ref
test/model,family,0.5B,tiny,CERTIFIED,900,A,mvp,2026-02-04T13:28:18.663+00:00,true,true,true,true,25.0,100.0,22.0,90.0,5.0,30.0,true,L3
"#;
    let temp = NamedTempFile::new().unwrap();
    std::fs::write(temp.path(), csv_data).unwrap();
    let rows = read_models_csv(temp.path()).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].kernel_proof_ref, Some("L3".to_string()));
}

/// Verify CSV with empty kernel_proof_ref column is None
#[test]
fn test_csv_with_empty_kernel_proof_ref() {
    let csv_data = r#"model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4,tps_gguf_cpu,tps_gguf_gpu,tps_apr_cpu,tps_apr_gpu,tps_st_cpu,tps_st_gpu,provenance_verified,kernel_proof_ref
test/model,family,0.5B,tiny,CERTIFIED,900,A,mvp,2026-02-04T13:28:18.663+00:00,true,true,true,true,25.0,100.0,22.0,90.0,5.0,30.0,true,
"#;
    let temp = NamedTempFile::new().unwrap();
    std::fs::write(temp.path(), csv_data).unwrap();
    let rows = read_models_csv(temp.path()).unwrap();
    assert_eq!(rows.len(), 1);
    assert!(rows[0].kernel_proof_ref.is_none());
}

/// Verify boolean "1" and "yes" values work in CSV
#[test]
fn test_csv_boolean_alternate_values() {
    let csv_data = r#"model_id,family,parameters,size_category,status,mqs_score,grade,certified_tier,last_certified,g1,g2,g3,g4,tps_gguf_cpu,tps_gguf_gpu,tps_apr_cpu,tps_apr_gpu,tps_st_cpu,tps_st_gpu,provenance_verified
test/model,family,0.5B,tiny,BLOCKED,246,F,quick,2026-02-04T13:28:18.663+00:00,1,yes,0,no,,,,,,,false
"#;
    let temp = NamedTempFile::new().unwrap();
    std::fs::write(temp.path(), csv_data).unwrap();
    let rows = read_models_csv(temp.path()).unwrap();
    assert_eq!(rows.len(), 1);
    assert!(rows[0].g1);
    assert!(rows[0].g2);
    assert!(!rows[0].g3);
    assert!(!rows[0].g4);
}

// ── FALSIFY-CERT-003: Grade derivation from MQS score ─────────────────────
//
// Prediction: grade is deterministically derived from mqs_score using fixed thresholds.
// Per Popper (1959), this test attempts to falsify the grade derivation algorithm.
//
// Grade thresholds (from derive_grade, aligned with apr-qa-certify::grade_from_score):
// A+: 950-1000
// A: 900-949
// B+: 850-899
// B: 800-849  (CERTIFIED threshold)
// C: 700-799
// F: 0-699

/// Verify grade derivation from MQS score matches fixed threshold boundaries
#[test]
fn test_falsify_cert_003_grade_derivation() {
    // Helper to derive grade from score
    let grade_for = |score: u32| -> String {
        CertificationRow {
            mqs_score: score,
            ..CertificationRow::default()
        }
        .derive_grade()
    };

    // A+ grade: 950-1000
    assert_eq!(grade_for(1000), "A+", "1000 should be A+");
    assert_eq!(grade_for(950), "A+", "950 (lower bound of A+) should be A+");

    // A grade: 900-949
    assert_eq!(grade_for(949), "A", "949 (upper bound of A) should be A");
    assert_eq!(grade_for(900), "A", "900 (lower bound) should be A");

    // B+ grade: 850-899
    assert_eq!(grade_for(899), "B+", "899 (upper bound of B+) should be B+");
    assert_eq!(grade_for(850), "B+", "850 (lower bound) should be B+");

    // B grade: 800-849
    assert_eq!(grade_for(849), "B", "849 (upper bound of B) should be B");
    assert_eq!(grade_for(800), "B", "800 (lower bound) should be B");

    // C grade: 700-799
    assert_eq!(grade_for(799), "C", "799 (upper bound of C) should be C");
    assert_eq!(grade_for(700), "C", "700 (lower bound) should be C");

    // F grade: 0-699
    assert_eq!(grade_for(699), "F", "699 (upper bound of F) should be F");
    assert_eq!(grade_for(500), "F", "500 should be F");
    assert_eq!(grade_for(0), "F", "0 should be F");
}
