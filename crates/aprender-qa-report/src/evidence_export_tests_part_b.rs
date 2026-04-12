/// Validate MQS score-to-grade mapping matches the grading contract
#[test]
fn test_falsify_oracle_002_grade_contract() {
    use crate::certification_data::CertificationRow;

    let grade_cases = [
        (950, "A+"),
        (900, "A"),
        (850, "B+"),
        (800, "B"),
        (700, "C"),
        (600, "F"),
        (500, "F"),
        (400, "F"),
        (300, "F"),
        (0, "F"),
    ];

    for (score, expected_grade) in grade_cases {
        let cert_row = CertificationRow {
            mqs_score: score,
            ..Default::default()
        };

        assert_eq!(
            cert_row.derive_grade(),
            expected_grade,
            "Grade mismatch for score {score}"
        );
    }
}

// FALSIFY-ORACLE-003: Evidence JSON schema compliance
//
// Falsification hypothesis: "Serialized JSON lacks required oracle fields"
// Oracle requires: $schema, model.hf_repo, mqs.score, mqs.grade, gates
/// Verify serialized evidence JSON contains all required oracle fields
#[test]
fn test_falsify_oracle_003_schema_compliance() {
    let export = EvidenceExport::builder()
        .model("Qwen/Qwen2.5-Coder-0.5B-Instruct", "qwen2", "0.5b")
        .mqs(850, "B", true)
        .gate("G1-MODEL-LOADS", true, "OK")
        .gate("G2-BASIC-INFERENCE", true, "OK")
        .gate("G3-NO-CRASHES", true, "OK")
        .gate("G4-OUTPUT-QUALITY", true, "OK")
        .build();

    let json = export.to_json().expect("serialize");

    // Verify required oracle fields present
    assert!(json.contains("\"$schema\""), "Missing $schema field");
    assert!(json.contains("\"hf_repo\""), "Missing model.hf_repo field");
    assert!(json.contains("\"score\""), "Missing mqs.score field");
    assert!(json.contains("\"grade\""), "Missing mqs.grade field");
    assert!(json.contains("\"gates\""), "Missing gates field");
    assert!(
        json.contains("\"gateway_passed\""),
        "Missing gateway_passed field"
    );

    // Verify specific values
    assert!(json.contains("Qwen/Qwen2.5-Coder-0.5B-Instruct"));
    assert!(json.contains("850"));
    assert!(json.contains("\"B\""));
}

// FALSIFY-ORACLE-004: CertificationRow↔EvidenceExport field mapping
//
// Falsification hypothesis: "Field names differ between CSV and JSON"
// Oracle must be able to map between CSV columns and JSON fields.
/// Verify CertificationRow and EvidenceExport field names align correctly
#[test]
fn test_falsify_oracle_004_field_mapping() {
    use crate::certification_data::CertificationRow;

    // Create equivalent data in both formats
    let export = EvidenceExport::builder()
        .model("test/model", "test-family", "1b")
        .playbook("test-1b-mvp", "1.0.0", "mvp")
        .summary(100, 85, 15, 0, 50000)
        .mqs(850, "B", true)
        .gate("G1-MODEL-LOADS", true, "OK")
        .gate("G2-BASIC-INFERENCE", true, "OK")
        .gate("G3-NO-CRASHES", true, "OK")
        .gate("G4-OUTPUT-QUALITY", true, "OK")
        .build();

    let cert_row = CertificationRow {
        model_id: "test/model".to_string(),
        family: "test-family".to_string(),
        parameters: "1B".to_string(),
        mqs_score: 850,
        grade: "B".to_string(),
        certified_tier: "mvp".to_string(),
        g1: true,
        g2: true,
        g3: true,
        g4: true,
        ..Default::default()
    };

    // Verify field mappings
    assert_eq!(export.model.hf_repo, cert_row.model_id);
    assert_eq!(export.model.family, cert_row.family);
    assert_eq!(export.mqs.score, cert_row.mqs_score);
    assert_eq!(export.mqs.grade, cert_row.grade);
    assert_eq!(export.playbook.tier, cert_row.certified_tier);

    // Verify gateway consistency
    assert_eq!(
        export.gates.get("G1-MODEL-LOADS").unwrap().passed,
        cert_row.g1
    );
    assert_eq!(
        export.gates.get("G2-BASIC-INFERENCE").unwrap().passed,
        cert_row.g2
    );
    assert_eq!(
        export.gates.get("G3-NO-CRASHES").unwrap().passed,
        cert_row.g3
    );
    assert_eq!(
        export.gates.get("G4-OUTPUT-QUALITY").unwrap().passed,
        cert_row.g4
    );
}

// FALSIFY-ORACLE-005: Evidence export reproducibility
//
// Falsification hypothesis: "Same input produces different exports"
// If two exports from identical MqsScore differ (except timestamp), broken.
/// Verify identical MqsScore inputs produce identical exports (except timestamp)
#[test]
fn test_falsify_oracle_005_reproducibility() {
    use crate::mqs::{CategoryScores, GatewayResult as MqsGateway, MqsScore};

    let mqs = MqsScore {
        model_id: "test/model".to_string(),
        raw_score: 850,
        normalized_score: 85.0,
        grade: "B".to_string(),
        gateways: vec![
            MqsGateway::passed("G1", "Model loads"),
            MqsGateway::passed("G2", "Inference works"),
        ],
        gateways_passed: true,
        categories: CategoryScores {
            qual: 180,
            perf: 140,
            stab: 190,
            comp: 130,
            edge: 120,
            regr: 90,
        },
        total_tests: 100,
        tests_passed: 85,
        tests_failed: 15,
        penalties: vec![],
        total_penalty: 0,
        proof_bonus: None,
    };

    let model = ModelMeta {
        hf_repo: "test/model".to_string(),
        family: "test".to_string(),
        size: "1b".to_string(),
        format: "safetensors".to_string(),
    };

    let playbook = PlaybookMeta {
        name: "test-1b".to_string(),
        version: "1.0.0".to_string(),
        tier: "mvp".to_string(),
    };

    let export1 = EvidenceExport::from_mqs_score(&mqs, vec![], model.clone(), playbook.clone());
    let export2 = EvidenceExport::from_mqs_score(&mqs, vec![], model, playbook);

    // All fields except timestamp should be identical
    assert_eq!(export1.mqs.score, export2.mqs.score);
    assert_eq!(export1.mqs.grade, export2.mqs.grade);
    assert_eq!(export1.mqs.gateway_passed, export2.mqs.gateway_passed);
    assert_eq!(
        export1.summary.total_scenarios,
        export2.summary.total_scenarios
    );
    assert_eq!(export1.summary.passed, export2.summary.passed);
    assert_eq!(export1.summary.failed, export2.summary.failed);
    assert_eq!(export1.model.hf_repo, export2.model.hf_repo);
    assert_eq!(export1.playbook.name, export2.playbook.name);
}
