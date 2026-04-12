#[test]
fn test_falsify_evidence_001_roundtrip_integrity() {
    let export = EvidenceExport::builder()
        .model("Qwen/Qwen2.5-Coder-0.5B-Instruct", "qwen2", "0.5b")
        .format("safetensors")
        .playbook("qwen2.5-coder-0.5b-mvp", "1.0.0", "mvp")
        .summary(47, 27, 20, 0, 1_134_808)
        .mqs(574, "F", true)
        .category_score("inference", 600)
        .category_score("conversion", 400)
        .gate("G1-MODEL-LOADS", true, "Model loaded")
        .gate("G2-BASIC-INFERENCE", true, "Inference works")
        .gate("G3-NO-CRASHES", true, "No crashes")
        .gate(
            "G4-OUTPUT-QUALITY",
            false,
            "Conversion diffs exceed tolerance",
        )
        .build();

    let json = export.to_json().expect("serialize");
    let roundtrip = EvidenceExport::from_json(&json).expect("deserialize");

    // Verify key fields survive round-trip
    assert_eq!(roundtrip.model.hf_repo, export.model.hf_repo);
    assert_eq!(roundtrip.model.family, export.model.family);
    assert_eq!(roundtrip.model.size, export.model.size);
    assert_eq!(roundtrip.playbook.name, export.playbook.name);
    assert_eq!(roundtrip.playbook.tier, export.playbook.tier);
    assert_eq!(
        roundtrip.summary.total_scenarios,
        export.summary.total_scenarios
    );
    assert_eq!(roundtrip.summary.passed, export.summary.passed);
    assert_eq!(roundtrip.mqs.score, export.mqs.score);
    assert_eq!(roundtrip.mqs.grade, export.mqs.grade);
    assert_eq!(roundtrip.gates.len(), export.gates.len());
}

#[test]
fn test_evidence_export_default() {
    let export = EvidenceExport::default();
    assert!(export.schema.contains("apr-qa-evidence"));
    assert_eq!(export.version, "1.0.0");
    assert!(export.model.hf_repo.is_empty());
    assert_eq!(export.mqs.score, 0);
}

#[test]
fn test_builder_model() {
    let export = EvidenceExport::builder()
        .model("org/model", "family", "1b")
        .build();

    assert_eq!(export.model.hf_repo, "org/model");
    assert_eq!(export.model.family, "family");
    assert_eq!(export.model.size, "1b");
}

#[test]
fn test_builder_playbook() {
    let export = EvidenceExport::builder()
        .playbook("test-playbook", "2.0.0", "full")
        .build();

    assert_eq!(export.playbook.name, "test-playbook");
    assert_eq!(export.playbook.version, "2.0.0");
    assert_eq!(export.playbook.tier, "full");
}

#[test]
fn test_builder_summary() {
    let export = EvidenceExport::builder()
        .summary(100, 80, 15, 5, 50000)
        .build();

    assert_eq!(export.summary.total_scenarios, 100);
    assert_eq!(export.summary.passed, 80);
    assert_eq!(export.summary.failed, 15);
    assert_eq!(export.summary.skipped, 5);
    assert!((export.summary.pass_rate - 0.8).abs() < 0.001);
    assert_eq!(export.summary.duration_ms, 50000);
}

#[test]
fn test_builder_mqs() {
    let export = EvidenceExport::builder()
        .mqs(850, "B", true)
        .category_score("inference", 900)
        .category_score("stability", 800)
        .build();

    assert_eq!(export.mqs.score, 850);
    assert_eq!(export.mqs.grade, "B");
    assert!(export.mqs.gateway_passed);
    assert_eq!(export.mqs.category_scores.get("inference"), Some(&900));
    assert_eq!(export.mqs.category_scores.get("stability"), Some(&800));
}

#[test]
fn test_builder_gates() {
    let export = EvidenceExport::builder()
        .gate("G1-MODEL-LOADS", true, "OK")
        .gate("G2-BASIC-INFERENCE", false, "Failed")
        .build();

    assert_eq!(export.gates.len(), 2);
    assert!(export.gates.get("G1-MODEL-LOADS").unwrap().passed);
    assert!(!export.gates.get("G2-BASIC-INFERENCE").unwrap().passed);
}

#[test]
fn test_calculate_pass_rate() {
    let export = EvidenceExport::builder()
        .summary(100, 75, 25, 0, 1000)
        .build();

    assert!((export.calculate_pass_rate() - 0.75).abs() < 0.001);
}

#[test]
fn test_calculate_pass_rate_empty() {
    let export = EvidenceExport::default();
    assert!((export.calculate_pass_rate() - 0.0).abs() < 0.001);
}

#[test]
fn test_all_gateways_passed() {
    let export = EvidenceExport::builder()
        .gate("G0-INTEGRITY", true, "OK")
        .gate("G1-MODEL-LOADS", true, "OK")
        .gate("G2-BASIC-INFERENCE", true, "OK")
        .gate("G3-NO-CRASHES", true, "OK")
        .gate("G4-OUTPUT-QUALITY", true, "OK")
        .build();

    assert!(export.all_gateways_passed());
}

#[test]
fn test_all_gateways_g0_failed() {
    let export = EvidenceExport::builder()
        .gate("G0-INTEGRITY", false, "Config mismatch")
        .gate("G1-MODEL-LOADS", true, "OK")
        .gate("G2-BASIC-INFERENCE", true, "OK")
        .gate("G3-NO-CRASHES", true, "OK")
        .gate("G4-OUTPUT-QUALITY", true, "OK")
        .build();

    assert!(!export.all_gateways_passed());
}

#[test]
fn test_all_gateways_one_failed() {
    let export = EvidenceExport::builder()
        .gate("G1-MODEL-LOADS", true, "OK")
        .gate("G2-BASIC-INFERENCE", true, "OK")
        .gate("G3-NO-CRASHES", false, "Crashed")
        .gate("G4-OUTPUT-QUALITY", true, "OK")
        .build();

    assert!(!export.all_gateways_passed());
}

#[test]
fn test_all_gateways_missing() {
    let export = EvidenceExport::builder()
        .gate("G1-MODEL-LOADS", true, "OK")
        .build();

    assert!(!export.all_gateways_passed());
}

#[test]
fn test_derive_status_certified() {
    let export = EvidenceExport::builder().mqs(850, "B", true).build();

    assert_eq!(export.derive_status(), "CERTIFIED");
}

#[test]
fn test_derive_status_blocked_low_score() {
    let export = EvidenceExport::builder().mqs(500, "F", true).build();

    assert_eq!(export.derive_status(), "BLOCKED");
}

#[test]
fn test_derive_status_blocked_gateway_failed() {
    let export = EvidenceExport::builder().mqs(900, "A", false).build();

    assert_eq!(export.derive_status(), "BLOCKED");
}

#[test]
fn test_derive_status_untested() {
    let export = EvidenceExport::default();
    assert_eq!(export.derive_status(), "UNTESTED");
}

/// Bug #58: Score 0 with evidence present must be BLOCKED, not UNTESTED.
/// A model that was tested and failed gateways has evidence but MQS=0.
#[test]
fn test_derive_status_blocked_score_zero_with_evidence() {
    let mut export = EvidenceExport::builder().mqs(0, "F", false).build();
    // Add some evidence to simulate a tested-but-failed model
    export.evidence.push(serde_json::json!({
        "gate_id": "G1-MODEL-LOADS",
        "outcome": "Falsified",
        "output": "failed to load"
    }));
    assert_eq!(
        export.derive_status(),
        "BLOCKED",
        "Score 0 with evidence should be BLOCKED, not UNTESTED"
    );
}

#[test]
fn test_to_json_contains_schema() {
    let export = EvidenceExport::default();
    let json = export.to_json().expect("serialize");
    assert!(json.contains("$schema"));
    assert!(json.contains("apr-qa-evidence.schema.json"));
}

#[test]
fn test_evidence_array() {
    let evidence = vec![
        serde_json::json!({"id": "1", "outcome": "Corroborated"}),
        serde_json::json!({"id": "2", "outcome": "Falsified"}),
    ];

    let export = EvidenceExport::builder().evidence(evidence).build();

    assert_eq!(export.evidence.len(), 2);
}

#[test]
fn test_builder_format() {
    let export = EvidenceExport::builder().format("gguf").build();

    assert_eq!(export.model.format, "gguf");
}

#[test]
fn test_summary_zero_total() {
    let export = EvidenceExport::builder().summary(0, 0, 0, 0, 0).build();

    assert_eq!(export.summary.pass_rate, 0.0);
}

#[test]
fn test_serde_gate_result() {
    let gate = GateResult {
        passed: true,
        reason: "Test passed".to_string(),
    };

    let json = serde_json::to_string(&gate).expect("serialize");
    assert!(json.contains("true"));
    assert!(json.contains("Test passed"));

    let deserialized: GateResult = serde_json::from_str(&json).expect("deserialize");
    assert!(deserialized.passed);
    assert_eq!(deserialized.reason, "Test passed");
}

#[test]
fn test_clone_export() {
    let export = EvidenceExport::builder()
        .model("test/model", "test", "1b")
        .mqs(750, "C", true)
        .build();

    let cloned = export.clone();
    assert_eq!(cloned.model.hf_repo, export.model.hf_repo);
    assert_eq!(cloned.mqs.score, export.mqs.score);
}

// FALSIFY-MQS-001: MqsScore conversion integrity
//
// Falsification hypothesis: "from_mqs_score loses critical data"
// If converted export doesn't preserve MQS score, grade, and gateway status, broken.
#[test]
fn test_falsify_mqs_001_conversion_integrity() {
    use crate::mqs::{CategoryScores, GatewayResult as MqsGateway, MqsScore};

    let mqs = MqsScore {
        model_id: "test/model".to_string(),
        raw_score: 850,
        normalized_score: 85.0,
        grade: "B".to_string(),
        gateways: vec![
            MqsGateway::passed("G1", "Model loads"),
            MqsGateway::passed("G2", "Inference works"),
            MqsGateway::passed("G3", "No crashes"),
            MqsGateway::failed("G4", "Output quality", "Conversion diffs"),
        ],
        gateways_passed: false,
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
        hf_repo: "Qwen/Qwen2.5-Coder-0.5B-Instruct".to_string(),
        family: "qwen2".to_string(),
        size: "0.5b".to_string(),
        format: "safetensors".to_string(),
    };

    let playbook = PlaybookMeta {
        name: "qwen2.5-coder-0.5b-mvp".to_string(),
        version: "1.0.0".to_string(),
        tier: "mvp".to_string(),
    };

    let evidence = vec![serde_json::json!({
        "id": "1",
        "outcome": "Corroborated",
        "metrics": {"duration_ms": 1000}
    })];

    let export = EvidenceExport::from_mqs_score(&mqs, evidence, model, playbook);

    // Verify MQS data preserved
    assert_eq!(export.mqs.score, 850);
    assert_eq!(export.mqs.grade, "B");
    assert!(!export.mqs.gateway_passed);

    // Verify summary computed correctly
    assert_eq!(export.summary.total_scenarios, 100);
    assert_eq!(export.summary.passed, 85);
    assert_eq!(export.summary.failed, 15);

    // Verify model metadata preserved
    assert_eq!(export.model.hf_repo, "Qwen/Qwen2.5-Coder-0.5B-Instruct");
    assert_eq!(export.model.family, "qwen2");

    // Verify playbook metadata preserved
    assert_eq!(export.playbook.name, "qwen2.5-coder-0.5b-mvp");
    assert_eq!(export.playbook.tier, "mvp");

    // Verify gates converted
    assert!(export.gates.contains_key("G1-MODEL-LOADS"));
    assert!(export.gates.get("G1-MODEL-LOADS").unwrap().passed);
    assert!(!export.gates.get("G4-OUTPUT-QUALITY").unwrap().passed);
}

#[test]
fn test_from_mqs_score_category_breakdown() {
    use crate::mqs::{CategoryScores, GatewayResult as MqsGateway, MqsScore};

    let mqs = MqsScore {
        model_id: "test".to_string(),
        raw_score: 500,
        normalized_score: 50.0,
        grade: "F".to_string(),
        gateways: vec![MqsGateway::passed("G1", "OK")],
        gateways_passed: true,
        categories: CategoryScores {
            qual: 100,
            perf: 80,
            stab: 100,
            comp: 70,
            edge: 80,
            regr: 70,
        },
        total_tests: 50,
        tests_passed: 25,
        tests_failed: 25,
        penalties: vec![],
        total_penalty: 0,
        proof_bonus: None,
    };

    let export = EvidenceExport::from_mqs_score(
        &mqs,
        vec![],
        ModelMeta {
            hf_repo: "test".to_string(),
            family: "test".to_string(),
            size: "1b".to_string(),
            format: "gguf".to_string(),
        },
        PlaybookMeta {
            name: "test".to_string(),
            version: "1.0.0".to_string(),
            tier: "smoke".to_string(),
        },
    );

    // Verify category scores converted (keys are lowercased)
    assert_eq!(export.mqs.category_scores.get("qual"), Some(&100));
    assert_eq!(export.mqs.category_scores.get("perf"), Some(&80));
    assert_eq!(export.mqs.category_scores.get("stab"), Some(&100));
}

#[test]
fn test_from_mqs_score_pass_rate() {
    use crate::mqs::{CategoryScores, MqsScore};

    let mqs = MqsScore {
        model_id: "test".to_string(),
        raw_score: 750,
        normalized_score: 75.0,
        grade: "C".to_string(),
        gateways: vec![],
        gateways_passed: true,
        categories: CategoryScores::default(),
        total_tests: 80,
        tests_passed: 60,
        tests_failed: 20,
        penalties: vec![],
        total_penalty: 0,
        proof_bonus: None,
    };

    let export = EvidenceExport::from_mqs_score(
        &mqs,
        vec![],
        ModelMeta {
            hf_repo: "t".to_string(),
            family: "t".to_string(),
            size: "1b".to_string(),
            format: "gguf".to_string(),
        },
        PlaybookMeta {
            name: "t".to_string(),
            version: "1".to_string(),
            tier: "smoke".to_string(),
        },
    );

    assert!((export.summary.pass_rate - 0.75).abs() < 0.001);
}

// ── PMAT-267: Oracle Integration Tests ──────────────────────────────────
//
// These tests verify the contract between EvidenceExport and CertificationRow,
// ensuring the oracle can correctly consume certification data.

// FALSIFY-ORACLE-001: Evidence→Certification data contract
//
// Falsification hypothesis: "EvidenceExport status derivation differs from CertificationRow"
// If derive_status() returns different values for equivalent inputs, oracle is broken.
#[test]
fn test_falsify_oracle_001_status_contract() {
    use crate::certification_data::{CertificationRow, ModelStatus};

    // Test case 1: CERTIFIED (MQS >= 800, gateways passed)
    let export = EvidenceExport::builder().mqs(850, "B", true).build();

    let cert_row = CertificationRow {
        mqs_score: 850,
        g1: true,
        g2: true,
        g3: true,
        g4: true,
        ..Default::default()
    };

    assert_eq!(export.derive_status(), "CERTIFIED");
    assert_eq!(cert_row.derive_status(), ModelStatus::Certified);

    // Test case 2: BLOCKED (MQS < 800)
    let export_blocked = EvidenceExport::builder().mqs(500, "F", true).build();

    let cert_row_blocked = CertificationRow {
        mqs_score: 500,
        g1: true,
        g2: true,
        g3: true,
        g4: true,
        ..Default::default()
    };

    assert_eq!(export_blocked.derive_status(), "BLOCKED");
    assert_eq!(cert_row_blocked.derive_status(), ModelStatus::Blocked);

    // Test case 3: BLOCKED (gateway failed)
    let export_gw_fail = EvidenceExport::builder().mqs(900, "A", false).build();

    let cert_row_gw_fail = CertificationRow {
        mqs_score: 900,
        g1: true,
        g2: true,
        g3: false, // Gateway failed
        g4: true,
        ..Default::default()
    };

    assert_eq!(export_gw_fail.derive_status(), "BLOCKED");
    assert_eq!(cert_row_gw_fail.derive_status(), ModelStatus::Blocked);
}

// FALSIFY-ORACLE-002: Grade derivation consistency
//
// Falsification hypothesis: "Grade thresholds differ between modules"
// Both modules must use identical grade thresholds.
