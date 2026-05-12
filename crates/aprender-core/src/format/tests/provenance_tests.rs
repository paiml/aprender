//! Harness tests for C-APR-PROVENANCE (AC-SHIP2-012 / FALSIFY-SHIP-022).
//!
//! These tests discharge INV-APR-PROV-001 and the JSON-serialization side
//! of INV-APR-PROV-002. The text-mode rendering half of INV-APR-PROV-002
//! is discharged by `apr-cli` tests on `output_metadata_text`.
//!
//! Contract: `contracts/apr-provenance-v1.yaml`.

use crate::format::v2::AprV2Metadata;

/// GATE-APR-PROV-001 / INV-APR-PROV-001: AprV2Metadata round-trips all
/// three provenance fields as NAMED JSON keys (not buried in `custom`).
#[test]
fn falsify_ship_022_apr_metadata_provenance_round_trip() {
    let meta = AprV2Metadata {
        license: Some("Apache-2.0".to_string()),
        data_source: Some("teacher-only".to_string()),
        data_license: Some("Apache-2.0".to_string()),
        ..Default::default()
    };

    let json = meta.to_json().expect("serialize AprV2Metadata");
    let reparsed = AprV2Metadata::from_json(&json).expect("deserialize AprV2Metadata");

    assert_eq!(
        reparsed.license,
        Some("Apache-2.0".to_string()),
        "license must round-trip byte-identically"
    );
    assert_eq!(
        reparsed.data_source,
        Some("teacher-only".to_string()),
        "data_source must round-trip byte-identically"
    );
    assert_eq!(
        reparsed.data_license,
        Some("Apache-2.0".to_string()),
        "data_license must round-trip byte-identically"
    );

    // Guard against provenance fields silently leaking into `custom`
    // (which would bypass the named-field invariant).
    assert!(
        !reparsed.custom.contains_key("license"),
        "license must be a named field, not in custom"
    );
    assert!(
        !reparsed.custom.contains_key("data_source"),
        "data_source must be a named field, not in custom"
    );
    assert!(
        !reparsed.custom.contains_key("data_license"),
        "data_license must be a named field, not in custom"
    );
}

/// GATE-APR-PROV-002 / INV-APR-PROV-002 (JSON half): when all three
/// provenance fields are None, the serialized JSON still contains each
/// key as an explicit null. This proves no `skip_serializing_if` has
/// been added to AprV2Metadata — FM-APR-PROV-SILENT-SKIP.
#[test]
fn falsify_ship_022_inspect_emits_provenance_keys() {
    let meta = AprV2Metadata {
        license: None,
        data_source: None,
        data_license: None,
        ..Default::default()
    };

    let json_bytes = meta.to_json().expect("serialize AprV2Metadata");
    let json_str = std::str::from_utf8(&json_bytes).expect("utf-8 JSON");

    let parsed: serde_json::Value = serde_json::from_str(json_str).expect("parse JSON");
    let obj = parsed.as_object().expect("JSON object at top level");

    for key in ["license", "data_source", "data_license"] {
        assert!(
            obj.contains_key(key),
            "AprV2Metadata JSON must emit key `{key}` even when None (no skip_serializing_if); \
             violating this silently hides provenance from auditors (FM-APR-PROV-SILENT-SKIP)"
        );
        assert!(
            obj[key].is_null(),
            "key `{key}` must serialize as null when None, got {:?}",
            obj[key]
        );
    }
}

/// GATE-APR-PROV-003 counter-test (partial): a struct with mixed
/// Some/None values still round-trips, and the None values survive as
/// None (not mangled to empty strings).
#[test]
fn falsify_ship_022_partial_provenance_round_trip() {
    let meta = AprV2Metadata {
        license: Some("CC-BY-4.0".to_string()),
        data_source: None,
        data_license: Some("CC-BY-4.0".to_string()),
        ..Default::default()
    };

    let json = meta.to_json().expect("serialize AprV2Metadata");
    let reparsed = AprV2Metadata::from_json(&json).expect("deserialize AprV2Metadata");

    assert_eq!(reparsed.license, Some("CC-BY-4.0".to_string()));
    assert_eq!(
        reparsed.data_source, None,
        "None must survive round-trip as None, not coerced to Some(\"\")"
    );
    assert_eq!(reparsed.data_license, Some("CC-BY-4.0".to_string()));
}

/// GATE-APR-PROV-004 / FALSIFY-SHIP-009 algorithm-level PARTIAL
/// discharge: the SAME AprV2Metadata + serde-JSON decision rule that
/// discharges AC-SHIP2-012 (MODEL-2 sovereign) also discharges
/// AC-SHIP1-009 (MODEL-1 teacher license & provenance recorded in
/// model.apr metadata).
///
/// This test constructs a teacher-representative metadata object —
/// paiml/qwen2.5-coder-7b-apache-q4k-v1 shipped at HF under the
/// Apache-2.0 license, distilled from Qwen2.5-Coder-7B-Instruct
/// (also Apache-2.0). Verifies the three required AC-SHIP1-009
/// fields (license, data_source, data_license) survive JSON
/// round-trip byte-identically. The fn is model-agnostic — no
/// teacher-specific logic beyond the input values.
#[test]
fn falsify_ship_009_apr_metadata_applies_to_model_1_teacher() {
    let teacher_meta = AprV2Metadata {
        license: Some("apache-2.0".to_string()),
        data_source: Some("qwen2.5-coder-7b-instruct".to_string()),
        data_license: Some("apache-2.0".to_string()),
        ..Default::default()
    };

    let json = teacher_meta
        .to_json()
        .expect("teacher AprV2Metadata must serialize");
    let reparsed = AprV2Metadata::from_json(&json).expect("teacher AprV2Metadata must deserialize");

    assert_eq!(
        reparsed.license,
        Some("apache-2.0".to_string()),
        "AC-SHIP1-009: teacher license must round-trip byte-identically"
    );
    assert_eq!(
        reparsed.data_source,
        Some("qwen2.5-coder-7b-instruct".to_string()),
        "AC-SHIP1-009: teacher data_source must round-trip byte-identically"
    );
    assert_eq!(
        reparsed.data_license,
        Some("apache-2.0".to_string()),
        "AC-SHIP1-009: teacher data_license must round-trip byte-identically"
    );

    assert!(
        !reparsed.custom.contains_key("license"),
        "license must remain a named field for MODEL-1 teacher too"
    );
    assert!(
        !reparsed.custom.contains_key("data_source"),
        "data_source must remain a named field for MODEL-1 teacher too"
    );
    assert!(
        !reparsed.custom.contains_key("data_license"),
        "data_license must remain a named field for MODEL-1 teacher too"
    );
}

/// GATE-APR-PROV-004 YAML binding: parses apr-provenance-v1.yaml and
/// asserts the gate block correctly binds AC-SHIP1-009 /
/// FALSIFY-SHIP-009 with DISCHARGED status (was PARTIAL_ALGORITHM_LEVEL
/// at v1.1.0; flipped to DISCHARGED at v1.2.0 on 2026-04-25 via live
/// `apr stamp` fixture-swap on the canonical lambda-labs staging
/// artifact). Falsifier: if the contract is edited to drop AC-SHIP1-009
/// binding or downgrade the discharge marker, this test fails.
#[test]
fn falsify_ship_009_gate_apr_prov_004_has_partial_discharge_marker() {
    const CONTRACT_YAML: &str = include_str!("../../../../../contracts/apr-provenance-v1.yaml");

    let doc: serde_yaml::Value =
        serde_yaml::from_str(CONTRACT_YAML).expect("apr-provenance-v1.yaml must parse as YAML");

    let gates = doc["gates"]
        .as_sequence()
        .expect("gates must be a sequence in apr-provenance-v1");
    let gate = gates
        .iter()
        .find(|g| g["id"].as_str() == Some("GATE-APR-PROV-004"))
        .expect("GATE-APR-PROV-004 must exist in apr-provenance-v1");

    assert_eq!(
        gate["falsification_id"].as_str(),
        Some("FALSIFY-SHIP-009"),
        "GATE-APR-PROV-004 must bind FALSIFY-SHIP-009",
    );
    assert_eq!(
        gate["binds_to"].as_str(),
        Some("AC-SHIP1-009"),
        "GATE-APR-PROV-004 must bind AC-SHIP1-009 (MODEL-1 teacher license/provenance)",
    );
    assert_eq!(
        gate["discharge_status"].as_str(),
        Some("DISCHARGED"),
        "GATE-APR-PROV-004 must advertise DISCHARGED \
         (live `apr stamp` fixture-swap on canonical lambda-labs staging \
         artifact at v1.2.0; previous PARTIAL_ALGORITHM_LEVEL at v1.1.0)",
    );
    assert!(
        gate["discharged_evidence"].is_mapping(),
        "GATE-APR-PROV-004 DISCHARGED status requires a discharged_evidence \
         block documenting the host, pre/post sha256, and tooling chain",
    );
    assert_eq!(
        gate["discharged_evidence"]["post_stamp"]["provenance_state"]["license"].as_str(),
        Some("Apache-2.0"),
        "discharged_evidence.post_stamp.provenance_state.license must equal Apache-2.0",
    );
    assert_eq!(
        gate["discharged_evidence"]["host"].as_str(),
        Some("noah-Lambda-Vector"),
        "discharged_evidence.host must pin to the lambda-labs RTX 4090 host",
    );
    assert_eq!(
        gate["ship_blocking"].as_bool(),
        Some(true),
        "GATE-APR-PROV-004 must be ship_blocking=true",
    );
    let evidence = gate["evidence_discharged_by"]
        .as_sequence()
        .expect("GATE-APR-PROV-004 must have evidence_discharged_by");
    assert!(
        !evidence.is_empty(),
        "GATE-APR-PROV-004 evidence_discharged_by must list at least one test",
    );
    let live_evidence = gate["discharged_evidence"]["evidence_discharged_by_live"]
        .as_sequence()
        .expect(
            "GATE-APR-PROV-004 DISCHARGED requires \
             discharged_evidence.evidence_discharged_by_live (live RTX 4090 evidence list)",
        );
    assert!(
        !live_evidence.is_empty(),
        "GATE-APR-PROV-004 evidence_discharged_by_live must list at least one live observation",
    );
}
