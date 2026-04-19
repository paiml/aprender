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
