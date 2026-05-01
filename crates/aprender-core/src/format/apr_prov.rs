// SHIP-TWO-001 — `apr-provenance-v1` algorithm-level PARTIAL
// discharge for GATE-APR-PROV-001..004 (closes 4/4).
//
// Contract: `contracts/apr-provenance-v1.yaml`.
// Spec: AC-SHIP1-009 (MODEL-1 teacher license + data provenance
// recorded in model.apr metadata).
//
// All four gates pin different invariants over the three named
// provenance fields: license, data_source, data_license.

/// Forbidden placeholder values that must be rejected as
/// "missing provenance".
pub const AC_APR_PROV_FORBIDDEN_PLACEHOLDERS: &[&[u8]] = &[
    b"unknown",
    b"UNKNOWN",
    b"Unknown",
    b"(missing)",
    b"TODO",
    b"todo",
    b"???",
];

// ===========================================================================
// GATE-APR-PROV-001 — round-trip serde for license/data_source/data_license
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AprProv001Verdict {
    Pass,
    Fail,
}

/// Pure verdict function for `GATE-APR-PROV-001`.
///
/// Pass iff the three named fields round-trip byte-identical
/// through serialize+deserialize.
#[must_use]
pub fn verdict_from_serde_round_trip(
    license_round_trip: bool,
    data_source_round_trip: bool,
    data_license_round_trip: bool,
) -> AprProv001Verdict {
    if license_round_trip && data_source_round_trip && data_license_round_trip {
        AprProv001Verdict::Pass
    } else {
        AprProv001Verdict::Fail
    }
}

// ===========================================================================
// GATE-APR-PROV-002 — apr inspect surfaces fields in text + JSON
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AprProv002Verdict {
    Pass,
    Fail,
}

/// Pure verdict function for `GATE-APR-PROV-002`.
///
/// Pass iff each of the three field names appears in BOTH text
/// and JSON outputs as a key-style token (followed by `:` or
/// `":` to disambiguate `license` from `data_license`).
#[must_use]
pub fn verdict_from_inspect_surfaces(
    text_output: &[u8],
    json_output: &[u8],
) -> AprProv002Verdict {
    let fields: &[&[u8]] = &[b"license", b"data_source", b"data_license"];
    for f in fields {
        // Build needle variants for "key-followed-by-colon" detection.
        let mut text_needle = f.to_vec();
        text_needle.push(b':');
        let mut json_needle = b"\"".to_vec();
        json_needle.extend_from_slice(f);
        json_needle.extend_from_slice(b"\":");
        if !contains_token_with_colon(text_output, &text_needle) {
            return AprProv002Verdict::Fail;
        }
        if !contains_subseq(json_output, &json_needle) {
            return AprProv002Verdict::Fail;
        }
    }
    AprProv002Verdict::Pass
}

/// Find `needle` (a key followed by `:`) where the byte
/// preceding the key is either start-of-string or a non-alphanumeric
/// boundary — disambiguates `license:` from `data_license:` for
/// the bare `license` field.
#[must_use]
fn contains_token_with_colon(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() {
        return false;
    }
    haystack
        .windows(needle.len())
        .enumerate()
        .any(|(idx, w)| {
            if w != needle {
                return false;
            }
            // Check left boundary: start-of-string OR previous byte is
            // not [a-zA-Z0-9_].
            if idx == 0 {
                return true;
            }
            let prev = haystack[idx - 1];
            !prev.is_ascii_alphanumeric() && prev != b'_'
        })
}

// ===========================================================================
// GATE-APR-PROV-003 — publish gate rejects missing/empty/unknown
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AprProv003Verdict {
    Pass,
    Fail,
}

/// Pure verdict function for `GATE-APR-PROV-003`.
///
/// Pass iff all three field values are present (non-empty), AND
/// none match the forbidden-placeholder list.
#[must_use]
pub fn verdict_from_publish_gate_field_validity(
    license: &[u8],
    data_source: &[u8],
    data_license: &[u8],
) -> AprProv003Verdict {
    for field in &[license, data_source, data_license] {
        if field.is_empty() {
            return AprProv003Verdict::Fail;
        }
        for forbidden in AC_APR_PROV_FORBIDDEN_PLACEHOLDERS {
            if *field == *forbidden {
                return AprProv003Verdict::Fail;
            }
        }
    }
    AprProv003Verdict::Pass
}

// ===========================================================================
// GATE-APR-PROV-004 — AC-SHIP1-009 dual discharge (same fields)
// ===========================================================================

/// `verdict_from_publish_gate_field_validity` is reused for
/// GATE-APR-PROV-004 directly. The contract pins this as the
/// SAME decision rule discharging both AC-SHIP1-009 and
/// AC-SHIP2-012; no separate verdict function is needed. This
/// dual-discharge is itself the algorithm-level claim.
///
/// To verify the dual discharge, callers can simply invoke
/// `verdict_from_publish_gate_field_validity` against the same
/// metadata for either AC. This module documents that the
/// contract intends this reuse.

#[cfg(test)]
mod tests {
    use super::*;

    // GATE-APR-PROV-001 ---------------------------------------------------------
    #[test]
    fn p001_pass_all_round_trip() {
        assert_eq!(verdict_from_serde_round_trip(true, true, true), AprProv001Verdict::Pass);
    }

    #[test]
    fn p001_fail_license_drift() {
        assert_eq!(verdict_from_serde_round_trip(false, true, true), AprProv001Verdict::Fail);
    }

    #[test]
    fn p001_fail_data_source_drift() {
        assert_eq!(verdict_from_serde_round_trip(true, false, true), AprProv001Verdict::Fail);
    }

    #[test]
    fn p001_fail_data_license_drift() {
        assert_eq!(verdict_from_serde_round_trip(true, true, false), AprProv001Verdict::Fail);
    }

    #[test]
    fn p001_fail_all_drift() {
        assert_eq!(verdict_from_serde_round_trip(false, false, false), AprProv001Verdict::Fail);
    }

    // GATE-APR-PROV-002 ---------------------------------------------------------
    #[test]
    fn p002_pass_both_surfaces_have_all_fields() {
        let text = b"license: Apache-2.0\ndata_source: codeparrot\ndata_license: permissive";
        let json = b"{\"license\":\"Apache-2.0\",\"data_source\":\"codeparrot\",\"data_license\":\"permissive\"}";
        assert_eq!(verdict_from_inspect_surfaces(text, json), AprProv002Verdict::Pass);
    }

    #[test]
    fn p002_fail_text_missing_license() {
        let text = b"data_source: x\ndata_license: y";
        let json = b"{\"license\":\"x\",\"data_source\":\"y\",\"data_license\":\"z\"}";
        assert_eq!(verdict_from_inspect_surfaces(text, json), AprProv002Verdict::Fail);
    }

    #[test]
    fn p002_fail_json_missing_data_source() {
        let text = b"license: x\ndata_source: y\ndata_license: z";
        let json = b"{\"license\":\"x\",\"data_license\":\"z\"}";
        assert_eq!(verdict_from_inspect_surfaces(text, json), AprProv002Verdict::Fail);
    }

    // GATE-APR-PROV-003 ---------------------------------------------------------
    #[test]
    fn p003_pass_all_real_values() {
        assert_eq!(
            verdict_from_publish_gate_field_validity(b"Apache-2.0", b"codeparrot", b"permissive"),
            AprProv003Verdict::Pass
        );
    }

    #[test]
    fn p003_fail_empty_license() {
        assert_eq!(
            verdict_from_publish_gate_field_validity(b"", b"x", b"y"),
            AprProv003Verdict::Fail
        );
    }

    #[test]
    fn p003_fail_unknown_license() {
        assert_eq!(
            verdict_from_publish_gate_field_validity(b"unknown", b"x", b"y"),
            AprProv003Verdict::Fail
        );
    }

    #[test]
    fn p003_fail_uppercase_unknown() {
        assert_eq!(
            verdict_from_publish_gate_field_validity(b"x", b"UNKNOWN", b"y"),
            AprProv003Verdict::Fail
        );
    }

    #[test]
    fn p003_fail_missing_placeholder() {
        assert_eq!(
            verdict_from_publish_gate_field_validity(b"x", b"y", b"(missing)"),
            AprProv003Verdict::Fail
        );
    }

    #[test]
    fn p003_fail_todo_placeholder() {
        assert_eq!(
            verdict_from_publish_gate_field_validity(b"TODO", b"x", b"y"),
            AprProv003Verdict::Fail
        );
    }

    // GATE-APR-PROV-004 — dual-discharge of AC-SHIP1-009 + AC-SHIP2-012 ---------
    #[test]
    fn p004_dual_discharge_uses_same_decision_rule() {
        // Per contract: GATE-APR-PROV-004 is discharged by the SAME
        // AprV2Metadata + serde-JSON decision rule that discharges
        // AC-SHIP2-012. Demonstrate by invoking the same verdict
        // function with both AC's metadata.
        let model_1_teacher = (b"Apache-2.0".as_slice(), b"codeparrot".as_slice(), b"permissive".as_slice());
        let model_2_pretrain = (b"MIT".as_slice(), b"thestack-python".as_slice(), b"permissive".as_slice());

        let v_ship1 = verdict_from_publish_gate_field_validity(
            model_1_teacher.0, model_1_teacher.1, model_1_teacher.2,
        );
        let v_ship2 = verdict_from_publish_gate_field_validity(
            model_2_pretrain.0, model_2_pretrain.1, model_2_pretrain.2,
        );

        assert_eq!(v_ship1, AprProv003Verdict::Pass);
        assert_eq!(v_ship2, AprProv003Verdict::Pass);
        // Same verdict function → contract claim "discharged by SAME rule".
    }

    // Shared primitive ----------------------------------------------------------
    #[test]
    fn placeholder_provenance_pin_count() {
        assert_eq!(AC_APR_PROV_FORBIDDEN_PLACEHOLDERS.len(), 7);
    }
}

// ===========================================================================
// Shared primitive
// ===========================================================================

#[must_use]
fn contains_subseq(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}
