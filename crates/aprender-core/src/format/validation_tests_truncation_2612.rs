//! Structural falsifiers for check 5, "Data section within file" (issue #2612).
//!
//! Measured on published `apr 0.63.0`, on `head -c 50000000` of a 1,115,528,900-byte
//! `.apr` (95.5% missing):
//!
//! ```text
//! apr validate  -> rc=0   checks 5-17 all "○ SKIP  Not implemented"
//! apr qa        -> rc=5   (the only command that caught it)
//! ```
//!
//! The same truncation on a GGUF exits 5 with `Truncated GGUF: file is 50000000
//! bytes but tensor data starts at 1117314624`. The capability existed for GGUF and
//! was ABSENT for APR — a missing check, not an unwired one.
//!
//! Before this fix `check_data_within_file` did not exist and check 5 was
//! `Skip("Not implemented")`, so `truncated_apr_fails_check_5` fails on the old
//! tree at the `is_fail()` assertion.

use super::*;
use crate::format::v2::{AprV2Metadata, AprV2Writer};

/// A structurally complete `.apr` with real tensor payload.
fn known_good_apr_bytes() -> Vec<u8> {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("truncation-fixture"));
    writer.add_f32_tensor("layer.0.weight", vec![16, 16], &[0.125_f32; 256]);
    writer.add_f32_tensor("layer.0.bias", vec![16], &[0.5_f32; 16]);
    writer.write().expect("fixture must serialize")
}

fn check_5(data: &[u8]) -> ValidationCheck {
    let mut validator = AprValidator::new();
    validator.validate_bytes(data);
    validator
        .report()
        .checks
        .iter()
        .find(|c| c.id == 5)
        .cloned()
        .expect("check 5 must be present in the report")
}

#[test]
fn intact_apr_passes_check_5() {
    let bytes = known_good_apr_bytes();
    let check = check_5(&bytes);
    assert!(
        check.status.is_pass(),
        "a complete .apr must pass the data-extent check, got {:?}",
        check.status
    );
}

#[test]
fn truncated_apr_fails_check_5() {
    let bytes = known_good_apr_bytes();
    let full_len = bytes.len();
    // Cut inside the data section, exactly the damage class measured on the
    // 1.1 GB file: header + metadata + index all survive.
    let truncated = &bytes[..full_len / 2];

    let check = check_5(truncated);
    assert!(
        check.status.is_fail(),
        "a .apr truncated from {full_len} to {} bytes must FAIL the data-extent check, got {:?}",
        truncated.len(),
        check.status
    );
    match &check.status {
        CheckStatus::Fail(reason) => {
            assert!(
                reason.contains("Truncated APR"),
                "the reason must name the defect, got: {reason}"
            );
            assert!(
                reason.contains(&truncated.len().to_string()),
                "the reason must cite the actual file length, got: {reason}"
            );
        }
        other => panic!("expected Fail, got {other:?}"),
    }
}

#[test]
fn truncated_apr_report_is_not_valid() {
    // The exit code the CLI derives comes from `is_valid()`; on 0.63.0 this was
    // `true` for a 95.5%-truncated file.
    let bytes = known_good_apr_bytes();
    let truncated = &bytes[..bytes.len() / 2];

    let mut validator = AprValidator::new();
    let report = validator.validate_bytes(truncated);
    assert!(
        !report.is_valid(),
        "a truncated .apr must not report VALID (failed checks: {})",
        report.failed_checks().len()
    );

    let mut ok_validator = AprValidator::new();
    let ok_report = ok_validator.validate_bytes(&bytes);
    assert!(
        ok_report.is_valid(),
        "no false positive: the intact fixture must still report VALID"
    );
}

/// One byte short is still short. Boundary, not a magnitude test.
#[test]
fn one_byte_short_still_fails() {
    let bytes = known_good_apr_bytes();
    let required = crate::format::v2::required_file_len(&bytes).expect("extent");
    // Trailing padding + the 4-byte footer sit past `required`, so trim to
    // exactly one byte below the declared extent.
    let cut = usize::try_from(required).expect("fixture fits usize") - 1;
    let check = check_5(&bytes[..cut]);
    assert!(
        check.status.is_fail(),
        "file one byte shorter than its declared extent must fail, got {:?}",
        check.status
    );
}

/// A non-APR buffer must be skipped, not failed: the invariant has no operands.
/// This is what keeps the check from firing on the GGUF path, which enforces its
/// own truncation gate.
#[test]
fn gguf_buffer_skips_check_5() {
    let mut gguf = vec![0u8; 128];
    gguf[0..4].copy_from_slice(b"GGUF");
    gguf[4..8].copy_from_slice(&3u32.to_le_bytes());
    let check = check_5(&gguf);
    assert!(
        matches!(check.status, CheckStatus::Skip(_)),
        "GGUF must skip the APR-only extent check, got {:?}",
        check.status
    );
}

/// Checks 6-25 are still declared stubs. This test states the scope of the fix
/// so a future reader does not mistake "check 5 implemented" for "Section A
/// implemented" — and it fails the moment someone implements one without
/// updating the ledger.
#[test]
fn scope_of_2612_is_check_5_only() {
    let bytes = known_good_apr_bytes();
    let mut validator = AprValidator::new();
    let report = validator.validate_bytes(&bytes);

    let stubbed: Vec<u8> = report
        .checks
        .iter()
        .filter(|c| (6..=25).contains(&c.id) && matches!(c.status, CheckStatus::Skip(_)))
        .map(|c| c.id)
        .collect();
    assert_eq!(
        stubbed.len(),
        20,
        "#2612 implemented check 5 only; checks 6-25 remain stubs (found {stubbed:?})"
    );
}
