//! `apr validate` exit-code falsifiers for a truncated `.apr` (issue #2612).
//!
//! Measured on the **published** `apr 0.63.0` from crates.io, on
//! `head -c 50000000` of a 1,115,528,900-byte `.apr` — 95.5% of the file gone:
//!
//! ```text
//! apr validate  -> rc=0   checks 5-17 all "○ SKIP  Not implemented"
//! apr qa        -> rc=5   (the only command that caught it)
//! ```
//!
//! `apr validate --quality` is the command CLAUDE.md tells users to run for
//! integrity. It graded the file F (3/100) and exited **0**, and every machine
//! consumer reads the exit code.
//!
//! The same truncation on a GGUF exits 5 (`Truncated GGUF: file is 50000000 bytes
//! but tensor data starts at 1117314624`), and that asymmetry is the whole ticket:
//! the capability existed for GGUF and was absent for APR.
//!
//! `--skip-contract` is the sharpest case. It short-circuits the data-quality
//! content gate entirely (`gate_apr_content` returns `Ok(())` on its first line),
//! so on the pre-#2612 tree it exits 0 on a 95%-truncated file no matter what the
//! tensors decode to. Structural truncation is arithmetic, not a contract opinion,
//! and the GGUF path does not let `--skip-contract` past it either.

use super::*;
use aprender::format::v2::{AprV2Metadata, AprV2Writer};
use std::io::Write;
use tempfile::NamedTempFile;

/// A structurally complete `.apr`, written by the real writer.
fn known_good_apr_bytes() -> Vec<u8> {
    let mut writer = AprV2Writer::new(AprV2Metadata::new("truncation-fixture"));
    // Varied values so the content gates have something real to look at; the
    // structural claim under test does not depend on them.
    let weight: Vec<f32> = (0..1024)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.031_25)
        .collect();
    let bias: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01 - 0.15).collect();
    writer.add_f32_tensor("layer.0.weight", vec![32, 32], &weight);
    writer.add_f32_tensor("layer.0.bias", vec![32], &bias);
    writer.write().expect("fixture must serialize")
}

/// Write `bytes` to a `.apr` temp file. The handle must outlive the call.
fn apr_file(bytes: &[u8]) -> NamedTempFile {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(bytes).expect("write fixture");
    file.flush().expect("flush fixture");
    file
}

fn truncated_apr_file() -> NamedTempFile {
    let bytes = known_good_apr_bytes();
    apr_file(&bytes[..bytes.len() / 2])
}

fn err_text(result: &Result<(), CliError>) -> String {
    match result {
        Ok(()) => String::new(),
        Err(e) => e.to_string(),
    }
}

#[test]
fn truncated_apr_exits_non_zero() {
    let file = truncated_apr_file();
    let result = run(file.path(), false, false, None, false, false);
    assert!(
        result.is_err(),
        "a truncated .apr must not validate clean — exit 0 here is #2612"
    );
    let msg = err_text(&result);
    assert!(
        msg.contains("Truncated APR"),
        "the failure must name truncation, not a downstream symptom: {msg}"
    );
}

#[test]
fn truncated_apr_exits_non_zero_with_quality_flag() {
    // `apr validate --quality` is the exact command CLAUDE.md documents.
    let file = truncated_apr_file();
    let result = run(file.path(), true, false, None, false, false);
    assert!(
        result.is_err(),
        "`apr validate --quality` on a truncated .apr must exit non-zero"
    );
    assert!(err_text(&result).contains("Truncated APR"));
}

#[test]
fn truncated_apr_exits_non_zero_with_skip_contract() {
    // The case no content gate can reach: --skip-contract returns Ok(()) from
    // `gate_apr_content` before looking at a single tensor.
    let file = truncated_apr_file();
    let result = run(file.path(), false, false, None, false, true);
    assert!(
        result.is_err(),
        "--skip-contract waives the data-quality contract, not the file's own arithmetic"
    );
    assert!(err_text(&result).contains("Truncated APR"));
}

#[test]
fn truncated_apr_exits_non_zero_as_json() {
    let file = truncated_apr_file();
    let result = run(file.path(), false, false, None, true, false);
    assert!(
        result.is_err(),
        "the --json consumer reads the exit code too"
    );
    assert!(err_text(&result).contains("Truncated APR"));
}

#[test]
fn truncated_apr_exits_non_zero_with_skip_contract_and_json() {
    let file = truncated_apr_file();
    let result = run(file.path(), true, false, None, true, true);
    assert!(result.is_err(), "no flag combination waives truncation");
    assert!(err_text(&result).contains("Truncated APR"));
}

#[test]
fn intact_apr_is_never_reported_as_truncated() {
    // No false positives. The intact fixture may still be rejected by the
    // data-quality content gates (that is a different gate with its own tests),
    // but it must never be called truncated.
    let bytes = known_good_apr_bytes();
    let file = apr_file(&bytes);
    let result = run(file.path(), true, false, None, false, false);
    assert!(
        !err_text(&result).contains("Truncated APR"),
        "intact .apr must not be reported as truncated: {}",
        err_text(&result)
    );
}

/// Parity statement: the GGUF path already refuses a truncated file, and after
/// #2612 the APR path refuses one too. Both are `ValidationFailed`, not a panic
/// and not a silent zero.
#[test]
fn apr_truncation_is_refused_the_way_gguf_truncation_is() {
    let file = truncated_apr_file();
    let result = run(file.path(), false, false, None, false, false);
    assert!(
        matches!(result, Err(CliError::ValidationFailed(_))),
        "expected ValidationFailed (the GGUF truncation verdict), got {result:?}"
    );
}
