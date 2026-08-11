use super::*;
use std::collections::HashMap;
use std::io::Write;
use tempfile::{tempdir, NamedTempFile};

// ========================================================================
// Path Validation Tests
// ========================================================================

#[test]
fn test_validate_path_not_found() {
    let result = validate_path(Path::new("/nonexistent/model.apr"));
    assert!(result.is_err());
    match result {
        Err(CliError::FileNotFound(_)) => {}
        _ => panic!("Expected FileNotFound error"),
    }
}

#[test]
fn test_validate_path_is_directory() {
    let dir = tempdir().expect("create temp dir");
    let result = validate_path(dir.path());
    assert!(result.is_err());
    match result {
        Err(CliError::NotAFile(_)) => {}
        _ => panic!("Expected NotAFile error"),
    }
}

#[test]
fn test_validate_path_valid_file() {
    let file = NamedTempFile::new().expect("create temp file");
    let result = validate_path(file.path());
    assert!(result.is_ok());
}

// ========================================================================
// Run Command Tests
// ========================================================================

#[test]
fn test_run_file_not_found() {
    let result = run(
        Path::new("/nonexistent/model.apr"),
        false,
        false,
        None,
        false,
        false,
    );
    assert!(result.is_err());
    match result {
        Err(CliError::FileNotFound(_)) => {}
        _ => panic!("Expected FileNotFound error"),
    }
}

#[test]
fn test_run_is_directory() {
    let dir = tempdir().expect("create temp dir");
    let result = run(dir.path(), false, false, None, false, false);
    assert!(result.is_err());
    match result {
        Err(CliError::NotAFile(_)) => {}
        _ => panic!("Expected NotAFile error"),
    }
}

#[test]
fn test_run_invalid_file() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"not a valid APR file").expect("write");

    let result = run(file.path(), false, false, None, false, false);
    // Should fail validation because file is not valid APR
    assert!(result.is_err());
}

#[test]
fn test_run_with_quality_flag() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"invalid data").expect("write");

    let result = run(file.path(), true, false, None, false, false);
    // Should fail but quality flag is handled
    assert!(result.is_err());
}

#[test]
fn test_run_with_min_score() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"invalid data").expect("write");

    let result = run(file.path(), false, false, Some(100), false, false);
    // Should fail before min_score check because file is invalid
    assert!(result.is_err());
}

#[test]
fn test_run_with_strict_flag() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"test data").expect("write");

    let result = run(file.path(), false, true, None, false, false);
    // Should fail with strict mode
    assert!(result.is_err());
}

#[test]
fn test_run_with_all_flags() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(b"test data").expect("write");

    let result = run(file.path(), true, true, Some(50), false, false);
    // Should fail with all flags enabled
    assert!(result.is_err());
}

#[test]
fn test_run_empty_file() {
    let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    // Empty file - no write

    let result = run(file.path(), false, false, None, false, false);
    // Empty file should fail validation
    assert!(result.is_err());
}

// ========================================================================
// Category Score Tests (using mocked reports via AprValidator)
// ========================================================================

#[test]
fn test_quality_assessment_display() {
    let mut category_scores = HashMap::new();
    category_scores.insert(Category::Structure, 25);
    category_scores.insert(Category::Physics, 20);
    category_scores.insert(Category::Tooling, 15);
    category_scores.insert(Category::Conversion, 10);

    let report = ValidationReport {
        checks: Vec::new(),
        total_score: 70,
        category_scores,
    };

    // Should not panic
    print_quality_assessment(&report);
}

#[test]
fn test_quality_assessment_missing_categories() {
    let report = ValidationReport {
        checks: Vec::new(),
        total_score: 0,
        category_scores: HashMap::new(),
    };

    // Should handle missing categories gracefully (default to 0)
    print_quality_assessment(&report);
}

#[test]
fn test_quality_assessment_all_score_ranges() {
    // High scores
    let mut high_scores = HashMap::new();
    high_scores.insert(Category::Structure, 25);
    high_scores.insert(Category::Physics, 25);
    high_scores.insert(Category::Tooling, 25);
    high_scores.insert(Category::Conversion, 25);

    let high_report = ValidationReport {
        checks: Vec::new(),
        total_score: 100,
        category_scores: high_scores,
    };

    // Low scores
    let mut low_scores = HashMap::new();
    low_scores.insert(Category::Structure, 5);

    let low_report = ValidationReport {
        checks: Vec::new(),
        total_score: 5,
        category_scores: low_scores,
    };

    // All should display without panic
    print_quality_assessment(&high_report);
    print_quality_assessment(&low_report);
}

// ========================================================================
// Print Summary Tests
// ========================================================================

#[test]
fn test_print_summary_valid_report() {
    let report = ValidationReport {
        checks: Vec::new(), // No failed checks
        total_score: 100,
        category_scores: HashMap::new(),
    };

    let result = print_summary(&report);
    assert!(result.is_ok());
}

#[test]
fn test_print_quality_assessment_empty() {
    let report = ValidationReport {
        checks: Vec::new(),
        total_score: 0,
        category_scores: HashMap::new(),
    };

    // Should not panic even with empty report
    print_quality_assessment(&report);
}

// ========================================================================
// Multi-Format Dispatch Tests (GGUF, SafeTensors)
// ========================================================================

#[test]
fn test_run_gguf_format_dispatch() {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

    // Create valid GGUF file with non-zero tensor data
    let floats: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![4, 4],
        dtype: GgmlType::F32,
        data,
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    file.write_all(&gguf_bytes).expect("write GGUF");

    // Should dispatch to GGUF validation path (RosettaStone::validate)
    let result = run(file.path(), false, false, None, false, false);
    // GGUF validation should succeed (physics constraints pass)
    assert!(result.is_ok(), "GGUF format dispatch should work");
}

#[test]
fn test_run_safetensors_format_dispatch() {
    // Create valid SafeTensors file manually
    let header_json = serde_json::json!({
        "test.weight": {
            "dtype": "F32",
            "shape": [2, 2],
            "data_offsets": [0, 16]
        }
    });
    let header_bytes = serde_json::to_vec(&header_json).expect("serialize header");
    let header_len = header_bytes.len() as u64;

    let mut st_bytes = Vec::new();
    st_bytes.extend_from_slice(&header_len.to_le_bytes());
    st_bytes.extend_from_slice(&header_bytes);
    // Add valid tensor data (4 floats = 16 bytes)
    let floats: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    for f in floats {
        st_bytes.extend_from_slice(&f.to_le_bytes());
    }

    let mut file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    file.write_all(&st_bytes).expect("write SafeTensors");

    // Should dispatch to SafeTensors validation path (RosettaStone::validate)
    let result = run(file.path(), false, false, None, false, false);
    // SafeTensors validation should succeed
    assert!(result.is_ok(), "SafeTensors format dispatch should work");
}

#[test]
fn test_run_gguf_format_detection_by_magic() {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

    // Create GGUF with .bin extension (magic detection, not extension)
    // Use valid non-zero tensor data
    let floats: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let tensor_data: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();

    let tensor = GgufTensor {
        name: "test.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: tensor_data,
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let mut file = NamedTempFile::with_suffix(".bin").expect("create temp file");
    file.write_all(&gguf_bytes).expect("write GGUF");

    // Should detect GGUF by magic bytes, not extension
    let result = run(file.path(), false, false, None, false, false);
    assert!(result.is_ok(), "Should detect GGUF by magic bytes");
}

#[test]
fn test_run_gguf_with_physics_violations() {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

    // Create GGUF with NaN values (physics violation)
    let nan_f32 = f32::NAN.to_le_bytes();
    let mut tensor_data = Vec::new();
    for _ in 0..4 {
        tensor_data.extend_from_slice(&nan_f32);
    }

    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: tensor_data,
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    file.write_all(&gguf_bytes).expect("write GGUF");

    // Should fail due to NaN physics violation
    let result = run(file.path(), false, false, None, false, false);
    assert!(result.is_err(), "Should fail with NaN tensors");
}

// ========================================================================
// PMAT-926: APR fail-closed content gates + --strict wiring
//
// Contract: apr-validate-fail-closed-v1.yaml
//   - F-VALIDATE-APR-DISPATCH-001 (content-broken .apr REJECTED)
//   - F-VALIDATE-STRICT-001        (--strict escalates warn → non-zero exit)
//
// Before PMAT-926 the `.apr` path routed to the stubbed `AprValidator`
// (magic/header/version/flags only) and `--strict` printed
// "not yet implemented, flag ignored" — so a semantically-broken `.apr`
// (all-zero lm_head, NaN/Inf, constant weight) was reported VALID and ran
// silently, exactly the class of garbage llama.cpp / Ollama load and run.
// These falsifiers go RED on the stub and GREEN on the fix; a valid model
// still passes (no false positives).
// ========================================================================

/// Build a syntactically-valid `.apr` file from `(name, shape, data)` tensors.
/// Header checksum + offsets are computed by `AprV2Writer`, so the file is a
/// genuinely-parseable model — only the *content* of the tensors varies.
fn write_apr_fixture(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> NamedTempFile {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer};

    let metadata = AprV2Metadata::new("test");
    let mut writer = AprV2Writer::new(metadata);
    for (name, shape, data) in tensors {
        writer.add_tensor_f32_owned(*name, shape.clone(), data.clone());
    }
    let mut apr_bytes = Vec::new();
    writer.write_to(&mut apr_bytes).expect("write APR fixture");

    let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    file.write_all(&apr_bytes).expect("write apr bytes");
    file
}

/// A healthy weight column with variation (passes all data-quality gates).
fn healthy_weights(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.013 - 0.5) * 0.2).collect()
}

/// FALSIFIER (F-VALIDATE-APR-DISPATCH-001): a valid-header `.apr` whose
/// `lm_head.weight` is entirely zero must be REJECTED. RED on the stub
/// (`AprValidator` never inspects tensor content → returns VALID), GREEN
/// after re-routing through the Rosetta content gates.
#[test]
fn pmat926_falsifier_all_zero_lm_head_apr_rejected() {
    // 8x4 lm_head, every weight zero (dead model). 4x4 healthy embed so the
    // file is otherwise well-formed.
    let file = write_apr_fixture(&[
        ("lm_head.weight", vec![8, 4], vec![0.0; 32]),
        ("model.embed_tokens.weight", vec![4, 4], healthy_weights(16)),
    ]);

    let result = run(file.path(), false, false, None, false, false);
    assert!(
        result.is_err(),
        "all-zero lm_head .apr must be REJECTED (F-VALIDATE-APR-DISPATCH-001), got Ok"
    );
}

/// FALSIFIER (F-VALIDATE-APR-DISPATCH-001): a `.apr` with NaN weights is
/// REJECTED. The incumbents load and run NaN weights silently.
#[test]
fn pmat926_falsifier_nan_tensor_apr_rejected() {
    let mut nan_block = healthy_weights(32);
    nan_block[5] = f32::NAN;
    nan_block[17] = f32::NAN;
    let file = write_apr_fixture(&[
        ("lm_head.weight", vec![8, 4], healthy_weights(32)),
        ("model.layers.0.mlp.down_proj.weight", vec![8, 4], nan_block),
    ]);

    let result = run(file.path(), false, false, None, false, false);
    assert!(
        result.is_err(),
        "NaN .apr must be REJECTED (F-VALIDATE-APR-DISPATCH-001), got Ok"
    );
}

/// NO-FALSE-POSITIVE: a healthy `.apr` (varied, non-zero, finite weights)
/// must still validate clean. Guards the fail-closed gate against rejecting
/// good models.
#[test]
fn pmat926_valid_apr_still_passes() {
    let file = write_apr_fixture(&[
        ("lm_head.weight", vec![8, 4], healthy_weights(32)),
        ("model.embed_tokens.weight", vec![8, 4], healthy_weights(32)),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![8, 4],
            healthy_weights(32),
        ),
    ]);

    let result = run(file.path(), false, false, None, false, false);
    assert!(
        result.is_ok(),
        "healthy .apr must still PASS (no false positives), got {result:?}"
    );
}

/// FALSIFIER (F-VALIDATE-STRICT-001): `--strict` escalates a warn-level
/// finding to a hard non-zero exit on the `.apr` path. Before PMAT-926
/// `--strict` was a documented no-op for APR ("flag ignored").
#[test]
fn pmat926_falsifier_strict_apr_all_zero_nonzero_exit() {
    // A standalone all-zero tensor (not an output-projection role) lands in
    // `all_zero_tensors` — a strict-blocking finding.
    let file = write_apr_fixture(&[
        ("lm_head.weight", vec![8, 4], healthy_weights(32)),
        (
            "model.layers.0.self_attn.q_proj.bias",
            vec![8],
            vec![0.0; 8],
        ),
    ]);

    // strict = true must fail closed.
    let strict_result = run(file.path(), false, true, None, false, false);
    assert!(
        strict_result.is_err(),
        "--strict on an all-zero-tensor .apr must exit non-zero (F-VALIDATE-STRICT-001), got Ok"
    );
}

/// `--skip-contract` bypasses the fail-closed content gate (parity with the
/// GGUF/SafeTensors path) — a broken `.apr` is accepted when the operator
/// explicitly opts out.
#[test]
fn pmat926_skip_contract_bypasses_content_gate() {
    let file = write_apr_fixture(&[
        ("lm_head.weight", vec![8, 4], vec![0.0; 32]),
        ("model.embed_tokens.weight", vec![4, 4], healthy_weights(16)),
    ]);

    let result = run(
        file.path(),
        false,
        false,
        None,
        false,
        /* skip_contract */ true,
    );
    assert!(
        result.is_ok(),
        "--skip-contract must bypass the content gate, got {result:?}"
    );
}

/// JSON path also fails closed on a content-broken `.apr` (the report is
/// still printed, but the exit code is non-zero).
#[test]
fn pmat926_json_apr_all_zero_lm_head_rejected() {
    let file = write_apr_fixture(&[
        ("lm_head.weight", vec![8, 4], vec![0.0; 32]),
        ("model.embed_tokens.weight", vec![4, 4], healthy_weights(16)),
    ]);

    let result = run(file.path(), false, false, None, /* json */ true, false);
    assert!(
        result.is_err(),
        "JSON .apr path must fail closed on content-broken model, got Ok"
    );
}

// ============================================================================
// #2394 findings 12 & 17: a score needs a denominator; a threshold needs a score
// ============================================================================

/// FALSIFIER (#2394 finding 17): `--min-score` on a format that computes no
/// score must be refused, not silently ignored.
///
/// `apr validate model.gguf --quality --min-score 100` exited 0. 100 is the
/// strictest threshold expressible, no score was printed anywhere in the
/// output, and nothing said the flag had been dropped — the dispatcher simply
/// never passed `min_score` to the Rosetta branch. A gate that cannot fail is
/// worse than no gate: it reports PASS to whoever wired it into CI.
#[test]
fn min_score_is_refused_on_formats_that_compute_no_score() {
    let dir = tempdir().expect("tempdir");
    let gguf = dir.path().join("tiny.gguf");
    std::fs::write(&gguf, b"GGUF\x03\x00\x00\x00rest-does-not-matter").expect("write");

    let err = super::run(
        &gguf,
        /* quality */ true,
        /* strict */ false,
        /* min_score */ Some(100),
        /* json */ false,
        /* skip_contract */ false,
    )
    .expect_err("--min-score 100 must not pass on a format with no score");
    let msg = err.to_string();
    assert!(
        msg.contains("--min-score"),
        "the refusal must name the flag it is refusing, got: {msg}"
    );
    assert!(
        msg.contains("no score"),
        "the refusal must say WHY (no score is computed), got: {msg}"
    );
}

/// ...and it must still discriminate: the `.apr` validator does produce a
/// score, so `--min-score` there is a real gate and must not be refused.
#[test]
fn min_score_is_accepted_on_the_format_that_does_compute_a_score() {
    let dir = tempdir().expect("tempdir");
    let apr = dir.path().join("tiny.apr");
    std::fs::write(&apr, b"APR\x00\x02\x00\x00\x00truncated").expect("write");

    // This file is broken, so validation fails — but it must fail on the FILE,
    // never on the flag.
    let msg = match super::run(&apr, false, false, Some(50), false, false) {
        Ok(()) => String::new(),
        Err(e) => e.to_string(),
    };
    assert!(
        !msg.contains("does not apply"),
        "--min-score must be honored for .apr, got refusal: {msg}"
    );
}

/// FALSIFIER (#2394 finding 12): the verdict line must not print a score
/// against a denominator that was never measured.
///
/// A healthy model printed `✓ VALID 3/100 points` — a green badge beside what
/// reads as 3%. 97 of the 100 checks are `Skip("Not implemented")` stubs that
/// never ran. Either reading of "3/100" is wrong, so the line must carry the
/// denominator that was actually measured and disclose the stubs.
#[test]
fn valid_verdict_reports_the_denominator_it_measured() {
    use aprender::format::validation::{Category, CheckStatus, ValidationCheck};

    let mut report = ValidationReport::new();
    for id in 1..=3u8 {
        report.add_check(ValidationCheck {
            id,
            name: "implemented check",
            category: Category::Structure,
            status: CheckStatus::Pass,
            points: 1,
        });
    }
    for id in 4..=100u8 {
        report.add_check(ValidationCheck {
            id,
            name: "stub",
            category: Category::Physics,
            status: CheckStatus::Skip("Not implemented".to_string()),
            points: 0,
        });
    }

    let line = summary_line(&report);
    assert!(
        !line.contains("/100"),
        "97 of those 100 checks never ran; the line must not score against them: {line}"
    );
    assert!(
        line.contains("3/3"),
        "the denominator must be the checks that ran: {line}"
    );
    assert!(
        line.contains("97"),
        "the line must disclose the 97 unimplemented checks: {line}"
    );
}

/// The same line must still read correctly when every check is implemented —
/// no phantom "not implemented" clause.
#[test]
fn valid_verdict_omits_the_stub_clause_when_nothing_was_skipped() {
    use aprender::format::validation::{Category, CheckStatus, ValidationCheck};

    let mut report = ValidationReport::new();
    report.add_check(ValidationCheck {
        id: 1,
        name: "implemented check",
        category: Category::Structure,
        status: CheckStatus::Pass,
        points: 1,
    });
    let line = summary_line(&report);
    assert!(line.contains("1/1"), "{line}");
    assert!(!line.contains("not implemented"), "{line}");
}
