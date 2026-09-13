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

// ============================================================================
// #1866 FALSIFIER: `apr validate --quality` graded a healthy model F at exit 0
// while the same run said passed:true, failed:0 and printed "✓ VALID".
//
// Reproduced on /home/noah/models/qwen2.5-coder-0.5b-instruct.apr at
// 6adeb6351 before the fix:
//
//     TOTAL: 3/100  Grade: F                                    (exit 0)
//     { "total_score": 3, "grade": "F", "failed": 0, "passed": true }
//     --min-score 50 -> error: Validation failed: Score 3/100 below minimum 50
//
// Contract: apr-validate-quality-threshold-v1
// ============================================================================

/// The report `apr validate` actually builds for that model: checks 1/2/3
/// PASS, check 11 WARNs on an unknown flag bit, 22 checks are
/// `Skip("Not implemented")` stubs.
fn healthy_apr_report() -> ValidationReport {
    use aprender::format::validation::ValidationCheck;

    fn push(report: &mut ValidationReport, id: u8, status: CheckStatus) {
        let points = u8::from(status.is_pass());
        report.add_check(ValidationCheck {
            id,
            name: "check",
            category: Category::Structure,
            status,
            points,
        });
    }

    let mut report = ValidationReport::new();
    push(&mut report, 1, CheckStatus::Pass);
    push(&mut report, 2, CheckStatus::Pass);
    push(&mut report, 3, CheckStatus::Pass);
    push(
        &mut report,
        11,
        CheckStatus::Warn("Unknown flag bits: 0x00000100".to_string()),
    );
    push(
        &mut report,
        4,
        CheckStatus::Skip("Footer not implemented".to_string()),
    );
    for id in 5..=25 {
        push(
            &mut report,
            id,
            CheckStatus::Skip("Not implemented".to_string()),
        );
    }
    report
}

/// A clean content-gate result, as `RosettaStone::validate` returns for that
/// model.
fn clean_content() -> Result<RosettaValidationReport, AprenderError> {
    Ok(RosettaValidationReport {
        format: FormatType::Apr,
        file_path: "model.apr".to_string(),
        is_valid: true,
        tensor_count: 339,
        failed_tensor_count: 0,
        total_nan_count: 0,
        total_inf_count: 0,
        all_zero_tensors: Vec::new(),
        tensors: Vec::new(),
        duration_ms: 1,
    })
}

/// FALSIFY-VALIDATE-QUALITY-006: the `--json` document cannot say
/// `grade: F`, `verdict: VALID` and `passed: true` at the same time.
#[test]
fn json_grade_passed_and_verdict_agree_on_a_healthy_model() {
    let report = healthy_apr_report();
    let (doc, passed) = apr_validation_json(
        Path::new("/models/qwen2.5-coder-0.5b-instruct.apr"),
        &report,
        &clean_content(),
        None,
    );

    assert!(passed, "no check failed and the content gate is clean");
    assert_eq!(doc["passed"], serde_json::json!(true));
    assert_eq!(doc["failed"], serde_json::json!(0));
    assert_eq!(doc["verdict"], serde_json::json!("VALID"));
    assert_ne!(
        doc["grade"],
        serde_json::json!("F"),
        "a document reporting passed:true and failed:0 must not grade F: {doc}"
    );
    assert_eq!(doc["grade"], serde_json::json!("C+"));
    // The score is measured against the 4 checks that ran, not the 26 declared.
    assert_eq!(doc["total_score"], serde_json::json!(75));
    assert_eq!(doc["checks_ran"], serde_json::json!(4));
    assert_eq!(doc["checks_not_implemented"], serde_json::json!(22));
}

/// The same document must flip every field together when a check really does
/// fail — otherwise the agreement above is just a constant.
#[test]
fn json_grade_passed_and_verdict_agree_on_a_broken_model() {
    use aprender::format::validation::ValidationCheck;

    let mut report = healthy_apr_report();
    report.add_check(ValidationCheck {
        id: 26,
        name: "Magic bytes valid",
        category: Category::Structure,
        status: CheckStatus::Fail("Invalid magic".to_string()),
        points: 0,
    });
    let (doc, passed) = apr_validation_json(
        Path::new("/models/broken.apr"),
        &report,
        &clean_content(),
        None,
    );

    assert!(!passed);
    assert_eq!(doc["passed"], serde_json::json!(false));
    assert_eq!(doc["verdict"], serde_json::json!("INVALID"));
    assert_eq!(doc["grade"], serde_json::json!("F"));
    assert_eq!(doc["failed"], serde_json::json!(1));
}

/// FALSIFY-VALIDATE-QUALITY-007: the `--quality` table must not print a
/// score against a denominator nothing was measured against, and must not
/// draw `0/25` for categories in which no check is even declared.
/// Colour is a terminal concern; the assertions below are about text.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[test]
fn quality_table_scores_only_what_ran() {
    let body = strip_ansi(&quality_assessment_body(&healthy_apr_report()));

    assert!(
        !body.contains("/100"),
        "the TOTAL line still bands against a 100 nothing was measured against: {body}"
    );
    assert!(
        !body.contains("0/25"),
        "categories that declare no checks must not be drawn as scoring zero: {body}"
    );
    assert!(
        body.contains("not implemented"),
        "the empty categories must be named as unimplemented: {body}"
    );
    assert!(body.contains("Grade: C+"), "{body}");
    assert!(body.contains("3/4 checks that ran"), "{body}");
}

/// FALSIFY-VALIDATE-QUALITY-008: `--min-score` thresholds the checks that ran.
///
/// `--min-score 50` used to reject this model with
/// `Score 3/100 below minimum 50` — while the identical run without the flag
/// printed `✓ VALID` and exited 0.
#[test]
fn min_score_thresholds_the_checks_that_ran() {
    let report = healthy_apr_report();

    check_min_score(&report, Some(50)).expect("3 of 4 checks that ran passed — that is 75%");
    check_min_score(&report, Some(75)).expect("75% must clear a 75 threshold");

    let err = check_min_score(&report, Some(80))
        .expect_err("75% must NOT clear an 80 threshold — the gate must still bite");
    let msg = format!("{err}");
    assert!(
        msg.contains("75/100"),
        "the refusal must quote the measured score: {msg}"
    );
}

/// A threshold against a score nothing produced is a gate that cannot fail;
/// refuse it rather than satisfy it silently.
#[test]
fn min_score_is_refused_when_no_check_ran() {
    use aprender::format::validation::ValidationCheck;

    let mut report = ValidationReport::new();
    for id in 1..=25u8 {
        report.add_check(ValidationCheck {
            id,
            name: "stub",
            category: Category::Structure,
            status: CheckStatus::Skip("Not implemented".to_string()),
            points: 0,
        });
    }
    let err = check_min_score(&report, Some(0))
        .expect_err("no check ran, so not even --min-score 0 was demonstrated");
    assert!(
        format!("{err}").contains("none of the 25 declared QA checks ran"),
        "{err}"
    );
}

// Issue #2612: truncated `.apr` must not validate clean.
#[path = "validate_tests_truncation_2612.rs"]
mod validate_tests_truncation_2612;
