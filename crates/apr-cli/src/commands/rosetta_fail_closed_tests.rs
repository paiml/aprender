
// ============================================================================
// Falsifiers for the `apr rosetta` gates that could not fail (issue #2382).
//
// Every test here asserts BEHAVIOUR of a verdict — the exit status, the verdict
// line, or the machine-readable document — for an input the command must reject.
// Each one turns RED against the 0.63.0 code it replaces.
// ============================================================================

// ── rosetta fingerprint: a diff that finds differences must fail ────────────

#[test]
fn falsifier_fingerprint_diff_all_tensors_missing_in_b_is_a_failure() {
    // The shipped repro: model A has 27 tensors, model B is a different model and
    // has none of them. 0.63.0 printed 27x "Missing in Model B" and then
    // "No statistical anomalies detected" / "passed": true, exit 0.
    let fps_a: Vec<TensorFingerprint> = (0..27)
        .map(|i| make_fingerprint(&format!("model.layers.{i}.weight"), 0.5, 1.0, 0, 0))
        .collect();
    let fps_b: Vec<TensorFingerprint> = vec![];

    let err = print_fingerprint_diff(&fps_a, &fps_b, false, false)
        .expect_err("27 of 27 tensors absent from B must not report success");
    let msg = err.to_string();
    assert!(
        msg.contains("27"),
        "failure must name how many tensors are missing, got: {msg}"
    );
    assert!(
        matches!(err, CliError::ValidationFailed(_)),
        "must be a validation failure so the process exits 5, got: {err:?}"
    );
}

#[test]
fn falsifier_fingerprint_diff_tensor_only_in_b_is_a_failure() {
    // The walker only iterates model A, so a tensor that exists solely in model B
    // was invisible: the extra 175 tensors of the shipped repro were never counted.
    let fps_a: Vec<TensorFingerprint> = vec![];
    let fps_b = vec![make_fingerprint("only_in_b.weight", 0.5, 1.0, 0, 0)];

    assert!(
        print_fingerprint_diff(&fps_a, &fps_b, false, false).is_err(),
        "a tensor present only in model B is a difference and must fail the diff"
    );
}

#[test]
fn falsifier_fingerprint_diff_identical_models_still_pass() {
    // Control: the fail-closed change must not turn every diff red.
    let fps_a = vec![
        make_fingerprint("model.embed_tokens.weight", 0.01, 0.02, 0, 0),
        make_fingerprint("model.norm.weight", 1.0, 0.1, 0, 0),
    ];
    assert!(
        print_fingerprint_diff(&fps_a, &fps_a, false, false).is_ok(),
        "a model compared against itself must still pass"
    );
}

#[test]
fn falsifier_missing_tensor_anomalies_are_counted_by_direction() {
    let anomalies = vec![
        missing_tensor_anomaly("a1", FIELD_MISSING_IN_B),
        missing_tensor_anomaly("a2", FIELD_MISSING_IN_B),
        missing_tensor_anomaly("b1", FIELD_MISSING_IN_A),
    ];
    assert_eq!(count_field(&anomalies, FIELD_MISSING_IN_B), 2);
    assert_eq!(count_field(&anomalies, FIELD_MISSING_IN_A), 1);
}

// ── rosetta verify: FAILED must not exit 0 ─────────────────────────────────

/// The report `verify_roundtrip` produces for the shipped repro: tensor count
/// mismatch, `max_diff` / `mean_diff` non-finite.
fn tensor_count_mismatch_report() -> VerificationReport {
    let mut report = VerificationReport::passing();
    report.is_equivalent = false;
    report.max_diff = f32::INFINITY;
    report.mean_diff = f32::INFINITY;
    report.failed_tensors = vec!["Tensor count mismatch".to_string()];
    report
}

#[test]
fn falsifier_verify_outcome_is_an_error_when_the_report_failed() {
    // 0.63.0 printed "Round-trip verification FAILED" and returned Ok(()), so a
    // `set -e` pipeline saw a broken round-trip as a pass.
    let report = tensor_count_mismatch_report();
    let err = verify_outcome(&report, 1e-5).expect_err("a FAILED report must not exit 0");
    let msg = err.to_string();
    assert!(
        msg.contains("FAILED"),
        "the error must say what failed, got: {msg}"
    );
}

#[test]
fn falsifier_verify_outcome_is_ok_when_the_report_passed() {
    // Control: an equivalent round-trip still exits 0.
    assert!(verify_outcome(&VerificationReport::passing(), 1e-5).is_ok());
}

// ── rosetta verify --json: non-finite values must not be bare `inf` ────────

#[test]
fn falsifier_verification_json_is_parseable_when_diffs_are_infinite() {
    // 0.63.0 emitted `"max_diff": inf`, which RFC 8259 has no literal for:
    // python's json rejects the document and jq silently rewrites it to f64::MAX.
    let report = tensor_count_mismatch_report();
    let text = serde_json::to_string(&verification_json(&report)).expect("serialize");
    assert!(
        !text.contains("inf"),
        "the document must not contain a bare inf literal: {text}"
    );

    let parsed: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("verify --json must be valid JSON, got {text:?}: {e}"));
    assert!(parsed["max_diff"].is_null(), "non-finite becomes null");
    assert!(parsed["mean_diff"].is_null(), "non-finite becomes null");
    assert_eq!(parsed["is_equivalent"], serde_json::json!(false));
}

#[test]
fn falsifier_verification_json_keeps_finite_numbers() {
    // Control: the passing path is unchanged and still numeric.
    let mut report = VerificationReport::passing();
    report.max_diff = 0.25;
    report.mean_diff = 0.125;
    let parsed = verification_json(&report);
    assert_eq!(parsed["max_diff"], serde_json::json!(0.25));
    assert_eq!(parsed["mean_diff"], serde_json::json!(0.125));
}

#[test]
fn falsifier_json_number_maps_non_finite_to_null() {
    assert!(json_number(f32::INFINITY).is_null());
    assert!(json_number(f32::NEG_INFINITY).is_null());
    assert!(json_number(f32::NAN).is_null());
    assert_eq!(json_number(1.5), serde_json::json!(1.5));
}

#[test]
fn falsifier_validate_stats_json_is_parseable_with_infinite_deviation() {
    // nan_count/inf_count anomalies carry deviation_sigma = INFINITY, which the
    // hand-rolled `{:.2}` printed as a bare `inf`.
    let anomalies = vec![StatisticalAnomaly {
        tensor: "model.layers.0.weight".to_string(),
        field: "nan_count".to_string(),
        expected: 0.0,
        actual: 12.0,
        deviation_sigma: f32::INFINITY,
    }];
    let doc = validate_stats_json(Path::new("/tmp/m.apr"), 3.0, false, 1, &anomalies);
    let text = serde_json::to_string(&doc).expect("serialize");
    assert!(!text.contains("inf"), "no bare inf literal: {text}");
    let parsed: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("validate-stats --json must be valid JSON: {e}"));
    assert!(parsed["anomaly_details"][0]["deviation"].is_null());
    assert_eq!(parsed["passed"], serde_json::json!(false));
}

// ── rosetta compare-inference: zero token pairs is not a 100% match ────────

#[test]
fn falsifier_compare_inference_zero_tokens_is_not_a_match() {
    // The shipped repro compared a 1.5B GGUF against a 0.5B APR, captured zero
    // token pairs, and reported "RESULT: INFERENCE MATCH (100%)".
    let line = inference_result_line(0, 0, 0.1);
    assert!(
        !line.contains("MATCH (100%)"),
        "0 token pairs must not read as a 100% match, got: {line}"
    );
    assert!(
        line.contains("VACUOUS"),
        "0 token pairs must be reported as vacuous, got: {line}"
    );
}

#[test]
fn falsifier_compare_inference_real_match_still_reads_as_a_match() {
    // Control: a genuine 5-of-5 match is unchanged.
    assert!(inference_result_line(5, 0, 0.1).contains("MATCH (100%)"));
}

#[test]
fn falsifier_compare_passed_is_false_with_zero_tokens() {
    // The `--json` surface: `"passed": true` for a comparison of nothing made this
    // command unusable as a CI gate.
    assert!(
        !compare_passed(0, 0, 0.1),
        "a comparison of zero token pairs never passes"
    );
    assert!(compare_passed(5, 0, 0.1), "a real 5/5 match still passes");
    assert!(
        !compare_passed(5, 5, 0.1),
        "a real 0/5 match still does not pass"
    );
}

#[test]
fn falsifier_validate_captured_tokens_rejects_vacuous_comparison() {
    // Both models emitted text, but no token pairs were captured. 0.63.0 returned
    // Ok(()) here, which is the whole reason the command exited 0.
    let err = validate_captured_tokens("4 2", "<unk><unk><unk>")
        .expect_err("zero token pairs compared must never be a success");
    let msg = err.to_string();
    assert!(
        msg.contains("VACUOUS"),
        "the error must name the vacuous comparison, got: {msg}"
    );
}

#[test]
fn falsifier_validate_captured_tokens_still_reports_a_silent_model() {
    // Controls: the pre-existing GH-188 diagnoses must survive.
    assert!(validate_captured_tokens("", "").is_err());
    assert!(validate_captured_tokens("", "some text").is_err());
    assert!(validate_captured_tokens("some text", "").is_err());
    assert!(validate_captured_tokens("120 tok/s", "real output").is_err());
}
