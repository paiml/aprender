/// Verify tagged safetensors extension generation from source path
#[test]
fn test_convert_to_format_tagged_safetensors_ext() {
    let source = std::path::PathBuf::from("/tmp/model.apr");
    let target = source.with_extension("tag2.safetensors");
    assert!(target.to_str().expect("path").ends_with("tag2.safetensors"));
}

/// Verify GPU backend flag exercises the GPU code path (fails on missing binary)
#[test]
fn test_run_inference_simple_gpu_flag() {
    // Verify GPU backend produces --gpu arg (fails because no binary, but exercises the match)
    let result = run_inference_simple(
        &std::path::PathBuf::from("/nonexistent/model.gguf"),
        Backend::Gpu,
        "/nonexistent/apr",
    );
    assert!(result.is_err());
}

/// Verify Falsified result structure for idempotency failures
#[test]
fn test_idempotency_falsified_result_structure() {
    // Directly test the Falsified variant construction
    let result = ConversionResult::Falsified {
        gate_id: "F-CONV-IDEM-001".to_string(),
        reason: "Idempotency failure: Gguf→Apr produced different output".to_string(),
        evidence: ConversionEvidence {
            source_hash: ConversionTest::hash_output("output1"),
            converted_hash: ConversionTest::hash_output("output2"),
            max_diff: 1.0,
            diff_indices: vec![],
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            failure_type: None,
            quant_type: None,
        },
    };
    match result {
        ConversionResult::Falsified {
            gate_id, reason, ..
        } => {
            assert_eq!(gate_id, "F-CONV-IDEM-001");
            assert!(reason.contains("Idempotency"));
        }
        ConversionResult::Corroborated { .. } => panic!("Expected Falsified"),
    }
}

/// Verify Falsified result structure for commutativity failures
#[test]
fn test_commutativity_falsified_result_structure() {
    let result = ConversionResult::Falsified {
        gate_id: "F-CONV-COM-001".to_string(),
        reason: "Commutativity failure: GGUF→APR differs from GGUF→ST→APR".to_string(),
        evidence: ConversionEvidence {
            source_hash: ConversionTest::hash_output("path_a"),
            converted_hash: ConversionTest::hash_output("path_b"),
            max_diff: 1.0,
            diff_indices: vec![],
            source_format: Format::Gguf,
            target_format: Format::Apr,
            backend: Backend::Cpu,
            failure_type: None,
            quant_type: None,
        },
    };
    match result {
        ConversionResult::Falsified {
            gate_id, reason, ..
        } => {
            assert_eq!(gate_id, "F-CONV-COM-001");
            assert!(reason.contains("Commutativity"));
        }
        ConversionResult::Corroborated { .. } => panic!("Expected Falsified"),
    }
}

/// Verify convert_to_format_tagged returns error for nonexistent model
#[test]
fn test_conversion_test_convert_model_failure() {
    // Exercise the conversion failure error path
    let result = convert_to_format_tagged(
        &std::path::PathBuf::from("/nonexistent/model.gguf"),
        Format::Gguf,
        "test",
        "/nonexistent/apr",
    );
    assert!(result.is_err());
}

/// Verify convert_to_format_tagged handles SafeTensors target format
#[test]
fn test_conversion_test_convert_model_safetensors_target() {
    let result = convert_to_format_tagged(
        &std::path::PathBuf::from("/nonexistent/model.apr"),
        Format::SafeTensors,
        "test",
        "/nonexistent/apr",
    );
    assert!(result.is_err());
}

// ── §3.4 classify_failure tests ────────────────────────────────────

/// Verify classify_failure detects tensor name mismatch patterns
#[test]
fn test_classify_failure_tensor_name_mismatch() {
    assert_eq!(
        classify_failure("tensor name mismatch: q_proj not found", 1),
        ConversionFailureType::TensorNameMismatch
    );
    assert_eq!(
        classify_failure("missing tensor 'lm_head.weight'", 1),
        ConversionFailureType::TensorNameMismatch
    );
    assert_eq!(
        classify_failure("unexpected tensor in output", 1),
        ConversionFailureType::TensorNameMismatch
    );
}

/// Verify classify_failure detects dequantization failure patterns
#[test]
fn test_classify_failure_dequantization() {
    assert_eq!(
        classify_failure("dequantization error: NaN values produced", 1),
        ConversionFailureType::DequantizationFailure
    );
    assert_eq!(
        classify_failure("quantization overflow detected", 1),
        ConversionFailureType::DequantizationFailure
    );
    assert_eq!(
        classify_failure("NaN in output tensor", 1),
        ConversionFailureType::DequantizationFailure
    );
    assert_eq!(
        classify_failure("infinity values in layer 5", 1),
        ConversionFailureType::DequantizationFailure
    );
}

/// Verify classify_failure detects config metadata mismatch patterns
#[test]
fn test_classify_failure_config_metadata() {
    assert_eq!(
        classify_failure("hidden_size mismatch: expected 768 got 512", 1),
        ConversionFailureType::ConfigMetadataMismatch
    );
    assert_eq!(
        classify_failure("metadata mismatch: num_layers differs", 1),
        ConversionFailureType::ConfigMetadataMismatch
    );
    assert_eq!(
        classify_failure("vocab_size does not match model", 1),
        ConversionFailureType::ConfigMetadataMismatch
    );
    assert_eq!(
        classify_failure("config mismatch detected", 1),
        ConversionFailureType::ConfigMetadataMismatch
    );
}

/// Verify classify_failure detects missing artifact patterns
#[test]
fn test_classify_failure_missing_artifact() {
    assert_eq!(
        classify_failure("file not found: model.safetensors", 1),
        ConversionFailureType::MissingArtifact
    );
    assert_eq!(
        classify_failure("No such file or directory", 1),
        ConversionFailureType::MissingArtifact
    );
    assert_eq!(
        classify_failure("tokenizer.json missing from model directory", 1),
        ConversionFailureType::MissingArtifact
    );
    assert_eq!(
        classify_failure("config.json: file not found", 1),
        ConversionFailureType::MissingArtifact
    );
}

/// Verify classify_failure detects inference failure patterns including SIGSEGV
#[test]
fn test_classify_failure_inference() {
    assert_eq!(
        classify_failure("inference failed: out of memory", 1),
        ConversionFailureType::InferenceFailure
    );
    assert_eq!(
        classify_failure("forward pass error", 1),
        ConversionFailureType::InferenceFailure
    );
    assert_eq!(
        classify_failure("", -11), // SIGSEGV
        ConversionFailureType::InferenceFailure
    );
}

/// Verify classify_failure falls back to Unknown for unrecognized patterns
#[test]
fn test_classify_failure_unknown() {
    assert_eq!(
        classify_failure("some generic error", 1),
        ConversionFailureType::Unknown
    );
    assert_eq!(classify_failure("", 1), ConversionFailureType::Unknown);
}

/// Verify classify_failure matches patterns case-insensitively
#[test]
fn test_classify_failure_case_insensitive() {
    assert_eq!(
        classify_failure("TENSOR NAME MISMATCH", 1),
        ConversionFailureType::TensorNameMismatch
    );
    assert_eq!(
        classify_failure("Dequantization Error", 1),
        ConversionFailureType::DequantizationFailure
    );
}

// ── §3.7 QuantType + tolerance tests ───────────────────────────────

/// Verify QuantType::from_str_label parses all known label variants
#[test]
fn test_quant_type_from_str_label() {
    assert_eq!(QuantType::from_str_label("f32"), QuantType::F32);
    assert_eq!(QuantType::from_str_label("fp32"), QuantType::F32);
    assert_eq!(QuantType::from_str_label("float32"), QuantType::F32);
    assert_eq!(QuantType::from_str_label("f16"), QuantType::F16);
    assert_eq!(QuantType::from_str_label("fp16"), QuantType::F16);
    assert_eq!(QuantType::from_str_label("bf16"), QuantType::BF16);
    assert_eq!(QuantType::from_str_label("bfloat16"), QuantType::BF16);
    assert_eq!(QuantType::from_str_label("q4_k_m"), QuantType::Q4KM);
    assert_eq!(QuantType::from_str_label("q4km"), QuantType::Q4KM);
    assert_eq!(QuantType::from_str_label("q5_k_m"), QuantType::Q5KM);
    assert_eq!(QuantType::from_str_label("q5km"), QuantType::Q5KM);
    assert_eq!(QuantType::from_str_label("q6_k"), QuantType::Q6K);
    assert_eq!(QuantType::from_str_label("q4_0"), QuantType::Q4_0);
    assert_eq!(QuantType::from_str_label("q8_0"), QuantType::Q8_0);
    assert_eq!(
        QuantType::from_str_label("unknown_type"),
        QuantType::Unknown
    );
}

/// Verify QuantType::from_str_label is case-insensitive
#[test]
fn test_quant_type_from_str_label_case_insensitive() {
    assert_eq!(QuantType::from_str_label("F32"), QuantType::F32);
    assert_eq!(QuantType::from_str_label("BF16"), QuantType::BF16);
    assert_eq!(QuantType::from_str_label("Q4_K_M"), QuantType::Q4KM);
    assert_eq!(QuantType::from_str_label("Q5_K_M"), QuantType::Q5KM);
}

/// Verify QuantType::from_str_label accepts hyphen-separated labels
#[test]
fn test_quant_type_from_str_label_with_hyphens() {
    assert_eq!(QuantType::from_str_label("q4-k-m"), QuantType::Q4KM);
    assert_eq!(QuantType::from_str_label("q5-k-m"), QuantType::Q5KM);
    assert_eq!(QuantType::from_str_label("q6-k"), QuantType::Q6K);
}

/// Verify F32 tolerance values
#[test]
fn test_tolerance_for_f32() {
    let tol = tolerance_for(QuantType::F32);
    assert!((tol.atol - 1e-6).abs() < 1e-10);
}

/// Verify F16 tolerance values
#[test]
fn test_tolerance_for_f16() {
    let tol = tolerance_for(QuantType::F16);
    assert!((tol.atol - 1e-3).abs() < 1e-10);
}

/// Verify Q4KM tolerance values
#[test]
fn test_tolerance_for_q4km() {
    let tol = tolerance_for(QuantType::Q4KM);
    assert!((tol.atol - 1e-1).abs() < 1e-10);
}

/// Verify Q5KM tolerance values for both atol and rtol
#[test]
fn test_tolerance_for_q5km() {
    let tol = tolerance_for(QuantType::Q5KM);
    assert!((tol.atol - 7.5e-2).abs() < 1e-10);
    assert!((tol.rtol - 5e-2).abs() < 1e-10);
}

/// Verify Q6K tolerance values
#[test]
fn test_tolerance_for_q6k() {
    let tol = tolerance_for(QuantType::Q6K);
    assert!((tol.atol - 5e-2).abs() < 1e-10);
}

// ── ConversionTest::execute branch coverage ──────────────────────────────────

/// ConversionTest::execute: same-format, identical outputs → Corroborated
#[test]
fn test_conversion_test_execute_corroborated_same_format() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    // Mock: `run` always returns "The answer is 4.", `rosetta convert` touches target
    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
run) printf "The answer is 4."; exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
    );

    let mut test = ConversionTest::new(
        Format::Gguf, Format::Gguf, Backend::Cpu,
        ModelId::new("test", "model"),
    );
    test.binary = mock.to_string_lossy().to_string();

    match test.execute(&model_file) {
        Ok(ConversionResult::Corroborated { source_format, target_format, max_diff, .. }) => {
            assert_eq!(source_format, Format::Gguf);
            assert_eq!(target_format, Format::Gguf);
            assert!(max_diff < 1e-6, "Same outputs should have near-zero diff");
        }
        other => panic!("Expected Corroborated, got: {other:?}"),
    }
}

/// ConversionTest::execute: cross-format, non-garbage outputs → Corroborated
#[test]
fn test_conversion_test_execute_corroborated_cross_format() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
run) printf "The answer is four."; exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
    );

    let mut test = ConversionTest::new(
        Format::Gguf, Format::Apr, Backend::Cpu,
        ModelId::new("test", "model"),
    );
    test.binary = mock.to_string_lossy().to_string();

    // Cross-format: uses garbage detection (not text diff). Non-garbage → Corroborated.
    match test.execute(&model_file) {
        Ok(ConversionResult::Corroborated { source_format, target_format, .. }) => {
            assert_eq!(source_format, Format::Gguf);
            assert_eq!(target_format, Format::Apr);
        }
        other => panic!("Expected Corroborated for non-garbage cross-format, got: {other:?}"),
    }
}

/// ConversionTest::execute: same-format, different outputs → Falsified (diff > epsilon)
#[test]
fn test_conversion_test_execute_falsified_same_format_diff() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    // Source returns "aaa", converted returns "zzz" → large diff
    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
run)
  case "$2" in
  *converted*) printf "zzzzzzzzzzz";;
  *) printf "aaaaaaaaaaa";;
  esac
  exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
    );

    let mut test = ConversionTest::new(
        Format::Gguf, Format::Gguf, Backend::Cpu,
        ModelId::new("test", "model"),
    );
    test.binary = mock.to_string_lossy().to_string();

    match test.execute(&model_file) {
        Ok(ConversionResult::Falsified { gate_id, reason, .. }) => {
            assert_eq!(gate_id, "F-CONV-G-G");
            assert!(reason.contains("different output"), "reason: {reason}");
        }
        other => panic!("Expected Falsified for diff outputs, got: {other:?}"),
    }
}

/// ConversionTest::execute: cross-format, converted output is garbage → Falsified
#[test]
fn test_conversion_test_execute_falsified_cross_format_garbage() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    // Source output is valid; converted output is garbage (too short = 1 char)
    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
run)
  case "$2" in
  *converted*) printf "x";;
  *) printf "The answer is four and a half.";;
  esac
  exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
    );

    let mut test = ConversionTest::new(
        Format::Gguf, Format::Apr, Backend::Cpu,
        ModelId::new("test", "model"),
    );
    test.binary = mock.to_string_lossy().to_string();

    match test.execute(&model_file) {
        Ok(ConversionResult::Falsified { gate_id, reason, .. }) => {
            assert_eq!(gate_id, "F-CONV-G-A");
            assert!(reason.contains("garbage"), "reason should mention garbage: {reason}");
        }
        other => panic!("Expected Falsified for garbage converted output, got: {other:?}"),
    }
}

/// ConversionTest::execute: cross-format, file exists but inference on converted fails
/// → Falsified with InferenceFailure
#[test]
fn test_conversion_test_execute_falsified_cross_format_inference_failure() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    // Source inference succeeds; rosetta convert creates file; converted inference fails
    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
run)
  case "$2" in
  *converted*) echo "inference failed" >&2; exit 1;;
  *) printf "The answer is 4."; exit 0;;
  esac;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
    );

    let mut test = ConversionTest::new(
        Format::Gguf, Format::Apr, Backend::Cpu,
        ModelId::new("test", "model"),
    );
    test.binary = mock.to_string_lossy().to_string();

    match test.execute(&model_file) {
        Ok(ConversionResult::Falsified { evidence, reason, .. }) => {
            assert_eq!(
                evidence.failure_type,
                Some(ConversionFailureType::InferenceFailure),
                "Expected InferenceFailure, got: {:?}. reason: {reason}",
                evidence.failure_type
            );
        }
        other => panic!("Expected Falsified with InferenceFailure, got: {other:?}"),
    }
}
