/// Verify compare_tensor_file returns ParseError for garbage data
#[test]
fn test_compare_tensor_file_invalid_safetensors() {
    let dir = make_test_dir("compare_tensor_invalid");
    let path = dir.join("bad.safetensors");
    std::fs::write(&path, b"garbage data").expect("write");

    let golden = GoldenOutput {
        input_hash: "h".to_string(),
        prompt: "p".to_string(),
        logits: vec![1.0],
        shape: vec![1],
        text: None,
        model_id: String::new(),
        transformers_version: String::new(),
    };

    let oracle = HfParityOracle::new("/tmp", "test");
    let result = oracle.compare_tensor_file(&path, &golden);
    assert!(matches!(result, Err(TensorDiff::ParseError { .. })));
    if let Err(TensorDiff::ParseError { message }) = result {
        assert!(message.contains("Failed to parse SafeTensors"));
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Verify compare_tensor_file returns ParseError when logits tensor is absent
#[test]
fn test_compare_tensor_file_missing_logits() {
    use safetensors::tensor::Dtype;
    use std::borrow::Cow;

    struct TestTensor {
        shape: Vec<usize>,
        data: Vec<u8>,
    }
    impl safetensors::tensor::View for TestTensor {
        fn dtype(&self) -> Dtype {
            Dtype::F32
        }
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }
        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    let dir = make_test_dir("compare_tensor_no_logits");
    let tensor = TestTensor {
        shape: vec![1],
        data: vec![0u8; 4],
    };
    let tensors = vec![("other_name".to_string(), tensor)];
    let bytes = safetensors::tensor::serialize(tensors, None).expect("serialize");
    let path = dir.join("no_logits.safetensors");
    std::fs::write(&path, bytes).expect("write");

    let golden = GoldenOutput {
        input_hash: "h".to_string(),
        prompt: "p".to_string(),
        logits: vec![1.0],
        shape: vec![1],
        text: None,
        model_id: String::new(),
        transformers_version: String::new(),
    };

    let oracle = HfParityOracle::new("/tmp", "test");
    let result = oracle.compare_tensor_file(&path, &golden);
    assert!(matches!(result, Err(TensorDiff::ParseError { .. })));
    if let Err(TensorDiff::ParseError { message }) = result {
        assert!(message.contains("Missing 'logits' tensor"));
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Verify compare_tensor_file returns ShapeMismatch when dimensions differ
#[test]
fn test_compare_tensor_file_shape_mismatch() {
    let dir = make_test_dir("compare_tensor_shape");
    // Actual has 3 logits
    let actual_path = dir.join("actual.safetensors");
    create_safetensors_file(&actual_path, &[1.0f32, 2.0, 3.0], &[1, 3]);

    // Golden has 5 logits
    let golden = GoldenOutput {
        input_hash: "h".to_string(),
        prompt: "p".to_string(),
        logits: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        shape: vec![1, 5],
        text: None,
        model_id: String::new(),
        transformers_version: String::new(),
    };

    let oracle = HfParityOracle::new("/tmp", "test");
    let result = oracle.compare_tensor_file(&actual_path, &golden);
    assert!(matches!(result, Err(TensorDiff::ShapeMismatch { .. })));

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================
// Oracle evaluate Tests (file-based, covering golden paths)
// ============================================================

/// Verify oracle evaluate returns Corroborated when output matches golden text
#[test]
fn test_oracle_evaluate_text_match() {
    let dir = make_test_dir("evaluate_text_match");
    let family = "eval-family";
    let prompt = "match prompt";
    let input_hash = hash_prompt(prompt);

    let family_dir = dir.join(family);
    std::fs::create_dir_all(&family_dir).expect("mkdir");

    // Create safetensors file
    let st_path = family_dir.join(format!("{input_hash}.safetensors"));
    create_safetensors_file(&st_path, &[1.0f32], &[1]);

    // Create metadata JSON with generated_text
    let meta_path = family_dir.join(format!("{input_hash}.json"));
    std::fs::write(
        &meta_path,
        r#"{"model": "m", "transformers_version": "4.0", "generated_text": "expected output"}"#,
    )
    .expect("write meta");

    let oracle = HfParityOracle::new(&dir, family);
    let result = oracle.evaluate(prompt, "expected output");
    assert!(result.is_corroborated());
    if let OracleResult::Corroborated { evidence } = result {
        assert!(evidence.contains("Text output matches HF golden"));
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Verify oracle evaluate returns Falsified when output differs from golden text
#[test]
fn test_oracle_evaluate_text_mismatch_falsified() {
    let dir = make_test_dir("evaluate_text_mismatch");
    let family = "eval-family2";
    let prompt = "mismatch prompt";
    let input_hash = hash_prompt(prompt);

    let family_dir = dir.join(family);
    std::fs::create_dir_all(&family_dir).expect("mkdir");

    let st_path = family_dir.join(format!("{input_hash}.safetensors"));
    create_safetensors_file(&st_path, &[1.0f32], &[1]);

    let meta_path = family_dir.join(format!("{input_hash}.json"));
    std::fs::write(
        &meta_path,
        r#"{"model": "m", "transformers_version": "4.0", "generated_text": "expected text"}"#,
    )
    .expect("write meta");

    let oracle = HfParityOracle::new(&dir, family);
    // Output differs and is not a file path
    let result = oracle.evaluate(prompt, "totally different output");
    assert!(result.is_falsified());
    if let OracleResult::Falsified { reason, evidence } = result {
        assert!(reason.contains("Text output differs"));
        assert!(evidence.contains("Expected:"));
        assert!(evidence.contains("Actual:"));
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Verify oracle evaluate returns Corroborated when tensor logits match golden
#[test]
fn test_oracle_evaluate_tensor_file_corroborated() {
    let dir = make_test_dir("evaluate_tensor_ok");
    let family = "tensor-family";
    let prompt = "tensor prompt";
    let input_hash = hash_prompt(prompt);

    let family_dir = dir.join(family);
    std::fs::create_dir_all(&family_dir).expect("mkdir");

    let logits = vec![1.0f32, 2.0, 3.0];
    let st_path = family_dir.join(format!("{input_hash}.safetensors"));
    create_safetensors_file(&st_path, &logits, &[1, 3]);
    // No metadata JSON (no expected text) so it falls through to tensor comparison

    // Create actual output file
    let actual_path = dir.join("actual_output.safetensors");
    create_safetensors_file(&actual_path, &logits, &[1, 3]);

    let oracle = HfParityOracle::new(&dir, family);
    let result = oracle.evaluate(prompt, actual_path.to_str().expect("utf8"));
    assert!(result.is_corroborated());
    if let OracleResult::Corroborated { evidence } = result {
        assert!(evidence.contains("Tensor parity verified"));
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Verify oracle evaluate returns Falsified when tensor logits diverge from golden
#[test]
fn test_oracle_evaluate_tensor_file_falsified() {
    let dir = make_test_dir("evaluate_tensor_fail");
    let family = "tensor-family2";
    let prompt = "tensor prompt fail";
    let input_hash = hash_prompt(prompt);

    let family_dir = dir.join(family);
    std::fs::create_dir_all(&family_dir).expect("mkdir");

    let golden_logits = vec![1.0f32, 2.0, 3.0, 4.0];
    let st_path = family_dir.join(format!("{input_hash}.safetensors"));
    create_safetensors_file(&st_path, &golden_logits, &[1, 4]);

    // Actual output differs drastically
    let actual_logits = vec![100.0f32, 200.0, 300.0, 400.0];
    let actual_path = dir.join("bad_output.safetensors");
    create_safetensors_file(&actual_path, &actual_logits, &[1, 4]);

    let oracle = HfParityOracle::new(&dir, family);
    let result = oracle.evaluate(prompt, actual_path.to_str().expect("utf8"));
    assert!(result.is_falsified());
    if let OracleResult::Falsified { reason, .. } = result {
        assert!(reason.contains("Tensor mismatch"));
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Verify oracle evaluate corroborates plain text when no golden text or tensor path
#[test]
fn test_oracle_evaluate_plain_text_no_golden_text_no_tensor() {
    // When golden exists but has no expected text, and output is not a
    // .safetensors file path, it should corroborate with the text fallback.
    let dir = make_test_dir("evaluate_plain_text");
    let family = "plain-family";
    let prompt = "plain prompt";
    let input_hash = hash_prompt(prompt);

    let family_dir = dir.join(family);
    std::fs::create_dir_all(&family_dir).expect("mkdir");

    let st_path = family_dir.join(format!("{input_hash}.safetensors"));
    create_safetensors_file(&st_path, &[1.0f32], &[1]);

    let oracle = HfParityOracle::new(&dir, family);
    let result = oracle.evaluate(prompt, "just some plain text output");
    assert!(result.is_corroborated());
    if let OracleResult::Corroborated { evidence } = result {
        assert!(evidence.contains("no tensor comparison"));
    }

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================
// truncate edge case Tests
// ============================================================

/// Verify truncate backs up to char boundary for 2-byte UTF-8 sequences
#[test]
fn test_truncate_mid_multibyte_char() {
    // "é" is 2 bytes in UTF-8. "éé" is 4 bytes.
    // Truncating at 3 should back up to byte 2 (end of first "é")
    let s = "éé";
    let truncated = truncate(s, 3);
    assert_eq!(truncated, "é");
}

/// Verify truncate returns empty string when max_len is zero
#[test]
fn test_truncate_at_zero() {
    assert_eq!(truncate("hello", 0), "");
}

/// Verify truncate backs up to char boundary for 3-byte CJK characters
#[test]
fn test_truncate_three_byte_mid_boundary() {
    // Each CJK character is 3 bytes. "漢字" = 6 bytes.
    // Truncating at 4 should back up to 3 (end of "漢")
    let s = "漢字";
    let truncated = truncate(s, 4);
    assert_eq!(truncated, "漢");
}

/// Verify truncate backs up to empty string for 4-byte emoji when max_len is 2
#[test]
fn test_truncate_four_byte_emoji() {
    // Emoji like U+1F600 is 4 bytes in UTF-8
    let s = "\u{1F600}abc";
    // Truncating at 2 bytes should back up to 0 since the first char is 4 bytes
    let truncated = truncate(s, 2);
    assert_eq!(truncated, "");
}

// ============================================================
// detect_systematic_bias edge case Tests
// ============================================================

/// Verify detect_systematic_bias returns None for mismatched-length slices
#[test]
fn test_detect_systematic_bias_mismatched_len() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0];
    assert!(HfParityOracle::detect_systematic_bias(&a, &b).is_none());
}

/// Verify detect_systematic_bias returns None when drift is within 10% threshold
#[test]
fn test_detect_systematic_bias_no_scale_drift_within_threshold() {
    // Scale ratio within 10%: std_a/std_e close to 1.0
    let a = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let b = vec![0.0, 1.05, 2.1, 3.15, 4.2]; // ~5% scale up
    let result = HfParityOracle::detect_systematic_bias(&a, &b);
    // Mean shift: mean_a=2.1, mean_e=2.0, shift=0.1, std_e=~1.41, 0.1/1.41=0.07 sigma (< 3)
    // Scale drift: std_a/std_e ≈ 1.05, (1.05-1)=0.05 < 0.1
    assert!(result.is_none());
}

// ============================================================
// Tolerance serialization Tests
// ============================================================

/// Verify Tolerance roundtrips through JSON serialization
#[test]
fn test_tolerance_serialize_deserialize() {
    let tol = Tolerance::fp16();
    let json = serde_json::to_string(&tol).expect("serialize");
    let tol2: Tolerance = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(tol, tol2);
}

/// Verify Tolerance Debug implementation includes field names
#[test]
fn test_tolerance_debug() {
    let tol = Tolerance::default();
    let debug = format!("{tol:?}");
    assert!(debug.contains("Tolerance"));
    assert!(debug.contains("atol_fp32"));
}

// ============================================================
// TensorDiff serialization Tests
// ============================================================

/// Verify TensorDiff::ShapeMismatch roundtrips through JSON serialization
#[test]
fn test_tensor_diff_serialize_deserialize_shape() {
    let diff = TensorDiff::ShapeMismatch {
        expected: 100,
        actual: 200,
    };
    let json = serde_json::to_string(&diff).expect("serialize");
    let diff2: TensorDiff = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(diff, diff2);
}

/// Verify TensorDiff::ValueMismatch roundtrips through JSON serialization
#[test]
fn test_tensor_diff_serialize_deserialize_value() {
    let diff = TensorDiff::ValueMismatch {
        num_mismatches: 5,
        total: 50,
        mismatch_ratio: 0.1,
        max_diff: 0.5,
        max_diff_idx: 7,
        expected_val: 1.0,
        actual_val: 1.5,
        mean_diff: 0.2,
    };
    let json = serde_json::to_string(&diff).expect("serialize");
    let diff2: TensorDiff = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(diff, diff2);
}

/// Verify TensorDiff::ParseError roundtrips through JSON serialization
#[test]
fn test_tensor_diff_serialize_deserialize_parse_error() {
    let diff = TensorDiff::ParseError {
        message: "something went wrong".to_string(),
    };
    let json = serde_json::to_string(&diff).expect("serialize");
    let diff2: TensorDiff = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(diff, diff2);
}

// ============================================================
// load_metadata_json Tests
// ============================================================

/// Verify load_metadata_json handles empty JSON with serde defaults
#[test]
fn test_load_metadata_json_minimal() {
    let dir = make_test_dir("meta_minimal");
    let path = dir.join("meta.json");
    // All fields use serde default
    std::fs::write(&path, r#"{}"#).expect("write");

    let result = HfParityOracle::load_metadata_json(&path);
    assert!(result.is_ok());
    let (model, version, text) = result.expect("ok");
    assert!(model.is_empty());
    assert!(version.is_empty());
    assert!(text.is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

/// Verify load_metadata_json extracts model, version, and generated_text fields
#[test]
fn test_load_metadata_json_full() {
    let dir = make_test_dir("meta_full");
    let path = dir.join("meta.json");
    std::fs::write(
            &path,
            r#"{"model": "bigmodel", "transformers_version": "5.0.0", "generated_text": "output here"}"#,
        )
        .expect("write");

    let result = HfParityOracle::load_metadata_json(&path);
    assert!(result.is_ok());
    let (model, version, text) = result.expect("ok");
    assert_eq!(model, "bigmodel");
    assert_eq!(version, "5.0.0");
    assert_eq!(text, Some("output here".to_string()));

    let _ = std::fs::remove_dir_all(&dir);
}

/// Verify load_metadata_json returns error for nonexistent path
#[test]
fn test_load_metadata_json_not_found() {
    let result = HfParityOracle::load_metadata_json(Path::new("/nonexistent/meta.json"));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to read metadata"));
}

/// Verify load_metadata_json returns error for malformed JSON content
#[test]
fn test_load_metadata_json_invalid_json() {
    let dir = make_test_dir("meta_invalid");
    let path = dir.join("meta.json");
    std::fs::write(&path, "{{{{invalid").expect("write");

    let result = HfParityOracle::load_metadata_json(&path);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .contains("Failed to parse metadata JSON")
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================
// Integration Tests (require golden corpus)
// ============================================================
