/// Verify default tolerance values match expected FP32 defaults
#[test]
fn test_tolerance_default() {
    let tol = Tolerance::default();
    assert!((tol.atol_fp32 - 1e-5).abs() < 1e-10);
    assert!((tol.rtol_fp32 - 1e-4).abs() < 1e-10);
    assert!((tol.atol_quant - 1e-2).abs() < 1e-10);
    assert!((tol.max_mismatch_ratio - 0.01).abs() < 1e-10);
}

/// Verify FP32 tolerance preset has correct absolute tolerance
#[test]
fn test_tolerance_fp32() {
    let tol = Tolerance::fp32();
    assert!((tol.atol_fp32 - 1e-5).abs() < 1e-10);
}

/// Verify FP16 tolerance preset uses relaxed absolute tolerance
#[test]
fn test_tolerance_fp16() {
    let tol = Tolerance::fp16();
    assert!((tol.atol_fp32 - 1e-3).abs() < 1e-10);
}

/// Verify INT8 tolerance preset uses wider absolute tolerance
#[test]
fn test_tolerance_int8() {
    let tol = Tolerance::int8();
    assert!((tol.atol_fp32 - 1e-1).abs() < 1e-10);
}

/// Verify INT4 tolerance preset allows largest absolute tolerance
#[test]
fn test_tolerance_int4() {
    let tol = Tolerance::int4();
    assert!((tol.atol_fp32 - 5e-1).abs() < 1e-10);
}

/// Verify identical values are always considered close
#[test]
fn test_tolerance_is_close_identical() {
    let tol = Tolerance::default();
    assert!(tol.is_close(1.0, 1.0));
    assert!(tol.is_close(0.0, 0.0));
    assert!(tol.is_close(-5.0, -5.0));
}

/// Verify values within absolute tolerance are considered close
#[test]
fn test_tolerance_is_close_within_atol() {
    let tol = Tolerance::default();
    // diff = 1e-6, bound = 1e-5 + 1e-4 * 1.0 = 1.1e-4 → close
    assert!(tol.is_close(1.000001, 1.0));
}

/// Verify values exceeding both absolute and relative tolerance are rejected
#[test]
fn test_tolerance_is_close_outside_tolerance() {
    let tol = Tolerance::default();
    // diff = 0.1, bound = 1e-5 + 1e-4 * 1.0 = 1.1e-4 → not close
    assert!(!tol.is_close(1.1, 1.0));
}

/// Verify relative tolerance dominates for large magnitude values
#[test]
fn test_tolerance_is_close_relative_tolerance() {
    let tol = Tolerance::default();
    // For large values, relative tolerance dominates
    // diff = 100, bound = 1e-5 + 1e-4 * 1_000_000 = 100.00001 → close
    assert!(tol.is_close(1_000_100.0, 1_000_000.0));
}

/// Verify only absolute tolerance applies when expected value is zero
#[test]
fn test_tolerance_is_close_zero_expected() {
    let tol = Tolerance::default();
    // When expected is 0, only atol matters
    assert!(tol.is_close(1e-6, 0.0));
    assert!(!tol.is_close(1e-4, 0.0));
}

// ============================================================
// TensorDiff Tests
// ============================================================

/// Verify ShapeMismatch display includes expected and actual sizes
#[test]
fn test_tensor_diff_display_shape_mismatch() {
    let diff = TensorDiff::ShapeMismatch {
        expected: 100,
        actual: 50,
    };
    let s = diff.to_string();
    assert!(s.contains("Shape mismatch"));
    assert!(s.contains("100"));
    assert!(s.contains("50"));
}

/// Verify ValueMismatch display includes count ratio and percentage
#[test]
fn test_tensor_diff_display_value_mismatch() {
    let diff = TensorDiff::ValueMismatch {
        num_mismatches: 10,
        total: 100,
        mismatch_ratio: 0.1,
        max_diff: 0.5,
        max_diff_idx: 42,
        expected_val: 1.0,
        actual_val: 1.5,
        mean_diff: 0.1,
    };
    let s = diff.to_string();
    assert!(s.contains("Value mismatch"));
    assert!(s.contains("10/100"));
    assert!(s.contains("10.00%"));
}

/// Verify ParseError display includes the error message
#[test]
fn test_tensor_diff_display_parse_error() {
    let diff = TensorDiff::ParseError {
        message: "file not found".to_string(),
    };
    let s = diff.to_string();
    assert!(s.contains("Parse error"));
    assert!(s.contains("file not found"));
}

// ============================================================
// HfParityOracle Construction Tests
// ============================================================

/// Verify oracle construction sets corpus path and model family
#[test]
fn test_oracle_new() {
    let oracle = HfParityOracle::new("/tmp/corpus", "llama-2-7b");
    assert_eq!(oracle.corpus_path(), Path::new("/tmp/corpus"));
    assert_eq!(oracle.model_family(), "llama-2-7b");
}

/// Verify oracle tolerance can be overridden via builder method
#[test]
fn test_oracle_with_tolerance() {
    let tol = Tolerance::int4();
    let oracle = HfParityOracle::new("/tmp", "test").with_tolerance(tol);
    assert!((oracle.tolerance().atol_fp32 - 0.5).abs() < 1e-10);
}

/// Verify oracle name returns the expected identifier
#[test]
fn test_oracle_name() {
    let oracle = HfParityOracle::new("/tmp", "test");
    assert_eq!(oracle.name(), "hf_parity");
}

// ============================================================
// Tensor Comparison Tests
// ============================================================

/// Verify identical tensor vectors pass closeness check
#[test]
fn test_tensors_close_identical() {
    let oracle = HfParityOracle::new("/tmp", "test");
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!(oracle.tensors_close(&a, &b).is_ok());
}

/// Verify tensors with sub-tolerance differences pass closeness check
#[test]
fn test_tensors_close_within_tolerance() {
    let oracle = HfParityOracle::new("/tmp", "test");
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![1.000001, 2.000001, 3.000001, 4.000001, 5.000001];
    assert!(oracle.tensors_close(&a, &b).is_ok());
}

/// Verify different-length tensors produce a ShapeMismatch error
#[test]
fn test_tensors_close_shape_mismatch() {
    let oracle = HfParityOracle::new("/tmp", "test");
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0];
    let result = oracle.tensors_close(&a, &b);
    assert!(matches!(result, Err(TensorDiff::ShapeMismatch { .. })));
}

/// Verify two empty tensors pass closeness check
#[test]
fn test_tensors_close_empty() {
    let oracle = HfParityOracle::new("/tmp", "test");
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    assert!(oracle.tensors_close(&a, &b).is_ok());
}

/// Verify tensors exceeding the mismatch ratio threshold produce ValueMismatch
#[test]
fn test_tensors_close_exceeds_mismatch_ratio() {
    let oracle = HfParityOracle::new("/tmp", "test");
    // 50% mismatch rate exceeds 1% threshold
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 2.0, 100.0, 100.0]; // 2/4 = 50% mismatch
    let result = oracle.tensors_close(&a, &b);
    assert!(matches!(result, Err(TensorDiff::ValueMismatch { .. })));
}

/// Verify tensors within the int4 mismatch ratio threshold pass
#[test]
fn test_tensors_close_within_mismatch_ratio() {
    // Use int4 tolerance which allows 10% mismatch (max_mismatch_ratio = 0.10)
    let oracle = HfParityOracle::new("/tmp", "test").with_tolerance(Tolerance::int4());
    // Create array with exactly 5% mismatch (within 10% threshold)
    let a: Vec<f32> = vec![1.0; 100];
    let mut b = a.clone();
    // Make 5 elements differ significantly (5% = 0.05 < 0.10 threshold)
    for i in 0..5 {
        b[i] = 100.0;
    }
    // With int4 tolerance, 5% mismatch ratio is WITHIN the 10% threshold
    // so this should pass (Ok)
    let result = oracle.tensors_close(&a, &b);
    assert!(result.is_ok(), "5% mismatch should be within 10% threshold");
}

// ============================================================
// Statistical Analysis Tests
// ============================================================

/// Verify divergence stats are zero for identical vectors
#[test]
fn test_compute_divergence_stats_identical() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0];
    let (max, mean, std) = HfParityOracle::compute_divergence_stats(&a, &b);
    assert!(max < 1e-10);
    assert!(mean < 1e-10);
    assert!(std < 1e-10);
}

/// Verify divergence stats compute correct max and mean for known diff
#[test]
fn test_compute_divergence_stats_with_diff() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 4.0]; // diff of 1.0 at index 2
    let (max, mean, _std) = HfParityOracle::compute_divergence_stats(&a, &b);
    assert!((max - 1.0).abs() < 1e-6);
    assert!((mean - 1.0 / 3.0).abs() < 1e-6);
}

/// Verify divergence stats return zeros for empty vectors
#[test]
fn test_compute_divergence_stats_empty() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let (max, mean, std) = HfParityOracle::compute_divergence_stats(&a, &b);
    assert!(max == 0.0);
    assert!(mean == 0.0);
    assert!(std == 0.0);
}

/// Verify divergence stats return zeros for mismatched-length vectors
#[test]
fn test_compute_divergence_stats_mismatched_len() {
    let a = vec![1.0, 2.0];
    let b = vec![1.0];
    let (max, mean, std) = HfParityOracle::compute_divergence_stats(&a, &b);
    assert!(max == 0.0);
    assert!(mean == 0.0);
    assert!(std == 0.0);
}

/// Verify no bias detected for identical vectors
#[test]
fn test_detect_systematic_bias_none() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert!(HfParityOracle::detect_systematic_bias(&a, &b).is_none());
}

/// Verify zero-std expected vector prevents sigma-based bias detection
#[test]
fn test_detect_systematic_bias_mean_shift() {
    let a = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    let b = vec![10.0, 10.0, 10.0, 10.0, 10.0]; // Large mean shift
    // With zero std in expected, we can't detect via sigma
    // This tests the edge case
    let result = HfParityOracle::detect_systematic_bias(&a, &b);
    // Zero std means no sigma-based detection
    assert!(result.is_none());
}

/// Verify mean shift bias is detected when variance exists
#[test]
fn test_detect_systematic_bias_with_variance() {
    let a = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let b = vec![10.0, 11.0, 12.0, 13.0, 14.0]; // Shift of 10
    let result = HfParityOracle::detect_systematic_bias(&a, &b);
    assert!(result.is_some());
    assert!(result.unwrap().contains("Mean shift"));
}

/// Verify scale drift bias is detected for 2x scaled vectors
#[test]
fn test_detect_systematic_bias_scale_drift() {
    let a = vec![0.0, 1.0, 2.0, 3.0, 4.0]; // std ≈ 1.41
    let b = vec![0.0, 2.0, 4.0, 6.0, 8.0]; // std ≈ 2.83 (2x scale)
    let result = HfParityOracle::detect_systematic_bias(&a, &b);
    assert!(result.is_some());
    assert!(result.unwrap().contains("Scale drift"));
}

/// Verify no bias detected for empty vectors
#[test]
fn test_detect_systematic_bias_empty() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    assert!(HfParityOracle::detect_systematic_bias(&a, &b).is_none());
}

// ============================================================
// Hash Function Tests
// ============================================================

/// Verify hash_prompt returns identical results for identical inputs
#[test]
fn test_hash_prompt_deterministic() {
    let h1 = hash_prompt("Hello, world!");
    let h2 = hash_prompt("Hello, world!");
    assert_eq!(h1, h2);
}

/// Verify hash_prompt produces different hashes for different inputs
#[test]
fn test_hash_prompt_different_inputs() {
    let h1 = hash_prompt("Hello");
    let h2 = hash_prompt("World");
    assert_ne!(h1, h2);
}

/// Verify hash_prompt output is 16 hex characters
#[test]
fn test_hash_prompt_format() {
    let h = hash_prompt("test");
    assert_eq!(h.len(), 16); // 16 hex chars = 64 bits
    assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
}

/// Verify hash_prompt handles empty string input
#[test]
fn test_hash_prompt_empty() {
    let h = hash_prompt("");
    assert_eq!(h.len(), 16);
}

/// Verify hash_prompt handles unicode input deterministically
#[test]
fn test_hash_prompt_unicode() {
    let h1 = hash_prompt("こんにちは");
    let h2 = hash_prompt("こんにちは");
    assert_eq!(h1, h2);
    assert_eq!(h1.len(), 16);
}

/// Verify hash_prompt matches Python generate_golden.py SHA-256 truncated values
#[test]
fn test_hash_prompt_cross_language_compatibility() {
    // These hashes are from the Python generate_golden.py script
    // Using SHA-256 truncated to 16 hex chars for cross-language consistency
    assert_eq!(hash_prompt("def fibonacci(n):"), "c839979da8b41875");
    assert_eq!(hash_prompt("2 + 2 ="), "154e0c9c61763891");
    assert_eq!(hash_prompt("fn main() {"), "72879bbc234f8df8");
    assert_eq!(hash_prompt("x"), "2d711642b726b044");
    assert_eq!(hash_prompt("1"), "6b86b273ff34fce1");
}

// ============================================================
// Truncate Tests
// ============================================================

/// Verify truncate preserves strings shorter than the limit
#[test]
fn test_truncate_short_string() {
    assert_eq!(truncate("hello", 10), "hello");
}

/// Verify truncate preserves strings at exactly the limit
#[test]
fn test_truncate_exact_length() {
    assert_eq!(truncate("hello", 5), "hello");
}

/// Verify truncate clips strings exceeding the byte limit
#[test]
fn test_truncate_long_string() {
    assert_eq!(truncate("hello world", 5), "hello");
}

/// Verify truncate handles empty strings
#[test]
fn test_truncate_empty() {
    assert_eq!(truncate("", 10), "");
}

/// Verify truncate respects multi-byte unicode character boundaries
#[test]
fn test_truncate_unicode_boundary() {
    // "こんにちは" is 15 bytes (3 bytes per char)
    let s = "こんにちは";
    let truncated = truncate(s, 6); // Should truncate to 2 chars (6 bytes)
    assert_eq!(truncated, "こん");
}

// ============================================================
// Oracle Evaluate Tests (without filesystem)
// ============================================================

/// Verify oracle returns corroborated when no golden output exists
#[test]
fn test_oracle_evaluate_no_golden() {
    let oracle = HfParityOracle::new("/nonexistent/path", "test");
    let result = oracle.evaluate("test prompt", "test output");
    assert!(result.is_corroborated());
    if let OracleResult::Corroborated { evidence } = result {
        assert!(evidence.contains("No golden output"));
    }
}

/// Verify oracle returns corroborated for plain text without tensors
#[test]
fn test_oracle_evaluate_text_no_tensor() {
    let oracle = HfParityOracle::new("/nonexistent", "test");
    let result = oracle.evaluate("prompt", "plain text output");
    assert!(result.is_corroborated());
}

// ============================================================
// Golden Output Tests
// ============================================================

/// Verify GoldenOutput serializes to JSON with expected fields
#[test]
fn test_golden_output_serialization() {
    let golden = GoldenOutput {
        input_hash: "abc123".to_string(),
        prompt: "test prompt".to_string(),
        logits: vec![1.0, 2.0, 3.0],
        shape: vec![1, 3],
        text: Some("generated".to_string()),
        model_id: "test-model".to_string(),
        transformers_version: "4.38.0".to_string(),
    };
    let json = serde_json::to_string(&golden).expect("serialize");
    assert!(json.contains("abc123"));
    assert!(json.contains("test prompt"));
}

/// Verify GoldenOutput deserializes from JSON with nullable text field
#[test]
fn test_golden_output_deserialization() {
    let json = r#"{
            "input_hash": "abc123",
            "prompt": "test",
            "logits": [1.0, 2.0],
            "shape": [1, 2],
            "text": null,
            "model_id": "model",
            "transformers_version": "4.38.0"
        }"#;
    let golden: GoldenOutput = serde_json::from_str(json).expect("deserialize");
    assert_eq!(golden.input_hash, "abc123");
    assert_eq!(golden.logits.len(), 2);
    assert!(golden.text.is_none());
}

// ============================================================
// Mutation-Killing Tests
// ============================================================
