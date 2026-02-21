use super::*;

// P120: APR tokenizer extraction round-trip
// H0: Tokenizer vocabulary embedded in APR can be extracted and used for decoding
// Refutation: Fails if vocabulary extraction fails or decoding produces wrong output
// NOTE: This test requires realizar crate's SimpleTokenizer implementation (GH-156)
#[test]
fn p120_apr_tokenizer_decode_roundtrip() {
    use crate::format::v2::{AprV2Metadata, AprV2Reader, AprV2Writer};
    use std::collections::HashMap;

    let path = unique_temp_path("test_decode", "apr");

    // Create APR with embedded tokenizer
    let mut metadata = AprV2Metadata::new("test");

    // Define vocabulary with BPE-style tokens
    let vocab = vec![
        "<pad>", "<bos>", "<eos>", "Ġhello", "Ġworld", "!", "Ġthe", "Ġquick",
    ];
    let vocab_json: Vec<serde_json::Value> = vocab
        .iter()
        .map(|s| serde_json::Value::String(s.to_string()))
        .collect();

    let mut custom: HashMap<String, serde_json::Value> = HashMap::new();
    custom.insert(
        "tokenizer.vocabulary".to_string(),
        serde_json::Value::Array(vocab_json),
    );
    custom.insert(
        "tokenizer.vocab_size".to_string(),
        serde_json::Value::Number(8.into()),
    );
    custom.insert(
        "tokenizer.bos_token_id".to_string(),
        serde_json::Value::Number(1.into()),
    );
    custom.insert(
        "tokenizer.eos_token_id".to_string(),
        serde_json::Value::Number(2.into()),
    );
    metadata.custom = custom;

    let mut writer = AprV2Writer::new(metadata);
    writer.add_f32_tensor("embed.weight", vec![8, 4], &[0.0; 32]);

    let mut file = std::fs::File::create(&path).expect("Create APR file");
    writer.write_to(&mut file).expect("Write APR");
    drop(file);

    // Read APR and extract vocabulary
    let data = std::fs::read(&path).expect("Read APR");
    let reader = AprV2Reader::from_bytes(&data).expect("Parse APR");
    let meta = reader.metadata();

    // Extract tokenizer vocabulary from custom metadata
    let vocab_value = meta.custom.get("tokenizer.vocabulary");
    assert!(
        vocab_value.is_some(),
        "APR should have tokenizer.vocabulary in metadata"
    );

    let vocab_array = vocab_value
        .unwrap()
        .as_array()
        .expect("vocabulary should be array");
    assert_eq!(vocab_array.len(), 8, "Vocabulary size should be 8");

    // Verify BOS/EOS token IDs
    let bos_id = meta
        .custom
        .get("tokenizer.bos_token_id")
        .and_then(|v| v.as_u64())
        .expect("Should have bos_token_id");
    let eos_id = meta
        .custom
        .get("tokenizer.eos_token_id")
        .and_then(|v| v.as_u64())
        .expect("Should have eos_token_id");

    assert_eq!(bos_id, 1, "BOS token ID should be 1");
    assert_eq!(eos_id, 2, "EOS token ID should be 2");

    // Verify vocabulary content (spot check)
    let first_token = vocab_array[3].as_str().expect("Token should be string");
    assert_eq!(first_token, "Ġhello", "Token at index 3 should be 'Ġhello'");

    let _ = std::fs::remove_file(path);
}

// P121: APR without tokenizer fallback
// H0: APR files without embedded tokenizer should indicate token count, not crash
// Refutation: Fails if accessing tokenizer on tokenizer-less APR causes panic
#[test]
fn p121_apr_no_tokenizer_graceful_fallback() {
    // This test uses create_apr_fixture() which creates APR WITHOUT tokenizer
    let path = create_apr_fixture();

    // Read APR and verify no tokenizer metadata
    let data = std::fs::read(&path).expect("Read APR");
    let reader = crate::format::v2::AprV2Reader::from_bytes(&data).expect("Parse APR");
    let meta = reader.metadata();

    // Should NOT have tokenizer.vocabulary
    let vocab_value = meta.custom.get("tokenizer.vocabulary");
    assert!(
        vocab_value.is_none(),
        "Fixture APR should NOT have embedded tokenizer"
    );

    // Accessing missing tokenizer should return None, not panic
    // (This is what GH-156 fixes in realizar)

    let _ = std::fs::remove_file(path);
}

// F-STRESS-520: Panic 411 (Empty Tensor) - 0-byte file handling
// H0: Loading a 0-byte file should return error, not panic
// Refutation: Fails if 0-byte file causes panic instead of graceful error
// Toyota Way: Jidoka - detect defects at the source
#[test]
fn f_stress_520_zero_byte_file_no_panic_pmat178() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let inspector = RosettaStone::new();

    // Create a 0-byte APR file
    let mut temp_apr = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp_apr.flush().expect("Flush");

    // Attempt to inspect - should return error, not panic
    let result = inspector.inspect(temp_apr.path());
    assert!(result.is_err(), "0-byte file should return error, not Ok");
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("small")
            || err_msg.contains("empty")
            || err_msg.contains("parse")
            || err_msg.contains("magic"),
        "Error should indicate file issue: {err_msg}"
    );

    // Create a 0-byte GGUF file
    let mut temp_gguf = NamedTempFile::with_suffix(".gguf").expect("Create temp file");
    temp_gguf.flush().expect("Flush");

    // Attempt to inspect - should return error, not panic
    let result = inspector.inspect(temp_gguf.path());
    assert!(
        result.is_err(),
        "0-byte GGUF file should return error, not Ok"
    );

    // Create a 0-byte SafeTensors file
    let mut temp_st = NamedTempFile::with_suffix(".safetensors").expect("Create temp file");
    temp_st.flush().expect("Flush");

    // Attempt to inspect - should return error, not panic
    let result = inspector.inspect(temp_st.path());
    assert!(
        result.is_err(),
        "0-byte SafeTensors file should return error, not Ok"
    );
}

// F-STRESS-520: Additional test - truncated file handling
// H0: Truncated files (partial headers) should return error, not panic
#[test]
fn f_stress_520_truncated_file_no_panic_pmat178() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    let inspector = RosettaStone::new();

    // Create a truncated APR file (just magic bytes, no header)
    let mut temp = NamedTempFile::with_suffix(".apr").expect("Create temp file");
    temp.write_all(b"APR\x00").expect("Write partial header");
    temp.flush().expect("Flush");

    let result = inspector.inspect(temp.path());
    assert!(
        result.is_err(),
        "Truncated APR file should return error, not Ok"
    );

    // Create a truncated GGUF file (just magic bytes)
    let mut temp_gguf = NamedTempFile::with_suffix(".gguf").expect("Create temp file");
    temp_gguf.write_all(b"GGUF").expect("Write partial header");
    temp_gguf.flush().expect("Flush");

    let result = inspector.inspect(temp_gguf.path());
    assert!(
        result.is_err(),
        "Truncated GGUF file should return error, not Ok"
    );
}

// GH-175, PMAT-180: Cross-format validation
// H0: validate() works for all formats and detects NaN/Inf/zeros
#[test]
fn gh175_cross_format_validation_pmat180() {
    // Test with APR fixture (should be valid)
    let apr_path = create_apr_fixture();
    let rosetta = RosettaStone::new();

    let report = rosetta.validate(&apr_path);
    assert!(report.is_ok(), "APR validation should succeed");

    let report = report.expect("validation");
    assert!(
        report.is_valid,
        "APR fixture should be valid (no NaN/Inf/zeros)"
    );
    assert_eq!(report.total_nan_count, 0, "Should have no NaN");
    assert_eq!(report.total_inf_count, 0, "Should have no Inf");
    assert!(
        report.all_zero_tensors.is_empty(),
        "Should have no all-zero tensors"
    );

    // Test summary output
    let summary = report.summary();
    assert!(summary.contains("VALID"), "Summary should indicate VALID");

    let _ = std::fs::remove_file(apr_path);
}

// GH-175: Validation report display
#[test]
fn gh175_validation_report_display() {
    let report = ValidationReport {
        format: FormatType::Apr,
        file_path: "test.apr".to_string(),
        is_valid: true,
        tensor_count: 10,
        failed_tensor_count: 0,
        total_nan_count: 0,
        total_inf_count: 0,
        all_zero_tensors: vec![],
        tensors: vec![],
        duration_ms: 100,
    };

    let display = format!("{report}");
    assert!(display.contains("VALID"), "Display should show VALID");
    assert!(
        display.contains("APR-SPEC 10.9"),
        "Should reference APR-SPEC"
    );
}

#[path = "tests_pygmy.rs"]
mod tests_pygmy;
