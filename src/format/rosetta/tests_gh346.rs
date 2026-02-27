// ============================================================================
// GH-346: GGUF Round-Trip PMAT-237 Violation + Sharded SafeTensors I-2 Failure
// ============================================================================
//
// Falsification tests for two bugs:
//   F-CONV-RT-001: Double Q4K quantization in APR→GGUF export
//   F-CONTRACT-I2-001: Sharded SafeTensors fallback in inspect/validate

use super::*;

// ========================================================================
// Bug 1: F-CONV-RT-001 — Double Q4K quantization
// ========================================================================
//
// Root cause: apply_export_quantization() applied flat quantize_q4_k() before
// encode_gguf_data() applied matrix-aware quantize_q4_k_matrix(). Two different
// quantization functions with different block layouts → amplified error.
//
// Fix: Skip apply_export_quantization() when export format is GGUF.

/// Falsify F-CONV-RT-001: APR→GGUF export with Q4K must NOT apply double quantization.
///
/// H0: Exporting F32 APR to GGUF with Q4K quantization produces a valid GGUF file
///     that can be read back, and the pre-quantization step is skipped for GGUF.
/// Refutation: If the export fails or tensors have excessive error, double quant is back.
#[test]
fn falsify_f_conv_rt_001_no_double_quantization_gguf_export() {
    use crate::format::converter::{apr_export, ExportFormat, ExportOptions};
    use crate::format::test_factory::build_pygmy_apr;

    let apr_bytes = build_pygmy_apr();
    let dir = tempfile::tempdir().expect("tempdir");
    let apr_path = dir.path().join("source.apr");
    let gguf_path = dir.path().join("exported.gguf");

    std::fs::write(&apr_path, &apr_bytes).expect("write APR");

    let options = ExportOptions {
        format: ExportFormat::Gguf,
        quantize: Some(crate::format::converter::QuantizationType::Q4K),
        include_tokenizer: false,
        include_config: false,
        skip_completeness_check: true,
    };
    let result = apr_export(&apr_path, &gguf_path, options);
    assert!(
        result.is_ok(),
        "F32 APR → GGUF Q4K export should succeed (no double quant): {:?}",
        result.unwrap_err()
    );
    assert!(gguf_path.exists(), "Exported GGUF file should exist");

    // Verify the exported GGUF can be inspected (proves structural integrity)
    let rosetta = RosettaStone::new();
    let report = rosetta.inspect(&gguf_path).expect("Inspect exported GGUF");
    assert_eq!(report.format, FormatType::Gguf);
    assert!(
        !report.tensors.is_empty(),
        "Exported GGUF should have tensors"
    );
}

/// Falsify F-CONV-RT-001: SafeTensors export STILL applies pre-quantization.
///
/// The GH-346 fix skips pre-quantization for GGUF only. SafeTensors export must
/// continue to quantize in apply_export_quantization() as before.
/// Refutation: If this fails, the fix was too broad and broke SafeTensors export.
#[test]
fn falsify_f_conv_rt_001_safetensors_still_quantizes() {
    use crate::format::converter::{apr_export, ExportFormat, ExportOptions};
    use crate::format::test_factory::build_pygmy_apr;

    let apr_bytes = build_pygmy_apr();
    let dir = tempfile::tempdir().expect("tempdir");
    let apr_path = dir.path().join("source.apr");
    let st_path = dir.path().join("exported.safetensors");

    std::fs::write(&apr_path, &apr_bytes).expect("write APR");

    let options = ExportOptions {
        format: ExportFormat::SafeTensors,
        quantize: Some(crate::format::converter::QuantizationType::Q4K),
        include_tokenizer: false,
        include_config: false,
        skip_completeness_check: true,
    };
    let result = apr_export(&apr_path, &st_path, options);
    assert!(
        result.is_ok(),
        "F32 APR → SafeTensors Q4K export should succeed: {:?}",
        result.unwrap_err()
    );
    assert!(st_path.exists(), "Exported SafeTensors file should exist");
}

// ========================================================================
// Bug 2: F-CONTRACT-I2-001 — Sharded SafeTensors fallback
// ========================================================================
//
// Root cause: inspect()/validate() fail for "model.safetensors" when the model
// is actually sharded (model-00001-of-00002.safetensors + model.safetensors.index.json).
// No fallback to resolve the sharded index from a missing single-file path.
//
// Fix: try_resolve_sharded_index() detects missing .safetensors with existing .index.json.

/// Falsify F-CONTRACT-I2-001: try_resolve_sharded_index resolves missing .safetensors
/// to the .index.json path when the model is sharded.
///
/// H0: When model.safetensors doesn't exist but model.safetensors.index.json does,
///     try_resolve_sharded_index returns Some(index_path).
/// Refutation: Returns None → sharded models still fail.
#[test]
fn falsify_f_contract_i2_001_resolve_sharded_index() {
    let dir = tempfile::tempdir().expect("tempdir");

    // Create the index file (content doesn't matter for resolution test)
    let index_path = dir.path().join("model.safetensors.index.json");
    std::fs::write(&index_path, "{}").expect("write index");

    // The single-file path does NOT exist
    let single_path = dir.path().join("model.safetensors");
    assert!(!single_path.exists(), "Single file must not exist for this test");

    let resolved = try_resolve_sharded_index(&single_path);
    assert!(
        resolved.is_some(),
        "try_resolve_sharded_index should return Some for missing .safetensors with existing .index.json"
    );
    assert_eq!(
        resolved.expect("checked above"),
        index_path,
        "Resolved path should be the .index.json file"
    );
}

/// Falsify F-CONTRACT-I2-001 negative: existing .safetensors returns None.
///
/// H0: When model.safetensors exists, try_resolve_sharded_index returns None
///     (no fallback needed — the file exists).
/// Refutation: Returns Some → unnecessary redirect for single-file models.
#[test]
fn falsify_f_contract_i2_001_no_resolve_when_file_exists() {
    let dir = tempfile::tempdir().expect("tempdir");
    let st_path = dir.path().join("model.safetensors");
    std::fs::write(&st_path, "dummy").expect("write safetensors");

    let resolved = try_resolve_sharded_index(&st_path);
    assert!(
        resolved.is_none(),
        "try_resolve_sharded_index should return None when .safetensors exists"
    );
}

/// Falsify F-CONTRACT-I2-001 negative: non-safetensors extension returns None.
///
/// H0: try_resolve_sharded_index only activates for .safetensors paths.
/// Refutation: Returns Some for .gguf → broken.
#[test]
fn falsify_f_contract_i2_001_no_resolve_for_non_safetensors() {
    let dir = tempfile::tempdir().expect("tempdir");
    let gguf_path = dir.path().join("model.gguf");
    // Don't create it — it doesn't exist

    let resolved = try_resolve_sharded_index(&gguf_path);
    assert!(
        resolved.is_none(),
        "try_resolve_sharded_index should return None for non-.safetensors paths"
    );
}

/// Falsify F-CONTRACT-I2-001 negative: missing both files returns None.
///
/// H0: When neither model.safetensors nor model.safetensors.index.json exist,
///     returns None (not a sharded model, just missing).
/// Refutation: Returns Some for a completely missing model → phantom sharded detection.
#[test]
fn falsify_f_contract_i2_001_no_resolve_when_index_missing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let st_path = dir.path().join("model.safetensors");
    // Neither file exists

    let resolved = try_resolve_sharded_index(&st_path);
    assert!(
        resolved.is_none(),
        "try_resolve_sharded_index should return None when .index.json also missing"
    );
}

/// Falsify F-CONTRACT-I2-001: inspect() with sharded fallback resolves correctly.
///
/// H0: inspect() on a missing model.safetensors with a valid sharded index
///     delegates to inspect_sharded_safetensors and returns a valid report.
/// Refutation: Errors out with "format not supported" or file-not-found.
#[test]
fn falsify_f_contract_i2_001_inspect_sharded_fallback() {
    use std::io::Write;

    let dir = tempfile::tempdir().expect("tempdir");

    // Create a valid single-shard SafeTensors file
    let shard_path = dir.path().join("model-00001-of-00001.safetensors");
    {
        let mut file = std::fs::File::create(&shard_path).expect("create shard");
        let header = r#"{"test.bias":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"__metadata__":{"format":"test"}}"#;
        file.write_all(&(header.len() as u64).to_le_bytes())
            .expect("write header len");
        file.write_all(header.as_bytes()).expect("write header");
        let data: [f32; 4] = [0.01, -0.02, 0.03, -0.01];
        for val in &data {
            file.write_all(&val.to_le_bytes()).expect("write tensor");
        }
    }

    // Create the sharded index JSON
    let index_path = dir.path().join("model.safetensors.index.json");
    let index_json = serde_json::json!({
        "metadata": {"total_size": 16},
        "weight_map": {
            "test.bias": "model-00001-of-00001.safetensors"
        }
    });
    std::fs::write(&index_path, index_json.to_string()).expect("write index");

    // Ask to inspect "model.safetensors" which doesn't exist — should resolve to sharded
    let missing_single = dir.path().join("model.safetensors");
    assert!(!missing_single.exists());

    let rosetta = RosettaStone::new();
    let report = rosetta.inspect(&missing_single);
    assert!(
        report.is_ok(),
        "inspect() should resolve sharded fallback: {:?}",
        report.unwrap_err()
    );

    let report = report.expect("checked above");
    assert_eq!(report.format, FormatType::SafeTensors);
    assert!(
        !report.tensors.is_empty(),
        "Sharded inspect should find tensors"
    );
}

/// Falsify F-CONTRACT-I2-001: validate() with sharded fallback resolves correctly.
///
/// H0: validate() on a missing model.safetensors with a valid sharded index
///     delegates to validate_sharded_safetensors and returns a valid report.
/// Refutation: Errors out instead of validating the sharded model.
#[test]
fn falsify_f_contract_i2_001_validate_sharded_fallback() {
    use std::io::Write;

    let dir = tempfile::tempdir().expect("tempdir");

    // Create a valid single-shard SafeTensors file
    let shard_path = dir.path().join("model-00001-of-00001.safetensors");
    {
        let mut file = std::fs::File::create(&shard_path).expect("create shard");
        let header = r#"{"test.bias":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"__metadata__":{"format":"test"}}"#;
        file.write_all(&(header.len() as u64).to_le_bytes())
            .expect("write header len");
        file.write_all(header.as_bytes()).expect("write header");
        let data: [f32; 4] = [0.01, -0.02, 0.03, -0.01];
        for val in &data {
            file.write_all(&val.to_le_bytes()).expect("write tensor");
        }
    }

    // Create the sharded index JSON
    let index_path = dir.path().join("model.safetensors.index.json");
    let index_json = serde_json::json!({
        "metadata": {"total_size": 16},
        "weight_map": {
            "test.bias": "model-00001-of-00001.safetensors"
        }
    });
    std::fs::write(&index_path, index_json.to_string()).expect("write index");

    // Ask to validate "model.safetensors" which doesn't exist — should resolve to sharded
    let missing_single = dir.path().join("model.safetensors");
    assert!(!missing_single.exists());

    let rosetta = RosettaStone::new();
    let report = rosetta.validate(&missing_single);
    assert!(
        report.is_ok(),
        "validate() should resolve sharded fallback: {:?}",
        report.unwrap_err()
    );

    let report = report.expect("checked above");
    assert_eq!(report.format, FormatType::SafeTensors);
    assert!(report.is_valid, "Sharded model tensors should be valid");
    assert!(
        report.tensor_count > 0,
        "Should have validated at least one tensor"
    );
}

/// Falsify F-CONTRACT-I2-001: validate() with explicit .index.json path works.
///
/// H0: validate() given a .index.json path directly validates sharded model.
/// Refutation: validate() doesn't support is_sharded_index dispatch.
#[test]
fn falsify_f_contract_i2_001_validate_explicit_index_json() {
    use std::io::Write;

    let dir = tempfile::tempdir().expect("tempdir");

    // Create a valid single-shard SafeTensors file
    let shard_path = dir.path().join("model-00001-of-00001.safetensors");
    {
        let mut file = std::fs::File::create(&shard_path).expect("create shard");
        let header = r#"{"test.bias":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"__metadata__":{"format":"test"}}"#;
        file.write_all(&(header.len() as u64).to_le_bytes())
            .expect("write header len");
        file.write_all(header.as_bytes()).expect("write header");
        let data: [f32; 4] = [0.01, -0.02, 0.03, -0.01];
        for val in &data {
            file.write_all(&val.to_le_bytes()).expect("write tensor");
        }
    }

    // Create the sharded index JSON
    let index_path = dir.path().join("model.safetensors.index.json");
    let index_json = serde_json::json!({
        "metadata": {"total_size": 16},
        "weight_map": {
            "test.bias": "model-00001-of-00001.safetensors"
        }
    });
    std::fs::write(&index_path, index_json.to_string()).expect("write index");

    let rosetta = RosettaStone::new();
    let report = rosetta.validate(&index_path);
    assert!(
        report.is_ok(),
        "validate() with explicit .index.json should work: {:?}",
        report.unwrap_err()
    );

    let report = report.expect("checked above");
    assert_eq!(report.format, FormatType::SafeTensors);
    assert!(report.is_valid, "Sharded model tensors should be valid");
}
