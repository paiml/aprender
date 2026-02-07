//! GH-218: Tests for sharded SafeTensors import
//!
//! Tests directory resolution and multi-shard tensor merging.

use crate::format::converter::import;
use crate::format::converter_types::{ImportOptions, Source};
use std::io::Write;

/// Create a minimal SafeTensors file with given tensor entries.
/// Each entry is (name, f32_values).
fn create_shard_file(path: &std::path::Path, tensors: &[(&str, &[f32])]) {
    let mut file = std::fs::File::create(path).expect("create shard file");

    // Build header JSON
    let mut header_parts = Vec::new();
    let mut offset = 0u64;
    for (name, values) in tensors {
        let byte_len = values.len() * 4;
        header_parts.push(format!(
            r#""{name}":{{"dtype":"F32","shape":[{len}],"data_offsets":[{start},{end}]}}"#,
            len = values.len(),
            start = offset,
            end = offset + byte_len as u64,
        ));
        offset += byte_len as u64;
    }
    let header = format!("{{{}}}", header_parts.join(","));

    // Write header length (8 bytes LE) + header + tensor data
    file.write_all(&(header.len() as u64).to_le_bytes())
        .expect("write header len");
    file.write_all(header.as_bytes()).expect("write header");
    for (_name, values) in tensors {
        for val in *values {
            file.write_all(&val.to_le_bytes())
                .expect("write tensor data");
        }
    }
}

/// Create a model.safetensors.index.json with given weight_map entries.
fn create_index_json(path: &std::path::Path, entries: &[(&str, &str)]) {
    let weight_map_entries: Vec<String> = entries
        .iter()
        .map(|(tensor, shard)| format!(r#""{tensor}": "{shard}""#))
        .collect();
    let json = format!(
        r#"{{"metadata": {{"total_size": 0}}, "weight_map": {{{}}}}}"#,
        weight_map_entries.join(", ")
    );
    std::fs::write(path, json).expect("write index json");
}

// ============================================================================
// resolve_source: directory handling
// ============================================================================

#[test]
fn test_resolve_source_directory_with_index() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let index_path = dir.path().join("model.safetensors.index.json");
    std::fs::write(&index_path, r#"{"weight_map": {}}"#).expect("write index");

    let source = Source::Local(dir.path().to_path_buf());
    let result = import::resolve_source(&source, false);
    assert!(result.is_ok(), "Should resolve directory with index.json");
    assert_eq!(result.expect("resolve"), index_path);
}

#[test]
fn test_resolve_source_directory_with_single_safetensors() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let st_path = dir.path().join("model.safetensors");
    create_shard_file(&st_path, &[("test.bias", &[0.1, -0.2])]);

    let source = Source::Local(dir.path().to_path_buf());
    let result = import::resolve_source(&source, false);
    assert!(
        result.is_ok(),
        "Should resolve directory with model.safetensors"
    );
    assert_eq!(result.expect("resolve"), st_path);
}

#[test]
fn test_resolve_source_directory_prefers_index_over_single() {
    let dir = tempfile::tempdir().expect("create tempdir");
    // Create both files
    let index_path = dir.path().join("model.safetensors.index.json");
    std::fs::write(&index_path, r#"{"weight_map": {}}"#).expect("write index");
    let st_path = dir.path().join("model.safetensors");
    create_shard_file(&st_path, &[("test.bias", &[0.1])]);

    let source = Source::Local(dir.path().to_path_buf());
    let result = import::resolve_source(&source, false).expect("resolve");
    // Should prefer index.json over single file
    assert_eq!(result, index_path);
}

#[test]
fn test_resolve_source_directory_empty_error() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let source = Source::Local(dir.path().to_path_buf());
    let result = import::resolve_source(&source, false);
    assert!(result.is_err(), "Empty directory should error");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("no model.safetensors.index.json") || err.contains("model.safetensors"),
        "Error should mention expected files, got: {err}"
    );
}

// ============================================================================
// load_sharded_safetensors: merge behavior
// ============================================================================

#[test]
fn test_load_sharded_safetensors_basic() {
    let dir = tempfile::tempdir().expect("create tempdir");

    // Create two shard files
    let shard1_path = dir.path().join("model-00001-of-00002.safetensors");
    create_shard_file(&shard1_path, &[("layer.0.weight", &[0.1, 0.2, 0.3, 0.4])]);

    let shard2_path = dir.path().join("model-00002-of-00002.safetensors");
    create_shard_file(&shard2_path, &[("layer.1.weight", &[0.5, 0.6, 0.7, 0.8])]);

    // Create index.json
    let index_path = dir.path().join("model.safetensors.index.json");
    create_index_json(
        &index_path,
        &[
            ("layer.0.weight", "model-00001-of-00002.safetensors"),
            ("layer.1.weight", "model-00002-of-00002.safetensors"),
        ],
    );

    let options = ImportOptions::default();
    let result = import::load_sharded_safetensors(&index_path, &options);
    assert!(result.is_ok(), "Sharded load should succeed: {result:?}");

    let load_result = result.expect("load result");
    assert_eq!(load_result.tensors.len(), 2, "Should have 2 tensors merged");
    assert!(
        load_result.tensors.contains_key("layer.0.weight"),
        "Should contain layer.0.weight"
    );
    assert!(
        load_result.tensors.contains_key("layer.1.weight"),
        "Should contain layer.1.weight"
    );

    // Verify tensor data is correct
    let (data0, shape0) = &load_result.tensors["layer.0.weight"];
    assert_eq!(shape0, &[4]);
    assert!((data0[0] - 0.1).abs() < 1e-6);

    let (data1, shape1) = &load_result.tensors["layer.1.weight"];
    assert_eq!(shape1, &[4]);
    assert!((data1[0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_load_sharded_safetensors_multiple_tensors_per_shard() {
    let dir = tempfile::tempdir().expect("create tempdir");

    // Single shard with two tensors
    let shard_path = dir.path().join("model-00001-of-00001.safetensors");
    create_shard_file(
        &shard_path,
        &[
            ("embed.weight", &[1.0, 2.0]),
            ("lm_head.weight", &[3.0, 4.0]),
        ],
    );

    let index_path = dir.path().join("model.safetensors.index.json");
    create_index_json(
        &index_path,
        &[
            ("embed.weight", "model-00001-of-00001.safetensors"),
            ("lm_head.weight", "model-00001-of-00001.safetensors"),
        ],
    );

    let options = ImportOptions::default();
    let result = import::load_sharded_safetensors(&index_path, &options).expect("load sharded");
    assert_eq!(result.tensors.len(), 2);
    assert!(result.tensors.contains_key("embed.weight"));
    assert!(result.tensors.contains_key("lm_head.weight"));
}

#[test]
fn test_load_sharded_missing_shard_error() {
    let dir = tempfile::tempdir().expect("create tempdir");

    // Create index referencing a shard that doesn't exist
    let index_path = dir.path().join("model.safetensors.index.json");
    create_index_json(
        &index_path,
        &[("layer.0.weight", "model-00001-of-00001.safetensors")],
    );

    let options = ImportOptions::default();
    let result = import::load_sharded_safetensors(&index_path, &options);
    assert!(result.is_err(), "Missing shard should error");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("not found"),
        "Error should mention shard not found, got: {err}"
    );
}

#[test]
fn test_load_sharded_empty_index_error() {
    let dir = tempfile::tempdir().expect("create tempdir");

    // Create index with no weight_map entries (empty)
    let index_path = dir.path().join("model.safetensors.index.json");
    std::fs::write(&index_path, r#"{"weight_map": {}}"#).expect("write index");

    let options = ImportOptions::default();
    let result = import::load_sharded_safetensors(&index_path, &options);
    assert!(result.is_err(), "Empty index should error");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("no shard files"),
        "Error should mention empty shards, got: {err}"
    );
}

// ============================================================================
// load_source_tensors: .index.json routing
// ============================================================================

#[test]
fn test_load_source_tensors_routes_index_json() {
    let dir = tempfile::tempdir().expect("create tempdir");

    let shard_path = dir.path().join("model-00001-of-00001.safetensors");
    create_shard_file(&shard_path, &[("test.bias", &[0.1, 0.2, 0.3])]);

    let index_path = dir.path().join("model.safetensors.index.json");
    create_index_json(
        &index_path,
        &[("test.bias", "model-00001-of-00001.safetensors")],
    );

    let options = ImportOptions::default();
    let result = import::load_source_tensors(&index_path, &options);
    assert!(
        result.is_ok(),
        "load_source_tensors should route .index.json: {result:?}"
    );
    let load_result = result.expect("load result");
    assert_eq!(load_result.tensors.len(), 1);
    assert!(load_result.tensors.contains_key("test.bias"));
}

#[test]
fn test_load_source_tensors_regular_json_still_errors() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let json_path = dir.path().join("config.json");
    std::fs::write(&json_path, r#"{"key": "value"}"#).expect("write json");

    let options = ImportOptions::default();
    let result = import::load_source_tensors(&json_path, &options);
    assert!(
        result.is_err(),
        "Non-index .json should still error: {result:?}"
    );
}
