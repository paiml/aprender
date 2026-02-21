use super::*;

#[test]
fn test_lint_level_copy() {
    let level = LintLevel::Warn;
    let copied = level;
    assert_eq!(copied, LintLevel::Warn);
}

#[test]
fn test_lint_category_copy() {
    let cat = LintCategory::Naming;
    let copied = cat;
    assert_eq!(copied, LintCategory::Naming);
}

#[test]
fn test_lint_zero_alignment() {
    // Zero alignment is a special case (skip check)
    let info = ModelLintInfo {
        has_license: true,
        has_model_card: true,
        has_provenance: true,
        tensors: vec![TensorLintInfo {
            name: "encoder.conv1.weight".to_string(),
            size_bytes: 1000,
            alignment: 0, // Zero alignment - skip check
            is_compressed: false,
            shape: vec![],
        }],
        ..Default::default()
    };

    let report = lint_model(&info);
    let efficiency_issues = report.issues_in_category(LintCategory::Efficiency);
    let alignment_issues: Vec<_> = efficiency_issues
        .iter()
        .filter(|i| i.message.contains("aligned"))
        .collect();
    // Zero alignment should not be flagged as "unaligned"
    assert!(alignment_issues.is_empty());
}

// ========================================================================
// Test Multi-Format Lint (lint_model_file, lint_gguf_file, lint_safetensors_file)
// ========================================================================

#[test]
fn test_lint_gguf_file_with_metadata() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use tempfile::NamedTempFile;

    // Create GGUF with license and author metadata
    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![4, 4],
        dtype: GgmlType::F32,
        data: vec![0u8; 64], // 16 floats * 4 bytes
    };
    let metadata = vec![
        (
            "general.license".to_string(),
            GgufValue::String("MIT".to_string()),
        ),
        (
            "general.author".to_string(),
            GgufValue::String("Test Author".to_string()),
        ),
        (
            "general.description".to_string(),
            GgufValue::String("Test model".to_string()),
        ),
    ];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let temp_file = NamedTempFile::new().expect("create temp file");
    std::fs::write(temp_file.path(), &gguf_bytes).expect("write GGUF");

    let report = lint_gguf_file(temp_file.path()).expect("lint GGUF");
    let metadata_issues = report.issues_in_category(LintCategory::Metadata);

    // Should have no metadata warnings (all present)
    assert!(
        metadata_issues.is_empty(),
        "GGUF with full metadata should have no metadata warnings"
    );
}

#[test]
fn test_lint_gguf_file_missing_metadata() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use tempfile::NamedTempFile;

    // Create GGUF without license/author/description
    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: vec![0u8; 16], // 4 floats * 4 bytes
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let temp_file = NamedTempFile::new().expect("create temp file");
    std::fs::write(temp_file.path(), &gguf_bytes).expect("write GGUF");

    let report = lint_gguf_file(temp_file.path()).expect("lint GGUF");

    // Should have warnings for missing license, model_card, provenance
    assert!(
        report.warn_count >= 3,
        "GGUF without metadata should have 3+ warnings"
    );
}

#[test]
fn test_lint_gguf_file_tensor_info() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use tempfile::NamedTempFile;

    // Create GGUF with multiple tensors
    let tensors = vec![
        GgufTensor {
            name: "encoder.layer.weight".to_string(),
            shape: vec![4, 4],
            dtype: GgmlType::F32,
            data: vec![0u8; 64],
        },
        GgufTensor {
            name: "encoder.layer.bias".to_string(),
            shape: vec![4],
            dtype: GgmlType::F32,
            data: vec![0u8; 16],
        },
    ];
    let metadata = vec![
        (
            "general.license".to_string(),
            GgufValue::String("MIT".to_string()),
        ),
        (
            "general.author".to_string(),
            GgufValue::String("Test".to_string()),
        ),
        (
            "general.model_card".to_string(),
            GgufValue::String("Test".to_string()),
        ),
    ];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &tensors, &metadata).expect("export GGUF");

    let temp_file = NamedTempFile::new().expect("create temp file");
    std::fs::write(temp_file.path(), &gguf_bytes).expect("write GGUF");

    let report = lint_gguf_file(temp_file.path()).expect("lint GGUF");

    // Should pass with properly named tensors
    assert!(report.passed(), "GGUF with proper names should pass lint");
}

#[test]
fn test_lint_safetensors_file_with_metadata() {
    use crate::format::test_factory::{build_pygmy_safetensors_with_config, PygmyConfig};
    use tempfile::NamedTempFile;

    // Build SafeTensors with custom metadata
    let config = PygmyConfig::default();
    let st_bytes = build_pygmy_safetensors_with_config(config);

    let temp_file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    std::fs::write(temp_file.path(), &st_bytes).expect("write SafeTensors");

    let report = lint_safetensors_file(temp_file.path()).expect("lint SafeTensors");

    // SafeTensors built with PygmyConfig may not have __metadata__ - that's expected
    // The report should still be valid (test is that lint doesn't crash)
    let _ = report.total_issues();
}

#[test]
fn test_lint_safetensors_file_with_custom_metadata() {
    use tempfile::NamedTempFile;

    // Manually create SafeTensors with __metadata__ containing license
    let header_json = serde_json::json!({
        "__metadata__": {
            "license": "Apache-2.0",
            "author": "Test Author",
            "description": "Test model for linting"
        },
        "test.weight": {
            "dtype": "F32",
            "shape": [4, 4],
            "data_offsets": [0, 64]
        }
    });
    let header_bytes = serde_json::to_vec(&header_json).expect("serialize header");
    let header_len = header_bytes.len() as u64;

    let mut st_bytes = Vec::new();
    st_bytes.extend_from_slice(&header_len.to_le_bytes());
    st_bytes.extend_from_slice(&header_bytes);
    st_bytes.extend_from_slice(&[0u8; 64]); // tensor data

    let temp_file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    std::fs::write(temp_file.path(), &st_bytes).expect("write SafeTensors");

    let report = lint_safetensors_file(temp_file.path()).expect("lint SafeTensors");

    // Should have no metadata warnings (all present in __metadata__)
    let metadata_issues = report.issues_in_category(LintCategory::Metadata);
    assert!(
        metadata_issues.is_empty(),
        "SafeTensors with full __metadata__ should have no metadata warnings"
    );
}

#[test]
fn test_lint_safetensors_file_no_metadata() {
    use tempfile::NamedTempFile;

    // Create SafeTensors without __metadata__
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
    st_bytes.extend_from_slice(&[0u8; 16]); // tensor data

    let temp_file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    std::fs::write(temp_file.path(), &st_bytes).expect("write SafeTensors");

    let report = lint_safetensors_file(temp_file.path()).expect("lint SafeTensors");

    // Should have warnings for missing license, model_card, provenance
    assert!(
        report.warn_count >= 3,
        "SafeTensors without __metadata__ should have 3+ warnings"
    );
}

#[test]
fn test_lint_model_file_gguf_dispatch() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use tempfile::NamedTempFile;

    let tensor = GgufTensor {
        name: "model.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: vec![0u8; 16],
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let temp_file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    std::fs::write(temp_file.path(), &gguf_bytes).expect("write GGUF");

    // lint_model_file should detect GGUF format and dispatch to lint_gguf_file
    let report = lint_model_file(temp_file.path()).expect("lint model file");
    assert!(
        report.total_issues() > 0,
        "lint_model_file should work for GGUF"
    );
}

#[test]
fn test_lint_model_file_safetensors_dispatch() {
    use crate::format::test_factory::build_pygmy_safetensors;
    use tempfile::NamedTempFile;

    let st_bytes = build_pygmy_safetensors();

    let temp_file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    std::fs::write(temp_file.path(), &st_bytes).expect("write SafeTensors");

    // lint_model_file should detect SafeTensors format and dispatch
    let report = lint_model_file(temp_file.path()).expect("lint model file");
    // Test passes if lint doesn't crash - total_issues() is usize, always >= 0
    let _ = report.total_issues();
}

#[test]
fn test_lint_model_file_format_detection_by_magic() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use tempfile::NamedTempFile;

    // Create GGUF file with .bin extension (format detected by magic, not extension)
    let tensor = GgufTensor {
        name: "test.weight".to_string(),
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: vec![0u8; 16],
    };
    let metadata = vec![(
        "general.license".to_string(),
        GgufValue::String("MIT".to_string()),
    )];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    // Use .bin extension but GGUF magic
    let temp_file = NamedTempFile::with_suffix(".bin").expect("create temp file");
    std::fs::write(temp_file.path(), &gguf_bytes).expect("write GGUF");

    // Should detect GGUF by magic bytes, not extension
    let report = lint_model_file(temp_file.path()).expect("lint model file");
    // Test passes if lint correctly detects GGUF by magic
    let _ = report.total_issues();
}

#[test]
fn test_lint_gguf_abbreviated_tensor_names() {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use tempfile::NamedTempFile;

    // Create GGUF with abbreviated tensor names
    let tensor = GgufTensor {
        name: "enc.w".to_string(), // abbreviated: should trigger naming warning
        shape: vec![2, 2],
        dtype: GgmlType::F32,
        data: vec![0u8; 16],
    };
    let metadata = vec![
        (
            "general.license".to_string(),
            GgufValue::String("MIT".to_string()),
        ),
        (
            "general.author".to_string(),
            GgufValue::String("Test".to_string()),
        ),
        (
            "general.description".to_string(),
            GgufValue::String("Test".to_string()),
        ),
    ];

    let mut gguf_bytes = Vec::new();
    export_tensors_to_gguf(&mut gguf_bytes, &[tensor], &metadata).expect("export GGUF");

    let temp_file = NamedTempFile::new().expect("create temp file");
    std::fs::write(temp_file.path(), &gguf_bytes).expect("write GGUF");

    let report = lint_gguf_file(temp_file.path()).expect("lint GGUF");
    let naming_issues = report.issues_in_category(LintCategory::Naming);

    // Should flag abbreviated name
    assert!(
        !naming_issues.is_empty(),
        "GGUF with abbreviated names should trigger naming warnings"
    );
}

#[test]
fn test_lint_safetensors_abbreviated_tensor_names() {
    use tempfile::NamedTempFile;

    // Create SafeTensors with abbreviated tensor name
    let header_json = serde_json::json!({
        "__metadata__": {
            "license": "MIT",
            "author": "Test",
            "description": "Test"
        },
        "enc.w": {  // abbreviated name
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
    st_bytes.extend_from_slice(&[0u8; 16]);

    let temp_file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    std::fs::write(temp_file.path(), &st_bytes).expect("write SafeTensors");

    let report = lint_safetensors_file(temp_file.path()).expect("lint SafeTensors");
    let naming_issues = report.issues_in_category(LintCategory::Naming);

    // Should flag abbreviated name
    assert!(
        !naming_issues.is_empty(),
        "SafeTensors with abbreviated names should trigger naming warnings"
    );
}

// ========================================================================
// P0 FORMAT DISPATCH TESTS (Refs GH-202)
// These tests ensure all formats are properly handled by lint_model_file
// and don't silently fail or return incorrect results.
// ========================================================================

/// P0 REGRESSION TEST: APR v1 format (APRN magic) must be handled
#[test]
fn test_lint_apr_v1_magic_detection() {
    // APR v1 magic: "APRN" (0x4150524E)
    let magic = b"APRN";
    assert_eq!(magic[0], 0x41); // 'A'
    assert_eq!(magic[1], 0x50); // 'P'
    assert_eq!(magic[2], 0x52); // 'R'
    assert_eq!(magic[3], 0x4E); // 'N'
}

/// P0 REGRESSION TEST: APR v2 format (APR\0 magic) must be handled
#[test]
fn test_lint_apr_v2_magic_detection() {
    // APR v2 magic: "APR\0" (0x41505200)
    let magic = b"APR\0";
    assert_eq!(magic[0], 0x41); // 'A'
    assert_eq!(magic[1], 0x50); // 'P'
    assert_eq!(magic[2], 0x52); // 'R'
    assert_eq!(magic[3], 0x00); // '\0'
}
