#![allow(clippy::disallowed_methods)]
//! Falsification Edge Case Matrix
//!
//! Tests edge cases for parsers, format detection, URI handling, and generation
//! parameters. These tests use synthetic data and do NOT require model files.
//!
//! Goal: Catch boundary violations, off-by-one errors, and malformed input handling
//! that historically caused bugs (GH-208, GH-220, GH-221, GH-222, PMAT-236/237).
//!
//! ~60 gates, no model-tests feature gate needed.

use std::path::Path;
use tempfile::NamedTempFile;

// =============================================================================
// GGUF Format Edge Cases (~15 tests)
// =============================================================================

#[test]
fn edge_gguf_empty_file() {
    let file = NamedTempFile::with_suffix(".gguf").expect("create temp");
    let result = aprender::format::rosetta::FormatType::from_magic(file.path());
    assert!(result.is_err(), "0-byte file must fail format detection");
}

#[test]
fn edge_gguf_under_magic_7_bytes() {
    let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp");
    std::io::Write::write_all(&mut file, &[0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00]).unwrap();
    // 7 bytes: has magic but incomplete — must NOT panic
    let _ = aprender::format::rosetta::FormatType::from_magic(file.path());
}

#[test]
fn edge_gguf_wrong_magic() {
    let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp");
    std::io::Write::write_all(&mut file, b"NOTGGUF\x00").unwrap();
    let result = aprender::format::rosetta::FormatType::from_magic(file.path());
    assert!(
        result.is_err() || result.unwrap() != aprender::format::rosetta::FormatType::Gguf,
        "Wrong magic must not detect as GGUF"
    );
}

#[test]
fn edge_gguf_magic_only_no_header() {
    let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp");
    std::io::Write::write_all(&mut file, b"GGUF\x03\x00\x00\x00").unwrap();
    let result = aprender::format::rosetta::FormatType::from_magic(file.path());
    assert_eq!(result.unwrap(), aprender::format::rosetta::FormatType::Gguf);
}

#[test]
fn edge_gguf_version_1_old() {
    let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp");
    std::io::Write::write_all(&mut file, b"GGUF\x01\x00\x00\x00").unwrap();
    let result = aprender::format::rosetta::FormatType::from_magic(file.path());
    assert_eq!(result.unwrap(), aprender::format::rosetta::FormatType::Gguf);
}

#[test]
fn edge_gguf_version_99_future() {
    let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp");
    std::io::Write::write_all(&mut file, b"GGUF\x63\x00\x00\x00").unwrap();
    let result = aprender::format::rosetta::FormatType::from_magic(file.path());
    assert_eq!(result.unwrap(), aprender::format::rosetta::FormatType::Gguf);
}

#[test]
fn edge_format_detection_from_extension_gguf() {
    let result = aprender::format::rosetta::FormatType::from_extension(Path::new("model.gguf"));
    assert_eq!(result.unwrap(), aprender::format::rosetta::FormatType::Gguf);
}

#[test]
fn edge_format_detection_from_extension_safetensors() {
    let result =
        aprender::format::rosetta::FormatType::from_extension(Path::new("model.safetensors"));
    assert_eq!(
        result.unwrap(),
        aprender::format::rosetta::FormatType::SafeTensors
    );
}

#[test]
fn edge_format_detection_from_extension_apr() {
    let result = aprender::format::rosetta::FormatType::from_extension(Path::new("model.apr"));
    assert_eq!(result.unwrap(), aprender::format::rosetta::FormatType::Apr);
}

#[test]
fn edge_format_detection_unknown_extension() {
    let result = aprender::format::rosetta::FormatType::from_extension(Path::new("model.bin"));
    assert!(result.is_err(), "Unknown extension must return error");
}

#[test]
fn edge_format_detection_no_extension() {
    let result = aprender::format::rosetta::FormatType::from_extension(Path::new("model"));
    assert!(result.is_err(), "No extension must return error");
}

#[test]
fn edge_format_detection_directory_path() {
    let result =
        aprender::format::rosetta::FormatType::from_extension(Path::new("/tmp/models/qwen/"));
    assert!(result.is_err(), "Directory path must return error");
}

#[test]
fn edge_format_detection_double_extension() {
    let result = aprender::format::rosetta::FormatType::from_extension(Path::new("model.tar.gguf"));
    assert_eq!(result.unwrap(), aprender::format::rosetta::FormatType::Gguf);
}

#[test]
fn edge_format_detection_uppercase_extension() {
    let result = aprender::format::rosetta::FormatType::from_extension(Path::new("MODEL.GGUF"));
    assert_eq!(result.unwrap(), aprender::format::rosetta::FormatType::Gguf);
}

#[test]
fn edge_format_detection_mixed_case() {
    let result = aprender::format::rosetta::FormatType::from_extension(Path::new("model.GgUf"));
    assert_eq!(result.unwrap(), aprender::format::rosetta::FormatType::Gguf);
}

// =============================================================================
// SafeTensors Edge Cases (~5 tests)
// =============================================================================

#[test]
fn edge_safetensors_empty_file() {
    let file = NamedTempFile::with_suffix(".safetensors").expect("create temp");
    let result = aprender::format::rosetta::FormatType::from_magic(file.path());
    assert!(
        result.is_err(),
        "Empty file must fail SafeTensors detection"
    );
}


include!("includes/falsification_edge_cases_include_01.rs");
