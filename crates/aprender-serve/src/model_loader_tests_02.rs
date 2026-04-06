//! Extended tests for model_loader module - Part 02
//!
//! Focus areas:
//! - File-based model detection (`detect_model`)
//! - Error handling edge cases
//!
//! Per EXTREME TDD methodology: Comprehensive edge case coverage.

use std::io::Write;
use std::path::Path;
use tempfile::NamedTempFile;

use crate::format::{FormatError, ModelFormat};
use crate::model_loader::{
    detect_model, detect_model_from_bytes, read_apr_model_type, validate_model_type, LoadError,
    ModelMetadata,
};

// ===== File-based detect_model tests =====

#[test]
fn test_detect_model_apr_file() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("tempfile");
    let mut data = b"APR\0".to_vec();
    data.extend_from_slice(&0x0002u16.to_le_bytes()); // LogisticRegression
    data.extend_from_slice(&[0x01, 0x00]); // version
    data.extend_from_slice(&[0u8; 100]); // padding
    file.write_all(&data).expect("write");
    file.flush().expect("flush");
    let metadata = detect_model(file.path()).expect("detect_model");
    assert_eq!(metadata.format, ModelFormat::Apr);
    assert_eq!(metadata.file_size, 108);
}

#[test]
fn test_detect_model_gguf_file() {
    let mut file = NamedTempFile::with_suffix(".gguf").expect("tempfile");
    let mut data = b"GGUF".to_vec();
    data.extend_from_slice(&[0u8; 100]);
    file.write_all(&data).expect("write");
    file.flush().expect("flush");
    let metadata = detect_model(file.path()).expect("detect_model");
    assert_eq!(metadata.format, ModelFormat::Gguf);
    assert_eq!(metadata.file_size, 104);
}

#[test]
fn test_detect_model_safetensors_file() {
    let mut file = NamedTempFile::with_suffix(".safetensors").expect("tempfile");
    let header_size: u64 = 500;
    let mut data = header_size.to_le_bytes().to_vec();
    data.extend_from_slice(&[0u8; 100]);
    file.write_all(&data).expect("write");
    file.flush().expect("flush");
    let metadata = detect_model(file.path()).expect("detect_model");
    assert_eq!(metadata.format, ModelFormat::SafeTensors);
    assert_eq!(metadata.file_size, 108);
}

#[test]
fn test_detect_model_file_not_found() {
    let result = detect_model(Path::new("/nonexistent/path/model.apr"));
    assert!(result.is_err());
    match result.unwrap_err() {
        LoadError::IoError(msg) => {
            assert!(msg.contains("No such file") || msg.contains("not found"));
        },
        other => panic!("Expected IoError, got {other:?}"),
    }
}

#[test]
fn test_detect_model_file_too_small() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("tempfile");
    file.write_all(b"APR").expect("write"); // Only 3 bytes
    file.flush().expect("flush");
    let result = detect_model(file.path());
    assert!(result.is_err());
    match result.unwrap_err() {
        LoadError::ParseError(msg) => {
            assert!(msg.contains("too small") && msg.contains("3 bytes"));
        },
        other => panic!("Expected ParseError, got {other:?}"),
    }
}

#[test]
fn test_detect_model_empty_file() {
    let file = NamedTempFile::with_suffix(".apr").expect("tempfile");
    let result = detect_model(file.path());
    assert!(result.is_err());
    match result.unwrap_err() {
        LoadError::ParseError(msg) => {
            assert!(msg.contains("too small") && msg.contains("0 bytes"));
        },
        other => panic!("Expected ParseError, got {other:?}"),
    }
}

#[test]
fn test_detect_model_exactly_7_bytes() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("tempfile");
    file.write_all(b"APR\0xyz").expect("write"); // 7 bytes
    file.flush().expect("flush");
    let result = detect_model(file.path());
    assert!(result.is_err());
    match result.unwrap_err() {
        LoadError::ParseError(msg) => {
            assert!(msg.contains("too small") && msg.contains("7 bytes"));
        },
        other => panic!("Expected ParseError, got {other:?}"),
    }
}

#[test]
fn test_detect_model_exactly_8_bytes() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("tempfile");
    file.write_all(b"APR\0xyzz").expect("write"); // 8 bytes
    file.flush().expect("flush");
    let metadata = detect_model(file.path()).expect("detect_model");
    assert_eq!(metadata.format, ModelFormat::Apr);
    assert_eq!(metadata.file_size, 8);
}

#[test]
fn test_detect_model_extension_mismatch() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("tempfile");
    let mut data = b"GGUF".to_vec(); // GGUF magic but .apr extension
    data.extend_from_slice(&[0u8; 100]);
    file.write_all(&data).expect("write");
    file.flush().expect("flush");
    let result = detect_model(file.path());
    assert!(result.is_err());
    match result.unwrap_err() {
        LoadError::FormatError(FormatError::ExtensionMismatch {
            detected,
            extension,
        }) => {
            assert_eq!(detected, ModelFormat::Gguf);
            assert_eq!(extension, "apr");
        },
        other => panic!("Expected ExtensionMismatch, got {other:?}"),
    }
}

#[test]
fn test_detect_model_unknown_extension_valid_magic() {
    let mut file = NamedTempFile::with_suffix(".bin").expect("tempfile");
    let mut data = b"APR\0".to_vec();
    data.extend_from_slice(&[0u8; 100]);
    file.write_all(&data).expect("write");
    file.flush().expect("flush");
    let metadata = detect_model(file.path()).expect("detect_model");
    assert_eq!(metadata.format, ModelFormat::Apr);
}

// ===== detect_model_from_bytes edge cases =====

#[test]
fn test_detect_model_from_bytes_exactly_7_bytes() {
    let data = b"APR\0xyz";
    let result = detect_model_from_bytes(data);
    assert!(result.is_err());
    match result.unwrap_err() {
        LoadError::ParseError(msg) => {
            assert!(msg.contains("too small") && msg.contains("7 bytes"));
        },
        other => panic!("Expected ParseError, got {other:?}"),
    }
}

#[test]
fn test_detect_model_from_bytes_exactly_8_bytes_apr() {
    let data = b"APR\0xyzz";
    let metadata = detect_model_from_bytes(data).expect("detect_model_from_bytes");
    assert_eq!(metadata.format, ModelFormat::Apr);
    assert_eq!(metadata.file_size, 8);
}

#[test]
fn test_detect_model_from_bytes_unknown_format() {
    let data = b"\x00\x00\x00\x00\x00\x00\x00\x00"; // Zero header = not SafeTensors
    let result = detect_model_from_bytes(data);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        LoadError::FormatError(FormatError::UnknownFormat)
    ));
}

#[test]
fn test_detect_model_from_bytes_safetensors_header_too_large() {
    let header_size: u64 = 200_000_000; // 200MB - too large
    let mut data = header_size.to_le_bytes().to_vec();
    data.extend_from_slice(&[0u8; 100]);
    let result = detect_model_from_bytes(&data);
    assert!(result.is_err());
    match result.unwrap_err() {
        LoadError::FormatError(FormatError::HeaderTooLarge { size }) => {
            assert_eq!(size, 200_000_000);
        },
        other => panic!("Expected HeaderTooLarge, got {other:?}"),
    }
}

#[test]
fn test_detect_model_from_bytes_apr_versions() {
    // APR v1
    let mut data = b"APR1".to_vec();
    data.extend_from_slice(&[0u8; 100]);
    assert_eq!(
        detect_model_from_bytes(&data).expect("v1").format,
        ModelFormat::Apr
    );
    // APR v2
    let mut data = b"APR2".to_vec();
    data.extend_from_slice(&[0u8; 100]);
    assert_eq!(
        detect_model_from_bytes(&data).expect("v2").format,
        ModelFormat::Apr
    );
}

// ===== LoadError conversion tests =====

#[test]
fn test_load_error_from_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let load_err: LoadError = io_err.into();
    assert!(matches!(load_err, LoadError::IoError(msg) if msg.contains("file not found")));
}

#[test]
fn test_load_error_from_io_error_permission_denied() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
    let load_err: LoadError = io_err.into();
    assert!(matches!(load_err, LoadError::IoError(msg) if msg.contains("access denied")));
}

#[test]
fn test_load_error_from_format_errors() {
    // TooShort
    let err: LoadError = FormatError::TooShort { len: 5 }.into();
    assert!(matches!(
        err,
        LoadError::FormatError(FormatError::TooShort { len: 5 })
    ));
    // UnknownFormat
    let err: LoadError = FormatError::UnknownFormat.into();
    assert!(matches!(
        err,
        LoadError::FormatError(FormatError::UnknownFormat)
    ));
    // HeaderTooLarge
    let err: LoadError = FormatError::HeaderTooLarge { size: 999 }.into();
    assert!(matches!(
        err,
        LoadError::FormatError(FormatError::HeaderTooLarge { size: 999 })
    ));
}

// ===== read_apr_model_type edge cases =====

#[test]
fn test_read_apr_model_type_too_short() {
    assert_eq!(read_apr_model_type(&[]), None);
    assert_eq!(read_apr_model_type(&[0x41]), None);
    assert_eq!(read_apr_model_type(b"APR\0\x01\x00\x01"), None); // 7 bytes
}

#[test]
fn test_read_apr_model_type_exactly_8_bytes() {
    // F-COV-95: APR v1 uses "APRN" magic
    let mut data = b"APRN".to_vec();
    data.extend_from_slice(&0x0003u16.to_le_bytes()); // DecisionTree
    data.extend_from_slice(&[0, 0]);
    assert_eq!(read_apr_model_type(&data), Some("DecisionTree".to_string()));
}

#[test]
fn test_read_apr_model_type_undefined_ids() {
    // F-COV-95: APR v1 uses "APRN" magic
    // Gap between SVM (0x000A) and NgramLM (0x0010)
    let mut data = b"APRN".to_vec();
    data.extend_from_slice(&0x000Bu16.to_le_bytes());
    data.extend_from_slice(&[0, 0]);
    assert_eq!(read_apr_model_type(&data), None);
    // Gap between NeuralCustom (0x0021) and ContentRecommender (0x0030)
    let mut data = b"APRN".to_vec();
    data.extend_from_slice(&0x0022u16.to_le_bytes());
    data.extend_from_slice(&[0, 0]);
    assert_eq!(read_apr_model_type(&data), None);
}

// ===== validate_model_type tests =====

#[test]
fn test_validate_model_type_cases() {
    assert!(validate_model_type("", "").is_ok());
    assert!(validate_model_type("Model", "Model").is_ok());
    assert!(validate_model_type("LogisticRegression", "LogisticRegression ").is_err());
}

// ===== ModelMetadata builder pattern tests =====

#[test]
fn test_model_metadata_builder_all_formats() {
    for format in [
        ModelFormat::Apr,
        ModelFormat::Gguf,
        ModelFormat::SafeTensors,
    ] {
        let meta = ModelMetadata::new(format);
        assert_eq!(meta.format, format);
    }
}

#[test]
fn test_model_metadata_with_builders() {
    let meta = ModelMetadata::new(ModelFormat::Apr).with_model_type(String::from("TestModel"));
    assert_eq!(meta.model_type, Some("TestModel".to_string()));

    let meta = ModelMetadata::new(ModelFormat::Gguf).with_version(String::from("1.2.3"));
    assert_eq!(meta.version, Some("1.2.3".to_string()));

    let meta = ModelMetadata::new(ModelFormat::SafeTensors)
        .with_input_dim(usize::MAX)
        .with_output_dim(usize::MAX);
    assert_eq!(meta.input_dim, Some(usize::MAX));
    assert_eq!(meta.output_dim, Some(usize::MAX));

    let meta = ModelMetadata::new(ModelFormat::Apr).with_file_size(u64::MAX);
    assert_eq!(meta.file_size, u64::MAX);
}

#[test]
fn test_model_metadata_chaining_preserves_values() {
    let meta = ModelMetadata::new(ModelFormat::Apr)
        .with_model_type("TypeA")
        .with_version("v1")
        .with_model_type("TypeB"); // Override
    assert_eq!(meta.model_type, Some("TypeB".to_string()));
    assert_eq!(meta.version, Some("v1".to_string()));
}
