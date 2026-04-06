//! Extended tests for model_loader module - Part 03
//!
//! Focus areas:
//! - LoadError Display and Error trait implementations
//! - Format error conversion edge cases
//!
//! Per EXTREME TDD methodology: Comprehensive edge case coverage.

use crate::format::{FormatError, ModelFormat};
use crate::model_loader::LoadError;

// ===== LoadError Display tests =====

#[test]
fn test_load_error_display_format_error() {
    let err = LoadError::FormatError(FormatError::TooShort { len: 4 });
    let msg = err.to_string();
    assert!(msg.contains("Format detection error"));
    assert!(msg.contains("4 bytes"));
}

#[test]
fn test_load_error_display_io_error() {
    let err = LoadError::IoError("disk failure".to_string());
    let msg = err.to_string();
    assert!(msg.contains("I/O error"));
    assert!(msg.contains("disk failure"));
}

#[test]
fn test_load_error_display_parse_error() {
    let err = LoadError::ParseError("corrupted header".to_string());
    let msg = err.to_string();
    assert!(msg.contains("Parse error"));
    assert!(msg.contains("corrupted header"));
}

#[test]
fn test_load_error_display_unsupported_type() {
    let err = LoadError::UnsupportedType("Transformer".to_string());
    let msg = err.to_string();
    assert!(msg.contains("Unsupported model type"));
    assert!(msg.contains("Transformer"));
}

#[test]
fn test_load_error_display_integrity_error() {
    let err = LoadError::IntegrityError("CRC32 verification failed".to_string());
    let msg = err.to_string();
    assert!(msg.contains("Integrity check failed"));
    assert!(msg.contains("CRC32"));
}

#[test]
fn test_load_error_display_type_mismatch() {
    let err = LoadError::TypeMismatch {
        expected: "RandomForest".to_string(),
        actual: "GradientBoosting".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("type mismatch"));
    assert!(msg.contains("RandomForest"));
    assert!(msg.contains("GradientBoosting"));
}

// ===== LoadError std::error::Error implementation =====

#[test]
fn test_load_error_is_error() {
    let err: &dyn std::error::Error = &LoadError::IoError("test".to_string());
    // Error trait requires Display + Debug
    assert!(!err.to_string().is_empty());
}

#[test]
fn test_load_error_error_trait_for_all_variants() {
    let variants: Vec<Box<dyn std::error::Error>> = vec![
        Box::new(LoadError::FormatError(FormatError::UnknownFormat)),
        Box::new(LoadError::IoError("io".to_string())),
        Box::new(LoadError::ParseError("parse".to_string())),
        Box::new(LoadError::UnsupportedType("unsupported".to_string())),
        Box::new(LoadError::IntegrityError("integrity".to_string())),
        Box::new(LoadError::TypeMismatch {
            expected: "A".to_string(),
            actual: "B".to_string(),
        }),
    ];

    for err in variants {
        // All should implement Display (via Error trait)
        assert!(!err.to_string().is_empty());
    }
}

// ===== LoadError from FormatError extension mismatch =====

#[test]
fn test_load_error_from_format_error_extension_mismatch() {
    let format_err = FormatError::ExtensionMismatch {
        detected: ModelFormat::Gguf,
        extension: "apr".to_string(),
    };
    let load_err: LoadError = format_err.into();
    match load_err {
        LoadError::FormatError(FormatError::ExtensionMismatch {
            detected,
            extension,
        }) => {
            assert_eq!(detected, ModelFormat::Gguf);
            assert_eq!(extension, "apr");
        },
        other => panic!("Expected FormatError::ExtensionMismatch, got {other:?}"),
    }
}

// ===== LoadError Debug trait =====

#[test]
fn test_load_error_debug_format_error() {
    let err = LoadError::FormatError(FormatError::TooShort { len: 2 });
    let debug_str = format!("{err:?}");
    assert!(debug_str.contains("FormatError"));
    assert!(debug_str.contains("TooShort"));
}

#[test]
fn test_load_error_debug_io_error() {
    let err = LoadError::IoError("test io".to_string());
    let debug_str = format!("{err:?}");
    assert!(debug_str.contains("IoError"));
    assert!(debug_str.contains("test io"));
}

#[test]
fn test_load_error_debug_parse_error() {
    let err = LoadError::ParseError("test parse".to_string());
    let debug_str = format!("{err:?}");
    assert!(debug_str.contains("ParseError"));
    assert!(debug_str.contains("test parse"));
}

#[test]
fn test_load_error_debug_unsupported_type() {
    let err = LoadError::UnsupportedType("TestType".to_string());
    let debug_str = format!("{err:?}");
    assert!(debug_str.contains("UnsupportedType"));
    assert!(debug_str.contains("TestType"));
}

#[test]
fn test_load_error_debug_integrity_error() {
    let err = LoadError::IntegrityError("checksum fail".to_string());
    let debug_str = format!("{err:?}");
    assert!(debug_str.contains("IntegrityError"));
    assert!(debug_str.contains("checksum fail"));
}

#[test]
fn test_load_error_debug_type_mismatch() {
    let err = LoadError::TypeMismatch {
        expected: "ExpectedType".to_string(),
        actual: "ActualType".to_string(),
    };
    let debug_str = format!("{err:?}");
    assert!(debug_str.contains("TypeMismatch"));
    assert!(debug_str.contains("ExpectedType"));
    assert!(debug_str.contains("ActualType"));
}

// ===== LoadError Clone trait =====

#[test]
fn test_load_error_clone_format_error() {
    let err = LoadError::FormatError(FormatError::TooShort { len: 3 });
    let cloned = err.clone();
    assert!(matches!(
        cloned,
        LoadError::FormatError(FormatError::TooShort { len: 3 })
    ));
}

#[test]
fn test_load_error_clone_io_error() {
    let err = LoadError::IoError("clone test".to_string());
    let cloned = err.clone();
    match cloned {
        LoadError::IoError(msg) => assert_eq!(msg, "clone test"),
        other => panic!("Expected IoError, got {other:?}"),
    }
}

#[test]
fn test_load_error_clone_parse_error() {
    let err = LoadError::ParseError("clone parse".to_string());
    let cloned = err.clone();
    match cloned {
        LoadError::ParseError(msg) => assert_eq!(msg, "clone parse"),
        other => panic!("Expected ParseError, got {other:?}"),
    }
}

#[test]
fn test_load_error_clone_unsupported_type() {
    let err = LoadError::UnsupportedType("CloneType".to_string());
    let cloned = err.clone();
    match cloned {
        LoadError::UnsupportedType(t) => assert_eq!(t, "CloneType"),
        other => panic!("Expected UnsupportedType, got {other:?}"),
    }
}

#[test]
fn test_load_error_clone_integrity_error() {
    let err = LoadError::IntegrityError("clone integrity".to_string());
    let cloned = err.clone();
    match cloned {
        LoadError::IntegrityError(msg) => assert_eq!(msg, "clone integrity"),
        other => panic!("Expected IntegrityError, got {other:?}"),
    }
}

#[test]
fn test_load_error_clone_type_mismatch() {
    let err = LoadError::TypeMismatch {
        expected: "E".to_string(),
        actual: "A".to_string(),
    };
    let cloned = err.clone();
    match cloned {
        LoadError::TypeMismatch { expected, actual } => {
            assert_eq!(expected, "E");
            assert_eq!(actual, "A");
        },
        other => panic!("Expected TypeMismatch, got {other:?}"),
    }
}

// ===== Display format edge cases =====

#[test]
fn test_load_error_display_empty_strings() {
    let err = LoadError::IoError(String::new());
    assert_eq!(err.to_string(), "I/O error: ");

    let err = LoadError::ParseError(String::new());
    assert_eq!(err.to_string(), "Parse error: ");

    let err = LoadError::UnsupportedType(String::new());
    assert_eq!(err.to_string(), "Unsupported model type: ");

    let err = LoadError::IntegrityError(String::new());
    assert_eq!(err.to_string(), "Integrity check failed: ");
}

#[test]
fn test_load_error_display_type_mismatch_empty() {
    let err = LoadError::TypeMismatch {
        expected: String::new(),
        actual: String::new(),
    };
    let msg = err.to_string();
    assert!(msg.contains("type mismatch"));
    assert!(msg.contains("expected"));
    assert!(msg.contains("got"));
}

#[test]
fn test_load_error_display_special_chars() {
    let err = LoadError::IoError("path/with/slashes & special <chars>".to_string());
    let msg = err.to_string();
    assert!(msg.contains("path/with/slashes & special <chars>"));
}

#[test]
fn test_load_error_display_unicode() {
    let err = LoadError::ParseError("unicode: \u{1F600} \u{4E2D}\u{6587}".to_string());
    let msg = err.to_string();
    assert!(msg.contains("\u{1F600}"));
    assert!(msg.contains("\u{4E2D}\u{6587}"));
}
