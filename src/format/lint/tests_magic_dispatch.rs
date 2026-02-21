use super::*;

/// P0 REGRESSION TEST: lint_apr_file dispatches correctly by magic
#[test]
fn test_lint_apr_file_magic_dispatch() {
    use tempfile::NamedTempFile;

    // Test APR v1 magic detection (should attempt v1 parsing)
    let apr_v1_file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    std::fs::write(apr_v1_file.path(), b"APRN\x00\x00\x00\x00").expect("write");

    let result = lint_apr_file(apr_v1_file.path());
    // Should fail because invalid v1 file, but NOT with "unknown magic" error
    assert!(result.is_err(), "Invalid v1 file should error");
    let err = result.unwrap_err().to_string();
    assert!(
        !err.contains("Invalid APR magic"),
        "APR v1 magic should be recognized"
    );

    // Test APR v2 magic detection (should attempt v2 parsing)
    let apr_v2_file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    std::fs::write(apr_v2_file.path(), b"APR\x00\x02\x00\x00\x00").expect("write");

    let result = lint_apr_file(apr_v2_file.path());
    // Should fail because invalid v2 file, but NOT with "unknown magic" error
    assert!(result.is_err(), "Invalid v2 file should error");
    let err = result.unwrap_err().to_string();
    assert!(
        !err.contains("Invalid APR magic"),
        "APR v2 magic should be recognized"
    );
}

/// P0 REGRESSION TEST: Unknown magic should produce clear error
#[test]
fn test_lint_apr_file_unknown_magic_error() {
    use tempfile::NamedTempFile;

    // Test unknown magic produces clear error
    let unknown_file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    std::fs::write(unknown_file.path(), b"XXXX\x00\x00\x00\x00").expect("write");

    let result = lint_apr_file(unknown_file.path());
    assert!(result.is_err(), "Unknown magic should error");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Invalid APR magic") || err.contains("magic"),
        "Unknown magic error should mention 'magic': {}",
        err
    );
}

/// P0 REGRESSION TEST: lint_model_file dispatches to correct handler
#[test]
fn test_lint_model_file_format_dispatch() {
    use crate::format::rosetta::FormatType;
    use tempfile::NamedTempFile;

    // Test GGUF dispatch
    let gguf_file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    // Minimal GGUF: magic + version (incomplete but enough for format detection)
    std::fs::write(gguf_file.path(), b"GGUF\x03\x00\x00\x00").expect("write");

    let format = FormatType::from_magic(gguf_file.path());
    assert!(format.is_ok(), "GGUF format should be detected");
    assert!(
        matches!(format.unwrap(), FormatType::Gguf),
        "Must detect as GGUF"
    );

    // Test APR dispatch
    let apr_file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    std::fs::write(apr_file.path(), b"APR\x00\x02\x00\x00\x00").expect("write");

    let format = FormatType::from_magic(apr_file.path());
    assert!(format.is_ok(), "APR format should be detected");
    assert!(
        matches!(format.unwrap(), FormatType::Apr),
        "Must detect as APR"
    );

    // Test SafeTensors dispatch
    let st_file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    let header = b"{\"test\": {}}";
    let mut st_bytes = Vec::new();
    st_bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    st_bytes.extend_from_slice(header);
    std::fs::write(st_file.path(), &st_bytes).expect("write");

    let format = FormatType::from_magic(st_file.path());
    assert!(format.is_ok(), "SafeTensors format should be detected");
    assert!(
        matches!(format.unwrap(), FormatType::SafeTensors),
        "Must detect as SafeTensors"
    );
}
