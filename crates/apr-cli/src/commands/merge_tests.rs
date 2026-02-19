use super::*;
use aprender::format::MergeReport;
use std::io::Write;
use tempfile::NamedTempFile;

// ========================================================================
// Validation Error Tests
// ========================================================================

#[test]
fn test_run_insufficient_files() {
    let file = NamedTempFile::with_suffix(".apr").expect("create temp file");

    let result = run(
        &[file.path().to_path_buf()],
        "average",
        Path::new("/tmp/merged.apr"),
        None,
        None,
        0.9,
        0.2,
        42,
        false,
    );
    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(msg.contains("at least 2"));
        }
        _ => panic!("Expected ValidationFailed error"),
    }
}

#[test]
fn test_run_empty_files() {
    let result = run(
        &[],
        "average",
        Path::new("/tmp/merged.apr"),
        None,
        None,
        0.9,
        0.2,
        42,
        false,
    );
    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(msg.contains("at least 2"));
        }
        _ => panic!("Expected ValidationFailed error"),
    }
}

#[test]
fn test_run_file_not_found() {
    let result = run(
        &[
            PathBuf::from("/nonexistent/model1.apr"),
            PathBuf::from("/nonexistent/model2.apr"),
        ],
        "average",
        Path::new("/tmp/merged.apr"),
        None,
        None,
        0.9,
        0.2,
        42,
        false,
    );
    assert!(result.is_err());
    match result {
        Err(CliError::FileNotFound(_)) => {}
        _ => panic!("Expected FileNotFound error"),
    }
}

#[test]
fn test_run_second_file_not_found() {
    let file1 = NamedTempFile::with_suffix(".apr").expect("create temp file");

    let result = run(
        &[
            file1.path().to_path_buf(),
            PathBuf::from("/nonexistent/model2.apr"),
        ],
        "average",
        Path::new("/tmp/merged.apr"),
        None,
        None,
        0.9,
        0.2,
        42,
        false,
    );
    assert!(result.is_err());
    match result {
        Err(CliError::FileNotFound(_)) => {}
        _ => panic!("Expected FileNotFound error"),
    }
}

#[test]
fn test_run_unknown_strategy() {
    let file1 = NamedTempFile::with_suffix(".apr").expect("create temp file");
    let file2 = NamedTempFile::with_suffix(".apr").expect("create temp file");

    let result = run(
        &[file1.path().to_path_buf(), file2.path().to_path_buf()],
        "unknown_strategy",
        Path::new("/tmp/merged.apr"),
        None,
        None,
        0.9,
        0.2,
        42,
        false,
    );
    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(msg.contains("Unknown merge strategy"));
        }
        _ => panic!("Expected ValidationFailed error"),
    }
}

#[test]
fn test_run_ties_without_base_model() {
    let file1 = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    let file2 = NamedTempFile::with_suffix(".safetensors").expect("create temp file");

    let result = run(
        &[file1.path().to_path_buf(), file2.path().to_path_buf()],
        "ties",
        Path::new("/tmp/merged.safetensors"),
        None,
        None, // no base model
        0.9,
        0.2,
        42,
        false,
    );
    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(
                msg.contains("base-model") || msg.contains("base_model") || msg.contains("TIES")
            );
        }
        _ => panic!("Expected ValidationFailed error for missing base model"),
    }
}

#[test]
fn test_run_dare_without_base_model() {
    let file1 = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    let file2 = NamedTempFile::with_suffix(".safetensors").expect("create temp file");

    let result = run(
        &[file1.path().to_path_buf(), file2.path().to_path_buf()],
        "dare",
        Path::new("/tmp/merged.safetensors"),
        None,
        None, // no base model
        0.9,
        0.2,
        42,
        false,
    );
    assert!(result.is_err());
    match result {
        Err(CliError::ValidationFailed(msg)) => {
            assert!(
                msg.contains("base-model") || msg.contains("base_model") || msg.contains("DARE")
            );
        }
        _ => panic!("Expected ValidationFailed error for missing base model"),
    }
}

#[test]
fn test_run_slerp_with_three_models() {
    let file1 = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    let file2 = NamedTempFile::with_suffix(".safetensors").expect("create temp file");
    let file3 = NamedTempFile::with_suffix(".safetensors").expect("create temp file");

    let result = run(
        &[
            file1.path().to_path_buf(),
            file2.path().to_path_buf(),
            file3.path().to_path_buf(),
        ],
        "slerp",
        Path::new("/tmp/merged.safetensors"),
        None,
        None,
        0.9,
        0.2,
        42,
        false,
    );
    assert!(result.is_err());
}

// ========================================================================
// Display Report Tests
// ========================================================================

#[test]
fn test_display_report_basic() {
    let report = MergeReport {
        model_count: 2,
        tensor_count: 100,
        output_size: 1024 * 1024 * 100, // 100MB
        strategy: MergeStrategy::Average,
        weights_used: None,
    };
    display_report(&report);
}

#[test]
fn test_display_report_with_weights() {
    let report = MergeReport {
        model_count: 3,
        tensor_count: 200,
        output_size: 1024 * 1024 * 500, // 500MB
        strategy: MergeStrategy::Weighted,
        weights_used: Some(vec![0.5, 0.3, 0.2]),
    };
    display_report(&report);
}

#[test]
fn test_display_report_large_merge() {
    let report = MergeReport {
        model_count: 5,
        tensor_count: 1000,
        output_size: 7 * 1024 * 1024 * 1024, // 7GB
        strategy: MergeStrategy::Average,
        weights_used: None,
    };
    display_report(&report);
}

// ========================================================================
// Invalid File Content Tests
// ========================================================================

#[test]
fn test_run_invalid_apr_files() {
    let mut file1 = NamedTempFile::with_suffix(".apr").expect("create temp file");
    let mut file2 = NamedTempFile::with_suffix(".apr").expect("create temp file");

    file1.write_all(b"not valid APR").expect("write to file");
    file2
        .write_all(b"also not valid APR")
        .expect("write to file");

    let result = run(
        &[file1.path().to_path_buf(), file2.path().to_path_buf()],
        "average",
        Path::new("/tmp/merged.apr"),
        None,
        None,
        0.9,
        0.2,
        42,
        false,
    );
    // Should fail because files are not valid APR
    assert!(result.is_err());
}

#[test]
fn test_run_with_weights() {
    let mut file1 = NamedTempFile::with_suffix(".apr").expect("create temp file");
    let mut file2 = NamedTempFile::with_suffix(".apr").expect("create temp file");

    file1.write_all(b"data1").expect("write");
    file2.write_all(b"data2").expect("write");

    let result = run(
        &[file1.path().to_path_buf(), file2.path().to_path_buf()],
        "weighted",
        Path::new("/tmp/merged.apr"),
        Some(vec![0.7, 0.3]),
        None,
        0.9,
        0.2,
        42,
        false,
    );
    // Will fail at actual merge, but tests weight parsing path
    assert!(result.is_err());
}
