use super::*;
use aprender::format::MergeReport;
use std::io::Write;
use tempfile::NamedTempFile;

// ========================================================================
// Validation Error Tests
// ========================================================================

// NOTE: the merge precondition is a `debug_assert!` (contract_pre_merge_weight_conservation),
// so it panics only in debug builds. In `--release` it is compiled out and `run` falls through to
// `validate_merge_inputs`, which returns `Err(ValidationFailed)`. These tests assert the rejection
// build-mode-agnostically: debug => panics ("Contract …"), release => Err. Either way <2 files is
// rejected. (Without the cfg_attr these `#[should_panic]` tests fail under `cargo test --release`.)

#[test]
#[cfg_attr(debug_assertions, should_panic(expected = "Contract"))]
fn test_run_insufficient_files() {
    // Merge requires >= 2 files.
    let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
    let result = run(
        &[file.path().to_path_buf()],
        "average",
        Some(Path::new("/tmp/merged.apr")),
        None,
        None,
        0.9,
        0.2,
        42,
        false,
        false,
    );
    // Debug: unreachable (panicked above). Release: must be a rejection, not a successful merge.
    #[cfg(not(debug_assertions))]
    assert!(
        result.is_err(),
        "merge with <2 files must return Err in release"
    );
    let _ = result;
}

#[test]
#[cfg_attr(debug_assertions, should_panic(expected = "Contract"))]
fn test_run_empty_files() {
    // Merge requires a non-empty file list.
    let result = run(
        &[],
        "average",
        Some(Path::new("/tmp/merged.apr")),
        None,
        None,
        0.9,
        0.2,
        42,
        false,
        false,
    );
    // Debug: unreachable (panicked above). Release: must be a rejection, not a successful merge.
    #[cfg(not(debug_assertions))]
    assert!(
        result.is_err(),
        "merge with empty file list must return Err in release"
    );
    let _ = result;
}

#[test]
fn test_run_file_not_found() {
    let result = run(
        &[
            PathBuf::from("/nonexistent/model1.apr"),
            PathBuf::from("/nonexistent/model2.apr"),
        ],
        "average",
        Some(Path::new("/tmp/merged.apr")),
        None,
        None,
        0.9,
        0.2,
        42,
        false,
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
        Some(Path::new("/tmp/merged.apr")),
        None,
        None,
        0.9,
        0.2,
        42,
        false,
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
        Some(Path::new("/tmp/merged.apr")),
        None,
        None,
        0.9,
        0.2,
        42,
        false,
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
        Some(Path::new("/tmp/merged.safetensors")),
        None,
        None, // no base model
        0.9,
        0.2,
        42,
        false,
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
        Some(Path::new("/tmp/merged.safetensors")),
        None,
        None, // no base model
        0.9,
        0.2,
        42,
        false,
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
        Some(Path::new("/tmp/merged.safetensors")),
        None,
        None,
        0.9,
        0.2,
        42,
        false,
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
        Some(Path::new("/tmp/merged.apr")),
        None,
        None,
        0.9,
        0.2,
        42,
        false,
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
        Some(Path::new("/tmp/merged.apr")),
        Some(vec![0.7, 0.3]),
        None,
        0.9,
        0.2,
        42,
        false,
        false,
    );
    // Will fail at actual merge, but tests weight parsing path
    assert!(result.is_err());
}
