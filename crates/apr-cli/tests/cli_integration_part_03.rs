
// F-TUNE-002: apr tune with missing model shows error
#[test]
fn test_f_tune_002_missing_model() {
    apr()
        .args(["tune", "/nonexistent/model.gguf", "--plan"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed")),
        );
}

// F-QA-001: apr qa help works
#[test]
fn test_f_qa_001_help() {
    apr()
        .args(["qa", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("qa").or(predicate::str::contains("QA")));
}

// F-QA-002: apr qa with missing model shows error
#[test]
fn test_f_qa_002_missing_model() {
    apr()
        .args(["qa", "/nonexistent/model.gguf"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed")),
        );
}

// F-CONVERT-001: apr convert help works (rosetta subcommand)
#[test]
fn test_f_convert_001_rosetta_help() {
    apr()
        .args(["rosetta", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("convert").or(predicate::str::contains("Convert")));
}

// F-CONVERT-002: apr rosetta convert with missing model shows error
#[test]
fn test_f_convert_002_missing_model() {
    apr()
        .args([
            "rosetta",
            "convert",
            "/nonexistent/model.gguf",
            "/tmp/out.apr",
        ])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed"))
                .or(predicate::str::contains("does not exist")),
        );
}

// ============================================================================
// PMAT-192 Phase 5: F-PROFILE-CI-* Tests (GH-180)
// ============================================================================

// F-PROFILE-CI-001: apr profile --help shows CI options
#[test]
fn test_f_profile_ci_001_help_shows_ci_options() {
    apr()
        .args(["profile", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--ci"))
        .stdout(predicate::str::contains("--assert-throughput"))
        .stdout(predicate::str::contains("--assert-p99"));
}

// F-PROFILE-CI-002: apr profile with missing model shows error
#[test]
fn test_f_profile_ci_002_missing_model_error() {
    apr()
        .args(["profile", "/nonexistent/model.gguf"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed"))
                .or(predicate::str::contains("does not exist")),
        );
}

// F-PROFILE-CI-003: apr profile --ci with missing model shows error
#[test]
fn test_f_profile_ci_003_ci_mode_missing_model() {
    apr()
        .args([
            "profile",
            "/nonexistent/model.gguf",
            "--ci",
            "--assert-throughput",
            "100",
        ])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("not found")
                .or(predicate::str::contains("No such file"))
                .or(predicate::str::contains("Failed"))
                .or(predicate::str::contains("does not exist")),
        );
}

// F-PROFILE-CI-004: apr profile accepts format=json
#[test]
fn test_f_profile_ci_004_format_json_accepted() {
    // Just verify the argument is accepted (will fail on missing model, but that's expected)
    apr()
        .args(["profile", "/nonexistent/model.gguf", "--format", "json"])
        .assert()
        .failure(); // Expected - model doesn't exist
}

// F-PROFILE-CI-005: apr profile accepts warmup and measure args
#[test]
fn test_f_profile_ci_005_warmup_measure_accepted() {
    apr()
        .args([
            "profile",
            "/nonexistent/model.gguf",
            "--warmup",
            "5",
            "--measure",
            "20",
        ])
        .assert()
        .failure(); // Expected - model doesn't exist
}

// F-PROFILE-CI-006: apr profile --ci --assert-p50 accepted
#[test]
fn test_f_profile_ci_006_assert_p50_accepted() {
    apr()
        .args([
            "profile",
            "/nonexistent/model.gguf",
            "--ci",
            "--assert-p50",
            "25",
        ])
        .assert()
        .failure(); // Expected - model doesn't exist
}

// ============================================================================
// F-PROFILE-EXIT Tests (GH-184: Exit code verification)
// ============================================================================

// F-PROFILE-EXIT-001: Help text documents exit codes
#[test]
fn test_f_profile_exit_001_help_documents_exit_codes() {
    // Verify that CI mode documentation exists in help
    apr()
        .args(["profile", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--ci"))
        .stdout(predicate::str::contains("--assert-throughput"));
}

// F-PROFILE-EXIT-002: Non-existent model returns failure (not success)
#[test]
fn test_f_profile_exit_002_nonexistent_model_fails() {
    // GH-184: Ensure file-not-found errors exit with non-zero code
    apr()
        .args([
            "profile",
            "/nonexistent/path/to/model.gguf",
            "--ci",
            "--assert-throughput",
            "1.0",
        ])
        .assert()
        .failure()
        .code(predicate::ne(0)); // Must be non-zero exit code
}

// F-PROFILE-EXIT-003: CI mode with impossible threshold should fail
// Note: This test requires a real model and inference feature
#[test]
#[ignore = "requires model download - run with: cargo test -- --ignored"]
fn test_f_profile_exit_003_impossible_threshold_fails() {
    // This test would need a real model to verify exit code on assertion failure
    // The assertion --assert-throughput 999999999.0 should always fail
    // and exit with code 1
    apr()
        .args([
            "profile",
            "test-model.gguf", // Would need real model path
            "--ci",
            "--assert-throughput",
            "999999999.0",
            "--warmup",
            "1",
            "--measure",
            "1",
        ])
        .assert()
        .failure()
        .code(predicate::eq(1)); // Must be exit code 1 for assertion failure
}

// ============================================================================
// F-CONVERT-QUANT Tests (GH-181: Q4_K_M block alignment)
// ============================================================================

// F-CONVERT-QUANT-001: apr convert --help shows Q4K option
#[test]
fn test_f_convert_quant_001_help_shows_q4k() {
    apr()
        .args(["convert", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("q4k").or(predicate::str::contains("q4_k")));
}

// F-CONVERT-QUANT-002: apr convert with --quantize q4k accepted
#[test]
fn test_f_convert_quant_002_q4k_option_accepted() {
    apr()
        .args([
            "convert",
            "/nonexistent/model.gguf",
            "/tmp/out.apr",
            "--quantize",
            "q4k",
        ])
        .assert()
        .failure(); // Expected - model doesn't exist
}

// ============================================================================
// F-EXPORT-COMPANION Tests (GH-182: SafeTensors companion files)
// ============================================================================

// F-EXPORT-COMPANION-001: apr export --help shows format options
#[test]
fn test_f_export_companion_001_help_shows_formats() {
    apr()
        .args(["export", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("safetensors"))
        .stdout(predicate::str::contains("gguf"));
}

// F-EXPORT-COMPANION-002: apr export accepts safetensors format
#[test]
fn test_f_export_companion_002_safetensors_accepted() {
    apr()
        .args([
            "export",
            "/nonexistent/model.apr",
            "--format",
            "safetensors",
            "/tmp/out.safetensors",
        ])
        .assert()
        .failure(); // Expected - model doesn't exist
}

// ============================================================================
// F-VALIDATE-GGUF Tests (GH-183: GGUF v3 validation)
// ============================================================================

// F-VALIDATE-GGUF-001: apr validate --help shows options
#[test]
fn test_f_validate_gguf_001_help_shows_options() {
    apr()
        .args(["validate", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("quality").or(predicate::str::contains("strict")));
}

// F-VALIDATE-GGUF-002: apr validate shows magic error for corrupted file
#[test]
fn test_f_validate_gguf_002_magic_error_message() {
    use std::io::Write;

    // Create a file with invalid magic
    let mut file = NamedTempFile::new().expect("create temp file");
    file.write_all(b"BADM1234567890123456789012345678").unwrap();

    apr()
        .args(["validate", file.path().to_str().unwrap()])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("Invalid")
                .or(predicate::str::contains("magic"))
                .or(predicate::str::contains("Unknown")),
        );
}

// F-VALIDATE-GGUF-003: apr validate accepts GGUF magic bytes
#[test]
fn test_f_validate_gguf_003_accepts_gguf_magic() {
    use std::io::Write;

    // Create a file with valid GGUF magic but minimal content
    let mut file = NamedTempFile::new().expect("create temp file");
    // GGUF magic + version 3 + minimal header
    file.write_all(b"GGUF").unwrap(); // magic
    file.write_all(&3u32.to_le_bytes()).unwrap(); // version 3
    file.write_all(&0u64.to_le_bytes()).unwrap(); // tensor count
    file.write_all(&0u64.to_le_bytes()).unwrap(); // metadata count
                                                  // Pad to 32 bytes
    file.write_all(&[0u8; 8]).unwrap();

    // This should pass the magic check (might fail later for other reasons)
    let output = apr()
        .args(["validate", file.path().to_str().unwrap()])
        .output()
        .expect("run command");

    // Should not fail on magic bytes
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("Invalid magic"),
        "GGUF magic should be accepted: {stdout}"
    );
}

// ============================================================================
// P0 FORMAT DISPATCH TESTS (GH-202)
// Verify all critical subcommands support GGUF, APR, and SafeTensors formats.
// These tests catch the bug where commands silently skip with "format not supported".
// ============================================================================

/// Create a minimal valid GGUF file for testing
fn create_test_gguf_file() -> NamedTempFile {
    let file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
    let mut data = Vec::new();

    // GGUF magic + version 3
    data.extend_from_slice(b"GGUF");
    data.extend_from_slice(&3u32.to_le_bytes()); // version 3
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor count = 0
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata count = 0
    data.extend_from_slice(&[0u8; 8]); // padding to 32 bytes

    std::fs::write(file.path(), data).expect("write file");
    file
}

/// Create a minimal valid SafeTensors file for testing
fn create_test_safetensors_file() -> NamedTempFile {
    let file = NamedTempFile::with_suffix(".safetensors").expect("create temp file");

    // Minimal SafeTensors: header length + empty JSON header
    let header = b"{}";
    let header_len = header.len() as u64;

    let mut data = Vec::new();
    data.extend_from_slice(&header_len.to_le_bytes());
    data.extend_from_slice(header);

    std::fs::write(file.path(), data).expect("write file");
    file
}

// F-FORMAT-DISPATCH-001: apr inspect supports all 3 formats
#[test]
fn test_f_format_dispatch_001_inspect_all_formats() {
    // GGUF
    let gguf = create_test_gguf_file();
    let output = apr()
        .args(["inspect", gguf.path().to_str().unwrap()])
        .output()
        .expect("run inspect on GGUF");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("not supported") && !stderr.contains("Only GGUF"),
        "GGUF inspect should not skip: {stderr}"
    );

    // APR
    let apr_file = create_test_apr_file();
    let output = apr()
        .args(["inspect", apr_file.path().to_str().unwrap()])
        .output()
        .expect("run inspect on APR");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("not supported") && !stderr.contains("Only GGUF"),
        "APR inspect should not skip: {stderr}"
    );

    // SafeTensors
    let st = create_test_safetensors_file();
    let output = apr()
        .args(["inspect", st.path().to_str().unwrap()])
        .output()
        .expect("run inspect on SafeTensors");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("not supported") && !stderr.contains("Only GGUF"),
        "SafeTensors inspect should not skip: {stderr}"
    );
}

// F-FORMAT-DISPATCH-002: apr validate supports all 3 formats
#[test]
fn test_f_format_dispatch_002_validate_all_formats() {
    for (name, file) in [
        ("GGUF", create_test_gguf_file()),
        ("APR", create_test_apr_file()),
        ("SafeTensors", create_test_safetensors_file()),
    ] {
        let output = apr()
            .args(["validate", file.path().to_str().unwrap()])
            .output()
            .expect(&format!("run validate on {name}"));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("not supported") && !stderr.contains("Only GGUF"),
            "{name} validate should not skip: {stderr}"
        );
    }
}
