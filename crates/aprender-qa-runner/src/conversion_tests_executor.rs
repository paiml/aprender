#[test]
fn test_conversion_execution_result_fields() {
    let result = ConversionExecutionResult {
        total: 10,
        passed: 5,
        failed: 2,
        duration_ms: 100,
        results: vec![],
        evidence: vec![],
    };
    assert_eq!(result.total, 10);
    assert_eq!(result.passed, 5);
    assert_eq!(result.failed, 2);
    assert_eq!(result.duration_ms, 100);
    assert!(result.results.is_empty());
    assert!(result.evidence.is_empty());
}

#[test]
fn test_conversion_test_compute_diff_same() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Same strings should have 0 diff
    assert!((test.compute_diff("hello", "hello") - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_conversion_test_compute_diff_different() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Completely different strings should have high diff
    let diff = test.compute_diff("abc", "xyz");
    assert!(diff > 0.5);
}

#[test]
fn test_conversion_test_compute_diff_empty_strings() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Empty strings should have 0 diff
    assert!((test.compute_diff("", "") - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_conversion_test_compute_diff_partial_match() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Partially matching strings
    let diff = test.compute_diff("abcd", "abXd");
    assert!(diff > 0.0 && diff < 1.0);
}

#[test]
fn test_conversion_test_find_diff_indices_with_diffs() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let indices = test.find_diff_indices("abcd", "aXcY");
    assert_eq!(indices.len(), 2);
    assert!(indices.contains(&1));
    assert!(indices.contains(&3));
}

#[test]
fn test_conversion_test_find_diff_indices_no_diffs() {
    let test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    let indices = test.find_diff_indices("same", "same");
    assert!(indices.is_empty());
}

#[test]
fn test_conversion_test_hash_output_consistency() {
    let hash1 = ConversionTest::hash_output("test string");
    let hash2 = ConversionTest::hash_output("test string");
    let hash3 = ConversionTest::hash_output("different string");

    // Same input should produce same hash
    assert_eq!(hash1, hash2);
    // Different input should produce different hash
    assert_ne!(hash1, hash3);
    // Hash should be 16 hex characters
    assert_eq!(hash1.len(), 16);
}

#[test]
fn test_classify_bug_empty_source_nonempty_target() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // If source is empty/whitespace and target is not, classify as unknown
    let bug = test.classify_bug("  ", "some output", false);
    assert_eq!(bug, Some(ConversionBugType::Unknown));
}

#[test]
fn test_classify_bug_both_empty_strings() {
    let test = SemanticConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    // Both empty should match
    let bug = test.classify_bug("", "", false);
    assert!(bug.is_none());
}

#[test]
fn test_generate_conversion_tests_full_count() {
    let model_id = ModelId::new("test", "model");
    let tests = generate_conversion_tests(&model_id);

    // 6 pairs x 2 backends = 12 tests
    assert_eq!(tests.len(), 12);
}

// ── Mock binary tests ────────────────────────────────────────────


/// Wait until a just-written script is actually spawnable.
///
/// `fs::write` closes our handle, but a CONCURRENT FORK elsewhere in the test
/// binary can inherit that write fd and hold it until its own exec. Spawning in
/// that window fails with ETXTBSY ("Text file busy") — observed on a loaded box
/// as `Expected Corroborated, got: Err(Io(Os { code: 26, kind: ExecutableFileBusy }))`
/// in `test_commutativity_execute_corroborated`. O_CLOEXEC closes the fd at the
/// child's exec, not before, so the window is real and only opens under load.
///
/// Absorbing it HERE, in the fixture, keeps the retry out of production code: the
/// code under test spawns exactly once, as it does in the field.
#[cfg(unix)]
fn wait_until_spawnable(path: &std::path::Path) {
    const ETXTBSY: i32 = 26;
    for _ in 0..100 {
        match std::process::Command::new(path).arg("--\u{2060}probe").output() {
            Err(e) if e.raw_os_error() == Some(ETXTBSY) => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            _ => return,
        }
    }
    panic!("mock at {} still ETXTBSY after 100 attempts", path.display());
}

fn create_mock_apr(dir: &std::path::Path, script: &str) -> std::path::PathBuf {
    let path = dir.join("mock_apr");
    std::fs::write(&path, format!("#!/bin/bash\n{script}")).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    // Flush filesystem metadata to avoid ETXTBSY in Docker overlayfs (CI containers)
    let _ = std::fs::File::open(&path).and_then(|f| f.sync_all());
    #[cfg(unix)]
    wait_until_spawnable(&path);
    path
}

#[test]
fn test_conversion_test_execute_corroborated_via_mock() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
    );

    let mut test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    test.binary = mock.to_string_lossy().to_string();

    if let Ok(conv) = test.execute(&model_file) {
        match conv {
            ConversionResult::Corroborated { max_diff, .. } => {
                assert!(max_diff < EPSILON);
            }
            ConversionResult::Falsified { .. } => {}
        }
    }
}

#[test]
fn test_conversion_test_execute_falsified_via_mock() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
run)
  case "$2" in
  *converted*) printf "Completely different output 99";;
  *) printf "The answer is 4";;
  esac
  exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
    );

    let mut test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    test.binary = mock.to_string_lossy().to_string();

    if let Ok(conv) = test.execute(&model_file) {
        match conv {
            ConversionResult::Falsified {
                gate_id, evidence, ..
            } => {
                assert_eq!(gate_id, "F-CONV-G-A");
                assert!(evidence.max_diff > EPSILON);
                assert_ne!(evidence.source_hash, evidence.converted_hash);
            }
            ConversionResult::Corroborated { .. } => {}
        }
    }
}

#[test]
fn test_conversion_test_execute_gpu_backend_via_mock() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.safetensors");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
    );

    let mut test = ConversionTest::new(
        Format::SafeTensors,
        Format::Gguf,
        Backend::Gpu,
        ModelId::new("test", "model"),
    );
    test.binary = mock.to_string_lossy().to_string();

    if let Ok(ConversionResult::Corroborated { backend, .. }) = &test.execute(&model_file) {
        assert_eq!(*backend, Backend::Gpu);
    }
}

#[test]
fn test_conversion_test_convert_model_failure_via_mock() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) printf "conversion error" >&2; exit 1;;
esac
exit 1"#,
    );

    let mut test = ConversionTest::new(
        Format::Gguf,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    test.binary = mock.to_string_lossy().to_string();

    if let Err(e) = test.execute(&model_file) {
        let msg = e.to_string();
        assert!(msg.contains("Conversion failed") || msg.contains("conversion error"));
    }
}

#[test]
fn test_semantic_test_execute_corroborated_via_mock() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.safetensors");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
    );

    let mut test = SemanticConversionTest::new(
        Format::SafeTensors,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    test.binary = mock.to_string_lossy().to_string();

    if let Ok(sem) = test.execute(&model_file) {
        if let SemanticTestResult::Corroborated {
            source_output,
            target_output,
        } = &sem
        {
            assert_eq!(source_output, target_output);
            assert!(sem.is_pass());
            assert!(sem.bug_type().is_none());
        }
    }
}

#[test]
fn test_semantic_test_execute_embedding_transposition_via_mock() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.safetensors");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
run)
  case "$2" in
  *semantic_test*) printf "PAD PAD PAD garbage tokens";;
  *) printf "The answer is 4";;
  esac
  exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
    );

    let mut test = SemanticConversionTest::new(
        Format::SafeTensors,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    test.binary = mock.to_string_lossy().to_string();

    if let Ok(sem) = test.execute(&model_file) {
        if let SemanticTestResult::Falsified { bug_type, .. } = &sem {
            assert_eq!(*bug_type, ConversionBugType::EmbeddingTransposition);
            assert!(!sem.is_pass());
        }
    }
}

#[test]
fn test_semantic_test_execute_tokenizer_missing_via_mock() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.safetensors");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
run)
  case "$2" in
  *semantic_test*) printf "output" >&1; printf "PMAT-172: missing embedded tokenizer" >&2;;
  *) printf "The answer is 4";;
  esac
  exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
    );

    let mut test = SemanticConversionTest::new(
        Format::SafeTensors,
        Format::Apr,
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    test.binary = mock.to_string_lossy().to_string();

    if let Ok(SemanticTestResult::Falsified {
        bug_type, stderr, ..
    }) = &test.execute(&model_file)
    {
        assert_eq!(*bug_type, ConversionBugType::TokenizerMissing);
        assert!(stderr.contains("PMAT-172"));
    }
}

#[test]
fn test_round_trip_execute_corroborated_via_mock() {
    let dir = tempfile::tempdir().unwrap();
    let model_file = dir.path().join("model.gguf");
    std::fs::write(&model_file, "fake").unwrap();

    let mock = create_mock_apr(
        dir.path(),
        r#"case "$1" in
run) printf "The answer is 4"; exit 0;;
rosetta) touch "$4"; exit 0;;
esac
exit 1"#,
    );

    let mut rt = RoundTripTest::new(
        vec![Format::Gguf, Format::Apr, Format::SafeTensors, Format::Gguf],
        Backend::Cpu,
        ModelId::new("test", "model"),
    );
    rt.binary = mock.to_string_lossy().to_string();

    if let Ok(ConversionResult::Corroborated { max_diff, .. }) = rt.execute(&model_file) {
        assert!((max_diff - 0.0).abs() < f64::EPSILON);
    }
}
