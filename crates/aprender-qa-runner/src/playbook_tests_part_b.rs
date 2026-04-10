/// Verify default_tolerance returns 1e-5
#[test]
fn test_default_tolerance() {
    assert!((default_tolerance() - 1e-5).abs() < 1e-10);
}

/// Verify default_warmup returns 3
#[test]
fn test_default_warmup() {
    assert_eq!(default_warmup(), 3);
}

/// Verify default_measure returns 10
#[test]
fn test_default_measure() {
    assert_eq!(default_measure(), 10);
}

/// Verify playbook parsing with fingerprint differential test config
#[test]
fn test_playbook_with_fingerprint() {
    let yaml = r#"
name: fingerprint-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
differential_tests:
  fingerprint:
    enabled: true
    tensors: "embed,lm_head"
    stats: ["mean", "std", "checksum"]
    gates: ["F-ROSETTA-FP-001", "F-ROSETTA-FP-002"]
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let diff = playbook
        .differential_tests
        .expect("Should have differential tests");

    let fp = diff.fingerprint.expect("Should have fingerprint");
    assert!(fp.enabled);
    assert_eq!(fp.tensors, "embed,lm_head");
    assert_eq!(fp.stats.len(), 3);
    assert_eq!(fp.gates.len(), 2);
}

/// Verify playbook parsing with validate_stats differential test config
#[test]
fn test_playbook_with_validate_stats() {
    let yaml = r#"
name: validate-stats-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
differential_tests:
  validate_stats:
    enabled: true
    reference: "reference.json"
    tolerance:
      layernorm: 0.001
      embedding: 0.1
      attention: 0.01
    gates: ["F-ROSETTA-STATS-001", "F-ROSETTA-STATS-002"]
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let diff = playbook
        .differential_tests
        .expect("Should have differential tests");

    let stats = diff.validate_stats.expect("Should have validate_stats");
    assert!(stats.enabled);
    assert_eq!(stats.reference, Some("reference.json".to_string()));
    assert!((stats.tolerance.layernorm - 0.001).abs() < 1e-10);
    assert!((stats.tolerance.embedding - 0.1).abs() < 1e-10);
    assert!((stats.tolerance.attention - 0.01).abs() < 1e-10);
    assert_eq!(stats.gates.len(), 2);
}

/// Verify default fingerprint tensors is "all"
#[test]
fn test_default_fingerprint_tensors() {
    assert_eq!(default_fingerprint_tensors(), "all");
}

/// Verify default fingerprint stats contains 5 stat types
#[test]
fn test_default_fingerprint_stats() {
    let stats = default_fingerprint_stats();
    assert_eq!(stats.len(), 5);
    assert!(stats.contains(&"mean".to_string()));
    assert!(stats.contains(&"checksum".to_string()));
}

/// Verify default tolerance values for layernorm, embedding, and attention
#[test]
fn test_default_tolerance_values() {
    assert!((default_layernorm_tolerance() - 0.001).abs() < 1e-10);
    assert!((default_embedding_tolerance() - 0.1).abs() < 1e-10);
    assert!((default_attention_tolerance() - 0.01).abs() < 1e-10);
}

/// Verify ProfileCiAssertions returns backend-specific throughput thresholds
#[test]
fn test_profile_ci_min_throughput_for() {
    // Test with all fields set
    let assertions = ProfileCiAssertions {
        min_throughput: Some(10.0),
        min_throughput_cpu: Some(5.0),
        min_throughput_gpu: Some(50.0),
        max_p99_ms: None,
        max_p50_ms: None,
    };

    assert_eq!(assertions.min_throughput_for("cpu"), Some(5.0));
    assert_eq!(assertions.min_throughput_for("gpu"), Some(50.0));
    assert_eq!(assertions.min_throughput_for("tpu"), Some(10.0));

    // Test with only min_throughput set (fallback)
    let assertions_fallback = ProfileCiAssertions {
        min_throughput: Some(20.0),
        min_throughput_cpu: None,
        min_throughput_gpu: None,
        max_p99_ms: None,
        max_p50_ms: None,
    };

    assert_eq!(assertions_fallback.min_throughput_for("cpu"), Some(20.0));
    assert_eq!(assertions_fallback.min_throughput_for("gpu"), Some(20.0));

    // Test with nothing set
    let assertions_none = ProfileCiAssertions {
        min_throughput: None,
        min_throughput_cpu: None,
        min_throughput_gpu: None,
        max_p99_ms: None,
        max_p50_ms: None,
    };

    assert_eq!(assertions_none.min_throughput_for("cpu"), None);
    assert_eq!(assertions_none.min_throughput_for("gpu"), None);
}

// ── §3.1 Playbook integrity lock tests ─────────────────────────────

/// Verify compute_playbook_hash produces consistent SHA-256 hex output
#[test]
fn test_compute_playbook_hash_consistent() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("test.playbook.yaml");
    std::fs::write(&path, "name: test\nversion: 1.0").expect("write");

    let hash1 = compute_playbook_hash(&path).expect("hash1");
    let hash2 = compute_playbook_hash(&path).expect("hash2");
    assert_eq!(hash1, hash2);
    assert_eq!(hash1.len(), 64); // SHA-256 hex
}

/// Verify compute_playbook_hash produces different hashes for different content
#[test]
fn test_compute_playbook_hash_differs() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path1 = dir.path().join("a.yaml");
    let path2 = dir.path().join("b.yaml");
    std::fs::write(&path1, "content-a").expect("write");
    std::fs::write(&path2, "content-b").expect("write");

    let hash1 = compute_playbook_hash(&path1).expect("hash1");
    let hash2 = compute_playbook_hash(&path2).expect("hash2");
    assert_ne!(hash1, hash2);
}

/// Verify verify_playbook_integrity passes when hash matches lock file
#[test]
fn test_verify_playbook_integrity_pass() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("test.playbook.yaml");
    std::fs::write(&path, "name: test\nversion: 1.0").expect("write");

    let hash = compute_playbook_hash(&path).expect("hash");
    let mut lock = PlaybookLockFile::default();
    lock.entries.insert(
        "test".to_string(),
        PlaybookLockEntry {
            sha256: hash,
            locked_fields: vec!["model.hf_repo".to_string()],
        },
    );

    assert!(verify_playbook_integrity(&path, &lock, "test").is_ok());
}

/// Verify verify_playbook_integrity fails on hash mismatch
#[test]
fn test_verify_playbook_integrity_fail_mismatch() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("test.playbook.yaml");
    std::fs::write(&path, "name: test\nversion: 1.0").expect("write");

    let mut lock = PlaybookLockFile::default();
    lock.entries.insert(
        "test".to_string(),
        PlaybookLockEntry {
            sha256: "wrong_hash".to_string(),
            locked_fields: vec![],
        },
    );

    let result = verify_playbook_integrity(&path, &lock, "test");
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Integrity check failed")
    );
}

/// Verify verify_playbook_integrity fails when entry is missing from lock file
#[test]
fn test_verify_playbook_integrity_missing_entry() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("test.playbook.yaml");
    std::fs::write(&path, "name: test").expect("write");

    let lock = PlaybookLockFile::default();
    let result = verify_playbook_integrity(&path, &lock, "test");
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("not found in lock file")
    );
}

/// Verify generate_lock_entry extracts name and computes hash
#[test]
fn test_generate_lock_entry() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("my-model.playbook.yaml");
    std::fs::write(&path, "name: my-model\nversion: 1.0").expect("write");

    let (name, entry) = generate_lock_entry(&path).expect("generate");
    assert_eq!(name, "my-model");
    assert_eq!(entry.sha256.len(), 64);
    assert!(!entry.locked_fields.is_empty());
}

/// Verify lock file survives save/load round-trip
#[test]
fn test_lock_file_save_load_roundtrip() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let lock_path = dir.path().join("playbook.lock.yaml");

    let mut lock = PlaybookLockFile::default();
    lock.entries.insert(
        "model-a".to_string(),
        PlaybookLockEntry {
            sha256: "abc123".to_string(),
            locked_fields: vec!["model.hf_repo".to_string()],
        },
    );

    save_lock_file(&lock, &lock_path).expect("save");
    let loaded = load_lock_file(&lock_path).expect("load");

    assert_eq!(loaded.entries.len(), 1);
    assert_eq!(loaded.entries["model-a"].sha256, "abc123");
}

/// Verify lock file survives serde YAML round-trip
#[test]
fn test_lock_file_serde_roundtrip() {
    let mut lock = PlaybookLockFile::default();
    lock.entries.insert(
        "test".to_string(),
        PlaybookLockEntry {
            sha256: "deadbeef".to_string(),
            locked_fields: vec!["a".to_string(), "b".to_string()],
        },
    );

    let yaml = serde_yaml::to_string(&lock).expect("serialize");
    let parsed: PlaybookLockFile = serde_yaml::from_str(&yaml).expect("deserialize");
    assert_eq!(parsed.entries["test"].sha256, "deadbeef");
    assert_eq!(parsed.entries["test"].locked_fields.len(), 2);
}

// ── §3.3 Skip mechanism tests ──────────────────────────────────────

/// Verify find_skip_files returns empty for directory with no skip files
#[test]
fn test_find_skip_files_empty_dir() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let skips = find_skip_files(dir.path(), "test-model");
    assert!(skips.is_empty());
}

/// Verify find_skip_files parses skip YAML correctly
#[test]
fn test_find_skip_files_with_skip() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let skip_path = dir.path().join("test-model.skip.yaml");
    std::fs::write(
        &skip_path,
        r#"- format_or_backend: gpu
  reason: "No GPU available"
  tracking_issue: "GH-123"
"#,
    )
    .expect("write");

    let skips = find_skip_files(dir.path(), "test-model");
    assert_eq!(skips.len(), 1);
    assert_eq!(skips[0].format_or_backend, "gpu");
    assert_eq!(skips[0].tracking_issue.as_deref(), Some("GH-123"));
}

/// Verify detect_implicit_skips identifies formats missing from playbook
#[test]
fn test_detect_implicit_skips() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let all = vec![Format::Gguf, Format::SafeTensors, Format::Apr];
    let skips: Vec<SkipReason> = vec![];
    let implicit = detect_implicit_skips(&playbook, &all, &skips);
    // safetensors and apr are missing from playbook formats
    assert_eq!(implicit.len(), 2);
    assert!(implicit.contains(&"safetensors".to_string()));
    assert!(implicit.contains(&"apr".to_string()));
}

/// Verify detect_implicit_skips excludes explicitly skipped formats
#[test]
fn test_detect_implicit_skips_with_explicit() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let all = vec![Format::Gguf, Format::SafeTensors, Format::Apr];
    // safetensors is explicitly skipped
    let skips = vec![SkipReason {
        format_or_backend: "safetensors".to_string(),
        reason: "Not supported".to_string(),
        tracking_issue: None,
    }];
    let implicit = detect_implicit_skips(&playbook, &all, &skips);
    // Only apr is implicitly skipped
    assert_eq!(implicit.len(), 1);
    assert_eq!(implicit[0], "apr");
}

/// Verify detect_implicit_skips returns empty when all formats covered
#[test]
fn test_detect_implicit_skips_all_covered() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf, safetensors, apr]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    let all = vec![Format::Gguf, Format::SafeTensors, Format::Apr];
    let skips: Vec<SkipReason> = vec![];
    let implicit = detect_implicit_skips(&playbook, &all, &skips);
    assert!(implicit.is_empty());
}

/// Verify SkipReason serialization round-trip preserves all fields
#[test]
fn test_skip_reason_serde() {
    let reason = SkipReason {
        format_or_backend: "gpu".to_string(),
        reason: "No GPU".to_string(),
        tracking_issue: Some("GH-100".to_string()),
    };
    let json = serde_json::to_string(&reason).expect("serialize");
    let parsed: SkipReason = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.format_or_backend, "gpu");
    assert_eq!(parsed.tracking_issue.as_deref(), Some("GH-100"));
}

/// Verify SkipType equality comparison
#[test]
fn test_skip_type_eq() {
    assert_eq!(SkipType::Explicit, SkipType::Explicit);
    assert_ne!(SkipType::Explicit, SkipType::Implicit);
}

// ── §3.4 Resource-aware scheduling tests ────────────────────────────

/// Verify SizeCategory max_workers for each tier
#[test]
fn test_size_category_max_workers() {
    assert_eq!(SizeCategory::Tiny.max_workers(), 4);
    assert_eq!(SizeCategory::Small.max_workers(), 4);
    assert_eq!(SizeCategory::Medium.max_workers(), 2);
    assert_eq!(SizeCategory::Large.max_workers(), 1);
    assert_eq!(SizeCategory::Xlarge.max_workers(), 1);
    assert_eq!(SizeCategory::Huge.max_workers(), 1);
}

/// Verify SizeCategory estimated memory for each tier
#[test]
fn test_size_category_estimated_memory() {
    assert_eq!(SizeCategory::Tiny.estimated_memory_gb(), 2);
    assert_eq!(SizeCategory::Small.estimated_memory_gb(), 4);
    assert_eq!(SizeCategory::Medium.estimated_memory_gb(), 8);
    assert_eq!(SizeCategory::Large.estimated_memory_gb(), 16);
    assert_eq!(SizeCategory::Xlarge.estimated_memory_gb(), 32);
    assert_eq!(SizeCategory::Huge.estimated_memory_gb(), 64);
}

/// Verify SizeCategory concurrent execution eligibility
#[test]
fn test_size_category_can_run_concurrent() {
    assert!(SizeCategory::Tiny.can_run_concurrent());
    assert!(SizeCategory::Small.can_run_concurrent());
    assert!(!SizeCategory::Medium.can_run_concurrent());
    assert!(!SizeCategory::Large.can_run_concurrent());
    assert!(!SizeCategory::Xlarge.can_run_concurrent());
    assert!(!SizeCategory::Huge.can_run_concurrent());
}

/// Verify SizeCategory default is Tiny
#[test]
fn test_size_category_default() {
    assert_eq!(SizeCategory::default(), SizeCategory::Tiny);
}

/// Verify SizeCategory YAML deserialization from playbook
#[test]
fn test_size_category_serde() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  size_category: large
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("parse");
    assert_eq!(playbook.model.size_category, SizeCategory::Large);
}
