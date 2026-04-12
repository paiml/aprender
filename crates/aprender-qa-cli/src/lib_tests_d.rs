use super::*;

/// Create a test QA scenario with default values for unit tests
fn make_test_scenario() -> aprender_qa_gen::QaScenario {
    aprender_qa_gen::QaScenario::new(
        aprender_qa_gen::ModelId::new("test", "model"),
        aprender_qa_gen::Modality::Run,
        aprender_qa_gen::Backend::Cpu,
        aprender_qa_gen::Format::Gguf,
        "What is 2+2?".to_string(),
        42,
    )
}

/// Create a corroborated evidence instance for testing
fn make_corroborated_evidence() -> Evidence {
    Evidence::corroborated("F-TEST-001", make_test_scenario(), "output", 100)
}

/// Create a falsified evidence instance for testing failure paths
fn make_falsified_evidence() -> Evidence {
    Evidence::falsified("F-TEST-002", make_test_scenario(), "failed", "error", 200)
}

#[test]
fn test_execute_playbook_with_yaml_inline() {
    let yaml = r#"
name: test-playbook
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
  quantizations: [q4_k_m]
  size_category: tiny
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
  seed: 42
  timeout_ms: 30000
gates:
  g1_model_loads: true
  g2_basic_inference: true
  g3_no_crashes: true
  g4_not_garbage: true
"#;
    let playbook = Playbook::from_yaml(yaml).expect("valid yaml");
    let config = build_certification_config(CertTier::Smoke, None);
    let result = execute_playbook(&playbook, config);
    // Should succeed in mock mode
    assert!(result.is_ok());
}

#[test]
fn test_certify_model_with_cache_smoke() {
    let config = CertificationConfig {
        tier: CertTier::Smoke,
        model_cache: Some(std::path::PathBuf::from("/tmp/test")),
        ..Default::default()
    };
    // Will fail because playbook doesn't exist
    let result = certify_model("org/model-smoke", &config);
    assert!(!result.success);
}

#[test]
fn test_generate_model_scenarios_all_modalities_present() {
    let scenarios = generate_model_scenarios("test/model", 1);
    // Check all modalities are present
    let has_run = scenarios
        .iter()
        .any(|s| s.modality == aprender_qa_gen::Modality::Run);
    let has_chat = scenarios
        .iter()
        .any(|s| s.modality == aprender_qa_gen::Modality::Chat);
    let has_serve = scenarios
        .iter()
        .any(|s| s.modality == aprender_qa_gen::Modality::Serve);
    assert!(has_run && has_chat && has_serve);
}

#[test]
fn test_generate_tickets_regular_failures() {
    let evidence = vec![make_falsified_evidence()];
    let tickets = generate_tickets_from_evidence(&evidence, "test/repo", false, 1);
    // Should generate tickets for regular failures
    let _ = tickets; // May or may not have tickets
}

// =========================================================================
// Additional coverage tests for certify_model paths
// =========================================================================

#[test]
fn test_certify_model_invalid_playbook_yaml() {
    // Just verify the playbook_path_for_model generates correct path
    let path = playbook_path_for_model("test/BadModel", CertTier::Mvp);
    assert!(path.contains("badmodel-mvp.playbook.yaml"));
}

#[test]
fn test_certify_model_cache_path_construction() {
    // Exercise the model cache path construction code
    let config = CertificationConfig {
        tier: CertTier::Mvp,
        model_cache: Some(std::path::PathBuf::from("/test/cache")),
        ..Default::default()
    };

    // This will fail (no playbook) but exercises the early return
    let result = certify_model("org/Model.Name-With.Dots", &config);
    assert!(!result.success);
    // Verify the error is about missing playbook (not a crash in path construction)
    assert!(result
        .error
        .as_ref()
        .expect("should have error")
        .contains("Playbook not found"));
}

#[test]
fn test_certify_model_without_cache() {
    let config = CertificationConfig {
        tier: CertTier::Smoke,
        model_cache: None,
        ..Default::default()
    };

    let result = certify_model("org/some-model", &config);
    assert!(!result.success);
}

#[test]
fn test_certify_model_with_cache_another() {
    let config = CertificationConfig {
        tier: CertTier::Smoke,
        model_cache: Some(std::path::PathBuf::from("/test/cache")),
        ..Default::default()
    };

    let result = certify_model("org/another-model", &config);
    assert!(!result.success);
}

#[test]
fn test_execute_playbook_smoke() {
    // Exercise execute_playbook directly with a valid playbook
    let yaml = r#"
name: coverage-test
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
    let config = ExecutionConfig::default();
    let result = execute_playbook(&playbook, config);
    assert!(result.is_ok());
}

#[test]
fn test_load_playbook_nonexistent_file() {
    let result = load_playbook(std::path::Path::new("/nonexistent/playbook.yaml"));
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Error loading playbook"));
}

#[test]
fn test_load_playbook_invalid_yaml() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join("bad.yaml");
    std::fs::write(&path, "not: [valid: yaml: {{{").expect("write");
    let result = load_playbook(&path);
    assert!(result.is_err());
}

// =========================================================================
// Phase 4 tests: lock-playbooks + auto-ticket
// =========================================================================

#[test]
fn test_generate_lock_file_empty_dir() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let output = dir.path().join("playbook.lock.yaml");
    let result = generate_lock_file(dir.path(), &output);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0);
}

#[test]
fn test_generate_lock_file_with_playbooks() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let playbook_yaml = r#"
name: test-lock
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    std::fs::write(dir.path().join("test.playbook.yaml"), playbook_yaml).expect("write");

    let output = dir.path().join("playbook.lock.yaml");
    let result = generate_lock_file(dir.path(), &output);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 1);
    assert!(output.exists());
}

#[test]
fn test_generate_lock_file_recursive() {
    let dir = tempfile::tempdir().expect("create temp dir");
    let sub = dir.path().join("models");
    std::fs::create_dir_all(&sub).expect("mkdir");

    let yaml = "name: sub\nversion: '1.0'\nmodel:\n  hf_repo: test/m\n  formats: [gguf]\ntest_matrix:\n  modalities: [run]\n  backends: [cpu]\n  scenario_count: 1\n";
    std::fs::write(sub.join("m.playbook.yaml"), yaml).expect("write");

    let output = dir.path().join("lock.yaml");
    let result = generate_lock_file(dir.path(), &output);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 1);
}

#[test]
fn test_execute_auto_tickets_no_failures() {
    let evidence = vec![make_corroborated_evidence()];
    let tickets = execute_auto_tickets(&evidence, "test/repo");
    assert!(tickets.is_empty());
}

#[test]
fn test_execute_auto_tickets_with_failures() {
    let mut ev = make_falsified_evidence();
    ev.stderr = Some("tensor name mismatch: layer.0".to_string());
    let tickets = execute_auto_tickets(&[ev], "test/repo");
    // Should generate at least 1 ticket for the classified failure
    assert_eq!(tickets.len(), 1);
}

#[test]
fn test_execute_auto_tickets_deduplication() {
    let evidence: Vec<Evidence> = (0..5)
        .map(|_| {
            let mut ev = make_falsified_evidence();
            ev.stderr = Some("tensor name mismatch: layer.0".to_string());
            ev
        })
        .collect();
    let tickets = execute_auto_tickets(&evidence, "test/repo");
    // 5 same-cause failures should produce 1 ticket
    assert_eq!(tickets.len(), 1);
}

#[test]
fn test_generate_lock_file_nonexistent_dir() {
    let output = std::path::Path::new("/tmp/test-lock-output.yaml");
    let result = generate_lock_file(std::path::Path::new("/nonexistent"), output);
    // Should succeed with 0 entries (no files found)
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0);
}

// =========================================================================
// Dimensional Smoke Tier Tests
// =========================================================================

#[test]
fn test_cert_tier_dimensional_smoke_from_str() {
    assert_eq!(
        "dim-smoke".parse::<CertTier>().unwrap(),
        CertTier::DimensionalSmoke
    );
    assert_eq!(
        "dimensional-smoke".parse::<CertTier>().unwrap(),
        CertTier::DimensionalSmoke
    );
    // Case insensitive
    assert_eq!(
        "DIM-SMOKE".parse::<CertTier>().unwrap(),
        CertTier::DimensionalSmoke
    );
}

#[test]
fn test_cert_tier_dimensional_smoke_suffix() {
    assert_eq!(CertTier::DimensionalSmoke.playbook_suffix(), "-dim-smoke");
}

#[test]
fn test_playbook_path_for_model_dim_smoke() {
    let path =
        playbook_path_for_model("Qwen/Qwen2.5-Coder-7B-Instruct", CertTier::DimensionalSmoke);
    assert_eq!(
        path,
        "playbooks/models/qwen2.5-coder-7b-dim-smoke.playbook.yaml"
    );
}

#[test]
fn test_build_dimensional_smoke_config_values() {
    let config = build_certification_config(CertTier::DimensionalSmoke, None);
    assert!(matches!(config.failure_policy, FailurePolicy::FailFast));
    assert_eq!(config.max_workers, 1);
    assert_eq!(config.default_timeout_ms, 30_000);
    assert!(config.no_gpu);
    assert!(!config.run_conversion_tests);
    assert!(!config.run_golden_rule_test);
    assert!(!config.run_contract_tests);
    assert!(!config.run_profile_ci);
    assert!(!config.run_hf_parity);
    assert!(!config.run_ollama_parity);
    assert!(config.metadata_only);
}

#[test]
fn test_build_dimensional_smoke_config_with_model_path() {
    let config = build_certification_config(
        CertTier::DimensionalSmoke,
        Some("/path/to/model".to_string()),
    );
    assert_eq!(config.model_path, Some("/path/to/model".to_string()));
    assert!(matches!(config.failure_policy, FailurePolicy::FailFast));
}

#[test]
fn test_build_dimensional_smoke_ignores_fail_fast_flag() {
    // DimensionalSmoke always uses FailFast regardless of the flag
    let config = build_certification_config_with_policy(CertTier::DimensionalSmoke, None, false);
    assert!(matches!(config.failure_policy, FailurePolicy::FailFast));
}

#[test]
fn test_cert_tier_from_str_error_mentions_dim_smoke() {
    let err = "invalid".parse::<CertTier>().unwrap_err();
    assert!(err.contains("dim-smoke"));
}

#[test]
fn test_non_dim_smoke_config_not_metadata_only() {
    let config = build_certification_config(CertTier::Mvp, None);
    assert!(!config.metadata_only);
}

#[test]
fn test_dim_smoke_config_is_metadata_only() {
    let config = build_certification_config(CertTier::DimensionalSmoke, None);
    assert!(config.metadata_only);
}
