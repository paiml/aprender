use super::*;

// =========================================================================
// Additional coverage tests for certify_model path
// =========================================================================

/// Helper to get the workspace root path for test playbooks
fn get_workspace_root() -> std::path::PathBuf {
    // Start from the manifest dir and go up to find the workspace root
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Create a minimal test scenario for unit testing
fn make_test_scenario() -> apr_qa_gen::QaScenario {
    apr_qa_gen::QaScenario::new(
        apr_qa_gen::ModelId::new("test", "model"),
        apr_qa_gen::Modality::Run,
        apr_qa_gen::Backend::Cpu,
        apr_qa_gen::Format::Gguf,
        "What is 2+2?".to_string(),
        42,
    )
}

/// Create a corroborated evidence instance for testing
fn make_corroborated_evidence() -> Evidence {
    Evidence::corroborated("F-TEST-001", make_test_scenario(), "output", 100)
}

#[test]
fn test_certify_model_no_cache() {
    let config = CertificationConfig {
        tier: CertTier::Mvp,
        model_cache: None,
        apr_binary: "apr".to_string(),
        output_dir: std::path::PathBuf::from("/tmp"),
        dry_run: false,
    };
    // Non-existent model, so will fail at playbook not found
    let result = certify_model("nonexistent/model", &config);
    // Will fail because playbook doesn't exist
    assert!(!result.success);
    assert!(result.error.is_some());
}

#[test]
fn test_certify_model_with_cache() {
    let config = CertificationConfig {
        tier: CertTier::Mvp,
        model_cache: Some(std::path::PathBuf::from("/nonexistent/cache")),
        apr_binary: "apr".to_string(),
        output_dir: std::path::PathBuf::from("/tmp"),
        dry_run: false,
    };
    // Non-existent model
    let result = certify_model("test/model", &config);
    // Will fail because playbook doesn't exist
    assert_eq!(result.model_id, "test/model");
}

#[test]
fn test_execute_playbook_with_config() {
    // Create a minimal playbook from YAML
    let yaml = r#"
name: test-playbook
version: "1.0.0"
description: "Test playbook"
model:
  hf_repo: "test/model"
  formats:
    - gguf
  quantizations:
    - q4_k_m
  size_category: tiny
test_matrix:
  modalities:
    - run
  backends:
    - cpu
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
    let config = build_certification_config(CertTier::Mvp, None);
    let result = execute_playbook(&playbook, config);
    assert!(result.is_ok());
    let exec_result = result.unwrap();
    assert_eq!(exec_result.playbook_name, "test-playbook");
}

#[test]
fn test_execute_playbook_with_dry_run() {
    let yaml = r#"
name: test-playbook-dry
version: "1.0.0"
description: "Test playbook dry run"
model:
  hf_repo: "test/model-dry"
  formats:
    - gguf
  quantizations:
    - q4_k_m
  size_category: tiny
test_matrix:
  modalities:
    - run
  backends:
    - cpu
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
    let mut config = build_certification_config(CertTier::Mvp, None);
    config.dry_run = true;
    let result = execute_playbook(&playbook, config);
    assert!(result.is_ok());
}

#[test]
fn test_load_playbook_with_workspace_path() {
    let workspace_root = get_workspace_root();
    let playbook_path =
        workspace_root.join("playbooks/models/qwen2.5-coder-0.5b-mvp.playbook.yaml");
    if playbook_path.exists() {
        let result = load_playbook(&playbook_path);
        assert!(result.is_ok());
        let playbook = result.unwrap();
        assert!(!playbook.name.is_empty());
    }
}

#[test]
fn test_certify_model_result_fields() {
    // Test that all ModelCertificationResult fields are properly set
    let config = CertificationConfig {
        tier: CertTier::Quick,
        ..Default::default()
    };
    let result = certify_model("nonexistent/test-model", &config);
    // Playbook won't exist, so we get error
    assert_eq!(result.model_id, "nonexistent/test-model");
    assert_eq!(result.mqs_score, 0);
    assert_eq!(result.grade, "-");
    assert!(result.error.is_some());
}

#[test]
fn test_certify_model_smoke_tier() {
    let config = CertificationConfig {
        tier: CertTier::Smoke,
        ..Default::default()
    };
    // Non-existent model
    let result = certify_model("test/model-smoke", &config);
    // Either succeeds or fails with "not found"
    assert!(result.error.is_some() || result.success);
}

#[test]
fn test_certify_model_standard_tier() {
    let config = CertificationConfig {
        tier: CertTier::Standard,
        ..Default::default()
    };
    let result = certify_model("test/model-standard", &config);
    // Standard tier uses base playbook without suffix
    assert!(!result.success); // Playbook won't exist
}

#[test]
fn test_playbook_path_for_model_various_tiers() {
    // Test all tier combinations
    let model = "Qwen/Qwen2.5-Coder-1.5B-Instruct";

    let smoke = playbook_path_for_model(model, CertTier::Smoke);
    assert!(smoke.contains("-smoke"));

    let mvp = playbook_path_for_model(model, CertTier::Mvp);
    assert!(mvp.contains("-mvp"));

    let quick = playbook_path_for_model(model, CertTier::Quick);
    assert!(quick.contains("-quick"));

    let standard = playbook_path_for_model(model, CertTier::Standard);
    assert!(!standard.contains("-standard")); // No suffix

    let deep = playbook_path_for_model(model, CertTier::Deep);
    assert!(!deep.contains("-deep")); // No suffix
}

#[test]
fn test_build_certification_config_with_model_path() {
    let config = build_certification_config(CertTier::Deep, Some("/path/to/models".to_string()));
    assert_eq!(config.model_path, Some("/path/to/models".to_string()));
    assert!(config.run_profile_ci); // Deep tier enables profile CI
}

#[test]
fn test_certification_config_all_fields_set() {
    let config = CertificationConfig {
        tier: CertTier::Deep,
        model_cache: Some(std::path::PathBuf::from("/cache")),
        apr_binary: "/usr/bin/apr".to_string(),
        output_dir: std::path::PathBuf::from("/output"),
        dry_run: true,
    };
    assert_eq!(config.tier, CertTier::Deep);
    assert!(config.model_cache.is_some());
    assert_eq!(config.apr_binary, "/usr/bin/apr");
    assert!(config.dry_run);
}

// --- Additional coverage tests ---

#[test]
fn test_list_all_models_returns_models() {
    let models = list_all_models();
    assert!(!models.is_empty());
    // Should have default models from registry
    assert!(models.len() >= 5);
}

#[test]
fn test_filter_models_by_size_small() {
    let models = list_all_models();
    let small = filter_models_by_size(&models, "small");
    // May or may not have small models depending on defaults
    for m in &small {
        assert!(format!("{:?}", m.size).to_lowercase().contains("small"));
    }
}

#[test]
fn test_filter_models_by_size_no_match() {
    let models = list_all_models();
    let none = filter_models_by_size(&models, "nonexistent");
    assert!(none.is_empty());
}

#[test]
fn test_generate_junit_report_basic() {
    let evidence = vec![make_corroborated_evidence()];
    let collector = collect_evidence(evidence);
    let mqs = calculate_mqs_score("test/model", &collector).unwrap();
    let junit = generate_junit_report("test/model", &collector, &mqs);
    assert!(junit.is_ok());
    assert!(junit.unwrap().contains("testsuite"));
}

#[test]
fn test_build_execution_config_with_model_path() {
    let config = PlaybookRunConfig {
        model_path: Some("/models/test.gguf".to_string()),
        ..Default::default()
    };
    let exec = build_execution_config(&config).unwrap();
    assert_eq!(exec.model_path, Some("/models/test.gguf".to_string()));
}

#[test]
fn test_build_execution_config_with_timeout() {
    let config = PlaybookRunConfig {
        timeout: 90000,
        ..Default::default()
    };
    let exec = build_execution_config(&config).unwrap();
    assert_eq!(exec.default_timeout_ms, 90000);
}

#[test]
fn test_build_execution_config_with_workers() {
    let config = PlaybookRunConfig {
        workers: 8,
        ..Default::default()
    };
    let exec = build_execution_config(&config).unwrap();
    assert_eq!(exec.max_workers, 8);
}

#[test]
fn test_cert_tier_from_str_all_values() {
    assert!(CertTier::from_str("smoke").is_ok());
    assert!(CertTier::from_str("mvp").is_ok());
    assert!(CertTier::from_str("quick").is_ok());
    assert!(CertTier::from_str("standard").is_ok());
    assert!(CertTier::from_str("deep").is_ok());
    assert!(CertTier::from_str("unknown").is_err());
}

#[test]
fn test_cert_tier_from_str_case_insensitive() {
    assert!(CertTier::from_str("SMOKE").is_ok());
    assert!(CertTier::from_str("MVP").is_ok());
    assert!(CertTier::from_str("Quick").is_ok());
}

#[test]
fn test_cert_tier_playbook_suffix_all() {
    assert_eq!(CertTier::Smoke.playbook_suffix(), "-smoke");
    assert_eq!(CertTier::Mvp.playbook_suffix(), "-mvp");
    assert_eq!(CertTier::Quick.playbook_suffix(), "-quick");
    assert_eq!(CertTier::Standard.playbook_suffix(), "");
    assert_eq!(CertTier::Deep.playbook_suffix(), "");
}

use std::str::FromStr;

// ── certify_model deeper paths ────────────────────────────────────────────────

#[test]
fn test_certify_model_load_playbook_failure_invalid_yaml() {
    // Create a real-but-invalid YAML file at the expected playbook path so that
    // certify_model passes the `!playbook_file.exists()` check and fails at
    // load_playbook() instead.
    let model_slug = "test-invalid-yaml-coverage-xxz";
    let playbook_path = format!("playbooks/models/{model_slug}-mvp.playbook.yaml");
    let playbook_dir = std::path::Path::new("playbooks/models");

    // Guard: can only run if playbooks/models/ exists in CWD (workspace root)
    if !playbook_dir.exists() {
        return;
    }

    std::fs::write(&playbook_path, "not: [valid: {{{ yaml}}}").expect("write invalid yaml");

    let config = CertificationConfig {
        tier: CertTier::Mvp,
        ..Default::default()
    };
    let result = certify_model(&format!("test/{model_slug}"), &config);

    // Clean up before assertions (avoids leaving debris on failure)
    let _ = std::fs::remove_file(&playbook_path);

    assert!(!result.success, "Expected failure on invalid YAML");
    let err = result.error.expect("should have error message");
    assert!(
        err.contains("Error loading playbook") || err.contains("yaml") || err.contains("parse"),
        "Expected YAML error, got: {err}"
    );
}

#[test]
fn test_certify_model_execute_playbook_success() {
    // Use an existing playbook (smollm2-135m-mvp) if available in CWD.
    // This exercises the load_playbook + execute_playbook + calculate_mqs_score path.
    let playbook_path = "playbooks/models/smollm2-135m-mvp.playbook.yaml";

    // Guard: only run when workspace playbooks are accessible
    if !std::path::Path::new(playbook_path).exists() {
        return;
    }

    let config = CertificationConfig {
        tier: CertTier::Mvp,
        model_cache: None,
        ..Default::default()
    };
    let result = certify_model("HuggingFaceTB/SmolLM2-135M", &config);
    // The playbook loads and executes; MQS may be 0 (no model file)
    // but success=true means execute_playbook succeeded and MQS was calculated
    // (or error path if MQS calc fails — either way, we reach beyond load_playbook)
    assert_eq!(result.model_id, "HuggingFaceTB/SmolLM2-135M");
    // Result may succeed or fail depending on model availability, but must not be
    // a "Playbook not found" error — we must have gotten past the first early return
    if let Some(ref err) = result.error {
        assert!(
            !err.contains("Playbook not found"),
            "Should not hit 'Playbook not found' — file exists: {err}"
        );
    }
}

/// Helper: minimal valid playbook YAML for certify_model tests
const CERTIFY_TEST_PLAYBOOK_YAML: &str = r#"
name: certify-coverage-test
version: "1.0.0"
description: "Minimal playbook for coverage testing"
model:
  hf_repo: "test/certify-coverage-unique-xxx"
  formats:
    - gguf
  quantizations:
    - q4_k_m
  size_category: tiny
test_matrix:
  modalities:
    - run
  backends:
    - cpu
  scenario_count: 1
  seed: 42
  timeout_ms: 5000
gates:
  g1_model_loads: true
  g2_basic_inference: true
  g3_no_crashes: true
  g4_not_garbage: true
"#;

#[test]
fn test_certify_model_reaches_execute_playbook_path() {
    // Create a valid playbook at the expected CWD-relative path so certify_model
    // passes all early returns and reaches the execute_playbook() call.
    // This covers lines 134-170 of lib_certification.rs.
    let playbook_dir = std::path::Path::new("playbooks/models");
    if !playbook_dir.exists() {
        return; // Guard: only run when workspace is accessible
    }

    let model_slug = "certify-coverage-unique-xxx";
    let playbook_path = format!("playbooks/models/{model_slug}-mvp.playbook.yaml");
    std::fs::write(&playbook_path, CERTIFY_TEST_PLAYBOOK_YAML).expect("write playbook");

    let config = CertificationConfig {
        tier: CertTier::Mvp,
        model_cache: None,
        ..Default::default()
    };
    let result = certify_model(&format!("test/{model_slug}"), &config);
    let _ = std::fs::remove_file(&playbook_path); // cleanup before assertions

    assert_eq!(result.model_id, format!("test/{model_slug}"));
    // Must NOT be a "Playbook not found" error — we've reached execute_playbook()
    if let Some(ref err) = result.error {
        assert!(
            !err.contains("Playbook not found"),
            "Expected to reach execute_playbook, got: {err}"
        );
    }
}

#[test]
fn test_certify_model_with_model_cache_reaches_execute_playbook() {
    // Same as above, but with model_cache=Some(path) to cover lines 134-142
    // (the model_cache_path building code).
    let playbook_dir = std::path::Path::new("playbooks/models");
    if !playbook_dir.exists() {
        return;
    }

    let model_slug = "certify-cache-coverage-yyy";
    let playbook_path = format!("playbooks/models/{model_slug}-smoke.playbook.yaml");
    std::fs::write(&playbook_path, CERTIFY_TEST_PLAYBOOK_YAML).expect("write playbook");

    let config = CertificationConfig {
        tier: CertTier::Smoke,
        model_cache: Some(std::path::PathBuf::from("/tmp/test-model-cache")),
        ..Default::default()
    };
    let result = certify_model(&format!("test/{model_slug}"), &config);
    let _ = std::fs::remove_file(&playbook_path); // cleanup

    assert_eq!(result.model_id, format!("test/{model_slug}"));
    if let Some(ref err) = result.error {
        assert!(
            !err.contains("Playbook not found"),
            "Expected to reach execute_playbook with model_cache, got: {err}"
        );
    }
}
