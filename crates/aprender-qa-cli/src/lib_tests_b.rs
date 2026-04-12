use super::*;

// =========================================================================
// Certification Tests
// =========================================================================

#[test]
fn test_cert_tier_from_str() {
    assert_eq!("smoke".parse::<CertTier>().unwrap(), CertTier::Smoke);
    assert_eq!("mvp".parse::<CertTier>().unwrap(), CertTier::Mvp);
    assert_eq!("quick".parse::<CertTier>().unwrap(), CertTier::Quick);
    assert_eq!("standard".parse::<CertTier>().unwrap(), CertTier::Standard);
    assert_eq!("deep".parse::<CertTier>().unwrap(), CertTier::Deep);
    // Case insensitive
    assert_eq!("SMOKE".parse::<CertTier>().unwrap(), CertTier::Smoke);
    assert_eq!("Quick".parse::<CertTier>().unwrap(), CertTier::Quick);
}

#[test]
fn test_cert_tier_from_str_invalid() {
    let result = "invalid".parse::<CertTier>();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Unknown tier"));
}

#[test]
fn test_cert_tier_playbook_suffix() {
    assert_eq!(CertTier::Smoke.playbook_suffix(), "-smoke");
    assert_eq!(CertTier::Mvp.playbook_suffix(), "-mvp");
    assert_eq!(CertTier::Quick.playbook_suffix(), "-quick");
    assert_eq!(CertTier::Standard.playbook_suffix(), "");
    assert_eq!(CertTier::Deep.playbook_suffix(), "");
}

#[test]
fn test_certification_config_default() {
    let config = CertificationConfig::default();
    assert_eq!(config.tier, CertTier::Quick);
    assert!(config.model_cache.is_none());
    assert_eq!(config.apr_binary, "apr");
    assert!(!config.dry_run);
}

#[test]
fn test_build_certification_config_no_model() {
    // Without model path, critical tests should still be enabled
    let config = build_certification_config(CertTier::Mvp, None);
    assert!(config.run_conversion_tests);
    assert!(config.run_golden_rule_test);
}

#[test]
fn test_build_certification_config_with_model() {
    // With model path, all critical tests should be enabled
    let config = build_certification_config(CertTier::Mvp, Some("/path/to/model".to_string()));
    assert!(config.run_conversion_tests);
    assert!(config.run_golden_rule_test);
    assert_eq!(config.model_path, Some("/path/to/model".to_string()));
}

#[test]
fn test_build_certification_config_profile_ci() {
    // MVP/Standard/Deep tiers should enable profile CI (Bug 203)
    let mvp = build_certification_config(CertTier::Mvp, None);
    assert!(mvp.run_profile_ci);

    let standard = build_certification_config(CertTier::Standard, None);
    assert!(standard.run_profile_ci);

    let deep = build_certification_config(CertTier::Deep, None);
    assert!(deep.run_profile_ci);
}

#[test]
fn test_playbook_path_for_model() {
    let path = playbook_path_for_model("Qwen/Qwen2.5-Coder-0.5B-Instruct", CertTier::Mvp);
    assert_eq!(
        path,
        "playbooks/models/qwen2.5-coder-0.5b-mvp.playbook.yaml"
    );

    let path = playbook_path_for_model("meta-llama/Llama-3-8B-Instruct", CertTier::Quick);
    assert_eq!(path, "playbooks/models/llama-3-8b-quick.playbook.yaml");

    let path = playbook_path_for_model("test/model-it", CertTier::Standard);
    assert_eq!(path, "playbooks/models/model.playbook.yaml");
}

#[test]
fn test_certify_model_nonexistent_playbook() {
    let config = CertificationConfig {
        tier: CertTier::Mvp,
        ..Default::default()
    };
    let result = certify_model("nonexistent/model", &config);
    assert!(!result.success);
    assert!(result.error.is_some());
    assert!(result.error.unwrap().contains("Playbook not found"));
}

#[test]
fn test_model_certification_result_fields() {
    let result = ModelCertificationResult {
        model_id: "test/model".to_string(),
        success: true,
        mqs_score: 850,
        grade: "A".to_string(),
        pass_rate: 95.0,
        gateway_failed: None,
        error: None,
    };
    assert!(result.success);
    assert_eq!(result.mqs_score, 850);
    assert_eq!(result.grade, "A");
}

#[test]
fn test_model_certification_result_with_gateway_failure() {
    let result = ModelCertificationResult {
        model_id: "test/model".to_string(),
        success: false,
        mqs_score: 0,
        grade: "-".to_string(),
        pass_rate: 0.0,
        gateway_failed: Some("G1: Model failed to load".to_string()),
        error: None,
    };
    assert!(!result.success);
    assert!(result.gateway_failed.is_some());
}

#[test]
fn test_certification_config_with_model_cache() {
    let config = CertificationConfig {
        tier: CertTier::Deep,
        model_cache: Some(std::path::PathBuf::from("/test/cache")),
        apr_binary: "custom-apr".to_string(),
        output_dir: std::path::PathBuf::from("/output"),
        dry_run: true,
    };
    assert_eq!(config.tier, CertTier::Deep);
    assert!(config.model_cache.is_some());
    assert!(config.dry_run);
}

#[test]
fn test_parse_evidence_empty_array() {
    let json = "[]";
    let result = parse_evidence(json);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}

#[test]
fn test_generate_tickets_black_swans_only() {
    let evidence = vec![make_falsified_evidence()];
    let tickets = generate_tickets_from_evidence(&evidence, "test/repo", true, 1);
    // May or may not have tickets depending on whether evidence qualifies as black swan
    let _ = tickets;
}

#[test]
fn test_generate_tickets_min_occurrences() {
    let evidence = vec![make_falsified_evidence()];
    let tickets = generate_tickets_from_evidence(&evidence, "test/repo", false, 5);
    // With only 1 evidence and min_occurrences=5, should have no tickets
    assert!(tickets.is_empty());
}

#[test]
fn test_playbook_run_config_with_all_options() {
    let config = PlaybookRunConfig {
        failure_policy: "collect-all".to_string(),
        dry_run: true,
        workers: 16,
        model_path: Some("/path/to/model".to_string()),
        timeout: 120_000,
        no_gpu: true,
        skip_conversion_tests: true,
        run_tool_tests: true,
        run_profile_ci: true,
        run_hf_parity: false,
        hf_parity_corpus_path: None,
        hf_parity_model_family: None,
        metadata_only: false,
    };
    assert!(config.dry_run);
    assert_eq!(config.workers, 16);
    assert!(config.run_tool_tests);
    assert!(config.run_profile_ci);
}

#[test]
fn test_build_execution_config_with_profile_ci() {
    let config = PlaybookRunConfig {
        run_profile_ci: true,
        ..Default::default()
    };
    let exec = build_execution_config(&config).unwrap();
    assert!(exec.run_profile_ci);
}

#[test]
fn test_build_certification_config_all_tiers() {
    // Test all tiers
    let tiers = [
        CertTier::Smoke,
        CertTier::Mvp,
        CertTier::Quick,
        CertTier::Standard,
        CertTier::Deep,
    ];

    for tier in tiers {
        let config = build_certification_config(tier, None);
        // All tiers should return valid config
        assert_eq!(config.failure_policy, FailurePolicy::CollectAll);
    }
}

#[test]
fn test_playbook_path_for_model_with_slash() {
    let path = playbook_path_for_model("org/model-name-Instruct", CertTier::Smoke);
    assert!(path.contains("smoke"));
    assert!(path.contains("model-name"));
    // Should strip -Instruct
    assert!(!path.contains("-Instruct") && !path.contains("-instruct"));
}

#[test]
fn test_playbook_path_for_model_deep_tier() {
    let path = playbook_path_for_model("test/model", CertTier::Deep);
    // Deep tier has no suffix
    assert!(path.ends_with(".playbook.yaml"));
    assert!(!path.contains("-deep"));
}

#[test]
fn test_certification_config_output_dir() {
    let config = CertificationConfig::default();
    assert_eq!(
        config.output_dir,
        std::path::PathBuf::from("certifications")
    );
}

#[test]
fn test_cli_result_debug() {
    let result = CliResult::Success("test".to_string());
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("Success"));
}

#[test]
fn test_model_certification_result_debug() {
    let result = ModelCertificationResult {
        model_id: "test".to_string(),
        success: true,
        mqs_score: 900,
        grade: "A".to_string(),
        pass_rate: 100.0,
        gateway_failed: None,
        error: None,
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("ModelCertificationResult"));
}

#[test]
fn test_playbook_run_config_debug() {
    let config = PlaybookRunConfig::default();
    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("PlaybookRunConfig"));
}

#[test]
fn test_certification_config_debug() {
    let config = CertificationConfig::default();
    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("CertificationConfig"));
}

#[test]
fn test_playbook_run_config_clone() {
    let config = PlaybookRunConfig::default();
    let cloned = config.clone();
    assert_eq!(config.failure_policy, cloned.failure_policy);
    assert_eq!(config.workers, cloned.workers);
}

#[test]
fn test_certification_config_clone() {
    let config = CertificationConfig::default();
    let cloned = config.clone();
    assert_eq!(config.tier, cloned.tier);
    assert_eq!(config.apr_binary, cloned.apr_binary);
}

#[test]
fn test_model_certification_result_clone() {
    let result = ModelCertificationResult {
        model_id: "test".to_string(),
        success: true,
        mqs_score: 800,
        grade: "B".to_string(),
        pass_rate: 80.0,
        gateway_failed: None,
        error: None,
    };
    let cloned = result.clone();
    assert_eq!(result.model_id, cloned.model_id);
    assert_eq!(result.mqs_score, cloned.mqs_score);
}

#[test]
fn test_cert_tier_default() {
    let tier = CertTier::default();
    assert_eq!(tier, CertTier::Quick);
}

#[test]
fn test_execute_tool_tests() {
    // Just verify function exists and returns results
    let results = execute_tool_tests("/nonexistent/model.gguf", true, 1000, false);
    // Should return empty or with failures since model doesn't exist
    let _ = results;
}

/// Create a minimal test scenario for unit testing
fn make_test_scenario() -> apr_qa_gen::QaScenario {
    apr_qa_gen::QaScenario::new(
        ModelId::new("test", "model"),
        apr_qa_gen::Modality::Run,
        apr_qa_gen::Backend::Cpu,
        apr_qa_gen::Format::Gguf,
        "What is 2+2?".to_string(),
        42,
    )
}

/// Create a falsified evidence instance for testing
fn make_falsified_evidence() -> Evidence {
    Evidence::falsified("F-TEST-002", make_test_scenario(), "failed", "error", 200)
}
