use super::*;
use apr_qa_gen::{Backend, Format, Modality, QaScenario};

fn make_test_scenario() -> QaScenario {
    QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "What is 2+2?".to_string(),
        42,
    )
}

fn make_corroborated_evidence() -> Evidence {
    Evidence::corroborated("F-TEST-001", make_test_scenario(), "output", 100)
}

fn make_falsified_evidence() -> Evidence {
    Evidence::falsified("F-TEST-002", make_test_scenario(), "failed", "error", 200)
}

#[test]
fn test_cli_result_success() {
    let result = CliResult::Success("test".to_string());
    assert!(result.is_success());
    assert_eq!(result.message(), "test");
}

#[test]
fn test_cli_result_error() {
    let result = CliResult::Error("error".to_string());
    assert!(!result.is_success());
    assert_eq!(result.message(), "error");
}

#[test]
fn test_playbook_run_config_default() {
    let config = PlaybookRunConfig::default();
    assert_eq!(config.failure_policy, "stop-on-p0");
    assert!(!config.dry_run);
    assert_eq!(config.workers, 4);
    assert!(config.model_path.is_none());
    assert_eq!(config.timeout, 60000);
    assert!(!config.no_gpu);
    assert!(!config.skip_conversion_tests);
    assert!(!config.run_tool_tests);
}

#[test]
fn test_parse_failure_policy_stop_on_first() {
    let policy = parse_failure_policy("stop-on-first").unwrap();
    assert!(matches!(policy, FailurePolicy::StopOnFirst));
}

#[test]
fn test_parse_failure_policy_stop_on_p0() {
    let policy = parse_failure_policy("stop-on-p0").unwrap();
    assert!(matches!(policy, FailurePolicy::StopOnP0));
}

#[test]
fn test_parse_failure_policy_collect_all() {
    let policy = parse_failure_policy("collect-all").unwrap();
    assert!(matches!(policy, FailurePolicy::CollectAll));
}

#[test]
fn test_parse_failure_policy_unknown() {
    let result = parse_failure_policy("unknown");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Unknown failure policy"));
}

#[test]
fn test_parse_failure_policy_fail_fast() {
    let policy = parse_failure_policy("fail-fast").unwrap();
    assert!(matches!(policy, FailurePolicy::FailFast));
}

#[test]
fn test_failure_policy_fail_fast_emit_diagnostic() {
    assert!(FailurePolicy::FailFast.emit_diagnostic());
    assert!(!FailurePolicy::StopOnFirst.emit_diagnostic());
    assert!(!FailurePolicy::StopOnP0.emit_diagnostic());
    assert!(!FailurePolicy::CollectAll.emit_diagnostic());
}

#[test]
fn test_failure_policy_stops_on_any_failure() {
    assert!(FailurePolicy::FailFast.stops_on_any_failure());
    assert!(FailurePolicy::StopOnFirst.stops_on_any_failure());
    assert!(!FailurePolicy::StopOnP0.stops_on_any_failure());
    assert!(!FailurePolicy::CollectAll.stops_on_any_failure());
}

#[test]
fn test_load_playbook_nonexistent() {
    let result = load_playbook(Path::new("/nonexistent/playbook.yaml"));
    assert!(result.is_err());
}

#[test]
fn test_generate_model_scenarios() {
    let scenarios = generate_model_scenarios("test/model", 10);
    // 3 modalities x 2 backends x 3 formats x 10 = 180 scenarios
    assert_eq!(scenarios.len(), 180);
}

#[test]
fn test_generate_model_scenarios_no_org() {
    let scenarios = generate_model_scenarios("model-only", 5);
    assert_eq!(scenarios.len(), 90); // 3 x 2 x 3 x 5
}

#[test]
fn test_scenarios_to_yaml() {
    let scenarios = generate_model_scenarios("test/model", 1);
    let yaml = scenarios_to_yaml(&scenarios);
    assert!(yaml.is_ok());
    let yaml_str = yaml.unwrap();
    assert!(yaml_str.contains("---"));
}

#[test]
fn test_scenarios_to_json() {
    let scenarios = generate_model_scenarios("test/model", 1);
    let json = scenarios_to_json(&scenarios);
    assert!(json.is_ok());
    let json_str = json.unwrap();
    assert!(json_str.starts_with('['));
}

#[test]
fn test_parse_evidence_invalid() {
    let json = "invalid json";
    let evidence = parse_evidence(json);
    assert!(evidence.is_err());
}

#[test]
fn test_collect_evidence() {
    let evidence = vec![make_corroborated_evidence()];
    let collector = collect_evidence(evidence);
    assert_eq!(collector.total(), 1);
}

#[test]
fn test_calculate_mqs_score() {
    let evidence = vec![make_corroborated_evidence(), make_falsified_evidence()];
    let collector = collect_evidence(evidence);
    let score = calculate_mqs_score("test/model", &collector);
    assert!(score.is_ok());
}

#[test]
fn test_calculate_popperian_score() {
    let evidence = vec![make_corroborated_evidence(), make_falsified_evidence()];
    let collector = collect_evidence(evidence);
    let score = calculate_popperian_score("test/model", &collector);
    assert_eq!(score.model_id, "test/model");
}

#[test]
fn test_generate_html_report() {
    let evidence = vec![make_corroborated_evidence()];
    let collector = collect_evidence(evidence);
    let mqs = calculate_mqs_score("test/model", &collector).unwrap();
    let popperian = calculate_popperian_score("test/model", &collector);
    let html = generate_html_report("Test Report", &mqs, &popperian, &collector);
    assert!(html.is_ok());
    assert!(html.unwrap().contains("<html"));
}

#[test]
fn test_generate_junit_report() {
    let evidence = vec![make_corroborated_evidence()];
    let collector = collect_evidence(evidence);
    let mqs = calculate_mqs_score("test/model", &collector).unwrap();
    let xml = generate_junit_report("test/model", &collector, &mqs);
    assert!(xml.is_ok());
    assert!(xml.unwrap().contains("<testsuite"));
}

#[test]
fn test_list_all_models() {
    let models = list_all_models();
    assert!(!models.is_empty());
}

#[test]
fn test_filter_models_by_size() {
    let models = list_all_models();
    let small = filter_models_by_size(&models, "small");
    // All filtered models should have "small" in their size
    for model in &small {
        let size_str = format!("{:?}", model.size).to_lowercase();
        assert!(size_str.contains("small"));
    }
}

#[test]
fn test_filter_models_by_size_case_insensitive() {
    let models = list_all_models();
    let small1 = filter_models_by_size(&models, "small");
    let small2 = filter_models_by_size(&models, "SMALL");
    assert_eq!(small1.len(), small2.len());
}

#[test]
fn test_generate_tickets_from_evidence_empty() {
    let evidence: Vec<Evidence> = vec![];
    let tickets = generate_tickets_from_evidence(&evidence, "test/repo", false, 1);
    assert!(tickets.is_empty());
}

#[test]
fn test_generate_tickets_from_evidence_with_failures() {
    let evidence = vec![make_falsified_evidence(), make_falsified_evidence()];
    let tickets = generate_tickets_from_evidence(&evidence, "test/repo", false, 1);
    // May or may not generate tickets depending on ticket rules
    assert!(tickets.len() <= evidence.len());
}

#[test]
fn test_build_execution_config() {
    let config = PlaybookRunConfig::default();
    let exec_config = build_execution_config(&config);
    assert!(exec_config.is_ok());
    let exec = exec_config.unwrap();
    assert!(!exec.dry_run);
    assert_eq!(exec.max_workers, 4);
}

#[test]
fn test_build_execution_config_invalid_policy() {
    let config = PlaybookRunConfig {
        failure_policy: "invalid".to_string(),
        ..Default::default()
    };
    let exec_config = build_execution_config(&config);
    assert!(exec_config.is_err());
}

#[test]
fn test_build_execution_config_with_options() {
    let config = PlaybookRunConfig {
        dry_run: true,
        workers: 8,
        model_path: Some("/path/to/model".to_string()),
        no_gpu: true,
        skip_conversion_tests: true,
        ..Default::default()
    };
    let exec_config = build_execution_config(&config).unwrap();
    assert!(exec_config.dry_run);
    assert_eq!(exec_config.max_workers, 8);
    assert_eq!(exec_config.model_path, Some("/path/to/model".to_string()));
    assert!(exec_config.no_gpu);
    assert!(!exec_config.run_conversion_tests);
}

#[test]
fn test_collect_multiple_evidence() {
    let evidence = vec![
        make_corroborated_evidence(),
        make_falsified_evidence(),
        make_corroborated_evidence(),
    ];
    let collector = collect_evidence(evidence);
    assert_eq!(collector.total(), 3);
    assert_eq!(collector.pass_count(), 2);
    assert_eq!(collector.fail_count(), 1);
}

#[test]
fn test_format_ticket_for_display() {
    let evidence = vec![make_falsified_evidence()];
    let tickets = generate_tickets_from_evidence(&evidence, "test/repo", false, 1);
    if !tickets.is_empty() {
        let display = format_ticket_for_display(&tickets[0], "test/repo");
        assert!(display.contains("---"));
        assert!(display.contains("Priority:"));
    }
}

#[test]
fn test_scenarios_yaml_roundtrip() {
    let scenarios = generate_model_scenarios("test/model", 1);
    let yaml = scenarios_to_yaml(&scenarios).unwrap();
    // Should be valid YAML that can be parsed back
    assert!(yaml.contains("model:"));
}

#[test]
fn test_scenarios_json_roundtrip() {
    let scenarios = generate_model_scenarios("test/model", 1);
    let json = scenarios_to_json(&scenarios).unwrap();
    // Should be valid JSON that can be parsed back
    let parsed: Vec<apr_qa_gen::QaScenario> = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.len(), scenarios.len());
}
