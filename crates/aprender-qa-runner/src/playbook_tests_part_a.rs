/// Verify default_formats returns gguf, safetensors, and apr
#[test]
fn test_default_formats() {
    let formats = default_formats();
    assert_eq!(formats.len(), 3);
    assert!(formats.contains(&Format::Gguf));
    assert!(formats.contains(&Format::SafeTensors));
    assert!(formats.contains(&Format::Apr));
}

/// Verify default_quantizations returns q4_k_m
#[test]
fn test_default_quantizations() {
    let quants = default_quantizations();
    assert_eq!(quants, vec!["q4_k_m"]);
}

/// Verify default_modalities returns run, chat, and serve
#[test]
fn test_default_modalities() {
    let modalities = default_modalities();
    assert_eq!(modalities.len(), 3);
    assert!(modalities.contains(&Modality::Run));
    assert!(modalities.contains(&Modality::Chat));
    assert!(modalities.contains(&Modality::Serve));
}

/// Verify default_backends returns cpu and gpu
#[test]
fn test_default_backends() {
    let backends = default_backends();
    assert_eq!(backends.len(), 2);
    assert!(backends.contains(&Backend::Cpu));
    assert!(backends.contains(&Backend::Gpu));
}

/// Verify default scenario count is 100
#[test]
fn test_default_scenario_count() {
    assert_eq!(default_scenario_count(), 100);
}

/// Verify default proptest count is 100
#[test]
fn test_default_proptest_count() {
    assert_eq!(default_proptest_count(), 100);
}

/// Verify default timeout is 60 seconds (60000 ms)
#[test]
fn test_default_timeout() {
    assert_eq!(default_timeout(), 60000);
}

/// Verify default severity is P1
#[test]
fn test_default_severity() {
    assert_eq!(default_severity(), "P1");
}

/// Verify TestMatrix::default populates modalities, backends, and scenario_count
#[test]
fn test_test_matrix_default() {
    let matrix = TestMatrix::default();
    assert_eq!(matrix.modalities.len(), 3);
    assert_eq!(matrix.backends.len(), 2);
    assert_eq!(matrix.scenario_count, 100);
}

/// Verify Playbook roundtrips through to_yaml preserving name and model
#[test]
fn test_playbook_to_yaml() {
    let yaml = r#"
name: test-playbook
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 5
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let output = playbook.to_yaml().expect("Failed to serialize");
    assert!(output.contains("test-playbook"));
    assert!(output.contains("test/model"));
}

/// Verify Playbook applies default formats and quantizations when omitted
#[test]
fn test_playbook_with_defaults() {
    // Test playbook that uses default values for model config
    let yaml = r#"
name: minimal
version: "1.0.0"
model:
  hf_repo: "org/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 100
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    assert_eq!(playbook.model.formats.len(), 3);
    assert_eq!(playbook.model.quantizations, vec!["q4_k_m"]);
    assert_eq!(playbook.test_matrix.scenario_count, 100);
}

/// Verify Playbook parses state machine with states, transitions, and guards
#[test]
fn test_playbook_with_state_machine() {
    let yaml = r#"
name: state-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
state_machine:
  initial: "ready"
  states:
    ready:
      on_enter:
        - action: "log 'entering ready'"
      transitions:
        - event: "start"
          target: "running"
          action: "initialize"
          guards:
            - "model_loaded"
    running:
      on_exit:
        - action: "cleanup"
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let state_machine = playbook.state_machine.expect("Should have state machine");
    assert_eq!(state_machine.initial, "ready");
    assert_eq!(state_machine.states.len(), 2);

    let ready_state = &state_machine.states["ready"];
    assert_eq!(ready_state.on_enter.len(), 1);
    assert_eq!(ready_state.transitions.len(), 1);

    let transition = &ready_state.transitions[0];
    assert_eq!(transition.event, "start");
    assert_eq!(transition.target, "running");
    assert!(transition.action.is_some());
    assert_eq!(transition.guards.len(), 1);
}

/// Verify Playbook parses property tests with generator, oracle, and count
#[test]
fn test_playbook_with_property_tests() {
    let yaml = r#"
name: prop-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
property_tests:
  - name: "arithmetic"
    generator: "random_arithmetic"
    oracle: "check_arithmetic"
    count: 50
  - name: "code"
    generator: "random_code"
    oracle: "check_code"
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    assert_eq!(playbook.property_tests.len(), 2);

    let first = &playbook.property_tests[0];
    assert_eq!(first.name, "arithmetic");
    assert_eq!(first.count, 50);

    let second = &playbook.property_tests[1];
    assert_eq!(second.name, "code");
    assert_eq!(second.count, 100); // default
}

/// Verify Playbook parses falsification gates with severity defaults
#[test]
fn test_playbook_with_falsification_gates() {
    let yaml = r#"
name: gate-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
falsification_gates:
  - id: F-QUAL-001
    description: "Output is valid"
    condition: "output.len() > 0"
    severity: P0
  - id: F-QUAL-002
    description: "No errors"
    condition: "!output.contains('error')"
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    assert_eq!(playbook.falsification_gates.len(), 2);

    let first = &playbook.falsification_gates[0];
    assert_eq!(first.severity, "P0");

    let second = &playbook.falsification_gates[1];
    assert_eq!(second.severity, "P1"); // default
}

/// Verify ModelConfig returns model-name for both org and name when no slash
#[test]
fn test_model_config_no_slash() {
    let config = ModelConfig {
        hf_repo: "model-name".to_string(),
        local_path: None,
        formats: vec![Format::Gguf],
        quantizations: vec![],
        size_category: SizeCategory::default(),
        expected_hidden_dim: None,
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };
    assert_eq!(config.hf_org(), "model-name");
    assert_eq!(config.hf_name(), "model-name");
}

/// Verify ModelConfig preserves optional local_path field
#[test]
fn test_model_config_with_local_path() {
    let config = ModelConfig {
        hf_repo: "org/model".to_string(),
        local_path: Some("/path/to/model".to_string()),
        formats: default_formats(),
        quantizations: default_quantizations(),
        size_category: SizeCategory::default(),
        expected_hidden_dim: None,
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };
    assert!(config.local_path.is_some());
}

/// Verify PlaybookStep stores timeout and expected exit code
#[test]
fn test_playbook_step() {
    let step = PlaybookStep {
        name: "test-step".to_string(),
        command: "echo test".to_string(),
        timeout_ms: default_timeout(),
        expected_exit_code: 0,
        expected_patterns: vec!["test".to_string()],
        forbidden_patterns: vec!["error".to_string()],
    };
    assert_eq!(step.timeout_ms, 60000);
    assert_eq!(step.expected_exit_code, 0);
}

/// Verify Playbook parses full YAML with model, test matrix, and gates
#[test]
fn test_playbook_parse() {
    let yaml = r#"
name: test-playbook
version: "1.0.0"
model:
  hf_repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct"
  formats: [gguf, safetensors]
test_matrix:
  modalities: [run, chat]
  backends: [cpu]
  scenario_count: 10
falsification_gates:
  - id: F-TEST-001
    description: "Output is non-empty"
    condition: "output.len() > 0"
"#;

    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse playbook");
    assert_eq!(playbook.name, "test-playbook");
    assert_eq!(playbook.model.hf_repo, "Qwen/Qwen2.5-Coder-1.5B-Instruct");
    assert_eq!(playbook.test_matrix.modalities.len(), 2);
    assert_eq!(playbook.falsification_gates.len(), 1);
}

/// Verify generate_scenarios produces correct count from matrix dimensions
#[test]
fn test_playbook_generate_scenarios() {
    let yaml = r#"
name: test-playbook
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 5
"#;

    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let scenarios = playbook.generate_scenarios();

    // 1 modality x 1 backend x 1 format x 5 scenarios = 5
    assert_eq!(scenarios.len(), 5);
}

/// Verify ModelConfig splits hf_repo into org and name correctly
#[test]
fn test_model_config_parse() {
    let config = ModelConfig {
        hf_repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct".to_string(),
        local_path: None,
        formats: vec![Format::Gguf],
        quantizations: vec!["q4_k_m".to_string()],
        size_category: SizeCategory::Small,
        expected_hidden_dim: None,
        expected_num_layers: None,
        expected_num_heads: None,
        expected_num_kv_heads: None,
        expected_vocab_size: None,
        expected_intermediate_dim: None,
        family: None,
        size_variant: None,
    };

    assert_eq!(config.hf_org(), "Qwen");
    assert_eq!(config.hf_name(), "Qwen2.5-Coder-1.5B-Instruct");
}

/// Verify total_tests computes modalities x backends x formats x scenario_count
#[test]
fn test_total_tests() {
    let yaml = r#"
name: test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf, safetensors, apr]
test_matrix:
  modalities: [run, chat, serve]
  backends: [cpu, gpu]
  scenario_count: 100
"#;

    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    // 3 modalities x 2 backends x 3 formats x 100 = 1800
    assert_eq!(playbook.total_tests(), 1800);
}

/// Verify Playbook parses differential test config with tensor diff and inference compare
#[test]
fn test_playbook_with_differential_tests() {
    let yaml = r#"
name: diff-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
differential_tests:
  tensor_diff:
    enabled: true
    filter: "embed,lm_head"
    gates: ["F-ROSETTA-DIFF-001", "F-ROSETTA-DIFF-002"]
  inference_compare:
    enabled: true
    prompt: "What is 2+2?"
    max_tokens: 10
    tolerance: 0.00001
    gates: ["F-ROSETTA-INF-001"]
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let diff = playbook
        .differential_tests
        .expect("Should have differential tests");

    let tensor = diff.tensor_diff.expect("Should have tensor diff");
    assert!(tensor.enabled);
    assert_eq!(tensor.filter, Some("embed,lm_head".to_string()));
    assert_eq!(tensor.gates.len(), 2);

    let inf = diff
        .inference_compare
        .expect("Should have inference compare");
    assert!(inf.enabled);
    assert_eq!(inf.prompt, Some("What is 2+2?".to_string()));
    assert_eq!(inf.max_tokens, 10);
}

/// Verify Playbook parses profile_ci with warmup, measure, and assertions
#[test]
fn test_playbook_with_profile_ci() {
    let yaml = r#"
name: profile-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
profile_ci:
  enabled: true
  warmup: 5
  measure: 20
  assertions:
    min_throughput: 10.0
    max_p99_ms: 500.0
    max_p50_ms: 200.0
  gates: ["F-PROFILE-CI-001", "F-PROFILE-CI-002"]
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let profile = playbook.profile_ci.expect("Should have profile CI");

    assert!(profile.enabled);
    assert_eq!(profile.warmup, 5);
    assert_eq!(profile.measure, 20);
    assert_eq!(profile.assertions.min_throughput, Some(10.0));
    assert_eq!(profile.assertions.max_p99_ms, Some(500.0));
    assert_eq!(profile.assertions.max_p50_ms, Some(200.0));
    assert_eq!(profile.gates.len(), 2);
}

/// Verify Playbook parses trace_payload with prompt and gates
#[test]
fn test_playbook_with_trace_payload() {
    let yaml = r#"
name: trace-test
version: "1.0.0"
model:
  hf_repo: "test/model"
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
trace_payload:
  enabled: true
  prompt: "Test prompt"
  gates: ["F-TRACE-PAYLOAD-001", "F-TRACE-PAYLOAD-002"]
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let trace = playbook.trace_payload.expect("Should have trace payload");

    assert!(trace.enabled);
    assert_eq!(trace.prompt, Some("Test prompt".to_string()));
    assert_eq!(trace.gates.len(), 2);
}

/// Verify default_max_tokens returns 10
#[test]
fn test_default_max_tokens() {
    assert_eq!(default_max_tokens(), 10);
}
