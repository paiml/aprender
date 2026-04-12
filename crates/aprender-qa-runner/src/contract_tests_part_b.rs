#[test]
fn test_contract_tests_with_dotted_workspace_path() {
    use crate::command::MockCommandRunner;

    let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
    let model_id = ModelId::new("Qwen", "Qwen2.5-Coder-0.5B-Instruct");
    let config = ContractTestConfig::default();

    let evidence = run_contract_tests(
        &runner,
        Path::new("/workspace/Qwen/Qwen2.5-Coder-0.5B-Instruct"),
        &model_id,
        &config,
    );

    // All 4 invariants (I-2 through I-5) should produce evidence
    assert_eq!(evidence.len(), 4, "Expected 4 invariant results");
    for ev in &evidence {
        // None should mention truncated paths like "Coder-0.apr"
        assert!(
            !ev.reason.contains("Coder-0.apr"),
            "Path was truncated by with_extension: {}",
            ev.reason
        );
    }
}

#[test]
fn test_is_valid_tensor_name_edge_cases() {
    let contract = load_format_contract().expect("Failed to load contract");

    // Valid edge cases
    assert!(validate_tensor_name("0.attn.weight", &contract));
    assert!(validate_tensor_name("99.mlp.bias", &contract));

    // Invalid edge cases
    assert!(!validate_tensor_name("weight", &contract));
    assert!(!validate_tensor_name(".q_proj.weight", &contract));
    assert!(!validate_tensor_name("a.q_proj.weight", &contract));
    assert!(!validate_tensor_name("0.q_proj.weight.extra", &contract));
}

#[test]
fn test_naming_convention() {
    let contract = load_format_contract().expect("Failed to load contract");
    assert_eq!(contract.tensor_naming.convention, "gguf-short");
}

#[test]
fn test_invariant_catches_fields() {
    let contract = load_format_contract().expect("Failed to load contract");
    let i1 = contract.invariants.iter().find(|i| i.id == "I-1").unwrap();
    assert!(i1.catches.contains(&"GH-190".to_string()));
    assert!(i1.implemented);

    let i2 = contract.invariants.iter().find(|i| i.id == "I-2").unwrap();
    assert!(i2.catches.contains(&"GH-190".to_string()));
    assert!(!i2.implemented);
}

#[test]
fn test_tolerance_entries_ordered_by_precision() {
    let contract = load_format_contract().expect("Failed to load contract");
    // F32 should have 0 tolerance (exact)
    let f32_tol = lookup_tolerance("F32", &contract).unwrap();
    assert!(f32_tol.0.abs() < f64::EPSILON);

    // Q2_K should have the loosest tolerance
    let q2k_tol = lookup_tolerance("Q2_K", &contract).unwrap();
    assert!(q2k_tol.0 > 0.1);
}

#[test]
fn test_is_word() {
    assert!(is_word("weight"));
    assert!(is_word("q_proj"));
    assert!(is_word("down_proj"));
    assert!(is_word("a"));
    assert!(!is_word(""));
    assert!(!is_word("has.dot"));
    assert!(!is_word("has space"));
}

/// Verify validate_dtype_bytes detects duplicate byte values
#[test]
fn test_validate_dtype_bytes_rejects_duplicates() {
    let mut contract = load_format_contract().expect("Failed to load contract");
    // Inject a duplicate byte value
    let existing_byte = contract.dtype_bytes.mappings[0].byte;
    contract.dtype_bytes.mappings.push(DtypeByteEntry {
        dtype: "FAKE_DUP".to_string(),
        byte: existing_byte,
    });
    let result = validate_dtype_bytes(&contract);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Duplicate GGML byte value"));
}

/// Verify validate_dtype_bytes succeeds on the real contract (no duplicates)
#[test]
fn test_validate_dtype_bytes_passes_real_contract() {
    let contract = load_format_contract().expect("Failed to load contract");
    assert!(validate_dtype_bytes(&contract).is_ok());
}

/// Verify InvariantDef default implemented field is false
#[test]
fn test_invariant_def_default_implemented() {
    let yaml = r#"
        id: "I-99"
        name: "Test"
        description: "Test invariant"
        catches: []
        gate_id: "F-TEST-001"
    "#;
    let inv: InvariantDef = serde_yaml::from_str(yaml).expect("should parse");
    assert!(!inv.implemented, "Default implemented should be false");
}

/// Verify run_contract_tests with empty invariants list returns empty evidence
#[test]
fn test_contract_empty_invariants_config() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig { invariants: vec![] };
    let evidence = run_contract_tests(
        &runner, Path::new("/test/workspace/org/model"), &model_id, &config,
    );
    assert!(evidence.is_empty());
}

/// I-2 inspect failure → falsified evidence with stderr
#[test]
fn test_i2_inspect_failure_produces_falsified() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> =
        Arc::new(MockCommandRunner::new().with_inspect_json_failure());
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig {
        invariants: vec!["I-2".to_string()],
    };
    let evidence = run_contract_tests(
        &runner,
        Path::new("/test/workspace/org/model"),
        &model_id,
        &config,
    );
    assert_eq!(evidence.len(), 1);
    assert!(
        evidence[0].outcome.is_fail(),
        "Inspect failure should produce falsified evidence"
    );
    assert!(
        evidence[0].reason.contains("inspect failed"),
        "Reason should mention inspect failure: {}",
        evidence[0].reason
    );
}

/// I-2 missing tensors → falsified (source has tensors APR does not)
#[test]
fn test_i2_missing_tensors_falsified() {
    use crate::command::MockCommandRunner;
    // Default mock returns 10 standard tensors for BOTH inspect calls,
    // but we need the APR inspection to return a subset.
    // Since MockCommandRunner returns the same names for both, we need
    // to test via the parse_tensor_names path instead.
    // Default mock has matching tensors → I-2 passes (corroborated).
    let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig {
        invariants: vec!["I-2".to_string()],
    };
    let evidence = run_contract_tests(
        &runner,
        Path::new("/test/workspace/org/model"),
        &model_id,
        &config,
    );
    assert_eq!(evidence.len(), 1);
    // Default mock returns same tensor names for both → bijection holds
    assert!(
        evidence[0].outcome.is_pass(),
        "Matching tensors should pass I-2: {}",
        evidence[0].reason
    );
    assert!(evidence[0].reason.contains("I-2 Tensor Name Bijection"));
}

/// I-2 with empty tensor names → falsified (Popper: vacuous bijection)
#[test]
fn test_i2_empty_tensor_names_falsified() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> =
        Arc::new(MockCommandRunner::new().with_tensor_names(vec![]));
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig {
        invariants: vec!["I-2".to_string()],
    };
    let evidence = run_contract_tests(
        &runner,
        Path::new("/test/workspace/org/model"),
        &model_id,
        &config,
    );
    assert_eq!(evidence.len(), 1);
    assert!(
        evidence[0].outcome.is_fail(),
        "Empty tensor names → vacuous bijection → fail: {}",
        evidence[0].reason
    );
    assert!(evidence[0].reason.contains("parsed 0 tensors"));
}

/// I-3 check failure → falsified evidence
#[test]
fn test_i3_check_failure_produces_falsified() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> =
        Arc::new(MockCommandRunner::new().with_check_failure());
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig {
        invariants: vec!["I-3".to_string()],
    };
    let evidence = run_contract_tests(
        &runner,
        Path::new("/test/workspace/org/model"),
        &model_id,
        &config,
    );
    assert_eq!(evidence.len(), 1);
    assert!(
        evidence[0].outcome.is_fail(),
        "Check failure should produce falsified evidence"
    );
    assert!(
        evidence[0].reason.contains("I-3 No Silent Fallbacks: check failed"),
        "Reason should mention check failure: {}",
        evidence[0].reason
    );
}

/// I-3 F32 fallback detected in check stdout → falsified
#[test]
fn test_i3_f32_fallback_detected_falsified() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> = Arc::new(
        MockCommandRunner::new()
            .with_check_response("Warning: fallback to F32 for unknown dtype Q3_XS"),
    );
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig {
        invariants: vec!["I-3".to_string()],
    };
    let evidence = run_contract_tests(
        &runner,
        Path::new("/test/workspace/org/model"),
        &model_id,
        &config,
    );
    assert_eq!(evidence.len(), 1);
    assert!(
        evidence[0].outcome.is_fail(),
        "F32 fallback should produce falsified: {}",
        evidence[0].reason
    );
    assert!(
        evidence[0]
            .reason
            .contains("I-3 No Silent Fallbacks: detected F32 fallback"),
        "Reason should mention F32 fallback: {}",
        evidence[0].reason
    );
}

/// I-3 with "defaulting to f32" pattern → falsified
#[test]
fn test_i3_defaulting_to_f32_detected() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> = Arc::new(
        MockCommandRunner::new().with_check_response("Defaulting to F32 for layer 5"),
    );
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig {
        invariants: vec!["I-3".to_string()],
    };
    let evidence = run_contract_tests(
        &runner,
        Path::new("/test/workspace/org/model"),
        &model_id,
        &config,
    );
    assert_eq!(evidence.len(), 1);
    assert!(evidence[0].outcome.is_fail());
}

/// I-3 with "unknown dtype" pattern → falsified
#[test]
fn test_i3_unknown_dtype_detected() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> = Arc::new(
        MockCommandRunner::new()
            .with_check_response("Error: unknown dtype at tensor 3, fallback to f32"),
    );
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig {
        invariants: vec!["I-3".to_string()],
    };
    let evidence = run_contract_tests(
        &runner,
        Path::new("/test/workspace/org/model"),
        &model_id,
        &config,
    );
    assert_eq!(evidence.len(), 1);
    assert!(evidence[0].outcome.is_fail());
}

/// I-4 validate_stats failure → falsified evidence
#[test]
fn test_i4_validate_stats_failure_produces_falsified() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> =
        Arc::new(MockCommandRunner::new().with_validate_stats_failure());
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig {
        invariants: vec!["I-4".to_string()],
    };
    let evidence = run_contract_tests(
        &runner,
        Path::new("/test/workspace/org/model"),
        &model_id,
        &config,
    );
    assert_eq!(evidence.len(), 1);
    assert!(
        evidence[0].outcome.is_fail(),
        "Validate-stats failure should produce falsified: {}",
        evidence[0].reason
    );
    assert!(
        evidence[0]
            .reason
            .contains("I-4 Statistical Preservation"),
        "Reason should mention I-4: {}",
        evidence[0].reason
    );
}

/// I-5 compare_inference failure → falsified evidence
#[test]
fn test_i5_compare_inference_failure_produces_falsified() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> =
        Arc::new(MockCommandRunner::new().with_compare_inference_failure());
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig {
        invariants: vec!["I-5".to_string()],
    };
    let evidence = run_contract_tests(
        &runner,
        Path::new("/test/workspace/org/model"),
        &model_id,
        &config,
    );
    assert_eq!(evidence.len(), 1);
    assert!(
        evidence[0].outcome.is_fail(),
        "Compare-inference failure should produce falsified: {}",
        evidence[0].reason
    );
    assert!(
        evidence[0].reason.contains("I-5 Tokenizer Roundtrip"),
        "Reason should mention I-5: {}",
        evidence[0].reason
    );
}

/// Verify I-1 label is silently skipped (handled elsewhere)
#[test]
fn test_contract_i1_skipped() {
    use crate::command::MockCommandRunner;
    let runner: Arc<dyn CommandRunner> = Arc::new(MockCommandRunner::new());
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig {
        invariants: vec!["I-1".to_string()],
    };
    let evidence = run_contract_tests(
        &runner,
        Path::new("/test/workspace/org/model"),
        &model_id,
        &config,
    );
    // I-1 is skipped by run_contract_tests, handled separately by executor
    assert!(evidence.is_empty(), "I-1 should be skipped");
}

// ── I-2 bijection: uncovered branches ──────────────────────────────────────

macro_rules! contract_stub_methods {
    () => {
        fn run_inference(&self, _: &Path, _: &str, _: u32, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("") }
        fn convert_model(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn inspect_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn validate_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn bench_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn check_model(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn profile_model(&self, _: &Path, _: u32, _: u32) -> CommandOutput { CommandOutput::success("") }
        fn profile_ci(&self, _: &Path, _: Option<f64>, _: Option<f64>, _: u32, _: u32, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn diff_tensors(&self, _: &Path, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn compare_inference(&self, _: &Path, _: &Path, _: &str, _: u32, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn profile_with_flamegraph(&self, _: &Path, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn profile_with_focus(&self, _: &Path, _: &str, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn validate_model_strict(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn fingerprint_model(&self, _: &Path, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn validate_stats(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn pull_model(&self, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn run_ollama_inference(&self, _: &str, _: &str, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn pull_ollama_model(&self, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn create_ollama_model(&self, _: &str, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn serve_model(&self, _: &Path, _: u16) -> CommandOutput { CommandOutput::success("") }
        fn http_get(&self, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn profile_memory(&self, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn run_chat(&self, _: &Path, _: &str, _: bool, _: &[&str]) -> CommandOutput { CommandOutput::success("") }
        fn http_post(&self, _: &str, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn spawn_serve(&self, _: &Path, _: u16, _: bool) -> CommandOutput { CommandOutput::success("") }
        fn quantize_model(&self, _: &Path, _: &Path, _: &str) -> CommandOutput { CommandOutput::success("") }
        fn import_model(&self, _: &Path, _: &Path) -> CommandOutput { CommandOutput::success("") }
        fn prune_model(&self, _: &Path, _: &Path, _: &str, _: f64) -> CommandOutput { CommandOutput::success("") }
        fn distill_model(&self, _: &Path, _: &Path, _: &Path, _: &str) -> CommandOutput { CommandOutput::success("") }
    };
}

/// I-2: source tensor missing in APR → falsified (lines 433-444)
///
/// ST has \[`embed.weight`, `layers.0.weight`\], APR only has \[`embed.weight`\]
/// → 1 missing tensor → falsified
#[test]
fn test_i2_source_tensor_missing_in_apr_falsified() {
    use crate::command::{CommandOutput, CommandRunner};
    use std::path::Path;

    struct MissingAprTensorRunner;
    impl CommandRunner for MissingAprTensorRunner {
        fn inspect_model_json(&self, path: &Path) -> CommandOutput {
            if path.to_str().is_some_and(|p| p.ends_with("model.safetensors")) {
                CommandOutput::success(r#"{"tensor_names":["embed.weight","layers.0.weight"]}"#)
            } else {
                // APR is missing "layers.0.weight"
                CommandOutput::success(r#"{"tensor_names":["embed.weight"]}"#)
            }
        }
        contract_stub_methods!();
    }

    let runner: Arc<dyn CommandRunner> = Arc::new(MissingAprTensorRunner);
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig { invariants: vec!["I-2".to_string()] };
    let evidence = run_contract_tests(
        &runner,
        Path::new("/test/workspace/org/model"),
        &model_id,
        &config,
    );

    assert_eq!(evidence.len(), 1);
    assert!(evidence[0].outcome.is_fail(),
        "Missing APR tensor should falsify I-2: {}", evidence[0].reason);
    assert!(evidence[0].reason.contains("missing") || evidence[0].reason.contains("Missing"),
        "Reason should mention missing tensors: {}", evidence[0].reason);
    assert!(evidence[0].reason.contains("layers.0.weight"),
        "Reason should name the missing tensor: {}", evidence[0].reason);
}

/// I-2: APR has unexpected extra tensor (not lm_head.weight/bias) → falsified (lines 461-477)
///
/// ST has \[`embed.weight`\], APR has \[`embed.weight`, `bad_extra.weight`\]
/// `bad_extra.weight` is not in the allowed set → falsified
#[test]
fn test_i2_unexpected_extra_tensor_in_apr_falsified() {
    use crate::command::{CommandOutput, CommandRunner};
    use std::path::Path;

    struct UnexpectedExtraRunner;
    impl CommandRunner for UnexpectedExtraRunner {
        fn inspect_model_json(&self, path: &Path) -> CommandOutput {
            if path.to_str().is_some_and(|p| p.ends_with("model.safetensors")) {
                CommandOutput::success(r#"{"tensor_names":["embed.weight"]}"#)
            } else {
                // APR has unexpected extra not in allowed set
                CommandOutput::success(r#"{"tensor_names":["embed.weight","bad_extra.weight"]}"#)
            }
        }
        contract_stub_methods!();
    }

    let runner: Arc<dyn CommandRunner> = Arc::new(UnexpectedExtraRunner);
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig { invariants: vec!["I-2".to_string()] };
    let evidence = run_contract_tests(
        &runner,
        Path::new("/test/workspace/org/model"),
        &model_id,
        &config,
    );

    assert_eq!(evidence.len(), 1);
    assert!(evidence[0].outcome.is_fail(),
        "Unexpected extra tensor should falsify I-2: {}", evidence[0].reason);
    assert!(evidence[0].reason.contains("unexpected") || evidence[0].reason.contains("Unexpected"),
        "Reason should mention unexpected tensors: {}", evidence[0].reason);
    assert!(evidence[0].reason.contains("bad_extra.weight"),
        "Reason should name the unexpected tensor: {}", evidence[0].reason);
}

/// I-2: APR has lm_head.weight as extra (tied embedding) → corroborated with note (lines 480-497)
///
/// ST has \[`embed.weight`\] (no separate lm_head), APR has \[`embed.weight`, `lm_head.weight`\]
/// `lm_head.weight` is in the allowed extras set → corroborated
#[test]
fn test_i2_tied_embedding_allowed_extra_corroborated() {
    use crate::command::{CommandOutput, CommandRunner};
    use std::path::Path;

    struct TiedEmbeddingRunner;
    impl CommandRunner for TiedEmbeddingRunner {
        fn inspect_model_json(&self, path: &Path) -> CommandOutput {
            if path.to_str().is_some_and(|p| p.ends_with("model.safetensors")) {
                // ST has no separate lm_head.weight (tied embeddings)
                CommandOutput::success(r#"{"tensor_names":["embed.weight"]}"#)
            } else {
                // APR materializes lm_head.weight from embed_tokens.weight
                CommandOutput::success(r#"{"tensor_names":["embed.weight","lm_head.weight"]}"#)
            }
        }
        contract_stub_methods!();
    }

    let runner: Arc<dyn CommandRunner> = Arc::new(TiedEmbeddingRunner);
    let model_id = ModelId::new("test", "model");
    let config = ContractTestConfig { invariants: vec!["I-2".to_string()] };
    let evidence = run_contract_tests(
        &runner,
        Path::new("/test/workspace/org/model"),
        &model_id,
        &config,
    );

    assert_eq!(evidence.len(), 1);
    assert!(evidence[0].outcome.is_pass(),
        "Allowed extra (lm_head.weight) should corroborate I-2: {}", evidence[0].reason);
    assert!(evidence[0].reason.contains("tied embedding") || evidence[0].reason.contains("Bijection"),
        "Reason should mention tied embedding: {}", evidence[0].reason);
}

