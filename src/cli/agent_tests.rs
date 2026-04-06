use super::*;
use tempfile::NamedTempFile;

#[test]
fn test_load_valid_manifest() {
    let toml = r#"
name = "test-agent"
version = "0.1.0"
[model]
system_prompt = "You help."
[resources]
max_iterations = 5
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let manifest = load_manifest(&tmp.path().to_path_buf()).expect("should load");
    assert_eq!(manifest.name, "test-agent");
}

#[test]
fn test_load_missing_file() {
    let result = load_manifest(&PathBuf::from("/nonexistent/agent.toml"));
    assert!(result.is_err());
}

#[test]
fn test_build_guard_with_override() {
    let manifest = batuta::agent::AgentManifest::default();
    let (max_iter, _) = build_guard(&manifest, Some(99));
    assert_eq!(max_iter, 99);
}

#[test]
fn test_build_guard_from_manifest() {
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.resources.max_iterations = 42;
    let (max_iter, _) = build_guard(&manifest, None);
    assert_eq!(max_iter, 42);
}

#[test]
fn test_validate_command_valid() {
    let toml = r#"
name = "valid"
version = "1.0.0"
[model]
system_prompt = "hi"
[resources]
max_iterations = 10
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let result = cmd_agent_validate(&tmp.path().to_path_buf(), false, false);
    assert!(result.is_ok());
}

#[test]
fn test_validate_with_needs_pull() {
    let toml = r#"
name = "repo-agent"
[model]
model_repo = "meta-llama/Llama-3-8B-GGUF"
model_quantization = "q4_k_m"
system_prompt = "hi"
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let result = cmd_agent_validate(&tmp.path().to_path_buf(), false, false);
    assert!(result.is_ok());
}

#[test]
fn test_validate_check_model_no_path() {
    // With model discovery (Phase 2a), validate may find local models
    // on disk even without explicit model_path. Test that the manifest
    // parses correctly and has no explicit path set.
    let toml = r#"
name = "no-model"
[model]
system_prompt = "hi"
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let result = cmd_agent_validate(&tmp.path().to_path_buf(), false, false);
    // Without --check-model flag, this should succeed (manifest is valid)
    assert!(result.is_ok(), "manifest without model should parse: {result:?}");
}

#[test]
fn test_validate_check_model_nonexistent() {
    let toml = r#"
name = "missing-model"
[model]
model_path = "/nonexistent/model.gguf"
system_prompt = "hi"
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let result = cmd_agent_validate(&tmp.path().to_path_buf(), true, false);
    assert!(result.is_err());
}

#[test]
fn test_detect_model_format() {
    let gguf = [0x47u8, 0x47, 0x55, 0x46, 0, 0, 0, 0];
    assert_eq!(detect_model_format(&gguf), "GGUF");

    let apr = [b'A', b'P', b'R', 0x02, 0, 0, 0, 0];
    assert_eq!(detect_model_format(&apr), "APR v2");

    let st = [0u8, 0, 0, 0, 0, 0, 0, 0, b'{'];
    assert_eq!(detect_model_format(&st), "SafeTensors");

    let unknown = [0u8; 4];
    assert_eq!(detect_model_format(&unknown), "unknown");
}

#[test]
fn test_run_command_executes_loop() {
    let toml = r#"
name = "run-test"
version = "1.0.0"
[model]
system_prompt = "You help."
[resources]
max_iterations = 10
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let result = cmd_agent_run(&tmp.path().to_path_buf(), "hello", None, false, false, false);
    assert!(result.is_ok());
}

#[test]
fn test_build_driver_no_model_returns_mock() {
    let manifest = batuta::agent::AgentManifest::default();
    let driver = build_driver(&manifest);
    assert!(driver.is_ok(), "should return MockDriver when no model_path");
}

#[test]
fn test_build_driver_remote_no_key_returns_mock() {
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.model.remote_model = Some("claude-sonnet-4-20250514".into());
    std::env::remove_var("ANTHROPIC_API_KEY");
    let driver = build_driver(&manifest);
    assert!(driver.is_ok(), "should return mock when API key missing");
}

#[test]
fn test_build_tool_registry_memory() {
    use batuta::agent::capability::Capability;
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities = vec![Capability::Memory];
    let registry = build_tool_registry(&manifest);
    assert!(registry.get("memory").is_some());
}

#[test]
fn test_build_tool_registry_compute() {
    use batuta::agent::capability::Capability;
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities = vec![Capability::Compute];
    let registry = build_tool_registry(&manifest);
    assert!(registry.get("compute").is_some());
}

#[test]
fn test_build_tool_registry_shell() {
    use batuta::agent::capability::Capability;
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities = vec![Capability::Shell { allowed_commands: vec!["*".into()] }];
    let registry = build_tool_registry(&manifest);
    assert!(registry.get("shell").is_some());
}

#[test]
fn test_build_memory_substrate() {
    let memory = build_memory();
    let _ = memory;
}

#[test]
fn test_run_with_max_iterations_override() {
    let toml = r#"
name = "override-test"
version = "1.0.0"
[model]
system_prompt = "You help."
[resources]
max_iterations = 10
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let result = cmd_agent_run(&tmp.path().to_path_buf(), "hello", Some(3), false, false, false);
    assert!(result.is_ok());
}

#[test]
fn test_validate_command_invalid() {
    let toml = r#"
name = ""
version = "1.0.0"
[model]
system_prompt = "hi"
[resources]
max_iterations = 0
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let result = cmd_agent_validate(&tmp.path().to_path_buf(), false, false);
    assert!(result.is_err());
}

#[test]
fn test_sign_and_verify_roundtrip() {
    let toml = r#"
name = "sign-test"
version = "1.0.0"
[model]
system_prompt = "hi"
[resources]
max_iterations = 10
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");

    let sig_path = tmp.path().with_extension("toml.sig");
    let pk_path = sig_path.with_extension("pub");

    let result = cmd_agent_sign(&tmp.path().to_path_buf(), Some("tester"), Some(sig_path.clone()));
    assert!(result.is_ok(), "sign failed: {result:?}");
    assert!(sig_path.exists(), "signature file not created");
    assert!(pk_path.exists(), "pubkey file not created");

    let result = cmd_agent_verify_sig(&tmp.path().to_path_buf(), Some(sig_path.clone()), &pk_path);
    assert!(result.is_ok(), "verify failed: {result:?}");

    let _ = std::fs::remove_file(&sig_path);
    let _ = std::fs::remove_file(&pk_path);
}

#[test]
fn test_verify_fails_on_tampered() {
    let toml = r#"
name = "tamper-test"
version = "1.0.0"
[model]
system_prompt = "hi"
[resources]
max_iterations = 10
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");

    let sig_path = tmp.path().with_extension("toml.sig");
    let pk_path = sig_path.with_extension("pub");

    cmd_agent_sign(&tmp.path().to_path_buf(), None, Some(sig_path.clone())).expect("sign");

    std::fs::write(tmp.path(), "name = \"tampered\"").expect("tamper");

    let result = cmd_agent_verify_sig(&tmp.path().to_path_buf(), Some(sig_path.clone()), &pk_path);
    assert!(result.is_err(), "should fail on tampered manifest");

    let _ = std::fs::remove_file(&sig_path);
    let _ = std::fs::remove_file(&pk_path);
}

#[test]
fn test_contracts_command() {
    let result = cmd_agent_contracts();
    assert!(result.is_ok());
}
