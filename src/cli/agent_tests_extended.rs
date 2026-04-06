use super::*;
use tempfile::NamedTempFile;

#[test]
fn test_status_command() {
    let toml = r#"
name = "status-test"
version = "1.0.0"
privacy = "Sovereign"

[model]
max_tokens = 2048
temperature = 0.5
system_prompt = "hi"

[resources]
max_iterations = 15
max_tool_calls = 30
max_cost_usd = 1.50

[[capabilities]]
type = "rag"

[[capabilities]]
type = "memory"
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let result = cmd_agent_status(&tmp.path().to_path_buf());
    assert!(result.is_ok());
}

#[test]
fn test_status_command_no_capabilities() {
    let toml = r#"
name = "empty-caps"
[model]
system_prompt = "x"
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let result = cmd_agent_status(&tmp.path().to_path_buf());
    assert!(result.is_ok());
}

#[test]
fn test_pool_command_fan_out() {
    let toml = r#"
name = "pool-agent"
version = "1.0.0"
[model]
system_prompt = "You help."
[resources]
max_iterations = 5
"#;
    let tmp1 = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp1.path(), toml).expect("write");
    let tmp2 = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp2.path(), toml).expect("write");

    let manifests = vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()];
    let result = cmd_agent_pool(&manifests, "hello", None);
    assert!(result.is_ok());
}

#[test]
fn test_pool_command_with_concurrency() {
    let toml = r#"
name = "pool-concurrent"
[model]
system_prompt = "hi"
[resources]
max_iterations = 3
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");

    let manifests = vec![tmp.path().to_path_buf()];
    let result = cmd_agent_pool(&manifests, "test", Some(1));
    assert!(result.is_ok());
}

#[test]
fn test_try_auto_pull_no_repo_is_noop() {
    let manifest = batuta::agent::AgentManifest::default();
    let result = try_auto_pull(&manifest);
    assert!(result.is_ok());
}

#[test]
fn test_try_auto_pull_with_repo_errors() {
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.model.model_repo = Some("test-org/fake-model".into());
    let result = try_auto_pull(&manifest);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("auto-pull failed"), "unexpected error: {err_msg}",);
}

#[test]
fn test_run_command_with_auto_pull_flag() {
    let toml = r#"
name = "auto-pull-test"
[model]
system_prompt = "hi"
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");

    let result = cmd_agent_run(&tmp.path().to_path_buf(), "hello", None, false, true, false);
    assert!(result.is_ok());
}

#[test]
fn test_validate_with_check_inference() {
    let toml = r#"
name = "g2-test"
version = "1.0.0"
[model]
system_prompt = "hi"
[resources]
max_iterations = 5
"#;
    let tmp = NamedTempFile::new().expect("tmp file");
    std::fs::write(tmp.path(), toml).expect("write");
    let result = cmd_agent_validate(&tmp.path().to_path_buf(), false, true);
    assert!(result.is_ok(), "G2 with mock driver should pass: {result:?}",);
}

#[test]
fn test_char_entropy() {
    use agent_helpers::char_entropy;

    let low = char_entropy("aaaaaaaaaa");
    assert!(low < 0.1, "repeated char entropy: {low}");

    let normal = char_entropy("Hello, I am operational.");
    assert!(normal > 2.0 && normal < 5.0, "normal text entropy: {normal}",);

    assert_eq!(char_entropy(""), 0.0);
}

#[test]
fn test_truncate_str() {
    use agent_helpers::truncate_str;

    assert_eq!(truncate_str("short", 10), "short");
    assert_eq!(truncate_str("hello world!", 8), "hello...");
    assert_eq!(truncate_str("", 5), "");
}

#[test]
fn test_build_tool_registry_spawn() {
    use batuta::agent::capability::Capability;
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities = vec![Capability::Spawn { max_depth: 2 }];
    let registry = build_tool_registry(&manifest);
    assert!(
        registry.get("spawn_agent").is_none(),
        "spawn requires register_spawn_tool with driver"
    );
}

#[test]
fn test_register_spawn_tool_with_driver() {
    use batuta::agent::capability::Capability;
    use batuta::agent::driver::mock::MockDriver;

    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities = vec![Capability::Spawn { max_depth: 3 }];
    let mut registry = build_tool_registry(&manifest);
    let driver: std::sync::Arc<dyn batuta::agent::driver::LlmDriver> =
        std::sync::Arc::new(MockDriver::single_response("x"));
    register_spawn_tool(&mut registry, &manifest, driver);
    assert!(registry.get("spawn_agent").is_some());
}

#[test]
#[cfg(feature = "agents-browser")]
fn test_build_tool_registry_browser() {
    use batuta::agent::capability::Capability;
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities = vec![Capability::Browser];
    let registry = build_tool_registry(&manifest);
    assert!(registry.get("browser").is_some());
}

#[test]
fn test_build_tool_registry_network() {
    use batuta::agent::capability::Capability;
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities =
        vec![Capability::Network { allowed_hosts: vec!["api.example.com".into()] }];
    let registry = build_tool_registry(&manifest);
    assert!(registry.get("network").is_some());
}

#[test]
#[cfg(feature = "rag")]
fn test_build_tool_registry_rag() {
    use batuta::agent::capability::Capability;
    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities = vec![Capability::Rag];
    let registry = build_tool_registry(&manifest);
    assert!(registry.get("rag").is_some());
}

#[test]
fn test_register_inference_tool_with_driver() {
    use batuta::agent::capability::Capability;
    use batuta::agent::driver::mock::MockDriver;

    let mut manifest = batuta::agent::AgentManifest::default();
    manifest.capabilities = vec![Capability::Inference];
    let mut registry = build_tool_registry(&manifest);
    let driver: std::sync::Arc<dyn batuta::agent::driver::LlmDriver> =
        std::sync::Arc::new(MockDriver::single_response("x"));
    register_inference_tool(&mut registry, &manifest, driver);
    assert!(registry.get("inference").is_some());
}

#[test]
fn test_register_inference_tool_no_capability() {
    use batuta::agent::driver::mock::MockDriver;

    let manifest = batuta::agent::AgentManifest::default();
    let mut registry = build_tool_registry(&manifest);
    let driver: std::sync::Arc<dyn batuta::agent::driver::LlmDriver> =
        std::sync::Arc::new(MockDriver::single_response("x"));
    register_inference_tool(&mut registry, &manifest, driver);
    assert!(
        registry.get("inference").is_none(),
        "should not register without Inference capability"
    );
}

#[test]
#[cfg(feature = "agents-mcp")]
fn test_register_mcp_tools_no_servers() {
    let manifest = batuta::agent::AgentManifest::default();
    let mut registry = build_tool_registry(&manifest);
    let rt =
        tokio::runtime::Builder::new_current_thread().enable_all().build().expect("tokio runtime");
    rt.block_on(register_mcp_tools(&mut registry, &manifest));
    assert!(registry.get("mcp_").is_none());
}

#[test]
fn test_register_spawn_tool_no_capability() {
    use batuta::agent::driver::mock::MockDriver;

    let manifest = batuta::agent::AgentManifest::default();
    let mut registry = build_tool_registry(&manifest);
    let driver: std::sync::Arc<dyn batuta::agent::driver::LlmDriver> =
        std::sync::Arc::new(MockDriver::single_response("x"));
    register_spawn_tool(&mut registry, &manifest, driver);
    assert!(registry.get("spawn_agent").is_none(), "should not register without Spawn capability");
}
