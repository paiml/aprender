//! Validation and edge-case tests for agent manifest.
//!
//! Extracted from `manifest_tests.rs` for QA-002 (≤500 lines).
//! Covers: model path resolution, auto-pull, validation edge cases,
//! resource quotas, model config defaults, MCP transport parsing.

use std::path::PathBuf;

use super::*;

#[test]
fn test_resolve_model_path_default_quant() {
    let mut config = ModelConfig::default();
    config.model_repo = Some("test/model".into());
    let resolved = config.resolve_model_path();
    assert!(resolved.is_some());
    assert!(resolved.expect("path").to_string_lossy().contains("q4_k_m"));
}

#[test]
fn test_resolve_model_path_none_without_explicit() {
    // With model discovery (Phase 2a), resolve_model_path may find models
    // on disk even with default config. Test that no explicit path is set.
    let config = ModelConfig::default();
    assert!(config.model_path.is_none(), "default should have no explicit model_path");
    assert!(config.model_repo.is_none(), "default should have no model_repo");
}

#[test]
fn test_needs_pull_no_repo() {
    let config = ModelConfig::default();
    assert!(config.needs_pull().is_none());
}

#[test]
fn test_needs_pull_with_explicit_path() {
    let mut config = ModelConfig::default();
    config.model_path = Some("/models/llama.gguf".into());
    config.model_repo = Some("meta-llama/Llama-3-8B".into());
    assert!(config.needs_pull().is_none());
}

#[test]
fn test_needs_pull_with_repo() {
    let mut config = ModelConfig::default();
    config.model_repo = Some("meta-llama/Llama-3-8B-GGUF".into());
    assert_eq!(config.needs_pull(), Some("meta-llama/Llama-3-8B-GGUF"),);
}

#[test]
fn test_auto_pull_no_repo() {
    let config = ModelConfig::default();
    let err = config.auto_pull(10).unwrap_err();
    assert!(matches!(err, AutoPullError::NoRepo), "expected NoRepo, got: {err}",);
}

#[test]
fn test_auto_pull_error_display() {
    let no_repo = AutoPullError::NoRepo;
    assert!(no_repo.to_string().contains("no model_repo"));

    let not_installed = AutoPullError::NotInstalled;
    assert!(not_installed.to_string().contains("not found"));
    assert!(not_installed.to_string().contains("apr-cli"));

    let subprocess = AutoPullError::Subprocess("boom".into());
    assert_eq!(subprocess.to_string(), "boom");

    let io_err = AutoPullError::Io("disk full".into());
    assert_eq!(io_err.to_string(), "disk full");
}

#[test]
fn test_auto_pull_apr_not_in_path() {
    let mut config = ModelConfig::default();
    config.model_repo = Some("test-org/nonexistent-model".into());

    let result = config.auto_pull(5);
    assert!(result.is_err());
}

#[test]
fn test_validate_zero_tool_calls() {
    let mut manifest = AgentManifest::default();
    manifest.resources.max_tool_calls = 0;
    let errors = manifest.validate().unwrap_err();
    assert!(errors.iter().any(|e| e.contains("max_tool_calls")));
}

#[test]
fn test_validate_zero_max_tokens() {
    let mut manifest = AgentManifest::default();
    manifest.model.max_tokens = 0;
    let errors = manifest.validate().unwrap_err();
    assert!(errors.iter().any(|e| e.contains("max_tokens")));
}

#[test]
fn test_validate_negative_temperature() {
    let mut manifest = AgentManifest::default();
    manifest.model.temperature = -0.5;
    let errors = manifest.validate().unwrap_err();
    assert!(errors.iter().any(|e| e.contains("temperature")));
}

#[test]
fn test_validate_multiple_errors() {
    let mut manifest = AgentManifest::default();
    manifest.name = String::new();
    manifest.resources.max_iterations = 0;
    manifest.resources.max_tool_calls = 0;
    let errors = manifest.validate().unwrap_err();
    assert!(errors.len() >= 3);
}

#[test]
fn test_auto_pull_error_is_std_error() {
    let err: Box<dyn std::error::Error> = Box::new(AutoPullError::NoRepo);
    assert!(err.to_string().contains("no model_repo"));
}

#[test]
fn test_resource_quota_defaults() {
    let q = ResourceQuota::default();
    assert_eq!(q.max_iterations, 20);
    assert_eq!(q.max_tool_calls, 50);
    assert_eq!(q.max_cost_usd, 0.0);
    assert!(q.max_tokens_budget.is_none());
}

#[test]
fn test_resource_quota_with_token_budget() {
    let toml = r#"
name = "budget-agent"
[model]
system_prompt = "hi"
[resources]
max_iterations = 10
max_tool_calls = 20
max_tokens_budget = 100000
"#;
    let manifest = AgentManifest::from_toml(toml).expect("parse");
    assert_eq!(manifest.resources.max_tokens_budget, Some(100000),);
}

#[test]
fn test_model_config_defaults() {
    let config = ModelConfig::default();
    assert!(config.model_path.is_none());
    assert!(config.remote_model.is_none());
    assert!(config.model_repo.is_none());
    assert!(config.model_quantization.is_none());
    assert_eq!(config.max_tokens, 4096);
    assert!((config.temperature - 0.3).abs() < 0.001);
    assert!(config.system_prompt.contains("helpful assistant"));
    assert!(config.context_window.is_none());
}

#[test]
fn test_auto_pull_with_repo_and_quant() {
    let mut config = ModelConfig::default();
    config.model_repo = Some("test-org/test-model-GGUF".into());
    config.model_quantization = Some("q6_k".into());

    let err = config.auto_pull(2).unwrap_err();
    assert!(
        matches!(err, AutoPullError::NotInstalled | AutoPullError::Subprocess(_)),
        "expected NotInstalled or Subprocess, got: {err}"
    );
}

#[test]
fn test_which_apr_not_in_path() {
    let result = which_apr();
    match result {
        Ok(path) => assert!(path.exists()),
        Err(AutoPullError::NotInstalled) => {}
        Err(other) => {
            panic!("expected NotInstalled, got: {other}")
        }
    }
}

#[test]
fn test_needs_pull_with_existing_file() {
    let mut config = ModelConfig::default();
    config.model_path = Some("/dev/null".into());
    assert!(config.needs_pull().is_none());
}

#[test]
fn test_validate_empty_toml() {
    let manifest = AgentManifest::from_toml("").expect("parse empty");
    assert_eq!(manifest.name, "unnamed-agent");
    assert!(manifest.validate().is_ok());
}

#[test]
fn test_context_window_override() {
    let toml = r#"
name = "ctx-agent"
[model]
system_prompt = "hi"
context_window = 8192
"#;
    let manifest = AgentManifest::from_toml(toml).expect("parse");
    assert_eq!(manifest.model.context_window, Some(8192));
}

#[cfg(feature = "agents-mcp")]
#[test]
fn test_mcp_websocket_transport_parse() {
    let toml = r#"
name = "ws-agent"
privacy = "Standard"
[model]
system_prompt = "hi"
[[capabilities]]
type = "memory"
[[mcp_servers]]
name = "ws-server"
transport = "web_socket"
url = "wss://example.com/mcp"
"#;
    let manifest = AgentManifest::from_toml(toml).expect("parse failed");
    assert_eq!(manifest.mcp_servers.len(), 1);
    assert!(matches!(manifest.mcp_servers[0].transport, McpTransport::WebSocket));
    assert!(manifest.validate().is_ok());
}

#[cfg(feature = "agents-mcp")]
#[test]
fn test_mcp_empty_server_name() {
    let toml = r#"
name = "bad-mcp"
privacy = "Standard"
[model]
system_prompt = "hi"
[[capabilities]]
type = "memory"
[[mcp_servers]]
name = ""
transport = "stdio"
command = ["echo"]
"#;
    let manifest = AgentManifest::from_toml(toml).expect("parse failed");
    let errors = manifest.validate().unwrap_err();
    assert!(errors.iter().any(|e| e.contains("name must not be empty")));
}

#[test]
fn test_example_manifests_valid() {
    let basic = include_str!("../../examples/agent.toml");
    let manifest = AgentManifest::from_toml(basic).expect("basic manifest parse");
    assert!(manifest.validate().is_ok());

    let full = include_str!("../../examples/agent-full.toml");
    let manifest = AgentManifest::from_toml(full).expect("full manifest parse");
    assert!(manifest.validate().is_ok());
    assert_eq!(manifest.capabilities.len(), 5);
}
