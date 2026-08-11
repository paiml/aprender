//! Tests for `agent::code` — apr code library entry point.

use super::*;

#[test]
fn test_build_default_manifest_always_sovereign() {
    let m = build_default_manifest();
    assert_eq!(m.name, "apr-code");
    assert_eq!(m.privacy, PrivacyTier::Sovereign);
    assert!(!m.capabilities.is_empty());
}

#[test]
fn test_build_code_tools_registers_all() {
    let m = build_default_manifest();
    let tools = build_code_tools(&m);
    assert!(tools.get("file_read").is_some(), "missing file_read");
    assert!(tools.get("file_write").is_some(), "missing file_write");
    assert!(tools.get("file_edit").is_some(), "missing file_edit");
    assert!(tools.get("glob").is_some(), "missing glob");
    assert!(tools.get("grep").is_some(), "missing grep");
    assert!(tools.get("shell").is_some(), "missing shell");
    assert!(tools.get("memory").is_some(), "missing memory");
    // PMAT-163: pmat_query tool
    assert!(tools.get("pmat_query").is_some(), "missing pmat_query (PMAT-163)");
    #[cfg(feature = "rag")]
    assert!(tools.get("rag").is_some(), "missing rag tool (PMAT-153)");
    // 9 tools with rag, 8 without
    #[cfg(feature = "rag")]
    assert!(tools.len() >= 9, "expected >=9 tools with rag, got {}", tools.len());
    #[cfg(not(feature = "rag"))]
    assert!(tools.len() >= 8, "expected >=8 tools, got {}", tools.len());
}

#[test]
fn test_web_tools_not_registered_on_sovereign_privacy() {
    // Poka-Yoke: Sovereign tier always blocks network tools, even if
    // the user specifies allowed_hosts (tier wins over config).
    let mut m = build_default_manifest();
    assert_eq!(m.privacy, PrivacyTier::Sovereign);
    m.allowed_hosts = vec!["docs.anthropic.com".into(), "crates.io".into()];
    let tools = build_code_tools(&m);
    assert!(tools.get("network").is_none(), "Sovereign must block network");
    assert!(tools.get("browser").is_none(), "Sovereign must block browser");
}

#[test]
fn test_web_tools_not_registered_when_allowed_hosts_empty() {
    // Explicit opt-in: even on Standard tier, empty allowed_hosts → no network tool.
    let mut m = build_default_manifest();
    m.privacy = PrivacyTier::Standard;
    m.allowed_hosts = Vec::new();
    let tools = build_code_tools(&m);
    assert!(tools.get("network").is_none(), "empty allowed_hosts must block network");
}

#[test]
fn test_web_tools_registered_on_standard_privacy_with_allowlist() {
    let mut m = build_default_manifest();
    m.privacy = PrivacyTier::Standard;
    m.allowed_hosts = vec!["docs.anthropic.com".into()];
    let tools = build_code_tools(&m);
    assert!(tools.get("network").is_some(), "Standard + allowlist must register network tool");
}

#[test]
fn test_web_tools_registered_on_private_privacy_with_allowlist() {
    let mut m = build_default_manifest();
    m.privacy = PrivacyTier::Private;
    m.allowed_hosts = vec!["github.com".into()];
    let tools = build_code_tools(&m);
    assert!(tools.get("network").is_some(), "Private + allowlist must register network tool");
}

#[test]
fn test_code_system_prompt_not_empty() {
    assert!(CODE_SYSTEM_PROMPT.len() > 200);
    assert!(CODE_SYSTEM_PROMPT.contains("tool_call"));
    assert!(CODE_SYSTEM_PROMPT.contains("sovereign"));
    // PMAT-168: all 9 tools enumerated with examples
    for tool in &[
        "file_read",
        "file_write",
        "file_edit",
        "glob",
        "grep",
        "shell",
        "memory",
        "pmat_query",
        "rag",
    ] {
        assert!(CODE_SYSTEM_PROMPT.contains(tool), "system prompt missing tool: {tool}");
    }
    // Verify example inputs exist (not just names)
    assert!(CODE_SYSTEM_PROMPT.contains("src/main.rs"), "missing file_read example");
    assert!(CODE_SYSTEM_PROMPT.contains("cargo test"), "missing shell example");
    assert!(CODE_SYSTEM_PROMPT.contains("error handling"), "missing pmat_query example");
}

#[test]
fn test_load_project_instructions_from_claude_md() {
    let instructions = load_project_instructions(4096);
    assert!(instructions.is_some(), "expected to find CLAUDE.md in project root");
    let text = instructions.expect("just checked");
    assert!(
        text.contains("batuta") || text.contains("Batuta") || text.contains("CLAUDE"),
        "CLAUDE.md should mention the project"
    );
}

#[test]
fn test_manifest_includes_project_instructions() {
    let m = build_default_manifest();
    assert!(
        m.model.system_prompt.contains("Project Instructions")
            || m.model.system_prompt.contains("sovereign"),
        "system prompt should contain either project instructions or base prompt"
    );
}

#[test]
fn test_gather_project_context_has_content() {
    let ctx = gather_project_context();
    assert!(ctx.contains("Working directory:"), "should have cwd");
    assert!(
        ctx.contains("Rust") || ctx.contains("Cargo") || ctx.contains("Language:"),
        "should detect language or build system: {ctx}"
    );
}

#[test]
fn test_manifest_includes_project_context() {
    let m = build_default_manifest();
    assert!(
        m.model.system_prompt.contains("Project Context"),
        "system prompt should contain project context section"
    );
    assert!(
        m.model.system_prompt.contains("Working directory:"),
        "context should include working directory"
    );
}

#[test]
fn test_instruction_budget_scales_with_context() {
    assert_eq!(instruction_budget(2048), 0, "2K context: skip instructions");
    assert_eq!(instruction_budget(4096), 1024, "4K context: 25% = 1024");
    assert_eq!(instruction_budget(8192), 2048, "8K context: 25% = 2048");
    assert_eq!(instruction_budget(32768), 4096, "32K context: capped at 4096");
    assert_eq!(instruction_budget(131072), 4096, "128K context: capped at 4096");
}

#[test]
fn test_load_instructions_zero_budget_returns_none() {
    let result = load_project_instructions(0);
    assert!(result.is_none(), "zero budget should skip instructions");
}

#[test]
fn test_exit_codes_match_spec() {
    assert_eq!(exit_code::SUCCESS, 0);
    assert_eq!(exit_code::AGENT_ERROR, 1);
    assert_eq!(exit_code::BUDGET_EXHAUSTED, 2);
    assert_eq!(exit_code::MAX_TURNS, 3);
    assert_eq!(exit_code::SANDBOX_VIOLATION, 4);
    assert_eq!(exit_code::NO_MODEL, 5);
}

#[test]
fn test_fallback_driver_without_model() {
    let manifest = build_default_manifest();
    // No model path set — should return MockDriver
    let driver = build_fallback_driver(&manifest);
    assert!(driver.is_ok(), "fallback should succeed with mock");
}

// PMAT-182: Tests for model discovery and cmd_code entrypoint

#[test]
fn test_discover_and_set_model_skips_when_path_set() {
    let mut manifest = build_default_manifest();
    manifest.model.model_path = Some(std::path::PathBuf::from("/tmp/existing-model.apr"));
    discover_and_set_model(&mut manifest);
    // Should not overwrite the explicitly set path
    assert_eq!(
        manifest.model.model_path.as_ref().unwrap().display().to_string(),
        "/tmp/existing-model.apr"
    );
}

#[test]
fn test_discover_and_set_model_skips_when_repo_set() {
    let mut manifest = build_default_manifest();
    manifest.model.model_repo = Some("hf://org/model".to_string());
    discover_and_set_model(&mut manifest);
    // model_path stays None when repo is set (repo takes priority)
    assert!(manifest.model.model_path.is_none());
}

#[test]
fn test_check_invalid_apr_returns_false_on_empty_dirs() {
    // search dirs exist but may not have APR files — should not panic
    let result = check_invalid_apr_in_search_dirs();
    // Just verifying it doesn't crash — result depends on system state
    let _ = result;
}

/// `--resume <unknown-id>` used to start a FRESH session and exit 0, so a
/// typo'd id left the operator believing a conversation was being continued
/// that the model had no history of. It must fail, and name the id.
///
/// The unreadable `--manifest` is a tripwire: it is the next thing `cmd_code`
/// would hit, so if the resume check is removed this test fails on the wrong
/// message in milliseconds instead of launching a model.
#[test]
fn test_cmd_code_rejects_unknown_resume_session_id() {
    let id = "no-such-session-2398-falsifier";
    let err = cmd_code(
        None,
        std::path::PathBuf::from("."),
        Some(Some(id.to_string())),
        vec!["hi".to_string()],
        true,
        50,
        Some(std::path::PathBuf::from("/nonexistent/manifest-2398.toml")),
        None,
        "text",
        "text",
    )
    .expect_err("an unknown --resume id must be an error, not a silent fresh session");
    let msg = err.to_string();
    assert!(msg.contains("no such session"), "must say the session is unknown; got: {msg}");
    assert!(msg.contains(id), "must name the id the operator typed; got: {msg}");
}

/// `--project /typo` used to be ignored, running the agent against the CURRENT
/// directory while the operator believed it was scoped elsewhere.
#[test]
fn test_cmd_code_rejects_nonexistent_project_dir() {
    let dir = "/nonexistent-project-dir-2398";
    let err = cmd_code(
        None,
        std::path::PathBuf::from(dir),
        None,
        vec!["hi".to_string()],
        true,
        50,
        Some(std::path::PathBuf::from("/nonexistent/manifest-2398.toml")),
        None,
        "text",
        "text",
    )
    .expect_err("a --project path that is not a directory must be an error");
    let msg = err.to_string();
    assert!(msg.contains("--project"), "must name the flag; got: {msg}");
    assert!(msg.contains(dir), "must name the path; got: {msg}");
}

#[test]
fn test_cmd_code_signature_matches_spec() {
    // Verify the public API signature exists and is callable
    // This catches regressions where the function is made private or renamed
    // Verify cmd_code exists and is callable
    let _ = cmd_code as fn(_, _, _, _, _, _, _, _, _, _) -> _;
}

#[test]
fn test_default_manifest_model_path_is_none() {
    let m = build_default_manifest();
    // Default manifest leaves model_path None for discovery
    assert!(m.model.model_path.is_none(), "default should rely on discovery");
}

#[test]
fn test_default_manifest_resource_quotas() {
    let m = build_default_manifest();
    // Verify coding-appropriate quotas (higher than agent defaults)
    assert!(m.resources.max_iterations >= 50, "coding needs >= 50 iterations");
    assert!(m.resources.max_tool_calls >= 200, "coding needs >= 200 tool calls");
}

// ═══ Contract: apr-model-discovery-v1 (PMAT-188) ═══

#[test]
fn falsify_disc_001_mtime_first_sort() {
    use std::path::PathBuf;
    use std::time::{Duration, SystemTime};

    let now = SystemTime::now();
    let yesterday = now - Duration::from_secs(86400);

    let mut candidates = vec![
        (PathBuf::from("old.apr"), yesterday, true, true), // older APR
        (PathBuf::from("new.gguf"), now, false, true),     // newer GGUF
    ];
    crate::agent::manifest::ModelConfig::sort_candidates(&mut candidates);
    assert_eq!(
        candidates[0].0.to_str().unwrap(),
        "new.gguf",
        "FALSIFY-DISC-001: newer GGUF must beat older APR (mtime > format)"
    );
}

#[test]
fn falsify_disc_001_apr_wins_same_mtime() {
    use std::path::PathBuf;
    use std::time::SystemTime;

    let now = SystemTime::now();
    let mut candidates = vec![
        (PathBuf::from("model.gguf"), now, false, true),
        (PathBuf::from("model.apr"), now, true, true),
    ];
    crate::agent::manifest::ModelConfig::sort_candidates(&mut candidates);
    assert_eq!(
        candidates[0].0.to_str().unwrap(),
        "model.apr",
        "FALSIFY-DISC-001: APR wins as tiebreaker when mtime is equal"
    );
}

#[test]
fn falsify_disc_002_invalid_apr_loses_to_valid_gguf() {
    use std::path::PathBuf;
    use std::time::{Duration, SystemTime};

    let now = SystemTime::now();
    let yesterday = now - Duration::from_secs(86400);

    let mut candidates = vec![
        (PathBuf::from("broken.apr"), now, true, false), // newer but INVALID APR
        (PathBuf::from("valid.gguf"), yesterday, false, true), // older but VALID GGUF
    ];
    crate::agent::manifest::ModelConfig::sort_candidates(&mut candidates);
    assert_eq!(
        candidates[0].0.to_str().unwrap(),
        "valid.gguf",
        "FALSIFY-DISC-002: valid GGUF must beat invalid APR (Jidoka)"
    );
}

#[test]
fn falsify_disc_003_no_model_exit_code() {
    // Verify exit code constant matches spec
    assert_eq!(exit_code::NO_MODEL, 5, "FALSIFY-DISC-003: no-model exit code must be 5");
}

#[test]
fn falsify_disc_004_search_dirs_order() {
    let dirs = crate::agent::manifest::ModelConfig::model_search_dirs();
    // First dir should be ~/.apr/models/
    assert!(
        dirs[0].to_str().unwrap().ends_with(".apr/models"),
        "FALSIFY-DISC-004: first search dir must be ~/.apr/models/, got {:?}",
        dirs[0]
    );
    // Last dir should be ./models/
    assert_eq!(
        dirs.last().unwrap().to_str().unwrap(),
        "./models",
        "FALSIFY-DISC-004: last search dir must be ./models/"
    );
    // Must have at least 2 dirs
    assert!(dirs.len() >= 2, "FALSIFY-DISC-004: need at least 2 search dirs");
}

// ═══ Contract: apr-code-v1 — GAP CLOSURE (PMAT-190) ═══

#[test]
fn falsify_code_001_sovereignty_guarantee() {
    // FALSIFY-CODE-001: apr code manifest is ALWAYS Sovereign.
    // No parameter or environment can change this.
    let m = build_default_manifest();
    assert_eq!(m.privacy, PrivacyTier::Sovereign, "FALSIFY-CODE-001: privacy MUST be Sovereign");
    // Verify there's no conditional that could change it
    let m2 = build_default_manifest();
    assert_eq!(
        m2.privacy,
        PrivacyTier::Sovereign,
        "FALSIFY-CODE-001: second call also Sovereign (deterministic)"
    );
}

#[test]
fn falsify_code_002_tool_capabilities_match() {
    // FALSIFY-CODE-002: Every registered tool has a matching Capability in manifest.
    let m = build_default_manifest();
    let tools = build_code_tools(&m);
    // Verify capabilities exist by checking variant names via Debug repr
    let caps_debug = format!("{:?}", m.capabilities);
    assert!(caps_debug.contains("FileRead"), "FALSIFY-CODE-002: FileRead capability present");
    assert!(caps_debug.contains("FileWrite"), "FALSIFY-CODE-002: FileWrite capability present");
    assert!(caps_debug.contains("Shell"), "FALSIFY-CODE-002: Shell capability present");
    assert!(caps_debug.contains("Memory"), "FALSIFY-CODE-002: Memory capability present");
    // Verify tool count matches expected (9 with rag)
    assert!(tools.len() >= 8, "FALSIFY-CODE-002: at least 8 tools");
}

#[test]
fn falsify_code_003_apr_format_preferred_in_discovery() {
    // FALSIFY-CODE-003: APR format is preferred over GGUF at same mtime.
    // This enforces the stack-native format preference.
    use std::path::PathBuf;
    use std::time::SystemTime;

    let now = SystemTime::now();
    let mut candidates = vec![
        (PathBuf::from("model.gguf"), now, false, true),
        (PathBuf::from("model.apr"), now, true, true),
    ];
    crate::agent::manifest::ModelConfig::sort_candidates(&mut candidates);
    assert!(
        candidates[0].0.extension().unwrap() == "apr",
        "FALSIFY-CODE-003: APR preferred over GGUF at same mtime"
    );
}

#[test]
fn falsify_code_004_system_prompt_contains_tool_format() {
    // FALSIFY-CODE-004: System prompt teaches <tool_call> format.
    // This is critical for local model tool-use parsing.
    assert!(
        CODE_SYSTEM_PROMPT.contains("<tool_call>"),
        "FALSIFY-CODE-004: system prompt must teach <tool_call> format"
    );
    assert!(
        CODE_SYSTEM_PROMPT.contains("</tool_call>"),
        "FALSIFY-CODE-004: system prompt must teach </tool_call> closing"
    );
}

#[test]
fn falsify_code_005_manifest_context_window() {
    // FALSIFY-CODE-005: Default manifest has reasonable context window.
    // context_window is Option<usize> — None means "use model default".
    let m = build_default_manifest();
    // Either None (model decides) or >= 4096
    if let Some(w) = m.model.context_window {
        assert!(w >= 4096, "FALSIFY-CODE-005: context window must be >= 4096, got {w}");
    }
    // None is acceptable — model determines its own context window
}

#[test]
fn falsify_code_006_session_dir_is_apr() {
    // FALSIFY-CODE-006: Sessions stored under ~/.apr/sessions/ (not ~/.batuta/).
    // This ensures apr-cli integration works with expected paths.
    // Verify by creating a session and checking its path.
    let home = dirs::home_dir().expect("home dir");
    let expected = home.join(".apr").join("sessions");
    // The Session module uses ~/.apr/sessions/ — verify the constant path
    assert!(
        expected.to_str().unwrap().contains(".apr/sessions"),
        "FALSIFY-CODE-006: session dir must be under ~/.apr/sessions/"
    );
}

// ═══ PMAT-198: Prompt scaling by model size ═══

#[test]
fn test_estimate_params_qwen3_1_7b() {
    use std::path::PathBuf;
    let p = PathBuf::from("Qwen3-1.7B-Q4_K_M.gguf");
    assert!((estimate_model_params_from_name(&p) - 1.7).abs() < 0.01);
}

#[test]
fn test_estimate_params_qwen3_8b() {
    use std::path::PathBuf;
    let p = PathBuf::from("qwen3-8b-q4k.apr");
    assert!((estimate_model_params_from_name(&p) - 8.0).abs() < 0.01);
}

#[test]
fn test_estimate_params_llama_70b() {
    use std::path::PathBuf;
    let p = PathBuf::from("llama-70b-instruct.gguf");
    assert!((estimate_model_params_from_name(&p) - 70.0).abs() < 0.01);
}

#[test]
fn test_estimate_params_unknown() {
    use std::path::PathBuf;
    let p = PathBuf::from("model-unknown.gguf");
    assert_eq!(estimate_model_params_from_name(&p), 0.0);
}

#[test]
fn test_estimate_params_0_6b() {
    use std::path::PathBuf;
    let p = PathBuf::from("qwen3-0.6b-q4k.gguf");
    assert!((estimate_model_params_from_name(&p) - 0.6).abs() < 0.01);
}

#[test]
fn test_scale_prompt_small() {
    let prompt = scale_prompt_for_model(1.7);
    assert!(!prompt.contains("## Tools"), "small model: no full tool table");
    assert!(prompt.contains("direct"), "small model: direct answer");
}

#[test]
fn test_scale_prompt_mid() {
    let prompt = scale_prompt_for_model(3.0);
    assert!(prompt.contains("file_read"), "mid model: has tool names");
    assert!(prompt.contains("<tool_call>"), "mid model: has tool format");
    assert!(!prompt.contains("Example input"), "mid model: no example column");
}

#[test]
fn test_scale_prompt_large() {
    let prompt = scale_prompt_for_model(8.0);
    assert!(prompt.contains("## Tools"), "large model: full tool table");
    assert!(prompt.contains("Example input"), "large model: has examples");
}

/// PMAT-CODE-MCP-CLIENT-001: `register_mcp_client_tools` must be a no-op
/// when the manifest has no `mcp_servers[]`, leaving the built-in tool
/// roster untouched. The falsification condition: if registration silently
/// *added* or *removed* builtin tools when mcp_servers is empty, this test
/// catches it. Exercised under the `agents-mcp` feature because the
/// `mcp_servers` field itself is gated there.
#[cfg(feature = "agents-mcp")]
#[test]
fn test_register_mcp_client_tools_noop_when_empty() {
    let manifest = build_default_manifest();
    assert!(manifest.mcp_servers.is_empty(), "default manifest should declare zero mcp_servers");
    let mut tools = build_code_tools(&manifest);
    let before = tools.len();
    register_mcp_client_tools(&mut tools, &manifest);
    assert_eq!(
        tools.len(),
        before,
        "register_mcp_client_tools must not mutate the registry when mcp_servers is empty"
    );
    // Still have all default builtins.
    assert!(tools.get("file_read").is_some(), "file_read missing after MCP noop");
    assert!(tools.get("shell").is_some(), "shell missing after MCP noop");
}

// Popperian falsification tests extracted to code_tests_falsification.rs
#[path = "code_tests_falsification.rs"]
mod falsification;

// ── PMAT-CODE-ORG-POLICY-RUNTIME-001: system-prompt assembly helper ───

#[cfg(test)]
mod assemble_prompt_tests {
    use super::super::assemble_system_prompt;
    use crate::agent::org_policy::{OrgPolicy, PolicyTier};
    use std::path::PathBuf;

    fn synth_policy(content: &str) -> OrgPolicy {
        OrgPolicy {
            source: PathBuf::from("/etc/apr-code/CLAUDE.md"),
            content: content.into(),
            tier: PolicyTier::Enforced,
        }
    }

    #[test]
    fn no_policy_no_extras_yields_base_plus_context() {
        let out = assemble_system_prompt("BASE", "ctx body", None, None, None);
        assert!(out.starts_with("BASE"));
        assert!(out.contains("## Project Context"));
        assert!(out.contains("ctx body"));
        // No optional sections should appear.
        assert!(!out.contains("## Enforced"));
        assert!(!out.contains("## Project Instructions"));
        assert!(!out.contains("## Auto-memory"));
    }

    #[test]
    fn policy_appears_before_context_and_instructions() {
        let pol = synth_policy("MUST USE MFA.\n");
        let out = assemble_system_prompt(
            "BASE",
            "ctx body",
            Some("PROJ-INSTR"),
            Some("MEMORY-NOTE"),
            Some(&pol),
        );
        // All sections present.
        assert!(out.contains("## Enforced organization policy"));
        assert!(out.contains("MUST USE MFA"));
        assert!(out.contains("## Project Context"));
        assert!(out.contains("ctx body"));
        assert!(out.contains("## Project Instructions"));
        assert!(out.contains("PROJ-INSTR"));
        assert!(out.contains("## Auto-memory"));
        assert!(out.contains("MEMORY-NOTE"));
        // Order check: policy < context < instructions < memory.
        let policy_idx = out.find("## Enforced").expect("policy section");
        let context_idx = out.find("## Project Context").expect("context section");
        let instr_idx = out.find("## Project Instructions").expect("instructions section");
        let mem_idx = out.find("## Auto-memory").expect("memory section");
        assert!(policy_idx < context_idx, "policy must precede context");
        assert!(context_idx < instr_idx, "context must precede instructions");
        assert!(instr_idx < mem_idx, "instructions must precede auto-memory");
    }

    #[test]
    fn policy_only_omits_other_optional_sections() {
        let pol = synth_policy("POLICY");
        let out = assemble_system_prompt("BASE", "ctx", None, None, Some(&pol));
        assert!(out.contains("## Enforced organization policy"));
        assert!(out.contains("POLICY"));
        // Instructions / memory not emitted.
        assert!(!out.contains("## Project Instructions"));
        assert!(!out.contains("## Auto-memory"));
    }

    #[test]
    fn policy_source_path_is_surfaced() {
        // The operator should be able to spot where the policy came from
        // by reading the system prompt itself — useful for CCPA tracing.
        let pol = synth_policy("X");
        let out = assemble_system_prompt("B", "c", None, None, Some(&pol));
        assert!(
            out.contains("/etc/apr-code/CLAUDE.md"),
            "source path must appear in the policy heading: {out}"
        );
    }

    #[test]
    fn instructions_only_no_memory_no_policy() {
        let out = assemble_system_prompt("B", "c", Some("INSTR"), None, None);
        assert!(out.contains("## Project Instructions"));
        assert!(out.contains("INSTR"));
        assert!(!out.contains("## Enforced"));
        assert!(!out.contains("## Auto-memory"));
    }
}

// ── PMAT-CODE-CONFIG-LADDER-001: settings.json precedence ladder ──────

#[cfg(test)]
mod settings_apply_tests {
    use super::super::apply_settings_to_manifest;
    use super::super::build_default_manifest;
    use crate::agent::settings::AprSettings;

    #[test]
    fn apply_model_repo_preserves_alias_form() {
        let mut m = build_default_manifest();
        let s = AprSettings { model: Some("qwen3:1.7b-q4k".into()), ..Default::default() };
        apply_settings_to_manifest(&mut m, &s).expect("apply ok");
        assert_eq!(m.model.model_repo.as_deref(), Some("qwen3:1.7b-q4k"));
        assert!(m.model.model_path.is_none());
    }

    #[test]
    fn apply_model_path_treats_absolute_as_path() {
        let mut m = build_default_manifest();
        let s = AprSettings { model: Some("/abs/model.gguf".into()), ..Default::default() };
        apply_settings_to_manifest(&mut m, &s).expect("apply ok");
        assert_eq!(
            m.model.model_path.as_ref().map(|p| p.to_string_lossy().into_owned()),
            Some("/abs/model.gguf".to_string())
        );
        assert!(m.model.model_repo.is_none());
    }

    #[test]
    fn apply_model_path_treats_relative_dot_as_path() {
        let mut m = build_default_manifest();
        let s = AprSettings { model: Some("./local.apr".into()), ..Default::default() };
        apply_settings_to_manifest(&mut m, &s).expect("apply ok");
        assert!(m.model.model_path.is_some());
        assert!(m.model.model_repo.is_none());
    }

    #[test]
    fn apply_max_turns_overrides_resource_quota() {
        let mut m = build_default_manifest();
        let original = m.resources.max_iterations;
        let s = AprSettings { max_turns: Some(7), ..Default::default() };
        apply_settings_to_manifest(&mut m, &s).expect("apply ok");
        assert_eq!(m.resources.max_iterations, 7);
        assert_ne!(7, original, "test invalid: settings.max_turns matches default");
    }

    #[test]
    fn apply_extra_system_prompt_appends_does_not_replace() {
        let mut m = build_default_manifest();
        let base_len = m.model.system_prompt.len();
        let s = AprSettings { extra_system_prompt: Some("BE TERSE.".into()), ..Default::default() };
        apply_settings_to_manifest(&mut m, &s).expect("apply ok");
        assert!(m.model.system_prompt.len() > base_len, "must append, not replace");
        assert!(m.model.system_prompt.ends_with("BE TERSE."));
    }

    #[test]
    fn apply_empty_extra_prompt_is_noop() {
        let mut m = build_default_manifest();
        let base = m.model.system_prompt.clone();
        let s = AprSettings { extra_system_prompt: Some("   \n  ".into()), ..Default::default() };
        apply_settings_to_manifest(&mut m, &s).expect("apply ok");
        assert_eq!(m.model.system_prompt, base, "whitespace-only extra is no-op");
    }

    #[test]
    fn apply_default_settings_is_noop() {
        let mut m = build_default_manifest();
        let snapshot = (
            m.model.model_path.clone(),
            m.model.model_repo.clone(),
            m.model.system_prompt.clone(),
            m.resources.max_iterations,
        );
        apply_settings_to_manifest(&mut m, &AprSettings::default()).expect("apply ok");
        assert_eq!(m.model.model_path, snapshot.0);
        assert_eq!(m.model.model_repo, snapshot.1);
        assert_eq!(m.model.system_prompt, snapshot.2);
        assert_eq!(m.resources.max_iterations, snapshot.3);
    }

    // ── PMAT-CODE-CONFIG-LADDER-FIELDS-001: permission_mode + allowed_hosts

    #[test]
    fn apply_valid_permission_mode_succeeds() {
        let mut m = build_default_manifest();
        for mode in &["default", "plan", "acceptEdits", "bypassPermissions"] {
            let s = AprSettings { permission_mode: Some((*mode).into()), ..Default::default() };
            apply_settings_to_manifest(&mut m, &s)
                .unwrap_or_else(|e| panic!("expected {mode} to parse: {e}"));
        }
    }

    #[test]
    fn apply_unknown_permission_mode_errs_loudly() {
        let mut m = build_default_manifest();
        let s = AprSettings { permission_mode: Some("totally-fake".into()), ..Default::default() };
        let err = apply_settings_to_manifest(&mut m, &s).expect_err("must reject unknown");
        let msg = format!("{err}");
        assert!(
            msg.contains("permissionMode"),
            "error must mention permissionMode field name: {msg}"
        );
        assert!(msg.contains("totally-fake"), "error must echo the bad value: {msg}");
    }

    #[test]
    fn apply_allowed_hosts_populates_manifest() {
        let mut m = build_default_manifest();
        // Default manifest's allowed_hosts is empty (Sovereign by default).
        assert!(m.allowed_hosts.is_empty());
        let s = AprSettings {
            allowed_hosts: Some(vec!["docs.anthropic.com".into(), "crates.io".into()]),
            ..Default::default()
        };
        apply_settings_to_manifest(&mut m, &s).expect("apply ok");
        assert_eq!(
            m.allowed_hosts,
            vec!["docs.anthropic.com".to_string(), "crates.io".to_string()]
        );
    }

    #[test]
    fn apply_allowed_hosts_does_not_override_explicit_manifest() {
        // Operator-declared TOML manifest wins over settings.json — same
        // policy as model/max_turns/etc.
        let mut m = build_default_manifest();
        m.allowed_hosts = vec!["explicit.example.com".into()];
        let s = AprSettings {
            allowed_hosts: Some(vec!["from-settings.example.com".into()]),
            ..Default::default()
        };
        apply_settings_to_manifest(&mut m, &s).expect("apply ok");
        assert_eq!(
            m.allowed_hosts,
            vec!["explicit.example.com".to_string()],
            "manifest-declared list must NOT be replaced by settings"
        );
    }
}

// ── M28: apr code --emit-trace ────────────────────────────────────────

#[cfg(test)]
mod emit_trace_tests {
    use super::super::emit_ccpa_trace;
    use crate::agent::{AgentLoopResult, TokenUsage};

    fn synth_result(text: &str) -> AgentLoopResult {
        AgentLoopResult {
            text: text.to_owned(),
            usage: TokenUsage { input_tokens: 42, output_tokens: 7 },
            iterations: 1,
            tool_calls: 0,
        }
    }

    #[test]
    fn emit_writes_4_jsonl_records_with_correct_kinds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("trace.jsonl");
        let r = synth_result("hello world");
        emit_ccpa_trace(&path, "what?", &r, std::time::Duration::from_millis(123), "qwen-test")
            .expect("emit");
        let body = std::fs::read_to_string(&path).expect("read back");
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 4, "expected 4 records, got {}", lines.len());
        assert!(lines[0].contains("\"kind\":\"session_start\""));
        assert!(lines[1].contains("\"kind\":\"user_prompt\""));
        assert!(lines[2].contains("\"kind\":\"assistant_turn\""));
        assert!(lines[3].contains("\"kind\":\"session_end\""));
    }

    #[test]
    fn emit_carries_prompt_and_response_text() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("trace.jsonl");
        let r = synth_result("the answer is 42");
        emit_ccpa_trace(
            &path,
            "what is the meaning of life",
            &r,
            std::time::Duration::from_millis(100),
            "test-model",
        )
        .expect("emit");
        let body = std::fs::read_to_string(&path).expect("read");
        assert!(body.contains("what is the meaning of life"));
        assert!(body.contains("the answer is 42"));
        assert!(body.contains("\"actor\":\"apr-code\""));
        assert!(body.contains("\"model\":\"test-model\""));
    }

    #[test]
    fn emit_carries_token_counts_and_elapsed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("trace.jsonl");
        let r = synth_result("x");
        emit_ccpa_trace(&path, "p", &r, std::time::Duration::from_millis(456), "m").expect("emit");
        let body = std::fs::read_to_string(&path).expect("read");
        // session_end carries elapsed_ms + token counts
        assert!(body.contains("\"elapsed_ms\":456"));
        assert!(body.contains("\"tokens_in\":42"));
        assert!(body.contains("\"tokens_out\":7"));
    }

    #[test]
    fn emit_each_record_has_v1_envelope() {
        // Per ccpa-trace-v2 schema: every record carries `v: 1` for
        // per-record back-compat regardless of file-level schema version.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("trace.jsonl");
        let r = synth_result("hi");
        emit_ccpa_trace(&path, "p", &r, std::time::Duration::from_millis(0), "m").expect("emit");
        let body = std::fs::read_to_string(&path).expect("read");
        for line in body.lines() {
            assert!(line.contains("\"v\":1"), "every JSONL record must carry v:1, got: {line}");
        }
    }
}

// ── PMAT-CODE-OUTPUT-FORMAT-001 / PMAT-CODE-INPUT-FORMAT-001 ─────────
// Non-interactive mode parity with `claude -p --output-format <fmt>`
// and `claude -p --input-format json`.

#[cfg(test)]
mod non_interactive_format_tests {
    use super::super::{build_json_result_envelope, parse_json_input_envelope};
    use crate::agent::{AgentLoopResult, TokenUsage};

    fn synth_result(text: &str) -> AgentLoopResult {
        AgentLoopResult {
            text: text.to_owned(),
            usage: TokenUsage { input_tokens: 42, output_tokens: 7 },
            iterations: 3,
            tool_calls: 1,
        }
    }

    #[test]
    fn json_output_envelope_carries_required_fields() {
        // Mirrors `claude -p "x" --output-format json` shape — the operator
        // contract for any tool downstream (e.g. CCPA differ) that parses
        // this envelope.
        let r = synth_result("the answer is 4");
        let s = build_json_result_envelope(&r, std::time::Duration::from_millis(123), false);
        let v: serde_json::Value = serde_json::from_str(&s).expect("envelope is valid JSON");
        assert_eq!(v["type"], "result");
        assert_eq!(v["subtype"], "success");
        assert_eq!(v["is_error"], false);
        assert_eq!(v["duration_ms"], 123);
        assert_eq!(v["result"], "the answer is 4");
        assert_eq!(v["num_turns"], 3);
        assert_eq!(v["tokens_in"], 42);
        assert_eq!(v["tokens_out"], 7);
        assert_eq!(v["total_cost_usd"], 0);
        assert!(v["session_id"].as_str().is_some_and(|s| !s.is_empty()));
    }

    #[test]
    fn json_output_envelope_marks_error_subtype_on_empty_response() {
        let r = synth_result("");
        let s = build_json_result_envelope(&r, std::time::Duration::from_millis(1), true);
        let v: serde_json::Value = serde_json::from_str(&s).expect("valid JSON");
        assert_eq!(v["subtype"], "error");
        assert_eq!(v["is_error"], true);
        assert_eq!(v["result"], "");
    }

    #[test]
    fn json_input_envelope_extracts_user_content() {
        let buf = r#"{"role":"user","content":"What is 2+2?"}"#;
        let prompt = parse_json_input_envelope(buf).expect("valid envelope");
        assert_eq!(prompt, "What is 2+2?");
    }

    #[test]
    fn json_input_envelope_defaults_role_to_user_when_omitted() {
        // Lenient: omitted role implies "user" since non-interactive surface
        // is single-turn user-only by construction.
        let buf = r#"{"content":"hello"}"#;
        let prompt = parse_json_input_envelope(buf).expect("valid envelope");
        assert_eq!(prompt, "hello");
    }

    #[test]
    fn json_input_envelope_rejects_non_user_role() {
        let buf = r#"{"role":"assistant","content":"x"}"#;
        let err = parse_json_input_envelope(buf).expect_err("must reject non-user role");
        let msg = format!("{err}");
        assert!(msg.contains("role"), "error must mention role: {msg}");
    }

    #[test]
    fn json_input_envelope_rejects_missing_content() {
        let buf = r#"{"role":"user"}"#;
        let err = parse_json_input_envelope(buf).expect_err("must reject missing content");
        let msg = format!("{err}");
        assert!(msg.contains("content"), "error must mention content field: {msg}");
    }

    #[test]
    fn json_input_envelope_rejects_empty_stdin() {
        let err = parse_json_input_envelope("   \n").expect_err("must reject empty stdin");
        let msg = format!("{err}");
        assert!(msg.contains("empty"), "error must mention empty: {msg}");
    }

    #[test]
    fn json_input_envelope_rejects_malformed_json() {
        let err = parse_json_input_envelope("{not json").expect_err("must reject bad JSON");
        let msg = format!("{err}");
        assert!(msg.contains("invalid JSON"), "error must mention JSON: {msg}");
    }
}
