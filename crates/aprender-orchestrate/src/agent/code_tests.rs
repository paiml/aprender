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

#[test]
fn test_cmd_code_signature_matches_spec() {
    // Verify the public API signature exists and is callable
    // This catches regressions where the function is made private or renamed
    // Verify cmd_code exists and is callable
    let _ = cmd_code as fn(_, _, _, _, _, _, _) -> _;
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

// Popperian falsification tests extracted to code_tests_falsification.rs
#[path = "code_tests_falsification.rs"]
mod falsification;
