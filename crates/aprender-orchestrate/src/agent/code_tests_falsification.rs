//! Popperian falsification of apr-code spec claims (PMAT-185).
//!
//! Each test attempts to BREAK a specific spec claim. If the test
//! passes, the claim survives. If it fails, the spec is wrong.
//!
//! These are NOT normal tests — they test the specification, not the code.
//! A passing test means "we tried to disprove this claim and couldn't."

use super::*;

/// FALSIFY-SPEC-001: Attempt to create a non-Sovereign manifest.
/// The build_default_manifest() MUST always return Sovereign.
/// If we could modify it through any parameter, sovereignty is broken.
#[test]
fn falsify_spec_001_sovereignty_cannot_be_overridden() {
    let m = build_default_manifest();
    assert_eq!(
        m.privacy,
        PrivacyTier::Sovereign,
        "FALSIFY-SPEC-001: sovereignty hardcoded in build_default_manifest"
    );
    assert_eq!(m.name, "apr-code", "FALSIFY-SPEC-001: manifest name is 'apr-code'");
}

/// FALSIFY-SPEC-002: Shell tool uses wildcard mode (not restrictive allowlist).
/// Spec says coding tasks need full shell access.
#[test]
fn falsify_spec_002_shell_wildcard_mode() {
    let m = build_default_manifest();
    let has_wildcard_shell = m
        .capabilities
        .iter()
        .any(|c| matches!(c, Capability::Shell { allowed_commands } if allowed_commands == &["*"]));
    assert!(has_wildcard_shell, "FALSIFY-SPEC-002: Shell must use wildcard mode for coding tasks");
}

/// FALSIFY-SPEC-003: File tools have unrestricted path access.
/// apr code needs to read/write any project file.
#[test]
fn falsify_spec_003_file_tools_unrestricted() {
    let m = build_default_manifest();
    let has_wildcard_read = m
        .capabilities
        .iter()
        .any(|c| matches!(c, Capability::FileRead { allowed_paths } if allowed_paths == &["*"]));
    let has_wildcard_write = m
        .capabilities
        .iter()
        .any(|c| matches!(c, Capability::FileWrite { allowed_paths } if allowed_paths == &["*"]));
    assert!(has_wildcard_read, "FALSIFY-SPEC-003: FileRead must have wildcard path access");
    assert!(has_wildcard_write, "FALSIFY-SPEC-003: FileWrite must have wildcard path access");
}

/// FALSIFY-SPEC-004: Max iterations is 50 for interactive mode.
#[test]
fn falsify_spec_004_max_iterations_50() {
    let m = build_default_manifest();
    assert_eq!(m.resources.max_iterations, 50, "FALSIFY-SPEC-004: max_iterations must be 50");
}

/// FALSIFY-SPEC-005: Temperature is 0.0 for deterministic coding.
#[test]
fn falsify_spec_005_temperature_zero() {
    let m = build_default_manifest();
    assert!(
        (m.model.temperature - 0.0).abs() < f32::EPSILON,
        "FALSIFY-SPEC-005: temperature must be 0.0, got {}",
        m.model.temperature
    );
}

/// FALSIFY-SPEC-006: Model discovery search dirs match spec §5.1.
#[test]
fn falsify_spec_006_search_dirs_complete() {
    let dirs = crate::agent::manifest::ModelConfig::model_search_dirs();
    let dir_strs: Vec<String> = dirs.iter().map(|d| d.display().to_string()).collect();
    let combined = dir_strs.join("|");
    assert!(combined.contains(".apr/models"), "FALSIFY-SPEC-006: must search ~/.apr/models/");
    assert!(combined.contains("models"), "FALSIFY-SPEC-006: must search ./models/");
}

/// FALSIFY-SPEC-007: Project instructions found in project root.
#[test]
fn falsify_spec_007_project_instructions_load() {
    let instructions = load_project_instructions(4096);
    assert!(instructions.is_some(), "FALSIFY-SPEC-007: must find project instructions");
}

/// FALSIFY-SPEC-008: System prompt contains all 9 tool names.
#[test]
fn falsify_spec_008_all_tools_in_prompt() {
    let required_tools = [
        "file_read",
        "file_write",
        "file_edit",
        "glob",
        "grep",
        "shell",
        "memory",
        "pmat_query",
        "rag",
    ];
    for tool in &required_tools {
        assert!(CODE_SYSTEM_PROMPT.contains(tool), "FALSIFY-SPEC-008: missing tool '{tool}'");
    }
}

/// FALSIFY-SPEC-009: Exit codes are distinct (injective mapping).
#[test]
fn falsify_spec_009_exit_codes_distinct() {
    let codes = [
        exit_code::SUCCESS,
        exit_code::AGENT_ERROR,
        exit_code::BUDGET_EXHAUSTED,
        exit_code::MAX_TURNS,
        exit_code::SANDBOX_VIOLATION,
        exit_code::NO_MODEL,
    ];
    let mut seen = std::collections::HashSet::new();
    for code in &codes {
        assert!(seen.insert(code), "FALSIFY-SPEC-009: duplicate exit code {code}");
    }
    assert_eq!(seen.len(), 6, "FALSIFY-SPEC-009: exactly 6 distinct exit codes");
}

/// FALSIFY-SPEC-010: Max cost is $0.00 (local inference is free).
#[test]
fn falsify_spec_010_zero_cost() {
    let m = build_default_manifest();
    assert!(
        (m.resources.max_cost_usd - 0.0).abs() < f64::EPSILON,
        "FALSIFY-SPEC-010: max_cost must be 0.0 (free local inference)"
    );
}

/// A minimal GGUF v3 header: `general.architecture` plus, when given,
/// `<arch>.context_length`. No tensors, so it is the whole file.
fn gguf_header(arch: &str, context_length: Option<u32>) -> Vec<u8> {
    fn key(out: &mut Vec<u8>, k: &str) {
        out.extend((k.len() as u64).to_le_bytes());
        out.extend(k.as_bytes());
    }
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes()); // tensors
    out.extend((1 + u64::from(context_length.is_some())).to_le_bytes());
    key(&mut out, "general.architecture");
    out.extend(8u32.to_le_bytes()); // string
    key(&mut out, arch);
    if let Some(n) = context_length {
        key(&mut out, &format!("{arch}.context_length"));
        out.extend(4u32.to_le_bytes()); // u32
        out.extend(n.to_le_bytes());
    }
    out
}

/// FALSIFY-4599-005: `apr code` sizes its window from the model, not a hard-coded 32K. An
/// explicit manifest value still wins; with neither, 32K.
#[test]
fn falsify_4599_005_code_window_is_the_models_context_length() {
    let dir = tempfile::tempdir().expect("tempdir");
    let qwen35 = dir.path().join("Qwen3.5-0.8B-Q4_K_M.gguf");
    std::fs::write(&qwen35, gguf_header("qwen35", Some(262_144))).expect("write");
    let no_key = dir.path().join("no-context.gguf");
    std::fs::write(&no_key, gguf_header("qwen35", None)).expect("write");
    let zero = dir.path().join("zero-context.gguf");
    std::fs::write(&zero, gguf_header("qwen35", Some(0))).expect("write");
    let not_gguf = dir.path().join("model.apr");
    std::fs::write(&not_gguf, b"APR\0not a gguf").expect("write");

    assert_eq!(build_default_manifest().model.context_window, None, "no hard-coded window");
    #[cfg(feature = "inference")]
    assert_eq!(code_context_window(None, &qwen35), 262_144, "the model's own context length");
    assert_eq!(code_context_window(Some(8192), &qwen35), 8192, "an explicit manifest wins");
    // The drivers launch with `code_driver_window`: the default manifest resolves to the model's
    // window, and a settings/manifest window still wins.
    let mut m = build_default_manifest();
    #[cfg(feature = "inference")]
    assert_eq!(code_driver_window(&m, &qwen35), 262_144, "the drivers' window is the model's");
    m.model.context_window = Some(8192);
    assert_eq!(code_driver_window(&m, &qwen35), 8192);
    assert_eq!(code_context_window(None, &no_key), CODE_DEFAULT_CONTEXT_WINDOW);
    assert_eq!(code_context_window(None, &zero), CODE_DEFAULT_CONTEXT_WINDOW, "0 is no window");
    assert_eq!(code_context_window(None, &not_gguf), CODE_DEFAULT_CONTEXT_WINDOW);
    assert_eq!(code_context_window(None, &dir.path().join("absent.gguf")), 32_768);
}
