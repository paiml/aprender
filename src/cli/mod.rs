//! CLI command logic - extracted for testability
//!
//! This module contains pure functions and testable logic extracted from main.rs.
//! Display functions remain in main.rs while business logic lives here.

#[cfg(feature = "agents")]
pub mod agent;
pub mod bug_hunter;
#[cfg(feature = "agents")]
pub mod code;
pub mod content;
pub mod data;
pub mod deploy;
pub mod experiment;
pub mod falsify;
pub mod hf;
pub mod oracle;
pub mod oracle_classic;
pub mod oracle_indexing;
pub mod parf;
pub mod pipeline_cmds;
pub mod playbook;
pub mod serve;
pub mod stack;
pub mod syntax;
pub mod viz;
pub mod workflow;

use crate::config::BatutaConfig;
use crate::types::{Language, PhaseStatus, WorkflowPhase, WorkflowState};
use std::path::PathBuf;

/// Get the workflow state file path
pub fn get_state_file_path() -> PathBuf {
    PathBuf::from(".batuta-state.json")
}

// Data Size Parsing

/// Try to parse a number with a size suffix (k, m, b)
fn parse_with_suffix(s: &str, suffix: char, multiplier: u64) -> Option<u64> {
    s.strip_suffix(suffix).and_then(|num_str| num_str.parse::<u64>().ok()).map(|n| n * multiplier)
}

/// Parse data size strings like "1M", "100K", "1000"
pub fn parse_data_size_value(s: &str) -> Option<u64> {
    let s = s.to_lowercase();

    parse_with_suffix(&s, 'm', 1_000_000)
        .or_else(|| parse_with_suffix(&s, 'k', 1_000))
        .or_else(|| parse_with_suffix(&s, 'b', 1_000_000_000))
        .or_else(|| s.parse::<u64>().ok())
}

// Transpiler Argument Building

/// Build transpiler command arguments
pub fn build_transpiler_args(
    config: &BatutaConfig,
    incremental: bool,
    cache: bool,
    ruchy: bool,
    modules: &Option<Vec<String>>,
) -> Vec<String> {
    contract_pre_transpile!(config);
    let input_path_str = config.source.path.to_string_lossy().to_string();
    let output_path_str = config.transpilation.output_dir.to_string_lossy().to_string();
    let modules_str = modules.as_ref().map(|m| m.join(",")).unwrap_or_default();

    let mut args =
        vec!["--input".to_string(), input_path_str, "--output".to_string(), output_path_str];

    if incremental || config.transpilation.incremental {
        args.push("--incremental".to_string());
    }

    if cache || config.transpilation.cache {
        args.push("--cache".to_string());
    }

    if ruchy || config.transpilation.use_ruchy {
        args.push("--ruchy".to_string());
    }

    if modules.is_some() {
        args.push("--modules".to_string());
        args.push(modules_str);
    }

    args
}

// Workflow Progress Calculation (planned for TUI dashboard)

/// Calculate workflow progress percentage
pub fn calculate_progress(state: &WorkflowState) -> f64 {
    state.progress_percentage()
}

/// Check if a phase is completed
pub fn is_phase_complete(state: &WorkflowState, phase: WorkflowPhase) -> bool {
    state.is_phase_completed(phase)
}

/// Get next recommended phase based on current state
pub fn get_next_phase(state: &WorkflowState) -> Option<WorkflowPhase> {
    state.current_phase
}

/// Count completed phases
pub fn count_completed_phases(state: &WorkflowState) -> usize {
    state.phases.values().filter(|info| info.status == PhaseStatus::Completed).count()
}

/// Check if any work has started
pub fn has_work_started(state: &WorkflowState) -> bool {
    state.phases.values().any(|info| info.status != PhaseStatus::NotStarted)
}

// Tool Selection Logic

/// Get needed tools for a language
pub fn get_needed_tools_for_language(lang: &Language) -> Vec<&'static str> {
    match lang {
        Language::Python => vec!["depyler"],
        Language::C | Language::Cpp => vec!["decy"],
        Language::Shell => vec!["bashrs"],
        _ => vec![],
    }
}

// TDG Score Grading

/// Grade for TDG score
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TdgGrade {
    APlus,
    A,
    B,
    C,
    D,
}

impl std::fmt::Display for TdgGrade {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TdgGrade::APlus => write!(f, "A+"),
            TdgGrade::A => write!(f, "A"),
            TdgGrade::B => write!(f, "B"),
            TdgGrade::C => write!(f, "C"),
            TdgGrade::D => write!(f, "D"),
        }
    }
}

/// Calculate TDG grade from score
pub fn calculate_tdg_grade(score: f64) -> TdgGrade {
    if score >= 90.0 {
        TdgGrade::APlus
    } else if score >= 80.0 {
        TdgGrade::A
    } else if score >= 70.0 {
        TdgGrade::B
    } else if score >= 60.0 {
        TdgGrade::C
    } else {
        TdgGrade::D
    }
}

// Validation Logic (planned for enhanced validation phase)

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub passed: bool,
    pub syscall_match: Option<bool>,
    pub test_passed: Option<bool>,
    pub benchmark_passed: Option<bool>,
    pub errors: Vec<String>,
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self {
            passed: true,
            syscall_match: None,
            test_passed: None,
            benchmark_passed: None,
            errors: vec![],
        }
    }
}
impl ValidationResult {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fail(&mut self, reason: &str) {
        self.passed = false;
        self.errors.push(reason.to_string());
    }

    pub fn set_syscall_match(&mut self, matched: bool) {
        self.syscall_match = Some(matched);
        if !matched {
            self.passed = false;
        }
    }
}

// Integration Pattern Parsing (planned for cross-tool integration)

/// Parse integration component pair
pub fn parse_integration_components(input: &str) -> Result<(&str, &str), &'static str> {
    let parts: Vec<&str> = input.split(',').map(|s| s.trim()).collect();
    if parts.len() != 2 {
        return Err("Expected two components separated by comma");
    }
    Ok((parts[0], parts[1]))
}

// Tests

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn test_CLI_001_state_file_path() {
        let path = get_state_file_path();
        assert_eq!(path, PathBuf::from(".batuta-state.json"));
    }

    #[test]
    fn test_CLI_002_parse_data_size_millions() {
        assert_eq!(parse_data_size_value("1m"), Some(1_000_000));
        assert_eq!(parse_data_size_value("10M"), Some(10_000_000));
        assert_eq!(parse_data_size_value("100m"), Some(100_000_000));
    }

    #[test]
    fn test_CLI_002_parse_data_size_thousands() {
        assert_eq!(parse_data_size_value("1k"), Some(1_000));
        assert_eq!(parse_data_size_value("100K"), Some(100_000));
        assert_eq!(parse_data_size_value("500k"), Some(500_000));
    }

    #[test]
    fn test_CLI_002_parse_data_size_billions() {
        assert_eq!(parse_data_size_value("1b"), Some(1_000_000_000));
        assert_eq!(parse_data_size_value("2B"), Some(2_000_000_000));
    }

    #[test]
    fn test_CLI_002_parse_data_size_raw_number() {
        assert_eq!(parse_data_size_value("1000"), Some(1000));
        assert_eq!(parse_data_size_value("999999"), Some(999999));
    }

    #[test]
    fn test_CLI_002_parse_data_size_invalid() {
        assert_eq!(parse_data_size_value("invalid"), None);
        assert_eq!(parse_data_size_value("abc123"), None);
        assert_eq!(parse_data_size_value(""), None);
    }

    #[test]
    fn test_CLI_003_build_transpiler_args_basic() {
        let mut config = BatutaConfig::default();
        // Override defaults for testing
        config.transpilation.incremental = false;
        config.transpilation.cache = false;
        config.transpilation.use_ruchy = false;
        let args = build_transpiler_args(&config, false, false, false, &None);

        assert!(args.contains(&"--input".to_string()));
        assert!(args.contains(&"--output".to_string()));
        assert!(!args.contains(&"--incremental".to_string()));
        assert!(!args.contains(&"--cache".to_string()));
        assert!(!args.contains(&"--ruchy".to_string()));
    }

    #[test]
    fn test_CLI_003_build_transpiler_args_with_flags() {
        let mut config = BatutaConfig::default();
        config.transpilation.incremental = false;
        config.transpilation.cache = false;
        let args = build_transpiler_args(&config, true, true, true, &None);

        assert!(args.contains(&"--incremental".to_string()));
        assert!(args.contains(&"--cache".to_string()));
        assert!(args.contains(&"--ruchy".to_string()));
    }

    #[test]
    fn test_CLI_003_build_transpiler_args_config_defaults() {
        // Test that config defaults are honored
        let config = BatutaConfig::default();
        let args = build_transpiler_args(&config, false, false, false, &None);

        // Default config has incremental=true, cache=true
        assert!(args.contains(&"--incremental".to_string()));
        assert!(args.contains(&"--cache".to_string()));
    }

    #[test]
    fn test_CLI_003_build_transpiler_args_with_modules() {
        let config = BatutaConfig::default();
        let modules = Some(vec!["mod1".to_string(), "mod2".to_string()]);
        let args = build_transpiler_args(&config, false, false, false, &modules);

        assert!(args.contains(&"--modules".to_string()));
        assert!(args.contains(&"mod1,mod2".to_string()));
    }

    #[test]
    fn test_CLI_004_workflow_state_new() {
        let state = WorkflowState::new();
        assert!(!has_work_started(&state));
        assert_eq!(count_completed_phases(&state), 0);
    }

    #[test]
    fn test_CLI_004_workflow_progress() {
        let state = WorkflowState::new();
        let progress = calculate_progress(&state);
        assert_eq!(progress, 0.0);
    }

    #[test]
    fn test_CLI_004_get_next_phase_new_state() {
        let state = WorkflowState::new();
        // New state has no current phase
        assert!(
            get_next_phase(&state).is_none()
                || get_next_phase(&state) == Some(WorkflowPhase::Analysis)
        );
    }

    #[test]
    fn test_CLI_005_tools_for_python() {
        let tools = get_needed_tools_for_language(&Language::Python);
        assert_eq!(tools, vec!["depyler"]);
    }

    #[test]
    fn test_CLI_005_tools_for_c() {
        let tools = get_needed_tools_for_language(&Language::C);
        assert_eq!(tools, vec!["decy"]);
    }

    #[test]
    fn test_CLI_005_tools_for_cpp() {
        let tools = get_needed_tools_for_language(&Language::Cpp);
        assert_eq!(tools, vec!["decy"]);
    }

    #[test]
    fn test_CLI_005_tools_for_shell() {
        let tools = get_needed_tools_for_language(&Language::Shell);
        assert_eq!(tools, vec!["bashrs"]);
    }

    #[test]
    fn test_CLI_005_tools_for_rust() {
        let tools = get_needed_tools_for_language(&Language::Rust);
        assert!(tools.is_empty());
    }

    #[test]
    fn test_CLI_006_tdg_grade_a_plus() {
        assert_eq!(calculate_tdg_grade(90.0), TdgGrade::APlus);
        assert_eq!(calculate_tdg_grade(95.0), TdgGrade::APlus);
        assert_eq!(calculate_tdg_grade(100.0), TdgGrade::APlus);
    }

    #[test]
    fn test_CLI_006_tdg_grade_a() {
        assert_eq!(calculate_tdg_grade(80.0), TdgGrade::A);
        assert_eq!(calculate_tdg_grade(85.0), TdgGrade::A);
        assert_eq!(calculate_tdg_grade(89.9), TdgGrade::A);
    }

    #[test]
    fn test_CLI_006_tdg_grade_b() {
        assert_eq!(calculate_tdg_grade(70.0), TdgGrade::B);
        assert_eq!(calculate_tdg_grade(75.0), TdgGrade::B);
        assert_eq!(calculate_tdg_grade(79.9), TdgGrade::B);
    }

    #[test]
    fn test_CLI_006_tdg_grade_c() {
        assert_eq!(calculate_tdg_grade(60.0), TdgGrade::C);
        assert_eq!(calculate_tdg_grade(65.0), TdgGrade::C);
        assert_eq!(calculate_tdg_grade(69.9), TdgGrade::C);
    }

    #[test]
    fn test_CLI_006_tdg_grade_d() {
        assert_eq!(calculate_tdg_grade(59.9), TdgGrade::D);
        assert_eq!(calculate_tdg_grade(50.0), TdgGrade::D);
        assert_eq!(calculate_tdg_grade(0.0), TdgGrade::D);
    }

    #[test]
    fn test_CLI_006_tdg_grade_display() {
        assert_eq!(format!("{}", TdgGrade::APlus), "A+");
        assert_eq!(format!("{}", TdgGrade::A), "A");
        assert_eq!(format!("{}", TdgGrade::B), "B");
        assert_eq!(format!("{}", TdgGrade::C), "C");
        assert_eq!(format!("{}", TdgGrade::D), "D");
    }

    #[test]
    fn test_CLI_007_validation_result_new() {
        let result = ValidationResult::new();
        assert!(result.passed);
        assert!(result.syscall_match.is_none());
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_CLI_007_validation_result_fail() {
        let mut result = ValidationResult::new();
        result.fail("Test failure");

        assert!(!result.passed);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0], "Test failure");
    }

    #[test]
    fn test_CLI_007_validation_result_syscall_match() {
        let mut result = ValidationResult::new();
        result.set_syscall_match(true);

        assert!(result.passed);
        assert_eq!(result.syscall_match, Some(true));
    }

    #[test]
    fn test_CLI_007_validation_result_syscall_mismatch() {
        let mut result = ValidationResult::new();
        result.set_syscall_match(false);

        assert!(!result.passed);
        assert_eq!(result.syscall_match, Some(false));
    }

    #[test]
    fn test_CLI_007_validation_result_multiple_failures() {
        let mut result = ValidationResult::new();
        result.fail("Error 1");
        result.fail("Error 2");
        result.set_syscall_match(false);

        assert!(!result.passed);
        assert_eq!(result.errors.len(), 2);
    }

    #[test]
    fn test_CLI_008_parse_integration_valid() {
        let result = parse_integration_components("aprender,realizar");
        assert!(result.is_ok());
        let (from, to) = result.expect("operation failed");
        assert_eq!(from, "aprender");
        assert_eq!(to, "realizar");
    }

    #[test]
    fn test_CLI_008_parse_integration_with_spaces() {
        let result = parse_integration_components(" trueno , aprender ");
        assert!(result.is_ok());
        let (from, to) = result.expect("operation failed");
        assert_eq!(from, "trueno");
        assert_eq!(to, "aprender");
    }

    #[test]
    fn test_CLI_008_parse_integration_invalid_single() {
        let result = parse_integration_components("aprender");
        assert!(result.is_err());
    }

    #[test]
    fn test_CLI_008_parse_integration_invalid_triple() {
        let result = parse_integration_components("a,b,c");
        assert!(result.is_err());
    }

    #[test]
    fn test_CLI_008_parse_integration_empty() {
        let result = parse_integration_components("");
        assert!(result.is_err());
    }
}
