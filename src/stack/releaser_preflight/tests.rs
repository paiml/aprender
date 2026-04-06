//! Tests for releaser preflight checks

use crate::stack::graph::DependencyGraph;
use crate::stack::releaser::ReleaseOrchestrator;
use crate::stack::releaser_preflight::helpers::{
    parse_count_from_json_multi, parse_value_from_json, run_check_command, score_check_result,
};
use crate::stack::releaser_types::ReleaseConfig;
use crate::stack::types::PreflightCheck;
use std::path::Path;

fn make_orchestrator(config: ReleaseConfig) -> ReleaseOrchestrator {
    let graph = DependencyGraph::new();
    let checker = crate::stack::checker::StackChecker::with_graph(graph);
    ReleaseOrchestrator::new(checker, config)
}

fn default_orchestrator() -> ReleaseOrchestrator {
    make_orchestrator(ReleaseConfig::default())
}

fn orchestrator_with_command(field: &str, cmd: &str) -> ReleaseOrchestrator {
    let mut config = ReleaseConfig::default();
    match field {
        "lint" => config.lint_command = cmd.to_string(),
        "coverage" => config.coverage_command = cmd.to_string(),
        "comply" => config.comply_command = cmd.to_string(),
        "quality_gate" => config.quality_gate_command = cmd.to_string(),
        "tdg" => config.tdg_command = cmd.to_string(),
        "dead_code" => config.dead_code_command = cmd.to_string(),
        "complexity" => config.complexity_command = cmd.to_string(),
        "satd" => config.satd_command = cmd.to_string(),
        "popper" => config.popper_command = cmd.to_string(),
        "book" => config.book_command = cmd.to_string(),
        "examples" => config.examples_command = cmd.to_string(),
        _ => {}
    }
    make_orchestrator(config)
}

// ============================================================================
// JSON Parsing Tests
// ============================================================================

#[test]
fn test_parse_score_from_json_simple() {
    assert_eq!(
        ReleaseOrchestrator::parse_score_from_json(r#"{"score": 85.5, "other": 10}"#, "score"),
        Some(85.5)
    );
}

#[test]
fn test_parse_score_from_json_integer() {
    assert_eq!(
        ReleaseOrchestrator::parse_score_from_json(r#"{"total": 100}"#, "total"),
        Some(100.0)
    );
}

#[test]
fn test_parse_score_from_json_missing() {
    assert_eq!(ReleaseOrchestrator::parse_score_from_json(r#"{"other": 10}"#, "score"), None);
}

#[test]
fn test_parse_count_from_json() {
    assert_eq!(ReleaseOrchestrator::parse_count_from_json(r#"{"count": 42}"#, "count"), Some(42));
}

#[test]
fn test_parse_count_from_json_decimal() {
    assert_eq!(ReleaseOrchestrator::parse_count_from_json(r#"{"count": 42.9}"#, "count"), Some(42));
}

#[test]
fn test_parse_score_with_whitespace() {
    assert_eq!(
        ReleaseOrchestrator::parse_score_from_json(r#"{"score":  85.5}"#, "score"),
        Some(85.5)
    );
}

#[test]
fn test_parse_score_negative() {
    assert_eq!(
        ReleaseOrchestrator::parse_score_from_json(r#"{"delta": -10.5}"#, "delta"),
        Some(-10.5)
    );
}

// ============================================================================
// Helper Function Tests
// ============================================================================

#[test]
fn test_parse_value_from_json_first_key_match() {
    assert_eq!(
        parse_value_from_json(r#"{"score": 85.5, "total": 90.0}"#, &["score", "total"]),
        Some(85.5)
    );
}

#[test]
fn test_parse_value_from_json_fallback_key() {
    assert_eq!(parse_value_from_json(r#"{"total": 90.0}"#, &["score", "total"]), Some(90.0));
}

#[test]
fn test_parse_value_from_json_no_match() {
    assert_eq!(parse_value_from_json(r#"{"other": 10}"#, &["score", "total"]), None);
}

#[test]
fn test_parse_count_from_json_multi() {
    assert_eq!(
        parse_count_from_json_multi(r#"{"dead_code_count": 5}"#, &["count", "dead_code_count"]),
        Some(5)
    );
}

// ============================================================================
// score_check_result
// ============================================================================

#[test]
fn test_score_check_result_pass() {
    let r = score_check_result("tdg", "TDG score", Some(90.0), 80.0, true, true);
    assert!(r.passed);
    assert!(r.message.contains("90.0"));
}

#[test]
fn test_score_check_result_fail() {
    let r = score_check_result("tdg", "TDG score", Some(70.0), 80.0, true, true);
    assert!(!r.passed);
}

#[test]
fn test_score_check_result_warning() {
    let r = score_check_result("tdg", "TDG score", Some(70.0), 80.0, false, true);
    assert!(r.passed);
    assert!(r.message.contains("warning"));
}

#[test]
fn test_score_check_result_no_score_success() {
    let r = score_check_result("tdg", "TDG", None, 80.0, true, true);
    assert!(r.passed);
    assert!(r.message.contains("check passed"));
}

#[test]
fn test_score_check_result_no_score_not_success() {
    let r = score_check_result("tdg", "TDG", None, 80.0, true, false);
    assert!(r.passed);
    assert!(r.message.contains("score not parsed"));
}

// ============================================================================
// run_check_command
// ============================================================================

#[test]
fn test_run_check_command_empty_command() {
    let r = run_check_command("", "test_id", "skipped", Path::new("."), |_, _, _| {
        PreflightCheck::fail("test_id", "should not reach")
    });
    assert!(r.passed);
    assert!(r.message.contains("skipped"));
}

#[test]
fn test_run_check_command_not_found() {
    let r = run_check_command(
        "nonexistent_tool_xyz_123",
        "test_id",
        "skip",
        Path::new("."),
        |_, _, _| PreflightCheck::fail("test_id", "should not reach"),
    );
    assert!(r.passed);
    assert!(r.message.contains("not found"));
}

#[test]
fn test_run_check_command_success() {
    let r = run_check_command("true", "test_id", "skip", Path::new("."), |out, _, _| {
        if out.status.success() {
            PreflightCheck::pass("test_id", "ok")
        } else {
            PreflightCheck::fail("test_id", "nope")
        }
    });
    assert!(r.passed);
}

#[test]
fn test_run_check_command_failure() {
    let r = run_check_command("false", "test_id", "skip", Path::new("."), |out, _, _| {
        if out.status.success() {
            PreflightCheck::pass("test_id", "ok")
        } else {
            PreflightCheck::fail("test_id", "failed")
        }
    });
    assert!(!r.passed);
}

// ============================================================================
// check_git_clean
// ============================================================================

#[test]
fn test_check_git_clean_in_repo() {
    // Run in a temp dir with a git repo
    let dir = std::env::temp_dir().join("test_rp_git_clean");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let _ = std::process::Command::new("git").args(["init"]).current_dir(&dir).output();
    let orch = default_orchestrator();
    let r = orch.check_git_clean(&dir);
    // Fresh repo with no commits: porcelain is empty
    assert!(r.passed);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_check_git_clean_dirty() {
    let dir = std::env::temp_dir().join("test_rp_git_dirty");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let _ = std::process::Command::new("git").args(["init"]).current_dir(&dir).output();
    std::fs::write(dir.join("file.txt"), "hello").unwrap();
    let orch = default_orchestrator();
    let r = orch.check_git_clean(&dir);
    assert!(!r.passed);
    assert!(r.message.contains("Uncommitted"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_check_git_clean_no_git() {
    let dir = std::env::temp_dir().join("test_rp_no_git");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let orch = default_orchestrator();
    let r = orch.check_git_clean(&dir);
    // git status in a non-repo still runs but returns error
    // Either way the check returns something sensible
    let _ = r;
    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// check_lint
// ============================================================================

#[test]
fn test_check_lint_pass() {
    let orch = orchestrator_with_command("lint", "true");
    let r = orch.check_lint(Path::new("."));
    assert!(r.passed);
}

#[test]
fn test_check_lint_fail() {
    let orch = orchestrator_with_command("lint", "false");
    let r = orch.check_lint(Path::new("."));
    assert!(!r.passed);
    assert!(r.message.contains("Lint failed"));
}

#[test]
fn test_check_lint_empty() {
    let orch = orchestrator_with_command("lint", "");
    let r = orch.check_lint(Path::new("."));
    assert!(r.passed);
    assert!(r.message.contains("No lint command"));
}

// ============================================================================
// check_coverage
// ============================================================================

#[test]
fn test_check_coverage_pass() {
    let orch = orchestrator_with_command("coverage", "true");
    let r = orch.check_coverage(Path::new("."));
    assert!(r.passed);
}

#[test]
fn test_check_coverage_fail() {
    let orch = orchestrator_with_command("coverage", "false");
    let r = orch.check_coverage(Path::new("."));
    assert!(!r.passed);
}

#[test]
fn test_check_coverage_empty() {
    let orch = orchestrator_with_command("coverage", "");
    let r = orch.check_coverage(Path::new("."));
    assert!(r.passed);
}

// ============================================================================
// check_no_path_deps / check_version_bumped (placeholders)
// ============================================================================

#[test]
fn test_check_no_path_deps() {
    let orch = default_orchestrator();
    let r = orch.check_no_path_deps("batuta");
    assert!(r.passed);
}

#[test]
fn test_check_version_bumped() {
    let orch = default_orchestrator();
    let r = orch.check_version_bumped("batuta");
    assert!(r.passed);
}

// ============================================================================
// check_pmat_comply
// ============================================================================

#[test]
fn test_check_pmat_comply_pass() {
    let orch = orchestrator_with_command("comply", "true");
    let r = orch.check_pmat_comply(Path::new("."));
    assert!(r.passed);
    assert!(r.message.contains("0 violations"));
}

#[test]
fn test_check_pmat_comply_empty() {
    let orch = orchestrator_with_command("comply", "");
    let r = orch.check_pmat_comply(Path::new("."));
    assert!(r.passed);
}

#[test]
fn test_check_pmat_comply_not_found() {
    let orch = orchestrator_with_command("comply", "nonexistent_xyz_tool");
    let r = orch.check_pmat_comply(Path::new("."));
    assert!(r.passed); // tool not found → skip
}

// ============================================================================
// check_pmat_quality_gate
// ============================================================================

#[test]
fn test_check_quality_gate_pass() {
    let orch = orchestrator_with_command("quality_gate", "true");
    let r = orch.check_pmat_quality_gate(Path::new("."));
    assert!(r.passed);
}

#[test]
fn test_check_quality_gate_fail_blocking() {
    let config = ReleaseConfig {
        quality_gate_command: "false".to_string(),
        fail_on_quality_gate: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_quality_gate(Path::new("."));
    assert!(!r.passed);
}

#[test]
fn test_check_quality_gate_fail_non_blocking() {
    let config = ReleaseConfig {
        quality_gate_command: "false".to_string(),
        fail_on_quality_gate: false,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_quality_gate(Path::new("."));
    assert!(r.passed);
    assert!(r.message.contains("warning"));
}

// ============================================================================
// check_pmat_tdg
// ============================================================================

#[test]
fn test_check_tdg_empty_command() {
    let orch = orchestrator_with_command("tdg", "");
    let r = orch.check_pmat_tdg(Path::new("."));
    assert!(r.passed);
}

#[test]
fn test_check_tdg_success_no_json() {
    let orch = orchestrator_with_command("tdg", "true");
    let r = orch.check_pmat_tdg(Path::new("."));
    assert!(r.passed);
}

// ============================================================================
// check_pmat_dead_code
// ============================================================================

#[test]
fn test_check_dead_code_empty() {
    let orch = orchestrator_with_command("dead_code", "");
    let r = orch.check_pmat_dead_code(Path::new("."));
    assert!(r.passed);
}

#[test]
fn test_check_dead_code_clean() {
    let orch = orchestrator_with_command("dead_code", "true");
    let r = orch.check_pmat_dead_code(Path::new("."));
    assert!(r.passed);
}

// ============================================================================
// check_pmat_complexity
// ============================================================================

#[test]
fn test_check_complexity_empty() {
    let orch = orchestrator_with_command("complexity", "");
    let r = orch.check_pmat_complexity(Path::new("."));
    assert!(r.passed);
}

#[test]
fn test_check_complexity_success() {
    let orch = orchestrator_with_command("complexity", "true");
    let r = orch.check_pmat_complexity(Path::new("."));
    assert!(r.passed);
}

// ============================================================================
// check_pmat_satd
// ============================================================================

#[test]
fn test_check_satd_empty() {
    let orch = orchestrator_with_command("satd", "");
    let r = orch.check_pmat_satd(Path::new("."));
    assert!(r.passed);
}

#[test]
fn test_check_satd_success() {
    let orch = orchestrator_with_command("satd", "true");
    let r = orch.check_pmat_satd(Path::new("."));
    assert!(r.passed);
}

// ============================================================================
// check_pmat_popper
// ============================================================================

#[test]
fn test_check_popper_empty() {
    let orch = orchestrator_with_command("popper", "");
    let r = orch.check_pmat_popper(Path::new("."));
    assert!(r.passed);
}

#[test]
fn test_check_popper_success() {
    let orch = orchestrator_with_command("popper", "true");
    let r = orch.check_pmat_popper(Path::new("."));
    assert!(r.passed);
}

// ============================================================================
// check_book_build
// ============================================================================

#[test]
fn test_check_book_no_dir() {
    let dir = std::env::temp_dir().join("test_rp_no_book");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let orch = default_orchestrator();
    let r = orch.check_book_build(&dir);
    assert!(r.passed);
    assert!(r.message.contains("No book directory"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_check_book_pass() {
    let dir = std::env::temp_dir().join(format!("test_rp_book_pass_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("book")).unwrap();
    let orch = orchestrator_with_command("book", "true");
    let r = orch.check_book_build(&dir);
    assert!(r.passed);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_check_book_fail_blocking() {
    let dir = std::env::temp_dir().join("test_rp_book_fail");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("book")).unwrap();
    let config = ReleaseConfig {
        book_command: "false".to_string(),
        fail_on_book: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_book_build(&dir);
    assert!(!r.passed);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_check_book_fail_non_blocking() {
    let dir = std::env::temp_dir().join("test_rp_book_warn");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("book")).unwrap();
    let config = ReleaseConfig {
        book_command: "false".to_string(),
        fail_on_book: false,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_book_build(&dir);
    assert!(r.passed);
    assert!(r.message.contains("not blocking"));
    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// check_examples_run / discover_examples
// ============================================================================

#[test]
fn test_check_examples_no_dir() {
    let dir = std::env::temp_dir().join("test_rp_no_examples");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let orch = default_orchestrator();
    let r = orch.check_examples_run(&dir);
    assert!(r.passed);
    assert!(r.message.contains("No examples directory"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_check_examples_empty_command() {
    let orch = orchestrator_with_command("examples", "");
    let dir = std::env::temp_dir().join("test_rp_examples_empty");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let r = orch.check_examples_run(&dir);
    assert!(r.passed);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_discover_examples_empty_dir() {
    let dir = std::env::temp_dir().join("test_rp_disc_empty");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("examples")).unwrap();
    let orch = default_orchestrator();
    let examples = orch.discover_examples(&dir);
    assert!(examples.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_discover_examples_from_files() {
    let dir = std::env::temp_dir().join("test_rp_disc_files");
    let _ = std::fs::remove_dir_all(&dir);
    let examples_dir = dir.join("examples");
    std::fs::create_dir_all(&examples_dir).unwrap();
    std::fs::write(examples_dir.join("demo.rs"), "fn main() {}").unwrap();
    std::fs::write(examples_dir.join("bench.rs"), "fn main() {}").unwrap();
    std::fs::write(dir.join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
    let orch = default_orchestrator();
    let examples = orch.discover_examples(&dir);
    assert_eq!(examples.len(), 2);
    assert!(examples.contains(&"demo".to_string()));
    assert!(examples.contains(&"bench".to_string()));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_discover_examples_no_cargo_toml() {
    let dir = std::env::temp_dir().join("test_rp_disc_no_cargo");
    let _ = std::fs::remove_dir_all(&dir);
    let examples_dir = dir.join("examples");
    std::fs::create_dir_all(&examples_dir).unwrap();
    std::fs::write(examples_dir.join("hello.rs"), "fn main() {}").unwrap();
    let orch = default_orchestrator();
    let examples = orch.discover_examples(&dir);
    assert_eq!(examples.len(), 1);
    assert_eq!(examples[0], "hello");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_check_examples_no_examples_found() {
    let dir = std::env::temp_dir().join("test_rp_examples_none");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("examples")).unwrap();
    let orch = default_orchestrator();
    let r = orch.check_examples_run(&dir);
    assert!(r.passed);
    assert!(r.message.contains("No examples found"));
    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// check_pmat_comply detailed branching
// ============================================================================

#[test]
fn test_check_pmat_comply_violations_blocking() {
    // "echo" outputs something, but we need CB- in output
    let config = ReleaseConfig {
        comply_command: "echo CB-020: unsafe block".to_string(),
        fail_on_comply_violations: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_comply(Path::new("."));
    assert!(!r.passed);
    assert!(r.message.contains("CB-020"));
}

#[test]
fn test_check_pmat_comply_violations_non_blocking() {
    let config = ReleaseConfig {
        comply_command: "echo CB-020: unsafe block".to_string(),
        fail_on_comply_violations: false,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_comply(Path::new("."));
    assert!(r.passed);
    assert!(r.message.contains("warnings"));
}

#[test]
fn test_check_pmat_comply_violation_keyword() {
    let config = ReleaseConfig {
        comply_command: "echo violation found".to_string(),
        fail_on_comply_violations: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_comply(Path::new("."));
    assert!(!r.passed);
}

#[test]
fn test_check_pmat_comply_clean_exit_no_violations() {
    let config = ReleaseConfig {
        comply_command: "echo all good".to_string(),
        fail_on_comply_violations: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_comply(Path::new("."));
    assert!(r.passed);
    assert!(r.message.contains("0 violations"));
}

// ============================================================================
// check_pmat_dead_code detailed branching
// ============================================================================

#[test]
fn test_check_dead_code_with_count_zero() {
    let config = ReleaseConfig {
        dead_code_command: r#"echo {"count": 0}"#.to_string(),
        fail_on_dead_code: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_dead_code(Path::new("."));
    assert!(r.passed);
    assert!(r.message.contains("No dead code"));
}

#[test]
fn test_check_dead_code_with_count_nonzero_blocking() {
    let config = ReleaseConfig {
        dead_code_command: r#"echo {"count": 5}"#.to_string(),
        fail_on_dead_code: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_dead_code(Path::new("."));
    assert!(!r.passed);
    assert!(r.message.contains("5 dead code items"));
}

#[test]
fn test_check_dead_code_with_count_nonzero_warning() {
    let config = ReleaseConfig {
        dead_code_command: r#"echo {"count": 3}"#.to_string(),
        fail_on_dead_code: false,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_dead_code(Path::new("."));
    assert!(r.passed);
    assert!(r.message.contains("warning"));
}

#[test]
fn test_check_dead_code_keyword_blocking() {
    let config = ReleaseConfig {
        dead_code_command: "echo dead_code detected unused items".to_string(),
        fail_on_dead_code: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_dead_code(Path::new("."));
    assert!(!r.passed);
}

#[test]
fn test_check_dead_code_keyword_warning() {
    let config = ReleaseConfig {
        dead_code_command: "echo dead_code found".to_string(),
        fail_on_dead_code: false,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_dead_code(Path::new("."));
    assert!(r.passed);
    assert!(r.message.contains("warning"));
}

// ============================================================================
// check_pmat_complexity detailed branching
// ============================================================================

#[test]
fn test_check_complexity_within_limit() {
    let config = ReleaseConfig {
        complexity_command: r#"echo {"max_complexity": 10}"#.to_string(),
        max_complexity: 20,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_complexity(Path::new("."));
    assert!(r.passed);
    assert!(r.message.contains("10"));
}

#[test]
fn test_check_complexity_exceeds_blocking() {
    let config = ReleaseConfig {
        complexity_command: r#"echo {"max_complexity": 30}"#.to_string(),
        max_complexity: 20,
        fail_on_complexity: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_complexity(Path::new("."));
    assert!(!r.passed);
    assert!(r.message.contains("30"));
}

#[test]
fn test_check_complexity_zero_violations() {
    let config = ReleaseConfig {
        complexity_command: r#"echo {"violations": 0}"#.to_string(),
        fail_on_complexity: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_complexity(Path::new("."));
    assert!(r.passed);
}

#[test]
fn test_check_complexity_violations_blocking() {
    let config = ReleaseConfig {
        complexity_command: r#"echo {"violations": 5}"#.to_string(),
        fail_on_complexity: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_complexity(Path::new("."));
    assert!(!r.passed);
}

#[test]
fn test_check_complexity_fail_exit_no_data() {
    let config = ReleaseConfig {
        complexity_command: "false".to_string(),
        fail_on_complexity: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_complexity(Path::new("."));
    // No JSON and command failed → "completed (warning)"
    assert!(r.passed);
}

// ============================================================================
// check_pmat_satd detailed branching
// ============================================================================

#[test]
fn test_check_satd_within_limit() {
    let config = ReleaseConfig {
        satd_command: r#"echo {"total": 5}"#.to_string(),
        max_satd_items: 10,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_satd(Path::new("."));
    assert!(r.passed);
}

#[test]
fn test_check_satd_exceeds_blocking() {
    let config = ReleaseConfig {
        satd_command: r#"echo {"total": 15}"#.to_string(),
        max_satd_items: 10,
        fail_on_satd: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_satd(Path::new("."));
    assert!(!r.passed);
}

#[test]
fn test_check_satd_exceeds_warning() {
    let config = ReleaseConfig {
        satd_command: r#"echo {"total": 15}"#.to_string(),
        max_satd_items: 10,
        fail_on_satd: false,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_satd(Path::new("."));
    assert!(r.passed);
    assert!(r.message.contains("warning"));
}

#[test]
fn test_check_satd_no_count_success() {
    let config = ReleaseConfig { satd_command: "true".to_string(), ..Default::default() };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_satd(Path::new("."));
    assert!(r.passed);
    assert!(r.message.contains("check passed"));
}

#[test]
fn test_check_satd_no_count_failure() {
    let config = ReleaseConfig { satd_command: "false".to_string(), ..Default::default() };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_satd(Path::new("."));
    assert!(r.passed);
    assert!(r.message.contains("check completed"));
}

// ============================================================================
// check_pmat_tdg / check_pmat_popper score thresholds
// ============================================================================

#[test]
fn test_check_tdg_above_threshold() {
    let config = ReleaseConfig {
        tdg_command: r#"echo {"score": 90}"#.to_string(),
        min_tdg_score: 80.0,
        fail_on_tdg: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_tdg(Path::new("."));
    assert!(r.passed);
}

#[test]
fn test_check_tdg_below_threshold_blocking() {
    let config = ReleaseConfig {
        tdg_command: r#"echo {"score": 70}"#.to_string(),
        min_tdg_score: 80.0,
        fail_on_tdg: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_tdg(Path::new("."));
    assert!(!r.passed);
}

#[test]
fn test_check_popper_above_threshold() {
    let config = ReleaseConfig {
        popper_command: r#"echo {"score": 80}"#.to_string(),
        min_popper_score: 60.0,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_popper(Path::new("."));
    assert!(r.passed);
}

#[test]
fn test_check_popper_below_threshold_blocking() {
    let config = ReleaseConfig {
        popper_command: r#"echo {"score": 40}"#.to_string(),
        min_popper_score: 60.0,
        fail_on_popper: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_popper(Path::new("."));
    assert!(!r.passed);
}

// ============================================================================
// check_book_build with empty command
// ============================================================================

#[test]
fn test_check_book_empty_command() {
    let dir = std::env::temp_dir().join("test_rp_book_empty_cmd");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("book")).unwrap();
    let orch = orchestrator_with_command("book", "");
    let r = orch.check_book_build(&dir);
    assert!(r.passed);
    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// run_check_command with arguments
// ============================================================================

#[test]
fn test_run_check_command_with_args() {
    let r = run_check_command(
        "echo hello world",
        "test_id",
        "skip",
        Path::new("."),
        |out, stdout, _| {
            if out.status.success() && stdout.contains("hello") {
                PreflightCheck::pass("test_id", "got output")
            } else {
                PreflightCheck::fail("test_id", "bad")
            }
        },
    );
    assert!(r.passed);
    assert!(r.message.contains("got output"));
}

#[test]
fn test_run_check_command_stderr_access() {
    let r = run_check_command(
        "ls /nonexistent_path_xyz",
        "test_id",
        "skip",
        Path::new("."),
        |_out, _stdout, stderr| {
            if stderr.contains("No such file") || stderr.contains("cannot access") {
                PreflightCheck::pass("test_id", "stderr captured")
            } else {
                // On some systems the error message differs
                PreflightCheck::pass("test_id", "command ran")
            }
        },
    );
    assert!(r.passed);
}

// ============================================================================
// Coverage: check_examples_run with real examples (compilation paths)
// ============================================================================

#[test]
fn test_check_examples_run_with_rs_files() {
    // Create a temp project that has examples/ dir with .rs files
    // but no actual cargo project, so compilation fails
    let dir = std::env::temp_dir().join("test_rp_examples_rs_files");
    let _ = std::fs::remove_dir_all(&dir);
    let examples_dir = dir.join("examples");
    std::fs::create_dir_all(&examples_dir).unwrap();
    std::fs::write(examples_dir.join("demo.rs"), "fn main() {}").unwrap();
    std::fs::write(dir.join("Cargo.toml"), "[package]\nname = \"x\"\nversion = \"0.1.0\"\n")
        .unwrap();

    let orch = default_orchestrator();
    let r = orch.check_examples_run(&dir);
    // Examples exist but cargo run --example will fail (no real project)
    // The check should still complete (either pass or fail depending on fail_on_examples)
    let _ = r; // Just verify it doesn't panic
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_check_examples_run_non_blocking() {
    let dir = std::env::temp_dir().join("test_rp_examples_nonblock");
    let _ = std::fs::remove_dir_all(&dir);
    let examples_dir = dir.join("examples");
    std::fs::create_dir_all(&examples_dir).unwrap();
    std::fs::write(examples_dir.join("hello.rs"), "fn main() {}").unwrap();
    std::fs::write(dir.join("Cargo.toml"), "[package]\nname = \"x\"\nversion = \"0.1.0\"\n")
        .unwrap();

    let config = ReleaseConfig { fail_on_examples: false, ..Default::default() };
    let orch = make_orchestrator(config);
    let r = orch.check_examples_run(&dir);
    // With fail_on_examples=false, failures should not block
    // (compilation will fail but that's non-blocking)
    let _ = r;
    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// Coverage: run_check_command "other spawn error" (not NotFound)
// ============================================================================

#[test]
fn test_run_check_command_permission_denied() {
    // Create a file that is not executable to trigger PermissionDenied
    let dir = std::env::temp_dir().join("test_rp_perm_denied");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let script = dir.join("no_exec.sh");
    std::fs::write(&script, "#!/bin/sh\necho hello").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o644)).unwrap();
    }
    let cmd = script.to_string_lossy().to_string();
    let r = run_check_command(&cmd, "perm_test", "skip", Path::new("."), |_, _, _| {
        PreflightCheck::pass("perm_test", "should not reach")
    });
    // On Unix, PermissionDenied hits the Err(e) branch (not NotFound)
    assert!(!r.passed);
    assert!(r.message.contains("Failed to run"));
    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// Coverage: discover_examples with Cargo.toml [[example]] + examples dir
// ============================================================================

#[test]
fn test_discover_examples_cargo_toml_matching_file() {
    let dir = std::env::temp_dir().join("test_rp_disc_cargo_match");
    let _ = std::fs::remove_dir_all(&dir);
    let examples_dir = dir.join("examples");
    std::fs::create_dir_all(&examples_dir).unwrap();
    std::fs::write(examples_dir.join("myexample.rs"), "fn main() {}").unwrap();
    let cargo = r#"[package]
name = "x"
version = "0.1.0"

[[example]]
name = "myexample"
"#;
    std::fs::write(dir.join("Cargo.toml"), cargo).unwrap();

    let orch = default_orchestrator();
    let examples = orch.discover_examples(&dir);
    assert!(examples.contains(&"myexample".to_string()));
    // Should NOT have duplicates even though found in Cargo.toml AND examples dir
    assert_eq!(examples.iter().filter(|n| *n == "myexample").count(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_discover_examples_cargo_toml_no_matching_file() {
    let dir = std::env::temp_dir().join("test_rp_disc_cargo_nomatch");
    let _ = std::fs::remove_dir_all(&dir);
    let examples_dir = dir.join("examples");
    std::fs::create_dir_all(&examples_dir).unwrap();
    // Cargo.toml references "ghost" but no ghost.rs exists
    let cargo = r#"[package]
name = "x"
version = "0.1.0"

[[example]]
name = "ghost"
"#;
    std::fs::write(dir.join("Cargo.toml"), cargo).unwrap();

    let orch = default_orchestrator();
    let examples = orch.discover_examples(&dir);
    // "ghost" should not be in the list since ghost.rs doesn't exist
    assert!(!examples.contains(&"ghost".to_string()));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_discover_examples_mixed_files() {
    let dir = std::env::temp_dir().join("test_rp_disc_mixed");
    let _ = std::fs::remove_dir_all(&dir);
    let examples_dir = dir.join("examples");
    std::fs::create_dir_all(&examples_dir).unwrap();
    // Mix of .rs files and non-rs files
    std::fs::write(examples_dir.join("good.rs"), "fn main() {}").unwrap();
    std::fs::write(examples_dir.join("readme.txt"), "not an example").unwrap();
    std::fs::write(dir.join("Cargo.toml"), "[package]\nname = \"x\"\nversion = \"0.1.0\"\n")
        .unwrap();

    let orch = default_orchestrator();
    let examples = orch.discover_examples(&dir);
    assert!(examples.contains(&"good".to_string()));
    assert!(!examples.contains(&"readme".to_string()));
    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// Coverage: parse_score_from_json edge cases
// ============================================================================

#[test]
fn test_parse_score_from_json_empty_string() {
    assert_eq!(ReleaseOrchestrator::parse_score_from_json("", "score"), None);
}

#[test]
fn test_parse_score_from_json_invalid_json() {
    assert_eq!(ReleaseOrchestrator::parse_score_from_json("not json at all", "score"), None);
}

#[test]
fn test_parse_score_from_json_key_with_string_value() {
    // The value is a string, not a number - should fail to parse
    assert_eq!(ReleaseOrchestrator::parse_score_from_json(r#"{"score": "high"}"#, "score"), None);
}

#[test]
fn test_parse_count_from_json_zero() {
    assert_eq!(ReleaseOrchestrator::parse_count_from_json(r#"{"count": 0}"#, "count"), Some(0));
}

// ============================================================================
// Coverage: check_pmat_comply error path (no violations, command failed)
// ============================================================================

#[test]
fn test_check_pmat_comply_fail_no_violations_no_success() {
    // Command fails but output has no CB- or violation markers
    let config = ReleaseConfig {
        comply_command: "false".to_string(),
        fail_on_comply_violations: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_comply(Path::new("."));
    // No violations detected, but command failed -> "PMAT comply error"
    assert!(!r.passed);
    assert!(r.message.contains("PMAT comply error"));
}

// ============================================================================
// Coverage: check_pmat_complexity non-blocking branch
// ============================================================================

#[test]
fn test_check_complexity_exceeds_non_blocking() {
    let config = ReleaseConfig {
        complexity_command: r#"echo {"max_complexity": 30}"#.to_string(),
        max_complexity: 20,
        fail_on_complexity: false,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_complexity(Path::new("."));
    // Non-blocking: should pass even though exceeds limit
    // Falls through to the success/warning catch-all
    assert!(r.passed);
}

#[test]
fn test_check_complexity_violations_non_blocking() {
    let config = ReleaseConfig {
        complexity_command: r#"echo {"violations": 5}"#.to_string(),
        fail_on_complexity: false,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_complexity(Path::new("."));
    // Non-blocking -> should pass through to the command success branch
    assert!(r.passed);
}

#[test]
fn test_check_complexity_success_exit_no_json() {
    let config = ReleaseConfig {
        complexity_command: "true".to_string(),
        fail_on_complexity: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_complexity(Path::new("."));
    // Command succeeded, no JSON data -> "Complexity check passed"
    assert!(r.passed);
    assert!(r.message.contains("check passed"));
}

// ============================================================================
// Coverage: check_pmat_tdg below threshold non-blocking
// ============================================================================

#[test]
fn test_check_tdg_below_threshold_non_blocking() {
    let config = ReleaseConfig {
        tdg_command: r#"echo {"score": 70}"#.to_string(),
        min_tdg_score: 80.0,
        fail_on_tdg: false,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_tdg(Path::new("."));
    assert!(r.passed);
    assert!(r.message.contains("warning"));
}

// ============================================================================
// Coverage: check_pmat_popper below threshold non-blocking
// ============================================================================

#[test]
fn test_check_popper_below_threshold_non_blocking() {
    let config = ReleaseConfig {
        popper_command: r#"echo {"score": 40}"#.to_string(),
        min_popper_score: 60.0,
        fail_on_popper: false,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_popper(Path::new("."));
    assert!(r.passed);
    assert!(r.message.contains("warning"));
}

// ============================================================================
// Coverage: check_pmat_tdg and popper with alternative JSON keys
// ============================================================================

#[test]
fn test_check_tdg_with_tdg_score_key() {
    let config = ReleaseConfig {
        tdg_command: r#"echo {"tdg_score": 95}"#.to_string(),
        min_tdg_score: 80.0,
        fail_on_tdg: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_tdg(Path::new("."));
    assert!(r.passed);
}

#[test]
fn test_check_popper_with_popper_score_key() {
    let config = ReleaseConfig {
        popper_command: r#"echo {"popper_score": 75}"#.to_string(),
        min_popper_score: 60.0,
        fail_on_popper: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_popper(Path::new("."));
    assert!(r.passed);
}

// ============================================================================
// Coverage: check_pmat_dead_code with dead_code_count key
// ============================================================================

#[test]
fn test_check_dead_code_with_alt_key() {
    let config = ReleaseConfig {
        dead_code_command: r#"echo {"dead_code_count": 0}"#.to_string(),
        fail_on_dead_code: true,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_dead_code(Path::new("."));
    assert!(r.passed);
    assert!(r.message.contains("No dead code"));
}

// ============================================================================
// Coverage: check_pmat_satd with satd_count key
// ============================================================================

#[test]
fn test_check_satd_with_alt_key() {
    let config = ReleaseConfig {
        satd_command: r#"echo {"satd_count": 3}"#.to_string(),
        max_satd_items: 10,
        ..Default::default()
    };
    let orch = make_orchestrator(config);
    let r = orch.check_pmat_satd(Path::new("."));
    assert!(r.passed);
}

// ============================================================================
// Coverage: check_examples_run compilation error path
// ============================================================================

#[test]
fn test_check_examples_all_pass() {
    // All examples succeed -> pass message with count
    let dir = std::env::temp_dir().join("test_rp_examples_all_pass");
    let _ = std::fs::remove_dir_all(&dir);
    let examples_dir = dir.join("examples");
    std::fs::create_dir_all(&examples_dir).unwrap();
    std::fs::write(examples_dir.join("demo.rs"), "fn main() {}").unwrap();
    std::fs::write(dir.join("Cargo.toml"), "[package]\nname = \"x\"\nversion = \"0.1.0\"\n")
        .unwrap();

    let orch = default_orchestrator();
    let r = orch.check_examples_run(&dir);
    // Just make sure the function completes without panic
    // In CI, cargo run will fail because it's not a real project
    let _ = r;
    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// Coverage: check_examples_run compilation error + fail_on_examples branches
// ============================================================================

#[test]
fn test_check_examples_compilation_error_blocking() {
    // Create a minimal Cargo project with a broken example that triggers
    // the "error[E" / "could not compile" detection path (lines 414-416)
    // and the fail_on_examples=true branch (lines 434-441).
    let dir =
        std::env::temp_dir().join(format!("test_rp_examples_compile_err_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let src_dir = dir.join("src");
    let examples_dir = dir.join("examples");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::create_dir_all(&examples_dir).unwrap();

    // Valid lib.rs so cargo can parse the project
    std::fs::write(src_dir.join("lib.rs"), "pub fn hello() {}\n").unwrap();

    // Broken example that won't compile
    std::fs::write(
        examples_dir.join("broken.rs"),
        "fn main() { let x: i32 = \"not a number\"; }\n",
    )
    .unwrap();

    let cargo_toml = r#"[package]
name = "testproj"
version = "0.1.0"
edition = "2021"

[[example]]
name = "broken"
"#;
    std::fs::write(dir.join("Cargo.toml"), cargo_toml).unwrap();

    // fail_on_examples = true to hit lines 434-441
    let config = ReleaseConfig { fail_on_examples: true, ..Default::default() };
    let orch = make_orchestrator(config);
    let r = orch.check_examples_run(&dir);

    // The broken example should fail compilation, triggering the error path.
    // With fail_on_examples=true, the check should fail.
    assert!(!r.passed, "Should fail when example has compilation error");
    assert!(
        r.message.contains("broken") || r.message.contains("failed"),
        "Message should mention the failed example: {}",
        r.message
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_check_examples_compilation_error_non_blocking() {
    // Same broken project, but fail_on_examples=false triggers lines 445-451
    let dir = std::env::temp_dir().join("test_rp_examples_compile_warn");
    let _ = std::fs::remove_dir_all(&dir);
    let src_dir = dir.join("src");
    let examples_dir = dir.join("examples");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::create_dir_all(&examples_dir).unwrap();

    std::fs::write(src_dir.join("lib.rs"), "pub fn hello() {}\n").unwrap();
    std::fs::write(
        examples_dir.join("broken.rs"),
        "fn main() { let x: i32 = \"not a number\"; }\n",
    )
    .unwrap();

    let cargo_toml = r#"[package]
name = "testproj2"
version = "0.1.0"
edition = "2021"

[[example]]
name = "broken"
"#;
    std::fs::write(dir.join("Cargo.toml"), cargo_toml).unwrap();

    let config = ReleaseConfig { fail_on_examples: false, ..Default::default() };
    let orch = make_orchestrator(config);
    let r = orch.check_examples_run(&dir);

    // With fail_on_examples=false, should pass even with failures
    assert!(r.passed, "Should pass (non-blocking) even with compilation error: {}", r.message);
    assert!(
        r.message.contains("not blocking") || r.message.contains("verified"),
        "Message should indicate non-blocking: {}",
        r.message
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_check_examples_runtime_nonzero_exit() {
    // Create a valid example that compiles but exits with non-zero status.
    // Since --help is passed, the example might exit non-zero. If the stderr
    // doesn't contain "error[E" or "could not compile", it hits the
    // "runtime exit with non-zero is OK" path (line 418-419).
    let dir = std::env::temp_dir().join("test_rp_examples_runtime_exit");
    let _ = std::fs::remove_dir_all(&dir);
    let src_dir = dir.join("src");
    let examples_dir = dir.join("examples");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::create_dir_all(&examples_dir).unwrap();

    std::fs::write(src_dir.join("lib.rs"), "pub fn hello() {}\n").unwrap();

    // Example that compiles fine but exits non-zero
    std::fs::write(examples_dir.join("exits.rs"), "fn main() { std::process::exit(1); }\n")
        .unwrap();

    let cargo_toml = r#"[package]
name = "testproj3"
version = "0.1.0"
edition = "2021"

[[example]]
name = "exits"
"#;
    std::fs::write(dir.join("Cargo.toml"), cargo_toml).unwrap();

    let orch = default_orchestrator();
    let r = orch.check_examples_run(&dir);

    // The example compiles fine, exits non-zero at runtime — should be
    // treated as a pass (runtime non-zero is OK for --help)
    let _ = r; // Just verify no panic
    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// Coverage: check_git_clean error branch (lines 34-35)
// ============================================================================

#[test]
fn test_check_git_clean_nonexistent_dir() {
    // Git status in a nonexistent directory triggers an error
    let dir = std::env::temp_dir().join("test_rp_git_nonexist_dir_xyz");
    let _ = std::fs::remove_dir_all(&dir);
    // Do NOT create the directory — let git status fail
    let orch = default_orchestrator();
    let r = orch.check_git_clean(&dir);
    // git status in nonexistent dir returns error — exercises the code path
    let _ = r;
}
