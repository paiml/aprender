#![allow(deprecated, clippy::unwrap_used)]

/// Integration tests for Batuta pipeline based on sovereign-ai-spec.md section 4.1 and 4.2
use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

/// Helper to create batuta command with drift check disabled (for pre-release testing)
fn batuta_cmd() -> Command {
    let mut cmd = Command::cargo_bin("batuta").unwrap();
    cmd.arg("--unsafe-skip-drift-check");
    cmd
}

/// Test full analysis workflow
#[test]
fn test_analyze_workflow() {
    let mut cmd = batuta_cmd();

    cmd.arg("analyze")
        .arg("--languages")
        .arg(".")
        .assert()
        .success()
        .stdout(predicate::str::contains("Analysis Results"))
        .stdout(predicate::str::contains("Rust"));
}

/// Test workflow state persistence
#[test]
fn test_workflow_state_tracking() {
    let temp_dir = TempDir::new().unwrap();

    // Create a small test project structure
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("main.rs"), "fn main() { println!(\"test\"); }").unwrap();

    let state_file = temp_dir.path().join(".batuta-state.json");

    // Run analyze to create state
    let mut cmd = batuta_cmd();
    cmd.current_dir(temp_dir.path()).arg("analyze").arg("--languages").arg(".").assert().success();

    // Verify state file exists
    assert!(state_file.exists(), "State file should be created");

    // Read and verify state structure
    let state_content = fs::read_to_string(&state_file).unwrap();
    assert!(state_content.contains("Analysis"));
    assert!(state_content.contains("Completed"));
}

/// Test status command
#[test]
fn test_status_command() {
    let mut cmd = batuta_cmd();

    cmd.arg("status").assert().success().stdout(predicate::str::contains("Workflow Status"));
}

/// Test report generation (markdown)
#[test]
fn test_report_generation_markdown() {
    let temp_dir = TempDir::new().unwrap();

    // Create a test source file
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    let test_file = src_dir.join("main.py");
    fs::write(&test_file, "import numpy as np\nx = np.array([1, 2, 3])\n").unwrap();

    // Run analyze first to create workflow data
    batuta_cmd().arg("analyze").arg(&src_dir).current_dir(temp_dir.path()).assert().success();

    let report_path = temp_dir.path().join("report.md");

    let mut cmd = batuta_cmd();
    cmd.arg("report")
        .arg("--format")
        .arg("markdown")
        .arg("--output")
        .arg(&report_path)
        .current_dir(temp_dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("Report generated successfully"));

    // Verify report file exists and has content
    assert!(report_path.exists(), "Report file should be created");
    let content = fs::read_to_string(&report_path).unwrap();
    assert!(content.contains("# Migration Report"));
    assert!(content.contains("## Summary"));
}

/// Test report generation (JSON)
#[test]
fn test_report_generation_json() {
    let temp_dir = TempDir::new().unwrap();

    // Create a test source file
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    let test_file = src_dir.join("main.py");
    fs::write(&test_file, "import numpy as np\nx = np.array([1, 2, 3])\n").unwrap();

    // Run analyze first to create workflow data
    batuta_cmd().arg("analyze").arg(&src_dir).current_dir(temp_dir.path()).assert().success();

    let report_path = temp_dir.path().join("report.json");

    let mut cmd = batuta_cmd();
    cmd.arg("report")
        .arg("--format")
        .arg("json")
        .arg("--output")
        .arg(&report_path)
        .current_dir(temp_dir.path())
        .assert()
        .success();

    // Verify JSON is valid
    assert!(report_path.exists());
    let content = fs::read_to_string(&report_path).unwrap();
    let _: serde_json::Value = serde_json::from_str(&content).expect("Report should be valid JSON");
}

/// Test report generation (HTML)
#[test]
fn test_report_generation_html() {
    let temp_dir = TempDir::new().unwrap();

    // Create a test source file
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    let test_file = src_dir.join("main.py");
    fs::write(&test_file, "import numpy as np\nx = np.array([1, 2, 3])\n").unwrap();

    // Run analyze first to create workflow data
    batuta_cmd().arg("analyze").arg(&src_dir).current_dir(temp_dir.path()).assert().success();

    // Now generate the report
    let report_path = temp_dir.path().join("report.html");
    batuta_cmd()
        .arg("report")
        .arg("--format")
        .arg("html")
        .arg("--output")
        .arg(report_path.to_str().unwrap())
        .current_dir(temp_dir.path())
        .assert()
        .success();

    // Verify HTML structure
    assert!(report_path.exists(), "Report file should exist at {:?}", report_path);
    let content = fs::read_to_string(&report_path).unwrap();
    assert!(content.contains("<!DOCTYPE html>"));
    assert!(content.contains("<html"));
    assert!(content.contains("Migration Report"));
}

/// Test reset workflow
#[test]
fn test_reset_workflow() {
    let temp_dir = TempDir::new().unwrap();

    // Create a test file
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("lib.rs"), "pub fn test() {}").unwrap();

    // First run analyze to create state
    batuta_cmd().current_dir(temp_dir.path()).arg("analyze").arg(".").assert().success();

    // Then reset with --yes flag
    let mut cmd = batuta_cmd();
    cmd.current_dir(temp_dir.path())
        .arg("reset")
        .arg("--yes")
        .assert()
        .success()
        .stdout(predicate::str::contains("reset successfully"));
}

/// Test help command
#[test]
fn test_help_command() {
    let mut cmd = batuta_cmd();

    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("orchestration"))
        .stdout(predicate::str::contains("analyze"))
        .stdout(predicate::str::contains("transpile"))
        .stdout(predicate::str::contains("optimize"))
        .stdout(predicate::str::contains("validate"))
        .stdout(predicate::str::contains("build"));
}

/// Test CLI version
#[test]
fn test_version_command() {
    let mut cmd = batuta_cmd();

    cmd.arg("--version").assert().success().stdout(predicate::str::contains("batuta"));
}

/// Test Renacer syscall tracing validation (BATUTA-011)
#[test]
fn test_renacer_validation() {
    use batuta::pipeline::{PipelineContext, PipelineStage, ValidationStage};
    use std::path::PathBuf;
    use tokio::runtime::Runtime;

    let temp_dir = TempDir::new().unwrap();

    // Create a simple test context
    let ctx = PipelineContext::new(PathBuf::from(temp_dir.path()), PathBuf::from(temp_dir.path()));

    // Create ValidationStage with syscall tracing enabled
    let stage = ValidationStage::new(true, false);

    // Run validation (will skip if binaries don't exist)
    let rt = Runtime::new().unwrap();
    let result = rt.block_on(stage.execute(ctx));

    // Should succeed even if binaries don't exist (graceful handling)
    assert!(result.is_ok(), "ValidationStage should handle missing binaries gracefully");

    let final_ctx = result.unwrap();
    assert!(
        final_ctx.metadata.contains_key("validation_completed"),
        "Validation should mark completion in metadata"
    );
}

// ============================================================================
// INIT COMMAND TESTS
// ============================================================================

/// Test init command with default output directory
#[test]
fn test_init_command_default() {
    let temp_dir = TempDir::new().unwrap();

    // Create test source file
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("main.py"), "print('hello')").unwrap();

    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("init")
        .arg("--source")
        .arg(".")
        .assert()
        .success()
        .stdout(predicate::str::contains("Initializing Batuta project"))
        .stdout(predicate::str::contains("Configuration Summary"));

    // Verify batuta.toml was created
    let config_path = temp_dir.path().join("batuta.toml");
    assert!(config_path.exists(), "batuta.toml should be created");

    // Verify configuration content
    let config_content = fs::read_to_string(&config_path).unwrap();
    assert!(config_content.contains("[project]"));
    assert!(config_content.contains("[transpilation]"));
    assert!(config_content.contains("[optimization]"));
}

/// Test init command with custom output directory
#[test]
fn test_init_command_custom_output() {
    let temp_dir = TempDir::new().unwrap();

    // Create test source file
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("test.rs"), "fn main() {}").unwrap();

    let output_dir = temp_dir.path().join("custom-output");

    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("init")
        .arg("--source")
        .arg(".")
        .arg("--output")
        .arg(&output_dir)
        .assert()
        .success();

    // Verify output directory structure
    assert!(output_dir.exists());
    assert!(output_dir.join("src").exists());

    // Verify config contains custom output path
    let config_path = temp_dir.path().join("batuta.toml");
    let config_content = fs::read_to_string(&config_path).unwrap();
    assert!(config_content.contains("custom-output"));
}

// ============================================================================
// ANALYZE COMMAND VARIANT TESTS
// ============================================================================

/// Test analyze with --tdg flag
#[test]
fn test_analyze_with_tdg() {
    let temp_dir = TempDir::new().unwrap();
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("main.rs"), "fn main() { println!(\"test\"); }").unwrap();

    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("analyze")
        .arg("--tdg")
        .arg(".")
        .assert()
        .success()
        .stdout(predicate::str::contains("TDG Score"));
}

/// Test analyze with --dependencies flag
#[test]
fn test_analyze_with_dependencies() {
    let temp_dir = TempDir::new().unwrap();

    // Create a Python project with requirements.txt
    fs::write(temp_dir.path().join("requirements.txt"), "numpy>=1.20.0\npandas\n").unwrap();
    fs::write(temp_dir.path().join("main.py"), "import numpy\nimport pandas\n").unwrap();

    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("analyze")
        .arg("--dependencies")
        .arg(".")
        .assert()
        .success()
        .stdout(predicate::str::contains("Dependencies"));
}

/// Test analyze with all flags
#[test]
fn test_analyze_with_all_flags() {
    let temp_dir = TempDir::new().unwrap();
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("lib.rs"), "pub fn test() {}").unwrap();

    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("analyze")
        .arg("--tdg")
        .arg("--languages")
        .arg("--dependencies")
        .arg(".")
        .assert()
        .success()
        .stdout(predicate::str::contains("Analysis Results"))
        .stdout(predicate::str::contains("TDG Score"))
        .stdout(predicate::str::contains("Language Detection"));
}

/// Test analyze with nonexistent path
#[test]
fn test_analyze_nonexistent_path() {
    batuta_cmd()
        .arg("analyze")
        .arg("--languages")
        .arg("/nonexistent/path/that/does/not/exist")
        .assert()
        .failure();
}

// ============================================================================
// WORKFLOW COMMAND TESTS
// ============================================================================

/// Test transpile command without prerequisites
#[test]
fn test_transpile_without_analysis() {
    let temp_dir = TempDir::new().unwrap();

    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("transpile")
        .assert()
        .success() // Should succeed but show warning
        .stdout(predicate::str::contains("Analysis phase not completed"));
}

/// Test optimize command without prerequisites
#[test]
fn test_optimize_without_transpile() {
    let temp_dir = TempDir::new().unwrap();

    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("optimize")
        .assert()
        .code(1) // Fails when prerequisite not met
        .stderr(predicate::str::contains("Transpilation phase not completed"));
}

/// Test validate command without prerequisites
#[test]
fn test_validate_without_optimize() {
    let temp_dir = TempDir::new().unwrap();

    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("validate")
        .assert()
        .code(1) // Fails when prerequisite not met
        .stderr(predicate::str::contains("Optimization phase not completed"));
}

/// Test build command without prerequisites
#[test]
fn test_build_without_validate() {
    let temp_dir = TempDir::new().unwrap();

    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("build")
        .assert()
        .code(1) // Fails when prerequisite not met
        .stderr(predicate::str::contains("Validation phase not completed"));
}

/// Test optimize with different profiles
#[test]
fn test_optimize_profiles() {
    let profiles = vec!["fast", "balanced", "aggressive"];

    for profile in profiles {
        let temp_dir = TempDir::new().unwrap();

        // Optimize command fails without prerequisite but still shows intent
        batuta_cmd()
            .current_dir(temp_dir.path())
            .arg("optimize")
            .arg("--profile")
            .arg(profile)
            .assert()
            .code(1)
            .stdout(predicate::str::contains("Optimizing code"));
    }
}

/// Test optimize with GPU and SIMD flags
#[test]
fn test_optimize_with_gpu_simd() {
    let temp_dir = TempDir::new().unwrap();

    // Without prerequisites, fails but accepts flags and shows intent
    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("optimize")
        .arg("--enable-gpu")
        .arg("--enable-simd")
        .arg("--gpu-threshold")
        .arg("1000")
        .assert()
        .code(1)
        .stdout(predicate::str::contains("Optimizing code"));
}

/// Test validate with different flags
#[test]
fn test_validate_with_flags() {
    let temp_dir = TempDir::new().unwrap();

    // Without prerequisites, fails but accepts flags and shows intent
    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("validate")
        .arg("--trace-syscalls")
        .arg("--diff-output")
        .arg("--run-original-tests")
        .arg("--benchmark")
        .assert()
        .code(1)
        .stdout(predicate::str::contains("Validating equivalence"));
}

/// Test build with different targets
#[test]
fn test_build_variants() {
    let temp_dir = TempDir::new().unwrap();

    // Without prerequisites, all variants fail but show intent
    // Test release build
    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("build")
        .arg("--release")
        .assert()
        .code(1)
        .stdout(predicate::str::contains("Building Rust project"));

    // Test WASM build
    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("build")
        .arg("--wasm")
        .assert()
        .code(1)
        .stdout(predicate::str::contains("Building Rust project"));

    // Test custom target
    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("build")
        .arg("--target")
        .arg("x86_64-unknown-linux-gnu")
        .assert()
        .code(1)
        .stdout(predicate::str::contains("Building Rust project"));
}

// ============================================================================
// PARF COMMAND TESTS
// ============================================================================

/// Test parf command basic execution
#[test]
fn test_parf_command_basic() {
    let temp_dir = TempDir::new().unwrap();
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("lib.rs"), "pub fn test_function() {}").unwrap();

    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("parf")
        .arg("src")
        .assert()
        .success()
        .stdout(predicate::str::contains("PARF Analysis"));
}

/// Test parf with --patterns flag
#[test]
fn test_parf_patterns() {
    let temp_dir = TempDir::new().unwrap();
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("lib.rs"), "// TODO: implement this\npub fn test() {}").unwrap();

    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("parf")
        .arg("src")
        .arg("--patterns")
        .assert()
        .success()
        .stdout(predicate::str::contains("Code Patterns Detected"));
}

/// Test parf with --dependencies flag
#[test]
fn test_parf_dependencies() {
    let temp_dir = TempDir::new().unwrap();
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("main.rs"), "mod lib;\nfn main() {}").unwrap();
    fs::write(src_dir.join("lib.rs"), "pub fn test() {}").unwrap();

    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("parf")
        .arg("src")
        .arg("--dependencies")
        .assert()
        .success()
        .stdout(predicate::str::contains("Dependencies"));
}

/// Test parf with --find flag
#[test]
fn test_parf_find_references() {
    let temp_dir = TempDir::new().unwrap();
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("lib.rs"), "pub fn my_function() {}\nfn caller() { my_function(); }")
        .unwrap();

    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("parf")
        .arg("src")
        .arg("--find")
        .arg("my_function")
        .assert()
        .success()
        .stdout(predicate::str::contains("References"));
}

/// Test parf with different output formats
#[test]
fn test_parf_output_formats() {
    let temp_dir = TempDir::new().unwrap();
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("lib.rs"), "pub fn test() {}").unwrap();

    // Test JSON format
    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("parf")
        .arg("src")
        .arg("--patterns")
        .arg("--format")
        .arg("json")
        .assert()
        .success();

    // Test Markdown format
    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("parf")
        .arg("src")
        .arg("--patterns")
        .arg("--format")
        .arg("markdown")
        .assert()
        .success();

    // Test text format (default)
    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("parf")
        .arg("src")
        .arg("--patterns")
        .arg("--format")
        .arg("text")
        .assert()
        .success();
}

/// Test parf output to file
#[test]
fn test_parf_output_file() {
    let temp_dir = TempDir::new().unwrap();
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    fs::write(src_dir.join("lib.rs"), "pub fn test() {}").unwrap();

    let output_file = temp_dir.path().join("parf-report.txt");

    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("parf")
        .arg("src")
        .arg("--patterns")
        .arg("--output")
        .arg(&output_file)
        .assert()
        .success()
        .stdout(predicate::str::contains("Report written to"));

    assert!(output_file.exists(), "PARF report file should be created");
}

// ============================================================================
// GLOBAL FLAG TESTS
// ============================================================================

/// Test --verbose flag
#[test]
fn test_verbose_flag() {
    batuta_cmd().arg("--verbose").arg("status").assert().success();
}

/// Test --debug flag
#[test]
fn test_debug_flag() {
    batuta_cmd().arg("--debug").arg("status").assert().success();
}

// ============================================================================
// REPORT FORMAT TESTS
// ============================================================================

/// Test report generation with text format
#[test]
fn test_report_generation_text() {
    let temp_dir = TempDir::new().unwrap();

    // Create a test source file
    let src_dir = temp_dir.path().join("src");
    fs::create_dir(&src_dir).unwrap();
    let test_file = src_dir.join("main.py");
    fs::write(&test_file, "import numpy as np\nx = np.array([1, 2, 3])\n").unwrap();

    // Run analyze first to create workflow data
    batuta_cmd().arg("analyze").arg(&src_dir).current_dir(temp_dir.path()).assert().success();

    let report_path = temp_dir.path().join("report.txt");

    batuta_cmd()
        .arg("report")
        .arg("--format")
        .arg("text")
        .arg("--output")
        .arg(&report_path)
        .current_dir(temp_dir.path())
        .assert()
        .success();

    assert!(report_path.exists());
    let content = fs::read_to_string(&report_path).unwrap();
    assert!(content.contains("MIGRATION REPORT"));
}

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================

/// Test transpile with missing prerequisites
#[test]
fn test_transpile_missing_config() {
    let temp_dir = TempDir::new().unwrap();

    // Transpile without config or analysis should still succeed with warning
    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("transpile")
        .assert()
        .success()
        .stdout(predicate::str::contains("Transpiling code"));
}

/// Test report generation without workflow
#[test]
fn test_report_without_workflow() {
    let temp_dir = TempDir::new().unwrap();

    let report_path = temp_dir.path().join("report.html");

    // Report command fails without workflow data
    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("report")
        .arg("--output")
        .arg(&report_path)
        .assert()
        .code(1)
        .stdout(predicate::str::contains("report"));
}

/// Test reset on empty workflow
#[test]
fn test_reset_empty_workflow() {
    let temp_dir = TempDir::new().unwrap();

    batuta_cmd()
        .current_dir(temp_dir.path())
        .arg("reset")
        .arg("--yes")
        .assert()
        .success()
        .stdout(predicate::str::contains("No workflow state found"));
}

// ============================================================================
// STACK COMMAND TESTS (PAIML Stack Orchestration)
// ============================================================================

/// Test stack command help
#[test]
fn test_stack_help() {
    batuta_cmd()
        .arg("stack")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("PAIML Stack dependency orchestration"))
        .stdout(predicate::str::contains("check"))
        .stdout(predicate::str::contains("release"))
        .stdout(predicate::str::contains("status"))
        .stdout(predicate::str::contains("sync"));
}

/// Test stack check help
#[test]
fn test_stack_check_help() {
    batuta_cmd()
        .arg("stack")
        .arg("check")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Check dependency health"))
        .stdout(predicate::str::contains("--strict"))
        .stdout(predicate::str::contains("--verify-published"))
        .stdout(predicate::str::contains("--format"));
}

/// Test stack release help
#[test]
fn test_stack_release_help() {
    batuta_cmd()
        .arg("stack")
        .arg("release")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Prepare a stack release"))
        .stdout(predicate::str::contains("--dry-run"))
        .stdout(predicate::str::contains("--bump"))
        .stdout(predicate::str::contains("--publish"));
}

/// Test stack status help
#[test]
fn test_stack_status_help() {
    batuta_cmd()
        .arg("stack")
        .arg("status")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Show stack status"))
        .stdout(predicate::str::contains("--simple"))
        .stdout(predicate::str::contains("--tree"));
}

/// Test stack sync help
#[test]
fn test_stack_sync_help() {
    batuta_cmd()
        .arg("stack")
        .arg("sync")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Sync stack dependencies"))
        .stdout(predicate::str::contains("--dry-run"))
        .stdout(predicate::str::contains("--align"));
}

/// Test stack check command on current workspace
#[test]
fn test_stack_check_current_workspace() {
    // This test runs on the batuta workspace itself
    batuta_cmd()
        .arg("stack")
        .arg("check")
        .assert()
        .success()
        .stdout(predicate::str::contains("PAIML Stack Health Check"));
}

/// Test stack release dry run without crate name shows error
#[test]
fn test_stack_release_no_crate() {
    batuta_cmd()
        .arg("stack")
        .arg("release")
        .arg("--dry-run")
        .assert()
        .success()
        .stdout(predicate::str::contains("Specify a crate name or use --all"));
}

/// Test stack sync without crate name shows error
#[test]
fn test_stack_sync_no_crate() {
    batuta_cmd()
        .arg("stack")
        .arg("sync")
        .arg("--dry-run")
        .assert()
        .success()
        .stdout(predicate::str::contains("Specify a crate name or use --all"));
}

/// Test stack check with JSON output format
#[test]
fn test_stack_check_json_format() {
    batuta_cmd()
        .arg("stack")
        .arg("check")
        .arg("--format")
        .arg("json")
        .assert()
        .success()
        .stdout(predicate::str::contains("\"timestamp\""))
        .stdout(predicate::str::contains("\"summary\""));
}

/// Test stack status with tree view
#[test]
fn test_stack_status_tree() {
    batuta_cmd()
        .arg("stack")
        .arg("status")
        .arg("--tree")
        .assert()
        .success()
        .stdout(predicate::str::contains("Dependency Tree"));
}

// ============================================================================
// SOVEREIGN STACK INTEGRATION TESTS (Initial Release Spec §2-6)
// ============================================================================

/// Test BLAKE3 content addressing
#[test]
fn test_sovereign_blake3_content_addressing() {
    use blake3;

    let model_data = b"transformer weights and biases";
    let hash = blake3::hash(model_data);
    let hash_hex = hash.to_hex().to_string();

    // BLAKE3 produces 256-bit (64 hex char) hashes
    assert_eq!(hash_hex.len(), 64);

    // Same data produces same hash (content addressing)
    let hash2 = blake3::hash(model_data);
    assert_eq!(hash.as_bytes(), hash2.as_bytes());

    // Different data produces different hash
    let different_data = b"different model weights";
    let different_hash = blake3::hash(different_data);
    assert_ne!(hash.as_bytes(), different_hash.as_bytes());
}

/// Test Ed25519 digital signatures (via pacha)
#[test]
fn test_sovereign_ed25519_signatures() {
    use pacha::signing::{sign_model, verify_model, SigningKey};

    let model_data = b"model weights for signing";

    // Generate signing key
    let signing_key = SigningKey::generate();

    // Sign model (returns Result<ModelSignature>)
    let signature = sign_model(model_data, &signing_key).expect("Signing should succeed");

    // Verify signature - returns Result<()>
    let verify_result = verify_model(model_data, &signature);
    assert!(verify_result.is_ok(), "Signature should be valid");

    // Tampered data should fail verification
    let tampered_data = b"tampered model weights";
    let tampered_verify_result = verify_model(tampered_data, &signature);
    assert!(tampered_verify_result.is_err(), "Tampered data should fail verification");
}

/// Test ChaCha20-Poly1305 encryption (via pacha)
#[test]
fn test_sovereign_chacha20_encryption() {
    use pacha::crypto::{decrypt_model, encrypt_model, is_encrypted};

    let model_data = b"sensitive model weights";
    let passphrase = "secure-passphrase-123";

    // Encrypt
    let encrypted = encrypt_model(model_data, passphrase).expect("Encryption should succeed");

    // Encrypted data should be different from original
    assert_ne!(&encrypted[..], model_data);

    // Should be marked as encrypted
    assert!(is_encrypted(&encrypted), "Data should be marked as encrypted");

    // Decrypt
    let decrypted = decrypt_model(&encrypted, passphrase).expect("Decryption should succeed");
    assert_eq!(&decrypted[..], model_data, "Decrypted data should match original");

    // Wrong passphrase should fail
    let wrong_passphrase = "wrong-passphrase";
    let decrypt_result = decrypt_model(&encrypted, wrong_passphrase);
    assert!(decrypt_result.is_err(), "Wrong passphrase should fail decryption");
}

/// Test privacy tier enforcement (Sovereign tier)
#[test]
fn test_sovereign_privacy_tier_enforcement() {
    use batuta::serve::{BackendSelector, PrivacyTier, ServingBackend};

    // Sovereign tier: only local backends allowed
    let sovereign_selector = BackendSelector::new().with_privacy(PrivacyTier::Sovereign);

    // Local backends should be valid
    assert!(
        sovereign_selector.is_valid(ServingBackend::Realizar),
        "Realizar should be valid in Sovereign tier"
    );
    assert!(
        sovereign_selector.is_valid(ServingBackend::Ollama),
        "Ollama should be valid in Sovereign tier"
    );
    assert!(
        sovereign_selector.is_valid(ServingBackend::LlamaCpp),
        "LlamaCpp should be valid in Sovereign tier"
    );

    // External APIs should be blocked
    assert!(
        !sovereign_selector.is_valid(ServingBackend::OpenAI),
        "OpenAI should be BLOCKED in Sovereign tier"
    );
    assert!(
        !sovereign_selector.is_valid(ServingBackend::Anthropic),
        "Anthropic should be BLOCKED in Sovereign tier"
    );
    assert!(
        !sovereign_selector.is_valid(ServingBackend::AzureOpenAI),
        "AzureOpenAI should be BLOCKED in Sovereign tier"
    );
}

/// Test privacy tier enforcement (Private tier)
#[test]
fn test_private_privacy_tier_enforcement() {
    use batuta::serve::{BackendSelector, PrivacyTier, ServingBackend};

    // Private tier: VPC/dedicated endpoints allowed
    let private_selector = BackendSelector::new().with_privacy(PrivacyTier::Private);

    // Local backends still valid
    assert!(
        private_selector.is_valid(ServingBackend::Realizar),
        "Realizar should be valid in Private tier"
    );

    // Azure OpenAI (VPC-capable) should be allowed
    assert!(
        private_selector.is_valid(ServingBackend::AzureOpenAI),
        "AzureOpenAI should be valid in Private tier"
    );

    // Public APIs should be blocked
    assert!(
        !private_selector.is_valid(ServingBackend::OpenAI),
        "Public OpenAI should be BLOCKED in Private tier"
    );
}

/// Test backend selection Poka-Yoke validation
#[test]
fn test_sovereign_backend_poka_yoke() {
    use batuta::serve::{BackendSelector, PrivacyTier, ServingBackend};

    // Create selector with Sovereign privacy
    let selector = BackendSelector::new().with_privacy(PrivacyTier::Sovereign);

    // Validate returns specific violation details
    let openai_validation = selector.validate(ServingBackend::OpenAI);
    assert!(openai_validation.is_err(), "OpenAI validation should return error in Sovereign tier");

    // Validation error should indicate tier violation
    let error_msg = openai_validation.unwrap_err().to_string();
    assert!(
        error_msg.contains("Tier") || error_msg.contains("tier") || error_msg.contains("privacy"),
        "Error should mention tier violation"
    );

    // Valid backend should pass
    let realizar_validation = selector.validate(ServingBackend::Realizar);
    assert!(realizar_validation.is_ok(), "Realizar validation should succeed in Sovereign tier");
}

/// Test backend recommendation based on privacy tier
#[test]
fn test_sovereign_backend_recommendation() {
    use batuta::serve::{BackendSelector, PrivacyTier};

    // Sovereign tier should only recommend local backends
    let sovereign_selector = BackendSelector::new().with_privacy(PrivacyTier::Sovereign);

    let recommendations = sovereign_selector.recommend();

    // Should have at least one recommendation
    assert!(!recommendations.is_empty(), "Should have backend recommendations");

    // All recommendations should be local backends
    for backend in &recommendations {
        assert!(
            sovereign_selector.is_valid(*backend),
            "All recommendations should be valid for Sovereign tier"
        );
    }
}

/// Test full sovereign stack workflow
#[test]
fn test_sovereign_full_workflow() {
    use batuta::serve::{BackendSelector, PrivacyTier, ServingBackend};
    use blake3;
    use pacha::crypto::{decrypt_model, encrypt_model};
    use pacha::signing::{sign_model, verify_model, SigningKey};

    // Step 1: Create model data
    let model_data = b"production ML model weights";

    // Step 2: Content addressing (BLAKE3)
    let content_hash = blake3::hash(model_data);
    assert_eq!(content_hash.to_hex().to_string().len(), 64);

    // Step 3: Sign model (Ed25519)
    let signing_key = SigningKey::generate();
    let signature = sign_model(model_data, &signing_key).expect("Signing should succeed");

    // Step 4: Encrypt for distribution (ChaCha20-Poly1305)
    let passphrase = "distribution-key";
    let encrypted = encrypt_model(model_data, passphrase).unwrap();

    // Step 5: Simulate secure transfer (data in transit)
    let received_encrypted = encrypted.clone();

    // Step 6: Decrypt at destination
    let decrypted = decrypt_model(&received_encrypted, passphrase).unwrap();
    assert_eq!(&decrypted[..], model_data);

    // Step 7: Verify signature before loading (returns Result<()>)
    let verify_result = verify_model(&decrypted, &signature);
    assert!(verify_result.is_ok(), "Model should be authentic after decryption");

    // Step 8: Verify content hash matches
    let received_hash = blake3::hash(&decrypted);
    assert_eq!(
        received_hash.as_bytes(),
        content_hash.as_bytes(),
        "Content hash should match after decryption"
    );

    // Step 9: Select backend with Sovereign privacy
    let selector = BackendSelector::new().with_privacy(PrivacyTier::Sovereign);

    let recommendations = selector.recommend();
    assert!(!recommendations.is_empty());

    // Step 10: Verify selected backend is local-only
    let selected = recommendations[0];
    assert!(
        matches!(
            selected,
            ServingBackend::Realizar
                | ServingBackend::Ollama
                | ServingBackend::LlamaCpp
                | ServingBackend::Llamafile
        ),
        "Selected backend should be local in Sovereign mode"
    );
}

// ============================================================================
// ORACLE --FORMAT CODE TESTS
// ============================================================================

/// Test --format code with a recipe outputs raw code containing `use`
#[test]
fn test_oracle_format_code_recipe() {
    batuta_cmd()
        .arg("oracle")
        .arg("--recipe")
        .arg("ml-random-forest")
        .arg("--format")
        .arg("code")
        .assert()
        .success()
        .stdout(predicate::str::contains("use "));
}

/// Test --format code output contains no ANSI escape sequences
#[test]
fn test_oracle_format_code_no_ansi() {
    let output = batuta_cmd()
        .arg("oracle")
        .arg("--recipe")
        .arg("ml-random-forest")
        .arg("--format")
        .arg("code")
        .output()
        .expect("failed to execute");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains('\x1b'), "Code output must not contain ANSI escape sequences");
}

/// Test --format code with a NL query that produces code
#[test]
fn test_oracle_format_code_nl_query() {
    // "Train a random forest" maps to aprender which has a code example
    batuta_cmd()
        .arg("oracle")
        .arg("Train a random forest classifier")
        .arg("--format")
        .arg("code")
        .assert()
        .success()
        .stdout(predicate::str::contains("aprender"));
}

/// Test --format code with --list exits with code 1
#[test]
fn test_oracle_format_code_no_code_exits_1() {
    batuta_cmd().arg("oracle").arg("--list").arg("--format").arg("code").assert().code(1);
}
