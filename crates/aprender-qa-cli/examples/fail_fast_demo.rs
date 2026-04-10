//! Fail-Fast Mode Demo (FF-REPORT-001)
//!
//! Demonstrates the --fail-fast feature for debugging and GitHub ticket creation.
//! When a test fails, generates comprehensive diagnostic reports using apr tooling.
//!
//! Run with:
//! ```bash
//! cargo run --example fail_fast_demo -p apr-qa-cli
//! ```

use apr_qa_runner::{
    DiagnosticResult, DiagnosticsBundle, EnvironmentContext, ExecutionConfig, FailFastReport,
    FailFastReporter, FailureDetails, FailurePolicy, ReproductionInfo,
};
use std::path::Path;

#[allow(clippy::too_many_lines)]
fn main() {
    println!("=== Fail-Fast Mode Demo (FF-REPORT-001) ===\n");

    // Demonstrate FailurePolicy::FailFast
    let policy = FailurePolicy::FailFast;

    println!("FailurePolicy::FailFast properties:");
    println!("  emit_diagnostic(): {}", policy.emit_diagnostic());
    println!(
        "  stops_on_any_failure(): {}",
        policy.stops_on_any_failure()
    );

    // Show how to configure ExecutionConfig with FailFast
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::FailFast,
        output_dir: Some("output".to_string()), // Diagnostic reports written here
        ..Default::default()
    };

    println!("\nExecutionConfig with FailFast:");
    println!("  failure_policy: {:?}", config.failure_policy);
    println!("  output_dir: {:?}", config.output_dir);
    println!("  default_timeout_ms: {}", config.default_timeout_ms);

    // Demonstrate FailFastReporter
    println!("\n=== FailFastReporter Demo ===\n");

    let reporter = FailFastReporter::new(Path::new("output"));
    println!("Created FailFastReporter with output_dir: output/");

    // Show what a diagnostic report looks like
    println!("\nDiagnostic Report Structure:");
    println!("  output/fail-fast-report/");
    println!("  ├── summary.md           # GitHub-ready markdown");
    println!("  ├── diagnostics.json     # Full machine-readable report");
    println!("  ├── check.json           # apr check output");
    println!("  ├── inspect.json         # apr inspect output");
    println!("  ├── trace.json           # apr trace output (if .apr)");
    println!("  ├── tensors.json         # apr tensors output");
    println!("  ├── environment.json     # OS, versions, git state");
    println!("  └── stderr.log           # Raw stderr capture");

    // Demonstrate creating a mock report structure
    println!("\n=== Sample Report Structure ===\n");

    let sample_report = FailFastReport {
        version: "1.0.0".to_string(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        failure: FailureDetails {
            gate_id: "G3-STABLE".to_string(),
            model: "Qwen/Qwen2.5-Coder-0.5B-Instruct".to_string(),
            format: "Apr".to_string(),
            backend: "Cpu".to_string(),
            outcome: "Crashed".to_string(),
            reason: "Process crashed with exit code -1".to_string(),
            exit_code: Some(-1),
            duration_ms: 52740,
            stderr: Some("SIGSEGV at 0x12345".to_string()),
        },
        environment: EnvironmentContext {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            apr_qa_version: env!("CARGO_PKG_VERSION").to_string(),
            apr_cli_version: "0.2.12".to_string(),
            git_commit: "abc123".to_string(),
            git_branch: "main".to_string(),
            git_dirty: false,
            rustc_version: "1.93.0".to_string(),
        },
        diagnostics: DiagnosticsBundle {
            check: Some(DiagnosticResult {
                command: "apr check /path/to/model.apr --json".to_string(),
                success: false,
                stdout: "{}".to_string(),
                stderr: "Error: failed pipeline check".to_string(),
                duration_ms: 2300,
                timed_out: false,
            }),
            inspect: None,
            trace: None,
            tensors: None,
            explain: None,
        },
        reproduction: ReproductionInfo {
            command: "apr-qa run playbook.yaml --fail-fast".to_string(),
            model_path: "/path/to/model.apr".to_string(),
            playbook: Some("playbook.yaml".to_string()),
        },
    };

    // Generate markdown preview
    let markdown = reporter.generate_markdown(&sample_report);
    println!("Sample Markdown Report (first 40 lines):");
    println!("----------------------------------------");
    for line in markdown.lines().take(40) {
        println!("{line}");
    }
    println!("...\n");

    // Compare with other policies
    println!("=== Policy Comparison ===\n");
    for (name, policy) in [
        ("StopOnFirst", FailurePolicy::StopOnFirst),
        ("StopOnP0", FailurePolicy::StopOnP0),
        ("CollectAll", FailurePolicy::CollectAll),
        ("FailFast", FailurePolicy::FailFast),
    ] {
        println!(
            "  {:<12} emit_diagnostic={:<5} stops_on_any={:<5}",
            name,
            policy.emit_diagnostic(),
            policy.stops_on_any_failure()
        );
    }

    println!("\n=== CLI Usage ===\n");
    println!("# Stop on first failure with diagnostic report generation:");
    println!("apr-qa run playbook.yaml --fail-fast\n");

    println!("# With full tracing for GitHub ticket:");
    println!("RUST_LOG=debug apr-qa run playbook.yaml --fail-fast 2>&1 | tee failure.log\n");

    println!("# View generated report:");
    println!("cat output/fail-fast-report/summary.md\n");

    println!("# Copy report to clipboard (Linux):");
    println!("cat output/fail-fast-report/summary.md | xclip -selection clipboard\n");

    println!("=== Demo Complete ===");
}
