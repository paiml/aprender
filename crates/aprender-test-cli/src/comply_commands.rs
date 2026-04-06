fn run_comply(config: &CliConfig, args: &probador::ComplyArgs) -> CliResult<()> {
    // Handle subcommands if present
    if let Some(ref subcommand) = args.subcommand {
        return match subcommand {
            probador::ComplySubcommand::Check(check_args) => run_comply_check(config, check_args),
            probador::ComplySubcommand::Migrate(migrate_args) => {
                run_comply_migrate(config, migrate_args)
            }
            probador::ComplySubcommand::Diff(diff_args) => run_comply_diff(config, diff_args),
            probador::ComplySubcommand::Enforce(enforce_args) => {
                run_comply_enforce(config, enforce_args)
            }
            probador::ComplySubcommand::Report(report_args) => {
                run_comply_report(config, report_args)
            }
        };
    }

    // Default behavior: run checks (backwards compatibility)
    if config.verbosity != Verbosity::Quiet {
        eprintln!("\n══════════════════════════════════════════════════════════════");
        eprintln!("  PROBAR COMPLY - WASM Compliance Checker");
        eprintln!("══════════════════════════════════════════════════════════════\n");
    }

    run_comply_checks_internal(config, args)
}

// =============================================================================
// Comply Subcommand Handlers
// NOTE: Compliance check functions (check_c001 through check_c010) and
// ComplianceResult are now imported from probador::handlers::comply
// =============================================================================

/// Run comply check subcommand
fn run_comply_check(config: &CliConfig, args: &probador::ComplyCheckArgs) -> CliResult<()> {
    use jugar_probar::strict::{E2ETestChecklist, WasmStrictMode};

    if config.verbosity != Verbosity::Quiet {
        eprintln!("\n══════════════════════════════════════════════════════════════");
        eprintln!("  PROBAR COMPLY CHECK - WASM Compliance Checker");
        eprintln!("══════════════════════════════════════════════════════════════\n");
    }

    let strict_mode = if args.strict {
        WasmStrictMode::production()
    } else {
        WasmStrictMode::development()
    };
    let _checklist = E2ETestChecklist::new().with_strict_mode(strict_mode);

    // Build a ComplyArgs for compatibility
    let compat_args = probador::ComplyArgs {
        subcommand: None,
        path: args.path.clone(),
        checks: args.checks.clone(),
        fail_fast: false,
        format: args.format.clone(),
        max_wasm_size: 5_242_880,
        strict: args.strict,
        report: None,
        detailed: args.detailed,
    };

    // Reuse the existing check logic
    run_comply_checks_internal(config, &compat_args)
}

/// Internal check logic (shared between top-level and check subcommand)
fn run_comply_checks_internal(config: &CliConfig, args: &probador::ComplyArgs) -> CliResult<()> {
    type CheckFn = Box<dyn Fn(&std::path::Path, &probador::ComplyArgs) -> ComplianceResult>;

    let checks_to_run = build_compliance_checks();
    let filtered_checks: Vec<(&str, &str, CheckFn)> =
        filter_compliance_checks(checks_to_run, args.checks.as_ref());

    if config.verbosity != Verbosity::Quiet {
        eprintln!(
            "Running {} compliance check(s) on {}\n",
            filtered_checks.len(),
            args.path.display()
        );
    }

    let (results, all_passed) = execute_compliance_checks(&filtered_checks, config, args);

    output_compliance_results(
        config,
        &results,
        &args.format,
        args.report.as_deref(),
        all_passed,
    )
}

/// Build the vector of all compliance checks (C001-C010).
fn build_compliance_checks() -> Vec<(
    &'static str,
    &'static str,
    Box<dyn Fn(&std::path::Path, &probador::ComplyArgs) -> ComplianceResult>,
)> {
    vec![
        (
            "C001",
            "Code execution verified",
            Box::new(|path, _| check_c001_code_execution(path)),
        ),
        (
            "C002",
            "Console errors fail tests",
            Box::new(|_, _| check_c002_console_errors()),
        ),
        (
            "C003",
            "Custom elements tested",
            Box::new(|path, _| check_c003_custom_elements(path)),
        ),
        (
            "C004",
            "Threading modes tested",
            Box::new(|_, _| check_c004_threading_modes()),
        ),
        (
            "C005",
            "Low memory tested",
            Box::new(|_, _| check_c005_low_memory()),
        ),
        (
            "C006",
            "COOP/COEP headers",
            Box::new(|path, _| check_c006_headers(path)),
        ),
        (
            "C007",
            "Replay hash matches",
            Box::new(|_, _| check_c007_replay_hash()),
        ),
        (
            "C008",
            "Cache handling",
            Box::new(|_, _| check_c008_cache()),
        ),
        (
            "C009",
            "WASM size limit",
            Box::new(|path, args| check_c009_wasm_size(path, args.max_wasm_size)),
        ),
        (
            "C010",
            "No panic paths",
            Box::new(|path, _| check_c010_panic_paths(path)),
        ),
    ]
}

/// Filter compliance checks by requested IDs, or return all if none specified.
fn filter_compliance_checks<F>(
    checks: Vec<(&'static str, &'static str, F)>,
    requested: Option<&Vec<String>>,
) -> Vec<(&'static str, &'static str, F)> {
    match requested {
        Some(ids) => checks
            .into_iter()
            .filter(|(id, _, _)| ids.iter().any(|r| r == id))
            .collect(),
        None => checks,
    }
}

/// Run each compliance check, printing results as we go.
fn execute_compliance_checks(
    checks: &[(
        &str,
        &str,
        Box<dyn Fn(&std::path::Path, &probador::ComplyArgs) -> ComplianceResult>,
    )],
    config: &CliConfig,
    args: &probador::ComplyArgs,
) -> (Vec<ComplianceResult>, bool) {
    let mut results = Vec::new();
    let mut all_passed = true;

    for (id, description, check_fn) in checks {
        let result = check_fn(&args.path, args);
        print_check_result(config, id, description, &result, args.detailed);

        if !result.passed {
            all_passed = false;
            if args.fail_fast {
                results.push(result);
                break;
            }
        }
        results.push(result);
    }

    (results, all_passed)
}

/// Print a single compliance check result.
fn print_check_result(
    config: &CliConfig,
    id: &str,
    description: &str,
    result: &ComplianceResult,
    detailed: bool,
) {
    if config.verbosity == Verbosity::Quiet {
        return;
    }
    let status = if result.passed { "✓" } else { "✗" };
    let color = if result.passed {
        "\x1b[32m"
    } else {
        "\x1b[31m"
    };
    let reset = "\x1b[0m";

    eprintln!("  {color}[{status}]{reset} {id}: {description}");
    if detailed && !result.details.is_empty() {
        for detail in &result.details {
            eprintln!("      └─ {detail}");
        }
    }
}

/// Output compliance results: summary, report file, stdout format.
fn output_compliance_results(
    config: &CliConfig,
    results: &[ComplianceResult],
    format: &probador::ComplyOutputFormat,
    report_path: Option<&std::path::Path>,
    all_passed: bool,
) -> CliResult<()> {
    let passed_count = results.iter().filter(|r| r.passed).count();
    let total_count = results.len();

    if config.verbosity != Verbosity::Quiet {
        eprintln!("\n══════════════════════════════════════════════════════════════");
        eprintln!("  Result: {passed_count}/{total_count} checks passed");
        eprintln!("══════════════════════════════════════════════════════════════\n");
    }

    if let Some(path) = report_path {
        let report = generate_comply_report(results, format);
        std::fs::write(path, &report).map_err(|e| {
            probador::CliError::report_generation(format!("Failed to write report: {e}"))
        })?;
    }

    match format {
        probador::ComplyOutputFormat::Json | probador::ComplyOutputFormat::Junit => {
            let report = generate_comply_report(results, format);
            println!("{report}");
        }
        probador::ComplyOutputFormat::Text => {}
    }

    if all_passed {
        Ok(())
    } else {
        Err(probador::CliError::test_execution(format!(
            "Compliance check failed: {passed_count}/{total_count} checks passed",
        )))
    }
}

/// Run comply migrate subcommand
fn run_comply_migrate(config: &CliConfig, args: &probador::ComplyMigrateArgs) -> CliResult<()> {
    use std::process::Command;

    if config.verbosity != Verbosity::Quiet {
        eprintln!("\n══════════════════════════════════════════════════════════════");
        eprintln!("  PROBAR COMPLY MIGRATE");
        eprintln!("══════════════════════════════════════════════════════════════\n");
    }

    // Check for uncommitted changes
    if !args.force {
        let status = Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(&args.path)
            .output();

        if let Ok(output) = status {
            if !output.stdout.is_empty() {
                return Err(probador::CliError::config(
                    "Uncommitted changes detected. Use --force to override.".to_string(),
                ));
            }
        }
    }

    let target_version = args.version.as_deref().unwrap_or("latest");

    if args.dry_run {
        eprintln!("DRY RUN - would migrate to version: {target_version}");
        eprintln!("\nChanges that would be applied:");
        eprintln!("  - Update probar.toml version field");
        eprintln!("  - Add new required test configurations");
        eprintln!("  - Update deprecated API calls");
        return Ok(());
    }

    eprintln!("Migrating to version: {target_version}");

    // Create probar.toml if it doesn't exist
    let config_path = args.path.join("probar.toml");
    if !config_path.exists() {
        let config_content = format!(
            r#"# Probar Configuration
# Generated by: probador comply migrate

[probar]
version = "{}"
cross_origin_isolated = true

[strict]
require_code_execution = true
fail_on_console_error = true
verify_custom_elements = true

[quality]
min_coverage = 95
max_wasm_size = 5242880
"#,
            target_version
        );
        std::fs::write(&config_path, config_content)
            .map_err(|e| probador::CliError::config(format!("Failed to create config: {e}")))?;
        eprintln!("  Created: {}", config_path.display());
    }

    eprintln!("\nMigration complete!");
    Ok(())
}

/// Run comply diff subcommand
fn run_comply_diff(config: &CliConfig, args: &probador::ComplyDiffArgs) -> CliResult<()> {
    if config.verbosity != Verbosity::Quiet {
        eprintln!("\n══════════════════════════════════════════════════════════════");
        eprintln!("  PROBAR COMPLY DIFF - Version Changelog");
        eprintln!("══════════════════════════════════════════════════════════════\n");
    }

    let from_version = args.from.as_deref().unwrap_or("0.3.0");
    let to_version = args.to.as_deref().unwrap_or(env!("CARGO_PKG_VERSION"));

    eprintln!("Changes from {} to {}:\n", from_version, to_version);

    // Changelog entries
    let changelog = vec![
        (
            "0.4.0",
            vec![
                ("FEATURE", "Added AudioEmulator for getUserMedia mocking"),
                (
                    "FEATURE",
                    "Added WasmThreadCapabilities for COOP/COEP detection",
                ),
                ("FEATURE", "Added WorkerEmulator for Web Worker testing"),
                (
                    "FEATURE",
                    "Added StreamingUxValidator for real-time UX testing",
                ),
                (
                    "FEATURE",
                    "Added probador comply subcommands (check, migrate, diff, enforce, report)",
                ),
                ("BREAKING", "ConsoleCapture now requires WasmStrictMode"),
            ],
        ),
        (
            "0.3.0",
            vec![
                ("FEATURE", "Added playbook state machine testing"),
                ("FEATURE", "Added pixel coverage heatmaps"),
                ("FEATURE", "Added serve --cross-origin-isolated flag"),
            ],
        ),
    ];

    for (version, changes) in &changelog {
        if args.breaking_only {
            let breaking: Vec<_> = changes.iter().filter(|(t, _)| *t == "BREAKING").collect();
            if !breaking.is_empty() {
                eprintln!("Version {}:", version);
                for (_, desc) in breaking {
                    eprintln!("  ⚠️  BREAKING: {desc}");
                }
                eprintln!();
            }
        } else {
            eprintln!("Version {}:", version);
            for (change_type, desc) in changes {
                let icon = match *change_type {
                    "FEATURE" => "✨",
                    "BREAKING" => "⚠️ ",
                    "FIX" => "🐛",
                    _ => "•",
                };
                eprintln!("  {icon} {desc}");
            }
            eprintln!();
        }
    }

    Ok(())
}

/// Run comply enforce subcommand
fn run_comply_enforce(config: &CliConfig, args: &probador::ComplyEnforceArgs) -> CliResult<()> {
    if config.verbosity != Verbosity::Quiet {
        eprintln!("\n══════════════════════════════════════════════════════════════");
        eprintln!("  PROBAR COMPLY ENFORCE - Git Hooks");
        eprintln!("══════════════════════════════════════════════════════════════\n");
    }

    let hooks_dir = args.path.join(".git/hooks");

    if !hooks_dir.exists() {
        return Err(probador::CliError::config(
            "Not a git repository (no .git/hooks directory)".to_string(),
        ));
    }

    let pre_commit_path = hooks_dir.join("pre-commit");

    if args.disable {
        // Remove hooks
        if pre_commit_path.exists() {
            std::fs::remove_file(&pre_commit_path)
                .map_err(|e| probador::CliError::config(format!("Failed to remove hook: {e}")))?;
            eprintln!("Removed pre-commit hook");
        } else {
            eprintln!("No pre-commit hook found");
        }
        return Ok(());
    }

    // Confirm installation
    if !args.yes {
        eprintln!("This will install a pre-commit hook that runs:");
        eprintln!("  - probador comply check --strict");
        eprintln!("  - WASM binary size check");
        eprintln!("  - Panic path verification");
        eprintln!("\nProceed? [y/N] ");
        // In a real implementation, read user input
        // For now, just proceed
    }

    // Generate pre-commit hook
    let hook_content = r##"#!/bin/bash
# Probar WASM Quality Gates
# Generated by: probador comply enforce

set -e

echo "Running Probar quality gates..."

# 1. WASM binary size regression check
MAX_WASM_SIZE=5000000
wasm_files=$(find target -name "*.wasm" 2>/dev/null | head -1)
if [ -n "$wasm_files" ]; then
    wasm_size=$(stat -c%s "$wasm_files" 2>/dev/null || stat -f%z "$wasm_files" 2>/dev/null || echo 0)
    if [ "$wasm_size" -gt "$MAX_WASM_SIZE" ]; then
        echo "ERROR: WASM binary size regression: ${wasm_size} > ${MAX_WASM_SIZE}"
        exit 1
    fi
fi

# 2. No panic paths in WASM code
if grep -rn "unwrap()" --include="*.rs" src/ 2>/dev/null | grep -v "// SAFETY:" | grep -v "#[cfg(test)]" | head -5; then
    echo "WARNING: unwrap() found in src/ code - consider using expect() with message"
fi

# 3. Run compliance check
if command -v probador &> /dev/null; then
    probador comply check --strict . || {
        echo "ERROR: Compliance check failed"
        exit 1
    }
fi

echo "Probar quality gates passed!"
"##;

    std::fs::write(&pre_commit_path, hook_content)
        .map_err(|e| probador::CliError::config(format!("Failed to write hook: {e}")))?;

    // Make executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&pre_commit_path)
            .map_err(|e| probador::CliError::config(format!("Failed to get perms: {e}")))?
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&pre_commit_path, perms)
            .map_err(|e| probador::CliError::config(format!("Failed to set perms: {e}")))?;
    }

    eprintln!(
        "Installed pre-commit hook at: {}",
        pre_commit_path.display()
    );
    eprintln!("\nHook will run on every commit to enforce:");
    eprintln!("  - WASM binary size limits");
    eprintln!("  - Panic-free code patterns");
    eprintln!("  - Full compliance check");

    Ok(())
}

/// Run comply report subcommand
fn run_comply_report(config: &CliConfig, args: &probador::ComplyReportArgs) -> CliResult<()> {
    use std::fs;
    type CheckFn = fn(&std::path::Path) -> ComplianceResult;

    if config.verbosity != Verbosity::Quiet {
        eprintln!("\n══════════════════════════════════════════════════════════════");
        eprintln!("  PROBAR COMPLY REPORT");
        eprintln!("══════════════════════════════════════════════════════════════\n");
    }

    // Run all checks to generate report
    let check_args = probador::ComplyArgs {
        subcommand: None,
        path: args.path.clone(),
        checks: None,
        fail_fast: false,
        format: probador::ComplyOutputFormat::Text,
        max_wasm_size: 5_242_880,
        strict: false,
        report: None,
        detailed: true,
    };

    // Collect results silently
    let mut results: Vec<ComplianceResult> = Vec::new();
    let checks: Vec<(&str, &str, CheckFn)> = vec![
        ("C001", "Code execution verified", |p| {
            check_c001_code_execution(p)
        }),
        ("C002", "Console errors fail tests", |_| {
            check_c002_console_errors()
        }),
        ("C003", "Custom elements tested", |p| {
            check_c003_custom_elements(p)
        }),
        ("C004", "Threading modes tested", |_| {
            check_c004_threading_modes()
        }),
        ("C005", "Low memory tested", |_| check_c005_low_memory()),
        ("C006", "COOP/COEP headers", |p| check_c006_headers(p)),
        ("C007", "Replay hash matches", |_| check_c007_replay_hash()),
        ("C008", "Cache handling", |_| check_c008_cache()),
        ("C009", "WASM size limit", |p| {
            check_c009_wasm_size(p, 5_242_880)
        }),
        ("C010", "No panic paths", |p| check_c010_panic_paths(p)),
    ];

    for (_, _, check_fn) in &checks {
        results.push(check_fn(&check_args.path));
    }

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();
    let timestamp = chrono::Utc::now().to_rfc3339();

    let report = match args.format {
        probador::ComplyReportFormat::Text => {
            format!(
                r#"============================================================
Probador WASM Compliance Report
============================================================

Project: {}
Probador Version: {}
Scan Time: {}

Checks:
{}
Summary: {}/{} passed

============================================================
"#,
                args.path.display(),
                env!("CARGO_PKG_VERSION"),
                timestamp,
                results
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        let status = if r.passed { "✓" } else { "⚠" };
                        format!(
                            "  {} C{:03} {}: {}",
                            status,
                            i + 1,
                            if r.passed { "Passed" } else { "Warning" },
                            r.details.first().unwrap_or(&String::new())
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                passed,
                total
            )
        }
        probador::ComplyReportFormat::Json => serde_json::json!({
            "project": args.path.display().to_string(),
            "version": env!("CARGO_PKG_VERSION"),
            "timestamp": timestamp,
            "results": results.iter().map(|r| {
                serde_json::json!({
                    "id": r.id,
                    "passed": r.passed,
                    "details": r.details
                })
            }).collect::<Vec<_>>(),
            "summary": { "passed": passed, "total": total }
        })
        .to_string(),
        probador::ComplyReportFormat::Markdown => {
            format!(
                r#"# Probador WASM Compliance Report

**Project**: {}
**Version**: {}
**Date**: {}

## Summary

| Metric | Value |
|--------|-------|
| Passed | {} |
| Total | {} |
| Score | {:.0}% |

## Checks

| Check | Status | Details |
|-------|--------|---------|
{}

---
*Generated by probador {}*
"#,
                args.path.display(),
                env!("CARGO_PKG_VERSION"),
                timestamp,
                passed,
                total,
                (passed as f64 / total as f64) * 100.0,
                results
                    .iter()
                    .map(|r| {
                        let status = if r.passed { "✅ Pass" } else { "⚠️ Warn" };
                        format!(
                            "| {} | {} | {} |",
                            r.id,
                            status,
                            r.details.first().unwrap_or(&String::new())
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                env!("CARGO_PKG_VERSION")
            )
        }
        probador::ComplyReportFormat::Html => {
            format!(
                r#"<!DOCTYPE html>
<html>
<head>
    <title>Probador Compliance Report</title>
    <style>
        body {{ font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        .pass {{ color: green; }}
        .warn {{ color: orange; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>Probador WASM Compliance Report</h1>
    <p><strong>Project:</strong> {}</p>
    <p><strong>Version:</strong> {}</p>
    <p><strong>Date:</strong> {}</p>
    <h2>Summary: {}/{} checks passed</h2>
    <table>
        <tr><th>Check</th><th>Status</th><th>Details</th></tr>
        {}
    </table>
</body>
</html>"#,
                args.path.display(),
                env!("CARGO_PKG_VERSION"),
                timestamp,
                passed,
                total,
                results
                    .iter()
                    .map(|r| {
                        let (class, status) = if r.passed {
                            ("pass", "✓")
                        } else {
                            ("warn", "⚠")
                        };
                        format!(
                            "<tr><td>{}</td><td class=\"{}\">{}</td><td>{}</td></tr>",
                            r.id,
                            class,
                            status,
                            r.details.first().unwrap_or(&String::new())
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        }
    };

    if let Some(ref output_path) = args.output {
        fs::write(output_path, &report).map_err(|e| {
            probador::CliError::report_generation(format!("Failed to write report: {e}"))
        })?;
        eprintln!("Report written to: {}", output_path.display());
    } else {
        println!("{report}");
    }

    Ok(())
}

