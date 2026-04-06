//! Stack drift detection and compliance command implementations.

use crate::ansi_colors::Colorize;
use crate::stack;
use std::path::PathBuf;

use super::{ComplyOutputFormat, StackOutputFormat};

pub(super) fn cmd_stack_drift(
    format: StackOutputFormat,
    fix: bool,
    workspace: Option<PathBuf>,
) -> anyhow::Result<()> {
    use stack::crates_io::CratesIoClient;
    use stack::drift::{format_drift_json, DriftChecker};

    if !matches!(format, StackOutputFormat::Json) {
        println!("{}", "🔍 PAIML Stack Ecosystem Drift (maintainer view)".bright_cyan().bold());
        println!("{}", "═".repeat(70).dimmed());
        println!();
    }

    let rt = tokio::runtime::Runtime::new()?;

    let drifts = rt.block_on(async {
        let mut client = CratesIoClient::new().with_persistent_cache();
        let mut checker = DriftChecker::new();
        checker.detect_drift(&mut client).await
    })?;

    if drifts.is_empty() {
        match format {
            StackOutputFormat::Json => {
                println!("{}", format_drift_json(&[])?);
            }
            _ => {
                println!(
                    "{}",
                    "✅ No drift detected - all stack crates are using latest versions!"
                        .green()
                        .bold()
                );
            }
        }
        return Ok(());
    }

    match format {
        StackOutputFormat::Json => {
            println!("{}", format_drift_json(&drifts)?);
        }
        StackOutputFormat::Text | StackOutputFormat::Markdown => {
            display_drift_table(&drifts);
            if fix {
                display_drift_fix_commands(&drifts, workspace);
            } else {
                println!();
                println!("{}", "Run with --fix to generate fix commands.".dimmed());
            }
        }
    }

    Ok(())
}

fn display_drift_table(drifts: &[stack::DriftReport]) {
    println!(
        "{:<20} {:>10} {:<15} {:>12} {:>12} {:>8}",
        "Crate".bright_yellow().bold(),
        "Version".bright_yellow().bold(),
        "Dependency".bright_yellow().bold(),
        "Uses".bright_yellow().bold(),
        "Latest".bright_yellow().bold(),
        "Severity".bright_yellow().bold()
    );
    println!("{}", "─".repeat(70).dimmed());

    for drift in drifts {
        let severity_colored = match drift.severity {
            stack::DriftSeverity::Major => "MAJOR".bright_red().bold(),
            stack::DriftSeverity::Minor => "MINOR".yellow(),
            stack::DriftSeverity::Patch => "PATCH".dimmed(),
        };
        println!(
            "{:<20} {:>10} {:<15} {:>12} {:>12} {}",
            drift.crate_name.cyan(),
            drift.crate_version.dimmed(),
            drift.dependency.bright_white(),
            drift.uses_version.red(),
            drift.latest_version.green(),
            severity_colored
        );
    }

    println!("{}", "─".repeat(70).dimmed());
    println!();

    let major_count =
        drifts.iter().filter(|d| matches!(d.severity, stack::DriftSeverity::Major)).count();
    let minor_count =
        drifts.iter().filter(|d| matches!(d.severity, stack::DriftSeverity::Minor)).count();

    println!(
        "📊 {} drift issues: {} {}, {} {}",
        drifts.len(),
        major_count.to_string().bright_red().bold(),
        "major".red(),
        minor_count.to_string().yellow(),
        "minor".yellow()
    );
}

fn display_drift_fix_commands(drifts: &[stack::DriftReport], workspace: Option<PathBuf>) {
    println!();
    println!("{}", "🔧 Fix Commands (run in each crate directory):".bright_cyan());
    println!("{}", "─".repeat(70).dimmed());

    let ws = workspace.unwrap_or_else(|| PathBuf::from(".."));

    let mut by_crate: std::collections::HashMap<&str, Vec<&stack::DriftReport>> =
        std::collections::HashMap::new();
    for drift in drifts {
        by_crate.entry(&drift.crate_name).or_default().push(drift);
    }

    for (crate_name, crate_drifts) in by_crate {
        let crate_path = ws.join(crate_name);
        println!();
        println!("# {} ({})", crate_name.cyan().bold(), crate_path.display());
        for drift in crate_drifts {
            let dep = &drift.dependency;
            let old_minor = extract_minor_version(&drift.uses_version);
            let new_minor = extract_minor_version(&drift.latest_version);
            println!(
                "sed -i 's/{} = \"\\([^0-9]*\\){}\\([^\"]*\\)\"/{} = \"\\1{}\\2\"/g' {}/Cargo.toml",
                dep,
                old_minor,
                dep,
                new_minor,
                crate_path.display()
            );
        }
    }

    println!();
    println!("{}", "After fixing, run 'cargo update' in each crate directory.".dimmed());
}

/// Extract minor version (e.g., "0.10" from "0.10.1" or "^0.10")
pub(super) fn extract_minor_version(version: &str) -> String {
    let cleaned = version
        .trim_start_matches('^')
        .trim_start_matches('~')
        .trim_start_matches('=')
        .trim_start_matches('>')
        .trim_start_matches('<')
        .trim();

    let parts: Vec<&str> = cleaned.split('.').collect();
    if parts.len() >= 2 {
        format!("{}.{}", parts[0], parts[1])
    } else {
        cleaned.to_string()
    }
}

pub(super) fn cmd_stack_comply(
    rule: Option<String>,
    fix: bool,
    dry_run: bool,
    format: ComplyOutputFormat,
    workspace: Option<PathBuf>,
    list_rules: bool,
) -> anyhow::Result<()> {
    use crate::comply::{ComplyConfig, ComplyReportFormat, StackComplyEngine};

    let workspace_path = workspace.unwrap_or_else(|| {
        std::env::current_dir()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."))
    });

    println!("{}", "🔍 PAIML Stack Compliance Check".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    let config = ComplyConfig::load_or_default(&workspace_path);
    let mut engine = StackComplyEngine::new(config);

    if list_rules {
        println!("{}", "Available compliance rules:".bright_white().bold());
        println!();
        for (id, description) in engine.available_rules() {
            println!("  {} - {}", id.bright_yellow(), description);
        }
        return Ok(());
    }

    println!("{}", "Discovering projects...".dimmed());
    engine.discover_projects(&workspace_path)?;
    println!(
        "  Found {} projects ({} PAIML crates)",
        engine.projects().len(),
        engine.projects().iter().filter(|p| p.is_paiml_crate).count()
    );
    println!();

    let report = if fix || dry_run {
        if dry_run {
            println!("{}", "⚠️  DRY RUN - No changes will be made".yellow().bold());
        } else {
            println!("{}", "🔧 Attempting to fix violations...".bright_yellow());
        }
        engine.fix_all(dry_run)
    } else if let Some(rule_id) = rule {
        println!("Checking rule: {}", rule_id.bright_yellow());
        engine.check_rule(&rule_id)
    } else {
        println!("{}", "Running all compliance checks...".dimmed());
        engine.check_all()
    };

    let output_format = match format {
        ComplyOutputFormat::Text => ComplyReportFormat::Text,
        ComplyOutputFormat::Json => ComplyReportFormat::Json,
        ComplyOutputFormat::Markdown => ComplyReportFormat::Markdown,
        ComplyOutputFormat::Html => ComplyReportFormat::Html,
    };

    println!();
    println!("{}", report.format(output_format));

    if report.is_compliant() {
        println!("{}", "✅ All compliance checks passed!".bright_green().bold());
    } else {
        println!(
            "{}",
            format!(
                "❌ {} violations found across {} projects",
                report.summary.total_violations, report.summary.failing_projects
            )
            .bright_red()
            .bold()
        );

        if report.summary.fixable_violations > 0 {
            println!(
                "   {} violations are auto-fixable (run with --fix)",
                report.summary.fixable_violations
            );
        }
    }

    Ok(())
}
