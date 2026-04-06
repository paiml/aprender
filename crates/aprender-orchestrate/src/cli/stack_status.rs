//! Stack status, sync, tree, quality, and gate command implementations.

use crate::ansi_colors::Colorize;
use crate::stack;
use std::io::IsTerminal;
use std::path::PathBuf;

use super::StackOutputFormat;

pub(super) fn cmd_stack_status(
    simple: bool,
    format: StackOutputFormat,
    tree: bool,
) -> anyhow::Result<()> {
    use stack::checker::{
        format_report_json, format_report_markdown, format_report_text, StackChecker,
    };
    use stack::crates_io::CratesIoClient;

    println!("{}", "📊 PAIML Stack Status".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    let workspace_path = PathBuf::from(".");

    let mut checker = StackChecker::from_workspace(&workspace_path)?.verify_published(true);

    let rt = tokio::runtime::Runtime::new()?;

    let report = rt.block_on(async {
        let mut client = CratesIoClient::new();
        checker.check(&mut client).await
    })?;

    if tree {
        println!("{}", "Dependency Tree:".bright_yellow().bold());
        if let Ok(order) = checker.topological_order() {
            for (i, name) in order.iter().enumerate() {
                println!("  {}. {}", i + 1, name.cyan());
            }
        }
        println!();
    }

    let output = match format {
        StackOutputFormat::Text => format_report_text(&report),
        StackOutputFormat::Json => format_report_json(&report)?,
        StackOutputFormat::Markdown => format_report_markdown(&report),
    };

    let is_tty = std::io::stdout().is_terminal();

    #[cfg(feature = "presentar-terminal")]
    let can_tui = true;
    #[cfg(not(feature = "presentar-terminal"))]
    let can_tui = false;

    if simple || !can_tui || !matches!(format, StackOutputFormat::Text) || !is_tty {
        println!("{}", output);
    } else {
        #[cfg(feature = "presentar-terminal")]
        stack::tui::run_dashboard(report)?;
    }

    Ok(())
}

pub(super) fn cmd_stack_sync(
    crate_name: Option<String>,
    all: bool,
    dry_run: bool,
    align: Option<String>,
) -> anyhow::Result<()> {
    println!("{}", "🔄 PAIML Stack Sync".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!();

    if dry_run {
        println!("{}", "⚠️  DRY RUN - No changes will be made".yellow().bold());
        println!();
    }

    if let Some(alignment) = &align {
        println!("Aligning dependency: {}", alignment.cyan());
    }

    if all {
        println!("Syncing all crates...");
    } else if let Some(name) = &crate_name {
        println!("Syncing crate: {}", name.cyan());
    } else {
        println!("{}", "❌ Specify a crate name or use --all".red());
        return Ok(());
    }

    let workspace_path = PathBuf::from(".");
    let checker = stack::checker::StackChecker::from_workspace(&workspace_path)?;
    let path_deps = checker.find_path_dependencies();

    if path_deps.is_empty() {
        println!(
            "{}",
            "✓ No path dependencies found - all deps use crates.io versions".bright_green()
        );
        return Ok(());
    }

    println!(
        "Found {} path dependencies to convert:\n",
        path_deps.len().to_string().bright_yellow()
    );

    for dep in &path_deps {
        let recommendation = dep.recommended.as_deref().unwrap_or("(version unknown)");
        println!(
            "  {} {} → {} depends on {} via path",
            "•".dimmed(),
            dep.crate_name.cyan(),
            dep.dependency.bright_yellow(),
            dep.current.dimmed()
        );
        println!("    Recommended: {}", recommendation.bright_green());
    }

    if dry_run {
        println!();
        println!("{}", "Dry run complete. Use without --dry-run to apply changes.".dimmed());
    } else {
        println!();
        println!("{}", "Path dependency conversion requires manual Cargo.toml edits.".yellow());
        println!(
            "Use {} to identify which dependencies need updating.",
            "batuta stack check --verify-published".cyan()
        );
    }

    Ok(())
}

pub(super) fn cmd_stack_tree(
    format: &str,
    health: bool,
    filter: Option<&str>,
) -> anyhow::Result<()> {
    use stack::tree::{build_tree, format_ascii, format_dot, format_json, OutputFormat};

    let output_format: OutputFormat = format.parse().map_err(|e: String| anyhow::anyhow!(e))?;

    let mut tree = build_tree();

    if let Some(layer_filter) = filter {
        tree.layers.retain(|l| l.name == layer_filter);
        tree.total_crates = tree.layers.iter().map(|l| l.components.len()).sum();
    }

    let output = match output_format {
        OutputFormat::Ascii => format_ascii(&tree, health),
        OutputFormat::Json => format_json(&tree)?,
        OutputFormat::Dot => format_dot(&tree),
    };

    println!("{}", output);
    Ok(())
}

pub(super) fn cmd_stack_quality(
    component: Option<String>,
    strict: bool,
    format: StackOutputFormat,
    workspace: Option<PathBuf>,
) -> anyhow::Result<()> {
    use stack::{
        format_quality_report_json, format_quality_report_text, QualityChecker, StackQualityReport,
    };

    let workspace_path = workspace.unwrap_or_else(|| {
        std::env::current_dir()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."))
    });

    let rt = tokio::runtime::Runtime::new()?;

    let report = match component {
        Some(comp_name) => {
            let checker = QualityChecker::new(workspace_path.clone());
            let quality = rt.block_on(async { checker.check_component(&comp_name).await })?;
            StackQualityReport::from_components(vec![quality])
        }
        None => {
            let components = check_all_stack_components(&rt, &workspace_path);
            StackQualityReport::from_components(components)
        }
    };

    let output = match format {
        StackOutputFormat::Json => format_quality_report_json(&report)?,
        StackOutputFormat::Text | StackOutputFormat::Markdown => {
            format_quality_report_text(&report)
        }
    };

    println!("{}", output);

    if strict && !report.is_all_a_plus() {
        anyhow::bail!("Strict mode: not all components meet A+ quality standard");
    }

    if !report.release_ready {
        anyhow::bail!(
            "Quality gate failed: {} component(s) below minimum threshold",
            report.blocked_components.len()
        );
    }

    Ok(())
}

/// Check all stack components from LAYER_DEFINITIONS.
fn check_all_stack_components(
    rt: &tokio::runtime::Runtime,
    workspace_path: &std::path::Path,
) -> Vec<stack::ComponentQuality> {
    use stack::tree::LAYER_DEFINITIONS;
    use stack::QualityChecker;

    let mut components = Vec::new();
    for (_layer_name, layer_components) in LAYER_DEFINITIONS {
        for comp_name in *layer_components {
            let comp_path = workspace_path.join(comp_name);
            if comp_path.join("Cargo.toml").exists() {
                let checker = QualityChecker::new(comp_path);
                match rt.block_on(async { checker.check_component(comp_name).await }) {
                    Ok(quality) => components.push(quality),
                    Err(e) => {
                        eprintln!("Warning: Failed to check {}: {}", comp_name, e);
                    }
                }
            }
        }
    }
    components
}

/// Collect quality data for all stack components in the workspace.
pub(super) fn collect_gate_components(
    workspace_path: &std::path::Path,
    rt: &tokio::runtime::Runtime,
    quiet: bool,
) -> Vec<stack::ComponentQuality> {
    use stack::{tree::LAYER_DEFINITIONS, QualityChecker};

    let mut components = Vec::new();
    for (_layer_name, layer_components) in LAYER_DEFINITIONS.iter() {
        for comp_name in *layer_components {
            let comp_path = workspace_path.join(comp_name);
            if comp_path.join("Cargo.toml").exists() {
                let checker = QualityChecker::new(comp_path);
                match rt.block_on(async { checker.check_component(comp_name).await }) {
                    Ok(quality) => components.push(quality),
                    Err(e) => {
                        if !quiet {
                            eprintln!("Warning: Failed to check {}: {}", comp_name, e);
                        }
                    }
                }
            }
        }
    }
    components
}

pub(super) fn cmd_stack_gate(workspace: Option<PathBuf>, quiet: bool) -> anyhow::Result<()> {
    use stack::StackQualityReport;

    let workspace_path = workspace.unwrap_or_else(|| {
        std::env::current_dir()
            .expect("Failed to get current directory")
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
    });

    if !quiet {
        eprintln!("{}", "🔒 Stack Quality Gate Check".bright_cyan().bold());
        eprintln!("{}", "═".repeat(60).dimmed());
    }

    let rt = tokio::runtime::Runtime::new()?;
    let components = collect_gate_components(&workspace_path, &rt, quiet);
    let report = StackQualityReport::from_components(components);

    if !report.release_ready {
        eprintln!();
        eprintln!("{}", "❌ QUALITY GATE FAILED".bright_red().bold());
        eprintln!();
        eprintln!(
            "The following {} component(s) are below A- threshold (SQI < 85):",
            report.blocked_components.len()
        );
        for comp in &report.blocked_components {
            eprintln!("  • {}", comp.bright_yellow());
        }
        eprintln!();
        eprintln!("Run 'batuta stack quality' for detailed breakdown.");
        eprintln!("Fix quality issues before committing or deploying.");
        eprintln!();

        anyhow::bail!(
            "Quality gate failed: {} blocked component(s)",
            report.blocked_components.len()
        );
    }

    if !quiet {
        eprintln!("{}", "✅ All components meet A- quality threshold".bright_green());
        eprintln!("   Stack Quality Index: {:.1}%", report.stack_quality_index);
    }

    Ok(())
}
