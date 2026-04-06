//! Build and report command handlers.
//!
//! Extracted from pipeline_cmds.rs for QA-002 compliance.

use crate::ansi_colors::Colorize;
use crate::config::BatutaConfig;
use crate::report;
use crate::types::{PhaseStatus, WorkflowPhase, WorkflowState};
use std::path::{Path, PathBuf};
use tracing::warn;

use super::ReportFormat;

fn run_cargo_build(
    project_dir: &Path,
    release: bool,
    target: Option<&str>,
    wasm: bool,
    extra_flags: &[String],
) -> anyhow::Result<()> {
    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("build").current_dir(project_dir);

    if wasm {
        cmd.arg("--target").arg("wasm32-unknown-unknown");
    } else if let Some(t) = target {
        cmd.arg("--target").arg(t);
    }
    if release {
        cmd.arg("--release");
    }
    for flag in extra_flags {
        cmd.arg(flag);
    }

    // Display the command being run
    let mut display = String::from("cargo build");
    if release {
        display.push_str(" --release");
    }
    if wasm {
        display.push_str(" --target wasm32-unknown-unknown");
    } else if let Some(t) = target {
        display.push_str(&format!(" --target {}", t));
    }
    for flag in extra_flags {
        display.push(' ');
        display.push_str(flag);
    }
    println!("{} {}", "Running:".bright_yellow(), display.cyan());
    println!();

    let status = cmd
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to execute cargo (is it in PATH?): {}", e))?;

    if status.success() {
        println!();
        println!("{}", "✅ Build completed successfully!".bright_green().bold());
        Ok(())
    } else {
        let code = status.code().map_or("signal".to_string(), |c| c.to_string());
        println!();
        println!("{} Build failed with exit code: {}", "✗".red(), code);
        anyhow::bail!("cargo build failed (exit {})", code)
    }
}

pub fn cmd_build(release: bool, target: Option<String>, wasm: bool) -> anyhow::Result<()> {
    println!("{}", "🔨 Building Rust project...".bright_cyan().bold());
    println!();

    // Load workflow state
    let state_file = crate::cli::get_state_file_path();
    let mut state = WorkflowState::load(&state_file).unwrap_or_else(|e| {
        warn!("Failed to load workflow state, starting fresh: {e}");
        WorkflowState::new()
    });

    // Check if validation phase is completed
    if !state.is_phase_completed(WorkflowPhase::Validation) {
        eprintln!("{}", "⚠️  Validation phase not completed!".yellow().bold());
        eprintln!();
        eprintln!("Run {} first to validate your project.", "batuta validate".cyan());
        eprintln!();
        crate::cli::workflow::display_workflow_progress(&state);
        anyhow::bail!("Prerequisite phase not completed: validation");
    }

    // Start deployment phase
    state.start_phase(WorkflowPhase::Deployment);
    state.save(&state_file)?;

    // Display build settings
    println!("{}", "Build Settings:".bright_yellow().bold());
    println!(
        "  {} Build mode: {}",
        "•".bright_blue(),
        if release { "release".green() } else { "debug".dimmed() }
    );
    if let Some(t) = &target {
        println!("  {} Target: {}", "•".bright_blue(), t.cyan());
    }
    println!(
        "  {} WebAssembly: {}",
        "•".bright_blue(),
        if wasm { "enabled".green() } else { "disabled".dimmed() }
    );
    println!();

    // Load project config to find the transpiled output directory
    let config_path = PathBuf::from("batuta.toml");
    let config = if config_path.exists() {
        BatutaConfig::load(&config_path)?
    } else {
        BatutaConfig::default()
    };

    let output_dir = &config.transpilation.output_dir;
    if !output_dir.join("Cargo.toml").exists() {
        println!("{} No Cargo.toml found in {}", "✗".red(), output_dir.display());
        println!();
        println!("Run {} first to generate the Rust project.", "batuta transpile".cyan());
        state.fail_phase(
            WorkflowPhase::Deployment,
            format!("No Cargo.toml in {}", output_dir.display()),
        );
        state.save(&state_file)?;
        anyhow::bail!("No Cargo.toml in transpiled output directory: {}", output_dir.display());
    }

    println!("  {} Project: {}", "•".bright_blue(), output_dir.display());
    println!();

    // Execute cargo build in the transpiled project
    match run_cargo_build(output_dir, release, target.as_deref(), wasm, &config.build.cargo_flags) {
        Ok(()) => {
            state.complete_phase(WorkflowPhase::Deployment);
            state.save(&state_file)?;
        }
        Err(e) => {
            state.fail_phase(WorkflowPhase::Deployment, e.to_string());
            state.save(&state_file)?;
            return Err(e);
        }
    }

    // Display workflow progress
    crate::cli::workflow::display_workflow_progress(&state);

    println!("{}", "🎉 Migration Complete!".bright_green().bold());
    println!();
    println!("{}", "💡 Next Steps:".bright_yellow().bold());
    println!(
        "  {} Run {} to generate migration report",
        "1.".bright_blue(),
        "batuta report".cyan()
    );
    println!("  {} Check your output directory for the final binary", "2.".bright_blue());
    println!("  {} Run {} to start fresh", "3.".bright_blue(), "batuta reset".cyan());
    println!();

    Ok(())
}

pub fn cmd_report(output: PathBuf, format: ReportFormat) -> anyhow::Result<()> {
    println!("{}", "📊 Generating migration report...".bright_cyan().bold());
    println!();

    // Load workflow state
    let state_file = crate::cli::get_state_file_path();
    let state = WorkflowState::load(&state_file).unwrap_or_else(|e| {
        warn!("Failed to load workflow state, starting fresh: {e}");
        WorkflowState::new()
    });

    // Check if any work has been done
    let has_started = state.phases.values().any(|info| info.status != PhaseStatus::NotStarted);
    if !has_started {
        eprintln!("{}", "⚠️  No workflow data found!".yellow().bold());
        eprintln!();
        eprintln!("Run {} first to generate analysis data.", "batuta analyze".cyan());
        eprintln!();
        anyhow::bail!("No workflow data found — run `batuta analyze` first");
    }

    // Load or create analysis
    let config_path = PathBuf::from("batuta.toml");
    let analysis = if config_path.exists() {
        let config = BatutaConfig::load(&config_path)?;
        crate::analyzer::analyze_project(&config.source.path, true, true, true)?
    } else {
        // Use current directory if no config
        crate::analyzer::analyze_project(&PathBuf::from("."), true, true, true)?
    };

    // Create report
    let project_name =
        analysis.root_path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown").to_string();

    let migration_report = report::MigrationReport::new(project_name, analysis, state);

    // Convert format enum
    let report_format = match format {
        ReportFormat::Html => report::ReportFormat::Html,
        ReportFormat::Markdown => report::ReportFormat::Markdown,
        ReportFormat::Json => report::ReportFormat::Json,
        ReportFormat::Text => report::ReportFormat::Text,
    };

    // Save report
    migration_report.save(&output, report_format)?;

    println!("{}", "✅ Report generated successfully!".bright_green().bold());
    println!();
    println!("{}: {:?}", "Output file".bold(), output);
    println!("{}: {:?}", "Format".bold(), format);
    println!();

    // Show preview for text-based formats
    if matches!(format, ReportFormat::Text | ReportFormat::Markdown) {
        println!("{}", "Preview (first 20 lines):".dimmed());
        println!("{}", "─".repeat(80).dimmed());
        let content = std::fs::read_to_string(&output)?;
        for line in content.lines().take(20) {
            println!("{}", line.dimmed());
        }
        if content.lines().count() > 20 {
            println!("{}", "...".dimmed());
        }
        println!("{}", "─".repeat(80).dimmed());
        println!();
    }

    println!("{}", "💡 Next Steps:".bright_green().bold());
    println!("  {} Open the report to view detailed analysis", "1.".bright_blue());
    if matches!(format, ReportFormat::Html) {
        println!(
            "  {} Open in browser: file://{}",
            "2.".bright_blue(),
            output.canonicalize()?.display()
        );
    }
    println!();

    Ok(())
}
