//! Pipeline command handlers.
//!
//! Split into submodules for QA-002 compliance (≤500 lines per file).

#[path = "pipeline_cmds_build.rs"]
mod build;
#[path = "pipeline_cmds_optimize.rs"]
mod optimize;
#[path = "pipeline_cmds_transpile.rs"]
mod transpile;
#[path = "pipeline_cmds_validate.rs"]
mod validate;

pub use build::{cmd_build, cmd_report};
pub use optimize::cmd_optimize;
pub use transpile::cmd_transpile;
pub use validate::cmd_validate;

use crate::analyzer::analyze_project;
use crate::ansi_colors::Colorize;
use crate::config::BatutaConfig;
use crate::types::{ProjectAnalysis, WorkflowPhase, WorkflowState};
use std::path::PathBuf;
use tracing::warn;

/// CLI report format
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum ReportFormat {
    /// HTML report with charts
    #[default]
    Html,
    /// Markdown report
    Markdown,
    /// JSON data
    Json,
    /// Plain text report
    Text,
}

/// CLI optimization profile
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum OptimizationProfile {
    /// Fast compilation, basic optimizations
    Fast,
    /// Balanced compilation and performance
    #[default]
    Balanced,
    /// Maximum performance, slower compilation
    Aggressive,
}

// ============================================================================
// Init Command
// ============================================================================

pub fn cmd_init(source: PathBuf, output: Option<PathBuf>) -> anyhow::Result<()> {
    println!("{}", "🚀 Initializing Batuta project...".bright_cyan().bold());
    println!();

    // Analyze the source project
    println!("{}", "Analyzing source project...".dimmed());
    let analysis = analyze_project(&source, true, true, true)?;

    println!("{} Source: {:?}", "✓".bright_green(), source);
    if let Some(lang) = &analysis.primary_language {
        println!("{} Detected language: {}", "✓".bright_green(), format!("{}", lang).cyan());
    }
    println!();

    // Determine output directory
    let output_dir = output.unwrap_or_else(|| {
        let mut dir = source.clone();
        dir.push("rust-output");
        dir
    });

    // Create configuration from analysis
    let mut config = BatutaConfig::from_analysis(&analysis);

    // Set output directory
    config.transpilation.output_dir = output_dir.clone();

    // Save configuration
    let config_path = source.join("batuta.toml");
    config.save(&config_path)?;

    println!("{} Created configuration: {:?}", "✓".bright_green(), config_path);

    // Create output directory structure
    std::fs::create_dir_all(&output_dir)?;
    std::fs::create_dir_all(output_dir.join("src"))?;

    println!("{} Created output directory: {:?}", "✓".bright_green(), output_dir);
    println!();

    // Display configuration summary
    display_init_summary(&config, &analysis);

    Ok(())
}

fn display_init_summary(config: &BatutaConfig, analysis: &ProjectAnalysis) {
    println!("{}", "📋 Configuration Summary".bright_yellow().bold());
    println!("{}", "=".repeat(50));
    println!();
    println!("{}: {}", "Project name".bold(), config.project.name.cyan());
    println!(
        "{}: {}",
        "Primary language".bold(),
        config.project.primary_language.as_ref().unwrap_or(&"Unknown".to_string()).cyan()
    );
    println!("{}: {:?}", "Output directory".bold(), config.transpilation.output_dir);
    println!();

    // Display transpilation settings
    println!("{}", "Transpilation:".bright_yellow());
    println!(
        "  {} Incremental: {}",
        "•".bright_blue(),
        config.transpilation.incremental.to_string().cyan()
    );
    println!("  {} Caching: {}", "•".bright_blue(), config.transpilation.cache.to_string().cyan());

    if analysis.has_ml_dependencies() {
        println!("  {} NumPy → Trueno: {}", "•".bright_blue(), "enabled".green());
        println!("  {} sklearn → Aprender: {}", "•".bright_blue(), "enabled".green());
        println!("  {} PyTorch → Realizar: {}", "•".bright_blue(), "enabled".green());
    }
    println!();

    // Display optimization settings
    println!("{}", "Optimization:".bright_yellow());
    println!("  {} Profile: {}", "•".bright_blue(), config.optimization.profile.cyan());
    println!(
        "  {} SIMD: {}",
        "•".bright_blue(),
        config.optimization.enable_simd.to_string().cyan()
    );
    println!(
        "  {} GPU: {}",
        "•".bright_blue(),
        if config.optimization.enable_gpu { "enabled".green() } else { "disabled".dimmed() }
    );
    println!();

    // Next steps
    println!("{}", "💡 Next Steps:".bright_green().bold());
    println!("  {} Edit {} to customize settings", "1.".bright_blue(), "batuta.toml".cyan());
    println!("  {} Run {} to convert your code", "2.".bright_blue(), "batuta transpile".cyan());
    println!("  {} Run {} to optimize performance", "3.".bright_blue(), "batuta optimize".cyan());
    println!();
}

// ============================================================================
// Analyze Command
// ============================================================================

pub fn cmd_analyze(
    path: PathBuf,
    tdg: bool,
    languages: bool,
    dependencies: bool,
) -> anyhow::Result<()> {
    contract_pre_analyze!(path);
    println!("{}", "🔍 Analyzing project...".bright_cyan().bold());
    println!();

    let state_file = super::get_state_file_path();
    let mut state = WorkflowState::load(&state_file).unwrap_or_else(|e| {
        warn!("Failed to load workflow state, starting fresh: {e}");
        WorkflowState::new()
    });

    state.start_phase(WorkflowPhase::Analysis);
    state.save(&state_file)?;

    let analysis = analyze_project(&path, tdg, languages, dependencies)?;

    // Display results based on flags
    display_analysis_results(&analysis);

    // Update and save workflow state
    state.complete_phase(WorkflowPhase::Analysis);

    // Create a default config if not exists
    let config_path = path.join("batuta.toml");
    if !config_path.exists() {
        let config = BatutaConfig::from_analysis(&analysis);
        config.save(&config_path)?;
        println!("{} Created default configuration: {:?}", "✓".bright_green(), config_path);
        println!();
    }

    state.save(&state_file)?;

    super::workflow::display_workflow_progress(&state);
    display_analyze_next_steps();

    Ok(())
}

/// Display project analysis results
pub fn display_analysis_results(analysis: &ProjectAnalysis) {
    println!("{}", "📊 Analysis Results".bright_green().bold());
    println!("{}", "=".repeat(50));
    println!();

    // File statistics
    println!("{}", "Files:".bright_yellow());
    println!("  {} Total files: {}", "•".bright_blue(), analysis.total_files.to_string().cyan());
    println!("  {} Total lines: {}", "•".bright_blue(), analysis.total_lines.to_string().cyan());
    println!();

    // Language information
    display_language_info(analysis);

    // Dependency information
    display_dependency_info(analysis);

    // TDG score
    display_tdg_score(analysis);
}

/// Display language detection results
fn display_language_info(analysis: &ProjectAnalysis) {
    if let Some(lang) = &analysis.primary_language {
        println!("{}", "Language Detection:".bright_yellow());
        println!("  {} Primary: {}", "•".bright_blue(), format!("{}", lang).cyan());

        if !analysis.languages.is_empty() {
            println!("  {} Breakdown:", "•".bright_blue());
            for stats in &analysis.languages {
                println!(
                    "    {} {}: {} files ({:.1}%)",
                    "·".dimmed(),
                    format!("{}", stats.language).cyan(),
                    stats.file_count,
                    stats.percentage
                );
            }
        }
        println!();
    }
}

/// Display dependency detection results
fn display_dependency_info(analysis: &ProjectAnalysis) {
    if !analysis.dependencies.is_empty() {
        println!("{}", "Dependencies:".bright_yellow());
        for dep in &analysis.dependencies {
            println!(
                "  {} {} ({})",
                "•".bright_blue(),
                format!("{:?}", dep.manager).cyan(),
                dep.file_path.display()
            );
        }
        println!();
    }

    if analysis.has_ml_dependencies() {
        println!("{}", "ML Stack Detection:".bright_yellow());
        println!(
            "  {} {}",
            "•".bright_blue(),
            "ML dependencies detected — Sovereign AI Stack converters available".green()
        );
        println!("    NumPy → Trueno (SIMD)");
        println!("    scikit-learn → Aprender (ML)");
        println!("    PyTorch → Realizar (Inference)");
        println!();
    }
}

/// Display TDG quality score
fn display_tdg_score(analysis: &ProjectAnalysis) {
    if let Some(tdg) = analysis.tdg_score {
        println!("{}", "Quality (TDG Score):".bright_yellow());
        let grade = if tdg >= 90.0 {
            "A+".green()
        } else if tdg >= 80.0 {
            "A".green()
        } else if tdg >= 70.0 {
            "B".cyan()
        } else if tdg >= 60.0 {
            "C".yellow()
        } else {
            "D".red()
        };
        println!("  {} Score: {:.1} (Grade: {})", "•".bright_blue(), tdg, grade);
        println!();
    }
}

fn display_analyze_next_steps() {
    println!("{}", "💡 Next Steps:".bright_green().bold());
    println!("  {} Run {} to convert your code", "1.".bright_blue(), "batuta transpile".cyan());
    println!(
        "  {} Run {} for detailed dependency analysis",
        "2.".bright_blue(),
        "batuta analyze --tdg".cyan()
    );
    println!();
}
