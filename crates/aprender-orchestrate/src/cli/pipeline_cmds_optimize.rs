//! Optimize command handler and helpers.
//!
//! Extracted from pipeline_cmds.rs for QA-002 compliance.

use crate::ansi_colors::Colorize;
use crate::config::BatutaConfig;
use crate::types::{WorkflowPhase, WorkflowState};
use std::path::{Path, PathBuf};
use tracing::warn;

use super::OptimizationProfile;

/// A detected compute pattern in transpiled source code.
pub(super) struct ComputePattern {
    pub file: String,
    pub kind: crate::backend::OpComplexity,
    pub description: String,
}

/// Scan transpiled Rust files for compute-intensive patterns.
fn scan_optimization_targets(output_dir: &Path) -> Vec<ComputePattern> {
    use crate::backend::OpComplexity;

    let mut patterns = Vec::new();
    let rs_files = collect_rs_files(output_dir);

    for path in &rs_files {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let file = path.strip_prefix(output_dir).unwrap_or(path).display().to_string();

        // High complexity: matrix operations
        for kw in &["matmul", "matrix_multiply", "gemm", "dot_product", "convolution"] {
            if content.contains(kw) {
                patterns.push(ComputePattern {
                    file: file.clone(),
                    kind: OpComplexity::High,
                    description: format!("matrix/convolution op: {}", kw),
                });
            }
        }

        // Medium complexity: reductions and aggregations
        for kw in &[".sum()", ".product()", ".fold(", "reduce(", ".norm("] {
            if content.contains(kw) {
                patterns.push(ComputePattern {
                    file: file.clone(),
                    kind: OpComplexity::Medium,
                    description: format!("reduction op: {}", kw.trim_matches('.')),
                });
            }
        }

        // Low complexity: element-wise operations in loops
        if content.contains(".iter()") && (content.contains(".map(") || content.contains(".zip(")) {
            patterns.push(ComputePattern {
                file: file.clone(),
                kind: OpComplexity::Low,
                description: "element-wise iter/map/zip pattern".to_string(),
            });
        }
    }

    patterns
}

/// Collect all .rs files under a directory.
fn collect_rs_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(collect_rs_files(&path));
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                files.push(path);
            }
        }
    }
    files
}

/// Run MoE backend analysis on detected compute patterns.
fn run_moe_analysis(
    enable_gpu: bool,
    enable_simd: bool,
    gpu_threshold: usize,
    patterns: &[ComputePattern],
) -> Vec<String> {
    use crate::pipeline::OptimizationStage;

    let stage = OptimizationStage::new(enable_gpu, enable_simd, gpu_threshold);
    let mut recommendations = stage.analyze_optimizations();

    // Add per-file recommendations based on detected patterns
    for pat in patterns {
        let backend = stage.backend_selector.select_with_moe(pat.kind, 10_000);
        recommendations.push(format!("{}: {} → {} backend", pat.file, pat.description, backend));
    }

    recommendations
}

/// Apply profile-specific [profile.release] settings to Cargo.toml.
fn apply_profile_optimizations(
    cargo_toml: &Path,
    profile: OptimizationProfile,
) -> anyhow::Result<Vec<String>> {
    let content = std::fs::read_to_string(cargo_toml)?;
    let mut applied = Vec::new();

    let (opt_level, lto, codegen_units, strip) = match profile {
        OptimizationProfile::Fast => ("2", "false", "16", "none"),
        OptimizationProfile::Balanced => ("3", "thin", "4", "none"),
        OptimizationProfile::Aggressive => ("3", "true", "1", "symbols"),
    };

    // Only append profile section if not already present
    if content.contains("[profile.release]") {
        applied.push(format!(
            "[profile.release] already exists — manual review recommended (profile: {:?})",
            profile
        ));
        return Ok(applied);
    }

    let section = format!(
        "\n[profile.release]\nopt-level = \"{}\"\nlto = \"{}\"\ncodegen-units = {}\nstrip = \"{}\"\n",
        opt_level, lto, codegen_units, strip
    );
    let mut new_content = content;
    new_content.push_str(&section);
    std::fs::write(cargo_toml, new_content)?;

    applied.push(format!("opt-level = \"{}\"", opt_level));
    applied.push(format!("lto = \"{}\"", lto));
    applied.push(format!("codegen-units = {}", codegen_units));
    applied.push(format!("strip = \"{}\"", strip));

    Ok(applied)
}

pub fn cmd_optimize(
    enable_gpu: bool,
    enable_simd: bool,
    profile: OptimizationProfile,
    gpu_threshold: usize,
) -> anyhow::Result<()> {
    println!("{}", "⚡ Optimizing code...".bright_cyan().bold());
    println!();

    // Load workflow state
    let state_file = crate::cli::get_state_file_path();
    let mut state = WorkflowState::load(&state_file).unwrap_or_else(|e| {
        warn!("Failed to load workflow state, starting fresh: {e}");
        WorkflowState::new()
    });

    // Check if transpilation phase is completed
    if !state.is_phase_completed(WorkflowPhase::Transpilation) {
        eprintln!("{}", "⚠️  Transpilation phase not completed!".yellow().bold());
        eprintln!();
        eprintln!("Run {} first to transpile your project.", "batuta transpile".cyan());
        eprintln!();
        crate::cli::workflow::display_workflow_progress(&state);
        anyhow::bail!("Prerequisite phase not completed: transpilation");
    }

    // Start optimization phase
    state.start_phase(WorkflowPhase::Optimization);
    state.save(&state_file)?;

    // Display optimization settings
    println!("{}", "Optimization Settings:".bright_yellow().bold());
    println!("  {} Profile: {:?}", "•".bright_blue(), profile);
    println!(
        "  {} SIMD vectorization: {}",
        "•".bright_blue(),
        if enable_simd { "enabled".green() } else { "disabled".dimmed() }
    );
    println!(
        "  {} GPU acceleration: {}",
        "•".bright_blue(),
        if enable_gpu {
            format!("enabled (threshold: {})", gpu_threshold).green()
        } else {
            "disabled".to_string().dimmed()
        }
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
    if !output_dir.exists() {
        println!("{} Output directory not found: {}", "✗".red(), output_dir.display());
        state.fail_phase(
            WorkflowPhase::Optimization,
            format!("Output directory not found: {}", output_dir.display()),
        );
        state.save(&state_file)?;
        anyhow::bail!("Transpiled output directory not found: {}", output_dir.display());
    }

    // Scan transpiled source and run MoE analysis
    let patterns = scan_optimization_targets(output_dir);
    let recommendations = run_moe_analysis(enable_gpu, enable_simd, gpu_threshold, &patterns);

    // Display MoE recommendations
    println!("{}", "MoE Backend Analysis:".bright_yellow().bold());
    if recommendations.is_empty() {
        println!("  {} No compute-intensive patterns detected", "•".dimmed());
    } else {
        for rec in &recommendations {
            println!("  {} {}", "→".bright_blue(), rec);
        }
    }
    println!();

    // Apply profile-specific Cargo.toml optimizations
    let cargo_toml = output_dir.join("Cargo.toml");
    if cargo_toml.exists() {
        let applied = apply_profile_optimizations(&cargo_toml, profile)?;
        println!("{}", "Cargo Profile Optimizations:".bright_yellow().bold());
        for opt in &applied {
            println!("  {} {}", "✓".bright_green(), opt);
        }
        println!();
    }

    // Summary
    println!(
        "{} Analyzed {} source patterns, generated {} recommendations",
        "✅".bright_green(),
        patterns.len(),
        recommendations.len()
    );

    state.complete_phase(WorkflowPhase::Optimization);
    state.save(&state_file)?;

    // Display workflow progress
    crate::cli::workflow::display_workflow_progress(&state);

    println!("{}", "💡 Next Steps:".bright_green().bold());
    println!("  {} Run {} to verify equivalence", "1.".bright_blue(), "batuta validate".cyan());
    println!(
        "  {} Run {} to build final binary",
        "2.".bright_blue(),
        "batuta build --release".cyan()
    );
    println!();

    Ok(())
}
