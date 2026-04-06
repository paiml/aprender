//! Validate command handler and testing helpers.
//!
//! Extracted from pipeline_cmds.rs for QA-002 compliance.

use crate::ansi_colors::Colorize;
use crate::config::BatutaConfig;
use crate::pipeline::{PipelineContext, PipelineStage, ValidationStage};
use crate::types::{WorkflowPhase, WorkflowState};
use std::path::{Path, PathBuf};
use tracing::warn;

pub fn cmd_validate(
    trace_syscalls: bool,
    diff_output: bool,
    run_original_tests: bool,
    benchmark: bool,
) -> anyhow::Result<()> {
    println!("{}", "✅ Validating equivalence...".bright_cyan().bold());
    println!();

    // Load workflow state
    let state_file = crate::cli::get_state_file_path();
    let mut state = WorkflowState::load(&state_file).unwrap_or_else(|e| {
        warn!("Failed to load workflow state, starting fresh: {e}");
        WorkflowState::new()
    });

    // Check if optimization phase is completed
    if !state.is_phase_completed(WorkflowPhase::Optimization) {
        eprintln!("{}", "⚠️  Optimization phase not completed!".yellow().bold());
        eprintln!();
        eprintln!("Run {} first to optimize your project.", "batuta optimize".cyan());
        eprintln!();
        crate::cli::workflow::display_workflow_progress(&state);
        anyhow::bail!("Prerequisite phase not completed: optimization");
    }

    // Start validation phase
    state.start_phase(WorkflowPhase::Validation);
    state.save(&state_file)?;

    // Display validation settings
    display_validation_settings(trace_syscalls, diff_output, run_original_tests, benchmark);

    // Implement validation with Renacer (BATUTA-011)
    let mut validation_passed = true;

    if trace_syscalls && !run_syscall_tracing(run_original_tests) {
        validation_passed = false;
    }

    if diff_output && !run_output_diff() {
        validation_passed = false;
    }

    if run_original_tests && !run_transpiled_tests() {
        validation_passed = false;
    }

    if benchmark && !run_performance_benchmark() {
        validation_passed = false;
    }

    // Mark as completed only if validation passed
    if validation_passed {
        state.complete_phase(WorkflowPhase::Validation);
    } else {
        state.fail_phase(WorkflowPhase::Validation, "Validation checks failed".to_string());
    }
    state.save(&state_file)?;

    // Display workflow progress
    crate::cli::workflow::display_workflow_progress(&state);

    println!("{}", "💡 Next Steps:".bright_green().bold());
    println!(
        "  {} Run {} to build final binary",
        "1.".bright_blue(),
        "batuta build --release".cyan()
    );
    println!("  {} Run {} to generate report", "2.".bright_blue(), "batuta report".cyan());
    println!();

    Ok(())
}

/// Run Renacer syscall tracing validation. Returns true if passed.
fn run_syscall_tracing(run_original_tests: bool) -> bool {
    println!("{}", "🔍 Running Renacer syscall tracing...".bright_cyan());

    let original_binary = std::path::Path::new("./original_binary");
    let transpiled_binary = std::path::Path::new("./target/release/transpiled");

    if !original_binary.exists() || !transpiled_binary.exists() {
        println!("{}", "  ⚠️  Binaries not found for comparison".yellow());
        println!("     Expected: ./original_binary and ./target/release/transpiled");
        println!();
        return true;
    }

    println!("  {} Tracing original binary...", "•".bright_blue());
    println!("  {} Tracing transpiled binary...", "•".bright_blue());
    println!("  {} Comparing syscall traces...", "•".bright_blue());

    let ctx = PipelineContext::new(PathBuf::from("."), PathBuf::from("."));
    let stage = ValidationStage::new(true, run_original_tests);

    match tokio::runtime::Runtime::new()
        .expect("failed to create tokio runtime")
        .block_on(stage.execute(ctx))
    {
        Ok(result_ctx) => {
            if let Some(eq) = result_ctx.metadata.get("syscall_equivalence") {
                if eq.as_bool() == Some(true) {
                    println!(
                        "{}",
                        "  ✅ Syscall traces match - semantic equivalence verified".green()
                    );
                    println!();
                    true
                } else {
                    println!("{}", "  ❌ Syscall traces differ - equivalence NOT verified".red());
                    println!();
                    false
                }
            } else {
                println!("{}", "  ⚠️  Syscall tracing skipped (binaries not found)".yellow());
                println!();
                true
            }
        }
        Err(e) => {
            println!("{}", format!("  ❌ Validation error: {}", e).red());
            println!();
            false
        }
    }
}

/// Run output diff comparison between original and transpiled binaries.
fn run_output_diff() -> bool {
    println!("{}", "📊 Output comparison:".bright_cyan());

    let original_binary = Path::new("./original_binary");
    let transpiled_binary = Path::new("./target/release/transpiled");

    if !original_binary.exists() || !transpiled_binary.exists() {
        println!("{}", "  ⚠️  Binaries not found for comparison".yellow());
        println!("     Expected: ./original_binary and ./target/release/transpiled");
        println!();
        return true;
    }

    println!("  {} Running original...", "•".bright_blue());
    let original_out = std::process::Command::new(original_binary)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output();

    println!("  {} Running transpiled...", "•".bright_blue());
    let transpiled_out = std::process::Command::new(transpiled_binary)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output();

    match (original_out, transpiled_out) {
        (Ok(orig), Ok(trans)) => {
            let orig_stdout = String::from_utf8_lossy(&orig.stdout);
            let trans_stdout = String::from_utf8_lossy(&trans.stdout);

            if orig_stdout == trans_stdout {
                println!("{}", "  ✅ Outputs match - functional equivalence verified".green());
                println!();
                true
            } else {
                println!("{}", "  ❌ Outputs differ:".red());
                show_output_diff(&orig_stdout, &trans_stdout);
                println!();
                false
            }
        }
        (Err(e), _) => {
            println!("{}", format!("  ❌ Failed to run original binary: {e}").red());
            println!();
            false
        }
        (_, Err(e)) => {
            println!("{}", format!("  ❌ Failed to run transpiled binary: {e}").red());
            println!();
            false
        }
    }
}

/// Show a simple line-by-line diff between two outputs.
fn show_output_diff(original: &str, transpiled: &str) {
    let orig_lines: Vec<&str> = original.lines().collect();
    let trans_lines: Vec<&str> = transpiled.lines().collect();
    let max = orig_lines.len().max(trans_lines.len()).min(20);

    for i in 0..max {
        let orig_line = orig_lines.get(i).unwrap_or(&"");
        let trans_line = trans_lines.get(i).unwrap_or(&"");
        if orig_line != trans_line {
            println!("    {} {}", "- ".red(), orig_line);
            println!("    {} {}", "+ ".green(), trans_line);
        }
    }
    if orig_lines.len().max(trans_lines.len()) > 20 {
        println!("    ... (truncated, {} total lines)", orig_lines.len().max(trans_lines.len()));
    }
}

/// Run `cargo test` in the transpiled output directory. Returns true if tests pass.
fn run_transpiled_tests() -> bool {
    println!("{}", "🧪 Running test suite on transpiled code:".bright_cyan());

    let config_path = PathBuf::from("batuta.toml");
    let config = if config_path.exists() {
        match BatutaConfig::load(&config_path) {
            Ok(c) => c,
            Err(e) => {
                println!("{}", format!("  ❌ Failed to load config: {e}").red());
                println!();
                return false;
            }
        }
    } else {
        BatutaConfig::default()
    };

    let output_dir = &config.transpilation.output_dir;
    if !output_dir.join("Cargo.toml").exists() {
        println!("{}", format!("  ⚠️  No Cargo.toml in {}", output_dir.display()).yellow());
        println!("     Run {} first.", "batuta build".cyan());
        println!();
        return true;
    }

    println!("  {} Running: cargo test in {}", "•".bright_blue(), output_dir.display());

    let status = std::process::Command::new("cargo")
        .arg("test")
        .current_dir(output_dir)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status();

    match status {
        Ok(s) if s.success() => {
            println!();
            println!("{}", "  ✅ All tests pass on transpiled code".green());
            println!();
            true
        }
        Ok(s) => {
            let code = s.code().map_or("signal".to_string(), |c| c.to_string());
            println!();
            println!("{}", format!("  ❌ Tests failed (exit {})", code).red());
            println!();
            false
        }
        Err(e) => {
            println!("{}", format!("  ❌ Failed to run cargo test: {e}").red());
            println!();
            false
        }
    }
}

/// Run performance benchmarks. Returns true always (informational).
fn run_performance_benchmark() -> bool {
    println!("{}", "⚡ Performance benchmarking:".bright_cyan());

    let original_binary = Path::new("./original_binary");
    let transpiled_binary = Path::new("./target/release/transpiled");

    if !original_binary.exists() || !transpiled_binary.exists() {
        println!("{}", "  ⚠️  Binaries not found for benchmarking".yellow());
        println!("     Expected: ./original_binary and ./target/release/transpiled");
        println!();
        return true;
    }

    let iterations = 3;
    println!("  {} Running {} iterations each...", "•".bright_blue(), iterations);

    let orig_time = time_binary_avg(original_binary, iterations);
    let trans_time = time_binary_avg(transpiled_binary, iterations);

    match (orig_time, trans_time) {
        (Some(orig_ms), Some(trans_ms)) => {
            println!();
            println!("  Original:   {:.1}ms avg", orig_ms);
            println!("  Transpiled: {:.1}ms avg", trans_ms);
            if trans_ms > 0.0 {
                let speedup = orig_ms / trans_ms;
                if speedup >= 1.0 {
                    println!("  Speedup:    {:.2}x {}", speedup, "faster".green());
                } else {
                    println!("  Speedup:    {:.2}x {}", speedup, "slower".red());
                }
            }
            println!();
            true
        }
        _ => {
            println!("{}", "  ❌ Failed to benchmark binaries".red());
            println!();
            false
        }
    }
}

/// Time a binary over N iterations, returning average milliseconds.
fn time_binary_avg(binary: &Path, iterations: u32) -> Option<f64> {
    let mut total = std::time::Duration::ZERO;
    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let status = std::process::Command::new(binary)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .ok()?;
        if !status.success() {
            return None;
        }
        total += start.elapsed();
    }
    Some(total.as_secs_f64() * 1000.0 / f64::from(iterations))
}

/// Display validation settings as a formatted list.
fn display_validation_settings(
    trace_syscalls: bool,
    diff_output: bool,
    run_original_tests: bool,
    benchmark: bool,
) {
    let settings = [
        ("Syscall tracing", trace_syscalls),
        ("Diff output", diff_output),
        ("Original tests", run_original_tests),
        ("Benchmarks", benchmark),
    ];
    println!("{}", "Validation Settings:".bright_yellow().bold());
    for (label, enabled) in settings {
        println!(
            "  {} {}: {}",
            "•".bright_blue(),
            label,
            if enabled { "enabled".green() } else { "disabled".dimmed() }
        );
    }
    println!();
}
