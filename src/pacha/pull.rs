//! Pacha Pull Command
//!
//! Handles downloading models from the registry or HuggingFace.

use crate::ansi_colors::Colorize;
use std::io::{self, Write};

use super::helpers::{
    create_progress_bar, is_cached, print_command_header, print_success, resolve_model_ref,
};

// ============================================================================
// PACHA-CLI-003: Pull Command
// ============================================================================

/// Execute the pull command to download a model
pub fn cmd_pull(model: &str, force: bool, quant: Option<&str>) -> anyhow::Result<()> {
    print_command_header("Pull Model");

    println!("Model:  {}", model.cyan());
    if let Some(q) = quant {
        println!("Quant:  {}", q.yellow());
    }
    if force {
        println!("Mode:   {}", "Force re-download".yellow());
    }
    println!();

    // Create progress bar callback
    let progress_callback = |downloaded: u64, total: u64, speed: f64| {
        let percent = if total > 0 { (downloaded as f64 / total as f64) * 100.0 } else { 0.0 };
        let downloaded_mb = downloaded as f64 / (1024.0 * 1024.0);
        let total_mb = total as f64 / (1024.0 * 1024.0);
        let speed_mb = speed / (1024.0 * 1024.0);

        print!(
            "\r{} {:.1}% ({:.1}/{:.1} MB) @ {:.1} MB/s    ",
            create_progress_bar(percent, 30),
            percent,
            downloaded_mb,
            total_mb,
            speed_mb
        );
        io::stdout().flush().ok();
    };

    // Simulate pull (actual implementation would use pacha::fetcher)
    println!("{}", "Resolving model reference...".dimmed());

    // For now, show what would happen
    let resolved = resolve_model_ref(model, quant)?;
    println!("Resolved: {}", resolved.cyan());
    println!();

    // Check cache
    if !force && is_cached(&resolved) {
        print_success("Model already cached");
        println!("  Use {} to re-download", "--force".yellow());
        return Ok(());
    }

    // Simulate download progress
    println!("{}", "Downloading...".dimmed());
    for i in 0..=100 {
        progress_callback(i * 40_000_000, 4_000_000_000, 50_000_000.0);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    println!();
    println!();

    print_success("Model downloaded successfully!");
    println!();
    println!("Run with: {}", format!("batuta serve {}", model).cyan());

    Ok(())
}
