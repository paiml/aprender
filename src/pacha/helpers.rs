//! Pacha Helper Functions
//!
//! Utility functions for the Pacha model registry CLI including
//! model resolution, caching, and display formatting.

use crate::ansi_colors::Colorize;

// ============================================================================
// Display Helpers
// ============================================================================

/// Print a styled command header
pub fn print_command_header(title: &str) {
    println!("{}", title.bright_cyan().bold());
    println!("{}", "=".repeat(60).dimmed());
    println!();
}

/// Print a success message with checkmark
pub fn print_success(message: impl std::fmt::Display) {
    println!("{} {}", "ok".bright_green().bold(), message);
}

// ============================================================================
// Model Resolution
// ============================================================================

/// Resolve a model reference to full URI
pub fn resolve_model_ref(model: &str, quant: Option<&str>) -> anyhow::Result<String> {
    // Check for known aliases
    let aliases = [
        ("llama3", "hf://meta-llama/Meta-Llama-3-8B-Instruct-GGUF"),
        ("llama3:8b", "hf://meta-llama/Meta-Llama-3-8B-Instruct-GGUF"),
        ("llama3:70b", "hf://meta-llama/Meta-Llama-3-70B-Instruct-GGUF"),
        ("mistral", "hf://mistralai/Mistral-7B-Instruct-v0.2-GGUF"),
        ("mixtral", "hf://mistralai/Mixtral-8x7B-Instruct-v0.1-GGUF"),
        ("phi3", "hf://microsoft/Phi-3-mini-4k-instruct-gguf"),
    ];

    for (alias, target) in &aliases {
        if model == *alias {
            let mut result = (*target).to_string();
            if let Some(q) = quant {
                result = format!("{}:{}", result, q.to_uppercase());
            }
            return Ok(result);
        }
    }

    // If already a full URI, return as-is
    if model.contains("://") {
        return Ok(model.to_string());
    }

    // Otherwise, assume pacha:// scheme
    Ok(format!("pacha://{}", model))
}

// ============================================================================
// Cache Operations
// ============================================================================

/// Check if a model is cached (simulation)
pub fn is_cached(model: &str) -> bool {
    // GH-47: model parameter not yet used — always returns false (simulation stub)
    let _ = model;
    false
}

/// Get cached models (simulation)
pub fn get_cached_models() -> Vec<(String, u64, String)> {
    vec![
        ("llama3:8b-q4_k_m".to_string(), 4_690_000_000, "2 hours ago".to_string()),
        ("mistral:7b-q4_k_m".to_string(), 4_110_000_000, "1 day ago".to_string()),
        ("phi3:mini-q4_k_m".to_string(), 2_390_000_000, "3 days ago".to_string()),
    ]
}

// ============================================================================
// Formatting
// ============================================================================

/// Format size in human-readable form
pub fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Create a progress bar string
pub fn create_progress_bar(percent: f64, width: usize) -> String {
    let filled = (percent / 100.0 * width as f64) as usize;
    let empty = width.saturating_sub(filled);

    format!(
        "{}[{}{}]{}",
        "\x1b[36m", // Cyan
        "=".repeat(filled),
        " ".repeat(empty),
        "\x1b[0m" // Reset
    )
}

/// Truncate string with ellipsis (delegates to batuta-common).
pub fn truncate_str(s: &str, max_len: usize) -> String {
    batuta_common::display::truncate_str(s, max_len)
}
