//! Output formatting utilities
//!
//! Toyota Way: Visualization - Make information visible and clear.

use colored::Colorize;

/// APR format magic bytes
pub(crate) const MAGIC_APRN: [u8; 4] = [0x41, 0x50, 0x52, 0x4E]; // "APRN" - aprender
pub(crate) const MAGIC_APR1: [u8; 4] = [0x41, 0x50, 0x52, 0x31]; // "APR1" - whisper.apr

/// Check if magic bytes are valid (supports both APRN and APR1)
pub(crate) fn is_valid_magic(magic: &[u8]) -> bool {
    magic.len() >= 4 && (magic[..4] == MAGIC_APRN || magic[..4] == MAGIC_APR1)
}

/// Get format name from magic bytes
pub(crate) fn format_name(magic: &[u8]) -> &'static str {
    if magic.len() >= 4 {
        if magic[..4] == MAGIC_APRN {
            return "APRN (aprender)";
        }
        if magic[..4] == MAGIC_APR1 {
            return "APR1 (whisper.apr)";
        }
    }
    "Unknown"
}

/// Print a section header
pub(crate) fn section(title: &str) {
    println!("\n{}", format!("=== {title} ===").cyan().bold());
}

/// Print a key-value pair
pub(crate) fn kv(key: &str, value: impl std::fmt::Display) {
    println!("  {}: {}", key.white().bold(), value);
}

/// Print a success message
pub(crate) fn success(msg: &str) {
    println!("{} {}", "[PASS]".green().bold(), msg);
}

/// Print a warning message
pub(crate) fn warning(msg: &str) {
    println!("{} {}", "[WARN]".yellow().bold(), msg);
}

/// Print a failure message
pub(crate) fn fail(msg: &str) {
    println!("{} {}", "[FAIL]".red().bold(), msg);
}

/// Print an info message
#[allow(dead_code)]
pub(crate) fn info(msg: &str) {
    println!("{} {}", "[INFO]".blue(), msg);
}

/// Format bytes as human-readable size
pub(crate) fn format_size(bytes: u64) -> String {
    humansize::format_size(bytes, humansize::BINARY)
}
