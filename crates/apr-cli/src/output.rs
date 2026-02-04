//! Output formatting utilities
//!
//! Toyota Way: Visualization - Make information visible and clear.

use colored::Colorize;

/// APR format magic bytes
pub(crate) const MAGIC_APRN: [u8; 4] = [0x41, 0x50, 0x52, 0x4E]; // "APRN" - aprender v1
pub(crate) const MAGIC_APR1: [u8; 4] = [0x41, 0x50, 0x52, 0x31]; // "APR1" - whisper.apr
pub(crate) const MAGIC_APR2: [u8; 4] = [0x41, 0x50, 0x52, 0x32]; // "APR2" - aprender v2
pub(crate) const MAGIC_APR0: [u8; 4] = [0x41, 0x50, 0x52, 0x00]; // "APR\0" - ONE TRUE APR format (v2)

/// GGUF format magic bytes
pub(crate) const MAGIC_GGUF: [u8; 4] = [0x47, 0x47, 0x55, 0x46]; // "GGUF" - llama.cpp quantized

/// Check if magic bytes are valid (supports APR, GGUF formats)
/// BUG-DEBUG-001 FIX: Now accepts GGUF format magic as valid
pub(crate) fn is_valid_magic(magic: &[u8]) -> bool {
    magic.len() >= 4
        && (magic[..4] == MAGIC_APRN
            || magic[..4] == MAGIC_APR1
            || magic[..4] == MAGIC_APR2
            || magic[..4] == MAGIC_APR0
            || magic[..4] == MAGIC_GGUF)
}

/// Get format name from magic bytes
/// BUG-PROBAR-001 FIX: Now returns correct name for GGUF format
pub(crate) fn format_name(magic: &[u8]) -> &'static str {
    if magic.len() >= 4 {
        if magic[..4] == MAGIC_APRN {
            return "APRN (aprender v1)";
        }
        if magic[..4] == MAGIC_APR1 {
            return "APR1 (whisper.apr)";
        }
        if magic[..4] == MAGIC_APR2 {
            return "APR2 (aprender v2)";
        }
        if magic[..4] == MAGIC_APR0 {
            return "APR v2 (ONE TRUE format)";
        }
        if magic[..4] == MAGIC_GGUF {
            return "GGUF (llama.cpp)";
        }
    }
    "Unknown"
}

/// Print a section header
pub(crate) fn section(title: &str) {
    println!("\n{}", format!("=== {title} ===").cyan().bold());
}

// Note: These functions are used by various commands and may appear unused
// when only some commands are being compiled/used.

/// Print a key-value pair
pub(crate) fn kv(key: &str, value: impl std::fmt::Display) {
    println!("  {}: {}", key.white().bold(), value);
}

/// Print a success message
#[allow(dead_code)]
pub(crate) fn success(msg: &str) {
    println!("{} {}", "[PASS]".green().bold(), msg);
}

/// Print a warning message
#[allow(dead_code)]
pub(crate) fn warning(msg: &str) {
    println!("{} {}", "[WARN]".yellow().bold(), msg);
}

/// Print a failure message
#[allow(dead_code)]
pub(crate) fn fail(msg: &str) {
    println!("{} {}", "[FAIL]".red().bold(), msg);
}

/// Print an info message
#[allow(dead_code)]
pub(crate) fn info(msg: &str) {
    println!("{} {}", "[INFO]".blue(), msg);
}

/// Print an error message
#[allow(dead_code)]
pub(crate) fn error(msg: &str) {
    eprintln!("{} {}", "[ERROR]".red().bold(), msg);
}

/// Print a warning message (alias for backward compatibility)
#[allow(dead_code)]
pub(crate) fn warn(msg: &str) {
    warning(msg);
}

/// Format bytes as human-readable size
pub(crate) fn format_size(bytes: u64) -> String {
    humansize::format_size(bytes, humansize::BINARY)
}
