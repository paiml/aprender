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

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== Magic Byte Detection ====================

    #[test]
    fn test_is_valid_magic_aprn() {
        assert!(is_valid_magic(&MAGIC_APRN));
    }

    #[test]
    fn test_is_valid_magic_apr1() {
        assert!(is_valid_magic(&MAGIC_APR1));
    }

    #[test]
    fn test_is_valid_magic_apr2() {
        assert!(is_valid_magic(&MAGIC_APR2));
    }

    #[test]
    fn test_is_valid_magic_apr0() {
        assert!(is_valid_magic(&MAGIC_APR0));
    }

    #[test]
    fn test_is_valid_magic_gguf() {
        assert!(is_valid_magic(&MAGIC_GGUF));
    }

    #[test]
    fn test_is_valid_magic_unknown() {
        assert!(!is_valid_magic(&[0x00, 0x00, 0x00, 0x00]));
    }

    #[test]
    fn test_is_valid_magic_too_short() {
        assert!(!is_valid_magic(&[0x41, 0x50]));
    }

    #[test]
    fn test_is_valid_magic_empty() {
        assert!(!is_valid_magic(&[]));
    }

    #[test]
    fn test_is_valid_magic_extra_bytes_ignored() {
        let mut bytes = MAGIC_GGUF.to_vec();
        bytes.extend_from_slice(&[0xFF, 0xFF]);
        assert!(is_valid_magic(&bytes));
    }

    // ==================== Format Name ====================

    #[test]
    fn test_format_name_aprn() {
        assert_eq!(format_name(&MAGIC_APRN), "APRN (aprender v1)");
    }

    #[test]
    fn test_format_name_apr1() {
        assert_eq!(format_name(&MAGIC_APR1), "APR1 (whisper.apr)");
    }

    #[test]
    fn test_format_name_apr2() {
        assert_eq!(format_name(&MAGIC_APR2), "APR2 (aprender v2)");
    }

    #[test]
    fn test_format_name_apr0() {
        assert_eq!(format_name(&MAGIC_APR0), "APR v2 (ONE TRUE format)");
    }

    #[test]
    fn test_format_name_gguf() {
        assert_eq!(format_name(&MAGIC_GGUF), "GGUF (llama.cpp)");
    }

    #[test]
    fn test_format_name_unknown() {
        assert_eq!(format_name(&[0x00, 0x00, 0x00, 0x00]), "Unknown");
    }

    #[test]
    fn test_format_name_too_short() {
        assert_eq!(format_name(&[0x41]), "Unknown");
    }

    #[test]
    fn test_format_name_empty() {
        assert_eq!(format_name(&[]), "Unknown");
    }

    // ==================== Output Formatting (no-panic) ====================

    #[test]
    fn test_section_does_not_panic() {
        section("Test Section");
    }

    #[test]
    fn test_kv_does_not_panic() {
        kv("key", "value");
    }

    #[test]
    fn test_kv_with_number() {
        kv("count", 42);
    }

    #[test]
    fn test_success_does_not_panic() {
        success("operation completed");
    }

    #[test]
    fn test_warning_does_not_panic() {
        warning("something may be wrong");
    }

    #[test]
    fn test_fail_does_not_panic() {
        fail("operation failed");
    }

    #[test]
    fn test_info_does_not_panic() {
        info("informational message");
    }

    #[test]
    fn test_error_does_not_panic() {
        error("error message");
    }

    #[test]
    fn test_warn_alias_does_not_panic() {
        warn("warning via alias");
    }

    // ==================== Format Size ====================

    #[test]
    fn test_format_size_zero() {
        let s = format_size(0);
        assert!(s.contains('0'));
    }

    #[test]
    fn test_format_size_bytes() {
        let s = format_size(512);
        assert!(s.contains("512"));
    }

    #[test]
    fn test_format_size_kib() {
        let s = format_size(1024);
        assert!(s.contains("KiB") || s.contains("1"));
    }

    #[test]
    fn test_format_size_mib() {
        let s = format_size(1024 * 1024);
        assert!(s.contains("MiB") || s.contains("1"));
    }

    #[test]
    fn test_format_size_gib() {
        let s = format_size(1024 * 1024 * 1024);
        assert!(s.contains("GiB") || s.contains("1"));
    }

    // ==================== Magic Byte Constants ====================

    #[test]
    fn test_magic_constants_are_4_bytes() {
        assert_eq!(MAGIC_APRN.len(), 4);
        assert_eq!(MAGIC_APR1.len(), 4);
        assert_eq!(MAGIC_APR2.len(), 4);
        assert_eq!(MAGIC_APR0.len(), 4);
        assert_eq!(MAGIC_GGUF.len(), 4);
    }

    #[test]
    fn test_magic_aprn_is_ascii_aprn() {
        assert_eq!(&MAGIC_APRN, b"APRN");
    }

    #[test]
    fn test_magic_apr1_is_ascii_apr1() {
        assert_eq!(&MAGIC_APR1, b"APR1");
    }

    #[test]
    fn test_magic_apr2_is_ascii_apr2() {
        assert_eq!(&MAGIC_APR2, b"APR2");
    }

    #[test]
    fn test_magic_gguf_is_ascii_gguf() {
        assert_eq!(&MAGIC_GGUF, b"GGUF");
    }
}
