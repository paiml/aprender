//! Shared helper functions for RAG oracle commands.

use crate::ansi_colors::Colorize;

/// Format a Unix timestamp (ms) as a human-readable string.
pub(super) fn format_timestamp(timestamp_ms: u64) -> String {
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    let duration = Duration::from_millis(timestamp_ms);
    let datetime = UNIX_EPOCH + duration;

    let age = SystemTime::now().duration_since(datetime).unwrap_or(Duration::ZERO);

    if age.as_secs() < 60 {
        "just now".to_string()
    } else if age.as_secs() < 3600 {
        format!("{} min ago", age.as_secs() / 60)
    } else if age.as_secs() < 86400 {
        format!("{} hours ago", age.as_secs() / 3600)
    } else {
        format!("{} days ago", age.as_secs() / 86400)
    }
}

/// Print a labeled statistic with the label in bright yellow.
pub(super) fn print_stat(label: &str, value: impl std::fmt::Display) {
    println!("{}: {}", label.bright_yellow(), value);
}
