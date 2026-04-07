//! Display utilities: column alignment, truncation, and builder traits.
//!
//! Provides consistent text formatting for terminal output across the Batuta stack.

use crate::fmt;

// =============================================================================
// ENUMS
// =============================================================================

/// Truncation strategy for text that exceeds column width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncateStrategy {
    /// Truncate from the end with ellipsis: "very long te…"
    #[default]
    End,
    /// Truncate from the start with ellipsis: "…ong text here"
    Start,
    /// Truncate in the middle: "hel…orld"
    Middle,
    /// Smart path truncation: keeps filename, truncates directories
    Path,
}

/// Column alignment for formatted output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColumnAlign {
    /// Left-align text.
    #[default]
    Left,
    /// Right-align text (common for numbers).
    Right,
    /// Center-align text.
    Center,
}

// =============================================================================
// TRUNCATION
// =============================================================================

/// Truncate a string to fit within a maximum width.
///
/// **Guarantee**: Output length will NEVER exceed `max_width` characters.
///
/// # Examples
/// ```
/// use batuta_common::display::{truncate, TruncateStrategy};
/// assert_eq!(truncate("hello world", 8, TruncateStrategy::End), "hello w…");
/// assert_eq!(truncate("hello world", 8, TruncateStrategy::Start), "…o world");
/// assert_eq!(truncate("hello world", 8, TruncateStrategy::Middle), "hel…orld");
/// assert_eq!(truncate("short", 10, TruncateStrategy::End), "short");
/// ```
#[must_use]
pub fn truncate(s: &str, max_width: usize, strategy: TruncateStrategy) -> String {
    if max_width == 0 {
        return String::new();
    }

    let char_count = s.chars().count();

    if char_count <= max_width {
        return s.to_string();
    }

    if max_width == 1 {
        return "\u{2026}".to_string();
    }

    match strategy {
        TruncateStrategy::End => {
            let chars: String = s.chars().take(max_width - 1).collect();
            format!("{chars}\u{2026}")
        }
        TruncateStrategy::Start => {
            let chars: String = s.chars().skip(char_count - max_width + 1).collect();
            format!("\u{2026}{chars}")
        }
        TruncateStrategy::Middle => {
            let left_len = (max_width - 1) / 2;
            let right_len = max_width - 1 - left_len;
            let left: String = s.chars().take(left_len).collect();
            let right: String = s.chars().skip(char_count - right_len).collect();
            format!("{left}\u{2026}{right}")
        }
        TruncateStrategy::Path => truncate_path(s, max_width),
    }
}

/// Smart path truncation that preserves the filename.
///
/// **Guarantee**: Output length will NEVER exceed `max_width` characters.
///
/// # Examples
/// ```
/// use batuta_common::display::truncate_path;
/// assert_eq!(truncate_path("/home/user/documents/file.txt", 20), "/home/user…/file.txt");
/// assert_eq!(truncate_path("/a/b/c.txt", 20), "/a/b/c.txt");
/// ```
#[must_use]
pub fn truncate_path(path: &str, max_width: usize) -> String {
    if max_width == 0 {
        return String::new();
    }

    let char_count = path.chars().count();

    if char_count <= max_width {
        return path.to_string();
    }

    if let Some(last_sep) = path.rfind('/') {
        let filename = &path[last_sep..];
        let filename_len = filename.chars().count();

        if filename_len >= max_width {
            return truncate(path, max_width, TruncateStrategy::End);
        }

        let dir_space = max_width.saturating_sub(filename_len).saturating_sub(1);

        if dir_space == 0 {
            let result = format!("\u{2026}{filename}");
            if result.chars().count() <= max_width {
                return result;
            }
            return truncate(path, max_width, TruncateStrategy::End);
        }

        let dir = &path[..last_sep];
        let dir_chars: Vec<char> = dir.chars().collect();

        if dir_chars.len() <= dir_space {
            return path.to_string();
        }

        let truncated_dir: String = dir_chars.iter().take(dir_space).collect();
        let result = format!("{truncated_dir}\u{2026}{filename}");

        if result.chars().count() <= max_width {
            result
        } else {
            truncate(path, max_width, TruncateStrategy::End)
        }
    } else {
        truncate(path, max_width, TruncateStrategy::End)
    }
}

// =============================================================================
// COLUMN FORMATTING
// =============================================================================

/// Format text into a fixed-width column with alignment and truncation.
///
/// **Guarantee**: Output length will NEVER exceed `width` characters.
///
/// # Examples
/// ```
/// use batuta_common::display::{format_column, ColumnAlign, TruncateStrategy};
/// assert_eq!(format_column("test", 8, ColumnAlign::Left, TruncateStrategy::End), "test    ");
/// assert_eq!(format_column("test", 8, ColumnAlign::Right, TruncateStrategy::End), "    test");
/// assert_eq!(format_column("test", 8, ColumnAlign::Center, TruncateStrategy::End), "  test  ");
/// ```
#[must_use]
pub fn format_column(
    text: &str,
    width: usize,
    align: ColumnAlign,
    truncate_strategy: TruncateStrategy,
) -> String {
    let char_count = text.chars().count();

    let truncated = if char_count > width {
        truncate(text, width, truncate_strategy)
    } else {
        text.to_string()
    };

    let truncated_len = truncated.chars().count();
    let padding = width.saturating_sub(truncated_len);

    match align {
        ColumnAlign::Left => {
            let mut result = truncated;
            for _ in 0..padding {
                result.push(' ');
            }
            result
        }
        ColumnAlign::Right => {
            let mut result = String::with_capacity(width);
            for _ in 0..padding {
                result.push(' ');
            }
            result.push_str(&truncated);
            result
        }
        ColumnAlign::Center => {
            let left_pad = padding / 2;
            let right_pad = padding - left_pad;
            let mut result = String::with_capacity(width);
            for _ in 0..left_pad {
                result.push(' ');
            }
            result.push_str(&truncated);
            for _ in 0..right_pad {
                result.push(' ');
            }
            result
        }
    }
}

/// Format bytes into a fixed-width column (SI units, right-aligned).
///
/// # Examples
/// ```
/// use batuta_common::display::format_bytes_column;
/// assert_eq!(format_bytes_column(1500, 6), " 1.50K");
/// ```
#[must_use]
pub fn format_bytes_column(bytes: u64, width: usize) -> String {
    let formatted = fmt::format_bytes_si(bytes);
    format_column(&formatted, width, ColumnAlign::Right, TruncateStrategy::End)
}

/// Format a percentage into a fixed-width column (right-aligned).
///
/// # Examples
/// ```
/// use batuta_common::display::format_percent_column;
/// assert_eq!(format_percent_column(45.3, 7), "  45.3%");
/// ```
#[must_use]
pub fn format_percent_column(value: f64, width: usize) -> String {
    let formatted = fmt::format_percent(value);
    format_column(&formatted, width, ColumnAlign::Right, TruncateStrategy::End)
}

/// Format a number into a fixed-width column (right-aligned, with commas).
#[must_use]
pub fn format_number_column(n: u64, width: usize) -> String {
    let formatted = fmt::format_number(n);
    format_column(&formatted, width, ColumnAlign::Right, TruncateStrategy::End)
}

// =============================================================================
// BUILDER TRAIT
// =============================================================================

/// Trait for types that have configurable width/height dimensions.
///
/// Eliminates the 14+ identical `dimensions()` builder methods duplicated across
/// pmat visualization and trueno-viz chart types.
///
/// # Examples
/// ```
/// use batuta_common::display::WithDimensions;
///
/// struct Chart { width: u32, height: u32 }
///
/// impl WithDimensions for Chart {
///     fn set_dimensions(&mut self, width: u32, height: u32) {
///         self.width = width;
///         self.height = height;
///     }
/// }
///
/// let chart = Chart { width: 80, height: 24 }.dimensions(120, 40);
/// assert_eq!(chart.width, 120);
/// assert_eq!(chart.height, 40);
/// ```
pub trait WithDimensions: Sized {
    /// Set the width and height on this type.
    fn set_dimensions(&mut self, width: u32, height: u32);

    /// Builder method: set dimensions and return self.
    #[must_use]
    fn dimensions(mut self, width: u32, height: u32) -> Self {
        self.set_dimensions(width, height);
        self
    }
}

// =============================================================================
// CONVENIENCE: truncate_str (ASCII "..." suffix)
// =============================================================================

/// Truncate a string to `max_len` with ASCII `"..."` suffix.
///
/// This is a convenience wrapper for CLI output where ASCII ellipsis is
/// preferred over Unicode ellipsis. If the string fits, it is returned as-is.
///
/// # Examples
/// ```
/// use batuta_common::display::truncate_str;
/// assert_eq!(truncate_str("hello world", 8), "hello...");
/// assert_eq!(truncate_str("short", 10), "short");
/// assert_eq!(truncate_str("ab", 3), "ab");
/// assert_eq!(truncate_str("abcdef", 3), "...");
/// ```
#[must_use]
pub fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }
    if max_len <= 3 {
        return ".".repeat(max_len);
    }
    let end = s
        .char_indices()
        .nth(max_len - 3)
        .map_or(max_len - 3, |(i, _)| i);
    format!("{}...", &s[..end])
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Truncation ---

    #[test]
    fn test_truncate_short_string_unchanged() {
        assert_eq!(truncate("hello", 10, TruncateStrategy::End), "hello");
    }

    #[test]
    fn test_truncate_end() {
        assert_eq!(truncate("hello world", 8, TruncateStrategy::End), "hello w\u{2026}");
    }

    #[test]
    fn test_truncate_start() {
        assert_eq!(
            truncate("hello world", 8, TruncateStrategy::Start),
            "\u{2026}o world"
        );
    }

    #[test]
    fn test_truncate_middle() {
        assert_eq!(
            truncate("hello world", 8, TruncateStrategy::Middle),
            "hel\u{2026}orld"
        );
    }

    #[test]
    fn test_truncate_zero_width() {
        assert_eq!(truncate("anything", 0, TruncateStrategy::End), "");
    }

    #[test]
    fn test_truncate_width_one() {
        assert_eq!(truncate("hello", 1, TruncateStrategy::End), "\u{2026}");
    }

    #[test]
    fn test_truncate_path_preserves_filename() {
        assert_eq!(
            truncate_path("/home/user/documents/file.txt", 20),
            "/home/user\u{2026}/file.txt"
        );
    }

    #[test]
    fn test_truncate_path_short_enough() {
        assert_eq!(truncate_path("/a/b/c.txt", 20), "/a/b/c.txt");
    }

    // --- Column formatting ---

    #[test]
    fn test_format_column_left() {
        assert_eq!(
            format_column("test", 8, ColumnAlign::Left, TruncateStrategy::End),
            "test    "
        );
    }

    #[test]
    fn test_format_column_right() {
        assert_eq!(
            format_column("test", 8, ColumnAlign::Right, TruncateStrategy::End),
            "    test"
        );
    }

    #[test]
    fn test_format_column_center() {
        assert_eq!(
            format_column("test", 8, ColumnAlign::Center, TruncateStrategy::End),
            "  test  "
        );
    }

    #[test]
    fn test_format_column_truncates() {
        assert_eq!(
            format_column("very long text", 8, ColumnAlign::Left, TruncateStrategy::End),
            "very lo\u{2026}"
        );
    }

    #[test]
    fn test_format_bytes_column() {
        assert_eq!(format_bytes_column(1500, 6), " 1.50K");
    }

    #[test]
    fn test_format_percent_column() {
        assert_eq!(format_percent_column(45.3, 7), "  45.3%");
    }

    // --- truncate_str ---

    #[test]
    fn test_truncate_str_short_unchanged() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_str_exact_fit() {
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_str_with_ellipsis() {
        assert_eq!(truncate_str("hello world", 8), "hello...");
    }

    #[test]
    fn test_truncate_str_min_len() {
        assert_eq!(truncate_str("abcdef", 3), "...");
    }

    #[test]
    fn test_truncate_str_len_4() {
        assert_eq!(truncate_str("abcdef", 4), "a...");
    }

    #[test]
    fn test_truncate_str_empty() {
        assert_eq!(truncate_str("", 5), "");
    }

    // --- WithDimensions trait ---

    #[test]
    fn test_with_dimensions_trait() {
        struct TestWidget {
            width: u32,
            height: u32,
        }

        impl WithDimensions for TestWidget {
            fn set_dimensions(&mut self, width: u32, height: u32) {
                self.width = width;
                self.height = height;
            }
        }

        let w = TestWidget {
            width: 0,
            height: 0,
        }
        .dimensions(120, 40);
        assert_eq!(w.width, 120);
        assert_eq!(w.height, 40);
    }
}
