//! Display Rules - Grammar of Graphics formatting primitives for TUI
//!
//! SPEC-024 Section 28: Display Rules Framework
#![allow(clippy::collapsible_if)] // Intentional for clarity in command parsing
#![allow(clippy::branches_sharing_code)] // Intentional for clarity in conditional flow
//!
//! All text and numbers in a TUI MUST follow consistent display rules:
//!
//! 1. **Numbers**: Use human-readable units (1.2G not 1234567890)
//! 2. **Truncation**: Smart truncation preserving meaningful content
//! 3. **Columns**: Width-aware formatting that never bleeds
//! 4. **Search**: O(1) fuzzy matching with relevance scoring
//!
//! # Performance Targets (from pzsh/aprender-shell patterns)
//! - Search: <1ms for 5000 items
//! - Format: <100µs per cell
//! - Truncate: <10µs per string

use std::borrow::Cow;

// =============================================================================
// BYTE FORMATTING (1000 vs 1024 base)
// =============================================================================

/// Format bytes with SI units (1000-based: KB, MB, GB)
///
/// # Examples
/// ```ignore
/// assert_eq!(format_bytes_si(1500), "1.5K");
/// assert_eq!(format_bytes_si(1_500_000), "1.5M");
/// assert_eq!(format_bytes_si(1_500_000_000), "1.5G");
/// ```
#[must_use]
pub fn format_bytes_si(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "K", "M", "G", "T", "P"];
    const BASE: f64 = 1000.0;

    if bytes == 0 {
        return "0B".to_string();
    }

    let bytes_f = bytes as f64;
    let exp = (bytes_f.log10() / BASE.log10()).floor() as usize;
    let exp = exp.min(UNITS.len() - 1);
    let value = bytes_f / BASE.powi(exp as i32);

    if exp == 0 {
        format!("{bytes}B")
    } else if value >= 100.0 {
        format!("{:.0}{}", value, UNITS[exp])
    } else if value >= 10.0 {
        format!("{:.1}{}", value, UNITS[exp])
    } else {
        format!("{:.2}{}", value, UNITS[exp])
    }
}

/// Format bytes with IEC units (1024-based: KiB, MiB, GiB)
#[must_use]
pub fn format_bytes_iec(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "Ki", "Mi", "Gi", "Ti", "Pi"];
    const BASE: f64 = 1024.0;

    if bytes == 0 {
        return "0B".to_string();
    }

    let bytes_f = bytes as f64;
    let exp = (bytes_f.log2() / 10.0).floor() as usize;
    let exp = exp.min(UNITS.len() - 1);
    let value = bytes_f / BASE.powi(exp as i32);

    if exp == 0 {
        format!("{bytes}B")
    } else if value >= 100.0 {
        format!("{:.0}{}", value, UNITS[exp])
    } else if value >= 10.0 {
        format!("{:.1}{}", value, UNITS[exp])
    } else {
        format!("{:.2}{}", value, UNITS[exp])
    }
}

/// Format bytes/second as transfer rate
#[must_use]
pub fn format_rate(bytes_per_sec: u64) -> String {
    format!("{}/s", format_bytes_si(bytes_per_sec))
}

// =============================================================================
// PERCENTAGE FORMATTING
// =============================================================================

/// Format percentage with smart precision
///
/// - 0-9.99%: 1 decimal (e.g., "5.2%")
/// - 10-99.9%: 1 decimal (e.g., "45.3%")
/// - 100%+: 0 decimal (e.g., "153%")
#[must_use]
pub fn format_percent(value: f32) -> String {
    if value >= 100.0 {
        format!("{value:.0}%")
    } else if value >= 10.0 {
        format!("{value:.1}%")
    } else if value >= 0.1 {
        format!("{value:.1}%")
    } else if value > 0.0 {
        format!("{value:.2}%")
    } else {
        "0%".to_string()
    }
}

/// Format percentage clamped to 0-100 range
#[must_use]
pub fn format_percent_clamped(value: f32) -> String {
    format_percent(value.clamp(0.0, 100.0))
}

/// Format percentage with fixed width (right-aligned)
#[must_use]
pub fn format_percent_fixed(value: f32, width: usize) -> String {
    let s = format_percent(value);
    if s.len() >= width {
        s
    } else {
        format!("{s:>width$}")
    }
}

// =============================================================================
// TIME/DURATION FORMATTING
// =============================================================================

/// Format duration in human-readable form
///
/// - <60s: "45s"
/// - <60m: "5m 30s" or "5:30"
/// - <24h: "3h 15m" or "3:15:00"
/// - ≥24h: "2d 5h" or just "2d"
#[must_use]
pub fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        let m = secs / 60;
        let s = secs % 60;
        if s == 0 {
            format!("{m}m")
        } else {
            format!("{m}:{s:02}")
        }
    } else if secs < 86400 {
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        if s == 0 && m == 0 {
            format!("{h}h")
        } else if s == 0 {
            format!("{h}:{m:02}")
        } else {
            format!("{h}:{m:02}:{s:02}")
        }
    } else {
        let d = secs / 86400;
        let h = (secs % 86400) / 3600;
        if h == 0 {
            format!("{d}d")
        } else {
            format!("{d}d {h}h")
        }
    }
}

/// Format duration compact (for tight columns)
#[must_use]
pub fn format_duration_compact(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86400 {
        format!("{}h", secs / 3600)
    } else {
        format!("{}d", secs / 86400)
    }
}

// =============================================================================
// FREQUENCY FORMATTING
// =============================================================================

/// Format frequency (MHz to GHz conversion)
#[must_use]
pub fn format_freq_mhz(mhz: u64) -> String {
    if mhz >= 1000 {
        let ghz = mhz as f64 / 1000.0;
        if ghz >= 10.0 {
            format!("{ghz:.1}G")
        } else {
            format!("{ghz:.2}G")
        }
    } else {
        format!("{mhz}M")
    }
}

// =============================================================================
// TEMPERATURE FORMATTING
// =============================================================================

/// Format temperature with unit
#[must_use]
pub fn format_temp_c(celsius: f32) -> String {
    if celsius >= 100.0 {
        format!("{celsius:.0}°C")
    } else {
        format!("{celsius:.1}°C")
    }
}

// =============================================================================
// TEXT TRUNCATION STRATEGIES
// =============================================================================

/// Truncation strategy
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TruncateStrategy {
    /// Truncate from end: "`long_text`..." → "long_..."
    #[default]
    End,
    /// Truncate from start: "`long_text`..." → "..._text"
    Start,
    /// Truncate from middle: "`long_text_here`" → "long_...here"
    Middle,
    /// Path-aware: "/home/user/very/long/path" → "/home/.../path"
    Path,
    /// Command-aware: "opt 9i94wsqoafn" → "opt …afn" (keep prefix + suffix)
    Command,
}

/// Truncate string to fit within width using specified strategy
///
/// # Arguments
/// * `s` - Input string
/// * `width` - Maximum width in characters
/// * `strategy` - Truncation strategy
///
/// # Returns
/// Truncated string with ellipsis if needed
#[must_use]
pub fn truncate(s: &str, width: usize, strategy: TruncateStrategy) -> Cow<'_, str> {
    let char_count = s.chars().count();

    if char_count <= width {
        return Cow::Borrowed(s);
    }

    if width <= 3 {
        return Cow::Owned("…".repeat(width.min(1)));
    }

    match strategy {
        TruncateStrategy::End => {
            let take = width - 1; // Leave room for ellipsis
            let truncated: String = s.chars().take(take).collect();
            Cow::Owned(format!("{truncated}…"))
        }
        TruncateStrategy::Start => {
            let skip = char_count - width + 1;
            let truncated: String = s.chars().skip(skip).collect();
            Cow::Owned(format!("…{truncated}"))
        }
        TruncateStrategy::Middle => {
            let half = (width - 1) / 2;
            let start: String = s.chars().take(half).collect();
            let end: String = s.chars().skip(char_count - half).collect();
            Cow::Owned(format!("{start}…{end}"))
        }
        TruncateStrategy::Path => truncate_path(s, width),
        TruncateStrategy::Command => truncate_command(s, width),
    }
}

/// Path-aware truncation: "/home/user/very/long/path" → "/home/.../path"
fn truncate_path(path: &str, width: usize) -> Cow<'_, str> {
    if path.chars().count() <= width {
        return Cow::Borrowed(path);
    }

    let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

    if parts.is_empty() {
        return Cow::Borrowed(path);
    }

    if parts.len() == 1 {
        return truncate(path, width, TruncateStrategy::End);
    }

    // Keep first and last parts, replace middle with ...
    let first = parts.first().unwrap_or(&"");
    let last = parts.last().unwrap_or(&"");

    let prefix = if path.starts_with('/') { "/" } else { "" };
    let result = format!("{prefix}{first}/…/{last}");

    // If still too long, truncate the result
    if result.chars().count() > width {
        let truncated: String = result.chars().take(width - 1).collect();
        Cow::Owned(format!("{truncated}…"))
    } else {
        Cow::Owned(result)
    }
}

/// Extract "key" arguments that contain identifiers (IDs, ports, paths).
///
/// Key patterns: contains digits, '=', or is a flag with numeric value following.
#[inline]
fn extract_key_args<'a>(args: &[&'a str]) -> Vec<&'a str> {
    let mut key_args: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i];
        let is_key = arg.contains('=')
            || arg.chars().any(|c| c.is_ascii_digit())
            || (arg.starts_with('-')
                && i + 1 < args.len()
                && args[i + 1].chars().any(|c| c.is_ascii_digit()));

        if is_key {
            if arg.starts_with('-') && !arg.contains('=') && i + 1 < args.len() {
                key_args.push(arg);
                key_args.push(args[i + 1]);
                i += 2;
            } else {
                key_args.push(arg);
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    key_args
}

/// Build suffix string from key args (most recent first), fitting within max_width.
#[inline]
fn build_suffix_from_key_args(key_args: &[&str], max_width: usize) -> String {
    let mut suffix = String::new();
    for &arg in key_args.iter().rev() {
        let arg_len = arg.chars().count();
        let new_len = if suffix.is_empty() {
            arg_len
        } else {
            suffix.chars().count() + 1 + arg_len
        };
        if new_len <= max_width {
            if suffix.is_empty() {
                suffix = arg.to_string();
            } else {
                suffix = format!("{arg} {suffix}");
            }
        } else {
            break;
        }
    }
    suffix
}

/// Simple end truncation with ellipsis.
#[inline]
fn simple_truncate(s: &str, width: usize) -> String {
    let truncated: String = s.chars().take(width - 1).collect();
    format!("{truncated}…")
}

/// Build truncated command from components.
fn build_command_with_args(
    basename: &str,
    first_arg: &str,
    key_args: &[&str],
    width: usize,
) -> String {
    let ellipsis = " … ";
    let base_len = basename.chars().count();

    let mut result = basename.to_string();
    let mut current_len = base_len;

    // Add first arg if space
    if !first_arg.is_empty() && current_len + 1 + first_arg.chars().count() + 4 < width {
        result.push(' ');
        result.push_str(first_arg);
        current_len = result.chars().count();
    }

    // Add key args suffix if space
    let space_for_keys = width.saturating_sub(current_len + ellipsis.chars().count());
    if !key_args.is_empty() && space_for_keys > 5 {
        let suffix = build_suffix_from_key_args(key_args, space_for_keys);
        if !suffix.is_empty() {
            result.push_str(ellipsis);
            result.push_str(&suffix);
        }
    }

    // Final safety: ensure we don't exceed width
    if result.chars().count() > width {
        simple_truncate(&result, width)
    } else {
        result
    }
}

/// Command-aware truncation using Basename + Key Args pattern (htop-style)
///
/// Strategy:
/// 1. Extract basename from executable path (`/usr/lib/firefox/firefox` → `firefox`)
/// 2. Identify "key args" with identifiers (childID, port, pid, etc.)
/// 3. Show: `basename [first-arg] … [key-args-from-end]`
///
/// # Examples
/// - `/usr/lib/firefox/firefox -contentproc -childID 5 -isForBrowser -prefsLen 31398`
///   → `firefox -contentproc … -childID 5 -isForBrowser` (width=45)
/// - `python /home/user/scripts/long/path/script.py --port=8080`
///   → `python script.py --port=8080` (width=30)
fn truncate_command(cmd: &str, width: usize) -> Cow<'_, str> {
    if cmd.chars().count() <= width {
        return Cow::Borrowed(cmd);
    }

    if width <= 3 {
        return Cow::Owned("…".repeat(width.min(1)));
    }

    if width < 12 {
        return Cow::Owned(simple_truncate(cmd, width));
    }

    // Parse command into parts
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        return Cow::Borrowed(cmd);
    }

    // Extract basename from first part (executable)
    let basename = parts[0].rsplit('/').next().unwrap_or(parts[0]);

    // Handle single-part command
    if parts.len() == 1 {
        return if basename.len() <= width {
            Cow::Owned(basename.to_string())
        } else {
            Cow::Owned(simple_truncate(basename, width))
        };
    }

    // Build with args using helper
    let args = &parts[1..];
    let key_args = extract_key_args(args);
    let first_arg = args.first().copied().unwrap_or("");
    Cow::Owned(build_command_with_args(
        basename, first_arg, &key_args, width,
    ))
}

// =============================================================================
// COLUMN FORMATTING (PREVENTS BLEEDING)
// =============================================================================

/// Column alignment
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ColumnAlign {
    #[default]
    Left,
    Right,
    Center,
}

/// Format a value to fit exactly within a column width
///
/// **GUARANTEE**: Output will NEVER exceed `width` characters
///
/// # Arguments
/// * `value` - The string to format
/// * `width` - Exact column width
/// * `align` - Alignment within column
/// * `truncate_strategy` - How to truncate if too long
#[must_use]
pub fn format_column(
    value: &str,
    width: usize,
    align: ColumnAlign,
    truncate_strategy: TruncateStrategy,
) -> String {
    let truncated = truncate(value, width, truncate_strategy);
    let len = truncated.chars().count();

    if len >= width {
        // Already at or over width, just take exactly width chars
        truncated.chars().take(width).collect()
    } else {
        let padding = width - len;
        match align {
            ColumnAlign::Left => format!("{truncated}{}", " ".repeat(padding)),
            ColumnAlign::Right => format!("{}{truncated}", " ".repeat(padding)),
            ColumnAlign::Center => {
                let left = padding / 2;
                let right = padding - left;
                format!("{}{truncated}{}", " ".repeat(left), " ".repeat(right))
            }
        }
    }
}

/// Format a number column (right-aligned, numeric formatting)
#[must_use]
pub fn format_number_column(value: f64, width: usize, decimals: usize) -> String {
    let formatted = if decimals == 0 {
        format!("{value:.0}")
    } else {
        format!("{value:.decimals$}")
    };
    format_column(&formatted, width, ColumnAlign::Right, TruncateStrategy::End)
}

/// Format a percentage column
#[must_use]
pub fn format_percent_column(value: f32, width: usize) -> String {
    let formatted = format_percent(value);
    format_column(&formatted, width, ColumnAlign::Right, TruncateStrategy::End)
}

/// Format a bytes column
#[must_use]
pub fn format_bytes_column(bytes: u64, width: usize) -> String {
    let formatted = format_bytes_si(bytes);
    format_column(&formatted, width, ColumnAlign::Right, TruncateStrategy::End)
}

// =============================================================================
// O(1) FUZZY SEARCH (from pzsh/aprender-shell patterns)
// =============================================================================

/// Search result with relevance scoring
#[derive(Debug, Clone)]
pub struct SearchResult<T> {
    /// The matched item
    pub item: T,
    /// Relevance score (0.0 - 1.0, higher is better)
    pub score: f32,
    /// Match positions (for highlighting)
    pub matches: Vec<usize>,
}

/// Fast fuzzy search with O(1) amortized lookup via pre-computed index
///
/// Performance targets:
/// - Build: O(n) where n = number of items
/// - Search: O(m * k) where m = query length, k = average results
/// - Memory: O(n * `avg_key_length`)
#[derive(Debug, Clone)]
pub struct FuzzyIndex<T: Clone> {
    /// Items to search
    items: Vec<T>,
    /// Pre-computed lowercase keys for fast comparison
    keys: Vec<String>,
    /// Trigram index for O(1) candidate lookup
    trigrams: std::collections::HashMap<[u8; 3], Vec<usize>>,
    /// Character index for single-char queries
    char_index: std::collections::HashMap<char, Vec<usize>>,
}

impl<T: Clone> FuzzyIndex<T> {
    /// Build a fuzzy search index from items
    ///
    /// # Arguments
    /// * `items` - Items to index
    /// * `key_fn` - Function to extract searchable key from each item
    pub fn new<F>(items: Vec<T>, key_fn: F) -> Self
    where
        F: Fn(&T) -> String,
    {
        let keys: Vec<String> = items
            .iter()
            .map(|item| key_fn(item).to_lowercase())
            .collect();

        let mut trigrams: std::collections::HashMap<[u8; 3], Vec<usize>> =
            std::collections::HashMap::new();
        let mut char_index: std::collections::HashMap<char, Vec<usize>> =
            std::collections::HashMap::new();

        for (idx, key) in keys.iter().enumerate() {
            // Build character index
            for ch in key.chars() {
                char_index.entry(ch).or_default().push(idx);
            }

            // Build trigram index
            let bytes = key.as_bytes();
            if bytes.len() >= 3 {
                for window in bytes.windows(3) {
                    let trigram: [u8; 3] = [window[0], window[1], window[2]];
                    trigrams.entry(trigram).or_default().push(idx);
                }
            }
        }

        // Deduplicate indices
        for indices in trigrams.values_mut() {
            indices.sort_unstable();
            indices.dedup();
        }
        for indices in char_index.values_mut() {
            indices.sort_unstable();
            indices.dedup();
        }

        Self {
            items,
            keys,
            trigrams,
            char_index,
        }
    }

    /// Search for items matching query
    ///
    /// # Arguments
    /// * `query` - Search query (case-insensitive)
    /// * `limit` - Maximum results to return
    ///
    /// # Returns
    /// Sorted results by relevance (highest first)
    #[must_use]
    pub fn search(&self, query: &str, limit: usize) -> Vec<SearchResult<T>> {
        if query.is_empty() {
            return Vec::new();
        }

        let query_lower = query.to_lowercase();
        let query_chars: Vec<char> = query_lower.chars().collect();

        // Get candidate indices
        let candidates = self.get_candidates(&query_lower);

        // Score and filter candidates
        let mut results: Vec<SearchResult<T>> = candidates
            .into_iter()
            .filter_map(|idx| {
                let key = &self.keys[idx];
                let (score, matches) = self.score_match(key, &query_chars);
                if score > 0.0 {
                    Some(SearchResult {
                        item: self.items[idx].clone(),
                        score,
                        matches,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by score (descending)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        results
    }

    /// Get candidate indices using index
    fn get_candidates(&self, query: &str) -> Vec<usize> {
        let bytes = query.as_bytes();

        // Use trigram index if query is long enough
        if bytes.len() >= 3 {
            let trigram: [u8; 3] = [bytes[0], bytes[1], bytes[2]];
            if let Some(indices) = self.trigrams.get(&trigram) {
                return indices.clone();
            }
        }

        // Fall back to character index
        if let Some(first_char) = query.chars().next() {
            if let Some(indices) = self.char_index.get(&first_char) {
                return indices.clone();
            }
        }

        // Fall back to scanning all items
        (0..self.items.len()).collect()
    }

    /// Score a match using fuzzy substring matching
    fn score_match(&self, key: &str, query_chars: &[char]) -> (f32, Vec<usize>) {
        if query_chars.is_empty() {
            return (0.0, Vec::new());
        }

        let key_chars: Vec<char> = key.chars().collect();
        let mut matches = Vec::new();
        let mut query_idx = 0;

        // Find subsequence matches
        for (key_idx, &key_char) in key_chars.iter().enumerate() {
            if query_idx < query_chars.len() && key_char == query_chars[query_idx] {
                matches.push(key_idx);
                query_idx += 1;
            }
        }

        // Must match all query characters
        if query_idx != query_chars.len() {
            return (0.0, Vec::new());
        }

        // Calculate score based on:
        // - Consecutive matches (bonus)
        // - Position of first match (earlier is better)
        // - Match density (matches / key length)

        let mut score = 1.0;

        // Consecutive bonus
        let mut consecutive = 0;
        for i in 1..matches.len() {
            if matches[i] == matches[i - 1] + 1 {
                consecutive += 1;
            }
        }
        score += consecutive as f32 * 0.1;

        // Early match bonus
        if !matches.is_empty() {
            score += (1.0 - matches[0] as f32 / key_chars.len() as f32) * 0.3;
        }

        // Exact prefix bonus
        if key.starts_with(&query_chars.iter().collect::<String>()) {
            score += 0.5;
        }

        // Exact match bonus
        if key_chars.len() == query_chars.len() {
            score += 1.0;
        }

        // Density factor
        score *= query_chars.len() as f32 / key_chars.len() as f32;

        (score.min(1.0), matches)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // BYTE FORMATTING TESTS
    // =========================================================================

    #[test]
    fn test_format_bytes_si() {
        assert_eq!(format_bytes_si(0), "0B");
        assert_eq!(format_bytes_si(100), "100B");
        assert_eq!(format_bytes_si(1000), "1.00K");
        assert_eq!(format_bytes_si(1500), "1.50K");
        assert_eq!(format_bytes_si(1_000_000), "1.00M");
        assert_eq!(format_bytes_si(1_500_000_000), "1.50G");
        assert_eq!(format_bytes_si(1_000_000_000_000), "1.00T");
    }

    #[test]
    fn test_format_bytes_si_large() {
        // Test 100+ values
        assert_eq!(format_bytes_si(150_000_000_000), "150G");
    }

    #[test]
    fn test_format_bytes_iec() {
        assert_eq!(format_bytes_iec(0), "0B");
        assert_eq!(format_bytes_iec(100), "100B");
        assert_eq!(format_bytes_iec(1024), "1.00Ki");
        assert_eq!(format_bytes_iec(1024 * 1024), "1.00Mi");
        assert_eq!(format_bytes_iec(1024 * 1024 * 1024), "1.00Gi");
    }

    #[test]
    fn test_format_rate() {
        assert_eq!(format_rate(1000), "1.00K/s");
        assert_eq!(format_rate(1_000_000), "1.00M/s");
    }

    // =========================================================================
    // PERCENTAGE FORMATTING TESTS
    // =========================================================================

    #[test]
    fn test_format_percent() {
        assert_eq!(format_percent(0.0), "0%");
        assert_eq!(format_percent(5.0), "5.0%");
        assert_eq!(format_percent(45.3), "45.3%");
        assert_eq!(format_percent(100.0), "100%");
        assert_eq!(format_percent(153.2), "153%");
    }

    #[test]
    fn test_format_percent_small() {
        assert_eq!(format_percent(0.05), "0.05%");
    }

    #[test]
    fn test_format_percent_clamped() {
        assert_eq!(format_percent_clamped(150.0), "100%");
        assert_eq!(format_percent_clamped(-10.0), "0%");
    }

    #[test]
    fn test_format_percent_fixed() {
        let result = format_percent_fixed(50.0, 8);
        assert_eq!(result.chars().count(), 8);
    }

    #[test]
    fn test_format_percent_fixed_no_padding() {
        let result = format_percent_fixed(100.0, 3);
        assert_eq!(result, "100%");
    }

    // =========================================================================
    // DURATION FORMATTING TESTS
    // =========================================================================

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(45), "45s");
        assert_eq!(format_duration(90), "1:30");
        assert_eq!(format_duration(3661), "1:01:01");
        assert_eq!(format_duration(86400), "1d");
        assert_eq!(format_duration(90000), "1d 1h");
    }

    #[test]
    fn test_format_duration_exact_minutes() {
        assert_eq!(format_duration(60), "1m");
        assert_eq!(format_duration(120), "2m");
    }

    #[test]
    fn test_format_duration_exact_hours() {
        assert_eq!(format_duration(3600), "1h");
        assert_eq!(format_duration(7200), "2h");
    }

    #[test]
    fn test_format_duration_hours_minutes() {
        assert_eq!(format_duration(3660), "1:01");
    }

    #[test]
    fn test_format_duration_compact() {
        assert_eq!(format_duration_compact(30), "30s");
        assert_eq!(format_duration_compact(120), "2m");
        assert_eq!(format_duration_compact(7200), "2h");
        assert_eq!(format_duration_compact(172800), "2d");
    }

    // =========================================================================
    // FREQUENCY FORMATTING TESTS
    // =========================================================================

    #[test]
    fn test_format_freq_mhz() {
        assert_eq!(format_freq_mhz(500), "500M");
        assert_eq!(format_freq_mhz(1000), "1.00G");
        assert_eq!(format_freq_mhz(3500), "3.50G");
        assert_eq!(format_freq_mhz(10500), "10.5G");
    }

    // =========================================================================
    // TEMPERATURE FORMATTING TESTS
    // =========================================================================

    #[test]
    fn test_format_temp_c() {
        assert_eq!(format_temp_c(45.5), "45.5°C");
        assert_eq!(format_temp_c(105.0), "105°C");
    }

    // =========================================================================
    // TRUNCATION TESTS
    // =========================================================================

    #[test]
    fn test_truncate_strategy_default() {
        assert_eq!(TruncateStrategy::default(), TruncateStrategy::End);
    }

    #[test]
    fn test_truncate_end() {
        assert_eq!(truncate("hello", 10, TruncateStrategy::End), "hello");
        assert_eq!(
            truncate("hello world", 8, TruncateStrategy::End),
            "hello w…"
        );
        assert_eq!(truncate("hi", 1, TruncateStrategy::End), "…");
    }

    #[test]
    fn test_truncate_start() {
        // "hello world" is 11 chars, width 8, skip = 11 - 8 + 1 = 4
        // Skips "hell", keeps "o world" with ellipsis = "…o world"
        assert_eq!(
            truncate("hello world", 8, TruncateStrategy::Start),
            "…o world"
        );
    }

    #[test]
    fn test_truncate_middle() {
        assert_eq!(
            truncate("hello_world_here", 10, TruncateStrategy::Middle),
            "hell…here"
        );
    }

    #[test]
    fn test_truncate_very_short() {
        assert_eq!(truncate("hello", 2, TruncateStrategy::End), "…");
        assert_eq!(truncate("hello", 3, TruncateStrategy::End), "…");
    }

    #[test]
    fn test_truncate_path() {
        assert_eq!(
            truncate("/home/user/documents/file.txt", 20, TruncateStrategy::Path),
            "/home/…/file.txt"
        );
    }

    #[test]
    fn test_truncate_path_single_part() {
        let result = truncate("filename", 5, TruncateStrategy::Path);
        assert!(result.chars().count() <= 5);
    }

    #[test]
    fn test_truncate_path_empty() {
        let result = truncate("", 10, TruncateStrategy::Path);
        assert_eq!(result, "");
    }

    #[test]
    fn test_truncate_path_no_slash() {
        let result = truncate("verylongfilename.txt", 10, TruncateStrategy::Path);
        assert!(result.chars().count() <= 10);
    }

    #[test]
    fn test_truncate_command() {
        let cmd = "/usr/bin/python script.py";
        let result = truncate(cmd, 20, TruncateStrategy::Command);
        assert!(result.starts_with("python"));

        let long_cmd = "/usr/lib/firefox/firefox -contentproc -parentBuildID 20240101 -childID 5 -isForBrowser";
        let result = truncate(long_cmd, 50, TruncateStrategy::Command);
        assert!(result.starts_with("firefox"));
        assert!(result.contains("…"));
        assert!(result.contains('5'));

        assert_eq!(
            truncate("command arg1 arg2", 8, TruncateStrategy::Command),
            "command…"
        );

        let very_long = "/usr/lib/firefox/firefox -contentproc -childID 5 -isForBrowser -prefsLen 31398 -prefMapSize 244787";
        let result = truncate(very_long, 40, TruncateStrategy::Command);
        assert!(
            result.chars().count() <= 40,
            "Result '{}' exceeds 40 chars",
            result
        );

        let with_eq = "python script.py --port=8080";
        let result = truncate(with_eq, 30, TruncateStrategy::Command);
        assert!(
            result.contains("8080") || result == "python script.py --port=8080",
            "Result '{}' should contain 8080 or fit entirely",
            result
        );
    }

    #[test]
    fn test_truncate_command_short_width() {
        let result = truncate("/usr/bin/python", 5, TruncateStrategy::Command);
        assert!(result.chars().count() <= 5);
    }

    #[test]
    fn test_truncate_command_single_word() {
        // When the input fits within width, it's returned unchanged
        let result = truncate("/usr/bin/python", 15, TruncateStrategy::Command);
        assert_eq!(result, "/usr/bin/python");

        // When it needs truncation, basename is used
        let result = truncate("/usr/bin/python", 10, TruncateStrategy::Command);
        assert!(result.chars().count() <= 10);
    }

    #[test]
    fn test_truncate_command_basename_only_too_long() {
        let result = truncate(
            "/usr/bin/verylongexecutablename",
            10,
            TruncateStrategy::Command,
        );
        assert!(result.chars().count() <= 10);
    }

    // =========================================================================
    // COLUMN FORMATTING TESTS
    // =========================================================================

    #[test]
    fn test_column_align_default() {
        assert_eq!(ColumnAlign::default(), ColumnAlign::Left);
    }

    #[test]
    fn test_format_column_never_bleeds() {
        let result = format_column(
            "very_long_text_that_should_be_truncated",
            10,
            ColumnAlign::Left,
            TruncateStrategy::End,
        );
        assert_eq!(result.chars().count(), 10);

        let result = format_column("short", 10, ColumnAlign::Right, TruncateStrategy::End);
        assert_eq!(result.chars().count(), 10);
        assert!(result.starts_with("     "));
    }

    #[test]
    fn test_format_column_center() {
        let result = format_column("hi", 10, ColumnAlign::Center, TruncateStrategy::End);
        assert_eq!(result.chars().count(), 10);
        // Centered: "    hi    "
    }

    #[test]
    fn test_format_number_column() {
        let result = format_number_column(1.23456, 8, 2);
        assert_eq!(result.chars().count(), 8);
    }

    #[test]
    fn test_format_number_column_no_decimals() {
        let result = format_number_column(42.0, 6, 0);
        assert_eq!(result.chars().count(), 6);
    }

    #[test]
    fn test_format_percent_column() {
        let result = format_percent_column(50.0, 8);
        assert_eq!(result.chars().count(), 8);
    }

    #[test]
    fn test_format_bytes_column() {
        let result = format_bytes_column(1_000_000, 8);
        assert_eq!(result.chars().count(), 8);
    }

    // =========================================================================
    // FUZZY SEARCH TESTS
    // =========================================================================

    #[test]
    fn test_fuzzy_search() {
        let items = vec![
            "firefox".to_string(),
            "thunderbird".to_string(),
            "chrome".to_string(),
            "chromium".to_string(),
            "firefox-developer".to_string(),
        ];

        let index = FuzzyIndex::new(items, |s| s.clone());

        let results = index.search("fire", 5);
        assert!(!results.is_empty());
        assert!(results[0].item.contains("fire"));

        let results = index.search("chro", 5);
        assert!(results.len() >= 2);
    }

    #[test]
    fn test_fuzzy_search_empty_query() {
        let items = vec!["test".to_string()];
        let index = FuzzyIndex::new(items, |s| s.clone());
        let results = index.search("", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_fuzzy_search_no_match() {
        let items = vec!["apple".to_string(), "banana".to_string()];
        let index = FuzzyIndex::new(items, |s| s.clone());
        let results = index.search("xyz", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_fuzzy_search_exact_match() {
        let items = vec!["test".to_string(), "testing".to_string()];
        let index = FuzzyIndex::new(items, |s| s.clone());
        let results = index.search("test", 5);
        assert!(!results.is_empty());
        // Exact match should score higher
        assert_eq!(results[0].item, "test");
    }

    #[test]
    fn test_fuzzy_search_single_char() {
        let items = vec![
            "apple".to_string(),
            "banana".to_string(),
            "apricot".to_string(),
        ];
        let index = FuzzyIndex::new(items, |s| s.clone());
        let results = index.search("a", 5);
        assert!(results.len() >= 2); // apple and apricot
    }
}

// =============================================================================
// DECLARATIVE DISPLAY RULES (SPEC-024 Appendix F)
// =============================================================================

/// Action to take based on display rule evaluation
///
/// SPEC-024 Appendix F.2.2: Display Rules Grammar
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DisplayAction {
    /// Show panel normally
    Show,
    /// Hide panel completely (do not render)
    Hide,
    /// Show placeholder text instead of content
    ShowPlaceholder(String),
    /// Use compact/minimal detail level
    Compact,
    /// Expand to show more detail
    Expand,
}

impl Default for DisplayAction {
    fn default() -> Self {
        Self::Show
    }
}

/// System capabilities for display rule evaluation
#[derive(Debug, Clone, Default)]
pub struct SystemCapabilities {
    /// NVIDIA GPU available
    pub has_nvidia: bool,
    /// AMD GPU available
    pub has_amd: bool,
    /// Apple Silicon available
    pub has_apple_silicon: bool,
    /// PSI (Pressure Stall Information) available
    pub has_psi: bool,
    /// Hardware sensors available
    pub has_sensors: bool,
    /// Battery present
    pub has_battery: bool,
    /// Container runtime detected
    pub in_container: bool,
}

impl SystemCapabilities {
    /// Detect system capabilities at startup
    #[cfg(target_os = "linux")]
    pub fn detect() -> Self {
        use std::path::Path;

        Self {
            has_nvidia: Path::new("/dev/nvidia0").exists()
                || Path::new("/proc/driver/nvidia").exists(),
            has_amd: Path::new("/sys/class/drm/card0/device/vendor").exists(),
            has_apple_silicon: false,
            has_psi: Path::new("/proc/pressure/cpu").exists(),
            has_sensors: Path::new("/sys/class/hwmon/hwmon0").exists(),
            has_battery: Path::new("/sys/class/power_supply/BAT0").exists()
                || Path::new("/sys/class/power_supply/BAT1").exists(),
            in_container: Path::new("/.dockerenv").exists() || std::env::var("container").is_ok(),
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn detect() -> Self {
        Self::default()
    }
}

/// Terminal size for display rule evaluation
#[derive(Debug, Clone, Copy, Default)]
pub struct TerminalSize {
    pub width: u16,
    pub height: u16,
}

/// Data availability flags for display rule evaluation
///
/// Each flag indicates whether the corresponding data source has valid data
#[derive(Debug, Clone, Default)]
pub struct DataAvailability {
    /// PSI data available and meaningful (>0.01% pressure)
    pub psi_available: bool,
    /// Sensor readings available
    pub sensors_available: bool,
    /// Sensor count (for compact vs full display)
    pub sensor_count: usize,
    /// GPU data available
    pub gpu_available: bool,
    /// Battery data available
    pub battery_available: bool,
    /// Treemap/files data ready (not scanning)
    pub treemap_ready: bool,
    /// Connection data available
    pub connections_available: bool,
    /// Connection count
    pub connection_count: usize,
}

/// Context for evaluating display rules
///
/// SPEC-024 Appendix F.2.3: Framework Trait
pub struct DisplayContext<'a> {
    /// System capabilities (detected at startup)
    pub system: &'a SystemCapabilities,
    /// Current terminal size
    pub terminal: TerminalSize,
    /// Data availability flags
    pub data: DataAvailability,
}

/// Trait for declarative display rules evaluation
///
/// SPEC-024 Appendix F.2.3: Every panel MUST implement this trait
/// to enable YAML-controlled visibility.
pub trait DisplayRules {
    /// Evaluate display rules and return action
    ///
    /// Called before rendering to determine if panel should be shown,
    /// hidden, or displayed in a modified state.
    fn evaluate(&self, ctx: &DisplayContext) -> DisplayAction;

    /// Get panel identifier for YAML configuration
    fn panel_id(&self) -> &'static str;
}

/// Default display rules for panels without custom logic
///
/// Behavior: Always show unless data is explicitly unavailable
pub struct DefaultDisplayRules {
    panel_id: &'static str,
}

impl DefaultDisplayRules {
    pub fn new(panel_id: &'static str) -> Self {
        Self { panel_id }
    }
}

impl DisplayRules for DefaultDisplayRules {
    fn evaluate(&self, _ctx: &DisplayContext) -> DisplayAction {
        DisplayAction::Show
    }

    fn panel_id(&self) -> &'static str {
        self.panel_id
    }
}

/// PSI panel display rules
///
/// Hide if PSI not available (non-Linux or disabled kernel config)
pub struct PsiDisplayRules;

impl DisplayRules for PsiDisplayRules {
    fn evaluate(&self, ctx: &DisplayContext) -> DisplayAction {
        if !ctx.system.has_psi || !ctx.data.psi_available {
            DisplayAction::Hide
        } else {
            DisplayAction::Show
        }
    }

    fn panel_id(&self) -> &'static str {
        "psi"
    }
}

/// Sensors panel display rules
///
/// Hide if no sensors, compact if few sensors
pub struct SensorsDisplayRules;

impl DisplayRules for SensorsDisplayRules {
    fn evaluate(&self, ctx: &DisplayContext) -> DisplayAction {
        if !ctx.system.has_sensors || !ctx.data.sensors_available {
            DisplayAction::Hide
        } else if ctx.data.sensor_count < 3 {
            DisplayAction::Compact
        } else {
            DisplayAction::Show
        }
    }

    fn panel_id(&self) -> &'static str {
        "sensors"
    }
}

/// GPU panel display rules
///
/// Hide if no GPU detected
pub struct GpuDisplayRules;

impl DisplayRules for GpuDisplayRules {
    fn evaluate(&self, ctx: &DisplayContext) -> DisplayAction {
        if ctx.data.gpu_available {
            DisplayAction::Show
        } else {
            DisplayAction::Hide
        }
    }

    fn panel_id(&self) -> &'static str {
        "gpu"
    }
}

/// Battery panel display rules
///
/// Hide if no battery (desktop systems)
pub struct BatteryDisplayRules;

impl DisplayRules for BatteryDisplayRules {
    fn evaluate(&self, ctx: &DisplayContext) -> DisplayAction {
        if !ctx.system.has_battery || !ctx.data.battery_available {
            DisplayAction::Hide
        } else {
            DisplayAction::Show
        }
    }

    fn panel_id(&self) -> &'static str {
        "battery"
    }
}

/// Files panel display rules
///
/// Show placeholder while scanning
pub struct FilesDisplayRules;

impl DisplayRules for FilesDisplayRules {
    fn evaluate(&self, ctx: &DisplayContext) -> DisplayAction {
        if ctx.data.treemap_ready {
            DisplayAction::Show
        } else {
            DisplayAction::ShowPlaceholder("Scanning filesystem...".to_string())
        }
    }

    fn panel_id(&self) -> &'static str {
        "files"
    }
}

#[cfg(test)]
mod display_rules_tests {
    use super::*;

    #[test]
    fn test_psi_hides_when_unavailable() {
        let rules = PsiDisplayRules;
        let system = SystemCapabilities {
            has_psi: false,
            ..Default::default()
        };
        let ctx = DisplayContext {
            system: &system,
            terminal: TerminalSize::default(),
            data: DataAvailability::default(),
        };

        assert_eq!(rules.evaluate(&ctx), DisplayAction::Hide);
    }

    #[test]
    fn test_psi_shows_when_available() {
        let rules = PsiDisplayRules;
        let system = SystemCapabilities {
            has_psi: true,
            ..Default::default()
        };
        let ctx = DisplayContext {
            system: &system,
            terminal: TerminalSize::default(),
            data: DataAvailability {
                psi_available: true,
                ..Default::default()
            },
        };

        assert_eq!(rules.evaluate(&ctx), DisplayAction::Show);
    }

    #[test]
    fn test_sensors_compact_with_few() {
        let rules = SensorsDisplayRules;
        let system = SystemCapabilities {
            has_sensors: true,
            ..Default::default()
        };
        let ctx = DisplayContext {
            system: &system,
            terminal: TerminalSize::default(),
            data: DataAvailability {
                sensors_available: true,
                sensor_count: 2,
                ..Default::default()
            },
        };

        assert_eq!(rules.evaluate(&ctx), DisplayAction::Compact);
    }

    #[test]
    fn test_battery_hides_on_desktop() {
        let rules = BatteryDisplayRules;
        let system = SystemCapabilities {
            has_battery: false,
            ..Default::default()
        };
        let ctx = DisplayContext {
            system: &system,
            terminal: TerminalSize::default(),
            data: DataAvailability::default(),
        };

        assert_eq!(rules.evaluate(&ctx), DisplayAction::Hide);
    }

    #[test]
    fn test_files_placeholder_while_scanning() {
        let rules = FilesDisplayRules;
        let system = SystemCapabilities::default();
        let ctx = DisplayContext {
            system: &system,
            terminal: TerminalSize::default(),
            data: DataAvailability {
                treemap_ready: false,
                ..Default::default()
            },
        };

        match rules.evaluate(&ctx) {
            DisplayAction::ShowPlaceholder(msg) => {
                assert!(msg.contains("Scanning"));
            }
            _ => panic!("Expected ShowPlaceholder"),
        }
    }

    // Additional tests for byte formatting

    #[test]
    fn test_format_bytes_si_zero() {
        assert_eq!(format_bytes_si(0), "0B");
    }

    #[test]
    fn test_format_bytes_si_small() {
        assert_eq!(format_bytes_si(500), "500B");
        assert_eq!(format_bytes_si(999), "999B");
    }

    #[test]
    fn test_format_bytes_si_kilobytes() {
        assert_eq!(format_bytes_si(1000), "1.00K");
        assert_eq!(format_bytes_si(1500), "1.50K");
        assert_eq!(format_bytes_si(15000), "15.0K");
    }

    #[test]
    fn test_format_bytes_si_megabytes() {
        assert_eq!(format_bytes_si(1_000_000), "1.00M");
        assert_eq!(format_bytes_si(1_500_000), "1.50M");
        assert_eq!(format_bytes_si(100_000_000), "100M");
    }

    #[test]
    fn test_format_bytes_si_gigabytes() {
        assert_eq!(format_bytes_si(1_000_000_000), "1.00G");
        assert_eq!(format_bytes_si(10_000_000_000), "10.0G");
    }

    #[test]
    fn test_format_bytes_si_terabytes() {
        assert_eq!(format_bytes_si(1_000_000_000_000), "1.00T");
    }

    #[test]
    fn test_format_bytes_iec_zero() {
        assert_eq!(format_bytes_iec(0), "0B");
    }

    #[test]
    fn test_format_bytes_iec_small() {
        assert_eq!(format_bytes_iec(500), "500B");
        assert_eq!(format_bytes_iec(1023), "1023B");
    }

    #[test]
    fn test_format_bytes_iec_kibibytes() {
        assert_eq!(format_bytes_iec(1024), "1.00Ki");
        assert_eq!(format_bytes_iec(1536), "1.50Ki");
    }

    #[test]
    fn test_format_bytes_iec_mebibytes() {
        assert_eq!(format_bytes_iec(1024 * 1024), "1.00Mi");
        assert_eq!(format_bytes_iec(1024 * 1024 * 100), "100Mi");
    }

    #[test]
    fn test_format_bytes_iec_gibibytes() {
        assert_eq!(format_bytes_iec(1024 * 1024 * 1024), "1.00Gi");
    }

    #[test]
    fn test_format_rate() {
        assert_eq!(format_rate(0), "0B/s");
        assert_eq!(format_rate(1500), "1.50K/s");
    }

    // Percentage formatting tests

    #[test]
    fn test_format_percent_small() {
        let result = format_percent(5.25);
        assert!(result.contains("5."));
    }

    #[test]
    fn test_format_percent_medium() {
        let result = format_percent(45.3);
        assert!(result.contains("45"));
    }

    #[test]
    fn test_format_percent_full() {
        let result = format_percent(100.0);
        assert!(result.contains("100"));
    }

    #[test]
    fn test_format_percent_over() {
        let result = format_percent(153.5);
        assert!(result.contains("153") || result.contains("154"));
    }

    // Truncation tests

    #[test]
    fn test_truncate_middle_short() {
        let result = truncate("hello", 10, TruncateStrategy::Middle);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_truncate_middle_exact() {
        let result = truncate("hello", 5, TruncateStrategy::Middle);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_truncate_middle_needs_truncation() {
        let result = truncate("hello world", 8, TruncateStrategy::Middle);
        assert!(result.chars().count() <= 8);
        assert!(result.contains("…"));
    }

    #[test]
    fn test_truncate_end_short() {
        let result = truncate("hello", 10, TruncateStrategy::End);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_truncate_end_needs_truncation() {
        let result = truncate("hello world", 8, TruncateStrategy::End);
        assert!(result.chars().count() <= 8);
    }

    #[test]
    fn test_truncate_start_short() {
        let result = truncate("hello", 10, TruncateStrategy::Start);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_truncate_start_needs_truncation() {
        let result = truncate("/very/long/path/to/file.txt", 15, TruncateStrategy::Start);
        assert!(result.chars().count() <= 15);
    }

    #[test]
    fn test_truncate_path_strategy() {
        let result = truncate(
            "/home/user/very/long/path/file.txt",
            20,
            TruncateStrategy::Path,
        );
        assert!(result.chars().count() <= 20);
    }

    #[test]
    fn test_truncate_command_strategy() {
        let result = truncate(
            "command --with-very-long-argument",
            15,
            TruncateStrategy::Command,
        );
        assert!(result.chars().count() <= 15);
    }

    #[test]
    fn test_truncate_very_small_width() {
        let result = truncate("hello world", 2, TruncateStrategy::End);
        assert!(result.chars().count() <= 2);
    }

    #[test]
    fn test_truncate_strategy_default() {
        let strategy = TruncateStrategy::default();
        assert_eq!(strategy, TruncateStrategy::End);
    }

    // Display action tests

    #[test]
    fn test_display_action_eq() {
        assert_eq!(DisplayAction::Show, DisplayAction::Show);
        assert_eq!(DisplayAction::Hide, DisplayAction::Hide);
        assert_eq!(DisplayAction::Compact, DisplayAction::Compact);
        assert_ne!(DisplayAction::Show, DisplayAction::Hide);
    }

    #[test]
    fn test_display_action_placeholder_eq() {
        assert_eq!(
            DisplayAction::ShowPlaceholder("Loading".to_string()),
            DisplayAction::ShowPlaceholder("Loading".to_string())
        );
        assert_ne!(
            DisplayAction::ShowPlaceholder("Loading".to_string()),
            DisplayAction::ShowPlaceholder("Other".to_string())
        );
    }

    #[test]
    fn test_display_action_debug() {
        let action = DisplayAction::Show;
        let debug = format!("{:?}", action);
        assert!(debug.contains("Show"));
    }

    #[test]
    fn test_display_action_clone() {
        let action = DisplayAction::ShowPlaceholder("test".to_string());
        let cloned = action.clone();
        assert_eq!(action, cloned);
    }

    // System capabilities tests

    #[test]
    fn test_system_capabilities_default() {
        let caps = SystemCapabilities::default();
        assert!(!caps.has_battery);
        assert!(!caps.has_nvidia);
        assert!(!caps.has_psi);
    }

    #[test]
    fn test_system_capabilities_clone() {
        let caps = SystemCapabilities {
            has_battery: true,
            has_nvidia: true,
            has_sensors: true,
            has_psi: true,
            ..Default::default()
        };
        let cloned = caps;
        assert!(cloned.has_battery);
        assert!(cloned.has_nvidia);
    }

    // Terminal size tests

    #[test]
    fn test_terminal_size_default() {
        let size = TerminalSize::default();
        assert_eq!(size.width, 0);
        assert_eq!(size.height, 0);
    }

    #[test]
    fn test_terminal_size_clone() {
        let size = TerminalSize {
            width: 120,
            height: 40,
        };
        let cloned = size;
        assert_eq!(cloned.width, 120);
        assert_eq!(cloned.height, 40);
    }

    // Data availability tests

    #[test]
    fn test_data_availability_default() {
        let data = DataAvailability::default();
        assert!(!data.psi_available);
        assert!(!data.gpu_available);
        assert!(!data.battery_available);
    }

    #[test]
    fn test_data_availability_clone() {
        let data = DataAvailability {
            psi_available: true,
            gpu_available: true,
            battery_available: true,
            sensors_available: true,
            treemap_ready: true,
            sensor_count: 10,
            ..Default::default()
        };
        let cloned = data;
        assert_eq!(cloned.sensor_count, 10);
    }

    // DisplayRules trait tests

    #[test]
    fn test_psi_panel_id() {
        let rules = PsiDisplayRules;
        assert_eq!(rules.panel_id(), "psi");
    }

    #[test]
    fn test_sensors_panel_id() {
        let rules = SensorsDisplayRules;
        assert_eq!(rules.panel_id(), "sensors");
    }

    #[test]
    fn test_gpu_panel_id() {
        let rules = GpuDisplayRules;
        assert_eq!(rules.panel_id(), "gpu");
    }

    #[test]
    fn test_battery_panel_id() {
        let rules = BatteryDisplayRules;
        assert_eq!(rules.panel_id(), "battery");
    }

    #[test]
    fn test_files_panel_id() {
        let rules = FilesDisplayRules;
        assert_eq!(rules.panel_id(), "files");
    }

    #[test]
    fn test_gpu_shows_when_available() {
        let rules = GpuDisplayRules;
        let system = SystemCapabilities::default();
        let ctx = DisplayContext {
            system: &system,
            terminal: TerminalSize::default(),
            data: DataAvailability {
                gpu_available: true,
                ..Default::default()
            },
        };
        assert_eq!(rules.evaluate(&ctx), DisplayAction::Show);
    }

    #[test]
    fn test_gpu_hides_when_unavailable() {
        let rules = GpuDisplayRules;
        let system = SystemCapabilities::default();
        let ctx = DisplayContext {
            system: &system,
            terminal: TerminalSize::default(),
            data: DataAvailability {
                gpu_available: false,
                ..Default::default()
            },
        };
        assert_eq!(rules.evaluate(&ctx), DisplayAction::Hide);
    }

    #[test]
    fn test_sensors_hides_when_no_sensors() {
        let rules = SensorsDisplayRules;
        let system = SystemCapabilities {
            has_sensors: false,
            ..Default::default()
        };
        let ctx = DisplayContext {
            system: &system,
            terminal: TerminalSize::default(),
            data: DataAvailability::default(),
        };
        assert_eq!(rules.evaluate(&ctx), DisplayAction::Hide);
    }

    #[test]
    fn test_sensors_shows_with_many() {
        let rules = SensorsDisplayRules;
        let system = SystemCapabilities {
            has_sensors: true,
            ..Default::default()
        };
        let ctx = DisplayContext {
            system: &system,
            terminal: TerminalSize::default(),
            data: DataAvailability {
                sensors_available: true,
                sensor_count: 10,
                ..Default::default()
            },
        };
        assert_eq!(rules.evaluate(&ctx), DisplayAction::Show);
    }

    #[test]
    fn test_battery_shows_when_available() {
        let rules = BatteryDisplayRules;
        let system = SystemCapabilities {
            has_battery: true,
            ..Default::default()
        };
        let ctx = DisplayContext {
            system: &system,
            terminal: TerminalSize::default(),
            data: DataAvailability {
                battery_available: true,
                ..Default::default()
            },
        };
        assert_eq!(rules.evaluate(&ctx), DisplayAction::Show);
    }

    #[test]
    fn test_files_shows_when_ready() {
        let rules = FilesDisplayRules;
        let system = SystemCapabilities::default();
        let ctx = DisplayContext {
            system: &system,
            terminal: TerminalSize::default(),
            data: DataAvailability {
                treemap_ready: true,
                ..Default::default()
            },
        };
        assert_eq!(rules.evaluate(&ctx), DisplayAction::Show);
    }

    // Additional formatting tests

    #[test]
    fn test_format_duration_seconds() {
        assert_eq!(format_duration(30), "30s");
        assert_eq!(format_duration(59), "59s");
    }

    #[test]
    fn test_format_duration_minutes() {
        assert_eq!(format_duration(60), "1m");
        assert_eq!(format_duration(90), "1:30");
        assert_eq!(format_duration(3599), "59:59");
    }

    #[test]
    fn test_format_duration_hours() {
        assert_eq!(format_duration(3600), "1h");
        assert_eq!(format_duration(7200), "2h");
        assert_eq!(format_duration(5400), "1:30");
    }

    #[test]
    fn test_format_duration_days() {
        assert_eq!(format_duration(86400), "1d");
        assert_eq!(format_duration(90000), "1d 1h");
    }

    #[test]
    fn test_format_duration_compact() {
        assert_eq!(format_duration_compact(30), "30s");
        assert_eq!(format_duration_compact(90), "1m");
        assert_eq!(format_duration_compact(7200), "2h");
        assert_eq!(format_duration_compact(90000), "1d");
    }

    #[test]
    fn test_format_freq_mhz() {
        assert_eq!(format_freq_mhz(800), "800M");
        assert_eq!(format_freq_mhz(1000), "1.00G");
        assert_eq!(format_freq_mhz(3600), "3.60G");
        assert_eq!(format_freq_mhz(10000), "10.0G");
    }

    #[test]
    fn test_format_temp_c() {
        assert_eq!(format_temp_c(45.5), "45.5°C");
        assert_eq!(format_temp_c(100.0), "100°C");
    }

    #[test]
    fn test_format_percent_clamped() {
        assert!(format_percent_clamped(150.0).contains("100"));
        assert!(format_percent_clamped(-10.0).contains('0'));
    }

    #[test]
    fn test_format_percent_fixed() {
        let result = format_percent_fixed(5.0, 8);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_truncate_strategy_debug() {
        let strategy = TruncateStrategy::End;
        let debug = format!("{:?}", strategy);
        assert!(debug.contains("End"));
    }

    #[test]
    fn test_truncate_strategy_clone() {
        let strategy = TruncateStrategy::Middle;
        let cloned = strategy;
        assert_eq!(strategy, cloned);
    }
}
