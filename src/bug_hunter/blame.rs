//! Git Blame Integration
//!
//! Provides git blame information for findings to help identify who introduced
//! bugs and when, enabling better triage and assignment.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

/// Git blame information for a single line.
#[derive(Debug, Clone, Default)]
pub struct BlameInfo {
    /// Author name
    pub author: String,
    /// Abbreviated commit hash
    pub commit: String,
    /// Date of the commit (YYYY-MM-DD format)
    pub date: String,
}

/// Cache for blame lookups to avoid repeated git calls.
#[derive(Debug, Default)]
pub struct BlameCache {
    /// Cache: (file, line) -> BlameInfo
    cache: HashMap<(String, usize), BlameInfo>,
}

impl BlameCache {
    /// Create a new empty blame cache.
    pub fn new() -> Self {
        Self { cache: HashMap::new() }
    }

    /// Get blame info for a specific file and line, using cache.
    pub fn get_blame(
        &mut self,
        project_path: &Path,
        file: &Path,
        line: usize,
    ) -> Option<BlameInfo> {
        let file_str = file.to_string_lossy().to_string();
        let key = (file_str.clone(), line);

        if let Some(cached) = self.cache.get(&key) {
            return Some(cached.clone());
        }

        // Not in cache, fetch from git
        let blame = get_blame_for_line(project_path, file, line)?;
        self.cache.insert(key, blame.clone());
        Some(blame)
    }

    /// Prefetch blame info for multiple lines in a file (batch optimization).
    pub fn prefetch_file(&mut self, project_path: &Path, file: &Path, lines: &[usize]) {
        if lines.is_empty() {
            return;
        }

        let file_str = file.to_string_lossy().to_string();

        // Check which lines we don't have cached
        let uncached: Vec<usize> = lines
            .iter()
            .filter(|&&l| !self.cache.contains_key(&(file_str.clone(), l)))
            .copied()
            .collect();

        if uncached.is_empty() {
            return;
        }

        // Batch fetch all blame info for the file
        if let Some(all_blames) = get_blame_for_file(project_path, file) {
            for (line, blame) in all_blames {
                self.cache.insert((file_str.clone(), line), blame);
            }
        }
    }
}

/// Get blame info for a specific line using git blame.
fn get_blame_for_line(project_path: &Path, file: &Path, line: usize) -> Option<BlameInfo> {
    let output = Command::new("git")
        .current_dir(project_path)
        .args([
            "blame",
            "-L",
            &format!("{},{}", line, line),
            "--porcelain",
            &file.to_string_lossy(),
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    parse_porcelain_blame(&String::from_utf8_lossy(&output.stdout))
}

/// Get blame info for an entire file (more efficient for multiple lines).
fn get_blame_for_file(project_path: &Path, file: &Path) -> Option<HashMap<usize, BlameInfo>> {
    let output = Command::new("git")
        .current_dir(project_path)
        .args(["blame", "--porcelain", &file.to_string_lossy()])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    Some(parse_porcelain_blame_full(&String::from_utf8_lossy(&output.stdout)))
}

/// Parse porcelain blame output for a single line.
fn parse_porcelain_blame(output: &str) -> Option<BlameInfo> {
    let mut author = String::new();
    let mut commit = String::new();
    let mut date = String::new();

    for line in output.lines() {
        if line.starts_with("author ") {
            author = line.strip_prefix("author ").unwrap_or("").to_string();
        } else if line.starts_with("author-time ") {
            // Convert Unix timestamp to YYYY-MM-DD
            if let Ok(timestamp) = line.strip_prefix("author-time ").unwrap_or("0").parse::<i64>() {
                date = format_timestamp(timestamp);
            }
        } else if commit.is_empty() && line.len() >= 40 {
            // Commit lines start with 40-char hex hash: <hash> <orig_line> <final_line> [count]
            let first_40: String = line.chars().take(40).collect();
            if first_40.chars().all(|c| c.is_ascii_hexdigit()) {
                commit = line[..7.min(line.len())].to_string();
            }
        }
    }

    if author.is_empty() && commit.is_empty() {
        return None;
    }

    Some(BlameInfo { author, commit, date })
}

/// Parse porcelain blame output for full file.
/// Check if a line is a porcelain blame commit header (40-char hex prefix).
fn is_commit_header(line: &str) -> bool {
    line.len() >= 40 && line.chars().take(40).all(|c| c.is_ascii_hexdigit())
}

/// Insert a blame entry if we have accumulated valid state.
fn flush_blame_entry(
    results: &mut HashMap<usize, BlameInfo>,
    line_num: usize,
    author: &str,
    commit: &str,
    date: &str,
) {
    if line_num > 0 && !commit.is_empty() {
        results.insert(
            line_num,
            BlameInfo {
                author: author.to_string(),
                commit: commit.to_string(),
                date: date.to_string(),
            },
        );
    }
}

fn parse_porcelain_blame_full(output: &str) -> HashMap<usize, BlameInfo> {
    let mut results = HashMap::new();
    let mut current_line = 0usize;
    let mut current_author = String::new();
    let mut current_commit = String::new();
    let mut current_date = String::new();

    for line in output.lines() {
        if is_commit_header(line) {
            flush_blame_entry(
                &mut results,
                current_line,
                &current_author,
                &current_commit,
                &current_date,
            );

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                current_commit = parts[0][..7.min(parts[0].len())].to_string();
                if let Ok(line_num) = parts[2].parse::<usize>() {
                    current_line = line_num;
                }
            }
            current_author.clear();
            current_date.clear();
        } else if let Some(author) = line.strip_prefix("author ") {
            current_author = author.to_string();
        } else if let Some(ts_str) = line.strip_prefix("author-time ") {
            if let Ok(timestamp) = ts_str.parse::<i64>() {
                current_date = format_timestamp(timestamp);
            }
        }
    }

    flush_blame_entry(&mut results, current_line, &current_author, &current_commit, &current_date);
    results
}

/// Format Unix timestamp as YYYY-MM-DD.
fn format_timestamp(timestamp: i64) -> String {
    if timestamp < 0 {
        return String::new();
    }

    // Convert to date using chrono if available, otherwise use simple calculation
    let secs_per_day = 86400u64;
    let days_since_epoch = (timestamp as u64) / secs_per_day;

    // Simple date calculation (approximate, good enough for display)
    let mut year = 1970i32;
    let mut remaining_days = days_since_epoch as i32;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let mut month = 1;
    let days_in_months = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    for days in &days_in_months {
        if remaining_days < *days {
            break;
        }
        remaining_days -= days;
        month += 1;
    }

    let day = remaining_days + 1;
    format!("{:04}-{:02}-{:02}", year, month, day)
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_porcelain_blame_basic() {
        let output = r#"abc1234567890123456789012345678901234567 1 1 1
author John Doe
author-mail <john@example.com>
author-time 1704067200
author-tz +0000
committer John Doe
committer-mail <john@example.com>
committer-time 1704067200
committer-tz +0000
summary Initial commit
filename src/main.rs
	fn main() {}
"#;

        let blame = parse_porcelain_blame(output).expect("unexpected failure");
        assert_eq!(blame.author, "John Doe");
        assert_eq!(blame.commit, "abc1234");
        assert_eq!(blame.date, "2024-01-01");
    }

    #[test]
    fn test_parse_porcelain_blame_empty() {
        let output = "";
        assert!(parse_porcelain_blame(output).is_none());
    }

    #[test]
    fn test_format_timestamp() {
        // 2024-01-01 00:00:00 UTC
        assert_eq!(format_timestamp(1704067200), "2024-01-01");
        // 2000-06-15
        assert_eq!(format_timestamp(961027200), "2000-06-15");
        // 1970-01-01
        assert_eq!(format_timestamp(0), "1970-01-01");
    }

    #[test]
    fn test_is_leap_year() {
        assert!(is_leap_year(2000)); // Divisible by 400
        assert!(is_leap_year(2024)); // Divisible by 4, not by 100
        assert!(!is_leap_year(1900)); // Divisible by 100, not by 400
        assert!(!is_leap_year(2023)); // Not divisible by 4
    }

    #[test]
    fn test_blame_cache_new() {
        let cache = BlameCache::new();
        assert!(cache.cache.is_empty());
    }

    #[test]
    fn test_parse_porcelain_blame_full() {
        let output = r#"abc1234567890123456789012345678901234567 1 1 1
author Alice
author-time 1704067200
summary Line 1
filename test.rs
	line 1 content
def5678901234567890123456789012345678901 2 2 1
author Bob
author-time 1704153600
summary Line 2
filename test.rs
	line 2 content
"#;

        let blames = parse_porcelain_blame_full(output);
        assert_eq!(blames.len(), 2);
        assert_eq!(blames.get(&1).expect("key not found").author, "Alice");
        assert_eq!(blames.get(&2).expect("key not found").author, "Bob");
    }

    // ================================================================
    // Additional coverage tests
    // ================================================================

    #[test]
    fn test_format_timestamp_negative() {
        // Negative timestamp should return empty string
        assert_eq!(format_timestamp(-1), "");
        assert_eq!(format_timestamp(-100000), "");
    }

    #[test]
    fn test_format_timestamp_leap_year_feb_29() {
        // Feb 29, 2000 00:00:00 UTC = 951782400
        let result = format_timestamp(951782400);
        assert_eq!(result, "2000-02-29");
    }

    #[test]
    fn test_format_timestamp_dec_31() {
        // Dec 31, 1999 00:00:00 UTC = 946598400
        let result = format_timestamp(946598400);
        assert_eq!(result, "1999-12-31");
    }

    #[test]
    fn test_format_timestamp_end_of_non_leap_feb() {
        // Feb 28, 2023 00:00:00 UTC = 1677542400
        let result = format_timestamp(1677542400);
        assert_eq!(result, "2023-02-28");
    }

    #[test]
    fn test_format_timestamp_various_months() {
        // Test each month boundary to exercise the days_in_months loop:
        // Mar 15, 2024 = 1710460800
        let result = format_timestamp(1710460800);
        assert_eq!(result, "2024-03-15");

        // Jul 1, 2024 = 1719792000
        let result = format_timestamp(1719792000);
        assert_eq!(result, "2024-07-01");

        // Nov 30, 2024 = 1732924800
        let result = format_timestamp(1732924800);
        assert_eq!(result, "2024-11-30");
    }

    #[test]
    fn test_is_leap_year_edge_cases() {
        // Non-century leap years
        assert!(is_leap_year(2004));
        assert!(is_leap_year(2008));
        assert!(is_leap_year(2012));
        assert!(is_leap_year(2016));
        assert!(is_leap_year(2020));

        // Century non-leap years
        assert!(!is_leap_year(1700));
        assert!(!is_leap_year(1800));
        assert!(!is_leap_year(2100));
        assert!(!is_leap_year(2200));

        // Century leap years (divisible by 400)
        assert!(is_leap_year(1600));
        assert!(is_leap_year(2400));
    }

    #[test]
    fn test_parse_porcelain_blame_no_hash_line() {
        // Output with author and author-time but no valid 40-char hex hash line.
        // The function should return Some since author is non-empty, but commit stays empty.
        let output = "author TestAuthor\nauthor-time 1704067200\nshort line\nfilename test.rs\n";
        let result = parse_porcelain_blame(output);
        assert!(result.is_some(), "Should return Some when author is present");
        let blame = result.expect("operation failed");
        assert_eq!(blame.author, "TestAuthor");
        assert_eq!(blame.commit, "", "No hash line means empty commit");
        assert_eq!(blame.date, "2024-01-01");
    }

    #[test]
    fn test_parse_porcelain_blame_no_hash_no_author() {
        // Output with no valid hash line and no author
        let output = "short line\nsome other data\n";
        let result = parse_porcelain_blame(output);
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_porcelain_blame_no_hash_with_author() {
        // Output with author but no valid commit hash - returns Some (author not empty)
        let output = "short line\nauthor Nobody\n";
        let result = parse_porcelain_blame(output);
        assert!(result.is_some());
        let blame = result.expect("operation failed");
        assert_eq!(blame.author, "Nobody");
        assert_eq!(blame.commit, ""); // No hash found
    }

    #[test]
    fn test_parse_porcelain_blame_only_author() {
        // Output with only author, no commit hash
        let output = "author SomeAuthor\n";
        let result = parse_porcelain_blame(output);
        // author is set but commit is empty => still returns Some since author is not empty
        assert!(result.is_some());
        let blame = result.expect("operation failed");
        assert_eq!(blame.author, "SomeAuthor");
        assert_eq!(blame.commit, "");
    }

    #[test]
    fn test_parse_porcelain_blame_non_hex_40char_line() {
        // 40-char line that is NOT all hex digits
        let output = "ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ 1 1 1\nauthor Test\n";
        let result = parse_porcelain_blame(output);
        // The 40-char line has non-hex chars, so commit remains empty
        // but author is set, so result is Some
        assert!(result.is_some());
        let blame = result.expect("operation failed");
        assert_eq!(blame.commit, "");
    }

    #[test]
    fn test_parse_porcelain_blame_invalid_timestamp() {
        let output =
            "abc1234567890123456789012345678901234567 1 1 1\nauthor Test\nauthor-time notanumber\n";
        let result = parse_porcelain_blame(output);
        assert!(result.is_some());
        let blame = result.expect("operation failed");
        assert_eq!(blame.author, "Test");
        assert_eq!(blame.date, ""); // Failed to parse, date stays empty
    }

    #[test]
    fn test_parse_porcelain_blame_full_empty_input() {
        let blames = parse_porcelain_blame_full("");
        assert!(blames.is_empty());
    }

    #[test]
    fn test_parse_porcelain_blame_full_single_entry() {
        let output = "abc1234567890123456789012345678901234567 1 5 1\nauthor Alice\nauthor-time 1704067200\n";
        let blames = parse_porcelain_blame_full(output);
        assert_eq!(blames.len(), 1);
        assert_eq!(blames.get(&5).expect("key not found").author, "Alice");
        assert_eq!(blames.get(&5).expect("key not found").commit, "abc1234");
    }

    #[test]
    fn test_parse_porcelain_blame_full_three_entries() {
        let output = concat!(
            "aaa1234567890123456789012345678901234567 1 1 1\n",
            "author A\n",
            "author-time 1000000000\n",
            "filename f.rs\n",
            "\tline 1\n",
            "bbb1234567890123456789012345678901234567 2 2 1\n",
            "author B\n",
            "author-time 1100000000\n",
            "filename f.rs\n",
            "\tline 2\n",
            "ccc1234567890123456789012345678901234567 3 3 1\n",
            "author C\n",
            "author-time 1200000000\n",
            "filename f.rs\n",
            "\tline 3\n",
        );
        let blames = parse_porcelain_blame_full(output);
        assert_eq!(blames.len(), 3);
        assert_eq!(blames.get(&1).expect("key not found").author, "A");
        assert_eq!(blames.get(&2).expect("key not found").author, "B");
        assert_eq!(blames.get(&3).expect("key not found").author, "C");
    }

    #[test]
    fn test_parse_porcelain_blame_full_invalid_timestamp() {
        let output =
            "abc1234567890123456789012345678901234567 1 3 1\nauthor Test\nauthor-time invalid\n";
        let blames = parse_porcelain_blame_full(output);
        assert_eq!(blames.len(), 1);
        let blame = blames.get(&3).expect("key not found");
        assert_eq!(blame.author, "Test");
        assert_eq!(blame.date, ""); // Failed to parse
    }

    #[test]
    fn test_parse_porcelain_blame_full_commit_line_less_than_3_parts() {
        // Hash line with < 3 whitespace-separated parts
        let output = "abc1234567890123456789012345678901234567 1\n";
        let blames = parse_porcelain_blame_full(output);
        // parts.len() < 3, so current_commit won't be set, current_line stays 0
        assert!(blames.is_empty());
    }

    #[test]
    fn test_blame_cache_direct_insert_and_get() {
        let mut cache = BlameCache::new();
        let key = ("src/main.rs".to_string(), 10);
        let blame = BlameInfo {
            author: "Test".to_string(),
            commit: "abc1234".to_string(),
            date: "2024-01-01".to_string(),
        };
        cache.cache.insert(key, blame);

        // Get blame from cache (cache hit path)
        // Note: get_blame calls git, so we test the cache hit via direct insert
        let cached = cache.cache.get(&("src/main.rs".to_string(), 10));
        assert!(cached.is_some());
        assert_eq!(cached.expect("unexpected failure").author, "Test");
    }

    #[test]
    fn test_blame_info_default() {
        let info = BlameInfo::default();
        assert_eq!(info.author, "");
        assert_eq!(info.commit, "");
        assert_eq!(info.date, "");
    }

    #[test]
    fn test_blame_info_clone() {
        let info = BlameInfo {
            author: "Author".to_string(),
            commit: "abc1234".to_string(),
            date: "2024-01-01".to_string(),
        };
        let cloned = info.clone();
        assert_eq!(info.author, cloned.author);
        assert_eq!(info.commit, cloned.commit);
        assert_eq!(info.date, cloned.date);
    }

    #[test]
    fn test_prefetch_file_empty_lines() {
        let mut cache = BlameCache::new();
        let path = Path::new("/nonexistent/project");
        let file = Path::new("src/main.rs");
        // Empty lines slice - should return early
        cache.prefetch_file(path, file, &[]);
        assert!(cache.cache.is_empty());
    }

    #[test]
    fn test_prefetch_file_all_cached() {
        let mut cache = BlameCache::new();
        let file_str = "src/main.rs".to_string();

        // Pre-populate cache for lines 1,2,3
        for line in 1..=3 {
            cache.cache.insert(
                (file_str.clone(), line),
                BlameInfo {
                    author: format!("Author{}", line),
                    commit: "abc1234".to_string(),
                    date: "2024-01-01".to_string(),
                },
            );
        }

        let path = Path::new("/nonexistent/project");
        let file = Path::new("src/main.rs");
        // All lines already cached - uncached will be empty, should return early
        cache.prefetch_file(path, file, &[1, 2, 3]);
        // Cache should still have 3 entries (no new fetches)
        assert_eq!(cache.cache.len(), 3);
    }

    #[test]
    fn test_get_blame_for_nonexistent_path() {
        let mut cache = BlameCache::new();
        let path = Path::new("/absolutely/nonexistent/project/path");
        let file = Path::new("nonexistent_file.rs");
        let result = cache.get_blame(path, file, 1);
        // Should return None since git blame will fail
        assert!(result.is_none());
    }

    #[test]
    fn test_get_blame_cache_hit() {
        let mut cache = BlameCache::new();
        let file_str = "src/test.rs".to_string();
        let key = (file_str, 42);
        cache.cache.insert(
            key,
            BlameInfo {
                author: "CachedAuthor".to_string(),
                commit: "ccc1234".to_string(),
                date: "2024-06-15".to_string(),
            },
        );

        // This should hit the cache branch
        let path = Path::new("/some/path");
        let file = Path::new("src/test.rs");
        let result = cache.get_blame(path, file, 42);
        assert!(result.is_some());
        let blame = result.expect("operation failed");
        assert_eq!(blame.author, "CachedAuthor");
    }

    #[test]
    fn test_prefetch_file_with_uncached_lines_nonexistent_project() {
        let mut cache = BlameCache::new();
        let path = Path::new("/nonexistent/project/path");
        let file = Path::new("src/main.rs");
        // Lines not in cache, but git blame will fail for nonexistent path
        cache.prefetch_file(path, file, &[1, 2, 3]);
        // No entries should be added since get_blame_for_file returns None
        assert!(cache.cache.is_empty());
    }
}
