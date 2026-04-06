//! Bug Hunter Configuration
//!
//! Handles loading and parsing of `.pmat/bug-hunter.toml` configuration files.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Bug Hunter configuration loaded from `.pmat/bug-hunter.toml`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BugHunterConfig {
    /// Allowlist entries for intentional patterns
    #[serde(default)]
    pub allow: Vec<AllowEntry>,

    /// Custom pattern definitions
    #[serde(default)]
    pub patterns: Vec<CustomPattern>,

    /// Trend tracking settings
    #[serde(default)]
    pub trend: TrendConfig,
}

/// An allowlist entry marking a pattern as intentional.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllowEntry {
    /// File glob pattern (e.g., "src/optim/*.rs")
    pub file: String,

    /// Pattern to allow (e.g., "unimplemented")
    pub pattern: String,

    /// Reason for allowing (documentation)
    #[serde(default)]
    pub reason: String,

    /// Optional: only allow in specific line ranges
    #[serde(default)]
    pub lines: Option<LineRange>,
}

/// Line range for scoped allowlist entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineRange {
    pub start: usize,
    pub end: usize,
}

/// A custom pattern definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomPattern {
    /// The pattern to match (regex or literal)
    pub pattern: String,

    /// Category for the finding
    #[serde(default = "default_category")]
    pub category: String,

    /// Severity level
    #[serde(default = "default_severity")]
    pub severity: String,

    /// Suspiciousness score (0.0-1.0)
    #[serde(default = "default_suspiciousness")]
    pub suspiciousness: f64,

    /// Optional description
    #[serde(default)]
    pub description: String,

    /// File glob to limit scope (optional)
    #[serde(default)]
    pub file_glob: Option<String>,

    /// Language filter (optional: "rust", "python", "typescript", "go")
    #[serde(default)]
    pub language: Option<String>,
}

fn default_category() -> String {
    "Custom".to_string()
}

fn default_severity() -> String {
    "Medium".to_string()
}

fn default_suspiciousness() -> f64 {
    0.5
}

/// Trend tracking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendConfig {
    /// Enable automatic trend snapshots
    #[serde(default)]
    pub enabled: bool,

    /// Snapshot interval in days
    #[serde(default = "default_interval")]
    pub interval_days: u32,

    /// Maximum snapshots to retain
    #[serde(default = "default_max_snapshots")]
    pub max_snapshots: usize,
}

fn default_interval() -> u32 {
    7
}

fn default_max_snapshots() -> usize {
    52
}

impl Default for TrendConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_days: default_interval(),
            max_snapshots: default_max_snapshots(),
        }
    }
}

impl BugHunterConfig {
    /// Load configuration from a project path.
    ///
    /// Looks for `.pmat/bug-hunter.toml` in the project root.
    pub fn load(project_path: &Path) -> Self {
        let config_path = project_path.join(".pmat").join("bug-hunter.toml");
        if config_path.exists() {
            match std::fs::read_to_string(&config_path) {
                Ok(content) => match toml::from_str(&content) {
                    Ok(config) => return config,
                    Err(e) => {
                        eprintln!("Warning: Failed to parse {}: {}", config_path.display(), e);
                    }
                },
                Err(e) => {
                    eprintln!("Warning: Failed to read {}: {}", config_path.display(), e);
                }
            }
        }
        Self::default()
    }

    /// Check if a finding should be allowed (skipped).
    pub fn is_allowed(&self, file_path: &Path, pattern: &str, line: usize) -> bool {
        let file_str = file_path.to_string_lossy();

        for entry in &self.allow {
            // Check pattern match
            if !entry.pattern.eq_ignore_ascii_case(pattern) && entry.pattern != "*" {
                continue;
            }

            // Check file glob
            if !glob_match(&entry.file, &file_str) {
                continue;
            }

            // Check line range if specified
            if let Some(ref range) = entry.lines {
                if line < range.start || line > range.end {
                    continue;
                }
            }

            return true;
        }

        false
    }
}

/// Simple glob matching (supports * and **).
fn glob_match(pattern: &str, path: &str) -> bool {
    if pattern == "*" || pattern == "**" {
        return true;
    }

    // Convert glob to regex-like matching
    let pattern_parts: Vec<&str> = pattern.split('/').collect();
    let path_parts: Vec<&str> = path.split('/').collect();

    glob_match_parts(&pattern_parts, &path_parts)
}

fn glob_match_parts(pattern: &[&str], path: &[&str]) -> bool {
    let Some((&p, pattern_rest)) = pattern.split_first() else {
        return path.is_empty();
    };

    if p == "**" {
        return glob_match_doublestar(pattern_rest, path);
    }

    let Some((&path_first, path_rest)) = path.split_first() else {
        return false;
    };

    segment_matches(p, path_first) && glob_match_parts(pattern_rest, path_rest)
}

/// Handle ** glob pattern: matches zero or more path segments
fn glob_match_doublestar(pattern_rest: &[&str], path: &[&str]) -> bool {
    for i in 0..=path.len() {
        if glob_match_parts(pattern_rest, path.get(i..).unwrap_or(&[])) {
            return true;
        }
    }
    false
}

fn segment_matches(pattern: &str, segment: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if !pattern.contains('*') {
        return pattern == segment;
    }

    // Simple wildcard matching
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 2 {
        let (prefix, suffix) = (parts[0], parts[1]);
        return segment.starts_with(prefix) && segment.ends_with(suffix);
    }

    // Fallback to exact match for complex patterns
    pattern == segment
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_match_simple() {
        assert!(glob_match("src/*.rs", "src/main.rs"));
        assert!(glob_match("src/*.rs", "src/lib.rs"));
        assert!(!glob_match("src/*.rs", "src/foo/bar.rs"));
    }

    #[test]
    fn test_glob_match_double_star() {
        assert!(glob_match("src/**/*.rs", "src/main.rs"));
        assert!(glob_match("src/**/*.rs", "src/foo/bar.rs"));
        assert!(glob_match("src/**/*.rs", "src/foo/bar/baz.rs"));
        assert!(!glob_match("src/**/*.rs", "test/main.rs"));
    }

    #[test]
    fn test_glob_match_star() {
        assert!(glob_match("*", "anything"));
        assert!(glob_match("**", "any/path/here"));
    }

    #[test]
    fn test_is_allowed() {
        let config = BugHunterConfig {
            allow: vec![AllowEntry {
                file: "src/optim/*.rs".to_string(),
                pattern: "unimplemented".to_string(),
                reason: "Batch optimizers don't support step()".to_string(),
                lines: None,
            }],
            ..Default::default()
        };

        assert!(config.is_allowed(Path::new("src/optim/admm.rs"), "unimplemented", 100));
        assert!(!config.is_allowed(Path::new("src/main.rs"), "unimplemented", 100));
        assert!(!config.is_allowed(Path::new("src/optim/admm.rs"), "placeholder", 100));
    }

    #[test]
    fn test_is_allowed_with_line_range() {
        let config = BugHunterConfig {
            allow: vec![AllowEntry {
                file: "src/foo.rs".to_string(),
                pattern: "TODO".to_string(),
                reason: "Known issue".to_string(),
                lines: Some(LineRange { start: 10, end: 20 }),
            }],
            ..Default::default()
        };

        assert!(config.is_allowed(Path::new("src/foo.rs"), "TODO", 15));
        assert!(!config.is_allowed(Path::new("src/foo.rs"), "TODO", 5));
        assert!(!config.is_allowed(Path::new("src/foo.rs"), "TODO", 25));
    }

    #[test]
    fn test_parse_config() {
        let toml = r#"
[[allow]]
file = "src/optim/*.rs"
pattern = "unimplemented"
reason = "Batch optimizers"

[[patterns]]
pattern = "PERF-TODO"
category = "PerformanceDebt"
severity = "High"
suspiciousness = 0.8
"#;

        let config: BugHunterConfig = toml::from_str(toml).expect("toml parse failed");
        assert_eq!(config.allow.len(), 1);
        assert_eq!(config.patterns.len(), 1);
        assert_eq!(config.patterns[0].pattern, "PERF-TODO");
        assert_eq!(config.patterns[0].suspiciousness, 0.8);
    }

    // ================================================================
    // Additional coverage tests
    // ================================================================

    #[test]
    fn test_load_nonexistent_path() {
        // load() with a path that has no .pmat/bug-hunter.toml should return default
        let config = BugHunterConfig::load(Path::new("/absolutely/nonexistent/path"));
        assert!(config.allow.is_empty());
        assert!(config.patterns.is_empty());
        assert!(!config.trend.enabled);
    }

    #[test]
    fn test_load_valid_toml() {
        use std::fs;
        let tmp = std::env::temp_dir().join("batuta_test_config_load_valid");
        let pmat_dir = tmp.join(".pmat");
        let _ = fs::create_dir_all(&pmat_dir);

        let toml_content = r#"
[[allow]]
file = "src/**/*.rs"
pattern = "todo"
reason = "Known issues"

[trend]
enabled = true
interval_days = 14
max_snapshots = 100
"#;
        fs::write(pmat_dir.join("bug-hunter.toml"), toml_content).expect("fs write failed");

        let config = BugHunterConfig::load(&tmp);
        assert_eq!(config.allow.len(), 1);
        assert_eq!(config.allow[0].pattern, "todo");
        assert!(config.trend.enabled);
        assert_eq!(config.trend.interval_days, 14);
        assert_eq!(config.trend.max_snapshots, 100);

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_load_invalid_toml() {
        use std::fs;
        let tmp = std::env::temp_dir().join("batuta_test_config_load_invalid");
        let pmat_dir = tmp.join(".pmat");
        let _ = fs::create_dir_all(&pmat_dir);

        // Write invalid TOML content
        fs::write(pmat_dir.join("bug-hunter.toml"), "{{invalid toml!!!").expect("fs write failed");

        // Should print warning and return default
        let config = BugHunterConfig::load(&tmp);
        assert!(config.allow.is_empty());
        assert!(config.patterns.is_empty());

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_load_unreadable_file() {
        // Path exists but .pmat dir exists with a directory instead of a file
        use std::fs;
        let tmp = std::env::temp_dir().join("batuta_test_config_load_unreadable");
        let pmat_dir = tmp.join(".pmat");
        let toml_as_dir = pmat_dir.join("bug-hunter.toml");
        let _ = fs::create_dir_all(&toml_as_dir); // Create as directory, not file

        // exists() returns true for directories, but read_to_string will fail
        let config = BugHunterConfig::load(&tmp);
        assert!(config.allow.is_empty());

        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_default_config() {
        let config = BugHunterConfig::default();
        assert!(config.allow.is_empty());
        assert!(config.patterns.is_empty());
        assert!(!config.trend.enabled);
        assert_eq!(config.trend.interval_days, 7);
        assert_eq!(config.trend.max_snapshots, 52);
    }

    #[test]
    fn test_trend_config_default() {
        let trend = TrendConfig::default();
        assert!(!trend.enabled);
        assert_eq!(trend.interval_days, 7);
        assert_eq!(trend.max_snapshots, 52);
    }

    #[test]
    fn test_custom_pattern_defaults() {
        let toml = r#"
[[patterns]]
pattern = "FIXME"
"#;

        let config: BugHunterConfig = toml::from_str(toml).expect("toml parse failed");
        let p = &config.patterns[0];
        assert_eq!(p.pattern, "FIXME");
        assert_eq!(p.category, "Custom");
        assert_eq!(p.severity, "Medium");
        assert!((p.suspiciousness - 0.5).abs() < f64::EPSILON);
        assert_eq!(p.description, "");
        assert!(p.file_glob.is_none());
        assert!(p.language.is_none());
    }

    #[test]
    fn test_custom_pattern_full_fields() {
        let toml = r#"
[[patterns]]
pattern = "HACK"
category = "TechDebt"
severity = "Critical"
suspiciousness = 0.9
description = "Hack workaround"
file_glob = "src/**/*.rs"
language = "rust"
"#;

        let config: BugHunterConfig = toml::from_str(toml).expect("toml parse failed");
        let p = &config.patterns[0];
        assert_eq!(p.pattern, "HACK");
        assert_eq!(p.category, "TechDebt");
        assert_eq!(p.severity, "Critical");
        assert!((p.suspiciousness - 0.9).abs() < f64::EPSILON);
        assert_eq!(p.description, "Hack workaround");
        assert_eq!(p.file_glob.as_deref(), Some("src/**/*.rs"));
        assert_eq!(p.language.as_deref(), Some("rust"));
    }

    #[test]
    fn test_is_allowed_wildcard_pattern() {
        let config = BugHunterConfig {
            allow: vec![AllowEntry {
                file: "**".to_string(),
                pattern: "*".to_string(),
                reason: "Allow everything".to_string(),
                lines: None,
            }],
            ..Default::default()
        };

        // Wildcard pattern "*" should match any pattern
        assert!(config.is_allowed(Path::new("src/anything.rs"), "any_pattern", 1));
        assert!(config.is_allowed(Path::new("tests/foo.rs"), "different", 999));
    }

    #[test]
    fn test_is_allowed_case_insensitive_pattern() {
        let config = BugHunterConfig {
            allow: vec![AllowEntry {
                file: "src/*.rs".to_string(),
                pattern: "TODO".to_string(),
                reason: "Known".to_string(),
                lines: None,
            }],
            ..Default::default()
        };

        // eq_ignore_ascii_case should match
        assert!(config.is_allowed(Path::new("src/main.rs"), "todo", 1));
        assert!(config.is_allowed(Path::new("src/main.rs"), "Todo", 1));
        assert!(config.is_allowed(Path::new("src/main.rs"), "TODO", 1));
    }

    #[test]
    fn test_is_allowed_no_entries() {
        let config = BugHunterConfig::default();
        assert!(!config.is_allowed(Path::new("src/main.rs"), "TODO", 1));
    }

    #[test]
    fn test_is_allowed_multiple_entries() {
        let config = BugHunterConfig {
            allow: vec![
                AllowEntry {
                    file: "src/a.rs".to_string(),
                    pattern: "TODO".to_string(),
                    reason: "".to_string(),
                    lines: None,
                },
                AllowEntry {
                    file: "src/b.rs".to_string(),
                    pattern: "FIXME".to_string(),
                    reason: "".to_string(),
                    lines: None,
                },
            ],
            ..Default::default()
        };

        assert!(config.is_allowed(Path::new("src/a.rs"), "TODO", 1));
        assert!(!config.is_allowed(Path::new("src/a.rs"), "FIXME", 1));
        assert!(config.is_allowed(Path::new("src/b.rs"), "FIXME", 1));
        assert!(!config.is_allowed(Path::new("src/b.rs"), "TODO", 1));
    }

    #[test]
    fn test_glob_match_exact_segment() {
        // No wildcards - exact match
        assert!(glob_match("src/main.rs", "src/main.rs"));
        assert!(!glob_match("src/main.rs", "src/lib.rs"));
    }

    #[test]
    fn test_glob_match_empty_pattern() {
        // Empty pattern should match empty path
        assert!(glob_match("", ""));
        // Empty pattern should not match non-empty path
        assert!(!glob_match("", "src/main.rs"));
    }

    #[test]
    fn test_glob_match_double_star_at_end() {
        // ** at end matches anything remaining
        assert!(glob_match("src/**", "src/foo.rs"));
        assert!(glob_match("src/**", "src/foo/bar.rs"));
        assert!(glob_match("src/**", "src/foo/bar/baz.rs"));
    }

    #[test]
    fn test_glob_match_double_star_at_beginning() {
        assert!(glob_match("**/main.rs", "src/main.rs"));
        assert!(glob_match("**/main.rs", "deep/nested/main.rs"));
        assert!(glob_match("**/main.rs", "main.rs")); // zero segments before
    }

    #[test]
    fn test_glob_match_star_segment_prefix_suffix() {
        // Pattern "*.rs" matches any segment ending in .rs
        assert!(glob_match("*.rs", "main.rs"));
        assert!(glob_match("*.rs", "lib.rs"));
        assert!(!glob_match("*.rs", "main.py"));
    }

    #[test]
    fn test_glob_match_deeper_paths() {
        assert!(glob_match("a/b/c", "a/b/c"));
        assert!(!glob_match("a/b/c", "a/b/d"));
        assert!(!glob_match("a/b/c", "a/b"));
        assert!(!glob_match("a/b", "a/b/c"));
    }

    #[test]
    fn test_segment_matches_no_wildcard() {
        assert!(segment_matches("main.rs", "main.rs"));
        assert!(!segment_matches("main.rs", "lib.rs"));
    }

    #[test]
    fn test_segment_matches_star() {
        assert!(segment_matches("*", "anything"));
        assert!(segment_matches("*", ""));
    }

    #[test]
    fn test_segment_matches_prefix_suffix() {
        assert!(segment_matches("test_*.rs", "test_main.rs"));
        assert!(segment_matches("test_*.rs", "test_.rs")); // empty middle
        assert!(!segment_matches("test_*.rs", "main.rs"));
    }

    #[test]
    fn test_segment_matches_complex_pattern() {
        // Multiple wildcards fall back to exact match
        assert!(segment_matches("a*b*c", "a*b*c")); // exact match of the pattern string
        assert!(!segment_matches("a*b*c", "aXbYc")); // won't match - fallback is exact
    }

    #[test]
    fn test_glob_match_double_star_zero_segments() {
        // ** matches zero segments
        assert!(glob_match("**/src/*.rs", "src/main.rs"));
        // ** matches one segment
        assert!(glob_match("**/src/*.rs", "foo/src/main.rs"));
        // ** matches multiple segments
        assert!(glob_match("**/src/*.rs", "a/b/src/main.rs"));
    }

    #[test]
    fn test_allow_entry_line_range_boundaries() {
        let config = BugHunterConfig {
            allow: vec![AllowEntry {
                file: "src/foo.rs".to_string(),
                pattern: "TODO".to_string(),
                reason: "".to_string(),
                lines: Some(LineRange { start: 10, end: 20 }),
            }],
            ..Default::default()
        };

        // Exactly at boundaries
        assert!(config.is_allowed(Path::new("src/foo.rs"), "TODO", 10)); // start boundary
        assert!(config.is_allowed(Path::new("src/foo.rs"), "TODO", 20)); // end boundary
        assert!(!config.is_allowed(Path::new("src/foo.rs"), "TODO", 9)); // just before
        assert!(!config.is_allowed(Path::new("src/foo.rs"), "TODO", 21)); // just after
    }

    #[test]
    fn test_parse_config_with_line_range() {
        let toml = r#"
[[allow]]
file = "src/main.rs"
pattern = "HACK"
reason = "Temporary workaround"

[allow.lines]
start = 50
end = 75
"#;

        let config: BugHunterConfig = toml::from_str(toml).expect("toml parse failed");
        assert_eq!(config.allow.len(), 1);
        let entry = &config.allow[0];
        assert!(entry.lines.is_some());
        let range = entry.lines.as_ref().expect("unexpected failure");
        assert_eq!(range.start, 50);
        assert_eq!(range.end, 75);
    }

    #[test]
    fn test_parse_config_with_trend() {
        let toml = r#"
[trend]
enabled = true
interval_days = 30
max_snapshots = 24
"#;

        let config: BugHunterConfig = toml::from_str(toml).expect("toml parse failed");
        assert!(config.trend.enabled);
        assert_eq!(config.trend.interval_days, 30);
        assert_eq!(config.trend.max_snapshots, 24);
    }

    #[test]
    fn test_parse_config_empty_toml() {
        let config: BugHunterConfig = toml::from_str("").expect("toml parse failed");
        assert!(config.allow.is_empty());
        assert!(config.patterns.is_empty());
        assert!(!config.trend.enabled);
        assert_eq!(config.trend.interval_days, 7);
        assert_eq!(config.trend.max_snapshots, 52);
    }

    #[test]
    fn test_glob_match_parts_empty_pattern_empty_path() {
        // Both empty -> true
        assert!(glob_match_parts(&[], &[]));
    }

    #[test]
    fn test_glob_match_parts_pattern_longer_than_path() {
        // Pattern has segments but path is empty
        assert!(!glob_match_parts(&["src", "main.rs"], &[]));
    }

    #[test]
    fn test_glob_match_doublestar_only() {
        // Just ** matches anything
        assert!(glob_match_doublestar(&[], &[]));
        assert!(glob_match_doublestar(&[], &["a", "b", "c"]));
    }

    #[test]
    fn test_glob_match_doublestar_with_rest() {
        // ** followed by pattern
        assert!(glob_match_doublestar(&["*.rs"], &["main.rs"]));
        assert!(glob_match_doublestar(&["*.rs"], &["src", "main.rs"]));
        assert!(!glob_match_doublestar(&["*.rs"], &["main.py"]));
    }
}
