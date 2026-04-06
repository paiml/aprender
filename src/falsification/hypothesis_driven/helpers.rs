//! Helper functions for hypothesis-driven checks.

use std::path::Path;

/// Check if project files contain any of the given patterns.
pub(super) fn check_for_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    crate::falsification::helpers::files_contain_pattern(
        project_path,
        &["src/**/*.rs", "**/*.yaml", "**/*.toml", "**/*.json", "**/*.md"],
        patterns,
    )
}

/// Check if CI configuration contains any of the given patterns.
pub(super) fn check_ci_for_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    crate::falsification::helpers::ci_contains_pattern(project_path, patterns)
}
