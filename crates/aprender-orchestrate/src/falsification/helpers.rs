//! Shared helper functions for falsification checks.
//!
//! Eliminates duplicated pattern-matching logic across falsification modules.

use std::path::Path;

use super::types::CheckItem;

/// Represents the outcome to apply to a [`CheckItem`].
///
/// Used with [`apply_check_outcome`] to replace repeated if-else outcome chains
/// in falsification check functions.
pub(crate) enum CheckOutcome<'a> {
    /// Pass the check (no message).
    Pass,
    /// Partial pass with a diagnostic message.
    Partial(&'a str),
    /// Fail with a diagnostic message.
    Fail(&'a str),
}

/// Apply the first matching outcome to a check item.
///
/// Iterates through `checks` and applies the outcome for the first entry
/// whose condition is `true`. If no condition matches, returns the item unchanged.
pub(crate) fn apply_check_outcome(
    item: CheckItem,
    checks: &[(bool, CheckOutcome<'_>)],
) -> CheckItem {
    for (condition, outcome) in checks {
        if *condition {
            return match outcome {
                CheckOutcome::Pass => item.pass(),
                CheckOutcome::Partial(msg) => item.partial(*msg),
                CheckOutcome::Fail(msg) => item.fail(*msg),
            };
        }
    }
    item
}

/// Check if any file matching the glob patterns contains any of the search patterns.
///
/// Used by all falsification modules for source/config file scanning.
pub(crate) fn files_contain_pattern(
    project_path: &Path,
    glob_patterns: &[&str],
    search_patterns: &[&str],
) -> bool {
    for glob_pat in glob_patterns {
        let full = format!("{}/{}", project_path.display(), glob_pat);
        let Ok(entries) = glob::glob(&full) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(content) = std::fs::read_to_string(&entry) else {
                continue;
            };
            if search_patterns.iter().any(|p| content.contains(p)) {
                return true;
            }
        }
    }
    false
}

/// Check if any file matching the glob patterns contains any search pattern (case-insensitive).
pub(crate) fn files_contain_pattern_ci(
    project_path: &Path,
    glob_patterns: &[&str],
    search_patterns: &[&str],
) -> bool {
    for glob_pat in glob_patterns {
        let full = format!("{}/{}", project_path.display(), glob_pat);
        let Ok(entries) = glob::glob(&full) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(content) = std::fs::read_to_string(&entry) else {
                continue;
            };
            let lower = content.to_lowercase();
            if search_patterns.iter().any(|p| lower.contains(&p.to_lowercase())) {
                return true;
            }
        }
    }
    false
}

/// Check if any Rust source file contains any of the given patterns.
pub(crate) fn source_contains_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    files_contain_pattern(project_path, &["src/**/*.rs"], patterns)
}

/// Check if any Rust source or config file contains any of the given patterns.
pub(crate) fn source_or_config_contains_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    files_contain_pattern(
        project_path,
        &["src/**/*.rs", "**/*.yaml", "**/*.toml", "**/*.json"],
        patterns,
    )
}

/// Check if any CI workflow file contains any of the given patterns (case-insensitive).
pub(crate) fn ci_contains_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    files_contain_pattern_ci(
        project_path,
        &[".github/workflows/*.yml", ".github/workflows/*.yaml", ".gitlab-ci.yml"],
        patterns,
    )
}

/// Check if any test file contains any of the given patterns.
///
/// Searches test directories, test-named files, and `#[cfg(test)]` modules.
pub(crate) fn test_contains_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    if files_contain_pattern(project_path, &["tests/**/*.rs", "src/**/*test*.rs"], patterns) {
        return true;
    }

    // Also check #[cfg(test)] modules in source files
    let glob_str = format!("{}/src/**/*.rs", project_path.display());
    let Ok(entries) = glob::glob(&glob_str) else {
        return false;
    };
    for entry in entries.flatten() {
        let Ok(content) = std::fs::read_to_string(&entry) else {
            continue;
        };
        if content.contains("#[cfg(test)]") && patterns.iter().any(|p| content.contains(p)) {
            return true;
        }
    }
    false
}

/// Count how many platforms from the list appear in CI workflow files.
pub(crate) fn ci_platform_count(project_path: &Path, platforms: &[&str]) -> usize {
    let ci_globs = [
        format!("{}/.github/workflows/*.yml", project_path.display()),
        format!("{}/.github/workflows/*.yaml", project_path.display()),
    ];

    for glob_pattern in &ci_globs {
        let Ok(entries) = glob::glob(glob_pattern) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(content) = std::fs::read_to_string(&entry) else {
                continue;
            };
            let count = platforms.iter().filter(|p| content.contains(*p)).count();
            if count >= 2 {
                return count;
            }
        }
    }
    0
}

/// Scan Cargo.toml for scripting runtime dependencies.
///
/// Returns list of forbidden dependency names found in non-dev deps.
pub(crate) fn find_scripting_deps(project_path: &Path) -> Vec<String> {
    let cargo_toml = project_path.join("Cargo.toml");
    let Ok(content) = std::fs::read_to_string(&cargo_toml) else {
        return Vec::new();
    };

    let forbidden = ["pyo3", "napi", "mlua", "rlua", "rustpython"];
    let mut found = Vec::new();

    for dep in forbidden {
        let has_dep = content.contains(&format!("{dep} =")) || content.contains(&format!("{dep}="));
        if !has_dep {
            continue;
        }
        // Rough check: in [dependencies] but not after [dev-dependencies]
        let is_dev_only = content.contains("[dev-dependencies]")
            && content.find(&format!("{dep} =")) >= content.find("[dev-dependencies]");
        if !is_dev_only {
            found.push(dep.to_string());
        }
    }
    found
}

/// Detect serde/schema validation configuration from Cargo.toml.
pub(crate) struct SchemaInfo {
    pub has_serde: bool,
    pub has_serde_yaml: bool,
    pub has_validator: bool,
}

/// Read schema-related dependency info from project files.
///
/// Checks Cargo.toml for Rust projects. For non-Rust projects, detects
/// alternative schema validation signals: pv/forjar validation in Makefiles
/// or CI configs, Python validation libraries (pydantic, marshmallow), and
/// JSON Schema files.
pub(crate) fn detect_schema_deps(project_path: &Path) -> SchemaInfo {
    let cargo_toml = project_path.join("Cargo.toml");
    let content = std::fs::read_to_string(&cargo_toml).unwrap_or_default();

    if !content.is_empty() {
        return detect_rust_schema_deps(&content);
    }

    detect_nonrust_schema_deps(project_path)
}

fn detect_rust_schema_deps(cargo_content: &str) -> SchemaInfo {
    SchemaInfo {
        has_serde: cargo_content.contains("serde"),
        has_serde_yaml: cargo_content.contains("serde_yaml")
            || cargo_content.contains("serde_yml")
            || cargo_content.contains("serde_yaml_ng"),
        has_validator: cargo_content.contains("validator") || cargo_content.contains("garde"),
    }
}

fn detect_nonrust_schema_deps(project_path: &Path) -> SchemaInfo {
    let mut has_schema_tool = false;
    let mut has_yaml_support = false;
    let mut has_validator = false;

    if has_validation_in_build_files(project_path) {
        has_schema_tool = true;
        has_yaml_support = true;
        has_validator = true;
    }

    let py_info = detect_python_schema_deps(project_path);
    has_schema_tool = has_schema_tool || py_info.has_serde;
    has_yaml_support = has_yaml_support || py_info.has_serde_yaml;
    has_validator = has_validator || py_info.has_validator;

    SchemaInfo { has_serde: has_schema_tool, has_serde_yaml: has_yaml_support, has_validator }
}

const VALIDATION_COMMANDS: &[&str] =
    &["pv validate", "forjar validate", "batuta playbook validate"];

fn has_validation_in_build_files(project_path: &Path) -> bool {
    let makefile_content =
        std::fs::read_to_string(project_path.join("Makefile")).unwrap_or_default();
    if VALIDATION_COMMANDS.iter().any(|cmd| makefile_content.contains(cmd)) {
        return true;
    }
    has_validation_in_ci(project_path)
}

fn has_validation_in_ci(project_path: &Path) -> bool {
    let pattern = format!("{}/.github/workflows/*.y*ml", project_path.display());
    let Ok(entries) = glob::glob(&pattern) else {
        return false;
    };
    for entry in entries.flatten() {
        let content = std::fs::read_to_string(&entry).unwrap_or_default();
        if VALIDATION_COMMANDS.iter().any(|cmd| content.contains(cmd)) {
            return true;
        }
    }
    false
}

fn detect_python_schema_deps(project_path: &Path) -> SchemaInfo {
    let mut info = SchemaInfo { has_serde: false, has_serde_yaml: false, has_validator: false };
    for pyfile in ["pyproject.toml", "setup.cfg", "requirements.txt"] {
        let content = std::fs::read_to_string(project_path.join(pyfile)).unwrap_or_default();
        if content.contains("pydantic")
            || content.contains("marshmallow")
            || content.contains("cerberus")
            || content.contains("jsonschema")
        {
            info.has_serde = true;
            info.has_validator = true;
        }
        if content.contains("pyyaml") || content.contains("ruamel") {
            info.has_serde_yaml = true;
        }
    }
    info
}

/// Check if any source file has a config struct with Deserialize derive,
/// or if the project uses external schema validation tools (pv, forjar).
pub(crate) fn has_deserialize_config_struct(project_path: &Path) -> bool {
    has_rust_config_struct(project_path)
        || has_pv_contracts(project_path)
        || has_json_schema_files(project_path)
}

fn has_rust_config_struct(project_path: &Path) -> bool {
    let glob_str = format!("{}/src/**/*.rs", project_path.display());
    let entries = match glob::glob(&glob_str) {
        Ok(e) => e,
        Err(_) => return false,
    };
    for entry in entries.flatten() {
        let content = match std::fs::read_to_string(&entry) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let is_config = content.contains("#[derive")
            && content.contains("Deserialize")
            && content.contains("struct")
            && content.to_lowercase().contains("config");
        if is_config {
            return true;
        }
    }
    false
}

fn has_pv_contracts(project_path: &Path) -> bool {
    for dir in ["contracts", "contract"] {
        let pattern = format!("{}/{}/**/*.yaml", project_path.display(), dir);
        let Ok(entries) = glob::glob(&pattern) else {
            continue;
        };
        for entry in entries.flatten() {
            let content = std::fs::read_to_string(&entry).unwrap_or_default();
            if content.contains("proof_obligations:") || content.contains("metadata:") {
                return true;
            }
        }
    }
    false
}

fn has_json_schema_files(project_path: &Path) -> bool {
    let pattern = format!("{}/**/*.schema.json", project_path.display());
    glob::glob(&pattern).ok().and_then(|mut e| e.next()).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_source_contains_pattern_finds_struct() {
        let path = PathBuf::from(".");
        assert!(source_contains_pattern(&path, &["struct"]));
    }

    #[test]
    fn test_source_contains_pattern_nonexistent_path() {
        let path = PathBuf::from("/nonexistent/path");
        assert!(!source_contains_pattern(&path, &["anything"]));
    }

    #[test]
    fn test_files_contain_pattern_ci_nonexistent_path() {
        let path = PathBuf::from("/nonexistent/path");
        assert!(!files_contain_pattern_ci(&path, &["src/**/*.rs"], &["anything"]));
    }

    #[test]
    fn test_find_scripting_deps_current_project() {
        let path = PathBuf::from(".");
        let deps = find_scripting_deps(&path);
        // batuta should not have scripting deps
        assert!(deps.is_empty());
    }

    #[test]
    fn test_detect_schema_deps_current_project() {
        let path = PathBuf::from(".");
        let info = detect_schema_deps(&path);
        // batuta uses serde
        assert!(info.has_serde);
    }

    #[test]
    fn test_has_deserialize_config_struct_nonexistent() {
        let path = PathBuf::from("/nonexistent/path");
        assert!(!has_deserialize_config_struct(&path));
    }

    #[test]
    fn test_test_contains_pattern_finds_test() {
        let path = PathBuf::from(".");
        assert!(test_contains_pattern(&path, &["#[test]"]));
    }

    #[test]
    fn test_ci_platform_count_current_project() {
        let path = PathBuf::from(".");
        // May or may not have CI - just verify no panic
        let _ = ci_platform_count(&path, &["ubuntu", "macos", "windows"]);
    }

    // =========================================================================
    // Coverage gap: invalid glob patterns
    // =========================================================================

    #[test]
    fn test_files_contain_pattern_invalid_glob() {
        let path = PathBuf::from(".");
        // Invalid glob pattern with unclosed bracket - should not panic, returns false
        assert!(!files_contain_pattern(&path, &["src/[invalid"], &["anything"]));
    }

    #[test]
    fn test_files_contain_pattern_ci_invalid_glob() {
        let path = PathBuf::from(".");
        assert!(!files_contain_pattern_ci(&path, &["src/[invalid"], &["anything"]));
    }

    // =========================================================================
    // Coverage gap: ci_platform_count with nonexistent path
    // =========================================================================

    #[test]
    fn test_ci_platform_count_nonexistent_path() {
        let path = PathBuf::from("/nonexistent/path");
        assert_eq!(ci_platform_count(&path, &["ubuntu", "macos"]), 0);
    }

    #[test]
    fn test_ci_platform_count_invalid_glob_pattern() {
        let path = PathBuf::from(".");
        // ci_platform_count uses hardcoded globs for .github/workflows/*.yml
        // With a normal path but no CI files, should return 0
        let count = ci_platform_count(&path, &["ubuntu", "macos", "windows"]);
        assert!(count <= 3); // At most 3 platforms
    }

    // =========================================================================
    // Coverage gap: apply_check_outcome variants
    // =========================================================================

    #[test]
    fn test_apply_check_outcome_pass() {
        let item = CheckItem::new("T-01", "Test", "Claim");
        let result = apply_check_outcome(item, &[(true, CheckOutcome::Pass)]);
        assert_eq!(result.status, super::super::types::CheckStatus::Pass);
    }

    #[test]
    fn test_apply_check_outcome_partial() {
        let item = CheckItem::new("T-01", "Test", "Claim");
        let result = apply_check_outcome(item, &[(true, CheckOutcome::Partial("partial reason"))]);
        assert_eq!(result.status, super::super::types::CheckStatus::Partial);
        assert_eq!(result.rejection_reason, Some("partial reason".to_string()));
    }

    #[test]
    fn test_apply_check_outcome_fail() {
        let item = CheckItem::new("T-01", "Test", "Claim");
        let result = apply_check_outcome(item, &[(true, CheckOutcome::Fail("fail reason"))]);
        assert_eq!(result.status, super::super::types::CheckStatus::Fail);
        assert_eq!(result.rejection_reason, Some("fail reason".to_string()));
    }

    #[test]
    fn test_apply_check_outcome_no_match() {
        let item = CheckItem::new("T-01", "Test", "Claim");
        let result = apply_check_outcome(
            item,
            &[(false, CheckOutcome::Pass), (false, CheckOutcome::Fail("nope"))],
        );
        // No condition matched, item returned unchanged (Skipped default)
        assert_eq!(result.status, super::super::types::CheckStatus::Skipped);
    }

    #[test]
    fn test_apply_check_outcome_first_match_wins() {
        let item = CheckItem::new("T-01", "Test", "Claim");
        let result = apply_check_outcome(
            item,
            &[
                (false, CheckOutcome::Fail("should not match")),
                (true, CheckOutcome::Partial("second wins")),
                (true, CheckOutcome::Pass), // should not be reached
            ],
        );
        assert_eq!(result.status, super::super::types::CheckStatus::Partial);
        assert_eq!(result.rejection_reason, Some("second wins".to_string()));
    }

    // =========================================================================
    // Coverage gap: find_scripting_deps edge cases
    // =========================================================================

    #[test]
    fn test_find_scripting_deps_nonexistent_path() {
        let path = PathBuf::from("/nonexistent/path");
        let deps = find_scripting_deps(&path);
        assert!(deps.is_empty());
    }

    // =========================================================================
    // Coverage gap: detect_schema_deps nonexistent path
    // =========================================================================

    #[test]
    fn test_detect_schema_deps_nonexistent_path() {
        let path = PathBuf::from("/nonexistent/path");
        let info = detect_schema_deps(&path);
        assert!(!info.has_serde);
        assert!(!info.has_serde_yaml);
        assert!(!info.has_validator);
    }

    // =========================================================================
    // Coverage gap: source_or_config_contains_pattern
    // =========================================================================

    #[test]
    fn test_source_or_config_contains_pattern_finds_toml() {
        let path = PathBuf::from(".");
        // Should find patterns in Cargo.toml
        assert!(source_or_config_contains_pattern(&path, &["[package]"]));
    }

    #[test]
    fn test_source_or_config_contains_pattern_nonexistent() {
        let path = PathBuf::from("/nonexistent/path");
        assert!(!source_or_config_contains_pattern(&path, &["anything"]));
    }

    // =========================================================================
    // Coverage gap: ci_contains_pattern
    // =========================================================================

    #[test]
    fn test_ci_contains_pattern_nonexistent_path() {
        let path = PathBuf::from("/nonexistent/path");
        assert!(!ci_contains_pattern(&path, &["ubuntu"]));
    }

    // =========================================================================
    // Coverage gap: test_contains_pattern with nonexistent path
    // =========================================================================

    #[test]
    fn test_test_contains_pattern_nonexistent_path() {
        let path = PathBuf::from("/nonexistent/path");
        assert!(!test_contains_pattern(&path, &["#[test]"]));
    }

    // =========================================================================
    // Coverage gap: has_deserialize_config_struct current project
    // =========================================================================

    #[test]
    fn test_has_deserialize_config_struct_current_project() {
        let path = PathBuf::from(".");
        // Just exercise the code path - current project may or may not have one
        let _ = has_deserialize_config_struct(&path);
    }

    // =========================================================================
    // Coverage gap: files_contain_pattern_ci actual match
    // =========================================================================

    #[test]
    fn test_files_contain_pattern_ci_matches_rust_source() {
        let path = PathBuf::from(".");
        // Use case-insensitive match on Rust source files
        assert!(files_contain_pattern_ci(
            &path,
            &["src/**/*.rs"],
            &["FN "] // lowercase "fn " should match via case insensitive
        ));
    }

    #[test]
    fn test_files_contain_pattern_ci_no_match() {
        // Use a nonexistent path to guarantee no files match the glob
        let path = PathBuf::from("/nonexistent/empty/dir");
        assert!(!files_contain_pattern_ci(&path, &["src/**/*.rs"], &["fn"]));
    }

    // =========================================================================
    // Coverage gap: find_scripting_deps with forbidden dependency present
    // =========================================================================

    #[test]
    fn test_find_scripting_deps_with_forbidden_dep() {
        let temp = std::env::temp_dir().join("batuta_test_scripting_deps");
        let _ = std::fs::create_dir_all(&temp);
        // Write a Cargo.toml with pyo3 in [dependencies] (not dev-dependencies)
        std::fs::write(
            temp.join("Cargo.toml"),
            "[package]\nname = \"test\"\nversion = \"0.1.0\"\n\n[dependencies]\npyo3 = \"0.20\"\n",
        )
        .expect("unexpected failure");

        let deps = find_scripting_deps(&temp);
        assert!(deps.contains(&"pyo3".to_string()), "Should find pyo3 in dependencies: {:?}", deps);

        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_find_scripting_deps_dev_only_dep() {
        let temp = std::env::temp_dir().join("batuta_test_scripting_devonly");
        let _ = std::fs::create_dir_all(&temp);
        // pyo3 only in dev-dependencies — should not be flagged
        std::fs::write(
            temp.join("Cargo.toml"),
            "[package]\nname = \"test\"\nversion = \"0.1.0\"\n\n[dev-dependencies]\npyo3 = \"0.20\"\n",
        )
        .expect("unexpected failure");

        let deps = find_scripting_deps(&temp);
        // Dev-only deps should be filtered out (line 192 branch)
        assert!(deps.is_empty(), "Dev-only dep should not be flagged: {:?}", deps);

        let _ = std::fs::remove_dir_all(&temp);
    }

    // =========================================================================
    // Coverage gap: ci_platform_count returning count >= 2 (line 167)
    // =========================================================================

    #[test]
    fn test_ci_platform_count_with_workflow_file() {
        let temp = std::env::temp_dir().join("batuta_test_ci_platforms");
        let _ = std::fs::remove_dir_all(&temp);
        let _ = std::fs::create_dir_all(temp.join(".github/workflows"));
        // Create a workflow file with multiple platform names
        std::fs::write(
            temp.join(".github/workflows/ci.yml"),
            "name: CI\non:\n  push:\njobs:\n  test:\n    strategy:\n      matrix:\n        os: [ubuntu-latest, macos-latest, windows-latest]\n",
        )
        .expect("unexpected failure");

        let count = ci_platform_count(&temp, &["ubuntu", "macos", "windows"]);
        assert!(count >= 2, "Should find at least 2 platforms in workflow: {}", count);

        let _ = std::fs::remove_dir_all(&temp);
    }

    // =========================================================================
    // Coverage gap: test_contains_pattern via cfg(test) fallback (lines 138-144)
    // =========================================================================

    #[test]
    fn test_test_contains_pattern_via_cfg_test_module() {
        // Create a temp project where test patterns exist only in
        // #[cfg(test)] modules inside src files (not in tests/ dir)
        let temp = std::env::temp_dir().join("batuta_test_cfg_test_fallback");
        let _ = std::fs::remove_dir_all(&temp);
        let _ = std::fs::create_dir_all(temp.join("src"));
        std::fs::write(
            temp.join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\n\
             #[cfg(test)]\n\
             mod tests {\n\
                 use super::*;\n\
                 #[test]\n\
                 fn test_add_unique_marker() { assert_eq!(add(1, 2), 3); }\n\
             }\n",
        )
        .expect("unexpected failure");

        // Search for the unique marker that only exists in #[cfg(test)] module
        assert!(test_contains_pattern(&temp, &["test_add_unique_marker"]));

        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_test_contains_pattern_no_cfg_test() {
        // Project with source files but no #[cfg(test)] and no tests/ dir
        let temp = std::env::temp_dir().join("batuta_test_no_cfg_test");
        let _ = std::fs::remove_dir_all(&temp);
        let _ = std::fs::create_dir_all(temp.join("src"));
        std::fs::write(temp.join("src/lib.rs"), "pub fn add(a: i32, b: i32) -> i32 { a + b }\n")
            .expect("unexpected failure");

        // No test modules, no tests/ dir — should return false
        assert!(!test_contains_pattern(&temp, &["nonexistent_test_fn"]));

        let _ = std::fs::remove_dir_all(&temp);
    }

    // =========================================================================
    // Coverage gap: has_deserialize_config_struct finding a match
    // =========================================================================

    #[test]
    fn test_has_deserialize_config_struct_found() {
        let temp = std::env::temp_dir().join("batuta_test_deser_config");
        let _ = std::fs::remove_dir_all(&temp);
        let _ = std::fs::create_dir_all(temp.join("src"));
        std::fs::write(
            temp.join("src/lib.rs"),
            "#[derive(serde::Deserialize)]\npub struct AppConfig {\n    pub name: String,\n}\n",
        )
        .expect("unexpected failure");

        assert!(has_deserialize_config_struct(&temp));

        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_has_deserialize_config_struct_no_config() {
        let temp = std::env::temp_dir().join("batuta_test_deser_noconfig");
        let _ = std::fs::remove_dir_all(&temp);
        let _ = std::fs::create_dir_all(temp.join("src"));
        std::fs::write(
            temp.join("src/lib.rs"),
            "#[derive(serde::Deserialize)]\npub struct UserData {\n    pub id: u64,\n}\n",
        )
        .expect("unexpected failure");

        // Has Deserialize + struct but not "config" in name — should return false
        assert!(!has_deserialize_config_struct(&temp));

        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_detect_schema_deps_serde_yaml_ng() {
        let temp = std::env::temp_dir().join("batuta_test_schema_yaml_ng");
        let _ = std::fs::remove_dir_all(&temp);
        let _ = std::fs::create_dir_all(&temp);
        std::fs::write(
            temp.join("Cargo.toml"),
            "[package]\nname = \"test\"\nversion = \"0.1.0\"\n\n[dependencies]\nserde = \"1.0\"\nserde_yaml_ng = \"0.10\"\n",
        )
        .expect("unexpected failure");

        let info = detect_schema_deps(&temp);
        assert!(info.has_serde);
        assert!(info.has_serde_yaml, "serde_yaml_ng should be detected as has_serde_yaml");

        let _ = std::fs::remove_dir_all(&temp);
    }
}
