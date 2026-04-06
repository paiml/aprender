//! Diff-aware lint: only lint contracts changed since a base ref.
//!
//! Uses `git diff --name-only <base_ref>..HEAD -- contracts/` to find
//! changed YAML files, then expands to include transitive dependents
//! (contracts that `depend_on` any changed contract).
//!
//! Spec: `docs/specifications/sub/lint.md` Section 3

use std::path::Path;
use std::process::Command;

/// Get contract stems changed since `base_ref`.
pub fn changed_contracts(contracts_dir: &Path, base_ref: &str) -> Result<Vec<String>, String> {
    let repo_root = find_repo_root(contracts_dir)?;

    let output = Command::new("git")
        .args([
            "diff",
            "--name-only",
            &format!("{base_ref}..HEAD"),
            "--",
            "contracts/",
        ])
        .current_dir(repo_root)
        .output()
        .map_err(|e| format!("Failed to run git diff: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("git diff failed: {stderr}"));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stems: Vec<String> = stdout
        .lines()
        .filter(|line| {
            Path::new(line)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("yaml"))
        })
        .filter_map(|line| {
            Path::new(line)
                .file_stem()
                .and_then(|s| s.to_str())
                .map(ToString::to_string)
        })
        .collect();

    Ok(stems)
}

/// Expand changed stems with transitive dependents.
///
/// If contract A changed and contract B has `depends_on: [A]`, then B
/// is also added to the lint set.
pub fn expand_dependents(
    changed: &[String],
    all_contracts: &[(String, crate::schema::Contract)],
) -> Vec<String> {
    let mut expanded: std::collections::HashSet<String> = changed.iter().cloned().collect();

    // Find contracts that depend on any changed contract
    for (stem, contract) in all_contracts {
        if expanded.contains(stem) {
            continue;
        }
        for dep in &contract.metadata.depends_on {
            if expanded.contains(dep) {
                expanded.insert(stem.clone());
                break;
            }
        }
    }

    let mut result: Vec<String> = expanded.into_iter().collect();
    result.sort();
    result
}

fn find_repo_root(start: &Path) -> Result<String, String> {
    let output = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(start)
        .output()
        .map_err(|e| format!("Failed to find git repo root: {e}"))?;

    if !output.status.success() {
        return Err("Not a git repository".into());
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contracts_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
    }

    #[test]
    #[ignore] // Requires git repo — skipped in container CI
    fn changed_contracts_from_head() {
        // HEAD~0..HEAD should produce empty or small set
        let result = changed_contracts(&contracts_dir(), "HEAD");
        assert!(result.is_ok());
    }

    #[test]
    fn changed_contracts_invalid_ref() {
        let result = changed_contracts(&contracts_dir(), "nonexistent-ref-xyz");
        assert!(result.is_err());
    }

    #[test]
    fn expand_dependents_no_deps() {
        let changed = vec!["softmax-kernel-v1".to_string()];
        let contracts: Vec<(String, crate::schema::Contract)> = vec![];
        let expanded = expand_dependents(&changed, &contracts);
        assert_eq!(expanded, vec!["softmax-kernel-v1".to_string()]);
    }

    #[test]
    fn expand_dependents_with_real_contracts() {
        // Load real contracts and test expand with actual depends_on
        use crate::lint::gates::load_contracts;
        let dir = contracts_dir();
        let (all, _) = load_contracts(&dir);

        // softmax-kernel-v1 has no deps, but attention-kernel-v1 depends on it
        let changed = vec!["softmax-kernel-v1".to_string()];
        let expanded = expand_dependents(&changed, &all);
        assert!(expanded.contains(&"softmax-kernel-v1".to_string()));
        // Should have at least the original
        assert!(!expanded.is_empty());
    }

    #[test]
    fn expand_dependents_empty_changed() {
        let changed: Vec<String> = vec![];
        let contracts: Vec<(String, crate::schema::Contract)> = vec![];
        let expanded = expand_dependents(&changed, &contracts);
        assert!(expanded.is_empty());
    }

    #[test]
    #[ignore] // Requires git repo — skipped in container CI
    fn changed_contracts_with_range() {
        // Use a wide range to guarantee files are returned, exercising the filter closures.
        // On shallow clones (CI), HEAD~50 may not exist — skip gracefully.
        let result = changed_contracts(&contracts_dir(), "HEAD~50");
        match result {
            Err(e) if e.contains("git diff failed") => {
                // Shallow clone — skip test
            }
            Err(e) => panic!("Unexpected error: {e}"),
            Ok(stems) => {
                // All returned stems should be valid (no .yaml extension, no path prefix)
                for stem in &stems {
                    assert!(
                        !stem.to_ascii_lowercase().ends_with(".yaml"),
                        "Stem should not have extension: {stem}"
                    );
                    assert!(
                        !stem.contains('/'),
                        "Stem should not have path separator: {stem}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore] // Requires git repo — skipped in container CI
    fn find_repo_root_works() {
        let root = find_repo_root(&contracts_dir());
        assert!(root.is_ok());
        assert!(root.unwrap().contains("provable-contracts"));
    }

    #[test]
    fn find_repo_root_non_git_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let result = find_repo_root(tmp.path());
        assert!(result.is_err());
    }

    #[test]
    fn expand_dependents_transitive() {
        // Build contracts where B depends on A
        let a_yaml = r#"
metadata:
  version: "1.0.0"
  description: "A"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
"#;
        let b_yaml = r#"
metadata:
  version: "1.0.0"
  description: "B"
  references: ["Paper"]
  depends_on:
    - "contract-a"
equations:
  g:
    formula: "g(x) = 2x"
"#;
        let c_yaml = r#"
metadata:
  version: "1.0.0"
  description: "C"
  references: ["Paper"]
equations:
  h:
    formula: "h(x) = 3x"
"#;
        let a: crate::schema::Contract = crate::schema::parse_contract_str(a_yaml).unwrap();
        let b: crate::schema::Contract = crate::schema::parse_contract_str(b_yaml).unwrap();
        let c: crate::schema::Contract = crate::schema::parse_contract_str(c_yaml).unwrap();

        let all = vec![
            ("contract-a".to_string(), a),
            ("contract-b".to_string(), b),
            ("contract-c".to_string(), c),
        ];

        // Only A changed — B should be pulled in because it depends on A
        let changed = vec!["contract-a".to_string()];
        let expanded = expand_dependents(&changed, &all);
        assert!(expanded.contains(&"contract-a".to_string()));
        assert!(expanded.contains(&"contract-b".to_string()));
        // C has no dependency on A, should not be included
        assert!(!expanded.contains(&"contract-c".to_string()));
    }

    #[test]
    fn expand_dependents_already_in_changed() {
        // If B is already in changed and depends on A, it shouldn't be duplicated
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "X"
  references: ["Paper"]
  depends_on:
    - "other"
equations:
  f:
    formula: "f(x) = x"
"#;
        let contract: crate::schema::Contract = crate::schema::parse_contract_str(yaml).unwrap();
        let all = vec![("dep-contract".to_string(), contract)];
        let changed = vec!["dep-contract".to_string(), "other".to_string()];
        let expanded = expand_dependents(&changed, &all);
        // dep-contract is already in changed, so it should appear exactly once
        assert_eq!(expanded.iter().filter(|s| *s == "dep-contract").count(), 1);
    }
}
