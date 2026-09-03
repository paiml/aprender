//! Shared recursive contract directory walker.
//!
//! Multiple `pv` subcommands need to load every YAML contract under
//! a directory tree (including subdirectories like `contracts/aprender/`,
//! `contracts/trueno/`, `contracts/patterns/`). This module provides the
//! canonical walker so commands agree on what counts as a contract and
//! which sidecar files to skip.

use std::path::Path;

use provable_contracts::schema::{parse_contract, Contract};

/// Walk `dir` recursively and collect every parseable `.yaml` contract
/// into `out` as `(stem, contract)` pairs.
///
/// Skips:
/// - `binding.yaml` / `binding.yml` (registry sidecar)
/// - any file whose stem contains `playbook` (playbook sidecars)
///
/// Silently drops unparseable files — callers wanting strict loading
/// should use `pv validate` or a bespoke loader.
pub fn collect_contracts(dir: &Path, out: &mut Vec<(String, Contract)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_contracts(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("yaml") {
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();
            if stem == "binding" || stem.contains("playbook") {
                continue;
            }
            if let Ok(c) = parse_contract(&path) {
                out.push((stem, c));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_from_missing_dir_is_empty() {
        let mut out = Vec::new();
        collect_contracts(Path::new("/nonexistent/path/to/contracts"), &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn collect_from_real_contracts_dir() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        if !dir.exists() {
            return; // skip on CI without contracts
        }
        let mut out = Vec::new();
        collect_contracts(&dir, &mut out);
        // Should find hundreds of contracts including subdirs.
        assert!(
            out.len() > 100,
            "expected > 100 contracts across the tree, got {}",
            out.len()
        );
        // Every entry must have a non-empty stem.
        for (stem, _) in &out {
            assert!(!stem.is_empty(), "empty stem in collected contracts");
            assert!(
                !stem.contains("playbook"),
                "playbook sidecar was not skipped: {stem}"
            );
            assert_ne!(stem, "binding", "binding sidecar was not skipped");
        }
    }

    #[test]
    fn collect_skips_binding_yaml() {
        let tmp = tempfile::tempdir().expect("temp dir is creatable");
        std::fs::write(
            tmp.path().join("binding.yaml"),
            "crates: []\nbindings: []\n",
        )
        .expect("fixture file is writable");
        std::fs::write(
            tmp.path().join("real-contract-v1.yaml"),
            "metadata:\n  version: \"1.0.0\"\n  description: \"t\"\n  references: [\"x\"]\nequations:\n  eq1:\n    formula: \"f(x)=x\"\nproof_obligations: []\nfalsification_tests: []\nkani_harnesses: []\n",
        )
        .expect("fixture file is writable");

        let mut out = Vec::new();
        collect_contracts(tmp.path(), &mut out);

        let stems: Vec<_> = out.iter().map(|(s, _)| s.as_str()).collect();
        assert!(!stems.contains(&"binding"), "binding should be skipped");
    }

    #[test]
    fn collect_recurses_into_subdirs() {
        let tmp = tempfile::tempdir().expect("temp dir is creatable");
        let sub = tmp.path().join("sub");
        std::fs::create_dir_all(&sub).expect("fixture subdirectory is creatable");
        let minimal = "metadata:\n  version: \"1.0.0\"\n  description: \"t\"\n  references: [\"x\"]\nequations:\n  eq1:\n    formula: \"f(x)=x\"\nproof_obligations: []\nfalsification_tests: []\nkani_harnesses: []\n";
        std::fs::write(tmp.path().join("top-v1.yaml"), minimal).expect("fixture file is writable");
        std::fs::write(sub.join("nested-v1.yaml"), minimal).expect("fixture file is writable");

        let mut out = Vec::new();
        collect_contracts(tmp.path(), &mut out);

        let stems: Vec<_> = out.iter().map(|(s, _)| s.clone()).collect();
        assert!(stems.contains(&"top-v1".to_string()));
        assert!(stems.contains(&"nested-v1".to_string()));
    }
}
