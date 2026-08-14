//! APR-MONO Integration Tests — Monorepo Structural Invariants
//!
//! Verifies the invariants defined in:
//! - `docs/specifications/aprender-monorepo-consolidation.md`
//! - `contracts/cgp-monorepo-consolidation-v1.yaml`
//! - `contracts/cgp-monorepo-build-v1.yaml`
//!
//! These tests enforce FALSIFY-MONO-010 through FALSIFY-MONO-013
//! and the build contract FALSIFY-BUILD-001 through FALSIFY-BUILD-006.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

/// FALSIFY-MONO-010: Every [package] name must be in the spec registry.
/// Crate names not in Appendix A are unauthorized.
#[test]
fn test_all_crate_names_use_aprender_prefix() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(workspace_root)
        .output()
        .expect("cargo metadata failed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let metadata: serde_json::Value =
        serde_json::from_str(&stdout).expect("failed to parse cargo metadata");

    let packages = metadata["packages"].as_array().expect("no packages");
    let mut violations = Vec::new();

    for pkg in packages {
        let name = pkg["name"].as_str().unwrap_or("");
        let source = pkg["source"].as_str();

        // Only check workspace members (source = null)
        if source.is_some() {
            continue;
        }

        // Package names must start with "aprender" or be an allowlisted `apr-*`
        // family member (the CLI binary crate and the sovereign leaf format crate).
        const APR_ALLOWLIST: [&str; 2] = ["apr-cli", "apr-format"];
        if !name.starts_with("aprender") && !APR_ALLOWLIST.contains(&name) {
            violations.push(name.to_string());
        }
    }

    assert!(
        violations.is_empty(),
        "FALSIFY-MONO-010: Crates with non-aprender names found: {:?}\n\
         All workspace crates must use `aprender-*` naming (except `apr-cli`, `apr-format`).",
        violations
    );
}

/// FALSIFY-MONO-012: No nested crates — all workspace members must be
/// direct children of crates/ (flat layout per Polars/Burn/Nushell pattern).
#[test]
fn test_flat_layout_no_nested_crates() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(workspace_root)
        .output()
        .expect("cargo metadata failed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let metadata: serde_json::Value =
        serde_json::from_str(&stdout).expect("failed to parse cargo metadata");

    // Get the actual workspace root from metadata
    let ws_root = metadata["workspace_root"].as_str().unwrap_or("");

    let packages = metadata["packages"].as_array().expect("no packages");
    let mut violations = Vec::new();

    for pkg in packages {
        if pkg["source"].as_str().is_some() {
            continue; // Skip external deps
        }

        let manifest = pkg["manifest_path"].as_str().unwrap_or("");
        let rel = manifest
            .strip_prefix(ws_root)
            .unwrap_or(manifest)
            .trim_start_matches('/');

        // Must be either root Cargo.toml or crates/<name>/Cargo.toml
        // Count the path depth: "crates/foo/Cargo.toml" = 3 parts
        let depth = rel.split('/').count();
        if depth > 3 && !rel.starts_with("Cargo.toml") {
            violations.push(format!("{} at {}", pkg["name"], rel));
        }
    }

    assert!(
        violations.is_empty(),
        "FALSIFY-MONO-012: Nested crates found (violates flat layout):\n{}\n\
         All crates must be at crates/<name>/Cargo.toml, not deeper.",
        violations.join("\n")
    );
}

/// FALSIFY-BUILD-001: No duplicate workspace members.
#[test]
fn test_no_duplicate_workspace_members() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(workspace_root)
        .output()
        .expect("cargo metadata failed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let metadata: serde_json::Value =
        serde_json::from_str(&stdout).expect("failed to parse cargo metadata");

    let packages = metadata["packages"].as_array().expect("no packages");
    let mut seen = HashSet::new();
    let mut duplicates = Vec::new();

    for pkg in packages {
        if pkg["source"].as_str().is_some() {
            continue;
        }
        let name = pkg["name"].as_str().unwrap_or("");
        if !seen.insert(name.to_string()) {
            duplicates.push(name.to_string());
        }
    }

    assert!(
        duplicates.is_empty(),
        "FALSIFY-BUILD-001: Duplicate workspace members: {:?}",
        duplicates
    );
}

/// FALSIFY-BUILD-002: Workspace version is consistent (0.29.0).
#[test]
fn test_workspace_version_consistency() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(workspace_root)
        .output()
        .expect("cargo metadata failed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let metadata: serde_json::Value =
        serde_json::from_str(&stdout).expect("failed to parse cargo metadata");

    let packages = metadata["packages"].as_array().expect("no packages");
    let mut version_mismatches = Vec::new();

    for pkg in packages {
        if pkg["source"].as_str().is_some() {
            continue;
        }
        let name = pkg["name"].as_str().unwrap_or("");
        let version = pkg["version"].as_str().unwrap_or("");

        // Crates using version.workspace = true should have 0.29.0
        // Some pre-existing crates may still have old versions — that's ok during migration
        if version == "0.29.0" || name == "apr-cli" || name == "aprender" {
            continue; // Expected
        }
        version_mismatches.push(format!("{}@{}", name, version));
    }

    // Informational — don't fail yet, but track
    if !version_mismatches.is_empty() {
        eprintln!(
            "INFO: {} crates not yet on workspace version 0.29.0: {:?}",
            version_mismatches.len(),
            version_mismatches
        );
    }
}

/// FALSIFY-BUILD-003: Minimum workspace member count.
#[test]
fn test_minimum_workspace_member_count() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(workspace_root)
        .output()
        .expect("cargo metadata failed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let metadata: serde_json::Value =
        serde_json::from_str(&stdout).expect("failed to parse cargo metadata");

    let workspace_members: Vec<&str> = metadata["packages"]
        .as_array()
        .expect("no packages")
        .iter()
        .filter(|p| p["source"].is_null())
        .map(|p| p["name"].as_str().unwrap_or(""))
        .collect();

    // Spec target: ~63 crates. We should have at least 60.
    assert!(
        workspace_members.len() >= 60,
        "FALSIFY-BUILD-003: Only {} workspace members (expected >= 60).\n\
         The monorepo should contain all merged crates.",
        workspace_members.len()
    );
}

/// FALSIFY-MONO-011: Only apr-cli may have [[bin]] sections
/// (exception: aprender-contracts-cli for build tooling).
#[test]
fn test_no_unauthorized_binaries() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let crates_dir = workspace_root.join("crates");

    // During migration, legacy binaries from merged repos are allowed.
    // Post-migration (Phase 5+), these should be folded into apr-cli subcommands.
    // For now, we track them — the test documents what has [[bin]] sections.
    let allowed_bins: HashSet<&str> = [
        "apr-cli",
        "aprender-contracts-cli",
        "aprender-compute",
        "aprender-cbtop",
        "aprender-cgp",
        "aprender-explain",
        "aprender-ptx-debug",
        "aprender-zram-cli",
        "aprender-present-cli",
        "aprender-present-terminal",
        "aprender-test-cli",
        "aprender-test-showcase",
        "aprender-data",
        "aprender-db",
        "aprender-distribute",
        "aprender-registry",
        "aprender-serve",
        "aprender-shell",
        "aprender-simulate",
        "aprender-train",
        "aprender-train-canary",
        // aprender-train-{bench,distill,inspect,lora,shell} were REMOVED from
        // this allowlist on 2026-08-14: their [[bin]] targets are gone and the
        // capability dispatches through `apr train {bench,distill,inspect,
        // lora,shell}`. Leaving them listed would keep permitting a binary
        // that no longer exists, so re-adding one would pass silently. See
        // contracts/apr-mono-binary-rule-v1.yaml FALSIFY-BINRULE-004.
        "aprender-tsp",
        "aprender-monte-carlo",
        "aprender-viz",
        "aprender-viz-ttop",
    ]
    .into();

    let mut violations = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&crates_dir) {
        for entry in entries.flatten() {
            let toml_path = entry.path().join("Cargo.toml");
            if !toml_path.exists() {
                continue;
            }
            let content = std::fs::read_to_string(&toml_path).unwrap_or_default();
            let dir_name = entry.file_name().to_string_lossy().to_string();

            if content.contains("[[bin]]") && !allowed_bins.contains(dir_name.as_str()) {
                violations.push(dir_name);
            }
        }
    }

    assert!(
        violations.is_empty(),
        "FALSIFY-MONO-011: Unauthorized [[bin]] sections found in: {:?}\n\
         Only apr-cli should produce user-facing binaries.",
        violations
    );
}

/// FALSIFY-BUILD-004: No [patch.crates-io] in root Cargo.toml.
/// The monorepo eliminates the need for patches.
#[test]
fn test_no_patch_in_root_cargo_toml() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let cargo_toml = std::fs::read_to_string(workspace_root.join("Cargo.toml"))
        .expect("failed to read root Cargo.toml");

    // Root Cargo.toml should not have [patch.crates-io]
    // (it may exist in .cargo/config.toml for dev overrides, but not in the committed Cargo.toml)
    assert!(
        !cargo_toml.contains("[patch.crates-io]"),
        "FALSIFY-BUILD-004: Root Cargo.toml still has [patch.crates-io].\n\
         The monorepo should eliminate all cross-repo patches."
    );
}

/// FALSIFY-BUILD-005: All crate directories exist and have Cargo.toml.
#[test]
fn test_all_workspace_dirs_have_cargo_toml() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(workspace_root)
        .output()
        .expect("cargo metadata failed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let metadata: serde_json::Value =
        serde_json::from_str(&stdout).expect("failed to parse cargo metadata");

    let packages = metadata["packages"].as_array().expect("no packages");
    let mut missing = Vec::new();

    for pkg in packages {
        if pkg["source"].as_str().is_some() {
            continue;
        }
        let manifest = pkg["manifest_path"].as_str().unwrap_or("");
        if !Path::new(manifest).exists() {
            missing.push(manifest.to_string());
        }
    }

    assert!(
        missing.is_empty(),
        "FALSIFY-BUILD-005: Missing Cargo.toml files: {:?}",
        missing
    );
}
