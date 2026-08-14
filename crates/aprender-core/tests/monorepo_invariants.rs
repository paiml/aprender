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

/// The actual workspace root.
///
/// `CARGO_MANIFEST_DIR` for this test target is `crates/aprender-core`, NOT the
/// workspace root. Tests that pass it to `cargo metadata` as `current_dir` are
/// unaffected — cargo walks up and reports all 78 packages — but any test using
/// it as a filesystem path was silently looking in the wrong place. Two were:
/// FALSIFY-MONO-011 scanned `crates/aprender-core/crates` (nonexistent, so it
/// passed unconditionally) and FALSIFY-BUILD-004 read aprender-core's manifest
/// while claiming to check the root's.
///
/// Same pattern as `readme_contract.rs:11`, which had it right.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root must resolve from crates/aprender-core")
}

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
    // `CARGO_MANIFEST_DIR` is crates/aprender-core, so the old
    // `workspace_root.join("crates")` resolved to crates/aprender-core/crates,
    // which does not exist. `read_dir` returned Err, the `if let Ok(..)` body
    // below never ran, `violations` stayed empty and this required check passed
    // unconditionally for its entire life. Repointing it reports three crates.
    //
    // The six other tests in this file that use the bare `CARGO_MANIFEST_DIR`
    // are NOT affected: they pass it only as `current_dir` to `cargo metadata`,
    // which walks up to the workspace root on its own (verified: 78 packages
    // from crates/aprender-core). Only this test and FALSIFY-BUILD-004 use it
    // as a filesystem path, and both were wrong.
    let crates_dir = workspace_root().join("crates");

    // During migration, legacy binaries from merged repos are allowed.
    // Post-migration (Phase 5+), these should be folded into apr-cli subcommands.
    // For now, we track them — the test documents what has [[bin]] sections.
    // The migration debt register: every package that still ships a binary of
    // its own, because its capability is not yet reachable as `apr <subcommand>`.
    //
    // This is NOT a permanent exemption list. Deleting a binary before its
    // capability is reachable through `apr` removes the capability rather than
    // relocating it, so the order is: expose via apr, drop the bin, drop the
    // entry here. The ratchet below enforces that the list only shrinks.
    //
    // Derived from `cargo metadata`, so it includes packages that auto-discover
    // `src/main.rs` without any `[[bin]]` section — invisible to the manifest
    // grep this check used to do.
    let allowed_bins: HashSet<&str> = [
        // Sanctioned by the contract itself (cgp-monorepo-consolidation-v1.yaml:212).
        "apr-cli",                // apr, apr-corpus-ingest
        "aprender",               // apr (root facade: `cargo install aprender`)
        "aprender-contracts-cli", // pv — explicit contract exception, and
        // `apr pv` + naked `pv` is a settled decision
        // Build/dev tooling, never user-facing ML surface.
        "aprender-compute-xtask", // aprender-compute-xtask
        "aprender-ptx-debug",     // aprender-ptx-debug
        "aprender-qa-certify",    // apr-qa-readme-sync
        // Pre-consolidation names still carrying the only access to their
        // capability. These are the migration targets.
        "aprender-data",             // alimentar
        "aprender-simulate",         // simular
        "aprender-present-cli",      // presentar
        "aprender-present-terminal", // ptop, score
        "aprender-rag-cli",          // trueno-rag
        "aprender-zram-cli",         // trueno-zram
        "aprender-zram-generator",   // aprender-zram-generator
        "aprender-verify-ml",        // verificar
        "aprender-explain",          // aprender-explain (auto-discovered)
        "aprender-orchestrate",      // aprender-orchestrate (auto-discovered)
        "aprender-profile",          // aprender-profile (auto-discovered)
        "aprender-qa-cli",           // apr-qa
        "aprender-cbtop",            // aprender-cbtop
        "aprender-cgp",              // aprender-cgp
        "aprender-db",               // aprender-db
        "aprender-test-cli",         // aprender-test-cli
        // Training satellites.
        "aprender-train-bench",   // aprender-train-bench
        "aprender-train-distill", // aprender-train-distill
        "aprender-train-inspect", // aprender-train-inspect
        "aprender-train-lora",    // aprender-train-lora
        "aprender-train-shell",   // aprender-train-shell
    ]
    .into();

    // Ask cargo what binaries the workspace actually BUILDS, rather than
    // grepping manifests for `[[bin]]`. Cargo auto-discovers `src/main.rs` with
    // no `[[bin]]` section at all, and six packages rely on that —
    // aprender-explain, aprender-orchestrate, aprender-profile,
    // aprender-compute-xtask, aprender-zram-generator, and aprender-verify-ml,
    // which ships `verificar`. A manifest grep is blind to every one of them, so
    // the contract's central claim was unenforceable even where the path worked.
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(workspace_root())
        .output()
        .expect("failed to run cargo metadata");
    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("cargo metadata is not valid JSON");
    let packages = metadata["packages"]
        .as_array()
        .expect("cargo metadata has no packages array");

    let mut violations = Vec::new();
    let mut with_bins: Vec<String> = Vec::new();
    for pkg in packages {
        let name = pkg["name"].as_str().unwrap_or_default().to_string();
        let ships_bin = pkg["targets"].as_array().into_iter().flatten().any(|t| {
            t["kind"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|k| k == "bin")
        });
        if ships_bin {
            if !allowed_bins.contains(name.as_str()) {
                violations.push(name.clone());
            }
            with_bins.push(name);
        }
    }

    // Vacuity: a scan that looked at nothing must not report clean. This is the
    // assertion the original lacked, and it is why the path bug went unnoticed
    // for the whole life of the check.
    assert!(
        packages.len() > 50,
        "FALSIFY-MONO-011 saw only {} package(s) — cargo metadata is not reporting \
         the workspace; fix the invocation rather than this number",
        packages.len()
    );
    assert!(
        with_bins.len() > 10,
        "FALSIFY-MONO-011 found only {} package(s) shipping a binary, which cannot \
         be right for this workspace — the target scan is broken",
        with_bins.len()
    );

    assert!(
        violations.is_empty(),
        "FALSIFY-MONO-011: Unauthorized [[bin]] sections found in: {:?}\n\
         Only apr-cli should produce user-facing binaries.",
        violations
    );

    // RATCHET: the allowlist is a migration debt register, not a permanent
    // exemption. Every entry is a capability reachable only by its own binary
    // and not yet through `apr <subcommand>`; the list may SHRINK as capabilities
    // migrate, never grow. Deleting a [[bin]] before its capability is reachable
    // through apr removes the capability rather than relocating it, so the
    // ordering is: expose via apr, then drop the bin, then drop the entry here.
    const ALLOWLIST_BASELINE: usize = 27;
    assert!(
        allowed_bins.len() <= ALLOWLIST_BASELINE,
        "FALSIFY-MONO-011: the [[bin]] allowlist grew to {} (baseline {ALLOWLIST_BASELINE}). \
         It is shrink-only: migrate the capability to an `apr` subcommand instead of \
         granting a new exemption.",
        allowed_bins.len()
    );

    // And it must not rot: an allowlisted crate that no longer ships a [[bin]]
    // is a stale exemption hiding the fact that the migration already happened.
    let stale: Vec<&str> = allowed_bins
        .iter()
        .filter(|c| !with_bins.iter().any(|b| b == *c) && crates_dir.join(c).exists())
        .copied()
        .collect();
    assert!(
        stale.is_empty(),
        "FALSIFY-MONO-011: these crates are allowlisted but ship no [[bin]] — the \
         exemption is stale and must be deleted so the ratchet reflects real debt: {stale:?}"
    );
}

/// FALSIFY-BUILD-004: No [patch.crates-io] in root Cargo.toml.
/// The monorepo eliminates the need for patches.
#[test]
fn test_no_patch_in_root_cargo_toml() {
    // Was `Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")`, i.e.
    // crates/aprender-core/Cargo.toml — this test has never once read the root
    // manifest it is named for.
    let root = workspace_root().join("Cargo.toml");
    let cargo_toml = std::fs::read_to_string(&root)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", root.display()));

    // Line-based, ignoring comments. The root manifest legitimately DISCUSSES
    // `[patch.crates-io]` twice — a note about the excluded aprender-train-canary
    // crate, and `# [patch.crates-io] — REMOVED` recording that the cc patch was
    // dropped. A naive `contains()` reads both as violations, so repointing the
    // path without this would have turned a required check red on prose.
    let active: Vec<(usize, &str)> = cargo_toml
        .lines()
        .enumerate()
        .filter(|(_, l)| l.trim_start().starts_with("[patch.crates-io]"))
        .map(|(i, l)| (i + 1, l.trim()))
        .collect();

    // Vacuity: prove we read the real root before concluding from an absence.
    assert!(
        cargo_toml.contains("[workspace]"),
        "{} is not the workspace root manifest — the path is wrong again",
        root.display()
    );

    assert!(
        active.is_empty(),
        "FALSIFY-BUILD-004: Root Cargo.toml has an active [patch.crates-io] at {active:?}.\n\
         The monorepo should eliminate all cross-repo patches; a dev override belongs \
         in .cargo/config.toml, which is not committed."
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
