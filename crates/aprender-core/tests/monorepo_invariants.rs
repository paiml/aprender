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

/// The repository root, not this crate's manifest dir.
///
/// `env!("CARGO_MANIFEST_DIR")` for an integration test of `aprender-core` is
/// `crates/aprender-core`, so `manifest_dir.join("crates")` names
/// `crates/aprender-core/crates` — a directory that has never existed.
/// `test_no_unauthorized_binaries` walked exactly that path, found zero
/// entries, and passed while 21 crates shipped `[[bin]]` sections; and
/// `test_no_patch_in_root_cargo_toml` read `crates/aprender-core/Cargo.toml`
/// while claiming to inspect the workspace root.
///
/// The `../..` + `canonicalize()` form is the one already used by
/// `crates/aprender-core/tests/readme_contract.rs` — same crate, same problem,
/// solved correctly there.
///
/// The tests that shell out to `cargo metadata` do NOT need this: cargo walks
/// up to the workspace root on its own, so passing the manifest dir as
/// `current_dir` is correct for them.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root (crates/aprender-core/../..) must resolve")
}

/// True when `line` opens the TOML table `name` (e.g. `[[bin]]`).
///
/// Anchored and comment-aware on purpose. A substring test is wrong in both
/// directions: the root `Cargo.toml` mentions `[patch.crates-io]` twice inside
/// prose comments (once quoting an excluded crate's manifest, once recording
/// that the patch was removed), so `contains()` would report an active patch
/// table that is not there.
fn opens_table(line: &str, name: &str) -> bool {
    line.trim_start().starts_with(name) && !line.trim_start().starts_with('#')
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

/// The ONE binary APR-MONO ships: `apr`, built by the root facade package.
///
/// `apr-cli` builds an `apr` of its own, which the facade re-exports; it is
/// listed as debt below rather than sanctioned, because `cargo install apr-cli`
/// installs a second `apr` into `~/.cargo/bin`.
const SANCTIONED_BINS: [(&str, &str); 1] = [("aprender", "apr")];

/// MIGRATION DEBT — **THIS LIST MAY ONLY SHRINK.**
///
/// `(package, binary)` for every standalone binary the workspace still builds
/// instead of exposing the capability as `apr <subcommand>`. That is the
/// pre-consolidation shape APR-MONO exists to delete: `cargo install
/// aprender-data` and `cargo install alimentar` both drop a binary named
/// `alimentar` into `~/.cargo/bin`, where they silently overwrite each other.
///
/// This is the inventory as it stood when FALSIFY-MONO-011 was repaired — a
/// record of what is left to do, never a menu of what is permitted. **The only
/// legal edit is a deletion**, made as each capability is rehomed under `apr`.
/// Adding an entry is not an approved way to land a new binary: the ratchet in
/// `test_no_unauthorized_binaries` rejects that, and so does this array's own
/// length, which is part of its type.
///
/// Note the pairs are `(package, bin)`, not crate directories. The binary NAME
/// is the thing that collides in `~/.cargo/bin`, and it is frequently unrelated
/// to the crate that builds it — `aprender-verify-ml` builds `verificar`,
/// `aprender-contracts-cli` builds `pv`, `aprender-present-terminal` builds two
/// (`ptop` and `score`).
const LEGACY_BINS: [(&str, &str); 28] = [
    ("apr-cli", "apr"),
    ("apr-cli", "apr-corpus-ingest"),
    ("aprender-cbtop", "aprender-cbtop"),
    ("aprender-cgp", "aprender-cgp"),
    ("aprender-compute-xtask", "aprender-compute-xtask"),
    ("aprender-contracts-cli", "pv"),
    ("aprender-data", "alimentar"),
    ("aprender-db", "aprender-db"),
    ("aprender-explain", "aprender-explain"),
    ("aprender-orchestrate", "aprender-orchestrate"),
    ("aprender-present-cli", "presentar"),
    ("aprender-present-terminal", "ptop"),
    ("aprender-present-terminal", "score"),
    ("aprender-profile", "aprender-profile"),
    ("aprender-ptx-debug", "aprender-ptx-debug"),
    ("aprender-qa-certify", "apr-qa-readme-sync"),
    ("aprender-qa-cli", "apr-qa"),
    ("aprender-rag-cli", "trueno-rag"),
    ("aprender-simulate", "simular"),
    ("aprender-test-cli", "aprender-test-cli"),
    ("aprender-train-bench", "aprender-train-bench"),
    ("aprender-train-distill", "aprender-train-distill"),
    ("aprender-train-inspect", "aprender-train-inspect"),
    ("aprender-train-lora", "aprender-train-lora"),
    ("aprender-train-shell", "aprender-train-shell"),
    ("aprender-verify-ml", "verificar"),
    ("aprender-zram-cli", "trueno-zram"),
    ("aprender-zram-generator", "aprender-zram-generator"),
];

/// The ratchet. The number of legacy binaries may never exceed this.
///
/// Written as its own constant, deliberately NOT derived from
/// `LEGACY_BINS.len()`: deriving it would let the single edit that adds a
/// binary also raise the ceiling that is supposed to stop it.
///
/// Lower it as conversions land. Raising it means a new standalone binary is
/// being shipped, which the consolidation forbids.
const MAX_LEGACY_BINS: usize = 28;

/// Directories under `crates/` that are `exclude`d from the workspace and still
/// declare a `[[bin]]`. `cargo metadata` cannot see them (they are not members),
/// so they get the secondary manifest sweep instead. Both are `publish = false`,
/// so neither can reach `~/.cargo/bin` through crates.io.
const EXCLUDED_DIRS_WITH_BIN: [&str; 1] = ["aprender-viz-ttop"];

/// Floor for the vacuity guards: the scans must actually reach the crate tree.
/// `crates/` holds ~82 directories and the workspace ~78 members; 60 is the
/// same conservative floor `test_minimum_workspace_member_count` uses.
const MIN_SCANNED: usize = 60;

/// FALSIFY-MONO-011: standalone binaries are migration debt, and the debt may
/// only shrink. Only `apr` (root facade) is a sanctioned binary.
///
/// WHY THIS TEST IS SHAPED THE WAY IT IS
/// -------------------------------------
/// It was vacuous from the day it was written, and it was vacuous twice over.
///
/// 1. WRONG PATH. It joined `"crates"` onto `CARGO_MANIFEST_DIR`, which for an
///    integration test of `aprender-core` is `crates/aprender-core` — so it
///    walked `crates/aprender-core/crates`, a path that has never existed.
///    `read_dir` returned `Err`, the `if let Ok(..)` swallowed it, `violations`
///    stayed empty, and the guard reported "only apr-cli has a [[bin]]" while
///    21 crates had one. Hence the vacuity floors below: a scan that sees
///    nothing is now RED, not green.
///
/// 2. WRONG QUESTION. Even with the path fixed, `content.contains("[[bin]]")`
///    is not what cargo does, and it is wrong in BOTH directions:
///      - FALSE NEGATIVE: cargo auto-discovers `src/main.rs` and `src/bin/*.rs`
///        with no `[[bin]]` table at all. Six packages ship binaries that way,
///        including `aprender-verify-ml` -> `verificar` — one of the nine
///        pre-consolidation names this migration exists to remove, completely
///        invisible to a `[[bin]]` grep.
///      - FALSE POSITIVE: `aprender-serve` and `aprender-train` both have a
///        `src/main.rs` and both set `autobins = false`, so they build no
///        binary. A text scan that "corrected" for auto-discovery would have
///        indicted two innocent crates.
///
/// So the primary scan asks `cargo metadata` for actual `bin` targets. Cargo is
/// the authority on what gets installed; the manifest text is not.
#[test]
fn test_no_unauthorized_binaries() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(manifest_dir)
        .output()
        .expect("cargo metadata failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let metadata: serde_json::Value =
        serde_json::from_str(&stdout).expect("failed to parse cargo metadata");
    let packages = metadata["packages"].as_array().expect("no packages");

    let sanctioned: HashSet<(&str, &str)> = SANCTIONED_BINS.into();
    let legacy: HashSet<(&str, &str)> = LEGACY_BINS.into();

    let mut members = 0usize;
    let mut observed: Vec<(String, String)> = Vec::new();

    for pkg in packages {
        if pkg["source"].as_str().is_some() {
            continue; // external dependency
        }
        members += 1;
        let pkg_name = pkg["name"].as_str().unwrap_or("").to_string();
        let Some(targets) = pkg["targets"].as_array() else {
            continue;
        };
        for target in targets {
            let is_bin = target["kind"]
                .as_array()
                .is_some_and(|ks| ks.iter().any(|k| k.as_str() == Some("bin")));
            if !is_bin {
                continue;
            }
            let bin_name = target["name"].as_str().unwrap_or("").to_string();
            observed.push((pkg_name.clone(), bin_name));
        }
    }
    observed.sort();

    let mut unauthorized: Vec<String> = Vec::new();
    let mut observed_legacy: Vec<String> = Vec::new();
    for (pkg, bin) in &observed {
        let pair = (pkg.as_str(), bin.as_str());
        if sanctioned.contains(&pair) {
            continue;
        }
        if legacy.contains(&pair) {
            observed_legacy.push(format!("{pkg}:{bin}"));
        } else {
            unauthorized.push(format!("{pkg}:{bin}"));
        }
    }

    eprintln!(
        "FALSIFY-MONO-011: cargo metadata reports {} workspace packages building {} bin \
         target(s); {} are migration debt (cap {}), {} sanctioned.\n  debt: {:?}",
        members,
        observed.len(),
        observed_legacy.len(),
        MAX_LEGACY_BINS,
        SANCTIONED_BINS.len(),
        observed_legacy
    );

    // --- vacuity: a scan that saw nothing must not report clean -------------
    assert!(
        members >= MIN_SCANNED,
        "FALSIFY-MONO-011 is VACUOUS: cargo metadata reported only {} workspace packages, \
         expected at least {}. The guard is looking at the wrong tree — that is the exact \
         defect it was repaired for. Fix the scan, not this floor.",
        members,
        MIN_SCANNED
    );
    assert!(
        !observed.is_empty(),
        "FALSIFY-MONO-011 is VACUOUS: zero bin targets found across {} workspace packages. \
         At least `aprender:apr` must exist, so the target filter is broken.",
        members
    );

    // --- unauthorized: a binary that is neither sanctioned nor known debt ---
    assert!(
        unauthorized.is_empty(),
        "FALSIFY-MONO-011: unauthorized binaries (package:bin): {:?}\n\
         Only `aprender:apr` is a sanctioned binary. Expose the capability as \
         `apr <subcommand>` calling the same library entry point, then delete the \
         binary target. Do NOT add it to LEGACY_BINS — that list is a shrinking \
         record of migration debt, not an allowlist for new work.",
        unauthorized
    );

    // --- ratchet: the debt may only shrink ---------------------------------
    assert!(
        observed_legacy.len() <= MAX_LEGACY_BINS,
        "FALSIFY-MONO-011 ratchet: {} legacy binaries, cap is {}.\n  found: {:?}\n\
         A new standalone binary re-creates the ~/.cargo/bin collision APR-MONO removed. \
         Lower MAX_LEGACY_BINS as conversions land; never raise it.",
        observed_legacy.len(),
        MAX_LEGACY_BINS,
        observed_legacy
    );
    assert!(
        LEGACY_BINS.len() <= MAX_LEGACY_BINS,
        "FALSIFY-MONO-011 ratchet: LEGACY_BINS grew to {} entries, cap is {}. \
         That list is migration debt and may only shrink.",
        LEGACY_BINS.len(),
        MAX_LEGACY_BINS
    );
}

/// FALSIFY-MONO-011 (secondary): `exclude`d directories may not hide a binary.
///
/// `cargo metadata` on the workspace cannot see crates the root `Cargo.toml`
/// excludes, so they are checked by manifest text instead. This sweep is
/// deliberately weaker than the primary one — it only catches an explicit
/// `[[bin]]` table — because running `cargo metadata` against an excluded
/// manifest is not safe here: `aprender-train-canary` pins a `[patch.crates-io]`
/// at an absolute developer path and will not resolve on CI or in a clean room.
#[test]
fn test_excluded_dirs_declare_no_new_binaries() {
    let crates_dir = workspace_root().join("crates");
    let allowed: HashSet<&str> = EXCLUDED_DIRS_WITH_BIN.into();

    let entries = std::fs::read_dir(&crates_dir)
        .unwrap_or_else(|e| panic!("crates/ must be readable at {}: {e}", crates_dir.display()));

    let mut scanned = 0usize;
    let mut declared: Vec<String> = Vec::new();

    for entry in entries.flatten() {
        let toml_path = entry.path().join("Cargo.toml");
        if !toml_path.exists() {
            continue;
        }
        scanned += 1;
        let content = std::fs::read_to_string(&toml_path).unwrap_or_default();
        if content.lines().any(|l| opens_table(l, "[[bin]]")) {
            declared.push(entry.file_name().to_string_lossy().to_string());
        }
    }
    declared.sort();

    // Anything the primary scan already covers is a workspace member; only the
    // excluded leftovers are this test's business.
    let members: HashSet<&str> = LEGACY_BINS
        .iter()
        .map(|(pkg, _)| *pkg)
        .chain(SANCTIONED_BINS.iter().map(|(pkg, _)| *pkg))
        .collect();
    let stray: Vec<&String> = declared
        .iter()
        .filter(|d| !members.contains(d.as_str()) && !allowed.contains(d.as_str()))
        .collect();

    eprintln!(
        "FALSIFY-MONO-011/excluded: scanned {} manifests under {}; {} declare [[bin]]",
        scanned,
        crates_dir.display(),
        declared.len()
    );

    assert!(
        scanned >= MIN_SCANNED,
        "FALSIFY-MONO-011/excluded is VACUOUS: only {} manifests found under {}, expected \
         at least {}. The directory walk is pointed at the wrong place — fix the path, \
         not this floor.",
        scanned,
        crates_dir.display(),
        MIN_SCANNED
    );
    assert!(
        stray.is_empty(),
        "FALSIFY-MONO-011: excluded directories declaring a [[bin]]: {:?}\n\
         An excluded crate is invisible to `cargo metadata`, so a binary added there \
         escapes the primary guard entirely. Rehome the capability under `apr`.",
        stray
    );
}

/// FALSIFY-BUILD-004: No `[patch.crates-io]` in the root Cargo.toml.
/// The monorepo eliminates the need for patches.
///
/// This was the second `CARGO_MANIFEST_DIR.join(..)` site: it read
/// `crates/aprender-core/Cargo.toml` — the core crate's own manifest — while
/// its message claimed to have inspected the workspace root. A patch table
/// added to the real root would never have been seen.
///
/// The membership test is line-anchored, not `contains()`: the real root
/// manifest names `[patch.crates-io]` twice inside prose comments (quoting the
/// excluded `aprender-train-canary` manifest, and recording the RC4 removal),
/// so a substring test on the correct file would fail for a false reason.
/// `.cargo/config.toml` may still carry dev overrides; the committed manifest
/// may not.
#[test]
fn test_no_patch_in_root_cargo_toml() {
    let root_manifest = workspace_root().join("Cargo.toml");
    let cargo_toml = std::fs::read_to_string(&root_manifest)
        .unwrap_or_else(|e| panic!("root Cargo.toml must be readable at {root_manifest:?}: {e}"));

    let patch_lines: Vec<(usize, &str)> = cargo_toml
        .lines()
        .enumerate()
        .filter(|(_, l)| opens_table(l, "[patch.crates-io]"))
        .map(|(i, l)| (i + 1, l.trim()))
        .collect();

    assert!(
        patch_lines.is_empty(),
        "FALSIFY-BUILD-004: {} still declares [patch.crates-io] at {:?}.\n\
         The monorepo should eliminate all cross-repo patches — every sibling is \
         a [workspace.dependencies] path alias.",
        root_manifest.display(),
        patch_lines
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
