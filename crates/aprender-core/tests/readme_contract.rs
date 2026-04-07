//! README.md Contract Enforcement
//!
//! Enforces: contracts/apr-docs-v1.yaml
//! FALSIFY-README-001 through FALSIFY-README-005 + FALSIFY-SVG-002/003
//!
//! These tests prevent README drift from the actual workspace state.

use std::path::Path;
use std::process::Command;

fn workspace_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn read_readme() -> String {
    std::fs::read_to_string(workspace_root().join("README.md"))
        .expect("README.md must exist at workspace root")
}

/// FALSIFY-README-001: README contains `cargo install aprender`
#[test]
fn test_readme_has_install_command() {
    let readme = read_readme();
    assert!(
        readme.contains("cargo install aprender"),
        "FALSIFY-README-001: README.md must contain 'cargo install aprender'"
    );
}

/// FALSIFY-README-002: README contains `apr run` example
#[test]
fn test_readme_has_apr_run_example() {
    let readme = read_readme();
    assert!(
        readme.contains("apr run"),
        "FALSIFY-README-002: README.md must contain 'apr run' usage example"
    );
}

/// FALSIFY-README-003: README does NOT reference `cargo install apr-cli`
#[test]
fn test_readme_no_apr_cli_install() {
    let readme = read_readme();
    assert!(
        !readme.contains("cargo install apr-cli"),
        "FALSIFY-README-003: README.md must not contain 'cargo install apr-cli'"
    );
}

/// FALSIFY-README-004: README does NOT reference old repos as installable
#[test]
fn test_readme_no_stale_install_refs() {
    let readme = read_readme();
    for old in &["cargo install trueno", "cargo install realizar",
                  "cargo install entrenar", "cargo install batuta"] {
        assert!(
            !readme.contains(old),
            "FALSIFY-README-004: README.md must not contain '{old}'"
        );
    }
}

/// FALSIFY-README-005: Crate count in README matches workspace
#[test]
fn test_readme_crate_count_matches_workspace() {
    let readme = read_readme();
    let ws_root = workspace_root();

    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(&ws_root)
        .output()
        .expect("cargo metadata failed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let metadata: serde_json::Value = serde_json::from_str(&stdout)
        .expect("failed to parse cargo metadata");

    let ws_count = metadata["packages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|p| p["source"].is_null())
        .count();

    let count_str = format!("**{}**", ws_count);
    assert!(
        readme.contains(&count_str),
        "FALSIFY-README-005: README says crate count that doesn't match workspace ({ws_count})"
    );
}

/// FALSIFY-SVG-002: Hero SVG is accessible
#[test]
fn test_hero_svg_accessible() {
    let svg = std::fs::read_to_string(workspace_root().join("docs/hero.svg"))
        .expect("docs/hero.svg must exist");
    assert!(svg.contains(r#"role="img""#), "FALSIFY-SVG-002: hero.svg missing role=img");
    assert!(svg.contains("aria-label"), "FALSIFY-SVG-002: hero.svg missing aria-label");
    assert!(svg.contains("<title>"), "FALSIFY-SVG-002: hero.svg missing <title>");
}

/// FALSIFY-README-006: Framework comparison section exists with real data
#[test]
fn test_readme_has_framework_comparison() {
    let readme = read_readme();
    assert!(readme.contains("## Framework Comparison"), "Missing Framework Comparison section");
    assert!(readme.contains("candle-vs-apr"), "Missing candle-vs-apr citation");
    assert!(readme.contains("ground-truth-apr-ludwig"), "Missing ludwig comparison citation");
    assert!(readme.contains("369.9"), "Missing realizr benchmark number (369.9 tok/s)");
    assert!(readme.contains("3,220"), "Missing batched throughput number (3,220 tok/s)");
}

/// FALSIFY-SVG-003: Hero SVG is valid XML
#[test]
fn test_hero_svg_valid() {
    let svg_path = workspace_root().join("docs/hero.svg");
    let content = std::fs::read_to_string(&svg_path).expect("read hero.svg");
    // Basic XML validation — starts with <svg, ends with </svg>
    assert!(content.trim().starts_with("<svg"), "FALSIFY-SVG-003: not valid SVG");
    assert!(content.trim().ends_with("</svg>"), "FALSIFY-SVG-003: not valid SVG");
}

/// FALSIFY-README-CRATE-001: Every crate has README.md
#[test]
fn test_every_crate_has_readme() {
    let ws_root = workspace_root();
    let crates_dir = ws_root.join("crates");
    let mut missing = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&crates_dir) {
        for entry in entries.flatten() {
            if !entry.path().join("Cargo.toml").exists() {
                continue; // Not a crate
            }
            if !entry.path().join("README.md").exists() {
                missing.push(entry.file_name().to_string_lossy().to_string());
            }
        }
    }

    assert!(
        missing.is_empty(),
        "FALSIFY-README-CRATE-001: Crates missing README.md: {:?}",
        missing
    );
}

/// FALSIFY-README-CRATE-002: Every README links to monorepo
#[test]
fn test_every_readme_links_monorepo() {
    let ws_root = workspace_root();
    let crates_dir = ws_root.join("crates");
    let mut no_link = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&crates_dir) {
        for entry in entries.flatten() {
            let readme = entry.path().join("README.md");
            if !readme.exists() {
                continue;
            }
            let content = std::fs::read_to_string(&readme).unwrap_or_default();
            if !content.contains("paiml/aprender") {
                no_link.push(entry.file_name().to_string_lossy().to_string());
            }
        }
    }

    assert!(
        no_link.is_empty(),
        "FALSIFY-README-CRATE-002: READMEs without monorepo link: {:?}",
        no_link
    );
}
